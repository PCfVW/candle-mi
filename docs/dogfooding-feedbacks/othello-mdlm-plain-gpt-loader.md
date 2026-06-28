# Loaders should cover plain-GPT backbones (the OthelloMDLM case)

**Date:** June 28, 2026
**Source:** askesis Othello-MDLM study — P4 (causal intervention / RQ3)
**Affected area:** model loaders (`src/diffusion/mdlm.rs`, `src/transformer/mod.rs`) + `MIBackend`
**Severity:** Feature — a new backbone family (plain GPT-2-style) + its `MIBackend` port

---

## The use case

The askesis Othello-MDLM study probes a masked-diffusion world model against ground truth.
P1–P3 are done in PyTorch (the trusted oracle): an autoregressive Othello-GPT control and a
bidirectional masked-diffusion model (`OthelloMDLM`) on the *same* backbone and data, probed for
the Li→Nanda board representation. The headline result is in hand — the linear player-relative
world model **transfers** to masked diffusion but is **substantially weaker** (~0.79 vs the AR's
0.96).

**P4 is the causal test (RQ3):** add a probe direction to the residual stream to flip a board cell
MINE↔YOURS, and measure whether the model's predicted legal moves shift to those of the
counterfactual board (vs a norm-matched random-direction control). The plan — and the program's
house method (*calibrate before claim*) — is:

1. implement P4 in **PyTorch** (the oracle), emitting fixtures;
2. reimplement it in **candle-mi**, cross-validated against those fixtures;
3. lean on the now-loaded model for **P5** (commitment / the denoising-`k` axis), where
   `generate_trajectory` + the diffusion logit-lens earn their keep.

Step 2 is the dogfooding. **The friction is not the intervention — candle-mi's hook surface already
does exactly what P4 needs. The friction is the loader: candle-mi has no path to load a plain
GPT-2-style backbone.**

## The gap

candle-mi (v0.1.14) has two model families, and `OthelloMDLM` fits neither:

| loader | positions | norm | conditioning | fits `OthelloMDLM`? |
|---|---|---|---|---|
| `GenericMdlm` (`src/diffusion/mdlm.rs`) | RoPE (`MdlmRope`) | weight-only LayerNorm | adaLN modulation + `sigma_map` | ❌ DiT/adaLN-locked |
| `GenericTransformer` (`src/transformer/mod.rs`) | RoPE | RMSNorm/LayerNorm | none (decoder) | ❌ no **learned absolute** pos-emb |

`GenericMdlm` is hard-wired to the DiT/adaLN `mdlm-owt` layout (`backbone.*` keys, fused QKV with
**no bias**, weight-only LayerNorm, a constant time-conditioning vector). `GenericTransformer` is
nicely configurable (causal masking is optional — it already serves the bidirectional Dream / a2d
diffusion decoders in the Qwen layout — and LayerNorm / plain-MLP are options), but its positional
encoding is **RoPE**. `OthelloMDLM` is a **GPT-2-style** backbone with **learned absolute
positional embeddings** (`nn.Embedding`), which RoPE cannot express. That single fact is the
blocker.

## The model to load: `OthelloMDLM`

A ~25.3 M-param bidirectional GPT-2-style transformer (nanoGPT/minGPT lineage). Config
(`OthelloMDLMConfig`): `vocab_size=62` (60 move cells + pad + `[MASK]`), `block_size=60`,
`n_layer=8`, `n_head=8`, `n_embd=512`, `dropout=0.0`, `causal=false`.

**Forward** (no causal mask, no time conditioning):

```
x = tok_emb[idx] + pos_emb[0..T]                          # learned token + absolute positions
for blk in blocks:
    x = x + proj(SDPA(qkv(ln1(x)), is_causal=false))      # full bidirectional attention
    x = x + mlp2(gelu_erf(mlp1(ln2(x))))                  # 4x GELU MLP
logits = head(ln_f(x))                                    # untied head, no bias
```

**Checkpoint** (`torch.save` dict): weights under `ckpt["model"]`, the config under
`ckpt["config"]`. State-dict keys (PyTorch `Linear` stores `weight` as `[out, in]` — **same
convention as candle `Linear`, so no transposes are needed**, only a name remap):

| PyTorch key (`ckpt["model"]`) | shape | candle target |
|---|---|---|
| `tok_emb.weight` | `[62, 512]` | token embedding |
| `pos_emb.weight` | `[60, 512]` | **learned** positional embedding |
| `blocks.{i}.ln1.{weight,bias}` | `[512]` | full LayerNorm (pre-attn) |
| `blocks.{i}.attn.qkv.{weight,bias}` | `[1536, 512]` / `[1536]` | fused QKV **with bias** |
| `blocks.{i}.attn.proj.{weight,bias}` | `[512, 512]` / `[512]` | attn output **with bias** |
| `blocks.{i}.ln2.{weight,bias}` | `[512]` | full LayerNorm (pre-MLP) |
| `blocks.{i}.mlp.0.{weight,bias}` | `[2048, 512]` / `[2048]` | MLP fc (`nn.Sequential` idx 0) |
| `blocks.{i}.mlp.2.{weight,bias}` | `[512, 2048]` / `[512]` | MLP proj (idx 2; idx 1 is GELU) |
| `ln_f.{weight,bias}` | `[512]` | final LayerNorm |
| `head.weight` | `[62, 512]` | untied head, **no bias** |

### Parity gotchas (these are what bite during the port)

These are exactly the deltas from `GenericMdlm`, and each is a silent-divergence risk:

- **GELU variant.** PyTorch `nn.GELU()` defaults to the **exact (erf)** GELU. candle's `Tensor::gelu()`
  is the **tanh** approximation; use **`Tensor::gelu_erf()`**. (`GenericMdlm` uses GELU-tanh — do not
  copy it here.)
- **Biases everywhere.** QKV, attn-proj, both MLP linears, and **both** LayerNorm affine params
  (weight *and* bias) are present. `GenericMdlm` omits attn biases and LayerNorm bias — the opposite.
- **Full LayerNorm** (weight + bias, eps `1e-5`), not weight-only.
- **No conditioning.** There is no timestep / `sigma` input at all — the masked-diffusion noise level
  is handled by the training loss, not a model input. Drop the adaLN path entirely.
- **Learned absolute positions**, added to the token embedding; no RoPE.
- **Attention scale** `1/sqrt(head_dim)` = `1/sqrt(64)`, standard softmax SDPA.

## Two implementation paths

1. **A small dedicated model** (recommended). A ~150-line `OthelloGpt` module
   (token + learned-pos embedding → N pre-LN blocks with fused-QKV bidirectional attention +
   `gelu_erf` MLP → final LN → untied head) implementing `MIBackend`. Faithful, self-contained, and
   the cleanest thing to differential-test. It is genuinely new surface area (no DiT, no RoPE).
2. **Generalize `GenericTransformer`.** Add (a) a *learned-absolute* positional-embedding option
   alongside RoPE and (b) confirm the LayerNorm + plain-GELU + with-bias combination round-trips.
   More reuse, but the learned-pos-emb addition touches the hot path of every model the struct
   serves — higher blast radius for a one-off backbone. Better as a follow-up if a second
   GPT-2-style model ever appears.

Either way, the checkpoint is converted **`.pt` → safetensors** first (a ~15-line Python script:
load `ckpt["model"]`, `safetensors.torch.save_file`, applying the name remap above), then loaded via
`VarBuilder::from_mmaped_safetensors`. (candle's `VarBuilder::from_pth` could read the `.pt`
directly, but a safetensors export with the remap baked in matches the candle-mi norm and keeps the
loader free of pickle.)

## What the model must expose for P4/P5

P4 maps **directly** onto the existing hook surface once `OthelloMDLM` implements `MIBackend`:

```rust
// P4 intervention: flip a cell by adding its probe direction at a chosen layer.
let mut hooks = HookSpec::new();
hooks.intervene(HookPoint::ResidPost(layer), Intervention::Add(steer_vec));
let cache = model.forward(&input_ids, &hooks)?;     // read shifted move logits
let logits = cache.output();
```

- **Reading** (the probe capture, already used in PyTorch P3): `hooks.capture(HookPoint::ResidPost(i))`
  for every block `i` — the per-layer residual stream the board probes are trained on.
- **Intervening** (P4): `Intervention::Add(v)` at `HookPoint::ResidPost(layer)` — exactly the
  Nanda-style vector edit; the random-direction control is the same call with a norm-matched `v`.
- **Trajectory** (P5): `generate_trajectory(&model, …)` works for any `&dyn MIBackend`, so once the
  loader lands, the denoising-`k` axis and the diffusion logit-lens come for free.

So the **only** new work is the loader + a faithful `MIBackend::forward` that populates the standard
`HookPoint`s (at minimum `Embed`, `ResidPre/Mid/Post(i)`, `AttnOut(i)`, `MlpOut(i)`, `FinalNorm`).
Note the crate lints (`unwrap_used`, `expect_used`, `panic`, `indexing_slicing` are **deny**): the
loader's tensor lookups and head-dim reshapes must be fallible (`?` / `.get(...)`), not indexed.

## The differential test (calibrate before claim)

The PyTorch P4 will emit fixtures so the candle port is a *reproduction*, not a re-derivation:

1. **Forward parity** — feed N fixed games (clean, `t=0`) through both; assert max-abs logit diff
   within the program bars (**~1e-3 CPU / 5e-3 GPU**), as the `mdlm-owt` port was checked against the
   `mdlm-owt-noflash` fp32 oracle (3e-5).
2. **Capture parity** — per-layer `ResidPost(i)` matches the PyTorch `output_hidden_states` to the
   same bar (this is what the board probes consume).
3. **Intervention parity** — a handful of canonical cases `(game, position, layer, cell, target-flip)`
   → the measured **legal-move-distribution shift** and the **random-direction control** value,
   reproduced within tolerance.

If forward + capture parity hold, the intervention parity is the real acceptance test for the
dogfood.

**Status (2026-06-28):** (1) and (2) are **ready**. The askesis exporter
(`reference/othello_mdlm/export_fixtures.py`) produced, from `epoch_4.pt`:

- `weights.safetensors` — the 101-tensor `state_dict` under the keys in the table above, fp32;
- `forward_capture.safetensors` — `input_ids [4, 60]`, `logits [4, 60, 62]`, `resid_post.0..7
  [4, 60, 512]`, fp32, t=0;

both with provenance (checkpoint, config, torch version, game indices) in the safetensors metadata.
Copy them from `askesis/reference/data/fixtures/othello_mdlm/` into the candle-mi test tree. The
**loader + `MIBackend::forward` depend on neither file** (only the checkpoint and this spec), so they
can start now; (3) the **intervention** fixtures arrive with the PyTorch P4 harness — not yet built
(P4 is the next askesis step).

## Why this is good dogfooding

- It exercises `MIBackend` / the hook system on a backbone family candle-mi has **never loaded** —
  not DiT, not RoPE, not Qwen — a real test that the abstraction isn't quietly DiT/Qwen-shaped.
- It **front-loads** the loader so **P5** (the commitment study, where `generate_trajectory` and the
  diffusion logit-lens are the whole point) is ready with no extra plumbing.
- It produces a second, *minimal* reference model in the crate — useful for fast hook/intervention
  tests that don't need a 648 MB `mdlm-owt` download.

## Concrete asks

1. A loader for a **plain GPT-2-style backbone** (learned absolute positions, full LayerNorm,
   with-bias attention, `gelu_erf` MLP, untied head, optional causal mask) — either a small dedicated
   module (preferred) or a `GenericTransformer` learned-pos-emb option.
2. A short **"adding a model" note** in `docs/` capturing the parity gotchas above (GELU variant,
   bias presence, LayerNorm affine, no-conditioning) — they are the recurring traps when porting a
   PyTorch backbone, and the next non-DiT model will hit the same ones.

*General rule:* the loaders currently encode two specific layouts (DiT/adaLN, RoPE/Qwen). A plain
GPT-2-style backbone — the most common interpretability target after LLaMA-family — should be a
first-class, documented option, not a per-study reimplementation.
