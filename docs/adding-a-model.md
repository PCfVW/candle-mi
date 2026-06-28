# Adding a model: porting a PyTorch backbone to candle-mi

This note captures the recurring traps when porting a PyTorch transformer to a
candle-mi `MIBackend`. It is framework-agnostic — the same checklist applies to
any LLaMA-, Qwen-, GPT-2-, or DiT-style backbone. The `OthelloGpt` port
(`src/diffusion/othello.rs`, a plain GPT-2-style backbone) is the worked example
throughout.

candle-mi already ships two configurable backbones — `GenericTransformer`
(RoPE / RMSNorm or LayerNorm decoder, optionally bidirectional) and
`GenericMdlm` (a DiT with adaLN). **Reach for a dedicated module when the new
architecture differs in something those two hard-code** (positional scheme,
norm placement, conditioning). For `OthelloGpt` the blocker was *learned
absolute* positional embeddings, which RoPE cannot express; generalizing
`GenericTransformer` would have touched the hot path of seven RoPE families for
one 25 M-param backbone, so a small faithful module was the better trade.

## The five silent-divergence traps

Each of these compiles and runs but produces *wrong* logits — no error, just
drift. They are the deltas most often missed when copying from a sibling
backend, and a forward-parity test is the only reliable catch.

| # | Trap | What to check in the PyTorch source | candle equivalent |
|---|------|-------------------------------------|-------------------|
| 1 | **GELU variant** | `nn.GELU()` (no arg) = **exact erf**; `nn.GELU(approximate='tanh')` = tanh. nanoGPT/minGPT lineage uses *both* across forks — read the line, do not assume. | `Tensor::gelu_erf()` for erf, `Tensor::gelu()` for tanh |
| 2 | **Bias presence** | Each `nn.Linear(..., bias=?)` and whether `nn.LayerNorm` is affine. GPT-2 has bias on QKV/proj/MLP and a no-bias head; DiT/MDLM often drops attention bias entirely. | `candle_nn::linear` (bias) vs `linear_no_bias` |
| 3 | **Norm type & affine** | `LayerNorm` (mean-subtracting, weight **and** bias) vs `RMSNorm` (weight only) vs weight-only `LayerNorm`. Also the `eps` (GPT-2/LLaMA use `1e-5`, some use `1e-6`). | `candle_nn::layer_norm` (full) / `rms_norm` / `LayerNorm::new_no_bias` |
| 4 | **Positional scheme** | Learned absolute (`nn.Embedding(block_size, d)`, added at input) vs RoPE (applied to Q/K per layer) vs ALiBi. They are mutually exclusive and not interchangeable. | add `pos_emb` rows at input vs a per-layer `RopeCache` |
| 5 | **Conditioning** | Is there a timestep / `sigma` / class input (DiT adaLN)? A plain LM has **none** — do not copy an adaLN sibling's modulation path. | drop the modulation entirely |

Two more that bite less often but are worth a glance: **attention scale**
(`1/sqrt(head_dim)` is standard, but some models fold a different scalar in) and
**weight tying** (is the head tied to `tok_emb`, or a separate `head.weight`?).

## Weight keys: prefer verbatim over remap

PyTorch `nn.Linear` stores `weight` as `[out, in]` — the **same** convention as
candle `Linear`, so **no transpose is needed**, only key navigation. candle's
`VarBuilder::pp(...)` chains map onto PyTorch module paths one-to-one:

| PyTorch module path | `VarBuilder` access | tensor key read |
|---------------------|---------------------|-----------------|
| `blocks[i].attn.qkv` | `vb.pp("blocks").pp(i).pp("attn").pp("qkv")` | `blocks.{i}.attn.qkv.{weight,bias}` |
| `blocks[i].mlp[0]` (`nn.Sequential`) | `vb.pp("blocks").pp(i).pp("mlp").pp("0")` | `blocks.{i}.mlp.0.{weight,bias}` |
| `head` (`bias=False`) | `vb.pp("head")` via `linear_no_bias` | `head.weight` |

Because of this, the cleanest export is the **state dict verbatim** — keep the
original key names and let the candle loader navigate them. No remap table to
keep in sync, and a complete load has no missing/unexpected keys. For
`OthelloGpt` this means the `.pt → safetensors` step is a pure lift
(`scripts/convert_othello_mdlm.py`), or — when an upstream study already emits
safetensors — no conversion at all.

## Populate the standard hook points

A faithful `MIBackend::forward` should populate the standard `HookPoint`s so the
MI tooling (probes, logit lens, steering) transfers unchanged:

- `Embed` (post token + position embedding),
- per block `i`: `ResidPre(i)`, `AttnQ/K/V(i)`, `AttnScores(i)`, `AttnPattern(i)`,
  `AttnOut(i)`, `ResidMid(i)`, `MlpPre(i)`, `MlpPost(i)`, `MlpOut(i)`,
  `ResidPost(i)`,
- `FinalNorm` (post final norm, pre head).

Follow the TransformerLens convention: `AttnOut`/`MlpOut` capture the sublayer
contribution **actually added to the residual stream**. Map the PyTorch
`output_hidden_states` list to `ResidPost(i)` — that residual-after-block tensor
is what board/feature probes are trained on.

Under the crate lints (`unwrap_used`, `expect_used`, `panic`, `indexing_slicing`
are **deny**), every tensor lookup and head-dim reshape must be fallible: use
`vb.get(...)?` and `?`, never indexing.

## Calibrate before claim: the differential test

Port as a *reproduction*, not a re-derivation. Have the PyTorch side emit fp32
fixtures and gate the candle port against them:

1. **Forward parity** — feed `N` fixed inputs through both; assert max-abs logit
   diff within the program bars (**~1e-3 CPU / 5e-3 GPU**).
2. **Capture parity** — each per-layer `ResidPost(i)` matches the PyTorch
   `output_hidden_states` to the same bar (this is what probes consume).
3. **Intervention parity** (when applicable) — a handful of canonical edits
   reproduced within tolerance; this reuses the already-tested hook machinery,
   so forward + capture parity are the real loader gates.

Export the fixtures as safetensors (`input_ids`, `logits`, `resid_post.{i}`) and
load them in a `#[ignore]` integration test that skips when the fixtures are
absent (see `tests/validate_othello_forward.rs`, pointed at its fixtures via an
environment variable). Keep large weights/fixtures **out** of the committed
crate — they are regenerable, and the published package excludes data.

> Worked result: the `OthelloGpt` port reproduced the fp32 oracle to
> **4.18e-5** (logits) / **2.59e-4** (worst of 8 `resid_post` layers) on CPU,
> well within the 1e-3 bar — and trap #1 (erf vs tanh GELU) was confirmed by
> reading the model source, not guessed.
