# Diffusion-LM support roadmap

Status of masked-diffusion-language-model support in candle-mi, and the planned
increments. This is the canonical, versioned record; the per-session plan files
under `~/.claude/plans/` are scratchpads, not engineering memory.

## Why a separate backend (and where the transformer backend is reused)

Masked-diffusion LMs split into two architecture families:

| Family | Examples | Tensor layout | Difference from a causal LM | candle-mi path |
|---|---|---|---|---|
| **DiT-style** | MDLM, SEDD | `backbone.*`, `adaLN`, fused `attn_qkv`, `mlp.0`/`mlp.2` | a whole new block (adaLN modulation) | dedicated `GenericMdlm` backend (`diffusion` feature) |
| **Decoder-style** | Dream (←Qwen2.5), Block Diffusion, LLaDA | standard `model.layers.*` decoder | **only the attention mask** (bidirectional) + the denoising loop | reuse `GenericTransformer` with a bidirectional flag (Stage 3) |

The auto-config path (`config.rs::parse_auto`, used by `from_pretrained` for
unknown `model_type`s) is a **causal-decoder** detector — it keys on the
LLaMA/Qwen layout (`layers.0.self_attn.*`, `mlp.gate_proj`, …). It therefore
**cannot** load DiT-style checkpoints (different tensor names and config keys),
but it **already infers** decoder-style ones (Dream is Qwen2.5) — so growing
coverage to Dream is mostly a one-field bidirectional flag, not a new backend.

## Stage 1 — MDLM forward pass + fp32 oracle — DONE

Commit `4a9c774` (not yet pushed; ROADMAP ratio fix in `dd5f161`).

- Standalone `diffusion` feature; `src/diffusion/{mod,config,rope,mdlm}.rs` →
  `GenericMdlm`, `MdlmConfig`, `SUPPORTED_DIFFUSION_MODEL_TYPES`.
- Ports `kuleshov-group/mdlm-owt`: bidirectional DiT, constant `adaLN`
  conditioning (`time_conditioning=false` → `c = silu(sigma_map(0))` precomputed
  at load; `time_conditioning=true` is rejected), weight-only `LayerNorm`, fused
  QKV, rotary on q/k only, plain GELU-tanh MLP, untied output head.
- `from_pretrained` dispatch on `model_type "mdlm"`; full `MIBackend` hook
  surface so the existing analysis primitives work unchanged.
- Validated against a from-first-principles fp32 oracle built on the
  flash-attn-free `TheQweaker/mdlm-owt-noflash` (byte-identical weights):
  top-10 logits exact, **max abs-diff 3.05×10⁻⁵ (CPU) / 1.34×10⁻⁵ (GPU)**.
  See `tests/validate_mdlm_forward.rs` + `scripts/mdlm_forward_validation.py`.
- Demo: `examples/quick_start_mdlm.rs` (masked fill-in → `" Paris"`).

## Stage 2 — MDLM denoising sampler + SAE-free MI examples — DONE

Commits `4a9c774`→ + the sampler/examples commits (not yet pushed).

- **SUBS ancestral sampler** (`src/diffusion/sample.rs`; port of the noflash
  `sample.py`): absorbing/masked diffusion, linear schedule `t: 1 → 0` over `K`
  steps, carry-over unmasking, zero-mask-probability (`[MASK]` logit → −∞),
  temperature, optional top-k; deterministic by seed. `generate` (final tokens)
  + `generate_trajectory` (per-step states = the `k` axis). Tested by model-free
  unit tests (SUBS / top-k / determinism) + a model invariant test (determinism,
  monotone unmasking, termination, prompt carry-over).
- **Diffusion-time logit lens** (`examples/diffusion_logit_lens.rs`): the
  `(layer × denoising-step)` slice of the `(k, ℓ, π)` object at a masked target
  position — watch the prediction crystallize over `k` (validated: target →
  "located", all layers converge by k=2).
- **Decoding-order analysis** (`examples/diffusion_decoding_order.rs`): random vs
  confidence vs entropy unmasking order, with per-order reveal-confidence and
  prediction-stability (SAE-free proxy for per-step feature stability). Validated:
  entropy > confidence > random on both metrics.

Both examples operate on raw residual-stream hooks (no SAE). They are
provenance-agnostic and are reused unchanged in Stage 3. The full
SAE-feature-stability decoding-order experiment needs a trained SAE (deferred).

## Stage 3 — Decoder-style DLMs via a bidirectional `GenericTransformer` — DONE

Commits `8c60a61` (3a) → `5260d67` (3d). Decoder-style masked-diffusion LMs reuse
the Qwen weight layout verbatim; the only forward delta is bidirectional attention.

- **3a** (`8c60a61`) — `TransformerConfig.bidirectional: bool`;
  `masks::create_bidirectional_mask` (cached all-zeros `[1,1,S,S]`); `mask_for_layer`
  short-circuits to it. Default `false` for every autoregressive family.
- **3b** (`399bab6`) — `from_hf_config` routes `model_type` `"Dream"` / `"a2d-qwen2"`
  (→ `parse_qwen2`) and `"a2d-qwen3"` (→ `parse_qwen3`) to `GenericTransformer` with
  `bidirectional = true`. The LM-head loader now prefers a materialized
  `lm_head.weight` even under `tie_word_embeddings` (A2D-converted checkpoints ship a
  separate head).
- **3c** (`87bb3a7`) — `check_auto_compatibility` short-circuits MDLM-shaped checkpoints
  (`backbone.*` / `adaLN`) with a single *"load with the `diffusion` feature"* hint
  instead of a wall of missing-tensor noise.
- **3d** (`5260d67`) — external fp32 oracle (`scripts/bidirectional_forward_validation.py`)
  on `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` (exact Qwen2.5 layout,
  fits 16 GB at F32). Loaded with stock `Qwen2ForCausalLM` + untied head + all-zeros 4D
  mask (the A2D model = Qwen2 + untied `lm_head` + bidirectional attention).
  **top-10 exact, max abs-diff 2.61×10⁻⁴ (CPU) / within 5×10⁻³ (GPU)**, checked at early
  positions where bidirectional ≠ causal. See `tests/validate_bidirectional_forward.rs`.

**Oracle choice:** the 0.5B `a2d-qwen2` validates the *same* bidirectional code path as
Dream-7B but fits 16 GB at F32 (tight parity), sidestepping the BF16-only caveat that
7B/8B models would impose. Its shipped `lm_head.weight` is value-identical to the
embeddings (max diff 0.0), so the tie question is moot.

## Stage 3e — LLaDA-8B (OLMo-style remap) — DEFERRED

LLaDA needs a weight-name remap subsystem (`model.transformer.blocks.{i}.{q,k,v}_proj`,
`attn_out`, `attn_norm`, `ff_proj`/`up_proj`/`ff_out`, `ff_norm`, `wte`/`ln_f`/top-level
`ff_out`) + OLMo→`TransformerConfig` translation (full MHA, no bias, SwiGLU, rms_eps 1e-5,
rope_theta 5e5, mask 126336), and is only checkable on CPU-F32 (8B) — a separable
follow-up. Sampling reuse is already free: the SUBS sampler takes `&dyn MIBackend` +
`mask_token_id`, so running the diffusion MI examples on Dream needs only the mask id
(151666) passed in — no backend change.

## Further out

- SAE / CLT / PLT training on diffusion activations (the DLM-Scope experiments
  that need a trained dictionary) — see the `askesis` reference-grade SAE-trainer
  decision; MDLM is the proving ground.
- SEDD (DiT-style) — likely loadable via `GenericMdlm` with a different output
  parameterization.
