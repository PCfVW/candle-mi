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

## Stage 3 — Decoder-style DLMs via a bidirectional `GenericTransformer` — AFTER Stage 2

- **3a** — add `bidirectional: bool` to `TransformerConfig`; `mask_for_layer`
  returns a zeros mask instead of the causal one (the trick `GenericMdlm` uses).
- **3b** — dispatch / auto-config wiring so decoder-style DLM `model_type`s run
  on `GenericTransformer` with `bidirectional = true`. Dream = Qwen2.5 (already
  auto-inferred); LLaDA's OLMo-ish naming (`transformer.blocks`/`ff_out`/`ln_f`)
  needs a small naming map — stretch.
- **3c** — defensive auto-compat error: detect MDLM-shaped checkpoints in
  `check_auto_compatibility` and emit *"enable the `diffusion` feature"* instead
  of a cryptic *"missing hidden_size"* when the `diffusion` feature is off.
- **3d** — Dream forward-parity oracle.

**Caveat:** Dream-7B / LLaDA-8B do not fit 16 GB at F32, so the GPU path runs
BF16 (looser parity than MDLM's 1.34×10⁻⁵); exact F32 survives only on CPU
(slow). This mirrors the existing Mistral-7B `#[ignore]` CPU-F32 + GPU pattern —
a known, manageable shape, but without MDLM's tight-everywhere guarantee.

## Further out

- SAE / CLT / PLT training on diffusion activations (the DLM-Scope experiments
  that need a trained dictionary) — see the `askesis` reference-grade SAE-trainer
  decision; MDLM is the proving ground.
- SEDD (DiT-style) — likely loadable via `GenericMdlm` with a different output
  parameterization.
