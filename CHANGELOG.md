# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **RWKV-6 (Finch) backend** — `RwkvConfig` with V6/V7 version dispatch,
  `GenericRwkv` struct implementing `MIBackend`, WKV-5/6 recurrence kernel,
  `TimeMixV6`/`ChannelMixV6` blocks, `RwkvState` and `RwkvDecay` hook
  points for mechanistic interpretability of recurrent state dynamics
- **RWKV-7 (Goose) backend** — WKV-7 kernel with generalized delta rule
  (`S_t = diag(exp(w)) * S + b^T(a @ S) + k^T v`), `TimeMixV7`/`ChannelMixV7`
  blocks, `LoraBlock` with tanh/sigmoid/identity middle activations, value
  residual mixing across layers, gate output correction, L2-norm key
  normalization, and plain squared-ReLU FFN (no receptance gate)
- `hf-fetch-model` integration for parallel multi-connection model downloads,
  replacing `hf-hub` v0.4 as the sole download backend; `from_pretrained()`
  and `resolve_safetensors_paths()` now use `hf-fetch-model` directly
- `download_model()` (async) and `download_model_blocking()` convenience
  functions that populate the standard HF cache
- `SUPPORTED_MODEL_TYPES` const for runtime model-type discovery
- `quick_start_transformer` and `fast_download` examples
- Python validation scripts (`scripts/rwkv6_validation.py`,
  `scripts/rwkv7_validation.py`) for reproducible reference output generation
- Integration tests for RWKV-6 (against plip-rs reference) and RWKV-7
  (against fla/flash-linear-attention reference), CPU F32 + GPU BF16
- RWKV clippy and test steps in CI publish workflow
- VRAM budget table and `config.json` field reference in rustdoc
- `MIError::Download` variant for download failures

### Fixed

- RWKV-7 `g_lora` sigmoid placement: sigmoid is the **middle** activation
  (between down and up projections), not applied after the full LoRA output;
  `down(x) -> sigmoid -> up` vs the incorrect `down(x) -> up -> sigmoid`
- Serialized GPU integration tests with `serial_test` to prevent CUDA OOM
  when running multiple model tests concurrently
- Pre-existing `cargo doc` link warnings resolved

### Changed

- Dropped `hf-hub` v0.4 dependency; all HuggingFace file resolution now
  goes through `hf-fetch-model` (parallel chunked downloads by default)
- `#[must_use]` policy applied across public API (Rule 17)
- Phase 1 audit remediation (code quality, documentation, consistency)

## [0.0.2-phase1] - 2026-02-25

### Added

- **Generic Transformer backend** — one config-driven forward pass covering
  6 model families: LLaMA, Qwen2, Gemma 2, Phi-3, StarCoder2, Mistral
- `TransformerConfig` with ~12 configuration axes parsed from HuggingFace
  `config.json` (norm type, activation, QKV layout, MLP layout, bias
  granularity, embedding scale, soft-capping, sliding window, etc.)
- Config parsers for `llama`, `qwen2`, `gemma`, `gemma2`, `phi3`,
  `starcoder2`, `mistral` — adding a new model family requires only a
  ~30-line parser function
- `GenericTransformer` struct implementing `MIBackend` with hook points
  at all 14 TransformerLens-equivalent locations (Embed, ResidPre, AttnQ/K/V,
  AttnScores, AttnPattern, AttnOut, ResidMid, MlpPre/Post, MlpOut,
  ResidPost, FinalNorm)
- Multi-head attention supporting GQA/MHA/MQA, separate and fused QKV
  projections, optional soft-capping, and sliding window (global,
  per-layer, or alternating)
- MLP variants: gated separate (LLaMA/Qwen/Gemma), gated fused (Phi-3),
  and plain (StarCoder2)
- Normalization: RmsNorm, LayerNorm, GemmaRmsNorm (weight + 1)
- RoPE via `candle_nn::rotary_emb::rope()` with pre-computed cos/sin cache
- `MIModel::from_pretrained(model_id)` for HuggingFace model loading
  with automatic config detection and sharded safetensors support
- `mmap` feature gate: `#![forbid(unsafe_code)]` by default, opt-in
  memory-mapped weight loading for 7B+ models (`features = ["mmap"]`)
- `Activation::GeluApprox` for PyTorch tanh-approximated GELU
  (`gelu_pytorch_tanh`)
- `AttentionCache` for per-layer attention pattern storage
- Integration tests validating all 6 model families on CPU (F32) and
  GPU (BF16) against Python HuggingFace reference outputs
- Hook overhead benchmark: +11.5% on GPU with full capture (194 hook
  points on LLaMA 3.2 1B), within noise on CPU

### Fixed

- Tokenizer `encode()` now adds special tokens (BOS) by default,
  matching HuggingFace convention; added `encode_raw()` for MI analyses
  needing raw tokenization
- StarCoder2 config now reads `norm_type` from `config.json` (LayerNorm,
  not RmsNorm) and uses `GeluApprox` activation

### Changed

- Clarified that plip-rs is a frozen predecessor project (v1.4.0) in
  `MIBackend` trait documentation

## [0.0.1] - 2026-02-23

### Added

- `MIError` typed error hierarchy with `thiserror` (`#[non_exhaustive]`)
- `MIBackend` trait and `MIModel` wrapper for dynamic dispatch over model backends
- `HookSpec`, `HookCache`, and `HookPoint` for activation capture and intervention
- `KVCache` and `ActivationCache` for inference state management
- `KnockoutSpec`, `SteeringSpec`, `StateKnockoutSpec`, `StateSteeringSpec` for interpretability interventions
- `CltInjectionSpec` for CLT feature injection (behind `clt` feature flag)
- `LogitLensAnalysis` and `SteeringCalibration` with dose-response curves
- `MITokenizer` enum supporting `HuggingFace` and RWKV World tokenizers
- Causal mask and generation mask utilities
- Token-to-character position mapping
- CI workflow (fmt, clippy pedantic, tests, feature-flag hygiene)
- Tag-triggered publish workflow with `workflow_dispatch` fallback

[0.0.2-phase1]: https://github.com/PCfVW/candle-mi/releases/tag/v0.0.2-phase1
[0.0.1]: https://github.com/PCfVW/candle-mi/releases/tag/v0.0.1
