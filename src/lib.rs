// SPDX-License-Identifier: MIT OR Apache-2.0

//! # candle-mi
//!
//! Mechanistic interpretability for language models in Rust, built on
//! [candle](https://github.com/huggingface/candle).
//!
//! candle-mi re-implements model forward passes with built-in hook points
//! (following the [`TransformerLens`](https://github.com/TransformerLensOrg/TransformerLens)
//! design), enabling activation capture, attention knockout, steering, logit
//! lens, and sparse-feature analysis (CLTs and SAEs) — all in pure Rust with
//! GPU acceleration.
//!
//! ## Supported backends
//!
//! | Backend | Models | Feature flag |
//! |---------|--------|-------------|
//! | [`GenericTransformer`] | `LLaMA`, `Qwen2`, `Qwen3`, Gemma, Gemma 2, `Phi-3`, `StarCoder2`, Mistral; bidirectional masked-diffusion decoders (Dream, `a2d-qwen2`, `a2d-qwen3`); + auto-config for unknown families | `transformer` |
//! | `GenericRwkv` | RWKV-6 (Finch), RWKV-7 (Goose) | `rwkv` |
//! | `GenericMdlm` | MDLM masked-diffusion `DiT` (bidirectional) | `diffusion` |
//! | `OthelloGpt` | Plain GPT-2-style backbone (learned absolute positions, full `LayerNorm`, exact-`GELU`); the `OthelloMDLM` world model | `diffusion` |
//! | `StoicheiaRnn` / `StoicheiaTransformer` | `AlgZoo` `ReLU` RNN, attention-only transformer (8–1,408 params) | `stoicheia` |
//!
//! See [`BACKENDS.md`](https://github.com/PCfVW/candle-mi/blob/main/BACKENDS.md)
//! for how to add a new model architecture.
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `transformer` | yes | Generic transformer backend (decoder-only) |
//! | `cuda` | yes | CUDA GPU acceleration |
//! | `rwkv` | no | RWKV-6/7 linear RNN backend |
//! | `rwkv-tokenizer` | no | RWKV world tokenizer (required for RWKV inference) |
//! | `diffusion` | no | MDLM masked-diffusion backend (bidirectional `DiT`; standalone) |
//! | `clt` | no | Cross-Layer Transcoder support |
//! | `sae` | no | Sparse Autoencoder support (NPZ via `anamnesis`) |
//! | `quantized` | no | Load quantized checkpoints (bitsandbytes `NF4`/`FP4`/`INT8`, `AWQ`, `GPTQ`) by dequantizing to `BF16` via `anamnesis` |
//! | `mmap` | no | Memory-mapped weight loading (required for sharded models) |
//! | `memory` | no | RAM/VRAM reporting (delegated to the `hypomnesis` crate) |
//! | `memory-debug` | no | Raw GPU-backend measurement values on stderr (via `hypomnesis`; implies `memory`) |
//! | `stoicheia` | no | `AlgZoo` tiny-model backends + MI analysis tools; agnostic `.safetensors`/`.pth` loading via `anamnesis` |
//! | `probing` | no | Linear probing via linfa (experimental) |
//! | `metal` | no | Apple Metal GPU acceleration |
//!
//! ## Quick start
//!
//! Load a model, run a forward pass, and inspect the output:
//!
//! ```no_run
//! use candle_mi::{HookSpec, MIModel};
//!
//! # fn main() -> candle_mi::Result<()> {
//! let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! let tokenizer = model.tokenizer().unwrap();
//!
//! let tokens = tokenizer.encode("The capital of France is")?;
//! let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//!
//! let cache = model.forward(&input, &HookSpec::new())?;
//! let logits = cache.output();  // [1, seq, vocab]
//!
//! let last_logits = logits.get(0)?.get(tokens.len() - 1)?;
//! let token_id = candle_mi::sample_token(&last_logits, 0.0)?;  // greedy
//! println!("{}", tokenizer.decode(&[token_id])?);  // " Paris"
//! # Ok(())
//! # }
//! ```
//!
//! ## Activation capture
//!
//! Use [`HookSpec::capture`] to snapshot tensors at any
//! [`HookPoint`] during the forward pass:
//!
//! ```no_run
//! use candle_mi::{HookPoint, HookSpec, MIModel};
//!
//! # fn main() -> candle_mi::Result<()> {
//! # let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # let tokenizer = model.tokenizer().unwrap();
//! # let tokens = tokenizer.encode("The capital of France is")?;
//! # let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//! let mut hooks = HookSpec::new();
//! hooks.capture(HookPoint::AttnPattern(5))       // post-softmax attention
//!      .capture(HookPoint::ResidPost(10));        // residual stream at layer 10
//!
//! let cache = model.forward(&input, &hooks)?;
//!
//! let attn = cache.require(&HookPoint::AttnPattern(5))?;   // [1, heads, seq, seq]
//! let resid = cache.require(&HookPoint::ResidPost(10))?;    // [1, seq, hidden]
//! # Ok(())
//! # }
//! ```
//!
//! ## Interventions
//!
//! Use [`HookSpec::intervene`] to modify activations mid-forward-pass.
//! Five intervention types are available: [`Intervention::Replace`],
//! [`Intervention::Add`], [`Intervention::Knockout`],
//! [`Intervention::Scale`], and [`Intervention::Zero`].
//!
//! ```no_run
//! use candle_mi::{HookPoint, HookSpec, Intervention, KnockoutSpec, create_knockout_mask};
//!
//! # fn main() -> candle_mi::Result<()> {
//! # let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # let tokenizer = model.tokenizer().unwrap();
//! # let tokens = tokenizer.encode("The capital of France is")?;
//! # let seq_len = tokens.len();
//! # let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//! // Knock out the attention edge: last token cannot attend to position 0
//! let spec = KnockoutSpec::new().layer(8).edge(seq_len - 1, 0);
//! let mask = create_knockout_mask(
//!     &spec, model.num_heads(), seq_len, model.device(), candle_core::DType::F32,
//! )?;
//!
//! let mut hooks = HookSpec::new();
//! hooks.intervene(HookPoint::AttnScores(8), Intervention::Knockout(mask));
//!
//! let ablated = model.forward(&input, &hooks)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Logit lens
//!
//! Project intermediate residual streams to vocabulary space using
//! [`MIModel::project_to_vocab`]:
//!
//! ```no_run
//! use candle_mi::{HookPoint, HookSpec, MIModel};
//!
//! # fn main() -> candle_mi::Result<()> {
//! # let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # let tokenizer = model.tokenizer().unwrap();
//! # let tokens = tokenizer.encode("The capital of France is")?;
//! # let seq_len = tokens.len();
//! # let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//! let mut hooks = HookSpec::new();
//! for layer in 0..model.num_layers() {
//!     hooks.capture(HookPoint::ResidPost(layer));
//! }
//! let cache = model.forward(&input, &hooks)?;
//!
//! for layer in 0..model.num_layers() {
//!     let resid = cache.require(&HookPoint::ResidPost(layer))?;
//!     let last = resid.get(0)?.get(seq_len - 1)?.unsqueeze(0)?;
//!     let logits = model.project_to_vocab(&last)?;
//!     let token_id = candle_mi::sample_token(&logits.flatten_all()?, 0.0)?;
//!     println!("Layer {layer:>2}: {}", tokenizer.decode(&[token_id])?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Fast downloads
//!
//! candle-mi uses [`hf-fetch-model`](https://github.com/PCfVW/hf-fetch-model)
//! for high-throughput parallel downloads from the `HuggingFace` Hub:
//!
//! ```rust,no_run
//! # async fn example() -> candle_mi::Result<()> {
//! // Async: parallel chunked download with progress bars
//! let _path = candle_mi::download_model("meta-llama/Llama-3.2-1B".to_owned()).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ```no_run
//! # fn main() -> candle_mi::Result<()> {
//! // Sync: blocking variant (uses local HF cache if already downloaded)
//! candle_mi::download_model_blocking("meta-llama/Llama-3.2-1B".to_owned())?;
//! let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Further reading
//!
//! - [`HOOKS.md`](https://github.com/PCfVW/candle-mi/blob/main/HOOKS.md) —
//!   complete hook point reference with shapes, intervention walkthrough, and
//!   worked examples.
//! - [`BACKENDS.md`](https://github.com/PCfVW/candle-mi/blob/main/BACKENDS.md) —
//!   how to add a new model architecture (auto-config, config parser, or
//!   custom `MIBackend`).
//! - [`examples/README.md`](https://github.com/PCfVW/candle-mi/blob/main/examples/README.md) —
//!   23 runnable examples covering inference, logit lens, attention patterns,
//!   knockout, steering, activation patching, `CounterFact` replication,
//!   CLT circuits, SAE encoding, RWKV inference, `AlgZoo` analysis, and more.

#![deny(warnings)]
// All warns → errors in CI
// Rule 5: safe by default. The only `unsafe` in the crate is the `mmap`
// safetensors loader and the CUDA pool-trim in `memory::sync_and_trim_gpu`
// (gated behind `cuda`). Since the hypomnesis migration, the `memory` feature
// carries NO unsafe on its own — all measurement FFI lives in hypomnesis — so
// `memory` without `cuda` stays `forbid(unsafe_code)`.
#![cfg_attr(
    not(any(feature = "mmap", all(feature = "memory", feature = "cuda"))),
    forbid(unsafe_code)
)]
// mmap, or memory+cuda (the pool-trim FFI): deny for scoped unsafe.
#![cfg_attr(
    any(feature = "mmap", all(feature = "memory", feature = "cuda")),
    deny(unsafe_code)
)]
// Test-code relaxations: the strict `unwrap_used`/`expect_used`/`indexing_slicing`/`panic`
// denies in `Cargo.toml` target *library* code (Rule 3). Inside `#[cfg(test)]` blocks,
// `unwrap()`, `expect("context")`, `assert_eq!(v[0], …)`, and `panic!` are the canonical
// Rust test idioms — failure-by-panic IS the test signal. Allow them only under
// `cfg(test)`; production builds remain strict.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::unreadable_literal,
        clippy::panic,
        clippy::wildcard_enum_match_arm,
        clippy::redundant_clone,
        clippy::needless_collect,
        clippy::format_push_string,
        clippy::doc_markdown,
        clippy::too_many_arguments,
        clippy::missing_const_for_fn,
    )
)]

pub mod backend;
pub mod cache;
#[cfg(feature = "clt")]
pub mod clt;
pub mod config;
#[cfg(feature = "diffusion")]
pub mod diffusion;
pub mod download;
pub mod error;
pub mod hooks;
pub mod interp;
#[cfg(feature = "memory")]
pub mod memory;
pub mod nn_ops;
#[cfg(feature = "rwkv")]
pub mod rwkv;
#[cfg(feature = "sae")]
pub mod sae;
pub mod sparse;
// Steering builders are only useful when a backend can apply their output; gate
// the module behind the same predicate as `hooks::apply_intervention` so builder
// and applier appear/disappear together. `sparse` stays ungated — it is shared
// data types (`FeatureId`/`SparseActivations`) consumed by the backend-independent
// `clt`/`sae` features, not a builder with a backend-gated applier.
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
pub mod steering;
#[cfg(feature = "stoicheia")]
pub mod stoicheia;
pub mod tokenizer;
#[cfg(feature = "transformer")]
pub mod transformer;
mod util;

/// Build-hygiene guard: asserts every `tests/*.rs` / `examples/*.rs` file is
/// registered in `Cargo.toml`. Test-only; see the module docs for rationale.
#[cfg(test)]
mod registration_guard;

// --- Public re-exports ---------------------------------------------------

// Backend
pub use backend::{
    GenerationResult, MIBackend, MIModel, TextForwardResult, extract_token_prob, sample_token,
};

// Config
pub use config::{
    Activation, CompatibilityReport, MlpLayout, NormType, QkvLayout, RopeScaling,
    SUPPORTED_MODEL_TYPES, TransformerConfig,
};

// Transformer backend
#[cfg(feature = "transformer")]
pub use transformer::GenericTransformer;

// Recurrent feedback (anacrousis)
#[cfg(feature = "transformer")]
pub use transformer::recurrent::{RecurrentFeedbackEntry, RecurrentPassSpec};

// RWKV backend
#[cfg(feature = "rwkv")]
pub use rwkv::{GenericRwkv, RwkvConfig, RwkvLoraDims, RwkvVersion};

// Diffusion backend (MDLM)
#[cfg(feature = "diffusion")]
pub use diffusion::{
    DiffusionSamplingConfig, GenericMdlm, MdlmConfig, OthelloGpt, OthelloGptConfig,
    SUPPORTED_DIFFUSION_MODEL_TYPES,
};

// Stoicheia (AlgZoo) backends — Phase A
#[cfg(feature = "stoicheia")]
pub use stoicheia::{
    StoicheiaArch, StoicheiaConfig, StoicheiaOutput, StoicheiaRnn, StoicheiaTask,
    StoicheiaTransformer,
};

// Stoicheia MI tooling — Phase B
#[cfg(feature = "stoicheia")]
pub use stoicheia::fast::RnnWeights;
#[cfg(feature = "stoicheia")]
pub use stoicheia::probing::NeuronRole;
#[cfg(feature = "stoicheia")]
pub use stoicheia::standardize::StandardizedRnn;

// Sparse feature types (shared by CLT and SAE)
pub use sparse::{FeatureId, SparseActivations};

// CLT (Cross-Layer Transcoder)
#[cfg(feature = "clt")]
pub use clt::{AttributionEdge, AttributionGraph, CltConfig, CltFeatureId, CrossLayerTranscoder};

// SAE (Sparse Autoencoder)
#[cfg(feature = "sae")]
pub use sae::{
    NormalizeActivations, SaeArchitecture, SaeConfig, SaeFeatureId, SparseAutoencoder, TopKStrategy,
};

// Cache
pub use cache::{ActivationCache, AttentionCache, FullActivationCache, KVCache};

// Error
pub use error::{MIError, Result};

// Hooks
pub use hooks::{HookCache, HookPoint, HookSpec, Intervention};

// Interpretability — intervention specs and results
pub use interp::intervention::{
    AblationResult, AttentionEdge, HeadSpec, InterventionType, KnockoutSpec, LayerSpec,
    StateAblationResult, StateKnockoutSpec, StateSteeringResult, StateSteeringSpec, SteeringResult,
    SteeringSpec, apply_steering, create_knockout_mask, kl_divergence,
    measure_attention_to_targets,
};

// Interpretability — logit lens
pub use interp::logit_lens::{LogitLensAnalysis, LogitLensResult, TokenPrediction};

// Interpretability — steering calibration
pub use interp::steering::{DoseResponseCurve, DoseResponsePoint, SteeringCalibration};

// Steering — contrastive activation steering (Maar et al. 2026)
// Gated with the `steering` module above (needs a backend to apply its output).
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
pub use steering::contrastive::{
    ContrastiveDirection, PositionStrategy, build_contrastive_direction, contrastive_intervention,
    position_delta,
};

// Utility — masks
pub use util::masks::{
    clear_mask_caches, create_bidirectional_mask, create_causal_mask, create_generation_mask,
};

// Utility — PCA
pub use util::pca::{PcaResult, pca_top_k};

// Utility — positioning
pub use util::positioning::{
    EncodingWithOffsets, PositionConversion, TokenWithOffset, convert_positions,
};

// Tokenizer
pub use tokenizer::MITokenizer;

// Memory reporting
#[cfg(feature = "memory")]
pub use memory::{MemoryReport, MemorySnapshot, sync_and_trim_gpu};

// Download
pub use download::{download_model, download_model_blocking, fetch_config_builder};
