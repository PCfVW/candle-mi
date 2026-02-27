// SPDX-License-Identifier: MIT OR Apache-2.0

//! # candle-mi
//!
//! Mechanistic interpretability for language models in Rust, built on
//! [candle](https://github.com/huggingface/candle).
//!
//! candle-mi re-implements model forward passes with built-in hook points
//! (following the `TransformerLens` design), enabling activation capture,
//! attention knockout, steering, logit lens, and sparse-feature analysis
//! (CLTs and SAEs) — all in pure Rust with GPU acceleration.
//!
//! ## Supported backends
//!
//! - **Generic Transformer** — covers `LLaMA`, `Qwen2`, Gemma 2, `Phi-3`,
//!   `StarCoder2`, Mistral, and more via configuration axes (feature:
//!   `transformer`).
//! - **Generic RWKV** (planned) — covers RWKV-6 and RWKV-7 linear RNN models
//!   (feature: `rwkv`).
//!
//! ## Fast downloads (optional)
//!
//! With the `fast-download` feature, candle-mi can download models from the
//! `HuggingFace` Hub with maximum throughput:
//!
//! ```toml
//! candle-mi = { version = "0.0.2", features = ["fast-download", "transformer"] }
//! ```
//!
//! ```rust,no_run
//! # #[cfg(feature = "fast-download")]
//! # async fn example() -> candle_mi::Result<()> {
//! // Pre-download with parallel chunks and progress via tracing
//! let path = candle_mi::download_model("meta-llama/Llama-3.2-1B".to_owned()).await?;
//!
//! // Load from cache (sync, no network needed)
//! let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use candle_mi::{HookPoint, HookSpec, MIModel};
//!
//! # fn main() -> candle_mi::Result<()> {
//! // Load a model (requires a concrete backend — Phase 1+).
//! let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//!
//! // Capture attention patterns at layer 5.
//! let mut hooks = HookSpec::new();
//! hooks.capture(HookPoint::AttnPattern(5));
//!
//! let tokens = candle_core::Tensor::zeros(
//!     (1, 10), candle_core::DType::U32, &candle_core::Device::Cpu,
//! )?;
//! let result = model.forward(&tokens, &hooks)?;
//! let attn = result.require(&HookPoint::AttnPattern(5))?;
//! # Ok(())
//! # }
//! ```

#![deny(warnings)] // All warns → errors in CI
#![cfg_attr(not(feature = "mmap"), forbid(unsafe_code))] // Rule 5: safe by default
#![cfg_attr(feature = "mmap", deny(unsafe_code))] // mmap: deny except one function

pub mod backend;
pub mod cache;
pub mod config;
#[cfg(feature = "fast-download")]
pub mod download;
pub mod error;
pub mod hooks;
pub mod interp;
pub mod tokenizer;
#[cfg(feature = "transformer")]
pub mod transformer;
pub mod util;

// --- Public re-exports ---------------------------------------------------

// Backend
pub use backend::{GenerationResult, MIBackend, MIModel};

// Config
pub use config::{
    Activation, MlpLayout, NormType, QkvLayout, SUPPORTED_MODEL_TYPES, TransformerConfig,
};

// Transformer backend
#[cfg(feature = "transformer")]
pub use transformer::GenericTransformer;

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
    SteeringSpec,
};

// Interpretability — logit lens
pub use interp::logit_lens::{LogitLensAnalysis, LogitLensResult, TokenPrediction};

// Interpretability — steering calibration
pub use interp::steering::{DoseResponseCurve, DoseResponsePoint, SteeringCalibration};

// Utility — masks
pub use util::masks::{create_causal_mask, create_generation_mask};

// Utility — positioning
pub use util::positioning::{EncodingWithOffsets, PositionConversion, TokenWithOffset};

// Tokenizer
pub use tokenizer::MITokenizer;

// Download (fast-download feature)
#[cfg(feature = "fast-download")]
pub use download::{download_model, download_model_blocking};
