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
//! - **Generic RWKV** — covers RWKV-6 and RWKV-7 linear RNN models
//!   (feature: `rwkv`).
//!
//! ## Quick start
//!
//! ```ignore
//! use candle_mi::{HookPoint, HookSpec, MIModel};
//!
//! // Load a model (requires a concrete backend — Phase 1+).
//! let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//!
//! // Capture attention patterns at layer 5.
//! let mut hooks = HookSpec::new();
//! hooks.capture(HookPoint::AttnPattern(5));
//!
//! let result = model.forward(&tokens, &hooks)?;
//! let attn = result.require(&HookPoint::AttnPattern(5))?;
//! ```

#![deny(warnings)] // All warns → errors in CI
#![forbid(unsafe_code)] // Rule 5
#![deny(elided_lifetimes_in_paths)] // Rule 1
#![deny(clippy::unwrap_used)] // Rule 3
#![deny(clippy::expect_used)] // Rule 3
#![deny(clippy::panic)] // Rule 3
#![deny(clippy::indexing_slicing)] // Rule 3
#![deny(clippy::wildcard_enum_match_arm)] // Rule 7
#![deny(clippy::match_wildcard_for_single_variants)] // Rule 7
#![warn(clippy::exhaustive_enums)] // Rule 11
#![warn(clippy::as_conversions)] // Rule 2
#![warn(clippy::cast_possible_truncation)] // Rule 2
#![warn(clippy::cast_precision_loss)] // Rule 2
#![warn(clippy::cast_sign_loss)] // Rule 2
#![warn(clippy::pedantic)] // General quality
#![warn(clippy::nursery)] // General quality
#![warn(missing_docs)] // Rule 12 prerequisite
#![warn(clippy::missing_docs_in_private_items)] // Document internal helpers
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::must_use_candidate)]

pub mod backend;
pub mod cache;
pub mod error;
pub mod hooks;
pub mod interp;
pub mod tokenizer;
pub mod util;

// --- Public re-exports ---------------------------------------------------

// Backend
pub use backend::{GenerationResult, MIBackend, MIModel};

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
