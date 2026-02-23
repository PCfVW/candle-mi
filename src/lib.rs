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

#![deny(warnings)]
#![warn(missing_docs)]

pub mod backend;
pub mod error;
pub mod hooks;

// --- Public re-exports ---------------------------------------------------

pub use backend::{GenerationResult, MIBackend, MIModel};
pub use error::{MIError, Result};
pub use hooks::{HookCache, HookPoint, HookSpec, Intervention};
