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

#![deny(warnings)]
#![warn(missing_docs)]
