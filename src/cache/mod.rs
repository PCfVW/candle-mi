// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activation, attention, and KV caching for efficient forward passes.
//!
//! - [`ActivationCache`] — per-layer last-token residual stream activations.
//! - [`AttentionCache`] — per-layer post-softmax attention patterns.
//! - [`FullActivationCache`] — all-position residual stream activations.
//! - [`KVCache`] — key/value cache for autoregressive generation.

mod activation;
mod attention;
mod kv;

pub use activation::{ActivationCache, FullActivationCache};
pub use attention::AttentionCache;
pub use kv::KVCache;
