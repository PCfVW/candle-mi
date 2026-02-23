// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activation and KV caching for efficient forward passes.
//!
//! - [`ActivationCache`] — per-layer last-token residual stream activations.
//! - [`FullActivationCache`] — all-position residual stream activations.
//! - [`KVCache`] — key/value cache for autoregressive generation.

mod activation;
mod kv;

pub use activation::{ActivationCache, FullActivationCache};
pub use kv::KVCache;
