// SPDX-License-Identifier: MIT OR Apache-2.0

//! RWKV gated-linear RNN backend.
//!
//! Covers RWKV-6 (Finch) and will cover RWKV-7 (Goose) linear RNN models.
//! Architecture differences are captured in [`RwkvConfig`] fields and
//! version-specific submodules.

pub mod config;

pub use config::{RwkvConfig, RwkvLoraDims, RwkvVersion, SUPPORTED_RWKV_MODEL_TYPES};
