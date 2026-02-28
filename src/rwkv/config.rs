// SPDX-License-Identifier: MIT OR Apache-2.0

//! RWKV configuration and `HuggingFace` `config.json` parsing.
//!
//! [`RwkvConfig`] captures the configuration axes that distinguish RWKV
//! versions (v6 Finch, v7 Goose).  Parsed from `HuggingFace` `config.json`
//! via [`from_hf_config`](RwkvConfig::from_hf_config).
//!
//! # Usage
//!
//! ```
//! use candle_mi::rwkv::RwkvConfig;
//!
//! let config_str = r#"{"model_type": "rwkv6", "hidden_size": 2048,
//!     "num_hidden_layers": 24, "num_attention_heads": 64,
//!     "vocab_size": 65536}"#;
//! let json: serde_json::Value = serde_json::from_str(config_str).unwrap();
//! let config = RwkvConfig::from_hf_config(&json).unwrap();
//! assert_eq!(config.num_layers, 24);
//! assert_eq!(config.num_heads, 32);
//! ```

use std::fmt;

use serde_json::Value;

use crate::config::{get_bool_or, get_f64_or, get_usize, get_usize_or};
use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// Supported RWKV model types
// ---------------------------------------------------------------------------

/// `model_type` strings accepted by [`RwkvConfig::from_hf_config`].
pub const SUPPORTED_RWKV_MODEL_TYPES: &[&str] = &["rwkv6"];

// ---------------------------------------------------------------------------
// RwkvVersion
// ---------------------------------------------------------------------------

/// RWKV architecture version.
///
/// Determines the recurrence formula (WKV kernel), token shift mechanism,
/// and channel mix variant.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RwkvVersion {
    /// RWKV-6 "Finch": data-dependent decay via `LoRA`, receptance-gated FFN.
    V6,
    /// RWKV-7 "Goose": generalized delta rule (`diag` + rank-1 state transition).
    V7,
}

impl fmt::Display for RwkvVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V6 => write!(f, "RWKV-6 (Finch)"),
            Self::V7 => write!(f, "RWKV-7 (Goose)"),
        }
    }
}

// ---------------------------------------------------------------------------
// RwkvLoraDims
// ---------------------------------------------------------------------------

/// Low-rank projection dimensions used in RWKV data-dependent mixing.
///
/// For RWKV-6 these are hardcoded (not in `config.json`).
/// For RWKV-7 they appear explicitly in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RwkvLoraDims {
    /// Dimension for the 5-component time-mix `LoRA` projection.
    /// RWKV-6: 32 (hardcoded).
    pub time_mix_extra_dim: usize,
    /// Dimension for the time-decay `LoRA` projection.
    /// RWKV-6: 64 (hardcoded).
    pub time_decay_extra_dim: usize,
}

// ---------------------------------------------------------------------------
// RwkvConfig
// ---------------------------------------------------------------------------

/// Configuration for a generic RWKV gated-linear RNN model.
///
/// Parsed from `HuggingFace` `config.json` via
/// [`from_hf_config`](Self::from_hf_config).
///
/// # Supported versions
///
/// | Version | Key config traits |
/// |---------|------------------|
/// | RWKV-6 (Finch) | Data-dependent decay (`LoRA`), receptance-gated FFN |
/// | RWKV-7 (Goose) | Generalized delta rule, plain FFN (planned) |
///
/// # `config.json` field reference (RWKV-6)
///
/// | Field | `config.json` key | Notes |
/// |-------|-------------------|-------|
/// | `hidden_size` | `hidden_size` | |
/// | `num_layers` | `num_hidden_layers` | |
/// | `head_dim` | `num_attention_heads` | Confusingly named: this is `head_size`, not head count |
/// | `vocab_size` | `vocab_size` | |
/// | `norm_eps` | `layer_norm_epsilon` | Default: 1e-5 |
/// | `head_size_divisor` | `head_size_divisor` | Default: 8; scales GroupNorm eps |
/// | `rescale_every` | `rescale_every` | Default: 6 |
/// | `intermediate_size` | `intermediate_size` | Default: `(hidden * 7/2) / 32 * 32` |
/// | `tie_word_embeddings` | `tie_word_embeddings` | Default: false |
#[derive(Debug, Clone)]
pub struct RwkvConfig {
    /// Architecture version.
    pub version: RwkvVersion,

    // --- Dimensions ----------------------------------------------------------
    /// Hidden dimension (`d_model`).
    pub hidden_size: usize,
    /// Number of RWKV blocks (layers).
    pub num_layers: usize,
    /// Per-head dimension (typically 64 for all current models).
    pub head_dim: usize,
    /// Number of heads (`hidden_size / head_dim`).
    pub num_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,

    // --- Normalization -------------------------------------------------------
    /// Epsilon for `LayerNorm` layers.
    pub norm_eps: f64,

    // --- FFN -----------------------------------------------------------------
    /// MLP intermediate dimension.
    ///
    /// RWKV-6: computed as `(hidden_size * 7/2) / 32 * 32` if not in config.
    /// RWKV-7: explicit in config (`hidden_size * hidden_ratio`).
    pub intermediate_size: usize,

    // --- Version-specific ----------------------------------------------------
    /// Rescale hidden states every N layers (v6: 6, v7: `None`).
    pub rescale_every: Option<usize>,
    /// Head-size divisor for `GroupNorm` epsilon scaling (v6: 8, v7: `None`).
    ///
    /// `GroupNorm` eps = `norm_eps * head_size_divisor^2`.
    pub head_size_divisor: Option<usize>,
    /// Low-rank projection dimensions for data-dependent mixing.
    pub lora_dims: RwkvLoraDims,
    /// Hidden ratio for FFN (v7 only; v6 uses the implicit 3.5x formula).
    pub hidden_ratio: Option<f64>,

    // --- Embeddings ----------------------------------------------------------
    /// Whether the LM head shares weights with the token embedding.
    pub tie_word_embeddings: bool,
}

impl RwkvConfig {
    /// Parse an [`RwkvConfig`] from a `HuggingFace` `config.json` value.
    ///
    /// Dispatches on the `model_type` field to a version-specific parser.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `model_type` is missing, unsupported,
    /// or if required fields are absent.
    pub fn from_hf_config(config: &Value) -> Result<Self> {
        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| MIError::Config("missing 'model_type' field".into()))?;

        match model_type {
            "rwkv6" => Self::parse_rwkv6(config),
            "rwkv7" => Err(MIError::Config(
                "RWKV-7 support is planned but not yet implemented".into(),
            )),
            other => Err(MIError::Config(format!(
                "unsupported RWKV model_type: '{other}'"
            ))),
        }
    }

    /// Compute the `GroupNorm` epsilon: `norm_eps * head_size_divisor^2`.
    ///
    /// Falls back to `norm_eps` if `head_size_divisor` is `None`.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    pub fn group_norm_eps(&self) -> f64 {
        self.head_size_divisor
            .map_or(self.norm_eps, |d| self.norm_eps * (d as f64).powi(2))
    }
}

// ---------------------------------------------------------------------------
// Per-version config parsers
// ---------------------------------------------------------------------------

impl RwkvConfig {
    /// Parse an RWKV-6 "Finch" config.
    ///
    /// # Notes
    ///
    /// The `HuggingFace` RWKV-6 config uses `num_attention_heads` to store
    /// what is actually `head_size` (the per-head dimension, typically 64),
    /// **not** the number of heads.  The actual head count is computed as
    /// `hidden_size / head_size`.
    ///
    /// The `LoRA` dimensions (`time_mix_extra_dim` = 32, `time_decay_extra_dim` = 64)
    /// are hardcoded and do not appear in `config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_rwkv6(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;

        // HF config confusingly stores head_size in "num_attention_heads".
        let head_dim = get_usize(config, "num_attention_heads")?;

        if head_dim == 0 {
            return Err(MIError::Config(
                "head_dim (num_attention_heads) is 0".into(),
            ));
        }
        let num_heads = hidden_size / head_dim;

        // intermediate_size: explicit in config, or computed as (hidden * 3.5) rounded to 32
        let intermediate_size =
            get_usize_or(config, "intermediate_size", (hidden_size * 7 / 2) / 32 * 32);

        Ok(Self {
            version: RwkvVersion::V6,
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            head_dim,
            num_heads,
            vocab_size: get_usize(config, "vocab_size")?,
            norm_eps: get_f64_or(config, "layer_norm_epsilon", 1e-5),
            intermediate_size,
            rescale_every: Some(get_usize_or(config, "rescale_every", 6)),
            head_size_divisor: Some(get_usize_or(config, "head_size_divisor", 8)),
            lora_dims: RwkvLoraDims {
                time_mix_extra_dim: 32,
                time_decay_extra_dim: 64,
            },
            hidden_ratio: None,
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", false),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Sample RWKV-6 config matching RWKV/v6-Finch-1B6-HF.
    fn rwkv6_config_json() -> Value {
        serde_json::json!({
            "model_type": "rwkv6",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "vocab_size": 65536,
            "layer_norm_epsilon": 1e-5,
            "head_size_divisor": 8,
            "rescale_every": 6,
            "tie_word_embeddings": false
        })
    }

    #[test]
    fn parse_rwkv6_basic() {
        let config = RwkvConfig::from_hf_config(&rwkv6_config_json()).unwrap();
        assert_eq!(config.version, RwkvVersion::V6);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.vocab_size, 65536);
        assert!((config.norm_eps - 1e-5).abs() < f64::EPSILON);
        // intermediate_size = (2048 * 7/2) / 32 * 32 = 7168 / 32 * 32 = 7168
        assert_eq!(config.intermediate_size, 7168);
        assert_eq!(config.rescale_every, Some(6));
        assert_eq!(config.head_size_divisor, Some(8));
        assert_eq!(config.lora_dims.time_mix_extra_dim, 32);
        assert_eq!(config.lora_dims.time_decay_extra_dim, 64);
        assert!(config.hidden_ratio.is_none());
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn rwkv6_group_norm_eps() {
        let config = RwkvConfig::from_hf_config(&rwkv6_config_json()).unwrap();
        // GroupNorm eps = 1e-5 * 8^2 = 1e-5 * 64 = 6.4e-4
        let expected = 1e-5 * 64.0;
        assert!((config.group_norm_eps() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn rwkv6_explicit_intermediate_size() {
        let json = serde_json::json!({
            "model_type": "rwkv6",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "vocab_size": 65536,
            "intermediate_size": 8192
        });
        let config = RwkvConfig::from_hf_config(&json).unwrap();
        assert_eq!(config.intermediate_size, 8192);
    }

    #[test]
    fn rwkv7_not_yet_implemented() {
        let json = serde_json::json!({
            "model_type": "rwkv7",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "vocab_size": 65536
        });
        let result = RwkvConfig::from_hf_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_model_type_errors() {
        let json = serde_json::json!({ "model_type": "gpt2" });
        let result = RwkvConfig::from_hf_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn missing_model_type_errors() {
        let json = serde_json::json!({ "hidden_size": 2048 });
        let result = RwkvConfig::from_hf_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn version_display() {
        assert_eq!(RwkvVersion::V6.to_string(), "RWKV-6 (Finch)");
        assert_eq!(RwkvVersion::V7.to_string(), "RWKV-7 (Goose)");
    }
}
