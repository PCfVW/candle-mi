// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transformer configuration and `HuggingFace` `config.json` parsing.
//!
//! [`TransformerConfig`] captures the ~12 configuration axes that distinguish
//! modern decoder-only transformer architectures (`LLaMA`, `Qwen2`, Gemma 2,
//! `Phi-3`, `StarCoder2`, Mistral, etc.).  One forward pass implementation
//! covers all of them; adding a new model family requires only a new
//! `parse_*` function (~30 lines).
//!
//! # Usage
//!
//! ```
//! use candle_mi::TransformerConfig;
//!
//! let config_str = r#"{"model_type": "llama", "hidden_size": 2048,
//!     "num_hidden_layers": 16, "num_attention_heads": 32,
//!     "num_key_value_heads": 8, "intermediate_size": 8192,
//!     "vocab_size": 32000, "rms_norm_eps": 1e-5,
//!     "rope_theta": 500000.0, "max_position_embeddings": 131072}"#;
//! let json: serde_json::Value = serde_json::from_str(config_str).unwrap();
//! let config = TransformerConfig::from_hf_config(&json).unwrap();
//! assert_eq!(config.num_layers, 16);
//! ```

use std::fmt;

use serde_json::Value;

use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Layer normalization variant.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Standard RMS normalization: `x * weight / sqrt(mean(x^2) + eps)`.
    RmsNorm,
    /// Standard layer normalization (weight + bias).
    LayerNorm,
    /// Gemma-style RMS norm that adds `1.0` to the learned weight:
    /// `x * (weight + 1) / sqrt(mean(x^2) + eps)`.
    GemmaRmsNorm,
}

impl fmt::Display for NormType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RmsNorm => write!(f, "RmsNorm"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::GemmaRmsNorm => write!(f, "GemmaRmsNorm"),
        }
    }
}

/// Activation function used in the MLP.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Sigmoid Linear Unit (used in `SwiGLU` gating).
    Silu,
    /// Gaussian Error Linear Unit — exact (erf) variant.
    Gelu,
    /// Gaussian Error Linear Unit — PyTorch tanh approximation.
    ///
    /// Used by Gemma 2, `StarCoder2`, and other models that specify
    /// `hidden_act: "gelu_pytorch_tanh"` in their `HuggingFace` config.
    GeluApprox,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Silu => write!(f, "SiLU"),
            Self::Gelu => write!(f, "GELU"),
            Self::GeluApprox => write!(f, "GELU (tanh approx)"),
        }
    }
}

/// Layout of the Q, K, V projections in the attention block.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvLayout {
    /// Three separate linear layers: `q_proj`, `k_proj`, `v_proj`.
    Separate,
    /// Single fused linear layer `qkv_proj`, split via `narrow()`.
    Fused,
}

impl fmt::Display for QkvLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Separate => write!(f, "Separate"),
            Self::Fused => write!(f, "Fused"),
        }
    }
}

/// Layout of the MLP (feed-forward network).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpLayout {
    /// Gated MLP with separate gate and up projections:
    /// `down(act(gate(x)) * up(x))`.
    GatedSeparate,
    /// Gated MLP with fused gate+up projection:
    /// `gate_up = fused(x)`, split, then `down(act(gate) * up)`.
    GatedFused,
    /// Plain (non-gated) MLP: `proj(act(fc(x)))`.
    Plain,
}

impl fmt::Display for MlpLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GatedSeparate => write!(f, "GatedSeparate"),
            Self::GatedFused => write!(f, "GatedFused"),
            Self::Plain => write!(f, "Plain"),
        }
    }
}

// ---------------------------------------------------------------------------
// TransformerConfig
// ---------------------------------------------------------------------------

/// Configuration for a generic decoder-only transformer.
///
/// Captures ~12 configuration axes that distinguish modern transformer
/// architectures.  Parsed from `HuggingFace` `config.json` via
/// [`from_hf_config`](Self::from_hf_config).
///
/// # Supported model families
///
/// | Family | Key config traits |
/// |--------|------------------|
/// | `LLaMA` 1/2/3 | Baseline: GQA, `SiLU`, `RmsNorm` |
/// | `Qwen` 2/2.5 | + QKV bias, conditional tied embeddings |
/// | Gemma / Gemma 2 | + `GemmaRmsNorm`, embedding scale, soft-capping, 4-norm |
/// | `Phi-3` / `Phi-4` | + Fused QKV, fused MLP |
/// | `StarCoder2` | + Plain MLP, GELU, bias everywhere |
/// | Mistral | + Sliding window attention |
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Config structs legitimately have many boolean axes
pub struct TransformerConfig {
    // --- Dimensions ----------------------------------------------------------
    /// Hidden dimension (`d_model`).
    pub hidden_size: usize,
    /// Number of transformer layers (decoder blocks).
    pub num_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA when < `num_attention_heads`).
    pub num_kv_heads: usize,
    /// Dimension per head (usually `hidden_size / num_attention_heads`).
    pub head_dim: usize,
    /// MLP intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,

    // --- Architecture axes ---------------------------------------------------
    /// Normalization variant.
    pub norm_type: NormType,
    /// Epsilon for normalization layers.
    pub norm_eps: f64,
    /// MLP activation function.
    pub activation: Activation,
    /// QKV projection layout (separate or fused).
    pub qkv_layout: QkvLayout,
    /// MLP layout (gated separate, gated fused, or plain).
    pub mlp_layout: MlpLayout,
    /// Whether Q, K, V projections have bias terms.
    pub qkv_bias: bool,
    /// Whether the output projection (`o_proj`) has a bias term.
    pub o_proj_bias: bool,
    /// Whether MLP projections have bias terms.
    pub mlp_bias: bool,
    /// Embedding scale factor (`Some(sqrt(hidden_size))` for Gemma models).
    pub embedding_scale: Option<f64>,
    /// Whether the LM head shares weights with the token embedding.
    pub tie_word_embeddings: bool,

    // --- Positional encoding -------------------------------------------------
    /// Base frequency for rotary position embeddings.
    pub rope_theta: f64,
    /// Maximum sequence length for position embeddings.
    pub max_position_embeddings: usize,

    // --- Gemma 2 extensions --------------------------------------------------
    /// Attention logit soft-capping: `tanh(scores / cap) * cap` before softmax.
    /// `Some(50.0)` for Gemma 2; `None` for most models.
    pub attn_logit_softcapping: Option<f64>,
    /// Final logit soft-capping: `tanh(logits / cap) * cap` after LM head.
    /// `Some(30.0)` for Gemma 2; `None` for most models.
    pub final_logit_softcapping: Option<f64>,
    /// Custom attention scaling factor.  When set, scale = `1/sqrt(scalar)`
    /// instead of the default `1/sqrt(head_dim)`.
    /// `Some(256.0)` for Gemma 2; `None` for most models.
    pub query_pre_attn_scalar: Option<f64>,
    /// Whether each layer has post-attention and post-feedforward norms
    /// (4 norms per layer instead of 2).  `true` for Gemma 2.
    pub use_post_norms: bool,

    // --- Sliding window attention --------------------------------------------
    /// Sliding window size.  `None` for global attention.
    pub sliding_window: Option<usize>,
    /// Whether sliding window alternates with global attention per layer.
    /// When `true`, even layers (0, 2, 4, ...) use sliding window and
    /// odd layers use global causal.  `true` for Gemma 2.
    pub alternating_sliding_window: bool,
}

// ---------------------------------------------------------------------------
// Config parsing — entry point
// ---------------------------------------------------------------------------

impl TransformerConfig {
    /// Parse a [`TransformerConfig`] from a `HuggingFace` `config.json` value.
    ///
    /// Dispatches on the `model_type` field to a family-specific parser.
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
            "llama" => Self::parse_llama(config),
            "qwen2" => Self::parse_qwen2(config),
            "gemma" => Self::parse_gemma(config),
            "gemma2" => Self::parse_gemma2(config),
            "phi3" => Self::parse_phi3(config),
            "starcoder2" => Self::parse_starcoder2(config),
            "mistral" => Self::parse_mistral(config),
            other => Err(MIError::Config(format!(
                "unsupported model_type: '{other}'"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-family config parsers
// ---------------------------------------------------------------------------

impl TransformerConfig {
    /// Parse a `LLaMA`-family config (`LLaMA` 1/2/3, `Code-LLaMA`).
    ///
    /// Simplest baseline: no bias, no embedding scale, no sliding window,
    /// separate LM head (unless `tie_word_embeddings` is set).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_llama(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;

        Ok(Self {
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_kv_heads: get_usize_or(config, "num_key_value_heads", num_attention_heads),
            head_dim: get_head_dim(config, hidden_size, num_attention_heads)?,
            intermediate_size: get_usize(config, "intermediate_size")?,
            vocab_size: get_usize(config, "vocab_size")?,

            norm_type: NormType::RmsNorm,
            norm_eps: get_f64_or(config, "rms_norm_eps", 1e-5),
            activation: Activation::Silu,
            qkv_layout: QkvLayout::Separate,
            mlp_layout: MlpLayout::GatedSeparate,
            qkv_bias: false,
            o_proj_bias: false,
            mlp_bias: false,
            embedding_scale: None,
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", false),

            rope_theta: get_f64_or(config, "rope_theta", 10_000.0),
            max_position_embeddings: get_usize_or(config, "max_position_embeddings", 4096),

            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            use_post_norms: false,
            sliding_window: None,
            alternating_sliding_window: false,
        })
    }

    /// Parse a Qwen2/Qwen2.5 config.
    ///
    /// Adds QKV bias and conditional tied embeddings on top of the
    /// `LLaMA` baseline.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_qwen2(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;

        Ok(Self {
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_kv_heads: get_usize_or(config, "num_key_value_heads", num_attention_heads),
            head_dim: get_head_dim(config, hidden_size, num_attention_heads)?,
            intermediate_size: get_usize(config, "intermediate_size")?,
            vocab_size: get_usize(config, "vocab_size")?,

            norm_type: NormType::RmsNorm,
            norm_eps: get_f64_or(config, "rms_norm_eps", 1e-6),
            activation: Activation::Silu,
            qkv_layout: QkvLayout::Separate,
            mlp_layout: MlpLayout::GatedSeparate,
            qkv_bias: get_bool_or(config, "attention_bias", true),
            o_proj_bias: false,
            mlp_bias: false,
            embedding_scale: None,
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", false),

            rope_theta: get_f64_or(config, "rope_theta", 1_000_000.0),
            max_position_embeddings: get_usize_or(config, "max_position_embeddings", 32_768),

            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            use_post_norms: false,
            sliding_window: None,
            alternating_sliding_window: false,
        })
    }

    /// Parse a Gemma config (Gemma 1, `CodeGemma`).
    ///
    /// Adds `GemmaRmsNorm` (weight + 1), sqrt embedding scale, and GELU.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_gemma(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;

        Ok(Self {
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_kv_heads: get_usize_or(config, "num_key_value_heads", num_attention_heads),
            head_dim: get_head_dim(config, hidden_size, num_attention_heads)?,
            intermediate_size: get_usize(config, "intermediate_size")?,
            vocab_size: get_usize(config, "vocab_size")?,

            norm_type: NormType::GemmaRmsNorm,
            norm_eps: get_f64_or(config, "rms_norm_eps", 1e-6),
            activation: Activation::GeluApprox,
            qkv_layout: QkvLayout::Separate,
            mlp_layout: MlpLayout::GatedSeparate,
            qkv_bias: false,
            o_proj_bias: false,
            mlp_bias: false,
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            // PROMOTE: embedding scale is sqrt(hidden_size); precision loss negligible for d_model <= 2^52
            embedding_scale: Some((hidden_size as f64).sqrt()),
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", true),

            rope_theta: get_f64_or(config, "rope_theta", 10_000.0),
            max_position_embeddings: get_usize_or(
                config,
                "max_position_embeddings",
                8192,
            ),

            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            use_post_norms: false,
            sliding_window: None,
            alternating_sliding_window: false,
        })
    }

    /// Parse a Gemma 2 config.
    ///
    /// Adds attention/final logit soft-capping, 4-norm layers,
    /// `query_pre_attn_scalar`, and alternating sliding window attention.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_gemma2(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;

        Ok(Self {
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_kv_heads: get_usize_or(config, "num_key_value_heads", num_attention_heads),
            head_dim: get_head_dim(config, hidden_size, num_attention_heads)?,
            intermediate_size: get_usize(config, "intermediate_size")?,
            vocab_size: get_usize(config, "vocab_size")?,

            norm_type: NormType::GemmaRmsNorm,
            norm_eps: get_f64_or(config, "rms_norm_eps", 1e-6),
            activation: Activation::GeluApprox,
            qkv_layout: QkvLayout::Separate,
            mlp_layout: MlpLayout::GatedSeparate,
            qkv_bias: false,
            o_proj_bias: false,
            mlp_bias: false,
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            // PROMOTE: embedding scale is sqrt(hidden_size); precision loss negligible for d_model <= 2^52
            embedding_scale: Some((hidden_size as f64).sqrt()),
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", true),

            rope_theta: get_f64_or(config, "rope_theta", 10_000.0),
            max_position_embeddings: get_usize_or(
                config,
                "max_position_embeddings",
                8192,
            ),

            attn_logit_softcapping: get_optional_f64(config, "attn_logit_softcapping"),
            final_logit_softcapping: get_optional_f64(config, "final_logit_softcapping"),
            query_pre_attn_scalar: get_optional_f64(config, "query_pre_attn_scalar")
                .or(Some(256.0)),
            use_post_norms: true,
            sliding_window: get_optional_usize(config, "sliding_window"),
            alternating_sliding_window: true,
        })
    }

    /// Parse a Phi-3 config.
    ///
    /// Adds fused QKV projection and fused gate+up MLP projection.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_phi3(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;

        Ok(Self {
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_kv_heads: get_usize_or(config, "num_key_value_heads", num_attention_heads),
            head_dim: get_head_dim(config, hidden_size, num_attention_heads)?,
            intermediate_size: get_usize(config, "intermediate_size")?,
            vocab_size: get_usize(config, "vocab_size")?,

            norm_type: NormType::RmsNorm,
            norm_eps: get_f64_or(config, "rms_norm_eps", 1e-5),
            activation: Activation::Silu,
            qkv_layout: QkvLayout::Fused,
            mlp_layout: MlpLayout::GatedFused,
            qkv_bias: false,
            o_proj_bias: false,
            mlp_bias: false,
            embedding_scale: None,
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", false),

            rope_theta: get_f64_or(config, "rope_theta", 10_000.0),
            max_position_embeddings: get_usize_or(config, "max_position_embeddings", 4096),

            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            use_post_norms: false,
            sliding_window: None,
            alternating_sliding_window: false,
        })
    }

    /// Parse a `StarCoder2` config.
    ///
    /// Adds plain (non-gated) MLP, GELU activation, and bias on all
    /// projections.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_starcoder2(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;
        let use_bias = get_bool_or(config, "use_bias", true);

        // StarCoder2 specifies norm_type in config (usually "layer_norm").
        let norm_type = match config.get("norm_type").and_then(Value::as_str) {
            Some("layer_norm") => NormType::LayerNorm,
            _ => NormType::RmsNorm,
        };

        Ok(Self {
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_kv_heads: get_usize_or(config, "num_key_value_heads", num_attention_heads),
            head_dim: get_head_dim(config, hidden_size, num_attention_heads)?,
            intermediate_size: get_usize(config, "intermediate_size")?,
            vocab_size: get_usize(config, "vocab_size")?,

            norm_type,
            norm_eps: get_f64_or(config, "norm_epsilon", 1e-5),
            activation: Activation::GeluApprox,
            qkv_layout: QkvLayout::Separate,
            mlp_layout: MlpLayout::Plain,
            qkv_bias: use_bias,
            o_proj_bias: use_bias,
            mlp_bias: use_bias,
            embedding_scale: None,
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", true),

            rope_theta: get_f64_or(config, "rope_theta", 10_000.0),
            max_position_embeddings: get_usize_or(config, "max_position_embeddings", 16_384),

            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            use_post_norms: false,
            sliding_window: get_optional_usize(config, "sliding_window"),
            alternating_sliding_window: false,
        })
    }

    /// Parse a Mistral config.
    ///
    /// LLaMA-like with sliding window attention on all layers.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_mistral(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;

        Ok(Self {
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_kv_heads: get_usize_or(config, "num_key_value_heads", num_attention_heads),
            head_dim: get_head_dim(config, hidden_size, num_attention_heads)?,
            intermediate_size: get_usize(config, "intermediate_size")?,
            vocab_size: get_usize(config, "vocab_size")?,

            norm_type: NormType::RmsNorm,
            norm_eps: get_f64_or(config, "rms_norm_eps", 1e-5),
            activation: Activation::Silu,
            qkv_layout: QkvLayout::Separate,
            mlp_layout: MlpLayout::GatedSeparate,
            qkv_bias: false,
            o_proj_bias: false,
            mlp_bias: false,
            embedding_scale: None,
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", false),

            rope_theta: get_f64_or(config, "rope_theta", 10_000.0),
            max_position_embeddings: get_usize_or(config, "max_position_embeddings", 32_768),

            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            use_post_norms: false,
            sliding_window: get_optional_usize(config, "sliding_window"),
            alternating_sliding_window: false,
        })
    }
}

// ---------------------------------------------------------------------------
// JSON extraction helpers
// ---------------------------------------------------------------------------

/// Extract a required `usize` field from a JSON object.
fn get_usize(config: &Value, key: &str) -> Result<usize> {
    let val = config
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| MIError::Config(format!("missing or invalid field '{key}'")))?;
    usize::try_from(val)
        .map_err(|_| MIError::Config(format!("field '{key}' value {val} overflows usize")))
}

/// Extract an optional `usize` field, returning a default if absent.
fn get_usize_or(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(default)
}

/// Extract an optional `usize` field, returning `None` if absent.
fn get_optional_usize(config: &Value, key: &str) -> Option<usize> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

/// Extract an `f64` field, returning a default if absent.
fn get_f64_or(config: &Value, key: &str, default: f64) -> f64 {
    config.get(key).and_then(Value::as_f64).unwrap_or(default)
}

/// Extract an optional `f64` field, returning `None` if absent.
fn get_optional_f64(config: &Value, key: &str) -> Option<f64> {
    config.get(key).and_then(Value::as_f64)
}

/// Extract a `bool` field, returning a default if absent.
fn get_bool_or(config: &Value, key: &str, default: bool) -> bool {
    config.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Extract `head_dim`, falling back to `hidden_size / num_attention_heads`.
fn get_head_dim(config: &Value, hidden_size: usize, num_attention_heads: usize) -> Result<usize> {
    // Explicit head_dim in config takes precedence.
    let explicit = config.get("head_dim").and_then(Value::as_u64).map(|hd| {
        usize::try_from(hd).map_err(|_| MIError::Config("head_dim overflows usize".into()))
    });

    match explicit {
        Some(result) => result,
        None if num_attention_heads == 0 => Err(MIError::Config(
            "num_attention_heads is 0, cannot compute head_dim".into(),
        )),
        None => Ok(hidden_size / num_attention_heads),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Helper to create a minimal LLaMA-style config JSON.
    fn llama_config_json() -> Value {
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072
        })
    }

    #[test]
    fn parse_llama_basic() {
        let config = TransformerConfig::from_hf_config(&llama_config_json()).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.norm_type, NormType::RmsNorm);
        assert_eq!(config.activation, Activation::Silu);
        assert_eq!(config.qkv_layout, QkvLayout::Separate);
        assert_eq!(config.mlp_layout, MlpLayout::GatedSeparate);
        assert!(!config.qkv_bias);
        assert!(!config.o_proj_bias);
        assert!(!config.mlp_bias);
        assert!(config.embedding_scale.is_none());
        assert!(!config.tie_word_embeddings);
        assert!((config.rope_theta - 500_000.0).abs() < f64::EPSILON);
        assert!(config.attn_logit_softcapping.is_none());
        assert!(config.sliding_window.is_none());
    }

    #[test]
    fn parse_qwen2_bias() {
        let json = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "vocab_size": 151936,
            "attention_bias": true,
            "tie_word_embeddings": true
        });
        let config = TransformerConfig::from_hf_config(&json).unwrap();
        assert!(config.qkv_bias);
        assert!(!config.o_proj_bias);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn parse_gemma2_extensions() {
        let json = serde_json::json!({
            "model_type": "gemma2",
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 9216,
            "vocab_size": 256000,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
            "sliding_window": 4096
        });
        let config = TransformerConfig::from_hf_config(&json).unwrap();
        assert_eq!(config.norm_type, NormType::GemmaRmsNorm);
        assert_eq!(config.head_dim, 256);
        assert!(config.embedding_scale.is_some());
        assert!((config.attn_logit_softcapping.unwrap() - 50.0).abs() < f64::EPSILON);
        assert!((config.final_logit_softcapping.unwrap() - 30.0).abs() < f64::EPSILON);
        assert!((config.query_pre_attn_scalar.unwrap() - 256.0).abs() < f64::EPSILON);
        assert!(config.use_post_norms);
        assert_eq!(config.sliding_window, Some(4096));
        assert!(config.alternating_sliding_window);
    }

    #[test]
    fn parse_phi3_fused() {
        let json = serde_json::json!({
            "model_type": "phi3",
            "hidden_size": 3072,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32064
        });
        let config = TransformerConfig::from_hf_config(&json).unwrap();
        assert_eq!(config.qkv_layout, QkvLayout::Fused);
        assert_eq!(config.mlp_layout, MlpLayout::GatedFused);
    }

    #[test]
    fn parse_starcoder2_bias_and_plain_mlp() {
        let json = serde_json::json!({
            "model_type": "starcoder2",
            "hidden_size": 3072,
            "num_hidden_layers": 30,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "intermediate_size": 12288,
            "vocab_size": 49152,
            "use_bias": true,
            "norm_type": "layer_norm"
        });
        let config = TransformerConfig::from_hf_config(&json).unwrap();
        assert_eq!(config.mlp_layout, MlpLayout::Plain);
        assert_eq!(config.activation, Activation::GeluApprox);
        assert_eq!(config.norm_type, NormType::LayerNorm);
        assert!(config.qkv_bias);
        assert!(config.o_proj_bias);
        assert!(config.mlp_bias);
    }

    #[test]
    fn parse_mistral_sliding_window() {
        let json = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "sliding_window": 4096
        });
        let config = TransformerConfig::from_hf_config(&json).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
        assert!(!config.alternating_sliding_window);
    }

    #[test]
    fn unsupported_model_type_errors() {
        let json = serde_json::json!({ "model_type": "bert" });
        let result = TransformerConfig::from_hf_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn missing_model_type_errors() {
        let json = serde_json::json!({ "hidden_size": 768 });
        let result = TransformerConfig::from_hf_config(&json);
        assert!(result.is_err());
    }
}
