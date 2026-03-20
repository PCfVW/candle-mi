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
use std::io::Read as _;
use std::path::Path;

use serde_json::Value;

use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// Supported model types
// ---------------------------------------------------------------------------

/// `model_type` strings accepted by
/// [`TransformerConfig::from_hf_config`].
///
/// Use this for cache discovery, UI filtering, or anywhere you need to know
/// which `HuggingFace` model families the generic transformer backend handles.
pub const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "gemma",
    "gemma2",
    "llama",
    "mistral",
    "phi3",
    "qwen2",
    "starcoder2",
];

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
    /// Gaussian Error Linear Unit — `PyTorch` tanh approximation.
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
///
/// # `config.json` field reference
///
/// ## Required fields (all families)
///
/// | Field | `config.json` key |
/// |-------|-------------------|
/// | — | `model_type` |
/// | `hidden_size` | `hidden_size` |
/// | `num_layers` | `num_hidden_layers` |
/// | `num_attention_heads` | `num_attention_heads` |
/// | `intermediate_size` | `intermediate_size` |
/// | `vocab_size` | `vocab_size` |
///
/// ## Optional fields (all families)
///
/// | Field | `config.json` key | Default |
/// |-------|-------------------|---------|
/// | `num_kv_heads` | `num_key_value_heads` | `num_attention_heads` |
/// | `head_dim` | `head_dim` | `hidden_size / num_attention_heads` |
/// | `norm_eps` | `rms_norm_eps` ¹ | 1e-5 ² |
/// | `rope_theta` | `rope_theta` | 10 000 ³ |
/// | `max_position_embeddings` | `max_position_embeddings` | 4 096 ⁴ |
/// | `tie_word_embeddings` | `tie_word_embeddings` | `false` ⁵ |
///
/// ¹ `StarCoder2` reads `norm_epsilon` instead.\
/// ² 1e-6 for Qwen2, Gemma, Gemma 2.\
/// ³ 1 000 000 for Qwen2.\
/// ⁴ 32 768 for Qwen2/Mistral; 16 384 for `StarCoder2`; 8 192 for
///   Gemma/Gemma 2; 4 096 for `LLaMA`/`Phi-3`.\
/// ⁵ `true` for Gemma, Gemma 2, `StarCoder2`.
///
/// ## Hardcoded architecture axes
///
/// The following fields are **set by the family-specific parser**, not
/// read from `config.json` (except where noted):
///
/// | Field | Description |
/// |-------|-------------|
/// | `norm_type` | [`RmsNorm`](NormType::RmsNorm) for most; [`GemmaRmsNorm`](NormType::GemmaRmsNorm) for Gemma/Gemma 2; read from `norm_type` key for `StarCoder2` (default [`RmsNorm`](NormType::RmsNorm), `"layer_norm"` → [`LayerNorm`](NormType::LayerNorm)) |
/// | `activation` | [`Silu`](Activation::Silu) for `LLaMA`/Qwen2/`Phi-3`/Mistral; [`GeluApprox`](Activation::GeluApprox) for Gemma/Gemma 2/`StarCoder2` |
/// | `qkv_layout` | [`Fused`](QkvLayout::Fused) for `Phi-3`; [`Separate`](QkvLayout::Separate) for all others |
/// | `mlp_layout` | [`GatedFused`](MlpLayout::GatedFused) for `Phi-3`; [`Plain`](MlpLayout::Plain) for `StarCoder2`; [`GatedSeparate`](MlpLayout::GatedSeparate) for all others |
/// | `embedding_scale` | `Some(sqrt(hidden_size))` for Gemma/Gemma 2; `None` for all others |
/// | `use_post_norms` | `true` for Gemma 2 (4 norms per layer); `false` for all others |
/// | `alternating_sliding_window` | `true` for Gemma 2; `false` for all others |
///
/// ## Per-family `config.json` extensions
///
/// **Qwen2** — reads `attention_bias` (default `true`) → `qkv_bias`.
///
/// **Gemma / Gemma 2** — hardcodes `embedding_scale` to `sqrt(hidden_size)`,
/// `tie_word_embeddings` defaults to `true`, and `norm_eps` defaults to 1e-6.
/// Gemma 2 additionally reads:
///
/// | `config.json` key | Field | Default |
/// |-------------------|-------|---------|
/// | `attn_logit_softcapping` | `attn_logit_softcapping` | `None` |
/// | `final_logit_softcapping` | `final_logit_softcapping` | `None` |
/// | `query_pre_attn_scalar` | `query_pre_attn_scalar` | `Some(256.0)` |
/// | `sliding_window` | `sliding_window` | `None` |
///
/// **`Phi-3`** — no extra `config.json` keys; fused QKV and fused gated MLP
/// are hardcoded.
///
/// **`StarCoder2`** — reads `use_bias` (default `true`) → `qkv_bias`,
/// `o_proj_bias`, and `mlp_bias`.  Reads `norm_type` (default `RmsNorm`,
/// `"layer_norm"` → `LayerNorm`).  Uses `norm_epsilon` key (not
/// `rms_norm_eps`).  Hardcodes [`Plain`](MlpLayout::Plain) MLP and
/// [`GeluApprox`](Activation::GeluApprox) activation.
///
/// **Mistral** — reads `sliding_window` (default `None`).  Otherwise
/// identical to `LLaMA`; `max_position_embeddings` defaults to 32 768.
#[derive(Debug, Clone, PartialEq)]
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
    /// See the [`TransformerConfig`] struct-level documentation for the
    /// full field reference (required/optional keys, defaults, and
    /// per-family extensions).
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

        // Keep in sync with SUPPORTED_MODEL_TYPES.
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
            // CAST: usize → f64, hidden_size fits in f64 mantissa (d_model <= 2^52)
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
            // CAST: usize → f64, hidden_size fits in f64 mantissa (d_model <= 2^52)
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
pub(crate) fn get_usize(config: &Value, key: &str) -> Result<usize> {
    let val = config
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| MIError::Config(format!("missing or invalid field '{key}'")))?;
    usize::try_from(val)
        .map_err(|_| MIError::Config(format!("field '{key}' value {val} overflows usize")))
}

/// Extract an optional `usize` field, returning a default if absent.
pub(crate) fn get_usize_or(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(default)
}

/// Extract an optional `usize` field, returning `None` if absent.
pub(crate) fn get_optional_usize(config: &Value, key: &str) -> Option<usize> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

/// Extract an `f64` field, returning a default if absent.
pub(crate) fn get_f64_or(config: &Value, key: &str, default: f64) -> f64 {
    config.get(key).and_then(Value::as_f64).unwrap_or(default)
}

/// Extract an optional `f64` field, returning `None` if absent.
pub(crate) fn get_optional_f64(config: &Value, key: &str) -> Option<f64> {
    config.get(key).and_then(Value::as_f64)
}

/// Extract a `bool` field, returning a default if absent.
pub(crate) fn get_bool_or(config: &Value, key: &str, default: bool) -> bool {
    config.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Extract `head_dim`, falling back to `hidden_size / num_attention_heads`.
pub(crate) fn get_head_dim(
    config: &Value,
    hidden_size: usize,
    num_attention_heads: usize,
) -> Result<usize> {
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
// Activation string parsing
// ---------------------------------------------------------------------------

/// Infer [`Activation`] from `hidden_activation` or `hidden_act` config fields.
///
/// Prefers `hidden_activation` (used by Gemma 2) over `hidden_act`.
/// Defaults to [`Activation::Silu`] when neither field is present.
fn parse_activation_str(config: &Value) -> Activation {
    let act_str = config
        .get("hidden_activation")
        .or_else(|| config.get("hidden_act"))
        .and_then(Value::as_str);
    match act_str {
        Some("gelu_pytorch_tanh") => Activation::GeluApprox,
        Some("gelu") => Activation::Gelu,
        _ => Activation::Silu,
    }
}

// ---------------------------------------------------------------------------
// Tensor name utilities
// ---------------------------------------------------------------------------

/// Extract tensor names from a single `.safetensors` file header.
///
/// Reads only the JSON header (first 8 bytes = length, then header bytes);
/// no weight data is loaded.
///
/// # Errors
///
/// Returns [`MIError::Io`] on read failure, [`MIError::Config`] if the
/// header is malformed.
pub fn tensor_names_from_safetensors(path: &Path) -> Result<Vec<String>> {
    let mut file = std::fs::File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    let header_len = usize::try_from(header_len)
        .map_err(|_| MIError::Config("safetensors header length overflows usize".into()))?;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: Value = serde_json::from_slice(&header_buf)
        .map_err(|e| MIError::Config(format!("failed to parse safetensors header: {e}")))?;
    let obj = header
        .as_object()
        .ok_or_else(|| MIError::Config("safetensors header is not a JSON object".into()))?;
    Ok(obj
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect())
}

/// Extract tensor names from a `model.safetensors.index.json` index file.
///
/// Reads the `weight_map` keys from the sharded model index.
///
/// # Errors
///
/// Returns [`MIError::Io`] on read failure, [`MIError::Config`] if the
/// index is malformed or missing `weight_map`.
pub fn tensor_names_from_index(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let index: Value = serde_json::from_str(&content)
        .map_err(|e| MIError::Config(format!("failed to parse safetensors index: {e}")))?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| MIError::Config("missing 'weight_map' in safetensors index".into()))?;
    Ok(weight_map.keys().cloned().collect())
}

// ---------------------------------------------------------------------------
// Auto-config: generic parser for unknown model families
// ---------------------------------------------------------------------------

impl TransformerConfig {
    /// Parse a [`TransformerConfig`] from a `HuggingFace` `config.json` value
    /// and safetensors tensor names.
    ///
    /// Two-tier dispatch:
    /// - **Known families** (listed in [`SUPPORTED_MODEL_TYPES`]): delegates to
    ///   the existing manually-validated parser via [`from_hf_config`](Self::from_hf_config).
    /// - **Unknown families**: auto-detects architecture axes from `config.json`
    ///   scalars and safetensors tensor names (QKV/MLP layout, bias flags, norm
    ///   type, post-norms), with `model_type`-based fixups for Gemma-family
    ///   traits.
    ///
    /// `tensor_names` should contain all tensor names from the model's
    /// safetensors file(s).  Use [`tensor_names_from_safetensors`] or
    /// [`tensor_names_from_index`] to obtain them without loading weights.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `model_type` is missing or if required
    /// dimension fields are absent.
    pub fn from_hf_config_auto(config: &Value, tensor_names: &[String]) -> Result<Self> {
        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| MIError::Config("missing 'model_type' field".into()))?;

        // Known families: use existing manually-validated parsers
        if SUPPORTED_MODEL_TYPES.contains(&model_type) {
            return Self::from_hf_config(config);
        }

        // Unknown families: auto-detect from config.json + tensor names
        Self::parse_auto(config, tensor_names, model_type)
    }

    /// Auto-detect a [`TransformerConfig`] from `config.json` scalars and
    /// safetensors tensor names.
    ///
    /// Uses a four-tier inference strategy:
    /// 1. Required scalars from `config.json`
    /// 2. Optional scalars from `config.json` with sensible defaults
    /// 3. Architecture axes inferred from layer-0 tensor names
    /// 4. `model_type`-based fixups (Gemma `RmsNorm`, embedding scale)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    fn parse_auto(config: &Value, tensor_names: &[String], model_type: &str) -> Result<Self> {
        // Helper: check if a tensor matching `layers.0.<suffix>` exists
        let has_layer0 = |suffix: &str| {
            tensor_names
                .iter()
                .any(|n| n.contains("layers.0.") && n.ends_with(suffix))
        };

        // --- Tier 1: Required scalars ---
        let hidden_size = get_usize(config, "hidden_size")?;
        let num_attention_heads = get_usize(config, "num_attention_heads")?;

        // --- Tier 2: Optional scalars ---
        let norm_eps = config
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .or_else(|| config.get("norm_epsilon").and_then(Value::as_f64))
            .unwrap_or(1e-5);

        let activation = parse_activation_str(config);

        // Sliding window: respect `use_sliding_window: false` (Qwen2)
        let sliding_window =
            if config.get("use_sliding_window").and_then(Value::as_bool) == Some(false) {
                None
            } else {
                get_optional_usize(config, "sliding_window")
            };

        // tie_word_embeddings: config.json field, fallback to tensor name check
        let tie_word_embeddings = config
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or_else(|| !tensor_names.iter().any(|n| n == "lm_head.weight"));

        // Gemma 2 extensions (Tier 2 — read from config.json if present)
        let attn_logit_softcapping = get_optional_f64(config, "attn_logit_softcapping");
        let final_logit_softcapping = get_optional_f64(config, "final_logit_softcapping");
        let query_pre_attn_scalar = get_optional_f64(config, "query_pre_attn_scalar");

        // --- Tier 3: Tensor name inference ---

        // QKV layout
        let qkv_layout = if has_layer0("self_attn.qkv_proj.weight") {
            QkvLayout::Fused
        } else {
            QkvLayout::Separate
        };

        // MLP layout
        let mlp_layout = if has_layer0("mlp.gate_up_proj.weight") {
            MlpLayout::GatedFused
        } else if has_layer0("mlp.gate_proj.weight") {
            MlpLayout::GatedSeparate
        } else if has_layer0("mlp.c_fc.weight") {
            MlpLayout::Plain
        } else {
            MlpLayout::GatedSeparate // safest default for decoder-only transformers
        };

        // Bias flags
        let qkv_bias = has_layer0("self_attn.q_proj.bias") || has_layer0("self_attn.qkv_proj.bias");
        let o_proj_bias = has_layer0("self_attn.o_proj.bias");
        let mlp_bias = has_layer0("mlp.down_proj.bias")
            || has_layer0("mlp.c_fc.bias")
            || has_layer0("mlp.gate_proj.bias")
            || has_layer0("mlp.gate_up_proj.bias");

        // Norm type: LayerNorm if norm layers have bias tensors
        let has_norm_bias = has_layer0("input_layernorm.bias");
        let base_norm_type = if has_norm_bias {
            NormType::LayerNorm
        } else {
            NormType::RmsNorm
        };

        // Post-norms (4-norm layers, Gemma 2 style)
        let use_post_norms = has_layer0("post_feedforward_layernorm.weight")
            || has_layer0("pre_feedforward_layernorm.weight");

        // --- Tier 4: model_type fixups ---
        let is_gemma = model_type.contains("gemma");

        let norm_type = if is_gemma {
            NormType::GemmaRmsNorm
        } else {
            base_norm_type
        };

        // CAST: usize → f64, hidden_size fits in f64 mantissa (d_model <= 2^52)
        // PROMOTE: embedding scale is sqrt(hidden_size); precision loss negligible for d_model <= 2^52
        let embedding_scale = if is_gemma {
            Some((hidden_size as f64).sqrt())
        } else {
            None
        };

        let alternating_sliding_window = is_gemma && use_post_norms;

        // Gemma 2-like models default query_pre_attn_scalar to 256
        let query_pre_attn_scalar = if is_gemma && use_post_norms {
            query_pre_attn_scalar.or(Some(256.0))
        } else {
            query_pre_attn_scalar
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
            norm_eps,
            activation,
            qkv_layout,
            mlp_layout,
            qkv_bias,
            o_proj_bias,
            mlp_bias,
            embedding_scale,
            tie_word_embeddings,

            rope_theta: get_f64_or(config, "rope_theta", 10_000.0),
            max_position_embeddings: get_usize_or(config, "max_position_embeddings", 4096),

            attn_logit_softcapping,
            final_logit_softcapping,
            query_pre_attn_scalar,
            use_post_norms,
            sliding_window,
            alternating_sliding_window,
        })
    }
}

// ---------------------------------------------------------------------------
// Auto-config compatibility check
// ---------------------------------------------------------------------------

/// Result of a compatibility check for auto-config loading.
///
/// Returned by [`TransformerConfig::check_auto_compatibility`].
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    /// Whether the model is loadable by `GenericTransformer`.
    pub compatible: bool,
    /// Human-readable issues found (empty if compatible).
    pub issues: Vec<String>,
}

impl CompatibilityReport {
    /// Returns `Ok(())` if compatible, or [`MIError::Config`] with a
    /// diagnostic summary of all issues.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] listing all detected incompatibilities.
    pub fn into_result(self) -> Result<()> {
        if self.compatible {
            Ok(())
        } else {
            Err(MIError::Config(format!(
                "model is not compatible with GenericTransformer:\n  - {}",
                self.issues.join("\n  - ")
            )))
        }
    }
}

impl TransformerConfig {
    /// Check whether `config.json` contains the required fields for auto-config.
    ///
    /// This is a lightweight check that does not require tensor names or
    /// downloading weights.  It validates that the five required scalar
    /// fields (`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
    /// `intermediate_size`, `vocab_size`) are present.
    ///
    /// A passing check does **not** guarantee full compatibility — use
    /// [`check_auto_compatibility`](Self::check_auto_compatibility) with
    /// tensor names for a definitive answer.
    #[must_use]
    pub fn check_config_fields(config: &Value) -> CompatibilityReport {
        let required = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "vocab_size",
        ];
        let mut issues = Vec::new();
        for key in &required {
            if config.get(*key).and_then(Value::as_u64).is_none() {
                issues.push(format!("missing or invalid required field '{key}'"));
            }
        }
        CompatibilityReport {
            compatible: issues.is_empty(),
            issues,
        }
    }

    /// Check whether a model is fully compatible with `GenericTransformer`
    /// auto-config loading.
    ///
    /// Validates both `config.json` fields and safetensors tensor names
    /// against the patterns `GenericTransformer::load()` expects.  Call
    /// this after downloading but before loading to get a clear diagnostic
    /// instead of a cryptic "tensor not found" error.
    ///
    /// Checks performed:
    /// - Required `config.json` scalars are present
    /// - Embedding tensor (`model.embed_tokens.weight`) exists
    /// - Layer-0 normalization tensors exist (`input_layernorm.weight`,
    ///   `post_attention_layernorm.weight`)
    /// - Final norm tensor (`model.norm.weight`) exists
    /// - At least one recognized attention projection pattern
    /// - At least one recognized MLP projection pattern
    /// - `lm_head.weight` exists when `tie_word_embeddings` is false
    #[must_use]
    pub fn check_auto_compatibility(
        config: &Value,
        tensor_names: &[String],
    ) -> CompatibilityReport {
        let mut issues = Vec::new();

        // --- Config field checks ---
        let field_report = Self::check_config_fields(config);
        issues.extend(field_report.issues);

        // --- Tensor name checks (with "did you mean?" hints) ---
        let has_tensor_issues = check_tensor_names(config, tensor_names, &mut issues);

        // --- Summary of actual naming convention (when tensor checks fail) ---
        if has_tensor_issues
            && !tensor_names.is_empty()
            && let Some(hint) = detect_naming_convention(tensor_names)
        {
            issues.push(hint);
        }

        CompatibilityReport {
            compatible: issues.is_empty(),
            issues,
        }
    }
}

/// Check safetensors tensor names against the patterns `GenericTransformer`
/// expects, appending actionable diagnostics (with "did you mean?" hints)
/// to `issues`.
///
/// Returns `true` if any tensor-name issue was found.
#[allow(clippy::too_many_lines)]
fn check_tensor_names(config: &Value, tensor_names: &[String], issues: &mut Vec<String>) -> bool {
    // Helper: check if a tensor name exists
    let has = |name: &str| tensor_names.iter().any(|n| n == name);
    let has_layer0 = |suffix: &str| {
        tensor_names
            .iter()
            .any(|n| n.contains("layers.0.") && n.ends_with(suffix))
    };

    // Helper: find tensors matching a keyword (for "did you mean?" hints)
    let find_matching = |keyword: &str, limit: usize| -> Vec<&str> {
        tensor_names
            .iter()
            .filter(|n| n.to_lowercase().contains(keyword))
            .take(limit)
            .map(String::as_str)
            .collect::<Vec<_>>()
    };

    let mut has_issues = false;

    // --- Embedding ---
    if !has("model.embed_tokens.weight") {
        has_issues = true;
        let found: Vec<&str> = tensor_names
            .iter()
            .filter(|n| n.contains("embed") || n.contains("wte") || n.contains("word_embeddings"))
            .take(3)
            .map(String::as_str)
            .collect();
        let hint = if found.is_empty() {
            String::new()
        } else {
            format!("; found embedding-like tensors: {}", found.join(", "))
        };
        issues.push(format!(
            "missing embedding tensor 'model.embed_tokens.weight'{hint}"
        ));
    }

    // --- Layer-0 normalization ---
    if !has_layer0("input_layernorm.weight") {
        has_issues = true;
        let found = find_matching("norm", 4);
        let hint = if found.is_empty() {
            String::new()
        } else {
            format!("; found norm-like tensors: {}", found.join(", "))
        };
        issues.push(format!(
            "missing normalization tensor \
             'model.layers.0.input_layernorm.weight'{hint}"
        ));
    }
    if !has_layer0("post_attention_layernorm.weight")
        && !has_layer0("pre_feedforward_layernorm.weight")
    {
        has_issues = true;
        issues.push(
            "missing normalization tensor \
             'model.layers.0.post_attention_layernorm.weight'"
                .into(),
        );
    }

    // --- Final norm ---
    if !has("model.norm.weight") {
        has_issues = true;
        let found: Vec<&str> = tensor_names
            .iter()
            .filter(|n| {
                (n.contains("ln_f") || n.contains("final_layer_norm") || n.contains("ln_out"))
                    && n.ends_with(".weight")
            })
            .take(2)
            .map(String::as_str)
            .collect();
        let hint = if found.is_empty() {
            String::new()
        } else {
            format!("; found final-norm-like tensors: {}", found.join(", "))
        };
        issues.push(format!(
            "missing final norm tensor 'model.norm.weight'{hint}"
        ));
    }

    // --- Attention projections ---
    let has_separate_attn = has_layer0("self_attn.q_proj.weight");
    let has_fused_attn = has_layer0("self_attn.qkv_proj.weight");
    if !has_separate_attn && !has_fused_attn {
        has_issues = true;
        let found = find_matching("attn", 4);
        let hint = if found.is_empty() {
            String::new()
        } else {
            format!("; found attention-like tensors: {}", found.join(", "))
        };
        issues.push(format!(
            "missing attention projections: expected \
             'self_attn.q_proj.weight' or 'self_attn.qkv_proj.weight'{hint}"
        ));
    }

    // --- MLP projections ---
    let has_gated_separate = has_layer0("mlp.gate_proj.weight");
    let has_gated_fused = has_layer0("mlp.gate_up_proj.weight");
    let has_plain = has_layer0("mlp.c_fc.weight");
    // Also accept down_proj as evidence of a recognized MLP
    let has_down = has_layer0("mlp.down_proj.weight");
    if !has_gated_separate && !has_gated_fused && !has_plain && !has_down {
        has_issues = true;
        let found: Vec<&str> = tensor_names
            .iter()
            .filter(|n| n.contains("mlp") || n.contains("ffn") || n.contains("fc"))
            .take(4)
            .map(String::as_str)
            .collect();
        let hint = if found.is_empty() {
            String::new()
        } else {
            format!("; found MLP-like tensors: {}", found.join(", "))
        };
        issues.push(format!(
            "missing MLP projections: expected 'mlp.gate_proj.weight', \
             'mlp.gate_up_proj.weight', or 'mlp.c_fc.weight'{hint}"
        ));
    }

    // --- LM head ---
    let tie = config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| !tensor_names.iter().any(|n| n == "lm_head.weight"));
    if !tie && !has("lm_head.weight") {
        issues.push("tie_word_embeddings is false but 'lm_head.weight' tensor is missing".into());
    }

    has_issues
}

/// Detect known non-standard weight naming conventions and produce a
/// human-readable hint explaining why the model is incompatible.
///
/// Returns `None` if the naming convention is unrecognized.
fn detect_naming_convention(tensor_names: &[String]) -> Option<String> {
    // Known non-standard prefix patterns
    let patterns: &[(&str, &str)] = &[
        (
            "transformer.h.",
            "GPT-2 / GPT-J / GPT-NeoX (uses 'transformer.h.{i}' prefix)",
        ),
        (
            "transformer.blocks.",
            "Falcon / MPT (uses 'transformer.blocks.{i}' prefix)",
        ),
        (
            "gpt_neox.layers.",
            "GPT-NeoX / Pythia (uses 'gpt_neox.layers.{i}' prefix)",
        ),
        (
            "transformer.layer.",
            "BLOOM (uses 'transformer.layer.{i}' prefix)",
        ),
    ];

    for &(prefix, description) in patterns {
        if tensor_names.iter().any(|n| n.starts_with(prefix)) {
            return Some(format!(
                "this model uses {description} — candle-mi currently requires \
                 HF-standard 'model.layers.{{i}}' weight naming. \
                 Support for this architecture is planned in Phase 9 \
                 (tensor name remapping)"
            ));
        }
    }

    // If no known pattern matched, show the first few tensor names as a
    // diagnostic aid
    if !tensor_names.iter().any(|n| n.starts_with("model.layers.")) {
        let sample: Vec<&str> = tensor_names.iter().take(5).map(String::as_str).collect();
        return Some(format!(
            "weight tensors use an unrecognized naming convention \
             (first 5: {}). candle-mi expects 'model.layers.{{i}}.self_attn.*' / \
             'model.layers.{{i}}.mlp.*' naming",
            sample.join(", ")
        ));
    }

    None
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

    // -----------------------------------------------------------------------
    // Auto-config validation: parse_auto() must match manual parsers
    // -----------------------------------------------------------------------
    //
    // For each of the 7 known transformer families, we verify that
    // parse_auto() produces the SAME TransformerConfig as the manual
    // parser.  Config JSON and tensor names are taken from real cached
    // models.
    //
    // Known exception — Phi-3 `sliding_window`: The Phi-3 config.json
    // contains "sliding_window": 2047 but the HuggingFace implementation
    // ignores it.  The manual parser sets None; the auto-parser reads
    // Some(2047).  We test all other fields and assert the sliding_window
    // difference explicitly.

    /// Helper: convert `&[&str]` to `Vec<String>` for tensor names.
    fn tensor_names(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn auto_config_matches_llama() {
        // LLaMA 3.2 1B — actual config.json + tensor names
        let json = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072,
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "tie_word_embeddings": true
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.weight",
        ]);

        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        let auto = TransformerConfig::parse_auto(&json, &names, "llama").unwrap();
        assert_eq!(auto, manual);
    }

    #[test]
    fn auto_config_matches_qwen2() {
        // Qwen2.5-Coder-3B-Instruct — actual config.json + tensor names
        let json = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "intermediate_size": 11008,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 32768,
            "hidden_act": "silu",
            "tie_word_embeddings": true,
            "sliding_window": 32768,
            "use_sliding_window": false
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.weight",
        ]);

        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        let auto = TransformerConfig::parse_auto(&json, &names, "qwen2").unwrap();
        assert_eq!(auto, manual);
    }

    #[test]
    fn auto_config_matches_gemma() {
        // CodeGemma 7B IT — actual config.json + tensor names
        let json = serde_json::json!({
            "model_type": "gemma",
            "hidden_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "intermediate_size": 24576,
            "vocab_size": 256000,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 8192,
            "hidden_activation": "gelu_pytorch_tanh"
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.weight",
        ]);

        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        let auto = TransformerConfig::parse_auto(&json, &names, "gemma").unwrap();
        assert_eq!(auto, manual);
    }

    #[test]
    fn auto_config_matches_gemma2() {
        // Gemma 2 2B — actual config.json + tensor names
        let json = serde_json::json!({
            "model_type": "gemma2",
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 9216,
            "vocab_size": 256000,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 8192,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
            "sliding_window": 4096
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.weight",
        ]);

        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        let auto = TransformerConfig::parse_auto(&json, &names, "gemma2").unwrap();
        assert_eq!(auto, manual);
    }

    #[test]
    fn auto_config_matches_phi3() {
        // Phi-3-mini-4k-instruct — actual config.json + tensor names
        //
        // Known exception: Phi-3 config.json contains "sliding_window": 2047
        // but the manual parser ignores it (sets None).  The auto-parser
        // reads it as Some(2047).  We verify all other fields match and
        // assert the sliding_window difference explicitly.
        let json = serde_json::json!({
            "model_type": "phi3",
            "hidden_size": 3072,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32064,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "sliding_window": 2047,
            "attention_bias": false
        });
        let names = tensor_names(&[
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.norm.weight",
        ]);

        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        let auto = TransformerConfig::parse_auto(&json, &names, "phi3").unwrap();

        // Known exception: sliding_window
        assert_eq!(manual.sliding_window, None);
        assert_eq!(auto.sliding_window, Some(2047));

        // All other fields must match — compare field by field excluding
        // sliding_window by creating copies with the same value.
        let mut auto_adjusted = auto;
        auto_adjusted.sliding_window = None;
        assert_eq!(auto_adjusted, manual);
    }

    #[test]
    fn auto_config_matches_starcoder2() {
        // StarCoder2-3B — actual config.json + tensor names
        let json = serde_json::json!({
            "model_type": "starcoder2",
            "hidden_size": 3072,
            "num_hidden_layers": 30,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "intermediate_size": 12288,
            "vocab_size": 49152,
            "norm_epsilon": 1e-5,
            "norm_type": "layer_norm",
            "rope_theta": 999999.4420358813,
            "max_position_embeddings": 16384,
            "hidden_act": "gelu_pytorch_tanh",
            "use_bias": true,
            "sliding_window": 4096
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.bias",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.c_fc.bias",
            "model.layers.0.mlp.c_fc.weight",
            "model.layers.0.mlp.c_proj.bias",
            "model.layers.0.mlp.c_proj.weight",
            "model.layers.0.post_attention_layernorm.bias",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.bias",
            "model.norm.weight",
        ]);

        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        let auto = TransformerConfig::parse_auto(&json, &names, "starcoder2").unwrap();
        assert_eq!(auto, manual);
    }

    #[test]
    fn auto_config_matches_mistral() {
        // Mistral 7B v0.1 — actual config.json + tensor names
        let json = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 32768,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "sliding_window": 4096
        });
        let names = tensor_names(&[
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.weight",
        ]);

        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        let auto = TransformerConfig::parse_auto(&json, &names, "mistral").unwrap();
        assert_eq!(auto, manual);
    }

    #[test]
    fn auto_config_unknown_model_type() {
        // Verify auto-config works for an unknown model_type using
        // LLaMA-like config.json + tensor names.
        let json = serde_json::json!({
            "model_type": "my_custom_llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "hidden_act": "silu"
        });
        let names = tensor_names(&[
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.weight",
        ]);

        // from_hf_config_auto should use auto-parser (not error)
        let config = TransformerConfig::from_hf_config_auto(&json, &names).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.norm_type, NormType::RmsNorm);
        assert_eq!(config.activation, Activation::Silu);
        assert_eq!(config.qkv_layout, QkvLayout::Separate);
        assert_eq!(config.mlp_layout, MlpLayout::GatedSeparate);
        assert!(!config.qkv_bias);
        assert!(!config.o_proj_bias);
        assert!(!config.mlp_bias);
        assert!(config.embedding_scale.is_none());
        assert!(!config.tie_word_embeddings);
        assert!(config.sliding_window.is_none());
    }

    #[test]
    fn auto_config_dispatches_known_families() {
        // Verify from_hf_config_auto delegates known families to manual parsers
        let json = llama_config_json();
        let names = tensor_names(&["model.embed_tokens.weight"]);

        let auto = TransformerConfig::from_hf_config_auto(&json, &names).unwrap();
        let manual = TransformerConfig::from_hf_config(&json).unwrap();
        assert_eq!(auto, manual);
    }

    // -----------------------------------------------------------------------
    // Compatibility check tests
    // -----------------------------------------------------------------------

    #[test]
    fn compatibility_check_passes_standard_model() {
        let json = serde_json::json!({
            "model_type": "my_custom",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "tie_word_embeddings": true
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.norm.weight",
        ]);
        let report = TransformerConfig::check_auto_compatibility(&json, &names);
        assert!(report.compatible, "issues: {:?}", report.issues);
    }

    #[test]
    fn compatibility_check_detects_missing_norms() {
        // OLMo-like: no norm weights at all
        let json = serde_json::json!({
            "model_type": "olmo",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 50304
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]);
        let report = TransformerConfig::check_auto_compatibility(&json, &names);
        assert!(!report.compatible);
        // Should detect missing input_layernorm, post_attention_layernorm, and model.norm
        assert!(report.issues.len() >= 3, "issues: {:?}", report.issues);
        assert!(
            report.issues.iter().any(|i| i.contains("input_layernorm")),
            "should mention input_layernorm"
        );
        assert!(
            report.issues.iter().any(|i| i.contains("model.norm")),
            "should mention model.norm"
        );
    }

    #[test]
    fn compatibility_check_detects_missing_config_fields() {
        let json = serde_json::json!({
            "model_type": "mystery",
            "hidden_size": 768
        });
        let names = tensor_names(&[]);
        let report = TransformerConfig::check_auto_compatibility(&json, &names);
        assert!(!report.compatible);
        // Missing: num_hidden_layers, num_attention_heads, intermediate_size, vocab_size
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("num_hidden_layers")),
            "should mention num_hidden_layers"
        );
    }

    #[test]
    fn compatibility_check_detects_missing_lm_head() {
        let json = serde_json::json!({
            "model_type": "custom",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "tie_word_embeddings": false
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.norm.weight",
            // Missing: lm_head.weight
        ]);
        let report = TransformerConfig::check_auto_compatibility(&json, &names);
        assert!(!report.compatible);
        assert!(
            report.issues.iter().any(|i| i.contains("lm_head")),
            "should mention lm_head"
        );
    }

    #[test]
    fn compatibility_check_config_only() {
        let good = serde_json::json!({
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32000
        });
        assert!(TransformerConfig::check_config_fields(&good).compatible);

        let bad = serde_json::json!({
            "hidden_size": 2048
        });
        let report = TransformerConfig::check_config_fields(&bad);
        assert!(!report.compatible);
        assert_eq!(report.issues.len(), 4); // missing 4 of 5 required fields
    }

    #[test]
    fn compatibility_into_result_error_message() {
        let json = serde_json::json!({
            "model_type": "olmo",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 50304
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ]);
        let result = TransformerConfig::check_auto_compatibility(&json, &names).into_result();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not compatible with GenericTransformer"),
            "error should explain incompatibility: {msg}"
        );
    }

    #[test]
    fn compatibility_check_shows_gpt2_naming_hint() {
        let json = serde_json::json!({
            "model_type": "gpt2",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 50257
        });
        let names = tensor_names(&[
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.ln_f.weight",
        ]);
        let report = TransformerConfig::check_auto_compatibility(&json, &names);
        assert!(!report.compatible);
        // Should detect GPT-2 naming
        assert!(
            report.issues.iter().any(|i| i.contains("GPT-2")),
            "should detect GPT-2 naming convention: {:?}",
            report.issues
        );
        // Should show found embedding-like tensors
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("transformer.wte.weight")),
            "should show found embedding tensor: {:?}",
            report.issues
        );
        // Should show found attention-like tensors
        assert!(
            report.issues.iter().any(|i| i.contains("c_attn")),
            "should show found attention tensor: {:?}",
            report.issues
        );
    }

    #[test]
    fn compatibility_check_shows_found_tensors_for_unknown_naming() {
        let json = serde_json::json!({
            "model_type": "custom_arch",
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
            "vocab_size": 30000
        });
        let names = tensor_names(&[
            "encoder.layer.0.attention.query.weight",
            "encoder.layer.0.attention.key.weight",
            "encoder.layer.0.ffn.dense.weight",
            "encoder.embeddings.weight",
        ]);
        let report = TransformerConfig::check_auto_compatibility(&json, &names);
        assert!(!report.compatible);
        // Should show the unrecognized naming hint with sample tensors
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("unrecognized naming convention")),
            "should flag unrecognized naming: {:?}",
            report.issues
        );
        // Should show found embedding-like tensor
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("encoder.embeddings.weight")),
            "should show found embedding: {:?}",
            report.issues
        );
    }

    #[test]
    fn compatibility_check_shows_found_norm_tensors() {
        // A model with HF-standard layer prefix but non-standard norm names
        let json = serde_json::json!({
            "model_type": "custom",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "tie_word_embeddings": true
        });
        let names = tensor_names(&[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.attention_norm.weight",
            "model.layers.0.ffn_norm.weight",
            "model.final_norm.weight",
        ]);
        let report = TransformerConfig::check_auto_compatibility(&json, &names);
        assert!(!report.compatible);
        // Should show the alternative norm tensors that were found
        assert!(
            report.issues.iter().any(|i| i.contains("attention_norm")),
            "should show found norm tensors: {:?}",
            report.issues
        );
    }
}
