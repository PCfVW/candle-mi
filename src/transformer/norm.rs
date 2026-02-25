// SPDX-License-Identifier: MIT OR Apache-2.0

//! Normalization variants for the generic transformer.
//!
//! Supports [`RmsNorm`](NormType::RmsNorm),
//! [`LayerNorm`](NormType::LayerNorm), and
//! [`GemmaRmsNorm`](NormType::GemmaRmsNorm) (which adds `1.0` to the
//! learned weight).

use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;

use crate::config::NormType;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Norm — enum-dispatched normalization
// ---------------------------------------------------------------------------

/// A normalization layer, selected at load time by [`NormType`].
// EXHAUSTIVE: internal dispatch enum; crate owns all three norm variants and matches exhaustively
#[allow(clippy::exhaustive_enums)]
pub enum Norm {
    /// Standard RMS normalization.
    Rms(candle_nn::RmsNorm),
    /// Standard layer normalization (weight + bias).
    Layer(candle_nn::LayerNorm),
    /// Gemma-style RMS norm: weight is stored as `w`, but applied as `(w + 1)`.
    GemmaRms(GemmaRmsNorm),
}

impl Norm {
    /// Apply normalization to the input tensor.
    ///
    /// # Shapes
    /// - `xs`: `[batch, seq, hidden_size]`
    /// - returns: `[batch, seq, hidden_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => Ok(norm.forward(xs)?),
            Self::Layer(norm) => Ok(norm.forward(xs)?),
            Self::GemmaRms(norm) => norm.forward(xs),
        }
    }
}

/// Create a [`Norm`] layer from the given configuration.
///
/// # Errors
///
/// Returns [`MIError::Model`] if weight loading fails.
#[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
pub fn create_norm(
    norm_type: NormType,
    hidden_size: usize,
    eps: f64,
    vb: VarBuilder<'_>,
) -> Result<Norm> {
    match norm_type {
        NormType::RmsNorm => {
            let norm = candle_nn::rms_norm(hidden_size, eps, vb)?;
            Ok(Norm::Rms(norm))
        }
        NormType::LayerNorm => {
            let config = candle_nn::LayerNormConfig {
                eps,
                ..Default::default()
            };
            let norm = candle_nn::layer_norm(hidden_size, config, vb)?;
            Ok(Norm::Layer(norm))
        }
        NormType::GemmaRmsNorm => {
            let norm = GemmaRmsNorm::load(hidden_size, eps, vb)?;
            Ok(Norm::GemmaRms(norm))
        }
    }
}

// ---------------------------------------------------------------------------
// GemmaRmsNorm — custom RMS norm that adds 1.0 to weight
// ---------------------------------------------------------------------------

/// Gemma-style RMS normalization.
///
/// Identical to standard `RmsNorm` except that the learned weight `w` is
/// applied as `(w + 1.0)`.  This is a Gemma-specific architectural choice.
pub struct GemmaRmsNorm {
    /// Learned weight vector.
    weight: Tensor,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl GemmaRmsNorm {
    /// Load the norm from a [`VarBuilder`] that provides `"weight"`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if the weight tensor cannot be loaded.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder convention
    fn load(hidden_size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }

    /// Apply Gemma RMS normalization.
    ///
    /// # Shapes
    /// - `xs`: `[batch, seq, hidden_size]`
    /// - returns: `[batch, seq, hidden_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();

        // PROMOTE: RMS norm variance computation requires F32 to avoid overflow
        let xs_f32 = if original_dtype == DType::F32 {
            xs.clone()
        } else {
            xs.to_dtype(DType::F32)?
        };

        // Compute RMS: sqrt(mean(x^2) + eps)
        let variance = xs_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normed = xs_f32.broadcast_div(&rms)?;

        // Gemma-specific: apply (weight + 1.0) instead of just weight
        let weight_plus_one = (&self.weight.to_dtype(DType::F32)? + 1.0)?;
        let result = normed.broadcast_mul(&weight_plus_one)?;

        // Convert back to original dtype
        if original_dtype == DType::F32 {
            Ok(result)
        } else {
            Ok(result.to_dtype(original_dtype)?)
        }
    }
}
