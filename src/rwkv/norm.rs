// SPDX-License-Identifier: MIT OR Apache-2.0

//! Normalization layers for RWKV models.
//!
//! RWKV uses standard `LayerNorm` (with bias) everywhere, plus a manual
//! `GroupNorm` after the WKV output.  These are implemented from scratch
//! because we need explicit weight/bias access for the `GroupNorm`
//! parameters stored alongside the attention block.

use candle_core::{D, Tensor};
use candle_nn::VarBuilder;

use crate::error::Result;

// ---------------------------------------------------------------------------
// LayerNorm (with bias)
// ---------------------------------------------------------------------------

/// Standard layer normalization with learned weight and bias.
///
/// RWKV uses `LayerNorm` (not `RmsNorm`) in all normalization positions.
/// We implement it manually rather than using `candle_nn::LayerNorm`
/// because we need the raw weight and bias tensors for weight loading
/// validation and `GroupNorm` parameter sharing.
pub struct LayerNorm {
    /// Learned scale parameter.
    weight: Tensor,
    /// Learned bias parameter.
    bias: Tensor,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl LayerNorm {
    /// Load a `LayerNorm` from weights.
    ///
    /// # Shapes
    /// - `weight`: `[size]`
    /// - `bias`: `[size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::error::MIError::Model) if weights
    /// cannot be loaded.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    /// Apply layer normalization.
    ///
    /// # Shapes
    /// - `x`: `[..., hidden_size]` -- input tensor (any leading dimensions)
    /// - returns: `[..., hidden_size]` -- same shape as input
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::error::MIError::Model) on tensor
    /// operation failure.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        Ok(x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

// ---------------------------------------------------------------------------
// GroupNorm (manual)
// ---------------------------------------------------------------------------

/// Apply group normalization.
///
/// Used after the WKV output for per-head normalization.
/// Not available in `candle_nn`, so implemented manually.
///
/// # Shapes
/// - `x`: `[batch_seq, channels]` -- input (flattened batch*seq dimension)
/// - `weight`: `[channels]` -- learned scale
/// - `bias`: `[channels]` -- learned bias
/// - returns: `[batch_seq, channels]`
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::error::MIError::Model) on tensor
/// operation failure.
pub fn group_norm(
    x: &Tensor,
    num_groups: usize,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let (n, c) = x.dims2()?;
    let channels_per_group = c / num_groups;

    // Reshape to [n, num_groups, channels_per_group]
    let x = x.reshape((n, num_groups, channels_per_group))?;
    let mean = x.mean_keepdim(2)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = x_centered.sqr()?.mean_keepdim(2)?;
    let x_normed = x_centered.broadcast_div(&(var + eps)?.sqrt()?)?;

    // Reshape back to [n, channels]
    let x_normed = x_normed.reshape((n, c))?;

    // Affine transform
    Ok(x_normed
        .broadcast_mul(&weight.unsqueeze(0)?)?
        .broadcast_add(&bias.unsqueeze(0)?)?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn layer_norm_basic() {
        let device = Device::Cpu;
        // 1D input: [1, 4]
        let x = Tensor::new(&[[1.0_f32, 2.0, 3.0, 4.0]], &device).unwrap();
        let weight = Tensor::ones(4, DType::F32, &device).unwrap();
        let bias = Tensor::zeros(4, DType::F32, &device).unwrap();

        let ln = LayerNorm {
            weight,
            bias,
            eps: 1e-5,
        };
        let out = ln.forward(&x).unwrap();

        // Mean=2.5, Var=1.25, so normed = (x-2.5)/sqrt(1.25+eps)
        let out_vec: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(out_vec.len(), 4);
        // First element: (1-2.5)/sqrt(1.25) ≈ -1.3416
        assert!((out_vec[0] - (-1.3416)).abs() < 0.01);
    }

    #[test]
    fn group_norm_basic() {
        let device = Device::Cpu;
        // [2, 4] input with 2 groups
        let x = Tensor::new(&[[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device).unwrap();
        let weight = Tensor::ones(4, DType::F32, &device).unwrap();
        let bias = Tensor::zeros(4, DType::F32, &device).unwrap();

        let out = group_norm(&x, 2, &weight, &bias, 1e-5).unwrap();
        let shape = out.dims2().unwrap();
        assert_eq!(shape, (2, 4));

        // Group 0 (channels 0,1): mean=1.5, group 1 (channels 2,3): mean=3.5
        // After norm each group should be zero-mean with unit variance
        let out_vec: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // First row group 0: (1-1.5)/sqrt(0.25), (2-1.5)/sqrt(0.25) = -1, 1
        assert!((out_vec[0] - (-1.0)).abs() < 0.01);
        assert!((out_vec[1] - 1.0).abs() < 0.01);
    }
}
