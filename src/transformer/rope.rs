// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rotary position embeddings (`RoPE`).
//!
//! Pre-computes `cos` and `sin` tensors at model load time and applies
//! them to query and key tensors during the forward pass.

use candle_core::{DType, Device, Tensor};

use crate::error::Result;

// ---------------------------------------------------------------------------
// RoPE cache — pre-computed cos/sin
// ---------------------------------------------------------------------------

/// Pre-computed cosine and sine tensors for rotary position embeddings.
pub struct RopeCache {
    /// Cosine values: `[max_position, head_dim / 2]`.
    cos: Tensor,
    /// Sine values: `[max_position, head_dim / 2]`.
    sin: Tensor,
}

impl RopeCache {
    /// Pre-compute the `RoPE` cache.
    ///
    /// # Shapes
    /// - `cos`: `[max_position, head_dim / 2]`
    /// - `sin`: `[max_position, head_dim / 2]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures.
    pub fn new(
        head_dim: usize,
        max_position: usize,
        theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies: theta^(-2i/d) for i in 0..half_dim
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                // Safe: f64 → f32 truncation is intentional for RoPE frequencies
                #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
                let freq_f32 = freq as f32;
                freq_f32
            })
            .collect();

        let inv_freq_tensor = Tensor::from_vec(inv_freq, (1, half_dim), device)?;

        // Position indices: [0, 1, 2, ..., max_position - 1]
        let positions: Vec<f32> = (0..max_position)
            .map(|p| {
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let pf = p as f32;
                pf
            })
            .collect();
        let pos_tensor = Tensor::from_vec(positions, (max_position, 1), device)?;

        // Outer product: [max_position, half_dim]
        let freqs = pos_tensor.matmul(&inv_freq_tensor)?;

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    /// Apply rotary embeddings to a query or key tensor.
    ///
    /// # Shapes
    /// - `x`: `[batch, n_heads, seq_len, head_dim]`
    /// - returns: `[batch, n_heads, seq_len, head_dim]`
    ///
    /// The `start_pos` parameter supports incremental generation (KV-cache):
    /// positions are offset by `start_pos` so that cached keys keep their
    /// original positional encoding.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation or shape errors.
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;

        // Slice cos/sin for the relevant positions
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        // Reshape for broadcasting: [1, 1, seq_len, half_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Split x into first half and second half along head_dim
        let x1 = x.narrow(candle_core::D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(candle_core::D::Minus1, half_dim, half_dim)?;

        // RoPE rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Ok(Tensor::cat(
            &[&rotated_x1, &rotated_x2],
            candle_core::D::Minus1,
        )?)
    }
}
