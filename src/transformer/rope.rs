// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rotary position embeddings (`RoPE`).
//!
//! Pre-computes `cos` and `sin` tensors at model load time and applies
//! them to query and key tensors during the forward pass.
//!
//! Uses `candle_nn::rotary_emb::rope()` for the actual rotation, matching
//! the reference implementation in plip-rs (frozen predecessor project, v1.4.0).

use candle_core::{DType, Device, Tensor};

use crate::config::RopeScaling;
use crate::error::Result;

/// Apply Llama 3 frequency-band rescaling to inverse frequencies in place.
///
/// Mirrors `HuggingFace`'s `_compute_llama3_parameters`: inverse frequencies
/// whose wavelength exceeds `low_freq_wavelen` are divided by `factor`
/// (low-frequency band), those below `high_freq_wavelen` are left intact
/// (high-frequency band), and the band between is smoothly interpolated.
///
/// Operates on `f64` for parity with the `PyTorch` reference (which computes
/// the rescaled frequencies in double precision before casting).
fn apply_llama3_scaling(
    inv_freq: &mut [f64],
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_position_embeddings: usize,
) {
    use std::f64::consts::PI;

    // CAST: usize → f64, context length fits in f64 mantissa (<= 2^52).
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let orig_max = original_max_position_embeddings as f64;
    let low_freq_wavelen = orig_max / low_freq_factor;
    let high_freq_wavelen = orig_max / high_freq_factor;

    for freq in inv_freq.iter_mut() {
        let wavelen = 2.0 * PI / *freq;
        if wavelen > low_freq_wavelen {
            // Low-frequency band: divide the inverse frequency by `factor`.
            *freq /= factor;
        } else if wavelen >= high_freq_wavelen {
            // Medium band (`high_freq_wavelen <= wavelen <= low_freq_wavelen`):
            // smooth interpolation between scaled and unscaled.
            let smooth =
                (orig_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            *freq = (1.0 - smooth) * *freq / factor + smooth * *freq;
        }
        // else high-frequency band: inverse frequency unchanged.
    }
}

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
    /// `scaling` applies a `rope_scaling` interpolation scheme (see
    /// [`RopeScaling`]):
    /// - [`RopeScaling::Linear`] divides every position index by `factor`
    ///   (uniform across frequencies).
    /// - [`RopeScaling::Llama3`] rescales the inverse frequencies by
    ///   frequency band (position-independent).
    /// - `None` is standard, unscaled `RoPE`.
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
        scaling: Option<RopeScaling>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Base inverse frequencies in f64 (theta^(-2i/d)); rescaled below for
        // Llama 3.  f64 throughout matches the PyTorch reference.
        let mut inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| {
                // CAST: usize → f64, loop index and head_dim fit in f64 mantissa
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let exponent = 2.0 * i as f64 / head_dim as f64;
                1.0 / theta.powf(exponent)
            })
            .collect();

        if let Some(RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        }) = scaling
        {
            apply_llama3_scaling(
                &mut inv_freq,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            );
        }

        // CAST: f64 → f32, precision loss acceptable for RoPE frequencies
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let inv_freq_f32: Vec<f32> = inv_freq.iter().map(|&f| f as f32).collect();
        let inv_freq_tensor =
            Tensor::from_vec(inv_freq_f32, (1, half_dim), device)?.to_dtype(dtype)?;

        // Position indices [0, 1, ..., max_position - 1].  Linear scaling
        // divides each position by `factor` (fractional, so built directly as
        // f32 rather than via an integer `arange`).
        let positions: Vec<f32> = match scaling {
            Some(RopeScaling::Linear { factor }) => (0..max_position)
                .map(|p| {
                    // CAST: usize → f64 then f32; max_position <= ~128K is exact in f64.
                    #[allow(
                        clippy::cast_precision_loss,
                        clippy::cast_possible_truncation,
                        clippy::as_conversions
                    )]
                    let scaled = (p as f64 / factor) as f32;
                    scaled
                })
                .collect(),
            // Llama3 scaling acts on the frequencies (applied above), and the
            // unscaled case both use raw integer positions.  Listed explicitly
            // rather than `_` so a future RopeScaling variant must opt in here.
            None | Some(RopeScaling::Llama3 { .. }) => (0..max_position)
                .map(|p| {
                    // CAST: usize → f32; max_position <= ~128K is exact in f32.
                    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                    let pos = p as f32;
                    pos
                })
                .collect(),
        };
        let pos_tensor = Tensor::from_vec(positions, (max_position, 1), device)?.to_dtype(dtype)?;

        // Outer product: [max_position, half_dim]
        let freqs = pos_tensor.matmul(&inv_freq_tensor)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    /// Apply rotary embeddings to a query or key tensor.
    ///
    /// Uses `candle_nn::rotary_emb::rope()` for the rotation.
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
        let (_, _, seq_len, _) = x.dims4()?;

        // Slice cos/sin for the relevant positions: [seq_len, half_dim]
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        // candle_nn::rotary_emb::rope() expects contiguous input
        Ok(candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)?)
    }
}
