// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention pattern cache for storing and querying per-layer attention weights.
//!
//! [`AttentionCache`] stores post-softmax attention patterns from each layer
//! of a forward pass, enabling downstream analysis such as attention knockout,
//! steering, and visualization.
//!
//! Each stored tensor has shape `[batch, heads, seq_q, seq_k]`.

use candle_core::{DType, Tensor};

use crate::error::{MIError, Result};

/// Stores per-layer attention weights from a forward pass.
///
/// Each tensor has shape `[batch, heads, seq_q, seq_k]` — the post-softmax
/// attention pattern for one layer.
///
/// # Example
///
/// ```
/// use candle_mi::AttentionCache;
/// use candle_core::{Device, Tensor};
///
/// let mut cache = AttentionCache::with_capacity(32);
/// // shape [batch=1, heads=8, seq=10, seq=10]
/// cache.push(Tensor::zeros((1, 8, 10, 10), candle_core::DType::F32, &Device::Cpu).unwrap());
///
/// // Query what position 5 attends to in layer 0
/// let attn_row = cache.attention_from_position(0, 5).unwrap();
/// ```
#[derive(Debug)]
pub struct AttentionCache {
    /// Attention patterns per layer, each shape `[batch, heads, seq_q, seq_k]`.
    patterns: Vec<Tensor>,
}

impl AttentionCache {
    /// Create an empty cache with capacity for `n_layers` layers.
    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            patterns: Vec::with_capacity(n_layers),
        }
    }

    /// Add an attention pattern for the next layer.
    ///
    /// # Shapes
    ///
    /// - `pattern`: `[batch, heads, seq_q, seq_k]`
    pub fn push(&mut self, pattern: Tensor) {
        self.patterns.push(pattern);
    }

    /// Number of cached layers.
    pub const fn n_layers(&self) -> usize {
        self.patterns.len()
    }

    /// Whether the cache is empty.
    pub const fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get the raw attention tensor for a specific layer.
    ///
    /// # Shapes
    ///
    /// - returns: `[batch, heads, seq_q, seq_k]`
    pub fn get_layer(&self, layer: usize) -> Option<&Tensor> {
        self.patterns.get(layer)
    }

    /// Get attention weights FROM a specific query position, averaged across heads.
    ///
    /// Returns a vector of length `seq_k` representing how much `position`
    /// attends to every key position, averaged over all attention heads
    /// (batch index 0).
    ///
    /// # Shapes
    ///
    /// - returns: `[seq_k]` as `Vec<f32>`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the layer is not in the cache or the
    /// position is out of range.
    pub fn attention_from_position(&self, layer: usize, position: usize) -> Result<Vec<f32>> {
        let pattern = self
            .patterns
            .get(layer)
            .ok_or_else(|| MIError::Hook(format!("layer {layer} not in attention cache")))?;

        let seq_q = pattern.dim(2)?;
        if position >= seq_q {
            return Err(MIError::Hook(format!(
                "position {position} out of range (seq_q={seq_q})"
            )));
        }

        // pattern: [batch, heads, seq_q, seq_k]
        // Select batch 0, all heads, the given query position, all key positions.
        // PROMOTE: averaging attention weights; compute in F32 for precision
        let attn_f32 = pattern.to_dtype(DType::F32)?;
        // narrow(dim=0, start=0, len=1) → [1, heads, seq_q, seq_k]
        let batch0 = attn_f32.narrow(0, 0, 1)?;
        // narrow(dim=2, start=position, len=1) → [1, heads, 1, seq_k]
        let row = batch0.narrow(2, position, 1)?;
        // squeeze dims 0 and 2 → [heads, seq_k]
        let row = row.squeeze(0)?.squeeze(1)?;
        // mean across heads (dim 0) → [seq_k]
        let avg = row.mean(0)?;
        let result: Vec<f32> = avg.to_vec1()?;
        Ok(result)
    }

    /// Get attention weights TO a specific key position, averaged across heads.
    ///
    /// Returns a vector of length `seq_q` representing how much every query
    /// position attends to `position`, averaged over all attention heads
    /// (batch index 0).
    ///
    /// # Shapes
    ///
    /// - returns: `[seq_q]` as `Vec<f32>`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the layer is not in the cache or the
    /// position is out of range.
    pub fn attention_to_position(&self, layer: usize, position: usize) -> Result<Vec<f32>> {
        let pattern = self
            .patterns
            .get(layer)
            .ok_or_else(|| MIError::Hook(format!("layer {layer} not in attention cache")))?;

        let seq_k = pattern.dim(3)?;
        if position >= seq_k {
            return Err(MIError::Hook(format!(
                "position {position} out of range (seq_k={seq_k})"
            )));
        }

        // pattern: [batch, heads, seq_q, seq_k]
        // PROMOTE: averaging attention weights; compute in F32 for precision
        let attn_f32 = pattern.to_dtype(DType::F32)?;
        let batch0 = attn_f32.narrow(0, 0, 1)?;
        // narrow(dim=3, start=position, len=1) → [1, heads, seq_q, 1]
        let col = batch0.narrow(3, position, 1)?;
        // squeeze dims 0 and 3 → [heads, seq_q]
        let col = col.squeeze(0)?.squeeze(2)?;
        // mean across heads (dim 0) → [seq_q]
        let avg = col.mean(0)?;
        let result: Vec<f32> = avg.to_vec1()?;
        Ok(result)
    }

    /// Get the top-k key positions that a given query position attends to most.
    ///
    /// Returns up to `k` pairs of `(key_position, weight)` sorted by
    /// descending attention weight, averaged across heads (batch index 0).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the layer is not in the cache or the
    /// position is out of range.
    pub fn top_attended_positions(
        &self,
        layer: usize,
        from_position: usize,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let attn = self.attention_from_position(layer, from_position)?;
        let mut indexed: Vec<(usize, f32)> = attn.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        Ok(indexed)
    }

    /// All cached patterns as a slice.
    pub fn patterns(&self) -> &[Tensor] {
        &self.patterns
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Build a tiny attention cache for testing.
    ///
    /// Creates a single-layer cache with shape `[1, 2, 4, 4]`:
    /// - 1 batch, 2 heads, 4 query positions, 4 key positions
    /// - Head 0: uniform attention (0.25 everywhere)
    /// - Head 1: identity-like (diagonal = 0.7, off-diagonal = 0.1)
    fn sample_cache() -> AttentionCache {
        let device = Device::Cpu;

        // Head 0: uniform 0.25
        // Head 1: diagonal-heavy
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            // Head 0 (uniform)
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
            // Head 1 (diagonal-heavy)
            0.70, 0.10, 0.10, 0.10,
            0.10, 0.70, 0.10, 0.10,
            0.10, 0.10, 0.70, 0.10,
            0.10, 0.10, 0.10, 0.70,
        ];

        let tensor = Tensor::from_vec(data, (1, 2, 4, 4), &device).unwrap();
        let mut cache = AttentionCache::with_capacity(1);
        cache.push(tensor);
        cache
    }

    #[test]
    fn empty_cache() {
        let cache = AttentionCache::with_capacity(32);
        assert_eq!(cache.n_layers(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_layer(0).is_none());
    }

    #[test]
    fn push_and_get_layer() {
        let cache = sample_cache();
        assert_eq!(cache.n_layers(), 1);
        assert!(!cache.is_empty());

        let layer0 = cache.get_layer(0).unwrap();
        assert_eq!(layer0.dims(), &[1, 2, 4, 4]);
        assert!(cache.get_layer(1).is_none());
    }

    #[test]
    fn attention_from_position_values() {
        let cache = sample_cache();

        // Position 0: head0=[0.25,0.25,0.25,0.25], head1=[0.70,0.10,0.10,0.10]
        // Average: [(0.25+0.70)/2, (0.25+0.10)/2, (0.25+0.10)/2, (0.25+0.10)/2]
        //        = [0.475, 0.175, 0.175, 0.175]
        let attn = cache.attention_from_position(0, 0).unwrap();
        assert_eq!(attn.len(), 4);
        assert!((attn[0] - 0.475).abs() < 1e-5);
        assert!((attn[1] - 0.175).abs() < 1e-5);
        assert!((attn[2] - 0.175).abs() < 1e-5);
        assert!((attn[3] - 0.175).abs() < 1e-5);
    }

    #[test]
    fn attention_from_position_out_of_range() {
        let cache = sample_cache();
        assert!(cache.attention_from_position(0, 10).is_err());
        assert!(cache.attention_from_position(5, 0).is_err());
    }

    #[test]
    fn attention_to_position_values() {
        let cache = sample_cache();

        // Key position 0: each query row's column-0 value
        // Head 0: all rows have 0.25 at col 0 → [0.25, 0.25, 0.25, 0.25]
        // Head 1: rows have [0.70, 0.10, 0.10, 0.10] at col 0
        // Average: [(0.25+0.70)/2, (0.25+0.10)/2, (0.25+0.10)/2, (0.25+0.10)/2]
        //        = [0.475, 0.175, 0.175, 0.175]
        let attn = cache.attention_to_position(0, 0).unwrap();
        assert_eq!(attn.len(), 4);
        assert!((attn[0] - 0.475).abs() < 1e-5);
        assert!((attn[1] - 0.175).abs() < 1e-5);
    }

    #[test]
    fn attention_to_position_out_of_range() {
        let cache = sample_cache();
        assert!(cache.attention_to_position(0, 10).is_err());
        assert!(cache.attention_to_position(5, 0).is_err());
    }

    #[test]
    fn top_attended_positions_sorted() {
        let cache = sample_cache();

        // From position 0, the top-1 should be key position 0 (weight 0.475)
        let top = cache.top_attended_positions(0, 0, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // position 0 has highest weight
        assert!((top[0].1 - 0.475).abs() < 1e-5);
    }

    #[test]
    fn top_attended_positions_k_larger_than_seq() {
        let cache = sample_cache();

        // k=100 but only 4 positions exist → returns all 4
        let top = cache.top_attended_positions(0, 0, 100).unwrap();
        assert_eq!(top.len(), 4);
    }

    #[test]
    fn patterns_slice() {
        let cache = sample_cache();
        assert_eq!(cache.patterns().len(), 1);
    }
}
