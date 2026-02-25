// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activation cache for storing intermediate transformer states.

use candle_core::{DType, Tensor};

use crate::error::{MIError, Result};

/// Stores per-layer last-token activations from a forward pass.
///
/// Each tensor has shape `[d_model]` — the residual stream activation
/// at the final sequence position for a given layer.
///
/// # Example
///
/// ```
/// use candle_mi::ActivationCache;
/// use candle_core::{Device, Tensor};
///
/// let mut cache = ActivationCache::with_capacity(32);
/// cache.push(Tensor::zeros(128, candle_core::DType::F32, &Device::Cpu).unwrap());
/// cache.push(Tensor::zeros(128, candle_core::DType::F32, &Device::Cpu).unwrap());
/// assert_eq!(cache.n_layers(), 2);
/// ```
#[derive(Debug)]
pub struct ActivationCache {
    /// Residual stream activations per layer, each shape `[d_model]`.
    activations: Vec<Tensor>,
}

impl ActivationCache {
    /// Create a new cache from collected activations.
    ///
    /// # Errors
    ///
    /// Currently infallible but returns `Result` for forward compatibility.
    pub const fn new(activations: Vec<Tensor>) -> Result<Self> {
        Ok(Self { activations })
    }

    /// Create an empty cache with capacity for `n_layers` layers.
    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            activations: Vec::with_capacity(n_layers),
        }
    }

    /// Add a layer's activation to the cache.
    pub fn push(&mut self, tensor: Tensor) {
        self.activations.push(tensor);
    }

    /// Get the activation for a specific layer.
    pub fn get_layer(&self, layer: usize) -> Option<&Tensor> {
        self.activations.get(layer)
    }

    /// Number of cached layers.
    pub const fn n_layers(&self) -> usize {
        self.activations.len()
    }

    /// Whether the cache is empty.
    pub const fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }

    /// All cached activations as a slice.
    pub fn activations(&self) -> &[Tensor] {
        &self.activations
    }

    /// Extract activations as `f32` vectors.
    ///
    /// Returns one `Vec<f32>` of shape `[d_model]` per layer.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if dtype conversion or flattening fails.
    pub fn to_f32_vecs(&self) -> Result<Vec<Vec<f32>>> {
        self.activations
            .iter()
            .map(|t| {
                let flat = t.flatten_all()?;
                let data: Vec<f32> = flat.to_dtype(DType::F32)?.to_vec1()?;
                Ok(data)
            })
            .collect()
    }
}

/// Stores all-position activations from a forward pass.
///
/// Unlike [`ActivationCache`] which stores only the last-token activation
/// per layer, this cache stores the full residual stream at every token
/// position. Each tensor has shape `[seq_len, d_model]`.
///
/// # Example
///
/// ```
/// use candle_mi::FullActivationCache;
/// use candle_core::{Device, Tensor};
///
/// let mut cache = FullActivationCache::with_capacity(32);
/// // shape [seq_len=10, d_model=128]
/// cache.push(Tensor::zeros((10, 128), candle_core::DType::F32, &Device::Cpu).unwrap());
///
/// // Get a single position's activation for CLT encoding
/// let act = cache.get_position(0, 5).unwrap(); // shape [d_model]
/// ```
#[derive(Debug)]
pub struct FullActivationCache {
    /// Residual stream activations per layer, each shape `[seq_len, d_model]`.
    activations: Vec<Tensor>,
}

impl FullActivationCache {
    /// Create an empty cache with capacity for `n_layers` layers.
    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            activations: Vec::with_capacity(n_layers),
        }
    }

    /// Add a layer's all-position activation to the cache.
    ///
    /// The tensor should have shape `[seq_len, d_model]`.
    pub fn push(&mut self, tensor: Tensor) {
        self.activations.push(tensor);
    }

    /// Get the full activation tensor for a specific layer.
    ///
    /// Returns shape `[seq_len, d_model]`, or `None` if the layer
    /// is not in the cache.
    pub fn get_layer(&self, layer: usize) -> Option<&Tensor> {
        self.activations.get(layer)
    }

    /// Get the activation at a specific layer and token position.
    ///
    /// Returns shape `[d_model]` — compatible with CLT `encode()`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the layer is not in the cache or
    /// the position is out of range.
    pub fn get_position(&self, layer: usize, position: usize) -> Result<Tensor> {
        let layer_tensor = self
            .activations
            .get(layer)
            .ok_or_else(|| MIError::Hook(format!("layer {layer} not in cache")))?;
        let seq_len = layer_tensor.dim(0)?;
        if position >= seq_len {
            return Err(MIError::Hook(format!(
                "position {position} out of range (seq_len={seq_len})"
            )));
        }
        Ok(layer_tensor.narrow(0, position, 1)?.squeeze(0)?)
    }

    /// Number of cached layers.
    pub const fn n_layers(&self) -> usize {
        self.activations.len()
    }

    /// Sequence length (from the first layer's tensor).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the cache is empty.
    pub fn seq_len(&self) -> Result<usize> {
        let first = self
            .activations
            .first()
            .ok_or_else(|| MIError::Hook("cache is empty".into()))?;
        Ok(first.dim(0)?)
    }

    /// Whether the cache is empty.
    pub const fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn cache_basic() {
        let device = Device::Cpu;
        let t1 = Tensor::zeros((2048,), DType::F32, &device).unwrap();
        let t2 = Tensor::zeros((2048,), DType::F32, &device).unwrap();

        let cache = ActivationCache::new(vec![t1, t2]).unwrap();

        assert_eq!(cache.n_layers(), 2);
        assert!(cache.get_layer(0).is_some());
        assert!(cache.get_layer(1).is_some());
        assert!(cache.get_layer(2).is_none());
    }

    #[test]
    fn cache_push() {
        let device = Device::Cpu;
        let mut cache = ActivationCache::with_capacity(2);

        assert!(cache.is_empty());

        let t = Tensor::zeros((2048,), DType::F32, &device).unwrap();
        cache.push(t);

        assert_eq!(cache.n_layers(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn full_cache_basic() {
        let device = Device::Cpu;
        let seq_len = 10;
        let d_model = 2304;

        let mut cache = FullActivationCache::with_capacity(2);
        assert!(cache.is_empty());

        let t1 = Tensor::zeros((seq_len, d_model), DType::F32, &device).unwrap();
        let t2 = Tensor::zeros((seq_len, d_model), DType::F32, &device).unwrap();
        cache.push(t1);
        cache.push(t2);

        assert_eq!(cache.n_layers(), 2);
        assert_eq!(cache.seq_len().unwrap(), seq_len);
        assert!(!cache.is_empty());

        // get_layer returns 2D tensor
        let layer0 = cache.get_layer(0).unwrap();
        assert_eq!(layer0.dims(), &[seq_len, d_model]);

        // get_position returns 1D tensor
        let pos = cache.get_position(0, 5).unwrap();
        assert_eq!(pos.dims(), &[d_model]);

        // out of range
        assert!(cache.get_position(0, seq_len).is_err());
        assert!(cache.get_position(5, 0).is_err());
    }
}
