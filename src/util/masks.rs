// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared attention mask utilities for candle-mi.
//!
//! Provides cached causal masks and generation masks used across
//! all model backends.
//!
//! ## Caching Strategy
//!
//! Masks are cached by `(seq_len, device_id, dtype)` to avoid recreating
//! large tensors (16MB+ for `seq_len`=2048) on every forward pass.
//! The cache uses shallow clones (Arc bump, no data copy) for efficiency.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use candle_core::{DType, Device, Tensor};

use crate::error::{MIError, Result};

/// Type alias for the causal mask cache to reduce type complexity.
type CausalMaskCache = LazyLock<Mutex<HashMap<(usize, usize, DType), Tensor>>>;

/// Cache for causal masks indexed by `(seq_len, device_ordinal, dtype)`.
///
/// Avoids recreating the same mask tensor repeatedly (saves 16MB+ per
/// forward pass for large sequences).
static CAUSAL_MASK_CACHE: CausalMaskCache = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get a numeric device identifier for cache keys.
///
/// Note: This simplified approach assumes a single device per type.
/// For multi-GPU scenarios, consider using device ordinal numbers.
const fn device_id(device: &Device) -> usize {
    match device {
        Device::Cpu => 0,
        Device::Cuda(_) => 1,
        Device::Metal(_) => 2,
    }
}

/// Type alias for the mask cache guard.
type CacheGuard = std::sync::MutexGuard<'static, HashMap<(usize, usize, DType), Tensor>>;

/// Acquire the mask cache lock, mapping poison errors to [`MIError`].
fn lock_cache() -> Result<CacheGuard> {
    CAUSAL_MASK_CACHE
        .lock()
        .map_err(|e| MIError::Hook(format!("mask cache lock poisoned: {e}")))
}

/// Create or retrieve a cached causal mask for the given sequence length.
///
/// Returns a tensor of shape `[1, 1, seq_len, seq_len]` where:
/// - `0.0` for positions that can attend (`j <= i`)
/// - `-inf` for positions that cannot attend (`j > i`)
///
/// # Shapes
///
/// - returns: `[1, 1, seq_len, seq_len]`
///
/// # Example
///
/// For `seq_len`=4, the mask looks like:
/// ```text
/// [[[[0, -inf, -inf, -inf],
///    [0,    0, -inf, -inf],
///    [0,    0,    0, -inf],
///    [0,    0,    0,    0]]]]
/// ```
///
/// # Errors
///
/// Returns [`MIError::Model`] if tensor creation fails, or
/// [`MIError::Hook`] if the cache lock is poisoned.
pub fn create_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let cache_key = (seq_len, device_id(device), dtype);

    // Try to get from cache first.
    {
        let cache = lock_cache()?;
        if let Some(cached) = cache.get(&cache_key) {
            return Ok(cached.clone()); // Shallow clone (Arc bump, no data copy)
        }
    }

    // Create new mask.
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    let mask_tensor = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)?;

    // Store in cache.
    {
        let mut cache = lock_cache()?;
        cache.insert(cache_key, mask_tensor.clone());
    }

    Ok(mask_tensor)
}

/// Create a causal mask for generation with KV-cache.
///
/// During generation with cache:
/// - `new_seq_len`: Number of new tokens being processed (usually 1)
/// - `total_seq_len`: Total sequence length (cached + new)
/// - `start_pos`: Starting position (equals cached sequence length)
///
/// The mask allows each new position to attend to:
/// - All cached positions (`0..start_pos`)
/// - All positions up to and including itself among new tokens
///
/// # Shapes
///
/// - returns: `[1, 1, new_seq_len, total_seq_len]`
///
/// # Special Case
///
/// For single token generation (`new_seq_len == 1`), returns an all-zeros
/// mask since the new token can see the entire cached context.
///
/// # Errors
///
/// Returns [`MIError::Model`] if tensor creation fails.
pub fn create_generation_mask(
    new_seq_len: usize,
    total_seq_len: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // Single token generation (most common case): all positions visible.
    if new_seq_len == 1 {
        let mask = Tensor::zeros((1, 1, 1, total_seq_len), dtype, device)?;
        return Ok(mask);
    }

    // Multi-token generation (e.g., prompt processing without cache).
    // Create mask where new_token[i] can see positions 0..(start_pos + i + 1).
    let mask: Vec<f32> = (0..new_seq_len)
        .flat_map(|i| {
            let visible_up_to = start_pos + i;
            (0..total_seq_len).map(move |j| {
                if j <= visible_up_to {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();

    let mask_tensor =
        Tensor::from_vec(mask, (1, 1, new_seq_len, total_seq_len), device)?.to_dtype(dtype)?;

    Ok(mask_tensor)
}

/// Clear all cached masks.
///
/// Useful for memory management in long-running applications or
/// when switching between different sequence lengths.
///
/// # Errors
///
/// Returns [`MIError::Hook`] if the cache lock is poisoned.
pub fn clear_mask_caches() -> Result<()> {
    lock_cache()?.clear();
    Ok(())
}

/// Get the current number of cached masks.
///
/// Useful for debugging and memory profiling.
///
/// # Errors
///
/// Returns [`MIError::Hook`] if the cache lock is poisoned.
pub fn mask_cache_size() -> Result<usize> {
    Ok(lock_cache()?.len())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mask = create_causal_mask(4, &device, dtype).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn causal_mask_values() {
        clear_mask_caches().unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mask = create_causal_mask(3, &device, dtype).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Row 0: [0, -inf, -inf]
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite() && data[1] < 0.0);
        assert!(data[2].is_infinite() && data[2] < 0.0);

        // Row 1: [0, 0, -inf]
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert!(data[5].is_infinite() && data[5] < 0.0);

        // Row 2: [0, 0, 0]
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
    }

    #[test]
    fn generation_mask_single_token() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mask = create_generation_mask(1, 5, 4, &device, dtype).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 1, 5]);

        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn generation_mask_multi_token() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // 2 new tokens, 3 cached + 2 new = 5 total
        let mask = create_generation_mask(2, 5, 3, &device, dtype).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 2, 5]);

        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Row 0 (new token at pos 3): can see 0,1,2,3, not 4
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 0.0);
        assert!(data[4].is_infinite() && data[4] < 0.0);

        // Row 1 (new token at pos 4): can see 0,1,2,3,4
        assert_eq!(data[5], 0.0);
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
        assert_eq!(data[9], 0.0);
    }
}
