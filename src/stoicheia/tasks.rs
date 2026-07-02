// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ground-truth task functions for `AlgZoo` model validation.
//!
//! These pure functions compute the correct answer for each `AlgZoo` task,
//! used to verify model predictions and measure accuracy.

use candle_core::{IndexOp, Tensor};

use crate::error::Result;

/// Compute the position of the second-largest value in each row.
///
/// # Shapes
/// - `input`: `[batch, seq_len]` (continuous floats)
/// - returns: `[batch]` (position indices, dtype U32)
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor operation failure.
pub fn second_argmax(input: &Tensor) -> Result<Tensor> {
    // argsort descending, take index 1 (second-largest position)
    let sorted_indices = input.arg_sort_last_dim(false)?;
    // INDEX: column 1 exists because seq_len >= 2 for this task
    let second = sorted_indices.i((.., 1))?;
    Ok(second)
}

/// Compute the position of the median value in each row.
///
/// # Shapes
/// - `input`: `[batch, seq_len]` (continuous floats)
/// - returns: `[batch]` (position indices, dtype U32)
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor operation failure.
pub fn argmedian(input: &Tensor) -> Result<Tensor> {
    let (_, seq_len) = input.dims2()?;
    let sorted_indices = input.arg_sort_last_dim(true)?;
    // INDEX: seq_len/2 is valid because seq_len >= 2
    let median_idx = seq_len / 2;
    let median_pos = sorted_indices.i((.., median_idx))?;
    Ok(median_pos)
}

/// Compute the median value of each row.
///
/// # Shapes
/// - `input`: `[batch, seq_len]` (continuous floats)
/// - returns: `[batch]` (scalar values, dtype F32)
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor operation failure.
pub fn median(input: &Tensor) -> Result<Tensor> {
    let (_, seq_len) = input.dims2()?;
    let (sorted, _indices) = input.sort_last_dim(true)?;
    let median_idx = seq_len / 2;
    // INDEX: median_idx < seq_len because seq_len >= 1
    let median_val = sorted.i((.., median_idx))?;
    Ok(median_val)
}

/// Compute the longest cycle length in each row's permutation.
///
/// Each row represents a function `f: {0..n-1} → {0..n-1}`.
/// The longest cycle is found by following chains `x → f(x) → f(f(x)) → ...`
///
/// # Shapes
/// - `input`: `[batch, seq_len]` (integers in `0..seq_len`, dtype U32 or I64)
/// - returns: `[batch]` (cycle lengths, dtype U32)
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor operation failure.
pub fn longest_cycle(input: &Tensor) -> Result<Tensor> {
    // This requires iterative graph traversal — do it on CPU
    let input_vec: Vec<Vec<u32>> = input.to_vec2()?;
    let mut results = Vec::with_capacity(input_vec.len());

    for row in &input_vec {
        let n = row.len();
        let mut visited = vec![false; n];
        let mut max_cycle = 0_u32;

        for start in 0..n {
            // INDEX: start is bounded by n (0..n loop)
            #[allow(clippy::indexing_slicing)]
            if visited[start] {
                continue;
            }
            let mut pos = start;
            let mut cycle_len = 0_u32;
            // EXPLICIT: graph traversal requires imperative loop
            #[allow(clippy::indexing_slicing)]
            while !visited[pos] {
                // INDEX: pos is bounded by n (initialized to start < n,
                // then set to row[pos] which is in 0..n by task definition)
                visited[pos] = true;
                // CAST: u32 → usize, values are small indices in 0..n
                #[allow(clippy::as_conversions)]
                let next = row[pos] as usize;
                pos = next;
                cycle_len += 1;
            }
            if cycle_len > max_cycle {
                max_cycle = cycle_len;
            }
        }
        results.push(max_cycle);
    }

    Ok(Tensor::new(&results[..], input.device())?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Build a `[batch, seq_len]` U32 tensor from a flat row-major vec.
    fn u32_tensor(rows: &[u32], batch: usize, seq_len: usize) -> Tensor {
        Tensor::from_vec(rows.to_vec(), (batch, seq_len), &Device::Cpu).unwrap()
    }

    #[test]
    fn longest_cycle_identity_is_one() {
        // f(x) = x for all x → every element is a fixed point (cycle length 1).
        let input = u32_tensor(&[0, 1, 2, 3], 1, 4);
        let out: Vec<u32> = longest_cycle(&input).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn longest_cycle_single_full_cycle() {
        // 0→1→2→3→0 is one 4-cycle.
        let input = u32_tensor(&[1, 2, 3, 0], 1, 4);
        let out: Vec<u32> = longest_cycle(&input).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![4]);
    }

    #[test]
    fn longest_cycle_two_two_cycles() {
        // 0↔1 and 2↔3 → the longest cycle is length 2.
        let input = u32_tensor(&[1, 0, 3, 2], 1, 4);
        let out: Vec<u32> = longest_cycle(&input).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn longest_cycle_mixed_takes_the_max() {
        // 0→1→2→0 (3-cycle) plus fixed point 3 → longest is 3.
        let input = u32_tensor(&[1, 2, 0, 3], 1, 4);
        let out: Vec<u32> = longest_cycle(&input).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![3]);
    }

    #[test]
    fn longest_cycle_batches_independently() {
        // Row 0: identity (1); row 1: full 4-cycle (4); row 2: two 2-cycles (2).
        let input = u32_tensor(&[0, 1, 2, 3, 1, 2, 3, 0, 1, 0, 3, 2], 3, 4);
        let out: Vec<u32> = longest_cycle(&input).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![1, 4, 2]);
    }

    #[test]
    fn second_argmax_picks_runner_up_position() {
        // Values 0.1, 0.9, 0.5, 0.3 → desc order positions [1, 2, 3, 0];
        // the second-largest is at position 2.
        let input = Tensor::new(&[[0.1_f32, 0.9, 0.5, 0.3]], &Device::Cpu).unwrap();
        let out: Vec<u32> = second_argmax(&input).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn median_and_argmedian_agree() {
        // Ascending sort of [1, 3, 2, 4] is [1, 2, 3, 4]; median index = 4/2 = 2 →
        // value 3.0 at original position 1.
        let input = Tensor::new(&[[1.0_f32, 3.0, 2.0, 4.0]], &Device::Cpu).unwrap();
        let med: Vec<f32> = median(&input).unwrap().to_vec1().unwrap();
        let pos: Vec<u32> = argmedian(&input).unwrap().to_vec1().unwrap();
        assert!((med[0] - 3.0).abs() < 1e-6, "median = {}", med[0]);
        assert_eq!(pos, vec![1]);
    }
}
