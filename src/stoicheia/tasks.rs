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
