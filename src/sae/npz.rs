// SPDX-License-Identifier: MIT OR Apache-2.0

//! NPZ bridge: thin adapter from [`anamnesis::parse_npz`] to candle tensors.
//!
//! `anamnesis` v0.3.0 provides a fast, framework-agnostic NPZ parser that returns
//! raw LE bytes (`HashMap<String, NpzTensor>`). This module converts those raw
//! bytes into `candle_core::Tensor` objects on a given `Device`.

use std::collections::HashMap;
use std::path::Path;

use anamnesis::{NpzDtype, NpzTensor, parse_npz};
use candle_core::{Device, Tensor};

use crate::error::{MIError, Result};

/// Convert an anamnesis [`NpzTensor`] (raw LE bytes) into a candle [`Tensor`].
///
/// `F32` data is loaded directly. `F64` is promoted to `F32` (acceptable
/// precision loss for SAE weight values).
///
/// # Shapes
///
/// Input: `NpzTensor` with shape `S` and dtype `F32` or `F64`.
/// Output: `Tensor` with shape `S` and dtype `F32`.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the dtype is neither `F32` nor `F64`.
fn npz_tensor_to_candle(npy: &NpzTensor, device: &Device) -> Result<Tensor> {
    match npy.dtype {
        NpzDtype::F32 => {
            // INDEX: chunks_exact(4) guarantees exactly 4 bytes per chunk
            #[allow(clippy::indexing_slicing)]
            let f32_data: Vec<f32> = npy
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Tensor::from_vec(f32_data, &*npy.shape, device)?)
        }
        NpzDtype::F64 => {
            // CAST: f64 → f32, precision loss acceptable for SAE weights
            #[allow(
                clippy::indexing_slicing,
                clippy::cast_possible_truncation,
                clippy::as_conversions
            )]
            let f32_data: Vec<f32> = npy
                .data
                .chunks_exact(8)
                .map(|c| {
                    // CAST: f64 → f32, precision loss acceptable for SAE weights
                    let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                    v as f32
                })
                .collect();
            Ok(Tensor::from_vec(f32_data, &*npy.shape, device)?)
        }
        // EXPLICIT: NpzDtype is #[non_exhaustive] — list known variants for
        // clippy::wildcard_enum_match_arm, plus catch-all for future additions.
        other @ (NpzDtype::Bool
        | NpzDtype::U8
        | NpzDtype::I8
        | NpzDtype::U16
        | NpzDtype::I16
        | NpzDtype::U32
        | NpzDtype::I32
        | NpzDtype::U64
        | NpzDtype::I64
        | NpzDtype::F16
        | NpzDtype::BF16
        | _) => Err(MIError::Config(format!(
            "unsupported NPZ dtype {other} for SAE weights (expected F32 or F64)"
        ))),
    }
}

/// Load all arrays from an NPZ file into a name → [`Tensor`] map.
///
/// Delegates NPZ parsing to [`anamnesis::parse_npz`], then converts each
/// [`NpzTensor`] into a candle [`Tensor`] via [`npz_tensor_to_candle`].
///
/// # Errors
///
/// Returns [`MIError::Config`] if any tensor has an unsupported dtype.
/// Returns [`MIError::Io`] if the file cannot be read.
pub fn load_npz(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let npz = parse_npz(path)?;
    let mut tensors = HashMap::with_capacity(npz.len());
    for (name, npy) in &npz {
        tensors.insert(name.clone(), npz_tensor_to_candle(npy, device)?);
    }
    Ok(tensors)
}
