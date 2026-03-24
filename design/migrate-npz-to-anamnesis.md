# Design: Migrate NPZ Parsing from Internal to anamnesis

**Status:** Implemented **|** **Date:** March 24, 2026 **|** **Relates to:** candle-mi v0.1.5 **|** **Origin:** anamnesis v0.3.0 now provides a fast, framework-agnostic NPZ parser

## Question

Should candle-mi remove its internal NPZ/NPY parser (`src/sae/npz.rs`, 415 lines) and depend on `anamnesis` for NPZ parsing?

## Context

candle-mi has a custom NPZ parser tightly coupled to candle: it parses NPY headers by hand, handles ZIP extraction via the `zip` crate, and converts raw bytes into `candle_core::Tensor` objects on a `Device`. This parser only supports F32 and F64, assumes little-endian, and rejects Fortran-order arrays.

anamnesis v0.3.0 now provides `parse_npz()` — a framework-agnostic NPZ parser that:
- Returns raw LE bytes (`HashMap<String, NpzTensor>`) instead of framework-specific tensors
- Supports all ML-relevant dtypes (F16, BF16, F32, F64, all integer types, Bool)
- Handles big-endian NPY files (byte-swap on read)
- Is **4.9× faster** than candle-mi's parser on the same 302 MB Gemma Scope file (84 ms vs 413 ms, approaching the 64 ms I/O floor)

The separation of concerns is correct: anamnesis parses the format, candle-mi places tensors on a device.

## Recommendation

### 1. Add `anamnesis` dependency (feature-gated)

In `Cargo.toml`:
```toml
anamnesis = { version = "0.3", features = ["npz"], optional = true }
```

Gate it behind the existing `sae` feature (or a new `npz` feature if preferred):
```toml
sae = ["dep:anamnesis"]
```

### 2. Remove the `zip` direct dependency

`zip` is no longer needed — anamnesis handles ZIP extraction internally.

### 3. Replace `src/sae/npz.rs` (~415 → ~50 lines)

The entire custom NPY parser (magic check, version dispatch, Python dict parser, `split_dict_entries`, `parse_shape_tuple`, `npy_to_tensor`) is replaced by a thin bridge:

```rust
use std::collections::HashMap;
use std::path::Path;

use anamnesis::{parse_npz, NpzDtype, NpzTensor};
use candle_core::{Device, Tensor};

use crate::error::{MIError, Result};

/// Convert an anamnesis `NpzTensor` (raw LE bytes) into a candle `Tensor`.
///
/// `F32` data is loaded directly. `F64` is promoted to `F32` (acceptable
/// precision loss for SAE weight values).
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
            #[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
            let f32_data: Vec<f32> = npy
                .data
                .chunks_exact(8)
                .map(|c| {
                    let v = f64::from_le_bytes([
                        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                    ]);
                    v as f32
                })
                .collect();
            Ok(Tensor::from_vec(f32_data, &*npy.shape, device)?)
        }
        other => Err(MIError::Config(format!(
            "unsupported NPZ dtype {other} for SAE weights (expected F32 or F64)"
        ))),
    }
}

/// Load all arrays from an NPZ file into a name → `Tensor` map.
pub fn load_npz(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let npz = parse_npz(path)?;
    let mut tensors = HashMap::with_capacity(npz.len());
    for (name, npy) in &npz {
        tensors.insert(name.clone(), npz_tensor_to_candle(npy, device)?);
    }
    Ok(tensors)
}
```

### 4. Add `From<AnamnesisError>` in `src/error.rs`

```rust
impl From<anamnesis::AnamnesisError> for MIError {
    fn from(e: anamnesis::AnamnesisError) -> Self {
        match e {
            anamnesis::AnamnesisError::Parse { reason } => Self::Config(reason),
            anamnesis::AnamnesisError::Unsupported { format, detail } => {
                Self::Config(format!("unsupported {format}: {detail}"))
            }
            anamnesis::AnamnesisError::Io(io_err) => Self::Io(io_err),
            // AnamnesisError is #[non_exhaustive] — forward-compatible catch-all
            _ => Self::Config(e.to_string()),
        }
    }
}
```

### 5. No changes to `from_npz` callers

`load_npz` keeps the same signature: `(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>>`. The `SparseAutoencoder::from_npz` method and `from_pretrained_npz` are completely unaffected.

## What gets deleted

| File / Item | Lines | Reason |
|---|---|---|
| `parse_npy()` | 52–158 | Replaced by anamnesis `parse_npy_header` |
| `parse_npy_header()` | 163–205 | Replaced by anamnesis header extraction |
| `split_dict_entries()` | 208–228 | No longer needed |
| `parse_shape_tuple()` | 231–247 | No longer needed |
| `npy_to_tensor()` | 252–286 | Replaced by `npz_tensor_to_candle` |
| `NpyArray` struct | 40–49 | Replaced by `anamnesis::NpzTensor` |
| `NPY_MAGIC` const | 37 | Internal to anamnesis |
| Unit tests (5) | 341–413 | Covered by anamnesis's own test suite |
| **Total removed** | **~365 lines** | |

## Performance impact

| Metric | Before (candle-mi internal) | After (anamnesis) |
|---|---|---|
| Gemma Scope 302 MB parse | 413 ms | 84 ms |
| Throughput | 731 MB/s | 3,586 MB/s |
| Improvement | — | **4.9× faster** |

The `npz_tensor_to_candle` bridge adds a `from_le_bytes` per-element pass to build `Vec<f32>`, but this is unavoidable given candle's `Tensor::from_vec<f32>` API. On LE machines, `f32::from_le_bytes` compiles to a no-op and the iterator becomes a memcpy.

## Risk assessment

- **API compatibility:** Zero risk — `load_npz` signature unchanged, `from_npz` callers unaffected.
- **Correctness:** anamnesis's cross-validation test verifies byte-exact match against NumPy on the same Gemma Scope file candle-mi uses.
- **Dependency weight:** anamnesis v0.3.0 with `npz` feature pulls `zip` v2 + `flate2` — the same dependency candle-mi already has directly. Net dependency count is unchanged.
- **MSRV:** Both projects require Rust 1.88+. No conflict.

## Bundling with v0.1.5

This migration targets candle-mi v0.1.5. The `Intervention::AddAtPositions` feature (see `design/add-at-positions.md`) may follow in v0.1.6.
