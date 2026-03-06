// SPDX-License-Identifier: MIT OR Apache-2.0

//! NPZ/NPY parser for loading SAE weights from `NumPy` archive files.
//!
//! NPZ is a ZIP archive containing `.npy` files. Each `.npy` file stores a
//! single tensor with a small header describing dtype, shape, and memory order.
//!
//! This parser supports the subset of NPY needed for SAE weights:
//! - NPY format versions 1.0 and 2.0
//! - `float32` (`<f4`) and `float64` (`<f8`) dtypes (promoted to F32)
//! - C-order (row-major) arrays
//!
//! # NPY Binary Format
//!
//! ```text
//! Offset  Size    Field
//! 0       6       Magic: \x93NUMPY
//! 6       1       Major version (1 or 2)
//! 7       1       Minor version (0)
//! 8       2|4     Header length (LE): 2 bytes for v1, 4 bytes for v2
//! 10|12   N       Header: Python dict literal (ASCII) padded to 64-byte alignment
//! 10+N|12+N ...   Raw data (contiguous, dtype-sized elements)
//! ```
//!
//! The header is a Python dict literal, e.g.:
//! `{'descr': '<f4', 'fortran_order': False, 'shape': (2304, 16384), }`

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::error::{MIError, Result};

/// NPY magic bytes: `\x93NUMPY`.
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// A parsed NPY array: shape, dtype info, and raw bytes.
struct NpyArray {
    /// Tensor shape (e.g., `[2304, 16384]`).
    shape: Vec<usize>,
    /// Bytes per element (4 for f32, 8 for f64).
    bytes_per_element: usize,
    /// Whether the source is f64 (needs conversion to f32).
    is_f64: bool,
    /// Raw data bytes (in source dtype).
    data: Vec<u8>,
}

/// Parse a single `.npy` byte stream into an [`NpyArray`].
fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    // Check magic.
    if bytes.len() < 10 {
        return Err(MIError::Config("NPY file too short for header".into()));
    }
    let magic = bytes
        .get(..6)
        .ok_or_else(|| MIError::Config("NPY file too short".into()))?;
    if magic != NPY_MAGIC {
        return Err(MIError::Config("invalid NPY magic bytes".into()));
    }

    let major = *bytes
        .get(6)
        .ok_or_else(|| MIError::Config("NPY: missing major version".into()))?;
    // minor at index 7, unused

    // Header length and data offset depend on version.
    let (header_len, data_offset) = match major {
        1 => {
            // 2-byte LE header length at offset 8.
            let lo = *bytes
                .get(8)
                .ok_or_else(|| MIError::Config("NPY v1: truncated header length".into()))?;
            let hi = *bytes
                .get(9)
                .ok_or_else(|| MIError::Config("NPY v1: truncated header length".into()))?;
            let len = u16::from_le_bytes([lo, hi]);
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            (len as usize, 10_usize + len as usize)
        }
        2 | 3 => {
            // 4-byte LE header length at offset 8.
            let b = bytes
                .get(8..12)
                .ok_or_else(|| MIError::Config("NPY v2: truncated header length".into()))?;
            let len = u32::from_le_bytes([
                *b.first()
                    .ok_or_else(|| MIError::Config("NPY v2: missing header byte".into()))?,
                *b.get(1)
                    .ok_or_else(|| MIError::Config("NPY v2: missing header byte".into()))?,
                *b.get(2)
                    .ok_or_else(|| MIError::Config("NPY v2: missing header byte".into()))?,
                *b.get(3)
                    .ok_or_else(|| MIError::Config("NPY v2: missing header byte".into()))?,
            ]);
            #[allow(clippy::as_conversions)]
            (len as usize, 12_usize + len as usize)
        }
        _ => {
            return Err(MIError::Config(format!("unsupported NPY version {major}")));
        }
    };

    // Extract header string.
    let header_bytes = bytes
        .get(data_offset - header_len..data_offset)
        .ok_or_else(|| MIError::Config("NPY header extends past end of file".into()))?;
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|e| MIError::Config(format!("NPY header is not valid UTF-8: {e}")))?;

    // Parse header fields.
    let (descr, fortran_order, shape) = parse_npy_header(header_str)?;

    if fortran_order {
        return Err(MIError::Config(
            "NPY Fortran-order arrays not supported (expected C-order)".into(),
        ));
    }

    let (bytes_per_element, is_f64) = match descr.as_str() {
        "<f4" | "=f4" | "f4" => (4, false),
        "<f8" | "=f8" | "f8" => (8, true),
        _ => {
            return Err(MIError::Config(format!(
                "unsupported NPY dtype '{descr}' (expected float32 or float64)"
            )));
        }
    };

    let data = bytes
        .get(data_offset..)
        .ok_or_else(|| MIError::Config("NPY data starts past end of file".into()))?;

    // Validate data size.
    let n_elements: usize = shape.iter().product();
    let expected_bytes = n_elements * bytes_per_element;
    if data.len() < expected_bytes {
        return Err(MIError::Config(format!(
            "NPY data too short: expected {expected_bytes} bytes for shape {shape:?}, got {}",
            data.len()
        )));
    }

    Ok(NpyArray {
        shape,
        bytes_per_element,
        is_f64,
        data: data
            .get(..expected_bytes)
            .ok_or_else(|| MIError::Config("NPY data slice out of bounds".into()))?
            .to_vec(),
    })
}

/// Parse the Python dict header string into (descr, `fortran_order`, shape).
///
/// Example header: `{'descr': '<f4', 'fortran_order': False, 'shape': (2304, 16384), }`
fn parse_npy_header(header: &str) -> Result<(String, bool, Vec<usize>)> {
    let trimmed = header.trim();
    // Strip outer braces.
    let inner = trimmed
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| MIError::Config(format!("NPY header not a dict: {trimmed}")))?
        .trim();

    let mut descr: Option<String> = None;
    let mut fortran_order = false;
    let mut shape: Option<Vec<usize>> = None;

    // Simple key-value parser for the Python dict literal.
    // Split on commas that are not inside parentheses.
    for kv in split_dict_entries(inner) {
        let kv = kv.trim();
        if kv.is_empty() {
            continue;
        }
        if let Some((key, value)) = kv.split_once(':') {
            let key = key.trim().trim_matches('\'').trim_matches('"');
            let value = value.trim();
            match key {
                "descr" => {
                    descr = Some(value.trim_matches('\'').trim_matches('"').to_owned());
                }
                "fortran_order" => {
                    fortran_order = value == "True";
                }
                "shape" => {
                    shape = Some(parse_shape_tuple(value)?);
                }
                _ => {} // Ignore unknown keys.
            }
        }
    }

    let descr = descr.ok_or_else(|| MIError::Config("NPY header missing 'descr' field".into()))?;
    let shape = shape.ok_or_else(|| MIError::Config("NPY header missing 'shape' field".into()))?;

    Ok((descr, fortran_order, shape))
}

/// Split a Python dict body on top-level commas (not inside parentheses).
fn split_dict_entries(s: &str) -> Vec<&str> {
    let mut entries = Vec::new();
    let mut depth = 0_usize;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                entries.push(s.get(start..i).unwrap_or(""));
                start = i + 1;
            }
            _ => {}
        }
    }
    // Last segment (may be empty if trailing comma).
    if start < s.len() {
        entries.push(s.get(start..).unwrap_or(""));
    }
    entries
}

/// Parse a Python shape tuple like `(2304, 16384)` or `(16384,)`.
fn parse_shape_tuple(s: &str) -> Result<Vec<usize>> {
    let inner = s
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| MIError::Config(format!("NPY shape not a tuple: {s}")))?;

    inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| MIError::Config(format!("NPY shape element parse error: {e}")))
        })
        .collect()
}

/// Convert an [`NpyArray`] into a candle [`Tensor`] on the given device.
///
/// F64 arrays are converted to F32. F32 arrays are loaded directly.
fn npy_to_tensor(npy: &NpyArray, device: &Device) -> Result<Tensor> {
    let n_elements: usize = npy.shape.iter().product();
    let mut f32_data = Vec::with_capacity(n_elements);

    for i in 0..n_elements {
        let start = i * npy.bytes_per_element;
        let val_f32 = if npy.is_f64 {
            let chunk = npy
                .data
                .get(start..start + 8)
                .ok_or_else(|| MIError::Config("NPY f64 data truncated".into()))?;
            // PROMOTE: f64 → f32 (precision loss is expected and acceptable)
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            let v = f64::from_le_bytes(
                <[u8; 8]>::try_from(chunk)
                    .map_err(|e| MIError::Config(format!("NPY f64 slice error: {e}")))?,
            ) as f32;
            v
        } else {
            let chunk = npy
                .data
                .get(start..start + 4)
                .ok_or_else(|| MIError::Config("NPY f32 data truncated".into()))?;
            f32::from_le_bytes(
                <[u8; 4]>::try_from(chunk)
                    .map_err(|e| MIError::Config(format!("NPY f32 slice error: {e}")))?,
            )
        };
        f32_data.push(val_f32);
    }

    let tensor = Tensor::from_vec(f32_data, &*npy.shape, device)?;
    Ok(tensor)
}

/// Load all arrays from an NPZ file into a name → [`Tensor`] map.
///
/// NPZ is a ZIP archive where each entry `{name}.npy` stores one array.
/// The `.npy` suffix is stripped from the key names.
///
/// # Errors
///
/// Returns [`MIError::Io`] if the file cannot be read.
/// Returns [`MIError::Config`] if any `.npy` entry is malformed.
pub fn load_npz(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| MIError::Config(format!("failed to open NPZ archive: {e}")))?;

    let mut tensors = HashMap::new();

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| MIError::Config(format!("failed to read NPZ entry {i}: {e}")))?;

        let name = entry.name().to_owned();
        if !std::path::Path::new(&name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("npy"))
        {
            continue;
        }

        // Read entire entry into memory.
        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .map_err(|e| MIError::Io(std::io::Error::other(e)))?;

        let npy = parse_npy(&buf)?;
        let tensor = npy_to_tensor(&npy, device)?;

        // Strip .npy suffix for the key name.
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_owned();
        tensors.insert(key, tensor);
    }

    Ok(tensors)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_header_basic() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2304, 16384), }";
        let (descr, fortran, shape) = parse_npy_header(header).unwrap();
        assert_eq!(descr, "<f4");
        assert!(!fortran);
        assert_eq!(shape, vec![2304, 16384]);
    }

    #[test]
    fn parse_header_1d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (16384,), }";
        let (_, _, shape) = parse_npy_header(header).unwrap();
        assert_eq!(shape, vec![16384]);
    }

    #[test]
    fn parse_header_f64() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (100,), }";
        let (descr, _, shape) = parse_npy_header(header).unwrap();
        assert_eq!(descr, "<f8");
        assert_eq!(shape, vec![100]);
    }

    #[test]
    fn parse_shape_tuple_basic() {
        assert_eq!(parse_shape_tuple("(3, 4)").unwrap(), vec![3, 4]);
        assert_eq!(parse_shape_tuple("(10,)").unwrap(), vec![10]);
        assert_eq!(parse_shape_tuple("(2, 3, 4)").unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn roundtrip_f32_npy() {
        // Build a minimal NPY v1 file with 4 f32 values.
        let shape = vec![2, 2];
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }";
        // Pad header to 64-byte alignment.
        let header_bytes = header.as_bytes();
        let total_before_pad = 10 + header_bytes.len();
        let padding = (64 - (total_before_pad % 64)) % 64;
        let padded_len = header_bytes.len() + padding;

        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(1); // major
        npy.push(0); // minor
        #[allow(clippy::cast_possible_truncation)]
        let len_bytes = (padded_len as u16).to_le_bytes();
        npy.extend_from_slice(&len_bytes);
        npy.extend_from_slice(header_bytes);
        // Pad with spaces, ending with newline.
        for _ in 0..padding.saturating_sub(1) {
            npy.push(b' ');
        }
        if padding > 0 {
            npy.push(b'\n');
        }
        // Data.
        for v in &values {
            npy.extend_from_slice(&v.to_le_bytes());
        }

        let parsed = parse_npy(&npy).unwrap();
        assert_eq!(parsed.shape, shape);
        assert!(!parsed.is_f64);

        let tensor = npy_to_tensor(&parsed, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[2, 2]);
        let data: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, values);
    }
}
