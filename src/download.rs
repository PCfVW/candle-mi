// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast model download via [`hf-fetch-model`](https://github.com/PCfVW/hf-fetch-model).
//!
//! This module provides [`download_model()`] and [`download_model_blocking()`],
//! convenience functions for downloading `HuggingFace` model repositories
//! with maximum throughput. Downloaded files are stored in the standard
//! `HuggingFace` cache directory (`~/.cache/huggingface/hub/`), ensuring
//! compatibility with [`MIModel::from_pretrained()`](crate::MIModel::from_pretrained).
//!
//! # Feature gate
//!
//! Requires `features = ["fast-download"]`:
//!
//! ```toml
//! candle-mi = { version = "...", features = ["fast-download"] }
//! ```
//!
//! # Usage pattern
//!
//! ```rust,no_run
//! # async fn example() -> candle_mi::Result<()> {
//! // 1. Pre-download the model (fast, async, with progress via tracing)
//! let path = candle_mi::download_model("meta-llama/Llama-3.2-1B".to_owned()).await?;
//! tracing::info!("model cached at {}", path.display());
//!
//! // 2. Load from cache (sync, no network needed)
//! let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use crate::error::{MIError, Result};

/// Downloads all files from a `HuggingFace` model repository.
///
/// Uses high-throughput parallel downloads for maximum speed. Files are
/// stored in the standard `HuggingFace` cache layout
/// (`~/.cache/huggingface/hub/`), so a subsequent call to
/// [`MIModel::from_pretrained()`](crate::MIModel::from_pretrained)
/// finds them without re-downloading.
///
/// Progress is reported via `tracing` at `info` level. To see progress
/// output, initialise a tracing subscriber before calling this function.
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"meta-llama/Llama-3.2-1B"`).
///
/// # Errors
///
/// Returns [`MIError::Download`] if the download fails for any reason
/// (network, authentication, repository not found, checksum mismatch).
pub async fn download_model(repo_id: String) -> Result<PathBuf> {
    let config = hf_fetch_model::FetchConfig::builder()
        .on_progress(|event| {
            tracing::info!(
                filename = %event.filename,
                percent = event.percent,
                bytes_downloaded = event.bytes_downloaded,
                bytes_total = event.bytes_total,
                files_remaining = event.files_remaining,
                "download progress",
            );
        })
        .build()
        .map_err(|e| MIError::Download(e.to_string()))?;

    hf_fetch_model::download_with_config(repo_id, &config)
        .await
        .map_err(|e| MIError::Download(e.to_string()))
}

/// Blocking version of [`download_model()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do **not** call from within an
/// existing async context (use [`download_model()`] instead).
///
/// # Arguments
///
/// * `repo_id` — The repository identifier (e.g., `"meta-llama/Llama-3.2-1B"`).
///
/// # Errors
///
/// Returns [`MIError::Download`] if the download or runtime creation fails.
pub fn download_model_blocking(repo_id: String) -> Result<PathBuf> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| MIError::Download(format!("failed to create tokio runtime: {e}")))?;
    rt.block_on(download_model(repo_id))
}
