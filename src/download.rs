// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast model download via [`hf-fetch-model`](https://github.com/PCfVW/hf-fetch-model).
//!
//! This module provides [`download_model()`] and [`download_model_blocking()`],
//! convenience functions for downloading `HuggingFace` model repositories
//! with maximum throughput. Downloaded files are stored in the standard
//! `HuggingFace` cache directory (`~/.cache/huggingface/hub/`), ensuring
//! compatibility with [`MIModel::from_pretrained()`](crate::MIModel::from_pretrained).
//!
//! Progress is displayed via `indicatif` progress bars: per-file bars show
//! bytes, throughput, and ETA; an overall bar tracks completed files.
//!
//! # Usage pattern
//!
//! ```rust,no_run
//! # async fn example() -> candle_mi::Result<()> {
//! // 1. Pre-download the model (fast, async, with progress bars)
//! let path = candle_mi::download_model("meta-llama/Llama-3.2-1B".to_owned()).await?;
//!
//! // 2. Load from cache (sync, no network needed)
//! let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use crate::error::{MIError, Result};

/// Returns a pre-configured [`hf_fetch_model::FetchConfigBuilder`] that reads
/// `HF_TOKEN` from the environment for gated/private `HuggingFace` repos.
///
/// hf-fetch-model 0.9.x requires explicit opt-in via `.token_from_env()` — the
/// public `download_files(...)` convenience wrappers build a no-token default
/// config that silently fails 401 on gated models (Llama, Mistral, Gemma,
/// Qwen, etc.). Every candle-mi call site that downloads from HF should start
/// from this helper so the token handling stays uniform.
///
/// Callers can chain further configuration (`.on_progress(...)`, etc.) before
/// calling `.build()`.
#[must_use]
pub fn fetch_config_builder() -> hf_fetch_model::FetchConfigBuilder {
    hf_fetch_model::FetchConfig::builder().token_from_env()
}

/// Downloads all files from a `HuggingFace` model repository.
///
/// Uses high-throughput parallel downloads for maximum speed. Files are
/// stored in the standard `HuggingFace` cache layout
/// (`~/.cache/huggingface/hub/`), so a subsequent call to
/// [`MIModel::from_pretrained()`](crate::MIModel::from_pretrained)
/// finds them without re-downloading.
///
/// Progress is displayed via `indicatif` progress bars showing per-file
/// bytes, throughput, and ETA.
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
    let progress = hf_fetch_model::progress::IndicatifProgress::new();

    let config = fetch_config_builder()
        .on_progress(move |event| progress.handle(event))
        .build()
        .map_err(|e| MIError::Download(e.to_string()))?;

    hf_fetch_model::download_with_config(repo_id, &config)
        .await
        .map(hf_fetch_model::DownloadOutcome::into_inner)
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
