// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for candle-mi.

/// Errors that can occur during MI operations.
///
/// This enum is `#[non_exhaustive]`: new variants will be added in future
/// releases as new backends and capabilities are added.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum MIError {
    /// Model loading or forward pass error (wraps candle).
    #[error("model error: {0}")]
    Model(#[from] candle_core::Error),

    /// Hook capture or lookup error.
    #[error("hook error: {0}")]
    Hook(String),

    /// Intervention validation or application error.
    #[error("intervention error: {0}")]
    Intervention(String),

    /// Model configuration parsing error.
    #[error("config error: {0}")]
    Config(String),

    /// Tokenizer error.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Model download error.
    ///
    /// Returned when downloading a model from the `HuggingFace` Hub fails.
    #[error("download error: {0}")]
    Download(String),

    /// Memory measurement error.
    ///
    /// Returned when a platform API for RAM or VRAM measurement fails.
    #[error("memory error: {0}")]
    Memory(String),
}

/// Bridge anamnesis errors into [`MIError`] when the `sae` feature is enabled.
///
/// [`AnamnesisError`](anamnesis::AnamnesisError) is `#[non_exhaustive]`, so the
/// catch-all arm ensures forward compatibility with future variants.
#[cfg(feature = "sae")]
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

/// Result type alias for candle-mi operations.
pub type Result<T> = std::result::Result<T, MIError>;
