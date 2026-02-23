// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for candle-mi.

/// Errors that can occur during MI operations.
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
}

/// Result type alias for candle-mi operations.
pub type Result<T> = std::result::Result<T, MIError>;
