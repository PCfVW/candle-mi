// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizer abstraction: dispatch between `HuggingFace` and RWKV backends.
//!
//! [`MITokenizer`] provides a unified encode/decode interface regardless of
//! the underlying tokenizer implementation.

#[cfg(feature = "rwkv-tokenizer")]
pub mod rwkv;

use crate::error::{MIError, Result};

/// Unified tokenizer supporting multiple backends.
///
/// Most models use the `HuggingFace` `tokenizers` crate. RWKV-6 models
/// ship their own vocabulary format and require a custom trie-based
/// tokenizer, which is available behind the `rwkv-tokenizer` feature.
///
/// # Example
///
/// ```no_run
/// use candle_mi::MITokenizer;
///
/// # fn main() -> candle_mi::Result<()> {
/// let tok = MITokenizer::from_hf_path("tokenizer.json")?;
/// let ids = tok.encode("fn main()")?;
/// let text = tok.decode(&ids)?;
/// assert!(!ids.is_empty());
/// # Ok(())
/// # }
/// ```
#[non_exhaustive]
pub enum MITokenizer {
    /// `HuggingFace` `tokenizers` backend.
    HuggingFace(Box<tokenizers::Tokenizer>),
    /// RWKV World tokenizer (trie-based greedy longest-match).
    #[cfg(feature = "rwkv-tokenizer")]
    Rwkv(rwkv::RwkvTokenizer),
}

impl MITokenizer {
    /// Load a `HuggingFace` tokenizer from a `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if the file cannot be loaded or parsed.
    pub fn from_hf_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| {
            MIError::Tokenizer(format!(
                "failed to load HF tokenizer from {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Ok(Self::HuggingFace(Box::new(tok)))
    }

    /// Wrap an already-loaded `HuggingFace` tokenizer.
    pub fn from_hf(tokenizer: tokenizers::Tokenizer) -> Self {
        Self::HuggingFace(Box::new(tokenizer))
    }

    /// Load an RWKV World tokenizer from a vocabulary file.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if the file cannot be loaded or parsed.
    #[cfg(feature = "rwkv-tokenizer")]
    pub fn from_rwkv_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tok = rwkv::RwkvTokenizer::from_file(path.as_ref())?;
        Ok(Self::Rwkv(tok))
    }

    /// Encode text into token IDs, adding special tokens (e.g. BOS for Gemma).
    ///
    /// Special tokens are added according to the tokenizer's configured
    /// post-processor, matching the `HuggingFace` convention for inference.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(tok) => {
                let encoding = tok
                    .encode(text, true)
                    .map_err(|e| MIError::Tokenizer(format!("HF encode failed: {e}")))?;
                Ok(encoding.get_ids().to_vec())
            }
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.encode(text),
        }
    }

    /// Encode text into token IDs **without** adding special tokens.
    ///
    /// Useful for MI analyses that need raw tokenization without BOS/EOS.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(tok) => {
                let encoding = tok
                    .encode(text, false)
                    .map_err(|e| MIError::Tokenizer(format!("HF encode failed: {e}")))?;
                Ok(encoding.get_ids().to_vec())
            }
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.encode(text),
        }
    }

    /// Decode token IDs back to a string.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if decoding fails.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::HuggingFace(tok) => tok
                .decode(ids, false)
                .map_err(|e| MIError::Tokenizer(format!("HF decode failed: {e}"))),
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.decode(ids),
        }
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(tok) => tok.get_vocab_size(true),
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.vocab_size(),
        }
    }
}

impl std::fmt::Debug for MITokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace(_) => f.debug_tuple("HuggingFace").field(&"...").finish(),
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => f.debug_tuple("Rwkv").field(tok).finish(),
        }
    }
}
