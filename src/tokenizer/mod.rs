// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizer abstraction: dispatch between `HuggingFace` and RWKV backends.
//!
//! [`MITokenizer`] provides a unified encode/decode interface regardless of
//! the underlying tokenizer implementation.

#[cfg(feature = "rwkv-tokenizer")]
mod rwkv;

use crate::error::{MIError, Result};
use crate::util::positioning::EncodingWithOffsets;

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
    #[must_use]
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

    /// Encode text into token IDs with character offset mapping.
    ///
    /// Returns an [`EncodingWithOffsets`] containing token IDs, token strings,
    /// and byte-offset ranges for each token. Special tokens are added
    /// (e.g., BOS for Gemma); special tokens receive a `(0, 0)` offset.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails or if the backend
    /// does not support offset mapping (RWKV).
    pub fn encode_with_offsets(&self, text: &str) -> Result<EncodingWithOffsets> {
        self.encode_with_offsets_inner(text, true)
    }

    /// Encode text into token IDs with character offset mapping, **without**
    /// adding special tokens.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails or if the backend
    /// does not support offset mapping (RWKV).
    pub fn encode_raw_with_offsets(&self, text: &str) -> Result<EncodingWithOffsets> {
        self.encode_with_offsets_inner(text, false)
    }

    /// Shared implementation for offset-bearing encode methods.
    fn encode_with_offsets_inner(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<EncodingWithOffsets> {
        match self {
            Self::HuggingFace(tok) => {
                let encoding = tok
                    .encode(text, add_special_tokens)
                    .map_err(|e| MIError::Tokenizer(format!("HF encode failed: {e}")))?;
                let ids = encoding.get_ids().to_vec();
                let tokens: Vec<String> = encoding
                    .get_tokens()
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                let offsets = encoding.get_offsets().to_vec();
                Ok(EncodingWithOffsets::new(ids, tokens, offsets))
            }
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(_) => Err(MIError::Tokenizer(
                "RWKV tokenizer does not support offset mapping".into(),
            )),
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

    /// Find the token ID for a word, trying `" word"` (with leading space) first,
    /// then bare `"word"`.
    ///
    /// This handles BPE tokenizers that represent word-initial tokens with a
    /// leading space (e.g., `" cat"` → single token).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if the word cannot be resolved to a
    /// single token in either form.
    pub fn find_token_id(&self, word: &str) -> Result<u32> {
        // Try with leading space first (most BPE tokenizers).
        let with_space = format!(" {word}");
        let ids = self.encode(&with_space)?;
        // ids[0] is BOS (if present), ids[1] would be the word token.
        if ids.len() == 2 {
            return ids
                .get(1)
                .copied()
                .ok_or_else(|| MIError::Tokenizer(format!("unexpected encoding for \" {word}\"")));
        }

        // Try bare word.
        let bare_ids = self.encode(word)?;
        if bare_ids.len() == 2 {
            return bare_ids
                .get(1)
                .copied()
                .ok_or_else(|| MIError::Tokenizer(format!("unexpected encoding for \"{word}\"")));
        }

        // Last resort: return last token.
        ids.last().copied().ok_or_else(|| {
            MIError::Tokenizer(format!("could not find single token ID for \"{word}\""))
        })
    }

    /// Decode a single token ID to its string representation.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if decoding fails.
    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.decode(&[token_id])
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
