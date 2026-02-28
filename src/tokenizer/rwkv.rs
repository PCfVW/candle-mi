// SPDX-License-Identifier: MIT OR Apache-2.0

//! RWKV World tokenizer (trie-based greedy longest-match).
//!
//! The RWKV-6 models use a custom tokenizer that is incompatible with the
//! `HuggingFace` `tokenizers` crate. This module implements the same algorithm
//! in Rust: a prefix trie built from the vocabulary, with greedy longest-match
//! encoding.
//!
//! Reference: `hf_rwkv_tokenizer.py` in `RWKV/v6-Finch-1B6-HF`

use std::collections::HashMap;
use std::path::Path;

use crate::error::{MIError, Result};

/// Type alias for encode-with-offsets return value.
///
/// Contains `(token_ids, byte_offset_pairs)` where each offset is `(start, end)`.
pub type EncodedWithOffsets = (Vec<u32>, Vec<(usize, usize)>);

/// A node in the prefix trie.
#[derive(Default)]
struct TrieNode {
    /// Child nodes keyed by byte value.
    children: HashMap<u8, Self>,
    /// If this node represents a complete token, store its ID.
    token_id: Option<u32>,
}

/// RWKV World tokenizer using trie-based greedy longest-match.
///
/// Builds a byte-level prefix trie from the vocabulary file and uses
/// greedy longest-match to tokenize input text.
pub struct RwkvTokenizer {
    /// Root of the trie.
    root: TrieNode,
    /// Token ID to byte sequence mapping.
    idx2token: Vec<Vec<u8>>,
    /// String representation to token ID mapping (for special token lookups).
    vocab_map: HashMap<String, u32>,
}

impl std::fmt::Debug for RwkvTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RwkvTokenizer")
            .field("vocab_size", &self.idx2token.len())
            .finish_non_exhaustive()
    }
}

impl RwkvTokenizer {
    /// Load the tokenizer from an RWKV vocabulary file.
    ///
    /// The vocabulary file format (one line per token):
    /// ```text
    /// <index> <python_repr> <byte_length>
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if the file cannot be read or parsed.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::as_conversions // idx is always < 65536
    )]
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            MIError::Tokenizer(format!("failed to read vocab file {}: {e}", path.display()))
        })?;

        let mut root = TrieNode::default();
        let mut idx2token: Vec<Vec<u8>> = Vec::new();
        let mut vocab_map = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Parse: <idx> <token_repr> <byte_length>
            let first_space = line.find(' ').ok_or_else(|| {
                MIError::Tokenizer(format!("invalid vocab line (no space): {line}"))
            })?;
            let idx: usize = line
                .get(..first_space)
                .ok_or_else(|| {
                    MIError::Tokenizer(format!("invalid vocab line (empty index): {line}"))
                })?
                .parse()
                .map_err(|e| {
                    MIError::Tokenizer(format!("invalid index in vocab line: {line}: {e}"))
                })?;

            let rest = line.get(first_space + 1..).ok_or_else(|| {
                MIError::Tokenizer(format!("invalid vocab line (nothing after index): {line}"))
            })?;
            let last_space = rest.rfind(' ').ok_or_else(|| {
                MIError::Tokenizer(format!("invalid vocab line (no second space): {line}"))
            })?;

            let token_repr = rest.get(..last_space).ok_or_else(|| {
                MIError::Tokenizer(format!("invalid vocab line (empty token repr): {line}"))
            })?;
            let expected_len: usize = rest
                .get(last_space + 1..)
                .ok_or_else(|| {
                    MIError::Tokenizer(format!("invalid vocab line (no length): {line}"))
                })?
                .trim()
                .parse()
                .map_err(|e| {
                    MIError::Tokenizer(format!("invalid byte length in vocab line: {line}: {e}"))
                })?;

            let token_bytes = parse_python_literal(token_repr).map_err(|e| {
                MIError::Tokenizer(format!("failed to parse token repr in line: {line}: {e}"))
            })?;

            if token_bytes.len() != expected_len {
                return Err(MIError::Tokenizer(format!(
                    "token length mismatch for idx {idx}: parsed {} bytes, expected {expected_len}",
                    token_bytes.len()
                )));
            }

            // Grow idx2token if needed.
            if idx >= idx2token.len() {
                idx2token.resize(idx + 1, Vec::new());
            }
            #[allow(clippy::indexing_slicing)] // idx < idx2token.len() guaranteed by resize
            idx2token[idx].clone_from(&token_bytes);

            // Build vocab map (for eos_token_id lookups).
            if let Ok(s) = String::from_utf8(token_bytes.clone()) {
                vocab_map.insert(s, idx as u32);
            }

            // Insert into trie.
            let token_id = idx as u32;
            let mut node = &mut root;
            for &byte in &token_bytes {
                node = node.children.entry(byte).or_default();
            }
            node.token_id = Some(token_id);
        }

        tracing::info!(
            "loaded RWKV vocabulary: {} tokens (max idx {})",
            vocab_map.len(),
            idx2token.len().saturating_sub(1),
        );

        Ok(Self {
            root,
            idx2token,
            vocab_map,
        })
    }

    /// Encode text into token IDs using greedy longest-match.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if no matching token is found at
    /// some byte position.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let src = text.as_bytes();
        let mut tokens = Vec::new();
        let mut idx = 0;

        while idx < src.len() {
            let (match_end, token_id) = self.find_longest_match(src, idx);
            if match_end == idx {
                let byte_val = src.get(idx).copied().unwrap_or(0);
                return Err(MIError::Tokenizer(format!(
                    "no matching token at byte position {idx} (byte value 0x{byte_val:02x})"
                )));
            }
            tokens.push(token_id);
            idx = match_end;
        }

        Ok(tokens)
    }

    /// Decode token IDs back to a string.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if a token ID is out of range
    /// or the resulting bytes are not valid UTF-8.
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let id_usize = id as usize;
            let token_bytes = self.idx2token.get(id_usize).ok_or_else(|| {
                MIError::Tokenizer(format!(
                    "token ID {id} out of range (vocab size {})",
                    self.idx2token.len()
                ))
            })?;
            bytes.extend_from_slice(token_bytes);
        }
        String::from_utf8(bytes).map_err(|e| MIError::Tokenizer(format!("UTF-8 decode error: {e}")))
    }

    /// Get vocabulary mapping (string to token ID) for special token lookups.
    #[must_use]
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab_map.clone()
    }

    /// Get vocabulary size (number of token entries).
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.idx2token.len()
    }

    /// Encode text and return token IDs with byte offsets.
    ///
    /// Each offset pair `(start, end)` is in bytes (not characters).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if no matching token is found.
    pub fn encode_with_offsets(&self, text: &str) -> Result<EncodedWithOffsets> {
        let src = text.as_bytes();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut idx = 0;

        while idx < src.len() {
            let (match_end, token_id) = self.find_longest_match(src, idx);
            if match_end == idx {
                let byte_val = src.get(idx).copied().unwrap_or(0);
                return Err(MIError::Tokenizer(format!(
                    "no matching token at byte position {idx} (byte value 0x{byte_val:02x})"
                )));
            }
            tokens.push(token_id);
            offsets.push((idx, match_end));
            idx = match_end;
        }

        Ok((tokens, offsets))
    }

    /// Find the longest matching token starting at position `start`.
    ///
    /// Returns `(end_position, token_id)`. If no match, returns `(start, 0)`.
    fn find_longest_match(&self, src: &[u8], start: usize) -> (usize, u32) {
        let mut node = &self.root;
        let mut best_end = start;
        let mut best_id = 0u32;
        let mut idx = start;

        while idx < src.len() {
            let byte = src.get(idx).copied().unwrap_or(0);
            if let Some(child) = node.children.get(&byte) {
                node = child;
                idx += 1;
                if let Some(token_id) = node.token_id {
                    best_end = idx;
                    best_id = token_id;
                }
            } else {
                break;
            }
        }

        (best_end, best_id)
    }
}

// ---------------------------------------------------------------------------
// Python literal parser
// ---------------------------------------------------------------------------

/// Parse a Python string/bytes literal into raw bytes.
///
/// Handles:
/// - `'...'` — Python string literal (with escape sequences)
/// - `b'...'` — Python bytes literal
/// - `"..."` — Python string literal (double quotes)
///
/// Escape sequences: `\xHH`, `\uHHHH`, `\UHHHHHHHH`, `\t`, `\n`, `\r`,
/// `\\`, `\'`, `\"`, `\0`, `\a`, `\b`, `\f`, `\v`
fn parse_python_literal(repr: &str) -> std::result::Result<Vec<u8>, String> {
    let repr = repr.trim();

    // Determine if it's a bytes literal (b'...') or string literal ('...')
    let (inner, is_bytes) = if let Some(stripped) = repr.strip_prefix("b'") {
        (
            stripped
                .strip_suffix('\'')
                .ok_or_else(|| format!("unterminated bytes literal: {repr}"))?,
            true,
        )
    } else if let Some(stripped) = repr.strip_prefix('\'') {
        (
            stripped
                .strip_suffix('\'')
                .ok_or_else(|| format!("unterminated string literal: {repr}"))?,
            false,
        )
    } else if let Some(stripped) = repr.strip_prefix('"') {
        (
            stripped
                .strip_suffix('"')
                .ok_or_else(|| format!("unterminated string literal: {repr}"))?,
            false,
        )
    } else {
        return Err(format!("unexpected token representation format: {repr}"));
    };

    let mut bytes = Vec::new();
    let mut chars = inner.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            parse_escape_sequence(&mut chars, &mut bytes, is_bytes, repr)?;
        } else {
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(encoded.as_bytes());
        }
    }

    Ok(bytes)
}

/// Parse a single escape sequence after a backslash.
fn parse_escape_sequence(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    bytes: &mut Vec<u8>,
    is_bytes: bool,
    repr: &str,
) -> std::result::Result<(), String> {
    match chars.next() {
        Some('x') => {
            let h1 = chars
                .next()
                .ok_or_else(|| format!("incomplete \\x escape in: {repr}"))?;
            let h2 = chars
                .next()
                .ok_or_else(|| format!("incomplete \\x escape in: {repr}"))?;
            let hex_str: String = [h1, h2].iter().collect();
            let byte = u8::from_str_radix(&hex_str, 16)
                .map_err(|_| format!("invalid hex in \\x escape: {hex_str}"))?;
            if is_bytes {
                bytes.push(byte);
            } else {
                let ch = char::from(byte);
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                bytes.extend_from_slice(encoded.as_bytes());
            }
        }
        Some('u') => {
            let hex_str = read_hex_chars(chars, 4, repr, "\\u")?;
            let codepoint = u32::from_str_radix(&hex_str, 16)
                .map_err(|_| format!("invalid hex in \\u escape: {hex_str}"))?;
            let ch = char::from_u32(codepoint)
                .ok_or_else(|| format!("invalid Unicode codepoint: U+{hex_str}"))?;
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(encoded.as_bytes());
        }
        Some('U') => {
            let hex_str = read_hex_chars(chars, 8, repr, "\\U")?;
            let codepoint = u32::from_str_radix(&hex_str, 16)
                .map_err(|_| format!("invalid hex in \\U escape: {hex_str}"))?;
            let ch = char::from_u32(codepoint)
                .ok_or_else(|| format!("invalid Unicode codepoint: U+{hex_str}"))?;
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(encoded.as_bytes());
        }
        Some('t') => bytes.push(b'\t'),
        Some('n') => bytes.push(b'\n'),
        Some('r') => bytes.push(b'\r'),
        Some('\\') => bytes.push(b'\\'),
        Some('\'') => bytes.push(b'\''),
        Some('"') => bytes.push(b'"'),
        Some('0') => bytes.push(0),
        Some('a') => bytes.push(0x07),
        Some('b') => bytes.push(0x08),
        Some('f') => bytes.push(0x0C),
        Some('v') => bytes.push(0x0B),
        Some(other) => {
            return Err(format!("unknown escape sequence \\{other} in: {repr}"));
        }
        None => {
            return Err(format!("trailing backslash in: {repr}"));
        }
    }
    Ok(())
}

/// Read `count` hex characters from the iterator.
fn read_hex_chars(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    count: usize,
    repr: &str,
    escape_name: &str,
) -> std::result::Result<String, String> {
    let mut hex_str = String::with_capacity(count);
    for _ in 0..count {
        hex_str.push(
            chars
                .next()
                .ok_or_else(|| format!("incomplete {escape_name} escape in: {repr}"))?,
        );
    }
    Ok(hex_str)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_strings() {
        assert_eq!(parse_python_literal("'hello'").unwrap(), b"hello");
        assert_eq!(parse_python_literal("'a'").unwrap(), b"a");
        assert_eq!(parse_python_literal("' '").unwrap(), b" ");
    }

    #[test]
    fn parse_escape_sequences() {
        assert_eq!(parse_python_literal("'\\t'").unwrap(), b"\t");
        assert_eq!(parse_python_literal("'\\n'").unwrap(), b"\n");
        assert_eq!(parse_python_literal("'\\r'").unwrap(), b"\r");
        assert_eq!(parse_python_literal("'\\x00'").unwrap(), vec![0u8]);
        // String literal '\\x7f' → U+007F → single UTF-8 byte 0x7F
        assert_eq!(parse_python_literal("'\\x7f'").unwrap(), vec![0x7F]);
        // String literal '\\x80' → U+0080 → two UTF-8 bytes [0xC2, 0x80]
        assert_eq!(parse_python_literal("'\\x80'").unwrap(), vec![0xC2, 0x80]);
        // String literal '\\xff' → U+00FF → two UTF-8 bytes [0xC3, 0xBF]
        assert_eq!(parse_python_literal("'\\xff'").unwrap(), vec![0xC3, 0xBF]);
        assert_eq!(parse_python_literal("'\\t\\t'").unwrap(), b"\t\t");
    }

    #[test]
    fn parse_bytes_literal() {
        assert_eq!(parse_python_literal("b'\\xff'").unwrap(), vec![0xFF]);
        assert_eq!(parse_python_literal("b'\\x00'").unwrap(), vec![0u8]);
    }

    #[test]
    fn trie_basic_encoding() {
        // Build a tiny trie for testing.
        let mut root = TrieNode::default();

        // Single bytes: 'h'=0, 'e'=1, 'l'=2, 'o'=3
        for (byte, id) in [(b'h', 0u32), (b'e', 1), (b'l', 2), (b'o', 3)] {
            let node = root.children.entry(byte).or_default();
            node.token_id = Some(id);
        }
        // Multi-byte: "he"=4, "hel"=5, "hello"=6
        {
            let h = root.children.entry(b'h').or_default();
            let he = h.children.entry(b'e').or_default();
            he.token_id = Some(4);
            let hel = he.children.entry(b'l').or_default();
            hel.token_id = Some(5);
            let hell = hel.children.entry(b'l').or_default();
            let hello = hell.children.entry(b'o').or_default();
            hello.token_id = Some(6);
        }

        let tokenizer = RwkvTokenizer {
            root,
            idx2token: vec![
                b"h".to_vec(),
                b"e".to_vec(),
                b"l".to_vec(),
                b"o".to_vec(),
                b"he".to_vec(),
                b"hel".to_vec(),
                b"hello".to_vec(),
            ],
            vocab_map: HashMap::new(),
        };

        // "hello" should match as one token (longest match)
        let ids = tokenizer.encode("hello").unwrap();
        assert_eq!(ids, vec![6]);

        // "helo" → "hel" + "o"
        let ids = tokenizer.encode("helo").unwrap();
        assert_eq!(ids, vec![5, 3]);

        // Decode round-trip
        let decoded = tokenizer.decode(&[6]).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn decode_out_of_range() {
        let tokenizer = RwkvTokenizer {
            root: TrieNode::default(),
            idx2token: vec![b"a".to_vec()],
            vocab_map: HashMap::new(),
        };
        assert!(tokenizer.decode(&[999]).is_err());
    }

    #[test]
    fn vocab_size() {
        let tokenizer = RwkvTokenizer {
            root: TrieNode::default(),
            idx2token: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            vocab_map: HashMap::new(),
        };
        assert_eq!(tokenizer.vocab_size(), 3);
    }

    #[test]
    fn encode_with_offsets_basic() {
        // Build a minimal trie: 'h'=0, 'i'=1
        let mut root = TrieNode::default();
        let h = root.children.entry(b'h').or_default();
        h.token_id = Some(0);
        let i = root.children.entry(b'i').or_default();
        i.token_id = Some(1);

        let tokenizer = RwkvTokenizer {
            root,
            idx2token: vec![b"h".to_vec(), b"i".to_vec()],
            vocab_map: HashMap::new(),
        };

        let (ids, offsets) = tokenizer.encode_with_offsets("hi").unwrap();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(offsets, vec![(0, 1), (1, 2)]);
    }
}
