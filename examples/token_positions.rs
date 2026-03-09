// SPDX-License-Identifier: MIT OR Apache-2.0

//! Character-to-token position mapping: demonstrate the positioning utilities
//! for mapping between character offsets and token indices.
//!
//! ```bash
//! # Run with a specific model's tokenizer
//! cargo run --example token_positions -- "meta-llama/Llama-3.2-1B"
//!
//! # Run on all cached models
//! cargo run --example token_positions
//! ```
//!
//! **What it does:**
//!
//! 1. Loads only a tokenizer (no model weights, no GPU) from a cached
//!    `HuggingFace` model snapshot.
//! 2. Tokenizes a sample text with
//!    [`encode_with_offsets`](candle_mi::MITokenizer::encode_with_offsets),
//!    producing an [`EncodingWithOffsets`](candle_mi::EncodingWithOffsets).
//! 3. Prints a token table showing each token's ID, string, and byte-offset
//!    range via [`tokens_with_offsets`](candle_mi::EncodingWithOffsets::tokens_with_offsets).
//! 4. Demonstrates character-to-token lookups:
//!    [`char_to_token`](candle_mi::EncodingWithOffsets::char_to_token),
//!    [`char_range_to_tokens`](candle_mi::EncodingWithOffsets::char_range_to_tokens),
//!    and [`token_to_char_range`](candle_mi::EncodingWithOffsets::token_to_char_range).
//! 5. Demonstrates batch conversion via
//!    [`convert_positions`](candle_mi::convert_positions) showing exact vs.
//!    fuzzy matching.
//!
//! This is a pure utility example — no model loading, no GPU, no `transformer`
//! feature required. It is essential for any MI analysis that maps
//! character-level annotations (e.g., entity spans) to token positions.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]

use candle_mi::{MITokenizer, SUPPORTED_MODEL_TYPES, convert_positions};
use std::path::{Path, PathBuf};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let text = "The Eiffel Tower is located in Paris, France.";
    let args: Vec<String> = std::env::args().collect();

    // If a model ID is provided, run only that model's tokenizer
    if args.len() > 1 {
        // INDEX: args[1] is safe — checked len() > 1 above
        #[allow(clippy::indexing_slicing)]
        let model_id = &args[1];
        return run_single_model(model_id, text);
    }

    // Otherwise, discover and run all cached models
    let cached = discover_cached_models();
    if cached.is_empty() {
        println!("No cached transformer models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!("  cargo run --example fast_download -- meta-llama/Llama-3.2-1B");
        return Ok(());
    }

    println!("Found {} supported model(s) in HF cache:\n", cached.len());

    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_with_snapshot(snapshot, text) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single model (by ID)
// ---------------------------------------------------------------------------

/// Load a model's tokenizer by ID and run the positioning demo.
fn run_single_model(model_id: &str, text: &str) -> candle_mi::Result<()> {
    println!("=== {model_id} ===");

    let cache_dir = hf_cache_dir().ok_or(candle_mi::MIError::Config(
        "HuggingFace cache directory not found".into(),
    ))?;
    let snapshot = find_snapshot(&cache_dir, model_id).ok_or(candle_mi::MIError::Config(
        format!("model {model_id} not found in HF cache"),
    ))?;

    run_with_snapshot(&snapshot, text)
}

/// Load tokenizer from a snapshot directory and run the demo.
fn run_with_snapshot(snapshot: &Path, text: &str) -> candle_mi::Result<()> {
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    run_positions(&tokenizer, text)
}

// ---------------------------------------------------------------------------
// Cache discovery (same pattern as other examples)
// ---------------------------------------------------------------------------

/// Return the `HuggingFace` Hub cache directory.
fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(cache).join("hub"));
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Find the first snapshot directory for a cached model.
fn find_snapshot(cache_dir: &Path, model_id: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots = cache_dir.join(dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Read `model_type` from a cached `config.json`.
fn read_model_type(snapshot: &Path) -> Option<String> {
    let config_path = snapshot.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    // BORROW: explicit .as_str() — serde_json::Value → &str
    json.get("model_type")?.as_str().map(String::from)
}

/// Scan the HF cache and return `(model_id, model_type, snapshot_path)`.
fn discover_cached_models() -> Vec<(String, String, PathBuf)> {
    let Some(cache_dir) = hf_cache_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return Vec::new();
    };

    let mut models = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(dir_name) = name.to_str() else {
            continue;
        };
        let Some(rest) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let model_id = rest.replacen("--", "/", 1);
        let Some(snapshot) = find_snapshot(&cache_dir, &model_id) else {
            continue;
        };
        let Some(model_type) = read_model_type(&snapshot) else {
            continue;
        };
        // BORROW: explicit .as_str() — String → &str for slice lookup
        if SUPPORTED_MODEL_TYPES.contains(&model_type.as_str()) {
            models.push((model_id, model_type, snapshot));
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

// ---------------------------------------------------------------------------
// Core positioning demo
// ---------------------------------------------------------------------------

/// Run the full positioning demo on a loaded tokenizer.
fn run_positions(tokenizer: &MITokenizer, text: &str) -> candle_mi::Result<()> {
    println!("  Text: \"{text}\"\n");

    // ── Tokenize with offsets ───────────────────────────────────────────
    let encoding = tokenizer.encode_with_offsets(text)?;

    // ── Token table ─────────────────────────────────────────────────────
    println!(
        "  {:>4}  {:>8}  {:>6}  {:>6}  {:>18}",
        "Idx", "Token ID", "Start", "End", "Token"
    );
    println!(
        "  {:->4}  {:->8}  {:->6}  {:->6}  {:->18}",
        "", "", "", "", ""
    );

    let tokens = encoding.tokens_with_offsets();
    for (idx, tok) in tokens.iter().enumerate() {
        let id = encoding.ids.get(idx).copied().unwrap_or(0);
        println!(
            "  {:>4}  {:>8}  {:>6}  {:>6}  {:>18}",
            idx,
            id,
            tok.start,
            tok.end,
            format!("\"{}\"", tok.token),
        );
    }

    // ── Character-to-token lookups ──────────────────────────────────────
    println!("\n  Character-to-token lookups:");
    // Key positions in "The Eiffel Tower is located in Paris, France."
    //  0: 'T', 4: 'E' (Eiffel), 11: 'T' (Tower), 31: 'P' (Paris), 38: 'F' (France)
    for char_pos in [0, 4, 11, 31, 38, 44] {
        let tok_idx = encoding.char_to_token(char_pos);
        let tok_str = tok_idx
            .and_then(|i| encoding.tokens.get(i))
            .map_or("(none)", String::as_str);
        println!("    char {char_pos:>2} → token {tok_idx:>6?}  \"{tok_str}\"");
    }

    // ── Character range → tokens ────────────────────────────────────────
    println!("\n  Character ranges to token sets:");

    // "Eiffel Tower" spans bytes 4..16
    let eiffel_tokens = encoding.char_range_to_tokens(4, 16);
    println!("    \"Eiffel Tower\" (chars 4-16) → tokens {eiffel_tokens:?}");

    // "Paris" spans bytes 31..36
    let paris_tokens = encoding.char_range_to_tokens(31, 36);
    println!("    \"Paris\" (chars 31-36) → tokens {paris_tokens:?}");

    // "France" spans bytes 38..44
    let france_tokens = encoding.char_range_to_tokens(38, 44);
    println!("    \"France\" (chars 38-44) → tokens {france_tokens:?}");

    // ── Token → character range (reverse) ───────────────────────────────
    println!("\n  Token-to-character reverse mapping:");
    for tok_idx in 0..tokens.len().min(5) {
        let range = encoding.token_to_char_range(tok_idx);
        let tok_str = encoding.tokens.get(tok_idx).map_or("?", String::as_str);
        println!("    token {tok_idx} (\"{tok_str}\") → chars {range:?}");
    }

    // ── Batch conversion with convert_positions ─────────────────────────
    println!("\n  Batch conversion (convert_positions):");
    let positions: &[usize] = &[4, 16, 31, 36, 99];
    let conversions = convert_positions(&encoding, positions);
    for conv in &conversions {
        let match_type = if conv.exact_match { "exact" } else { "fuzzy" };
        let tok_str = conv
            .token
            .as_deref()
            .map_or_else(|| String::from("(none)"), |s| format!("\"{s}\""));
        println!(
            "    char {:>3} → token {:>6?}  {:<18} [{match_type}]",
            conv.char_pos, conv.token_idx, tok_str,
        );
    }

    println!();
    Ok(())
}
