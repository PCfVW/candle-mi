// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quick start: discover cached transformers, run inference, print top
//! predictions.
//!
//! ```bash
//! cargo run --release --example quick_start_transformer
//! ```
//!
//! **What it does:**
//!
//! 1. Scans the local `HuggingFace` Hub cache (`~/.cache/huggingface/hub/`)
//!    for models whose `model_type` is in
//!    [`SUPPORTED_MODEL_TYPES`](candle_mi::SUPPORTED_MODEL_TYPES).
//! 2. For each cached model, loads it via
//!    [`MIModel::from_pretrained`](candle_mi::MIModel::from_pretrained)
//!    (cache hit only — no downloads are triggered).  Sharded models
//!    (multi-file safetensors) are **skipped** unless the `mmap` feature
//!    is enabled.
//! 3. Tokenizes the prompt *"The capital of France is"*, runs a single
//!    forward pass with an empty [`HookSpec`](candle_mi::HookSpec) (zero
//!    overhead), and prints the top-5 next-token predictions.
//!
//! Each model is dropped before the next one loads, so GPU memory is reused.

use candle_mi::{HookSpec, MIModel, MITokenizer, SUPPORTED_MODEL_TYPES};
use std::path::{Path, PathBuf};

fn main() {
    let prompt = "The capital of France is";

    // 1. Discover cached models
    let cached = discover_cached_models();
    if cached.is_empty() {
        println!("No cached transformer models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!(
            "  python -c \"from huggingface_hub import snapshot_download; \
             snapshot_download('meta-llama/Llama-3.2-1B')\""
        );
        println!();
        println!("Or with Rust:");
        println!("  cargo run --example fast_download -- meta-llama/Llama-3.2-1B");
        return;
    }

    println!(
        "Found {} supported transformer(s) in HF cache:\n",
        cached.len()
    );

    // 2. Iterate and run each model
    for (model_id, model_type, snapshot) in &cached {
        println!("--- {model_id} (model_type: {model_type}) ---");

        if let Err(e) = run_model(model_id, snapshot, prompt) {
            println!("  Skipped: {e}\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Cache discovery
// ---------------------------------------------------------------------------

/// Return the `HuggingFace` Hub cache directory.
fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(cache).join("hub"));
    }
    // Windows
    if let Ok(home) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    // Unix / WSL
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

/// Scan the HF cache and return `(model_id, model_type, snapshot_path)` tuples
/// for supported transformer models.
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

        // HF cache dirs are named models--org--repo
        let Some(rest) = dir_name.strip_prefix("models--") else {
            continue;
        };

        // Convert back: models--org--repo → org/repo
        let model_id = rest.replacen("--", "/", 1);

        // Check for a snapshot with a config.json we can read
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

    // Sort by model_id for deterministic output
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

// ---------------------------------------------------------------------------
// Per-model inference
// ---------------------------------------------------------------------------

/// Load a model, tokenize a prompt, run a forward pass, and print top-5
/// predictions.
fn run_model(model_id: &str, snapshot: &Path, prompt: &str) -> candle_mi::Result<()> {
    // Load model (uses HF cache — no download triggered)
    let model = MIModel::from_pretrained(model_id)?;
    println!(
        "  {} layers, {} hidden, device: {:?}",
        model.num_layers(),
        model.hidden_size(),
        model.device()
    );

    // Load tokenizer from local snapshot
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    // Encode and forward
    let token_ids = tokenizer.encode(prompt)?;
    let input = candle_core::Tensor::new(&token_ids[..], model.device())?.unsqueeze(0)?; // [1, seq]
    println!("  Prompt: \"{prompt}\"  ({} tokens)", token_ids.len());

    let hooks = HookSpec::new();
    let cache = model.forward(&input, &hooks)?;
    let logits = cache.output(); // [1, seq, vocab]

    // Top-5 predictions for the last token
    let seq_len = token_ids.len();
    let last_logits = logits.get(0)?.get(seq_len - 1)?; // [vocab]
    print_top_k(&last_logits, &tokenizer, 5)?;
    println!();

    Ok(())
}

/// Print top-k token predictions from a logits vector.
fn print_top_k(
    logits: &candle_core::Tensor,
    tokenizer: &MITokenizer,
    k: usize,
) -> candle_mi::Result<()> {
    let logits_f32: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;

    // Argsort descending
    let mut indexed: Vec<(usize, f32)> = logits_f32.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Top-{k} predictions:");
    for (rank, (idx, score)) in indexed.iter().take(k).enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let token_text = tokenizer.decode(&[*idx as u32])?;
        println!(
            "    #{}: {:>8.3}  \"{}\"",
            rank + 1,
            score,
            token_text.trim()
        );
    }
    Ok(())
}
