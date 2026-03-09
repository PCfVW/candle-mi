// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention head knockout: ablate attention at a target layer and measure
//! the impact on predictions.
//!
//! ```bash
//! cargo run --release --features transformer --example attention_knockout
//! ```
//!
//! **What it does:**
//!
//! 1. Scans the local `HuggingFace` Hub cache for supported transformers.
//! 2. For each cached model, runs a **baseline** forward pass and an
//!    **ablated** forward pass where all attention heads at a middle layer
//!    are knocked out for the last token position (the query position that
//!    produces the next-token prediction).
//! 3. Builds an [`AblationResult`](candle_mi::AblationResult) and prints:
//!    - KL divergence between baseline and ablated distributions,
//!    - logit diff for the expected answer token,
//!    - top-10 tokens whose probabilities changed the most.
//!
//! The knockout works by adding a pre-softmax mask of `-inf` to the
//! attention scores via [`Intervention::Knockout`](candle_mi::Intervention),
//! zeroing out the targeted attention edges after softmax.
//!
//! Each model is dropped before the next one loads, so GPU memory is
//! reused.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use candle_mi::interp::intervention::create_knockout_mask;
use candle_mi::{
    AblationResult, HookPoint, HookSpec, Intervention, KnockoutSpec, MIModel, MITokenizer,
    SUPPORTED_MODEL_TYPES,
};
use std::path::{Path, PathBuf};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let prompt = "The capital of France is";

    // 1. Discover cached models
    let cached = discover_cached_models();
    if cached.is_empty() {
        println!("No cached transformer models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!("  cargo run --example fast_download -- meta-llama/Llama-3.2-1B");
        return Ok(());
    }

    println!(
        "Found {} supported transformer(s) in HF cache:\n",
        cached.len()
    );

    // 2. Run knockout experiment on each model
    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_model(model_id, snapshot, prompt) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cache discovery (same pattern as quick_start_transformer)
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
// Per-model knockout experiment
// ---------------------------------------------------------------------------

/// Load a model, run baseline vs. ablated, and print the analysis.
fn run_model(model_id: &str, snapshot: &Path, prompt: &str) -> candle_mi::Result<()> {
    let model = MIModel::from_pretrained(model_id)?;
    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    println!(
        "  {} layers, {} heads, {} hidden, device: {:?}",
        n_layers,
        n_heads,
        model.hidden_size(),
        model.device()
    );

    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    // Encode prompt
    let token_ids = tokenizer.encode(prompt)?;
    let seq_len = token_ids.len();
    let input = candle_core::Tensor::new(&token_ids[..], model.device())?.unsqueeze(0)?; // [1, seq]
    println!("  Prompt: \"{prompt}\" ({seq_len} tokens)");

    // Target a middle layer for knockout
    let target_layer = n_layers / 2;
    println!("  Knockout: all {n_heads} heads at layer {target_layer}, last query position\n");

    // --- Baseline forward pass ---
    let baseline_cache = model.forward(&input, &HookSpec::new())?;
    let baseline_logits = baseline_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]

    // --- Build knockout: all heads at target_layer, FROM last position ---
    let spec = KnockoutSpec::new()
        .layer(target_layer)
        .from_position(seq_len - 1);

    let mask = create_knockout_mask(
        &spec,
        n_heads,
        seq_len,
        model.device(),
        candle_core::DType::F32,
    )?;

    let mut ablated_hooks = HookSpec::new();
    ablated_hooks.intervene(
        HookPoint::AttnScores(target_layer),
        Intervention::Knockout(mask),
    );

    // --- Ablated forward pass ---
    let ablated_cache = model.forward(&input, &ablated_hooks)?;
    let ablated_logits = ablated_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]

    // --- Analysis ---
    let result = AblationResult::new(baseline_logits, ablated_logits, spec);

    let kl = result.kl_divergence()?;
    println!("  KL divergence (baseline || ablated): {kl:.6}");

    // Find "Paris" token for logit diff
    let paris_tokens = tokenizer.encode(" Paris")?;
    if let Some(&paris_id) = paris_tokens.last() {
        let diff = result.logit_diff(paris_id)?;
        println!("  Logit diff for \" Paris\" (token {paris_id}): {diff:+.4}");
    }

    // Top changed tokens
    let changed = result.top_changed_tokens(10)?;
    println!("\n  Top-10 most changed tokens:");
    println!(
        "  {:>4}  {:>15}  {:>10}  {:>10}  {:>10}",
        "Rank", "Token", "Baseline", "Ablated", "|Diff|"
    );
    for (rank, &(token_id, baseline_p, ablated_p, abs_diff)) in changed.iter().enumerate() {
        let token_text = tokenizer.decode(&[token_id])?;
        println!(
            "  {:>4}  {:>15}  {:>9.4}%  {:>9.4}%  {:>9.4}%",
            rank + 1,
            format!("\"{}\"", token_text.trim()),
            baseline_p * 100.0,
            ablated_p * 100.0,
            abs_diff * 100.0
        );
    }
    println!();

    Ok(())
}
