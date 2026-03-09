// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention head knockout: ablate attention at a target layer and measure
//! the impact on predictions.
//!
//! ```bash
//! # Run on a specific model
//! cargo run --release --features transformer --example attention_knockout -- "meta-llama/Llama-3.2-1B"
//!
//! # Run on all cached models (no argument)
//! cargo run --release --features transformer,mmap --example attention_knockout
//!
//! # With real memory reporting (RAM + VRAM)
//! cargo run --release --features transformer,memory --example attention_knockout -- "meta-llama/Llama-3.2-1B"
//! ```
//!
//! **What it does:**
//!
//! 1. Loads a transformer and runs a **baseline** forward pass and an
//!    **ablated** forward pass where a specific attention edge (last token
//!    → first token) is knocked out across all heads at a middle layer.
//! 2. Builds an [`AblationResult`](candle_mi::AblationResult) and prints:
//!    - KL divergence between baseline and ablated distributions,
//!    - logit diff for the expected answer token,
//!    - top-10 tokens whose probabilities changed the most.
//!
//! The knockout works by adding a pre-softmax mask of `-inf` to the
//! attention scores via [`Intervention::Knockout`](candle_mi::Intervention),
//! zeroing out the targeted attention edges after softmax.
//!
//! Pass a model ID to run a single model; omit to run all cached models.
//! Each model is dropped before the next one loads, so GPU memory is
//! reused.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use candle_mi::interp::intervention::create_knockout_mask;
use candle_mi::interp::logit_lens::format_probability;
use candle_mi::{
    AblationResult, HookPoint, HookSpec, Intervention, KnockoutSpec, MIModel, MITokenizer,
    SUPPORTED_MODEL_TYPES,
};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let prompt = "The capital of France is";
    let args: Vec<String> = std::env::args().collect();

    // If a model ID is provided, run only that model
    if args.len() > 1 {
        // INDEX: args[1] is safe — checked len() > 1 above
        #[allow(clippy::indexing_slicing)]
        let model_id = &args[1];
        return run_single_model(model_id, prompt);
    }

    // Otherwise, discover and run all cached models
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

    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_model(model_id, snapshot, prompt) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single model (by ID)
// ---------------------------------------------------------------------------

/// Load a model by ID, run knockout experiment, and print results.
fn run_single_model(model_id: &str, prompt: &str) -> candle_mi::Result<()> {
    println!("=== {model_id} ===");

    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    // CAST: usize → f64, values are small enough for exact representation
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let weight_mb = estimate_weight_mb(n_layers, hidden);
    println!(
        "  Layers: {n_layers}, heads: {n_heads}, hidden: {hidden}, device: {:?}",
        model.device()
    );
    println!("  Estimated F32 weight size: {weight_mb:.0} MB");
    println!("  Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    run_knockout(&model, tokenizer, prompt)
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
// Per-model knockout experiment (cache discovery mode)
// ---------------------------------------------------------------------------

/// Load a model from a snapshot, run knockout, and print the analysis.
fn run_model(model_id: &str, snapshot: &Path, prompt: &str) -> candle_mi::Result<()> {
    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    // CAST: usize → f64, values are small enough for exact representation
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let weight_mb = estimate_weight_mb(n_layers, hidden);
    println!(
        "  {} layers, {} heads, {} hidden, device: {:?}",
        n_layers,
        n_heads,
        hidden,
        model.device()
    );
    println!("  Estimated F32 weight size: {weight_mb:.0} MB  |  Load: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Memory");
    }

    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    run_knockout(&model, &tokenizer, prompt)
}

// ---------------------------------------------------------------------------
// Core knockout logic
// ---------------------------------------------------------------------------

/// Run baseline vs. ablated forward passes and print analysis.
fn run_knockout(model: &MIModel, tokenizer: &MITokenizer, prompt: &str) -> candle_mi::Result<()> {
    let n_layers = model.num_layers();
    let n_heads = model.num_heads();

    // Encode prompt
    let token_ids = tokenizer.encode(prompt)?;
    let seq_len = token_ids.len();
    let input = candle_core::Tensor::new(&token_ids[..], model.device())?.unsqueeze(0)?; // [1, seq]
    println!("  Prompt: \"{prompt}\" ({seq_len} tokens)");

    // Target a middle layer for knockout.
    //
    // Experiment: for each head at the target layer, knock out the
    // attention edge from the last token (query) TO position 0 (the first
    // token).  This tests how much the model relies on each head's
    // attention to the beginning of the prompt for its next-token
    // prediction.
    //
    // We knock out a single edge (last → 0) across ALL heads rather than
    // all edges for a single head, because zeroing an entire query row
    // produces NaN in softmax (the row sums to zero).
    let target_layer = n_layers / 2;
    println!(
        "  Knockout: all {n_heads} heads at layer {target_layer}, \
         edge (last → position 0)\n"
    );

    // --- Baseline forward pass ---
    let t1 = Instant::now();
    let baseline_cache = model.forward(&input, &HookSpec::new())?;
    let baseline_time = t1.elapsed();
    let baseline_logits = baseline_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]

    // --- Build knockout: all heads, single edge last→0 ---
    let spec = KnockoutSpec::new().layer(target_layer).edge(seq_len - 1, 0);

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
    let t2 = Instant::now();
    let ablated_cache = model.forward(&input, &ablated_hooks)?;
    let ablated_time = t2.elapsed();
    let ablated_logits = ablated_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]

    println!("  Baseline forward : {baseline_time:.2?}");
    println!("  Ablated forward  : {ablated_time:.2?}");

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
            "  {:>4}  {:>15}  {:>10}  {:>10}  {:>10}",
            rank + 1,
            format!("\"{}\"", token_text.trim()),
            format_probability(baseline_p),
            format_probability(ablated_p),
            format_probability(abs_diff),
        );
    }
    println!();

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough estimate of F32 weight memory in MB.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    let params_per_layer = 12.0 * (hidden as f64) * (hidden as f64);
    let total_params = (n_layers as f64) * params_per_layer;
    total_params * 4.0 / 1_000_000.0
}
