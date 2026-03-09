// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention patterns: inspect per-head attention distributions at every layer.
//!
//! ```bash
//! # Run on a specific model
//! cargo run --release --features transformer --example attention_patterns -- "meta-llama/Llama-3.2-1B"
//!
//! # Run on all cached models (no argument)
//! cargo run --release --features transformer,mmap --example attention_patterns
//!
//! # With real memory reporting (RAM + VRAM)
//! cargo run --release --features transformer,memory --example attention_patterns -- "meta-llama/Llama-3.2-1B"
//! ```
//!
//! **What it does:**
//!
//! 1. Loads a transformer and captures the post-softmax attention pattern
//!    ([`HookPoint::AttnPattern`](candle_mi::HookPoint::AttnPattern)) at every
//!    layer in a single forward pass.
//! 2. Builds an [`AttentionCache`](candle_mi::AttentionCache) and queries it
//!    with [`attention_from_position`], [`attention_to_position`], and
//!    [`top_attended_positions`] to analyze how the **last token** attends to
//!    previous positions and how **position 0** receives attention.
//! 3. Prints per-layer summaries showing the top-5 attended positions for the
//!    last token, and identifies the layer with the strongest last→first
//!    attention (connecting to the knockout experiment).
//!
//! [`attention_from_position`]: candle_mi::AttentionCache::attention_from_position
//! [`attention_to_position`]: candle_mi::AttentionCache::attention_to_position
//! [`top_attended_positions`]: candle_mi::AttentionCache::top_attended_positions
//!
//! Pass a model ID to run a single model; omit to run all cached models.
//! Each model is dropped before the next one loads, so GPU memory is
//! reused.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use candle_mi::{AttentionCache, HookPoint, HookSpec, MIModel, MITokenizer, SUPPORTED_MODEL_TYPES};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use std::fmt::Write;
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

/// Load a model by ID, capture attention patterns, and print analysis.
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

    run_attention_analysis(&model, tokenizer, prompt)
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
// Per-model attention analysis (cache discovery mode)
// ---------------------------------------------------------------------------

/// Load a model from a snapshot, capture attention, and print analysis.
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

    run_attention_analysis(&model, &tokenizer, prompt)
}

// ---------------------------------------------------------------------------
// Core attention analysis logic
// ---------------------------------------------------------------------------

/// Capture attention patterns at all layers and analyze them.
fn run_attention_analysis(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prompt: &str,
) -> candle_mi::Result<()> {
    let n_layers = model.num_layers();

    // Encode prompt and decode individual tokens for display
    let token_ids = tokenizer.encode(prompt)?;
    let seq_len = token_ids.len();
    let input = candle_core::Tensor::new(&token_ids[..], model.device())?.unsqueeze(0)?; // [1, seq]

    let token_strings: Vec<String> = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .decode(&[id])
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();

    println!("  Prompt: \"{prompt}\" ({seq_len} tokens)");
    print!("  Tokens:");
    for (i, tok) in token_strings.iter().enumerate() {
        print!(" [{i}]=\"{tok}\"");
    }
    println!("\n");

    // Capture AttnPattern at every layer
    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::AttnPattern(layer));
    }

    // Single forward pass with all captures
    let t1 = Instant::now();
    let cache = model.forward(&input, &hooks)?;
    let forward_time = t1.elapsed();
    println!("  Forward pass ({n_layers} AttnPattern captures): {forward_time:.2?}\n");

    // Build AttentionCache from captured patterns
    let mut attn_cache = AttentionCache::with_capacity(n_layers);
    for layer in 0..n_layers {
        let pattern = cache.require(&HookPoint::AttnPattern(layer))?;
        attn_cache.push(pattern.clone());
    }

    let last_pos = seq_len - 1;

    // --- Per-layer analysis: what does the last token attend to? ---
    println!("  === Last token (pos {last_pos}) attention — top-5 per layer ===\n");
    println!(
        "  {:>5}  {:>5} {:>12}  {:>5} {:>12}  {:>5} {:>12}  {:>5} {:>12}  {:>5} {:>12}",
        "Layer", "#1", "weight", "#2", "weight", "#3", "weight", "#4", "weight", "#5", "weight"
    );

    // Track which layer has the strongest attention to position 0
    let mut max_attn_to_pos0: f32 = 0.0;
    let mut max_attn_layer: usize = 0;

    for layer in 0..n_layers {
        let top5 = attn_cache.top_attended_positions(layer, last_pos, 5)?;

        // Track attention to position 0
        let attn_from = attn_cache.attention_from_position(layer, last_pos)?;
        if let Some(&attn_to_0) = attn_from.first() {
            if attn_to_0 > max_attn_to_pos0 {
                max_attn_to_pos0 = attn_to_0;
                max_attn_layer = layer;
            }
        }

        // Format the top-5 row
        let mut row = format!("  {layer:>5}");
        for (pos, weight) in &top5 {
            let tok = token_strings.get(*pos).map_or("?", |s| s.as_str());
            // Truncate token to 5 chars for display
            let tok_display: String = tok.chars().take(5).collect();
            let _ = write!(row, "  {pos:>2}:{tok_display:<5} {weight:>6.3}");
        }
        println!("{row}");
    }

    // --- Incoming attention to position 0 across layers ---
    println!("\n  === Attention TO position 0 (first token) from last token ===\n");
    println!("  {:>5}  {:>12}", "Layer", "Attn to pos 0");
    println!("  {:->5}  {:->12}", "", "");

    for layer in 0..n_layers {
        let attn_from = attn_cache.attention_from_position(layer, last_pos)?;
        let attn_to_0 = attn_from.first().copied().unwrap_or(0.0);
        let marker = if layer == max_attn_layer {
            " ← peak"
        } else {
            ""
        };
        println!("  {layer:>5}  {attn_to_0:>12.6}{marker}");
    }

    // --- Summary ---
    println!("\n  Peak last→first attention: layer {max_attn_layer} ({max_attn_to_pos0:.4})");

    // Show incoming attention to pos 0 from ALL query positions at the peak layer
    let incoming = attn_cache.attention_to_position(max_attn_layer, 0)?;
    println!("\n  Incoming attention to pos 0 at layer {max_attn_layer} (from each query):");
    for (q_pos, &weight) in incoming.iter().enumerate() {
        let tok = token_strings.get(q_pos).map_or("?", |s| s.as_str());
        println!("    pos {q_pos} \"{tok}\": {weight:.4}");
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
