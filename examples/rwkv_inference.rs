// SPDX-License-Identifier: MIT OR Apache-2.0

//! RWKV linear RNN inference with state hooks and knockout.
//!
//! RWKV (<https://www.rwkv.com/>) is a linear-complexity recurrent architecture
//! that replaces attention with a WKV (Weighted Key-Value) recurrence.
//! This example demonstrates RWKV-specific MI capabilities that differ from
//! transformer attention:
//!
//! - **Recurrent state capture** — `RwkvState` is a `[batch, heads, head_dim,
//!   head_dim]` matrix per layer (vs. transformers' `[batch, heads, seq, seq]`
//!   attention patterns).
//! - **Data-dependent decay** — `RwkvDecay` controls per-position memory
//!   retention, enabling selective forgetting.
//! - **State knockout** — skipping the kv write at a position makes that token
//!   invisible to all future tokens (the RNN analogue of all-edge attention
//!   knockout in transformers).
//!
//! Supports both RWKV-6 (Finch) and RWKV-7 (Goose) models.
//!
//! ```bash
//! # Run on all cached RWKV models
//! cargo run --release --features rwkv --example rwkv_inference
//!
//! # Run on a specific model
//! cargo run --release --features rwkv --example rwkv_inference -- "RWKV/RWKV7-Goose-World3-1.5B-HF"
//!
//! # RWKV-6 requires the rwkv-tokenizer feature (no tokenizer.json in HF repo)
//! cargo run --release --features rwkv,rwkv-tokenizer --example rwkv_inference -- "RWKV/v6-Finch-1B6-HF"
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_docs_in_private_items)]

use std::path::{Path, PathBuf};

use candle_core::{DType, IndexOp, Tensor};

use candle_mi::rwkv::SUPPORTED_RWKV_MODEL_TYPES;
use candle_mi::{
    HookPoint, HookSpec, MIModel, MITokenizer, StateAblationResult, StateKnockoutSpec,
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Single model specified on the command line.
    if let Some(model_id) = args.get(1) {
        return run_single_model(model_id);
    }

    // Otherwise, discover and run all cached RWKV models.
    let cached = discover_cached_rwkv_models();
    if cached.is_empty() {
        println!("No cached RWKV models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!(
            "  cargo run --features rwkv --example fast_download -- RWKV/RWKV7-Goose-World3-1.5B-HF"
        );
        return Ok(());
    }

    println!("Found {} cached RWKV model(s):\n", cached.len());
    for (id, model_type, _) in &cached {
        println!("  {id}  ({model_type})");
    }
    println!();

    for (model_id, _, _) in &cached {
        if let Err(e) = run_single_model(model_id) {
            eprintln!("  Error on {model_id}: {e}\n");
        }
    }

    Ok(())
}

/// Load a model by ID and run the full demonstration.
fn run_single_model(model_id: &str) -> candle_mi::Result<()> {
    println!("=== RWKV Inference: {model_id} ===\n");

    let t_start = std::time::Instant::now();
    let model = MIModel::from_pretrained(model_id)?;

    let n_layers = model.num_layers();
    let hidden = model.hidden_size();
    let n_heads = model.num_heads();
    let device = model.device().clone();

    println!("  Layers:  {n_layers}");
    println!("  Hidden:  {hidden}");
    println!("  Heads:   {n_heads}");
    println!("  Device:  {device:?}");
    println!("  Loaded in {:.2}s\n", t_start.elapsed().as_secs_f64());

    // RWKV-7 HF repos include tokenizer.json; RWKV-6 repos do not.
    // Try the bundled HF tokenizer first, then fall back to the RWKV World
    // tokenizer vocab file from the HF cache.
    let owned_tokenizer: MITokenizer;
    let tokenizer = if let Some(t) = model.tokenizer() {
        t
    } else {
        owned_tokenizer = load_rwkv_tokenizer(model_id)?;
        &owned_tokenizer
    };

    let prompt = "The capital of France is";
    let token_ids = tokenizer.encode(prompt)?;
    let seq_len = token_ids.len();
    println!("  Prompt:  \"{prompt}\"");
    println!("  Tokens:  {seq_len}\n");

    // ------------------------------------------------------------------
    // Section 1: Basic inference — top-10 predictions
    // ------------------------------------------------------------------

    println!("--- 1. Basic Inference (top-10) ---\n");

    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let hooks = HookSpec::new();
    let result = model.forward(&input, &hooks)?;

    let logits = result.output();
    let last_logits = logits.i((0, seq_len - 1))?;
    let last_logits_f32 = last_logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = last_logits_f32.to_vec1()?;

    let mut indexed: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  {:>4}  {:>8}  {:>10}  Token", "Rank", "ID", "Logit");
    println!("  {}", "-".repeat(40));
    for (rank, (idx, logit)) in indexed.iter().take(10).enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let token_str = tokenizer
            .decode(&[*idx as u32])
            .unwrap_or_else(|_| format!("[{idx}]"));
        println!(
            "  {:>4}  {:>8}  {:>10.4}  \"{}\"",
            rank + 1,
            idx,
            logit,
            token_str.replace('\n', "\\n")
        );
    }
    println!();

    // ------------------------------------------------------------------
    // Section 2: RWKV-specific hook capture
    // ------------------------------------------------------------------

    println!("--- 2. RWKV Hook Capture ---\n");

    let mid_layer = n_layers / 2;

    let mut capture_hooks = HookSpec::new();
    capture_hooks.capture(HookPoint::RwkvState(mid_layer));
    capture_hooks.capture(HookPoint::RwkvDecay(mid_layer));
    capture_hooks.capture(HookPoint::ResidPost(mid_layer));

    let capture_result = model.forward(&input, &capture_hooks)?;

    // RwkvState: [batch, heads, head_dim, head_dim]
    let state = capture_result.require(&HookPoint::RwkvState(mid_layer))?;
    println!("  RwkvState(L{mid_layer}):  {:?}", state.dims());
    println!("    → Recurrent state matrix per head (unique to RWKV)");

    // RwkvDecay: [batch, seq, heads, head_dim]
    let decay = capture_result.require(&HookPoint::RwkvDecay(mid_layer))?;
    println!("  RwkvDecay(L{mid_layer}):  {:?}", decay.dims());
    println!("    → Data-dependent decay controlling memory retention");

    // ResidPost: [batch, seq, hidden] (same as transformers)
    let resid = capture_result.require(&HookPoint::ResidPost(mid_layer))?;
    println!("  ResidPost(L{mid_layer}): {:?}", resid.dims());
    println!("    → Residual stream (shared with transformer hook system)");
    println!();

    // ------------------------------------------------------------------
    // Section 3: State knockout — make position 0 invisible
    // ------------------------------------------------------------------

    println!("--- 3. State Knockout (position 0 at L{mid_layer}) ---\n");

    let knockout_spec = StateKnockoutSpec::new().position(0).layer(mid_layer);

    let mut knockout_hooks = HookSpec::new();
    knockout_hooks.set_state_knockout(knockout_spec.clone());

    let ablated_result = model.forward(&input, &knockout_hooks)?;

    // Build StateAblationResult for analysis.
    let baseline_last = logits.i((0, seq_len - 1))?;
    let ablated_logits = ablated_result.output();
    let ablated_last = ablated_logits.i((0, seq_len - 1))?;

    let ablation = StateAblationResult::new(baseline_last, ablated_last, knockout_spec);

    let kl = ablation.kl_divergence()?;
    println!("  KL divergence (baseline → ablated): {kl:.6}");

    let top_changed = ablation.top_changed_tokens(5)?;
    println!("\n  Top-5 changed tokens:");
    println!(
        "  {:>8}  {:>10}  {:>10}  {:>10}  Token",
        "ID", "Baseline", "Ablated", "Delta"
    );
    println!("  {}", "-".repeat(55));
    for (token_id, baseline_logit, ablated_logit, delta) in &top_changed {
        let token_str = tokenizer
            .decode(&[*token_id])
            .unwrap_or_else(|_| format!("[{token_id}]"));
        println!(
            "  {:>8}  {:>10.4}  {:>10.4}  {:>+10.4}  \"{}\"",
            token_id,
            baseline_logit,
            ablated_logit,
            delta,
            token_str.replace('\n', "\\n")
        );
    }

    println!(
        "\n  Total elapsed: {:.2}s\n",
        t_start.elapsed().as_secs_f64()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// RWKV tokenizer fallback
// ---------------------------------------------------------------------------

/// Load the RWKV World tokenizer from the HF cache for models that lack
/// a `tokenizer.json` (e.g., RWKV-6 Finch).
#[cfg(feature = "rwkv-tokenizer")]
fn load_rwkv_tokenizer(model_id: &str) -> candle_mi::Result<MITokenizer> {
    /// RWKV World tokenizer vocab filename (present in RWKV-6 and RWKV-7 repos).
    const RWKV_VOCAB_FILE: &str = "rwkv_vocab_v20230424.txt";

    let cache_dir = hf_cache_dir().ok_or_else(|| {
        candle_mi::MIError::Tokenizer("cannot find HuggingFace cache directory".into())
    })?;
    let snapshot = find_snapshot(&cache_dir, model_id).ok_or_else(|| {
        candle_mi::MIError::Tokenizer(format!("{model_id} not found in HF cache"))
    })?;
    let vocab_path = snapshot.join(RWKV_VOCAB_FILE);
    if vocab_path.exists() {
        MITokenizer::from_rwkv_path(&vocab_path)
    } else {
        Err(candle_mi::MIError::Tokenizer(format!(
            "no tokenizer.json or {RWKV_VOCAB_FILE} found for {model_id}"
        )))
    }
}

/// Fallback when the `rwkv-tokenizer` feature is not enabled.
#[cfg(not(feature = "rwkv-tokenizer"))]
fn load_rwkv_tokenizer(model_id: &str) -> candle_mi::Result<MITokenizer> {
    Err(candle_mi::MIError::Tokenizer(format!(
        "{model_id} has no tokenizer.json; enable `rwkv-tokenizer` feature for RWKV World tokenizer"
    )))
}

// ---------------------------------------------------------------------------
// Cache discovery (RWKV models)
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

/// Scan the HF cache and return `(model_id, model_type, snapshot_path)` for RWKV models.
fn discover_cached_rwkv_models() -> Vec<(String, String, PathBuf)> {
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
        if SUPPORTED_RWKV_MODEL_TYPES.contains(&model_type.as_str()) {
            models.push((model_id, model_type, snapshot));
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}
