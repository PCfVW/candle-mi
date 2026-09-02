// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activation patching (causal tracing): identify which layers are causally
//! responsible for a model's factual prediction.
//!
//! ```bash
//! # Run on a specific model
//! cargo run --release --features transformer --example activation_patching -- "meta-llama/Llama-3.2-1B"
//!
//! # Run on all cached models (no argument)
//! cargo run --release --features transformer,mmap --example activation_patching
//!
//! # With JSON output (includes full layer × position heatmap grid)
//! cargo run --release --features transformer --example activation_patching -- "meta-llama/Llama-3.2-1B" --output examples/results/activation_patching/llama-3.2-1b.json
//!
//! # With real memory reporting (RAM + VRAM)
//! cargo run --release --features transformer,memory --example activation_patching -- "meta-llama/Llama-3.2-1B"
//! ```
//!
//! **What it does:**
//!
//! 1. Runs a **clean** forward pass on "The capital of France is", capturing
//!    the residual stream at every layer via
//!    [`HookPoint::ResidPost`](candle_mi::HookPoint) and building a
//!    [`FullActivationCache`](candle_mi::FullActivationCache).
//! 2. Runs a **corrupted** forward pass on "The capital of Poland is" (same
//!    structure, different country) for its logits. It needs no captures: the
//!    patch is expressed against the recipient's own activation in flight.
//! 3. For each layer, runs a **patching** pass: the corrupted forward pass
//!    with the clean residual stream at the **subject token position** only
//!    restored via [`Intervention::PatchAt`](candle_mi::Intervention). This
//!    isolates the effect of the subject token's representation at each layer.
//! 4. Prints a layer-by-layer recovery table showing how much the "Paris"
//!    prediction recovers when clean information is injected at each layer.
//! 5. Runs a **full grid sweep** over every (layer × token position)
//!    combination (Meng et al. Figure 1e). For each cell, the clean residual
//!    at that position is patched into the corrupted forward pass. The result
//!    is a 2D heatmap of recovery percentages — the "causal trace."
//! 6. With `--output`, writes the heatmap grid and metadata to JSON for
//!    plotting (e.g., with the bundled Mathematica script).
//!
//! This is the standard "causal tracing" technique from:
//!
//! > Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.
//! > "Locating and Editing Factual Associations in GPT."
//! > *Advances in Neural Information Processing Systems* (NeurIPS), 2022.
//! > <https://arxiv.org/abs/2202.05262>
//! > (Section 2.1 "Causal Tracing of Factual Associations", Figure 1e)
//!
//! Layers with the highest recovery are the causal site for factual recall.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use candle_mi::interp::intervention::kl_divergence;
use candle_mi::interp::logit_lens::format_probability;
use candle_mi::{
    FullActivationCache, HookPoint, HookSpec, Intervention, MIModel, MITokenizer,
    SUPPORTED_MODEL_TYPES,
};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use clap::Parser;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "activation_patching")]
#[command(about = "Causal tracing: patch clean residuals into corrupted passes")]
struct Args {
    /// `HuggingFace` model ID (omit to run all cached models)
    model: Option<String>,

    /// Write structured JSON output (including heatmap grid) to this file
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

/// Full causal tracing output for one model, including the Figure 1e grid.
#[derive(Serialize)]
struct JsonOutput {
    /// Model identifier (e.g., "google/gemma-2-2b")
    model_id: String,
    /// Clean prompt text
    clean_prompt: String,
    /// Corrupted prompt text
    corrupted_prompt: String,
    /// Index of the first differing token
    subject_pos: usize,
    /// Decoded token strings for the clean prompt
    tokens: Vec<String>,
    /// Number of layers
    n_layers: usize,
    /// Sequence length (number of tokens)
    seq_len: usize,
    /// KL(clean || corrupted) baseline
    corrupted_kl: f32,
    /// Recovery % for subject-position-only sweep, indexed by layer
    subject_recovery: Vec<f32>,
    /// Full heatmap grid: `grid[layer][position]` = recovery %
    grid: Vec<Vec<f32>>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// Prompt pairs for causal tracing.
///
/// Each pair has a clean prompt and a list of candidate corrupted prompts.
/// The first candidate whose tokenization has the same length as the clean
/// prompt is used. This handles tokenizers that split names differently.
struct PromptPair {
    /// Short label for this prompt pair (used in output)
    label: &'static str,
    /// The clean factual prompt
    clean: &'static str,
    /// Candidate corrupted prompts (first length-matching one is used)
    corrupted_candidates: &'static [&'static str],
}

/// Default prompt pair: "The capital of France is" → "Paris".
const PROMPT_FRANCE: PromptPair = PromptPair {
    label: "France→Paris",
    clean: "The capital of France is",
    corrupted_candidates: &[
        "The capital of Poland is",
        "The capital of Brazil is",
        "The capital of Russia is",
        "The capital of Canada is",
        "The capital of Turkey is",
    ],
};

/// Meng et al. (2022) original prompt: "The Space Needle is in downtown" → "Seattle".
const PROMPT_SPACE_NEEDLE: PromptPair = PromptPair {
    label: "Space Needle→Seattle",
    clean: "The Space Needle is in downtown",
    corrupted_candidates: &[
        "The Eiffel Tower is in downtown",
        "The Colosseum is located in downtown",
        "The Big Ben is located in downtown",
        "The Great Wall is located in downtown",
    ],
};

/// All prompt pairs to run.
const PROMPT_PAIRS: &[&PromptPair] = &[&PROMPT_FRANCE, &PROMPT_SPACE_NEEDLE];

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();

    // If a model ID is provided, run only that model
    if let Some(ref model_id) = args.model {
        return run_single_model(model_id, args.output.as_deref());
    }

    // Otherwise, discover and run all cached models
    let cached = discover_cached_models();
    if cached.is_empty() {
        println!("No cached transformer models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!("  cargo run --example fast_download -- meta-llama/Llama-3.2-1B");
        return Ok(());
    }

    if args.output.is_some() {
        println!("Note: --output is only used with a specific model ID.");
    }

    println!(
        "Found {} supported transformer(s) in HF cache:\n",
        cached.len()
    );

    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_model(model_id, snapshot) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single model (by ID)
// ---------------------------------------------------------------------------

/// Load a model by ID, run activation patching, and print results.
fn run_single_model(model_id: &str, output_path: Option<&Path>) -> candle_mi::Result<()> {
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

    for pair in PROMPT_PAIRS {
        println!("\n  --- Prompt: {} ---", pair.label);
        let corrupted_prompt = find_corrupted_prompt(tokenizer, pair)?;
        // Build a per-pair output path: e.g. "dir/model.json" → "dir/model_france.json"
        let pair_output = output_path.map(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("json");
            // BORROW: to_lowercase() — &str → String for filename sanitisation
            let suffix = pair
                .label
                .split('→')
                .next()
                .unwrap_or("unknown")
                .to_lowercase();
            let suffix = suffix.trim();
            p.with_file_name(format!("{stem}_{suffix}.{ext}"))
        });
        run_patching(
            &model,
            tokenizer,
            pair.clean,
            &corrupted_prompt,
            model_id,
            pair_output.as_deref(),
        )?;
    }
    Ok(())
}

/// Pick the first corrupted prompt whose tokenization matches the clean
/// prompt's token count.
fn find_corrupted_prompt(tokenizer: &MITokenizer, pair: &PromptPair) -> candle_mi::Result<String> {
    let clean_len = tokenizer.encode(pair.clean)?.len();
    for &candidate in pair.corrupted_candidates {
        if tokenizer.encode(candidate)?.len() == clean_len {
            return Ok(candidate.into());
        }
    }
    Err(candle_mi::MIError::Tokenizer(format!(
        "no corrupted prompt candidate matches clean prompt token count for \"{}\"",
        pair.label
    )))
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
// Per-model patching experiment (cache discovery mode)
// ---------------------------------------------------------------------------

/// Load a model from a snapshot, run patching, and print the analysis.
fn run_model(model_id: &str, snapshot: &Path) -> candle_mi::Result<()> {
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

    for pair in PROMPT_PAIRS {
        println!("\n  --- Prompt: {} ---", pair.label);
        match find_corrupted_prompt(&tokenizer, pair) {
            Ok(corrupted_prompt) => {
                run_patching(
                    &model,
                    &tokenizer,
                    pair.clean,
                    &corrupted_prompt,
                    model_id,
                    None,
                )?;
            }
            Err(e) => {
                println!("  Skipped: {e}");
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core activation patching logic
// ---------------------------------------------------------------------------

/// Run clean, corrupted, and patching passes, then print the recovery table.
///
/// When `output_path` is `Some`, also writes a JSON file with the full
/// layer × position heatmap grid (Meng et al. Figure 1e).
fn run_patching(
    model: &MIModel,
    tokenizer: &MITokenizer,
    clean_prompt: &str,
    corrupted_prompt: &str,
    model_id: &str,
    output_path: Option<&Path>,
) -> candle_mi::Result<()> {
    let n_layers = model.num_layers();

    // Encode both prompts
    let clean_ids = tokenizer.encode(clean_prompt)?;
    let corrupted_ids = tokenizer.encode(corrupted_prompt)?;
    let seq_len = clean_ids.len();

    if clean_ids.len() != corrupted_ids.len() {
        return Err(candle_mi::MIError::Config(format!(
            "clean ({}) and corrupted ({}) prompts have different token counts",
            clean_ids.len(),
            corrupted_ids.len(),
        )));
    }

    // Find the subject token position (first token that differs)
    let subject_pos = clean_ids
        .iter()
        .zip(corrupted_ids.iter())
        .position(|(a, b)| a != b)
        .ok_or_else(|| {
            candle_mi::MIError::Config("clean and corrupted prompts are identical".into())
        })?;

    // INDEX: subject_pos is a valid index — it came from .position() over
    // clean_ids/corrupted_ids which have length seq_len (checked equal above)
    #[allow(clippy::indexing_slicing)]
    let clean_subject_id = clean_ids[subject_pos];
    #[allow(clippy::indexing_slicing)]
    let corrupted_subject_id = corrupted_ids[subject_pos];
    let clean_subject = tokenizer.decode(&[clean_subject_id])?;
    let corrupted_subject = tokenizer.decode(&[corrupted_subject_id])?;
    println!("  Clean: \"{clean_prompt}\" ({seq_len} tokens)");
    println!("  Corrupted: \"{corrupted_prompt}\"");
    println!("  Subject position: {subject_pos} (\"{clean_subject}\" → \"{corrupted_subject}\")");

    let clean_input = candle_core::Tensor::new(&clean_ids[..], model.device())?.unsqueeze(0)?;
    let corrupted_input =
        candle_core::Tensor::new(&corrupted_ids[..], model.device())?.unsqueeze(0)?;

    // ── Step 1: Clean forward pass ──────────────────────────────────────
    let mut capture_hooks = HookSpec::new();
    for layer in 0..n_layers {
        capture_hooks.capture(HookPoint::ResidPost(layer));
    }

    let t1 = Instant::now();
    let clean_cache = model.forward(&clean_input, &capture_hooks)?;
    let clean_time = t1.elapsed();
    let clean_logits = clean_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]
    println!("  Clean forward ({n_layers} captures): {clean_time:.2?}");

    // Build FullActivationCache from clean captures
    let mut clean_acts = FullActivationCache::with_capacity(n_layers);
    for layer in 0..n_layers {
        let resid = clean_cache.require(&HookPoint::ResidPost(layer))?; // [1, seq, hidden]
        clean_acts.push(resid.get(0)?); // [seq, hidden]
    }

    // ── Step 2: Corrupted forward pass ──────────────────────────────────
    // No captures: `Intervention::PatchAt` overwrites one position of whatever
    // reaches the hook point, so the recipient's own residual stream never has
    // to be read back out and spliced by hand.
    let t2 = Instant::now();
    let corrupted_cache = model.forward(&corrupted_input, &HookSpec::new())?;
    let corrupted_time = t2.elapsed();
    let corrupted_logits = corrupted_cache.output().get(0)?.get(seq_len - 1)?;
    println!("  Corrupted forward (no captures): {corrupted_time:.2?}");

    // Baseline metrics
    let corrupted_kl = kl_divergence(&clean_logits, &corrupted_logits)?;
    println!("  KL(clean || corrupted): {corrupted_kl:.6}");

    // Find "Paris" token
    let paris_id = tokenizer.encode(" Paris")?.into_iter().last();
    if let Some(pid) = paris_id {
        // PROMOTE: softmax requires F32 for numerical stability
        let clean_p = extract_prob(
            &candle_nn::ops::softmax_last_dim(&clean_logits.to_dtype(candle_core::DType::F32)?)?,
            pid,
        )?;
        let corrupted_p = extract_prob(
            &candle_nn::ops::softmax_last_dim(
                &corrupted_logits.to_dtype(candle_core::DType::F32)?,
            )?,
            pid,
        )?;
        println!(
            "  P(Paris) clean: {}  |  corrupted: {}",
            format_probability(clean_p),
            format_probability(corrupted_p),
        );
    }

    // ── Step 3: Patching sweep ──────────────────────────────────────────
    // For each layer, run the corrupted forward pass but restore the clean
    // residual at the subject token position only.
    println!(
        "\n  {:>5}  {:>10}  {:>10}  {:>12}",
        "Layer", "KL", "Recovery", "P(Paris)"
    );
    println!("  {:->5}  {:->10}  {:->10}  {:->12}", "", "", "", "");

    let t3 = Instant::now();
    let mut best_layer = 0;
    let mut best_recovery = f32::NEG_INFINITY;
    let mut subject_recovery = Vec::with_capacity(n_layers);

    for layer in 0..n_layers {
        // The clean row to inject: corrupted everywhere else, clean at
        // subject_pos. `PatchAt` leaves every other position in flight alone.
        let clean_row = clean_acts.get_position(layer, subject_pos)?; // [hidden]

        let mut patch_hooks = HookSpec::new();
        patch_hooks.intervene(
            HookPoint::ResidPost(layer),
            Intervention::PatchAt {
                position: subject_pos,
                value: clean_row,
            },
        );

        let patched_cache = model.forward(&corrupted_input, &patch_hooks)?;
        let patched_logits = patched_cache.output().get(0)?.get(seq_len - 1)?;

        let patched_kl = kl_divergence(&clean_logits, &patched_logits)?;

        // Recovery = 1 - (patched_kl / corrupted_kl), as percentage
        let recovery = if corrupted_kl > 1e-10 {
            (1.0 - patched_kl / corrupted_kl) * 100.0
        } else {
            100.0
        };

        subject_recovery.push(recovery);
        if recovery > best_recovery {
            best_recovery = recovery;
            best_layer = layer;
        }

        let paris_str = if let Some(pid) = paris_id {
            // PROMOTE: softmax requires F32 for numerical stability
            let patched_probs = candle_nn::ops::softmax_last_dim(
                &patched_logits.to_dtype(candle_core::DType::F32)?,
            )?;
            format_probability(extract_prob(&patched_probs, pid)?)
        } else {
            String::from("--")
        };

        println!("  {layer:>5}  {patched_kl:>10.6}  {recovery:>9.1}%  {paris_str:>12}");
    }
    let patch_time = t3.elapsed();

    println!("\n  Best recovery: layer {best_layer} ({best_recovery:.1}%)");
    println!("  Patching sweep ({n_layers} passes): {patch_time:.2?}");

    // ── Step 4: Full grid sweep (layer × position) — Figure 1e ────────
    println!("\n  --- Full causal trace (layer × position) ---");
    let t4 = Instant::now();
    let mut grid: Vec<Vec<f32>> = Vec::with_capacity(n_layers);

    for layer in 0..n_layers {
        let mut row = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            let clean_row = clean_acts.get_position(layer, pos)?; // [hidden]

            let mut patch_hooks = HookSpec::new();
            patch_hooks.intervene(
                HookPoint::ResidPost(layer),
                Intervention::PatchAt {
                    position: pos,
                    value: clean_row,
                },
            );

            let patched_cache = model.forward(&corrupted_input, &patch_hooks)?;
            let patched_logits = patched_cache.output().get(0)?.get(seq_len - 1)?;
            let patched_kl = kl_divergence(&clean_logits, &patched_logits)?;

            let recovery = if corrupted_kl > 1e-10 {
                (1.0 - patched_kl / corrupted_kl) * 100.0
            } else {
                100.0
            };
            row.push(recovery);
        }
        grid.push(row);
    }
    let grid_time = t4.elapsed();

    // Decode each token for column headers
    let token_labels: Vec<String> = clean_ids
        .iter()
        .map(|&tid| tokenizer.decode(&[tid]))
        .collect::<candle_mi::Result<Vec<_>>>()?;

    // Print header row
    print!("\n  {:>5}", "Layer");
    for (pos, label) in token_labels.iter().enumerate() {
        // BORROW: chars().take() — safe Unicode truncation
        let short: String = label.chars().take(8).collect();
        print!("  {:>10}", format!("[{pos}]{short}"));
    }
    println!();

    // Print separator
    print!("  {:->5}", "");
    for _ in 0..seq_len {
        print!("  {:->10}", "");
    }
    println!();

    // Print each layer row
    for (layer, row) in grid.iter().enumerate() {
        print!("  {layer:>5}");
        for &val in row {
            print!("  {val:>9.1}%");
        }
        println!();
    }

    // CAST: usize → display, small values
    println!(
        "\n  Full grid ({n_layers}×{seq_len} = {} passes): {grid_time:.2?}",
        n_layers * seq_len
    );

    // ── Step 5: JSON output ───────────────────────────────────────────
    if let Some(path) = output_path {
        let output = JsonOutput {
            // BORROW: to_owned() — &str → String for JSON serialization
            model_id: model_id.to_owned(),
            clean_prompt: clean_prompt.to_owned(),
            corrupted_prompt: corrupted_prompt.to_owned(),
            subject_pos,
            tokens: token_labels,
            n_layers,
            seq_len,
            corrupted_kl,
            subject_recovery,
            grid,
        };
        write_json(path, &output)?;
        println!("\n  JSON written to {}", path.display());
    }

    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the probability of a specific token from a probability tensor.
fn extract_prob(probs: &candle_core::Tensor, token_id: u32) -> candle_mi::Result<f32> {
    let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    // CAST: u32 → usize, token ID is a valid index
    #[allow(clippy::as_conversions)]
    let idx = token_id as usize;
    Ok(probs_vec.get(idx).copied().unwrap_or(0.0))
}

/// Rough estimate of F32 weight memory in MB.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    let params_per_layer = 12.0 * (hidden as f64) * (hidden as f64);
    let total_params = (n_layers as f64) * params_per_layer;
    total_params * 4.0 / 1_000_000.0
}

/// Serialize `output` as pretty JSON and write to `path`.
fn write_json(path: &Path, output: &JsonOutput) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    std::fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;
    Ok(())
}
