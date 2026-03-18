// SPDX-License-Identifier: MIT OR Apache-2.0

//! Steering convergence: does residual stream injection converge to natural
//! computation, or does the model take a different internal path?
//!
//! ```bash
//! # Run on a specific model
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B"
//!
//! # With JSON output
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B" --output results.json
//!
//! # Custom threshold and strength sweep
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B" --threshold 0.90 --max-strength 8.0
//!
//! # Custom prompts — measure attractor depth of planning (rhyme)
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B" --prompt "Twinkle twinkle little star, how I wonder what you" --contrastive "Twinkle twinkle little star, how I wonder where you" --target-token " are"
//! ```
//!
//! **What it does:**
//!
//! 1. Runs a **baseline** forward pass on "The capital of France is", capturing
//!    [`HookPoint::ResidPost`](candle_mi::HookPoint) at every layer.
//! 2. Runs a **contrastive** forward pass on "The capital of Germany is" and
//!    computes per-layer steering vectors (France residual − Germany residual
//!    at the last token position).
//! 3. For each injection layer (0..n_layers), injects the layer-specific
//!    steering vector via [`Intervention::Add`](candle_mi::Intervention) and
//!    captures all layers — producing a 16×16 **convergence matrix** of cosine
//!    similarities between steered and natural activations.
//! 4. Identifies the **absorption boundary** — the earliest layer after
//!    injection where cosine similarity exceeds a threshold (default 0.95).
//! 5. Runs a **strength sweep** at the most effective injection layer to show
//!    how increasing perturbation strength shifts the absorption boundary.
//!
//! This answers: when externally controlled, does the model converge to its
//! natural attractor state, or does it find an alternative internal path?
//!
//! Inspired by Jyothir S V, Siddhartha Jalagam, Yann LeCun, and Vlad Sobal.
//! "Gradient-based Planning with World Models." arXiv:2312.17227, 2023.
//! — reframed as MI observation of external control in language models.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use candle_core::{DType, Device, Tensor};
use candle_mi::interp::intervention::kl_divergence;
use candle_mi::interp::logit_lens::format_probability;
use candle_mi::{HookPoint, HookSpec, Intervention, MIModel, MITokenizer};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use clap::Parser;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "steering_convergence")]
#[command(
    about = "Steering convergence: does residual stream injection converge to natural computation?"
)]
struct Args {
    /// `HuggingFace` model ID
    #[arg(default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// Cosine similarity threshold for absorption boundary
    #[arg(long, default_value_t = 0.95)]
    threshold: f32,

    /// Maximum steering strength for the strength sweep
    #[arg(long, default_value_t = 6.0)]
    max_strength: f32,

    /// Number of strength steps in the sweep
    #[arg(long, default_value_t = 12)]
    strength_steps: usize,

    /// Custom clean prompt (overrides the default "The capital of France is")
    #[arg(long)]
    prompt: Option<String>,

    /// Custom contrastive prompt (must have same token count as --prompt)
    #[arg(long)]
    contrastive: Option<String>,

    /// Custom target token to track (e.g., " are", " mat"); default " Paris"
    #[arg(long)]
    target_token: Option<String>,

    /// Write structured JSON output to this file
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    prompt: String,
    contrastive_prompt: String,
    n_layers: usize,
    hidden_size: usize,
    target_token: String,
    target_token_id: u32,
    baseline_p_target: f32,
    baseline_top5: Vec<JsonPrediction>,
    /// `convergence_matrix[inj][obs]` = cosine similarity
    convergence_matrix: Vec<Vec<f32>>,
    layer_summaries: Vec<JsonLayerSummary>,
    best_injection_layer: usize,
    strength_sweep: Vec<JsonStrengthPoint>,
    threshold: f32,
}

#[derive(Serialize)]
struct JsonPrediction {
    token: String,
    token_id: u32,
    probability: f32,
}

#[derive(Serialize)]
struct JsonLayerSummary {
    injection_layer: usize,
    p_target: f32,
    kl_divergence: f32,
    absorption_layer: Option<usize>,
}

#[derive(Serialize)]
struct JsonStrengthPoint {
    strength: f32,
    p_target: f32,
    kl_divergence: f32,
    absorption_layer: Option<usize>,
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const CLEAN_PROMPT: &str = "The capital of France is";

/// Contrastive candidates — first one whose token count matches is used.
const CONTRASTIVE_CANDIDATES: &[&str] = &[
    "The capital of Germany is",
    "The capital of Poland is",
    "The capital of Brazil is",
    "The capital of Russia is",
    "The capital of Canada is",
];

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();
    run_model(&args.model, &args)
}

// ---------------------------------------------------------------------------
// Core experiment
// ---------------------------------------------------------------------------

fn run_model(model_id: &str, args: &Args) -> candle_mi::Result<()> {
    println!("=== {model_id} ===");

    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(&Device::cuda_if_available(0).unwrap_or(Device::Cpu))?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    let device = model.device().clone();

    println!("  Layers: {n_layers}, heads: {n_heads}, hidden: {hidden}, device: {device:?}");
    println!(
        "  Estimated F32 weight size: {:.0} MB",
        estimate_weight_mb(n_layers, hidden)
    );
    println!("  Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    // Resolve prompt, contrastive, and target token
    let clean_prompt = args.prompt.as_deref().unwrap_or(CLEAN_PROMPT);
    let clean_tokens = tokenizer.encode(clean_prompt)?;
    let seq_len = clean_tokens.len();

    let (contrastive_prompt, contrastive_tokens) = if let Some(ref c) = args.contrastive {
        let tokens = tokenizer.encode(c)?;
        if tokens.len() != seq_len {
            return Err(candle_mi::MIError::Tokenizer(format!(
                "--contrastive has {} tokens but --prompt has {} tokens (must match)",
                tokens.len(),
                seq_len
            )));
        }
        (c.as_str(), tokens)
    } else {
        find_contrastive_prompt(tokenizer, &clean_tokens)?
    };

    println!("  Prompt: \"{clean_prompt}\" ({seq_len} tokens)");
    println!("  Contrastive: \"{contrastive_prompt}\"");

    // Find target token (custom or default " Paris")
    let target_str = args.target_token.as_deref().unwrap_or(" Paris");
    let target_tokens = tokenizer.encode(target_str)?;
    let target_id = *target_tokens
        .last()
        .ok_or(candle_mi::MIError::Tokenizer(format!(
            "could not encode target token \"{target_str}\""
        )))?;
    let target_text = tokenizer.decode(&[target_id])?;
    println!("  Target token: \"{target_text}\" (id {target_id})");

    // Build input tensors
    let clean_input = Tensor::new(&clean_tokens[..], &device)?.unsqueeze(0)?;
    let contrastive_input = Tensor::new(&contrastive_tokens[..], &device)?.unsqueeze(0)?;

    // Set up capture hooks for all layers
    let mut capture_hooks = HookSpec::new();
    for layer in 0..n_layers {
        capture_hooks.capture(HookPoint::ResidPost(layer));
    }

    let t1 = Instant::now();

    // -----------------------------------------------------------------------
    // Step 1: Baseline run
    // -----------------------------------------------------------------------
    println!("\n  Step 1: Baseline forward pass...");
    let baseline_cache = model.forward(&clean_input, &capture_hooks)?;
    let baseline_logits = last_token_logits(baseline_cache.output(), seq_len)?;
    let baseline_probs = softmax_1d(&baseline_logits)?;
    let baseline_p_target = extract_prob(&baseline_probs, target_id)?;
    let baseline_top5 = top_k_predictions(&baseline_probs, tokenizer, 5)?;

    // Store baseline residuals at last token position
    let mut baseline_resid: Vec<Tensor> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let resid = baseline_cache.require(&HookPoint::ResidPost(layer))?;
        // resid: [1, seq_len, hidden] → [hidden]
        baseline_resid.push(resid.get(0)?.get(seq_len - 1)?);
    }

    println!(
        "    P(\"{target_text}\") = {}",
        format_probability(baseline_p_target)
    );
    print!("    Top-5:");
    for p in &baseline_top5 {
        print!(
            "  \"{}\" {}",
            p.token.trim(),
            format_probability(p.probability)
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 2: Contrastive run → steering vectors
    // -----------------------------------------------------------------------
    println!("  Step 2: Contrastive run → extracting steering vectors...");
    let contrastive_cache = model.forward(&contrastive_input, &capture_hooks)?;

    let mut steering_vectors: Vec<Tensor> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let contrastive_resid = contrastive_cache.require(&HookPoint::ResidPost(layer))?;
        // contrastive_resid: [1, seq_len, hidden] → [hidden]
        let contrastive_last = contrastive_resid.get(0)?.get(seq_len - 1)?;
        // steering = baseline − contrastive (points from Germany toward France)
        steering_vectors.push((&baseline_resid[layer] - &contrastive_last)?);
    }

    // -----------------------------------------------------------------------
    // Step 3: Injection layer sweep
    // -----------------------------------------------------------------------
    println!("  Step 3: Injection layer sweep ({n_layers} forward passes)...");

    // steered_resids[inj_layer][obs_layer] = Tensor [hidden]
    let mut steered_resids: Vec<Vec<Tensor>> = Vec::with_capacity(n_layers);
    let mut steered_logits_per_layer: Vec<Tensor> = Vec::with_capacity(n_layers);

    for inj_layer in 0..n_layers {
        let delta = build_position_delta(
            &steering_vectors[inj_layer],
            seq_len,
            hidden,
            seq_len - 1,
            &device,
        )?;

        let mut hooks = HookSpec::new();
        hooks.intervene(HookPoint::ResidPost(inj_layer), Intervention::Add(delta));
        for layer in 0..n_layers {
            hooks.capture(HookPoint::ResidPost(layer));
        }

        let steered_cache = model.forward(&clean_input, &hooks)?;
        let steered_logits = last_token_logits(steered_cache.output(), seq_len)?;
        steered_logits_per_layer.push(steered_logits);

        let mut layer_resids = Vec::with_capacity(n_layers);
        for obs_layer in 0..n_layers {
            let resid = steered_cache.require(&HookPoint::ResidPost(obs_layer))?;
            layer_resids.push(resid.get(0)?.get(seq_len - 1)?);
        }
        steered_resids.push(layer_resids);
    }

    // -----------------------------------------------------------------------
    // Step 4: Compute convergence matrix
    // -----------------------------------------------------------------------
    println!("  Step 4: Computing convergence matrix...");

    let mut convergence_matrix: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    for inj_layer in 0..n_layers {
        let mut row = Vec::with_capacity(n_layers);
        for obs_layer in 0..n_layers {
            let sim = cosine_similarity(
                &baseline_resid[obs_layer],
                &steered_resids[inj_layer][obs_layer],
            )?;
            row.push(sim);
        }
        convergence_matrix.push(row);
    }

    // -----------------------------------------------------------------------
    // Step 5: Absorption boundary + per-layer summary
    // -----------------------------------------------------------------------
    println!("  Step 5: Analyzing absorption boundaries...");

    let mut layer_summaries: Vec<JsonLayerSummary> = Vec::with_capacity(n_layers);
    #[allow(unused_assignments)]
    let mut best_layer = 0_usize;

    for inj_layer in 0..n_layers {
        let steered_probs = softmax_1d(&steered_logits_per_layer[inj_layer])?;
        let p_target = extract_prob(&steered_probs, target_id)?;
        let kl = kl_divergence(&baseline_logits, &steered_logits_per_layer[inj_layer])?;

        // Find absorption boundary: first layer AFTER injection where sim >= threshold
        let absorption =
            find_absorption_boundary(&convergence_matrix[inj_layer], inj_layer, args.threshold);

        layer_summaries.push(JsonLayerSummary {
            injection_layer: inj_layer,
            p_target,
            kl_divergence: kl,
            absorption_layer: absorption,
        });
    }

    // Best layer for strength sweep = deepest layer that still achieves absorption.
    // This is the most informative site: right at the absorption boundary, where
    // increasing strength is most likely to push past the attractor's basin.
    best_layer = 0;
    for s in &layer_summaries {
        if s.absorption_layer.is_some() {
            best_layer = s.injection_layer;
        }
    }
    // Fallback: if no layer absorbs, pick the one with lowest KL divergence
    if layer_summaries.iter().all(|s| s.absorption_layer.is_none()) {
        let mut min_kl = f32::MAX;
        for s in &layer_summaries {
            if s.kl_divergence < min_kl {
                min_kl = s.kl_divergence;
                best_layer = s.injection_layer;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Strength sweep at best injection layer
    // -----------------------------------------------------------------------
    println!(
        "  Step 6: Strength sweep at layer {best_layer} ({} steps)...",
        args.strength_steps
    );

    let mut strength_sweep: Vec<JsonStrengthPoint> = Vec::with_capacity(args.strength_steps);
    let step_size = args.max_strength / args.strength_steps as f32;

    for step in 1..=args.strength_steps {
        // CAST: usize → f32, step is small (≤12)
        #[allow(clippy::as_conversions)]
        let strength = step as f32 * step_size;

        let scaled = (&steering_vectors[best_layer] * f64::from(strength))?;
        let delta = build_position_delta(&scaled, seq_len, hidden, seq_len - 1, &device)?;

        let mut hooks = HookSpec::new();
        hooks.intervene(HookPoint::ResidPost(best_layer), Intervention::Add(delta));
        for layer in 0..n_layers {
            hooks.capture(HookPoint::ResidPost(layer));
        }

        let cache = model.forward(&clean_input, &hooks)?;
        let s_logits = last_token_logits(cache.output(), seq_len)?;
        let s_probs = softmax_1d(&s_logits)?;
        let p_target = extract_prob(&s_probs, target_id)?;
        let kl = kl_divergence(&baseline_logits, &s_logits)?;

        // Compute convergence row for this strength
        let mut conv_row = Vec::with_capacity(n_layers);
        for obs_layer in 0..n_layers {
            let resid = cache.require(&HookPoint::ResidPost(obs_layer))?;
            let steered_last = resid.get(0)?.get(seq_len - 1)?;
            conv_row.push(cosine_similarity(
                &baseline_resid[obs_layer],
                &steered_last,
            )?);
        }
        let absorption = find_absorption_boundary(&conv_row, best_layer, args.threshold);

        strength_sweep.push(JsonStrengthPoint {
            strength,
            p_target,
            kl_divergence: kl,
            absorption_layer: absorption,
        });
    }

    let total_time = t1.elapsed();
    println!("\n  Total experiment time: {total_time:.2?}");

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------

    print_convergence_matrix(&convergence_matrix, n_layers);
    print_layer_summary(&layer_summaries, &target_text);
    print_strength_sweep(&strength_sweep, best_layer, &target_text);
    print_interpretation(&convergence_matrix, &layer_summaries, args.threshold);

    // -----------------------------------------------------------------------
    // JSON output
    // -----------------------------------------------------------------------
    if let Some(ref path) = args.output {
        let output = JsonOutput {
            model_id: model_id.to_owned(),
            prompt: clean_prompt.to_owned(),
            contrastive_prompt: contrastive_prompt.to_owned(),
            n_layers,
            hidden_size: hidden,
            target_token: target_text.clone(),
            target_token_id: target_id,
            baseline_p_target,
            baseline_top5,
            convergence_matrix,
            layer_summaries,
            best_injection_layer: best_layer,
            strength_sweep,
            threshold: args.threshold,
        };
        write_json(path, &output)?;
        println!("\n  JSON output written to {}", path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers: tensor operations
// ---------------------------------------------------------------------------

/// Cosine similarity between two `[hidden]` tensors.
fn cosine_similarity(a: &Tensor, b: &Tensor) -> candle_mi::Result<f32> {
    // PROMOTE: F32 for dot product precision
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let dot: f32 = (&a * &b)?.sum_all()?.to_scalar()?;
    let norm_a: f32 = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_b: f32 = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        return Ok(0.0);
    }
    Ok(dot / denom)
}

/// Build a `[1, seq_len, hidden]` delta tensor with `vector` at `position`,
/// zeros elsewhere. Uses the narrow+cat pattern from CLT injection.
///
/// # Shapes
/// - `vector`: `[hidden]`
/// - returns: `[1, seq_len, hidden]`
fn build_position_delta(
    vector: &Tensor,
    seq_len: usize,
    hidden: usize,
    position: usize,
    device: &Device,
) -> candle_mi::Result<Tensor> {
    let zeros = Tensor::zeros((1, seq_len, hidden), DType::F32, device)?;
    let scaled_3d = vector.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, hidden]

    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if position > 0 {
        parts.push(zeros.narrow(1, 0, position)?);
    }
    parts.push(scaled_3d);
    if position + 1 < seq_len {
        parts.push(zeros.narrow(1, position + 1, seq_len - position - 1)?);
    }

    Ok(Tensor::cat(&parts, 1)?)
}

/// Extract logits for the last token position.
///
/// # Shapes
/// - `output`: `[1, seq_len, vocab]`
/// - returns: `[vocab]`
fn last_token_logits(output: &Tensor, seq_len: usize) -> candle_mi::Result<Tensor> {
    Ok(output.get(0)?.get(seq_len - 1)?)
}

/// Softmax over a 1D logits tensor.
///
/// # Shapes
/// - `logits`: `[vocab]`
/// - returns: `[vocab]` (probabilities)
fn softmax_1d(logits: &Tensor) -> candle_mi::Result<Tensor> {
    // PROMOTE: F32 for softmax numerical stability
    let logits = logits.to_dtype(DType::F32)?;
    let max_val: f64 = logits.max(0)?.to_scalar::<f32>()?.into();
    let shifted = (logits - max_val)?;
    let exp = shifted.exp()?;
    let sum: f64 = exp.sum_all()?.to_scalar::<f32>()?.into();
    Ok((exp / sum)?)
}

/// Extract probability for a specific token ID from a probability tensor.
fn extract_prob(probs: &Tensor, token_id: u32) -> candle_mi::Result<f32> {
    // CAST: u32 → usize, token_id fits in usize
    #[allow(clippy::as_conversions)]
    let idx = token_id as usize;
    Ok(probs.get(idx)?.to_scalar()?)
}

/// Get top-k predictions from a probability tensor.
fn top_k_predictions(
    probs: &Tensor,
    tokenizer: &MITokenizer,
    k: usize,
) -> candle_mi::Result<Vec<JsonPrediction>> {
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut results = Vec::with_capacity(k);
    for &(idx, prob) in indexed.iter().take(k) {
        // CAST: usize → u32, vocab indices fit in u32
        #[allow(clippy::as_conversions)]
        let token_id = idx as u32;
        let token = tokenizer.decode(&[token_id])?;
        results.push(JsonPrediction {
            token,
            token_id,
            probability: prob,
        });
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Helpers: analysis
// ---------------------------------------------------------------------------

/// Find the absorption boundary: the earliest observation layer strictly after
/// `injection_layer` where cosine similarity >= threshold.
fn find_absorption_boundary(
    convergence_row: &[f32],
    injection_layer: usize,
    threshold: f32,
) -> Option<usize> {
    for obs in (injection_layer + 1)..convergence_row.len() {
        // INDEX: obs is bounded by convergence_row.len()
        if convergence_row[obs] >= threshold {
            return Some(obs);
        }
    }
    None
}

/// Find a contrastive prompt whose token count matches the clean prompt.
fn find_contrastive_prompt<'a>(
    tokenizer: &MITokenizer,
    clean_tokens: &[u32],
) -> candle_mi::Result<(&'a str, Vec<u32>)> {
    let target_len = clean_tokens.len();
    for &candidate in CONTRASTIVE_CANDIDATES {
        let tokens = tokenizer.encode(candidate)?;
        if tokens.len() == target_len {
            return Ok((candidate, tokens));
        }
    }
    Err(candle_mi::MIError::Tokenizer(format!(
        "no contrastive candidate matches clean prompt length ({target_len} tokens)"
    )))
}

// ---------------------------------------------------------------------------
// Helpers: output
// ---------------------------------------------------------------------------

fn print_convergence_matrix(matrix: &[Vec<f32>], n_layers: usize) {
    println!("\n=== Convergence Matrix (cosine similarity: steered vs natural) ===");
    println!("Rows = injection layer, Cols = observation layer\n");

    // Header
    print!("  Inj\\Obs");
    for obs in 0..n_layers {
        print!("  {:>5}", obs);
    }
    println!();
    print!("  -------");
    for _ in 0..n_layers {
        print!("  -----");
    }
    println!();

    // Rows
    for (inj, row) in matrix.iter().enumerate() {
        print!("  {:>5}  ", inj);
        for &sim in row {
            if sim >= 0.999 {
                print!("  1.000");
            } else if sim >= 0.95 {
                print!("  {sim:.3}");
            } else if sim >= 0.90 {
                print!("  {sim:.3}");
            } else {
                print!("  {sim:.3}");
            }
        }
        println!();
    }
}

fn print_layer_summary(summaries: &[JsonLayerSummary], target_text: &str) {
    println!("\n=== Injection Layer Summary ===");
    println!(
        "  {:>5}  {:>10}  {:>8}  {:>12}",
        "Layer",
        format!("P({target_text})"),
        "KL Div",
        "Absorption"
    );
    println!(
        "  {:>5}  {:>10}  {:>8}  {:>12}",
        "-----", "----------", "--------", "----------"
    );

    for s in summaries {
        let absorption = match s.absorption_layer {
            Some(l) => format!("Layer {l}"),
            None => "--".to_owned(),
        };
        println!(
            "  {:>5}  {:>10}  {:>8.4}  {:>12}",
            s.injection_layer,
            format_probability(s.p_target),
            s.kl_divergence,
            absorption,
        );
    }
}

fn print_strength_sweep(sweep: &[JsonStrengthPoint], best_layer: usize, target_text: &str) {
    println!("\n=== Strength Sweep at Layer {best_layer} ===");
    println!(
        "  {:>8}  {:>10}  {:>8}  {:>12}",
        "Strength",
        format!("P({target_text})"),
        "KL Div",
        "Absorption"
    );
    println!(
        "  {:>8}  {:>10}  {:>8}  {:>12}",
        "--------", "----------", "--------", "----------"
    );

    for pt in sweep {
        let absorption = match pt.absorption_layer {
            Some(l) => format!("Layer {l}"),
            None => "--".to_owned(),
        };
        println!(
            "  {:>8.2}  {:>10}  {:>8.4}  {:>12}",
            pt.strength,
            format_probability(pt.p_target),
            pt.kl_divergence,
            absorption,
        );
    }
}

fn print_interpretation(matrix: &[Vec<f32>], summaries: &[JsonLayerSummary], threshold: f32) {
    println!("\n=== Interpretation ===");

    // Count how many injection layers achieve absorption
    let absorbed: Vec<&JsonLayerSummary> = summaries
        .iter()
        .filter(|s| s.absorption_layer.is_some())
        .collect();

    // CAST: usize → f32, n_layers is small
    #[allow(clippy::as_conversions)]
    let frac = absorbed.len() as f32 / summaries.len() as f32 * 100.0;

    println!(
        "  {}/{} injection layers achieve absorption (threshold {threshold:.2})",
        absorbed.len(),
        summaries.len()
    );
    println!("  ({frac:.0}% of layers converge back to natural computation)\n");

    if !absorbed.is_empty() {
        // Average absorption depth (layers after injection)
        let avg_depth: f32 = absorbed
            .iter()
            .map(|s| {
                // CAST: usize → f32, layer indices are small
                #[allow(clippy::as_conversions)]
                let depth = s.absorption_layer.unwrap_or(0) as f32 - s.injection_layer as f32;
                depth
            })
            .sum::<f32>()
            / absorbed.len() as f32;

        println!("  Average absorption depth: {avg_depth:.1} layers after injection");
        println!(
            "  → The model absorbs external perturbations within ~{:.0} layers on average.",
            avg_depth.ceil()
        );
    }

    // Check diagonal pattern
    let n = matrix.len();
    if n >= 4 {
        let early_sim = matrix[0][n / 2]; // inject early, observe middle
        let mid_sim = matrix[n / 2][n - 1]; // inject middle, observe late
        let late_sim = matrix[n - 2][n - 1]; // inject late, observe last

        println!();
        if early_sim > threshold && mid_sim > threshold {
            println!("  Pattern: ATTRACTOR — the model converges to its natural state");
            println!("  regardless of where the steering is injected.");
        } else if early_sim > threshold && late_sim < 0.9 {
            println!("  Pattern: DEPTH-DEPENDENT — early injections converge (the model");
            println!("  has enough layers to course-correct), but late injections diverge.");
        } else if early_sim < 0.9 && mid_sim < 0.9 {
            println!("  Pattern: MULTIPLE PATHS — the model reaches the same output");
            println!("  through different internal trajectories.");
        } else {
            println!("  Pattern: MIXED — convergence depends on the injection site.");
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: utilities
// ---------------------------------------------------------------------------

/// Rough estimate of F32 weight size in MB.
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    // CAST: usize → f64, values are small
    #[allow(clippy::as_conversions)]
    let params = (12.0 * n_layers as f64 * (hidden as f64).powi(2)) + (hidden as f64 * 128_000.0);
    params * 4.0 / 1_048_576.0
}

fn write_json(path: &Path, output: &JsonOutput) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(candle_mi::MIError::Io)?;
    }
    std::fs::write(path, json).map_err(candle_mi::MIError::Io)?;
    Ok(())
}
