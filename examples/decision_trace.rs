// SPDX-License-Identifier: MIT OR Apache-2.0

//! Depth-axis irrevocability: the running action margin `logit(correct) −
//! logit(alternative)` by layer at the planning site, and whether *later* layers
//! erode or flip it (CLT-free).
//!
//! Follows the means-ends commitment-onset and DLA results. At the planning-site
//! token we logit-lens **every** layer's residual and track the signed action
//! margin `d[L] = logit(correct) − logit(alternative)`. To quantify "how often,
//! and by how much, a later layer threatens the decision" we use **peak-vs-final**
//! statistics (bias-free, unlike an absolute zero-crossing — `on` is generically a
//! more frequent token than `off`): the margin **peaks** at some layer, and the
//! **late erosion** is `peak − final`. A decision is **flipped by late layers**
//! when the correct action led at the peak (`peak > 0`) yet the readout is wrong
//! (`final ≤ 0`). This is the depth-axis analogue of STRIPS non-backtracking — a
//! commitment in *layers*, not in *tokens*.
//!
//! Items are the means-ends `on_off` set (`means_ends_generator.py --controlled`).
//! No CLT is used.
//!
//! ```bash
//! cargo run --release --features transformer,mmap --example decision_trace -- \
//!     --model meta-llama/Llama-3.2-1B \
//!     --output docs/experiments/means-ends-prolepsis/decision_trace_llama32_1b.json
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Tensor;
use clap::Parser;
use serde::{Deserialize, Serialize};

use candle_mi::{HookPoint, HookSpec, MIModel};

/// An item counts as "eroded" when the late layers give back at least this many
/// logits from the peak margin.
const EROSION_EPS: f32 = 0.05;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "decision_trace")]
#[command(about = "Depth-axis irrevocability: action margin by layer, peak-vs-final erosion")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Means-ends items JSON (each `{correct, alternative, prompt, ...}`).
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/step_b_items.json"
    )]
    items: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/decision_trace.json"
    )]
    output: PathBuf,
}

// ── Input ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Item {
    #[serde(default)]
    device: Option<String>,
    /// Goal-correct action token.
    correct: String,
    /// Contrastive action token.
    alternative: String,
    prompt: String,
}

// ── Output ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ItemTrace {
    #[serde(skip_serializing_if = "Option::is_none")]
    device: Option<String>,
    correct: String,
    /// `logit(correct) − logit(alternative)` at the final layer (the readout).
    final_diff: f32,
    /// Final-layer decision is correct (`final_diff > 0`).
    correct_decision: bool,
    /// Layer of the maximum margin along the trajectory.
    peak_layer: usize,
    /// Maximum margin along the trajectory.
    peak_diff: f32,
    /// `peak_diff − final_diff` — how many logits the later layers gave back.
    late_erosion: f32,
    /// `peak_diff > 0` but `final_diff ≤ 0`: the correct action led at the peak yet
    /// lost at the readout (a late-layer flip).
    flipped_by_late: bool,
    /// Last layer's step `d[last] − d[last−1]` (< 0 ⇒ the last layer threatens).
    last_delta: f32,
    /// Signed margin `logit(correct) − logit(alternative)` by layer.
    diff_by_layer: Vec<f32>,
}

#[derive(Serialize)]
struct Aggregate {
    n_layers: usize,
    n_items: usize,
    /// Items whose final-layer decision is correct.
    n_correct: usize,
    /// Median peak-margin layer over correct items.
    peak_layer_median: Option<f64>,
    /// Mean `peak − final` over correct items (logits the late layers gave back).
    mean_late_erosion: f32,
    /// Fraction of correct items eroded by more than `EROSION_EPS` from peak.
    frac_eroded: f64,
    /// Mean of the last layer's step over correct items.
    last_layer_mean_delta: f32,
    /// Fraction of correct items whose last layer reduces the margin.
    frac_last_layer_threatens: f64,
    /// Count of items (over all) the late layers flipped from leading to wrong.
    n_flipped_by_late: usize,
    /// Of the incorrect items, the fraction that were leading at the peak.
    frac_wrong_that_are_late_flips: f64,
    /// Mean per-layer step `d[L] − d[L−1]` over correct items (layer 0 = 0).
    mean_delta_by_layer: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    model: String,
    n_layers: usize,
    erosion_eps: f32,
    n_pairs_total: usize,
    aggregate: Aggregate,
    items: Vec<ItemTrace>,
    elapsed_secs: f64,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn read_to_string(path: &Path) -> candle_mi::Result<String> {
    fs::read_to_string(path)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display())))
}

fn count_to_f64(count: usize) -> candle_mi::Result<f64> {
    let as_u32 = u32::try_from(count)
        .map_err(|e| candle_mi::MIError::Config(format!("count {count} exceeds u32: {e}")))?;
    Ok(f64::from(as_u32))
}

fn token_to_usize(id: u32) -> candle_mi::Result<usize> {
    usize::try_from(id)
        .map_err(|e| candle_mi::MIError::Config(format!("token id {id} exceeds usize: {e}")))
}

/// `logit(correct) − logit(alternative)` from a `[hidden]` residual via the real
/// final-norm + unembedding.
fn margin_at(
    model: &MIModel,
    resid_site: &Tensor,
    correct_id: u32,
    alt_id: u32,
) -> candle_mi::Result<f32> {
    let logits = model.project_to_vocab(&resid_site.unsqueeze(0)?)?; // [1, vocab]
    let row = logits.get(0)?;
    let c = row.get(token_to_usize(correct_id)?)?.to_scalar::<f32>()?;
    let a = row.get(token_to_usize(alt_id)?)?.to_scalar::<f32>()?;
    Ok(c - a)
}

fn median_usize(values: &[usize]) -> candle_mi::Result<Option<f64>> {
    if values.is_empty() {
        return Ok(None);
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        Ok(Some(count_to_f64(*v.get(mid).unwrap_or(&0))?))
    } else {
        let a = count_to_f64(*v.get(mid.saturating_sub(1)).unwrap_or(&0))?;
        let b = count_to_f64(*v.get(mid).unwrap_or(&0))?;
        Ok(Some(a.midpoint(b)))
    }
}

fn write_json(path: &Path, output: &Output) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;
    eprintln!("\nOutput written to {}", path.display());
    Ok(())
}

// ── Per-item trace ─────────────────────────────────────────────────────────────

fn trace_item(model: &MIModel, item: &Item, n_layers: usize) -> candle_mi::Result<ItemTrace> {
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    let correct_id = tokenizer.find_token_id(&item.correct)?;
    let alt_id = tokenizer.find_token_id(&item.alternative)?;

    let ids = tokenizer.encode(&item.prompt)?;
    let seq_len = ids.len();
    let site = seq_len - 1;
    let input = Tensor::new(&ids[..], model.device())?.unsqueeze(0)?;

    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::ResidPost(layer));
    }
    let result = model.forward(&input, &hooks)?;

    let mut diff_by_layer = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let resid = result
            .require(&HookPoint::ResidPost(layer))?
            .get(0)?
            .get(site)?;
        diff_by_layer.push(margin_at(model, &resid, correct_id, alt_id)?);
    }

    let final_diff = diff_by_layer.last().copied().unwrap_or(0.0);
    let correct_decision = final_diff > 0.0;

    // Peak margin and late erosion (peak − final), both bias-free.
    let mut peak_layer = 0usize;
    let mut peak_diff = f32::NEG_INFINITY;
    for (layer, &d) in diff_by_layer.iter().enumerate() {
        if d > peak_diff {
            peak_diff = d;
            peak_layer = layer;
        }
    }
    let late_erosion = peak_diff - final_diff;
    let flipped_by_late = peak_diff > 0.0 && final_diff <= 0.0;
    let last_delta = {
        let last = diff_by_layer.last().copied().unwrap_or(0.0);
        let penult = diff_by_layer
            .get(n_layers.saturating_sub(2))
            .copied()
            .unwrap_or(0.0);
        last - penult
    };

    Ok(ItemTrace {
        device: item.device.clone(),
        correct: item.correct.clone(),
        final_diff,
        correct_decision,
        peak_layer,
        peak_diff,
        late_erosion,
        flipped_by_late,
        last_delta,
        diff_by_layer,
    })
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let t_start = Instant::now();

    let items: Vec<Item> = {
        let json = read_to_string(&args.items)?;
        serde_json::from_str(&json).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to parse {}: {e}", args.items.display()))
        })?
    };

    eprintln!("=== Decision trace (depth-axis irrevocability, CLT-free) ===\n");
    eprintln!("Model: {}", args.model);
    eprintln!("Items: {}\n", items.len());

    let model = MIModel::from_pretrained(&args.model)?;
    let n_layers = model.num_layers();
    eprintln!("  {n_layers} layers, device={:?}\n", model.device());

    let mut traces: Vec<ItemTrace> = Vec::with_capacity(items.len());
    for item in &items {
        traces.push(trace_item(&model, item, n_layers)?);
    }

    // Correct-decision subset (the erosion/peak denominator).
    let correct: Vec<&ItemTrace> = traces.iter().filter(|t| t.correct_decision).collect();
    let n_correct = correct.len();
    let n_correct_f64 = count_to_f64(n_correct.max(1))?;

    let peak_layers: Vec<usize> = correct.iter().map(|t| t.peak_layer).collect();
    let mean_late_erosion = {
        let sum: f64 = correct.iter().map(|t| f64::from(t.late_erosion)).sum();
        // CAST: mean erosion to f32 for the JSON scalar.
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let v = (sum / n_correct_f64) as f32;
        v
    };
    let frac_eroded = count_to_f64(
        correct
            .iter()
            .filter(|t| t.late_erosion > EROSION_EPS)
            .count(),
    )? / n_correct_f64;
    let last_layer_mean_delta = {
        let sum: f64 = correct.iter().map(|t| f64::from(t.last_delta)).sum();
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let v = (sum / n_correct_f64) as f32;
        v
    };
    let frac_last_layer_threatens =
        count_to_f64(correct.iter().filter(|t| t.last_delta < 0.0).count())? / n_correct_f64;

    let n_flipped_by_late = traces.iter().filter(|t| t.flipped_by_late).count();
    let n_wrong = traces.len() - n_correct;
    let frac_wrong_that_are_late_flips = if n_wrong > 0 {
        count_to_f64(n_flipped_by_late)? / count_to_f64(n_wrong)?
    } else {
        0.0
    };

    // Mean per-layer step Δd over correct items.
    let mut mean_delta = vec![0.0_f32; n_layers];
    for t in &correct {
        for layer in 1..n_layers {
            let cur = t.diff_by_layer.get(layer).copied().unwrap_or(0.0);
            let prev = t.diff_by_layer.get(layer - 1).copied().unwrap_or(0.0);
            if let Some(slot) = mean_delta.get_mut(layer) {
                *slot += cur - prev;
            }
        }
    }
    if n_correct > 0 {
        for d in &mut mean_delta {
            // CAST: averaged Δ back to f32 for the JSON curve.
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            {
                *d = (f64::from(*d) / n_correct_f64) as f32;
            }
        }
    }

    let aggregate = Aggregate {
        n_layers,
        n_items: traces.len(),
        n_correct,
        peak_layer_median: median_usize(&peak_layers)?,
        mean_late_erosion,
        frac_eroded,
        last_layer_mean_delta,
        frac_last_layer_threatens,
        n_flipped_by_late,
        frac_wrong_that_are_late_flips,
        mean_delta_by_layer: mean_delta,
    };

    eprintln!(
        "=== Depth-axis irrevocability ({n_correct}/{} correct) ===",
        traces.len()
    );
    if let Some(pl) = aggregate.peak_layer_median {
        // CAST: layer fraction for display only.
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let frac = pl / n_layers as f64;
        eprintln!("  peak-margin layer (median): {pl:.1} / {n_layers}  ({frac:.2} depth)");
    }
    eprintln!(
        "  late erosion (peak − final): mean {:+.2} logits, {:.0}% of items eroded",
        aggregate.mean_late_erosion,
        aggregate.frac_eroded * 100.0
    );
    eprintln!(
        "  last layer reduces the margin: {:.0}% of items (mean Δ_last = {:+.3})",
        aggregate.frac_last_layer_threatens * 100.0,
        aggregate.last_layer_mean_delta
    );
    eprintln!(
        "  flipped by late layers (led at peak, wrong at readout): {} items ({:.0}% of the {} wrong)",
        aggregate.n_flipped_by_late,
        aggregate.frac_wrong_that_are_late_flips * 100.0,
        n_wrong
    );
    eprintln!("\n  mean margin-step by layer (Δd):");
    for (layer, d) in aggregate.mean_delta_by_layer.iter().enumerate() {
        let glyph = if *d >= 0.0 { '+' } else { '-' };
        // CAST: bar length for display; small, truncation/sign-loss intended.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::as_conversions
        )]
        let bar_len = ((d.abs() * 8.0) as usize).min(40);
        eprintln!(
            "    L{layer:>2}  {d:>+8.3}  {}",
            glyph.to_string().repeat(bar_len)
        );
    }

    let output = Output {
        model: args.model.clone(),
        n_layers,
        erosion_eps: EROSION_EPS,
        n_pairs_total: traces.len(),
        aggregate,
        items: traces,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    };
    write_json(&args.output, &output)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}
