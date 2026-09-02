// SPDX-License-Identifier: MIT OR Apache-2.0

//! Contrastive activation patching of the goal→action signal (CLT-free).
//!
//! The CLT-free causal mirror of the means-ends planning-site / commitment-onset
//! results. For each token-aligned `bright`/`dark` goal-flip pair (clean = on-goal,
//! corrupt = off-goal; they differ at exactly the goal word), we patch the clean
//! residual into the corrupt forward pass at each `(position, layer)` and measure
//! how far the **action logit-diff** `logit(on) − logit(off)` is restored toward
//! the clean run. The position where this recovers traces *where the goal signal
//! flows*; the **planning-site (last-token)** layer-recovery curve gives a
//! **causal onset** — the layer at which patching the decision token alone
//! restores the goal-correct action — comparable to the logit-lens onset.
//!
//! Items come from `scripts/means_ends_generator.py --contrastive`. No CLT is
//! used; this is plain residual-stream patching (`Intervention::PatchAt` at
//! `HookPoint::ResidPost`).
//!
//! ```bash
//! cargo run --release --features transformer,mmap --example contrastive_patch -- \
//!     --model meta-llama/Llama-3.2-1B \
//!     --output docs/experiments/means-ends-prolepsis/contrastive_patch_llama32_1b.json
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

use candle_mi::{HookPoint, HookSpec, Intervention, MIModel};

/// A planning-site patch "recovers" the goal-correct action at the threshold
/// fraction of the clean-vs-corrupt logit-diff gap.
const RECOVERY_THRESHOLD: f32 = 0.5;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "contrastive_patch")]
#[command(about = "CLT-free contrastive activation patching of the goal→action signal")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Contrastive pairs JSON (output of `means_ends_generator.py --contrastive`).
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/step_b_contrastive_pairs.json"
    )]
    items: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/contrastive_patch.json"
    )]
    output: PathBuf,
}

// ── Input ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Pair {
    device: String,
    clean_prompt: String,
    corrupt_prompt: String,
    /// Goal-correct action for the clean prompt (e.g. `on`).
    clean_action: String,
    /// Goal-correct action for the corrupt prompt (e.g. `off`).
    corrupt_action: String,
}

// ── Output ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct PairResult {
    device: String,
    kept: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    skip_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    clean_logit_diff: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    corrupt_logit_diff: Option<f32>,
    /// Token positions that differ between clean and corrupt (the goal word).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    goal_positions: Vec<usize>,
    /// Layer at which patching the planning-site residual reaches the threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    causal_onset_layer: Option<usize>,
    /// Recovery by layer when patching the last (planning-site) token only.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    planning_site_curve: Vec<f32>,
    /// Recovery by layer when patching the goal-word position(s) only.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    goal_pos_curve: Vec<f32>,
}

#[derive(Serialize)]
struct Aggregate {
    n_layers: usize,
    /// Mean planning-site recovery curve over kept pairs.
    planning_site_curve: Vec<f32>,
    /// Mean goal-position recovery curve over kept pairs.
    goal_pos_curve: Vec<f32>,
    /// Median causal-onset layer over kept pairs.
    causal_onset_median: Option<f64>,
}

#[derive(Serialize)]
struct Output {
    model: String,
    n_layers: usize,
    recovery_threshold: f32,
    n_pairs_total: usize,
    n_pairs_kept: usize,
    aggregate: Aggregate,
    pairs: Vec<PairResult>,
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

/// `logit(on) − logit(off)` at the last position of an output `[1, seq, vocab]`.
fn logit_diff(output: &Tensor, seq_len: usize, on_id: u32, off_id: u32) -> candle_mi::Result<f32> {
    let last = output.get(0)?.get(seq_len - 1)?; // [vocab]
    let on = last.get(token_to_usize(on_id)?)?.to_scalar::<f32>()?;
    let off = last.get(token_to_usize(off_id)?)?.to_scalar::<f32>()?;
    Ok(on - off)
}

/// Capture `ResidPost(L)` for every layer; return the per-layer `[seq, hidden]`
/// residuals and the output logits.
fn forward_capture(
    model: &MIModel,
    input: &Tensor,
    n_layers: usize,
) -> candle_mi::Result<(Vec<Tensor>, Tensor)> {
    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::ResidPost(layer));
    }
    let result = model.forward(input, &hooks)?;
    let output = result.output().clone();
    let mut acts = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        acts.push(result.require(&HookPoint::ResidPost(layer))?.get(0)?); // [seq, hidden]
    }
    Ok((acts, output))
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

// ── Per-pair patching ─────────────────────────────────────────────────────────

/// A skipped pair (validation or competence failure) carries only its reason.
fn skipped(device: &str, reason: String) -> PairResult {
    PairResult {
        device: device.to_owned(),
        kept: false,
        skip_reason: Some(reason),
        clean_logit_diff: None,
        corrupt_logit_diff: None,
        goal_positions: Vec::new(),
        causal_onset_layer: None,
        planning_site_curve: Vec::new(),
        goal_pos_curve: Vec::new(),
    }
}

fn patch_pair(model: &MIModel, pair: &Pair, n_layers: usize) -> candle_mi::Result<PairResult> {
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;

    let clean_ids = tokenizer.encode(&pair.clean_prompt)?;
    let corrupt_ids = tokenizer.encode(&pair.corrupt_prompt)?;

    // Token-alignment: equal length, and the differing positions are the goal word.
    if clean_ids.len() != corrupt_ids.len() {
        return Ok(skipped(
            &pair.device,
            format!(
                "not token-aligned (clean {} vs corrupt {} tokens)",
                clean_ids.len(),
                corrupt_ids.len()
            ),
        ));
    }
    let seq_len = clean_ids.len();
    let goal_positions: Vec<usize> = clean_ids
        .iter()
        .zip(corrupt_ids.iter())
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .collect();
    if goal_positions.is_empty() {
        return Ok(skipped(
            &pair.device,
            "clean and corrupt prompts are identical".into(),
        ));
    }

    let on_id = tokenizer.find_token_id(&pair.clean_action)?;
    let off_id = tokenizer.find_token_id(&pair.corrupt_action)?;

    let clean_input = Tensor::new(&clean_ids[..], &device)?.unsqueeze(0)?;
    let corrupt_input = Tensor::new(&corrupt_ids[..], &device)?.unsqueeze(0)?;

    let (clean_acts, clean_out) = forward_capture(model, &clean_input, n_layers)?;
    // The corrupt pass needs no captures: `Intervention::PatchAt` overwrites one
    // position of the residual stream in flight, so the recipient's own
    // activations never have to be read back out and spliced.
    let corrupt_out = model
        .forward(&corrupt_input, &HookSpec::new())?
        .into_output();

    let clean_d = logit_diff(&clean_out, seq_len, on_id, off_id)?;
    let corrupt_d = logit_diff(&corrupt_out, seq_len, on_id, off_id)?;
    // Gate on the *metric*: the goal flip must flip the on/off preference
    // (`clean` prefers `on`, `corrupt` prefers `off`). Strict full-vocab top-1
    // is too harsh here — the goal-only frame's argmax is often a function word
    // even when `on` ≻ `off`, and the logit-diff is what we patch.
    if !(clean_d > 0.0 && corrupt_d < 0.0) {
        return Ok(skipped(
            &pair.device,
            format!(
                "goal does not flip on/off preference (clean_d={clean_d:+.2}, corrupt_d={corrupt_d:+.2})"
            ),
        ));
    }
    let gap = clean_d - corrupt_d;
    if gap.abs() < 1e-6 {
        return Ok(skipped(
            &pair.device,
            "clean/corrupt logit-diff gap ≈ 0".into(),
        ));
    }

    // Patch grid: recovery[layer][pos]. Track the two position-roles we report.
    let mut planning_site_curve = Vec::with_capacity(n_layers);
    let mut goal_pos_curve = Vec::with_capacity(n_layers);
    let output_pos = seq_len - 1;
    for layer in 0..n_layers {
        let src = clean_acts
            .get(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("missing clean layer {layer}")))?;

        let recovery_at = |pos: usize| -> candle_mi::Result<f32> {
            let clean_row = src.get(pos)?; // [hidden]
            let mut hooks = HookSpec::new();
            hooks.intervene(
                HookPoint::ResidPost(layer),
                Intervention::PatchAt {
                    position: pos,
                    value: clean_row,
                },
            );
            let out = model.forward(&corrupt_input, &hooks)?;
            let patched_d = logit_diff(out.output(), seq_len, on_id, off_id)?;
            Ok((patched_d - corrupt_d) / gap)
        };

        planning_site_curve.push(recovery_at(output_pos)?);
        // Goal-position recovery = mean over the differing positions.
        let mut sum = 0.0_f32;
        for &gp in &goal_positions {
            sum += recovery_at(gp)?;
        }
        // CAST: small positive count → f32 for an average.
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let n_goal = goal_positions.len() as f32;
        goal_pos_curve.push(sum / n_goal);
    }

    let causal_onset_layer = planning_site_curve
        .iter()
        .position(|&r| r >= RECOVERY_THRESHOLD);

    Ok(PairResult {
        device: pair.device.clone(),
        kept: true,
        skip_reason: None,
        clean_logit_diff: Some(clean_d),
        corrupt_logit_diff: Some(corrupt_d),
        goal_positions,
        causal_onset_layer,
        planning_site_curve,
        goal_pos_curve,
    })
}

// ── Aggregation ───────────────────────────────────────────────────────────────

/// Element-wise mean of equal-length curves; empty input → empty.
fn mean_curve(curves: &[&[f32]], n_layers: usize) -> candle_mi::Result<Vec<f32>> {
    if curves.is_empty() {
        return Ok(Vec::new());
    }
    let n = count_to_f64(curves.len())?;
    let mut out = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let mut sum = 0.0_f64;
        for c in curves {
            sum += f64::from(*c.get(layer).unwrap_or(&0.0));
        }
        // CAST: averaged recovery back to f32 for the JSON curve.
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        out.push((sum / n) as f32);
    }
    Ok(out)
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

    let pairs: Vec<Pair> = {
        let json = read_to_string(&args.items)?;
        serde_json::from_str(&json).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to parse {}: {e}", args.items.display()))
        })?
    };

    eprintln!("=== Contrastive activation patching (goal→action, CLT-free) ===\n");
    eprintln!("Model: {}", args.model);
    eprintln!("Pairs: {}\n", pairs.len());

    let model = MIModel::from_pretrained(&args.model)?;
    let n_layers = model.num_layers();
    eprintln!("  {n_layers} layers, device={:?}\n", model.device());

    let mut results: Vec<PairResult> = Vec::with_capacity(pairs.len());
    for pair in &pairs {
        let r = patch_pair(&model, pair, n_layers)?;
        if r.kept {
            eprintln!(
                "  [keep] {:<10} clean_d={:+.2} corrupt_d={:+.2} causal-onset L{:?}",
                r.device,
                r.clean_logit_diff.unwrap_or(0.0),
                r.corrupt_logit_diff.unwrap_or(0.0),
                r.causal_onset_layer
            );
        } else {
            eprintln!(
                "  [skip] {:<10} {}",
                r.device,
                r.skip_reason.as_deref().unwrap_or("?")
            );
        }
        results.push(r);
    }

    // Aggregate over kept pairs.
    let kept: Vec<&PairResult> = results.iter().filter(|r| r.kept).collect();
    let ps_curves: Vec<&[f32]> = kept
        .iter()
        .map(|r| r.planning_site_curve.as_slice())
        .collect();
    let goal_curves: Vec<&[f32]> = kept.iter().map(|r| r.goal_pos_curve.as_slice()).collect();
    let onsets: Vec<usize> = kept.iter().filter_map(|r| r.causal_onset_layer).collect();
    let aggregate = Aggregate {
        n_layers,
        planning_site_curve: mean_curve(&ps_curves, n_layers)?,
        goal_pos_curve: mean_curve(&goal_curves, n_layers)?,
        causal_onset_median: median_usize(&onsets)?,
    };

    eprintln!(
        "\n=== Planning-site recovery by layer (mean over {} kept) ===",
        kept.len()
    );
    for (layer, r) in aggregate.planning_site_curve.iter().enumerate() {
        // CAST: bar length in [0, 30] for display; truncation/sign-loss intended.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::as_conversions
        )]
        let bar_len = ((*r).clamp(0.0, 1.0) * 30.0) as usize;
        let bar = "#".repeat(bar_len);
        let mark = if *r >= RECOVERY_THRESHOLD { " <-" } else { "" };
        eprintln!("  L{layer:>2}  {r:+.2}  {bar}{mark}");
    }
    eprintln!(
        "\ncausal onset (planning-site recovery ≥ {RECOVERY_THRESHOLD}, median): {:?}  ({}/{} pairs kept)",
        aggregate.causal_onset_median,
        kept.len(),
        pairs.len()
    );

    let output = Output {
        model: args.model.clone(),
        n_layers,
        recovery_threshold: RECOVERY_THRESHOLD,
        n_pairs_total: pairs.len(),
        n_pairs_kept: kept.len(),
        aggregate,
        pairs: results,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    };
    write_json(&args.output, &output)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}
