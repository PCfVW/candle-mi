// SPDX-License-Identifier: MIT OR Apache-2.0

//! MLP-vs-attention direct logit attribution of the goal→action signal (CLT-free).
//!
//! The mechanism behind compute-then-readout (probe (b), following the contrastive
//! activation patching of `examples/contrastive_patch`). For each token-aligned
//! `bright`/`dark` goal-flip pair we decompose the action logit-diff
//! `logit(on) − logit(off)` at the planning-site token into per-layer `AttnOut`
//! and `MlpOut` contributions, and isolate the **goal-driven** part by contrasting
//! clean (on-goal) vs corrupt (off-goal).
//!
//! Attribution is by **component ablation** through the real readout: for the
//! planning-site final residual `R_final` and a component `c` (a block's output),
//! `contrib(c) = logit_diff(proj(R_final)) − logit_diff(proj(R_final − c))`, where
//! `proj` is `MIModel::project_to_vocab` (final-norm + unembed). The **goal-driven**
//! contribution is `contrib_clean − contrib_corrupt`. No CLT is used.
//!
//! ```bash
//! cargo run --release --features transformer,mmap --example action_dla -- \
//!     --model meta-llama/Llama-3.2-1B \
//!     --output docs/experiments/means-ends-prolepsis/action_dla_llama32_1b.json
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

/// Cumulative goal-driven contribution fraction defining the DLA onset.
const ONSET_FRACTION: f32 = 0.5;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "action_dla")]
#[command(about = "CLT-free MLP-vs-attention DLA of the goal→action signal")]
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
        default_value = "docs/experiments/means-ends-prolepsis/action_dla.json"
    )]
    output: PathBuf,
}

// ── Input ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Pair {
    device: String,
    clean_prompt: String,
    corrupt_prompt: String,
    clean_action: String,
    corrupt_action: String,
}

// ── Output ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct PairResult {
    device: String,
    kept: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    skip_reason: Option<String>,
    /// `Σ contribs` ÷ `(base − embed)` per prompt (clean) — ≈ 1 ⇒ additive.
    #[serde(skip_serializing_if = "Option::is_none")]
    additivity_ratio: Option<f32>,
    /// Goal-driven attention contribution to `logit(on)−logit(off)` by layer.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    goal_attn: Vec<f32>,
    /// Goal-driven MLP contribution by layer.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    goal_mlp: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dla_onset_layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attn_total: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mlp_total: Option<f32>,
}

#[derive(Serialize)]
struct Aggregate {
    n_layers: usize,
    goal_attn_curve: Vec<f32>,
    goal_mlp_curve: Vec<f32>,
    dla_onset_median: Option<f64>,
    attn_total: f32,
    mlp_total: f32,
}

#[derive(Serialize)]
struct Output {
    model: String,
    n_layers: usize,
    onset_fraction: f32,
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

/// `logit(on) − logit(off)` from a `[1, vocab]` logits row.
fn logit_diff_row(logits: &Tensor, on_id: u32, off_id: u32) -> candle_mi::Result<f32> {
    let row = logits.get(0)?; // [vocab]
    let on = row.get(token_to_usize(on_id)?)?.to_scalar::<f32>()?;
    let off = row.get(token_to_usize(off_id)?)?.to_scalar::<f32>()?;
    Ok(on - off)
}

/// `logit(on) − logit(off)` after projecting a `[hidden]` planning-site residual
/// through the model's final norm + unembedding.
fn proj_logit_diff(
    model: &MIModel,
    resid_site: &Tensor,
    on_id: u32,
    off_id: u32,
) -> candle_mi::Result<f32> {
    let logits = model.project_to_vocab(&resid_site.unsqueeze(0)?)?; // [1, vocab]
    logit_diff_row(&logits, on_id, off_id)
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

/// Element-wise mean of equal-length curves.
fn mean_curve(curves: &[&[f32]], n_layers: usize) -> candle_mi::Result<Vec<f32>> {
    if curves.is_empty() {
        return Ok(vec![0.0; n_layers]);
    }
    let n = count_to_f64(curves.len())?;
    let mut out = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let mut sum = 0.0_f64;
        for c in curves {
            sum += f64::from(*c.get(layer).unwrap_or(&0.0));
        }
        // CAST: averaged contribution back to f32 for the JSON curve.
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        out.push((sum / n) as f32);
    }
    Ok(out)
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

// ── Per-prompt DLA ─────────────────────────────────────────────────────────────

/// Per-prompt direct logit attribution at the planning site.
struct PromptDla {
    /// `R_final = ResidPost(last)[site]` — `[hidden]`.
    r_final: Tensor,
    /// `logit(on) − logit(off)` at the planning site (the actual readout).
    base_diff: f32,
    /// Per-layer attention `AttnOut(L)[site]` — `[hidden]`.
    attn: Vec<Tensor>,
    /// Per-layer MLP `MlpOut(L)[site]` — `[hidden]`.
    mlp: Vec<Tensor>,
}

/// Forward a prompt, capturing `AttnOut`/`MlpOut`/`ResidPost(last)` at the
/// planning-site (last) position.
fn forward_dla(
    model: &MIModel,
    prompt: &str,
    n_layers: usize,
    on_id: u32,
    off_id: u32,
) -> candle_mi::Result<PromptDla> {
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    let ids = tokenizer.encode(prompt)?;
    let seq_len = ids.len();
    let site = seq_len - 1;
    let input = Tensor::new(&ids[..], model.device())?.unsqueeze(0)?;

    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::AttnOut(layer));
        hooks.capture(HookPoint::MlpOut(layer));
    }
    hooks.capture(HookPoint::ResidPost(n_layers - 1));
    let result = model.forward(&input, &hooks)?;

    let r_final = result
        .require(&HookPoint::ResidPost(n_layers - 1))?
        .get(0)?
        .get(site)?; // [hidden]
    let base_diff = proj_logit_diff(model, &r_final, on_id, off_id)?;

    let mut attn = Vec::with_capacity(n_layers);
    let mut mlp = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        attn.push(
            result
                .require(&HookPoint::AttnOut(layer))?
                .get(0)?
                .get(site)?,
        );
        mlp.push(
            result
                .require(&HookPoint::MlpOut(layer))?
                .get(0)?
                .get(site)?,
        );
    }
    Ok(PromptDla {
        r_final,
        base_diff,
        attn,
        mlp,
    })
}

/// Root-mean-square of a `[hidden]` vector.
fn rms(v: &Tensor) -> candle_mi::Result<f32> {
    let ms = v.sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok(ms.sqrt())
}

/// Frozen-final-norm direct logit attribution of one component to
/// `logit(on) − logit(off)`. Projecting a component alone applies the final
/// RMSNorm with *its own* scale; rescaling by `rms(c) / rms(R_final)` restores the
/// shared final-residual scale, so the per-component contributions are additive
/// (per-component *ablation* is not, because RMSNorm renormalises on removal).
fn contrib(
    model: &MIModel,
    component: &Tensor,
    rms_final: f32,
    on_id: u32,
    off_id: u32,
) -> candle_mi::Result<f32> {
    let ld = proj_logit_diff(model, component, on_id, off_id)?;
    Ok((rms(component)? / rms_final) * ld)
}

fn skipped(device: &str, reason: String) -> PairResult {
    PairResult {
        device: device.to_owned(),
        kept: false,
        skip_reason: Some(reason),
        additivity_ratio: None,
        goal_attn: Vec::new(),
        goal_mlp: Vec::new(),
        dla_onset_layer: None,
        attn_total: None,
        mlp_total: None,
    }
}

fn dla_pair(model: &MIModel, pair: &Pair, n_layers: usize) -> candle_mi::Result<PairResult> {
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    let on_id = tokenizer.find_token_id(&pair.clean_action)?;
    let off_id = tokenizer.find_token_id(&pair.corrupt_action)?;

    let clean = forward_dla(model, &pair.clean_prompt, n_layers, on_id, off_id)?;
    let corrupt = forward_dla(model, &pair.corrupt_prompt, n_layers, on_id, off_id)?;

    // Keep-gate on the metric: the goal flip must flip the on/off preference.
    if !(clean.base_diff > 0.0 && corrupt.base_diff < 0.0) {
        return Ok(skipped(
            &pair.device,
            format!(
                "goal does not flip on/off preference (clean={:+.2}, corrupt={:+.2})",
                clean.base_diff, corrupt.base_diff
            ),
        ));
    }

    // Frozen final-norm scale per prompt (shared across that prompt's components).
    let rms_clean = rms(&clean.r_final)?;
    let rms_corrupt = rms(&corrupt.r_final)?;

    let mut goal_attn = Vec::with_capacity(n_layers);
    let mut goal_mlp = Vec::with_capacity(n_layers);
    // Additivity sanity (clean): Σ component contribs (incl. embedding) vs base_diff.
    let mut sum_contribs_clean = 0.0_f32;
    let mut embed_clean = clean.r_final.clone();
    for layer in 0..n_layers {
        let attn_c = clean
            .attn
            .get(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("clean attn {layer}")))?;
        let mlp_c = clean
            .mlp
            .get(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("clean mlp {layer}")))?;
        let attn_k = corrupt
            .attn
            .get(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("corrupt attn {layer}")))?;
        let mlp_k = corrupt
            .mlp
            .get(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("corrupt mlp {layer}")))?;

        let attn_clean = contrib(model, attn_c, rms_clean, on_id, off_id)?;
        let mlp_clean = contrib(model, mlp_c, rms_clean, on_id, off_id)?;
        let attn_corrupt = contrib(model, attn_k, rms_corrupt, on_id, off_id)?;
        let mlp_corrupt = contrib(model, mlp_k, rms_corrupt, on_id, off_id)?;

        goal_attn.push(attn_clean - attn_corrupt);
        goal_mlp.push(mlp_clean - mlp_corrupt);

        sum_contribs_clean += attn_clean + mlp_clean;
        embed_clean = ((embed_clean - attn_c)? - mlp_c)?;
    }

    // Additivity ratio (clean): (Σ block contribs + embedding contrib) ÷ base_diff.
    // The frozen-norm DLA is additive by construction, so this should be ≈ 1.
    let embed_contrib = contrib(model, &embed_clean, rms_clean, on_id, off_id)?;
    let additivity_ratio = if clean.base_diff.abs() > 1e-6 {
        Some((sum_contribs_clean + embed_contrib) / clean.base_diff)
    } else {
        None
    };

    // Onset: first layer where cumulative goal-driven sum ≥ ONSET_FRACTION × total.
    let total: f32 = goal_attn.iter().chain(goal_mlp.iter()).sum();
    let dla_onset_layer = if total.abs() > 1e-6 {
        let target = ONSET_FRACTION * total;
        let mut cum = 0.0_f32;
        let mut onset = None;
        for layer in 0..n_layers {
            cum += goal_attn.get(layer).copied().unwrap_or(0.0)
                + goal_mlp.get(layer).copied().unwrap_or(0.0);
            // Compare by sign of `total` so it works whether total is ±.
            if (total > 0.0 && cum >= target) || (total < 0.0 && cum <= target) {
                onset = Some(layer);
                break;
            }
        }
        onset
    } else {
        None
    };

    Ok(PairResult {
        device: pair.device.clone(),
        kept: true,
        skip_reason: None,
        additivity_ratio,
        attn_total: Some(goal_attn.iter().sum()),
        mlp_total: Some(goal_mlp.iter().sum()),
        goal_attn,
        goal_mlp,
        dla_onset_layer,
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

    let pairs: Vec<Pair> = {
        let json = read_to_string(&args.items)?;
        serde_json::from_str(&json).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to parse {}: {e}", args.items.display()))
        })?
    };

    eprintln!("=== MLP-vs-attention DLA (goal→action, CLT-free) ===\n");
    eprintln!("Model: {}", args.model);
    eprintln!("Pairs: {}\n", pairs.len());

    let model = MIModel::from_pretrained(&args.model)?;
    let n_layers = model.num_layers();
    eprintln!("  {n_layers} layers, device={:?}\n", model.device());

    let mut results: Vec<PairResult> = Vec::with_capacity(pairs.len());
    for pair in &pairs {
        let r = dla_pair(&model, pair, n_layers)?;
        if r.kept {
            eprintln!(
                "  [keep] {:<10} onset L{:?}  attn_total={:+.2} mlp_total={:+.2} (additivity {:.2})",
                r.device,
                r.dla_onset_layer,
                r.attn_total.unwrap_or(0.0),
                r.mlp_total.unwrap_or(0.0),
                r.additivity_ratio.unwrap_or(0.0),
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

    let kept: Vec<&PairResult> = results.iter().filter(|r| r.kept).collect();
    let attn_curves: Vec<&[f32]> = kept.iter().map(|r| r.goal_attn.as_slice()).collect();
    let mlp_curves: Vec<&[f32]> = kept.iter().map(|r| r.goal_mlp.as_slice()).collect();
    let onsets: Vec<usize> = kept.iter().filter_map(|r| r.dla_onset_layer).collect();
    let attn_total: f32 = kept.iter().filter_map(|r| r.attn_total).sum();
    let mlp_total: f32 = kept.iter().filter_map(|r| r.mlp_total).sum();
    let aggregate = Aggregate {
        n_layers,
        goal_attn_curve: mean_curve(&attn_curves, n_layers)?,
        goal_mlp_curve: mean_curve(&mlp_curves, n_layers)?,
        dla_onset_median: median_usize(&onsets)?,
        attn_total,
        mlp_total,
    };

    eprintln!(
        "\n=== Goal-driven DLA by layer (mean over {} kept) ===",
        kept.len()
    );
    eprintln!("  {:>4}  {:>10}  {:>10}", "L", "goal_attn", "goal_mlp");
    for layer in 0..n_layers {
        let a = aggregate.goal_attn_curve.get(layer).copied().unwrap_or(0.0);
        let m = aggregate.goal_mlp_curve.get(layer).copied().unwrap_or(0.0);
        eprintln!("  {layer:>4}  {a:>+10.3}  {m:>+10.3}");
    }
    eprintln!(
        "\nDLA onset (cumulative ≥ {ONSET_FRACTION}, median): {:?}  | attn_total={:+.2} mlp_total={:+.2}  ({}/{} kept)",
        aggregate.dla_onset_median,
        aggregate.attn_total,
        aggregate.mlp_total,
        kept.len(),
        pairs.len()
    );

    let output = Output {
        model: args.model.clone(),
        n_layers,
        onset_fraction: ONSET_FRACTION,
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
