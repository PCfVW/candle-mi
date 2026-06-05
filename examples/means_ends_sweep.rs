// SPDX-License-Identifier: MIT OR Apache-2.0

//! Means-ends prolepsis — Step B: suppress-plus-inject planning-site sweep.
//!
//! The actual prolepsis test for the means-ends `on_off` cell (Step A established
//! the model commits; the vocab scan found injectable features). For each item we
//! **suppress** the goal-correct action feature and **inject** the alternative
//! action feature *at a swept token position*, across all downstream layers, and
//! measure `P(alternative)` at the output. The position where injection maximally
//! redirects the commitment is the **planning site** (the figure13 / COLM
//! protocol). Permuting the Initial/Goal clause order dissociates *where* the
//! commitment lives — `goal`-bound, `information-completion` (the second clause,
//! = STRIPS's precondition antecedent), or `output-adjacent` (the `stem`).
//!
//! Items come from `scripts/means_ends_generator.py --controlled`
//! (`order`-tagged, clause-segment-annotated, device-once). Default features are
//! the vocab-scan picks (`docs/experiments/means-ends-prolepsis/action_token_inject_candidates.json`):
//! inject `on` = L25:78640, suppress `off` = L24:92568; per item the inject side
//! is the *alternative*'s feature and the suppress side is the *correct*'s.
//!
//! ```bash
//! cargo run --features clt,transformer,mmap --release --example means_ends_sweep -- \
//!     --strength-grid 2,5,10,25,50 \
//!     --output docs/experiments/means-ends-prolepsis/step_b_sweep_gemma2_2b_2.5m.json
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Tensor;
use clap::Parser;
use serde::{Deserialize, Serialize};

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{MIModel, MITokenizer, extract_token_prob};

/// Spike is "localized" (a real commitment site) when `P(alt)` at the best cell
/// reaches this multiple of the no-intervention baseline (COLM-style gate).
const LOCALIZED_RATIO: f32 = 2.0;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "means_ends_sweep")]
#[command(about = "Step B: suppress-plus-inject planning-site sweep for the on_off cell")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// `HuggingFace` `CLT` repository.
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-2.5M")]
    clt_repo: String,

    /// Controlled Step-B items JSON (output of `means_ends_generator.py --controlled`).
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/step_b_items.json"
    )]
    items: PathBuf,

    /// Inject feature for the `on` action, `L<layer>:<index>`.
    #[arg(long, default_value = "L25:78640")]
    feature_on: String,

    /// Inject feature for the `off` action, `L<layer>:<index>`.
    #[arg(long, default_value = "L24:92568")]
    feature_off: String,

    /// Steering strengths to sweep (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "2,5,10,25,50")]
    strength_grid: Vec<f32>,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/step_b_sweep_gemma2_2b_2.5m.json"
    )]
    output: PathBuf,
}

// ── Input items ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Segments {
    initial: String,
    goal: String,
    stem: String,
}

#[derive(Deserialize)]
struct Item {
    id: usize,
    order: String,
    device: String,
    /// Goal-correct (committed) action token.
    correct: String,
    /// Contrastive action token (the one we inject).
    alternative: String,
    segments: Segments,
    prompt: String,
}

// ── Output ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone, Copy)]
struct SweepCell {
    strength: f32,
    position: usize,
    prob: f32,
}

/// Inclusive token-index span `[first, last]` of a clause in the tokenized prompt.
#[derive(Serialize, Clone, Copy)]
struct Span {
    first: usize,
    last: usize,
}

#[derive(Serialize)]
struct ItemResult {
    id: usize,
    order: String,
    device: String,
    /// `commit_off` (inject `on`, strong) or `commit_on` (inject `off`, weak).
    direction: String,
    correct: String,
    alternative: String,
    inject_feature: CltFeatureId,
    suppress_feature: CltFeatureId,
    baseline_prob: f32,
    spike_strength: f32,
    spike_position: usize,
    spike_prob: f32,
    spike_ratio: f32,
    /// Clause the spike position falls in: `initial` / `goal` / `stem` / `other`.
    spike_clause: String,
    localized: bool,
    initial_span: Span,
    goal_span: Span,
    stem_span: Span,
    grid: Vec<SweepCell>,
}

#[derive(Serialize)]
struct GroupSummary {
    /// `<order>/<direction>`.
    key: String,
    n: usize,
    /// Spike-clause histogram within the group.
    clause_counts: BTreeMap<String, usize>,
    modal_clause: String,
    mean_spike_ratio: f64,
    localized_fraction: f64,
}

#[derive(Serialize)]
struct Output {
    model: String,
    clt_repo: String,
    feature_on: CltFeatureId,
    feature_off: CltFeatureId,
    strength_grid: Vec<f32>,
    localized_ratio: f32,
    n_items: usize,
    summaries: Vec<GroupSummary>,
    items: Vec<ItemResult>,
    elapsed_secs: f64,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse a `L<layer>:<index>` feature id.
fn parse_feature(s: &str) -> candle_mi::Result<CltFeatureId> {
    let rest = s.strip_prefix('L').ok_or_else(|| {
        candle_mi::MIError::Config(format!("feature must start with 'L', got '{s}'"))
    })?;
    let (layer_str, index_str) = rest.split_once(':').ok_or_else(|| {
        candle_mi::MIError::Config(format!("feature must be 'L<layer>:<index>', got '{s}'"))
    })?;
    let layer = layer_str
        .parse()
        .map_err(|e| candle_mi::MIError::Config(format!("invalid layer in '{s}': {e}")))?;
    let index = index_str
        .parse()
        .map_err(|e| candle_mi::MIError::Config(format!("invalid index in '{s}': {e}")))?;
    Ok(CltFeatureId { layer, index })
}

/// Load + parse the controlled items JSON.
fn load_items(path: &Path) -> candle_mi::Result<Vec<Item>> {
    let json = fs::read_to_string(path).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
    })?;
    serde_json::from_str(&json)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to parse {}: {e}", path.display())))
}

/// Lossless count → `f64` (small counts).
fn count_to_f64(count: usize) -> candle_mi::Result<f64> {
    let as_u32 = u32::try_from(count)
        .map_err(|e| candle_mi::MIError::Config(format!("count {count} exceeds u32: {e}")))?;
    Ok(f64::from(as_u32))
}

/// Character `[start, end)` of `seg` within `prompt`.
fn char_range(prompt: &str, seg: &str) -> candle_mi::Result<(usize, usize)> {
    let start = prompt.find(seg).ok_or_else(|| {
        candle_mi::MIError::Config(format!("segment not found in prompt: {seg:?}"))
    })?;
    Ok((start, start + seg.len()))
}

/// Inclusive token-index span of the tokens whose start char lies in `[cs, ce)`.
/// Returns `(first, last)`; falls back to `(0, 0)` if no token matches.
fn span_for(offsets: &[(usize, usize)], cs: usize, ce: usize) -> Span {
    let mut first: Option<usize> = None;
    let mut last = 0usize;
    for (i, &(ts, te)) in offsets.iter().enumerate() {
        // Skip zero-length tokens (e.g. BOS at (0, 0)).
        if te <= ts {
            continue;
        }
        if ts >= cs && ts < ce {
            first.get_or_insert(i);
            last = i;
        }
    }
    let first = first.unwrap_or(0);
    Span { first, last }
}

/// Classify a token index into a clause label via the precomputed spans.
const fn clause_of(pos: usize, initial: Span, goal: Span, stem: Span) -> &'static str {
    if pos >= initial.first && pos <= initial.last {
        "initial"
    } else if pos >= goal.first && pos <= goal.last {
        "goal"
    } else if pos >= stem.first && pos <= stem.last {
        "stem"
    } else {
        "other"
    }
}

/// Build the downstream `(feature, target_layer)` entries for one feature.
fn downstream_entries(feat: CltFeatureId, n_layers: usize) -> Vec<(CltFeatureId, usize)> {
    (feat.layer..n_layers).map(|l| (feat, l)).collect()
}

/// Serialize the results to JSON, creating parent dirs as needed.
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

// ── Per-item sweep ────────────────────────────────────────────────────────────

/// Run the strength×position sweep for one item; returns its scored result.
#[allow(clippy::too_many_arguments)]
fn sweep_item(
    model: &MIModel,
    clt: &CrossLayerTranscoder,
    tokenizer: &MITokenizer,
    item: &Item,
    feature_on: CltFeatureId,
    feature_off: CltFeatureId,
    n_layers: usize,
    strength_grid: &[f32],
    device: &candle_core::Device,
) -> candle_mi::Result<ItemResult> {
    // Inject = the alternative's feature; suppress = the correct (committed) one.
    let inject_feature = if item.alternative == "on" {
        feature_on
    } else {
        feature_off
    };
    let suppress_feature = if item.correct == "on" {
        feature_on
    } else {
        feature_off
    };
    let inject_token_id = tokenizer.find_token_id(&item.alternative)?;

    // Tokenize once with offsets (adds BOS, matching `encode`); reuse ids + offsets.
    let enc = tokenizer.encode_with_offsets(&item.prompt)?;
    let token_ids = enc.ids;
    let seq_len = token_ids.len();
    let input = Tensor::new(&token_ids[..], device)?.unsqueeze(0)?;

    // Clause → token spans.
    let (ci, ce) = char_range(&item.prompt, &item.segments.initial)?;
    let (gi, ge) = char_range(&item.prompt, &item.segments.goal)?;
    let (si, se) = char_range(&item.prompt, &item.segments.stem)?;
    let initial_span = span_for(&enc.offsets, ci, ce);
    let goal_span = span_for(&enc.offsets, gi, ge);
    let stem_span = span_for(&enc.offsets, si, se);

    // Baseline P(alternative), no intervention.
    let baseline = extract_token_prob(
        model.forward(&input, &candle_mi::HookSpec::new())?.output(),
        inject_token_id,
    )?;

    let suppress_entries = downstream_entries(suppress_feature, n_layers);
    let inject_entries = downstream_entries(inject_feature, n_layers);

    let mut grid: Vec<SweepCell> = Vec::with_capacity(strength_grid.len() * seq_len);
    let mut best = SweepCell {
        strength: 0.0,
        position: 0,
        prob: -1.0,
    };
    for &strength in strength_grid {
        for position in 0..seq_len {
            let mut hooks = clt.prepare_hook_injection(
                &suppress_entries,
                position,
                seq_len,
                -strength,
                device,
            )?;
            let inject_hooks =
                clt.prepare_hook_injection(&inject_entries, position, seq_len, strength, device)?;
            hooks.extend(&inject_hooks);
            let prob =
                extract_token_prob(model.forward(&input, &hooks)?.output(), inject_token_id)?;
            grid.push(SweepCell {
                strength,
                position,
                prob,
            });
            if prob > best.prob {
                best = SweepCell {
                    strength,
                    position,
                    prob,
                };
            }
        }
    }

    let spike_ratio = if baseline > 0.0 {
        best.prob / baseline
    } else {
        0.0
    };
    let spike_clause = clause_of(best.position, initial_span, goal_span, stem_span).to_owned();
    let direction = format!("commit_{}", item.correct);

    Ok(ItemResult {
        id: item.id,
        order: item.order.clone(),
        device: item.device.clone(),
        direction,
        correct: item.correct.clone(),
        alternative: item.alternative.clone(),
        inject_feature,
        suppress_feature,
        baseline_prob: baseline,
        spike_strength: best.strength,
        spike_position: best.position,
        spike_prob: best.prob,
        spike_ratio,
        spike_clause,
        localized: spike_ratio >= LOCALIZED_RATIO,
        initial_span,
        goal_span,
        stem_span,
        grid,
    })
}

/// Summarize one `<order>/<direction>` group.
fn summarize(key: String, results: &[&ItemResult]) -> candle_mi::Result<GroupSummary> {
    let n = results.len();
    let mut clause_counts: BTreeMap<String, usize> = BTreeMap::new();
    for r in results {
        *clause_counts.entry(r.spike_clause.clone()).or_default() += 1;
    }
    let modal_clause = clause_counts
        .iter()
        .max_by_key(|(_, count)| **count)
        .map_or_else(|| "none".to_owned(), |(k, _)| k.clone());
    let (mean_ratio, localized_frac) = if n == 0 {
        (0.0, 0.0)
    } else {
        let n_f64 = count_to_f64(n)?;
        let sum_ratio: f64 = results.iter().map(|r| f64::from(r.spike_ratio)).sum();
        let localized = results.iter().filter(|r| r.localized).count();
        (sum_ratio / n_f64, count_to_f64(localized)? / n_f64)
    };
    Ok(GroupSummary {
        key,
        n,
        clause_counts,
        modal_clause,
        mean_spike_ratio: mean_ratio,
        localized_fraction: localized_frac,
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

    let feature_on = parse_feature(&args.feature_on)?;
    let feature_off = parse_feature(&args.feature_off)?;

    eprintln!("=== Means-Ends Prolepsis — Step B (suppress-plus-inject sweep) ===\n");
    eprintln!("Model:   {}", args.model);
    eprintln!("CLT:     {}", args.clt_repo);
    eprintln!("inject on  = {feature_on}");
    eprintln!("inject off = {feature_off}");
    eprintln!("strengths: {:?}\n", args.strength_grid);

    let items = load_items(&args.items)?;
    if items.is_empty() {
        return Err(candle_mi::MIError::Config(format!(
            "no items in {}",
            args.items.display()
        )));
    }
    let n_items = items.len();

    eprintln!("Loading model...");
    let model = MIModel::from_pretrained(&args.model)?;
    let n_layers = model.num_layers();
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!("  {n_layers} layers, device={device:?}");

    eprintln!("Opening CLT and caching steering vectors...");
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    clt.cache_steering_vectors_all_downstream(&[feature_on, feature_off], &device)?;

    eprintln!(
        "Sweeping {n_items} items ({} strengths)...\n",
        args.strength_grid.len()
    );
    let mut results: Vec<ItemResult> = Vec::with_capacity(n_items);
    for item in &items {
        let r = sweep_item(
            &model,
            &clt,
            tokenizer,
            item,
            feature_on,
            feature_off,
            n_layers,
            &args.strength_grid,
            &device,
        )?;
        eprintln!(
            "  [{:>2}] {:<13} {:<10} dev={:<9} baseline={:.4} spike={:.4} (x{:.1}) @pos {} [{}] s={}",
            r.id,
            r.order,
            r.direction,
            r.device,
            r.baseline_prob,
            r.spike_prob,
            r.spike_ratio,
            r.spike_position,
            r.spike_clause,
            r.spike_strength
        );
        results.push(r);
    }

    // Per-(order × direction) summary — the dissociation table.
    let mut keys: Vec<String> = results
        .iter()
        .map(|r| format!("{}/{}", r.order, r.direction))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let mut summaries: Vec<GroupSummary> = Vec::with_capacity(keys.len());
    for key in keys {
        let group: Vec<&ItemResult> = results
            .iter()
            .filter(|r| format!("{}/{}", r.order, r.direction) == key)
            .collect();
        summaries.push(summarize(key, &group)?);
    }

    eprintln!("\n=== Dissociation (spike clause by order × direction) ===");
    eprintln!(
        "  {:<28} {:>3} {:>8} {:>10}  clause histogram",
        "group", "n", "modalCl", "localized"
    );
    for s in &summaries {
        eprintln!(
            "  {:<28} {:>3} {:>8} {:>9.2}  {:?}",
            s.key, s.n, s.modal_clause, s.localized_fraction, s.clause_counts
        );
    }

    let output = Output {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        feature_on,
        feature_off,
        strength_grid: args.strength_grid.clone(),
        localized_ratio: LOCALIZED_RATIO,
        n_items,
        summaries,
        items: results,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    };
    write_json(&args.output, &output)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}
