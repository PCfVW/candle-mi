// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLT vs PLT planning-site comparison on Llama 3.2 1B — shared harness.
//!
//! Two modes, selected via `--schema`:
//!
//! - `--schema clt` — **Step A** (Jacopin replication). Loads
//!   `mntss/clt-llama-3.2-1b-524k`, runs the `-ee` rhyming-couplet
//!   `figure13_planning_poems.rs` protocol verbatim (suppress `-ee` group
//!   features + inject `"that"` at strength 10.0, 31-position sweep). Writes
//!   `docs/experiments/clt-vs-plt-planning-site/clt_step_a_llama.json`. Locks
//!   the paper result in the same harness that Step B reuses.
//!
//! - `--schema both` (default) — **Step B** (method-matched comparison).
//!   Runs BOTH the `CLT` and `PLT` transcoders side-by-side with two causal
//!   protocols each: (1) *suppress-only* zeroes out the top-5 features that
//!   point at `unembed("that")` at each sweep position (V3 Step 1.7 clean
//!   formulation); (2) *suppress+inject* mirrors Step A's protocol but with
//!   decoder-projection-derived features — suppress top-5 + inject the
//!   top-1 feature at strength 10.0.
//!   Four position sweeps total (2 arms × 2 protocols). Collects the full V3
//!   Step 1.7 instrumentation payload: top-20 decoder-projection rankings,
//!   top-20 decoder vectors, all-layer × all-position activation traces for
//!   each top-20 feature, 32-bin pre-activation histograms at the spike
//!   layer and its two neighbours, the PLT `W_skip · x` projection at the
//!   spike position (PLT-only), and both CLT decoder-slice metrics
//!   (same-layer and max-over-target-layers) in parallel. Writes
//!   `docs/experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json`.
//!
//! Device: CUDA-or-bust. Hard-fails if the device selector falls back to
//! CPU (keeps the `CLT`/`PLT` comparison on the same device family).
//!
//! Reference value: `P("that") = 0.687` on candle 0.9 with this CLT
//! (matches `figure13_planning_poems.rs` on the same build; see the
//! memory note `project_llama_planning_reference.md`). plip-rs §Q2 and
//! the Jacopin COLM 2026 submission report `0.777` — the gap is
//! runtime-stack drift, not a bug.
//!
//! Run:
//! ```bash
//! # Step B full comparison (default, ~3 min on CUDA):
//! cargo run --release --features clt,transformer,mmap --example clt_vs_plt_planning_site
//!
//! # Step A only (paper replication, ~25 s):
//! cargo run --release --features clt,transformer,mmap \
//!   --example clt_vs_plt_planning_site -- --schema clt
//! ```
//!
//! Plan: [`docs/roadmaps/PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`] Step B;
//! instrumentation spec: V3 Step 1.7.

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, IndexOp, Tensor};
use clap::Parser;
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookPoint, HookSpec, MIModel, extract_token_prob};

// ── Constants ───────────────────────────────────────────────────────────────

/// Reference `P("that")` at the spike on Llama 3.2 1B with
/// `mntss/clt-llama-3.2-1b-524k` in the candle-mi/candle 0.9 stack. Matches
/// `examples/figure13_planning_poems.rs` on the same build.
const LLAMA_REFERENCE_MAX_PROB: f32 = 0.687;

/// Tight tolerance for drift from the reference value.
const LLAMA_REFERENCE_TOL: f32 = 1e-2;

/// Hard-fail threshold for the Step A sanity gate: a spike weaker than this
/// indicates the harness is broken, not noise.
const LLAMA_SPIKE_HARD_MIN: f32 = 0.50;

// ── Llama preset ────────────────────────────────────────────────────────────

/// The `-ee` rhyming prompt. Final line primes a rhyme the transcoder has to
/// "plan" at the comma on line 3 — that is the planning site the sweep finds.
const PROMPT: &str = "The birds were singing in the tree,\n\
                      And everything was wild and free.\n\
                      The river ran down to the sea,\n\
                      There is so much we cannot";
const SUPPRESS_WORD: &str = "free";
const INJECT_WORD: &str = "that";
/// Jacopin-paper suppress features (used by Step A only).
const SUPPRESS_FEATURES: &[(usize, usize)] = &[
    (13, 30985), // "he"
    (9, 5488),   // "be"
    (14, 27874), // "ne"
    (13, 32049), // "we"
];
/// Jacopin-paper inject feature (used by Step A only).
const INJECT_FEATURE: (usize, usize) = (14, 13043); // "that"
const DEFAULT_STRENGTH: f32 = 10.0;
const DEFAULT_TOP_K: usize = 5;
/// Step B tracks the top-20 features per arm (a superset of the top-5 used
/// for suppression). V3 Step 1.7 (E) uses the top-20 for qualitative /
/// cross-layer-binding inspection without re-running forward passes.
const STEP_B_TOP_N: usize = 20;
/// Spike-layer neighbours to histogram (layer 14 is the known Jacopin spike
/// layer on Llama; 13 and 15 bracket it).
const HISTOGRAM_NEIGHBOUR_OFFSETS: &[i64] = &[-1, 0, 1];
/// 32-bin fixed-edge histogram is the V3 Step 1.7 (D) spec.
const HISTOGRAM_N_BINS: usize = 32;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "clt_vs_plt_planning_site")]
#[command(about = "CLT vs PLT planning-site comparison on Llama 3.2 1B")]
struct Args {
    /// Mode. `both` (default) runs Step B (full comparison, both arms,
    /// both protocols); `clt` runs Step A (Jacopin replication, CLT only).
    #[arg(long, default_value = "both")]
    schema: String,

    /// `HuggingFace` model ID.
    #[arg(long, default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// `HuggingFace` `CLT` repository.
    #[arg(long, default_value = "mntss/clt-llama-3.2-1b-524k")]
    clt_repo: String,

    /// `HuggingFace` `PLT` repository (used by Step B).
    #[arg(long, default_value = "mntss/transcoder-Llama-3.2-1B")]
    plt_repo: String,

    /// Steering strength for both the suppress and inject hooks.
    #[arg(long, default_value_t = DEFAULT_STRENGTH)]
    strength: f32,

    /// Number of top decoder-projection features used for Step A's
    /// auxiliary ranking. Step B always records
    #[doc = concat!("top-", stringify!(20), ".")]
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// Output JSON path override. Defaults per mode:
    /// `--schema clt` → `clt_step_a_llama.json`;
    /// `--schema both` → `clt_vs_plt_llama.json`.
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Step A output types ─────────────────────────────────────────────────────

#[derive(Serialize)]
struct StepAOutput {
    schema: String,
    model: String,
    transcoder_repo: String,
    prompt: String,
    tokens: Vec<String>,
    suppress_word: String,
    inject_word: String,
    inject_token_id: u32,
    suppress_features: Vec<CltFeatureId>,
    inject_feature: CltFeatureId,
    strength: f32,
    top_k_target_layer: usize,
    top_k_features: Vec<FeatureScore>,
    baseline_prob: f32,
    baseline_logit: f32,
    spike_position: usize,
    spike_token: String,
    max_prob: f32,
    max_logit: f32,
    delta_prob: f32,
    delta_logit: f32,
    reference_max_prob: f32,
    reference_tolerance: f32,
    sweep: Vec<PositionResult>,
}

// ── Step B output types ─────────────────────────────────────────────────────

#[derive(Serialize)]
struct StepBOutput {
    experiment: &'static str,
    step: &'static str,
    runtime_seconds: f64,
    device_name: String,
    model: String,
    prompt: String,
    tokens: Vec<String>,
    inject_word: String,
    inject_token_id: u32,
    top_k_target_layer: usize,
    baseline_prob: f32,
    baseline_logit: f32,
    arms: ArmOutputs,
    sanity_gates: SanityGates,
}

#[derive(Serialize)]
struct ArmOutputs {
    clt: ArmOutput,
    plt: ArmOutput,
}

#[derive(Serialize)]
struct ArmOutput {
    schema: String,
    transcoder_repo: String,
    n_layers: usize,
    n_features_per_layer: usize,
    top_20_features_same_layer: Vec<FeatureScore>,
    top_20_features_max_over_target: Option<Vec<FeatureScore>>,
    top_20_decoder_vectors_same_layer: Vec<Vec<f32>>,
    pre_activation_histograms: HashMap<String, Histogram>,
    all_layer_activation_trace: ActivationTrace,
    w_skip_projection_at_spike: Option<f32>,
    suppress_only: CausalTestResult,
    suppress_inject: CausalTestResult,
}

#[derive(Serialize)]
struct CausalTestResult {
    protocol: String,
    suppress_features: Vec<CltFeatureId>,
    inject_feature: Option<CltFeatureId>,
    strength: f32,
    spike_position: usize,
    spike_token: String,
    max_prob: f32,
    max_logit: f32,
    delta_prob: f32,
    delta_logit: f32,
    sweep: Vec<PositionResult>,
}

#[derive(Serialize)]
struct Histogram {
    bin_edges: Vec<f32>,
    counts: Vec<u64>,
}

#[derive(Serialize)]
struct ActivationTrace {
    feature_ids: Vec<CltFeatureId>,
    layer_indices: Vec<usize>,
    positions: Vec<usize>,
    /// Shape `[n_features][n_layers][seq_len]` — dense post-ReLU
    /// activations retrieved from `encode()`'s sparse output (absent
    /// features default to `0.0`).
    values: Vec<Vec<Vec<f32>>>,
}

#[derive(Serialize)]
struct SanityGates {
    clt_reference_max_prob: f32,
    clt_reference_tolerance: f32,
    clt_hard_min: f32,
    clt_suppress_inject_gate_triggered: bool,
    plt_reference_max_prob: Option<f32>,
}

// ── Shared output types ─────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct FeatureScore {
    feature: CltFeatureId,
    cosine: f32,
}

#[derive(Serialize, Clone)]
struct PositionResult {
    position: usize,
    token: String,
    prob: f32,
    logit: f32,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const fn feature_id(pair: (usize, usize)) -> CltFeatureId {
    CltFeatureId {
        layer: pair.0,
        index: pair.1,
    }
}

fn default_output_path(step: Step) -> PathBuf {
    let filename = match step {
        Step::A => "clt_step_a_llama.json",
        Step::B => "clt_vs_plt_llama.json",
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("experiments")
        .join("clt-vs-plt-planning-site")
        .join(filename)
}

#[derive(Copy, Clone)]
enum Step {
    A,
    B,
}

/// Extract the raw logit at `token_id` from the last sequence position.
fn extract_token_logit(logits: &Tensor, token_id: u32) -> candle_mi::Result<f32> {
    // PROMOTE: logits may arrive in BF16/F16; F32 for scalar extraction.
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let last_logits = match logits_f32.dims().len() {
        1 => logits_f32,
        2 => {
            let seq_len = logits_f32.dim(0)?;
            logits_f32.i(seq_len - 1)?
        }
        3 => {
            let seq_len = logits_f32.dim(1)?;
            logits_f32.i((0, seq_len - 1))?
        }
        n => {
            return Err(candle_mi::MIError::Config(format!(
                "extract_token_logit: expected 1-3 dims, got {n}"
            )));
        }
    };
    // CAST: u32 → usize, token ID used as 1-D tensor index
    #[allow(clippy::as_conversions)]
    let logit = last_logits.i(token_id as usize)?.to_scalar::<f32>()?;
    Ok(logit)
}

/// Fixed-edge histogram over `values`. Returns `HISTOGRAM_N_BINS + 1` bin
/// edges (min → max, inclusive) and `HISTOGRAM_N_BINS` counts. Values outside
/// `[min, max]` are clipped to the edge bins; NaN is dropped.
fn fixed_edge_histogram(values: &[f32], min: f32, max: f32) -> Histogram {
    // CAST: usize → f32 for bin-width arithmetic (HISTOGRAM_N_BINS = 32, exact)
    #[allow(clippy::as_conversions)]
    let n_bins_f = HISTOGRAM_N_BINS as f32;
    let width = (max - min) / n_bins_f;
    let mut bin_edges: Vec<f32> = Vec::with_capacity(HISTOGRAM_N_BINS + 1);
    for i in 0..=HISTOGRAM_N_BINS {
        // CAST: usize → f32, `i <= 32`, exact
        #[allow(clippy::as_conversions)]
        let i_f = i as f32;
        // Fused multiply-add: min + width * i_f, as clippy::suboptimal_flops suggests.
        bin_edges.push(width.mul_add(i_f, min));
    }
    let mut counts: Vec<u64> = vec![0; HISTOGRAM_N_BINS];
    for &v in values {
        if v.is_nan() {
            continue;
        }
        let mut idx = if width > 0.0 {
            ((v - min) / width).floor()
        } else {
            0.0
        };
        if idx < 0.0 {
            idx = 0.0;
        }
        // CAST: f32 → usize via clamp, bounded by HISTOGRAM_N_BINS - 1 below
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let mut bin = idx as usize;
        if bin >= HISTOGRAM_N_BINS {
            bin = HISTOGRAM_N_BINS - 1;
        }
        // INDEX: `bin` clamped to `< HISTOGRAM_N_BINS`; `counts` has exactly that length.
        if let Some(slot) = counts.get_mut(bin) {
            *slot += 1;
        }
    }
    Histogram { bin_edges, counts }
}

/// Describe the device in a form suitable for provenance logging.
fn describe_device(device: &candle_core::Device) -> String {
    match device {
        candle_core::Device::Cpu => "cpu".to_owned(),
        candle_core::Device::Cuda(_) => "cuda:0".to_owned(),
        candle_core::Device::Metal(_) => "metal".to_owned(),
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // BORROW: &str view of the CLI-parsed String for match discrimination
    match args.schema.as_str() {
        "clt" => run_step_a(&args),
        "both" => run_step_b(&args),
        other => Err(candle_mi::MIError::Config(format!(
            "--schema must be `clt` (Step A) or `both` (Step B), got `{other}`"
        ))),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step A — CLT-only Jacopin replication
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
fn run_step_a(args: &Args) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    eprintln!("=== Step A: CLT baseline on Llama 3.2 1B ===\n");
    eprintln!("Loading model: {}", args.model);
    let model = MIModel::from_pretrained(&args.model)?;
    let device = model.device().clone();
    if !device.is_cuda() {
        return Err(candle_mi::MIError::Config(
            "Step A requires CUDA (the CLT vs PLT comparison must run on the \
             same device family)."
                .into(),
        ));
    }
    let n_layers = model.num_layers();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!(
        "  {n_layers} layers, {} hidden, device={device:?}",
        model.hidden_size()
    );

    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;

    let suppress_features: Vec<CltFeatureId> =
        SUPPRESS_FEATURES.iter().copied().map(feature_id).collect();
    let inject_feature = feature_id(INJECT_FEATURE);
    // BORROW: clone() — owned Vec to pass separately from the expanded all-features list
    let mut all_features: Vec<CltFeatureId> = suppress_features.clone();
    all_features.push(inject_feature);

    let prompt_with_space = format!("{PROMPT} ");
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();
    let token_strs: Vec<String> = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();
    eprintln!("Tokens ({seq_len}): {token_strs:?}");

    let inject_token_id = tokenizer.find_token_id(INJECT_WORD)?;
    let inject_token_str = tokenizer.decode_token(inject_token_id)?;
    eprintln!("Inject token: \"{inject_token_str}\" (id={inject_token_id})");

    let top_k_target_layer = inject_feature.layer;
    eprintln!(
        "Ranking top-{} CLT features by cosine(decoder_row, unembed(\"{INJECT_WORD}\")) \
         at layer {top_k_target_layer}...",
        args.top_k
    );
    let direction = model.backend().embedding_vector(inject_token_id)?;
    let top_k_scores =
        clt.score_features_by_decoder_projection(&direction, top_k_target_layer, args.top_k, true)?;
    let top_k_features: Vec<FeatureScore> = top_k_scores
        .iter()
        .map(|(fid, cos)| FeatureScore {
            feature: *fid,
            cosine: *cos,
        })
        .collect();
    for (rank, fs) in top_k_features.iter().enumerate() {
        eprintln!(
            "  {:>2}. L{:<2}:{:<6}  cos={:+.4}",
            rank + 1,
            fs.feature.layer,
            fs.feature.index,
            fs.cosine
        );
    }

    eprintln!("Caching decoder vectors for all downstream layers...");
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    let suppress_entries: Vec<(CltFeatureId, usize)> = suppress_features
        .iter()
        .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
        .collect();
    let inject_entries: Vec<(CltFeatureId, usize)> = (inject_feature.layer..n_layers)
        .map(|l| (inject_feature, l))
        .collect();

    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let baseline_out = model.forward(&input, &HookSpec::new())?;
    let baseline_prob = extract_token_prob(baseline_out.output(), inject_token_id)?;
    let baseline_logit = extract_token_logit(baseline_out.output(), inject_token_id)?;
    eprintln!(
        "Baseline P(\"{inject_token_str}\") = {baseline_prob:.6e}  \
         logit = {baseline_logit:+.6}"
    );

    eprintln!(
        "\nSweeping {seq_len} positions (strength={})...",
        args.strength
    );
    let sweep = sweep_suppress_inject(
        &model,
        &clt,
        &input,
        seq_len,
        &token_strs,
        &suppress_entries,
        &inject_entries,
        args.strength,
        inject_token_id,
        baseline_prob,
        &device,
        /*verbose=*/ true,
    )?;

    let (spike_position, spike) = pick_spike(&sweep)?;

    if spike.prob < LLAMA_SPIKE_HARD_MIN {
        return Err(candle_mi::MIError::Config(format!(
            "CLT sanity failed: max P(\"{INJECT_WORD}\") = {:.4} < \
             hard-min {LLAMA_SPIKE_HARD_MIN:.2}. Expected ~{LLAMA_REFERENCE_MAX_PROB:.3} \
             (candle-mi reference; figure13_planning_poems.rs reproduces the same \
             number). Check model/CLT repo pinning.",
            spike.prob
        )));
    }
    let band_diff = (spike.prob - LLAMA_REFERENCE_MAX_PROB).abs();
    if band_diff > LLAMA_REFERENCE_TOL {
        eprintln!(
            "\nWARN: max_prob {:.4} drifted {band_diff:.4} from \
             reference {LLAMA_REFERENCE_MAX_PROB:.3} (tol ±{LLAMA_REFERENCE_TOL:.2}).",
            spike.prob
        );
    } else {
        eprintln!(
            "\nOK: max_prob {:.4} within ±{LLAMA_REFERENCE_TOL:.2} of \
             reference {LLAMA_REFERENCE_MAX_PROB:.3}.",
            spike.prob
        );
    }

    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(Step::A));
    let output = StepAOutput {
        schema: "clt".into(),
        model: args.model.clone(),
        transcoder_repo: args.clt_repo.clone(),
        prompt: PROMPT.into(),
        tokens: token_strs,
        suppress_word: SUPPRESS_WORD.into(),
        inject_word: INJECT_WORD.into(),
        inject_token_id,
        suppress_features,
        inject_feature,
        strength: args.strength,
        top_k_target_layer,
        top_k_features,
        baseline_prob,
        baseline_logit,
        spike_position,
        spike_token: spike.token.clone(),
        max_prob: spike.prob,
        max_logit: spike.logit,
        delta_prob: spike.prob - baseline_prob,
        delta_logit: spike.logit - baseline_logit,
        reference_max_prob: LLAMA_REFERENCE_MAX_PROB,
        reference_tolerance: LLAMA_REFERENCE_TOL,
        sweep,
    };
    write_json(&output, &output_path)?;

    eprintln!(
        "\nSpike: pos {spike_position} (\"{}\"), P={:.4}, logit={:+.4}",
        output.spike_token.replace('\n', "\\n"),
        output.max_prob,
        output.max_logit
    );
    eprintln!(
        "ΔP = {:+.4}, Δlogit = {:+.4}",
        output.delta_prob, output.delta_logit
    );
    eprintln!("Total elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Step B — CLT + PLT method-matched comparison
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
fn run_step_b(args: &Args) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    eprintln!("=== Step B: CLT vs PLT method-matched comparison ===\n");
    eprintln!("Loading model: {}", args.model);
    let model = MIModel::from_pretrained(&args.model)?;
    let device = model.device().clone();
    if !device.is_cuda() {
        return Err(candle_mi::MIError::Config(
            "Step B requires CUDA (the CLT vs PLT comparison must run on the \
             same device family)."
                .into(),
        ));
    }
    let n_layers = model.num_layers();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!(
        "  {n_layers} layers, {} hidden, device={device:?}",
        model.hidden_size()
    );

    // --- Shared preamble: tokenize, direction, baseline ---
    let prompt_with_space = format!("{PROMPT} ");
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();
    let token_strs: Vec<String> = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();
    eprintln!("Tokens ({seq_len}): {token_strs:?}");

    let inject_token_id = tokenizer.find_token_id(INJECT_WORD)?;
    let inject_token_str = tokenizer.decode_token(inject_token_id)?;
    eprintln!("Inject token: \"{inject_token_str}\" (id={inject_token_id})");

    let direction = model.backend().embedding_vector(inject_token_id)?;

    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

    // --- One baseline forward with ResidMid captures at every layer ---
    eprintln!("\nCapturing baseline residuals at every layer...");
    let mut capture_spec = HookSpec::new();
    for layer in 0..n_layers {
        capture_spec.capture(HookPoint::ResidMid(layer));
    }
    let baseline_cache = model.forward(&input, &capture_spec)?;
    let baseline_prob = extract_token_prob(baseline_cache.output(), inject_token_id)?;
    let baseline_logit = extract_token_logit(baseline_cache.output(), inject_token_id)?;
    eprintln!(
        "Baseline P(\"{inject_token_str}\") = {baseline_prob:.6e}  \
         logit = {baseline_logit:+.6}"
    );
    // BORROW: clone() — detach per-layer residual tensors from the HookCache
    // so we can drop the cache once the two arms start running.
    let mut residuals: Vec<Tensor> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let t = baseline_cache
            .get(&HookPoint::ResidMid(layer))
            .ok_or_else(|| {
                candle_mi::MIError::Config(format!(
                    "HookCache missing ResidMid({layer}) — capture_spec wiring bug"
                ))
            })?
            .clone();
        residuals.push(t);
    }
    drop(baseline_cache);

    let top_k_target_layer: usize = 14;
    let suppress_inject_strength = args.strength;

    // --- CLT arm ---
    eprintln!("\n── CLT arm (mntss/clt-llama-3.2-1b-524k) ──");
    let clt_arm = run_arm_step_b(
        "clt",
        &args.clt_repo,
        &model,
        &residuals,
        &input,
        seq_len,
        &token_strs,
        inject_token_id,
        &direction,
        baseline_prob,
        baseline_logit,
        suppress_inject_strength,
        top_k_target_layer,
        n_layers,
        /*include_max_over_target=*/ true,
        &device,
    )?;

    // --- PLT arm ---
    eprintln!("\n── PLT arm (mntss/transcoder-Llama-3.2-1B) ──");
    let plt_arm = run_arm_step_b(
        "plt",
        &args.plt_repo,
        &model,
        &residuals,
        &input,
        seq_len,
        &token_strs,
        inject_token_id,
        &direction,
        baseline_prob,
        baseline_logit,
        suppress_inject_strength,
        top_k_target_layer,
        n_layers,
        /*include_max_over_target=*/ false,
        &device,
    )?;

    // NOTE: Step B does NOT reuse Step A's LLAMA_REFERENCE_MAX_PROB=0.687 gate.
    // That reference was for Step A's Jacopin protocol (cross-layer -ee suppress
    // features). Step B uses decoder-projection-derived top-5 suppress features
    // at the target layer, producing a numerically different signal even with
    // the same transcoder. The gate field in the output JSON records this
    // intentionally (clt_suppress_inject_gate_triggered=false, no reference).
    let clt_gate_triggered = false;

    let runtime_seconds = t_start.elapsed().as_secs_f64();
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(Step::B));
    let output = StepBOutput {
        experiment: "clt-vs-plt-planning-site",
        step: "b",
        runtime_seconds,
        device_name: describe_device(&device),
        model: args.model.clone(),
        prompt: PROMPT.into(),
        tokens: token_strs,
        inject_word: INJECT_WORD.into(),
        inject_token_id,
        top_k_target_layer,
        baseline_prob,
        baseline_logit,
        arms: ArmOutputs {
            clt: clt_arm,
            plt: plt_arm,
        },
        sanity_gates: SanityGates {
            clt_reference_max_prob: LLAMA_REFERENCE_MAX_PROB,
            clt_reference_tolerance: LLAMA_REFERENCE_TOL,
            clt_hard_min: LLAMA_SPIKE_HARD_MIN,
            clt_suppress_inject_gate_triggered: clt_gate_triggered,
            plt_reference_max_prob: None,
        },
    };
    write_json(&output, &output_path)?;

    eprintln!("\n=== Step B complete ===");
    eprintln!(
        "  CLT suppress-only    ΔP = {:+.6e}  Δlogit = {:+.4}  spike pos {}",
        output.arms.clt.suppress_only.delta_prob,
        output.arms.clt.suppress_only.delta_logit,
        output.arms.clt.suppress_only.spike_position
    );
    eprintln!(
        "  CLT suppress+inject  ΔP = {:+.6e}  Δlogit = {:+.4}  spike pos {}",
        output.arms.clt.suppress_inject.delta_prob,
        output.arms.clt.suppress_inject.delta_logit,
        output.arms.clt.suppress_inject.spike_position
    );
    eprintln!(
        "  PLT suppress-only    ΔP = {:+.6e}  Δlogit = {:+.4}  spike pos {}",
        output.arms.plt.suppress_only.delta_prob,
        output.arms.plt.suppress_only.delta_logit,
        output.arms.plt.suppress_only.spike_position
    );
    eprintln!(
        "  PLT suppress+inject  ΔP = {:+.6e}  Δlogit = {:+.4}  spike pos {}",
        output.arms.plt.suppress_inject.delta_prob,
        output.arms.plt.suppress_inject.delta_logit,
        output.arms.plt.suppress_inject.spike_position
    );
    eprintln!("Total elapsed: {runtime_seconds:.2?}s");
    Ok(())
}

/// Run the per-arm Step B pipeline. Consumes one transcoder, produces one
/// `ArmOutput`. The model residuals are shared between arms.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_arm_step_b(
    arm_label: &str,
    repo: &str,
    model: &MIModel,
    residuals: &[Tensor],
    input: &Tensor,
    seq_len: usize,
    token_strs: &[String],
    inject_token_id: u32,
    direction: &Tensor,
    baseline_prob: f32,
    baseline_logit: f32,
    strength: f32,
    top_k_target_layer: usize,
    n_layers: usize,
    include_max_over_target: bool,
    device: &candle_core::Device,
) -> candle_mi::Result<ArmOutput> {
    let t_arm = std::time::Instant::now();

    eprintln!("Opening transcoder: {repo}");
    let mut transcoder = CrossLayerTranscoder::open(repo)?;
    let n_features_per_layer = transcoder.config().n_features_per_layer;

    // --- Top-20 same-layer ranking ---
    eprintln!(
        "  Ranking top-{STEP_B_TOP_N} features by cosine(decoder_row, unembed(\"{INJECT_WORD}\")) \
         at layer {top_k_target_layer}..."
    );
    let top_20_scores = transcoder.score_features_by_decoder_projection(
        direction,
        top_k_target_layer,
        STEP_B_TOP_N,
        true,
    )?;
    let top_20_features_same_layer: Vec<FeatureScore> = top_20_scores
        .iter()
        .map(|(fid, cos)| FeatureScore {
            feature: *fid,
            cosine: *cos,
        })
        .collect();
    for (rank, fs) in top_20_features_same_layer.iter().take(5).enumerate() {
        eprintln!(
            "    {:>2}. L{:<2}:{:<6}  cos={:+.4}",
            rank + 1,
            fs.feature.layer,
            fs.feature.index,
            fs.cosine
        );
    }

    // --- Top-20 max-over-target-layers (CLT only) ---
    let top_20_features_max_over_target = if include_max_over_target {
        eprintln!(
            "  Ranking top-{STEP_B_TOP_N} features by max cosine across target layers \
             0..{n_layers} (CLT slice-ambiguity control)..."
        );
        Some(rank_top_k_max_over_target(
            &mut transcoder,
            direction,
            n_layers,
            STEP_B_TOP_N,
        )?)
    } else {
        None
    };

    // --- Top-20 decoder vectors (same-layer) ---
    eprintln!("  Extracting top-{STEP_B_TOP_N} decoder vectors at layer {top_k_target_layer}...");
    let top_20_fids: Vec<CltFeatureId> = top_20_features_same_layer
        .iter()
        .map(|fs| fs.feature)
        .collect();
    let dec_map = transcoder.extract_decoder_vectors(&top_20_fids, top_k_target_layer)?;
    let mut top_20_decoder_vectors_same_layer: Vec<Vec<f32>> = Vec::with_capacity(STEP_B_TOP_N);
    for fid in &top_20_fids {
        let tensor = dec_map.get(fid).ok_or_else(|| {
            candle_mi::MIError::Config(format!("extract_decoder_vectors missed feature {fid:?}"))
        })?;
        // BORROW: .to_vec1()? moves tensor values from CPU to an owned Vec<f32>
        top_20_decoder_vectors_same_layer.push(tensor.to_vec1::<f32>()?);
    }

    // --- All-layer activation trace for top-20 features ---
    eprintln!("  Building {STEP_B_TOP_N}×{n_layers}×{seq_len} activation trace...");
    let all_layer_activation_trace = build_activation_trace(
        &mut transcoder,
        residuals,
        &top_20_fids,
        n_layers,
        seq_len,
        device,
    )?;

    // --- Pre-activation histograms at spike layer ± 1 ---
    eprintln!("  Computing pre-activation histograms at L14 ± 1...");
    let pre_activation_histograms = build_pre_activation_histograms(
        &mut transcoder,
        residuals,
        top_k_target_layer,
        seq_len,
        n_layers,
        device,
    )?;

    // --- PLT W_skip projection (arm_label == "plt" only) ---
    let w_skip_projection_at_spike = if arm_label == "plt" {
        eprintln!(
            "  Computing W_skip · residual[spike] projection onto unembed(\"{INJECT_WORD}\")..."
        );
        Some(compute_w_skip_projection(
            &mut transcoder,
            residuals,
            direction,
            top_k_target_layer,
            /*spike_position=*/ seq_len - 1,
            device,
        )?)
    } else {
        None
    };

    // --- Suppress and inject feature selection (top-5 + top-1 from same-layer ranking) ---
    let suppress_features: Vec<CltFeatureId> = top_20_features_same_layer
        .iter()
        .take(DEFAULT_TOP_K)
        .map(|fs| fs.feature)
        .collect();
    let inject_feature = top_20_features_same_layer
        .first()
        .map(|fs| fs.feature)
        .ok_or_else(|| candle_mi::MIError::Config("top-20 ranking returned 0 features".into()))?;

    // Cache decoder vectors for both protocols (suppress + inject across all downstream).
    let mut all_cache_features: Vec<CltFeatureId> = suppress_features.clone();
    all_cache_features.push(inject_feature);
    eprintln!("  Caching decoder vectors for suppress+inject features...");
    transcoder.cache_steering_vectors_all_downstream(&all_cache_features, device)?;

    // Per-layer schemas (PltBundle) cache a single entry per feature (at its
    // own source layer); cross-layer schemas (CltSplit) cache one entry per
    // downstream target. Intervention entries must match the cache structure.
    let is_cross_layer = transcoder.config().schema.is_cross_layer();
    let suppress_entries: Vec<(CltFeatureId, usize)> = if is_cross_layer {
        suppress_features
            .iter()
            .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
            .collect()
    } else {
        suppress_features
            .iter()
            .map(|feat| (*feat, feat.layer))
            .collect()
    };
    let inject_entries: Vec<(CltFeatureId, usize)> = if is_cross_layer {
        (inject_feature.layer..n_layers)
            .map(|l| (inject_feature, l))
            .collect()
    } else {
        vec![(inject_feature, inject_feature.layer)]
    };

    // --- Causal test 1: suppress-only ---
    eprintln!("  Sweeping {seq_len} positions (suppress-only, strength={strength})...");
    let sweep_only_positions = sweep_suppress_only(
        model,
        &transcoder,
        input,
        seq_len,
        token_strs,
        &suppress_entries,
        strength,
        inject_token_id,
        device,
    )?;
    let (spike_only_pos, spike_only) = pick_spike(&sweep_only_positions)?;
    let suppress_only = CausalTestResult {
        protocol: "suppress_only".into(),
        suppress_features: suppress_features.clone(),
        inject_feature: None,
        strength,
        spike_position: spike_only_pos,
        spike_token: spike_only.token.clone(),
        max_prob: spike_only.prob,
        max_logit: spike_only.logit,
        delta_prob: spike_only.prob - baseline_prob,
        delta_logit: spike_only.logit - baseline_logit,
        sweep: sweep_only_positions,
    };

    // --- Causal test 2: suppress + inject ---
    eprintln!("  Sweeping {seq_len} positions (suppress+inject, strength={strength})...");
    let sweep_inject_positions = sweep_suppress_inject(
        model,
        &transcoder,
        input,
        seq_len,
        token_strs,
        &suppress_entries,
        &inject_entries,
        strength,
        inject_token_id,
        baseline_prob,
        device,
        /*verbose=*/ false,
    )?;
    let (spike_inject_pos, spike_inject) = pick_spike(&sweep_inject_positions)?;
    let suppress_inject = CausalTestResult {
        protocol: "suppress_inject".into(),
        suppress_features,
        inject_feature: Some(inject_feature),
        strength,
        spike_position: spike_inject_pos,
        spike_token: spike_inject.token.clone(),
        max_prob: spike_inject.prob,
        max_logit: spike_inject.logit,
        delta_prob: spike_inject.prob - baseline_prob,
        delta_logit: spike_inject.logit - baseline_logit,
        sweep: sweep_inject_positions,
    };

    eprintln!(
        "  Arm elapsed: {:.2?}  (suppress-only ΔP={:+.6e}, suppress+inject ΔP={:+.6e})",
        t_arm.elapsed(),
        suppress_only.delta_prob,
        suppress_inject.delta_prob
    );

    Ok(ArmOutput {
        schema: arm_label.into(),
        transcoder_repo: repo.into(),
        n_layers,
        n_features_per_layer,
        top_20_features_same_layer,
        top_20_features_max_over_target,
        top_20_decoder_vectors_same_layer,
        pre_activation_histograms,
        all_layer_activation_trace,
        w_skip_projection_at_spike,
        suppress_only,
        suppress_inject,
    })
}

/// Rank the top-`k` features by max cosine across all downstream target
/// layers. Loop pattern borrowed from `examples/clt_probe.rs:476-500`.
fn rank_top_k_max_over_target(
    transcoder: &mut CrossLayerTranscoder,
    direction: &Tensor,
    n_layers: usize,
    k: usize,
) -> candle_mi::Result<Vec<FeatureScore>> {
    let mut all_hits: Vec<(CltFeatureId, f32)> = Vec::new();
    for target_layer in 0..n_layers {
        // Ask for up to `k` per target layer (more than enough for the global top-k).
        let hits =
            transcoder.score_features_by_decoder_projection(direction, target_layer, k, true)?;
        for (fid, cosine) in hits {
            if let Some(existing) = all_hits.iter_mut().find(|(f, _)| *f == fid) {
                if cosine > existing.1 {
                    existing.1 = cosine;
                }
            } else {
                all_hits.push((fid, cosine));
            }
        }
    }
    all_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    all_hits.truncate(k);
    Ok(all_hits
        .into_iter()
        .map(|(feature, cosine)| FeatureScore { feature, cosine })
        .collect())
}

/// For each of `features`, record the post-`ReLU` activation at every
/// `(layer, position)`. Uses `encode` (sparse) and materialises 0.0 for
/// features absent from the sparse output.
//
// The triple nested `Vec<Vec<Vec<f32>>>` is pre-allocated at a known
// `[n_features][n_layers][seq_len]` shape and every `slot`, `layer`, `pos`
// write uses indices produced from the same-length iteration counters. The
// three `indexing_slicing` hits on `values[slot][layer][pos] = *act` are
// noise at this callsite, and `needless_range_loop` on the two outer loops
// is unavoidable because the iteration counter is also used to drive the
// transcoder's stateful `load_encoder(layer)` side effect (not just to index
// `values`). Suppress both intentionally.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn build_activation_trace(
    transcoder: &mut CrossLayerTranscoder,
    residuals: &[Tensor],
    features: &[CltFeatureId],
    n_layers: usize,
    seq_len: usize,
    device: &candle_core::Device,
) -> candle_mi::Result<ActivationTrace> {
    let n_features = features.len();
    // Pre-allocate [feature][layer][position] = 0.0.
    let mut values: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0_f32; seq_len]; n_layers]; n_features];

    let feature_to_slot: HashMap<CltFeatureId, usize> = features
        .iter()
        .enumerate()
        .map(|(i, fid)| (*fid, i))
        .collect();

    for layer in 0..n_layers {
        transcoder.load_encoder(layer, device)?;
        let layer_res = residuals.get(layer).ok_or_else(|| {
            candle_mi::MIError::Config(format!("residuals missing layer {layer}"))
        })?;
        for pos in 0..seq_len {
            let residual = layer_res.i((0, pos))?;
            let sparse = transcoder.encode(&residual, layer)?;
            for (fid, act) in &sparse.features {
                if let Some(&slot) = feature_to_slot.get(fid) {
                    // INDEX: slot < n_features; layer < n_layers; pos < seq_len — all guarded above.
                    values[slot][layer][pos] = *act;
                }
            }
        }
    }

    Ok(ActivationTrace {
        feature_ids: features.to_vec(),
        layer_indices: (0..n_layers).collect(),
        positions: (0..seq_len).collect(),
        values,
    })
}

/// Compute `HISTOGRAM_N_BINS`-bin pre-activation histograms at the spike
/// layer and its two neighbours. Flattens all `(position, feature)`
/// pre-activation values per layer into one histogram.
fn build_pre_activation_histograms(
    transcoder: &mut CrossLayerTranscoder,
    residuals: &[Tensor],
    spike_layer: usize,
    seq_len: usize,
    n_layers: usize,
    device: &candle_core::Device,
) -> candle_mi::Result<HashMap<String, Histogram>> {
    let mut histograms: HashMap<String, Histogram> = HashMap::new();

    for &offset in HISTOGRAM_NEIGHBOUR_OFFSETS {
        // CAST: i64 → isize for signed arithmetic; HISTOGRAM_NEIGHBOUR_OFFSETS is {-1, 0, 1}, trivially in range
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let offset_isize = offset as isize;
        // CAST: usize → isize for signed arithmetic; spike_layer is a layer index, fits easily
        #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
        let target_signed = spike_layer as isize + offset_isize;
        if target_signed < 0 {
            continue;
        }
        // CAST: isize → usize after positivity check above
        #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
        let target_layer = target_signed as usize;
        if target_layer >= n_layers {
            continue;
        }

        transcoder.load_encoder(target_layer, device)?;
        let layer_res = residuals.get(target_layer).ok_or_else(|| {
            candle_mi::MIError::Config(format!("residuals missing layer {target_layer}"))
        })?;

        // Collect flattened pre-activations across all positions for this layer.
        let mut flat: Vec<f32> = Vec::new();
        for pos in 0..seq_len {
            let residual = layer_res.i((0, pos))?;
            let pre = transcoder.encode_pre_activation(&residual, target_layer)?;
            let mut v: Vec<f32> = pre.to_vec1()?;
            flat.append(&mut v);
        }

        // Min/max bin edges so every value lands inside.
        let (mn, mx) = flat
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            });
        let (mn, mx) = if mn.is_finite() && mx.is_finite() && mx > mn {
            (mn, mx)
        } else {
            (0.0, 1.0)
        };

        histograms.insert(
            format!("layer_{target_layer}"),
            fixed_edge_histogram(&flat, mn, mx),
        );
    }

    Ok(histograms)
}

/// Compute `(W_skip @ residual_at_spike) · direction` for the PLT arm.
fn compute_w_skip_projection(
    transcoder: &mut CrossLayerTranscoder,
    residuals: &[Tensor],
    direction: &Tensor,
    spike_layer: usize,
    spike_position: usize,
    device: &candle_core::Device,
) -> candle_mi::Result<f32> {
    let w_skip = transcoder.load_skip_matrix(spike_layer, device)?;
    let layer_res = residuals.get(spike_layer).ok_or_else(|| {
        candle_mi::MIError::Config(format!("residuals missing layer {spike_layer}"))
    })?;
    let residual = layer_res.i((0, spike_position))?;
    // PROMOTE: residual may arrive BF16/F16; F32 to match W_skip's F32.
    let residual_f32 = residual.to_dtype(DType::F32)?;
    // W_skip @ residual → [d_model]
    let skip_vec = w_skip.matmul(&residual_f32.unsqueeze(1)?)?.squeeze(1)?;
    // direction is on the model device, skip_vec on `device` (same). Project.
    // PROMOTE: direction may be F16/BF16 from embedding_vector.
    let direction_f32 = direction.to_dtype(DType::F32)?;
    let dot = (&skip_vec * &direction_f32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok(dot)
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared sweep helpers
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn sweep_suppress_inject(
    model: &MIModel,
    transcoder: &CrossLayerTranscoder,
    input: &Tensor,
    seq_len: usize,
    token_strs: &[String],
    suppress_entries: &[(CltFeatureId, usize)],
    inject_entries: &[(CltFeatureId, usize)],
    strength: f32,
    inject_token_id: u32,
    baseline_prob: f32,
    device: &candle_core::Device,
    verbose: bool,
) -> candle_mi::Result<Vec<PositionResult>> {
    let mut sweep: Vec<PositionResult> = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let mut combined =
            transcoder.prepare_hook_injection(suppress_entries, pos, seq_len, -strength, device)?;
        let inject_hooks =
            transcoder.prepare_hook_injection(inject_entries, pos, seq_len, strength, device)?;
        combined.extend(&inject_hooks);
        let result = model.forward(input, &combined)?;
        let prob = extract_token_prob(result.output(), inject_token_id)?;
        let logit = extract_token_logit(result.output(), inject_token_id)?;
        // BORROW: String::clone — owned token string for the sweep entry.
        let token = token_strs.get(pos).map_or_else(String::new, String::clone);
        if verbose {
            let display = token.replace('\n', "\\n");
            eprintln!(
                "    pos {pos:>3}  {display:<20}  P={prob:.6e}  logit={logit:+.4}  \
                 ΔP={:+.6e}",
                prob - baseline_prob
            );
        }
        sweep.push(PositionResult {
            position: pos,
            token,
            prob,
            logit,
        });
    }
    Ok(sweep)
}

#[allow(clippy::too_many_arguments)]
fn sweep_suppress_only(
    model: &MIModel,
    transcoder: &CrossLayerTranscoder,
    input: &Tensor,
    seq_len: usize,
    token_strs: &[String],
    suppress_entries: &[(CltFeatureId, usize)],
    strength: f32,
    inject_token_id: u32,
    device: &candle_core::Device,
) -> candle_mi::Result<Vec<PositionResult>> {
    let mut sweep: Vec<PositionResult> = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let hooks =
            transcoder.prepare_hook_injection(suppress_entries, pos, seq_len, -strength, device)?;
        let result = model.forward(input, &hooks)?;
        let prob = extract_token_prob(result.output(), inject_token_id)?;
        let logit = extract_token_logit(result.output(), inject_token_id)?;
        // BORROW: String::clone — owned token string for the sweep entry.
        let token = token_strs.get(pos).map_or_else(String::new, String::clone);
        sweep.push(PositionResult {
            position: pos,
            token,
            prob,
            logit,
        });
    }
    Ok(sweep)
}

/// Return `(index_of_max, &entry)` over a non-empty sweep. Uses `.get()`
/// on the index for the `indexing_slicing` lint; the Err arm is unreachable
/// because `spike_idx` was just produced by `.max_by()` on the same slice.
fn pick_spike(sweep: &[PositionResult]) -> candle_mi::Result<(usize, &PositionResult)> {
    let spike_idx = sweep
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.prob
                .partial_cmp(&b.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .ok_or_else(|| candle_mi::MIError::Config("empty sweep".into()))?;
    // INDEX: spike_idx produced by max_by over the same slice; Err arm unreachable.
    let spike = sweep
        .get(spike_idx)
        .ok_or_else(|| candle_mi::MIError::Config("spike index out of range".into()))?;
    Ok((spike_idx, spike))
}

// ═══════════════════════════════════════════════════════════════════════════
// Output
// ═══════════════════════════════════════════════════════════════════════════

fn write_json<T: Serialize>(output: &T, path: &Path) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialize: {e}")))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| candle_mi::MIError::Config(format!("create {}: {e}", parent.display())))?;
    }
    fs::write(path, &json).map_err(|e| candle_mi::MIError::Config(format!("write output: {e}")))?;
    eprintln!("Output written to {}", path.display());
    Ok(())
}
