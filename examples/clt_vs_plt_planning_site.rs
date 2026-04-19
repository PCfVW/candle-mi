// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLT vs PLT planning-site comparison on Llama 3.2 1B — shared harness.
//!
//! **Step A** (this commit): reproduce the `CLT` baseline
//! (`mntss/clt-llama-3.2-1b-524k`) in the exact same harness, device, and code
//! path that the `PLT` arm will run through in Step B. Locking the `CLT` number
//! in this harness — rather than citing it from
//! [`examples/figure13_planning_poems.rs`](figure13_planning_poems.rs) — is
//! what makes the subsequent `CLT` vs `PLT` comparison apples-to-apples.
//!
//! **Step B** (next): wire `mntss/transcoder-Llama-3.2-1B` through the same
//! pipeline via `TranscoderSchema::PltBundle` and capture the full V3 Step 1.7
//! instrumentation payload. Until then, `--schema plt` exits with a clear
//! pointer.
//!
//! **Plan:** `docs/roadmaps/PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`.
//!
//! Protocol (identical to `figure13_planning_poems.rs` Llama preset):
//! - Prompt: rhyming couplets ending with `-ee`; final line "`There is so much
//!   we cannot`".
//! - Suppress `-ee` group features `{(13,30985), (9,5488), (14,27874),
//!   (13,32049)}`.
//! - Inject `(14,13043)` which decodes to `"that"`.
//! - Strength `10.0`, position sweep `0..seq_len`.
//!
//! Auxiliary instrumentation (new in Step A, carried forward to Step B):
//! - Top-`k` features ranked by cosine decoder projection onto
//!   `unembed("that")` at the inject layer. Not used for suppression — it
//!   records what the transcoder *thinks* points at `"that"` so Step B can
//!   diff the `CLT` and `PLT` rankings directly.
//! - Raw logit of `"that"` recorded alongside probability, so Δlogit is
//!   available for later decomposition.
//!
//! Device: CUDA-or-bust. The `MIModel::from_pretrained` path silently falls
//! back to CPU; this example hard-errors if CUDA is unavailable to keep the
//! `CLT`/`PLT` comparison on the same device family.
//!
//! Run:
//! ```bash
//! cargo run --release --features clt,transformer,mmap \
//!   --example clt_vs_plt_planning_site -- --schema clt
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, IndexOp, Tensor};
use clap::Parser;
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookSpec, MIModel, extract_token_prob};

// ── Constants ───────────────────────────────────────────────────────────────

/// Reference `P("that")` at the spike on Llama 3.2 1B with
/// `mntss/clt-llama-3.2-1b-524k` in the candle-mi/candle 0.9 stack. This
/// matches `examples/figure13_planning_poems.rs`'s observed max on the same
/// build — cross-check whenever either number drifts.
///
/// Historical note: plip-rs §Q2 / Jacopin COLM 2026 submission reports
/// `P("that") = 0.777` on a different runtime stack. The ~0.09 gap is stack
/// drift (candle 0.9 vs the original), not a bug; it reproduces identically
/// between both candle-mi examples.
const LLAMA_REFERENCE_MAX_PROB: f32 = 0.687;

/// Tight tolerance for drift from the reference value. Both candle-mi
/// examples on the same build produce the same number to better than `1e-3`;
/// anything larger indicates a regression worth investigating.
const LLAMA_REFERENCE_TOL: f32 = 1e-2;

/// Hard-fail threshold: a spike this weak means the harness is broken, not
/// that the spike drifted within noise. `0.50` is well above the `~0.1`
/// ambient we would see from an un-steered forward.
const LLAMA_SPIKE_HARD_MIN: f32 = 0.50;

// ── Llama preset (mirrors `figure13_planning_poems::LLAMA`) ─────────────────

/// The `-ee` rhyming prompt. Final line primes a rhyme the transcoder has to
/// "plan" at the comma on line 3 — that is the planning site the sweep finds.
const PROMPT: &str = "The birds were singing in the tree,\n\
                      And everything was wild and free.\n\
                      The river ran down to the sea,\n\
                      There is so much we cannot";
const SUPPRESS_WORD: &str = "free";
const INJECT_WORD: &str = "that";
const SUPPRESS_FEATURES: &[(usize, usize)] = &[
    (13, 30985), // "he"
    (9, 5488),   // "be"
    (14, 27874), // "ne"
    (13, 32049), // "we"
];
const INJECT_FEATURE: (usize, usize) = (14, 13043); // "that"
const DEFAULT_STRENGTH: f32 = 10.0;
const DEFAULT_TOP_K: usize = 5;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "clt_vs_plt_planning_site")]
#[command(about = "CLT vs PLT planning-site comparison on Llama 3.2 1B")]
struct Args {
    /// Transcoder family to run. `clt` (default) runs Step A; `plt` is
    /// reserved for Step B (currently exits with a pointer to the plan).
    #[arg(long, default_value = "clt")]
    schema: String,

    /// `HuggingFace` model ID.
    #[arg(long, default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// `HuggingFace` `CLT` repository (used when `--schema clt`).
    #[arg(long, default_value = "mntss/clt-llama-3.2-1b-524k")]
    clt_repo: String,

    /// `HuggingFace` `PLT` repository (used when `--schema plt`, Step B).
    #[arg(long, default_value = "mntss/transcoder-Llama-3.2-1B")]
    plt_repo: String,

    /// Steering strength for both the suppress and inject hooks.
    #[arg(long, default_value_t = DEFAULT_STRENGTH)]
    strength: f32,

    /// Top-`k` features to record from the decoder-projection ranking at the
    /// inject layer.
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// Output JSON path. Defaults to
    /// `docs/experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json`.
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Output types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ComparisonOutput {
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
    /// Layer at which decoder-projection top-`k` features were ranked.
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

#[derive(Serialize)]
struct FeatureScore {
    feature: CltFeatureId,
    cosine: f32,
}

#[derive(Serialize)]
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

fn default_output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("experiments")
        .join("clt-vs-plt-planning-site")
        .join("clt_vs_plt_llama.json")
}

/// Extract the raw logit at `token_id` from the last sequence position.
///
/// Mirrors `extract_token_prob`'s 1/2/3-dim support so callers can pass the
/// raw `HookCache::output()` tensor without reshaping.
fn extract_token_logit(logits: &Tensor, token_id: u32) -> candle_mi::Result<f32> {
    // PROMOTE: logits may arrive in BF16/F16; F32 for scalar extraction
    // and to match `extract_token_prob`'s internal dtype.
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
        "clt" => run_clt_arm(&args),
        "plt" => Err(candle_mi::MIError::Config(
            "--schema plt is deferred to Step B of \
             docs/roadmaps/PLAN-PLT-LLAMA-PLANNING-SIGNAL.md — \
             PLT wiring lands after the CLT baseline is locked."
                .into(),
        )),
        other => Err(candle_mi::MIError::Config(format!(
            "--schema must be `clt` or `plt`, got `{other}`"
        ))),
    }
}

#[allow(clippy::too_many_lines)]
// Step A is a flat "load → score → cache → tokenize → sweep → classify → write"
// pipeline mirroring `figure13_planning_poems.rs`. Extracting helpers would
// fragment the linear experiment story; the pedantic length lint is
// suppressed intentionally. Step B will split this into per-schema arms.
fn run_clt_arm(args: &Args) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    eprintln!("=== Step A: CLT baseline on Llama 3.2 1B ===\n");
    eprintln!("Loading model: {}", args.model);
    let model = MIModel::from_pretrained(&args.model)?;
    let device = model.device().clone();
    if !device.is_cuda() {
        return Err(candle_mi::MIError::Config(
            "Step A requires CUDA (the CLT vs PLT comparison must run on the \
             same device family). candle-mi's device selector fell back to CPU \
             — check your build features and driver."
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

    // --- Open CLT ---
    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;

    // --- Resolve feature IDs ---
    let suppress_features: Vec<CltFeatureId> =
        SUPPRESS_FEATURES.iter().copied().map(feature_id).collect();
    let inject_feature = feature_id(INJECT_FEATURE);
    // BORROW: clone() — owned Vec to pass separately from the expanded all-features list
    let mut all_features: Vec<CltFeatureId> = suppress_features.clone();
    all_features.push(inject_feature);

    // --- Tokenize ---
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

    // --- Decoder-projection top-k at the inject layer ---
    //
    // Auxiliary to the suppress+inject intervention: this ranks what the CLT's
    // decoder says points at `"that"` at layer 14, independent of the
    // pre-specified suppress set. Step B will compute the same ranking for the
    // PLT and the two lists can be diffed directly.
    let top_k_target_layer = inject_feature.layer;
    eprintln!(
        "Ranking top-{} CLT features by cosine(decoder_row, unembed(\"{INJECT_WORD}\")) \
         at layer {top_k_target_layer}...",
        args.top_k
    );
    let direction = model.backend().embedding_vector(inject_token_id)?;
    let top_k_scores = clt.score_features_by_decoder_projection(
        &direction,
        top_k_target_layer,
        args.top_k,
        true, // cosine
    )?;
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

    // --- Cache steering vectors for suppress + inject across all downstream ---
    eprintln!("Caching decoder vectors for all downstream layers...");
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    // --- Build feature entries for all downstream layers ---
    let suppress_entries: Vec<(CltFeatureId, usize)> = suppress_features
        .iter()
        .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
        .collect();
    let inject_entries: Vec<(CltFeatureId, usize)> = (inject_feature.layer..n_layers)
        .map(|l| (inject_feature, l))
        .collect();
    eprintln!(
        "Suppress: {} entries across {} features; inject: {} entries (layers {}-{})",
        suppress_entries.len(),
        suppress_features.len(),
        inject_entries.len(),
        inject_feature.layer,
        n_layers - 1
    );

    // --- Baseline (no intervention) ---
    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let baseline_out = model.forward(&input, &HookSpec::new())?;
    let baseline_prob = extract_token_prob(baseline_out.output(), inject_token_id)?;
    let baseline_logit = extract_token_logit(baseline_out.output(), inject_token_id)?;
    eprintln!(
        "Baseline P(\"{inject_token_str}\") = {baseline_prob:.6e}  \
         logit = {baseline_logit:+.6}"
    );

    // --- Position sweep ---
    eprintln!(
        "\nSweeping {seq_len} positions (strength={})...",
        args.strength
    );
    let mut sweep: Vec<PositionResult> = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let mut combined =
            clt.prepare_hook_injection(&suppress_entries, pos, seq_len, -args.strength, &device)?;
        let inject_hooks =
            clt.prepare_hook_injection(&inject_entries, pos, seq_len, args.strength, &device)?;
        combined.extend(&inject_hooks);
        let result = model.forward(&input, &combined)?;
        let prob = extract_token_prob(result.output(), inject_token_id)?;
        let logit = extract_token_logit(result.output(), inject_token_id)?;
        // BORROW: String::clone — sweep entry needs an owned token string
        // because the outer token_strs Vec is iterated later for display.
        let token = token_strs.get(pos).map_or_else(String::new, String::clone);
        let display = token.replace('\n', "\\n");
        eprintln!(
            "  pos {pos:>3}  {display:<20}  P={prob:.6e}  logit={logit:+.4}  \
             ΔP={:+.6e}",
            prob - baseline_prob
        );
        sweep.push(PositionResult {
            position: pos,
            token,
            prob,
            logit,
        });
    }

    // --- Locate the spike ---
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
    // INDEX: spike_idx was just produced by max_by over sweep.iter(), so it
    // is in-bounds by construction. Use .get() anyway to satisfy the crate's
    // indexing_slicing lint without an #[allow]; the Err arm is unreachable.
    let spike = sweep
        .get(spike_idx)
        .ok_or_else(|| candle_mi::MIError::Config("spike index out of range".into()))?;
    let spike_position = spike.position;
    // BORROW: clone() — spike.token borrowed from sweep; we need an owned copy
    let spike_token = spike.token.clone();
    let max_prob = spike.prob;
    let max_logit = spike.logit;

    // --- Sanity gates ---
    if max_prob < LLAMA_SPIKE_HARD_MIN {
        return Err(candle_mi::MIError::Config(format!(
            "CLT sanity failed: max P(\"{INJECT_WORD}\") = {max_prob:.4} < \
             hard-min {LLAMA_SPIKE_HARD_MIN:.2}. Expected ~{LLAMA_REFERENCE_MAX_PROB:.3} \
             (candle-mi reference on the same CLT; figure13_planning_poems.rs \
             reproduces the same number). Check model/CLT repo pinning."
        )));
    }
    let band_diff = (max_prob - LLAMA_REFERENCE_MAX_PROB).abs();
    if band_diff > LLAMA_REFERENCE_TOL {
        eprintln!(
            "\nWARN: max_prob {max_prob:.4} drifted {band_diff:.4} from \
             reference {LLAMA_REFERENCE_MAX_PROB:.3} (tol ±{LLAMA_REFERENCE_TOL:.2}); \
             spike reproduced but numerical band loosened — worth a second look."
        );
    } else {
        eprintln!(
            "\nOK: max_prob {max_prob:.4} within ±{LLAMA_REFERENCE_TOL:.2} of \
             reference {LLAMA_REFERENCE_MAX_PROB:.3}."
        );
    }

    // --- Serialize ---
    let output_path = args.output.clone().unwrap_or_else(default_output_path);
    let output = ComparisonOutput {
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
        spike_token,
        max_prob,
        max_logit,
        delta_prob: max_prob - baseline_prob,
        delta_logit: max_logit - baseline_logit,
        reference_max_prob: LLAMA_REFERENCE_MAX_PROB,
        reference_tolerance: LLAMA_REFERENCE_TOL,
        sweep,
    };
    write_output(&output, &output_path)?;

    eprintln!(
        "\nSpike: pos {spike_position} (\"{}\"), P={max_prob:.4}, logit={max_logit:+.4}",
        output.spike_token.replace('\n', "\\n")
    );
    eprintln!(
        "ΔP = {:+.4}, Δlogit = {:+.4}",
        output.delta_prob, output.delta_logit
    );
    eprintln!("Total elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

fn write_output(output: &ComparisonOutput, path: &Path) -> candle_mi::Result<()> {
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
