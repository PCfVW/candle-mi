// SPDX-License-Identifier: MIT OR Apache-2.0

//! Commitment-onset layer at the planning site (logit-lens + CLT-activation).
//!
//! Complements the Figure-13 position sweep (`means_ends_sweep` /
//! `figure13_planning_poems`), which locates *where in the sequence* the
//! commitment lives (the planning site = last content token). This example
//! locates *at which layer* the commitment forms, two complementary ways, both
//! read at the planning-site position:
//!
//! 1. **Logit-lens** — project each layer's `ResidPost` through the unembedding
//!    and track `P(committed token)`; the **onset** is the earliest layer at
//!    which the committed token becomes the top-1 prediction (and stays).
//! 2. **CLT feature-activation** — for the layer-`L` feature that best encodes
//!    the committed token (per `scripts/pick_per_layer_feature.py`), read its
//!    activation from `ResidMid(L)` via the CLT encoder; the **onset** is the
//!    earliest layer at which that activation exceeds a threshold.
//!
//! The single-layer causal injection cannot measure onset (a CLT feature only
//! exists from its source layer downward, and our planning features are late),
//! so these two readout signals are the right instrument.
//!
//! ```bash
//! # means-ends (aggregate over the 48-item set)
//! cargo run --release --features clt,transformer,mmap --example commitment_onset -- \
//!     --items docs/experiments/means-ends-prolepsis/step_b_items.json \
//!     --per-layer-features docs/experiments/means-ends-prolepsis/per_layer_features_gemma2_2b_2.5m.json \
//!     --output docs/experiments/means-ends-prolepsis/onset_gemma2_2b_2.5m.json
//!
//! # a single rhyme prompt (committed token defaults to the planning-site argmax)
//! cargo run --release --features clt,transformer,mmap --example commitment_onset -- \
//!     --prompt "<poem>" \
//!     --per-layer-features <scan_per_layer.json> --output <out.json>
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Tensor;
use clap::Parser;
use serde::{Deserialize, Serialize};

use candle_mi::clt::CrossLayerTranscoder;
use candle_mi::{HookPoint, HookSpec, MIModel};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "commitment_onset")]
#[command(about = "Planning-site commitment-onset layer (logit-lens + CLT-activation)")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// `HuggingFace` `CLT` repository.
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-2.5M")]
    clt_repo: String,

    /// Means-ends items JSON (each `{correct, prompt, ...}`); mutually exclusive
    /// with `--prompt`.
    #[arg(long)]
    items: Option<PathBuf>,

    /// Single prompt (rhyme cells), mutually exclusive with `--items`. The
    /// committed token defaults to the planning-site argmax unless
    /// `--committed-token` is given.
    #[arg(long)]
    prompt: Option<String>,

    /// Committed token (e.g. `" that"`). Optional: in `--prompt` mode it defaults
    /// to the model's final-layer top-1 at the planning site; in `--items` mode
    /// it overrides each item's `correct` if given.
    #[arg(long)]
    committed_token: Option<String>,

    /// Per-layer best-feature map from `pick_per_layer_feature.py`
    /// (`{token: {layer: {index, cosine}}}`).
    #[arg(long)]
    per_layer_features: PathBuf,

    /// CLT activation onset threshold (first layer whose activation exceeds this).
    #[arg(long, default_value_t = 0.0)]
    act_threshold: f32,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/onset.json"
    )]
    output: PathBuf,
}

// ── Input ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Item {
    /// Committed (goal-correct) token, e.g. `on` / `off`.
    correct: String,
    prompt: String,
}

#[derive(Deserialize, Clone, Copy)]
struct FeatEntry {
    index: usize,
    #[allow(dead_code)]
    cosine: f32,
}

/// `token -> (layer-string -> best feature)`.
type PerLayerFeatures = HashMap<String, HashMap<String, FeatEntry>>;

// ── Output ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct LayerStat {
    layer: usize,
    /// Mean `P(committed)` across items (logit-lens at this layer).
    logitlens_p_mean: f64,
    /// Fraction of items where the committed token is top-1 at this layer.
    logitlens_top1_frac: f64,
    /// Mean CLT activation of the committed-token feature at this layer.
    cltact_mean: f64,
    /// How many items had a per-layer feature for the committed token here.
    cltact_covered: usize,
}

#[derive(Serialize)]
struct Output {
    model: String,
    clt_repo: String,
    n_layers: usize,
    n_items: usize,
    act_threshold: f32,
    /// Median over items of the first layer where the committed token is top-1.
    onset_layer_logitlens_median: Option<f64>,
    /// Median over items of the first layer where the CLT activation exceeds the
    /// threshold.
    onset_layer_cltact_median: Option<f64>,
    per_layer: Vec<LayerStat>,
    elapsed_secs: f64,
}

// ── Captured residuals for one item ───────────────────────────────────────────

/// Planning-site residuals for one item: `resid_post[L]` and `resid_mid[L]` are
/// each `[hidden]` at the last token.
struct ItemCapture {
    correct: String,
    committed_id: u32,
    resid_post: Vec<Tensor>,
    resid_mid: Vec<Tensor>,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn read_to_string(path: &Path) -> candle_mi::Result<String> {
    fs::read_to_string(path)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display())))
}

/// Lossless small-count `usize -> f64`.
fn count_to_f64(count: usize) -> candle_mi::Result<f64> {
    let as_u32 = u32::try_from(count)
        .map_err(|e| candle_mi::MIError::Config(format!("count {count} exceeds u32: {e}")))?;
    Ok(f64::from(as_u32))
}

/// Median of a slice of layer indices (already small, sorted copy).
fn median(values: &[usize]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v: Vec<usize> = values.to_vec();
    v.sort_unstable();
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        Some(count_to_f64(*v.get(mid)?).ok()?)
    } else {
        let a = count_to_f64(*v.get(mid.checked_sub(1)?)?).ok()?;
        let b = count_to_f64(*v.get(mid)?).ok()?;
        Some(a.midpoint(b))
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

// ── Phase 1: capture planning-site residuals for one item ──────────────────────

fn capture_item(
    model: &MIModel,
    item: &Item,
    n_layers: usize,
    committed_override: Option<&str>,
    derive_argmax: bool,
) -> candle_mi::Result<ItemCapture> {
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;

    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::ResidPost(layer));
        hooks.capture(HookPoint::ResidMid(layer));
    }
    let result = model.forward_text(&item.prompt, &hooks)?;
    let cache = result.cache();
    let seq_len = result.encoding().tokens.len();
    let site = seq_len - 1; // planning site = last content token

    let mut resid_post = Vec::with_capacity(n_layers);
    let mut resid_mid = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        // [1, seq, hidden] -> [hidden] at the planning site.
        let rp = cache
            .require(&HookPoint::ResidPost(layer))?
            .get(0)?
            .get(site)?;
        let rm = cache
            .require(&HookPoint::ResidMid(layer))?
            .get(0)?
            .get(site)?;
        resid_post.push(rp);
        resid_mid.push(rm);
    }

    // Resolve the committed token: either the model's own final-layer top-1 at
    // the planning site (rhyme cells, where the planned word is not known a
    // priori), or the supplied token / the item's `correct` (means-ends).
    let (correct, committed_id) = if derive_argmax {
        let last = resid_post
            .get(n_layers - 1)
            .ok_or_else(|| candle_mi::MIError::Config("no final-layer residual captured".into()))?;
        let id = argmax_vocab(model, last)?;
        (tokenizer.decode_token(id)?, id)
    } else {
        // BORROW: committed token string — the override, else the item's `correct`.
        let s = committed_override
            .unwrap_or(item.correct.as_str())
            .to_owned();
        let id = tokenizer.find_token_id(&s)?;
        (s, id)
    };

    Ok(ItemCapture {
        correct,
        committed_id,
        resid_post,
        resid_mid,
    })
}

/// Vocab argmax of a planning-site residual via the unembedding.
fn argmax_vocab(model: &MIModel, resid_post: &Tensor) -> candle_mi::Result<u32> {
    let hidden = resid_post.unsqueeze(0)?; // [1, hidden]
    let logits = model.project_to_vocab(&hidden)?;
    // PROMOTE: argmax in F32 for numerical stability.
    let v: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let idx = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .ok_or_else(|| candle_mi::MIError::Config("empty logits vector".into()))?;
    u32::try_from(idx)
        .map_err(|e| candle_mi::MIError::Config(format!("argmax index {idx} exceeds u32: {e}")))
}

// ── Phase 2 helpers: per-layer logit-lens and CLT activation ───────────────────

/// `(P(committed), is_top1)` from a planning-site residual via logit-lens.
fn logit_lens_at(
    model: &MIModel,
    resid_post: &Tensor,
    committed_id: u32,
) -> candle_mi::Result<(f32, bool)> {
    let hidden = resid_post.unsqueeze(0)?; // [1, hidden]
    let logits = model.project_to_vocab(&hidden)?;
    // PROMOTE: softmax / argmax in F32 for numerical stability.
    let logits_f32 = logits.to_dtype(candle_core::DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;
    let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    // committed_id came from `find_token_id`, so it indexes the vocab.
    let idx = usize::try_from(committed_id).map_err(|e| {
        candle_mi::MIError::Config(format!("token id {committed_id} exceeds usize: {e}"))
    })?;
    let p = probs_vec.get(idx).copied().unwrap_or(0.0);
    // Top-1 check without a full sort: is any prob strictly greater?
    let is_top1 = !probs_vec.iter().any(|&v| v > p) && p > 0.0;
    Ok((p, is_top1))
}

/// CLT activation (post-ReLU) of one feature index at a planning-site residual.
fn clt_activation_at(
    clt: &CrossLayerTranscoder,
    resid_mid: &Tensor,
    layer: usize,
    feature_index: usize,
) -> candle_mi::Result<f32> {
    let pre = clt.encode_pre_activation(resid_mid, layer)?;
    let value = pre.get(feature_index)?.to_scalar::<f32>()?;
    Ok(value.max(0.0)) // ReLU
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

    // --- Resolve input items ---
    let items: Vec<Item> = if let Some(ref items_path) = args.items {
        let json = read_to_string(items_path)?;
        serde_json::from_str(&json).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to parse {}: {e}", items_path.display()))
        })?
    } else if let Some(ref prompt) = args.prompt {
        // `--committed-token` optional in prompt mode: when absent the committed
        // token defaults to the model's final-layer top-1 at the planning site.
        vec![Item {
            correct: args.committed_token.clone().unwrap_or_default(),
            prompt: prompt.clone(),
        }]
    } else {
        return Err(candle_mi::MIError::Config(
            "provide --items or --prompt".into(),
        ));
    };
    // Derive the committed token from the final-layer argmax only in prompt mode
    // with no explicit `--committed-token` (rhyme cells); means-ends items always
    // carry their `correct` action.
    let derive_argmax = args.items.is_none() && args.committed_token.is_none();

    let per_layer: PerLayerFeatures = {
        let json = read_to_string(&args.per_layer_features)?;
        serde_json::from_str(&json).map_err(|e| {
            candle_mi::MIError::Config(format!(
                "failed to parse {}: {e}",
                args.per_layer_features.display()
            ))
        })?
    };

    eprintln!("=== Commitment-onset (logit-lens + CLT-activation) ===\n");
    eprintln!("Model:   {}", args.model);
    eprintln!("CLT:     {}", args.clt_repo);
    eprintln!("Items:   {}\n", items.len());

    let model = MIModel::from_pretrained(&args.model)?;
    let n_layers = model.num_layers();
    let device = model.device().clone();
    eprintln!("  {n_layers} layers, device={device:?}");

    // --- Phase 1: capture planning-site residuals for every item ---
    eprintln!("Phase 1: capturing planning-site residuals...");
    let committed_override = if args.items.is_some() {
        // For item mode, the override applies only if explicitly set.
        args.committed_token.as_deref()
    } else {
        None // prompt mode already baked `committed_token` into the item
    };
    let mut caps: Vec<ItemCapture> = Vec::with_capacity(items.len());
    for item in &items {
        caps.push(capture_item(
            &model,
            item,
            n_layers,
            committed_override,
            derive_argmax,
        )?);
    }
    if derive_argmax && let Some(first) = caps.first() {
        eprintln!(
            "  committed token (final-layer argmax): {:?}",
            first.correct
        );
    }

    // --- Phase 2: per-layer logit-lens + CLT activation ---
    eprintln!("Phase 2: per-layer logit-lens + CLT activation...");
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_items = caps.len();

    // Running per-item onsets: first layer where the committed token is top-1
    // (logit-lens) / where the CLT activation exceeds the threshold.
    let mut onset_ll_item: Vec<Option<usize>> = vec![None; n_items];
    let mut onset_act_item: Vec<Option<usize>> = vec![None; n_items];
    let mut stats: Vec<LayerStat> = Vec::with_capacity(n_layers);
    // The CLT-activation side needs the encoder; JumpReLU CLTs whose threshold
    // tensor is absent cannot load it. Degrade gracefully to logit-lens-only
    // (which needs no CLT) rather than aborting the whole run.
    let mut clt_enc_failed = false;

    for layer in 0..n_layers {
        let layer_key = layer.to_string();
        let enc_loaded = match clt.load_encoder(layer, &device) {
            Ok(()) => true,
            Err(e) => {
                if !clt_enc_failed {
                    eprintln!(
                        "warning: CLT encoder unavailable ({e}); reporting logit-lens onset only"
                    );
                    clt_enc_failed = true;
                }
                false
            }
        };

        let mut sum_p = 0.0_f64;
        let mut n_top1 = 0usize;
        let mut sum_act = 0.0_f64;
        let mut n_cov = 0usize;

        for (i, cap) in caps.iter().enumerate() {
            let rp = cap.resid_post.get(layer).ok_or_else(|| {
                candle_mi::MIError::Config(format!("missing ResidPost[{layer}] for item {i}"))
            })?;
            let (p, is_top1) = logit_lens_at(&model, rp, cap.committed_id)?;
            sum_p += f64::from(p);
            if is_top1 {
                n_top1 += 1;
                if let Some(slot) = onset_ll_item.get_mut(i) {
                    slot.get_or_insert(layer);
                }
            }

            // CLT activation: look up the per-layer feature for this committed token.
            if enc_loaded
                && let Some(entry) = per_layer.get(&cap.correct).and_then(|m| m.get(&layer_key))
            {
                let rm = cap.resid_mid.get(layer).ok_or_else(|| {
                    candle_mi::MIError::Config(format!("missing ResidMid[{layer}] for item {i}"))
                })?;
                let act = clt_activation_at(&clt, rm, layer, entry.index)?;
                sum_act += f64::from(act);
                n_cov += 1;
                if act > args.act_threshold
                    && let Some(slot) = onset_act_item.get_mut(i)
                {
                    slot.get_or_insert(layer);
                }
            }
        }

        let n_items_f64 = count_to_f64(n_items)?;
        let cltact_mean = if n_cov > 0 {
            sum_act / count_to_f64(n_cov)?
        } else {
            0.0
        };
        stats.push(LayerStat {
            layer,
            logitlens_p_mean: sum_p / n_items_f64,
            logitlens_top1_frac: count_to_f64(n_top1)? / n_items_f64,
            cltact_mean,
            cltact_covered: n_cov,
        });
    }

    // --- Per-item onsets -> medians (drop items that never reached the onset) ---
    let onset_ll: Vec<usize> = onset_ll_item.into_iter().flatten().collect();
    let onset_act: Vec<usize> = onset_act_item.into_iter().flatten().collect();

    // --- Report ---
    eprintln!("\n=== Commitment-onset trajectory (planning site) ===");
    eprintln!(
        "  {:>5}  {:>12}  {:>10}  {:>12}  {:>4}",
        "Layer", "P(committed)", "top1_frac", "CLT_act", "cov"
    );
    for s in &stats {
        eprintln!(
            "  {:>5}  {:>12.4}  {:>10.2}  {:>12.4}  {:>4}",
            s.layer, s.logitlens_p_mean, s.logitlens_top1_frac, s.cltact_mean, s.cltact_covered
        );
    }
    let onset_ll_med = median(&onset_ll);
    let onset_act_med = median(&onset_act);
    eprintln!(
        "\nonset (logit-lens, median first top-1 layer): {onset_ll_med:?}  over {}/{} items",
        onset_ll.len(),
        n_items
    );
    eprintln!(
        "onset (CLT-act > {}, median first layer):     {onset_act_med:?}  over {}/{} items",
        args.act_threshold,
        onset_act.len(),
        n_items
    );

    let output = Output {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        n_layers,
        n_items,
        act_threshold: args.act_threshold,
        onset_layer_logitlens_median: onset_ll_med,
        onset_layer_cltact_median: onset_act_med,
        per_layer: stats,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    };
    write_json(&args.output, &output)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}
