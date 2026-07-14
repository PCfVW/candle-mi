// SPDX-License-Identifier: MIT OR Apache-2.0

//! Experiment 1 of the BlackboxNLP newline-experiments spec: the **newline
//! feature census** (correlational).
//!
//! Question: at the newline positions of a Figure-13 poem, do ANY `CLT`
//! features carry anticipatory content about the upcoming rhyme, or is the
//! anticipatory signal confined to the trailing-space planning site adjacent
//! to emission?
//!
//! This binary is **stage 1** of a two-stage pipeline. It forwards each cell's
//! standard four-line prompt once, selects the census positions (every newline
//! token, two mid-line content controls, and the trailing-space positive
//! control), `CLT`-encodes the residual at every layer at each position, and
//! for every collected feature records:
//!
//! - its activation at that position (native activation function of the `CLT`,
//!   i.e. plain `ReLU` for `mntss` / `JumpReLU` for `BlueLightAI`, via the same
//!   [`top_k`](candle_mi::clt::CrossLayerTranscoder::top_k) path used by
//!   `clt_probe`), and
//! - **c3**: the cosine of the feature's decoder vector (projected to the
//!   writeable layer closest to the LM head) to the natural target word's token
//!   embedding.
//!
//! It deliberately does **not** compute the census-membership (c1) or
//! decoder-top-20 rime-group (c2) fields, nor the registered `plan_like`
//! decision: those require `CMUdict` rime grouping and the raw vocab-scan
//! JSONs, and are added by the stage-2 post-processor
//! `scripts/newline_census_classify.py`, which reuses the existing
//! `vocab_scan_cmudict_filter.py` machinery. The stage-1 JSON is marked
//! `"classified": false` so the post-processor can detect un-augmented files.
//!
//! Token indexing matches the `figure13_planning_poems` convention exactly
//! (position = index into the tokenized prompt with the trailing space
//! appended, `BOS` included where the tokenizer prepends it). The full token
//! list and the detected newline indices are written into the output JSON, so
//! the Llama merged `,\n` token and the Qwen3 no-`BOS` offset are auditable per
//! cell.
//!
//! ```bash
//! # Gemma 2 2B, 426K CLT (one of the seven Table 2 cells)
//! cargo run --release --features clt,transformer,mmap --example figure13_newline_census -- \
//!     --preset gemma2-2b-426k \
//!     --output docs/experiments/figure13-newline/census_gemma2-2b-426k.json
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use clap::Parser;
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookPoint, HookSpec, MIModel};

// Shared Figure-13 cell presets (see module docs).  `#[path]`-included rather
// than declared as its own example: the file lives in a `main.rs`-free
// subdirectory of `examples/`, so Cargo's example auto-discovery skips it.
#[path = "figure13_common/presets.rs"]
mod presets;

use presets::select_preset;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "figure13_newline_census")]
#[command(about = "Experiment 1: newline feature census (correlational, stage 1 of 2)")]
struct Args {
    /// Preset name; one of the seven Table 2 cells (all presets except
    /// `gemma2-2b-2.5m`).
    #[arg(long, default_value = "gemma2-2b-426k")]
    preset: String,

    /// `HuggingFace` model ID (overrides preset).
    #[arg(long)]
    model: Option<String>,

    /// `HuggingFace` CLT repository (overrides preset).
    #[arg(long)]
    clt_repo: Option<String>,

    /// Prompt text (overrides preset). A single trailing space is appended so
    /// the final positive-control position is the trailing space before the
    /// natural rhyme word.
    #[arg(long)]
    prompt: Option<String>,

    /// Natural target word for the c3 cosine (overrides preset
    /// `suppress_word`, the poem's natural rhyme word).
    #[arg(long)]
    natural_word: Option<String>,

    /// Features to keep per (position, layer). `0` (the default) keeps **all**
    /// active features — the amended-spec requirement, so a weak rhyme feature
    /// cannot be silently truncated below stronger generic features. A positive
    /// value caps to that many top features per layer (legacy behaviour). The
    /// per-position list is the union across layers with per-layer provenance.
    #[arg(long, default_value_t = 0)]
    top_k: usize,

    /// Explicit control positions (comma-separated token indices), overriding
    /// the default heuristic (second token of the line after each of the first
    /// two newlines).
    #[arg(long, value_delimiter = ',')]
    control_positions: Vec<usize>,

    /// Output JSON path (defaults to stdout).
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Output types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct CensusOutput {
    /// Resolved `HuggingFace` model ID.
    model: String,
    /// Resolved `HuggingFace` CLT repository ID.
    clt_repo: String,
    /// Preset name this census was run for.
    preset: String,
    /// The four-line prompt (without the appended trailing space).
    prompt: String,
    /// Natural target word used for the c3 cosine.
    natural_word: String,
    /// Token ID of `natural_word` used for the c3 cosine.
    natural_token_id: u32,
    /// Every prompt token, in order (index = position). Includes any `BOS`.
    tokens: Vec<String>,
    /// Token indices treated as newlines (includes Llama merged `,\n` tokens).
    newline_positions: Vec<usize>,
    /// The per-(position, layer) top-K used.
    top_k: usize,
    /// Pipeline-stage marker; always `"rust-census-v1"` for this binary.
    stage: &'static str,
    /// `false` until `scripts/newline_census_classify.py` adds the c1/c2 and
    /// `plan_like` fields.
    classified: bool,
    /// One entry per selected census position.
    positions: Vec<PositionCensus>,
}

#[derive(Serialize)]
struct PositionCensus {
    /// Token index into `tokens`.
    position: usize,
    /// Decoded token text at this position (may contain a newline).
    token: String,
    /// Role of this position: `"newline"`, `"control"`, or `"final"`.
    role: &'static str,
    /// Collected features (union of each layer's top-K), most-active first.
    features: Vec<FeatureCensus>,
}

#[derive(Serialize)]
struct FeatureCensus {
    /// Source layer of the feature.
    layer: usize,
    /// Feature index within the layer.
    index: usize,
    /// Feature activation at this position (native activation function).
    activation: f32,
    /// c3: cosine of the feature's decoder vector (to the writeable layer
    /// closest to the LM head) to `natural_word`'s token embedding.
    cos_to_target: f32,
}

// ── Position role tags ───────────────────────────────────────────────────────

const ROLE_NEWLINE: &str = "newline";
const ROLE_CONTROL: &str = "control";
const ROLE_FINAL: &str = "final";

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let preset = select_preset(&args.preset)?;
    // BORROW: .to_owned() — convert &'static str to owned String for storage.
    let model_id = args.model.unwrap_or_else(|| preset.model.to_owned());
    let clt_repo = args.clt_repo.unwrap_or_else(|| preset.clt_repo.to_owned());
    let prompt = args.prompt.unwrap_or_else(|| preset.prompt.to_owned());
    let natural_word = args
        .natural_word
        .unwrap_or_else(|| preset.suppress_word.to_owned());

    eprintln!("=== Experiment 1: newline feature census (stage 1) ===\n");
    eprintln!("Preset:       {}", args.preset);
    eprintln!("Model:        {model_id}");
    eprintln!("CLT:          {clt_repo}");
    eprintln!("Natural word: \"{natural_word}\" (c3 cosine target)");
    if args.top_k == 0 {
        eprintln!("Per layer:    ALL active features (un-truncated)\n");
    } else {
        eprintln!("Per layer:    top {} by activation\n", args.top_k);
    }

    run_census(
        &args.preset,
        &model_id,
        &clt_repo,
        &prompt,
        &natural_word,
        args.top_k,
        &args.control_positions,
        args.output.as_deref(),
    )
}

/// Load model + CLT, select census positions, encode features, compute c3, and
/// write the stage-1 JSON.
#[allow(clippy::too_many_arguments)]
fn run_census(
    preset_name: &str,
    model_id: &str,
    clt_repo_name: &str,
    prompt: &str,
    natural_word: &str,
    top_k: usize,
    control_positions: &[usize],
    output_path: Option<&Path>,
) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    // --- Load model + CLT ---
    eprintln!("Loading model...");
    let model = MIModel::from_pretrained(model_id)?;
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;

    eprintln!("Opening CLT: {clt_repo_name}...");
    let mut clt = CrossLayerTranscoder::open(clt_repo_name)?;
    let n_layers = clt.config().n_layers;
    // The writeable layer closest to the LM head — the decoder-projection
    // target used by the vocabulary scan and thus by the c3 cosine.
    let target_layer = n_layers - 1;

    // --- Tokenize (trailing space appended: matches figure13 convention) ---
    let prompt_with_space = format!("{prompt} ");
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

    // --- Natural target embedding for c3 (unit-normalised, F32, CPU) ---
    let natural_token_id = tokenizer.find_token_id(natural_word)?;
    let target_unit = unit_vector_cpu(&model.backend().embedding_vector(natural_token_id)?)?;

    // --- Select census positions ---
    let newline_positions = detect_newline_positions(&token_strs);
    if newline_positions.is_empty() {
        return Err(candle_mi::MIError::Config(
            "no newline tokens found in prompt (cannot run the census)".into(),
        ));
    }
    let controls = if control_positions.is_empty() {
        default_controls(&newline_positions, seq_len)
    } else {
        control_positions.to_vec()
    };
    let final_pos = seq_len - 1;
    let selected = selected_positions(&newline_positions, &controls, final_pos);

    eprintln!("Tokens ({seq_len}):");
    for (i, tok) in token_strs.iter().enumerate() {
        // BORROW: .as_str() — String → &str for the display replace.
        let shown = tok.as_str().replace('\n', "\\n");
        let role = selected
            .iter()
            .find(|(p, _)| *p == i)
            .map_or("", |(_, r)| *r);
        let marker = if role.is_empty() {
            String::new()
        } else {
            format!("   <- {role}")
        };
        eprintln!("  {i:>3}  {shown}{marker}");
    }
    eprintln!(
        "\nNewline positions: {newline_positions:?}   controls: {controls:?}   final: {final_pos}\n"
    );

    // --- Forward once, capturing ResidMid at every layer ---
    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::ResidMid(layer));
    }
    eprintln!("Running forward pass ({n_layers} captures)...");
    let cache = model.forward(&input, &hooks)?;

    // --- Encode top-K per (position, layer) ---
    // Collect per-position feature lists and the set of unique features (for a
    // single batched c3 pass).
    let mut positions: Vec<PositionCensus> = Vec::with_capacity(selected.len());
    // Dedup unique features across all positions so each decoder vector is
    // fetched once for the c3 cosine.
    let mut unique_features: BTreeSet<(usize, usize)> = BTreeSet::new();

    for &(pos, role) in &selected {
        let mut feats: Vec<(CltFeatureId, f32)> = Vec::new();
        for layer in 0..n_layers {
            clt.load_encoder(layer, &device)?;
            // `Tensor::get` selects along dim 0: first the batch row, then the
            // position row. pos < seq_len for every selected position.
            let resid = cache
                .require(&HookPoint::ResidMid(layer))?
                .get(0)?
                .get(pos)?;
            // top_k == 0 → keep all active features (amended-spec default);
            // a positive cap uses the partial-sort top_k path.
            let sparse = if top_k == 0 {
                clt.encode(&resid, layer)?
            } else {
                clt.top_k(&resid, layer, top_k)?
            };
            for (fid, act) in sparse.features {
                unique_features.insert((fid.layer, fid.index));
                feats.push((fid, act));
            }
        }
        // Most-active first (stable, deterministic ordering).
        feats.sort_by(|a, b| b.1.total_cmp(&a.1));

        positions.push(PositionCensus {
            position: pos,
            // pos < seq_len, so token_strs has an entry; `.get` avoids the
            // crate-wide `indexing_slicing` deny while preserving the invariant.
            token: token_strs.get(pos).cloned().unwrap_or_default(),
            role,
            // c3 filled in the second pass once decoders are cached.
            features: feats
                .into_iter()
                .map(|(fid, act)| FeatureCensus {
                    layer: fid.layer,
                    index: fid.index,
                    activation: act,
                    cos_to_target: 0.0,
                })
                .collect(),
        });
    }

    // --- c3: batch-cache decoder vectors, then cosine each unique feature ---
    eprintln!(
        "Computing c3 cosines for {} unique features...",
        unique_features.len()
    );
    let pairs: Vec<(CltFeatureId, usize)> = unique_features
        .iter()
        .map(|&(layer, index)| (CltFeatureId { layer, index }, target_layer))
        .collect();
    clt.cache_steering_vectors(&pairs, &device)?;

    let mut cos_by_feature: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    for (fid, _) in &pairs {
        let dec = clt.decoder_vector(fid, target_layer, &device)?;
        let dec_unit = unit_vector_cpu(&dec)?;
        let cos = dot(&dec_unit, &target_unit);
        cos_by_feature.insert((fid.layer, fid.index), cos);
    }

    // Backfill c3 into every occurrence.
    for pos in &mut positions {
        for feat in &mut pos.features {
            if let Some(&cos) = cos_by_feature.get(&(feat.layer, feat.index)) {
                feat.cos_to_target = cos;
            }
        }
    }

    // --- Assemble + write ---
    let output = CensusOutput {
        model: model_id.into(),
        clt_repo: clt_repo_name.into(),
        preset: preset_name.into(),
        prompt: prompt.into(),
        natural_word: natural_word.into(),
        natural_token_id,
        tokens: token_strs,
        newline_positions,
        top_k,
        stage: "rust-census-v1",
        classified: false,
        positions,
    };
    write_output(&output, output_path)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

// ── Position selection ───────────────────────────────────────────────────────

/// Token indices whose decoded text contains a newline. Catches both a bare
/// `"\n"` token and the Llama merged `",\n"` token.
fn detect_newline_positions(tokens: &[String]) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.contains('\n'))
        .map(|(i, _)| i)
        .collect()
}

/// Default base-rate controls: the second token of the line following each of
/// the first two newlines (i.e. `newline_index + 2`), skipping any that would
/// collide with a newline, the final position, or exceed the sequence.
fn default_controls(newline_positions: &[usize], seq_len: usize) -> Vec<usize> {
    let final_pos = seq_len.saturating_sub(1);
    let mut controls = Vec::new();
    for &nl in newline_positions.iter().take(2) {
        let candidate = nl + 2;
        if candidate < final_pos
            && !newline_positions.contains(&candidate)
            && !controls.contains(&candidate)
        {
            controls.push(candidate);
        }
    }
    controls
}

/// Merge the newline, control, and final positions into a single ordered,
/// de-duplicated list of `(position, role)` pairs. Newlines win over controls
/// on collision; the final position wins over both.
fn selected_positions(
    newline_positions: &[usize],
    controls: &[usize],
    final_pos: usize,
) -> Vec<(usize, &'static str)> {
    // BTreeMap keeps positions sorted and de-duplicated; later inserts of a
    // higher-priority role overwrite lower-priority ones.
    let mut by_pos: BTreeMap<usize, &'static str> = BTreeMap::new();
    for &c in controls {
        by_pos.insert(c, ROLE_CONTROL);
    }
    for &n in newline_positions {
        by_pos.insert(n, ROLE_NEWLINE);
    }
    by_pos.insert(final_pos, ROLE_FINAL);
    by_pos.into_iter().collect()
}

// ── Vector helpers ───────────────────────────────────────────────────────────

/// Unit-normalise a `[d_model]` tensor into a host `Vec<f32>` on CPU.
///
/// A zero vector is returned unchanged (all zeros) rather than producing
/// `NaN`s from a divide-by-zero.
fn unit_vector_cpu(v: &Tensor) -> candle_mi::Result<Vec<f32>> {
    // PROMOTE: embedding / decoder vectors may be BF16 on disk; F32 for the
    // dot-product precision the cosine needs.
    let host: Vec<f32> = v
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let norm = host.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 1e-10 {
        return Ok(host);
    }
    Ok(host.iter().map(|x| x / norm).collect())
}

/// Dot product of two equal-length host vectors. If lengths differ, the excess
/// tail of the longer vector is ignored (the shared prefix is dotted).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Output ───────────────────────────────────────────────────────────────────

/// Serialise the census to JSON; write to file (creating parents) or stdout.
fn write_output(output: &CensusOutput, path: Option<&Path>) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to serialize census JSON: {e}")))?;

    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
            })?;
        }
        fs::write(p, &json).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to write census output: {e}"))
        })?;
        eprintln!("\nOutput written to {}", p.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
