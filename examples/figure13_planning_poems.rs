// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replication of Anthropic's "Planning in Poems" Figure 13: suppress + inject
//! position sweep.
//!
//! Suppresses natural rhyme group CLT features while injecting an alternative
//! feature, sweeping the injection position across the prompt to locate
//! planning sites.
//!
//! Three built-in presets select model, CLT, prompt, features, and strength:
//!
//! | Preset | Model | CLT | Suppress | Inject |
//! |--------|-------|-----|----------|--------|
//! | `llama3.2-1b-524k` | Llama 3.2 1B | 524K | -ee group: L13:30985 + L9:5488 + L14:27874 + L13:32049 | L14:13043 ("that") |
//! | `gemma2-2b-426k` | Gemma 2 2B | 426K | L16:13725 + L25:9385 ("-out") | L22:10243 ("around") |
//! | `gemma2-2b-2.5m` | Gemma 2 2B | 2.5M | L25:57092 + L23:49923 + L20:77102 ("-out") | L25:82839 ("can") |
//!
//! ```bash
//! # Llama 3.2 1B (default)
//! cargo run --release --features clt,transformer --example figure13_planning_poems
//!
//! # Gemma 2 2B, 426K CLT
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-426k
//!
//! # Gemma 2 2B, 2.5M CLT (word-level features)
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-2.5m
//! ```
//!
//! Outputs JSON suitable for direct import into Mathematica via
//! `Import["output.json"]`.

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::Tensor;
use clap::Parser;
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookSpec, MIModel, extract_token_prob};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "figure13_planning_poems")]
#[command(about = "Anthropic Figure 13 replication: suppress + inject position sweep")]
struct Args {
    /// Preset: "llama3.2-1b-524k", "gemma2-2b-426k", or "gemma2-2b-2.5m"
    #[arg(long, default_value = "llama3.2-1b-524k")]
    preset: String,

    /// `HuggingFace` model ID (overrides preset)
    #[arg(long)]
    model: Option<String>,

    /// `HuggingFace` CLT repository (overrides preset)
    #[arg(long)]
    clt_repo: Option<String>,

    /// Prompt text (overrides preset)
    #[arg(long)]
    prompt: Option<String>,

    /// Word to suppress (overrides preset)
    #[arg(long)]
    suppress_word: Option<String>,

    /// Word to inject (overrides preset)
    #[arg(long)]
    inject_word: Option<String>,

    /// Suppress features in "layer:index" format; repeatable (overrides preset)
    #[arg(long)]
    suppress_feature: Vec<String>,

    /// Inject feature in "layer:index" format (overrides preset)
    #[arg(long)]
    inject_feature: Option<String>,

    /// Steering strength (overrides preset)
    #[arg(long)]
    strength: Option<f32>,

    /// Output file path (defaults to stdout)
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Output types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SweepOutput {
    model: String,
    clt_repo: String,
    prompt: String,
    tokens: Vec<String>,
    suppress_word: String,
    inject_word: String,
    suppress_features: Vec<CltFeatureId>,
    inject_feature: CltFeatureId,
    strength: f32,
    baseline_prob: f32,
    sweep: Vec<PositionResult>,
}

#[derive(Serialize)]
struct PositionResult {
    position: usize,
    token: String,
    prob: f32,
}

// ── Presets ──────────────────────────────────────────────────────────────────

struct Preset {
    model: &'static str,
    clt_repo: &'static str,
    prompt: &'static str,
    suppress_word: &'static str,
    inject_word: &'static str,
    suppress_features: &'static [(usize, usize)],
    inject_feature: (usize, usize),
    strength: f32,
}

/// Llama 3.2 1B with 524K CLT.
///
/// Suppress -ee group: L13:30985 ("he"), L9:5488 ("be"), L14:27874 ("ne"),
/// L13:32049 ("we").  Inject "that" (L14:13043) from -at group.
/// From plip-rs validation: P("that") reaches 0.777 at the last position.
const LLAMA: Preset = Preset {
    model: "meta-llama/Llama-3.2-1B",
    clt_repo: "mntss/clt-llama-3.2-1b-524k",
    prompt: "The birds were singing in the tree,\n\
             And everything was wild and free.\n\
             The river ran down to the sea,\n\
             There is so much we cannot",
    suppress_word: "free",
    inject_word: "that",
    suppress_features: &[(13, 30985), (9, 5488), (14, 27874), (13, 32049)],
    inject_feature: (14, 13043),
    strength: 10.0,
};

/// Gemma 2 2B with 426K CLT.
///
/// Suppress "-out" group: L16:13725 ("about") + L25:9385 ("out").
/// Inject "around" (L22:10243).
/// From plip-rs validation: P("around") reaches 0.483 at planning site.
const GEMMA: Preset = Preset {
    model: "google/gemma-2-2b",
    clt_repo: "mntss/clt-gemma-2-2b-426k",
    prompt: "The stars were twinkling in the night,\n\
             The lanterns cast a golden light.\n\
             She wandered in the dark about,\n\
             And found a hidden passage",
    suppress_word: "out",
    inject_word: "around",
    suppress_features: &[(16, 13725), (25, 9385)],
    inject_feature: (22, 10243),
    strength: 10.0,
};

/// Gemma 2 2B with 2.5M CLT (word-level granularity).
///
/// Suppress "-out" words: L25:57092 ("about") + L23:49923 ("out") + L20:77102
/// ("without"). Inject "can" (L25:82839).
/// From plip-rs validation: P("can") reaches 0.425 at planning site.
const GEMMA_2M: Preset = Preset {
    model: "google/gemma-2-2b",
    clt_repo: "mntss/clt-gemma-2-2b-2.5M",
    prompt: "The stars were twinkling in the night,\n\
             The lanterns cast a golden light.\n\
             She wandered in the dark about,\n\
             And found a hidden passage",
    suppress_word: "out",
    inject_word: "can",
    suppress_features: &[(25, 57092), (23, 49923), (20, 77102)],
    inject_feature: (25, 82839),
    strength: 10.0,
};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Parse a "layer:index" string into a [`CltFeatureId`].
fn parse_feature(s: &str) -> candle_mi::Result<CltFeatureId> {
    let (layer_str, index_str) = s.split_once(':').ok_or_else(|| {
        candle_mi::MIError::Config(format!(
            "feature must be in 'layer:index' format, got '{s}'"
        ))
    })?;
    let layer: usize = layer_str.parse().map_err(|e| {
        candle_mi::MIError::Config(format!("invalid layer number '{layer_str}': {e}"))
    })?;
    let index: usize = index_str.parse().map_err(|e| {
        candle_mi::MIError::Config(format!("invalid feature index '{index_str}': {e}"))
    })?;
    Ok(CltFeatureId { layer, index })
}

const fn feature_id(pair: (usize, usize)) -> CltFeatureId {
    CltFeatureId {
        layer: pair.0,
        index: pair.1,
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

    // --- Select preset ---
    let preset = select_preset(&args.preset)?;

    // --- Resolve experiment parameters (CLI overrides preset) ---
    // BORROW: .to_owned() — convert &'static str to String for owned storage
    let model_id = args.model.unwrap_or_else(|| preset.model.to_owned());
    let clt_repo = args.clt_repo.unwrap_or_else(|| preset.clt_repo.to_owned());
    let prompt = args.prompt.unwrap_or_else(|| preset.prompt.to_owned());
    let suppress_word = args
        .suppress_word
        .unwrap_or_else(|| preset.suppress_word.to_owned());
    let inject_word = args
        .inject_word
        .unwrap_or_else(|| preset.inject_word.to_owned());
    let strength = args.strength.unwrap_or(preset.strength);

    let suppress_features: Vec<CltFeatureId> = if args.suppress_feature.is_empty() {
        preset
            .suppress_features
            .iter()
            .copied()
            .map(feature_id)
            .collect()
    } else {
        args.suppress_feature
            .iter()
            .map(|s| parse_feature(s))
            .collect::<candle_mi::Result<Vec<_>>>()?
    };

    let inject_feature = match &args.inject_feature {
        Some(s) => parse_feature(s)?,
        None => feature_id(preset.inject_feature),
    };

    eprintln!("=== Figure 13: Suppress + Inject Position Sweep ===\n");
    eprintln!("Preset:   {}", args.preset);
    eprintln!("Model:    {model_id}");
    eprintln!("CLT:      {clt_repo}");
    eprintln!("Suppress: \"{suppress_word}\" features {suppress_features:?}");
    eprintln!("Inject:   \"{inject_word}\" feature {inject_feature}");
    eprintln!("Strength: {strength}\n");

    run_experiment(
        &model_id,
        &clt_repo,
        &prompt,
        &suppress_word,
        &inject_word,
        &suppress_features,
        inject_feature,
        strength,
        args.output.as_deref(),
    )
}

/// Select a built-in preset by name.
fn select_preset(name: &str) -> candle_mi::Result<&'static Preset> {
    match name {
        "llama3.2-1b-524k" => Ok(&LLAMA),
        "gemma2-2b-426k" => Ok(&GEMMA),
        "gemma2-2b-2.5m" => Ok(&GEMMA_2M),
        other => Err(candle_mi::MIError::Config(format!(
            "unknown preset '{other}' \
             (expected 'llama3.2-1b-524k', 'gemma2-2b-426k', or 'gemma2-2b-2.5m')"
        ))),
    }
}

/// Load model + CLT, run the position sweep, print summary, and write output.
#[allow(clippy::too_many_arguments)]
fn run_experiment(
    model_id: &str,
    clt_repo_name: &str,
    prompt: &str,
    suppress_word: &str,
    inject_word: &str,
    suppress_features: &[CltFeatureId],
    inject_feature: CltFeatureId,
    strength: f32,
    output_path: Option<&Path>,
) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    // --- Load model ---
    eprintln!("Loading model...");
    let model = MIModel::from_pretrained(model_id)?;
    let n_layers = model.num_layers();
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!(
        "Model: {n_layers} layers, {} hidden, device={device:?}",
        model.hidden_size()
    );

    // --- Open CLT + cache steering vectors ---
    eprintln!("Opening CLT: {clt_repo_name}...");
    let mut clt = CrossLayerTranscoder::open(clt_repo_name)?;
    let mut all_features: Vec<CltFeatureId> = suppress_features.to_vec();
    all_features.push(inject_feature);
    eprintln!("Caching decoder vectors for all downstream layers...");
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    // --- Tokenize ---
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
    eprintln!("Tokens ({seq_len}): {token_strs:?}");

    let inject_token_id = tokenizer.find_token_id(inject_word)?;
    let inject_token_str = tokenizer.decode_token(inject_token_id)?;
    eprintln!("Inject token: \"{inject_token_str}\" (id={inject_token_id})");

    // --- Build feature entries for all downstream layers ---
    let suppress_entries: Vec<(CltFeatureId, usize)> = suppress_features
        .iter()
        .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
        .collect();
    let inject_entries: Vec<(CltFeatureId, usize)> = (inject_feature.layer..n_layers)
        .map(|l| (inject_feature, l))
        .collect();
    eprintln!(
        "Suppress: {} entries across {} features",
        suppress_entries.len(),
        suppress_features.len()
    );
    eprintln!(
        "Inject: {} entries (layers {}–{})",
        inject_entries.len(),
        inject_feature.layer,
        n_layers - 1
    );

    // --- Baseline (no intervention) ---
    eprintln!("\nRunning baseline...");
    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let result = model.forward(&input, &HookSpec::new())?;
    let baseline_prob = extract_token_prob(result.output(), inject_token_id)?;
    eprintln!("Baseline P(\"{inject_token_str}\") = {baseline_prob:.6e}");

    // --- Position sweep ---
    let positions = sweep_positions(
        &model,
        &clt,
        &input,
        seq_len,
        &token_strs,
        &suppress_entries,
        &inject_entries,
        strength,
        inject_token_id,
        baseline_prob,
        &device,
    )?;

    // --- Summary ---
    print_sweep_summary(&positions, baseline_prob, &token_strs);

    // --- JSON output ---
    let output = SweepOutput {
        model: model_id.into(),
        clt_repo: clt_repo_name.into(),
        prompt: prompt.into(),
        tokens: token_strs,
        suppress_word: suppress_word.into(),
        inject_word: inject_word.into(),
        suppress_features: suppress_features.to_vec(),
        inject_feature,
        strength,
        baseline_prob,
        sweep: positions,
    };
    write_sweep_output(&output, output_path)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

/// Run the position sweep and print progress.
#[allow(clippy::too_many_arguments)]
fn sweep_positions(
    model: &MIModel,
    clt: &CrossLayerTranscoder,
    input: &Tensor,
    seq_len: usize,
    token_strs: &[String],
    suppress_entries: &[(CltFeatureId, usize)],
    inject_entries: &[(CltFeatureId, usize)],
    strength: f32,
    inject_token_id: u32,
    baseline_prob: f32,
    device: &candle_core::Device,
) -> candle_mi::Result<Vec<PositionResult>> {
    eprintln!("\nSweeping {seq_len} positions (strength={strength})...");
    let mut positions: Vec<PositionResult> = Vec::with_capacity(seq_len);

    for pos in 0..seq_len {
        let mut combined =
            clt.prepare_hook_injection(suppress_entries, pos, seq_len, -strength, device)?;
        let inject_hooks =
            clt.prepare_hook_injection(inject_entries, pos, seq_len, strength, device)?;
        combined.extend(&inject_hooks);

        let result = model.forward(input, &combined)?;
        let p_inject = extract_token_prob(result.output(), inject_token_id)?;

        positions.push(PositionResult {
            position: pos,
            token: token_strs.get(pos).cloned().unwrap_or_default(),
            prob: p_inject,
        });

        let delta = p_inject - baseline_prob;
        let marker = if delta > baseline_prob * 10.0 && delta > 1e-12 {
            " ***"
        } else if delta > baseline_prob && delta > 1e-12 {
            " *"
        } else {
            ""
        };
        // BORROW: explicit .as_str() — String to &str for display
        let display = token_strs
            .get(pos)
            .map_or("?", String::as_str)
            .replace('\n', "\\n");
        eprintln!("  pos {pos:>3}  {display:<20}  P={p_inject:.6e}  delta={delta:+.6e}{marker}");
    }

    Ok(positions)
}

/// Print the sweep summary to stderr.
fn print_sweep_summary(positions: &[PositionResult], baseline_prob: f32, token_strs: &[String]) {
    let (max_pos, max_p) = positions
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.prob
                .partial_cmp(&b.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or((0, 0.0), |(i, p)| (i, p.prob));

    let ratio = if baseline_prob > 0.0 {
        max_p / baseline_prob
    } else {
        0.0
    };

    eprintln!("\n=== Results ===");
    eprintln!("Baseline:   {baseline_prob:.6e}");
    // BORROW: explicit .as_str() — String to &str for display
    eprintln!(
        "Max P:      {max_p:.6e} at position {max_pos} (\"{}\")  ratio={ratio:.1}x",
        token_strs
            .get(max_pos)
            .map_or("?", String::as_str)
            .replace('\n', "\\n")
    );
}

/// Serialize sweep results to JSON; write to file or stdout.
fn write_sweep_output(output: &SweepOutput, path: Option<&Path>) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;

    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
            })?;
        }
        fs::write(p, &json)
            .map_err(|e| candle_mi::MIError::Config(format!("write output: {e}")))?;
        eprintln!("\nOutput written to {}", p.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
