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
//! | `llama3.2-1b-524k` | Llama 3.2 1B | 524K | L5:19894 ("cat") | L14:13043 ("that") |
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
#![allow(clippy::too_many_lines)]

use std::fs;
use std::path::PathBuf;

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
/// Suppress "cat" (L5:19894), inject "that" (L14:13043).
/// From plip-rs validation: P("that") reaches 0.98 at planning site.
const LLAMA: Preset = Preset {
    model: "meta-llama/Llama-3.2-1B",
    clt_repo: "mntss/clt-llama-3.2-1b-524k",
    prompt: "A little mouse ran through the house,\n\
             And found some cheese behind the door.\n\
             She shared it with a friendly cat,\n\
             Who wore a tiny velvet",
    suppress_word: "cat",
    inject_word: "that",
    suppress_features: &[(5, 19894)],
    inject_feature: (14, 13043),
    strength: 15.0,
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
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(candle_mi::MIError::Config(format!(
            "feature must be in 'layer:index' format, got '{s}'"
        )));
    }
    let layer: usize = parts[0].parse().map_err(|e| {
        candle_mi::MIError::Config(format!("invalid layer number '{}': {e}", parts[0]))
    })?;
    let index: usize = parts[1].parse().map_err(|e| {
        candle_mi::MIError::Config(format!("invalid feature index '{}': {e}", parts[1]))
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
    let preset = match args.preset.as_str() {
        "llama3.2-1b-524k" => &LLAMA,
        "gemma2-2b-426k" => &GEMMA,
        "gemma2-2b-2.5m" => &GEMMA_2M,
        other => {
            return Err(candle_mi::MIError::Config(format!(
                "unknown preset '{other}' (expected 'llama3.2-1b-524k', 'gemma2-2b-426k', or 'gemma2-2b-2.5m')"
            )));
        }
    };

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
        preset.suppress_features.iter().copied().map(feature_id).collect()
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
    eprintln!(
        "Suppress: \"{}\" features {:?}",
        suppress_word, suppress_features
    );
    eprintln!("Inject:   \"{inject_word}\" feature {inject_feature}");
    eprintln!("Strength: {strength}\n");

    // --- Load model ---
    let t_start = std::time::Instant::now();
    eprintln!("Loading model...");
    let model = MIModel::from_pretrained(&model_id)?;
    let n_layers = model.num_layers();
    let device = model.device().clone();

    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;

    eprintln!(
        "Model: {} layers, {} hidden, device={:?}",
        n_layers,
        model.hidden_size(),
        device
    );

    // --- Open CLT ---
    eprintln!("Opening CLT: {clt_repo}...");
    let mut clt = CrossLayerTranscoder::open(&clt_repo)?;

    // --- Cache steering vectors for all features ---
    let mut all_features: Vec<CltFeatureId> = suppress_features.clone();
    all_features.push(inject_feature);
    eprintln!("Caching decoder vectors for all downstream layers...");
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    // --- Tokenize ---
    let prompt_with_space = format!("{prompt} ");
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();

    // Decode individual tokens for display.
    let token_strs: Vec<String> = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();

    eprintln!("Tokens ({seq_len}): {:?}", token_strs);

    // --- Find inject word token ID ---
    let inject_token_id = tokenizer.find_token_id(&inject_word)?;
    let inject_token_str = tokenizer.decode_token(inject_token_id)?;
    eprintln!(
        "Inject token: \"{}\" (id={})",
        inject_token_str, inject_token_id
    );

    // --- Build feature entries (feature, target_layer) for all downstream layers ---
    let suppress_entries: Vec<(CltFeatureId, usize)> = suppress_features
        .iter()
        .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
        .collect();
    let inject_entries: Vec<(CltFeatureId, usize)> = (inject_feature.layer..n_layers)
        .map(|l| (inject_feature, l))
        .collect();

    eprintln!("Suppress: {} entries across {} features", suppress_entries.len(), suppress_features.len());
    eprintln!(
        "Inject: {} entries (layers {}–{})",
        inject_entries.len(),
        inject_feature.layer,
        n_layers - 1
    );

    // --- Baseline (no intervention) ---
    eprintln!("\nRunning baseline...");
    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let hooks = HookSpec::new();
    let result = model.forward(&input, &hooks)?;
    let baseline_prob = extract_token_prob(result.output(), inject_token_id)?;
    eprintln!("Baseline P(\"{inject_token_str}\") = {baseline_prob:.6e}");

    // --- Position sweep ---
    eprintln!(
        "\nSweeping {} positions (strength={})...",
        seq_len, strength
    );
    let mut positions: Vec<PositionResult> = Vec::with_capacity(seq_len);

    for pos in 0..seq_len {
        // Build suppress hooks (negative strength).
        let mut combined =
            clt.prepare_hook_injection(&suppress_entries, pos, seq_len, -strength, &device)?;

        // Build inject hooks (positive strength) and merge.
        let inject_hooks =
            clt.prepare_hook_injection(&inject_entries, pos, seq_len, strength, &device)?;
        combined.extend(&inject_hooks);

        let result = model.forward(&input, &combined)?;
        let p_inject = extract_token_prob(result.output(), inject_token_id)?;

        positions.push(PositionResult {
            position: pos,
            token: token_strs.get(pos).cloned().unwrap_or_default(),
            prob: p_inject,
        });

        // Progress indicator.
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
        eprintln!(
            "  pos {:>3}  {:<20}  P={:.6e}  delta={:+.6e}{}",
            pos, display, p_inject, delta, marker
        );
    }

    // --- Summary ---
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

    // --- JSON output ---
    let output = SweepOutput {
        model: model_id,
        clt_repo,
        prompt,
        tokens: token_strs,
        suppress_word,
        inject_word,
        suppress_features,
        inject_feature,
        strength,
        baseline_prob,
        sweep: positions,
    };

    let json = serde_json::to_string_pretty(&output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;

    if let Some(ref p) = args.output {
        fs::write(p, &json)
            .map_err(|e| candle_mi::MIError::Config(format!("write output: {e}")))?;
        eprintln!("\nOutput written to {}", p.display());
    } else {
        println!("{json}");
    }

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}
