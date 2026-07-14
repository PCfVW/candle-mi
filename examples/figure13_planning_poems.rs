// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replication of Anthropic's "Planning in Poems" Figure 13: suppress + inject
//! position sweep.
//!
//! Suppresses natural rhyme group CLT features while injecting an alternative
//! feature, sweeping the injection position across the prompt to locate
//! planning sites.
//!
//! Eight built-in presets select model, CLT, prompt, features, and strength:
//!
//! | Preset | Model | CLT | Suppress | Inject |
//! |--------|-------|-----|----------|--------|
//! | `llama3.2-1b-524k` | Llama 3.2 1B | mntss 524K | -ee group: L13:30985 + L9:5488 + L14:27874 + L13:32049 | L14:13043 (`that`) |
//! | `gemma2-2b-426k` | Gemma 2 2B | mntss 426K | L16:13725 + L25:9385 (-out) | L22:10243 (`around`) |
//! | `gemma2-2b-2.5m` | Gemma 2 2B | mntss 2.5M | L25:57092 + L23:49923 + L20:77102 (-out) | L25:82839 (`can`) |
//! | `qwen3-1.7b-20k-ation` | `Qwen3-1.7B-Base` | `BlueLightAI` 20K | L15:263 + L18:3801 + L18:4404 (-ation) | L21:3908 (-self, `cos→" myself"` = 0.39) |
//! | `qwen3-1.7b-20k-teen`  | `Qwen3-1.7B-Base` | `BlueLightAI` 20K | L27:16975 + L20:3668 + L18:10986 (-teen) | L15:263 (-ation cluster-broad) |
//! | `qwen3-0.6b-20k-ation` | `Qwen3-0.6B-Base` | `BlueLightAI` 20K | L19:9578 + L0:8867 + L25:4979 (-ation) | L22:4081 (-self, `cos→" myself"` = 0.42) |
//! | `qwen3-0.6b-20k-teen`  | `Qwen3-0.6B-Base` | `BlueLightAI` 20K | L27:16425 + L23:15839 + L26:6308 (-teen) | L19:9578 (-ation cluster-broad) |
//! | `qwen3-0.6b-16k-ation` | `Qwen3-0.6B-Base` | `BlueLightAI`-dev 16K | L23:11154 + L20:10987 + L14:10719 (-ation) | L22:8011 (-self, `cos→" myself"` = 0.30) |
//!
//! The five `qwen3-*` presets are populated from `JumpReLU` `CltSplit`
//! vocabulary scans against `BlueLightAI`'s `Qwen3` `CLT`s
//! (`bluelightai/clt-qwen3-{1.7b,0.6b}-base-20k`,
//! `bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test`).  Per pairing, suppress
//! is the top-3 features (by `max_cosine`) of the prompt's natural rhyme
//! group.  Inject is one of two complementary picks: for `-ation` poems the
//! inject feature targets `" myself"`-cosine specifically (a narrow,
//! word-level inject works because the prompt has no natural `-self`
//! prior to displace); for `-teen` poems the inject feature is the
//! cluster-broad top-1 `EY1 SH AH0 N` feature (which empirically beat the
//! word-narrow `" duration"` pick by 3–14× because the suppress side
//! already clears the natural `-teen` prior).  Regenerate picks with
//! `examples/vocab_scan` + `scripts/vocab_scan_cmudict_filter.py` +
//! `scripts/pick_features.py` + `scripts/pick_inject_feature.py`.
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
//!
//! # Qwen3 1.7B Base, BlueLightAI 20K CLT, -ation suppress + -self inject
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-1.7b-20k-ation
//!
//! # Qwen3 0.6B Base, BlueLightAI-dev 16K CLT, -ation suppress + -self inject
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-0.6b-16k-ation
//! ```
//!
//! Outputs JSON suitable for direct import into Mathematica via
//! `Import["output.json"]`.

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::Tensor;
use clap::Parser;
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookSpec, MIModel, extract_token_prob};

// Shared Figure-13 cell presets (see module docs).  `#[path]`-included rather
// than declared as its own example: the file lives in a `main.rs`-free
// subdirectory of `examples/`, so Cargo's example auto-discovery skips it.
#[path = "figure13_common/presets.rs"]
mod presets;

use presets::{feature_id, parse_feature, select_preset};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "figure13_planning_poems")]
#[command(about = "Anthropic Figure 13 replication: suppress + inject position sweep")]
struct Args {
    /// Preset name: one of `llama3.2-1b-524k`, `gemma2-2b-426k`,
    /// `gemma2-2b-2.5m`, `qwen3-1.7b-20k-ation`, `qwen3-1.7b-20k-teen`,
    /// `qwen3-0.6b-20k-ation`, `qwen3-0.6b-20k-teen`, or
    /// `qwen3-0.6b-16k-ation`.
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

    /// Steering strength (overrides preset).  Ignored when `--strength-grid`
    /// is set.
    #[arg(long)]
    strength: Option<f32>,

    /// Strength grid (comma-separated `f32`s) for a 2D position × strength
    /// sweep, e.g. `--strength-grid 0.5,1,2.5,5,10,25,50,100`.  When set,
    /// `--strength` is ignored, the position sweep is rerun for each
    /// strength, and the output gains `sweep_grid` + `best_*` fields; the
    /// top-level `sweep` is populated from the best (strength, position)
    /// row in the grid (for `Mathematica` `Import` backward compat).
    #[arg(long, value_delimiter = ',')]
    strength_grid: Vec<f32>,

    /// Skip the suppress side of the intervention; inject the inject feature
    /// alone at each position.  Tests whether the redirect requires both
    /// halves (suppress + inject) or whether the inject decoder vector
    /// suffices as a pure additive steering direction.  Recorded in the
    /// output JSON as `no_suppress: true`.
    #[arg(long, default_value_t = false)]
    no_suppress: bool,

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
    /// `true` when `--no-suppress` was passed; suppression skipped, inject
    /// applied alone.
    #[serde(default)]
    no_suppress: bool,
    /// In single-strength mode this is the chosen strength; in `sweep_grid`
    /// mode it is the *best* strength (the one yielding the highest
    /// `prob / baseline_prob` across all positions).
    strength: f32,
    baseline_prob: f32,
    /// Per-position results at `strength` above (i.e. the best strength's
    /// row when `sweep_grid` is `Some`).  Kept at the top level for
    /// backwards-compatible `Mathematica` `Import` of legacy single-strength
    /// outputs.
    sweep: Vec<PositionResult>,
    /// Full 2D grid present only when `--strength-grid` was passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_grid: Option<Vec<StrengthRow>>,
    /// Top-position probability ratio (`prob / baseline_prob`) at the best
    /// (strength, position) cell.  Present only in grid mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    best_ratio: Option<f32>,
    /// Position index of the best cell in the grid.  Present only in grid
    /// mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    best_position: Option<usize>,
}

#[derive(Serialize, Clone)]
struct PositionResult {
    position: usize,
    token: String,
    prob: f32,
}

#[derive(Serialize, Clone)]
struct StrengthRow {
    strength: f32,
    sweep: Vec<PositionResult>,
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

    let suppress_features: Vec<CltFeatureId> = if args.no_suppress {
        // EXPLICIT: empty Vec disables the suppress half of the intervention.
        Vec::new()
    } else if args.suppress_feature.is_empty() {
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
    if args.no_suppress {
        eprintln!("Suppress: SKIPPED (--no-suppress; inject-only mode)");
    } else {
        eprintln!("Suppress: \"{suppress_word}\" features {suppress_features:?}");
    }
    eprintln!("Inject:   \"{inject_word}\" feature {inject_feature}");
    if args.strength_grid.is_empty() {
        eprintln!("Strength: {strength} (single-strength mode)\n");
    } else {
        eprintln!("Strength: grid {:?} (2D sweep mode)\n", args.strength_grid);
    }

    run_experiment(
        &model_id,
        &clt_repo,
        &prompt,
        &suppress_word,
        &inject_word,
        &suppress_features,
        inject_feature,
        strength,
        &args.strength_grid,
        args.no_suppress,
        args.output.as_deref(),
    )
}

/// Load model + CLT, run the position sweep, print summary, and write output.
///
/// `strength` is the single-strength choice; `strength_grid` selects 2D mode
/// when non-empty (and `strength` is ignored).
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
    strength_grid: &[f32],
    no_suppress: bool,
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

    // --- Position sweep (single-strength) or 2D grid (position × strength) ---
    let (final_strength, best_positions, sweep_grid_opt, best_meta) = if strength_grid.is_empty() {
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
        (strength, positions, None, None)
    } else {
        let mut rows: Vec<StrengthRow> = Vec::with_capacity(strength_grid.len());
        // (strength, position, ratio) of the best cell across the grid.
        let mut best: (f32, usize, f32) = (0.0, 0, 0.0);
        for &s in strength_grid {
            eprintln!("\n--- Strength {s} ---");
            let positions = sweep_positions(
                &model,
                &clt,
                &input,
                seq_len,
                &token_strs,
                &suppress_entries,
                &inject_entries,
                s,
                inject_token_id,
                baseline_prob,
                &device,
            )?;
            for p in &positions {
                let ratio = if baseline_prob > 0.0 {
                    p.prob / baseline_prob
                } else {
                    0.0
                };
                if ratio > best.2 {
                    best = (s, p.position, ratio);
                }
            }
            // BORROW: positions moves into the row; no clone.
            rows.push(StrengthRow {
                strength: s,
                sweep: positions,
            });
        }
        // EXPLICIT: linear search across at most ~10 rows; building an index
        // would obscure intent.
        // BORROW: r.sweep.clone() — `rows` is still needed below for the
        // `sweep_grid` field; we need the best row's positions duplicated at
        // the top level for Mathematica `Import` backward compat.
        let best_row_sweep = rows
            .iter()
            .find(|r| (r.strength - best.0).abs() < f32::EPSILON)
            .map_or_else(Vec::new, |r| r.sweep.clone());
        (best.0, best_row_sweep, Some(rows), Some((best.2, best.1)))
    };

    // --- Summary (best row in grid mode; the only row in single mode) ---
    if let Some((ratio, pos)) = best_meta {
        eprintln!(
            "\n=== Best cell across the strength grid: strength={final_strength}, \
             position={pos}, ratio={ratio:.2}x ===",
        );
    }
    print_sweep_summary(&best_positions, baseline_prob, &token_strs);

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
        no_suppress,
        strength: final_strength,
        baseline_prob,
        sweep: best_positions,
        sweep_grid: sweep_grid_opt,
        best_ratio: best_meta.map(|(r, _)| r),
        best_position: best_meta.map(|(_, p)| p),
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
