// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replication of Maar et al. (2026) "What's the plan?" contrastive activation
//! steering for rhyme planning.
//!
//! Maar's protocol:
//!
//! 1. For each of N positive prompts (first-line couplets whose natural
//!    completion is a target rhyme family) and N negative prompts (first
//!    lines from a contrasting family), capture the residual stream at a
//!    chosen layer at a chosen position.
//! 2. Compute `direction = mean(positive_residuals) − mean(negative_residuals)`,
//!    L2-normalise (Maar's documented `m = 1.5` magnitude implies unit
//!    direction).
//! 3. For each held-out evaluation prompt: greedy-sample the next token both
//!    without and with `Intervention::Add(strength × direction)` at
//!    `HookPoint::ResidPost(layer)`.  Record `P(target_word)`, the top-1
//!    token, and whether the top-1 is in the target rhyme family.
//! 4. Sweep `(layer, strength)` cells in a 2D grid; report the best cell
//!    plus the full grid.
//!
//! Four built-in presets, one per (model, rhyme-family) cell.
//!
//! | Preset | Model | Rhyme family | Role |
//! |--------|-------|--------------|------|
//! | `llama32-3b-rhyme-ee` | `meta-llama/Llama-3.2-3B` | `-ee` | **calibration** (Maar's own tested model) |
//! | `gemma2-2b-rhyme-ee` | `google/gemma-2-2b` | `-ee` | primary rebuttal cell |
//! | `gemma2-2b-rhyme-out` | `google/gemma-2-2b` | `-out` | primary rebuttal cell |
//! | `llama32-1b-rhyme-ee` | `meta-llama/Llama-3.2-1B` | `-ee` | new data (Maar didn't test 1B) |
//!
//! Each preset references a `prompts_file` JSON path that must exist; the
//! prompts JSONs ship in Commit 4 of the v0.1.12 release sequence.  Running
//! a preset before its prompts file exists produces a clear error pointing
//! at the file path.
//!
//! ```bash
//! # Calibration: Llama 3.2 3B at strength 1.5 (Maar's documented value),
//! # full layer sweep, "first newline" position strategy
//! cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
//!     --preset llama32-3b-rhyme-ee \
//!     --strength-grid 1.5 \
//!     --position-strategy first-newline \
//!     --output docs/experiments/maar-replication/llama32_3b_rhyme_ee_grid.json
//!
//! # Full 2D grid (default strengths -5,-2,-1,0.5,1,1.5,2,5,10)
//! cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
//!     --preset gemma2-2b-rhyme-ee \
//!     --output docs/experiments/maar-replication/gemma2_rhyme_ee_grid.json
//! ```
//!
//! Reference: Maar, Paperno, McDougall, Nanda. *What's the plan?* ICLR 2026
//! (poster). arXiv 2601.20164. OpenReview Z10pxu0Q7X.

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::as_conversions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_lossless)]

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::Tensor;
use clap::Parser;
use serde::{Deserialize, Serialize};

use candle_mi::{
    HookPoint, HookSpec, MIModel, PositionStrategy, build_contrastive_direction,
    contrastive_intervention, extract_token_prob,
};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "maar_contrastive_steering")]
#[command(
    about = "Maar et al. (2026) contrastive activation steering replication: 2D layer × strength sweep"
)]
struct Args {
    /// Preset name: one of `llama32-3b-rhyme-ee` (calibration),
    /// `gemma2-2b-rhyme-ee`, `gemma2-2b-rhyme-out`, `llama32-1b-rhyme-ee`.
    #[arg(long, default_value = "llama32-3b-rhyme-ee")]
    preset: String,

    /// `HuggingFace` model ID (overrides preset).
    #[arg(long)]
    model: Option<String>,

    /// Prompts file path (overrides preset).  Schema documented in
    /// `examples/results/maar_contrastive_steering/README.md`.
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    /// Layer grid: comma-separated explicit layer indices (e.g.
    /// `--layer-grid 8,12,16,20`).  When unset, sweeps every layer
    /// `0..n_layers`.
    #[arg(long, value_delimiter = ',')]
    layer_grid: Vec<usize>,

    /// Strength grid: comma-separated signed `f32`s.  Default
    /// `-5,-2,-1,0.5,1,1.5,2,5,10` covers Maar's `m = 1.5` plus a sign
    /// sweep (catches inverted-label bugs).
    #[arg(long, value_delimiter = ',', default_values_t = vec![-5.0_f32, -2.0, -1.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0])]
    strength_grid: Vec<f32>,

    /// Position strategy: `last` | `first-newline` | `explicit:N`.
    /// Default `first-newline` matches Maar's `Gemma 2 9B` documented choice.
    #[arg(long, default_value = "first-newline")]
    position_strategy: String,

    /// L2-normalise the contrastive direction to a unit vector (default
    /// `true`).  Pass `--no-normalise` to use the raw `mean(pos) − mean(neg)`
    /// difference without normalisation.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    normalise: bool,

    /// Output JSON path (required for committed grid runs).
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Preset table ────────────────────────────────────────────────────────────

struct Preset {
    /// HuggingFace model id.
    model: &'static str,
    /// Rhyme-family tag (e.g. "-ee", "-out") — appears in the output JSON
    /// for filtering / cross-cell aggregation.
    rhyme_family: &'static str,
    /// Path to the prompts JSON file (relative to the repo root).
    prompts_file: &'static str,
}

const LLAMA32_3B_RHYME_EE: Preset = Preset {
    model: "meta-llama/Llama-3.2-3B",
    rhyme_family: "-ee",
    prompts_file: "examples/results/maar_contrastive_steering/prompts/llama32_3b_rhyme_ee.json",
};

const GEMMA2_2B_RHYME_EE: Preset = Preset {
    model: "google/gemma-2-2b",
    rhyme_family: "-ee",
    prompts_file: "examples/results/maar_contrastive_steering/prompts/gemma2_rhyme_ee.json",
};

const GEMMA2_2B_RHYME_OUT: Preset = Preset {
    model: "google/gemma-2-2b",
    rhyme_family: "-out",
    prompts_file: "examples/results/maar_contrastive_steering/prompts/gemma2_rhyme_out.json",
};

const LLAMA32_1B_RHYME_EE: Preset = Preset {
    model: "meta-llama/Llama-3.2-1B",
    rhyme_family: "-ee",
    prompts_file: "examples/results/maar_contrastive_steering/prompts/llama32_1b_rhyme_ee.json",
};

fn select_preset(name: &str) -> candle_mi::Result<&'static Preset> {
    match name {
        "llama32-3b-rhyme-ee" => Ok(&LLAMA32_3B_RHYME_EE),
        "gemma2-2b-rhyme-ee" => Ok(&GEMMA2_2B_RHYME_EE),
        "gemma2-2b-rhyme-out" => Ok(&GEMMA2_2B_RHYME_OUT),
        "llama32-1b-rhyme-ee" => Ok(&LLAMA32_1B_RHYME_EE),
        other => Err(candle_mi::MIError::Config(format!(
            "unknown preset '{other}' (expected one of: 'llama32-3b-rhyme-ee', \
             'gemma2-2b-rhyme-ee', 'gemma2-2b-rhyme-out', 'llama32-1b-rhyme-ee')"
        ))),
    }
}

// ── Prompts JSON schema ─────────────────────────────────────────────────────

// EXPLICIT: `template` field is parsed for schema documentation / future use
// (the example does not currently substitute the template, since the prompt
// strings come pre-substituted in `positive` / `negative` / `eval`).
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct PromptsFile {
    family: String,
    template: String,
    positive: Vec<String>,
    negative: Vec<String>,
    eval: Vec<EvalPrompt>,
    source: String,
    #[serde(default)]
    source_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EvalPrompt {
    prompt: String,
    target_token: String,
    target_rhyme_words: Vec<String>,
}

// ── Output JSON schema ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct MaarOutput {
    model: String,
    preset: String,
    rhyme_family: String,
    n_layers: usize,
    hidden_size: usize,
    n_positive_prompts: usize,
    n_negative_prompts: usize,
    n_eval_prompts: usize,
    position_strategy: String,
    normalise: bool,
    prompts_source: String,
    prompts_source_url: Option<String>,
    baseline: BaselineSummary,
    grid: Vec<CellResult>,
    best_cell: Option<BestCell>,
    elapsed_seconds: f64,
}

#[derive(Serialize)]
struct BaselineSummary {
    mean_p_target: f32,
    hit_rate: f32,
    per_prompt: Vec<EvalResult>,
}

#[derive(Serialize)]
struct CellResult {
    layer: usize,
    strength: f32,
    mean_p_target: f32,
    hit_rate: f32,
    direction_norm: f32,
    per_prompt: Vec<EvalResult>,
}

#[derive(Serialize, Clone)]
struct EvalResult {
    prompt: String,
    target_token: String,
    p_target: f32,
    top1_token_id: u32,
    top1_token_text: String,
    is_hit: bool,
}

#[derive(Serialize)]
struct BestCell {
    layer: usize,
    strength: f32,
    mean_p_target: f32,
    hit_rate: f32,
    /// Ratio `hit_rate / max(baseline_hit_rate, 1e-9)`.  Reported separately
    /// because near-zero baselines are common on smaller models, and the
    /// ratio is what the calibration criterion looks at.
    hit_rate_ratio: f32,
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
    let preset = select_preset(&args.preset)?;

    // BORROW: .to_owned() — &'static str → String for owned model id.
    let model_id = args
        .model
        .clone()
        .unwrap_or_else(|| preset.model.to_owned());

    // Resolve prompts file path: CLI override > preset default.
    let prompts_path: PathBuf = args
        .prompt_file
        .clone()
        .unwrap_or_else(|| PathBuf::from(preset.prompts_file));

    let prompts = load_prompts(&prompts_path)?;
    if prompts.family != preset.rhyme_family {
        eprintln!(
            "Warning: prompts file family '{}' does not match preset family '{}'; \
             continuing anyway.",
            prompts.family, preset.rhyme_family
        );
    }

    let position_strategy = parse_position_strategy(&args.position_strategy)?;

    eprintln!("=== Maar contrastive activation steering ===");
    eprintln!("Preset:           {}", args.preset);
    eprintln!("Model:            {model_id}");
    eprintln!("Rhyme family:     {}", prompts.family);
    eprintln!("Prompts source:   {}", prompts.source);
    eprintln!(
        "Prompts:          {} positive + {} negative + {} eval",
        prompts.positive.len(),
        prompts.negative.len(),
        prompts.eval.len()
    );
    eprintln!("Position:         {position_strategy:?}");
    eprintln!("Normalise:        {}", args.normalise);
    eprintln!("Strength grid:    {:?}", args.strength_grid);
    if args.layer_grid.is_empty() {
        eprintln!("Layer grid:       (all layers)");
    } else {
        eprintln!("Layer grid:       {:?}", args.layer_grid);
    }
    eprintln!();

    let t_start = std::time::Instant::now();
    let model = MIModel::from_pretrained(&model_id)?;
    let n_layers = model.num_layers();
    let hidden = model.hidden_size();
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!("Loaded: {n_layers} layers, hidden={hidden}, device={device:?}");

    // BORROW: collect &str views once for the library API (which takes &[&str]).
    let positive_refs: Vec<&str> = prompts.positive.iter().map(String::as_str).collect();
    let negative_refs: Vec<&str> = prompts.negative.iter().map(String::as_str).collect();

    // Resolve target token IDs for each eval prompt.
    let eval_token_ids = prompts
        .eval
        .iter()
        .map(|e| {
            tokenizer
                .find_token_id(&e.target_token)
                .map(|id| (id, e.target_token.clone()))
        })
        .collect::<candle_mi::Result<Vec<_>>>()?;
    let eval_rhyme_token_ids: Vec<Vec<u32>> = prompts
        .eval
        .iter()
        .map(|e| {
            e.target_rhyme_words
                .iter()
                .map(|w| tokenizer.find_token_id(w))
                .collect::<candle_mi::Result<Vec<_>>>()
        })
        .collect::<candle_mi::Result<Vec<_>>>()?;

    // Baseline pass (no intervention).
    eprintln!("\nRunning baseline (no intervention)...");
    let baseline = run_eval_pass(
        &model,
        tokenizer,
        &prompts.eval,
        &eval_token_ids,
        &eval_rhyme_token_ids,
        None,
    )?;
    eprintln!(
        "Baseline: mean_p_target = {:.4e}, hit_rate = {:.2}%",
        baseline.0,
        baseline.1 * 100.0_f32
    );

    let baseline_summary = BaselineSummary {
        mean_p_target: baseline.0,
        hit_rate: baseline.1,
        per_prompt: baseline.2,
    };

    let layers: Vec<usize> = if args.layer_grid.is_empty() {
        (0..n_layers).collect()
    } else {
        args.layer_grid.clone()
    };

    let mut grid: Vec<CellResult> = Vec::with_capacity(layers.len() * args.strength_grid.len());
    let mut best: Option<(usize, f32, f32, f32)> = None;
    // EXPLICIT: outer loop is `layer` so we build the direction once per
    // layer and reuse it across all strength rows — avoids re-running the
    // direction-building forward passes for each strength.
    for &layer in &layers {
        eprintln!("\n--- Layer {layer} ---");
        let direction = build_contrastive_direction(
            &model,
            tokenizer,
            &positive_refs,
            &negative_refs,
            layer,
            position_strategy,
            args.normalise,
        )?;
        let norm_sq = (&direction.vector * &direction.vector)?.sum_all()?;
        // PROMOTE: scalar extraction needs a known dtype; vector is F32.
        let direction_norm = norm_sq
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?
            .sqrt();

        for &strength in &args.strength_grid {
            let intervention = contrastive_intervention(&direction, strength)?;
            let hook = HookPoint::ResidPost(layer);
            let mut hooks = HookSpec::new();
            hooks.intervene(hook, intervention);

            let cell = run_eval_pass(
                &model,
                tokenizer,
                &prompts.eval,
                &eval_token_ids,
                &eval_rhyme_token_ids,
                Some(&hooks),
            )?;

            eprintln!(
                "  s={strength:>5.1}  mean_p_target={:.4e}  hit_rate={:.2}%  dir_norm={:.4}",
                cell.0,
                cell.1 * 100.0_f32,
                direction_norm
            );

            grid.push(CellResult {
                layer,
                strength,
                mean_p_target: cell.0,
                hit_rate: cell.1,
                direction_norm,
                per_prompt: cell.2,
            });

            // Update best by hit_rate (primary metric per Maar); break ties
            // by mean_p_target (secondary).
            let candidate = (layer, strength, cell.0, cell.1);
            best = Some(match best {
                None => candidate,
                Some(prev) if cell.1 > prev.3 => candidate,
                Some(prev) if (cell.1 - prev.3).abs() < f32::EPSILON && cell.0 > prev.2 => {
                    candidate
                }
                Some(prev) => prev,
            });
        }
    }

    let best_cell = best.map(|(layer, strength, mean_p_target, hit_rate)| {
        let hit_rate_ratio = if baseline_summary.hit_rate > 1e-9_f32 {
            hit_rate / baseline_summary.hit_rate
        } else {
            // EXPLICIT: baseline hit_rate is effectively zero; report the
            // ratio as `hit_rate / 1e-9` (a large number) so the caller can
            // see absolute improvement vs no-baseline.
            hit_rate / 1e-9_f32
        };
        BestCell {
            layer,
            strength,
            mean_p_target,
            hit_rate,
            hit_rate_ratio,
        }
    });

    let elapsed = t_start.elapsed().as_secs_f64();

    let output = MaarOutput {
        model: model_id,
        preset: args.preset.clone(),
        rhyme_family: prompts.family.clone(),
        n_layers,
        hidden_size: hidden,
        n_positive_prompts: prompts.positive.len(),
        n_negative_prompts: prompts.negative.len(),
        n_eval_prompts: prompts.eval.len(),
        position_strategy: format!("{position_strategy:?}"),
        normalise: args.normalise,
        prompts_source: prompts.source.clone(),
        prompts_source_url: prompts.source_url.clone(),
        baseline: baseline_summary,
        grid,
        best_cell,
        elapsed_seconds: elapsed,
    };

    eprintln!("\nTotal elapsed: {elapsed:.2}s");

    write_output(&output, args.output.as_deref())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn load_prompts(path: &Path) -> candle_mi::Result<PromptsFile> {
    let body = fs::read_to_string(path).map_err(|e| {
        candle_mi::MIError::Config(format!(
            "load_prompts: failed to read {}: {e}",
            path.display()
        ))
    })?;
    let parsed: PromptsFile = serde_json::from_str(&body).map_err(|e| {
        candle_mi::MIError::Config(format!(
            "load_prompts: failed to parse {}: {e}",
            path.display()
        ))
    })?;
    if parsed.positive.is_empty() {
        return Err(candle_mi::MIError::Config(format!(
            "load_prompts: {} has empty positive set",
            path.display()
        )));
    }
    if parsed.negative.is_empty() {
        return Err(candle_mi::MIError::Config(format!(
            "load_prompts: {} has empty negative set",
            path.display()
        )));
    }
    if parsed.eval.is_empty() {
        return Err(candle_mi::MIError::Config(format!(
            "load_prompts: {} has empty eval set",
            path.display()
        )));
    }
    Ok(parsed)
}

fn parse_position_strategy(s: &str) -> candle_mi::Result<PositionStrategy> {
    if s == "last" {
        return Ok(PositionStrategy::Last);
    }
    if s == "first-newline" {
        return Ok(PositionStrategy::FirstNewline);
    }
    if let Some(n_str) = s.strip_prefix("explicit:") {
        let n: usize = n_str.parse().map_err(|e| {
            candle_mi::MIError::Config(format!(
                "parse_position_strategy: invalid Explicit index '{n_str}': {e}"
            ))
        })?;
        return Ok(PositionStrategy::Explicit(n));
    }
    Err(candle_mi::MIError::Config(format!(
        "parse_position_strategy: unknown strategy '{s}' \
         (expected 'last', 'first-newline', or 'explicit:N')"
    )))
}

/// Run the model on each eval prompt (optionally with hooks), measure
/// `P(target_token)` and the top-1 token, return `(mean_p_target, hit_rate,
/// per_prompt_results)`.
fn run_eval_pass(
    model: &MIModel,
    tokenizer: &candle_mi::MITokenizer,
    eval_prompts: &[EvalPrompt],
    eval_token_ids: &[(u32, String)],
    eval_rhyme_token_ids: &[Vec<u32>],
    hooks: Option<&HookSpec>,
) -> candle_mi::Result<(f32, f32, Vec<EvalResult>)> {
    let mut per_prompt: Vec<EvalResult> = Vec::with_capacity(eval_prompts.len());
    let mut sum_p_target = 0.0_f32;
    let mut hits: usize = 0;
    let empty_hooks = HookSpec::new();

    for (i, eval) in eval_prompts.iter().enumerate() {
        let tokens = tokenizer.encode(&eval.prompt)?;
        if tokens.is_empty() {
            return Err(candle_mi::MIError::Config(format!(
                "run_eval_pass: eval prompt #{i} encoded to zero tokens"
            )));
        }
        let input = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
        let cache = model.forward(&input, hooks.unwrap_or(&empty_hooks))?;
        let logits = cache.output();

        // INDEX: eval_token_ids has the same length as eval_prompts (built
        // together at the start of run()); i is in 0..eval_prompts.len().
        let (target_id, target_text) = &eval_token_ids[i];
        let p_target = extract_token_prob(logits, *target_id)?;
        sum_p_target += p_target;

        // Argmax over the last-position logits to identify top-1.
        let last_logits = last_position_logits(logits)?;
        let top1_id = argmax_token(&last_logits)?;
        let top1_text = tokenizer.decode_token(top1_id).unwrap_or_default();

        // INDEX: eval_rhyme_token_ids[i] is the per-prompt rhyme-word list.
        let is_hit = eval_rhyme_token_ids[i].contains(&top1_id);
        if is_hit {
            hits += 1;
        }

        per_prompt.push(EvalResult {
            prompt: eval.prompt.clone(),
            target_token: target_text.clone(),
            p_target,
            top1_token_id: top1_id,
            top1_token_text: top1_text,
            is_hit,
        });
    }

    // CAST: usize → f32, lossless for small N.
    let n = eval_prompts.len() as f32;
    let mean_p = if n > 0.0 { sum_p_target / n } else { 0.0 };
    let hit_rate = if n > 0.0 { hits as f32 / n } else { 0.0 };
    Ok((mean_p, hit_rate, per_prompt))
}

/// Slice the last-position logits from a `[1, seq, vocab]` tensor → `[vocab]`.
fn last_position_logits(logits_3d: &Tensor) -> candle_mi::Result<Tensor> {
    let dims = logits_3d.dims();
    if dims.len() != 3 {
        return Err(candle_mi::MIError::Model(candle_core::Error::Msg(format!(
            "last_position_logits: expected 3-D [1, seq, vocab]; got {dims:?}"
        ))));
    }
    let seq = dims[1];
    if seq == 0 {
        return Err(candle_mi::MIError::Model(candle_core::Error::Msg(
            "last_position_logits: seq == 0".into(),
        )));
    }
    let last = logits_3d.get(0)?.get(seq - 1)?;
    Ok(last)
}

/// Argmax over a 1-D logits tensor → token id.  Promotes to `F32` so the
/// argmax is dtype-stable across backends.
fn argmax_token(logits_1d: &Tensor) -> candle_mi::Result<u32> {
    // PROMOTE: argmax may have backend-specific dtype handling; force F32.
    let f32_logits = logits_1d.to_dtype(candle_core::DType::F32)?;
    let values: Vec<f32> = f32_logits.to_vec1()?;
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    // CAST: usize → u32, token IDs fit in 32 bits for all tested vocabularies
    // (max ~256K for Gemma); the upstream extract_token_prob already uses u32.
    u32::try_from(best_idx).map_err(|e| {
        candle_mi::MIError::Config(format!(
            "argmax_token: token id {best_idx} exceeds u32: {e}"
        ))
    })
}

fn write_output(output: &MaarOutput, path: Option<&Path>) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output).map_err(|e| {
        candle_mi::MIError::Config(format!("write_output: JSON serialization failed: {e}"))
    })?;
    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                candle_mi::MIError::Config(format!(
                    "write_output: failed to create {}: {e}",
                    parent.display()
                ))
            })?;
        }
        fs::write(p, &json)
            .map_err(|e| candle_mi::MIError::Config(format!("write_output: write: {e}")))?;
        eprintln!("Output written to {}", p.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
