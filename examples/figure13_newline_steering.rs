// SPDX-License-Identifier: MIT OR Apache-2.0

//! Experiment 2 of the BlackboxNLP newline-experiments spec: **full-line
//! steering at the newline** (causal), the composition-horizon analogue of
//! Anthropic's Figure 13.
//!
//! Unlike `figure13_planning_poems` (which truncates the prompt so the rhyme
//! word is the *next token* — no line to compose), this example truncates the
//! prompt **after the newline that ends line 3**, so the model must **compose
//! line 4** and choose its final rhyme word many tokens downstream. Steering is
//! applied at the final prompt token — the line-3 newline — which is therefore
//! both "the newline" and "the last token" by construction, giving a
//! newline-time signal the best possible chance to shape the composed line.
//!
//! This is the geometry that makes the planning question meaningful: with a
//! line between the steering site and the rhyme, a null result is evidence that
//! the model is *below the planning floor* (improvises at emission) rather than
//! merely lacking a horizon to plan over.
//!
//! Metrics (spec §Exp 2):
//! - **m2**: the greedy composed line per condition (baseline / suppress-only /
//!   inject-only / suppress+inject), steering at the newline.
//! - **m1** (raw): `--k-samples` sampled lines per condition (temperature,
//!   fixed seed) — the final-word rime classification + Clopper-Pearson CIs are
//!   computed by a Python post-pass reusing the `CMUdict` machinery.
//! - **m3**: teacher-force the baseline greedy line up to its final-word slot,
//!   report `P(inject)` and `P(natural)` at that slot under each condition.
//! - **m4**: the true Figure-13 analogue — sweep the steering position over
//!   every token of the teacher-forced context and record m3 per position.
//!
//! candle-mi is KV-cache-free, so the steered key/value at the newline is
//! reproduced by re-applying the hook at the fixed newline position at every
//! generation step (`route: "recompute-per-step"` in the output).
//!
//! ```bash
//! cargo run --release --features clt,transformer,mmap --example figure13_newline_steering -- \
//!     --preset gemma2-2b-426k --strength 10 \
//!     --output docs/experiments/figure13-newline/fullline_gemma2-2b-426k.json
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookSpec, MIModel, MITokenizer, extract_token_prob};

#[path = "figure13_common/presets.rs"]
mod presets;

use presets::{feature_id, parse_feature, select_preset};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "figure13_newline_steering")]
#[command(
    about = "Experiment 2: full-line steering at the newline (composition-horizon Figure 13)"
)]
struct Args {
    /// Preset name (one of the Exp-2 cells; any figure13 preset works).
    #[arg(long, default_value = "gemma2-2b-426k")]
    preset: String,

    /// `HuggingFace` model ID (overrides preset).
    #[arg(long)]
    model: Option<String>,

    /// `HuggingFace` CLT repository (overrides preset).
    #[arg(long)]
    clt_repo: Option<String>,

    /// Truncated prompt override (default: the preset's first three lines plus
    /// the line-3 newline, so the model composes line 4).
    #[arg(long)]
    prompt: Option<String>,

    /// Natural rhyme word (overrides preset `suppress_word`).
    #[arg(long)]
    suppress_word: Option<String>,

    /// Alternative inject word (overrides preset `inject_word`).
    #[arg(long)]
    inject_word: Option<String>,

    /// Suppress features `layer:index`; repeatable (overrides preset).
    #[arg(long)]
    suppress_feature: Vec<String>,

    /// Inject feature `layer:index` (overrides preset).
    #[arg(long)]
    inject_feature: Option<String>,

    /// Steering strength.
    #[arg(long)]
    strength: Option<f32>,

    /// Max new tokens per generated line.
    #[arg(long, default_value_t = 15)]
    generate: usize,

    /// Sampled lines per condition for m1.
    #[arg(long, default_value_t = 20)]
    k_samples: usize,

    /// Sampling temperature for m1.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// RNG seed for m1 sampling (logged in the output).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output JSON path (defaults to stdout).
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Conditions ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Condition {
    name: &'static str,
    suppress: bool,
    inject: bool,
}

const CONDITIONS: [Condition; 4] = [
    Condition {
        name: "baseline",
        suppress: false,
        inject: false,
    },
    Condition {
        name: "suppress-only",
        suppress: true,
        inject: false,
    },
    Condition {
        name: "inject-only",
        suppress: false,
        inject: true,
    },
    Condition {
        name: "suppress+inject",
        suppress: true,
        inject: true,
    },
];

// ── Output types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SteeringOutput {
    model: String,
    clt_repo: String,
    preset: String,
    /// The truncated prompt (ends at the line-3 newline).
    prompt: String,
    tokens: Vec<String>,
    /// Index of the steering site (the final prompt token = line-3 newline).
    newline_index: usize,
    suppress_word: String,
    inject_word: String,
    suppress_features: Vec<CltFeatureId>,
    inject_feature: CltFeatureId,
    strength: f32,
    /// KV-cache route used; always `"recompute-per-step"` for candle-mi.
    route: &'static str,
    temperature: f32,
    seed: u64,
    /// Baseline greedy composed line (tokens decoded), for reference.
    baseline_greedy_line: String,
    /// Position in the full sequence that predicts the baseline line's final
    /// word (the m3 teacher-forcing slot).
    final_word_slot: usize,
    /// Per-condition m1/m2/m3 results.
    conditions: Vec<ConditionResult>,
    /// m4: per-position teacher-forced final-slot probabilities at `strength`.
    m4_position_sweep: Vec<PositionProb>,
}

#[derive(Serialize)]
struct ConditionResult {
    condition: String,
    /// m2: greedy composed line under this condition.
    greedy_line: String,
    /// m3: `P(inject)` at the baseline final-word slot under this condition.
    m3_p_inject: f32,
    /// m3: `P(natural)` at the baseline final-word slot under this condition.
    m3_p_natural: f32,
    /// m1 (raw): sampled composed lines (classified by the Python post-pass).
    sampled_lines: Vec<String>,
}

#[derive(Serialize)]
struct PositionProb {
    position: usize,
    token: String,
    p_inject: f32,
    p_natural: f32,
}

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

    // BORROW: .to_owned() — &'static str → owned String for storage.
    let model_id = args.model.unwrap_or_else(|| preset.model.to_owned());
    let clt_repo = args.clt_repo.unwrap_or_else(|| preset.clt_repo.to_owned());
    let suppress_word = args
        .suppress_word
        .unwrap_or_else(|| preset.suppress_word.to_owned());
    let inject_word = args
        .inject_word
        .unwrap_or_else(|| preset.inject_word.to_owned());
    let strength = args.strength.unwrap_or(preset.strength);
    let prompt = args
        .prompt
        .unwrap_or_else(|| truncate_after_line3(preset.prompt));

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

    eprintln!("=== Experiment 2: composition-horizon newline steering ===\n");
    eprintln!("Preset:   {}", args.preset);
    eprintln!("Model:    {model_id}");
    eprintln!("CLT:      {clt_repo}");
    eprintln!("Prompt (truncated after line-3 newline):\n{prompt}\n");
    eprintln!("Suppress: \"{suppress_word}\" {suppress_features:?}");
    eprintln!("Inject:   \"{inject_word}\" {inject_feature}");
    eprintln!("Strength: {strength}\n");

    run_experiment(&Config {
        preset_name: &args.preset,
        model_id: &model_id,
        clt_repo: &clt_repo,
        prompt: &prompt,
        suppress_word: &suppress_word,
        inject_word: &inject_word,
        suppress_features: &suppress_features,
        inject_feature,
        strength,
        max_new: args.generate,
        k_samples: args.k_samples,
        temperature: args.temperature,
        seed: args.seed,
        output: args.output.as_deref(),
    })
}

/// Immutable run configuration (keeps `run_experiment` under the argument cap).
struct Config<'a> {
    preset_name: &'a str,
    model_id: &'a str,
    clt_repo: &'a str,
    prompt: &'a str,
    suppress_word: &'a str,
    inject_word: &'a str,
    suppress_features: &'a [CltFeatureId],
    inject_feature: CltFeatureId,
    strength: f32,
    max_new: usize,
    k_samples: usize,
    temperature: f32,
    seed: u64,
    output: Option<&'a Path>,
}

fn run_experiment(cfg: &Config<'_>) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    let model = MIModel::from_pretrained(cfg.model_id)?;
    let device = model.device().clone();
    let n_layers = model.num_layers();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;

    let mut clt = CrossLayerTranscoder::open(cfg.clt_repo)?;
    let mut all_features: Vec<CltFeatureId> = cfg.suppress_features.to_vec();
    all_features.push(cfg.inject_feature);
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    // Downstream-layer injection entries (feature fires at all layers ≥ its own).
    let suppress_entries: Vec<(CltFeatureId, usize)> = cfg
        .suppress_features
        .iter()
        .flat_map(|f| (f.layer..n_layers).map(move |l| (*f, l)))
        .collect();
    let inject_entries: Vec<(CltFeatureId, usize)> = (cfg.inject_feature.layer..n_layers)
        .map(|l| (cfg.inject_feature, l))
        .collect();

    // Tokenize the truncated prompt; the steering site is the final token.
    let prompt_tokens = tokenizer.encode(cfg.prompt)?;
    let prompt_len = prompt_tokens.len();
    if prompt_len == 0 {
        return Err(candle_mi::MIError::Config("empty prompt".into()));
    }
    let newline_index = prompt_len - 1;
    let token_strs: Vec<String> = prompt_tokens
        .iter()
        .map(|&id| {
            tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();

    let inject_id = tokenizer.find_token_id(cfg.inject_word)?;
    let natural_id = tokenizer.find_token_id(cfg.suppress_word)?;

    eprintln!("Prompt tokens ({prompt_len}); steering site = newline @ {newline_index}");

    // --- Baseline greedy composed line + m3 teacher-forcing slot ---
    let baseline_gen = generate(
        &model,
        &clt,
        &prompt_tokens,
        newline_index,
        &[],
        &[],
        0.0,
        cfg.max_new,
        tokenizer,
        None,
        &device,
    )?;
    let baseline_line = tokenizer.decode(&baseline_gen).unwrap_or_default();
    let final_word_start = final_word_start_index(prompt_len, &baseline_gen, tokenizer);
    eprintln!("Baseline greedy line: {baseline_line:?}");
    eprintln!("Final-word slot (predicts the rhyme word): position {final_word_start}\n");

    // Teacher-forced context: prompt + baseline line up to (not incl.) final word.
    let mut context: Vec<u32> = prompt_tokens.clone();
    // INDEX-free: extend with the generated prefix before the final word.
    let prefix_len = final_word_start.saturating_sub(prompt_len);
    context.extend(baseline_gen.iter().take(prefix_len));

    // --- Per condition: m2 (greedy line), m3 (final-slot P), m1 (sampled) ---
    let mut condition_results: Vec<ConditionResult> = Vec::with_capacity(CONDITIONS.len());
    for cond in CONDITIONS {
        let sup: &[(CltFeatureId, usize)] = if cond.suppress {
            &suppress_entries
        } else {
            &[]
        };
        let inj: &[(CltFeatureId, usize)] = if cond.inject { &inject_entries } else { &[] };

        // m2: greedy composed line under this condition.
        let line_tokens = generate(
            &model,
            &clt,
            &prompt_tokens,
            newline_index,
            sup,
            inj,
            cfg.strength,
            cfg.max_new,
            tokenizer,
            None,
            &device,
        )?;
        let greedy_line = tokenizer.decode(&line_tokens).unwrap_or_default();

        // m3: teacher-forced final-slot probabilities under this condition.
        let hooks = build_steer_hooks(
            &clt,
            sup,
            inj,
            newline_index,
            context.len(),
            cfg.strength,
            &device,
        )?;
        let logits = model.forward(&input_tensor(&context, &device)?, &hooks)?;
        let m3_p_inject = extract_token_prob(logits.output(), inject_id)?;
        let m3_p_natural = extract_token_prob(logits.output(), natural_id)?;

        // m1: sampled composed lines (deterministic per (condition, seed)).
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut sampled_lines: Vec<String> = Vec::with_capacity(cfg.k_samples);
        for _ in 0..cfg.k_samples {
            let s = generate(
                &model,
                &clt,
                &prompt_tokens,
                newline_index,
                sup,
                inj,
                cfg.strength,
                cfg.max_new,
                tokenizer,
                Some((&mut rng, cfg.temperature)),
                &device,
            )?;
            sampled_lines.push(tokenizer.decode(&s).unwrap_or_default());
        }

        eprintln!(
            "  [{:<15}] greedy={:?}  P(inject)={m3_p_inject:.4e}  P(natural)={m3_p_natural:.4e}",
            cond.name, greedy_line
        );
        condition_results.push(ConditionResult {
            condition: cond.name.to_owned(),
            greedy_line,
            m3_p_inject,
            m3_p_natural,
            sampled_lines,
        });
    }

    // --- m4: sweep steering position over the teacher-forced context ---
    eprintln!(
        "\nm4 position sweep (suppress+inject at strength {}):",
        cfg.strength
    );
    let mut m4: Vec<PositionProb> = Vec::with_capacity(context.len());
    for pos in 0..context.len() {
        let hooks = build_steer_hooks(
            &clt,
            &suppress_entries,
            &inject_entries,
            pos,
            context.len(),
            cfg.strength,
            &device,
        )?;
        let logits = model.forward(&input_tensor(&context, &device)?, &hooks)?;
        let p_inject = extract_token_prob(logits.output(), inject_id)?;
        let p_natural = extract_token_prob(logits.output(), natural_id)?;
        let token = token_label(&token_strs, &baseline_gen, tokenizer, prompt_len, pos);
        m4.push(PositionProb {
            position: pos,
            token,
            p_inject,
            p_natural,
        });
    }
    if let Some(best) = m4.iter().max_by(|a, b| a.p_inject.total_cmp(&b.p_inject)) {
        eprintln!(
            "  best P(inject)={:.4e} at position {} (\"{}\")",
            best.p_inject,
            best.position,
            best.token.replace('\n', "\\n")
        );
    }

    let output = SteeringOutput {
        model: cfg.model_id.into(),
        clt_repo: cfg.clt_repo.into(),
        preset: cfg.preset_name.into(),
        prompt: cfg.prompt.into(),
        tokens: token_strs,
        newline_index,
        suppress_word: cfg.suppress_word.into(),
        inject_word: cfg.inject_word.into(),
        suppress_features: cfg.suppress_features.to_vec(),
        inject_feature: cfg.inject_feature,
        strength: cfg.strength,
        route: "recompute-per-step",
        temperature: cfg.temperature,
        seed: cfg.seed,
        baseline_greedy_line: baseline_line,
        final_word_slot: final_word_start,
        conditions: condition_results,
        m4_position_sweep: m4,
    };
    write_output(&output, cfg.output)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

// ── Generation + steering ─────────────────────────────────────────────────────

/// Build the combined suppress(−s) + inject(+s) hooks at `steer_pos` for the
/// current `seq_len`. Empty entries → no steering (baseline).
fn build_steer_hooks(
    clt: &CrossLayerTranscoder,
    suppress: &[(CltFeatureId, usize)],
    inject: &[(CltFeatureId, usize)],
    steer_pos: usize,
    seq_len: usize,
    strength: f32,
    device: &Device,
) -> candle_mi::Result<HookSpec> {
    let mut hooks = HookSpec::new();
    if !suppress.is_empty() {
        let h = clt.prepare_hook_injection(suppress, steer_pos, seq_len, -strength, device)?;
        hooks.extend(&h);
    }
    if !inject.is_empty() {
        let h = clt.prepare_hook_injection(inject, steer_pos, seq_len, strength, device)?;
        hooks.extend(&h);
    }
    Ok(hooks)
}

/// Generate a composed line, re-applying the newline steering at the fixed
/// `steer_pos` at every step (candle-mi is KV-cache-free). Stops at the first
/// generated newline or after `max_new` tokens. Returns the generated tokens
/// (excluding the prompt).
fn generate(
    model: &MIModel,
    clt: &CrossLayerTranscoder,
    prompt_tokens: &[u32],
    steer_pos: usize,
    suppress: &[(CltFeatureId, usize)],
    inject: &[(CltFeatureId, usize)],
    strength: f32,
    max_new: usize,
    tokenizer: &MITokenizer,
    mut sampler: Option<(&mut StdRng, f32)>,
    device: &Device,
) -> candle_mi::Result<Vec<u32>> {
    let mut current: Vec<u32> = prompt_tokens.to_vec();
    let start = current.len();
    for _ in 0..max_new {
        let seq_len = current.len();
        let hooks = build_steer_hooks(clt, suppress, inject, steer_pos, seq_len, strength, device)?;
        let cache = model.forward(&input_tensor(&current, device)?, &hooks)?;
        let logits = last_position_logits(cache.output())?;
        let next = match sampler.as_mut() {
            Some((rng, temp)) => sample_token(&logits, *temp, rng)?,
            None => argmax_token(&logits)?,
        };
        // Stop at the first generated newline (tokenizer-agnostic: Gemma emits a
        // bare "\n", Llama a merged ",\n", etc.). The newline token is not
        // included in the returned line.
        let is_newline = tokenizer.decode_token(next).is_ok_and(|s| s.contains('\n'));
        if is_newline {
            break;
        }
        current.push(next);
    }
    Ok(current.split_off(start))
}

fn input_tensor(tokens: &[u32], device: &Device) -> candle_mi::Result<Tensor> {
    Ok(Tensor::new(tokens, device)?.unsqueeze(0)?)
}

/// Last-position logits `[vocab]` from a `[1, seq, vocab]` tensor.
fn last_position_logits(logits_3d: &Tensor) -> candle_mi::Result<Tensor> {
    let seq = logits_3d.dim(1)?;
    Ok(logits_3d.get(0)?.get(seq - 1)?)
}

fn argmax_token(logits_1d: &Tensor) -> candle_mi::Result<u32> {
    // PROMOTE: force F32 so the argmax is dtype-stable across backends.
    let values: Vec<f32> = logits_1d.to_dtype(DType::F32)?.to_vec1()?;
    let mut best = 0_usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    // CAST: usize → u32, token ids fit in u32 for all tested vocabularies.
    u32::try_from(best)
        .map_err(|e| candle_mi::MIError::Config(format!("token id {best} exceeds u32: {e}")))
}

/// Temperature multinomial sample from 1-D logits.
fn sample_token(logits_1d: &Tensor, temperature: f32, rng: &mut StdRng) -> candle_mi::Result<u32> {
    // PROMOTE: softmax over logits needs F32 for numerical stability.
    let values: Vec<f32> = logits_1d.to_dtype(DType::F32)?.to_vec1()?;
    let temp = if temperature <= 0.0 { 1.0 } else { temperature };
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| ((v - max) / temp).exp()).collect();
    let sum: f32 = exps.iter().sum();
    // Draw u in [0, sum) and walk the cumulative distribution.
    let threshold = rng.r#gen::<f32>() * sum;
    let mut acc = 0.0_f32;
    for (i, &e) in exps.iter().enumerate() {
        acc += e;
        if acc >= threshold {
            // CAST: usize → u32, token ids fit in u32.
            return u32::try_from(i).map_err(|err| {
                candle_mi::MIError::Config(format!("token id {i} exceeds u32: {err}"))
            });
        }
    }
    // EXPLICIT: floating-point rounding can leave the walk just short of the
    // threshold; fall back to the last index.
    let last = exps.len().saturating_sub(1);
    u32::try_from(last)
        .map_err(|e| candle_mi::MIError::Config(format!("token id {last} exceeds u32: {e}")))
}

// ── Prompt + word-boundary helpers ────────────────────────────────────────────

/// Truncate a four-line preset prompt to its first three lines plus the line-3
/// newline, so the model must compose line 4.
fn truncate_after_line3(prompt: &str) -> String {
    let lines: Vec<&str> = prompt.split('\n').collect();
    if lines.len() < 3 {
        // EXPLICIT: fewer than 3 lines — return unchanged with a trailing
        // newline so a steering site still exists.
        return format!("{prompt}\n");
    }
    // BORROW: join the first three lines; append the line-3 newline.
    let head: Vec<&str> = lines.into_iter().take(3).collect();
    format!("{}\n", head.join("\n"))
}

/// Whether a decoded token begins a new word (leading space or metaspace).
fn is_word_start(token: &str) -> bool {
    token.starts_with(' ') || token.starts_with('\u{2581}') || token.starts_with('\u{0120}')
}

/// Index (in the full sequence) of the first token of the baseline line's final
/// word — the token the m3 slot predicts. Falls back to `prompt_len` (steer at
/// the newline predicting the first generated token) when no word start is
/// found.
fn final_word_start_index(prompt_len: usize, gen_tokens: &[u32], tokenizer: &MITokenizer) -> usize {
    let mut last_rel = 0_usize;
    for (i, &id) in gen_tokens.iter().enumerate() {
        let s = tokenizer.decode_token(id).unwrap_or_default();
        if i == 0 || is_word_start(&s) {
            last_rel = i;
        }
    }
    prompt_len + last_rel
}

/// Human label for a sweep position: prompt token text, or `gen[i]` for
/// positions inside the teacher-forced composed prefix.
fn token_label(
    prompt_tokens: &[String],
    gen_tokens: &[u32],
    tokenizer: &MITokenizer,
    prompt_len: usize,
    pos: usize,
) -> String {
    if pos < prompt_len {
        prompt_tokens.get(pos).cloned().unwrap_or_default()
    } else {
        gen_tokens
            .get(pos - prompt_len)
            .map(|&id| tokenizer.decode_token(id).unwrap_or_default())
            .unwrap_or_default()
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

fn write_output(output: &SteeringOutput, path: Option<&Path>) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to serialize steering JSON: {e}"))
    })?;
    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
            })?;
        }
        fs::write(p, &json).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to write steering output: {e}"))
        })?;
        eprintln!("\nOutput written to {}", p.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
