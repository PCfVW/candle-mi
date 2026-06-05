// SPDX-License-Identifier: MIT OR Apache-2.0

//! Means-ends prolepsis — baseline feasibility (Step A) for the *linguistic*
//! transposition of the gridworld experiment.
//!
//! The gridworld pilot showed Gemma 2 2B base is at chance on coordinate-based
//! action selection (spatial modality). This cell moves the planning task into
//! the linguistic modality where small models are competent: goal-directed
//! means-ends action selection (the STRIPS operator-selection primitive).
//!
//! Items come from `scripts/means_ends_generator.py`: each states a current
//! state and a goal, then ends at the planning site (the content token before
//! the action, **no trailing space**). The model's next token should be the
//! goal-correct single-token action.
//!
//! The set is **goal-contrastive**: every device appears in both directions
//! (e.g. want-bright → `on`, want-dark → `off`), so a pass requires the model
//! to respond to the *goal*, not merely emit the high-frequency collocation
//! ("turn the lamp on"). The override sides (`off` / `closed` / `down`) are the
//! discriminating cases — see the per-(family, token) breakdown.
//!
//! No CLT, no intervention — pure forward passes, like the gridworld Step A.
//!
//! ```bash
//! python scripts/means_ends_generator.py
//! cargo run --features transformer,mmap --release --example means_ends_prolepsis
//! cargo run --features transformer,mmap --release --example means_ends_prolepsis -- \
//!     --model meta-llama/Llama-3.2-1B \
//!     --output docs/experiments/means-ends-prolepsis/baseline_llama32_1b.json
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Tensor};
use clap::Parser;
use serde::{Deserialize, Serialize};

use candle_mi::{HookCache, HookSpec, MIModel};

/// Per-metric pass threshold (Step A gate): both accuracies must reach this.
const THRESHOLD: f64 = 0.80;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "means_ends_prolepsis")]
#[command(about = "Means-ends prolepsis baseline feasibility (linguistic planning cell)")]
struct Args {
    /// `HuggingFace` model ID (base model, since the CLT is base-only).
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Path to the means-ends items JSON (output of `means_ends_generator.py`).
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/means_ends_items.json"
    )]
    items: PathBuf,

    /// Output JSON path for the per-item results.
    #[arg(
        long,
        default_value = "docs/experiments/means-ends-prolepsis/baseline_gemma2_2b.json"
    )]
    output: PathBuf,

    /// Number of per-item lines to echo to stderr during the run.
    #[arg(long, default_value_t = 20)]
    max_print: usize,
}

// ── Dataset ───────────────────────────────────────────────────────────────────

/// One goal-contrastive means-ends item, deserialized from the generator JSON.
/// `prompt` ends at the planning site (no trailing space); the model's next
/// token should be `correct`.
#[derive(Deserialize)]
struct Item {
    /// Full prompt, ending at the planning site.
    prompt: String,
    /// Goal-correct single-token action.
    correct: String,
    /// Contrastive (goal-incorrect) action for the same device/family.
    alternative: String,
    /// Action family (`on_off` / `open_closed` / `up_down`).
    family: String,
}

// ── JSON output ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct Output {
    model: String,
    items_path: String,
    threshold: f64,
    n_items: usize,
    full_vocab_top1_accuracy: f64,
    forced_choice_accuracy: f64,
    mean_p_correct: f64,
    passed: bool,
    /// Accuracy breakdown per action family.
    per_family: Vec<GroupStats>,
    /// Accuracy breakdown per (family, correct-token) — the anti-collapse view.
    per_token: Vec<GroupStats>,
    items: Vec<ItemResult>,
    elapsed_secs: f64,
}

#[derive(Serialize)]
struct GroupStats {
    /// Group label (a family, or `family/token`).
    key: String,
    /// Number of items in the group.
    n: usize,
    /// Full-vocabulary top-1 accuracy within the group.
    full_vocab_top1_accuracy: f64,
    /// Forced-choice accuracy within the group.
    forced_choice_accuracy: f64,
    /// Mean probability on the correct token within the group.
    mean_p_correct: f64,
}

#[derive(Serialize)]
struct ItemResult {
    prompt: String,
    family: String,
    correct: String,
    correct_token_id: u32,
    alternative: String,
    alternative_token_id: u32,
    top1_token: String,
    top1_token_trimmed: String,
    top1_prob: f32,
    full_vocab_top1_correct: bool,
    forced_choice_correct: bool,
    p_correct: f32,
    p_alternative: f32,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Load + parse the means-ends items JSON.
fn load_items(path: &Path) -> candle_mi::Result<Vec<Item>> {
    let json = fs::read_to_string(path).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
    })?;
    serde_json::from_str(&json)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to parse {}: {e}", path.display())))
}

/// Probability of `token_id` within a softmax probability vector.
fn prob_of(probs: &[f32], token_id: u32) -> candle_mi::Result<f32> {
    let idx = usize::try_from(token_id).map_err(|e| {
        candle_mi::MIError::Config(format!("token id {token_id} exceeds usize: {e}"))
    })?;
    probs.get(idx).copied().ok_or_else(|| {
        candle_mi::MIError::Config(format!(
            "token id {token_id} out of vocab range (len {})",
            probs.len()
        ))
    })
}

/// Index and value of the maximum probability (full-vocabulary argmax).
fn argmax_prob(probs: &[f32]) -> candle_mi::Result<(usize, f32)> {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, &p)| (i, p))
        .ok_or_else(|| candle_mi::MIError::Config("empty probability vector".into()))
}

/// Lossless count → `f64` (item counts are small, well within `u32`).
fn count_to_f64(count: usize) -> candle_mi::Result<f64> {
    let as_u32 = u32::try_from(count)
        .map_err(|e| candle_mi::MIError::Config(format!("count {count} exceeds u32: {e}")))?;
    Ok(f64::from(as_u32))
}

/// Softmax of the output-position logits, returned as a dense `Vec<f32>`.
///
/// # Shapes
/// - `cache.output()`: `[batch, seq, vocab]`
/// - returns: `[vocab]` as a host vector
fn output_probs(cache: &HookCache, output_pos: usize) -> candle_mi::Result<Vec<f32>> {
    let logits = cache
        .output()
        .get(0)?
        .narrow(0, output_pos, 1)?
        .squeeze(0)?
        // PROMOTE: softmax over logits requires F32 (model output may be BF16)
        .to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits.unsqueeze(0)?)?.squeeze(0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    Ok(probs_vec)
}

/// Accuracy breakdown for one group of items.
fn group_stats(key: String, subset: &[&ItemResult]) -> candle_mi::Result<GroupStats> {
    let n = subset.len();
    let (full, forced, mean_p) = if n == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let f = subset.iter().filter(|r| r.full_vocab_top1_correct).count();
        let fc = subset.iter().filter(|r| r.forced_choice_correct).count();
        let sum_p: f32 = subset.iter().map(|r| r.p_correct).sum();
        let n_f64 = count_to_f64(n)?;
        (
            count_to_f64(f)? / n_f64,
            count_to_f64(fc)? / n_f64,
            f64::from(sum_p) / n_f64,
        )
    };
    Ok(GroupStats {
        key,
        n,
        full_vocab_top1_accuracy: full,
        forced_choice_accuracy: forced,
        mean_p_correct: mean_p,
    })
}

/// Group results by a key function and compute per-group stats (sorted by key).
fn grouped(
    results: &[ItemResult],
    key_of: impl Fn(&ItemResult) -> String,
) -> candle_mi::Result<Vec<GroupStats>> {
    let mut by_key: BTreeMap<String, Vec<&ItemResult>> = BTreeMap::new();
    for r in results {
        by_key.entry(key_of(r)).or_default().push(r);
    }
    by_key
        .into_iter()
        .map(|(k, v)| group_stats(k, &v))
        .collect()
}

/// Validate that every distinct action token is a single token; return the id
/// map. Fails (listing offenders) if any is multi-token.
fn check_action_tokens(
    tokenizer: &candle_mi::MITokenizer,
    items: &[Item],
) -> candle_mi::Result<HashMap<String, u32>> {
    let mut tokens: BTreeSet<&str> = BTreeSet::new();
    for it in items {
        tokens.insert(it.correct.as_str()); // BORROW: .as_str() — String → &str for the set
        tokens.insert(it.alternative.as_str());
    }
    eprintln!("Action token check (single-token requirement):");
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut bad: Vec<String> = Vec::new();
    for &tok in &tokens {
        if let Ok(id) = tokenizer.find_token_id(tok) {
            eprintln!("  \"{tok}\" -> id {id}");
            ids.insert(tok.to_owned(), id);
        } else {
            eprintln!("  \"{tok}\" -> NOT a single token  FAIL");
            bad.push(tok.to_owned());
        }
    }
    if bad.is_empty() {
        Ok(ids)
    } else {
        Err(candle_mi::MIError::Tokenizer(format!(
            "multi-token action tokens (fix the generator): {}",
            bad.join(", ")
        )))
    }
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

    eprintln!("=== Means-Ends Prolepsis — baseline feasibility ===\n");
    let items = load_items(&args.items)?;
    if items.is_empty() {
        return Err(candle_mi::MIError::Config(format!(
            "no items in {}",
            args.items.display()
        )));
    }
    let n_items = items.len();
    eprintln!("Loaded {n_items} items from {}", args.items.display());

    eprintln!("Loading model {}...", args.model);
    let model = MIModel::from_pretrained(&args.model)?;
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!("  device = {device:?}\n");

    let token_ids = check_action_tokens(tokenizer, &items)?;
    eprintln!();

    let mut results: Vec<ItemResult> = Vec::with_capacity(n_items);
    for (printed, item) in items.iter().enumerate() {
        let correct_id = *token_ids.get(&item.correct).ok_or_else(|| {
            candle_mi::MIError::Config(format!("no token id for \"{}\"", item.correct))
        })?;
        let alt_id = *token_ids.get(&item.alternative).ok_or_else(|| {
            candle_mi::MIError::Config(format!("no token id for \"{}\"", item.alternative))
        })?;

        let prompt_ids = tokenizer.encode(&item.prompt)?;
        let seq_len = prompt_ids.len();
        let input = Tensor::new(&prompt_ids[..], &device)?.unsqueeze(0)?;
        let cache = model.forward(&input, &HookSpec::new())?;
        let probs = output_probs(&cache, seq_len - 1)?;

        let (top_idx, top_prob) = argmax_prob(&probs)?;
        let top_id = u32::try_from(top_idx).map_err(|e| {
            candle_mi::MIError::Config(format!("vocab index {top_idx} exceeds u32: {e}"))
        })?;
        let top1_token = tokenizer.decode_token(top_id)?;
        let top1_trimmed = top1_token.trim().to_owned();
        let full_correct = top1_trimmed.eq_ignore_ascii_case(&item.correct);

        let p_correct = prob_of(&probs, correct_id)?;
        let p_alt = prob_of(&probs, alt_id)?;
        let forced_correct = p_correct > p_alt;

        if printed < args.max_print {
            let m1 = if full_correct { "✓" } else { "✗" };
            let m2 = if forced_correct { "✓" } else { "✗" };
            eprintln!(
                "  [{:<9}] correct=\"{}\"  top1=\"{}\" ({:.4}) {m1}  forced {m2}  P(correct)={:.4} P(alt)={:.4}",
                item.family, item.correct, top1_trimmed, top_prob, p_correct, p_alt
            );
        }

        results.push(ItemResult {
            prompt: item.prompt.clone(),
            family: item.family.clone(),
            correct: item.correct.clone(),
            correct_token_id: correct_id,
            alternative: item.alternative.clone(),
            alternative_token_id: alt_id,
            top1_token,
            top1_token_trimmed: top1_trimmed,
            top1_prob: top_prob,
            full_vocab_top1_correct: full_correct,
            forced_choice_correct: forced_correct,
            p_correct,
            p_alternative: p_alt,
        });
    }

    // --- Aggregate ---
    let full_count = results.iter().filter(|r| r.full_vocab_top1_correct).count();
    let forced_count = results.iter().filter(|r| r.forced_choice_correct).count();
    let n_f64 = count_to_f64(n_items)?;
    let full_acc = count_to_f64(full_count)? / n_f64;
    let forced_acc = count_to_f64(forced_count)? / n_f64;
    let sum_p: f32 = results.iter().map(|r| r.p_correct).sum();
    let mean_p = f64::from(sum_p) / n_f64;
    let passed = full_acc >= THRESHOLD && forced_acc >= THRESHOLD;

    let per_family = grouped(&results, |r| r.family.clone())?;
    let per_token = grouped(&results, |r| format!("{}/{}", r.family, r.correct))?;

    eprintln!("\n=== Per (family, token) ===");
    eprintln!(
        "  {:<18} {:>3} {:>10} {:>10} {:>8}",
        "group", "n", "top1", "forced", "mean_P"
    );
    for s in &per_token {
        eprintln!(
            "  {:<18} {:>3} {:>10.3} {:>10.3} {:>8.4}",
            s.key, s.n, s.full_vocab_top1_accuracy, s.forced_choice_accuracy, s.mean_p_correct
        );
    }
    eprintln!("\n=== Per family ===");
    for s in &per_family {
        eprintln!(
            "  {:<12} n={:<3} top1={:.3} forced={:.3} meanP={:.4}",
            s.key, s.n, s.full_vocab_top1_accuracy, s.forced_choice_accuracy, s.mean_p_correct
        );
    }

    eprintln!("\n=== Overall ===");
    eprintln!("  full-vocab top-1 accuracy = {full_acc:.3}  ({full_count}/{n_items})");
    eprintln!("  forced-choice accuracy    = {forced_acc:.3}  ({forced_count}/{n_items})");
    eprintln!("  mean P(correct)           = {mean_p:.4}");
    eprintln!("  threshold (both)          = {THRESHOLD:.2}");
    eprintln!(
        "  VERDICT: {}",
        if passed {
            "PASS — linguistic means-ends cell clears the gate."
        } else {
            "FAIL — below threshold."
        }
    );

    let output = Output {
        model: args.model.clone(),
        items_path: args.items.display().to_string(),
        threshold: THRESHOLD,
        n_items,
        full_vocab_top1_accuracy: full_acc,
        forced_choice_accuracy: forced_acc,
        mean_p_correct: mean_p,
        passed,
        per_family,
        per_token,
        items: results,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    };
    write_json(&args.output, &output)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}
