// SPDX-License-Identifier: MIT OR Apache-2.0

//! Factual routing: measure how activation patching changes attention
//! routing from the output position to the subject position.
//!
//! ```bash
//! # Default: Llama 3.2 1B, 89 gold pairs
//! cargo run --release --features transformer --example factual_routing
//!
//! # Quick test with 5 pairs
//! cargo run --release --features transformer --example factual_routing -- --limit 5
//!
//! # With JSON output and memory reporting
//! cargo run --release --features transformer,memory --example factual_routing -- --output examples/results/factual_routing/llama-3.2-1b.json
//! ```
//!
//! **What it does:**
//!
//! 1. For each `CounterFact` gold pair, captures `AttnPattern` and `ResidPost`
//!    at every layer via [`MIModel::forward_text`](candle_mi::MIModel::forward_text).
//! 2. Patches the subject position with counterfactual activations (same as
//!    `counterfact_patching`), but captures `AttnPattern` during the patched
//!    forward pass.
//! 3. Computes per-head attention deltas: how much does each head change its
//!    attention from the last token to the fact-span tokens?
//! 4. Identifies "factual routing heads" — analogous to L21:H5 for rhyme
//!    planning in Gemma 2 2B (`attention_routing` example).
//!
//! This tests whether factual recall in small models uses attention-mediated
//! routing (regime 3 from `steering_convergence`), connecting the
//! early-commitment pattern found in `counterfact_patching` to the
//! attention-routing mechanism found in `attention_routing`.
//!
//! **Hypothesis:** If specific attention heads consistently change their
//! routing when facts are patched, factual recall uses the same
//! irrevocable-commitment mechanism as rhyme planning — a general property,
//! not task-specific.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use candle_mi::{FullActivationCache, HookPoint, HookSpec, Intervention, MIModel, sample_token};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use clap::Parser;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "factual_routing")]
#[command(about = "Attention routing in factual recall — identify factual routing heads")]
struct Args {
    /// `HuggingFace` model ID
    #[arg(default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// Write structured JSON output to this file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Number of contiguous layers per patching block
    #[arg(long, default_value = "4")]
    block_size: usize,

    /// Run only the first N prompt pairs (for quick testing)
    #[arg(long)]
    limit: Option<usize>,

    /// Path to `CounterFact` gold pairs JSON file
    #[arg(
        long,
        default_value = "data/counterfact_transluce_test_both_correct.json"
    )]
    data: PathBuf,

    /// Suppress per-pair and total runtime reporting
    #[arg(long)]
    no_runtime: bool,

    /// Number of top routing heads to report per block
    #[arg(long, default_value = "10")]
    top_k: usize,
}

// ---------------------------------------------------------------------------
// CounterFact prompt pairs (reused from counterfact_patching)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct CounterFactPair {
    original_prompt: String,
    counterfactual_prompt: String,
    gt_original_target: String,
    gt_counterfactual_target: String,
}

impl CounterFactPair {
    fn fact_span_len(&self) -> usize {
        self.original_prompt
            .find("\n\n")
            .unwrap_or(self.original_prompt.len())
    }

    fn label(&self) -> String {
        let end = self.fact_span_len().min(35);
        // BORROW: chars — safe Unicode truncation
        self.original_prompt.chars().take(end).collect()
    }
}

fn load_pairs(path: &Path) -> candle_mi::Result<Vec<CounterFactPair>> {
    let data = std::fs::read_to_string(path).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
    })?;
    let pairs: Vec<CounterFactPair> = serde_json::from_str(&data).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to parse {}: {e}", path.display()))
    })?;
    Ok(pairs)
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    block_size: usize,
    n_layers: usize,
    n_heads: usize,
    n_blocks: usize,
    total_time_secs: f64,
    results: Vec<PairRoutingResult>,
    aggregate: AggregateRouting,
}

#[derive(Serialize)]
struct PairRoutingResult {
    label: String,
    gt_original_target: String,
    gt_counterfactual_target: String,
    original_continuation: String,
    cf_continuation: String,
    fact_positions: Vec<usize>,
    time_secs: f64,
    blocks: Vec<BlockRouting>,
}

#[derive(Serialize)]
struct BlockRouting {
    layers: Vec<usize>,
    is_different: bool,
    top_heads: Vec<HeadDelta>,
    total_routing_shift: f32,
}

#[derive(Serialize, Clone)]
struct HeadDelta {
    layer: usize,
    head: usize,
    baseline_attn: f32,
    patched_attn: f32,
    delta: f32,
}

#[derive(Serialize)]
struct AggregateRouting {
    head_frequency: Vec<HeadFrequency>,
    avg_routing_by_block: Vec<BlockAvg>,
}

#[derive(Serialize)]
struct HeadFrequency {
    layer: usize,
    head: usize,
    appearances_in_top10: usize,
    avg_abs_delta: f32,
}

#[derive(Serialize)]
struct BlockAvg {
    layers: Vec<usize>,
    avg_total_routing_shift: f32,
    pct_is_different: f32,
}

// ---------------------------------------------------------------------------
// Attention extraction (adapted from attention_routing)
// ---------------------------------------------------------------------------

/// Extract per-head attention weights from query_pos to multiple key positions,
/// averaged over all key positions.
///
/// # Shapes
///
/// - `AttnPattern(layer)`: `[1, n_heads, seq_len, seq_len]`
/// - returns: `[n_layers][n_heads]` averaged attention weights
fn extract_attention_to_span(
    cache: &candle_mi::HookCache,
    n_layers: usize,
    n_heads: usize,
    query_pos: usize,
    key_positions: &[usize],
) -> candle_mi::Result<Vec<Vec<f32>>> {
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let pattern = cache.require(&HookPoint::AttnPattern(layer))?;
        // pattern: [1, n_heads, seq_len, seq_len]
        // Extract the query row: [n_heads, seq_len] at query_pos
        let query_row = pattern
            .get(0)? // [n_heads, seq_len, seq_len]
            .narrow(1, query_pos, 1)? // [n_heads, 1, seq_len]
            .squeeze(1)?; // [n_heads, seq_len]
        // PROMOTE: attention may be BF16; extraction needs F32
        let query_row_f32 = query_row.to_dtype(candle_core::DType::F32)?;
        let row_data: Vec<Vec<f32>> = query_row_f32.to_vec2()?;

        let mut head_avgs = Vec::with_capacity(n_heads);
        for head_row in &row_data {
            // Average attention over all key positions
            let mut sum = 0.0_f32;
            for &kp in key_positions {
                // INDEX: kp bounded by seq_len (from label_spans output)
                sum += head_row.get(kp).copied().unwrap_or(0.0);
            }
            // CAST: usize → f32 for averaging, key_positions.len() is small
            #[allow(clippy::as_conversions)]
            let avg = if key_positions.is_empty() {
                0.0
            } else {
                sum / key_positions.len() as f32
            };
            head_avgs.push(avg);
        }
        result.push(head_avgs);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Patching helpers (reused from counterfact_patching)
// ---------------------------------------------------------------------------

/// Build a tensor that is `base` everywhere except at `patch_pos` where it
/// takes values from `patch_vector`.
fn replace_position(
    base: &candle_core::Tensor,
    patch_vector: &candle_core::Tensor,
    patch_pos: usize,
    seq_len: usize,
    hidden: usize,
    device: &candle_core::Device,
) -> candle_mi::Result<candle_core::Tensor> {
    let mut mask_data = vec![0.0_f32; seq_len * hidden];
    for i in 0..hidden {
        // INDEX: patch_pos * hidden + i bounded by seq_len * hidden
        #[allow(clippy::indexing_slicing)]
        {
            mask_data[patch_pos * hidden + i] = 1.0;
        }
    }
    let mask = candle_core::Tensor::from_vec(mask_data, (seq_len, hidden), device)?;
    let patch_broadcast = patch_vector.unsqueeze(0)?.broadcast_as((seq_len, hidden))?;
    let one_minus_mask = (1.0 - &mask)?;
    let result = (base * &one_minus_mask)? + (patch_broadcast * &mask)?;
    Ok(result?)
}

/// Build a `HookSpec` that replaces `ResidPost` at every layer in `block_layers`,
/// AND captures `AttnPattern` at all layers.
fn build_routing_patch(
    orig_acts: &FullActivationCache,
    cf_acts: &FullActivationCache,
    block_layers: &[usize],
    orig_pos: usize,
    cf_pos: usize,
    orig_seq_len: usize,
    hidden: usize,
    n_layers: usize,
    device: &candle_core::Device,
) -> candle_mi::Result<HookSpec> {
    let mut hooks = HookSpec::new();

    // Interventions: replace ResidPost at each layer in the block
    for &layer in block_layers {
        let orig_resid = orig_acts
            .get_layer(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("layer {layer} not in orig cache")))?;
        let cf_vector = cf_acts.get_position(layer, cf_pos)?;
        let patched = replace_position(
            orig_resid,
            &cf_vector,
            orig_pos,
            orig_seq_len,
            hidden,
            device,
        )?
        .unsqueeze(0)?; // [1, seq, hidden]
        hooks.intervene(HookPoint::ResidPost(layer), Intervention::Replace(patched));
    }

    // Captures: AttnPattern at ALL layers (not just the block)
    for layer in 0..n_layers {
        hooks.capture(HookPoint::AttnPattern(layer));
    }

    Ok(hooks)
}

fn compute_blocks(n_layers: usize, block_size: usize) -> Vec<Vec<usize>> {
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < n_layers {
        let end = (start + block_size).min(n_layers);
        blocks.push((start..end).collect());
        start = end;
    }
    blocks
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();

    // Load model
    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    println!("Loading {}...", args.model);
    let t0 = Instant::now();
    let model = MIModel::from_pretrained(&args.model)?;
    let load_time = t0.elapsed();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Config("model has no tokenizer".into()))?;

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    let blocks = compute_blocks(n_layers, args.block_size);
    let n_blocks = blocks.len();

    println!(
        "  Layers: {n_layers}, heads: {n_heads}, hidden: {hidden}, device: {:?}",
        model.device()
    );
    if !args.no_runtime {
        println!("  Load time: {load_time:.2?}");
    }

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let pairs = load_pairs(&args.data)?;
    println!(
        "  Data: {} gold pairs from {}",
        pairs.len(),
        args.data.display()
    );
    println!("  Block size: {}, blocks: {n_blocks}", args.block_size);
    let n_pairs = args.limit.unwrap_or(pairs.len()).min(pairs.len());

    let mut all_results: Vec<PairRoutingResult> = Vec::with_capacity(n_pairs);

    // Aggregate: count how often each (layer, head) appears in top-10
    let mut head_appearances: HashMap<(usize, usize), usize> = HashMap::new();
    let mut head_abs_deltas: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
    let mut block_shifts: Vec<Vec<f32>> = (0..n_blocks).map(|_| Vec::new()).collect();
    let mut block_different: Vec<usize> = vec![0; n_blocks];
    let mut block_total: Vec<usize> = vec![0; n_blocks];

    let total_start = Instant::now();

    for (pair_idx, pair) in pairs.iter().take(n_pairs).enumerate() {
        let pair_start = Instant::now();
        let label = pair.label();
        println!("\n  --- [{}/{}] {} ---", pair_idx + 1, n_pairs, label);

        // ── Step 1: Baseline capture (AttnPattern + ResidPost) ──────────
        let mut baseline_hooks = HookSpec::new();
        for layer in 0..n_layers {
            baseline_hooks.capture(HookPoint::AttnPattern(layer));
            baseline_hooks.capture(HookPoint::ResidPost(layer));
        }

        let orig_result = model.forward_text(&pair.original_prompt, &baseline_hooks)?;
        let orig_seq_len = orig_result.seq_len();

        // Identify fact-span positions via label_spans
        let fact_len = pair.fact_span_len();
        let labels = orig_result.encoding().label_spans(&[("fact", 0..fact_len)]);
        let fact_positions: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, l)| l.starts_with("fact"))
            .map(|(i, _)| i)
            .collect();

        // Find the subject_final position (last fact token)
        let subject_pos = labels
            .iter()
            .rposition(|l| l.starts_with("fact"))
            .unwrap_or(0);

        let last_pos = orig_seq_len - 1;

        // Extract baseline attention: last token → fact span
        let baseline_attn = extract_attention_to_span(
            orig_result.cache(),
            n_layers,
            n_heads,
            last_pos,
            &fact_positions,
        )?;

        // Build FullActivationCache from ResidPost
        let mut orig_acts = FullActivationCache::with_capacity(n_layers);
        for layer in 0..n_layers {
            let resid = orig_result.require(&HookPoint::ResidPost(layer))?;
            orig_acts.push(resid.get(0)?);
        }

        // Get baseline prediction
        let orig_logits = orig_result.output().get(0)?.get(last_pos)?;
        let orig_tid = sample_token(&orig_logits, 0.0)?;
        let orig_continuation = tokenizer.decode(&[orig_tid])?;

        // ── Step 2: Counterfactual capture (ResidPost only) ─────────────
        let mut cf_hooks = HookSpec::new();
        for layer in 0..n_layers {
            cf_hooks.capture(HookPoint::ResidPost(layer));
        }
        let cf_result = model.forward_text(&pair.counterfactual_prompt, &cf_hooks)?;
        let cf_seq_len = cf_result.seq_len();

        let mut cf_acts = FullActivationCache::with_capacity(n_layers);
        for layer in 0..n_layers {
            let resid = cf_result.require(&HookPoint::ResidPost(layer))?;
            cf_acts.push(resid.get(0)?);
        }

        let cf_logits = cf_result.output().get(0)?.get(cf_seq_len - 1)?;
        let cf_tid = sample_token(&cf_logits, 0.0)?;
        let cf_continuation = tokenizer.decode(&[cf_tid])?;

        println!(
            "  GT: {} -> {}  |  Pred: {} -> {}",
            pair.gt_original_target,
            pair.gt_counterfactual_target,
            orig_continuation.trim(),
            cf_continuation.trim()
        );
        println!(
            "  Fact positions: {:?} (subject_final: {subject_pos})",
            &fact_positions
        );

        // Map subject position to CF sequence
        let cf_subject_pos = subject_pos.min(cf_seq_len.saturating_sub(1));

        // ── Step 3: Patched attention capture (per block) ───────────────
        let orig_input = candle_core::Tensor::new(&orig_result.encoding().ids[..], model.device())?
            .unsqueeze(0)?;

        let mut block_results: Vec<BlockRouting> = Vec::with_capacity(n_blocks);

        for (bi, block) in blocks.iter().enumerate() {
            let patch_hooks = build_routing_patch(
                &orig_acts,
                &cf_acts,
                block,
                subject_pos,
                cf_subject_pos,
                orig_seq_len,
                hidden,
                n_layers,
                model.device(),
            )?;

            let patched_cache = model.forward(&orig_input, &patch_hooks)?;

            // Extract patched attention
            let patched_attn = extract_attention_to_span(
                &patched_cache,
                n_layers,
                n_heads,
                last_pos,
                &fact_positions,
            )?;

            // Check if prediction changed
            let patched_logits = patched_cache.output().get(0)?.get(last_pos)?;
            let patched_tid = sample_token(&patched_logits, 0.0)?;
            let ablated = tokenizer.decode(&[patched_tid])?;
            let is_different = ablated != orig_continuation;

            // Compute deltas
            let mut all_deltas: Vec<HeadDelta> = Vec::with_capacity(n_layers * n_heads);
            let mut total_shift = 0.0_f32;

            for layer in 0..n_layers {
                for head in 0..n_heads {
                    // INDEX: layer/head bounded by n_layers/n_heads
                    #[allow(clippy::indexing_slicing)]
                    let b = baseline_attn[layer][head];
                    #[allow(clippy::indexing_slicing)]
                    let p = patched_attn[layer][head];
                    let delta = p - b;
                    total_shift += delta.abs();
                    all_deltas.push(HeadDelta {
                        layer,
                        head,
                        baseline_attn: b,
                        patched_attn: p,
                        delta,
                    });
                }
            }

            // Sort by |delta| descending
            all_deltas.sort_by(|a, b| {
                b.delta
                    .abs()
                    .partial_cmp(&a.delta.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_heads: Vec<HeadDelta> = all_deltas.iter().take(args.top_k).cloned().collect();

            // Print top-3 for this block
            // INDEX: block is non-empty by construction
            #[allow(clippy::indexing_slicing)]
            let first = block[0];
            #[allow(clippy::indexing_slicing)]
            let last_layer = block[block.len() - 1];
            let marker = if is_different { "CHANGED" } else { "same" };
            print!("  Block [{first:>2}-{last_layer:>2}]:");
            for hd in top_heads.iter().take(3) {
                print!(" L{}:H{}{:+.4}", hd.layer, hd.head, hd.delta);
            }
            println!("  total={total_shift:.4}  {marker}");

            // Update aggregates
            for hd in top_heads.iter().take(10) {
                let key = (hd.layer, hd.head);
                *head_appearances.entry(key).or_insert(0) += 1;
                head_abs_deltas.entry(key).or_default().push(hd.delta.abs());
            }
            // INDEX: bi bounded by n_blocks
            #[allow(clippy::indexing_slicing)]
            {
                block_shifts[bi].push(total_shift);
                block_total[bi] += 1;
                if is_different {
                    block_different[bi] += 1;
                }
            }

            block_results.push(BlockRouting {
                layers: block.clone(),
                is_different,
                top_heads,
                total_routing_shift: total_shift,
            });
        }

        let pair_time = pair_start.elapsed();
        if !args.no_runtime {
            println!("  Pair time: {pair_time:.2?}");
        }

        all_results.push(PairRoutingResult {
            label,
            gt_original_target: pair.gt_original_target.clone(),
            gt_counterfactual_target: pair.gt_counterfactual_target.clone(),
            original_continuation: orig_continuation,
            cf_continuation,
            fact_positions,
            time_secs: pair_time.as_secs_f64(),
            blocks: block_results,
        });
    }

    let total_time = total_start.elapsed();

    // ── Aggregate ───────────────────────────────────────────────────────
    if args.no_runtime {
        println!("\n  === Aggregate ({n_pairs} pairs) ===\n");
    } else {
        println!("\n  === Aggregate ({n_pairs} pairs, {total_time:.1?}) ===\n");
    }

    // Top routing heads by frequency
    let mut freq_list: Vec<((usize, usize), usize)> = head_appearances.into_iter().collect();
    freq_list.sort_by(|a, b| b.1.cmp(&a.1));

    println!("  Top Factual Routing Heads:");
    let mut head_freq_output: Vec<HeadFrequency> = Vec::new();
    for &((layer, head), count) in freq_list.iter().take(args.top_k) {
        let deltas = head_abs_deltas
            .get(&(layer, head))
            .cloned()
            .unwrap_or_default();
        // CAST: usize → f32 for averaging
        #[allow(clippy::as_conversions)]
        let avg_delta = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f32>() / deltas.len() as f32
        };
        // CAST: usize → f64 for percentage
        #[allow(clippy::as_conversions)]
        let pct = count as f64 / n_pairs as f64 * 100.0;
        println!(
            "    L{layer:>2}:H{head:<2}   appeared {count:>3}/{n_pairs} ({pct:>5.1}%)  avg |delta| = {avg_delta:.5}"
        );
        head_freq_output.push(HeadFrequency {
            layer,
            head,
            appearances_in_top10: count,
            avg_abs_delta: avg_delta,
        });
    }

    // Routing shift by block
    println!("\n  Routing Shift by Block:");
    let mut block_avg_output: Vec<BlockAvg> = Vec::new();
    for (bi, block) in blocks.iter().enumerate() {
        // INDEX: bi bounded by n_blocks
        #[allow(clippy::indexing_slicing)]
        let shifts = &block_shifts[bi];
        // CAST: usize → f32 for averaging
        #[allow(clippy::as_conversions)]
        let avg_shift = if shifts.is_empty() {
            0.0
        } else {
            shifts.iter().sum::<f32>() / shifts.len() as f32
        };
        #[allow(clippy::indexing_slicing)]
        let diff = block_different[bi];
        #[allow(clippy::indexing_slicing)]
        let total = block_total[bi];
        // CAST: usize → f32 for percentage
        #[allow(clippy::as_conversions)]
        let pct = if total > 0 {
            diff as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        #[allow(clippy::indexing_slicing)]
        let first = block[0];
        #[allow(clippy::indexing_slicing)]
        let last_layer = block[block.len() - 1];
        println!(
            "    Block [{first:>2}-{last_layer:>2}]:  avg shift = {avg_shift:.4}  (is_different: {pct:.1}%)"
        );
        block_avg_output.push(BlockAvg {
            layers: block.clone(),
            avg_total_routing_shift: avg_shift,
            pct_is_different: pct,
        });
    }

    // ── JSON output ─────────────────────────────────────────────────────
    if let Some(path) = &args.output {
        let output = JsonOutput {
            model_id: args.model.clone(),
            block_size: args.block_size,
            n_layers,
            n_heads,
            n_blocks,
            total_time_secs: total_time.as_secs_f64(),
            results: all_results,
            aggregate: AggregateRouting {
                head_frequency: head_freq_output,
                avg_routing_by_block: block_avg_output,
            },
        };
        write_json(path, &output)?;
        println!("\n  JSON written to {}", path.display());
    }

    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_json(path: &Path, output: &JsonOutput) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    std::fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;
    Ok(())
}
