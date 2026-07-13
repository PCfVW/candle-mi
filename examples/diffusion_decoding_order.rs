// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding-order analysis for `MDLM`.
//!
//! ```bash
//! cargo run --features diffusion,mmap --release --example diffusion_decoding_order
//! ```
//!
//! A masked-diffusion model can reveal masked positions in any order.  This
//! example fills the same masked completion three ways — **random**,
//! **confidence** (reveal the highest top1−top2 margin first), and **entropy**
//! (reveal the lowest-entropy position first) — one token per step, greedily.
//!
//! For each order it reports two SAE-free signals that the brief's decoding-order
//! experiment is built around:
//!
//! - **reveal confidence**: mean top-1 probability of each token at the moment
//!   it is revealed (confidence/entropy orders front-load confident tokens);
//! - **prediction stability**: mean fraction of *still-masked* positions whose
//!   top-1 prediction is unchanged from the previous step (a residual-free proxy
//!   for the per-step feature stability the full SAE experiment measures).
//!
//! The full version (top-K SAE-feature Jaccard across steps) needs a trained
//! SAE and is deferred; this example demonstrates the order-controlled decoder
//! and the per-step signals it exposes.

use candle_core::{DType, IndexOp, Tensor};
use candle_mi::{HookSpec, MIModel, MITokenizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `HuggingFace` repo id of the MDLM masked-diffusion checkpoint.
const MODEL_ID: &str = "TheQweaker/mdlm-owt-noflash";
/// Number of masked positions to fill after the prompt.
const GEN_LEN: usize = 6;
/// RNG seed for the random order (fixes the comparison).
const SEED: u64 = 0;

/// Which masked position to reveal next at each step.
#[derive(Debug, Clone, Copy)]
enum Order {
    /// Reveal a uniformly random masked position.
    Random,
    /// Reveal the position with the largest top1 − top2 probability margin.
    Confidence,
    /// Reveal the position with the lowest predictive entropy.
    Entropy,
}

impl Order {
    /// Short display label.
    const fn label(self) -> &'static str {
        match self {
            Self::Random => "random    ",
            Self::Confidence => "confidence",
            Self::Entropy => "entropy   ",
        }
    }
}

fn main() -> candle_mi::Result<()> {
    let model = MIModel::from_pretrained(MODEL_ID)?;
    let Ok(tokenizer) = MITokenizer::from_hf_cache("openai-community/gpt2")
        .or_else(|_| MITokenizer::from_hf_cache("gpt2"))
    else {
        println!("GPT-2 tokenizer not found in the HuggingFace cache.");
        println!("Get tokenizer.json from https://huggingface.co/openai-community/gpt2");
        println!(
            "e.g. (dogfooding hf-fm):  hf-fm download-file openai-community/gpt2 tokenizer.json"
        );
        return Ok(());
    };

    // MDLM convention: [MASK] is the final vocab id. Decoder-style diffusion LMs
    // (Dream, a2d-qwen2) use a distinct `<|mask|>` token id instead — supply it
    // when reusing this on them.
    let mask_id = u32::try_from(model.vocab_size() - 1).map_err(|e| {
        candle_mi::MIError::Model(candle_core::Error::Msg(format!("mask id overflow: {e}")))
    })?;

    let prompt_text = "The weather today is";
    let prompt = tokenizer.encode_raw(prompt_text)?;
    let seq_len = prompt.len() + GEN_LEN;

    println!(
        "MDLM decoding-order analysis — prompt {prompt_text:?}, filling {GEN_LEN} positions\n"
    );
    println!(
        "{:<10} | {:>11} | {:>11} | completion",
        "order", "reveal conf", "pred stab"
    );
    println!("{}", "-".repeat(72));

    for order in [Order::Random, Order::Confidence, Order::Entropy] {
        let result = decode_with_order(&model, mask_id, &prompt, seq_len, order)?;
        let filled = result.tokens.get(prompt.len()..).unwrap_or(&[]);
        let text = tokenizer.decode(filled)?;
        println!(
            "{} | {:>11.3} | {:>11.3} | {:?}",
            order.label(),
            result.mean_reveal_confidence,
            result.mean_prediction_stability,
            text.trim()
        );
    }

    println!(
        "\nConfidence/entropy orders front-load confident, stable positions; random does not."
    );
    Ok(())
}

/// Result of one order-controlled decode.
struct DecodeResult {
    /// Final token ids (prompt prefix + filled positions).
    tokens: Vec<u32>,
    /// Mean top-1 probability of each token when it was revealed.
    mean_reveal_confidence: f32,
    /// Mean fraction of still-masked positions whose top-1 prediction is
    /// unchanged between consecutive steps.
    mean_prediction_stability: f32,
}

/// Fill the masked positions one per step, choosing the next position by `order`
/// and revealing it greedily (`SUBS` argmax).
#[allow(clippy::too_many_lines)] // flat decode loop: forward -> stats -> choose -> reveal
fn decode_with_order(
    model: &MIModel,
    mask_id: u32,
    prompt: &[u32],
    seq_len: usize,
    order: Order,
) -> candle_mi::Result<DecodeResult> {
    // CAST: u32 -> usize, vocab index used to forbid the [MASK] logit
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    let mask_idx = mask_id as usize;

    let mut x = vec![mask_id; seq_len];
    for (i, &token) in prompt.iter().take(seq_len).enumerate() {
        if let Some(slot) = x.get_mut(i) {
            *slot = token;
        }
    }
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut conf_sum = 0.0_f32;
    let mut reveals = 0_u32;
    let mut stab_sum = 0.0_f32;
    let mut stab_steps = 0_u32;
    let mut prev_top1: Option<Vec<Option<u32>>> = None;

    loop {
        let masked: Vec<usize> = (0..seq_len)
            .filter(|&i| x.get(i).copied() == Some(mask_id))
            .collect();
        if masked.is_empty() {
            break;
        }

        let input = Tensor::new(x.as_slice(), model.device())?.unsqueeze(0)?;
        let cache = model.forward(&input, &HookSpec::new())?;
        let logits = cache
            .output()
            .i(0)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;

        // Per masked position: (pos, top1 token, p1, margin, entropy).
        let mut stats: Vec<(usize, u32, f32, f32, f32)> = Vec::with_capacity(masked.len());
        for &pos in &masked {
            if let Some(row) = logits.get(pos) {
                let (token, p1, margin, entropy) = dist_stats(row, mask_idx);
                stats.push((pos, token, p1, margin, entropy));
            }
        }

        // Prediction stability vs the previous step (over still-masked positions).
        if let Some(prev) = &prev_top1 {
            let mut same = 0_u32;
            let mut total = 0_u32;
            for &(pos, token, ..) in &stats {
                if let Some(&Some(prev_token)) = prev.get(pos) {
                    total += 1;
                    if prev_token == token {
                        same += 1;
                    }
                }
            }
            if total > 0 {
                // CAST: u32 -> f32, small counts are exact in f32
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                {
                    stab_sum += same as f32 / total as f32;
                }
                stab_steps += 1;
            }
        }

        // Record this step's top-1 predictions for the next step's comparison.
        let mut cur_top1 = vec![None; seq_len];
        for &(pos, token, ..) in &stats {
            if let Some(slot) = cur_top1.get_mut(pos) {
                *slot = Some(token);
            }
        }
        prev_top1 = Some(cur_top1);

        // Choose the next position to reveal.
        let chosen = match order {
            Order::Random => {
                let idx = rng.gen_range(0..stats.len());
                stats.get(idx).map(|s| s.0)
            }
            Order::Confidence => stats
                .iter()
                .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
                .map(|s| s.0),
            Order::Entropy => stats
                .iter()
                .min_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal))
                .map(|s| s.0),
        };
        let Some(chosen_pos) = chosen else { break };

        if let Some(&(_, token, p1, ..)) = stats.iter().find(|s| s.0 == chosen_pos) {
            conf_sum += p1;
            reveals += 1;
            if let Some(slot) = x.get_mut(chosen_pos) {
                *slot = token;
            }
        }
    }

    // CAST: u32 -> f32, small counts are exact in f32
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let mean_reveal_confidence = if reveals > 0 {
        conf_sum / reveals as f32
    } else {
        0.0
    };
    // CAST: u32 -> f32, small counts are exact in f32
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let mean_prediction_stability = if stab_steps > 0 {
        stab_sum / stab_steps as f32
    } else {
        0.0
    };

    Ok(DecodeResult {
        tokens: x,
        mean_reveal_confidence,
        mean_prediction_stability,
    })
}

/// Top-1 token, its probability, the top1−top2 margin, and the entropy of a
/// `SUBS`-masked logit row (the `[MASK]` index is excluded from the softmax).
// EXPLICIT: keep the readable `-= p * p.ln()`; fused `mul_add` buys no accuracy
// for a displayed entropy scalar.
#[allow(clippy::suboptimal_flops)]
fn dist_stats(row: &[f32], mask_idx: usize) -> (u32, f32, f32, f32) {
    let max = row
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mask_idx)
        .map(|(_, &l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row
        .iter()
        .enumerate()
        .map(|(i, &l)| if i == mask_idx { 0.0 } else { (l - max).exp() })
        .collect();
    let sum: f32 = exps.iter().sum::<f32>().max(f32::MIN_POSITIVE);

    let mut top1 = (0_usize, -1.0_f32);
    let mut top2 = -1.0_f32;
    let mut entropy = 0.0_f32;
    for (i, &e) in exps.iter().enumerate() {
        let p = e / sum;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
        if p > top1.1 {
            top2 = top1.1;
            top1 = (i, p);
        } else if p > top2 {
            top2 = p;
        }
    }
    let margin = top1.1 - top2.max(0.0);
    // CAST: usize -> u32, vocab index fits in u32
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    (top1.0 as u32, top1.1, margin, entropy)
}

// (GPT-2 tokenizer discovery now lives in `MITokenizer::from_hf_cache`.)
