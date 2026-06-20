// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quick start: `MDLM` masked-diffusion fill-in-the-blank.
//!
//! ```bash
//! cargo run --features diffusion,mmap --release --example quick_start_mdlm
//! ```
//!
//! **What it does:**
//!
//! 1. Loads `kuleshov-group/mdlm-owt` via
//!    [`MIModel::from_pretrained`](candle_mi::MIModel::from_pretrained)
//!    (downloads to the `HuggingFace` cache on first run, ~648 MB).
//! 2. Locates a `GPT-2` `tokenizer.json` in the cache — the `MDLM` repo ships
//!    none, so the plain `GPT-2` tokenizer is used (vocab `0..=50256`, with
//!    `[MASK]` = `50257`).
//! 3. Masks the word `" Paris"` in *"The capital of France is Paris."*, runs a
//!    single bidirectional forward pass, applies the `SUBS` rule (forbid
//!    `[MASK]`), and prints the model's fill-in for the masked position.
//!
//! `MDLM` is bidirectional, so a single denoising step already recovers the
//! masked word from the surrounding context.

use candle_core::{IndexOp, Tensor};
use candle_mi::{HookSpec, MIModel, MITokenizer};

/// `HuggingFace` repo id of the MDLM masked-diffusion checkpoint.
const MODEL_ID: &str = "kuleshov-group/mdlm-owt";

fn main() -> candle_mi::Result<()> {
    // 1. Load MDLM (cache hit if already downloaded).
    let model = MIModel::from_pretrained(MODEL_ID)?;
    println!(
        "{MODEL_ID}: {} blocks, {} hidden, {} heads, device {:?}",
        model.num_layers(),
        model.hidden_size(),
        model.num_heads(),
        model.device()
    );

    // 2. GPT-2 tokenizer (MDLM ships none).
    let Ok(tokenizer) = MITokenizer::from_hf_cache("openai-community/gpt2")
        .or_else(|_| MITokenizer::from_hf_cache("gpt2"))
    else {
        println!("\nGPT-2 tokenizer not found in the HuggingFace cache.");
        println!("Fetch it (dogfooding hf-fm):");
        println!("  hf-fm download-file openai-community/gpt2 tokenizer.json");
        return Ok(());
    };

    // 3. Mask the word " Paris" and let MDLM fill it back in.
    let text = "The capital of France is Paris.";
    let target = " Paris";
    // MDLM convention: [MASK] is the final vocab id (50257). Decoder-style
    // diffusion LMs (Dream, a2d-qwen2) instead use a distinct `<|mask|>` token
    // id (e.g. 151666 / 151665) — supply that when reusing this elsewhere.
    let mask_id = model.vocab_size() - 1;
    let mask_u32 = u32::try_from(mask_id).map_err(|e| {
        candle_mi::MIError::Model(candle_core::Error::Msg(format!("mask id overflow: {e}")))
    })?;

    let mut ids = tokenizer.encode_raw(text)?;
    let target_id = tokenizer
        .encode_raw(target)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            candle_mi::MIError::Tokenizer(format!("target {target:?} did not tokenize"))
        })?;
    let pos = ids.iter().position(|&t| t == target_id).ok_or_else(|| {
        candle_mi::MIError::Tokenizer(format!("target {target:?} not found in {text:?}"))
    })?;
    if let Some(slot) = ids.get_mut(pos) {
        *slot = mask_u32;
    }

    println!("\nPrompt (masked): {text:?}  — masking {target:?} at position {pos}");

    let input = Tensor::new(&ids[..], model.device())?.unsqueeze(0)?; // [1, seq]
    let cache = model.forward(&input, &HookSpec::new())?;
    let logits = cache.output(); // [1, seq, vocab]

    // SUBS: forbid the [MASK] token, then greedily decode the masked position.
    let last = logits.i((0, pos))?; // [vocab]
    let vocab = model.vocab_size();
    let suppress: Vec<f32> = (0..vocab)
        .map(|i| if i == mask_id { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    let suppress = Tensor::new(suppress, model.device())?;
    let masked = last.broadcast_add(&suppress)?;
    let pred = candle_mi::sample_token(&masked, 0.0)?;

    println!(
        "MDLM fills [MASK] -> {:?}  (expected \" Paris\")",
        tokenizer.decode(&[pred])?
    );
    Ok(())
}

// (GPT-2 tokenizer discovery now lives in `MITokenizer::from_hf_cache`.)
