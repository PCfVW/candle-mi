// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quick start: load a transformer, run a forward pass, print top predictions.
//!
//! ```bash
//! cargo run --release --example quick_start_transformer
//! ```
//!
//! On first run, downloads Llama 3.2 1B (~2.5 GB) from `HuggingFace` Hub.
//! Subsequent runs use the local cache.

use candle_mi::{HookPoint, HookSpec, MIModel, MITokenizer};

fn main() -> candle_mi::Result<()> {
    // 1. Load model (auto-selects GPU if available, else CPU)
    let model_id = "meta-llama/Llama-3.2-1B";
    println!("Loading {model_id}...");
    let model = MIModel::from_pretrained(model_id)?;
    println!(
        "  {} layers, {} hidden, {} vocab, device: {:?}",
        model.num_layers(),
        model.hidden_size(),
        model.vocab_size(),
        model.device()
    );

    // 2. Load tokenizer
    let api = hf_hub::api::sync::Api::new().map_err(|e| {
        candle_mi::MIError::Model(candle_core::Error::Msg(format!("HF Hub API: {e}")))
    })?;
    let tokenizer_path = api
        .model(model_id.to_string())
        .get("tokenizer.json")
        .map_err(|e| candle_mi::MIError::Tokenizer(format!("tokenizer.json: {e}")))?;
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    // 3. Encode a prompt
    let prompt = "The capital of France is";
    let token_ids = tokenizer.encode(prompt)?;
    let input = candle_core::Tensor::new(&token_ids[..], model.device())?.unsqueeze(0)?; // [1, seq_len]
    println!("\nPrompt: \"{prompt}\"  ({} tokens)", token_ids.len());

    // 4. Forward pass — no hooks (zero overhead)
    let hooks = HookSpec::new();
    let cache = model.forward(&input, &hooks)?;
    let logits = cache.output(); // [1, seq_len, vocab_size]

    // 5. Get top-5 predictions for the last token
    let seq_len = token_ids.len();
    let last_logits = logits.get(0)?.get(seq_len - 1)?; // [vocab_size]
    print_top_k(&last_logits, &tokenizer, 5)?;

    // 6. Forward pass WITH hook capture — attention pattern at layer 0
    let mut hooks_with_capture = HookSpec::new();
    hooks_with_capture.capture(HookPoint::AttnPattern(0));
    let cache = model.forward(&input, &hooks_with_capture)?;
    let attn = cache.require(&HookPoint::AttnPattern(0))?;
    println!(
        "\nAttention pattern at layer 0: {:?}  (shape: [batch, heads, seq_q, seq_k])",
        attn.dims()
    );

    Ok(())
}

/// Print top-k token predictions from a logits vector.
fn print_top_k(
    logits: &candle_core::Tensor,
    tokenizer: &MITokenizer,
    k: usize,
) -> candle_mi::Result<()> {
    let logits_f32: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;

    // Argsort descending
    let mut indexed: Vec<(usize, f32)> = logits_f32.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop-{k} predictions:");
    for (rank, (idx, score)) in indexed.iter().take(k).enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let token_text = tokenizer.decode(&[*idx as u32])?;
        println!("  #{}: {:>8.3}  \"{}\"", rank + 1, score, token_text.trim());
    }
    Ok(())
}
