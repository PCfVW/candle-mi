// SPDX-License-Identifier: MIT OR Apache-2.0

//! Auto-config dogfooding: download a model and test auto-config loading.
//!
//! ```bash
//! cargo run --release --features transformer --example auto_config_dogfood -- "allenai/OLMo-1B-hf"
//! ```
//!
//! **What it does:**
//!
//! 1. Downloads the model using `hf-fetch-model` (fast parallel download).
//! 2. Inspects `config.json` and reports whether auto-config will be used.
//! 3. Loads the model via [`MIModel::from_pretrained()`], which uses
//!    auto-config for unknown model families (includes compatibility check).
//! 4. Runs a short forward pass to verify the model works end-to-end.

fn main() {
    tracing_subscriber::fmt::init();

    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "allenai/OLMo-1B-hf".to_string());

    if let Err(e) = run(&model_id) {
        eprintln!("\nError: {e}");
        std::process::exit(1);
    }
}

fn run(model_id: &str) -> candle_mi::Result<()> {
    download_phase(model_id)?;
    inspect_config(model_id)?;
    let model = load_phase(model_id)?;
    forward_phase(&model)?;

    eprintln!("\n=== Done ===");
    Ok(())
}

/// Download the model and measure elapsed time.
///
/// Uses `hf_fetch_model` directly (without indicatif progress bars) to
/// avoid terminal artifacts when files are already cached.
fn download_phase(model_id: &str) -> candle_mi::Result<()> {
    eprintln!("=== Step 1: Download ===");
    eprintln!("Model: {model_id}");
    eprintln!("(Files will be cached in ~/.cache/huggingface/hub/)\n");

    let t0 = std::time::Instant::now();
    let fetch_config = candle_mi::fetch_config_builder()
        .build()
        .map_err(|e| candle_mi::MIError::Download(e.to_string()))?;
    let outcome =
        hf_fetch_model::download_files_with_config_blocking(model_id.to_owned(), &fetch_config)
            .map_err(|e| candle_mi::MIError::Download(e.to_string()))?;
    let elapsed = t0.elapsed();
    eprintln!("Download complete in {elapsed:.2?}");
    let path = outcome.into_inner();
    // BORROW: `path` is a HashMap; grab any value to show the cache root
    if let Some(first) = path.values().next() {
        if let Some(snapshot_dir) = first.parent() {
            eprintln!("Cache path: {}\n", snapshot_dir.display());
        }
    }
    Ok(())
}

/// Read and display key `config.json` fields.
fn inspect_config(model_id: &str) -> candle_mi::Result<()> {
    eprintln!("=== Step 2: Inspect config.json ===");

    let fetch_config = candle_mi::fetch_config_builder()
        .build()
        .map_err(|e| candle_mi::MIError::Download(e.to_string()))?;
    let files =
        hf_fetch_model::download_files_with_config_blocking(model_id.to_owned(), &fetch_config)
            .map_err(|e| candle_mi::MIError::Download(e.to_string()))?
            .into_inner();

    let config_path = files
        .get("config.json")
        .ok_or_else(|| candle_mi::MIError::Config("config.json not found".into()))?;

    let config_str = std::fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to parse config.json: {e}")))?;

    if let Some(mt) = json.get("model_type").and_then(serde_json::Value::as_str) {
        eprintln!("model_type: \"{mt}\"");
        let is_known = candle_mi::SUPPORTED_MODEL_TYPES.contains(&mt);
        eprintln!(
            "Known family: {is_known} (will use {})",
            if is_known {
                "manual parser"
            } else {
                "AUTO-CONFIG"
            }
        );
    }

    for key in &[
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "hidden_act",
    ] {
        if let Some(v) = json.get(*key) {
            eprintln!("  {key}: {v}");
        }
    }
    eprintln!();
    Ok(())
}

/// Load the model via `from_pretrained` and report metadata.
fn load_phase(model_id: &str) -> candle_mi::Result<candle_mi::MIModel> {
    eprintln!("=== Step 3: Load model ===");
    let t1 = std::time::Instant::now();
    let model = candle_mi::MIModel::from_pretrained(model_id)?;
    let elapsed = t1.elapsed();
    eprintln!("Model loaded in {elapsed:.2?}");
    eprintln!(
        "  layers={}, hidden={}, vocab={}, heads={}",
        model.num_layers(),
        model.hidden_size(),
        model.vocab_size(),
        model.num_heads()
    );
    eprintln!("  device: {:?}\n", model.device());
    Ok(model)
}

/// Run a short forward pass and print top-5 predictions.
fn forward_phase(model: &candle_mi::MIModel) -> candle_mi::Result<()> {
    eprintln!("=== Step 4: Forward pass ===");
    let hooks = candle_mi::HookSpec::new();
    let tokens = candle_core::Tensor::new(&[1u32, 2, 3, 4, 5], model.device())?;
    let tokens = tokens.unsqueeze(0)?; // [1, 5]

    let t2 = std::time::Instant::now();
    let cache = model.forward(&tokens, &hooks)?;
    let elapsed = t2.elapsed();
    let logits = cache.output();
    eprintln!("Forward pass succeeded in {elapsed:.2?}");
    eprintln!("  logits shape: {:?}", logits.shape());

    // Show top-5 predictions for the last token
    let last_logits = logits.squeeze(0)?;
    let seq_len = last_logits.dim(0)?;
    // INDEX: seq_len > 0 guaranteed by 5-token input above
    let last_pos = last_logits.get(seq_len - 1)?;
    // PROMOTE: logits may be BF16; extract as F32 for display
    let logits_vec: Vec<f32> = last_pos.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\n  Top-5 token predictions (last position):");
    for (rank, (tok_id, logit)) in indexed.iter().take(5).enumerate() {
        eprintln!("    #{}: token_id={tok_id}, logit={logit:.4}", rank + 1);
    }
    Ok(())
}
