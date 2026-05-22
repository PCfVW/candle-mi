// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `Qwen/Qwen3-1.7B-Base` forward-pass parity against the
//! from-first-principles Python oracle in `scripts/qwen3_forward_validation.py`.
//!
//! Consumes the frozen reference JSON (`scripts/qwen3_forward_reference.json`)
//! and verifies that candle-mi's [`GenericTransformer`] produces matching
//! output when fed the same input prompts.  Acceptance bar:
//!
//! - Detected config has `use_qk_norm == true`.
//! - `(hidden_size, num_layers, vocab_size, head_dim)` match the Python run.
//! - Per test case: top-10 logit indices match exactly; magnitudes within
//!   `abs diff < 1e-3` (CPU vs CPU `F32`) or `< 5e-3` (GPU `F32` vs CPU
//!   `F32` — looser to absorb CUDA-vs-CPU rounding noise documented for
//!   RWKV-7).
//!
//! Two test wrappers (one CPU, one GPU), both `#[ignore]`-gated and serial.
//! GPU test skips cleanly when no `CUDA` device is available.
//!
//! Requires `Qwen/Qwen3-1.7B-Base` (~3.2 GiB) cached in
//! `~/.cache/huggingface/hub/`.
//!
//! Run CPU:
//!   `cargo test --test validate_qwen3_forward --features transformer -- --ignored qwen3_1_7b_forward_parity_cpu`
//!
//! Run GPU:
//!   `cargo test --test validate_qwen3_forward --features transformer -- --ignored qwen3_1_7b_forward_parity_gpu`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    unsafe_code,
    missing_docs
)]

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{GenericTransformer, HookSpec, MIBackend, TransformerConfig};
use serial_test::serial;

const MODEL_ID: &str = "Qwen/Qwen3-1.7B-Base";
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("qwen3_forward_reference.json")
}

fn hf_cache_dir() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return PathBuf::from(cache).join("hub");
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    panic!("Cannot find HuggingFace cache directory");
}

fn find_snapshot(model_id: &str) -> Option<PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

fn safetensors_paths(snapshot: &std::path::Path) -> Vec<PathBuf> {
    let single = snapshot.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }
    let index_path = snapshot.join("model.safetensors.index.json");
    let index_str = std::fs::read_to_string(&index_path).unwrap_or_else(|_| {
        panic!(
            "no model.safetensors or index.json in {}",
            snapshot.display()
        )
    });
    let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    let mut shard_names: Vec<String> = weight_map
        .values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    shard_names.sort();
    shard_names.dedup();
    shard_names.iter().map(|name| snapshot.join(name)).collect()
}

fn cuda_device() -> Option<Device> {
    Device::cuda_if_available(0).ok().filter(Device::is_cuda)
}

/// Run the Qwen3 forward-parity check on the given `device`.  Panics on any
/// mismatch.  `abs_diff_bar` is the per-rank acceptance threshold for the
/// top-10 logit magnitudes (CPU is tighter than GPU).
#[allow(clippy::too_many_lines)] // Flat sequence — load → assert → iterate cases.
fn run_qwen3_forward_parity(device: &Device, device_name: &str, abs_diff_bar: f32) {
    // --- Load the frozen reference JSON ---
    let reference_str = std::fs::read_to_string(reference_path()).expect(
        "failed to read qwen3_forward_reference.json — run scripts/qwen3_forward_validation.py first",
    );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let model_repo = reference["model_repo"].as_str().unwrap();
    let ref_hidden = reference["hidden_size"].as_u64().unwrap() as usize;
    let ref_layers = reference["num_layers"].as_u64().unwrap() as usize;
    let ref_vocab = reference["vocab_size"].as_u64().unwrap() as usize;
    let ref_head_dim = reference["head_dim"].as_u64().unwrap() as usize;
    let ref_use_qk_norm = reference["use_qk_norm"].as_bool().unwrap();
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(model_repo, MODEL_ID, "oracle JSON model_repo mismatch");
    assert!(
        ref_use_qk_norm,
        "Qwen3 reference must report use_qk_norm == true"
    );

    println!("Validating Qwen3 forward parity ({device_name}) against Python oracle:");
    println!("  model:  {model_repo}");
    println!(
        "  hidden_size={ref_hidden}, num_layers={ref_layers}, \
         vocab_size={ref_vocab}, head_dim={ref_head_dim}"
    );
    println!(
        "  {} test cases, abs-diff bar = {abs_diff_bar:.0e}",
        test_cases.len()
    );

    // --- Load model from HF cache ---
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    // Sanity: candle-mi must detect QK norm.
    assert!(
        config.use_qk_norm,
        "candle-mi config parsing must detect use_qk_norm=true for Qwen3"
    );
    assert_eq!(config.hidden_size, ref_hidden);
    assert_eq!(config.num_layers, ref_layers);
    assert_eq!(config.vocab_size, ref_vocab);
    assert_eq!(config.head_dim, ref_head_dim);

    let dtype = DType::F32;
    let paths = safetensors_paths(&snapshot);

    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };
    let model = GenericTransformer::load(config.clone(), device, dtype, vb).unwrap();

    assert_eq!(model.num_layers(), ref_layers);
    assert_eq!(model.hidden_size(), ref_hidden);
    assert_eq!(model.vocab_size(), ref_vocab);

    // --- Run each test case ---
    let mut max_abs_diff: f32 = 0.0;
    for tc in test_cases {
        let prompt = tc["prompt"].as_str().unwrap();
        let ref_tokens: Vec<u32> = tc["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let ref_top10 = tc["top_10"].as_array().unwrap();

        println!("\nPrompt: {prompt:?}");
        println!("  expected tokens: {ref_tokens:?}");

        // Use the Python-tokenized IDs directly so any tokenizer-version drift
        // doesn't taint the forward-pass comparison.
        let input = Tensor::new(&ref_tokens[..], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let hooks = HookSpec::new();
        let result = model.forward(&input, &hooks).unwrap();
        let logits = result.output();

        let (batch, out_seq, vocab) = logits.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(out_seq, ref_tokens.len());
        assert_eq!(vocab, ref_vocab);

        // Last-token logits, F32 on CPU.
        let last_logits: Vec<f32> = logits
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .i((0, out_seq - 1))
            .unwrap()
            .to_vec1()
            .unwrap();

        // Compute Rust's top-10.
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Compare top-10 indices + magnitudes.
        for (rank, ref_item) in ref_top10.iter().enumerate() {
            let ref_idx = ref_item["index"].as_u64().unwrap() as usize;
            let ref_logit = ref_item["logit"].as_f64().unwrap() as f32;

            let (rust_idx, rust_logit) = indexed[rank];

            assert_eq!(
                rust_idx, ref_idx,
                "rank {rank}: top-10 index mismatch (Rust {rust_idx}, Python {ref_idx}) \
                 for prompt {prompt:?}"
            );

            let diff = (rust_logit - ref_logit).abs();
            assert!(
                diff < abs_diff_bar,
                "rank {rank}: logit abs-diff {diff:.2e} >= {abs_diff_bar:.0e} \
                 (Rust {rust_logit}, Python {ref_logit}) for prompt {prompt:?}"
            );
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }

        println!(
            "  Rust top-1: ({}, {:.4}) — matches Python",
            indexed[0].0, indexed[0].1
        );
    }

    println!(
        "\nAll {} test cases passed on {device_name}; max abs-diff across all top-10 logits = {:.2e} (bar: {:.0e})",
        test_cases.len(),
        max_abs_diff,
        abs_diff_bar
    );
}

#[test]
#[ignore = "requires Qwen/Qwen3-1.7B-Base cached (~3.2 GiB); run with --ignored"]
#[serial]
fn qwen3_1_7b_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_qwen3_forward_parity(&Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

#[test]
#[ignore = "requires Qwen/Qwen3-1.7B-Base cached (~3.2 GiB) and a CUDA device; run with --ignored"]
#[serial]
fn qwen3_1_7b_forward_parity_gpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: no CUDA device available");
            return;
        }
    };
    run_qwen3_forward_parity(&device, "CUDA", ABS_DIFF_BAR_GPU);
}
