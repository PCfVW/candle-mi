// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests: load RWKV-7 Goose model from the `HuggingFace` cache
//! and validate forward-pass outputs against Python reference data.
//!
//! These tests require `RWKV/RWKV7-Goose-World3-1.5B-HF` in the local HF cache.
//!
//! Run CPU tests:
//!   `cargo test --test validate_rwkv7 --no-default-features --features rwkv,rwkv-tokenizer`
//!
//! Run all (CPU + GPU):
//!   `cargo test --test validate_rwkv7 --features rwkv,rwkv-tokenizer`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    unsafe_code,
    missing_docs
)]

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::rwkv::{GenericRwkv, RwkvConfig, RwkvVersion};
use candle_mi::{HookPoint, HookSpec, MIBackend, MITokenizer};
use serial_test::serial;

const MODEL_ID: &str = "RWKV/RWKV7-Goose-World3-1.5B-HF";
const VOCAB_FILE: &str = "rwkv_vocab_v20230424.txt";

// ---------------------------------------------------------------------------
// Reference data
// ---------------------------------------------------------------------------

/// Parsed reference data from `scripts/rwkv7_reference.json`.
struct ReferenceData {
    test_prompt: String,
    token_ids: Vec<u32>,
    top_predictions: Vec<(u32, String, f32)>, // (token_id, token_str, logit)
}

fn load_reference() -> ReferenceData {
    let json_str =
        std::fs::read_to_string("scripts/rwkv7_reference.json").expect("reference JSON not found");
    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let test_prompt = json["test_prompt"].as_str().unwrap().to_string();

    let token_ids: Vec<u32> = json["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let top_predictions: Vec<(u32, String, f32)> = json["top_predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let id = p["token_id"].as_u64().unwrap() as u32;
            let token = p["token"].as_str().unwrap().to_string();
            let logit = p["logit"].as_f64().unwrap() as f32;
            (id, token, logit)
        })
        .collect();

    ReferenceData {
        test_prompt,
        token_ids,
        top_predictions,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the `HuggingFace` cache directory.
fn hf_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return std::path::PathBuf::from(cache).join("hub");
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    if let Ok(home) = std::env::var("HOME") {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    panic!("Cannot find HuggingFace cache directory");
}

/// Find the snapshot directory for a given model ID.
fn find_snapshot(model_id: &str) -> Option<std::path::PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Get a CUDA device if available, or None.
fn cuda_device() -> Option<Device> {
    Device::cuda_if_available(0).ok().filter(Device::is_cuda)
}

/// Collect safetensors paths for a model snapshot (single or sharded).
fn safetensors_paths(snapshot: &std::path::Path) -> Vec<std::path::PathBuf> {
    let single = snapshot.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }

    // Sharded: parse model.safetensors.index.json
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

/// Load the RWKV-7 model and tokenizer from the local HF cache.
fn load_rwkv7_on(device: &Device) -> (GenericRwkv, MITokenizer, RwkvConfig) {
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));

    // Parse config
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = RwkvConfig::from_hf_config(&json).unwrap();

    // DType: BF16 for CUDA, F32 for CPU
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    // Resolve safetensors paths
    let paths = safetensors_paths(&snapshot);

    // Load weights (mmap for both single and sharded)
    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };

    // Build model
    let model = GenericRwkv::load(config.clone(), device, dtype, vb).unwrap();

    // Load RWKV World tokenizer
    let vocab_path = snapshot.join(VOCAB_FILE);
    let tokenizer = MITokenizer::from_rwkv_path(&vocab_path).unwrap();

    (model, tokenizer, config)
}

/// Run a forward pass and return top-k `(token_id, token_string, logit)` for the last position.
fn top_k_last_token(
    model: &GenericRwkv,
    tokenizer: &MITokenizer,
    device: &Device,
    prompt: &str,
    k: usize,
) -> Vec<(u32, String, f32)> {
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();

    let input = Tensor::new(&token_ids[..], device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let hooks = HookSpec::new();
    let result = model.forward(&input, &hooks).unwrap();

    let logits = result.output();
    let (batch, out_seq, _vocab) = logits.dims3().unwrap();
    assert_eq!(batch, 1);
    assert_eq!(out_seq, seq_len);

    // Move to CPU F32 for inspection
    let logits_cpu = logits
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Get logits for the last token position
    let last_logits: Vec<f32> = logits_cpu.i((0, seq_len - 1)).unwrap().to_vec1().unwrap();

    // Sort by logit value descending
    let mut indexed: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Decode top-k
    indexed
        .iter()
        .take(k)
        .map(|(idx, logit)| {
            let token = tokenizer.decode(&[*idx as u32]).unwrap();
            (*idx as u32, token, *logit)
        })
        .collect()
}

fn print_top_k(device_name: &str, prompt: &str, top_k: &[(u32, String, f32)]) {
    println!(
        "RWKV-7 ({device_name}) — Top {} for '{prompt}':",
        top_k.len()
    );
    for (rank, (id, token, logit)) in top_k.iter().enumerate() {
        println!("  {}: id={id} '{}' (logit={logit:.4})", rank + 1, token);
    }
}

// ===========================================================================
// Config parsing
// ===========================================================================

#[test]
fn rwkv7_config_parse() {
    let Some(snapshot) = find_snapshot(MODEL_ID) else {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = RwkvConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.num_heads, 32); // 2048 / 64
    assert_eq!(config.vocab_size, 65536);
    assert_eq!(config.version, RwkvVersion::V7);
}

// ===========================================================================
// Tokenizer validation
// ===========================================================================

#[test]
fn rwkv7_tokenizer() {
    let Some(snapshot) = find_snapshot(MODEL_ID) else {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    };

    let vocab_path = snapshot.join(VOCAB_FILE);
    let tokenizer = MITokenizer::from_rwkv_path(&vocab_path).unwrap();

    let reference = load_reference();
    let tokens = tokenizer.encode(&reference.test_prompt).unwrap();

    assert_eq!(
        tokens, reference.token_ids,
        "Token IDs don't match reference for '{}'",
        reference.test_prompt
    );
}

// ===========================================================================
// CPU forward pass
// ===========================================================================

#[test]
fn rwkv7_forward_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_rwkv7_on(&device);
    let reference = load_reference();

    let top_k = top_k_last_token(&model, &tokenizer, &device, &reference.test_prompt, 10);
    print_top_k("CPU", &reference.test_prompt, &top_k);

    // Top-1 should be "if" (token 1942)
    let (top1_id, top1_token, top1_logit) = &top_k[0];
    assert_eq!(
        *top1_id, 1942,
        "Expected top-1 token ID 1942 ('if'), got {top1_id} ('{top1_token}')"
    );

    // Check logit is close to reference (7.559)
    // F32 CPU should be very close to the Python F32 reference
    let ref_logit = reference.top_predictions[0].2;
    let logit_diff = (*top1_logit - ref_logit).abs();
    assert!(
        logit_diff < 1.0,
        "Top-1 logit {top1_logit:.4} differs from reference {ref_logit:.4} by {logit_diff:.4}"
    );

    // Validate top-5 token IDs match reference
    for (rank, (ref_id, ref_token, _ref_logit)) in
        reference.top_predictions.iter().take(5).enumerate()
    {
        let (got_id, got_token, _) = &top_k[rank];
        assert_eq!(
            got_id,
            ref_id,
            "Rank {}: expected token {ref_id} ('{ref_token}'), got {got_id} ('{got_token}')",
            rank + 1
        );
    }
}

// ===========================================================================
// GPU forward pass
// ===========================================================================

#[test]
#[serial]
fn rwkv7_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device");
        return;
    };

    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let (model, tokenizer, _config) = load_rwkv7_on(&device);
    let reference = load_reference();

    let top_k = top_k_last_token(&model, &tokenizer, &device, &reference.test_prompt, 10);
    print_top_k("GPU", &reference.test_prompt, &top_k);

    // Top-1 should be "if" (token 1942)
    let (top1_id, top1_token, top1_logit) = &top_k[0];
    assert_eq!(
        *top1_id, 1942,
        "Expected top-1 token ID 1942 ('if'), got {top1_id} ('{top1_token}')"
    );

    // BF16 has lower precision, so allow wider tolerance
    let ref_logit = reference.top_predictions[0].2;
    let logit_diff = (*top1_logit - ref_logit).abs();
    assert!(
        logit_diff < 2.0,
        "Top-1 logit {top1_logit:.4} differs from reference {ref_logit:.4} by {logit_diff:.4}"
    );

    // Top-5 token IDs should match
    for (rank, (ref_id, ref_token, _ref_logit)) in
        reference.top_predictions.iter().take(5).enumerate()
    {
        let (got_id, got_token, _) = &top_k[rank];
        assert_eq!(
            got_id,
            ref_id,
            "Rank {}: expected token {ref_id} ('{ref_token}'), got {got_id} ('{got_token}')",
            rank + 1
        );
    }
}

// ===========================================================================
// Hook capture: RwkvState + RwkvDecay shape
// ===========================================================================

#[test]
fn rwkv7_hook_capture_state() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, config) = load_rwkv7_on(&device);
    let reference = load_reference();

    let token_ids = tokenizer.encode(&reference.test_prompt).unwrap();
    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Capture RwkvState, RwkvDecay, and ResidPre at layer 0
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::RwkvState(0));
    hooks.capture(HookPoint::RwkvDecay(0));
    hooks.capture(HookPoint::ResidPre(0));

    let result = model.forward(&input, &hooks).unwrap();

    // RwkvState should be [batch, num_heads, head_dim, head_dim]
    let state = result.require(&HookPoint::RwkvState(0)).unwrap();
    let state_dims = state.dims4().unwrap();
    assert_eq!(state_dims.0, 1, "batch");
    assert_eq!(state_dims.1, config.num_heads, "num_heads");
    assert_eq!(state_dims.2, config.head_dim, "head_dim");
    assert_eq!(state_dims.3, config.head_dim, "head_dim");

    // RwkvDecay should be [batch, seq_len, num_heads, head_dim]
    let decay = result.require(&HookPoint::RwkvDecay(0)).unwrap();
    let decay_dims = decay.dims4().unwrap();
    assert_eq!(decay_dims.0, 1, "batch");
    assert_eq!(decay_dims.1, token_ids.len(), "seq_len");
    assert_eq!(decay_dims.2, config.num_heads, "num_heads");
    assert_eq!(decay_dims.3, config.head_dim, "head_dim");

    // ResidPre should be [batch, seq_len, hidden_size]
    let resid = result.require(&HookPoint::ResidPre(0)).unwrap();
    let resid_dims = resid.dims3().unwrap();
    assert_eq!(resid_dims.0, 1, "batch");
    assert_eq!(resid_dims.1, token_ids.len(), "seq_len");
    assert_eq!(resid_dims.2, config.hidden_size, "hidden_size");
}
