// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for SAE (Sparse Autoencoder) support.
//!
//! Requires cached in `~/.cache/huggingface/hub/`:
//! - `google/gemma-2-2b`
//! - `jbloom/Gemma-2-2B-Residual-Stream-SAEs`
//!
//! Tests are `#[ignore]`-gated and require a CUDA GPU with **at least 16 GiB VRAM**.
//!
//! Run:
//!   `cargo test --test validate_sae --features sae,transformer -- --ignored --test-threads=1`

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
use candle_mi::sae::SparseAutoencoder;
use candle_mi::{
    GenericTransformer, HookPoint, HookSpec, MIBackend, MITokenizer, TransformerConfig,
};
use serial_test::serial;

// ---------------------------------------------------------------------------
// Helpers (duplicated from validate_clt.rs — Rust integration tests are
// separate crates, so sharing is not straightforward)
// ---------------------------------------------------------------------------

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

fn find_snapshot(model_id: &str) -> Option<std::path::PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

fn cuda_device() -> Option<Device> {
    Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
}

fn safetensors_paths(snapshot: &std::path::Path) -> Vec<std::path::PathBuf> {
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

fn load_gemma2(device: &Device) -> (GenericTransformer, MITokenizer, TransformerConfig) {
    let snapshot =
        find_snapshot("google/gemma-2-2b").expect("google/gemma-2-2b not found in HF cache");
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();
    let dtype = DType::F32;
    let paths = safetensors_paths(&snapshot);
    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };
    let model = GenericTransformer::load(config.clone(), device, dtype, vb).unwrap();
    let tokenizer = MITokenizer::from_hf_path(snapshot.join("tokenizer.json")).unwrap();
    (model, tokenizer, config)
}

// ---------------------------------------------------------------------------
// SAE constants
// ---------------------------------------------------------------------------

const SAE_REPO: &str = "jbloom/Gemma-2-2B-Residual-Stream-SAEs";
const SAE_ID: &str = "gemma-2-2b-res-jb/blocks.0.hook_resid_post";
const HOOK_LAYER: usize = 0;

// ===========================================================================
// Test: SAE loading and config detection
// ===========================================================================

#[test]
#[ignore]
#[serial]
fn sae_load_detects_config() {
    let device = Device::Cpu;
    let sae = SparseAutoencoder::from_pretrained(SAE_REPO, SAE_ID, &device).unwrap();
    let cfg = sae.config();

    assert_eq!(cfg.d_in, 2304, "Gemma 2 2B hidden dim is 2304");
    assert_eq!(cfg.d_sae, 16384, "SAE dictionary size should be 16384");
    assert_eq!(cfg.hook_name, "blocks.0.hook_resid_post");
    assert_eq!(cfg.hook_point, HookPoint::ResidPost(0));

    println!(
        "SAE config: d_in={}, d_sae={}, arch={:?}",
        cfg.d_in, cfg.d_sae, cfg.architecture
    );
}

// ===========================================================================
// Test: SAE encoding on real Gemma 2 2B activations
// ===========================================================================

#[test]
#[ignore]
#[serial]
fn sae_encode_gemma2_residuals() {
    let device = cuda_device().expect("CUDA required for SAE encoding test");

    // Load Gemma 2 2B.
    let (model, tokenizer, _config) = load_gemma2(&device);

    // Tokenize.
    let prompt = "The capital of France is";
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();
    println!("Prompt: '{prompt}' -> {seq_len} tokens: {token_ids:?}");

    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Capture ResidPost at HOOK_LAYER.
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::ResidPost(HOOK_LAYER));
    let result = model.forward(&input, &hooks).unwrap();

    let resid_post = result.require(&HookPoint::ResidPost(HOOK_LAYER)).unwrap(); // [1, seq, 2304]
    println!("resid_post shape: {:?}", resid_post.dims());

    // Load SAE.
    let sae = SparseAutoencoder::from_pretrained(SAE_REPO, SAE_ID, &device).unwrap();
    assert_eq!(sae.d_in(), 2304);

    // --- Dense encode ---
    let encoded = sae.encode(resid_post).unwrap(); // [1, seq, 16384]
    assert_eq!(encoded.dims(), &[1, seq_len, 16384]);

    // Check sparsity: most values should be zero (ReLU activation).
    let encoded_last = encoded.i((0, seq_len - 1)).unwrap(); // [16384]
    let values: Vec<f32> = encoded_last.to_vec1().unwrap();
    let n_active = values.iter().filter(|&&v| v > 0.0).count();
    println!(
        "Active features at last position: {n_active} / {}",
        values.len()
    );

    // SAEs should produce sparse output: at most ~5% active.
    assert!(
        n_active < values.len() / 2,
        "SAE should produce sparse output, got {n_active}/{} active",
        values.len()
    );
    assert!(n_active > 0, "SAE should have at least one active feature");

    // All activations should be non-negative (ReLU).
    assert!(
        values.iter().all(|&v| v >= 0.0),
        "ReLU SAE should produce non-negative activations"
    );

    // --- Sparse encode (single position) ---
    let resid_last = resid_post.i((0, seq_len - 1)).unwrap(); // [2304]
    let sparse = sae.encode_sparse(&resid_last).unwrap();

    assert_eq!(
        sparse.len(),
        n_active,
        "sparse and dense should agree on count"
    );

    // Features should be sorted descending.
    for window in sparse.features.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "features not sorted descending: {} >= {}",
            window[0].1,
            window[1].1
        );
    }

    // Top activation should be finite and positive.
    let top = &sparse.features[0];
    assert!(
        top.1.is_finite() && top.1 > 0.0,
        "top activation should be finite and positive, got {}",
        top.1
    );
    assert!(top.0.index < 16384, "feature index should be < d_sae");

    println!(
        "Top-5 features: {:?}",
        &sparse.features[..5.min(sparse.len())]
    );

    // --- Decode ---
    let decoded = sae.decode(&encoded).unwrap(); // [1, seq, 2304]
    assert_eq!(decoded.dims(), resid_post.dims());

    // --- Reconstruction error ---
    let mse = sae.reconstruction_error(resid_post).unwrap();
    println!("Reconstruction MSE: {mse:.6}");

    // MSE should be reasonable (not zero, not huge).
    assert!(
        mse > 0.0,
        "reconstruction error should be > 0 (lossy encoding)"
    );
    assert!(mse < 100.0, "reconstruction error seems too large: {mse}");

    // --- Decoder vector ---
    let dec_vec = sae.decoder_vector(top.0.index).unwrap();
    assert_eq!(dec_vec.dims(), &[2304]);

    let norm: f32 = dec_vec
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    assert!(
        norm.is_finite() && norm > 0.0,
        "decoder vector should have finite positive norm, got {norm}"
    );
    println!("Top feature {} decoder norm: {norm:.4}", top.0.index);

    // --- Norms for reference ---
    let resid_norm: f32 = resid_last
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let encoded_norm: f32 = encoded_last
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let decoded_last = decoded.i((0, seq_len - 1)).unwrap();
    let decoded_norm: f32 = decoded_last
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();

    println!(
        "Norms — resid: {resid_norm:.4}, encoded: {encoded_norm:.4}, decoded: {decoded_norm:.4}"
    );

    // Decoded norm should be in the same ballpark as original residual norm.
    let ratio = decoded_norm / resid_norm;
    assert!(
        (0.5..2.0).contains(&ratio),
        "decoded/original norm ratio {ratio:.2} is outside [0.5, 2.0]"
    );

    drop(sae);
    drop(result);
    drop(model);
}

// ===========================================================================
// Test: SAE injection changes model output
// ===========================================================================

#[test]
#[ignore]
#[serial]
fn sae_injection_shifts_logits() {
    let device = cuda_device().expect("CUDA required for SAE injection test");

    // Load Gemma 2 2B.
    let (model, tokenizer, _config) = load_gemma2(&device);

    let prompt = "The capital of France is";
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();
    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // --- Baseline forward pass ---
    let mut baseline_hooks = HookSpec::new();
    baseline_hooks.capture(HookPoint::ResidPost(HOOK_LAYER));
    let baseline_result = model.forward(&input, &baseline_hooks).unwrap();
    let baseline_logits = baseline_result.output().clone();

    // --- Find top features to inject ---
    let resid_post = baseline_result
        .require(&HookPoint::ResidPost(HOOK_LAYER))
        .unwrap();
    let resid_last = resid_post.i((0, seq_len - 1)).unwrap();

    let sae = SparseAutoencoder::from_pretrained(SAE_REPO, SAE_ID, &device).unwrap();
    let sparse = sae.encode_sparse(&resid_last).unwrap();

    assert!(
        !sparse.is_empty(),
        "need at least one feature for injection"
    );

    // Inject top feature at large strength to ensure measurable logit shift.
    let top_feature = sparse.features[0].0.index;
    let injection_hooks = sae
        .prepare_hook_injection(&[(top_feature, 50.0)], seq_len - 1, seq_len, &device)
        .unwrap();

    let injected_result = model.forward(&input, &injection_hooks).unwrap();
    let injected_logits = injected_result.output();

    // Compare logits at last position.
    let baseline_last = baseline_logits.i((0, seq_len - 1)).unwrap();
    let injected_last = injected_logits.i((0, seq_len - 1)).unwrap();

    let diff = (&injected_last - &baseline_last).unwrap();
    let max_diff: f32 = diff.abs().unwrap().max(0).unwrap().to_scalar().unwrap();

    println!("Max logit diff from injecting feature {top_feature} at strength 50.0: {max_diff:.4}");
    assert!(
        max_diff > 0.1,
        "injection should shift logits noticeably, got max diff {max_diff}"
    );

    drop(sae);
    drop(baseline_result);
    drop(injected_result);
    drop(model);
}

// ===========================================================================
// Test: SAE vs Python reference (scripts/sae_reference.json)
// ===========================================================================

#[test]
#[ignore]
#[serial]
fn sae_vs_python_reference() {
    let ref_path = std::path::Path::new("scripts/sae_reference.json");
    if !ref_path.exists() {
        println!(
            "SKIP: scripts/sae_reference.json not found. Run scripts/sae_validation.py first."
        );
        return;
    }

    let device = cuda_device().expect("CUDA required for SAE reference test");

    // Parse Python reference.
    let ref_text = std::fs::read_to_string(ref_path).unwrap();
    let reference: serde_json::Value = serde_json::from_str(&ref_text).unwrap();

    let py_d_in = reference["d_in"].as_u64().unwrap() as usize;
    let py_d_sae = reference["d_sae"].as_u64().unwrap() as usize;
    let py_mse = reference["reconstruction_mse"].as_f64().unwrap();
    let py_n_active = reference["n_active_last_pos"].as_u64().unwrap() as usize;
    let py_top_features: Vec<(usize, f64)> = reference["top_features_last_pos"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["index"].as_u64().unwrap() as usize,
                v["value"].as_f64().unwrap(),
            )
        })
        .collect();

    println!(
        "Python reference: d_in={py_d_in}, d_sae={py_d_sae}, MSE={py_mse:.6}, active={py_n_active}"
    );

    // Load Gemma 2 2B.
    let (model, tokenizer, _config) = load_gemma2(&device);

    // Match Python prompt.
    let prompt = reference["prompt"].as_str().unwrap();
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();
    let py_n_tokens = reference["n_tokens"].as_u64().unwrap() as usize;
    assert_eq!(
        seq_len, py_n_tokens,
        "token count mismatch: Rust={seq_len}, Python={py_n_tokens}"
    );

    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Forward pass capturing resid_post at HOOK_LAYER.
    let hook_layer = reference["hook_layer"].as_u64().unwrap() as usize;
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::ResidPost(hook_layer));
    let result = model.forward(&input, &hooks).unwrap();
    let resid_post = result.require(&HookPoint::ResidPost(hook_layer)).unwrap();

    // Load SAE.
    let sae = SparseAutoencoder::from_pretrained(SAE_REPO, SAE_ID, &device).unwrap();
    assert_eq!(sae.d_in(), py_d_in, "d_in mismatch");
    assert_eq!(sae.d_sae(), py_d_sae, "d_sae mismatch");

    // Encode.
    let encoded = sae.encode(resid_post).unwrap();

    // Check active feature count at last position.
    let encoded_last = encoded.i((0, seq_len - 1)).unwrap();
    let values: Vec<f32> = encoded_last.to_vec1().unwrap();
    let n_active = values.iter().filter(|&&v| v > 0.0).count();

    println!("Rust: {n_active} active features, Python: {py_n_active}");
    // Allow some tolerance — float differences can toggle features near threshold.
    let active_diff = (n_active as i64 - py_n_active as i64).unsigned_abs();
    assert!(
        active_diff <= 10,
        "active feature count differs too much: Rust={n_active}, Python={py_n_active}"
    );

    // Compare top features.
    let mut rust_indexed: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .filter(|&(_, v)| *v > 0.0)
        .map(|(i, v)| (i, *v))
        .collect();
    rust_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = py_top_features.len().min(rust_indexed.len());
    println!("\nTop-{top_k} comparison (Rust vs Python):");
    let mut feature_matches = 0;
    for i in 0..top_k {
        let (rust_idx, rust_val) = rust_indexed[i];
        let (py_idx, py_val) = py_top_features[i];
        let match_str = if rust_idx == py_idx { "MATCH" } else { "DIFF" };
        if rust_idx == py_idx {
            feature_matches += 1;
        }
        println!(
            "  #{}: Rust feature {} ({:.4}) vs Python feature {} ({:.4}) — {match_str}",
            i + 1,
            rust_idx,
            rust_val,
            py_idx,
            py_val
        );
    }

    // At least 50% of top features should match (accounting for float differences).
    assert!(
        feature_matches >= top_k / 2,
        "only {feature_matches}/{top_k} top features match between Rust and Python"
    );

    // Reconstruction MSE comparison.
    let rust_mse = sae.reconstruction_error(resid_post).unwrap();
    println!("\nReconstruction MSE — Rust: {rust_mse:.6}, Python: {py_mse:.6}");

    // MSE should be within an order of magnitude (float precision + implementation diffs).
    let mse_ratio = rust_mse / py_mse;
    assert!(
        (0.1..10.0).contains(&mse_ratio),
        "MSE ratio {mse_ratio:.2} is outside [0.1, 10.0]"
    );

    // Compare norms.
    if let Some(py_resid_norm) = reference.get("resid_last_norm").and_then(|v| v.as_f64()) {
        let resid_last = resid_post.i((0, seq_len - 1)).unwrap();
        let rust_resid_norm: f32 = resid_last
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        let norm_diff = (f64::from(rust_resid_norm) - py_resid_norm).abs();
        println!(
            "Residual norm — Rust: {:.4}, Python: {:.4}, diff: {:.6}",
            rust_resid_norm, py_resid_norm, norm_diff
        );
        // F32 residual norms should be very close.
        assert!(
            norm_diff < 1.0,
            "residual norm differs too much: diff={norm_diff:.4}"
        );
    }

    drop(sae);
    drop(result);
    drop(model);
}
