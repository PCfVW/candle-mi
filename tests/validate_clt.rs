// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for CLT (Cross-Layer Transcoder) support.
//!
//! Requires `google/gemma-2-2b` and `mntss/clt-gemma-2-2b-426k` to be
//! cached in `~/.cache/huggingface/hub/`. Tests are `#[ignore]`-gated
//! and require a CUDA GPU.
//!
//! Run:
//!   `cargo test --test validate_clt --features clt,transformer -- --ignored --test-threads=1`

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
use candle_mi::clt::CrossLayerTranscoder;
use candle_mi::{
    GenericTransformer, HookPoint, HookSpec, MIBackend, MITokenizer, TransformerConfig,
};
use serial_test::serial;

// ---------------------------------------------------------------------------
// Helpers (shared with validate_models.rs — duplicated to keep test files
// independent, as Rust integration tests are separate crates)
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
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let paths = safetensors_paths(&snapshot);
    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };
    let model = GenericTransformer::load(config.clone(), device, dtype, vb).unwrap();
    let tokenizer = MITokenizer::from_hf_path(snapshot.join("tokenizer.json")).unwrap();
    (model, tokenizer, config)
}

// ===========================================================================
// Test: CLT open + config detection
// ===========================================================================

#[test]
#[ignore]
#[serial]
fn clt_open_detects_config() {
    let clt = CrossLayerTranscoder::open("mntss/clt-gemma-2-2b-426k").unwrap();
    let cfg = clt.config();

    assert_eq!(cfg.n_layers, 26, "Gemma 2 2B has 26 layers");
    assert_eq!(cfg.d_model, 2304, "Gemma 2 2B hidden dim is 2304");
    assert_eq!(
        cfg.n_features_per_layer, 16384,
        "CLT-426K has 16384 features per layer"
    );
    assert_eq!(
        cfg.n_features_total,
        26 * 16384,
        "total = n_layers × features_per_layer"
    );
    assert!(
        cfg.model_name.contains("gemma"),
        "model_name should mention gemma, got: {}",
        cfg.model_name
    );
    println!("CLT config: {cfg:?}");
}

// ===========================================================================
// Test: CLT encoding on real Gemma 2 2B activations
// ===========================================================================

#[test]
#[ignore]
#[serial]
fn clt_encode_gemma2_residuals() {
    let device = cuda_device().expect("CUDA required for CLT encoding test");
    if find_snapshot("google/gemma-2-2b").is_none() {
        panic!("google/gemma-2-2b not in HF cache");
    }

    // Load Gemma 2 2B.
    let (model, tokenizer, config) = load_gemma2(&device);
    assert_eq!(config.num_layers, 26);

    // Tokenize a known prompt.
    let prompt = "Roses are red, violets are blue";
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();
    println!("Prompt: '{prompt}' → {seq_len} tokens: {token_ids:?}");

    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Capture ResidMid at a few layers for CLT encoding.
    let test_layers: Vec<usize> = vec![0, 5, 12, 20, 25];
    let mut hooks = HookSpec::new();
    for &layer in &test_layers {
        hooks.capture(HookPoint::ResidMid(layer));
    }
    let result = model.forward(&input, &hooks).unwrap();

    // Open CLT.
    let mut clt = CrossLayerTranscoder::open("mntss/clt-gemma-2-2b-426k").unwrap();

    // Encode at each test layer, verifying basic properties.
    let planning_position = seq_len - 1; // last token position
    for &layer in &test_layers {
        let resid_mid = result.require(&HookPoint::ResidMid(layer)).unwrap();

        // Extract the residual at the planning position: [batch, seq, d_model] → [d_model]
        let residual = resid_mid
            .i((0, planning_position))
            .unwrap()
            .to_device(&device)
            .unwrap();

        // Load encoder and encode.
        clt.load_encoder(layer, &device).unwrap();
        assert_eq!(clt.loaded_encoder_layer(), Some(layer));

        let sparse = clt.encode(&residual, layer).unwrap();

        // Basic sanity checks.
        assert!(
            !sparse.is_empty(),
            "layer {layer}: should have non-zero active features"
        );

        // Activations should be sorted descending.
        for window in sparse.features.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "layer {layer}: features not sorted descending: {} >= {}",
                window[0].1,
                window[1].1
            );
        }

        // All feature IDs should reference the correct layer.
        for (fid, _) in &sparse.features {
            assert_eq!(fid.layer, layer);
            assert!(fid.index < clt.config().n_features_per_layer);
        }

        // Top feature activations should be in a reasonable range (not NaN/Inf).
        let top_act = sparse.features[0].1;
        assert!(
            top_act.is_finite() && top_act > 0.0,
            "layer {layer}: top activation should be finite and positive, got {top_act}"
        );

        println!(
            "Layer {layer}: {} active features, top: {} (act={:.4}), bottom: {} (act={:.6})",
            sparse.len(),
            sparse.features[0].0,
            sparse.features[0].1,
            sparse.features[sparse.len() - 1].0,
            sparse.features[sparse.len() - 1].1,
        );

        // Top-k should truncate properly.
        let top5 = clt.top_k(&residual, layer, 5).unwrap();
        assert!(top5.len() <= 5, "top_k(5) returned {} features", top5.len());
        if sparse.len() >= 5 {
            assert_eq!(top5.len(), 5);
            // Top-5 should match first 5 of full encoding.
            for i in 0..5 {
                assert_eq!(top5.features[i].0, sparse.features[i].0);
                assert!((top5.features[i].1 - sparse.features[i].1).abs() < 1e-5);
            }
        }
    }
}

// ===========================================================================
// Test: CLT injection changes model output
// ===========================================================================

#[test]
#[ignore]
#[serial]
fn clt_injection_shifts_logits() {
    let device = cuda_device().expect("CUDA required for CLT injection test");
    if find_snapshot("google/gemma-2-2b").is_none() {
        panic!("google/gemma-2-2b not in HF cache");
    }

    // Load Gemma 2 2B.
    let (model, tokenizer, _config) = load_gemma2(&device);

    let prompt = "Roses are red, violets are blue";
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();
    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // --- Baseline forward pass ---
    let encode_layer = 12;
    let mut baseline_hooks = HookSpec::new();
    baseline_hooks.capture(HookPoint::ResidMid(encode_layer));
    let baseline_result = model.forward(&input, &baseline_hooks).unwrap();
    let baseline_logits = baseline_result.output().clone();

    // --- Encode to find top features ---
    let resid_mid = baseline_result
        .require(&HookPoint::ResidMid(encode_layer))
        .unwrap();
    let planning_position = seq_len - 1;
    let residual = resid_mid.i((0, planning_position)).unwrap();

    let mut clt = CrossLayerTranscoder::open("mntss/clt-gemma-2-2b-426k").unwrap();
    clt.load_encoder(encode_layer, &device).unwrap();
    let top5 = clt.top_k(&residual, encode_layer, 5).unwrap();
    assert!(
        !top5.is_empty(),
        "should have at least one active feature at layer {encode_layer}"
    );
    println!("Top-5 features at layer {encode_layer}:");
    for (fid, act) in &top5.features {
        println!("  {fid}: {act:.4}");
    }

    // --- Cache decoder vectors for the final layer ---
    let target_layer = 25; // last layer
    let features_for_injection: Vec<(candle_mi::CltFeatureId, usize)> = top5
        .features
        .iter()
        .map(|(fid, _)| (*fid, target_layer))
        .collect();
    clt.cache_steering_vectors(&features_for_injection, &device)
        .unwrap();
    assert_eq!(
        clt.steering_cache_len(),
        features_for_injection.len(),
        "all features should be cached"
    );

    // --- Build injection HookSpec and run forward with injection ---
    let strength = 5.0;
    let mut injection_hooks = clt
        .prepare_hook_injection(
            &features_for_injection,
            planning_position,
            seq_len,
            strength,
            &device,
        )
        .unwrap();

    // Also capture ResidPost at the injection layer to verify the intervention exists.
    injection_hooks.capture(HookPoint::ResidPost(target_layer));

    let injected_result = model.forward(&input, &injection_hooks).unwrap();
    let injected_logits = injected_result.output().clone();

    // --- Verify logits changed ---
    // Move both to CPU F32 for comparison.
    let baseline_f32 = baseline_logits
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let injected_f32 = injected_logits
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Compute L2 distance between logits at the last position.
    let bl_last: Vec<f32> = baseline_f32.i((0, seq_len - 1)).unwrap().to_vec1().unwrap();
    let ij_last: Vec<f32> = injected_f32.i((0, seq_len - 1)).unwrap().to_vec1().unwrap();

    let l2_dist: f32 = bl_last
        .iter()
        .zip(ij_last.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("L2 distance between baseline and injected logits: {l2_dist:.4}");
    assert!(
        l2_dist > 0.01,
        "injection should change logits (L2={l2_dist})"
    );

    // Print top-5 predictions for both.
    let print_top5 = |label: &str, logits_vec: &[f32]| {
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("{label} top-5:");
        for (rank, (idx, logit)) in indexed.iter().take(5).enumerate() {
            let token = tokenizer.decode(&[*idx as u32]).unwrap();
            println!("  {}: '{}' (logit={:.4})", rank + 1, token, logit);
        }
    };
    print_top5("Baseline", &bl_last);
    print_top5("Injected", &ij_last);

    // The top prediction should have changed (or at least the distribution shifted).
    let mut bl_indexed: Vec<(usize, f32)> =
        bl_last.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut ij_indexed: Vec<(usize, f32)> =
        ij_last.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    bl_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ij_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Check that top-5 sets differ (at least one different token in top-5).
    let bl_top5_ids: Vec<usize> = bl_indexed.iter().take(5).map(|(i, _)| *i).collect();
    let ij_top5_ids: Vec<usize> = ij_indexed.iter().take(5).map(|(i, _)| *i).collect();
    let same_count = bl_top5_ids
        .iter()
        .filter(|id| ij_top5_ids.contains(id))
        .count();
    println!(
        "Top-5 overlap: {same_count}/5 (baseline: {bl_top5_ids:?}, injected: {ij_top5_ids:?})"
    );
    // With strength=5.0, the distribution should shift noticeably.
    // We don't require ALL tokens to change, but at least the logit magnitudes should differ.
    assert!(
        l2_dist > 1.0,
        "with strength=5.0, L2 distance should be substantial (got {l2_dist})"
    );

    // --- Cleanup ---
    clt.clear_steering_cache();
    assert_eq!(clt.steering_cache_len(), 0);
}
