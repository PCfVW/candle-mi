// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: loading + running a **bitsandbytes NF4** quantized
//! checkpoint via candle-mi's anamnesis-backed dequant path (`quantized`
//! feature), validated against a PyTorch + bitsandbytes oracle.
//!
//! candle-mi cannot consume quantized weights directly (candle_nn loads plain
//! float safetensors).  With the `quantized` feature, `from_pretrained` detects
//! `quantization_config` and routes the weights through anamnesis
//! (`parse` → `remember_to_bytes(BF16)`) before building the `VarBuilder`.
//!
//! Target: `medmekk/Llama-3.2-1B-Instruct-bnb-nf4` — NF4, single-file, and a
//! **llama** model (a family already exact-parity-validated), so a mismatch
//! isolates to the dequant, not the forward.
//!
//! The oracle (`scripts/bnb_nf4_validation.py`) runs the *same* NF4 checkpoint
//! through bitsandbytes in **F32** (the checkpoint's `bnb_4bit_compute_dtype`).
//! candle-mi loads it through anamnesis, which dequantizes NF4 → **BF16**, so
//! the bar is a **BF16-weight tier** (~1.0 at logit ≈ 20), not the ~1e-5 of
//! full-precision families — the residual is BF16-vs-F32 weight precision, not a
//! dequant error (anamnesis's own cross-validation proves its NF4 decode is
//! bit-exact to bitsandbytes at BF16).  What this proves: candle-mi loads a real
//! quantized checkpoint and runs it correctly — **exact top-1 match** on every
//! prompt, magnitudes within the BF16-weight bar.
//!
//! The quant repo ships no tokenizer, so the oracle records the token ids and
//! the Rust side feeds them directly (as the other `validate_*_forward` tests
//! do), keeping tokenization out of the comparison.
//!
//! `#[ignore]`-gated; requires the model cached, a CUDA device, and the
//! `quantized` feature.
//!
//! Run:
//!   `cargo test --test validate_bnb_loading --features transformer,quantized -- --ignored --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    missing_docs
)]

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{HookSpec, MIBackend, MIModel};

const MODEL_ID: &str = "medmekk/Llama-3.2-1B-Instruct-bnb-nf4";
/// anamnesis dequantizes NF4 → **BF16** (its `TargetDtype` is BF16-only), so
/// candle loads BF16-precision weights, while the F32 oracle dequantizes NF4 →
/// F32 directly.  anamnesis's own cross-validation proves its BF16 output is
/// bit-exact to bitsandbytes-at-BF16, so the residual here is purely BF16-vs-F32
/// weight precision propagated through 16 layers (~1.0 at logit ≈ 20, observed
/// max ≈ 1.01) — the inherent floor of a BF16 dequant path, not a bug.  The
/// headline correctness guarantee is the **exact top-1 match** on every prompt.
const ABS_DIFF_BAR: f32 = 1.5;

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("bnb_nf4_forward_reference.json")
}

fn model_is_cached(model_id: &str) -> bool {
    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return false;
    };
    let dir = std::path::Path::new(&home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")));
    dir.exists()
}

#[test]
#[ignore = "requires medmekk/Llama-3.2-1B-Instruct-bnb-nf4 cached, a CUDA device, and the `quantized` feature; run with --ignored"]
fn bnb_nf4_llama_forward_parity() {
    if !model_is_cached(MODEL_ID) {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let reference_str = std::fs::read_to_string(reference_path())
        .expect("read bnb_nf4_forward_reference.json — run scripts/bnb_nf4_validation.py first");
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();
    assert_eq!(reference["model_repo"].as_str().unwrap(), MODEL_ID);
    let ref_vocab = reference["vocab_size"].as_u64().unwrap() as usize;
    let test_cases = reference["test_cases"].as_array().unwrap();

    // Loads via the anamnesis dequant path (quantization_config → NF4 → BF16).
    let model =
        MIModel::from_pretrained(MODEL_ID).expect("load bnb-NF4 model via anamnesis dequant path");

    println!("Validating bnb-NF4 forward parity (anamnesis dequant) vs bitsandbytes oracle:");
    println!("  model:  {MODEL_ID}");
    println!(
        "  num_layers={}, hidden_size={}, vocab_size={}",
        model.num_layers(),
        model.hidden_size(),
        model.vocab_size()
    );
    assert_eq!(model.vocab_size(), ref_vocab);

    let device = model.device().clone();
    let mut max_abs_diff: f32 = 0.0;
    let mut failures: Vec<String> = Vec::new();

    for tc in test_cases {
        let prompt = tc["prompt"].as_str().unwrap();
        let ref_tokens: Vec<u32> = tc["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let ref_top10 = tc["top_10"].as_array().unwrap();

        let input = Tensor::new(&ref_tokens[..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hooks = HookSpec::new();
        let cache = model.forward(&input, &hooks).unwrap();
        let logits = cache.output();

        let (batch, out_seq, vocab) = logits.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(out_seq, ref_tokens.len());
        assert_eq!(vocab, ref_vocab);

        let last: Vec<f32> = logits
            .i((0, out_seq - 1))
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut indexed: Vec<(usize, f32)> =
            last.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ref_top1_idx = ref_top10[0]["index"].as_u64().unwrap() as usize;
        let ref_top1_logit = ref_top10[0]["logit"].as_f64().unwrap() as f32;
        println!("\nPrompt: {prompt:?}  ({} tokens)", ref_tokens.len());
        println!(
            "  bnb oracle top-1: ({ref_top1_idx}, {ref_top1_logit:.4})   \
             candle top-1: ({}, {:.4})",
            indexed[0].0, indexed[0].1
        );

        // Top-1 token must match (the headline correctness check for bf16).
        if indexed[0].0 != ref_top1_idx {
            failures.push(format!(
                "{prompt:?}: top-1 mismatch (candle {}, oracle {ref_top1_idx})",
                indexed[0].0
            ));
        }

        // Magnitudes: compare per *token* (candle's logit for the oracle's
        // token), so a bf16 reordering of near-tied ranks does not inflate.
        let mut prompt_max_diff: f32 = 0.0;
        for ref_item in ref_top10 {
            let ref_idx = ref_item["index"].as_u64().unwrap() as usize;
            let ref_logit = ref_item["logit"].as_f64().unwrap() as f32;
            let diff = (last[ref_idx] - ref_logit).abs();
            if diff >= ABS_DIFF_BAR {
                failures.push(format!(
                    "{prompt:?} token {ref_idx}: logit abs-diff {diff:.3e} >= {ABS_DIFF_BAR:.2} \
                     (candle {:.4}, oracle {ref_logit:.4})",
                    last[ref_idx]
                ));
            }
            prompt_max_diff = prompt_max_diff.max(diff);
            max_abs_diff = max_abs_diff.max(diff);
        }
        println!("  max abs-diff over top-10 tokens: {prompt_max_diff:.3e}");
    }

    println!(
        "\n{} test cases; max abs-diff across all top-10 tokens = {:.3e} (bar: {:.2})",
        test_cases.len(),
        max_abs_diff,
        ABS_DIFF_BAR
    );

    assert!(
        failures.is_empty(),
        "bnb-NF4 forward parity FAILED ({} divergences):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
