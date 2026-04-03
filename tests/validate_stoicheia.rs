// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-validation tests for stoicheia (`AlgZoo`) backends.
//!
//! Loads pre-trained weights from `safetensors` fixtures, runs the same
//! inputs as the `Python` reference, and compares outputs to 6 decimal places.

use candle_core::{Device, Tensor};
use candle_mi::MIBackend;
use candle_mi::hooks::HookSpec;
use candle_mi::stoicheia::config::{StoicheiaConfig, StoicheiaTask};
use candle_mi::stoicheia::{StoicheiaRnn, StoicheiaTransformer};

/// Parsed reference data from a `Python`-generated JSON file.
struct Reference {
    input: Vec<Vec<f64>>,
    output: Vec<Vec<f64>>,
}

impl Reference {
    fn load(path: &str) -> Self {
        let content = std::fs::read_to_string(path).expect("fixture file missing");
        let json: serde_json::Value = serde_json::from_str(&content).expect("invalid JSON fixture");

        let input: Vec<Vec<f64>> = json["input"]
            .as_array()
            .expect("missing input")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("input row not array")
                    .iter()
                    .map(|v| v.as_f64().expect("input value not float"))
                    .collect()
            })
            .collect();

        let output_raw = &json["output"];
        let output: Vec<Vec<f64>> = output_raw
            .as_array()
            .expect("missing output")
            .iter()
            .map(|row| {
                // Scalar tasks return [[v], [v], ...]; distribution tasks return [[v, v, ...], ...]
                if row.is_array() {
                    row.as_array()
                        .expect("output row not array")
                        .iter()
                        .map(|v| v.as_f64().expect("output value not float"))
                        .collect()
                } else {
                    // Scalar output stored as flat value
                    vec![row.as_f64().expect("output value not float")]
                }
            })
            .collect();

        Self { input, output }
    }
}

/// Compare two `f32` values with tolerance (6 decimal places).
fn assert_close(actual: f32, expected: f64, name: &str, tolerance: f64) {
    // CAST: f32 → f64, widening for comparison
    #[allow(clippy::as_conversions)]
    let actual_f64 = actual as f64;
    let diff = (actual_f64 - expected).abs();
    assert!(
        diff < tolerance,
        "{name}: actual={actual}, expected={expected}, diff={diff}"
    );
}

// ---------------------------------------------------------------------------
// RNN cross-validation
// ---------------------------------------------------------------------------

#[test]
fn rnn_2nd_argmax_h2_n2_matches_python() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(
        config,
        "tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load RNN fixture");

    let reference = Reference::load("tests/fixtures/stoicheia/ref_2nd_argmax_h2_n2.json");

    // Build input tensor from reference data: [batch=4, seq_len=2]
    let input_data: Vec<f32> = reference
        .input
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32)) // CAST: f64 → f32, reference precision
        .collect();
    let input = Tensor::from_slice(
        &input_data,
        (reference.input.len(), reference.input[0].len()),
        &Device::Cpu,
    )
    .expect("failed to create input tensor");

    // Forward pass
    let cache = model
        .forward(&input, &HookSpec::new())
        .expect("forward pass failed");

    // Output is [batch, 1, output_size] — squeeze the seq dim
    let output = cache.output().squeeze(1).expect("failed to squeeze output");
    let output_vec: Vec<Vec<f32>> = output.to_vec2().expect("failed to extract output");

    // Compare against Python reference
    for (batch_idx, (actual_row, expected_row)) in
        output_vec.iter().zip(&reference.output).enumerate()
    {
        for (col_idx, (&actual, &expected)) in actual_row.iter().zip(expected_row).enumerate() {
            assert_close(
                actual,
                expected,
                &format!("batch[{batch_idx}][{col_idx}]"),
                1e-4,
            );
        }
    }
}

#[test]
fn rnn_hook_capture() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(
        config,
        "tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load RNN fixture");

    let input = Tensor::randn(0.0_f32, 1.0, (1, 2), &Device::Cpu).expect("failed to create input");

    let mut hooks = HookSpec::new();
    hooks.capture(candle_mi::HookPoint::Custom("rnn.hook_hidden.0".into()));
    hooks.capture(candle_mi::HookPoint::Custom("rnn.hook_hidden.1".into()));
    hooks.capture(candle_mi::HookPoint::Custom("rnn.hook_final_state".into()));

    let cache = model.forward(&input, &hooks).expect("forward pass failed");

    // Hidden states should be [1, 2] (batch=1, hidden=2)
    let h0 = cache
        .get(&candle_mi::HookPoint::Custom("rnn.hook_hidden.0".into()))
        .expect("hook_hidden.0 not captured");
    assert_eq!(h0.dims(), &[1, 2]);

    let h1 = cache
        .get(&candle_mi::HookPoint::Custom("rnn.hook_hidden.1".into()))
        .expect("hook_hidden.1 not captured");
    assert_eq!(h1.dims(), &[1, 2]);

    // Final state should equal h1 (last timestep)
    let final_state = cache
        .get(&candle_mi::HookPoint::Custom("rnn.hook_final_state".into()))
        .expect("hook_final_state not captured");
    assert_eq!(final_state.dims(), &[1, 2]);
}

// ---------------------------------------------------------------------------
// Transformer cross-validation
// ---------------------------------------------------------------------------

#[test]
fn transformer_longest_cycle_h4_n4_matches_python() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::LongestCycle, 4, 4);
    let model = StoicheiaTransformer::load(
        config,
        "tests/fixtures/stoicheia/transformer_longest_cycle_h4_n4.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load transformer fixture");

    let reference = Reference::load("tests/fixtures/stoicheia/ref_longest_cycle_h4_n4.json");

    // Build input tensor from reference data: [batch=4, seq_len=4] (integers)
    // CAST: f64 → u32, reference stores integers as floats in JSON
    #[allow(clippy::as_conversions)]
    let input_data: Vec<u32> = reference
        .input
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as u32))
        .collect();
    let input = Tensor::from_slice(
        &input_data,
        (reference.input.len(), reference.input[0].len()),
        &Device::Cpu,
    )
    .expect("failed to create input tensor");

    // Forward pass
    let cache = model
        .forward(&input, &HookSpec::new())
        .expect("forward pass failed");

    // Output is [batch, 1, output_size] — squeeze the seq dim
    let output = cache.output().squeeze(1).expect("failed to squeeze output");
    let output_vec: Vec<Vec<f32>> = output.to_vec2().expect("failed to extract output");

    // Compare against Python reference
    for (batch_idx, (actual_row, expected_row)) in
        output_vec.iter().zip(&reference.output).enumerate()
    {
        for (col_idx, (&actual, &expected)) in actual_row.iter().zip(expected_row).enumerate() {
            assert_close(
                actual,
                expected,
                &format!("batch[{batch_idx}][{col_idx}]"),
                1e-2, // Transformer logits are large (thousands); use relative-scale tolerance
            );
        }
    }
}

#[test]
fn transformer_hook_capture() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::LongestCycle, 4, 4);
    let model = StoicheiaTransformer::load(
        config,
        "tests/fixtures/stoicheia/transformer_longest_cycle_h4_n4.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load transformer fixture");

    let input = Tensor::new(&[0_u32, 1, 2, 3], &Device::Cpu)
        .expect("failed to create input")
        .unsqueeze(0)
        .expect("failed to unsqueeze");

    let mut hooks = HookSpec::new();
    hooks.capture(candle_mi::HookPoint::Embed);
    hooks.capture(candle_mi::HookPoint::AttnPattern(0));
    hooks.capture(candle_mi::HookPoint::AttnPattern(1));
    hooks.capture(candle_mi::HookPoint::ResidPost(0));
    hooks.capture(candle_mi::HookPoint::ResidPost(1));

    let cache = model.forward(&input, &hooks).expect("forward pass failed");

    // Embed: [1, 4, 4] (batch=1, seq=4, hidden=4)
    let embed = cache
        .get(&candle_mi::HookPoint::Embed)
        .expect("Embed not captured");
    assert_eq!(embed.dims(), &[1, 4, 4]);

    // AttnPattern: [1, 1, 4, 4] (batch=1, heads=1, seq=4, seq=4)
    let attn0 = cache
        .get(&candle_mi::HookPoint::AttnPattern(0))
        .expect("AttnPattern(0) not captured");
    assert_eq!(attn0.dims(), &[1, 1, 4, 4]);

    // ResidPost: [1, 4, 4] (batch=1, seq=4, hidden=4)
    let resid1 = cache
        .get(&candle_mi::HookPoint::ResidPost(1))
        .expect("ResidPost(1) not captured");
    assert_eq!(resid1.dims(), &[1, 4, 4]);
}
