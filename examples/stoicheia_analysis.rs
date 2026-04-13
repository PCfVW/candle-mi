// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stoicheia MI analysis — full Phase B pipeline on an `AlgZoo` RNN.
//!
//! ```bash
//! cargo run --features stoicheia --release --example stoicheia_analysis -- \
//!     --weights path/to/rnn_2nd_argmax_h2_n2.safetensors \
//!     --hidden-size 2 --seq-len 2
//! ```
//!
//! **What it does:**
//!
//! 1. Loads an `AlgZoo` RNN from a `safetensors` file
//! 2. Extracts raw `f32` weights (fast-path kernel)
//! 3. Standardizes weights so `|W_ih[j]| = 1`
//! 4. Measures baseline accuracy on random inputs
//! 5. Runs single-neuron ablation sweep
//! 6. Probes each neuron's functional role
//! 7. Enumerates piecewise-linear regions
//! 8. Runs surprise accounting with an oracle estimator

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use std::time::Instant;

use candle_core::Device;
use candle_mi::stoicheia::StoicheiaRnn;
use candle_mi::stoicheia::ablation;
use candle_mi::stoicheia::config::{StoicheiaConfig, StoicheiaTask};
use candle_mi::stoicheia::fast::{self, RnnWeights};
use candle_mi::stoicheia::piecewise;
use candle_mi::stoicheia::probing;
use candle_mi::stoicheia::standardize;
use candle_mi::stoicheia::surprise;
use clap::Parser;

/// Full MI analysis of an AlgZoo RNN (Phase B pipeline).
#[derive(Parser)]
#[command(name = "stoicheia_analysis")]
#[command(about = "Full MI analysis pipeline for AlgZoo RNNs")]
struct Args {
    /// Path to the `.safetensors` weight file.
    #[arg(long)]
    weights: String,

    /// Hidden dimension.
    #[arg(long, default_value = "2")]
    hidden_size: usize,

    /// Sequence length.
    #[arg(long, default_value = "2")]
    seq_len: usize,

    /// Number of random samples for analysis.
    #[arg(long, default_value = "1000")]
    samples: usize,

    /// Number of probe inputs per probe type.
    #[arg(long, default_value = "500")]
    probes: usize,

    /// Suppress runtime reporting.
    #[arg(long)]
    no_runtime: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();
    let total_start = Instant::now();

    let config =
        StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, args.hidden_size, args.seq_len);
    println!("Model: {config}");
    println!("Weights: {}", args.weights);

    // 1. Load model + extract raw weights
    let t0 = Instant::now();
    let model = StoicheiaRnn::load(config.clone(), &args.weights, &Device::Cpu)?;
    let weights = RnnWeights::from_model(&model)?;
    if !args.no_runtime {
        println!("  Load time: {:.2?}", t0.elapsed());
    }

    // 2. Standardize
    let t0 = Instant::now();
    let std_rnn = standardize::standardize_rnn(&model)?;
    let quality = standardize::standardization_quality(&std_rnn);
    if !args.no_runtime {
        println!("  Standardize time: {:.2?}", t0.elapsed());
    }
    println!("\n=== Weight Standardization ===");
    println!("  Max deviation from +/-1: {quality:.8}");
    println!("  Sign table (W_ih):");
    for (j, &w) in std_rnn.weight_ih.iter().enumerate() {
        let sign = if w > 0.0 { "+" } else { "-" };
        println!("    Neuron {j}: {sign}1 (exact: {w:.6})");
    }

    // 3. Generate test inputs and targets
    let n = args.samples;
    // CAST: usize → f32, small test indices
    #[allow(clippy::as_conversions)]
    let inputs: Vec<f32> = (0..n * config.seq_len)
        .map(|i| ((i as f32) * 0.618_034).sin() * 3.0)
        .collect();
    let targets = compute_targets(&inputs, n, config.seq_len);

    // 4. Baseline accuracy
    let t0 = Instant::now();
    let baseline = fast::accuracy(&weights, &inputs, &targets, n, &config)?;
    if !args.no_runtime {
        println!("  Accuracy time: {:.2?}", t0.elapsed());
    }
    println!("\n=== Accuracy ===");
    println!("  Baseline: {:.2}% ({n} samples)", baseline * 100.0);

    // 5. Ablation sweep
    let t0 = Instant::now();
    let sweep = ablation::ablate_neurons(&weights, &inputs, &targets, n, &config)?;
    if !args.no_runtime {
        println!("  Ablation time: {:.2?}", t0.elapsed());
    }
    println!("\n=== Neuron Ablation ===");
    for r in &sweep.results {
        println!(
            "  Neuron {}: accuracy {:.2}% (delta {:+.2}%)",
            r.neuron,
            r.ablated_accuracy * 100.0,
            r.accuracy_delta * 100.0
        );
    }

    // 6. Neuron probing
    let t0 = Instant::now();
    let probe_report = probing::probe_neurons(&weights, &config, args.probes)?;
    if !args.no_runtime {
        println!("  Probing time: {:.2?}", t0.elapsed());
    }
    println!("\n=== Neuron Roles ===");
    for p in &probe_report.neurons {
        println!(
            "  Neuron {}: {:?} (correlation: {:.3})",
            p.neuron, p.role, p.correlation
        );
    }

    // 7. Region enumeration
    let t0 = Instant::now();
    let region_map = piecewise::classify_regions(&weights, &inputs, n, &config)?;
    if !args.no_runtime {
        println!("  Region classification time: {:.2?}", t0.elapsed());
    }
    println!("\n=== Piecewise-Linear Regions ===");
    println!("  Distinct regions: {}", region_map.regions.len());
    for (i, r) in region_map.regions.iter().enumerate().take(10) {
        println!(
            "  Region {i}: {}/{n} inputs ({:.1}%), {} active neurons",
            r.count,
            r.count as f32 / n as f32 * 100.0,
            r.pattern.count_active()
        );
    }

    // 8. Surprise accounting
    let t0 = Instant::now();
    let oracle = surprise::OracleEstimator::new(StoicheiaTask::SecondArgmax, config.seq_len);
    let report = surprise::surprise_accounting(&weights, &oracle, &config, n)?;
    if !args.no_runtime {
        println!("  Surprise accounting time: {:.2?}", t0.elapsed());
    }
    println!("\n=== Surprise Accounting ===");
    println!("  Estimator: {}", report.estimator_description);
    println!("  Model accuracy:    {:.2}%", report.model_accuracy * 100.0);
    println!(
        "  Estimate accuracy: {:.2}%",
        report.estimate_accuracy * 100.0
    );
    println!("  Agreement rate:    {:.2}%", report.agreement_rate * 100.0);
    println!(
        "  Disagreement rate: {:.2}%",
        report.disagreement_rate * 100.0
    );
    println!(
        "  Chance accuracy:   {:.2}%",
        report.chance_accuracy * 100.0
    );
    println!("  Parameters:        {}", report.param_count);

    // Summary
    let total_time = total_start.elapsed();
    if args.no_runtime {
        println!("\n=== Done ===");
    } else {
        println!("\n=== Done ({total_time:.2?}) ===");
    }

    Ok(())
}

/// Compute second-argmax targets from flat inputs.
fn compute_targets(inputs: &[f32], n: usize, seq_len: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let start = i * seq_len;
            let end = start + seq_len;
            // INDEX: bounded by n * seq_len = inputs.len()
            #[allow(clippy::indexing_slicing)]
            let slice = &inputs[start..end];

            // Find position of second-largest value
            let mut max_pos = 0;
            let mut second_pos = 0;
            let mut max_val = f32::NEG_INFINITY;
            let mut second_val = f32::NEG_INFINITY;
            for (j, &x) in slice.iter().enumerate() {
                if x > max_val {
                    second_val = max_val;
                    second_pos = max_pos;
                    max_val = x;
                    max_pos = j;
                } else if x > second_val {
                    second_val = x;
                    second_pos = j;
                }
            }
            let _ = (max_pos, second_val);
            // CAST: usize → u32, positions ≤ seq_len ≤ 10
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            {
                second_pos as u32
            }
        })
        .collect()
}
