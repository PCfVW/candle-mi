// SPDX-License-Identifier: MIT OR Apache-2.0

//! Character count helix: replicate the core finding from
//! [Gurnee et al. (2025)](https://transformer-circuits.pub/2025/linebreaks/index.html)
//! "When Models Manipulate Manifolds" (Transformer Circuits).
//!
//! ```bash
//! # Run with default model (Gemma 2 2B, requires mmap for sharded weights)
//! cargo run --release --features transformer,mmap --example character_count_helix
//!
//! # Specify a different model (non-sharded models don't need mmap)
//! cargo run --release --features transformer --example character_count_helix -- "meta-llama/Llama-3.2-1B"
//!
//! # Export JSON for external plotting
//! cargo run --release --features transformer,mmap --example character_count_helix -- --output results.json
//!
//! # Analyse a specific layer
//! cargo run --release --features transformer,mmap --example character_count_helix -- --layer 2
//!
//! # Compare variance across layers 0-3
//! cargo run --release --features transformer,mmap --example character_count_helix -- --all-layers
//!
//! # Use your own prose file
//! cargo run --release --features transformer,mmap --example character_count_helix -- --text mytext.txt
//! ```
//!
//! **What it does:**
//!
//! 1. Loads a transformer model and a prose passage (built-in or from `--text`).
//! 2. Strips newlines and re-wraps at widths 20, 30, 40, ... 150 characters.
//! 3. For each width, runs a forward pass capturing
//!    [`HookPoint::ResidPost`](candle_mi::HookPoint::ResidPost) at the target
//!    layer, then computes each token's line character count from byte offsets.
//! 4. Averages residual-stream vectors by character count (1..=150).
//! 5. Runs [`pca_top_k`](candle_mi::pca_top_k) on the mean vectors (6 PCs).
//! 6. Computes a 150x150 cosine-similarity matrix on the mean vectors.
//! 7. Prints a summary and optionally writes structured JSON output.
//!
//! The JSON output can be visualised with the companion Mathematica script
//! `examples/results/character_count_helix/helix_plot.wl`.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use candle_mi::{HookPoint, HookSpec, MIModel, MITokenizer, PcaResult, pca_top_k};
use clap::Parser;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "character_count_helix")]
#[command(about = "Character count helix: replicate the manifold geometry finding")]
struct Args {
    /// `HuggingFace` model ID (default: google/gemma-2-2b)
    model: Option<String>,

    /// Write structured JSON output to this file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Which layer to capture (0-indexed, default: 1)
    #[arg(long, default_value_t = 1)]
    layer: usize,

    /// Compare explained variance across layers 0-3
    #[arg(long)]
    all_layers: bool,

    /// Path to a plain-text file to use as prose input (default: built-in passage)
    #[arg(long)]
    text: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HelixOutput {
    model_id: String,
    layer: usize,
    max_char_count: usize,
    line_widths: Vec<usize>,
    explained_variance: Vec<f32>,
    total_variance_top6: f32,
    projections: Vec<Projection>,
    cosine_similarity: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct Projection {
    char_count: usize,
    pc: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Prose passage
// ---------------------------------------------------------------------------

/// Public-domain prose for wrapping experiments.
///
/// Source: adapted from freely available scientific and encyclopaedic text.
/// All ASCII, no special characters. Approximately 500 words.
const fn prose_passage() -> &'static str {
    concat!(
        "The surface of the Earth is divided into several large regions ",
        "called tectonic plates. These plates float on the semi-fluid ",
        "asthenosphere beneath them and move very slowly over geological ",
        "time. The boundaries where plates meet are sites of significant ",
        "geological activity including earthquakes and volcanic eruptions. ",
        "Mountains form when two continental plates collide and push the ",
        "crust upward. The Himalayas for example arose from the collision ",
        "between the Indian plate and the Eurasian plate millions of years ",
        "ago and continue to rise by a few millimetres each year. Ocean ",
        "trenches mark places where one plate slides beneath another in a ",
        "process called subduction. The Mariana Trench in the western ",
        "Pacific is the deepest known point on Earth reaching nearly eleven ",
        "kilometres below sea level. Mid-ocean ridges are underwater ",
        "mountain chains where new oceanic crust is created as magma rises ",
        "from the mantle and solidifies. The Mid-Atlantic Ridge runs down ",
        "the centre of the Atlantic Ocean and separates the North American ",
        "plate from the Eurasian plate in the north and the South American ",
        "plate from the African plate in the south. Transform boundaries ",
        "occur where plates slide horizontally past each other. The San ",
        "Andreas Fault in California is a well-known transform boundary ",
        "between the Pacific plate and the North American plate. Plate ",
        "tectonics also explains the distribution of fossils across ",
        "continents. Identical fossil species found on separate continents ",
        "provide evidence that those landmasses were once joined together ",
        "in a supercontinent called Pangaea which began to break apart ",
        "roughly two hundred million years ago. The movement of plates ",
        "influences ocean currents and climate patterns over long periods. ",
        "When continents drift toward the poles ice sheets can form and ",
        "global temperatures may drop leading to ice ages. Conversely when ",
        "most landmasses cluster near the equator the planet tends to be ",
        "warmer overall. Volcanic activity associated with plate boundaries ",
        "releases gases into the atmosphere which can affect global climate ",
        "on shorter timescales. Large eruptions inject sulfur dioxide into ",
        "the stratosphere where it forms aerosol particles that reflect ",
        "sunlight and temporarily cool the planet. The eruption of Mount ",
        "Tambora in eighteen fifteen caused the so-called year without a ",
        "summer in eighteen sixteen when crops failed across Europe and ",
        "North America. Scientists monitor plate movements using satellite ",
        "based systems and networks of seismometers. These measurements ",
        "help predict where future earthquakes are most likely to occur ",
        "and inform building codes in vulnerable regions. Understanding ",
        "plate tectonics is fundamental to geology geophysics and the ",
        "study of natural hazards. It provides the framework for explaining ",
        "why earthquakes volcanoes and mountain ranges are found where they ",
        "are and how the surface of our planet has changed over billions of ",
        "years of geological history.",
    )
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();
    let model_id = args.model.as_deref().unwrap_or("google/gemma-2-2b");

    println!("=== Character Count Helix ===\n");

    // 1. Load model
    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();
    let n_layers = model.num_layers();
    let hidden = model.hidden_size();
    println!(
        "Model: {model_id} ({n_layers} layers, hidden={hidden}, device={:?})",
        model.device(),
    );
    println!("Load time: {load_time:.2?}\n");

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    // 2. Prepare prose (from file or built-in)
    let prose_owned: String;
    if let Some(ref text_path) = args.text {
        prose_owned = std::fs::read_to_string(text_path).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to read {}: {e}", text_path.display()))
        })?;
        println!("Prose source: {}", text_path.display());
    } else {
        prose_owned = prose_passage().to_string();
        println!("Prose source: built-in (tectonic plates, ~500 words)");
    }
    let prose = strip_newlines(&prose_owned);
    let widths: Vec<usize> = (20..=150).step_by(10).collect(); // 14 widths

    // 3. Multi-layer comparison if requested
    if args.all_layers {
        let max_layer = 3.min(n_layers.saturating_sub(1));
        println!("--- Layer comparison (0..={max_layer}) ---\n");
        for layer in 0..=max_layer {
            let means = collect_means(&model, tokenizer, &prose, &widths, layer)?;
            if means.is_empty() {
                println!("  Layer {layer}: no valid character counts collected");
                continue;
            }
            let matrix = build_mean_matrix(&means, hidden, model.device())?;
            let pca = pca_top_k(&matrix, 6.min(matrix.dim(0)?), 50)?;
            let total: f32 = pca.explained_variance_ratio.iter().sum();
            println!(
                "  Layer {layer}: top-6 variance = {:.1}%  [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                // CAST: f32 * 100.0 for display percentage
                total * 100.0,
                pca.explained_variance_ratio.first().copied().unwrap_or(0.0) * 100.0,
                pca.explained_variance_ratio.get(1).copied().unwrap_or(0.0) * 100.0,
                pca.explained_variance_ratio.get(2).copied().unwrap_or(0.0) * 100.0,
                pca.explained_variance_ratio.get(3).copied().unwrap_or(0.0) * 100.0,
                pca.explained_variance_ratio.get(4).copied().unwrap_or(0.0) * 100.0,
                pca.explained_variance_ratio.get(5).copied().unwrap_or(0.0) * 100.0,
            );
        }
        println!();
    }

    // 4. Main analysis on the target layer
    let layer = args.layer;
    if layer >= n_layers {
        return Err(candle_mi::MIError::Config(format!(
            "layer {layer} out of range (model has {n_layers} layers)"
        )));
    }

    run_analysis(
        &model,
        tokenizer,
        &prose,
        &widths,
        layer,
        model_id,
        args.output.as_deref(),
    )
}

/// Run the main PCA analysis on a single layer and optionally write JSON.
#[allow(clippy::too_many_arguments)]
fn run_analysis(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose: &str,
    widths: &[usize],
    layer: usize,
    model_id: &str,
    json_path: Option<&Path>,
) -> candle_mi::Result<()> {
    let hidden = model.hidden_size();
    println!("--- Main analysis: layer {layer} ---\n");

    let t1 = Instant::now();
    let means = collect_means(model, tokenizer, prose, widths, layer)?;
    let collect_time = t1.elapsed();
    println!(
        "Collected {} valid character counts across {} widths in {collect_time:.2?}",
        means.len(),
        widths.len(),
    );

    if means.is_empty() {
        println!("No valid character counts found. Check the model and prose.");
        return Ok(());
    }

    // 5. PCA
    let matrix = build_mean_matrix(&means, hidden, model.device())?;
    let n_samples = matrix.dim(0)?;
    let n_components = 6.min(n_samples);
    let pca = pca_top_k(&matrix, n_components, 50)?;
    let total: f32 = pca.explained_variance_ratio.iter().sum();

    println!(
        "PCA: top-{n_components} components capture {:.1}% of variance",
        total * 100.0
    );
    for (i, ratio) in pca.explained_variance_ratio.iter().enumerate() {
        println!("  PC{}: {:.1}%", i + 1, ratio * 100.0);
    }
    println!();

    // 6. Project means into PC space
    let projections = project_to_pcs(&matrix, &pca)?;

    // 7. Cosine similarity matrix
    let cosine_sim = cosine_similarity_matrix(&matrix)?;

    // 8. Print cosine similarity summary (ringing pattern check)
    print_ringing_summary(&cosine_sim, &means);

    // 9. JSON output
    if let Some(path) = json_path {
        let sorted_counts: Vec<usize> = sorted_char_counts(&means);
        let proj_entries: Vec<Projection> = sorted_counts
            .iter()
            .zip(projections.iter())
            .map(|(&cc, pcs)| Projection {
                char_count: cc,
                pc: pcs.clone(),
            })
            .collect();

        let output = HelixOutput {
            model_id: model_id.to_string(),
            layer,
            max_char_count: sorted_counts.last().copied().unwrap_or(0),
            line_widths: widths.to_vec(),
            explained_variance: pca.explained_variance_ratio,
            total_variance_top6: total,
            projections: proj_entries,
            cosine_similarity: cosine_sim,
        };
        write_json(&output, path)?;
        println!("JSON written to {}", path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Activation collection
// ---------------------------------------------------------------------------

/// For each line width, run a forward pass and accumulate residual vectors
/// by character count. Returns a map from char_count to (sum_vector, count).
fn collect_means(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose: &str,
    widths: &[usize],
    layer: usize,
) -> candle_mi::Result<HashMap<usize, (Vec<f64>, usize)>> {
    let mut accum: HashMap<usize, (Vec<f64>, usize)> = HashMap::new();
    let hidden = model.hidden_size();

    for &width in widths {
        let wrapped = word_wrap(prose, width);
        let encoding = tokenizer.encode_with_offsets(&wrapped)?;

        // Build input tensor
        let input = candle_core::Tensor::new(&encoding.ids[..], model.device())?.unsqueeze(0)?; // [1, seq]

        // Capture ResidPost at the target layer
        let mut hooks = HookSpec::new();
        hooks.capture(HookPoint::ResidPost(layer));

        let cache = model.forward(&input, &hooks)?;
        let resid = cache.require(&HookPoint::ResidPost(layer))?; // [1, seq, hidden]
        let resid_2d = resid.squeeze(0)?; // [seq, hidden]

        // PROMOTE: extract as F32 for accumulation into F64 sums
        let resid_f32 = resid_2d.to_dtype(candle_core::DType::F32)?;
        let resid_data: Vec<Vec<f32>> = (0..encoding.ids.len())
            .map(|i| -> candle_mi::Result<Vec<f32>> { Ok(resid_f32.get(i)?.to_vec1()?) })
            .collect::<candle_mi::Result<Vec<_>>>()?;

        // For each token, compute line character count from byte offsets
        for (tok_idx, &(start, end)) in encoding.offsets.iter().enumerate() {
            // Skip BOS/special tokens with (0, 0) offset
            if start == 0 && end == 0 {
                continue;
            }

            // Line char count = bytes since last newline before this token
            let char_count = compute_line_char_count(&wrapped, start);
            if char_count == 0 || char_count > 150 {
                continue;
            }

            let entry = accum
                .entry(char_count)
                .or_insert_with(|| (vec![0.0_f64; hidden], 0));
            // INDEX: tok_idx is bounded by encoding.offsets.len() == encoding.ids.len() == resid_data.len()
            #[allow(clippy::indexing_slicing)]
            let token_resid = &resid_data[tok_idx];
            for (acc, &val) in entry.0.iter_mut().zip(token_resid.iter()) {
                *acc += f64::from(val);
            }
            entry.1 += 1;
        }
    }

    Ok(accum)
}

/// Compute how many characters into the current line a byte offset falls.
///
/// Scans backwards from `byte_offset` to find the last `\n` (or start of
/// string) and returns the distance.
fn compute_line_char_count(text: &str, byte_offset: usize) -> usize {
    // BORROW: explicit .as_bytes() for byte-level scanning
    let bytes = text.as_bytes();
    let pos = byte_offset.min(bytes.len());

    // Scan backwards for the last newline
    let mut last_newline = 0; // start of string if no newline found
    for i in (0..pos).rev() {
        // INDEX: i < pos <= bytes.len(), safe
        #[allow(clippy::indexing_slicing)]
        if bytes[i] == b'\n' {
            last_newline = i + 1;
            break;
        }
    }

    pos - last_newline
}

// ---------------------------------------------------------------------------
// Mean matrix construction
// ---------------------------------------------------------------------------

/// Build a `[n_valid, hidden]` matrix of mean residual vectors, sorted by
/// character count.
fn build_mean_matrix(
    accum: &HashMap<usize, (Vec<f64>, usize)>,
    hidden: usize,
    device: &candle_core::Device,
) -> candle_mi::Result<candle_core::Tensor> {
    let sorted = sorted_char_counts(accum);
    let n = sorted.len();

    let mut data = Vec::with_capacity(n * hidden);
    for cc in &sorted {
        let Some((sum, count)) = accum.get(cc) else {
            continue;
        };
        // CAST: usize → f64, count is small; exact in f64 mantissa
        #[allow(clippy::as_conversions)]
        let count_f64 = *count as f64;
        for &s in sum {
            // CAST: f64 → f32, precision loss acceptable for mean vector
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let mean_val = (s / count_f64) as f32;
            data.push(mean_val);
        }
    }

    let tensor = candle_core::Tensor::from_vec(data, (n, hidden), device)?;
    Ok(tensor)
}

/// Return sorted character counts from the accumulator.
fn sorted_char_counts(accum: &HashMap<usize, (Vec<f64>, usize)>) -> Vec<usize> {
    let mut counts: Vec<usize> = accum.keys().copied().collect();
    counts.sort_unstable();
    counts
}

// ---------------------------------------------------------------------------
// PCA projection
// ---------------------------------------------------------------------------

/// Project mean vectors into PC space, returning `[n_valid][n_components]`.
fn project_to_pcs(
    matrix: &candle_core::Tensor,
    pca: &PcaResult,
) -> candle_mi::Result<Vec<Vec<f32>>> {
    // matrix: [n, hidden], components: [k, hidden]
    // projection = matrix @ components^T → [n, k]
    // CONTIGUOUS: transpose may produce non-unit strides
    let comp_t = pca.components.t()?.contiguous()?; // [hidden, k]
    let projected = matrix.matmul(&comp_t)?; // [n, k]

    let n = projected.dim(0)?;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<f32> = projected.get(i)?.to_vec1()?;
        result.push(row);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

/// Compute the `[n, n]` cosine similarity matrix on CPU.
fn cosine_similarity_matrix(matrix: &candle_core::Tensor) -> candle_mi::Result<Vec<Vec<f32>>> {
    let n = matrix.dim(0)?;
    // PROMOTE: F32 for dot products
    let mat = matrix.to_dtype(candle_core::DType::F32)?;

    // Precompute norms
    let norms: Vec<f32> = (0..n)
        .map(|i| -> candle_mi::Result<f32> {
            let row = mat.get(i)?;
            let norm: f32 = row.sqr()?.sum_all()?.to_scalar()?;
            Ok(norm.sqrt())
        })
        .collect::<candle_mi::Result<Vec<_>>>()?;

    // Dot product matrix: mat @ mat^T
    // CONTIGUOUS: transpose may produce non-unit strides
    let mat_t = mat.t()?.contiguous()?;
    let dots = mat.matmul(&mat_t)?; // [n, n]

    let mut result = Vec::with_capacity(n);
    for (i, &norm_i) in norms.iter().enumerate() {
        let dots_row = dots.get(i)?; // [n]
        let mut row = Vec::with_capacity(n);
        for (j, &norm_j) in norms.iter().enumerate() {
            let dot: f32 = dots_row.get(j)?.to_scalar()?;
            let denom = norm_i * norm_j;
            if denom > 1e-12 {
                row.push(dot / denom);
            } else {
                row.push(0.0);
            }
        }
        result.push(row);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Text utilities
// ---------------------------------------------------------------------------

/// Strip all newlines from text, replacing them with spaces.
fn strip_newlines(text: &str) -> String {
    text.replace('\n', " ")
}

/// Word-wrap text at `width` characters per line, breaking at word boundaries.
///
/// Lines are separated by `\n`. No line will exceed `width` characters unless
/// a single word is longer than `width`.
fn word_wrap(text: &str, width: usize) -> String {
    let mut result = String::with_capacity(text.len() + text.len() / width);
    let mut line_len = 0;

    for word in text.split_whitespace() {
        let wlen = word.len();
        if line_len > 0 && line_len + 1 + wlen > width {
            result.push('\n');
            line_len = 0;
        }
        if line_len > 0 {
            result.push(' ');
            line_len += 1;
        }
        result.push_str(word);
        line_len += wlen;
    }

    result
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Print a summary of the cosine similarity ringing pattern.
fn print_ringing_summary(cosine_sim: &[Vec<f32>], accum: &HashMap<usize, (Vec<f64>, usize)>) {
    let n = cosine_sim.len();
    if n < 10 {
        println!("Too few character counts ({n}) for ringing analysis.\n");
        return;
    }

    // Average cosine similarity at each offset distance
    let max_offset = n.min(50);
    let mut avg_at_offset = vec![0.0_f64; max_offset];
    let mut count_at_offset = vec![0_usize; max_offset];

    for (i, row_i) in cosine_sim.iter().enumerate() {
        for (j, &sim_val) in row_i.iter().enumerate() {
            let offset = i.abs_diff(j);
            if let (Some(avg), Some(cnt)) = (
                avg_at_offset.get_mut(offset),
                count_at_offset.get_mut(offset),
            ) {
                *avg += f64::from(sim_val);
                *cnt += 1;
            }
        }
    }

    println!("Cosine similarity by offset distance (ringing pattern):");
    for d in 0..max_offset.min(20) {
        let cnt = avg_at_offset.get(d).copied().unwrap_or(0.0);
        let count = count_at_offset.get(d).copied().unwrap_or(0);
        if count > 0 {
            // CAST: usize → f64 for division
            #[allow(clippy::as_conversions)]
            let avg = cnt / count as f64;
            let bar_len = ((avg + 1.0) * 20.0).clamp(0.0, 40.0);
            // CAST: f64 → usize for repeat count; clamped to [0, 40]
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let bar = "#".repeat(bar_len as usize);
            println!("  d={d:3}: {avg:+.3}  {bar}");
        }
    }

    let total_tokens: usize = accum.values().map(|(_, c)| c).sum();
    println!("\nTotal tokens accumulated: {total_tokens}");
    println!("Distinct character counts: {n}\n");
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

/// Write structured JSON output to a file.
fn write_json(output: &HelixOutput, path: &Path) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    std::fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;
    Ok(())
}
