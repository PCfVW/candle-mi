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
//! # Quick variance scan across all layers
//! cargo run --release --features transformer,mmap --example character_count_helix -- --scan-layers all
//!
//! # Full PCA analysis on a specific layer with JSON export
//! cargo run --release --features transformer,mmap --example character_count_helix -- --pca-layers 10..11 --output helix.json
//!
//! # Full PCA analysis on several layers (JSON files named helix_L10.json, etc.)
//! cargo run --release --features transformer,mmap --example character_count_helix -- --pca-layers 10..15 --output helix.json
//!
//! # Scan all layers then full analysis on the best ones
//! cargo run --release --features transformer,mmap --example character_count_helix -- --scan-layers all --pca-layers 10..13
//!
//! # Use your own prose file
//! cargo run --release --features transformer,mmap --example character_count_helix -- --text mytext.txt
//!
//! # Use a directory of text files (each file is a separate batch)
//! cargo run --release --features transformer,mmap --example character_count_helix -- --text-dir texts/
//!
//! # Sweep mode: one layer per run, auto-resume (first run → layer 0, next → layer 1, ...)
//! cargo run --release --features transformer,mmap --example character_count_helix -- --sweep --output helix_sweep.json
//!
//! # With memory reporting (GPU name, per-process VRAM before/after)
//! cargo run --release --features transformer,mmap,memory --example character_count_helix
//!
//! # With DXGI debug output (raw adapter/VRAM values on stderr, Windows only)
//! cargo run --release --features transformer,mmap,dxgi-debug --example character_count_helix
//! ```
//!
//! **What it does:**
//!
//! 1. Loads a transformer model and prose text (built-in, `--text`, or `--text-dir`).
//! 2. Strips newlines and re-wraps at widths 20, 30, 40, ... 150 characters.
//! 3. For each text and width, runs one or more forward passes (long
//!    sequences are chunked to `--max-tokens` to fit in VRAM) capturing
//!    [`HookPoint::ResidPost`](candle_mi::HookPoint::ResidPost) at the target
//!    layer, then computes each token's line character count from byte offsets.
//! 4. Averages residual-stream vectors by character count (1..=150).
//! 5. Runs [`pca_top_k`](candle_mi::pca_top_k) on the mean vectors (6 PCs).
//! 6. Computes an `n x n` cosine-similarity matrix on the mean vectors
//!    (where `n` is the number of distinct character counts, at most 150).
//! 7. Prints a summary and optionally writes structured JSON output.
//!
//! **CLI flags — "what to analyse" vs "how to iterate":**
//!
//! `--scan-layers` and `--pca-layers` select *what* to analyse:
//! - `--scan-layers` runs a lightweight variance scan (top-6 explained
//!   variance only, no JSON) — useful for finding which layers carry the
//!   strongest helix signal.
//! - `--pca-layers` runs the full analysis (PCA projections, cosine
//!   similarity matrix, ringing summary, optional JSON output).
//!
//! `--sweep` controls *how* to iterate: it runs the same full analysis as
//! `--pca-layers` but one layer per invocation, auto-resuming from the
//! output JSON file. Results are appended to a JSON array, so repeated
//! runs walk through layers 0, 1, 2, ... automatically. Requires `--output`.
//!
//! The JSON output can be visualised with the companion Mathematica script
//! `examples/results/character_count_helix/helix_plot.wl`.

#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]

use candle_mi::{HookPoint, HookSpec, MIModel, MITokenizer, PcaResult, pca_top_k};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use clap::Parser;
use serde::{Deserialize, Serialize};
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

    /// **What to analyse** — lightweight variance scan across a layer range.
    /// Reports top-6 explained variance only (no projections, no JSON).
    /// Use `all` to scan every layer, or `START..END` for a specific range
    /// (exclusive end, e.g. `5..15` scans layers 5 through 14).
    #[arg(long, value_name = "all | START..END")]
    scan_layers: Option<String>,

    /// **What to analyse** — full PCA analysis (projections, cosine
    /// similarity, ringing summary, optional JSON) across a layer range.
    /// Accepts the same values as `--scan-layers`.
    /// Default when neither flag is given: full analysis on layer 1.
    #[arg(long, value_name = "all | START..END")]
    pca_layers: Option<String>,

    /// Path to a plain-text file to use as prose input (default: built-in passage)
    #[arg(long)]
    text: Option<PathBuf>,

    /// Path to a directory of `.txt` files (each file is processed as a separate batch)
    #[arg(long)]
    text_dir: Option<PathBuf>,

    /// Maximum tokens per forward pass (longer sequences are chunked). Default: 4096.
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// **How to iterate** — sweep layers one at a time, auto-resuming from
    /// the output JSON file. Runs the same full analysis as `--pca-layers`
    /// but one layer per invocation. On first run starts at layer 0;
    /// subsequent runs read the file and pick the next untested layer.
    /// Requires `--output`.
    #[arg(long)]
    sweep: bool,
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
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
    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();
    let n_layers = model.num_layers();
    let hidden = model.hidden_size();
    println!(
        "Model: {model_id} ({n_layers} layers, hidden={hidden}, device={:?})",
        model.device(),
    );
    println!("Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    println!();

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    // 2. Prepare prose texts (from directory, single file, or built-in)
    let prose_texts = load_prose_texts(&args)?;
    let widths: Vec<usize> = (20..=150).step_by(10).collect(); // 14 widths

    // 2b. Sweep mode: auto-resume from existing JSON file
    if args.sweep {
        let output_path = args.output.as_ref().ok_or_else(|| {
            candle_mi::MIError::Config("--sweep requires --output".into())
        })?;
        run_sweep(
            &model, tokenizer, &prose_texts, &widths,
            model_id, output_path, n_layers, args.max_tokens,
        )?;
        print_finished(t0);
        return Ok(());
    }

    // 3. Lightweight variance scan if requested
    let scan_range = parse_layer_range(args.scan_layers.as_deref(), "scan-layers", n_layers)?;
    if let Some((start, end)) = scan_range {
        run_variance_scan(
            &model,
            tokenizer,
            &prose_texts,
            &widths,
            hidden,
            start,
            end,
            args.max_tokens,
        )?;
    }

    // 4. Full PCA analysis
    let pca_range = parse_layer_range(args.pca_layers.as_deref(), "pca-layers", n_layers)?;
    // Default: analyse layer 1 when neither flag is given
    let pca_layers: Vec<usize> = if let Some((start, end)) = pca_range {
        (start..end).collect()
    } else if scan_range.is_none() {
        vec![1.min(n_layers.saturating_sub(1))]
    } else {
        vec![]
    };

    let multi_pca = pca_layers.len() > 1;
    for &layer in &pca_layers {
        // When analysing multiple layers, embed the layer number in the filename
        let layer_path = if multi_pca {
            args.output.as_deref().map(|p| {
                let stem = p.file_stem().unwrap_or_default().to_string_lossy();
                let ext = p.extension().unwrap_or_default().to_string_lossy();
                p.with_file_name(format!("{stem}_L{layer}.{ext}"))
            })
        } else {
            args.output.clone()
        };
        run_analysis(
            &model,
            tokenizer,
            &prose_texts,
            &widths,
            layer,
            model_id,
            layer_path.as_deref(),
            args.max_tokens,
        )?;
    }

    print_finished(t0);
    Ok(())
}

/// Load prose texts from `--text-dir`, `--text`, or the built-in passage.
fn load_prose_texts(args: &Args) -> candle_mi::Result<Vec<String>> {
    if let Some(ref dir_path) = args.text_dir {
        let texts = load_text_dir(dir_path)?;
        println!(
            "Prose source: {} files from {}",
            texts.len(),
            dir_path.display()
        );
        Ok(texts)
    } else if let Some(ref text_path) = args.text {
        let raw = std::fs::read_to_string(text_path).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to read {}: {e}", text_path.display()))
        })?;
        println!("Prose source: {}", text_path.display());
        Ok(vec![strip_newlines(&raw)])
    } else {
        println!("Prose source: built-in (tectonic plates, ~500 words)");
        Ok(vec![strip_newlines(prose_passage())])
    }
}

/// Sweep one layer per invocation, auto-resuming from the output JSON file.
#[allow(clippy::too_many_arguments)]
fn run_sweep(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose_texts: &[String],
    widths: &[usize],
    model_id: &str,
    output_path: &Path,
    n_layers: usize,
    max_tokens: usize,
) -> candle_mi::Result<()> {
    let existing = read_sweep_file(output_path)?;
    let next_layer = if existing.is_empty() {
        0
    } else {
        let max_done = existing.iter().map(|e| e.layer).max().unwrap_or(0);
        max_done + 1
    };

    if next_layer >= n_layers {
        println!("All {n_layers} layers already analysed. Nothing to do.");
        return Ok(());
    }

    println!(
        "Sweep: layer {next_layer}/{n_layers} ({} already done)\n",
        existing.len()
    );

    let result = run_analysis_returning(
        model, tokenizer, prose_texts, widths, next_layer, model_id, max_tokens,
    )?;

    let mut all = existing;
    if let Some(entry) = result {
        all.push(entry);
    }
    write_json_array(&all, output_path)?;
    println!(
        "JSON written to {} ({} layer(s) total)",
        output_path.display(),
        all.len()
    );
    Ok(())
}

/// Run a lightweight variance scan across layers `start..end`, printing a summary table.
#[allow(clippy::too_many_arguments)]
fn run_variance_scan(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose_texts: &[String],
    widths: &[usize],
    hidden: usize,
    start: usize,
    end: usize,
    max_tokens: usize,
) -> candle_mi::Result<()> {
    println!("--- Variance scan [{start}, {end}) ---\n");
    for layer in start..end {
        let t_layer = Instant::now();
        let means = collect_means_multi(model, tokenizer, prose_texts, widths, layer, max_tokens)?;
        if means.is_empty() {
            println!("  Layer {layer:>2}: no valid character counts collected");
            continue;
        }
        let matrix = build_mean_matrix(&means, hidden, model.device())?;
        let pca = pca_top_k(&matrix, 6.min(matrix.dim(0)?), 50)?;
        let total: f32 = pca.explained_variance_ratio.iter().sum();
        let elapsed = t_layer.elapsed();
        println!(
            "  Layer {layer:>2}: top-6 variance = {:.1}%  [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]  ({elapsed:.2?})",
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
    Ok(())
}

/// Run the main PCA analysis on a single layer and optionally write JSON.
#[allow(clippy::too_many_arguments)]
fn run_analysis(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose_texts: &[String],
    widths: &[usize],
    layer: usize,
    model_id: &str,
    json_path: Option<&Path>,
    max_tokens: usize,
) -> candle_mi::Result<()> {
    let hidden = model.hidden_size();
    println!("--- Full PCA analysis: layer {layer} ---\n");

    let t1 = Instant::now();
    let means = collect_means_multi(model, tokenizer, prose_texts, widths, layer, max_tokens)?;
    let collect_time = t1.elapsed();
    println!(
        "Collected {} valid character counts across {} texts x {} widths in {collect_time:.2?}",
        means.len(),
        prose_texts.len(),
        widths.len(),
    );

    if means.is_empty() {
        println!("No valid character counts found. Check the model and prose.");
        return Ok(());
    }

    // PCA
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

    // Project means into PC space
    let projections = project_to_pcs(&matrix, &pca)?;

    // Cosine similarity matrix
    let cosine_sim = cosine_similarity_matrix(&matrix)?;

    // Ringing pattern check
    print_ringing_summary(&cosine_sim, &means);

    // JSON output
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

/// Run PCA analysis on a single layer and return the `HelixOutput` (for sweep mode).
#[allow(clippy::too_many_arguments)]
fn run_analysis_returning(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose_texts: &[String],
    widths: &[usize],
    layer: usize,
    model_id: &str,
    max_tokens: usize,
) -> candle_mi::Result<Option<HelixOutput>> {
    let hidden = model.hidden_size();
    println!("--- Full PCA analysis: layer {layer} ---\n");

    let t1 = Instant::now();
    let means = collect_means_multi(model, tokenizer, prose_texts, widths, layer, max_tokens)?;
    let collect_time = t1.elapsed();
    println!(
        "Collected {} valid character counts across {} texts x {} widths in {collect_time:.2?}",
        means.len(),
        prose_texts.len(),
        widths.len(),
    );

    if means.is_empty() {
        println!("No valid character counts found. Check the model and prose.");
        return Ok(None);
    }

    // PCA
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

    // Project means into PC space
    let projections = project_to_pcs(&matrix, &pca)?;

    // Cosine similarity matrix
    let cosine_sim = cosine_similarity_matrix(&matrix)?;

    // Ringing pattern check
    print_ringing_summary(&cosine_sim, &means);

    let sorted_counts: Vec<usize> = sorted_char_counts(&means);
    let proj_entries: Vec<Projection> = sorted_counts
        .iter()
        .zip(projections.iter())
        .map(|(&cc, pcs)| Projection {
            char_count: cc,
            pc: pcs.clone(),
        })
        .collect();

    Ok(Some(HelixOutput {
        model_id: model_id.to_string(),
        layer,
        max_char_count: sorted_counts.last().copied().unwrap_or(0),
        line_widths: widths.to_vec(),
        explained_variance: pca.explained_variance_ratio,
        total_variance_top6: total,
        projections: proj_entries,
        cosine_similarity: cosine_sim,
    }))
}

// ---------------------------------------------------------------------------
// Activation collection
// ---------------------------------------------------------------------------

/// Collect means across multiple prose texts, merging their accumulators.
fn collect_means_multi(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose_texts: &[String],
    widths: &[usize],
    layer: usize,
    max_tokens: usize,
) -> candle_mi::Result<HashMap<usize, (Vec<f64>, usize)>> {
    let mut merged: HashMap<usize, (Vec<f64>, usize)> = HashMap::new();
    let hidden = model.hidden_size();

    let n_texts = prose_texts.len();
    for (text_idx, prose) in prose_texts.iter().enumerate() {
        let t_text = Instant::now();
        let partial = collect_means(model, tokenizer, prose, widths, layer, max_tokens)?;
        let tokens_in_text: usize = partial.values().map(|(_, c)| c).sum();
        eprintln!(
            "    text {}/{n_texts}: {tokens_in_text} tokens across {} widths ({:.1?})",
            text_idx + 1,
            widths.len(),
            t_text.elapsed(),
        );
        for (cc, (sum_vec, count)) in partial {
            let entry = merged
                .entry(cc)
                .or_insert_with(|| (vec![0.0_f64; hidden], 0));
            for (acc, val) in entry.0.iter_mut().zip(sum_vec.iter()) {
                *acc += val;
            }
            entry.1 += count;
        }
    }

    Ok(merged)
}

/// For each line width, run forward passes in chunks and accumulate residual
/// vectors by character count. Returns a map from `char_count` to
/// `(sum_vector, count)`.
///
/// Long sequences are split into `max_tokens`-sized chunks so that VRAM
/// stays bounded. Each chunk is processed independently; the byte offsets
/// from `encode_with_offsets` are relative to the full wrapped text, so
/// `compute_line_char_count` works correctly across chunk boundaries.
fn collect_means(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prose: &str,
    widths: &[usize],
    layer: usize,
    max_tokens: usize,
) -> candle_mi::Result<HashMap<usize, (Vec<f64>, usize)>> {
    let mut accum: HashMap<usize, (Vec<f64>, usize)> = HashMap::new();
    let hidden = model.hidden_size();

    for &width in widths {
        let wrapped = word_wrap(prose, width);
        let encoding = tokenizer.encode_with_offsets(&wrapped)?;

        let full_len = encoding.ids.len();
        let n_chunks = full_len.div_ceil(max_tokens);
        if n_chunks > 1 {
            eprintln!("      width {width}: {full_len} tokens → {n_chunks} chunks of ≤{max_tokens}");
        }

        for chunk_idx in 0..n_chunks {
            let chunk_start = chunk_idx * max_tokens;
            let chunk_end = full_len.min(chunk_start + max_tokens);
            // INDEX: chunk_start < chunk_end <= full_len == encoding.ids.len()
            #[allow(clippy::indexing_slicing)]
            let ids = &encoding.ids[chunk_start..chunk_end];
            // INDEX: offsets.len() == ids.len() from encode_with_offsets
            #[allow(clippy::indexing_slicing)]
            let offsets = &encoding.offsets[chunk_start..chunk_end];
            let chunk_len = ids.len();

            // Build input tensor
            let input = candle_core::Tensor::new(ids, model.device())?.unsqueeze(0)?; // [1, chunk_len]

            // Capture ResidPost at the target layer
            let mut hooks = HookSpec::new();
            hooks.capture(HookPoint::ResidPost(layer));

            let cache = model.forward(&input, &hooks)?;
            let resid = cache.require(&HookPoint::ResidPost(layer))?; // [1, chunk_len, hidden]
            let resid_2d = resid.squeeze(0)?; // [chunk_len, hidden]

            // PROMOTE: extract as F32 for accumulation into F64 sums
            let resid_f32 = resid_2d.to_dtype(candle_core::DType::F32)?;
            let resid_data: Vec<Vec<f32>> = (0..chunk_len)
                .map(|i| -> candle_mi::Result<Vec<f32>> { Ok(resid_f32.get(i)?.to_vec1()?) })
                .collect::<candle_mi::Result<Vec<_>>>()?;

            // For each token, compute line character count from byte offsets
            // (offsets are relative to the full wrapped text, not the chunk)
            for (tok_idx, &(start, end)) in offsets.iter().enumerate() {
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
                // INDEX: tok_idx is bounded by offsets.len() == chunk_len == resid_data.len()
                #[allow(clippy::indexing_slicing)]
                let token_resid = &resid_data[tok_idx];
                for (acc, &val) in entry.0.iter_mut().zip(token_resid.iter()) {
                    *acc += f64::from(val);
                }
                entry.1 += 1;
            }
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
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
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
// CLI helpers
// ---------------------------------------------------------------------------

/// Parse a layer range spec into a `(start, end)` pair (exclusive end).
///
/// Accepted values: `"all"` → `(0, n_layers)`, `"START..END"` → parsed range.
/// Returns `None` when `spec` is `None`.
///
/// # Errors
///
/// Returns an error if the range string is malformed or out of bounds.
fn parse_layer_range(
    spec: Option<&str>,
    flag_name: &str,
    n_layers: usize,
) -> candle_mi::Result<Option<(usize, usize)>> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    if spec.eq_ignore_ascii_case("all") {
        return Ok(Some((0, n_layers)));
    }
    // Expect "START..END"
    let Some((start_str, end_str)) = spec.split_once("..") else {
        return Err(candle_mi::MIError::Config(format!(
            "invalid --{flag_name} value \"{spec}\": expected \"all\" or \"START..END\""
        )));
    };
    let start: usize = start_str.parse().map_err(|_| {
        candle_mi::MIError::Config(format!(
            "invalid --{flag_name} start \"{start_str}\": expected integer"
        ))
    })?;
    let end: usize = end_str.parse().map_err(|_| {
        candle_mi::MIError::Config(format!(
            "invalid --{flag_name} end \"{end_str}\": expected integer"
        ))
    })?;
    if start >= end || end > n_layers {
        return Err(candle_mi::MIError::Config(format!(
            "--{flag_name} {start}..{end} is out of range (model has {n_layers} layers)"
        )));
    }
    Ok(Some((start, end)))
}

// ---------------------------------------------------------------------------
// Text utilities
// ---------------------------------------------------------------------------

/// Load all `.txt` files from a directory, sorted by name, with newlines stripped.
///
/// # Errors
///
/// Returns an error if the directory cannot be read or any file fails to load.
fn load_text_dir(dir: &Path) -> candle_mi::Result<Vec<String>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| candle_mi::MIError::Config(format!("cannot read {}: {e}", dir.display())))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();

    if paths.is_empty() {
        return Err(candle_mi::MIError::Config(format!(
            "no .txt files found in {}",
            dir.display()
        )));
    }

    let mut texts = Vec::with_capacity(paths.len());
    for path in &paths {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
        })?;
        println!("  loaded: {} ({} bytes)", path.display(), raw.len());
        texts.push(strip_newlines(&raw));
    }
    Ok(texts)
}

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

/// Print a finish line with wall-clock time and elapsed duration.
fn print_finished(t0: Instant) {
    let now = std::time::SystemTime::now();
    let secs_since_epoch = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // CAST: u64 → i64, Unix timestamp fits in i64 until year 292 billion
    #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
    let secs = secs_since_epoch as i64;
    let (h, m, s) = utc_hms(secs);
    println!(
        "\nFinished at {:02}:{:02}:{:02} UTC — total runtime: {:.2?}",
        h, m, s, t0.elapsed()
    );
}

/// Extract `(hour, minute, second)` from a Unix timestamp.
const fn utc_hms(epoch_secs: i64) -> (i64, i64, i64) {
    let day_secs = epoch_secs.rem_euclid(86_400);
    (day_secs / 3600, (day_secs % 3600) / 60, day_secs % 60)
}

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
        let sum = avg_at_offset.get(d).copied().unwrap_or(0.0);
        let count = count_at_offset.get(d).copied().unwrap_or(0);
        if count > 0 {
            // CAST: usize → f64, count is small; exact in f64 mantissa
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            let avg = sum / count as f64;
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

/// Read an existing sweep JSON file (array of `HelixOutput`).
/// Returns an empty `Vec` if the file does not exist.
fn read_sweep_file(path: &Path) -> candle_mi::Result<Vec<HelixOutput>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
    })?;
    let entries: Vec<HelixOutput> = serde_json::from_str(&raw).map_err(|e| {
        candle_mi::MIError::Config(format!(
            "failed to parse {} as JSON array: {e}", path.display()
        ))
    })?;
    Ok(entries)
}

/// Write an array of `HelixOutput` entries to a file (sweep mode).
fn write_json_array(entries: &[HelixOutput], path: &Path) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(entries)
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
