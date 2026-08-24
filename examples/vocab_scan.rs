// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vocabulary scan: enumerate `CLT` features by decoder-cosine projection
//! against the base model's embedding matrix.
//!
//! For each `CLT` feature, computes the cosine similarity between the
//! feature's decoder vector (projected to the final target layer, i.e.
//! the direction it writes into the residual stream closest to the LM
//! head) and every token's embedding row, then keeps the top-K tokens
//! per feature.  The output JSON is consumed by the
//! [`scripts/vocab_scan_cmudict_filter.py`](../scripts/vocab_scan_cmudict_filter.py)
//! post-processor, which labels phonologically-clean features using
//! `CMUdict`.
//!
//! Algorithm (mirrors plip-rs's `mode_explore_vocabulary`):
//!
//! 1. Load and normalise the base model's `embed_tokens.weight`
//!    `[vocab_size, d_model]`, transpose to `[d_model, vocab_size]` on
//!    device.
//! 2. For each requested source layer L, fetch the decoder matrix
//!    `[n_features, d_model]` at `target_layer = n_layers - 1` (the
//!    feature's writeable layer closest to the LM head).
//! 3. Chunk features (4096 at a time), normalise each chunk row-wise,
//!    matmul against the transposed embedding to get
//!    `[chunk_size, vocab_size]` cosines, transfer to `CPU`, extract top-K
//!    per feature with a sliding-minimum heap.
//! 4. Decode token IDs back to text via the model's tokenizer.
//! 5. Sort all features by `max_cosine` descending; dump JSON.
//!
//! Memory budget on `RTX 5060 Ti 16 GB` for `Qwen3-1.7B-Base` /
//! `bluelightai/clt-qwen3-1.7b-base-20k`:
//! - Transposed embedding `F32` `GPU`: ~1.16 `GiB`
//! - One layer's decoder `F32` `GPU`: 160 `MiB`
//! - One chunk's cosine result `F32` `GPU`: 2.4 `GiB`
//! - Peak ~3.8 `GiB`; comfortable headroom for the other 12 `GiB`.
//!
//! Pre-flight: the base model must be cached locally
//! (`hf-fm download <model_repo>`) so the embedding can be read from
//! `~/.cache/huggingface/hub/`; the `CLT` decoder files are fetched on
//! demand by [`CrossLayerTranscoder::decoder_matrix`].
//!
//! Usage:
//!   `cargo run --release --features clt,transformer,mmap --example vocab_scan -- \`
//!   `  --model Qwen/Qwen3-1.7B-Base --clt-repo bluelightai/clt-qwen3-1.7b-base-20k \`
//!   `  --output docs/experiments/figure13-qwen3-1.7b-20k/vocab_scan_qwen3_raw.json`

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{D, Device, Tensor};
use clap::Parser;
use safetensors::SafeTensors;
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "vocab_scan")]
#[command(about = "Enumerate CLT features by decoder-cosine projection against \
                   the base model's vocabulary")]
struct Args {
    /// `HuggingFace` model ID providing the embedding matrix
    /// (must be cached locally; not auto-downloaded).
    #[arg(long)]
    model: String,

    /// `HuggingFace` `CLT` repository.
    #[arg(long)]
    clt_repo: String,

    /// Comma-separated list of source layers to scan (default: all layers).
    #[arg(long)]
    layers: Option<String>,

    /// Sample step over features (default: 1 = scan every feature).
    #[arg(long, default_value_t = 1)]
    sample_step: usize,

    /// Number of top-cosine tokens to keep per feature.
    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// `CLT` decoder chunk size (features per GPU matmul); 4096 is the
    /// plip-rs precedent for `RTX 5060 Ti 16 GB`.
    #[arg(long, default_value_t = 4096)]
    chunk_size: usize,

    /// Force `CPU` device (skip `CUDA` even if available).
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Output JSON path (defaults to stdout).
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Output schema (mirrors plip-rs's `ExploreFeatureResult`) ────────────────

#[derive(Serialize)]
struct ExploreTokenScore {
    token_id: u32,
    text: String,
    cosine: f32,
}

#[derive(Serialize)]
struct ExploreFeatureResult {
    feature: CltFeatureId,
    max_cosine: f32,
    top_tokens: Vec<ExploreTokenScore>,
}

#[derive(Serialize)]
struct ExploreOutput {
    model: String,
    clt_repo: String,
    layers: Vec<usize>,
    sample_step: usize,
    top_k: usize,
    vocab_size: usize,
    d_model: usize,
    n_features_scanned: usize,
    total_runtime_secs: f32,
    features: Vec<ExploreFeatureResult>,
}

// ── HF-cache discovery helpers ──────────────────────────────────────────────

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
    // BORROW: explicit .unwrap_or_else fallback to "~/.cache/huggingface/hub"
    // on POSIX. We don't panic — the caller error-paths off `find_snapshot`.
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

fn find_snapshot(model_id: &str) -> Option<PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Locate the safetensors file containing `model.embed_tokens.weight` and
/// extract that tensor as an `F32` flat `Vec`, returning
/// `(values, vocab_size, d_model)`.
///
/// Supports both single-file (`model.safetensors`) and sharded
/// (`model.safetensors.index.json`) layouts.
fn load_embedding_matrix(snapshot: &Path) -> Result<(Vec<f32>, usize, usize), String> {
    let tensor_name = "model.embed_tokens.weight";

    // Single-file layout
    let single = snapshot.join("model.safetensors");
    if single.exists() {
        let data = fs::read(&single).map_err(|e| format!("read {}: {e}", single.display()))?;
        let st = SafeTensors::deserialize(&data)
            .map_err(|e| format!("parse {}: {e}", single.display()))?;
        if st.tensor(tensor_name).is_ok() {
            return extract_embedding(&st, tensor_name);
        }
    }

    // Sharded layout
    let idx_path = snapshot.join("model.safetensors.index.json");
    let idx_str =
        fs::read_to_string(&idx_path).map_err(|e| format!("read {}: {e}", idx_path.display()))?;
    let idx: serde_json::Value =
        serde_json::from_str(&idx_str).map_err(|e| format!("parse {}: {e}", idx_path.display()))?;
    let shard = idx
        .get("weight_map")
        .and_then(|m| m.get(tensor_name))
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("tensor '{tensor_name}' missing from weight_map"))?;
    let shard_path = snapshot.join(shard);
    let data = fs::read(&shard_path).map_err(|e| format!("read {}: {e}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&data)
        .map_err(|e| format!("parse {}: {e}", shard_path.display()))?;
    extract_embedding(&st, tensor_name)
}

fn extract_embedding(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<f32>, usize, usize), String> {
    let view = st
        .tensor(name)
        .map_err(|e| format!("tensor '{name}': {e}"))?;
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(format!(
            "expected 2D embedding, got shape {shape:?} for '{name}'"
        ));
    }
    // INDEX: shape.len() == 2 verified just above.
    #[allow(clippy::indexing_slicing)]
    let (vocab_size, d_model) = (shape[0], shape[1]);
    let bytes = view.data();
    // EXHAUSTIVE: safetensors exposes many dtypes; embedding matrices in the
    // wild are BF16 (Qwen3, Gemma, Llama 3.x BF16 checkpoints) or F32 (older
    // releases); other dtypes surface as an explicit error.
    #[allow(clippy::wildcard_enum_match_arm)]
    let values: Vec<f32> = match view.dtype() {
        safetensors::Dtype::BF16 => bf16_bytes_to_f32(bytes),
        safetensors::Dtype::F32 => {
            // CONTIGUOUS: safetensors stores F32 little-endian; as_chunks(4) +
            // f32::from_le_bytes is the standard zero-copy-ish reader.
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
        other => {
            return Err(format!(
                "unsupported embedding dtype {other:?} (only BF16 / F32 implemented)"
            ));
        }
    };
    Ok((values, vocab_size, d_model))
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // INDEX: bounds i*2 + 1 < n*2 == bytes.len() by construction of n.
        #[allow(clippy::indexing_slicing)]
        let bf16_bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        // CAST: u16 → u32 then left-shifted to upper half of f32; this is the
        // canonical BF16 → F32 widening (BF16 has the same exponent layout as F32).
        let f32_bits = u32::from(bf16_bits) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

// ── Token decoding helper ───────────────────────────────────────────────────

/// Lazily decode a token ID to text via the model's tokenizer, with a cache.
fn decode_token_cached(
    cache: &mut HashMap<u32, String>,
    tokenizer: &tokenizers::Tokenizer,
    token_id: u32,
) -> String {
    // BORROW: .clone() on the cached String — Tokenizer::decode returns owned
    // String; clone keeps the cache populated for repeated lookups.
    cache
        .entry(token_id)
        .or_insert_with(|| {
            tokenizer
                .decode(&[token_id], true)
                .unwrap_or_else(|_| format!("<{token_id}>"))
        })
        .clone()
}

// ── Main scan ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
// EXPLICIT: the scan body is a flat sequence (parse layers → load embedding
// → loop layers → chunked matmul → top-K extraction → JSON dump). Extracting
// helpers would scatter the data flow and the helpers would have no other
// call sites.
fn run_scan(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();

    // 1. Device selection.
    let device = if args.cpu {
        Device::Cpu
    } else {
        // BORROW: .or_else fallback to CPU when no CUDA device is available.
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };
    eprintln!("=== Vocab scan ===");
    eprintln!("Device:   {device:?}");
    eprintln!("Model:    {}", args.model);
    eprintln!("CLT:      {}", args.clt_repo);

    // 2. Locate model snapshot.
    let snapshot = find_snapshot(&args.model).ok_or_else(|| {
        format!(
            "model '{}' not found in local HF cache; run \
             `hf-fm download {}` first",
            args.model, args.model
        )
    })?;
    eprintln!("Snapshot: {}", snapshot.display());

    // 3. Load tokenizer.
    let tok_path = snapshot.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("load tokenizer from {}: {e}", tok_path.display()))?;

    // 4. Load + normalise + transpose embedding matrix on device.
    eprintln!("Loading embedding matrix...");
    let (embed_flat, vocab_size, d_model) = load_embedding_matrix(&snapshot)?;
    eprintln!("  shape: [{vocab_size}, {d_model}], transferring to {device:?}");
    let embed = Tensor::from_vec(embed_flat, (vocab_size, d_model), &device)
        .map_err(|e| format!("embed Tensor::from_vec: {e}"))?;
    // CAST: 1e-16 is the canonical small epsilon used by the plip-rs reference
    // to avoid division-by-zero on near-degenerate embedding rows.
    let norms = embed
        .sqr()
        .and_then(|t| t.sum_keepdim(D::Minus1))
        .and_then(|t| t.affine(1.0, 1e-16))
        .and_then(|t| t.sqrt())
        .map_err(|e| format!("embed norms: {e}"))?;
    let embed_normed = embed
        .broadcast_div(&norms)
        .map_err(|e| format!("embed normalise: {e}"))?;
    drop(embed);
    // CONTIGUOUS: .t() yields non-unit strides; matmul requires contiguous.
    let embed_t = embed_normed
        .t()
        .and_then(|t| t.contiguous())
        .map_err(|e| format!("embed transpose: {e}"))?;
    drop(embed_normed);
    eprintln!("  embedding ready on {device:?} (normalised, transposed)");

    // 5. Open the CLT.
    eprintln!("Opening CLT...");
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)
        .map_err(|e| format!("open CLT '{}': {e}", args.clt_repo))?;
    let n_layers = clt.config().n_layers;
    let n_features = clt.config().n_features_per_layer;
    let final_target = n_layers - 1;
    let layers: Vec<usize> = match &args.layers {
        Some(s) => s
            .split(',')
            .map(|x| {
                x.trim()
                    .parse::<usize>()
                    .map_err(|e| format!("invalid layer '{x}': {e}"))
            })
            .collect::<Result<_, _>>()?,
        None => (0..n_layers).collect(),
    };
    eprintln!(
        "  {} layers, {} features/layer, scanning {} layer(s): {layers:?}",
        n_layers,
        n_features,
        layers.len()
    );
    eprintln!(
        "  sample_step={}, top_k={}, chunk_size={}",
        args.sample_step, args.top_k, args.chunk_size
    );

    // 6. Per-layer scan.
    let mut all_results: Vec<ExploreFeatureResult> = Vec::new();
    let mut tok_cache: HashMap<u32, String> = HashMap::new();

    for &source_layer in &layers {
        let layer_t0 = Instant::now();
        eprintln!("Layer {source_layer}: loading decoder slice (target_layer={final_target})...");
        let dec = clt
            .decoder_matrix(source_layer, final_target, &device)
            .map_err(|e| format!("decoder_matrix(L{source_layer}): {e}"))?;
        let feat_indices: Vec<usize> = (0..n_features).step_by(args.sample_step).collect();
        let n_sampled = feat_indices.len();
        eprintln!("  scanning {n_sampled} features vs {vocab_size} tokens...");

        for chunk_feats in feat_indices.chunks(args.chunk_size) {
            // Extract feature rows as a contiguous chunk tensor.
            let chunk_indices: Vec<u32> = chunk_feats
                .iter()
                // CAST: usize → u32, feature index bounded by n_features_per_layer
                // (20480 for BlueLightAI Qwen3 20K).
                .map(|&i| i.try_into().unwrap_or(u32::MAX))
                .collect();
            let idx_tensor = Tensor::new(chunk_indices.as_slice(), &device)
                .map_err(|e| format!("idx_tensor: {e}"))?;
            let chunk = dec
                .index_select(&idx_tensor, 0)
                .map_err(|e| format!("index_select chunk: {e}"))?;
            // Normalise chunk row-wise.
            let chunk_norms = chunk
                .sqr()
                .and_then(|t| t.sum_keepdim(D::Minus1))
                .and_then(|t| t.affine(1.0, 1e-16))
                .and_then(|t| t.sqrt())
                .map_err(|e| format!("chunk norms: {e}"))?;
            let chunk_normed = chunk
                .broadcast_div(&chunk_norms)
                .map_err(|e| format!("chunk normalise: {e}"))?;
            // Cosines: [chunk_len, vocab_size]
            let cosines = chunk_normed
                .matmul(&embed_t)
                .map_err(|e| format!("matmul: {e}"))?;
            let cos_cpu: Vec<f32> = cosines
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| format!("cosines to CPU: {e}"))?;
            drop(cosines);

            // Per-feature top-K extraction (sliding-minimum heap).
            for (local_idx, &fi) in chunk_feats.iter().enumerate() {
                let row_start = local_idx * vocab_size;
                // INDEX: row_start..row_start+vocab_size is in-bounds because
                // cos_cpu.len() == chunk_feats.len() * vocab_size.
                #[allow(clippy::indexing_slicing)]
                let row = &cos_cpu[row_start..row_start + vocab_size];
                let top = extract_top_k(row, args.top_k);
                // CAST: f32 max_cosine is already f32; no cast needed but keep
                // the explicit max via first element after sort.
                let max_cosine = top.first().map_or(0.0, |&(_, c)| c);
                let top_tokens: Vec<ExploreTokenScore> = top
                    .into_iter()
                    .map(|(tid, cosine)| {
                        let text = decode_token_cached(&mut tok_cache, &tokenizer, tid);
                        ExploreTokenScore {
                            token_id: tid,
                            text,
                            cosine,
                        }
                    })
                    .collect();
                all_results.push(ExploreFeatureResult {
                    feature: CltFeatureId {
                        layer: source_layer,
                        index: fi,
                    },
                    max_cosine,
                    top_tokens,
                });
            }
        }
        eprintln!(
            "  Layer {source_layer} done in {:.1}s",
            layer_t0.elapsed().as_secs_f32()
        );
    }

    // 7. Sort by max_cosine descending.
    all_results.sort_by(|a, b| {
        b.max_cosine
            .partial_cmp(&a.max_cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 8. Stderr summary.
    let preview_n = 20.min(all_results.len());
    eprintln!(
        "\nTop {preview_n} most word-specific features (of {} scanned):",
        all_results.len()
    );
    for (rank, r) in all_results.iter().take(preview_n).enumerate() {
        let toks: String = r
            .top_tokens
            .iter()
            .take(10)
            .map(|t| format!("{}({:.3})", t.text.trim(), t.cosine))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "  #{:3}  L{}:{}  max={:.4}  [{toks}]",
            rank + 1,
            r.feature.layer,
            r.feature.index,
            r.max_cosine
        );
    }

    // 9. Write JSON.
    let elapsed = t0.elapsed().as_secs_f32();
    eprintln!("\nTotal time: {elapsed:.1}s");
    let output = ExploreOutput {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        layers,
        sample_step: args.sample_step,
        top_k: args.top_k,
        vocab_size,
        d_model,
        n_features_scanned: all_results.len(),
        total_runtime_secs: elapsed,
        features: all_results,
    };
    let json = serde_json::to_string(&output).map_err(|e| format!("serialise JSON: {e}"))?;
    if let Some(ref p) = args.output {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        fs::write(p, &json).map_err(|e| format!("write {}: {e}", p.display()))?;
        eprintln!("Written to {}", p.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

/// Sliding-minimum top-K extraction over a single row of cosine scores.
/// Returns `(token_id, cosine)` pairs sorted by cosine descending.
fn extract_top_k(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(row.len());
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    let mut min_cos = f32::NEG_INFINITY;
    let mut min_i: usize = 0;
    for (tid, &cos) in row.iter().enumerate() {
        // CAST: usize → u32 — token ID is bounded by tokenizer vocab_size,
        // which is typically <= 256K (fits in u32 with huge headroom).
        let tid_u32 = u32::try_from(tid).unwrap_or(u32::MAX);
        if top.len() < k {
            top.push((tid_u32, cos));
            if top.len() == k {
                // Initialise running minimum after the initial fill.
                min_cos = f32::INFINITY;
                for (i, &(_, c)) in top.iter().enumerate() {
                    if c < min_cos {
                        min_cos = c;
                        min_i = i;
                    }
                }
            }
        } else if cos > min_cos {
            // INDEX: min_i bounded by top.len() == k just above.
            #[allow(clippy::indexing_slicing)]
            {
                top[min_i] = (tid_u32, cos);
            }
            // Rescan to find the new minimum.
            min_cos = f32::INFINITY;
            for (i, &(_, c)) in top.iter().enumerate() {
                if c < min_cos {
                    min_cos = c;
                    min_i = i;
                }
            }
        }
    }
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    if let Err(e) = run_scan(&args) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
