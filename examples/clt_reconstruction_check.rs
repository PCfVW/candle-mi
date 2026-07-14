// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which residual was the `CLT` encoder trained to read? Decide it by
//! reconstruction, not convention.
//!
//! A cross-layer transcoder is trained so that, feeding each layer's encoder
//! the correct residual, the summed decoder outputs reconstruct every layer's
//! **MLP output**. This diagnostic reconstructs `MlpOut(T)` at chosen target
//! layers `T` two ways —
//!
//! - **Mid**: each source encoder reads `ResidMid(L)` (post-attention,
//!   pre-MLP; the circuit-tracer convention candle-mi's census uses), and
//! - **Post**: each source encoder reads `ResidPost(L)` (the layer output,
//!   post-MLP; the residual plip-rs's `forward_with_cache` caches) —
//!
//! and compares each reconstruction to the model's actual `MlpOut(T)` by
//! cosine similarity and relative L2 error. The convention whose
//! reconstruction matches is the one the `CLT` was trained on; the other is
//! the wrong encoder input.
//!
//! `reconstruction(T) = Σ_{L ≤ T} Σ_{f active at L} act_f · decoder_{L→T}[f]`
//!
//! ```bash
//! cargo run --release --features clt,transformer,mmap --example clt_reconstruction_check
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use clap::Parser;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookPoint, HookSpec, MIModel};

/// The Gemma "about" completion prompt (identical to the census cell prompt).
const GEMMA_ABOUT_PROMPT: &str = "The stars were twinkling in the night,\n\
                                  The lanterns cast a golden light.\n\
                                  She wandered in the dark about,\n\
                                  And found a hidden passage";

#[derive(Parser)]
#[command(name = "clt_reconstruction_check")]
#[command(
    about = "Decide the CLT encoder's input residual (ResidMid vs ResidPost) by reconstruction"
)]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// `HuggingFace` CLT repository.
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Prompt (a trailing space is appended, matching the census/detection).
    #[arg(long, default_value = GEMMA_ABOUT_PROMPT)]
    prompt: String,

    /// Target layers whose `MlpOut` to reconstruct (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec![8_usize, 16, 25])]
    targets: Vec<usize>,
}

/// Which residual each source encoder reads.
#[derive(Clone, Copy)]
enum Convention {
    Mid,
    Post,
}

impl Convention {
    const fn label(self) -> &'static str {
        match self {
            Self::Mid => "ResidMid (MLP input) ",
            Self::Post => "ResidPost (layer out)",
        }
    }

    const fn hook(self, layer: usize) -> HookPoint {
        match self {
            Self::Mid => HookPoint::ResidMid(layer),
            Self::Post => HookPoint::ResidPost(layer),
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    eprintln!("=== CLT encoder-input reconstruction check ===\n");
    eprintln!("Model: {}", args.model);
    eprintln!("CLT:   {}\n", args.clt_repo);

    let model = MIModel::from_pretrained(&args.model)?;
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_layers = clt.config().n_layers;

    let prompt_with_space = format!("{} ", args.prompt);
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();
    let pos = seq_len - 1; // planning position (trailing space)
    eprintln!("Prompt tokens: {seq_len}; reconstructing at planning position {pos}\n");

    // One forward capturing ResidMid, ResidPost, and MlpOut at every layer.
    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::ResidMid(layer));
        hooks.capture(HookPoint::ResidPost(layer));
        hooks.capture(HookPoint::MlpOut(layer));
    }
    let cache = model.forward(&Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?, &hooks)?;

    eprintln!(
        "{:<24} {:>8} {:>10} {:>12} {:>12}",
        "convention", "target", "cosine", "rel_L2", "actual_norm"
    );
    eprintln!("{}", "-".repeat(70));

    for &target in &args.targets {
        if target >= n_layers {
            eprintln!("(skipping target {target}: out of range)");
            continue;
        }
        // Actual MLP output at the target layer, planning position.
        let actual = host_vec(
            &cache
                .require(&HookPoint::MlpOut(target))?
                .get(0)?
                .get(pos)?,
        )?;
        let actual_norm = l2(&actual);

        for conv in [Convention::Mid, Convention::Post] {
            let recon = reconstruct(&mut clt, &cache, conv, target, pos, &device)?;
            let cos = cosine(&recon, &actual);
            let rel = rel_l2(&recon, &actual);
            eprintln!(
                "{:<24} {target:>8} {cos:>10.4} {rel:>12.4} {actual_norm:>12.4}",
                conv.label()
            );
        }
    }

    eprintln!(
        "\nInterpretation: the convention with high cosine (→1) and low rel_L2 (→0)\n\
         is the residual the CLT encoder was trained to read."
    );
    Ok(())
}

/// Reconstruct `MlpOut(target)` at position `pos` under one input convention.
fn reconstruct(
    clt: &mut CrossLayerTranscoder,
    cache: &candle_mi::HookCache,
    conv: Convention,
    target: usize,
    pos: usize,
    device: &Device,
) -> candle_mi::Result<Vec<f32>> {
    // Encode every source layer L ≤ target under the chosen convention;
    // collect (feature, activation). `load_encoder` keeps one layer resident,
    // so (re)load per source immediately before encoding it.
    let mut acts: HashMap<CltFeatureId, f32> = HashMap::new();
    for source in 0..=target {
        clt.load_encoder(source, device)?;
        let resid = cache.require(&conv.hook(source))?.get(0)?.get(pos)?;
        let sparse = clt.encode(&resid, source)?;
        for (fid, act) in sparse.features {
            acts.insert(fid, act);
        }
    }

    // Decoder vectors for every active feature, projected to the target layer.
    // `extract_decoder_vectors` loads each source decoder file exactly once.
    let fids: Vec<CltFeatureId> = acts.keys().copied().collect();
    let decoders = clt.extract_decoder_vectors(&fids, target)?;

    // Weighted sum: Σ act_f · decoder_f.
    let d_model = clt.config().d_model;
    let mut recon = vec![0.0_f32; d_model];
    for (fid, act) in &acts {
        if let Some(dvec) = decoders.get(fid) {
            let host = host_vec(dvec)?;
            for (r, d) in recon.iter_mut().zip(host.iter()) {
                *r += act * d;
            }
        }
    }
    Ok(recon)
}

/// Copy a `[d_model]` tensor to a host `f32` vector on CPU.
fn host_vec(t: &Tensor) -> candle_mi::Result<Vec<f32>> {
    // PROMOTE: residual / decoder tensors may be BF16 on device; F32 for the
    // host-side dot products.
    Ok(t.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?)
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = l2(a);
    let nb = l2(b);
    if na <= 1e-10 || nb <= 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Relative L2 error `‖recon − actual‖ / ‖actual‖`.
fn rel_l2(recon: &[f32], actual: &[f32]) -> f32 {
    let diff_sq: f32 = recon
        .iter()
        .zip(actual.iter())
        .map(|(r, a)| (r - a) * (r - a))
        .sum();
    let na = l2(actual);
    if na <= 1e-10 {
        return f32::INFINITY;
    }
    diff_sq.sqrt() / na
}
