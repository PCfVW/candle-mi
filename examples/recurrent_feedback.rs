// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recurrent feedback (anacrousis) for rhyme completion.
//!
//! Demonstrates `RecurrentPassSpec`, `RecurrentFeedbackEntry`,
//! `forward_recurrent()`, and `generate_recurrent()` on a rhyme completion
//! task.  The recurrence re-runs commitment layers with directional feedback
//! derived from token embeddings, giving the model extra depth to sustain
//! planning signals through generation.
//!
//! Inspired by the structural correspondence between DRC planning
//! (Taufeeque et al., "Planning in a recurrent neural network that plays
//! Sokoban", arXiv:2407.15421, 2024) and transformer planning (Lindsey et al.,
//! "On the Biology of a Large Language Model", Anthropic, 2025).
//!
//! See also: Eric Jacopin, "Replicating 'Planning in Poems' with Open Tools"
//! (plip-rs, anacrousis branch) for the full 28-condition experiment.
//!
//! ```bash
//! # Default: Llama 3.2 1B, layers 8-15, strength 2.0, prefill-only
//! cargo run --release --features transformer --example recurrent_feedback
//!
//! # Sustained feedback (applied at every generation step)
//! cargo run --release --features transformer --example recurrent_feedback -- --sustained
//!
//! # Custom layer range and strength
//! cargo run --release --features transformer --example recurrent_feedback -- \
//!     --loop-start 14 --loop-end 15 --strength 1.0 --sustained
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_docs_in_private_items)]

use std::path::PathBuf;

use candle_core::{DType, IndexOp, Tensor};
use clap::Parser;

use candle_mi::{
    GenericTransformer, HookSpec, MIBackend, MITokenizer, RecurrentPassSpec, TransformerConfig,
    sample_token,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "recurrent_feedback")]
#[command(about = "Recurrent feedback (anacrousis) for rhyme completion")]
struct Args {
    /// `HuggingFace` model ID
    #[arg(long, default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// First layer of the recurrent block (inclusive)
    #[arg(long, default_value_t = 8)]
    loop_start: usize,

    /// Last layer of the recurrent block (inclusive)
    #[arg(long, default_value_t = 15)]
    loop_end: usize,

    /// Feedback strength (amplification factor)
    #[arg(long, default_value_t = 2.0)]
    strength: f32,

    /// Apply feedback at every generation step (sustained mode)
    #[arg(long)]
    sustained: bool,

    /// Maximum number of couplets to test (default: all 15)
    #[arg(long)]
    max_couplets: Option<usize>,
}

// ---------------------------------------------------------------------------
// Couplet definitions (canonical 15 from plip-rs anacrousis experiments)
// ---------------------------------------------------------------------------

struct CoupletDef {
    id: u32,
    target_word: &'static str,
    line1: &'static str,
    rhyme_family: &'static [&'static str],
}

fn couplet_defs() -> Vec<CoupletDef> {
    vec![
        CoupletDef {
            id: 1,
            target_word: "light",
            line1: "The moon casts silver light,",
            rhyme_family: &[
                "light", "night", "bright", "sight", "might", "flight", "right", "tight", "white",
                "bite", "kite", "quite", "knight", "delight", "blight", "plight", "slight",
                "fright", "height",
            ],
        },
        CoupletDef {
            id: 2,
            target_word: "play",
            line1: "The children laugh and play,",
            rhyme_family: &[
                "play", "day", "way", "say", "stay", "sway", "ray", "bay", "may", "lay", "pay",
                "gray", "away", "display", "pray", "stray", "clay", "hay", "decay", "delay",
            ],
        },
        CoupletDef {
            id: 3,
            target_word: "sound",
            line1: "The thunder makes a sound,",
            rhyme_family: &[
                "sound", "ground", "found", "round", "bound", "around", "mound", "pound", "hound",
                "wound", "profound", "abound", "astound",
            ],
        },
        CoupletDef {
            id: 4,
            target_word: "rain",
            line1: "The clouds bring heavy rain,",
            rhyme_family: &[
                "rain", "pain", "gain", "main", "vain", "plain", "chain", "train", "brain",
                "strain", "remain", "again", "drain", "lane", "crane", "bane", "wane", "reign",
                "feign",
            ],
        },
        CoupletDef {
            id: 5,
            target_word: "time",
            line1: "The old clock measures time,",
            rhyme_family: &[
                "time", "rhyme", "climb", "crime", "dime", "lime", "mime", "prime", "chime",
                "sublime", "paradigm", "thyme",
            ],
        },
        CoupletDef {
            id: 6,
            target_word: "air",
            line1: "The geese fly through the air,",
            rhyme_family: &[
                "air", "there", "fair", "care", "bare", "dare", "rare", "share", "stare", "where",
                "pair", "aware", "compare", "despair", "prayer", "hair", "chair", "bear", "wear",
                "spare", "snare", "glare",
            ],
        },
        CoupletDef {
            id: 7,
            target_word: "gold",
            line1: "The sunset gleams like gold,",
            rhyme_family: &[
                "gold",
                "old",
                "bold",
                "cold",
                "fold",
                "hold",
                "told",
                "sold",
                "mold",
                "behold",
                "unfold",
                "rolled",
                "controlled",
            ],
        },
        CoupletDef {
            id: 8,
            target_word: "fire",
            line1: "The embers feed the fire,",
            rhyme_family: &[
                "fire", "hire", "wire", "desire", "tire", "inspire", "acquire", "higher", "entire",
                "admire", "liar", "dire", "sire", "pyre", "mire", "conspire", "expire",
            ],
        },
        CoupletDef {
            id: 9,
            target_word: "stone",
            line1: "The castle walls of stone,",
            rhyme_family: &[
                "stone", "bone", "tone", "lone", "zone", "throne", "phone", "own", "known",
                "blown", "grown", "shown", "moan", "groan", "clone", "drone",
            ],
        },
        CoupletDef {
            id: 10,
            target_word: "dream",
            line1: "I wandered through a dream,",
            rhyme_family: &[
                "dream", "stream", "seem", "team", "beam", "cream", "gleam", "scheme", "theme",
                "extreme", "esteem", "scream", "steam",
            ],
        },
        CoupletDef {
            id: 11,
            target_word: "strange",
            line1: "The silence felt so strange,",
            rhyme_family: &[
                "strange", "change", "range", "arrange", "exchange", "grange",
            ],
        },
        CoupletDef {
            id: 12,
            target_word: "love",
            line1: "I never knew such love,",
            rhyme_family: &["love", "above", "dove", "of", "shove", "glove", "thereof"],
        },
        CoupletDef {
            id: 13,
            target_word: "truth",
            line1: "She spoke the honest truth,",
            rhyme_family: &[
                "truth", "youth", "tooth", "booth", "smooth", "sleuth", "ruth", "uncouth",
            ],
        },
        CoupletDef {
            id: 14,
            target_word: "world",
            line1: "He traveled all the world,",
            rhyme_family: &[
                "world", "curled", "unfurled", "whirled", "hurled", "swirled", "pearled", "furled",
                "twirled",
            ],
        },
        CoupletDef {
            id: 15,
            target_word: "earth",
            line1: "The seeds lay in the earth,",
            rhyme_family: &[
                "earth", "birth", "worth", "mirth", "berth", "girth", "dearth", "rebirth", "hearth",
            ],
        },
    ]
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
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mode = if args.sustained {
        "sustained"
    } else {
        "prefill-only"
    };
    println!("=== Recurrent Feedback (Anacrousis) ===\n");
    println!("  Model:     {}", args.model);
    println!("  Layers:    {}–{}", args.loop_start, args.loop_end);
    println!("  Strength:  {:.1}", args.strength);
    println!("  Mode:      {mode}\n");

    // --- Load model ---
    let t_start = std::time::Instant::now();
    eprintln!("Loading model...");
    let (model, tokenizer, _config) = load_transformer(&args.model)?;

    let n_layers = model.num_layers();
    let device = model.embedding_vector(0)?.device().clone();
    eprintln!(
        "  {} layers, {} hidden, device={device:?} ({:.1}s)\n",
        n_layers,
        model.hidden_size(),
        t_start.elapsed().as_secs_f64()
    );

    // --- Prepare couplets ---
    let couplets = couplet_defs();
    let max_couplets = args.max_couplets.unwrap_or(couplets.len());
    let limit = max_couplets.min(couplets.len());
    let couplets = couplets.get(..limit).unwrap_or(&couplets);

    let max_tokens: usize = 30;
    #[allow(clippy::unreadable_literal)]
    let stop_tokens: Vec<u32> = vec![
        128001, // <|end_of_text|>
        128009, // <|eot_id|>
    ];

    // --- Run experiment ---
    println!(
        "  {:>3}  {:>10}  {:>10}  {:>10}  {:>10}  Line 2 (recurrent)",
        "ID", "Target", "Baseline", "Recurrent", "Result"
    );
    println!("  {}", "-".repeat(80));

    let mut baseline_rhymes = 0_usize;
    let mut recurrent_rhymes = 0_usize;

    for couplet in couplets {
        let prompt = format!("{}\n", couplet.line1);
        let prompt_ids = tokenizer.encode(&prompt)?;
        let planning_pos = prompt_ids.len() - 1;

        let rhyme_set: Vec<String> = couplet
            .rhyme_family
            .iter()
            .map(|w| (*w).to_lowercase())
            .collect();

        // --- Baseline generation ---
        let baseline_tokens =
            generate_baseline(&model, &prompt_ids, max_tokens, &stop_tokens, &device)?;
        // INDEX: prompt_ids.len() is always <= baseline_tokens.len()
        // because generate_baseline starts with prompt_tokens.to_vec()
        let baseline_gen = baseline_tokens.get(prompt_ids.len()..).unwrap_or(&[]);
        let baseline_text = tokenizer.decode(baseline_gen)?;
        let baseline_line = baseline_text.lines().next().unwrap_or("");
        let baseline_word = extract_last_word(baseline_line);
        let baseline_ok = word_rhymes(&baseline_word, &rhyme_set);
        if baseline_ok {
            baseline_rhymes += 1;
        }

        // --- Recurrent generation ---
        let rhyme_dir = averaged_rhyme_direction(&model, &tokenizer, couplet.rhyme_family)?;
        let mut spec = RecurrentPassSpec::no_feedback(args.loop_start, args.loop_end)
            .with_sustained(args.sustained);
        spec.add_feedback(planning_pos, rhyme_dir, args.strength);

        let recurrent_tokens =
            model.generate_recurrent(&prompt_ids, max_tokens, 0.0, &stop_tokens, &spec)?;
        // INDEX: prompt_ids.len() is always <= recurrent_tokens.len()
        // because generate_recurrent starts with prompt_tokens.to_vec()
        let recurrent_gen = recurrent_tokens.get(prompt_ids.len()..).unwrap_or(&[]);
        let recurrent_text = tokenizer.decode(recurrent_gen)?;
        let recurrent_line = recurrent_text.lines().next().unwrap_or("");
        let recurrent_word = extract_last_word(recurrent_line);
        let recurrent_ok = word_rhymes(&recurrent_word, &rhyme_set);
        if recurrent_ok {
            recurrent_rhymes += 1;
        }

        let result = match (baseline_ok, recurrent_ok) {
            (false, true) => "RESCUED",
            (true, false) => "REGRESS",
            (true, true) => "OK",
            (false, false) => "-",
        };

        println!(
            "  {:>3}  {:>10}  {:>10}  {:>10}  {:>10}  {}",
            couplet.id,
            couplet.target_word,
            baseline_word,
            recurrent_word,
            result,
            recurrent_line.trim()
        );
    }

    // --- Summary ---
    println!("\n  Baseline:  {baseline_rhymes}/{}", couplets.len());
    println!("  Recurrent: {recurrent_rhymes}/{}", couplets.len());
    match recurrent_rhymes.cmp(&baseline_rhymes) {
        std::cmp::Ordering::Greater => {
            println!(
                "  Improvement: +{} couplet(s) rescued",
                recurrent_rhymes - baseline_rhymes
            );
        }
        std::cmp::Ordering::Less => {
            println!(
                "  Degradation: -{} couplet(s) lost",
                baseline_rhymes - recurrent_rhymes
            );
        }
        std::cmp::Ordering::Equal => {
            println!("  No change in rhyme success rate");
        }
    }

    println!("\n  Total elapsed: {:.2}s", t_start.elapsed().as_secs_f64());

    Ok(())
}

// ---------------------------------------------------------------------------
// Model loading (direct GenericTransformer — needed for generate_recurrent)
// ---------------------------------------------------------------------------

fn load_transformer(
    model_id: &str,
) -> candle_mi::Result<(GenericTransformer, MITokenizer, TransformerConfig)> {
    let files = hf_fetch_model::download_files_blocking(model_id.to_owned())
        .map(hf_fetch_model::DownloadOutcome::into_inner)
        .map_err(|e| candle_mi::MIError::Download(e.to_string()))?;

    let config_path = files
        .get("config.json")
        .ok_or_else(|| candle_mi::MIError::Config("config.json not found".into()))?;
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| candle_mi::MIError::Config(format!("read config.json: {e}")))?;
    let json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| candle_mi::MIError::Config(format!("parse config.json: {e}")))?;

    let config = TransformerConfig::from_hf_config(&json)?;

    let device = candle_core::Device::cuda_if_available(0).map_err(candle_mi::MIError::Model)?;
    let dtype = DType::F32;

    let weights_paths = resolve_safetensors_paths(&files)?;
    let vb = create_var_builder(&weights_paths, dtype, &device)?;

    let model = GenericTransformer::load(config.clone(), &device, dtype, vb)?;

    let tokenizer_path = files
        .get("tokenizer.json")
        .ok_or_else(|| candle_mi::MIError::Tokenizer("tokenizer.json not found".into()))?;
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    Ok((model, tokenizer, config))
}

/// Resolve safetensors file paths from a downloaded file map.
fn resolve_safetensors_paths(
    files: &std::collections::HashMap<String, PathBuf>,
) -> candle_mi::Result<Vec<PathBuf>> {
    // Try sharded first
    if let Some(index_path) = files.get("model.safetensors.index.json") {
        let index_str = std::fs::read_to_string(index_path)
            .map_err(|e| candle_mi::MIError::Config(format!("read index: {e}")))?;
        let index: serde_json::Value = serde_json::from_str(&index_str)
            .map_err(|e| candle_mi::MIError::Config(format!("parse index: {e}")))?;
        let weight_map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| candle_mi::MIError::Config("missing weight_map in index".into()))?;

        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .map(String::from)
            .collect();
        shard_names.sort();
        shard_names.dedup();

        let mut paths = Vec::with_capacity(shard_names.len());
        for name in &shard_names {
            // BORROW: explicit .as_str() — String → &str for HashMap lookup
            let path = files
                .get(name.as_str())
                .ok_or_else(|| candle_mi::MIError::Config(format!("shard {name} not found")))?;
            paths.push(path.clone());
        }
        return Ok(paths);
    }

    // Single file
    let path = files
        .get("model.safetensors")
        .ok_or_else(|| candle_mi::MIError::Config("model.safetensors not found".into()))?;
    Ok(vec![path.clone()])
}

/// Create a `VarBuilder` from safetensors file paths.
fn create_var_builder(
    paths: &[PathBuf],
    dtype: DType,
    device: &candle_core::Device,
) -> candle_mi::Result<candle_nn::VarBuilder<'static>> {
    #[cfg(feature = "mmap")]
    {
        // SAFETY: safetensors files must not be modified while loaded.
        #[allow(unsafe_code)]
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
        Ok(vb)
    }
    #[cfg(not(feature = "mmap"))]
    {
        if paths.len() > 1 {
            return Err(candle_mi::MIError::Config(
                "sharded models require the `mmap` feature".into(),
            ));
        }
        let path = paths
            .first()
            .ok_or_else(|| candle_mi::MIError::Config("no safetensors files".into()))?;
        let data = std::fs::read(path)
            .map_err(|e| candle_mi::MIError::Config(format!("read {}: {e}", path.display())))?;
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(data, dtype, device)?;
        Ok(vb)
    }
}

// ---------------------------------------------------------------------------
// Generation helpers
// ---------------------------------------------------------------------------

/// Greedy baseline generation (standard forward, no recurrence).
fn generate_baseline(
    model: &GenericTransformer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    stop_tokens: &[u32],
    device: &candle_core::Device,
) -> candle_mi::Result<Vec<u32>> {
    let mut tokens = prompt_tokens.to_vec();

    for _ in 0..max_tokens {
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let cache = model.forward(&input, &HookSpec::new())?;
        let logits = cache.output();
        let seq_len = logits.dim(1)?;
        let last_logits = logits.i((.., seq_len - 1, ..))?.squeeze(1)?.flatten_all()?;
        let next_token = sample_token(&last_logits, 0.0)?;
        if stop_tokens.contains(&next_token) {
            break;
        }
        tokens.push(next_token);
    }
    Ok(tokens)
}

/// Compute L2-normalised average of embedding vectors for rhyme family words.
fn averaged_rhyme_direction(
    model: &GenericTransformer,
    tokenizer: &MITokenizer,
    rhyme_words: &[&str],
) -> candle_mi::Result<Tensor> {
    let mut embeddings: Vec<Tensor> = Vec::new();

    for word in rhyme_words {
        let with_space = format!(" {word}");
        let ids = tokenizer.encode_raw(&with_space)?;
        let Some(&token_id) = ids.last() else {
            continue;
        };
        let emb = model.embedding_vector(token_id)?;
        embeddings.push(emb);
    }

    if embeddings.is_empty() {
        return Err(candle_mi::MIError::Tokenizer(
            "no valid embeddings for rhyme family".into(),
        ));
    }

    let stacked = Tensor::stack(&embeddings, 0)?;
    let avg = stacked.mean(0)?;
    // PROMOTE: ensure F32 for norm computation
    let avg_f32 = avg.to_dtype(DType::F32)?;
    let norm: f32 = avg_f32.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
    if norm > 1e-8 {
        Ok(avg_f32.affine(1.0 / f64::from(norm), 0.0)?)
    } else {
        Ok(avg_f32)
    }
}

/// Extract the last word-like token from generated text.
fn extract_last_word(text: &str) -> String {
    text.split_whitespace()
        .next_back()
        .unwrap_or("")
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

/// Check if a word matches any member of the rhyme family.
fn word_rhymes(word: &str, rhyme_family: &[String]) -> bool {
    let clean = word
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase();
    rhyme_family.contains(&clean)
}
