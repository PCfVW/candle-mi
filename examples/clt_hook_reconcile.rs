// SPDX-License-Identifier: MIT OR Apache-2.0

//! Machinery-reconciliation positive control for the newline census
//! (Experiment 1, amended spec 2026-07-13).
//!
//! plip-rs "detection V2" documented specific `(feature, prompt, activation)`
//! triples on Gemma 2 2B × `mntss/clt-gemma-2-2b-426k` — e.g. the `-out`
//! feature `L25:9385` reads **0.247** at the trailing-space planning site of
//! the "about" completion prompt. The census encode path must reproduce those
//! numbers before any null it reports is trustworthy.
//!
//! The one degree of freedom is **which residual the `CLT` encoder reads**.
//! candle-mi's census / `clt_probe` read `ResidMid` (post-attention, pre-MLP,
//! the circuit-tracer convention); plip-rs's `forward_with_cache` caches the
//! **layer output** (`ResidPost`, post-MLP) and encodes that. This example
//! encodes each tracked feature under **all three** hook points
//! (`ResidPre` / `ResidMid` / `ResidPost`) at the planning position, so the
//! hook that reproduces the documented activation is identified directly
//! rather than assumed.
//!
//! ```bash
//! # -out / -about features on the Gemma "about" prompt (default)
//! cargo run --release --features clt,transformer,mmap --example clt_hook_reconcile
//!
//! # -ow "go" feature (0.983) on the "so" prompt
//! cargo run --release --features clt,transformer,mmap --example clt_hook_reconcile -- --scenario go
//!
//! # -oo "ou" feature (0.359) on the "who" prompt
//! cargo run --release --features clt,transformer,mmap --example clt_hook_reconcile -- --scenario ou
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use candle_core::Tensor;
use clap::Parser;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookPoint, HookSpec, MIModel};

// ── Scenarios ────────────────────────────────────────────────────────────────

/// A tracked feature with the plip-rs detection-V2 activation to reproduce.
struct Tracked {
    /// Human label (the rhyme word the feature decodes toward).
    label: &'static str,
    /// Feature layer.
    layer: usize,
    /// Feature index within the layer.
    index: usize,
    /// plip-rs detection-V2 activation at the planning site (`None` if
    /// undocumented for this feature).
    expected: Option<f32>,
}

/// A (model, CLT, prompt, tracked-features) reconciliation scenario.
struct Scenario {
    /// Scenario name for `--scenario`.
    name: &'static str,
    /// `HuggingFace` model id.
    model: &'static str,
    /// `HuggingFace` CLT repository id.
    clt_repo: &'static str,
    /// Completion prompt (the trailing space is appended by the runner, so the
    /// last token is the space before the gap word — plip-rs convention).
    prompt: &'static str,
    /// Features to track under each hook point.
    features: &'static [Tracked],
}

/// The Gemma "about" completion prompt — identical to the `gemma2-2b-426k`
/// census prompt and to plip-rs's `("-out", "about", "out")` candidate.
const GEMMA_ABOUT_PROMPT: &str = "The stars were twinkling in the night,\n\
                                  The lanterns cast a golden light.\n\
                                  She wandered in the dark about,\n\
                                  And found a hidden passage";

/// The Gemma "so" completion prompt — plip-rs's `("-ow", "so", "go")`
/// candidate (track the `-ow` "go" feature, documented at 0.983).
const GEMMA_SO_PROMPT: &str = "A sailor sailed across the bay,\n\
                               And dreamed of home throughout the day.\n\
                               The world keeps spinning even so,\n\
                               There is so much we do not";

/// The Gemma "who" completion prompt — plip-rs's `("-oo", "who", "ou")`
/// candidate (track the `-oo` "ou" feature, documented at 0.359).
const GEMMA_WHO_PROMPT: &str = "The sun goes up, the sun goes down,\n\
                                The moon shines bright above the town.\n\
                                Nobody knows or remembers who,\n\
                                Would come to find a way back";

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "out-about",
        model: "google/gemma-2-2b",
        clt_repo: "mntss/clt-gemma-2-2b-426k",
        prompt: GEMMA_ABOUT_PROMPT,
        features: &[
            Tracked {
                label: "out",
                layer: 25,
                index: 9385,
                expected: Some(0.247),
            },
            Tracked {
                label: "about",
                layer: 16,
                index: 13725,
                expected: None,
            },
        ],
    },
    Scenario {
        name: "go",
        model: "google/gemma-2-2b",
        clt_repo: "mntss/clt-gemma-2-2b-426k",
        prompt: GEMMA_SO_PROMPT,
        features: &[Tracked {
            label: "go",
            layer: 25,
            index: 4505,
            expected: Some(0.983),
        }],
    },
    Scenario {
        name: "ou",
        model: "google/gemma-2-2b",
        clt_repo: "mntss/clt-gemma-2-2b-426k",
        prompt: GEMMA_WHO_PROMPT,
        features: &[Tracked {
            label: "ou",
            layer: 25,
            index: 5927,
            expected: Some(0.359),
        }],
    },
];

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "clt_hook_reconcile")]
#[command(
    about = "Reconcile the census encode hook point against plip-rs detection-V2 activations"
)]
struct Args {
    /// Scenario: `out-about` (default), `go`, or `ou`.
    #[arg(long, default_value = "out-about")]
    scenario: String,

    /// Also print the full per-position activation profile (every token
    /// position) under each hook, not just the planning position.
    #[arg(long, default_value_t = false)]
    all_positions: bool,
}

// ── Hook points under test ───────────────────────────────────────────────────

/// The three residual read points, in stack order.
const fn hook_at(kind: HookKind, layer: usize) -> HookPoint {
    match kind {
        HookKind::Pre => HookPoint::ResidPre(layer),
        HookKind::Mid => HookPoint::ResidMid(layer),
        HookKind::Post => HookPoint::ResidPost(layer),
    }
}

#[derive(Clone, Copy)]
enum HookKind {
    Pre,
    Mid,
    Post,
}

impl HookKind {
    const ALL: [Self; 3] = [Self::Pre, Self::Mid, Self::Post];

    const fn label(self) -> &'static str {
        match self {
            Self::Pre => "ResidPre ",
            Self::Mid => "ResidMid ",
            Self::Post => "ResidPost",
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let scenario = SCENARIOS
        .iter()
        .find(|s| s.name == args.scenario)
        .ok_or_else(|| {
            candle_mi::MIError::Config(format!(
                "unknown scenario '{}' (expected 'out-about', 'go', or 'ou')",
                args.scenario
            ))
        })?;

    eprintln!(
        "=== CLT hook-point reconciliation (scenario: {}) ===\n",
        scenario.name
    );
    eprintln!("Model: {}", scenario.model);
    eprintln!("CLT:   {}\n", scenario.clt_repo);

    let model = MIModel::from_pretrained(scenario.model)?;
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    let mut clt = CrossLayerTranscoder::open(scenario.clt_repo)?;

    // Trailing space: the last token is the space before the gap word (plip-rs
    // convention, matching the census).
    let prompt_with_space = format!("{} ", scenario.prompt);
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();
    let planning_pos = seq_len - 1;
    let token_strs: Vec<String> = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();
    eprintln!(
        "Prompt tokens ({seq_len}); planning position = {planning_pos} (\"{}\")\n",
        token_strs
            .get(planning_pos)
            .map_or("?", String::as_str)
            .replace('\n', "\\n")
    );

    // One forward capturing all three hook points at every tracked layer.
    let mut hooks = HookSpec::new();
    for feat in scenario.features {
        for kind in HookKind::ALL {
            hooks.capture(hook_at(kind, feat.layer));
        }
    }
    let cache = model.forward(&Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?, &hooks)?;

    for feat in scenario.features {
        let fid = CltFeatureId {
            layer: feat.layer,
            index: feat.index,
        };
        clt.load_encoder(feat.layer, &device)?;

        eprintln!(
            "── feature L{}:{} (\"{}\") ──",
            feat.layer, feat.index, feat.label
        );
        if let Some(exp) = feat.expected {
            eprintln!("   plip-rs detection-V2 expected activation: {exp:.4}");
        }
        eprintln!("   planning position ({planning_pos}):");
        for kind in HookKind::ALL {
            let act =
                encode_feature_at(&clt, &cache, &hook_at(kind, feat.layer), fid, planning_pos)?;
            let flag = match feat.expected {
                // Within 10% relative (stack-drift tolerance) of the documented value.
                Some(exp) if exp > 0.0 && (act - exp).abs() <= 0.1 * exp => {
                    "  <== matches expected"
                }
                _ => "",
            };
            eprintln!("     {}  act = {act:.4}{flag}", kind.label());
        }

        if args.all_positions {
            eprintln!("   full per-position profile:");
            for pos in 0..seq_len {
                let tok = token_strs
                    .get(pos)
                    .map_or("?", String::as_str)
                    .replace('\n', "\\n");
                let pre =
                    encode_feature_at(&clt, &cache, &HookPoint::ResidPre(feat.layer), fid, pos)?;
                let mid =
                    encode_feature_at(&clt, &cache, &HookPoint::ResidMid(feat.layer), fid, pos)?;
                let post =
                    encode_feature_at(&clt, &cache, &HookPoint::ResidPost(feat.layer), fid, pos)?;
                eprintln!(
                    "     pos {pos:>3} {tok:<14}  pre={pre:.4}  mid={mid:.4}  post={post:.4}"
                );
            }
        }
        eprintln!();
    }

    Ok(())
}

/// Encode the residual captured at `hook` and position `pos`, returning the
/// activation of feature `fid` (0.0 if the feature is not active there).
fn encode_feature_at(
    clt: &CrossLayerTranscoder,
    cache: &candle_mi::HookCache,
    hook: &HookPoint,
    fid: CltFeatureId,
    pos: usize,
) -> candle_mi::Result<f32> {
    // `Tensor::get` selects along dim 0: batch row then position row.
    let residual = cache.require(hook)?.get(0)?.get(pos)?;
    let sparse = clt.encode(&residual, fid.layer)?;
    // The CLT encode returns only active features; a missing feature reads 0.
    let act = sparse
        .features
        .iter()
        .find(|(f, _)| *f == fid)
        .map_or(0.0, |(_, a)| *a);
    Ok(act)
}
