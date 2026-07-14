// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared Figure-13 cell presets.
//!
//! This module is `#[path]`-included by the Figure-13 example binaries
//! (`figure13_planning_poems`, `figure13_newline_census`,
//! `figure13_newline_steering`) so the annotated preset table lives in exactly
//! one place. It is **not** a standalone example: it has no `main`, and it
//! lives in a subdirectory of `examples/` without a `main.rs`, so Cargo's
//! example auto-discovery skips it.
//!
//! Each [`Preset`] fixes a (model, `CLT`, prompt, natural rhyme word,
//! alternative inject word, feature set, strength) cell. The eight presets are
//! the union of every cell used across the Figure-13 sweeps; the seven that
//! make up Table 2 of the BlackboxNLP paper (`tab:cells`) are enumerated in
//! [`CENSUS_CELLS`] (all eight minus `gemma2-2b-2.5m`, the word-level
//! granularity variant that is not one of the seven reported cells).
//!
//! Every consumer includes the whole table but uses a different subset of the
//! helpers, so the module allows `dead_code` rather than gating each item per
//! consumer.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]

use candle_mi::clt::CltFeatureId;

// ── Preset table ─────────────────────────────────────────────────────────────

/// One Figure-13 experimental cell: a fixed (model, `CLT`, prompt, feature)
/// tuple with a natural rhyme word to suppress and an alternative word to
/// inject.
pub struct Preset {
    /// `HuggingFace` model id (e.g. `google/gemma-2-2b`).
    pub model: &'static str,
    /// `HuggingFace` `CLT` repository id (e.g. `mntss/clt-gemma-2-2b-426k`).
    pub clt_repo: &'static str,
    /// Four-line rhyming prompt ending mid-fourth-line, so the natural rhyme
    /// word is the next token after a trailing space.
    pub prompt: &'static str,
    /// The prompt's natural rhyme word (the word suppression removes).
    pub suppress_word: &'static str,
    /// The alternative-group word the inject feature steers toward.
    pub inject_word: &'static str,
    /// Suppress features as `(layer, index)` pairs — the natural rhyme group's
    /// top features by decoder cosine.
    pub suppress_features: &'static [(usize, usize)],
    /// Inject feature as a `(layer, index)` pair — the alternative-group pick.
    pub inject_feature: (usize, usize),
    /// Default steering strength for single-strength runs.
    pub strength: f32,
}

/// The seven cells reported in Table 2 (`tab:cells`) of the paper, by preset
/// name. This is all eight presets **except** `gemma2-2b-2.5m`.
pub const CENSUS_CELLS: &[&str] = &[
    "gemma2-2b-426k",
    "llama3.2-1b-524k",
    "qwen3-0.6b-16k-ation",
    "qwen3-0.6b-20k-teen",
    "qwen3-1.7b-20k-teen",
    "qwen3-0.6b-20k-ation",
    "qwen3-1.7b-20k-ation",
];

/// Llama 3.2 1B with 524K CLT.
///
/// Suppress -ee group: L13:30985 ("he"), L9:5488 ("be"), L14:27874 ("ne"),
/// L13:32049 ("we").  Inject "that" (L14:13043) from -at group.
/// From plip-rs validation: P("that") reaches 0.777 at the last position.
pub const LLAMA: Preset = Preset {
    model: "meta-llama/Llama-3.2-1B",
    clt_repo: "mntss/clt-llama-3.2-1b-524k",
    prompt: "The birds were singing in the tree,\n\
             And everything was wild and free.\n\
             The river ran down to the sea,\n\
             There is so much we cannot",
    suppress_word: "free",
    inject_word: "that",
    suppress_features: &[(13, 30985), (9, 5488), (14, 27874), (13, 32049)],
    inject_feature: (14, 13043),
    strength: 10.0,
};

/// Gemma 2 2B with 426K CLT.
///
/// Suppress "-out" group: L16:13725 ("about") + L25:9385 ("out").
/// Inject "around" (L22:10243).
/// From plip-rs validation: P("around") reaches 0.483 at planning site.
pub const GEMMA: Preset = Preset {
    model: "google/gemma-2-2b",
    clt_repo: "mntss/clt-gemma-2-2b-426k",
    prompt: "The stars were twinkling in the night,\n\
             The lanterns cast a golden light.\n\
             She wandered in the dark about,\n\
             And found a hidden passage",
    suppress_word: "out",
    inject_word: "around",
    suppress_features: &[(16, 13725), (25, 9385)],
    inject_feature: (22, 10243),
    strength: 10.0,
};

/// Gemma 2 2B with 2.5M CLT (word-level granularity).
///
/// Suppress "-out" words: L25:57092 ("about") + L23:49923 ("out") + L20:77102
/// ("without"). Inject "can" (L25:82839).
/// From plip-rs validation: P("can") reaches 0.425 at planning site.
///
/// **Not** one of the seven Table 2 cells (see [`CENSUS_CELLS`]).
pub const GEMMA_2M: Preset = Preset {
    model: "google/gemma-2-2b",
    clt_repo: "mntss/clt-gemma-2-2b-2.5M",
    prompt: "The stars were twinkling in the night,\n\
             The lanterns cast a golden light.\n\
             She wandered in the dark about,\n\
             And found a hidden passage",
    suppress_word: "out",
    inject_word: "can",
    suppress_features: &[(25, 57092), (23, 49923), (20, 77102)],
    inject_feature: (25, 82839),
    strength: 10.0,
};

// ── Shared `Qwen3` prompts ──────────────────────────────────────────────────
//
// Each `Qwen3` preset reuses one of two four-line rhyming-couplet prompts,
// each ending mid-fourth-line so the planning site sits at the trailing-space
// spike before the natural rhyme word.

/// Four-line `-ation` poem; natural last word is *duration* (or another
/// `EY1 SH AH0 N`-rime word).  Used by all three `qwen3-*-ation` presets.
const QWEN3_ATION_PROMPT: &str = "At every grand celebration,\n\
                                  Each careful preparation,\n\
                                  Brings joy beyond expectation,\n\
                                  And then the brief";

/// Four-line `-teen` poem; natural last word is *seventeen* (a `-teen`
/// numeral not yet present in the prompt).  Used by both `qwen3-*-teen`
/// presets.
const QWEN3_TEEN_PROMPT: &str = "She counted thirteen, then fourteen,\n\
                                 Followed shortly by fifteen,\n\
                                 And carefully whispered sixteen,\n\
                                 Before she reached";

/// `Qwen3-1.7B-Base` × `BlueLightAI` 20K `CLT`, `-ation` suppress + `-self`
/// inject.  Suppress features = top 3 `EY1 SH AH0 N` features by `max_cosine`
/// (broad rime coverage).  Inject feature = `L21:3908`, picked by
/// **highest cosine to `" myself"` directly** (`cos = 0.39`) rather than
/// by overall `max_cosine`; the original top-`EH1 L F` pick `L23:11747`
/// only had `cos = 0.25` to `" myself"` even though its rime-membership
/// score was higher.  Vocab scan summary: 84 clean `-ation` features,
/// 7 clean `-self` features.
pub const QWEN3_1_7B_20K_ATION: Preset = Preset {
    model: "Qwen/Qwen3-1.7B-Base",
    clt_repo: "bluelightai/clt-qwen3-1.7b-base-20k",
    prompt: QWEN3_ATION_PROMPT,
    suppress_word: "duration",
    inject_word: "myself",
    suppress_features: &[(15, 263), (18, 3801), (18, 4404)],
    inject_feature: (21, 3908),
    strength: 10.0,
};

/// `Qwen3-1.7B-Base` × `BlueLightAI` 20K `CLT`, `-teen` suppress + `-ation`
/// inject.  Suppress features = top 3 `IY1 N` features by `max_cosine`.
/// Inject feature = `L15:263`, the top **`EY1 SH AH0 N`-rime feature by
/// `max_cosine`** (i.e. the *cluster-broad* pick).  We empirically compared
/// against a `" duration"`-cosine-specific pick (`L24:17759`, `cos = 0.22`)
/// and found the cluster-broad feature gives a 14× larger planning-site
/// redirect (16.42× at `s = 5` vs 1.17× at `s = 25`).  Interpretation: the
/// duration-specific feature is too narrow to displace the model's strong
/// `-teen` prior; the broad `-ation`-cluster feature offers the model a
/// well-defined alternative *region* of token space to commit to instead.
/// Vocab scan summary: 30 clean `-teen` features, 84 clean `-ation` features.
pub const QWEN3_1_7B_20K_TEEN: Preset = Preset {
    model: "Qwen/Qwen3-1.7B-Base",
    clt_repo: "bluelightai/clt-qwen3-1.7b-base-20k",
    prompt: QWEN3_TEEN_PROMPT,
    suppress_word: "seventeen",
    inject_word: "duration",
    suppress_features: &[(27, 16975), (20, 3668), (18, 10986)],
    inject_feature: (15, 263),
    strength: 10.0,
};

/// `Qwen3-0.6B-Base` × `BlueLightAI` 20K `CLT`, `-ation` suppress + `-self`
/// inject.  Suppress features = top 3 `EY1 SH AH0 N` features by `max_cosine`.
/// Inject feature = `L22:4081`, which happens to be **both** the top
/// `EH1 L F` feature by overall `max_cosine` **and** the feature with the
/// highest cosine to `" myself"` (`cos = 0.42`) — no swap needed.  Vocab
/// scan summary: 71 clean `-ation` features, 8 clean `-self` features.
pub const QWEN3_0_6B_20K_ATION: Preset = Preset {
    model: "Qwen/Qwen3-0.6B-Base",
    clt_repo: "bluelightai/clt-qwen3-0.6b-base-20k",
    prompt: QWEN3_ATION_PROMPT,
    suppress_word: "duration",
    inject_word: "myself",
    suppress_features: &[(19, 9578), (0, 8867), (25, 4979)],
    inject_feature: (22, 4081),
    strength: 10.0,
};

/// `Qwen3-0.6B-Base` × `BlueLightAI` 20K `CLT`, `-teen` suppress + `-ation`
/// inject.  Suppress features = top 3 `IY1 N` features by `max_cosine`.
/// Inject feature = `L19:9578`, the top **`EY1 SH AH0 N`-rime feature by
/// `max_cosine`** (cluster-broad pick).  We empirically compared against a
/// `" duration"`-cosine-specific pick (`L15:2229`, `cos = 0.19`) and found
/// the cluster-broad feature gives a 3× larger planning-site redirect
/// (157× at `s = 1` vs 49.5× at `s = 0.5`).  Same conclusion as the
/// 1.7B sibling: narrow features ≠ better inject targets when the
/// suppress side is doing the heavy lifting on the natural rhyme prior.
/// Vocab scan summary: 33 clean `-teen` features, 71 clean `-ation` features.
pub const QWEN3_0_6B_20K_TEEN: Preset = Preset {
    model: "Qwen/Qwen3-0.6B-Base",
    clt_repo: "bluelightai/clt-qwen3-0.6b-base-20k",
    prompt: QWEN3_TEEN_PROMPT,
    suppress_word: "seventeen",
    inject_word: "duration",
    suppress_features: &[(27, 16425), (23, 15839), (26, 6308)],
    inject_feature: (19, 9578),
    strength: 10.0,
};

/// `Qwen3-0.6B-Base` × `BlueLightAI`-dev 16K `CLT`, `-ation` suppress +
/// `-self` inject.  Suppress features = top 3 `EY1 SH AH0 N` features by
/// `max_cosine`.  Inject feature = `L22:8011`, picked by **highest cosine
/// to `" myself"` directly** (`cos = 0.30`, top tokens all `my` variants);
/// the previous top-`EH1 L F` pick `L15:6772` had no `" myself"` in its
/// top-20 at all.  Vocab scan summary: 27 clean `-ation` features, 3 clean
/// `-self` features.
pub const QWEN3_0_6B_16K_ATION: Preset = Preset {
    model: "Qwen/Qwen3-0.6B-Base",
    clt_repo: "bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test",
    prompt: QWEN3_ATION_PROMPT,
    suppress_word: "duration",
    inject_word: "myself",
    suppress_features: &[(23, 11154), (20, 10987), (14, 10719)],
    inject_feature: (22, 8011),
    strength: 10.0,
};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Select a built-in preset by name.
///
/// # Errors
///
/// Returns [`MIError::Config`](candle_mi::MIError::Config) if `name` is not one
/// of the eight known preset names.
pub fn select_preset(name: &str) -> candle_mi::Result<&'static Preset> {
    match name {
        "llama3.2-1b-524k" => Ok(&LLAMA),
        "gemma2-2b-426k" => Ok(&GEMMA),
        "gemma2-2b-2.5m" => Ok(&GEMMA_2M),
        "qwen3-1.7b-20k-ation" => Ok(&QWEN3_1_7B_20K_ATION),
        "qwen3-1.7b-20k-teen" => Ok(&QWEN3_1_7B_20K_TEEN),
        "qwen3-0.6b-20k-ation" => Ok(&QWEN3_0_6B_20K_ATION),
        "qwen3-0.6b-20k-teen" => Ok(&QWEN3_0_6B_20K_TEEN),
        "qwen3-0.6b-16k-ation" => Ok(&QWEN3_0_6B_16K_ATION),
        other => Err(candle_mi::MIError::Config(format!(
            "unknown preset '{other}' (expected one of: 'llama3.2-1b-524k', \
             'gemma2-2b-426k', 'gemma2-2b-2.5m', 'qwen3-1.7b-20k-ation', \
             'qwen3-1.7b-20k-teen', 'qwen3-0.6b-20k-ation', \
             'qwen3-0.6b-20k-teen', 'qwen3-0.6b-16k-ation')"
        ))),
    }
}

/// Convert a `(layer, index)` pair into a [`CltFeatureId`].
#[must_use]
pub const fn feature_id(pair: (usize, usize)) -> CltFeatureId {
    CltFeatureId {
        layer: pair.0,
        index: pair.1,
    }
}

/// Parse a `"layer:index"` string into a [`CltFeatureId`].
///
/// # Errors
///
/// Returns [`MIError::Config`](candle_mi::MIError::Config) if the string is not
/// in `"layer:index"` form, or if either component is not a valid `usize`.
pub fn parse_feature(s: &str) -> candle_mi::Result<CltFeatureId> {
    let (layer_str, index_str) = s.split_once(':').ok_or_else(|| {
        candle_mi::MIError::Config(format!(
            "feature must be in 'layer:index' format, got '{s}'"
        ))
    })?;
    let layer: usize = layer_str.parse().map_err(|e| {
        candle_mi::MIError::Config(format!("invalid layer number '{layer_str}': {e}"))
    })?;
    let index: usize = index_str.parse().map_err(|e| {
        candle_mi::MIError::Config(format!("invalid feature index '{index_str}': {e}"))
    })?;
    Ok(CltFeatureId { layer, index })
}
