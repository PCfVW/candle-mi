// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gridworld prolepsis — does early-irrevocable-commitment ("prolepsis") in
//! rhyme planning transfer to 2D-gridworld action planning?
//!
//! This example hosts the staged experiment of
//! `docs/roadmaps/PLAN-GRIDWORLD-PROLEPSIS.md`. It is driven by subcommands,
//! one per step:
//!
//! - **`scaffold`** (Step 0) — tokenization sanity check + prompt-formatter
//!   dry-run. Verifies the four mapped action tokens (`black`, `kind`, `well`,
//!   `round`) are single tokens in the Gemma 2 2B vocabulary and prints the
//!   formatted planning prompts.
//! - **`baseline`** (Step A) — baseline feasibility. Runs the prompts through
//!   the model (no intervention) and measures whether Gemma 2 2B reliably
//!   produces the correct mapped action, both as full-vocabulary top-1 and as a
//!   forced choice among the four action tokens. Prepends `--few-shot` solved
//!   demonstrations drawn balanced across the four actions from a fixed pool,
//!   with their order randomized per instance (`--seed`) to defeat the
//!   primacy/copying bias; the test instance ends at the `"):"` planning site.
//!
//! The cardinal-action-to-token mapping is a CLI parameter so the same binary
//! drives the Step B baseline and the Step C permutation without a rebuild:
//!
//! | Mapping | Up | Down | Left | Right |
//! |---------|----|------|------|-------|
//! | `baseline`  | `black` | `kind` | `well`  | `round` |
//! | `permuted`  | `round` | `well` | `kind`  | `black` |
//!
//! Each instance is rendered with the prompt template below. The cue ends at
//! the `"):"` token (no trailing space) — that token is the planning site at
//! which Step B probes the spike, and the model's next token is the
//! space-prefixed action token (e.g. `▁black`). This is the action-planning
//! analogue of the rhyme planning site in `figure13_planning_poems`.
//!
//! ```text
//! Grid: 5x5. Agent: (ax,ay). Goal: (gx,gy). Walls: none.
//! Map: Up→m_up, Down→m_down, Left→m_left, Right→m_right.
//! Best next move (m_up/m_down/m_left/m_right):
//! ```
//!
//! ```bash
//! # Step 0 — tokenization check + prompt dry-run (loads cached Gemma 2 2B)
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- scaffold
//!
//! # Step 0 — formatter only, no model load
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- scaffold --skip-token-check
//!
//! # Step A — baseline feasibility on the 100 instances
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- baseline
//!
//! # Step A — permuted mapping
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- baseline --mapping permuted
//!
//! # Step A — zero-shot (no demonstrations)
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- baseline --few-shot 0
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Tensor};
use clap::{Args as ClapArgs, Parser, Subcommand};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use candle_mi::{HookCache, HookSpec, MIModel, MITokenizer};

/// Baseline (Step A) exit threshold: both accuracies must reach this.
const THRESHOLD: f64 = 0.80;

/// Human-readable description of the prompt template, recorded in the JSON so
/// Step B can confirm it inherits the exact format Step A validated. The cue
/// ends at `"):"` (no trailing space) so the model's next token is the
/// space-prefixed action token (e.g. `▁black`), which is also the planning site.
const PROMPT_FORMAT: &str = "Grid: {N}x{N}. Agent: ({ax},{ay}). Goal: ({gx},{gy}). \
                             Walls: none.\nMap: Up→{up}, Down→{down}, Left→{left}, \
                             Right→{right}.\nBest next move ({up}/{down}/{left}/{right}):";

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "gridworld_prolepsis")]
#[command(about = "Gridworld prolepsis experiment (PLAN-GRIDWORLD-PROLEPSIS.md)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// One subcommand per experiment step.
#[derive(Subcommand)]
#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: dispatched exhaustively in `run`
enum Command {
    /// Step 0: tokenization sanity check + prompt-formatter dry-run.
    Scaffold(ScaffoldArgs),
    /// Step A: baseline feasibility (top-1 accuracy, no intervention).
    Baseline(BaselineArgs),
}

/// Options shared by every step.
#[derive(ClapArgs)]
struct CommonArgs {
    /// Path to the gridworld instances JSON (output of
    /// `scripts/gridworld_generator.py`).
    #[arg(
        long,
        default_value = "docs/experiments/gridworld-prolepsis/gridworld_instances.json"
    )]
    instances: PathBuf,

    /// Action-to-token mapping: `baseline` (`Up→black, Down→kind, Left→well,
    /// Right→round`) or `permuted` (`Up→round, Down→well, Left→kind,
    /// Right→black`).
    #[arg(long, default_value = "baseline")]
    mapping: String,

    /// Override the token mapped to `Up` (applied on top of `--mapping`).
    #[arg(long)]
    map_up: Option<String>,

    /// Override the token mapped to `Down` (applied on top of `--mapping`).
    #[arg(long)]
    map_down: Option<String>,

    /// Override the token mapped to `Left` (applied on top of `--mapping`).
    #[arg(long)]
    map_left: Option<String>,

    /// Override the token mapped to `Right` (applied on top of `--mapping`).
    #[arg(long)]
    map_right: Option<String>,

    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Grid side length `N` used when rendering the `NxN` header. Must match
    /// the `--grid-size` passed to `scripts/gridworld_generator.py`.
    #[arg(long, default_value_t = 5)]
    grid_size: u32,

    /// Grid encoding: `coords` (`(x,y)` tuples) or `ascii` (rendered grid).
    #[arg(long, default_value = "coords")]
    encoding: String,
}

/// `scaffold` (Step 0) options.
#[derive(ClapArgs)]
struct ScaffoldArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Number of formatted prompts to print as a dry-run sample.
    #[arg(long, default_value_t = 5)]
    max_print: usize,

    /// Skip the tokenization sanity check (and the model load it requires);
    /// run the prompt formatter alone.
    #[arg(long, default_value_t = false)]
    skip_token_check: bool,
}

/// `baseline` (Step A) options.
#[derive(ClapArgs)]
struct BaselineArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Output JSON path for the per-instance baseline results.
    #[arg(
        long,
        default_value = "docs/experiments/gridworld-prolepsis/baseline_gemma2_2b_2.5m.json"
    )]
    output: PathBuf,

    /// Number of few-shot demonstrations to prepend (0 = zero-shot), drawn
    /// balanced across the four actions and capped at the pool size (24).
    #[arg(long, default_value_t = 4)]
    few_shot: usize,

    /// RNG seed for the per-instance demonstration-order shuffle.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Number of per-instance lines to echo to stderr during the run.
    #[arg(long, default_value_t = 10)]
    max_print: usize,
}

// ── Domain types ──────────────────────────────────────────────────────────────

/// A single gridworld instance, deserialized from the generator's JSON.
#[derive(Deserialize)]
struct GridInstance {
    /// Agent cell as `[x, y]`, `0 <= x, y < grid_size`.
    agent: [u32; 2],
    /// Goal cell as `[x, y]`, `0 <= x, y < grid_size`.
    goal: [u32; 2],
    /// Ground-truth dominant first action: `Up`, `Down`, `Left`, or `Right`.
    correct_action: String,
    /// Zero-based index assigned by the generator.
    instance_id: usize,
}

/// The four cardinal moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: internal dispatch enum, matched exhaustively in this example
enum CardinalAction {
    Up,
    Down,
    Left,
    Right,
}

impl CardinalAction {
    /// All four actions in a fixed order (matches the generator's `ACTIONS`).
    const ALL: [Self; 4] = [Self::Up, Self::Down, Self::Left, Self::Right];

    /// The canonical string label for this action.
    const fn label(self) -> &'static str {
        match self {
            Self::Up => "Up",
            Self::Down => "Down",
            Self::Left => "Left",
            Self::Right => "Right",
        }
    }

    /// Parse an action from its canonical label.
    fn from_label(label: &str) -> candle_mi::Result<Self> {
        match label {
            "Up" => Ok(Self::Up),
            "Down" => Ok(Self::Down),
            "Left" => Ok(Self::Left),
            "Right" => Ok(Self::Right),
            other => Err(candle_mi::MIError::Config(format!(
                "unknown correct_action '{other}' (expected Up, Down, Left, or Right)"
            ))),
        }
    }
}

/// How the grid state is rendered into the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: internal dispatch enum, matched exhaustively in this example
enum Encoding {
    /// `(x, y)` coordinate tuples (`Agent: (2,3). Goal: (5,5).`).
    Coords,
    /// A rendered `NxN` grid of `A`/`G`/`.` with an orientation legend.
    Ascii,
}

impl Encoding {
    /// The canonical label for this encoding.
    const fn label(self) -> &'static str {
        match self {
            Self::Coords => "coords",
            Self::Ascii => "ascii",
        }
    }

    /// Parse an encoding from its label.
    fn from_label(label: &str) -> candle_mi::Result<Self> {
        match label {
            "coords" => Ok(Self::Coords),
            "ascii" => Ok(Self::Ascii),
            other => Err(candle_mi::MIError::Config(format!(
                "unknown --encoding '{other}' (expected 'coords' or 'ascii')"
            ))),
        }
    }
}

/// The cardinal-action-to-output-token mapping.
struct ActionMapping {
    /// Token emitted for an `Up` move.
    up: String,
    /// Token emitted for a `Down` move.
    down: String,
    /// Token emitted for a `Left` move.
    left: String,
    /// Token emitted for a `Right` move.
    right: String,
}

impl ActionMapping {
    /// Baseline mapping (`PLAN-GRIDWORLD-PROLEPSIS.md`, "The structural
    /// constraint"): the four 2.5M-CLT high-redirect tokens assigned to
    /// cardinal actions.
    fn baseline() -> Self {
        Self {
            up: "black".to_owned(),
            down: "kind".to_owned(),
            left: "well".to_owned(),
            right: "round".to_owned(),
        }
    }

    /// Permuted mapping for the Step C spatial-vs-lexical permutation test:
    /// the same four tokens re-assigned to different cardinal actions.
    fn permuted() -> Self {
        Self {
            up: "round".to_owned(),
            down: "well".to_owned(),
            left: "kind".to_owned(),
            right: "black".to_owned(),
        }
    }

    /// The token mapped to a given action.
    fn token_for(&self, action: CardinalAction) -> &str {
        match action {
            CardinalAction::Up => &self.up,
            CardinalAction::Down => &self.down,
            CardinalAction::Left => &self.left,
            CardinalAction::Right => &self.right,
        }
    }
}

// ── JSON output (Step A) ──────────────────────────────────────────────────────

/// Top-level baseline-results document.
#[derive(Serialize)]
struct BaselineOutput {
    /// `HuggingFace` model ID.
    model: String,
    /// Mapping preset name (`baseline` / `permuted`); overrides not reflected.
    mapping_name: String,
    /// The resolved action → token → token-id assignment actually used.
    mapping: Vec<MappedToken>,
    /// Grid side length used to render prompts.
    grid_size: u32,
    /// Grid encoding used (`coords` / `ascii`).
    encoding: String,
    /// Number of few-shot demonstrations prepended to each prompt.
    few_shot: usize,
    /// RNG seed for the per-instance demonstration-order shuffle.
    seed: u64,
    /// Per-metric pass threshold (`0.80`).
    threshold: f64,
    /// Human-readable prompt template (placeholders, not a rendered prompt).
    prompt_format: String,
    /// The fully rendered prompt for the first instance (illustrative).
    example_prompt: String,
    /// Number of instances scored.
    n_instances: usize,
    /// Fraction whose full-vocabulary argmax decodes to the correct action.
    full_vocab_top1_accuracy: f64,
    /// Fraction whose correct token outranks the other three action tokens.
    forced_choice_accuracy: f64,
    /// Mean probability assigned to the correct action token.
    mean_p_correct: f64,
    /// `true` when both accuracies reach `threshold`.
    passed: bool,
    /// Per-action-class breakdown of both accuracies.
    per_action: Vec<PerActionStats>,
    /// Per-instance records.
    instances: Vec<InstanceResult>,
    /// Wall-clock seconds for the whole baseline run.
    elapsed_secs: f64,
}

/// One row of the resolved mapping.
#[derive(Serialize)]
struct MappedToken {
    /// Cardinal action label.
    action: String,
    /// Token string the action maps to.
    token: String,
    /// Single-token id of `token` in the model vocabulary.
    token_id: u32,
}

/// Per-action-class accuracy breakdown.
#[derive(Serialize)]
struct PerActionStats {
    /// Cardinal action label.
    action: String,
    /// Number of instances whose correct action is this one.
    n: usize,
    /// Full-vocabulary top-1 accuracy within this class.
    full_vocab_top1_accuracy: f64,
    /// Forced-choice accuracy within this class.
    forced_choice_accuracy: f64,
    /// Mean probability on the correct token within this class.
    mean_p_correct: f64,
}

/// Probability assigned to one action token at the output position.
#[derive(Serialize)]
struct ActionProb {
    /// Cardinal action label.
    action: String,
    /// Token string.
    token: String,
    /// Probability at the output position.
    prob: f32,
}

/// One instance's scored result.
#[derive(Serialize)]
struct InstanceResult {
    /// Instance id from the generator.
    instance_id: usize,
    /// Agent cell `[x, y]`.
    agent: [u32; 2],
    /// Goal cell `[x, y]`.
    goal: [u32; 2],
    /// Ground-truth cardinal action label.
    correct_action: String,
    /// Token the correct action maps to.
    correct_token: String,
    /// Single-token id of `correct_token`.
    correct_token_id: u32,
    /// Raw decoded full-vocabulary argmax token.
    top1_token: String,
    /// `top1_token` with surrounding whitespace trimmed.
    top1_token_trimmed: String,
    /// Probability of the full-vocabulary argmax token.
    top1_prob: f32,
    /// `true` when `top1_token_trimmed` matches `correct_token` (case-insensitive).
    full_vocab_top1_correct: bool,
    /// Action whose token had the highest probability among the four.
    forced_choice_action: String,
    /// `true` when `forced_choice_action` equals `correct_action`.
    forced_choice_correct: bool,
    /// Probability on the correct action token.
    p_correct: f32,
    /// Probability on each of the four action tokens.
    action_probs: Vec<ActionProb>,
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Resolve the mapping preset, then apply any per-action CLI overrides.
fn resolve_mapping(common: &CommonArgs) -> candle_mi::Result<ActionMapping> {
    let mut mapping = match common.mapping.as_str() {
        // BORROW: explicit .as_str() — String → &str for the preset match
        "baseline" => ActionMapping::baseline(),
        "permuted" => ActionMapping::permuted(),
        other => {
            return Err(candle_mi::MIError::Config(format!(
                "unknown --mapping '{other}' (expected 'baseline' or 'permuted')"
            )));
        }
    };
    if let Some(token) = &common.map_up {
        mapping.up.clone_from(token);
    }
    if let Some(token) = &common.map_down {
        mapping.down.clone_from(token);
    }
    if let Some(token) = &common.map_left {
        mapping.left.clone_from(token);
    }
    if let Some(token) = &common.map_right {
        mapping.right.clone_from(token);
    }
    Ok(mapping)
}

/// One few-shot demonstration: `(agent, goal, correct_action)`.
type Demo = ([u32; 2], [u32; 2], CardinalAction);

/// Balanced pool of hand-written demonstrations, six per cardinal action, each
/// an unambiguous move (`|dx| != |dy|`). `select_demos` draws a balanced subset
/// and randomizes its order per instance to defeat the few-shot primacy bias
/// (at fixed order the model copies the first demonstration's answer token).
const DEMO_POOL: [Demo; 24] = [
    // Up (dy > 0 dominant)
    ([2, 0], [2, 4], CardinalAction::Up),
    ([0, 1], [0, 3], CardinalAction::Up),
    ([4, 0], [4, 3], CardinalAction::Up),
    ([1, 0], [2, 4], CardinalAction::Up),
    ([3, 1], [2, 4], CardinalAction::Up),
    ([0, 0], [1, 3], CardinalAction::Up),
    // Down (dy < 0 dominant)
    ([2, 4], [2, 0], CardinalAction::Down),
    ([0, 3], [0, 1], CardinalAction::Down),
    ([4, 4], [4, 1], CardinalAction::Down),
    ([1, 4], [2, 0], CardinalAction::Down),
    ([3, 3], [2, 0], CardinalAction::Down),
    ([4, 3], [3, 0], CardinalAction::Down),
    // Left (dx < 0 dominant)
    ([4, 2], [0, 2], CardinalAction::Left),
    ([3, 0], [0, 0], CardinalAction::Left),
    ([4, 4], [1, 4], CardinalAction::Left),
    ([4, 1], [0, 2], CardinalAction::Left),
    ([3, 3], [0, 2], CardinalAction::Left),
    ([4, 0], [1, 1], CardinalAction::Left),
    // Right (dx > 0 dominant)
    ([0, 2], [4, 2], CardinalAction::Right),
    ([1, 0], [4, 0], CardinalAction::Right),
    ([0, 4], [3, 4], CardinalAction::Right),
    ([0, 1], [4, 2], CardinalAction::Right),
    ([1, 3], [4, 2], CardinalAction::Right),
    ([0, 0], [3, 1], CardinalAction::Right),
];

/// Draw a balanced subset of `k` demonstrations (as even as possible across the
/// four actions, remainder distributed in `CardinalAction::ALL` order) and
/// shuffle their order with an RNG seeded from `seed` and `instance_id`, so each
/// instance sees the same demonstrations in a different (reproducible) order.
fn select_demos(k: usize, seed: u64, instance_id: usize) -> candle_mi::Result<Vec<Demo>> {
    let base = k / CardinalAction::ALL.len();
    let rem = k % CardinalAction::ALL.len();
    let mut chosen: Vec<Demo> = Vec::with_capacity(k);
    for (i, action) in CardinalAction::ALL.iter().enumerate() {
        let want = base + usize::from(i < rem);
        chosen.extend(
            DEMO_POOL
                .iter()
                .copied()
                .filter(|d| d.2 == *action)
                .take(want),
        );
    }
    let id64 = u64::try_from(instance_id).map_err(|e| {
        candle_mi::MIError::Config(format!("instance_id {instance_id} exceeds u64: {e}"))
    })?;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(id64));
    chosen.shuffle(&mut rng);
    Ok(chosen)
}

/// Render the planning prompt for one (agent, goal) under a given mapping.
///
/// The cue ends at `"):"` (no trailing space): that `"):"` token is the
/// planning site at which Step B probes the spike, and the model's next token
/// is the space-prefixed action token (e.g. `▁black`). A trailing space would
/// instead tokenize as a standalone `▁`, after which the space-prefixed answer
/// is unreachable. A few-shot demonstration therefore appends its answer with a
/// leading space (see `build_prompt`), giving that same space-prefixed token,
/// which matches the `find_token_id` scoring.
fn format_prompt(
    agent: [u32; 2],
    goal: [u32; 2],
    grid_size: u32,
    mapping: &ActionMapping,
    encoding: Encoding,
) -> String {
    let [ax, ay] = agent;
    let [gx, gy] = goal;
    let up = &mapping.up;
    let down = &mapping.down;
    let left = &mapping.left;
    let right = &mapping.right;
    let state = match encoding {
        Encoding::Coords => {
            format!(
                "Grid: {grid_size}x{grid_size}. Agent: ({ax},{ay}). Goal: ({gx},{gy}). Walls: none."
            )
        }
        Encoding::Ascii => format!(
            "Grid (A=agent, G=goal, .=empty; up=north, right=east):\n{}",
            render_ascii(agent, goal, grid_size)
        ),
    };
    format!(
        "{state}\n\
         Map: Up→{up}, Down→{down}, Left→{left}, Right→{right}.\n\
         Best next move ({up}/{down}/{left}/{right}):"
    )
}

/// Render the grid as rows of `A`/`G`/`.`, top row = highest `y` (north up),
/// columns left→right in increasing `x`. No trailing newline.
fn render_ascii(agent: [u32; 2], goal: [u32; 2], grid_size: u32) -> String {
    let [ax, ay] = agent;
    let [gx, gy] = goal;
    let mut grid = String::new();
    for row in 0..grid_size {
        let y = grid_size - 1 - row;
        for x in 0..grid_size {
            let cell = if x == ax && y == ay {
                'A'
            } else if x == gx && y == gy {
                'G'
            } else {
                '.'
            };
            grid.push(cell);
        }
        if row + 1 < grid_size {
            grid.push('\n');
        }
    }
    grid
}

/// Build the full model input: the given `demos` (already selected and ordered)
/// followed by the test instance, which ends at the `"):"` planning site. Each
/// demonstration appends `" <token>"` (leading space) so the answer is the
/// space-prefixed action token; demonstrations are separated by a blank line.
fn build_prompt(
    agent: [u32; 2],
    goal: [u32; 2],
    grid_size: u32,
    mapping: &ActionMapping,
    encoding: Encoding,
    demos: &[Demo],
) -> String {
    let mut prompt = String::new();
    for &(ex_agent, ex_goal, ex_action) in demos {
        prompt.push_str(&format_prompt(
            ex_agent, ex_goal, grid_size, mapping, encoding,
        ));
        prompt.push(' ');
        prompt.push_str(mapping.token_for(ex_action));
        prompt.push_str("\n\n");
    }
    prompt.push_str(&format_prompt(agent, goal, grid_size, mapping, encoding));
    prompt
}

/// Load + parse the instances JSON and validate every `correct_action` label.
fn load_instances(path: &Path) -> candle_mi::Result<Vec<GridInstance>> {
    let json = fs::read_to_string(path).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
    })?;
    let instances: Vec<GridInstance> = serde_json::from_str(&json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to parse {}: {e}", path.display()))
    })?;
    for instance in &instances {
        CardinalAction::from_label(&instance.correct_action)?;
    }
    Ok(instances)
}

/// Verify that each token is a single token in the model's vocabulary.
///
/// Uses `find_token_id`, which encodes the space-prefixed then bare form and
/// only succeeds when the result is exactly one token.
fn check_tokenization(tokenizer: &MITokenizer, tokens: &[&str]) -> candle_mi::Result<()> {
    eprintln!("Tokenization sanity check (single-token requirement):");
    let mut failures: Vec<String> = Vec::new();
    for &word in tokens {
        if let Ok(id) = tokenizer.find_token_id(word) {
            let decoded = tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"));
            eprintln!("  \"{word}\" -> id {id} (\"{decoded}\")  OK");
        } else {
            eprintln!("  \"{word}\" -> NOT a single token  FAIL");
            // BORROW: .to_owned() — &str → String to collect the failing word
            failures.push(word.to_owned());
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(candle_mi::MIError::Tokenizer(format!(
            "tokens not single-token in this vocabulary: {}",
            failures.join(", ")
        )))
    }
}

/// Probability of `token_id` within a softmax probability vector.
fn prob_of(probs: &[f32], token_id: u32) -> candle_mi::Result<f32> {
    let idx = usize::try_from(token_id).map_err(|e| {
        candle_mi::MIError::Config(format!("token id {token_id} exceeds usize: {e}"))
    })?;
    probs.get(idx).copied().ok_or_else(|| {
        candle_mi::MIError::Config(format!(
            "token id {token_id} out of vocab range (len {})",
            probs.len()
        ))
    })
}

/// Lossless count → `f64` (instance counts are small, well within `u32`).
fn count_to_f64(count: usize) -> candle_mi::Result<f64> {
    let as_u32 = u32::try_from(count)
        .map_err(|e| candle_mi::MIError::Config(format!("count {count} exceeds u32: {e}")))?;
    Ok(f64::from(as_u32))
}

/// Index and value of the maximum probability (full-vocabulary argmax).
fn argmax_prob(probs: &[f32]) -> candle_mi::Result<(usize, f32)> {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, &p)| (i, p))
        .ok_or_else(|| candle_mi::MIError::Config("empty probability vector".into()))
}

/// Softmax of the output-position logits, returned as a dense `Vec<f32>`.
///
/// # Shapes
/// - `cache.output()`: `[batch, seq, vocab]`
/// - returns: `[vocab]` as a host vector
fn output_probs(cache: &HookCache, output_pos: usize) -> candle_mi::Result<Vec<f32>> {
    let logits = cache
        .output()
        .get(0)? // [seq, vocab]
        .narrow(0, output_pos, 1)? // [1, vocab]
        .squeeze(0)? // [vocab]
        // PROMOTE: softmax over logits requires F32 (model output may be BF16)
        .to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits.unsqueeze(0)?)?.squeeze(0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    Ok(probs_vec)
}

/// Serialize the baseline results to JSON, creating parent dirs as needed.
fn write_json(path: &Path, output: &BaselineOutput) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;
    eprintln!("\nOutput written to {}", path.display());
    Ok(())
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Scaffold(args) => run_scaffold(&args),
        Command::Baseline(args) => run_baseline(&args),
    }
}

// ── Step 0: scaffold ──────────────────────────────────────────────────────────

fn run_scaffold(args: &ScaffoldArgs) -> candle_mi::Result<()> {
    let common = &args.common;
    let mapping = resolve_mapping(common)?;
    let encoding = Encoding::from_label(&common.encoding)?;
    eprintln!("=== Gridworld Prolepsis — Step 0 (scaffold) ===\n");
    eprintln!("Mapping ({}):", common.mapping);
    for action in CardinalAction::ALL {
        eprintln!(
            "  {:<6} → \"{}\"",
            action.label(),
            mapping.token_for(action)
        );
    }
    eprintln!();

    // --- Tokenization sanity check (the active mapping's distinct tokens) ---
    if args.skip_token_check {
        eprintln!("Tokenization sanity check: SKIPPED (--skip-token-check)\n");
    } else {
        let mut tokens: Vec<&str> = vec![&mapping.up, &mapping.down, &mapping.left, &mapping.right];
        tokens.sort_unstable();
        tokens.dedup();

        eprintln!("Loading model {} (tokenizer only used)...", common.model);
        let model = MIModel::from_pretrained(&common.model)?;
        let tokenizer = model.tokenizer().ok_or_else(|| {
            candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into())
        })?;
        check_tokenization(tokenizer, &tokens)?;
        eprintln!();
    }

    // --- Load instances ---
    let instances = load_instances(&common.instances)?;
    eprintln!(
        "Loaded {} instances from {}",
        instances.len(),
        common.instances.display()
    );
    for action in CardinalAction::ALL {
        let count = instances
            .iter()
            .filter(|i| i.correct_action == action.label())
            .count();
        eprintln!("  {:<6} {count}", action.label());
    }
    eprintln!();

    // --- Prompt formatter dry-run ---
    let n_print = args.max_print.min(instances.len());
    eprintln!("Sample of {n_print} formatted prompt(s):\n");
    for instance in instances.iter().take(n_print) {
        let action = CardinalAction::from_label(&instance.correct_action)?;
        let target = mapping.token_for(action);
        let prompt = format_prompt(
            instance.agent,
            instance.goal,
            common.grid_size,
            &mapping,
            encoding,
        );
        eprintln!(
            "--- instance {} (correct: {} → \"{target}\") ---",
            instance.instance_id, instance.correct_action
        );
        eprintln!("{prompt}");
        eprintln!();
    }

    Ok(())
}

// ── Step A: baseline feasibility ────────────────────────────────────────────

fn run_baseline(args: &BaselineArgs) -> candle_mi::Result<()> {
    let t_start = Instant::now();
    let common = &args.common;
    let mapping = resolve_mapping(common)?;
    let encoding = Encoding::from_label(&common.encoding)?;

    eprintln!("=== Gridworld Prolepsis — Step A (baseline feasibility) ===\n");
    eprintln!("Mapping ({}):", common.mapping);
    for action in CardinalAction::ALL {
        eprintln!(
            "  {:<6} → \"{}\"",
            action.label(),
            mapping.token_for(action)
        );
    }
    eprintln!();

    // --- Load model ---
    eprintln!("Loading model {}...", common.model);
    let model = MIModel::from_pretrained(&common.model)?;
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!("  device = {device:?}\n");

    // --- Resolve the four target token ids (also validates single-token) ---
    let mut target_ids: Vec<(CardinalAction, u32)> = Vec::with_capacity(CardinalAction::ALL.len());
    let mut mapping_json: Vec<MappedToken> = Vec::with_capacity(CardinalAction::ALL.len());
    eprintln!("Action token ids:");
    for action in CardinalAction::ALL {
        let word = mapping.token_for(action);
        let id = tokenizer.find_token_id(word)?;
        eprintln!("  {:<6} \"{word}\" → id {id}", action.label());
        target_ids.push((action, id));
        mapping_json.push(MappedToken {
            action: action.label().to_owned(),
            token: word.to_owned(),
            token_id: id,
        });
    }
    eprintln!();

    // --- Load instances ---
    let instances = load_instances(&common.instances)?;
    if instances.is_empty() {
        return Err(candle_mi::MIError::Config(format!(
            "no instances in {}",
            common.instances.display()
        )));
    }
    let n_instances = instances.len();
    let few_shot_k = args.few_shot.min(DEMO_POOL.len());
    if few_shot_k < args.few_shot {
        eprintln!(
            "(--few-shot {} capped to the {few_shot_k}-demonstration pool)",
            args.few_shot
        );
    }
    eprintln!(
        "Scoring {n_instances} instances from {} ({few_shot_k}-shot, seed {}, randomized order)...\n",
        common.instances.display(),
        args.seed
    );

    // --- Per-instance forward passes ---
    let mut results: Vec<InstanceResult> = Vec::with_capacity(n_instances);
    for (printed, instance) in instances.iter().enumerate() {
        let action = CardinalAction::from_label(&instance.correct_action)?;
        let correct_token = mapping.token_for(action).to_owned();
        let correct_id = target_ids
            .iter()
            .find(|&&(a, _)| a == action)
            .map(|&(_, id)| id)
            .ok_or_else(|| {
                candle_mi::MIError::Config(format!("no token id for action {}", action.label()))
            })?;

        let demos = select_demos(few_shot_k, args.seed, instance.instance_id)?;
        let prompt = build_prompt(
            instance.agent,
            instance.goal,
            common.grid_size,
            &mapping,
            encoding,
            &demos,
        );
        let token_ids = tokenizer.encode(&prompt)?;
        let seq_len = token_ids.len();
        let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
        let cache = model.forward(&input, &HookSpec::new())?;
        let probs = output_probs(&cache, seq_len - 1)?;

        // Full-vocabulary top-1.
        let (top_idx, top_prob) = argmax_prob(&probs)?;
        let top_id = u32::try_from(top_idx).map_err(|e| {
            candle_mi::MIError::Config(format!("vocab index {top_idx} exceeds u32: {e}"))
        })?;
        let top1_token = tokenizer.decode_token(top_id)?;
        let top1_trimmed = top1_token.trim().to_owned();
        let full_correct = top1_trimmed.eq_ignore_ascii_case(&correct_token);

        // Forced choice among the four action tokens.
        let mut action_probs: Vec<ActionProb> = Vec::with_capacity(target_ids.len());
        let mut best: Option<(CardinalAction, f32)> = None;
        for &(a, id) in &target_ids {
            let p = prob_of(&probs, id)?;
            action_probs.push(ActionProb {
                action: a.label().to_owned(),
                token: mapping.token_for(a).to_owned(),
                prob: p,
            });
            if best.is_none_or(|(_, bp)| p > bp) {
                best = Some((a, p));
            }
        }
        let forced_action = best.map_or(action, |(a, _)| a);
        let forced_correct = forced_action == action;
        let p_correct = prob_of(&probs, correct_id)?;

        if printed < args.max_print {
            let mark = if full_correct { "✓" } else { "✗" };
            eprintln!(
                "  [{:>3}] {:<5} correct=\"{}\"  top1=\"{}\" ({:.4}) {mark}  forced=\"{}\" {}  P(correct)={:.4}",
                instance.instance_id,
                instance.correct_action,
                correct_token,
                top1_trimmed,
                top_prob,
                forced_action.label(),
                if forced_correct { "✓" } else { "✗" },
                p_correct,
            );
        }

        results.push(InstanceResult {
            instance_id: instance.instance_id,
            agent: instance.agent,
            goal: instance.goal,
            correct_action: instance.correct_action.clone(),
            correct_token,
            correct_token_id: correct_id,
            top1_token,
            top1_token_trimmed: top1_trimmed,
            top1_prob: top_prob,
            full_vocab_top1_correct: full_correct,
            forced_choice_action: forced_action.label().to_owned(),
            forced_choice_correct: forced_correct,
            p_correct,
            action_probs,
        });
    }

    // --- Aggregate ---
    let full_correct_count = results.iter().filter(|r| r.full_vocab_top1_correct).count();
    let forced_correct_count = results.iter().filter(|r| r.forced_choice_correct).count();
    let n_f64 = count_to_f64(n_instances)?;
    let full_acc = count_to_f64(full_correct_count)? / n_f64;
    let forced_acc = count_to_f64(forced_correct_count)? / n_f64;
    let sum_p: f32 = results.iter().map(|r| r.p_correct).sum();
    let mean_p = f64::from(sum_p) / n_f64;
    let passed = full_acc >= THRESHOLD && forced_acc >= THRESHOLD;

    let per_action = CardinalAction::ALL
        .iter()
        .map(|action| per_action_stats(action.label(), &results))
        .collect::<candle_mi::Result<Vec<_>>>()?;

    // --- Summary ---
    eprintln!("\n=== Step A results ===");
    eprintln!(
        "  {:<6} {:>5} {:>14} {:>14} {:>12}",
        "action", "n", "top1_acc", "forced_acc", "mean_P"
    );
    for s in &per_action {
        eprintln!(
            "  {:<6} {:>5} {:>14.3} {:>14.3} {:>12.4}",
            s.action, s.n, s.full_vocab_top1_accuracy, s.forced_choice_accuracy, s.mean_p_correct
        );
    }
    eprintln!(
        "  {:<6} {n_instances:>5} {full_acc:>14.3} {forced_acc:>14.3} {mean_p:>12.4}",
        "ALL"
    );
    eprintln!(
        "\n  full-vocab top-1 accuracy = {full_acc:.3}  ({full_correct_count}/{n_instances})"
    );
    eprintln!(
        "  forced-choice accuracy    = {forced_acc:.3}  ({forced_correct_count}/{n_instances})"
    );
    eprintln!("  threshold (both)          = {THRESHOLD:.2}");
    eprintln!(
        "  VERDICT: {}",
        if passed {
            "PASS — Step A clears the gate; Step B may proceed."
        } else {
            "FAIL — baseline below threshold; do not proceed to Step B on this format."
        }
    );

    // --- JSON output ---
    let example_prompt = match instances.first() {
        Some(i) => {
            let demos = select_demos(few_shot_k, args.seed, i.instance_id)?;
            build_prompt(
                i.agent,
                i.goal,
                common.grid_size,
                &mapping,
                encoding,
                &demos,
            )
        }
        None => String::new(),
    };
    let output = BaselineOutput {
        model: common.model.clone(),
        mapping_name: common.mapping.clone(),
        mapping: mapping_json,
        grid_size: common.grid_size,
        encoding: encoding.label().to_owned(),
        few_shot: few_shot_k,
        seed: args.seed,
        threshold: THRESHOLD,
        prompt_format: PROMPT_FORMAT.to_owned(),
        example_prompt,
        n_instances,
        full_vocab_top1_accuracy: full_acc,
        forced_choice_accuracy: forced_acc,
        mean_p_correct: mean_p,
        passed,
        per_action,
        instances: results,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    };
    write_json(&args.output, &output)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

/// Compute the accuracy breakdown for one action class.
fn per_action_stats(label: &str, results: &[InstanceResult]) -> candle_mi::Result<PerActionStats> {
    let subset: Vec<&InstanceResult> = results
        .iter()
        .filter(|r| r.correct_action == label)
        .collect();
    let n = subset.len();
    let (full_acc, forced_acc, mean_p) = if n == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let full = subset.iter().filter(|r| r.full_vocab_top1_correct).count();
        let forced = subset.iter().filter(|r| r.forced_choice_correct).count();
        let sum_p: f32 = subset.iter().map(|r| r.p_correct).sum();
        let n_f64 = count_to_f64(n)?;
        (
            count_to_f64(full)? / n_f64,
            count_to_f64(forced)? / n_f64,
            f64::from(sum_p) / n_f64,
        )
    };
    Ok(PerActionStats {
        action: label.to_owned(),
        n,
        full_vocab_top1_accuracy: full_acc,
        forced_choice_accuracy: forced_acc,
        mean_p_correct: mean_p,
    })
}
