// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gridworld prolepsis — Step 0 scaffolding (prompt formatter + tokenization
//! sanity check).
//!
//! Step 0 of `docs/roadmaps/PLAN-GRIDWORLD-PROLEPSIS.md`: infrastructure only,
//! no model intervention. This binary does two things:
//!
//! 1. **Tokenization sanity check.** Loads the Gemma 2 2B tokenizer and
//!    verifies that each mapped action token (`black`, `kind`, `well`,
//!    `round` for the baseline mapping) is a single token in the vocabulary.
//! 2. **Prompt formatter dry-run.** Reads the gridworld instances JSON emitted
//!    by `scripts/gridworld_generator.py` and prints the formatted planning
//!    prompts for the chosen action mapping.
//!
//! The cardinal-action-to-token mapping is a CLI parameter so the same binary
//! drives the Step B baseline and the Step C permutation without a rebuild:
//!
//! | Mapping | Up | Down | Left | Right |
//! |---------|----|------|------|-------|
//! | `baseline`  | `black` | `kind` | `well`  | `round` |
//! | `permuted`  | `round` | `well` | `kind`  | `black` |
//!
//! Each instance is rendered with the prompt template (the trailing space
//! after the colon is the planning site, structurally analogous to the
//! trailing-space rhyme planning site in `figure13_planning_poems`):
//!
//! ```text
//! Grid: 5x5. Agent: (ax,ay). Goal: (gx,gy). Walls: none.
//! Map: Up→m_up, Down→m_down, Left→m_left, Right→m_right.
//! Best next move (m_up/m_down/m_left/m_right):
//! ```
//!
//! ```bash
//! # Full Step-0 check (loads cached Gemma 2 2B for the tokenizer)
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis
//!
//! # Permuted mapping
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- --mapping permuted
//!
//! # Formatter-only dry-run (skips the model load)
//! cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- --skip-token-check
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;

use candle_mi::{MIModel, MITokenizer};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "gridworld_prolepsis")]
#[command(about = "Gridworld prolepsis Step 0: prompt formatter + tokenization check")]
struct Args {
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

    /// `HuggingFace` model ID used for the tokenization sanity check.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Grid side length `N` used when rendering the `NxN` header. Must match
    /// the `--grid-size` passed to `scripts/gridworld_generator.py`.
    #[arg(long, default_value_t = 5)]
    grid_size: u32,

    /// Number of formatted prompts to print as a dry-run sample.
    #[arg(long, default_value_t = 5)]
    max_print: usize,

    /// Skip the tokenization sanity check (and the model load it requires);
    /// run the prompt formatter alone.
    #[arg(long, default_value_t = false)]
    skip_token_check: bool,
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

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Resolve the mapping preset, then apply any per-action CLI overrides.
fn resolve_mapping(args: &Args) -> candle_mi::Result<ActionMapping> {
    let mut mapping = match args.mapping.as_str() {
        // BORROW: explicit .as_str() — String → &str for the preset match
        "baseline" => ActionMapping::baseline(),
        "permuted" => ActionMapping::permuted(),
        other => {
            return Err(candle_mi::MIError::Config(format!(
                "unknown --mapping '{other}' (expected 'baseline' or 'permuted')"
            )));
        }
    };
    if let Some(token) = &args.map_up {
        mapping.up.clone_from(token);
    }
    if let Some(token) = &args.map_down {
        mapping.down.clone_from(token);
    }
    if let Some(token) = &args.map_left {
        mapping.left.clone_from(token);
    }
    if let Some(token) = &args.map_right {
        mapping.right.clone_from(token);
    }
    Ok(mapping)
}

/// Render the planning prompt for one instance under a given mapping.
///
/// The trailing space after the final colon is deliberate: it is the planning
/// site at which the prolepsis spike is probed in Step B.
fn format_prompt(instance: &GridInstance, grid_size: u32, mapping: &ActionMapping) -> String {
    let [ax, ay] = instance.agent;
    let [gx, gy] = instance.goal;
    let up = &mapping.up;
    let down = &mapping.down;
    let left = &mapping.left;
    let right = &mapping.right;
    format!(
        "Grid: {grid_size}x{grid_size}. Agent: ({ax},{ay}). Goal: ({gx},{gy}). Walls: none.\n\
         Map: Up→{up}, Down→{down}, Left→{left}, Right→{right}.\n\
         Best next move ({up}/{down}/{left}/{right}): "
    )
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

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mapping = resolve_mapping(&args)?;
    eprintln!("=== Gridworld Prolepsis — Step 0 ===\n");
    eprintln!("Mapping ({}):", args.mapping);
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

        eprintln!("Loading model {} (tokenizer only used)...", args.model);
        let model = MIModel::from_pretrained(&args.model)?;
        let tokenizer = model.tokenizer().ok_or_else(|| {
            candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into())
        })?;
        check_tokenization(tokenizer, &tokens)?;
        eprintln!();
    }

    // --- Load instances ---
    let json = fs::read_to_string(&args.instances).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to read {}: {e}", args.instances.display()))
    })?;
    let instances: Vec<GridInstance> = serde_json::from_str(&json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to parse {}: {e}", args.instances.display()))
    })?;

    // Validate every label up front so bad data fails loudly, not mid-print.
    for instance in &instances {
        CardinalAction::from_label(&instance.correct_action)?;
    }

    eprintln!(
        "Loaded {} instances from {}",
        instances.len(),
        args.instances.display()
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
        let prompt = format_prompt(instance, args.grid_size, &mapping);
        eprintln!(
            "--- instance {} (correct: {} → \"{target}\") ---",
            instance.instance_id, instance.correct_action
        );
        eprintln!("{prompt}");
        eprintln!();
    }

    Ok(())
}
