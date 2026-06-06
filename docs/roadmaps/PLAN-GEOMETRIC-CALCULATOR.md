# Geometric Calculator — Minimum-Architecture Sweep — Experiment Plan

**Scope:** a single experimental question, tracked separately from the master
roadmap, targeting candle-mi **v0.1.13**. (`v0.2.0` is reserved for the NLnet
grant work; this is a `v0.1.x` patch.)

**The question:** does the "geometric calculator" mechanism — circular (Fourier)
number representations manipulated by a parallel modular-addition module —
replicate on the consumer-scale models in our cache, and *what is the minimum
architecture* (width, depth, family) at which it appears?

**Why a dedicated document.** This is a method/mechanism replication with its own
narrative arc (calibrate on a known-good anchor, then descend a model ladder),
distinct from release coordination. Separating it lets us track the scientific
target without infrastructure noise — the same convention used by
[`PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`](PLAN-PLT-LLAMA-PLANNING-SIGNAL.md).

---

## Status (2026-06-03)

Design phase. No code written yet. This document is the pre-flight plan; it will
be reviewed before any implementation begins.

| Step | Status | Artefact (planned) |
|---|---|---|
| 0 — Library prerequisites (`MlpActPost` hook + `fourier.rs`) | ⏳ Not started | `src/hooks.rs`, `src/transformer/mlp.rs`, `src/util/fourier.rs`, `tests/validate_mlp_neurons.rs` |
| A — Calibrate the circle-finder on the anchor | ⏳ Not started | `geometric_calculator` example + `docs/experiments/geometric-calculator/llama3.2-3b.json` |
| B — Phase-1 representation sweep (all 3 pilot models) | ⏳ Not started | per-model JSON + `findings.md` representation table |
| C — Phase-2 computation (anchor first, then descent) | ⏳ Not started | per-model JSON + `findings.md` computation table |
| D — Minimum-architecture readout + figures + write-up + ship | ⏳ Not started | `findings.md`, `geometric_calculator_plot.wl` + `plots/*.png`, README row, `CHANGELOG.md` |

**Result in one sentence (to be filled at Step D).** _TBD — the width/family/depth
cell at which circles + parallel modular addition both appear._

---

## Context

Three pieces of prior work define the space:

1. **Feucht, Haklay, Bhalla, Wurgaft, Rager, Sarfati, Merullo, McGrath, Lewis,
   Lubana, Fel, Geiger — _Arithmetic in the Wild: Llama uses Base-10 Addition to
   Reason About Cyclic Concepts_** ([arXiv:2605.01148](https://arxiv.org/abs/2605.01148);
   Goodfire blog _A geometric calculator inside a neural network_, Neural Geometry
   Series, 2026-05-14). A general-purpose addition module in **Llama-3.1-8B,
   layer 18 MLP** (~28 neurons, ~0.2% of the layer) manipulates circular number
   representations, solving day/month/cyclic arithmetic by **parallel modular
   addition** on circles of period **2, 5, 10** (base-10 residue system). The
   thesis: networks don't merely *store* geometric representations — they
   *compute* with them.
2. **Kantamneni & Tegmark — _Language Models Use Trigonometry to Do Addition_**
   (2025) — the substrate. Numbers are represented on helices/circles; addition
   proceeds by a "Clock" algorithm. Documented across GPT-J, Llama-3.1-8B, and
   Pythia — i.e. the *representation* half is known to be broad, not 8B-specific.
3. **The gap for us** — nobody has charted *where on the model ladder* the
   computation half (the modular-addition module, not just the circles) first
   appears. The candle-mi cache holds a clean width axis at fixed depth that
   isolates exactly this.

**Working stance.** This is empirical coverage in a statistical framework, not
hypothesis testing with a theorem under risk. Every model fills a cell in the
minimum-architecture matrix. A null result (circles present, no modular-addition
module) at a given scale is itself a finding — provided we can distinguish it
from an implementation bug (see *Reverse-engineering provenance*).

---

## The hard constraint: model availability vs F32 on 16 GB

The headline model **cannot** be run faithfully on the target hardware
(RTX 5060 Ti, 16 GB) under candle-mi's F32-everywhere numerical standard:

- Llama-3.1-8B @ F32 ≈ 32 GB weights — does not fit.
- @ BF16 ≈ 16 GB — fits weights only, no headroom for activations/hooks.
- The cache holds only **FP8 / AWQ-quantized** 8B variants; candle-mi has no
  quantized-inference path, and they violate the F32 parity standard regardless.

Therefore this is a **consumer-scale method replication**, not a bit-exact
Llama-3.1-8B-layer-18 reproduction. Layer indices and neuron counts will differ
by design; that difference is part of the finding.

---

## Pilot model ladder

All three are present in the cache, run in already-validated transformer arms,
and fit comfortably at F32 on 16 GB. Config numbers below are read from the
cached `config.json` files (PowerShell over `~/.cache/huggingface/hub`,
2026-06-03), not reconstructed:

| Role | Model | type | layers | d_model | d_ff | heads (q/kv) | MLP / norm |
|---|---|---|---|---|---|---|---|
| **Calibrate (anchor)** | `meta-llama/Llama-3.2-3B` | llama | 28 | 3072 | 8192 | 24/8 | gated SwiGLU, RMSNorm |
| Descent — mid | `Qwen/Qwen3-1.7B-Base` | qwen3 | 28 | 2048 | 6144 | 16/8 | gated SwiGLU, +QK-norm |
| Descent — floor | `Qwen/Qwen3-0.6B-Base` | qwen3 | 28 | 1024 | 3072 | 16/8 | gated SwiGLU, +QK-norm |

**Why this set (decided 2026-06-03).** A "minimal pilot first" scope, plus
Qwen3-1.7B added so the descent has a mid-rung. A property falls out of it:
**all three are 28 layers**, so the pilot holds *depth fixed* and cleanly sweeps
**width** (3072 → 2048 → 1024) with one **family** swap at the top
(llama ↔ qwen3). Note both pilot models use the *same* MLP layout
(`GatedSeparate` SwiGLU); the QK-norm difference is in attention, not the MLP. So
the pilot validates `MlpActPost` parity **cross-family** (llama vs qwen3) on one
MLP layout — the other two layouts (`GatedFused`, `Plain`) are covered by a
synthetic-weight unit test (see *Validation*), not by a pilot model run.

**Why Llama-3.2-3B as the anchor.** Closest architectural cousin to the paper's
Llama-3.1-8B (same family, tokenizer, SwiGLU/RMSNorm/GQA); the largest Llama that
fits at F32; already validated in candle-mi.

**Calibration is honest about which half is certain.** Per Kantamneni & Tegmark,
the *representation* half (numbers on circles) is near-certain to exist even at
sub-2B scale — so the circle-finder is calibrated against a signal we expect to
be present. The *computation* half (a localized parallel modular-addition module)
is the genuine open question and is **not** assumed for calibration.

**Deferred to post-pilot expansion** (out of scope here): the *depth* axis
(`Llama-3.2-1B`, L=16/d=2048 — shallow-wide), the *plain-MLP* architecture
(`StarCoder2-3B`, non-gated + LayerNorm — cleanest neuron basis), and the
*literature floor* (`EleutherAI/pythia-70m`, L=6 — needs a new `gpt_neox`
config parser + oracle, the family Kantamneni & Tegmark used).

---

## Library prerequisites (Step 0 — infrastructure, not science)

Two genuine gaps in candle-mi must close before the experiment can run. Both are
surfaced from a read of the current source, not assumed.

### Gap A — `HookPoint::MlpActPost` (the per-neuron activation hook) — **critical**

The paper's centrepiece (the neuron-explorer heatmap; "28 neurons, each tied to
one circle it reads/writes") lives in the **d_mlp** activation space. candle-mi
does not expose it:

- `MlpPre(layer)` is captured in [`src/transformer/mod.rs`](../../src/transformer/mod.rs)
  *after* the pre-MLP norm and *before* `Mlp::forward` — it is the **normed
  d_model MLP input**, not a d_mlp pre-activation.
- `MlpPost(layer)` is captured *after* `Mlp::forward` — it is the **d_model MLP
  output** (already projected back down by `down_proj`), not a d_mlp
  post-activation.
- The d_mlp intermediate `act(gate(x)) · up(x)` (gated) / `act(fc(x))` (plain) —
  the thing fed into the down-projection, i.e. the classic "neuron activations" —
  is a local inside [`Mlp::forward`](../../src/transformer/mlp.rs) and is never
  captured.

> **Naming-debt note (fix in this PR).** The *source* doc-comments on
> `MlpPre`/`MlpPost` in `src/hooks.rs` say "MLP pre-activation"/"post-activation",
> which implies TransformerLens d_mlp semantics, but the implementation captures
> residual-level d_model tensors. (HOOKS.md is already accurate — it documents
> `MlpPre` as "MLP input", shape `[batch, seq, hidden]` — so only the source
> comment is stale.) Crucially, **`MlpPre`'s current d_model semantics are
> load-bearing**: the `clt_vs_plt_planning_site` example uses `MlpPre` as the
> GemmaScope PLT input hook (`PltInputHook::MlpPre → HookPoint::MlpPre`, post-LN2).
> We therefore do **not** repurpose `MlpPre`/`MlpPost`; we add a new, unambiguous
> variant, *clarify* the stale `src/hooks.rs` comments to say "MLP block input /
> output (d_model)", and add a HOOKS.md row for the new variant.

**Design:**

- New variant `HookPoint::MlpActPost(usize)` in [`src/hooks.rs`](../../src/hooks.rs),
  `Display` string `blocks.{i}.mlp.hook_act_post`, with a matching `parse_hook_string`
  arm. Definition: **the d_mlp-dimensional vector fed into the down-projection**
  (`gate ⊙ up` for `GatedSeparate`/`GatedFused`; `act(fc(x))` for `Plain`),
  shape `[batch, seq, d_mlp]`.
- Capture happens *inside* `Mlp::forward`. The v0.1.11 hook-overhead diagnostic
  established that the hook architecture is not a bottleneck (an empty `HookSpec`
  costs ~0.6% on Llama-3.2-1B — not zero). To avoid materializing the large
  `[batch, seq, d_mlp]` tensor on every forward, gate the capture: `mod.rs` calls
  a `Mlp::forward_capturing(&self, x) -> Result<(Tensor /*out, d_model*/, Tensor
  /*neurons, d_mlp*/)>` **only** when `hooks.is_captured(&MlpActPost(layer))`,
  then `cache.store`s the neurons tensor (keeping cache writes in `mod.rs`,
  consistent with every other hook point); otherwise the existing `forward` path
  runs untouched.
- **Capture-only for the pilot.** Interventions on `MlpActPost` are *not* in
  scope — the paper's causal steering is residual-level (adding circular-basis
  vectors to the residual stream via the existing `Intervention::Add`), so
  neuron-level intervention is unnecessary here. Neuron ablation is a clean
  follow-on, explicitly deferred.

**Validation (mandatory before Step A).** Mirror the existing `validate_*`
oracle pattern:
- `scripts/mlp_neurons_validation.py` — load Llama-3.2-3B and Qwen3-1.7B-Base via
  `transformers` in F32 on CPU, register a forward hook on the MLP that dumps the
  d_mlp pre-down-projection vector for fixed prompts at chosen layers, write
  `scripts/mlp_neurons_reference.json` (top-K neuron indices + magnitudes).
- `tests/validate_mlp_neurons.rs` — `#[ignore]`-gated; assert candle-mi's
  `MlpActPost` capture matches: top-K neuron indices exact, magnitudes
  abs-diff < 1e-4 (CPU) / < 5e-3 (GPU). Required-features `["transformer"]`.
  Covers the `GatedSeparate` layout on both Llama-3.2-3B and Qwen3-1.7B-Base.
- Unit test in `src/transformer/mlp.rs` (no download): on synthetic weights,
  assert `forward_capturing` returns the correct d_mlp intermediate for **all
  three** layouts (`GatedSeparate`, `GatedFused`, `Plain`) — covering the two
  layouts no pilot model exercises.

Without this passing, a "no modular-addition module" null at small scale is
uninterpretable — we could not tell a genuine architectural absence from a hook
that captures the wrong tensor.

### Gap B — `src/util/fourier.rs` (DFT + circle fit) — self-contained, low-risk

No FFT/DFT/circular-fit exists anywhere in the crate (grep-confirmed). Needed to
(a) detect the periods {2, 5, 10, 100} in number representations and
(b) score per-neuron periodicity for clustering. Number sweeps are small
(≤ a few hundred points), so a naive DFT is fine — no new dependency.

Proposed API (candle-tensor based, mirroring the style of
[`src/util/pca.rs`](../../src/util/pca.rs)):

```rust
/// A least-squares circle of a given integer period fitted to per-number
/// activations.
pub struct CircleFit {
    pub period: usize,
    pub r2: f32,             // variance explained by the [cos, sin] pair
    pub cos_dir: Tensor,     // [d_model] direction for cos(2πn/period)
    pub sin_dir: Tensor,     // [d_model] direction for sin(2πn/period)
}

/// Aggregate DFT power per integer frequency k = 0..=n/2 over the sample axis
/// of an `[n_samples, d]` matrix (mean over feature dimension). Used to find
/// which periods dominate the number representation.
pub fn dft_power_spectrum(samples: &Tensor) -> Result<Vec<f32>>;

/// Least-squares fit of a ⋅ cos(2πn/period) + b ⋅ sin(2πn/period) per feature,
/// over rows indexed by integer n. Returns the fitted circle + R².
pub fn circle_fit(samples: &Tensor, period: usize) -> Result<CircleFit>;

/// Power a single neuron's firing curve (one value per number) carries at each
/// candidate period. Used to cluster neurons by the circle they read.
pub fn neuron_periodicity(firings: &[f32], periods: &[usize]) -> Vec<(usize, f32)>;
```

Unit tests: synthetic pure-period signals recover their period and R² ≈ 1; a
flat signal scores ~0 everywhere.

---

## Folder structure

Follows the convention established by `PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`: a
per-experiment folder under `docs/experiments/` whose name matches the example.

```
candle-mi/
├── docs/
│   ├── roadmaps/
│   │   └── PLAN-GEOMETRIC-CALCULATOR.md            ★ this document
│   └── experiments/
│       └── geometric-calculator/                   ★ NEW per-experiment folder
│           ├── findings.md                         ← write-up (Step D)
│           ├── llama3.2-3b.json                     ← anchor: Phase 1 + Phase 2
│           ├── qwen3-1.7b.json                      ← descent mid
│           └── qwen3-0.6b.json                      ← descent floor
├── scripts/
│   ├── mlp_neurons_validation.py                   ← d_mlp oracle (Step 0)
│   └── mlp_neurons_reference.json                  ← deterministic reference
├── src/
│   ├── hooks.rs                                    ← + MlpActPost variant
│   ├── transformer/mlp.rs                          ← + forward_capturing
│   └── util/fourier.rs                             ← NEW: DFT + circle fit
├── examples/
│   ├── geometric_calculator.rs                     ← the experiment
│   └── results/
│       └── geometric_calculator/                   ★ figures (convention)
│           ├── geometric_calculator_plot.wl        ← Mathematica plotting script
│           └── plots/                              ← rendered PNGs (committed)
└── tests/
    └── validate_mlp_neurons.rs                     ← MlpActPost parity test
```

Note `examples/results/<name>/` is the established home for the JSON outputs +
`.wl` script + `plots/*.png`; `docs/experiments/geometric-calculator/` holds the
narrative `findings.md` and the canonical result JSON. (Some prior experiments
keep JSON under `docs/experiments/`, others under `examples/results/`; this plan
puts the result JSON in `docs/experiments/` and points the `.wl` `jsonFile` path
there, to avoid duplicating large JSON across two trees.)

---

## Experiment

Working file: `examples/geometric_calculator.rs`. Always run with the `mmap`
feature (user convention):

```
cargo run --features transformer,mmap --release --example geometric_calculator -- <args>
```

### Number-token handling (applies to all steps)

- Build a per-model set of **single-token** non-negative integers via
  `encode_raw` / `find_token_id` (multi-token numbers are excluded — circles are
  read from single-token representations). Require ≥ ~50 single-token integers
  for a meaningful DFT; log the count and the dropped numbers (no silent caps).
- Cyclic-task tokens (days 0–6, months 1–12) are used for the Phase-2 *task*
  prompts, not for circle-finding.
- Base models need few-shot context for the cyclic task; the example ships a
  fixed few-shot preamble per task and records it in the output JSON.

### Step A — Calibrate the circle-finder on the anchor (Llama-3.2-3B)

Goal: lock the Phase-1 harness against a model where the representation is
expected to be present, before trusting it on the descent rungs.

- Sweep single-token integers; at each layer capture `ResidPost` at the number
  token position; average per number.
- Run `pca_top_k` (reuse [`character_count_helix`](../../examples/character_count_helix.rs)'s
  bin → PCA → projection pipeline) and `dft_power_spectrum` along the number axis.
- **Exit criterion:** at some layer, `dft_power_spectrum` shows clear peaks at
  periods that include {10, 100} (base-10) and ≥ one of {2, 5}; PCA projections
  trace visible loops. If the anchor shows no circles, stop and debug the harness
  (do not proceed to the descent).

### Step B — Phase-1 representation sweep (all three models)

Goal: chart *where* circles live as width shrinks, depth fixed.

- For each model, sweep all 28 layers; record, per layer (and per **fractional
  depth** = layer / 28, trivially aligned here), the circle-score for each period
  in {2, 5, 10, 100} via `circle_fit` R² and DFT power.
- **Exit criterion:** a representation table in `findings.md` with one row per
  (model × period) giving peak-R² and peak layer. Cheap — no neuron hook, no
  steering.

### Step C — Phase-2 computation (anchor first, then descent)

Run only on models whose Phase-1 shows circles **and** which can do the task
above chance. Goal: is there a *localized parallel modular-addition module*?

- **Neuron clustering.** With `MlpActPost`, capture d_mlp firings across the
  integer sweep at the candidate module layer(s); score each neuron's periodicity
  via `neuron_periodicity`; cluster neurons by their dominant period. Report the
  count per period and the total "circle neuron" fraction (cf. the paper's ~0.2%).
- **Read vs write circle (anchor + mid; floor if time).** *Read* = correlation of
  a neuron's firing with input-circle phase (from capture). *Write* = projection
  of the neuron's down-projection column onto the fitted circular directions —
  requires a read-only **MLP weight accessor** (a small public method returning
  the per-layer `down_proj` / `gate_proj` matrices, analogous to
  `CrossLayerTranscoder::decoder_matrix`). Mark this accessor **Phase-2b**: the
  read-circle analysis alone already demonstrates the mechanism; the write-circle
  analysis is the fuller replication.
- **Causal steering along circles.** Construct the per-period circular-basis
  directions (the `cos_dir`/`sin_dir` from `circle_fit`) and add them to the
  residual at the module layer/position via `Intervention::Add`, sweeping
  magnitude. Measure ΔP on the cyclic-task answer token (e.g. the predicted month
  for "sixteen months after August"). This is the causal analogue of the blog's
  steering-slider demo.
- **Exit criterion:** per model, a computation row giving {circle-neuron count,
  dominant periods, localizing fractional depth, causal steering ΔP}. A model
  with circles but flat steering and no periodic neuron cluster is recorded as
  "representation only".

### Step D — Minimum-architecture readout, write-up, ship

- `findings.md`: the width/family/depth cell where {2, 5, 10} circles **and** a
  causal modular-addition module both appear; framed as a coverage-matrix
  extension, not hypothesis testing.
- README "Paper replications" row for Feucht et al., pointing at the example.
- `CHANGELOG.md` under `[Unreleased]`: `MlpActPost` hook, `fourier.rs`, the
  example, the findings doc.
- Tag `v0.1.13` only after the full pre-commit + preflight gate is green
  (`scripts/preflight.ps1`). The documented `-Full` trigger is "adding a new
  model family", which this is not — but it alters the MLP forward/hook hot path
  that `bench_hook_*` measures, so running `-Full` once before the tag is prudent
  to confirm no forward-path regression.

---

## Figures (deliverable **and** visual validation)

The Goodfire blog and the arXiv PDF lean heavily on figures (circle plots, the
neuron-explorer heatmap, the steering slider). We reproduce them, and the figures
are not decoration — they are the qualitative validation that the replication
worked, exactly as [`helix_plot.wl`](../../examples/results/character_count_helix/helix_plot.wl)
visually confirms the manifold result. The quantitative side (circle R², DFT
power, steering ΔP) lives in the JSON; the figures make it legible.

**Convention (followed exactly).** The example emits JSON; a companion Mathematica
`<name>_plot.wl` under `examples/results/geometric_calculator/` imports it
(`Import[..., "RawJSON"]`) and `Export`s PNGs to a `plots/` subfolder — the same
pattern as `helix_plot.wl`, `attention_routing_plot.wl`, `convergence_plot.wl`.
The user runs Mathematica on Windows and the repo already ships `.wl` scripts, so
no new plotting dependency is introduced.

| Goodfire / paper artefact | candle-mi figure | Template to reuse | JSON fields |
|---|---|---|---|
| Circular number representation | 2D scatter of number activations projected onto `(cos_dir, sin_dir)` per period, coloured by `n` — should trace a clean loop; plus a PC1–PC3 view | `L12_helix_pc123.png` | `projections`, `circle_fit` dirs |
| "Which periods" decomposition | DFT power-spectrum bars vs frequency (peaks at 1/2, 1/5, 1/10, 1/100) | `L12_variance_bars.png` | `dft_power_spectrum` |
| Addition-module localization (Fig 2 — shared across tasks) | per-layer × period **R² heatmap**, one per model; the three side-by-side **are the minimum-architecture panel** | `L12_cosine_heatmap.png` | per-layer `circle_fit` R² |
| Neuron-explorer heatmap | neuron firing heatmap: neurons sorted/clustered by dominant period (rows) × number input (cols), cell = `MlpActPost` activation — periodic banding | heatmap variant of the cosine template | `MlpActPost` firing matrix |
| Neuron period clusters | neuron-count-per-period bar chart (cf. the paper's ~0.2% / 28-neuron figure) | `L12_variance_bars.png` | `neuron_periodicity` cluster counts |
| Steering slider (interactive demo 2) | static **dose-response**: steering magnitude × P(answer token), plus "answer token walks around the cycle" | `*_strength_sweep_p.png` (steering_convergence) | steering sweep |
| Modular-sum walkthrough (interactive demo 1) | *optional* schematic: three small circles showing `(a mod p) + (b mod p) = res mod p` | bespoke | per-circle decomposition |

**Honest framing.** The two interactive demos become **static** figures (a
magnitude sweep, a labelled circle). We reproduce their *content*, not their
interactivity — stated as such in `findings.md`.

**Cross-model small-multiples are the headline.** The localization heatmap and the
circle plots rendered for all three pilot models on shared axes (fractional depth)
turn "does it replicate" into a single readable panel — and that panel is the
minimum-architecture finding.

---

## Deliverables

- `src/hooks.rs` (`MlpActPost`), `src/transformer/mlp.rs` (`forward_capturing`),
  `src/util/fourier.rs`.
- `scripts/mlp_neurons_validation.py` + `scripts/mlp_neurons_reference.json`;
  `tests/validate_mlp_neurons.rs`.
- `examples/geometric_calculator.rs`.
- `examples/results/geometric_calculator/geometric_calculator_plot.wl` +
  `plots/*.png` (circle plots, DFT bars, per-layer R² heatmap, neuron firing
  heatmap, neuron-cluster bars, steering dose-response) — one set per pilot model
  plus the cross-model minimum-architecture panel.
- `docs/experiments/geometric-calculator/{findings.md, llama3.2-3b.json,
  qwen3-1.7b.json, qwen3-0.6b.json}`.
- **`Cargo.toml`** — new `[[example]]` (`geometric_calculator`,
  `required-features = ["transformer"]`) and `[[test]]` (`validate_mlp_neurons`,
  `required-features = ["transformer"]`) entries, matching the existing pattern.
- **`HOOKS.md`** — new table row for `MlpActPost` (`blocks.{i}.mlp.hook_act_post`,
  `[batch, seq, d_mlp]`); plus the stale `src/hooks.rs` `MlpPre`/`MlpPost`
  doc-comment fix.
- README row + CHANGELOG entry.

---

## Reverse-engineering provenance

Same rigour bar as the PLT/CLT work: the d_mlp neuron capture must be proven
correct against a from-first-principles Python oracle *before* any scientific
claim, so that a small-scale null ("no modular-addition module") is
distinguishable from an implementation bug.

| Phase | Purpose | Artefact |
|---|---|---|
| 1. Format/semantics discovery | Confirm what `gate ⊙ up` / `act(fc(x))` is, per layout | This document's *Gap A* analysis of `src/transformer/mlp.rs` |
| 2. Python reference oracle | Dump d_mlp pre-down-projection vectors from `transformers` in F32 | `scripts/mlp_neurons_validation.py` + `_reference.json` |
| 3. Rust implementation | Capture the same tensor in candle-mi | `MlpActPost` + `forward_capturing` |
| 4. Cross-validation | Prove parity to a numerical bar | `tests/validate_mlp_neurons.rs` (top-K exact, abs-diff < 1e-4 CPU) |

**Gate:** do not enter Step A before Phases 2–4 pass on **both** Llama-3.2-3B
and Qwen3-1.7B-Base (the two families in the pilot), with the synthetic-weight
unit test green for all three MLP layouts.

---

## Scope discipline — what this document does NOT cover

- **Bit-exact Llama-3.1-8B layer-18 reproduction** — impossible at F32 on 16 GB;
  see *The hard constraint*. This is a consumer-scale method replication.
- **The depth axis** (`Llama-3.2-1B`, L=16) and **plain-MLP architecture**
  (`StarCoder2-3B`) — post-pilot expansion; the pilot holds depth fixed at 28.
- **GPT-NeoX / Pythia / OLMo** — needs new `gpt_neox` / `olmo` config parsers +
  oracles; deferred. (Pythia is the Kantamneni & Tegmark family and a true L=6
  floor — the highest-value future addition.)
- **Neuron-level interventions** (ablation/steering on `MlpActPost`) — the
  geometric-calculator steering is residual-level; neuron intervention is a
  follow-on.
- **A general FFT** — `fourier.rs` ships only the small-n DFT + circle-fit the
  experiment needs.

---

## References

- Feucht et al., _Arithmetic in the Wild: Llama uses Base-10 Addition to Reason
  About Cyclic Concepts_ — [arXiv:2605.01148](https://arxiv.org/abs/2605.01148);
  Goodfire blog _A geometric calculator inside a neural network_ (Neural Geometry
  Series, 2026-05-14).
- Kantamneni & Tegmark, _Language Models Use Trigonometry to Do Addition_ (2025).
- Gurnee et al., _When Models Manipulate Manifolds_ (Transformer Circuits, 2025) —
  the manifold-geometry finding already replicated in
  [`examples/character_count_helix.rs`](../../examples/character_count_helix.rs),
  the structural template for the circle-finder here.
- candle-mi internals: [`src/util/pca.rs`](../../src/util/pca.rs),
  [`src/hooks.rs`](../../src/hooks.rs),
  [`src/transformer/mlp.rs`](../../src/transformer/mlp.rs),
  [`HOOKS.md`](../../HOOKS.md).
