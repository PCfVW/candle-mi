# Gridworld Prolepsis — Transfer to Action Planning — Experiment Plan

**Scope:** a single experimental question, tracked separately from the master
roadmap, targeting candle-mi **v0.1.14**. (`v0.2.0` is reserved for the NLnet
grant work; this is a `v0.1.x` patch.)

**The question:** does the prolepsis pattern observed in rhyme planning (the
COLM 2026 submission, Gemma 2 2B + Llama 3.2 1B) transfer to action planning
in a 2D gridworld domain, and if so, does the commitment bind to the **spatial
direction** or to the **mapped output token**?

**Why a dedicated document.** This is a method-transfer replication with its own
narrative arc (rhyme planning to action planning, with a permutation-test
distinguishing two commitment hypotheses), distinct from release coordination.
Separating it lets us track the scientific target without infrastructure noise,
the same convention used by
[`PLAN-GEOMETRIC-CALCULATOR.md`](PLAN-GEOMETRIC-CALCULATOR.md) and
[`PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`](PLAN-PLT-LLAMA-PLANNING-SIGNAL.md).

---

## Status (2026-06-04)

Design phase. No code written yet. This document is the pre-flight plan; it
will be reviewed before any implementation begins.

| Step | Status | Artefact (planned) |
|---|---|---|
| 0 — Gridworld generator + prompt formatter | ⏳ Not started | `scripts/gridworld_generator.py`, prompt module in `examples/gridworld_prolepsis.rs` |
| A — Baseline feasibility on Gemma 2 2B | ⏳ Not started | `docs/experiments/gridworld-prolepsis/baseline_gemma2_2b_2.5m.json` |
| B — Prolepsis replication (suppress-plus-inject) | ⏳ Not started | `docs/experiments/gridworld-prolepsis/prolepsis_gemma2_2b_2.5m.json` |
| C — Permutation test (spatial vs lexical commitment) | ⏳ Not started | `docs/experiments/gridworld-prolepsis/permutation_gemma2_2b_2.5m.json` |
| D — Irrevocability test (Appendix G analogue) | ⏳ Not started | `docs/experiments/gridworld-prolepsis/irrevocability_gemma2_2b_2.5m.json` |
| E — Write-up + figures + ship | ⏳ Not started | `findings.md`, `gridworld_prolepsis_plot.wl` + `plots/*.png`, README row, `CHANGELOG.md` |

**Result in one sentence (to be filled at Step E).** _TBD — whether the
prolepsis spike geometry replicates in gridworld action planning, and whether
the commitment layer is mapping-invariant (spatial) or mapping-dependent
(lexical)._

---

## Context

Three pieces of prior work define the space:

1. **The COLM 2026 submission** — _What Is the Minimum Architecture for
   Prolepsis? Early Irrevocable Commitment Across Tasks in Small Transformers_
   ([arXiv:2604.15010](https://arxiv.org/abs/2604.15010)). Establishes
   prolepsis as a structural motif in rhyme planning on Gemma 2 2B and Llama
   3.2 1B: early CLT feature activation at the planning site, sustained
   propagation via attention routing (L21:H5 in Gemma), no downstream
   correction. The suppress-plus-inject protocol at the planning site reaches
   P(target) between 0.483 (Gemma 2 2B, 426K CLT, "around") and 0.834 (Llama
   3.2 1B, 524K CLT, "that"). The Appendix G irrevocability test shows the
   L21:H5 routing delta locked at +0.023 across the correction-strength sweep.

2. **Taufeeque et al.,** _Planning in a recurrent neural network that plays
   Sokoban_ ([arXiv:2407.15421](https://arxiv.org/abs/2407.15421), 2024) and
   _Path channels and plan extension kernels_
   ([arXiv:2506.10138](https://arxiv.org/abs/2506.10138), 2025). The
   transformer equivalent question is open: does an LLM solving a planning
   task commit to the first action at an early layer and sustain that
   commitment, in the manner the COLM paper found for rhyme? Taufeeque's
   choice of a recurrent network was deliberate (path channels and
   backtracking are easier to localise in a recurrent substrate); the
   transformer side is unmapped.

3. **The vocab-scan-as-feasibility-probe insight from the COLM paper, inverted.**
   The COLM paper's Phase-1 was a vocab scan to verify which rhyme groups have
   clean CLT features. In rhyme planning the prompt structure dictated which
   words could appear at the rhyme position, so the vocab scan was a hard
   feasibility gate. In action planning we have full degrees of freedom on the
   action-token names. We therefore **skip the vocab scan entirely** by mapping
   the four cardinal actions to four tokens we already know have validated CLT
   features at the 2.5M scale (see "The structural constraint" below).

**Working stance.** This is method-transfer with two distinguishable outcomes
(replicates / does not replicate) and a built-in disambiguation sub-experiment
(spatial vs lexical commitment via the permutation test). Either outcome on the
main question is publishable; the disambiguation question is interesting
regardless of the main outcome.

---

## The structural constraint: action tokens with validated CLT features

The COLM paper's Appendix D §D.2 reports for Gemma 2 2B at the 2.5M CLT scale
that **16 of 264 pairs** reached P(inject) ≥ 0.1 in the suppress-plus-inject
sweep. The high-redirect words include `black` (P = 0.522, 3.4 × 10¹¹× ratio,
the best absolute P in the 2.5M scan), `kind` (3.78 × 10¹²× ratio, the best
ratio in the 2.5M scan), `well`, `round`, `can`, `that`.

Pick four for the cardinal action mapping. Suggested **baseline mapping**:

| Cardinal action | Mapped token | Source (COLM Appendix D §D.2) |
|---|---|---|
| Up | `black` | P = 0.522 |
| Down | `kind` | best ratio in the 2.5M scan |
| Left | `well` | P ≥ 0.1 row |
| Right | `round` | P ≥ 0.1 row |

The permutation test (Step C) uses a **permuted mapping** that re-assigns the
same four tokens to different cardinal actions:

| Cardinal action | Permuted mapped token |
|---|---|
| Up | `round` |
| Down | `well` |
| Left | `kind` |
| Right | `black` |

**Tokenization sanity check (Step 0 prerequisite).** Confirm all four mapped
tokens (`black`, `kind`, `well`, `round`) tokenize as single tokens in the
Gemma 2 2B tokenizer. This is implicit in their appearance as 2.5M CLT
high-redirect features but should be verified explicitly before Step A
(`encode_raw` + length check, the same pattern used by the COLM-paper
vocabulary scan).

---

## Pilot model and CLT

Single cell for the pilot:

| Role | Model | Type | Layers | CLT |
|---|---|---|---|---|
| **Anchor** | `google/gemma-2-2b` | gemma2 | 26 | `mntss/clt-gemma-2-2b-2.5M` |

**Why Gemma 2 2B at the 2.5M CLT.** The deepest validated word-level CLT
feature coverage in the candle-mi cache; all four chosen action tokens
(`black`, `kind`, `well`, `round`) are validated above the P ≥ 0.1 threshold
in the COLM paper's Appendix D §D.2 sweep. Already validated in candle-mi as
of v0.1.7.

**Deferred to post-pilot expansion** (out of scope here): Llama 3.2 1B 524K
(would need a new vocab scan since Appendix D §D.3 reports only `that`
reaching P ≥ 0.1 there); Qwen3 0.6B / 1.7B with BlueLightAI 16K / 20K
transcoders (would need new vocab scans for action-token coverage); the 426K
Gemma CLT (lower feature resolution).

---

## Library prerequisites

No new candle-mi internals are required. The infrastructure for
suppress-plus-inject ([`figure13_planning_poems`](../../examples/figure13_planning_poems.rs)),
correction tests ([`correction_test`](../../examples/correction_test.rs)), and
CLT loading is already in place. Two thin pieces of new scaffolding:

### Gap A — Gridworld instance generator (Step 0)

A small Python script that emits gridworld instances with a single unambiguous
correct first move.

- `scripts/gridworld_generator.py` (new; ~80 lines).
- Inputs: grid size N (default 5), number of instances I (default 100),
  random seed.
- Output JSON: a list of
  `{agent: [x, y], goal: [x, y], correct_action: "Up"|"Down"|"Left"|"Right",
  instance_id: int}`, written to
  `docs/experiments/gridworld-prolepsis/gridworld_instances.json`.
- Filter: keep only instances where the Manhattan-distance-dominant action is
  **unique** (e.g., agent (2,3) to goal (4,4) has Right strictly dominant;
  agent (3,3) to goal (4,4) has Right or Up equally valid, drop). This
  guarantees every instance has a single ground-truth correct action.
- No walls, no obstacles, no diagonal moves in the pilot.

### Gap B — Prompt formatter (Rust side, in the example file)

A small module in `examples/gridworld_prolepsis.rs` (no new `src/` code):

- Reads the gridworld instances JSON.
- For each instance, emits a prompt of the form:

  ```
  Grid: 5x5. Agent: ({ax},{ay}). Goal: ({gx},{gy}). Walls: none.
  Map: Up→{m_up}, Down→{m_down}, Left→{m_left}, Right→{m_right}.
  Best next move ({m_up}/{m_down}/{m_left}/{m_right}): 
  ```

- The mapping `{m_up, m_down, m_left, m_right}` is a CLI parameter so the same
  binary runs the baseline (Step B) and the permutation (Step C) without
  rebuild.

No `src/` code is added in this experiment. The MLP/hooks/CLT machinery the
example calls is already in place.

---

## Folder structure

Follows the convention established by
[`PLAN-GEOMETRIC-CALCULATOR.md`](PLAN-GEOMETRIC-CALCULATOR.md) and
[`PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`](PLAN-PLT-LLAMA-PLANNING-SIGNAL.md): a
per-experiment folder under `docs/experiments/` whose name matches the example.

```
candle-mi/
├── docs/
│   ├── roadmaps/
│   │   └── PLAN-GRIDWORLD-PROLEPSIS.md          ★ this document
│   └── experiments/
│       └── gridworld-prolepsis/                  ★ NEW per-experiment folder
│           ├── findings.md                       ← write-up (Step E)
│           ├── gridworld_instances.json          ← Step 0 output
│           ├── baseline_gemma2_2b_2.5m.json      ← Step A: feasibility
│           ├── prolepsis_gemma2_2b_2.5m.json     ← Step B: spike geometry
│           ├── permutation_gemma2_2b_2.5m.json   ← Step C: spatial vs lexical
│           └── irrevocability_gemma2_2b_2.5m.json ← Step D: locked routing
├── scripts/
│   └── gridworld_generator.py                    ← NEW: instance generator
└── examples/
    ├── gridworld_prolepsis.rs                    ← NEW: the experiment
    └── results/
        └── gridworld_prolepsis/                  ★ figures (convention)
            ├── gridworld_prolepsis_plot.wl       ← Mathematica plotting script
            └── plots/                            ← rendered PNGs (committed)
```

Convention: the per-instance JSON (output of `gridworld_generator.py`) lives
under `docs/experiments/gridworld-prolepsis/` alongside the four results JSONs;
the Mathematica `.wl` script and the rendered PNGs live under
`examples/results/gridworld_prolepsis/`. This matches the split used in
`PLAN-GEOMETRIC-CALCULATOR.md`.

---

## Experiment

Working file: `examples/gridworld_prolepsis.rs`. Always run with the `mmap`
feature (user convention):

```
cargo run --features clt,transformer,mmap --release --example gridworld_prolepsis -- <args>
```

### Step 0 — Gridworld generator + prompt formatter (infrastructure, not science)

- Implement `scripts/gridworld_generator.py` per Gap A.
- Implement the prompt-formatter module inside `examples/gridworld_prolepsis.rs`
  per Gap B.
- Verify the tokenization sanity check: `black`, `kind`, `well`, `round` each
  tokenize as a single token in the Gemma 2 2B tokenizer.

### Step A — Baseline feasibility on Gemma 2 2B

Goal: confirm Gemma 2 2B reliably produces the correct mapped action on simple
unambiguous gridworld instances. The prolepsis test is only meaningful against
a baseline where the model gets the right answer.

- Run the prompt format on the 100 gridworld instances with the **baseline
  mapping** (`Up→black, Down→kind, Left→well, Right→round`).
- Measure top-1 accuracy at the output position (without any intervention).
- **Exit criterion:** baseline top-1 accuracy ≥ 0.80 on the 100 instances. If
  below threshold, change the prompt format (add few-shot examples, explicit
  chain-of-thought, alternative wording) until it crosses. Record the final
  prompt format together with per-instance outputs in
  `baseline_gemma2_2b_2.5m.json`.

### Step B — Prolepsis replication on planning (suppress-plus-inject)

Goal: do we see the planning-site spike geometry from the COLM paper Figure 1
on action-planning prompts?

Only run if Step A passes.

- For each instance, capture `ResidPost` at every layer at the trailing-space
  position immediately before the action token (the "planning site"
  structurally analogous to the rhyme planning case).
- Apply the suppress-plus-inject protocol: suppress the correct-action feature
  (e.g., the `round` CLT feature on a Right-correct instance), inject an
  alternative-action feature (e.g., the `kind` CLT feature). Sweep the
  injection position across all token positions in the prompt.
- Measure P(injected action token) at the output position for each injection
  position.
- **Exit criterion:** at some injection position, P(injected action token) is
  at least 2× its baseline floor (analogous to the COLM paper's 70%
  localisation-rate threshold), with the peak at or near the trailing-space
  planning site. If flat across all positions, prolepsis does not transfer in
  this domain at this scale — record the finding in `prolepsis_gemma2_2b_2.5m.
  json` and skip to Step E.

### Step C — Permutation test (spatial vs lexical commitment)

Goal: when the model commits at the planning site, is the commitment to the
spatial direction or to the mapped output token?

Only run if Step B yields a spike.

- Re-run Step B with the **permuted mapping**
  (`Up→round, Down→well, Left→kind, Right→black`).
- For each instance class (Up-correct, Down-correct, Left-correct,
  Right-correct), compare the commitment layer (where the per-layer P(target)
  spike peaks at the planning-site position) under the two mappings.
- **Exit criterion (interpretation):**
  - **Spatial commitment hypothesis:** the commitment layer is the same under
    both mappings (mapping-invariant). The model commits to "Up / Down / Left
    / Right" at layer L, and the lookup-and-emit-token resolution happens
    later in the network.
  - **Lexical commitment hypothesis:** the commitment layer shifts with the
    mapping. The model commits to the mapped output token (`black`, `kind`,
    `well`, `round`) at layer L, and the spatial reasoning happens upstream.

The two hypotheses are empirically distinguishable on the same dataset.
Record both per-mapping commitment layers in
`permutation_gemma2_2b_2.5m.json`.

### Step D — Irrevocability test (Appendix G analogue)

Goal: is the commitment, once made, irrevocable, in the manner the Gemma 2 2B
L21:H5 routing delta was locked under contradictory injection (COLM Appendix
G)?

Only run if Step B (and ideally Step C) yields a spike.

- Identify the commitment layer L from Step B (and Step C if applicable).
- Inject a contradictory action-token feature (an alternative mapped token's
  feature) at post-commitment layers (L+1 through n_layers) at up to 2× the
  commitment strength.
- Measure P(correct mapped token) and P(injected mapped token) as functions of
  correction strength.
- **Exit criterion:** the correction sweep produces the same three-outcome
  framework as COLM Appendix G:
  - P(correct) stays at ~0 at all strengths: irrevocability is architectural.
  - P(correct) rises above threshold: commitment is overridable.
  - P(commit) drops, P(correct) does not rise: correction disrupts but does
    not redirect.

The expected outcome under the prolepsis-transfers hypothesis is the first
(irrevocability is architectural). Record the correction-sweep table in
`irrevocability_gemma2_2b_2.5m.json`.

### Step E — Write-up, figures, ship

- `findings.md`: the answers to (i) does prolepsis transfer to action planning,
  (ii) is the commitment spatial or lexical, (iii) what is the irrevocability
  outcome; framed as method-transfer with a built-in disambiguation
  sub-experiment, not hypothesis testing under risk.
- README row in the "Paper extensions" (or equivalent) section pointing at the
  example.
- `CHANGELOG.md` under `[Unreleased]`: the example, the script, the findings
  doc.
- Tag `v0.1.14` only after the full pre-commit + preflight gate is green
  (`scripts/preflight.ps1`). The documented `-Full` trigger ("adding a new
  model family") does not apply here; the standard fast-path preflight is
  sufficient.

---

## Predictions (preregistration-style, recorded before any run)

Marking these before Step A so the result is judged against an explicit prior.

1. **Goal encoding depth.** Step B's commitment layer L_action will be
   **deeper** than the COLM paper's commitment layer for rhyme planning
   (L14–L22 band on Gemma 2 2B). Goal-state encoding from the prompt requires
   more semantic processing than phonological-constraint encoding.

2. **Spatial vs lexical commitment.** Step C will yield the **spatial**
   commitment hypothesis: the commitment layer is mapping-invariant. The
   model's planning circuit resolves the abstract action before resolving the
   token-mapping. Prediction strength: moderate. The alternative (lexical
   commitment, mapping-dependent) is also plausible if the model treats the
   prompt as a token-completion task without abstracting over the spatial
   directions.

3. **Irrevocability.** Step D will yield the **architectural irrevocability**
   outcome (P(correct) at ~0 across the correction sweep), matching the COLM
   Appendix G Gemma result. If the planning circuit's commitment is structural
   at this scale, it should be irrevocable in the same sense.

Each prediction is independent and falsifiable. A reverse outcome on any one
of the three is a finding, not a failure of the experiment.

---

## Figures (deliverable and visual validation)

Convention (followed exactly, matching `PLAN-GEOMETRIC-CALCULATOR.md`,
`helix_plot.wl`, `attention_routing_plot.wl`, `convergence_plot.wl`): the
example emits JSON; a companion Mathematica `gridworld_prolepsis_plot.wl`
under `examples/results/gridworld_prolepsis/` imports it
(`Import[..., "RawJSON"]`) and `Export`s PNGs to a `plots/` subfolder.

| Result artefact | candle-mi figure | Template to reuse | JSON source |
|---|---|---|---|
| Step B spike geometry (replicate COLM Fig 1) | per-instance P(injected target) vs injection position curves | `figure13_planning_poems` figure pattern | `prolepsis_gemma2_2b_2.5m.json` |
| Step B planning-site localisation across instances | per-layer × per-mapping-token P(injected target) heatmap at the planning-site position | `attention_routing` heatmap template | `prolepsis_gemma2_2b_2.5m.json` |
| Step C commitment-layer comparison | side-by-side baseline-vs-permuted per-layer spike curves at the planning-site position, on shared axes | bespoke two-panel small multiples | `permutation_gemma2_2b_2.5m.json` |
| Step D irrevocability curve | correction-strength sweep of P(correct mapped token) vs P(committed mapped token) | `correction_test` plot pattern | `irrevocability_gemma2_2b_2.5m.json` |

**Honest framing.** The COLM paper's Figure 1 spike is the qualitative target;
Step B either replicates that shape or does not. The Step C two-panel
comparison is the headline figure for the spatial-vs-lexical question.

---

## Deliverables

- `scripts/gridworld_generator.py`.
- `examples/gridworld_prolepsis.rs`.
- `examples/results/gridworld_prolepsis/gridworld_prolepsis_plot.wl` +
  `plots/*.png` (Step B spike curves, per-layer heatmap, Step C two-panel
  comparison, Step D correction-sweep curve).
- `docs/experiments/gridworld-prolepsis/{findings.md, gridworld_instances.
  json, baseline_gemma2_2b_2.5m.json, prolepsis_gemma2_2b_2.5m.json,
  permutation_gemma2_2b_2.5m.json, irrevocability_gemma2_2b_2.5m.json}`.
- **`Cargo.toml`** — new `[[example]]` (`gridworld_prolepsis`,
  `required-features = ["clt", "transformer"]`).
- README row + CHANGELOG entry.

No new `src/` code, no new tests, no new HOOKS.md row. This is a pure
application of existing candle-mi infrastructure (CLT loading, suppress-plus-
inject, correction sweep) to a new task domain.

---

## Reverse-engineering provenance

Different from PLT/CLT/MlpActPost work: no new tensor capture is introduced,
so there is no Python-vs-Rust parity oracle to write. The validation gate here
is **baseline correctness on the new prompt format** (Step A).

| Phase | Purpose | Artefact |
|---|---|---|
| 1. Prompt-format discovery | Confirm Gemma 2 2B handles in-context mapping plus simple spatial reasoning on unambiguous instances | `baseline_gemma2_2b_2.5m.json`, top-1 accuracy ≥ 0.80 |
| 2. Existing-feature validation | Already done in the COLM paper's Appendix D §D.2 sweep for the 2.5M CLT (P ≥ 0.1 threshold cleared for all four chosen tokens) | COLM paper Appendix D §D.2 |
| 3. Intervention-semantics validation | Already done in `figure13_planning_poems` (v0.1.7+) and `correction_test` (v0.1.7+) | existing candle-mi examples |

**Gate:** do not enter Step B before Step A's baseline top-1 accuracy crosses
0.80. A flat spike at Step B is only interpretable against a correct baseline;
otherwise it could be a prompt-comprehension failure rather than a
prolepsis-does-not-transfer finding.

---

## Scope discipline — what this document does NOT cover

- **Multi-step planning.** Pilot is single-action only (the first move from a
  given state). Multi-step plans (output the full sequence of moves) are a
  follow-on experiment.
- **Walls, obstacles, and diagonal moves.** Pilot uses a clean 5x5 grid with
  no obstacles and four cardinal actions. Walls add state-tracking demand;
  diagonal moves expand the action vocabulary beyond what is conveniently
  single-token across both pilot and post-pilot models.
- **Other models.** Pilot is Gemma 2 2B with the 2.5M CLT only. Llama 3.2 1B
  524K and the Qwen3 BlueLightAI cells are post-pilot expansion (each requires
  a new vocab scan because the 2.5M-validated tokens may not clear P ≥ 0.1
  there).
- **Training new CLTs.** This plan uses existing validated features only.
  Planning-oriented CLT training is a separate roadmap (out of scope here;
  see also the v0.2.0 NLnet scope).
- **Other planning domains.** Blocksworld with single-token operator names,
  Sokoban with single-token directional moves, and MindGames-style multi-step
  reasoning are all interesting follow-ons but are out of scope for the pilot.

---

## References

- The COLM 2026 submission — _What Is the Minimum Architecture for Prolepsis?
  Early Irrevocable Commitment Across Tasks in Small Transformers_
  ([arXiv:2604.15010](https://arxiv.org/abs/2604.15010)). Appendix D §D.2 for
  the 2.5M CLT high-redirect token list, Appendix G for the irrevocability
  test protocol.
- Lindsey et al., _On the Biology of a Large Language Model_
  ([Transformer Circuits, 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems)) —
  the planning-in-poems precedent and Figure 13 protocol.
- Taufeeque et al., _Planning in a recurrent neural network that plays
  Sokoban_ ([arXiv:2407.15421](https://arxiv.org/abs/2407.15421), 2024), and
  _Path channels and plan extension kernels_
  ([arXiv:2506.10138](https://arxiv.org/abs/2506.10138), 2025) — the recurrent
  precedent for planning-circuit identification.
- candle-mi internals: [`figure13_planning_poems`](../../examples/figure13_planning_poems.rs),
  [`correction_test`](../../examples/correction_test.rs),
  [`src/clt/`](../../src/clt/), [`HOOKS.md`](../../HOOKS.md).
