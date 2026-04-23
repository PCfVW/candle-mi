# AlgZoo Surprise-Accounting Bakeoff — Experiment Plan

**Scope:** a single, self-contained research artefact built on candle-mi v0.1.8's stoicheia Phase B tooling, targeting the [ARC AlgZoo open invitation](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) (Hilton et al., 2026). Timeline: May–August 2026, shipping before the September 2026 ARC hiring re-open.

**The question:** can we implement and fairly compare multiple mechanistic estimators of AlgZoo RNN accuracy under both the **MSE-vs-compute** metric (random-sampling baseline) and the **surprise-accounting** metric (`explanation_bits ≤ selection_bits`), and is the trade-off curve different from what ARC's internal results imply? Secondary: does the bridging framework between **planning entropy** (Jacopin, Allerton 2018 / ISITA 2020) and ARC's **surprise accounting** (Christiano et al.) yield a novel cross-domain evaluation principle — or do the two frameworks reduce to the same quantity under careful inspection?

**Why a dedicated document.** candle-mi v0.1.10 is the Gemma arm of the prolepsis planning-signal experiment (a different scientific question, unrelated model family, unrelated metric). The AlgZoo work has its own cadence, its own external audience (ARC), and its own publication arc. Separating the experiment from other release coordination lets us track progress against the scientific target without conflating timelines.

**September-ready deliverable.** A shippable artefact — example file, reproducible JSON output, findings writeup — that a visiting-researcher application to ARC can point at as concrete evidence of "contribution to our research agenda." See *Success criteria* below.

---

## Context

Three pieces of prior work define the space:

1. **AlgZoo (Hilton et al., 2026)** — [blog post](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/), [repo](https://github.com/alignment-research-center/alg-zoo). Tiny trained models (8–1,408 parameters) as test cases for mechanistic interpretability. Four task families: 2nd argmax (RNN), argmedian (RNN), median (RNN), longest cycle (transformer). Three analysis techniques demonstrated on 2nd argmax RNNs: piecewise linear decomposition (M₂,₂, 10 params), approximate symmetries (M₄,₃, 32 params), feature extraction + leave-one-out-max subcircuits (M₁₆,₁₀, 432 params). The explicit open challenge: *"Design a method for mechanistically estimating the accuracy of M₁₆,₁₀ that matches the performance of random sampling in terms of mean squared error versus compute."*
2. **ARC's surprise-accounting framework** ([Formal verification, heuristic explanations and surprise accounting](https://www.alignment.org/research/), Christiano et al.) — a mechanistic estimate of model behaviour counts as a "full understanding" when the **bits of surprise** from the explanation (explanation length in the MDL sense, plus residual divergence from observed behaviour) does not exceed the **bits of selection** that chose the model from the training distribution. Introduced as an alternative to MSE-vs-compute when the question is understanding rather than prediction.
3. **Planning Entropy (Jacopin, Allerton 2018 / ISITA 2020)** — entropy `H(P) = -Σ pᵢ log pᵢ` computed over agent **plan distributions**, used as a design-level metric for AI planners ("how surprising are the agent's plans to the observer?"). Originally motivated by Ubisoft game-design requests for "planning for surprises." Structurally an information-theoretic evaluation of predictability over a distribution of structured objects.

The two surprise frameworks both measure `-log p` quantities under distributions of structured objects, but they differ in **what has the distribution** (ARC: model behaviours given explanations; Jacopin: plans given an agent's policy) and in **what the null baseline is** (ARC: bits of selection pressure; Jacopin: uniform-random planner). The bridge is that both frame the evaluation question as: *"is the observed entity more predictable under my mechanistic model than under a null baseline, and by how many bits?"* This plan's Step E formalises that bridge as a cross-domain framing.

---

## Prerequisites

Everything this experiment needs already ships in candle-mi v0.1.8:

- **`StoicheiaRnn` backend** (`src/stoicheia/mod.rs`) — loads 2nd-argmax / argmedian / median RNNs from AlgZoo's `.pth` or `.safetensors` weights; cross-validated against Python reference to 1e-4.
- **`fast.rs` kernel** — raw f32 forward pass at 18–25× candle-tensor throughput. The **enabling infrastructure** for MSE-vs-compute sweeps at 10⁸+ samples on a 16 GB consumer GPU.
- **`MechanisticEstimator` trait** (`src/stoicheia/surprise.rs`) — abstract interface that concrete estimators implement. `OracleEstimator` baseline already exists.
- **`surprise_accounting()` + `surprise_accounting_noisy()` harness** — evaluates any `MechanisticEstimator` against ARC's framework; supports weight-perturbation variants.
- **Phase B analysis primitives**: `piecewise.rs` (ReLU region enumeration with 320-bit compact patterns), `ablation.rs` (single + pairwise neuron sweeps), `probing.rs` (Pearson-correlation role classification), `standardize.rs` (exact-equivalence weight rescaling).

**New code required for this plan:** four concrete `MechanisticEstimator` implementations (Step B), a sweep harness for MSE-vs-compute at graded budgets (Step C), and the cross-framework theoretical writeup (Step E). No new library primitives — the trait and evaluation machinery are already in place.

**External assets:** AlgZoo's published weights at `gs://arc-ml-public/alg/zoo` (MIT-0 licensed). Pulled via standard `gsutil`, re-uploaded to the candle-mi author's HuggingFace namespace if bandwidth-prohibitive for reproducers; otherwise fetched on first run.

---

## Folder structure

```
candle-mi/
├── examples/
│   └── stoicheia_surprise_bakeoff.rs              ★ new
├── docs/experiments/
│   └── algzoo-surprise-bakeoff/
│       ├── findings.md                            ★ final writeup (Step F)
│       ├── m2_2_bakeoff.json                      ★ per-model primary + surprise data
│       ├── m4_3_bakeoff.json
│       ├── m16_10_bakeoff.json
│       └── bridging-framework.md                  ★ Step E theoretical essay
└── src/stoicheia/
    └── estimators/                                ★ new module
        ├── mod.rs
        ├── piecewise_exact.rs
        ├── piecewise_pruned.rs
        ├── ablation_sensitivity.rs
        └── role_composition.rs
```

Gated behind the existing `stoicheia` feature flag; no new Cargo dependencies anticipated.

---

## Experiment

Working file: `examples/stoicheia_surprise_bakeoff.rs`. Output: per-model JSON under `docs/experiments/algzoo-surprise-bakeoff/`. Findings: `docs/experiments/algzoo-surprise-bakeoff/findings.md`.

### Step A — Random-sampling baseline

Goal: establish the MSE-vs-compute curve for uniform random sampling on each target model. This curve is the **null baseline** every mechanistic estimator in Step C is scored against.

- Targets: **M₂,₂** (10 params, N=2), **M₄,₃** (32 params, N=3), **M₁₆,₁₀** (432 params, N=10). Optional stretch: longest-cycle transformer at h=4.
- Load each model via `StoicheiaRnn::load` (or `StoicheiaTransformer::load` for the stretch target).
- Use `fast.rs` to draw i.i.d. Gaussian inputs at budgets `[10³, 10⁴, 10⁵, 10⁶, 10⁷, 10⁸]`. The 18–25× kernel is load-bearing here — 10⁸ candle-tensor samples would cost weeks on consumer hardware.
- For each budget: compute the estimated accuracy (fraction correct); compute MSE vs ground truth accuracy (established at 10⁹ samples for M₁₆,₁₀, exhaustive enumeration for M₂,₂ and M₄,₃).
- Serialise: `baseline_mse_vs_compute` field in per-model JSON.

**Exit criterion:** three monotonically-decreasing MSE-vs-compute curves serialised. Sanity check: the 10⁸-sample MSE is at least an order of magnitude below the 10³-sample MSE for each model.

**Time estimate:** ~1 week. The 18–25× kernel makes this almost trivially cheap; bulk of the time is fixture setup and the exhaustive/ground-truth runs.

### Step B — Four mechanistic estimators

Goal: implement four concrete `MechanisticEstimator` implementations, each using a different Phase B primitive. Tests pass; unit cross-validation against oracle matches to 1e-4 for each.

1. **`PiecewiseExactEstimator`** — uses `piecewise.rs::classify_regions()` to enumerate every ReLU activation region the model reaches on its input distribution; within each region the model is a linear function and its per-region accuracy is closed-form. **Exact** for M₂,₂ (≤16 regions) and M₄,₃ (≤256 regions expected; bounded by 2^H · N for H hidden units). **Infeasible** for M₁₆,₁₀ without pruning.
2. **`PiecewisePrunedEstimator`** — same machinery, but enumerates only the top-k most-visited regions (k configurable, budget-graded). The estimator's MSE at each compute budget is measured by the budget spent on region enumeration. This is the direct candidate for the M₁₆,₁₀ open-problem entry.
3. **`AblationSensitivityEstimator`** — uses `ablation.rs::ablate_neurons()` to compute each neuron's single-neuron causal effect on accuracy; builds a per-input accuracy predictor from the vector of active-neuron effects. Compute budget = number of ablation sweeps × number of inputs.
4. **`RoleCompositionEstimator`** — uses `probing.rs::probe_neurons()` to classify each hidden unit by Pearson-correlated role (running-max tracker, leave-one-out-max, comparator, etc.); predicts accuracy from the composition of roles present in the active neuron set. Compute budget = probing passes × probe set size.

Each estimator implements `MechanisticEstimator` (trait defined in `src/stoicheia/surprise.rs`); all four composite with the existing `surprise_accounting()` evaluation harness.

**Exit criterion:** all four estimators compile behind the `stoicheia` feature, each has a unit test cross-validating against `OracleEstimator` output on M₂,₂ to 1e-4 tolerance.

**Time estimate:** ~4 weeks. `PiecewiseExactEstimator` is the cheapest (most of the logic exists in `piecewise.rs`). `AblationSensitivityEstimator` and `RoleCompositionEstimator` are roughly equal effort. `PiecewisePrunedEstimator` depends on the exact variant.

### Step C — MSE-vs-compute comparison

Goal: evaluate every (model, estimator) pair at graded compute budgets, plot against the Step A baseline.

- For each (model, estimator) pair: measure estimator MSE at compute budgets `[10³, 10⁴, ..., 10⁸]` (same scale as Step A).
- "Compute" for each estimator is the count of primitive operations the estimator performs to produce its prediction — region enumerations for piecewise, ablation sweeps for sensitivity, probe runs for role composition.
- Plot: one chart per model, with five curves (random baseline + four estimators) on the same MSE-vs-compute axes.
- Serialise: `estimator_mse_vs_compute` field in per-model JSON, keyed by estimator name.

**Exit criterion:** three comparison plots rendered (SVG or PNG). At least one estimator should be competitive with random sampling on M₂,₂ (the tractable model); whether any is competitive on M₁₆,₁₀ is the open scientific question this plan exists to answer.

**Time estimate:** ~2 weeks. Bulk of the time is the M₁₆,₁₀ runs at 10⁷–10⁸ budget; 18–25× kernel keeps it tractable.

### Step D — Surprise accounting

Goal: evaluate each (model, estimator) pair against ARC's surprise-accounting framework. Does `explanation_bits + residual_surprise ≤ selection_bits`?

- For each (model, estimator): invoke `surprise_accounting()`. Record explanation bits (length of the explanation in MDL sense — piecewise model needs log(N_regions) · region_size bits, ablation needs H · log(effect_precision), etc.), residual surprise (bits of divergence between estimator prediction and observed model behaviour on a held-out sample), and selection bits (ARC-provided or computed from training specs).
- Tabulate: per (model, estimator): `{explanation_bits, residual_surprise, selection_bits, bound_satisfied, tightness}`.
- Also run `surprise_accounting_noisy()` at three noise levels (perturb weights by σ ∈ {0.01, 0.05, 0.1} of their standard deviation) to measure how sensitive each estimator is to weight perturbation — ARC's blog specifically mentions this as an evaluation lens.

**Exit criterion:** for each model, the surprise-accounting table is populated. For M₂,₂ and M₄,₃ (blog says "fully understood"), `PiecewiseExactEstimator` should satisfy the bound with meaningful tightness; if it doesn't, that's itself a finding worth reporting.

**Time estimate:** ~2 weeks.

### Step E — Bridging framework essay

Goal: write `bridging-framework.md` — a standalone theoretical essay (4–6 pages) that formalises the connection between planning entropy and surprise accounting.

Content outline:
1. **Planning entropy recap** (1 page). `H(P_plans)` as an agent-design metric; Allerton 2018 / ISITA 2020 results; the Ubisoft-originated question about "plans for surprises."
2. **Surprise accounting recap** (1 page). Christiano et al.'s framework; `explanation_bits ≤ selection_bits`; the relationship to MDL and algorithmic information theory.
3. **Structural common ground** (1–2 pages). Both frameworks are `-log p` under a distribution over structured objects. Both define a null baseline against which predictability is measured. Both turn "understanding" into a budget problem.
4. **Structural differences** (1 page). Planning entropy distributes over action sequences; surprise accounting distributes over model behaviours given explanations. Planning entropy's null baseline is uniform; surprise accounting's is bits of selection. Planning entropy is open-loop; surprise accounting is closed-loop (estimator prediction vs ground truth).
5. **Synthesis — a cross-domain evaluation principle?** (1 page). Is there a unified principle under which both reduce to the same quantity? Or are they two specialisations of a more general "predictability-vs-null-baseline budget" framework? This is the section where a novel research claim can land.

**Exit criterion:** essay committed under `docs/experiments/algzoo-surprise-bakeoff/bridging-framework.md`; at least one falsifiable claim in §5; references to Allerton 2018, ISITA 2020, Christiano et al., and all relevant ARC blog posts.

**Time estimate:** ~2 weeks (writing is usually slower than expected).

### Step F — Write-up and ship

Goal: package everything as a September-ready artefact.

- Consolidate into `findings.md`. Narrative arc: *"Can Phase B primitives implement mechanistic estimators that compete with random sampling under the MSE-vs-compute metric, and are any of them admissible under surprise accounting? Cross-framework connection to Planning Entropy elaborated in companion essay."*
- Headline table: per-model, per-estimator, MSE-at-10⁶ vs baseline-at-10⁶, surprise-bound-satisfied, explanation-bits.
- README Paper-replications row pointing at the bakeoff.
- CHANGELOG entry under whichever release hosts this (`[0.1.10]` if bundled with Gemma arm, `[0.1.11]` if shipped separately).
- Optional: arxiv preprint with the bridging-framework essay as the core contribution, bakeoff as the empirical demonstration.

**Exit criterion:** all three output JSONs committed, `findings.md` committed, `bridging-framework.md` committed, README + CHANGELOG updated, release shipped.

**Time estimate:** ~2 weeks.

---

## Timeline

Total budget: ~13 weeks → fits between late-April 2026 and mid-August 2026 with ~1 month buffer before September.

| Month | Steps | Deliverable |
|---|---|---|
| **May 2026** | Step A (baseline), Step B.1 (`PiecewiseExactEstimator`), Step B.4 (`RoleCompositionEstimator`) | Two estimators compile + tested; baseline curves for all three models |
| **June 2026** | Step B.2 (`PiecewisePrunedEstimator`), Step B.3 (`AblationSensitivityEstimator`), Step C (MSE-vs-compute) | All four estimators live; three comparison plots |
| **July 2026** | Step D (surprise accounting), Step E (bridging framework essay) | Surprise-accounting tables complete; theoretical essay drafted |
| **August 2026** | Step F (writeup, ship) + buffer | `findings.md` + release shipped; buffer for polish |

This is the **full-version** schedule. Fallback is the minimum-viable cut, below.

---

## Minimum-viable fallback

If life intervenes and the full 13-week scope doesn't fit, the scientific contribution still lands with this reduced scope (shippable in ~7 weeks):

- **Step A**: all three models (required — the baseline curves are the reference point).
- **Step B**: `PiecewiseExactEstimator` + `RoleCompositionEstimator` only (two estimators).
- **Step C**: restrict to M₂,₂ and M₄,₃ (skip M₁₆,₁₀ comparison curves).
- **Step D**: surprise accounting on M₂,₂ and M₄,₃ (where piecewise-exact is tractable).
- **Step E**: skip the bridging essay OR include as a single section in `findings.md` rather than a standalone essay.
- **Step F**: ship `findings.md` with the reduced results table.

Even this reduced artefact is:
- The first external (non-ARC) surprise-accounting implementation published.
- The first reproducible MSE-vs-compute baseline on AlgZoo models from a consumer-GPU stack.
- A concrete entry in the leaderboard ARC's open-challenge posting implies.

---

## Success criteria — "September-ready"

A September 2026 visiting-researcher application to ARC should be able to point at:

- ✅ A crates.io release (v0.1.10 or v0.1.11) with the bakeoff example and primitives committed.
- ✅ `docs/experiments/algzoo-surprise-bakeoff/findings.md` with a concrete results table.
- ✅ At least one falsifiable claim in the Planning-Entropy ↔ surprise-accounting bridging framework (full version) or a one-section-in-findings version of the same (fallback).
- ✅ An arxiv preprint or ARC-blog-style writeup that a reader can read in ≤20 minutes.

**Nice-to-have** (don't block on these):
- A preliminary email exchange with Jacob Hilton about the bakeoff results (follows the Olah playbook from v0.1.9 — ship first, email second).
- A v0.1.11 release dedicated to the bakeoff, separate from the Gemma arm, if timing works.

---

## Scope discipline — what this document does NOT cover

- **Training new AlgZoo models.** ARC's published weights at `gs://arc-ml-public/alg/zoo` are reused as-is. Training cost is not in candle-mi's consumer-GPU envelope.
- **Cross-task generalisation to longest-cycle transformer.** Noted as optional stretch target in Step A; if it ships, it's a bonus. If not, it's a natural v0.1.12 follow-up.
- **Attempting a full-understanding claim on M₁₆,₁₀.** The blog's explicit open challenge is in play, but "beats random sampling on MSE-vs-compute on M₁₆,₁₀" is an empirical outcome we report whichever way it goes — we don't promise to beat it.
- **Planning-Entropy → surprise-accounting theorem.** The bridging framework essay (Step E) is a *framework* essay, not a *theorem* paper. A formal reduction of one framework to the other would be a separate v0.1.13+ research project.
- **The Gemma arm of the planning-signal experiment (v0.1.10).** Tracked separately in `PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`. Parallel tracks; dependencies are minimal.

---

## References

- Hilton et al., *AlgZoo: Uninterpreted Models with Fewer than 1,500 Parameters* (2026) — [blog](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/), [repo](https://github.com/alignment-research-center/alg-zoo).
- Christiano et al., *Formal verification, heuristic explanations and surprise accounting* — [ARC blog](https://www.alignment.org/research/).
- Jacopin, *Statistical Planning: Entropy of Centralized Planning for Multi-Agent Systems*, Allerton Conference 2018.
- Jacopin, *Entropy to Control Planning in Video-Games*, ISITA 2020.
- candle-mi v0.1.8 `stoicheia` module: [`src/stoicheia/`](../../src/stoicheia/) — Phase B tooling that this experiment composes over.
- Sister experiment plan: [`PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`](PLAN-PLT-LLAMA-PLANNING-SIGNAL.md) — same plan format; v0.1.9 reference.
- Release playbook: [`release-sequence.md`](release-sequence.md) — for shipping the v0.1.10 / v0.1.11 that hosts this.
