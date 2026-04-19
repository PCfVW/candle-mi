# PLT Planning Signal on Llama 3.2 1B — Experiment Plan

**Scope:** a single experimental question, tracked separately from the [v0.1.9 master roadmap](candle_mi_v019_roadmap_V3.md).

**The question:** does a Per-Layer Transcoder (PLT) detect the planning-site signal on Llama 3.2 1B, and how does its causal effect on `P(planned_word)` compare to the Cross-Layer Transcoder (CLT) result already documented in the COLM 2026 submission?

**Why a dedicated document.** The v0.1.9 master roadmap bundles many things — PLT loader infrastructure, Gemma follow-up, a conditional Qwen-3 scale sweep. The planning-signal question has its own cadence and narrative arc. Separating the experiment from the release coordination lets us track progress against the scientific target without the infrastructure noise.

---

## Status (2026-04-19)

The experiment design below is preserved as a historical plan. Steps A–D are
complete; Step E is pending the v0.1.9 tag.

| Step | Status | Artefact |
|---|---|---|
| A — Reproduce Jacopin CLT result on Llama 3.2 1B | ✅ Complete | [`clt_step_a_llama.json`](../experiments/clt-vs-plt-planning-site/clt_step_a_llama.json) — CLT suppress+inject (Jacopin cross-layer features) reaches ΔP("that") = +0.687 at position 30, within candle 0.9 stack drift of the paper's +0.777. |
| B — PLT arm + full V3 Step 1.7 instrumentation | ✅ Complete | [`clt_vs_plt_llama.json`](../experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json) — PLT suppress-only ΔP = +0.986; method-matched CLT (same-layer top-5) ΔP = +5.7×10⁻⁷. Single prompt (PLAN proposed two; one delivered). |
| C — Classify outcome against V3 Appendix A | ✅ Complete | [`findings.md`](../experiments/clt-vs-plt-planning-site/findings.md) — **Outcome C** on the primary metric (different spikes). |
| D — (A)–(F) discrimination battery | ✅ Complete, plus Follow-up 1 | (B) confirmed dominant: CLT max-over-target top-5 suppress recovers ΔP = +0.87–0.92 at pos 30, reclassifying to **Outcome B** under method-matched-per-transcoder-capabilities ranking. Findings.md has the full battery and the follow-up table. |
| E — Write up and ship with v0.1.9 | ⏳ Pending | README paper-replications row, CHANGELOG [0.1.9] consolidation, v0.1.9 tag. |

**Result in one sentence.** Both transcoder classes detect the Llama 3.2 1B rhyming-couplet planning site at comparable ΔP when each is ranked via a method that respects its decoder topology (same-layer for PLT, max-over-target for CLT); naïve single-layer decoder-projection undersells the CLT by seven orders of magnitude on this prompt.

---

## Context

Three pieces of prior work define the space:

1. **Jacopin, [*What Is the Minimum Architecture for Prolepsis?*](https://arxiv.org/abs/2604.15010)** (COLM 2026 submission) — six residual-stream methods fail to detect the planning circuit on Gemma 2 2B. A Cross-Layer Transcoder (CLT) succeeds on both Gemma 2 2B (70% localisation across 136 pairs) and Llama 3.2 1B (`P("that") = 0.777` at L14, 133,879× ratio on the best prompt). The paper concludes that learned sparse feature decomposition is necessary to observe the planning circuit; the six tested methods did not include PLTs.
2. **Hanna & Ameisen, *Latent Planning Emerges with Scale*** ([arXiv:2604.12493](https://arxiv.org/abs/2604.12493), ICLR 2026) — PLTs detect planning features on Qwen-3, with consistent success only at 14B+ parameters and nascent signal at 4B–8B.
3. **The gap** — no one has run PLTs on the exact models where CLT ground truth exists. `mntss/transcoder-Llama-3.2-1B` makes this experiment possible for the first time, at a scale (1B) where Hanna & Ameisen's own scale-emergence prior would predict weak or absent PLT signal in Qwen-3.

Framing (per working stance): this is extensive empirical coverage in a statistical framework, not hypothesis testing with a theorem-under-risk. Every outcome fills a cell in the coverage matrix below:

|  | Llama 3.2 1B | Gemma 2 2B | Qwen-3 14B |
|---|---|---|---|
| 6 residual-stream methods | ✗ (Jacopin) | ✗ (Jacopin) | — |
| CLT | ✓ (Jacopin) | ✓ (Jacopin) | — (no open CLT) |
| PLT | **? (this plan)** | ? (v0.1.10) | ✓ (Hanna & Ameisen) |

---

## Prerequisites

These land in [v0.1.9 V3 roadmap](candle_mi_v019_roadmap_V3.md) and are infrastructure, not science. This experiment is unblocked when V3 Step 1.5 passes:

- **V3 Steps 1.1–1.3** — `TranscoderSchema` enum (`CltSplit` / `PltBundle` / `GemmaScopeNpz`), schema-aware loader helpers (`decoder_row`, `decoder_file_and_tensor_name`), unit tests for both active schemas.
- **V3 Step 1.4** — from-first-principles Python reference oracle for Llama 3.2 1B PLT (`scripts/plt_llama_validation.py` + `scripts/plt_llama_reference.json`). Reverse-engineering validation — mirrors plip-rs's `clt_reference.py` methodology.
- **V3 Step 1.5** — Rust parity test (`tests/validate_plt.rs`) confirms candle-mi's PLT encode matches Python to abs-diff < 1e-4.

---

## Reverse-engineering provenance

Before the experiment can run, the PLT format must be understood to the same rigour that plip-rs achieved for CLTs (90/90 top-10 parity, max relative error 1.2×10⁻⁶). This section maps the four-phase methodology plip-rs used for CLTs onto candle-mi's PLT work and names where each phase lives.

| Phase | Purpose | plip-rs (CLT) artefact | candle-mi (PLT) artefact |
|---|---|---|---|
| **1. Format discovery** | Identify files, tensor names, shapes, dtypes before coding any loader | [`examples/inspect_clt.rs`](../../../plip-rs/examples/inspect_clt.rs) — downloads files, iterates, prints headers | **Done today** via `hf-fm list-files` + `hf-fm inspect` HTTP-range reads — see V3 pre-flight section. Strict improvement over plip-rs: no downloads required. |
| **2. Python reference oracle** | Prove the encoder formula is understood by implementing it from first principles; produce a deterministic reference for Rust parity | [`scripts/clt_reference.py`](../../../plip-rs/scripts/clt_reference.py) — loads raw safetensors, computes `ReLU(W_enc @ x + b_enc)` directly; outputs `clt_reference_426k.json` | V3 Step 1.4: `scripts/plt_llama_validation.py` — same philosophy, adapted for the PltBundle schema (un-suffixed tensor names, BF16 native). Circuit-tracer is optional secondary oracle only. |
| **3. Rust implementation** | Load the format in candle-mi's MI framework | [`src/clt.rs`](../../../plip-rs/src/clt.rs) | V3 Steps 1.1–1.2: `TranscoderSchema::PltBundle` + schema-aware `decoder_row` and `decoder_file_and_tensor_name` helpers in `src/clt/mod.rs`. |
| **4. Cross-validation** | Prove the Rust implementation matches the Python oracle to a numerical bar | plip-rs integration tests consuming `clt_reference_426k.json` | V3 Step 1.5: `tests/validate_plt.rs` — abs-diff < 1e-4 (F32, CUDA), top-10 exact match. |

**Why this matters for Llama 3.2 1B specifically.** Llama 3.2 1B is small (1.2B params, 16 layers) and was not trained on data specifically curated for rhyming-couplet planning — the Jacopin paper already reports that Llama "searches but doesn't commit" on this task (main.tex §Q4). When the experiment in the next section runs, we may observe a weak or absent planning signal. **We must be able to distinguish "the model's planning signal is genuinely weak at this scale" from "our PLT implementation has a bug."** The reverse-engineering rigour — especially Phase 2's from-first-principles oracle — is what makes that distinction possible. Without the parity test passing, a null result on the planning signal is uninterpretable.

**Gate:** entering Step A of the experiment before Phases 1–4 are all complete and documented is not recommended. Phase 1 is already done (today). Phases 2–4 land as V3 Steps 1.1–1.5.

---

## Folder structure

candle-mi conventions (documented in [`scripts/README.md`](../../scripts/README.md) and observable in the existing `data/`, `experiments/`, `docs/audit/`, `docs/conventions/`, `docs/dogfooding-feedbacks/`, `docs/roadmaps/` layout) place:

- Python validation infrastructure → `scripts/` with pattern `<feature>_validation.py` + `<feature>_reference.json` + optional `<feature>_comparison.md`.
- Integration tests → `tests/` with pattern `validate_<feature>.rs`.
- Rust examples → `examples/`.
- Durable documentation → `docs/<kind>/` where `<kind>` ∈ `{audit, conventions, dogfooding-feedbacks, roadmaps}`.
- Large post-completion archives (figures, multi-MB JSON, video) → `experiments/YYYYMMDD-candle-mi-vX-Y-Z-<description>.7z`.

**One net-new addition** in this plan: a `docs/experiments/` subfolder (sibling of `docs/audit/`, etc.) for in-flight experiment artefacts — writeup plus supporting JSON that is too small / too iterative for the archive convention in `experiments/`. All paths in this plan and in V3 Step 1.7 follow the structure below:

```
candle-mi/
├── docs/
│   ├── audit/                                      (existing)
│   ├── conventions/                                (existing)
│   ├── dogfooding-feedbacks/                       (existing)
│   ├── roadmaps/                                   (existing — V1/V2/V3, PLAN-PLT-LLAMA-PLANNING-SIGNAL)
│   └── experiments/                                ★ NEW — sibling of audit/, conventions/, ...
│       └── clt-vs-plt-planning-site/               ★ per-experiment folder (matches example name)
│           ├── findings.md                         ← the writeup (populated in Step E)
│           ├── clt_vs_plt_llama.json               ← Llama arm primary + instrumentation (v0.1.9)
│           └── clt_vs_plt_gemma.json               ← Gemma arm primary + instrumentation (v0.1.10)
├── scripts/
│   ├── plt_llama_validation.py                     ← from-first-principles PLT oracle (V3 Step 1.4)
│   ├── plt_llama_reference.json                    ← deterministic reference output
│   └── plt_llama_comparison.md                     ← Rust vs Python writeup (pattern: clt_position_sweep_comparison.md)
├── examples/
│   └── clt_vs_plt_planning_site.rs                 ← the experiment (V3 Step 1.7, Step B of this plan)
└── tests/
    └── validate_plt.rs                             ← PLT loader + Python parity test (V3 Step 1.5)
```

**Conventions honoured:**
- **Script naming follows candle-mi, not plip-rs.** candle-mi uses `<feature>_validation.py` + `<feature>_reference.json` + `<feature>_comparison.md`. plip-rs used `<feature>_reference.py` (no separate validation). The methodology is identical; the naming follows the host project.
- **No `outputs/` directory.** candle-mi does not have one; the plip-rs `outputs/` equivalent is `docs/experiments/<slug>/`.
- **Per-experiment folder name matches the example.** `docs/experiments/clt-vs-plt-planning-site/` ↔ `examples/clt_vs_plt_planning_site.rs`. Obvious correspondence on filesystem browse; future experiments (routing-heads-via-PLT, irrevocability-via-PLT) get their own sibling folders.

**When to archive.** After `v0.1.9` ships with Step E complete and `findings.md` finalised, the per-experiment folder can optionally be snapshotted to `experiments/20260420-candle-mi-v0-1-9-clt-vs-plt-planning-site.7z` following the existing candle-mi convention. This is post-hoc and optional — the `docs/experiments/clt-vs-plt-planning-site/` folder remains the canonical live location during development and for small artefacts.

---

## Experiment

Working file: `examples/clt_vs_plt_planning_site.rs` (Llama arm). Output: `docs/experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json`. Findings: `docs/experiments/clt-vs-plt-planning-site/findings.md`.

### Step A — Reproduce the Jacopin CLT result on Llama 3.2 1B in candle-mi

Goal: lock the CLT baseline in the same harness, same device, same code path that the PLT arm will use. No new analysis — just confirm the numbers from the paper reproduce in candle-mi's harness so that CLT and PLT are run apples-to-apples.

- Load `mntss/clt-llama-3.2-1b-524k`.
- Run the suppress + inject position sweep on the two rhyming-couplet prompts from `examples/figure13_planning_poems.rs`.
- Record: spike (layer, position), top-5 planning-aligned features, full position-sweep profile, ΔP(planned_word), Δlogit(planned_word).

**Exit criterion:** CLT numbers reproduce the paper's Llama results to within noise. Reference: paper §Q2 reports `P("that") = 0.777 at L14, 133,879× ratio` on the best prompt, "all strong injections peak at the planning site" across 44 pairs.

### Step B — Run the identical protocol with PLT

Goal: get the PLT data point. Apples-to-apples with Step A.

- Load `mntss/transcoder-Llama-3.2-1B` (via `TranscoderSchema::PltBundle`).
- Same prompts, same suppress + inject protocol, same device. Suppress top-5 planning-aligned PLT features at the spike position.
- Record the same observables as Step A.
- Capture the full V3 Step 1.7 instrumentation payload:
  - Full activation trace across **all layers** for the top-20 planning-aligned features in both transcoders (not just the top-5 used for suppression).
  - Encoder output distribution — 32-bin histogram of encoder pre-activation at the spike layer and the two neighbours, per transcoder.
  - Both CLT decoder-slice metrics in parallel (same-layer and max-over-target-layers).
  - Llama PLT `W_skip · x` projection onto `unembed(planned_word)` at the spike position.
  - Top-20 feature IDs + decoder vectors exported alongside top-5.

**Exit criterion:** experiment runs to completion on one CUDA forward pass per arm. JSON dumped to `docs/experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json` with all instrumentation fields populated.

### Step C — Classify the outcome

Map the primary metric (`ΔP_CLT` vs `ΔP_PLT`) onto the four outcomes in [V3 Appendix A](candle_mi_v019_roadmap_V3.md):

| Signature | Outcome | One-line interpretation |
|---|---|---|
| ΔP_PLT ≈ ΔP_CLT, same spike (layer, position) | **B** | PLT sufficient for detection and intervention on this model. |
| ΔP_PLT ≈ 0 while ΔP_CLT is strong | **A** | PLT misses the spike. Cross-layer binding apparently required at 1B. |
| 0 < ΔP_PLT < ΔP_CLT, same spike | **D** | PLT detects but cannot control cleanly. Detection/intervention separable. |
| PLT spike at different (layer, position) vs CLT | **C** | PLT finds a different decomposition. Triggers Step D. |

**Exit criterion:** outcome label assigned in `docs/experiments/clt-vs-plt-planning-site/findings.md` with the primary-metric table populated.

### Step D — If and only if Outcome C: run the (A)–(F) discrimination battery

Do **not** log-and-move-on if the spike locations disagree. Investigate using the already-captured data.

Full specification in [V3 Step 1.7 Investigation commitment](candle_mi_v019_roadmap_V3.md). Summary:

- **(A) Dictionary resolution mismatch** — subsample PLT features to CLT per-layer resolution; re-rank.
- **(B) CLT decoder-projection ambiguity** — compare same-layer vs max-over-slices metrics (both captured in Step B).
- **(C) Cross-layer binding** — inspect the per-layer activation trace of each top-20 feature.
- **(D) Activation-function regime** — examine encoder histograms + Llama PLT `W_skip` contribution.
- **(E) Feature-granularity mismatch** — qualitative inspection of top-20 feature decoder vectors.
- **(F) Training-objective divergence** — flagged, not resolved without re-training.

**Epistemic stance:** (A)–(D) are mechanically resolvable on the already-captured data; (E) is best-effort semantic interpretation; (F) is acknowledged beyond scope.

**Exit criterion:** subsections (A)–(F) in `docs/experiments/clt-vs-plt-planning-site/findings.md` are populated with data, interpretation, or an explicit "out of scope" acknowledgement.

### Step E — Write up and ship

- Finalise `docs/experiments/clt-vs-plt-planning-site/findings.md` narrative. Frame as coverage-matrix extension of the Jacopin/Lindsey empirical protocol, not as hypothesis testing.
- Update README Paper replications table with a Hanna & Ameisen row pointing to `examples/clt_vs_plt_planning_site.rs`.
- Update `CHANGELOG.md` under `[0.1.9]`.
- Ships with V3 Stage 1 commit sequence items 7–9 — no separate release.

---

## Deliverables

- `examples/clt_vs_plt_planning_site.rs` — Llama arm, complete with V3 instrumentation payload.
- `docs/experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json` — primary metric + instrumentation captures.
- `docs/experiments/clt-vs-plt-planning-site/findings.md` — outcome classification and, if Outcome C, (A)–(F) discrimination battery.
- README row + CHANGELOG entry.

---

## Scope discipline — what this document does NOT cover

- **Gemma 2 2B PLT arm** — ships in v0.1.10, separately tracked under V3 Step 1.6. Gemma provides the second data point that turns a singleton Llama result into a two-point pattern; it is load-bearing for a publishable statistical claim but not blocking for v0.1.9.
- **Routing-head analysis under PLT intervention** (paper Q3 via PLT) — future work. Requires attention-pattern capture under PLT suppress+inject, then comparison against the paper's Llama routing-head tables (paper §Q3 and Appendix E).
- **Irrevocability test under PLT intervention** (paper Appendix G via PLT) — future work.
- **Qwen-3 scale sweep** — conditional Stage 2 in V3, gated on the decision checkpoint that runs after this experiment.

This document tracks property **(1) early commitment** of the prolepsis motif only. The full three-property motif (early commitment + routing-mediated propagation + irrevocability) under PLT decomposition is a separate investigation for a future release.

---

## References

- Jacopin, *What Is the Minimum Architecture for Prolepsis?* (COLM 2026 submission) — [arXiv:2604.15010](https://arxiv.org/abs/2604.15010), particularly §Q1 (six methods), §Q2 (Llama 3.2 1B and Gemma 2 2B spike replication), §Q4 (Llama searches but commits unreliably), §Appendix G (irrevocability direct test).
- Hanna & Ameisen, *Latent Planning Emerges with Scale* (ICLR 2026) — [arXiv:2604.12493](https://arxiv.org/abs/2604.12493).
- candle-mi v0.1.9 master roadmap: [`candle_mi_v019_roadmap_V3.md`](candle_mi_v019_roadmap_V3.md).
- CLT ground truth for Llama 3.2 1B in candle-mi: [`examples/figure13_planning_poems.rs`](../../examples/figure13_planning_poems.rs).
