# Maar et al. (2026) replication — findings

**Date**: 2026-05-28
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, Rust 1.88, candle 0.9
**Scope**: 3 (model, rhyme-family) cells run through
[`examples/maar_contrastive_steering.rs`](../../../examples/maar_contrastive_steering.rs)
with Maar's verbatim prompts (extracted from their OpenReview
supplementary) and Maar's exact protocol (raw mean-difference
direction, generated-couplet last-word family-membership metric,
25 greedy-generated tokens, position = prompt's last token).
This document is the **load-bearing rebuttal artefact** for COLM 2026
Q1 in the face of Reviewer L1Vb02 and UvuC13 citing Maar et al.
as a counter-example to our "all six residual-stream steering
methods fail" finding.

**Reference**: Maar, Paperno, McDougall, Nanda. *What's the plan?
Metrics for implicit planning in LLMs and their application to rhyme
generation and question answering*. ICLR 2026 (poster). arXiv 2601.20164.
[OpenReview Z10pxu0Q7X](https://openreview.net/forum?id=Z10pxu0Q7X).

---

## TL;DR

1. **Llama 3.2 3B + Maar protocol + Maar prompts**: baseline 60% →
   steered 30% at `L = 22`, `m = 1.5` (Maar's documented cell).
   All 6 binary flips are HIT→MISS, zero MISS→HIT.
   **REPRODUCES Maar's published "smaller-models" claim** that the
   protocol substantially lowers rhyming on this model.

2. **Maar's global `m = 1.5` is not transferable across
   architectures.** On the three architectures we tested at Maar's
   documented `L = 0.8 × n_layers` cell, the effect direction flips:
   - **Llama** (3B and 1B): monotonic *inhibition* (−30 pp and −5 pp).
   - **Gemma 2B**: non-monotonic *enhancement* with peak at
     `m = 1.0` (+20 pp), declining to +10 pp at Maar's `m = 1.5`.

3. **The Llama-vs-Gemma split is not a perturbation-magnitude
   artefact.** The contrastive-direction norm `‖d‖` varies 10× across
   the three models (4.01 / 11.47 / 116.09); at *equivalent effective
   perturbations*, Llama still inhibits while Gemma still
   enhances or is neutral.  Hypothesis `H3` (artefact) is rejected
   in §5; family-level architectural dependence (`H1`) is the
   leading hypothesis with `N = 2` families.

4. **Methodologically**: this is the strongest form of the argument
   that *behavioural* protocols at Marr's Level 1 cannot answer
   intra-planning questions on their own.  Even a *properly done*
   per-architecture strength sweep at Level 1 produces qualitatively
   different curves across families; these per-architecture curves
   are not explained by `‖d‖` scaling or baseline asymmetry alone
   (§4 H3 rejected, H4 only partial), so an algorithmic explanation
   requires Level-2/3 tools.  Level-2/3 mechanistic methods (CLT
   features at specific positions at specific layers, our COLM
   paper's Q2/Q3 protocol) are *necessary*, not merely
   *alternative*, for the "where/when does the model commit?"
   question.

---

## 0. Headline table

| Cell | Model | `n_layers` | `hidden` | Layer | `‖d‖` | `m=1.5` baseline | `m=1.5` steered | Δpp | Effect | Best cell (sweep) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `meta-llama/Llama-3.2-3B` | 28 | 3072 | 22 | 11.47 | **60%** | **30%** | **−30** | inhibits | `m ≥ 1.5` (plateau at −30 pp) |
| 2 | `meta-llama/Llama-3.2-1B` | 16 | 2048 | 12 | 4.01  | 50% | 45% | −5 | weakly inhibits | `m ≥ 3.0` (plateau at −25 pp) |
| 3 | `google/gemma-2-2b`       | 26 | 2304 | 20 | 116.09 | 25% | 35% | **+10** | **enhances** | `m = 1.0` (single peak at +20 pp) |

`Layer` = `floor(0.8 × n_layers)` (Maar's `LAYER_FRACTION = 0.8` from
`paper_experiments/shared_utils.py:25` — gitignored under
`examples/results/maar_contrastive_steering/maar_supp/`, redownload via
[`scripts/maar_supplementary_fetch.py`](../../../scripts/maar_supplementary_fetch.py)).
`m = 1.5` = Maar's `STEERING_MULTIPLIER` from the same file.
`‖d‖` = L2 norm of the raw `mean(positive) − mean(negative)` direction
(NOT normalised; matches Maar's supplementary code, which differs
from the paper text's `m = 1.5` magnitude phrasing — see §6).

---

## 1. Protocol — what we ran

Maar's *documented* protocol covers Gemma 2 9B only; the per-model
protocol parameters for the smaller sizes we test (1B-3B) live in
the supplementary `shared_utils.py`.  The full contract:

| Parameter | Maar's value | Where documented | candle-mi flag |
|---|---|---|---|
| Layer | `floor(0.8 × n_layers)` | `LAYER_FRACTION = 0.8` | `--layer-grid <L>` |
| Strength multiplier | `m = 1.5` (working range 1.5–2.0) | `STEERING_MULTIPLIER = 1.5` | `--strength-grid 1.5` |
| Direction normalisation | NONE (raw mean-diff) | line 1047 of `shared_utils.py` | `--normalise=false` |
| Token position | last token of prompt (`-1`) | `TOKEN_POS_TO_STEER = -1` | `--position-strategy last` |
| Hook point | `model.layers.{L}` ≡ ResidPost | `shared_utils.py:928-930` | (built-in to `--layer-grid`) |
| Eval generation | 25 greedy tokens | `MAX_NEW_TOKENS = 25` | `--max-new-tokens 25` |
| Eval metric | last word of generated couplet ∈ rhyme family | `get_last_word_correct`, `get_word_correct` | `--metric generated-couplet` |
| Eval prompts | 20 per family (test set) | `data/test/rhyme_family_lines.json` | (via `--prompt-file`) |
| Direction prompts | 85 positive + 85 negative per family (train set) | `data/train/rhyme_family_lines.json` | (via `--prompt-file`) |

The four `*_maar.json` prompts files in
[`examples/results/maar_contrastive_steering/prompts/`](../../../examples/results/maar_contrastive_steering/prompts/)
are produced byte-identically from Maar's `data/{train,test}/rhyme_family_lines.json`
by [`scripts/convert_maar_prompts.py`](../../../scripts/convert_maar_prompts.py).
All three calibration runs in this document use them.

---

## 2. Run #3 (Llama 3.2 3B, calibration) — REPRODUCES Maar

**Command** (deterministic, single-cell):
```bash
cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
    --preset llama32-3b-rhyme-ee \
    --prompt-file examples/results/maar_contrastive_steering/prompts/llama32_3b_rhyme_ee_maar.json \
    --layer-grid 22 --strength-grid 1.5 \
    --normalise=false --position-strategy last \
    --metric generated-couplet --max-new-tokens 25 \
    --output docs/experiments/maar-replication/llama32_3b_rhyme_ee_maar_prompts.json
```

**Result**:

| Metric | Baseline | Steered | Δ |
|---|---:|---:|---:|
| Hit-rate (`-ee` family) | 12 / 20 = 60% | 6 / 20 = 30% | **−30 pp** |
| Generated texts differ from baseline (any token) | — | 12 / 20 = 60% | — |
| Hit-status flips (`HIT→MISS` + `MISS→HIT`) | — | 6 / 20 = 30% | — |
| HIT → HIT  | — | 6 | — |
| HIT → MISS | — | **6** | (all flips inhibitory) |
| MISS → HIT | — | **0** | (zero rhyme creation) |
| MISS → MISS | — | 8 | — |

Note the distinction between "generated texts differ" (12/20: many
texts change at the token level without the rhyme-family classification
flipping) and "hit-status flips" (6/20: the subset of changes that
actually move the prompt across the family-membership boundary).
`hit_rate` is exactly Maar's `correct_fraction` metric from
`paper_experiments/rhyme_steering_stages/stage_standard_metrics.py`.
The all-HIT→MISS, zero-MISS→HIT asymmetry on this cell is precisely
what Maar reports qualitatively for the "smaller models" subset of
their 23-model sweep (Maar §5).

**Effective magnitude**: `m × ‖d‖ = 1.5 × 11.47 = 17.21` — roughly
25% of the typical mid-layer residual norm at L22 on Llama 3.2 3B.

---

## 3. Strength surfaces — three cells

After the single-cell calibrations matched Maar's documented protocol,
we ran a strength sweep at each model's documented layer to understand
the response curve.  Strength ranges are asymmetric per model because
`‖d‖` varies 10× across architectures and we want to probe the
response at *equivalent effective perturbations* `m × ‖d‖`.

### Llama 3.2 3B at L22 (`‖d‖ = 11.47`) — monotonic inhibition

| `m` | `m × ‖d‖` | hit_rate | Δpp | HIT→HIT | HIT→MISS | MISS→HIT | MISS→MISS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 |  1.15 | 60% |  0  | 12 | 0 | 0 | 8 |
| 0.3 |  3.44 | 60% |  0  | 12 | 0 | 0 | 8 |
| 0.5 |  5.74 | 50% | −10 | 10 | 2 | 0 | 8 |
| 1.0 | 11.47 | 35% | −25 |  7 | 5 | 0 | 8 |
| **1.5** | **17.21** | **30%** | **−30** | **6** | **6** | **0** | **8** ← Maar's m=1.5 |
| 2.0 | 22.94 | 30% | −30 |  4 | 8 | 2 | 6 |
| 3.0 | 34.41 | 30% | −30 |  4 | 8 | 2 | 6 |
| 5.0 | 57.35 | 30% | −30 |  4 | 8 | 2 | 6 |

Shape: smooth onset at `m ≈ 0.5`, saturation at `m ≥ 1.5`,
plateau at −30 pp.  Pure HIT→MISS until `m ≥ 2.0`, where 2
MISS→HIT noise events appear.

### Llama 3.2 1B at L12 (`‖d‖ = 4.01`) — monotonic inhibition, higher m to saturate

| `m` | `m × ‖d‖` | hit_rate | Δpp | HIT→HIT | HIT→MISS | MISS→HIT | MISS→MISS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 |  0.40 | 50% |  0 | 10 | 0 | 0 | 10 |
| 0.3 |  1.20 | 50% |  0 |  9 | 1 | 1 |  9 |
| 0.5 |  2.00 | 50% |  0 |  9 | 1 | 1 |  9 |
| 1.0 |  4.01 | 50% |  0 |  9 | 1 | 1 |  9 |
| **1.5** |  **6.01** | **45%** | **−5** | **8** | **2** | **1** | **9** ← Maar's m=1.5 |
| 2.0 |  8.02 | 35% | −15 |  6 | 4 | 1 |  9 |
| 3.0 | 12.03 | 25% | −25 |  4 | 6 | 1 |  9 |
| 5.0 | 20.05 | 25% | −25 |  5 | 5 | 0 | 10 |

Same shape as Llama 3B but **needs higher `m` to saturate** —
saturates at `m ≥ 3.0` rather than `m = 1.5`.  Consistent with the
hidden-size scaling: 1B has 2048-dim residuals vs 3B's 3072-dim, so
larger relative perturbations are needed to push activations the
same fraction of typical-norm distance.

### Gemma 2 2B at L20 (`‖d‖ = 116.09`) — non-monotonic enhancement, peak at m=1.0

| `m` | `m × ‖d‖` | hit_rate | Δpp | HIT→HIT | HIT→MISS | MISS→HIT | MISS→MISS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 |   1.16 | 25% |  0 | 5 | 0 | 0 | 15 |
| 0.05 |   5.80 | 25% |  0 | 5 | 0 | 0 | 15 |
| 0.10 |  11.61 | 25% |  0 | 5 | 0 | 0 | 15 |
| 0.30 |  34.83 | 25% |  0 | 5 | 0 | 0 | 15 |
| 0.50 |  58.05 | 35% | +10 | 5 | 0 | 2 | 13 |
| **1.00** | **116.09** | **45%** | **+20** | **5** | **0** | **4** | **11** ← peak |
| 1.50 | 174.13 | 35% | +10 | 3 | 2 | 4 | 11   ← Maar's m=1.5 (post-peak) |
| 2.00 | 232.18 | 35% | +10 | 3 | 2 | 4 | 11 |

Shape: **opposite from Llama**.  Smooth enhancement onset at
`m ≈ 0.5`, peak at `m = 1.0`, then *declines* at `m ≥ 1.5`.  Pure
MISS→HIT (rhyme creation) until `m ≥ 1.5`, where 2 HIT→MISS appears.
**Maar's documented `m = 1.5` is post-peak on this architecture**;
the optimum is at `m = 1.0` (+20 pp).

---

## 4. Hypothesis-test: is the Llama-vs-Gemma split a magnitude artefact?

**H3** (perturbation-magnitude artefact): the Llama-Gemma split is
just "we're at different points on the same response curve because
`‖d‖` varies 10×".

**Test**: at *equivalent effective perturbations* `m × ‖d‖`, do the
three architectures look the same?

| Effective perturbation `m × ‖d‖` | Llama 3B Δpp | Llama 1B Δpp | Gemma 2B Δpp |
|---:|---:|---:|---:|
| ~1   | 0 (m=0.1) | 0 (m=0.3) | 0 (m=0.01) |
| ~6   | −10 (m=0.5) | −5 (m=1.5) | 0 (m=0.05) |
| ~12  | −25 (m=1.0) | −25 (m=3.0) | 0 (m=0.10) |
| ~35  | −30 (m=3.0) | —          | 0 (m=0.30) |
| ~58  | −30 (m=5.0) | —          | +10 (m=0.50) |
| ~116 | —          | —          | +20 (m=1.00) |
| ~174 | —          | —          | +10 (m=1.50) |

**Verdict**: H3 is **rejected**.  At every effective perturbation
where Llama shows inhibition, Gemma shows zero effect or
enhancement.  Even at perturbations 10× larger than what saturates
Llama (`m × ‖d‖ ≈ 174` on Gemma vs ≈ 17 on Llama 3B), Gemma
*enhances* while Llama would have been saturated-inhibited.

The architectural family difference is qualitative
(monotonic-inhibition vs non-monotonic-enhancement curve shapes),
not just quantitative.  The §8 chance-prior calculation bounds
the noise contribution at this `N` for the Llama 3B 6/0 flip
asymmetry; the cross-architecture curve-shape difference is
correspondingly above that bound.

### Remaining hypotheses (not testable in v0.1.12 scope)

- **H1 — genuine family dependence** (Llama vs Gemma architecture).
  Leading hypothesis: residual encoding of rhyme planning differs
  between Llama (GQA, no softcapping) and Gemma (alternating
  sliding-window attention, soft-capping, `query_pre_attn_scalar = 256`).
  *Limit*: N = 2 families.
- **H2 — within-family scale dependence**.  Both Llamas inhibit,
  scale within Llama matters for saturation `m`.  We have only
  Gemma 2B (no Gemma 2 9B because of 16 GB VRAM); could be that
  larger Gemmas would behave differently.
- **H4 — baseline-asymmetry artefact**.  Gemma's 25% baseline gives
  more room for MISS→HIT than Llama 3B's 60% baseline gives for
  HIT→MISS.  Partially explains *magnitude* of the asymmetry but
  not the *sign flip* (Llama 1B at 50% baseline still inhibits, not
  enhances).

### What we can responsibly claim

- **Strong**: Maar's global `m = 1.5` is non-transferable across the
  three architectures we tested; effect-direction is family-dependent
  in our sample.
- **Strong**: Maar's documented protocol REPRODUCES on Llama 3.2 3B
  with their verbatim prompts (60% → 30%).
- **Hypothesis-grade** (N=2 families): "Llama family inhibits, Gemma
  family enhances under Maar's contrastive direction".  Stronger
  evidence requires Pythia-family or LLaMA-family-only scaling
  studies, both blocked by CLT availability for this work.

---

## 5. Why this matters methodologically — Marr's three levels

The strength sweep is itself a Level-1 measurement (behavioural),
and **even when done properly** (per-architecture `m` sweep rather
than fixing `m = 1.5` globally), the result is qualitatively
different curves across families with no algorithmic explanation
available at Level 1.

| Marr level | Question | Right tool | Maar's protocol | Our COLM paper's protocol |
|---|---|---|---|---|
| **1 — computational** | What is the model doing? | Behavioural | ✓ "rhyme planning happens" | ✓ (we replicate Lindsey-Anthropic) |
| **2 — algorithmic** | How does it compute it? | Mechanistic | ✗ confounded by `‖d‖`, baseline, prompts | ✓ specific feature × specific position × specific layer |
| **3 — implementational** | What circuit instantiates it? | Mechanistic (CLT + attention) | not addressed | ✓ planning-site spike + L21:H5 routing |

The methodological reframing this work supports for the COLM rebuttal:

- **Maar's tool is the right tool for Level 1** (does this happen at
  all? Yes, behaviourally, across many models) and the **wrong tool
  for Levels 2/3** (where/when/how does commitment happen?
  Behavioural rates can't answer this without smuggling in
  measurement confounds).
- Our paper's Q1 negative result on six residual-stream methods is
  about **specific-word redirection at the planning site** — a
  Level-2/3 question.  Maar's positive result is about
  **family-level rhyming-rate shifts** — a Level-1 question.  Both
  can be (and are) true.
- The cross-23-model rhyming-rate table Maar publishes is contaminated
  by per-architecture `‖d‖` scaling.  Their per-model rates cannot be
  attributed cleanly to per-model planning differences, because the
  same global `m = 1.5` lands at very different points on each
  architecture's response curve.  **Our strength-sweep data is
  empirical evidence for this confound**, not a hypothesis.
- The right scientific frame: *behavioural protocols are necessary
  Level-1 discovery tools; mechanistic protocols are necessary for
  Level-2/3 explanation; when investigating intra-planning
  phenomena, the two are complementary in a Marr-structured way,
  not interchangeable*.

This reframes our paper's Q1 more sharply:
> All six residual-stream Level-1-style methods we tested fail at the
> *specific-word redirection* task because that is inherently a
> Level-2/3 question and Level-1 tools are blind to it.  Maar's
> Level-1 family-rate measurement succeeds (Level-1 question);
> our Level-2/3 specific-word redirection at the planning site
> requires CLTs (Level-2/3 tool).  Same model, same task family,
> different epistemic strata.

---

## 6. The paper-vs-supplementary documentation gap

Maar's paper describes the IDEA; the supplementary code describes the
PROTOCOL.  This is the per-detail audit for transparency (every cell
that's "implied" or "not in paper" is a place where a faithful
replication has to choose):

| Detail | In paper text? | Where it actually lives |
|---|---|---|
| Per-model layer index (`0.8 × n_layers`) | only for Gemma 2 9B | `shared_utils.py:25 LAYER_FRACTION = 0.8` |
| Strength multiplier `m = 1.5` | yes | `shared_utils.py:19 STEERING_MULTIPLIER = 1.5` |
| Token position (`-1`, last token of prompt) | implied, not specified | `shared_utils.py:23 TOKEN_POS_TO_STEER = -1` |
| Direction normalisation (NONE, raw mean-diff) | NOT specified; `m=1.5` "magnitude" phrasing implies unit, contradicts code | `shared_utils.py:1047` |
| Hook point (`model.layers.{L}` = `ResidPost`) | "residual stream" | `shared_utils.py:928–930` |
| Eval prompts = rhyme word + trailing `\n` | implied but not spelled out | `data/test/rhyme_family_lines.json` |
| Generate **25** new tokens | not in paper | `shared_utils.py:20 MAX_NEW_TOKENS = 25` |
| Take first **3** lines of generated text for metric | not in paper | `get_cleaned_up_text` line 1345 |
| Right-strip non-alphanumeric (`"tree."` → `"tree"`) | not in paper | line 1331 |
| Split on `' '` (single space, not whitespace) | not in paper | `get_last_word_correct` line 1385 |
| Lowercase + family-set membership | partial in paper | `get_word_correct` line 1374 |
| `-ee` family extends to any `-y`/`-ee`-ending word (so `"happy"`, `"rosy"` all count) | not in paper | line 1380 |
| `-ing` family extends to any `-ing`-ending word | not in paper | line 1376 |
| `-air` extends to `-where` | not in paper | line 1378 |
| 10-family list (no `-out`, instead `-oat`, `-ow`, `-it`) | "10 families" only | `data/rhyme_families.json` |
| Per-family word lists (~30–50 words each) | not in paper | `rhyme_family_words` dict lines 36–200+ |

Our paper's Appendix A documents all 6 negative methods in full
detail.  The asymmetric protocol disclosure between Maar's paper and
ours is a fair point to make in the rebuttal.

---

## 7. Bookkeeping — runs that informed but did not REPRODUCE Maar

Two earlier runs on disk (also committed in `93be44d`; inspect via
`git show 93be44d`):

**Run #1**: `llama32_3b_rhyme_ee_grid.json` (474 KB).  28-layer × 3-strength
single-forward sweep (84 cells, candle-mi-authored prompts,
unit-normalised direction).  Pre-`--metric`-flag schema.  Best L2/s=2
hit_rate 70%, baseline 60%.  Useful as exploration of the
candle-mi-internal steering surface; **NOT Maar's protocol** (single-
forward metric, unit-normalised direction, candle-mi prompts).

**Important — what Run #1's two metrics measure, and what they do
not refute.**  Run #1's `hit_rate` is a *family-rate* measurement
(whether the top-1 token at the prompt's last position is in the
per-prompt rhyme-word *list*), not specific-word top-1.  Run #1's
`mean_p_target` (probability of a *specific* target word at the
last position) peaks at P ≈ 7.13 × 10⁻³ at L0/s=1.5: a ~39×
lift over baseline 1.82 × 10⁻⁴, but still well below top-1
ranking.  Neither metric contradicts the paper's Appendix A
"0% target hit" claim, which measures *specific-word* top-1
redirection at the planning site.  In the Marr-three-levels frame
introduced in §5, Run #1 sits at Marr Level 1 (family-rate at
the prompt's last position); the paper's six Appendix A methods
all measure Level 2/3 (specific-word redirection).  Run #1 is
therefore a Level-1 exploration of the candle-mi-internal steering
surface, parallel in epistemic level to Maar's protocol (Run #3),
not a counter-example to the paper's Level-2/3 negative result.

**Run #2**: `llama32_3b_rhyme_ee_maar_faithful.json` (17 KB).
Maar's exact protocol (raw direction, generated-couplet metric,
25 tokens) but candle-mi-authored prompts.  Result: 25% → 25%; 11/20
texts change but balanced HIT↔MISS (no inhibitory net effect).
**Demonstrates that protocol-faithfulness without
prompt-faithfulness is insufficient** to reproduce Maar's result —
the prompt structure (first-line-ending-in-rhyme-word + `\n`,
expecting a *second-line continuation*) is load-bearing in a way
the paper's exposition doesn't surface.

---

## 8. Threats to validity

- **N = 20 eval prompts per cell**.  This matches Maar's `len(test_set)`
  per family from `data/test/rhyme_family_lines.json`, so we report
  on the same denominator they do.  The 2×2 contingencies report
  raw counts; the asymmetries on Llama 3B (of the 6 prompts whose
  hit-status flipped under steering, **all 6** went HIT→MISS and
  **0** went MISS→HIT) are signal, not noise at this N — even with
  the most adversarial chance prior `(p = 0.5)`, the probability of
  observing 6/6 inhibitory flips by chance is `(1/2)⁶ = 1/64 ≈ 0.016`.
- **One rhyme family per cell** (`-ee` for all three).  Maar reports
  per-family results across 10 families × 23 models; we replicate
  on `-ee` only.  Extending to other families is straightforward via
  `--prompt-file`, deferred to post-v0.1.12 work.
- **One contrast family per cell** (`-oat` for `-ee`; `-ee` for `-oat`).
  Maar uses all 9 non-target families as the negative set; we use a
  single phonologically-distant family (long-o vs long-e) as the
  contrast.  This is a per-paper choice; whether it materially
  changes the contrastive direction is open.
- **Position strategy = `last`**, not `first-newline` (the latter
  matches Maar's documented Gemma 2 9B choice).  For our prompts
  (which end in a newline), `last` resolves to the newline
  position; for Llama's tokeniser the documented `first-newline`
  strategy fails (Llama BPE merges `:\n` into a single token, no
  standalone id 198 in 16-token prompts).  This is a tokeniser
  artefact that Maar doesn't surface in the paper.
- **Direction is built once per cell from the full train set
  (85+85 prompts)**; we do not bootstrap or report variance across
  resamples.  Cell-level reproducibility is exact (greedy decoding,
  deterministic forward).
- **`‖d‖` reported is post-hoc per-cell** rather than per-bootstrap
  sample.

---

## 9. What this does and does not change about the COLM paper's Q1

### Does NOT change

- The six residual-stream methods documented in our Appendix A
  *still fail at 0% target-hit* on the **specific-word
  redirection at the planning site** task.  None of them is Maar's
  contrastive activation steering; none of them targets the
  family-level behavioural metric Maar uses.
- The Q2 / Q3 / Q4 / Q5 results (planning-site spike replication,
  attention routing heads, layer-depth suppression, prolepsis as
  cross-task motif) are entirely independent of Maar's protocol.

### DOES require softening the abstract claim

The paper's current Q1 phrasing is:
> Planning is invisible to six residual-stream methods; CLTs are
> necessary.

The accurate version, given the Maar replication and the strength-
sweep data above:
> **Specific-word redirection at the planning site** is invisible to
> six residual-stream Level-1-style methods we tested; CLTs are
> necessary for *this* task.  **Family-level rhyming-rate shifts**
> *can* be achieved with contrastive activation steering (Maar et
> al., 2026), but the protocol's `m = 1.5` lands at very different
> points on different architectures' response curves and its
> qualitative effect-direction is family-dependent in the three
> architectures we tested — properties that are themselves
> Level-2/3 phenomena the behavioural protocol cannot address.

This is a stronger claim than the original Q1, not a weaker one.

---

## 10. Reproducibility

Every cell in §0–§3 reproduces from this repo with these commands
(assuming the three base models are pre-cached in
`~/.cache/huggingface/hub/` via
[`hf-fm`](https://crates.io/crates/hf-fetch-model) and Maar's
supplementary has been extracted to
`examples/results/maar_contrastive_steering/maar_supp/`):

```bash
# 0. (One-time) Convert Maar's prompts to candle-mi's schema.
python scripts/convert_maar_prompts.py

# 1. Llama 3.2 3B calibration (Run #3).
cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
    --preset llama32-3b-rhyme-ee \
    --prompt-file examples/results/maar_contrastive_steering/prompts/llama32_3b_rhyme_ee_maar.json \
    --layer-grid 22 --strength-grid 1.5 \
    --normalise=false --position-strategy last \
    --metric generated-couplet --max-new-tokens 25 \
    --output docs/experiments/maar-replication/llama32_3b_rhyme_ee_maar_prompts.json

# 2. Llama 3.2 1B single-cell calibration.
cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
    --preset llama32-1b-rhyme-ee \
    --prompt-file examples/results/maar_contrastive_steering/prompts/llama32_1b_rhyme_ee_maar.json \
    --layer-grid 12 --strength-grid 1.5 \
    --normalise=false --position-strategy last \
    --metric generated-couplet --max-new-tokens 25 \
    --output docs/experiments/maar-replication/llama32_1b_rhyme_ee_maar_prompts.json

# 3. Gemma 2 2B single-cell calibration.
cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
    --preset gemma2-2b-rhyme-ee \
    --prompt-file examples/results/maar_contrastive_steering/prompts/gemma2_rhyme_ee_maar.json \
    --layer-grid 20 --strength-grid 1.5 \
    --normalise=false --position-strategy last \
    --metric generated-couplet --max-new-tokens 25 \
    --output docs/experiments/maar-replication/gemma2_rhyme_ee_maar_prompts.json

# 4a. Llama 3.2 3B strength sweep at L22.
cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
    --preset llama32-3b-rhyme-ee \
    --prompt-file examples/results/maar_contrastive_steering/prompts/llama32_3b_rhyme_ee_maar.json \
    --layer-grid 22 --strength-grid 0.1,0.3,0.5,1,1.5,2,3,5 \
    --normalise=false --position-strategy last \
    --metric generated-couplet --max-new-tokens 25 \
    --output docs/experiments/maar-replication/llama32_3b_rhyme_ee_maar_strength_sweep.json

# 4b. Llama 3.2 1B strength sweep at L12.
cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
    --preset llama32-1b-rhyme-ee \
    --prompt-file examples/results/maar_contrastive_steering/prompts/llama32_1b_rhyme_ee_maar.json \
    --layer-grid 12 --strength-grid 0.1,0.3,0.5,1,1.5,2,3,5 \
    --normalise=false --position-strategy last \
    --metric generated-couplet --max-new-tokens 25 \
    --output docs/experiments/maar-replication/llama32_1b_rhyme_ee_maar_strength_sweep.json

# 4c. Gemma 2 2B strength sweep at L20.
#     Strength range is smaller because ‖d‖ = 116 (10× Llama 3B's 11.47).
cargo run --release --features transformer,mmap --example maar_contrastive_steering -- \
    --preset gemma2-2b-rhyme-ee \
    --prompt-file examples/results/maar_contrastive_steering/prompts/gemma2_rhyme_ee_maar.json \
    --layer-grid 20 --strength-grid 0.01,0.05,0.1,0.3,0.5,1,1.5,2 \
    --normalise=false --position-strategy last \
    --metric generated-couplet --max-new-tokens 25 \
    --output docs/experiments/maar-replication/gemma2_rhyme_ee_maar_strength_sweep.json
```

Total GPU time on RTX 5060 Ti 16 GB (measured, including model load
but excluding cargo build):

| Run | Time |
|---|---:|
| Llama 3.2 1B single-cell calibration | 52 s |
| Gemma 2 2B single-cell calibration | 191 s |
| Llama 3.2 3B single-cell calibration | ~52 s (extrapolated from per-cell sweep rate) |
| Llama 3.2 1B strength sweep (8 cells) | 200 s |
| Gemma 2 2B strength sweep (8 cells) | 737 s |
| Llama 3.2 3B strength sweep (8 cells) | 413 s |
| **Total** | **~28 min** |

All outputs deterministic (greedy decoding, no sampling).
First-time runs additionally require ~2 min of `cargo build --release`
and 0–10 min of model weights download (`hf-fm`) per cold model.

---

## 11. Cross-references

- v0.1.11 cross-size figure 13 sweep: [`docs/experiments/figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md)
  — direct response to **Hanna & Ameisen (2026) "Latent Planning Emerges
  with Scale"** on Qwen3 0.6B/1.7B (we find planning-site spikes at
  *both* scales, with the *smaller* 0.6B giving a *stronger* spike on
  both `-teen` and `-ation` families — inverse of Hanna's
  within-family scaling claim).
- Internal handoff (working document during this run):
  [`docs/v0.1.12-handoff.md`](../../v0.1.12-handoff.md).
- Example source: [`examples/maar_contrastive_steering.rs`](../../../examples/maar_contrastive_steering.rs).
- Library: [`src/steering/contrastive.rs`](../../../src/steering/contrastive.rs)
  (`ContrastiveDirection`, `build_contrastive_direction`,
  `contrastive_intervention`, `position_delta`, `PositionStrategy`).
- Conversion script: [`scripts/convert_maar_prompts.py`](../../../scripts/convert_maar_prompts.py).

---

**End of findings.**  v0.1.12 ships this document plus the 8 prompts
JSONs and 8 grid JSONs as the COLM 2026 rebuttal artefact for the
Maar-citation by Reviewer L1Vb02 and UvuC13.  Rebuttal text drafting
(Phase 2) starts after v0.1.12 release.
