# Prolepsis in planning — interim brief

**Question.** Does the prolepsis pattern from rhyme planning (Anthropic's
"planning in poems" Figure 13; the COLM 2026 *Minimum Architecture for
Prolepsis* work) — early, sustained, irrevocable commitment to a single-token
output at a planning site — transfer to **action planning**? Roadmap:
[PLAN-GRIDWORLD-PROLEPSIS.md](../../roadmaps/PLAN-GRIDWORLD-PROLEPSIS.md).

## Where we are

**Cell 1 — gridworld (negative; a modality finding).** The original pilot asked
a model to pick the dominant first move from `(x, y)` coordinates. Base
`google/gemma-2-2b` is at **chance (~0.30)** here, across coordinate, ASCII-grid,
and direct-direction encodings and 0–20-shot (randomized) — and it is *not* the
token mapping or the encoding (direct direction words and ASCII both stay at
chance). The blocker is spatial reasoning: single-action gridworld is really
coordinate comparison, which a transformer has no inductive prior for (cf.
Taufeeque et al.'s Sokoban planner, which uses a 2-D image + ConvLSTM). So
prolepsis was never *testable* here — a precondition failure, not "prolepsis is
rhyme-only." Harness: [gridworld_prolepsis.rs](../../../examples/gridworld_prolepsis.rs);
results in [docs/experiments/gridworld-prolepsis/](../gridworld-prolepsis/).

**Cell 2 — means-ends (the working cell).** Moving the *same* planning primitive
(STRIPS / means-ends operator selection) into the **linguistic** modality —
goal-contrastive prompts whose completing token is the goal-correct action
(*"… We want the room to be dark. Turn the lamp"* → `off`) — base `gemma-2-2b`
does it at **ceiling**: the `on_off` cell scores **1.00 / 1.00** (full-vocab
top-1 / forced-choice), and **0.96 / 0.97** on `Llama-3.2-1B`, **including the
discriminating `off`-override** that beats the lexical "turn it on" prior. Across
a 200-item, balanced, multi-family set, goal-conditioning **generalizes** on
forced-choice (≥ 0.84 both models); the stricter top-1 is gated by lexical
realizability (the model prefers `closed`/`opened` over `shut`), which is a
measurement artifact, not a planning failure. Generator:
[means_ends_generator.py](../../../scripts/means_ends_generator.py); scorer:
[means_ends_prolepsis.rs](../../../examples/means_ends_prolepsis.rs); results:
[baseline_gemma2_2b.json](baseline_gemma2_2b.json),
[baseline_llama32_1b.json](baseline_llama32_1b.json).

**Step B prerequisite cleared — features exist.** A full vocab scan of the 2.5M
Gemma CLT (all 26 layers, 2,555,904 features) confirms the `on_off` cell is
**injectable on both sides**: `on` = **L25:78640** (decoder→embedding cosine
0.51, clean), `off` = **L24:92568** (0.36, clean). `open` is the single cleanest
action feature (0.54) but `closed` has none (0.15) — so `open_closed` can't do a
bidirectional intervention, which is *why* `on_off` is the locked cell. Details:
[action_token_inject_candidates.json](action_token_inject_candidates.json).

## What this means

We have a faithful Figure-13 replication target in a new, **explicitly
action-predictive** domain. It is still a single-token completion task
(by design — that is what lets the suppress-plus-inject method apply unchanged),
but the *content* of the commitment is a **goal-conditioned action choice** —
the means-ends operator-selection atom — and we have shown it is genuinely
goal-driven, not collocational. Scope is honest: this is single-operator
selection (the *first* commitment), not multi-step plan search — which is exactly
the right scope for a *prolepsis* study and faithful to how rhyme planning works.

## Classical-planning framing: the STRIPS commitment rule

The sharper statement of the hypothesis: we are testing whether a transformer's
forward pass **instantiates the STRIPS operator-commitment rule** (Fikes &
Nilsson 1971), as stated by Ghallab, Nau & Traverso (*Automated Planning: Theory
and Practice*, 2004, §4.4, p. 76): *"If the current state satisfies all of an
operator's preconditions, STRIPS commits to executing that operator and will not
backtrack over this commitment."* That single sentence maps clause-by-clause
onto three already-designed probes:

- **"if the current state satisfies the preconditions" → *when* it commits.**
  The antecedent is precondition-satisfaction — commitment once state + goal are
  encoded, i.e. the *information-completion* locus. The Initial/Goal
  order-permutation (Step B) tests exactly this: does the planning-site spike
  track where the preconditions become checkable, or sit at a fixed pre-output
  slot?
- **"commits to executing that operator" → *that* it commits.** The operator is
  the action token; the planning-site spike is the commitment event.
- **"will not backtrack over this commitment" → *that it won't revise*.** This is
  the irrevocability / correction sweep (COLM Appendix-G analogue):
  "commitment locked under contradictory injection" is the neural-substrate
  restatement of non-backtracking.

This places transformer action-selection on the classical **commitment-strategy
axis** — at the **STRIPS / eager** end (commit early, don't backtrack) rather
than the **least-commitment / partial-order** end (TWEAK, Chapman 1987; SNLP,
McAllester & Rosenblitt 1991; UCPOP, Penberthy & Weld 1992; Weld 1994), which
was invented precisely to avoid premature commitment. It also yields a
falsifiable failure prediction: STRIPS's non-backtracking commitment is the very
source of its **incompleteness**, witnessed by the **Sussman anomaly** (subgoal
interactions requiring the first operator to be undone). If a transformer commits
STRIPS-style, it should fail *where STRIPS fails* — recasting "LLMs can't plan"
(Kambhampati et al.) as a candidate *mechanistic diagnosis* (eager-commitment
planning, with eager-commitment incompleteness) rather than a flat verdict.

**Two honesty checks.** (i) This is a correspondence to be *earned*. **Step B
result (below):** the redirect spike sits at the **planning site** (the last
content token / decision slot), order-invariant — the *same* localization as all
seven rhyme cells, where (per this codebase's Fig-13 operationalization) the
planning site **is** the trailing position immediately before the word. So it is
**not** "completion dressed as commitment" in any sense that distinguishes it
from the rhyme planning Anthropic reports: the action commitment is read at the
planning site, exactly as rhyme is. The genuinely distinguishing axis is *depth*
— the **commitment-onset layer**, reported below. (ii) Single-operator selection
cannot exhibit the Sussman failure — that is strictly a **multi-operator
follow-on cell** (does the model fail on subgoal-interacting instances exactly as
ground-STRIPS does?).

## Step B — RESULT: planning-site redirect transfers to means-ends

The suppress-plus-inject **position sweep** (`examples/means_ends_sweep`, inject =
the *alternative* action's feature, suppress = the *correct* action's, both at the
swept position across all downstream layers) was run on the controlled 48-item
`on_off` set across **three CLTs**. The redirect peaks **at the planning site**
(the last content token, where the next token is the action) — the canonical
Figure-13 shape, here in the **action domain**.

> **A correction to the earlier draft of this file.** A prior revision labelled the
> spike-at-the-last-token a "condition artifact / not proleptic / readout
> collapse," on the mistaken belief that the planning site must sit several tokens
> *before* the word (à la a newline). Re-reading the candle-mi Fig-13 replications
> (`docs/experiments/figure13-qwen3-cross-size.md`, Finding #1) settled it: in this
> codebase — and in the COLM paper it underpins — **the planning site *is* the
> trailing/last position**; all seven rhyme cells spike there. So the means-ends
> spike-at-the-planning-site is the *same* positive shape, not an artifact. That
> mislabel is retracted.

**Cross-CLT position sweep** (`step_b_sweep_{gemma2_2b_2.5m,gemma2_2b_426k,llama32_1b_524k}.json`):

| Base model (n_layers) | CLT | spike at planning site | ratio (max / mean) | best abs. P(inject) | baseline P(inject) |
|---|---|---:|---:|---:|---|
| Gemma 2 2B (26) | mntss 2.5M | **48/48** | 403× / 35.6× | **0.968** | 0.001–0.564 |
| Llama 3.2 1B (16) | mntss 524K | **48/48** | 250× / 22.9× | **0.958** | 0.002–0.692 |
| Gemma 2 2B (26) | mntss 426K | 32/48 | 55× / 7.1× | 0.842 | 0.001–0.564 |

Two of three CLTs — including the **16-layer Llama** — give a clean 48/48
planning-site replication. The 426K shortfall is **CLT-substrate-specific, not a
planning deficit**: its strong direction (inject `on`) localizes 24/24, but the
weak `off` feature scatters; Llama's `off` is even weaker (cos 0.17) yet still
localizes 24/24, so this is the 426K substrate, not the model or the domain.

**Representative Figure-13 curve** (2.5M, `speaker`, commit-`off` → inject `on`,
strength 50; baseline P(`on`)=0.013): flat at baseline across every earlier
position, a single spike at the planning site —

```
P(on) by steering location:   pos 0–13: 0.013 (flat at baseline)
                              pos 14 (planning site, "speaker"): 0.515  ██████████████████████
```

**Reading the numbers honestly.** The ratios (×7–×400) are far below the rhyme
cells' ×10⁵–10¹² *because* the alternative action has a real baseline
(P≈10⁻³–0.5), not the ~10⁻⁷ of an off-rhyme word — so absolute steered
probability, **0.96–0.97, at the top of the rhyme range, is the fair
cross-domain metric**; ratio penalises a sane baseline.

### Commitment-onset layer (`examples/commitment_onset`)

Where the position sweep finds *where in the sequence* (the planning site), this
finds *at which layer* the committed action is decided — at the planning site,
two ways: **logit-lens** P(committed) by layer (onset = first top-1 layer) and the
**CLT feature-activation** of the per-layer best-encoding feature (onset = first
layer above 0).

| Base model (CLT) | n_layers | logit-lens onset | CLT-act onset | depth (LL) |
|---|---:|---|---|---:|
| Gemma 2 2B (2.5M) | 26 | L21.5 (46/48) | L24 (4/48) | **0.83** |
| Gemma 2 2B (426K) | 26 | L21.5 (46/48) | L22 (18/48) | **0.83** |
| Llama 3.2 1B (524K) | 16 | L13 (43/48) | L12 (24/48) | **0.81** |
| Qwen3-1.7B (20K) | 28 | L25 (46/48) | L16 (41/48) | **0.89** |
| Qwen3-0.6B (20K) | 28 | L22 (48/48) | L25 (36/48) | **0.79** |

**The action commitment forms late and remarkably consistently — 0.79–0.89 of
depth across four base models (0.6B→2B) and three families (Gemma, Llama,
Qwen3).** So: *committed at the planning site (early in sequence), computed late
in depth, at the decision token.*

**Rhyme vs action — same depth.** The one cleanly-comparable rhyme cell (Llama
3.2 1B `-ee` poem, planned word " see", `commitment_onset --prompt`) commits at
**L14/16 = 0.875** — squarely in the action band. So on the same model, rhyme and
action crystallize at the *same* late depth. (Qwen3 base models do **not** greedily
produce a rhyme at the planning site on the figure13 poems — argmax is a
subword/function continuation, `est`/`the` — so no clean natural-rhyme-onset
there; they are *steerable* to a rhyme, that's the figure13 result, but not
naturally planning one. Llama is the clean rhyme cell.)

**CLT-act is a secondary, noisy signal — read logit-lens as primary.** It is
computed over the subset of items whose committed-token feature fires, with a
lenient `>0` threshold and possible contextual bleed (the state clause contains
`on`/`off`). It corroborates logit-lens on Gemma/Llama (±1–2.5 layers) but
**diverges on Qwen3** in *both* directions (1.7B: CLT-act L16 *before* logit-lens
L25 — the tantalizing compute-then-readout shape; 0.6B: CLT-act L25 *after* L22).
We therefore do **not** claim compute-then-readout from CLT-act; the clean test is
a CLT-free per-layer MLP/attention direct-logit-attribution at the planning site
(does an MLP write the action direction in mid-layers before it surfaces?).

**Tooling note.** The mntss Gemma CLTs are mis-flagged `CltSplitJumpReLU` by the
`features/index.json.gz` sidecar heuristic but ship plain-ReLU encoders (no
`threshold` tensor); `load_encoder` now falls back to ReLU when the threshold is
genuinely absent (GemmaScope stays strict), so the encoder/`encode()` path works
for these CLTs.

### Causal mirror — contrastive activation patching (CLT-free)

`examples/contrastive_patch` (no CLT) patches the clean residual into a
token-aligned `bright`/`dark` goal-flip pair (`means_ends_generator.py
--contrastive`, goal-only frame `"We want the {entity} to be {goal}. Turn the
{device}"`) at each `(position, layer)` and measures restoration of the action
logit-diff `logit(on) − logit(off)`. The **causal onset** = the layer at which
patching the **planning-site (last) token alone** restores ≥ 50 % of the
clean-vs-corrupt gap. Pairs are kept when the goal flip flips the on/off
preference (`clean_d > 0 > corrupt_d`).

| Base model (n_layers) | causal onset (patch) | logit-lens onset (readout) | kept |
|---|---:|---:|---:|
| Gemma 2 2B (26) | **0.58** | 0.83 | 7/8 |
| Llama 3.2 1B (16) | **0.66** | 0.81 | 6/8 |
| Qwen3-1.7B (28) | **0.61** | 0.89 | 7/8 |
| Qwen3-0.6B (28) | 0.61 | 0.79 | 2/8 |

**In every model the goal-determined action signal is *causally* present at the
decision token ~3–8 layers (0.15–0.28 of depth) BEFORE the action surfaces as the
top-1 readout.** This is **compute-then-readout**, causal and CLT-free: the
goal→action composition completes mid-stack (~0.58–0.66 depth) and the late
logit-lens onset (~0.79–0.89) is a *lexicalization lag*, not late computation — so
the means-ends action is genuinely *computed* from the goal, not retrieved.
Caveats: Qwen3-0.6B kept only 2/8 pairs (the goal-only preference-flip gate is
strict on the smallest model) — suggestive, not robust; the two onsets use
different thresholds (patch recovery ≥ 0.5 vs first top-1), so the *direction* and
the ~3–8-layer gap are the robust claim, not the exact depths. The mechanism split
(MLP vs attention DLA) is the deliberate next step, not done here.

### Mechanism — MLP-vs-attention DLA (CLT-free)

`examples/action_dla` (no CLT) decomposes the action logit-diff
`logit(on) − logit(off)` at the planning-site token into per-layer `AttnOut`/`MlpOut`
contributions via **frozen-final-norm direct logit attribution**
(`contrib(c) = (rms(c)/rms(R_final)) · logit_diff(project_to_vocab(c))` — additive
by construction; the per-component *ablation* variant is **not** additive under
RMSNorm, which the additivity sanity check caught: 0.15 → 1.00 after the fix). The
**goal-driven** contribution is `contrib_clean − contrib_corrupt` on the same
`bright`/`dark` pairs and keep-gate as (a).

| Base model (n_layers) | DLA onset | (a) causal onset | logit-lens onset | attn total | **MLP total** | kept |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 2 2B (26) | 0.88 | 0.58 | 0.83 | +29.8 | **+68.5** | 7/8 |
| Llama 3.2 1B (16) | 0.75 | 0.66 | 0.81 | +6.7 | **+9.7** | 6/8 |
| Qwen3-1.7B (28) | 0.86 | 0.61 | 0.89 | +4.7 | **+28.2** | 7/8 |
| Qwen3-0.6B (28) | 0.77 | 0.61 | 0.79 | +1.0 | **+5.3** | 2/8 |

**The goal→action signal is written predominantly by the MLPs in every model**
(MLP ≫ attention, 1.4×–6×) — the means-ends action is **composed in the MLPs**, not
attention-copied from a stored associate. That is the *compute* verdict,
mechanistically; attention plays a secondary (goal-routing) role. The onsets form a
pipeline: the goal is **causally present** at the decision token early (~0.6, from
(a)), the **MLPs compose it into the action-logit direction** in the upper-mid stack
(DLA onset ~0.75–0.88), and it **surfaces as top-1** last (logit-lens ~0.79–0.89) —
so the late readout is the *tail of MLP composition*, not late arrival of the goal.
Caveats: additivity = 1.00 (validated); Qwen3-0.6B is thin (2/8 kept) but
MLP-dominance still holds; the three onsets use different thresholds, so the
*ordering* and the *MLP-dominance* are the robust claims, not the exact depths.
**This closes the CLT-free mechanism arc (a) + (b).**

### Depth-axis (ir)revocability — does a later layer overturn the decision?

`examples/decision_trace` (no CLT) logit-lenses every layer's planning-site
residual and tracks the signed action margin `d[L] = logit(correct) −
logit(alternative)`. To avoid the `on ≻ off` frequency baseline (which makes an
absolute zero-crossing meaningless), it uses **peak-vs-final** statistics: the
margin peaks at some layer, the **late erosion** is `peak − final`, and a decision
is **flipped by late layers** when the correct action led at the peak (`peak > 0`)
yet lost at the readout (`final ≤ 0`).

| Base model (n_layers) | peak layer | late erosion (peak−final) | last-layer Δ | failures that are late-flips |
|---|---:|---:|---:|---:|
| Llama 3.2 1B (16) | 12 (0.75) | +5.9 (95%) | −1.7 (68%) | **7/7 (100%)** |
| Gemma 2 2B (26) | 23 (0.88) | +13.9 (100%) | −9.5 (98%) | **3/3 (100%)** |
| Qwen3-1.7B (28) | 24 (0.86) | +7.7 (100%) | −2.0 (93%) | **2/2 (100%)** |
| Qwen3-0.6B (28) | 22 (0.79) | +6.9 (100%) | −1.1 (86%) | **5/5 (100%)** |

**The decision is *strongest at the mid-late peak* (~0.75–0.88 depth) and the later
layers systematically walk it back** (95–100 % of items eroded; the last layer is
the single biggest eroder). And **in 100 % of the model's errors, across all four
models, the goal-correct `off` was winning at the peak and the late layers flipped
it to the lexical prior `on`** — every failure is an `off`-correct item (verified;
not the `on`-bias artifact), so the mid-stack *computed the right operator* and the
final layers *relapsed to the "turn it on" prior*. The readout is therefore a
**worse operator-selector than the mid-stack peak**. So means-ends is
depth-**revocable** — the opposite of STRIPS non-backtracking: there is
"backtracking" in *depth*, it is directional (toward the lexical prior), and that
backtracking *is* the failure mode. **Caveat:** the erosion *magnitude* (esp.
Gemma's +13.9 / −9.5) is largely **final-layer calibration** — Gemma-2 logit
soft-capping and the general final-norm temperature compress the margin — not
deliberate reconsideration; the robust claims are **sign-based** (the late layers
reduce the margin and, in failures, flip it toward the prior).

### Versified means-ends — 79-couplet competence sweep (base models)

Scaling the single versified couplet to **79 couplets × 4 base models**
(`means_ends_generator.py --versified` → `means_ends_prolepsis` →
`scripts/classify_versified.py`, CMUdict rhyme via `nltk`). Each couplet's final
word must satisfy **both** the rhyme (line-1 anchor) and the goal (`W` = the dual
word, `W'` = goal-wrong antonym).

| Base model | goal-lean `P(W)>P(W')` | rhyme | dual-hit (`C==W`) |
|---|---:|---:|---:|
| gemma-2-2b | 99% | 61% | **57%** |
| Qwen3-1.7B | 86% | 43% | **41%** |
| Llama-3.2-1B | 85% | 34% | **32%** |
| Qwen3-0.6B | 84% | 27% | **25%** |

**Goal selection is robust (84–99%); joint rhyme+goal is hard and uneven (25–57%).**
But the aggregate **overstates** planning: for most families `W` was the *default*
goal word and line 1 was anchored to rhyme with it, so rhyme is satisfied
**incidentally**, not planned. The only family where rhyme **forces a non-default
word** (`glow_dim`: rhyme demands `glow`, default goal word is `bright`) is the real
test — `glow` produced **gemma 5/8, Qwen3-1.7B 2/8, Llama 0/8, Qwen3-0.6B 0/8**.
This **inverts the single-prompt impression** (where Qwen3-1.7B emitted `glow` and
gemma `so`): at n=8 gemma is *best* and Qwen mostly fails, the smalls fail entirely
— the one prompt was noise. **Conclusion:** under genuine pressure only gemma plans
the non-default rhyming word more than half the time; the rest default to the plain
goal word and drop the rhyme — weak evidence of rhyme-*planning*, strong evidence of
goal-*selection*. **Design lesson (for a v2):** every family must force a non-default
rhyming word (like `glow_dim`); most of this set does not, which is why the aggregate
looks better than the mechanism warrants.

## Provenance

Commits (on `main`, unpushed): `da1216a` Step 0 · `40171ff` Step A (gridworld
negative + means-ends cell) · `57cbb9b` vocab-scan candidates · `2758a8d` Step B
harness (its message's "condition under revision" framing is superseded by the
RESULT section above). The 3 GB raw vocab scans are gitignored; per-CLT on/off
candidate picks and per-layer-feature maps are committed.
