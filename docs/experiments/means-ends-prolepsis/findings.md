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

**Two honesty checks.** (i) This is a correspondence to be *earned*: if Step B
shows the spike at the stem (shallow lookup), the GNT-antecedent prediction
breaks and it is "completion dressed as commitment." (ii) Single-operator
selection cannot exhibit the Sussman failure — that is strictly a
**multi-operator follow-on cell** (does the model fail on subgoal-interacting
instances exactly as ground-STRIPS does?).

## Where we're heading — Step B

Run the suppress-plus-inject **position sweep** on the `on_off` cell (inject
`on` L25:78640 / suppress `off` L24:92568), looking for the planning-site spike.
The novel lever is **permuting the Initial-state and Goal-state order**
(Initial→Goal vs Goal→Initial), which dissociates *where* the commitment lives:
**goal-bound** (tracks the goal clause), **information-completion** (the second
clause), or **output-adjacent** (a fixed pre-action slot). This is what
adjudicates "genuine proleptic planning" (early, on the goal clause) versus
"shallow lookup at the stem." Both prompt orders already pass Step A, so the
permutation is viable; it is designed and de-risked but **not yet built**. After
that: the irrevocability test (Appendix-G analogue) and a write-up.

## Step B status (in progress — condition revision needed)

The suppress-plus-inject harness exists (`scripts/means_ends_generator.py
--controlled` → `step_b_items.json`; `examples/means_ends_sweep.rs`; inject
`on` = L25:78640 / suppress `off` = L24:92568). The first run found the redirect
spike **only at the final/readout token, flat at every prompt position,
order-invariant**. **This is a condition artifact, not a prolepsis verdict:** our
prompt ends with the action token as the *immediate next token*, so the planning
site and the readout collapse — there are no intervening tokens to "write
toward," unlike Anthropic's poem (whose planning site is the newline *before* the
already-written line, separated from the rhyme word by the line). Anthropic's
Fig 13 is itself a fixed-prompt steering-location sweep, and candle-mi's
`figure13_planning_poems` shares that valid structure — **the flaw is specific to
the means-ends prompt having no span between commit and output.** Fix: redesign
so the action ends a *generated* span (a justification/phrase), then sweep the
pre-span planning site. (TODO: confirm figure13's spike sits at the early
planning-site position.)

## Provenance

Commits (on `main`, unpushed): `da1216a` Step 0 · `40171ff` Step A (gridworld
negative + means-ends cell) · `57cbb9b` vocab-scan candidates. The 3 GB raw
vocab scan is gitignored; the small candidate summary is committed.
