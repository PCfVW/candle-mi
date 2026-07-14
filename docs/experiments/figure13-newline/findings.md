# Newline experiments — findings (Exp 1 census · reconciliation · Exp 2 steering)

**Date**: 2026-07-13 – 07-14
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, candle 0.11
**Spec**: `BlackboxNLP 2026/Figure-13/docs/newline-experiments-spec.md` (amended 2026-07-13)
**Paper**: BlackboxNLP 2026 Reproducibility Challenge, `Figure-13/main.tex`

Experiment 1 asks: at the newline positions of a Figure-13 poem, do any CLT
features carry anticipatory content about the upcoming rhyme — or is the
newline free of plan content (interpretation a, improvisation)?

Two-stage pipeline: [`figure13_newline_census`](../../../examples/figure13_newline_census.rs)
(forward + encode, **all active features per position**, activation + c3) →
[`newline_census_classify.py`](../../../scripts/newline_census_classify.py)
(c1/c2 via CMUdict + vocab-scan join; registered `plan_like` decision).

---

## 0. Machinery reconciliation (do this before trusting any null)

The amended spec requires the census encode path to reproduce plip-rs
detection-V2's documented `(feature, position, activation)` triples before any
null is publishable. It does **not** — and finding out why resolved a latent
hook mismatch in the reference implementation.

The one free variable is *which residual the CLT encoder reads*. Two diagnostics
settle it:

**[`clt_hook_reconcile`](../../../examples/clt_hook_reconcile.rs)** — encode the
three documented Gemma features under all three hook points at the planning site:

| feature | plip-rs (documented) | candle-mi `ResidPre` | `ResidMid` | `ResidPost` |
|---|---|---|---|---|
| out `L25:9385` | 0.247 | 0.000 | 0.000 | **0.2323** |
| go `L25:4505` | 0.983 | 0.000 | 0.000 | **0.9808** |
| ou `L25:5927` | 0.359 | 0.000 | 0.000 | **0.3047** |

plip-rs's numbers reproduce **only under `ResidPost`** (the layer output,
post-MLP); `ResidMid` (post-attention, pre-MLP) gives exactly 0.000.

**[`clt_reconstruction_check`](../../../examples/clt_reconstruction_check.rs)** —
decide which residual the CLT was *trained* to read by reconstructing the actual
MLP output, `recon(T) = Σ_{L≤T} encode(resid(L))·decoder(L→T)`:

| target layer | `ResidMid` cosine / rel-L2 | `ResidPost` cosine / rel-L2 |
|---|---|---|
| 8  | **0.775 / 0.64** | 0.569 / 1.17 |
| 16 | **0.810 / 0.59** | 0.688 / 0.96 |
| 25 | **0.945 / 0.33** | 0.451 / 2.26 |

`ResidMid` reconstructs the MLP output at every layer (cosine 0.945 at layer 25);
`ResidPost` fails (rel-L2 2.26 — worse than predicting zero).

**Conclusion.** The mntss CLT encoder reads **`ResidMid`** (the MLP input), which
is what candle-mi's census uses and the circuit-tracer convention the spec cites.
plip-rs's *detection-V2* read `ResidPost` — the wrong encoder input — because
`get_activations`→`forward_with_cache` caches the layer output. Its documented
0.247/0.983/0.359 "positives" are **artifacts of the wrong hook**. (plip-rs's
*main* planning results survive: the decoder/injection side correctly uses
`ResidPost`, so P(around)=0.483 and P(that)=0.853 reproduced; only the
exploratory encoder-read was wrong.) The spec's "0.247 positive control" is thus
not recoverable by design; under the correct hook those features read 0.000, and
the census null below is valid — not a machinery failure.

This is itself a reproducibility finding: a latent encoder-hook mismatch in the
reference detection code.

---

## 1. Census results (all 7 cells, un-truncated)

`plan_like` = active at the position **and** (c2.ii: decoder-top-20 contains ≥2
words of the natural rhyme group, **or** c3: decoder cosine to the natural target
≥ 0.30). All active features per (position, layer) are classified — not top-K —
so a weak rhyme feature cannot be truncated below stronger generic features.

| Cell | newline (plan-like / active) | control | final (positive ctrl) |
|---|---|---|---|
| Gemma 2 2B × mntss 426K (-out) | 0 / 709 | 2 / 719 | 0 / 269 |
| Llama 3.2 1B × mntss 524K (-ee) | 6 / 591 | 11 / 535 | 6 / 216 |
| Qwen3-0.6B × BLA-dev 16K (-ation) | 128 / 34065 | 96 / 23476 | 28 / 8735 |
| Qwen3-0.6B × BLA 20K (-teen) | 11 / 42965 | 9 / 30010 | 7 / 10633 |
| Qwen3-1.7B × BLA 20K (-teen) | 66 / 132868 | 36 / 86626 | 18 / 28282 |
| Qwen3-0.6B × BLA 20K (-ation) | 233 / 42751 | 157 / 30306 | 60 / 11662 |
| Qwen3-1.7B × BLA 20K (-ation) | 1184 / 156215 | 652 / 94162 | 322 / 46669 |

The **absolute** newline counts diverge wildly by CLT type, but that is a
**density artifact**: mntss CLTs are plain-ReLU and sparse (hundreds of active
features/position); BlueLightAI CLTs are JumpReLU and dense (tens of *thousands*).
More active features → more features that *incidentally* decode to a common rime.
The un-truncation mattered only for the dense Qwen cells (e.g. 16K -ation:
34065 features vs 2069 under the old top-30 cap); it left the sparse Gemma/Llama
cells essentially unchanged.

## 2. The right metric: plan-like *rate* (enrichment vs base rate)

The registered question is whether the newline carries plan content *above base
rate* — so the diagnostic is the plan-like **fraction**, and whether it is
*enriched* at the newline relative to mid-line controls and the final token.

| Cell | newline rate | control rate | final rate | newline enriched? |
|---|---|---|---|---|
| Gemma 426K (-out) | 0.00% | 0.28% | 0.00% | no |
| Llama 524K (-ee) | 1.02% | 2.06% | 2.78% | no (lowest) |
| Qwen3-0.6B 16K (-ation) | 0.38% | 0.41% | 0.32% | no |
| Qwen3-0.6B 20K (-teen) | 0.026% | 0.030% | 0.066% | no |
| Qwen3-1.7B 20K (-teen) | 0.050% | 0.042% | 0.064% | no |
| Qwen3-0.6B 20K (-ation) | 0.545% | 0.518% | 0.514% | no (flat) |
| Qwen3-1.7B 20K (-ation) | 0.758% | 0.692% | 0.690% | no (flat) |

**In no cell does the newline plan-like rate exceed both its base-rate controls
and the final position.** It is flat or lowest at the newline everywhere. Whatever
rhyme-decoding features fire, they fire at a roughly constant background rate
across positions — they are not concentrated at the newline.

## 3. Mapping to the registered predictions

- **Interpretation (a), improvisation — supported across all 7 cells** when
  "plan-like at the newline" is read as *enrichment* (the intended sense): no
  cell shows newline-specific concentration of anticipatory features.
- The **absolute** presence of some plan-like features at Llama/Qwen newlines
  (the literal ">0" reading of the registered criterion) is a base-rate effect,
  not evidence of a resident newline plan — it is matched at the controls.
- **Gemma is the cleanest case and a "decoder-only regime"** (new row in the
  amended registered-predictions table): its Figure-13 rhyme features are
  decoder-defined write-directions that are encoder-silent everywhere under the
  correct `ResidMid` hook (§0), so both the newline and the positive control
  read 0. The "suppress + inject" protocol on this cell is bidirectional
  decoder-direction steering, decoupled from feature activations — which
  mechanically explains its emission-adjacent effective site.

## 4. Caveats

- **c2.ii is loose in dense codes.** With 10⁴–10⁵ active JumpReLU features, many
  incidentally carry ≥2 rime-mates in their decoder-top-20. The rate analysis
  (§2) controls for this by comparing against base-rate positions; the absolute
  counts (§1) should not be read as "plan features found".
- **c1 is null for Gemma/Llama** (no `phonological_clean` vocab-scan on disk for
  those cells); it does not enter the `plan_like` decision.
- **Single prompt per cell.** The census is one forward per cell, as specified;
  the enrichment comparison is within-prompt (newline vs control vs final).

## 5. Reproduce

Set `$env:HF_TOKEN` first (the login-cache file alone is insufficient for gated
models via `from_pretrained`):

```powershell
$env:HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()

# Reconciliation (§0)
cargo run --release --features clt,transformer,mmap --example clt_hook_reconcile
cargo run --release --features clt,transformer,mmap --example clt_reconstruction_check

# Census + classify, one cell (§1)
cargo run --release --features clt,transformer,mmap --example figure13_newline_census -- `
    --preset gemma2-2b-426k --output docs/experiments/figure13-newline/census_gemma2-2b-426k.json
python scripts/newline_census_classify.py docs/experiments/figure13-newline/census_gemma2-2b-426k.json `
    --raw-scan docs/experiments/means-ends-prolepsis/vocab_scan_gemma2_2b_426k_raw.json --inject-word around
```

---

# Experiment 2 — full-line steering at the newline (causal, composition horizon)

**The faithful Figure-13 analogue.** Unlike Exp 1 / `figure13_planning_poems`
(prompts truncated so the rhyme word is the *next token* — no line to compose),
Exp 2 truncates the prompt **after the line-3 newline**, so the model must
**compose line 4** and choose its rhyme word many tokens downstream. Steering is
applied at the final prompt token — the line-3 newline — which is thus both "the
newline" and "the last token" by construction. This is Anthropic's geometry:
with a line between the steering site and the rhyme, a null is evidence the model
is *below the planning floor* (improvises at emission), not that it lacks a
horizon. Harness: [`figure13_newline_steering.rs`](../../../examples/figure13_newline_steering.rs)
(m2/m3/m4) + [`newline_steering_classify.py`](../../../scripts/newline_steering_classify.py)
(m1). candle-mi is KV-cache-free, so the newline hook is re-applied at every
generation step (`route: "recompute-per-step"`).

## 6. m4 — position sweep of `P(inject)` (the Figure-13 curve)

At each cell's Table-2 best strength (2.5M at 10; k = 20 sampled lines):

| Cell | s | `P(inject)` at **newline** | best `P(inject)` (at emission) |
|---|---|---|---|
| Gemma 2 2B × mntss **426K** (-out→around) | 25 | 9.3 × 10⁻⁷ | **0.810** @ pos 30 (" with") |
| Gemma 2 2B × mntss **2.5M** *(word-level)* (-out→can) | 10 | 5.5 × 10⁻⁷ | **0.579** @ pos 30 (" with") |
| Llama 3.2 1B × mntss 524K (-ee→that) | 25 | 9.4 × 10⁻⁴ | **0.998** @ pos 28 (" shining") |
| Qwen3-0.6B × BLA-dev 16K (-ation→myself) | 25 | 2.8 × 10⁻⁶ | 0.009 @ pos 20 |
| Qwen3-0.6B × BLA 20K (-teen→duration), control | 1 | 1.5 × 10⁻⁷ | 0.000 @ pos 22 |

**In every cell the curve is flat at baseline everywhere except an
emission-adjacent spike; the newline is inert (at baseline level).** The
Figure-13 *shape* (flat + single spike) replicates, but the spike is at
**emission**, not the newline — the opposite of Claude 3.5 Haiku. Decisively,
the **word-level 2.5M CLT** — whose features are the closest analogue to
Anthropic's "ordinary features representing that word" — shows the **same**
inert newline (5.5 × 10⁻⁷) and emission-only spike (0.579). So planning is
absent even at the resolution where Anthropic found planned-word features.

## 7. m1 — does steering redirect the *composed line's* rhyme?

Final-word CMUdict-rime group of the k = 20 sampled lines, inject-group fraction
with exact Clopper-Pearson 95% CIs, baseline vs suppress+inject:

| Cell | baseline inject-group | suppress+inject inject-group | redirect? |
|---|---|---|---|
| Gemma 426K | 0% [0, 17] | 10% [1, 32] | **no** (CIs overlap) |
| Gemma 2.5M (word-level) | 0% [0, 17] | 0% [0, 17] | **no** |
| Llama 524K | 0% [0, 17] | 0% [0, 17] | **no** |
| Qwen 16K | 0% [0, 17] | 0% [0, 17] | **no** |
| Qwen 20K-teen (control) | 0% [0, 17] | 0% [0, 17] | **no** |

**No cell moves the inject-group fraction beyond CI overlap** — the registered
criterion for *improvisation supported*, uniformly.

## 8. m2 — surface leakage, not replanning

The greedy lines expose the mechanism. Injecting at the newline changes the line
*content* — the inject word surfaces, almost always at the **start** of the
line — but the **rhyme ending stays in the natural group or drifts to "other"**:

| Cell | baseline greedy line | suppress+inject greedy line |
|---|---|---|
| Gemma 426K | "Her heart was filled with **doubt**." (-out) | "**Around** the town, she went **about**." (-out) |
| Gemma 2.5M | "Her heart was filled with **doubt**." (-out) | "**Can't** find her way back to the **house**." (other) |
| Llama 524K | "And the sun was shining **bright**." (other) | "**That's** where the birds were singing in the **tree**." (-ee) |

The injected feature leaks its token into the composition (forward-planning-*like*
surface behaviour), but the model never installs a plan that redirects the
**rhyme**. Only m2+m3+m4 together — impossible without the composition horizon —
separate surface leakage from genuine planning.

## 9. Conclusion — small open models sit below the planning floor

Across **3 model families, 2 CLT pipelines, and both group-level and word-level
CLT granularities**, the composition-horizon Figure-13 test gives one answer:
- **m4**: the newline is causally inert; the only spike is emission-adjacent.
- **m1**: no CI-separated redirect of the composed line's rhyme.
- **m2**: newline injection produces surface leakage, not rhyme replanning.

The Figure-13 *signature* is reproducible at 0.6B–2B, but the planning *site*
is emission, not the newline — even at word-level CLT resolution. These models,
which are the complete open-CLT set, **define the current floor: they improvise
the rhyme at emission and do not plan it at the newline.** Anthropic's
newline-localized planning is, on this evidence, a capability above this scale.

Open framing question (for the paper): the newline injection could not redirect
the rhyme (m1), i.e. the emission-time commitment resisted an upstream nudge —
a *late-committing* echo of prolepsis rather than newline planning. Whether
"late but hard-to-budge" is improvisation or a distinct regime is the natural
next question.

## 10. Reproduce (Exp 2)

```powershell
$env:HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
cargo run --release --features clt,transformer,mmap --example figure13_newline_steering -- `
    --preset gemma2-2b-426k --strength 25 --k-samples 20 `
    --output docs/experiments/figure13-newline/fullline_gemma2-2b-426k.json
python scripts/newline_steering_classify.py docs/experiments/figure13-newline/fullline_gemma2-2b-426k.json
```

## 11. Status

- **Experiment 1.5** (bridge) — not triggered: Exp 1 surfaced no newline-enriched
  plan-like features, and Exp 2 confirms no newline redirect, so there is nothing
  to suppress/inject at a census-identified newline site.
- **Committed artifacts**: `findings.md`, all `fullline_*.json`, and the small
  mntss `census_gemma2-2b-426k.json` / `census_llama3.2-1b-524k.json`. The dense
  BlueLightAI (Qwen) `census_qwen3-*.json` are 53–237 MB each and are
  `.gitignore`d — kept locally, regenerable via `figure13_newline_census`.
- Copy `fullline_*.json` (and, from disk, the Qwen `census_*.json`) into
  `Figure-13/data/`; the m4 arrays drive the paper's position-sweep-with-horizon
  figure.
