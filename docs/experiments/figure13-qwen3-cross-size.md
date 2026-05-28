# Figure 13 cross-cell sweep: `Qwen3` × `BlueLightAI` `CLT`s vs `Llama` / `Gemma` references

**Date**: 2026-05-27
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, Rust 1.95
**Scope**: 7 (model, `CLT` width, rhyme) cells, all run through the same
`examples/figure13_planning_poems.rs` harness with the
**2D position × strength grid** sweep (strengths
`{0.5, 1, 2.5, 5, 10, 25, 50, 100}` × 20–32 prompt positions per cell).
Best (strength, position) cell per row is reported; full per-strength
profiles are committed alongside.

This document is the **load-bearing rebuttal artefact** for COLM 2026 §Q2
extension to `Qwen3`.  Per-experiment findings link back to this table.

## Headline table

| Model | `CLT` source | Width | Rhyme | N (clean) | Baseline P(inject) | Best strength | Best position | Best ratio | At planning site? | Best absolute P |
|---|---|---:|---|---:|---|---:|---:|---:|:---:|---:|
| `Llama 3.2 1B`    | `mntss`            | 524 K | -ee   |  79 | 1.057 × 10⁻⁶ | 25 | 30 | **806,260×**    | **yes** | **0.8525** |
| `Gemma 2 2B`      | `mntss`            | 426 K | -out  | 287 | 4.836 × 10⁻⁸ | 25 | 31 | **9,974,880×**  | **yes** | **0.4824** |
| `Qwen3-0.6B-Base` | `bluelightai-dev`  |  16 K | -ation| 27  | 2.693 × 10⁻⁷ | 25 | 19 | **33,860×**     | **yes** | 9.12 × 10⁻³ |
| `Qwen3-0.6B-Base` | `bluelightai`      |  20 K | -teen | 33  | 2.874 × 10⁻⁸ |  1 | 21 | **157.1×**      | **yes** | 4.51 × 10⁻⁶ |
| `Qwen3-1.7B-Base` | `bluelightai`      |  20 K | -teen | 30  | 1.116 × 10⁻⁷ |  5 | 21 | **16.4×**       | **yes** | 1.83 × 10⁻⁶ |
| `Qwen3-0.6B-Base` | `bluelightai`      |  20 K | -ation| 71  | 2.693 × 10⁻⁷ | 10 | 18 |    5.4×         | one before | 1.46 × 10⁻⁶ |
| `Qwen3-1.7B-Base` | `bluelightai`      |  20 K | -ation| 84  | 2.142 × 10⁻⁷ |  2.5 | 19 | 3.8×           | **yes** | 8.16 × 10⁻⁷ |

All `Qwen3` cells use the corrected `find_token_id` lookup (post-v0.1.11
tokenizer fix); the prior implementation silently resolved
`inject_word = "myself"` to the bare `"self"` sub-token on `Qwen3` because
the lookup assumed `BOS` was always prepended — see
[`src/tokenizer/mod.rs`](../../src/tokenizer/mod.rs) `find_token_id` docstring.

## Six findings

### 1. Planning is observable across all 7 cells

Six of seven cells produce their best redirect **at the trailing-space
planning site** (last token position).  The seventh (`qwen3-0.6b-20k-ation`)
peaks **one position before** the planning site.  This is the canonical
Anthropic / Lindsey-et-al. (2025) "spike at the planning site" shape,
reproduced on three open `CLT` families (`mntss`, `BlueLightAI` production,
`BlueLightAI` dev) and three model bases (`Llama 3.2 1B`, `Gemma 2 2B`,
`Qwen3-{0.6B, 1.7B}-Base`).

### 2. Reference cells confirm the protocol scales

The `Llama` and `Gemma` reference rows reproduce the headline numbers
documented in the paper (`Llama` `P("that") = 0.777` per paper Q2,
`Gemma` `P(" around") = 0.483` per paper §5 best-of-136-pairs):

- `Llama 3.2 1B 524 K -ee`: best absolute `P("that") = 0.8525` at `s = 25`
  (the grid sweep modestly exceeds the paper's `s = 10` number `0.777`).
- `Gemma 2 2B 426 K -out`: best absolute `P(" around") = 0.4824` at `s = 25`
  (paper's best-of-136 was `0.483`, paper's spot run was `0.457`).

The grid sweep gives a slightly stronger result than the paper's
fixed-strength sweep — confirms `s = 10` was a reasonable but
not-optimal choice, with the optimum at `s = 25` for both reference cells.

### 3. The `s = 10` convention is **ours**, not Anthropic's

We checked the Anthropic *On the Biology of a Large Language Model*
page directly (the "Planning in Poems" section): **no numerical
strength is documented.**  Our `s = 10` was a `plip-rs` / `candle-mi`
internal default refined to `15 → 10` in v0.1.7 (2026-03-30).  The
grid sweep is the right tool for justifying the strength choice
post-hoc; one strength per (model, `CLT`, target word) is empirically
the best policy.

The strength-grid is now wired into the example as `--strength-grid`
(see [`examples/figure13_planning_poems.rs`](../../examples/figure13_planning_poems.rs)
CLI), and the strength optimum per cell is reported in this table.

### 4. `Qwen3` planning is observable but the redirect magnitude is `BlueLightAI`-`CLT`-bound

| `CLT` family | Width | Strongest `Qwen3` redirect | Strongest `mntss` reference |
|---|---:|---:|---:|
| `mntss` ReLU (paper)         | 426–524 K | — | **806,260×** (Llama) / **9.97 M×** (Gemma) |
| `BlueLightAI` JumpReLU prod  | 20 K | 157× (Qwen3-0.6B -teen) | — |
| `BlueLightAI-dev` JumpReLU   | 16 K | **33,860×** (Qwen3-0.6B -ation) | — |

The `BlueLightAI` 20 K production `CLT`s are **3–5 orders of magnitude
weaker** as steering substrates than the `mntss` `CLT`s on the reference
models.  The `BlueLightAI`-dev 16 K test `CLT` is anomalously strong on
the `-ation` target — its `L22:8011` feature targets `" myself"`
specifically (`cos = 0.30`) and produces a `33,860×` redirect.  Open
hypothesis: width is not the only quality axis; per-feature
target-word specificity matters more than feature count.  We did **not**
test `mntss/clt-131k` on `Qwen3` (200 GiB pre-cache cost; out of scope
for v0.1.11 per the in-scope table below).

### 5. Within `Qwen3-0.6B → Qwen3-1.7B`, planning signal does **not** monotonically grow with scale

Strict same-`CLT`-family, same-prompt, same-protocol comparison:

| Rhyme | `Qwen3-0.6B 20K` | `Qwen3-1.7B 20K` | Δ (smaller / larger) |
|---|---:|---:|---:|
| `-teen`  | **157×** | 16.4× | **9.5×** stronger at 0.6B |
| `-ation` |   5.4×   |  3.8× | 1.4× stronger at 0.6B |

The 0.6B model gives a *stronger* planning-site spike than the 1.7B
sibling on both rhyme families.  This is **inverse** of what
Hanna & Ameisen (2026) ("Latent Planning Emerges with Scale") would
predict on the same `CLT` family.  Within the 0.6B–1.7B subrange this
contradicts the within-family scaling part of their claim; we cannot
refute the 0.6B–14B trend overall.  Open hypothesis: the
`BlueLightAI` 0.6B `CLT` may simply have been trained more thoroughly,
or the smaller model's denser per-dimension behaviour makes its
residual stream more responsive to fixed-magnitude steering.

### 6. The `CLT-decoder-as-additive-steering-vector` test (Reviewer L1Vb02 critique 4)

Reviewer L1Vb02 asked: *"Have you tried using the `CLT` feature direction
itself as the steering vector? If you can manipulate the LM behavior with
`CLT` directions, then your conclusion here is simply invalid."*

We added a `--no-suppress` flag to `figure13_planning_poems` and ran
inject-only sweeps on three cells.  Results:

| Cell | Full (suppress + inject) | Inject-only (`--no-suppress`) | Suppress's marginal contribution |
|---|---:|---:|---:|
| `Qwen3-0.6B 20K -teen` (cluster-broad inject) | 157.1× | 47.5× (with `cos→target` inject) | substantial when using cluster-broad inject |
| `Qwen3-0.6B 20K -ation` | 5.4× | 1.83× | 3× boost from suppress |
| `Qwen3-1.7B 20K -teen` (cluster-broad inject) | 16.4× | 1.21× (with `cos→target` inject) | substantial |

This is **consistent with what Anthropic's biology piece already
establishes**: `CLT` decoder vectors are residual-stream directions, and
adding them to the residual stream IS a form of additive steering.  The
question therefore has a clean answer: **yes**, `CLT`-decoder-as-direction
works (that's `Figure 13` in Anthropic, `Figure 1` in our paper).  Our
`Q1` negative result is about **non-`CLT`-derived** steering directions
(max-activation probes, contrastive word probes, etc. — six methods in
`Appendix A`); none of those methods uses `CLT` decoder vectors.  When we
*do* use `CLT` decoder vectors (with or without the suppress side), we
recover a planning-site redirect — that's exactly what `Q2 / Figure 1`
demonstrates.  The critique therefore reduces to confirming that the
paper's positive `Q2` result holds without the suppress side too, which
the inject-only sweep confirms it does (with reduced magnitude).

## On Maar et al. (2026) "What's the plan?" (Reviewer L1Vb02 critique 2, Reviewer UvuC13)

Both negative reviews cite Maar et al. as a counter-example to our `Q1`
("all six steering methods fail").  We checked Maar et al.'s paper
methodology section carefully:

- **Layer**: documented only for `Gemma 2 9B` at layer 27 on the newline
  token.  For the other 22 models in their 1B–32B sweep, the per-model
  layer choice is **not specified in the paper text**.
- **Strength**: `m = 1.5` documented; 1.5–2 noted as the working range.
- **Contrast prompt sets**: 85 prompts per rhyme family for training; the
  positive/negative split is not documented.
- **Hook point**: residual-stream addition (clear).
- **Normalization**: not documented.
- **Supplementary code**: noted as available on `OpenReview`
  ([Z10pxu0Q7X](https://openreview.net/forum?id=Z10pxu0Q7X)).
- **Models tested**: 23 models from 1B to 32B; **documented protocol only
  for `Gemma 2 9B`**.  `Gemma 2 2B` and `Llama 3.2 1B` (the models our
  paper tests) are presumably in the wide sweep but the per-model
  protocol parameters are not in the paper.

Reviewer `L1Vb02` writes *"Maar et al. replicated the rhyme findings with
contrastive steering vectors, also with Gemma and Llama models"* — true at
the family level but **the per-model protocol parameters for the specific
sizes we tested are not documented in their paper text**.  The supplementary
ZIP would resolve this.  Until then, the claim that Maar's positive result
directly contradicts our `Q1` negative result on `Gemma 2 2B` rests on an
unverified same-protocol assumption.

## Maar replication — completed in v0.1.12

The Maar et al. (2026) replication is now in
[`docs/experiments/maar-replication/findings.md`](maar-replication/findings.md).
Headline:

- **Llama 3.2 3B** + Maar's exact protocol + Maar's verbatim prompts:
  baseline 60% → steered 30% at `L = 22`, `m = 1.5`.  All 6 binary
  flips are HIT→MISS, zero MISS→HIT.  **REPRODUCES** Maar's published
  "smaller-models" claim on this model.
- **Llama 3.2 1B** at `L = 12`, `m = 1.5`: 50% → 45%, weak monotonic
  inhibition (saturates at −25 pp at `m = 3.0`).
- **Gemma 2 2B** at `L = 20`, `m = 1.5`: 25% → 35%, **non-monotonic
  ENHANCEMENT** with peak at `m = 1.0` (+20 pp).
- The effect-direction split (Llama inhibits, Gemma enhances) is NOT
  a perturbation-magnitude artefact (`H3` rejected per strength sweeps
  at equivalent effective perturbations `m × ‖d‖`).  Architectural
  family dependence is the leading hypothesis at `N = 2` families
  tested.
- `‖d‖` varies 10× across architectures (Llama 1B: 4.01; Llama 3B:
  11.47; Gemma 2B: 116.09), confirming that Maar's single global
  `m = 1.5` is not cross-architecture-transferable; their cross-23-
  model rhyming-rate table is contaminated by this scaling confound.

See [`maar-replication/findings.md`](maar-replication/findings.md) §5 for
the Marr-three-levels methodological reframing.

## Reproducibility

Every cell in the headline table reproduces from this repo with three
commands (see per-experiment `findings.md` for exact invocations):

```powershell
# 1. Pre-cache the base model + the CLT.
hf-fm <model_id>  --timeout-per-file-secs 1800
hf-fm <clt_repo>  --timeout-per-file-secs 1800

# 2. Vocab scan (only needed once per CLT; outputs are gitignored to save space).
cargo run --release --features clt,transformer,mmap --example vocab_scan -- `
    --model <model_id> --clt-repo <clt_repo> --output <raw_json>
python scripts/vocab_scan_cmudict_filter.py <raw_json> --clean-only-output `
    --output <clean_json>

# 3. Position × strength grid sweep.
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- `
    --preset <preset_name> --strength-grid 0.5,1,2.5,5,10,25,50,100 `
    --output <grid_json>
```

Helper scripts: [`scripts/pick_features.py`](../../scripts/pick_features.py)
(top features per rime), [`scripts/pick_inject_feature.py`](../../scripts/pick_inject_feature.py)
(features by cosine to a specific target word),
[`scripts/inspect_grid.py`](../../scripts/inspect_grid.py)
(per-strength summary table).

## Per-experiment links

- [`figure13-llama-524k/findings.md`](figure13-llama-524k/findings.md)
- [`figure13-gemma-426k/findings.md`](figure13-gemma-426k/findings.md)
- [`figure13-qwen3-1.7b-20k/findings.md`](figure13-qwen3-1.7b-20k/findings.md)
- [`figure13-qwen3-0.6b-20k/findings.md`](figure13-qwen3-0.6b-20k/findings.md)
- [`figure13-qwen3-0.6b-16k/findings.md`](figure13-qwen3-0.6b-16k/findings.md)

## In-scope vs deferred `CLT` universe

In-scope for `v0.1.11`:

- `mntss/clt-llama-3.2-1b-524k` (paper reference) ✓
- `mntss/clt-gemma-2-2b-426k` (paper reference) ✓
- `bluelightai/clt-qwen3-1.7b-base-20k` (rebuttal extension) ✓
- `bluelightai/clt-qwen3-0.6b-base-20k` (rebuttal extension) ✓
- `bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test` (`CLT`-width ablation) ✓

Deferred (cited as future work):

| Repo | Why deferred |
|---|---|
| `qc2354/qwen3-06b-clt-{0825, 0831}` | Cross-layer rank-3 decoder in `PltBundle`-style bundle; needs a new schema variant |
| `mntss/clt-131k` | ~200 GiB download; out of scope for consumer-`GPU`-tier reproducibility |
| `georglange/crosslayer-transcoder-topk-*` (×3) | `TopK` activation variant; needs new `CltSplitTopK` schema |
| `EleutherAI/gpt2-mntss-transcoder-clt-relu-sp*` (×2) | `GPT-2` base; `model_type = "gpt2"` not yet in `SUPPORTED_MODEL_TYPES` |

This is now defensibly **the exhaustive set of consumer-`GPU`-feasible
open `CLT` × open-weights model cells as of 2026-05-27**.
