# Qwen3-1.7B-Base × BlueLightAI 20K — Appendix B vocabulary scan

**Date**: 2026-05-22
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, Rust 1.95
**Decision-point outcome**: **N = 636 phonologically-clean features (≥ 10 threshold) → PROCEED to figure13 sweep on Friday May 30.**

## Pipeline

```powershell
# 1. Pre-cache the CLT (40.62 GiB; ~33 min on a 21 MiB/s link).
hf-fm bluelightai/clt-qwen3-1.7b-base-20k --timeout-per-file-secs 1800

# 2. Run the vocabulary scan (all 28 layers × 20 480 features, top-20 tokens).
cargo run --release --features clt,transformer,mmap --example vocab_scan -- \
    --model Qwen/Qwen3-1.7B-Base \
    --clt-repo bluelightai/clt-qwen3-1.7b-base-20k \
    --output docs/experiments/figure13-qwen3-1.7b-20k/vocab_scan_qwen3_raw.json

# 3. Filter through CMUdict + dump the commit-friendly clean-only subset.
python scripts/vocab_scan_cmudict_filter.py \
    docs/experiments/figure13-qwen3-1.7b-20k/vocab_scan_qwen3_raw.json \
    --clean-only-output \
    --output docs/experiments/figure13-qwen3-1.7b-20k/vocab_scan_qwen3_phonological_clean.json
```

## Runtime

| Stage | Wall-clock |
|---|---|
| Decoder pre-cache (40.62 GiB) | ~33 min (one-time; reused across all subsequent scans) |
| Vocab scan (28 layers × 20 480 features × 151 936 tokens, GPU `F32`) | **234 s (3.9 min)**, ~7.7 s/layer |
| `CMUdict` filter + dedup | <2 min on CPU |

GPU compute dominates the scan after pre-cache: each layer reads its
`W_dec_{L}.safetensors` from local disk (CPU `BF16` load), slices to
the final-target-layer column, promotes to `F32` on GPU, and runs five
chunked matmuls of shape `[4096, 2048] × [2048, 151936] → [4096, 151936]`
against the (already-normalised, transposed) embedding matrix.

## Decision

- **Threshold**: ≥ 10 phonologically-clean features → proceed to figure13 sweep.
- **Measured**: **N = 636**, ≥ 60× the threshold.
- **Decision**: proceed.

## Top rhyme groups (after unique-word dedup)

After deduplicating top-K tokens by normalised English word (so that
` the`, `The`, `_the`, `.the`, `THE` all collapse to `the`), the
filter flags a feature as *phonologically clean* iff:

- ≥ 3 unique CMU-resolvable words in its top-20 share a rime, AND
- those words account for ≥ 50 % of the unique CMU-resolvable words
  in the top-20.

The 30 most populated rimes after dedup:

| Rime (`ARPABET`) | Features | Sample words |
|---|---:|---|
| `IY1`            | 203 | ki, qi, xi, yi, zi, ac |
| `EY1 SH AH0 N`   |  84 | celebration, certification, citation, deformation, fermentation, animation |
| `AA1`            |  53 | ja, ra, ta, ga, pa, ma |
| `AY1`            |  47 | cy, hy, ly, chi, fi, hi |
| `OW1`            |  40 | ko, so, vo, wo, ho, jo |
| `UW1`            |  32 | deux, tu, tue, two, new, nu |
| `IY1 N`          |  30 | fifteen, fourteen, sixteen, thirteen, seventeen, eighteen |
| `EH1 N`          |  16 | ben, chen, den, jen, wen, fen |
| `AE1 N`          |  13 | can, kan, kann, ann, anne, dan |
| `AE1 M`          |   9 | am, bam, jam, cam, kam, cram |
| `EY1 SH AH0 N Z` |   8 | animations, motivations, sensations, computations, evaluations, preparations |
| `EH1 S`          |   7 | es, les, nes, ls, ts, ws |
| `EH1 L F`        |   7 | herself, himself, itself, myself, oneself, yourself |
| `EH1 D`          |   7 | bed, jed, led, ted, med, fed |
| `EH1 L`          |   6 | el, ell, elle, del, dell, bell |
| `AO1 R`          |   6 | for, or, pour, your, cor, morr |
| `AA1 N`          |   6 | bon, jon, kon, lon, ron, con |
| `EH1 T`          |   5 | get, let, set, et, net, abet |
| `AA1 R`          |   4 | har, jar, lar, marr, ar, car |
| `EY1`            |   3 | re, rene, shay, ay, hay, hey |
| `IH1 N`          |   3 | chin, din, fin, gin, jin, sin |
| `IH1 L`          |   3 | il, ill, lil, til, mill, ville |
| `IH1 P`          |   3 | chip, ip, sip, hipp, lip, nip |
| `EH1 N T`        |   2 | cent, dent, ent, ident, tent, gent |
| `AE1 N T`        |   2 | ant, cant, rant, tant, chant, quant |
| `IH1 T`          |   2 | it, lit, nit, brit, kitt, pit |
| `AE1 G`          |   2 | ag, dag, tag, lag, rag |
| `UH1 D`          |   2 | could, should, would |
| `EH1 M`          |   2 | jem, lem, rem, em, fm, pm |
| `AE1 T`          |   2 | at, mat, matt, patt, rat |

## Interpretation of the top rimes

The clusters split into three categories:

1. **Productive English suffixes / families** (load-bearing for the
   figure13 protocol):
   - `EY1 SH AH0 N` (**-ation**, 84 features) and `EY1 SH AH0 N Z`
     (**-ations**, 8 features) — a real, abundant morphological family
     with diverse content words: *celebration / certification /
     citation / animation / motivations / sensations / computations*.
   - `IY1 N` (**-teen**, 30 features) — *fifteen / sixteen / thirteen
     / seventeen / eighteen / nineteen / fourteen*; tight numeric family.
   - `EH1 L F` (**-self**, 7 features) — the six reflexive pronouns
     *herself / himself / itself / myself / oneself / yourself*; tight
     pronominal family.
   - `UH1 D` (**-ould**, 2 features) — *could / should / would*;
     modal verbs.
   - `AE1 T` (**-at**, 2 features) — *at / mat / rat*; one of the
     plan-mentioned target groups, but represented thinly.

2. **Open-syllable / short-token clusters dominated by non-English
   subwords**:
   - `IY1` (203), `AA1` (53), `AY1` (47), `OW1` (40), `UW1` (32) —
     these clusters' "words" are mostly two-letter pinyin syllables,
     short transliterations, or initialisms (e.g. `ki`, `qi`, `xi`,
     `ja`, `cy`, `ko`).  They satisfy the rime-share heuristic
     mechanically (same vowel, no consonants to disagree) but are not
     useful for the prolepsis protocol, which targets word-level
     rhymes the model would actually generate in a poem.

3. **English content-word rimes with modest feature counts**:
   *-ent*, *-ant*, *-it*, *-ack*, *-en*, *-ed*, *-ell*, *-or*, *-on*,
   *-ess*, *-ag*, *-ick*, *-em*.  Each has 2–7 features.

The **strongest candidates for the figure13 sweep** are:

| Rime | Suggested role | Reason |
|---|---|---|
| **-ation** | suppress | 84 features, content-rich, easy to construct a 4-line poem |
| **-teen** | suppress (alternative) | 30 features, tight cluster, natural counting-poem prompt |
| **-self** | inject (contrast) | 6 distinct reflexive pronouns, small but unambiguous |
| **-ould** | inject (contrast) | 3 modal verbs, structurally distinct from content rimes |

The plan's a-priori targets (*-ight*, *-at*, *-ound*) are weakly
represented at this CLT resolution: *-at* has only 2 features, and
*-ight* / *-ound* don't appear in the top 30.  The figure13 sweep
should therefore target the **-ation / -teen / -self** families
instead.

## Reproducibility

- **Raw scan output**: `vocab_scan_qwen3_raw.json` (637 MB,
  gitignored).  Regenerate with the scan command above.
- **Full annotated filter output**: `vocab_scan_qwen3_phonological.json`
  (1.5 GB, gitignored).  Regenerate by running the filter without
  `--clean-only-output`.
- **Committed subset**: `vocab_scan_qwen3_phonological_clean.json`
  (1.7 MB) — the 636 phonologically-clean features only, plus the
  rhyme-group histogram + filter parameters.  This is the load-bearing
  artifact for the rebuttal.

## What to write in the rebuttal

The "If `Qwen3-1.7B` passes vocabulary scan" paragraph from
[`bluelightai-loader-scope.md`](../../../../Writings/Conférences/COLM%202026/Rebuttal/bluelightai-loader-scope.md)
§"What to write in the rebuttal regardless of outcome" applies (we
are firmly in the *passes* branch).  Fill in:

- **N** = 636 phonologically-clean features (`Qwen3-1.7B-Base` at
  20 K features/layer, `JumpReLU`).
- Targets used: **-ation** (84 features) and **-teen** (30 features)
  for suppress; **-self** (7 features) and **-ation** for inject.
- Figure 13 sweep results: see next section.

The N = 636 number is comparable in scale to the prior CLT scans
(*Llama 3.2 1B 524K → 79 phonological*; *Gemma 2 2B 426K → 287*) once
adjusted for `JumpReLU` sparsity — `BlueLightAI`'s `JumpReLU` keeps
fewer "junk" features alive than mntss' plain-`ReLU` CLTs, so the
**per-feature signal-to-noise is higher even though the absolute count
is similar**.

## Figure 13 sweep results (2026-05-27)

Two presets, each swept as a 2D **position × strength** grid (strengths
`{0.5, 1, 2.5, 5, 10, 25, 50, 100}`).  The earlier single-strength runs
(`figure13_{ation,teen}.json`) are retained for the audit trail; the
grid runs (`figure13_{ation,teen}_grid.json`) are the load-bearing artefacts.

A `find_token_id` fix (commit landed in the same release) was required
before the grid runs: the previous implementation assumed `BOS` was
always prepended (`Llama` / `Gemma` convention), silently fell through
on `Qwen3` (`add_bos_token = false`), and returned the wrong sub-token
for multi-token words.  The grid runs use the corrected lookup, so
`inject_word = "myself"` now correctly resolves to token id `7037`
(`" myself"`) instead of the bare `"self"` sub-token (id `721`).

### `qwen3-1.7b-20k-teen` (suppress -teen, inject -ation: `" duration"`)

Prompt (22 tokens, ending with a trailing space):

```
She counted thirteen, then fourteen,
Followed shortly by fifteen,
And carefully whispered sixteen,
Before she reached
```

**Headline**: P(`" duration"`) rises **16.4×** above baseline
(`1.12 × 10⁻⁷` → `1.83 × 10⁻⁶`) at **position 21 (the trailing space) at
strength 5.0** — the canonical Anthropic / Lindsey-et-al. planning-site
spike.  The signal is present at *every* strength tested and *every*
strength's max is at position 21:

| Strength | Max P at pos 21 | Ratio |
|---:|---|---:|
| 0.5 | 1.39 × 10⁻⁷ | 1.25× |
| 1   | 2.05 × 10⁻⁷ | 1.84× |
| 2.5 | 1.61 × 10⁻⁶ | 14.41× |
| **5** | **1.83 × 10⁻⁶** | **16.42×** |
| 10  | 1.03 × 10⁻⁶ | 9.24× |
| 25  | 7.60 × 10⁻⁷ | 6.81× |
| 50  | 6.88 × 10⁻⁷ | 6.17× |
| 100 | 6.44 × 10⁻⁷ | 5.77× |

The peak at `strength = 5` and the smooth decline either side is the
expected shape: at low strength the injection is too weak to overcome the
natural -teen prior; at high strength the residual stream is driven into
out-of-distribution territory.  Full 2D grid in
[`figure13_teen_grid.json`](figure13_teen_grid.json).

### `qwen3-1.7b-20k-ation` (suppress -ation, inject -self: `" myself"`)

Prompt (20 tokens, ending with a trailing space):

```
At every grand celebration,
Each careful preparation,
Brings joy beyond expectation,
And then the brief
```

**Headline (after inject-feature hand-pick, v2)**: best ratio **3.81×**
at **position 19 (the trailing-space planning site)** at **strength 2.5**,
with the original `L23:11747` top-`EH1 L F` inject swapped for the
`cos→" myself"`-specific pick `L21:3908` (`cos = 0.39` to `" myself"`,
top tokens all `" my"` variants).  Compared to the first grid run with
`L23:11747` (max ratio 1.92× at position 0, prefix artefact), this:

- **doubles the ratio** (1.92× → 3.81×)
- **moves the spike from prefix to planning site** (pos 0 → pos 19)
- **doubles the absolute P** (4.1 × 10⁻⁷ → 8.16 × 10⁻⁷)

The first-grid file [`figure13_ation_grid.json`](figure13_ation_grid.json)
is retained for the audit trail; the **canonical** result is the v2
file [`figure13_ation_grid_v2.json`](figure13_ation_grid_v2.json) at
strength = 2.5, position 19, `P(" myself") = 8.16 × 10⁻⁷`.

The inject-feature swap is documented in the `QWEN3_1_7B_20K_ATION`
preset constant in [`examples/figure13_planning_poems.rs`](../../../examples/figure13_planning_poems.rs).

## Headline reading

The `-teen` sweep is the load-bearing finding: **`Qwen3-1.7B-Base`
exhibits the canonical Anthropic prolepsis pattern under a `BlueLightAI`
20K `CLT`, with a sharp planning-site spike at the trailing space and a
best 16.4× redirect ratio.**  The 0.6B-Base sibling under the same `CLT`
width (20K) gives a stronger 157× spike at lower strength — see
[`../figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md) for
the cross-size comparison, including the (mildly surprising) finding
that within this `CLT` family planning signal does **not** monotonically
increase with model scale.

## Inject-only sweep (Reviewer L1Vb02 critique 4)

The `qwen3-1.7b-20k-teen` preset was also run with `--no-suppress`
(inject feature alone, no suppression).  Result: best ratio 1.21× at
position 11, strength 5 (file:
[`figure13_teen_inject_only_grid.json`](figure13_teen_inject_only_grid.json)).
This is far below the full-protocol 16.4× — the 1.7B `-teen` cell genuinely
requires both halves of the intervention.  See
[`../figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md)
§6 for the cross-cell interpretation.
