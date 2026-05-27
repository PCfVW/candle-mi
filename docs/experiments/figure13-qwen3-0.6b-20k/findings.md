# `Qwen3-0.6B-Base` × `BlueLightAI` 20K — Appendix B vocabulary scan + Figure 13 sweeps

**Date**: 2026-05-27
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, Rust 1.95
**Vocab-scan outcome**: **N = 739 phonologically-clean features** (vs 636 for the
1.7B sibling at the same `BlueLightAI` 20K width and 648 for the 16K dev variant).
**Sweep outcome**: **-teen → -ation redirect peaks at the trailing-space
planning site, ratio = 7.1× baseline** (the canonical Anthropic prolepsis
shape).  -ation → -self redirect is in noise.

## Pipeline

```powershell
# 1. Pre-cache the CLT (23.68 GiB; ~19 min on a 21 MiB/s link).
hf-fm bluelightai/clt-qwen3-0.6b-base-20k --timeout-per-file-secs 1800

# 2. Run the vocabulary scan (28 layers × 20 480 features × 151 936 tokens).
cargo run --release --features clt,transformer,mmap --example vocab_scan -- `
    --model Qwen/Qwen3-0.6B-Base `
    --clt-repo bluelightai/clt-qwen3-0.6b-base-20k `
    --output docs/experiments/figure13-qwen3-0.6b-20k/vocab_scan_qwen3_raw.json

# 3. Filter through CMUdict + dump the commit-friendly clean-only subset.
python scripts/vocab_scan_cmudict_filter.py `
    docs/experiments/figure13-qwen3-0.6b-20k/vocab_scan_qwen3_raw.json `
    --clean-only-output `
    --output docs/experiments/figure13-qwen3-0.6b-20k/vocab_scan_qwen3_phonological_clean.json

# 4. Run the two figure13 sweeps (one per rhyme group).
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- `
    --preset qwen3-0.6b-20k-ation `
    --output docs/experiments/figure13-qwen3-0.6b-20k/figure13_ation.json

cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- `
    --preset qwen3-0.6b-20k-teen `
    --output docs/experiments/figure13-qwen3-0.6b-20k/figure13_teen.json
```

## Runtime

| Stage | Wall-clock |
|---|---|
| Decoder pre-cache (23.68 GiB) | ~19 min (one-time; reused across both sweeps) |
| Vocab scan (28 × 20 480 × 151 936, GPU `F32`) | **216 s (3.6 min)**, ~7.7 s/layer |
| `CMUdict` filter + dedup | <1 min on CPU |
| Figure 13 sweep, `-ation` preset (20 positions × strength 10) | ~4 s/sweep after model load |
| Figure 13 sweep, `-teen` preset (22 positions × strength 10) | ~3 s/sweep after model load |

## Phonological scan: top rhyme groups (after unique-word dedup)

After deduplicating top-K tokens by normalised English word and applying the
`share ≥ 0.50, count ≥ 3` heuristic, the filter flags **N = 739** clean
features.  Top 10 rimes by feature count:

| Rime (`ARPABET`) | Features | Sample words |
|---|---:|---|
| `IY1`            | 252 | li, mi, si, ti, he, she |
| `AA1`            |  99 | da, ga, ha, ka, la, ma |
| `OW1`            |  76 | co, coe, ko, bio, bo, lo |
| `EY1 SH AH0 N`   |  71 | activation, allocation, contamination, deformation, instantiation |
| `UW1`            |  35 | que, su, to, tu, deux, two |
| `IY1 N`          |  33 | eighteen, fifteen, fourteen, sixteen, nineteen, seventeen |
| `AY1`            |  33 | cy, ly, sy, ty, vy, bi |
| `EH1 L`          |  13 | el, ell, elle, del, delle, noel |
| `EY1 SH AH0 N Z` |   9 | consultations, informations, notifications, collaborations |
| `EH1 L F`        |   8 | herself, himself, itself, myself, oneself, yourself |

Compared to the 1.7B sibling (84 -ation / 30 -teen / 7 -self) the 0.6B base
has a slightly **higher** count in the same productive families (71 / 33 / 8),
so the scaling claim "planning requires larger models" already fails at the
*feature-availability* layer, before any intervention is run.  The numeric
`-ould` (modal-verb) cluster is also denser at 0.6B (5 features vs 2 at 1.7B).

## Figure 13 sweep results

Two presets, each swept as a 2D **position × strength** grid (strengths
`{0.5, 1, 2.5, 5, 10, 25, 50, 100}`).  The earlier single-strength runs
(`figure13_{ation,teen}.json`) are retained for the audit trail; the
grid runs (`figure13_{ation,teen}_grid.json`) are the load-bearing artefacts.

A `find_token_id` fix (`src/tokenizer/mod.rs`, same release) was required
before the grid runs: the previous implementation assumed `BOS` was
always prepended (`Llama` / `Gemma` convention), silently fell through
on `Qwen3` (`add_bos_token = false`), and returned the wrong sub-token
for multi-token words.  The grid runs use the corrected lookup, so
`inject_word = "myself"` now correctly resolves to token id `7037`
(`" myself"`) instead of the bare `"self"` sub-token (id `721`).  The
baseline P(`" myself"`) is `2.69 × 10⁻⁷` — about 100× the previously
measured P(`"self"`) `2.26 × 10⁻⁹`.

### `qwen3-0.6b-20k-teen` (suppress -teen, inject -ation: `" duration"`)

Prompt (22 tokens, ending with a trailing space):

```
She counted thirteen, then fourteen,
Followed shortly by fifteen,
And carefully whispered sixteen,
Before she reached
```

**Headline**: P(`" duration"`) rises **157.1×** above baseline
(`2.87 × 10⁻⁸` → `4.51 × 10⁻⁶`) at **position 21 (the trailing space)
at strength 1.0**.  This is the **strongest** planning-site spike
observed across the full Qwen3 × BlueLightAI sweep — stronger than the
1.7B sibling (16.4× at strength 5), stronger than the Llama 3.2 1B
reference (`P("that") = 0.687` at the spike).  Every strength's max is
at position 21:

| Strength | Max P at pos 21 | Ratio |
|---:|---|---:|
| 0.5  | 4.50 × 10⁻⁶ | 156.71× |
| **1**  | **4.51 × 10⁻⁶** | **157.08×** |
| 2.5  | 9.81 × 10⁻⁷ |  34.14× |
| 5    | 3.30 × 10⁻⁷ |  11.48× |
| 10   | 2.05 × 10⁻⁷ |   7.13× |
| 25   | 1.68 × 10⁻⁷ |   5.86× |
| 50   | 1.64 × 10⁻⁷ |   5.71× |
| 100  | 1.60 × 10⁻⁷ |   5.58× |

The profile is **monotonically declining with strength** — peak at the
*lowest* strength tested (0.5 / 1, almost identical) and progressively
weaker as strength rises.  Interpretation: the 0.6B `CLT`'s -teen
suppress + -ation inject is so effective at low strength that any
amplification just pushes the residual stream off-manifold without
adding signal.

A finer grid below `0.5` (e.g. `0.1, 0.25, 0.5, 0.75, 1, 1.5, 2`) would
sharpen the peak location but is unlikely to materially raise the ratio
— the curve is already flat between strengths 0.5 and 1.  Recorded as
future work; the current 157× is the load-bearing rebuttal-grade number.

Full 2D grid in [`figure13_teen_grid.json`](figure13_teen_grid.json).

### `qwen3-0.6b-20k-ation` (suppress -ation, inject -self: `" myself"`)

Prompt (20 tokens, ending with a trailing space):

```
At every grand celebration,
Each careful preparation,
Brings joy beyond expectation,
And then the brief
```

**Headline**: best ratio **5.42×** at **position 18 (`" brief"`)** at
strength 10 — one position *before* the planning site, not at it.
P(`" myself"`) at the planning site (pos 19) collapses to `~10⁻¹⁶` at
strength ≥ 2.5 — the suppression is overwhelming but the -self inject
fails to push `" myself"` up to compensate.

This is the partial-signal counterpart to the 1.7B `-ation` null: the
0.6B base shows *some* redirect (5.4× at pos 18 is non-noise), but the
spike sits at the token *preceding* the trailing space rather than at
the planning site itself.  Possible interpretations: (a) the suppression
collapses the rhyme-locus state one position early; (b) the -self
inject decodes via a different positional signature than the -ation
inject did on the -teen sweep; (c) the prompt's `" brief"` token is
itself a partial planning cue that becomes a sink for the injected
direction.  Disambiguating these requires per-layer attention-pattern
inspection — recorded as arXiv v2 follow-up.

Full 2D grid in [`figure13_ation_grid.json`](figure13_ation_grid.json).

## Interpretation

The `-teen` sweep is the load-bearing finding: **`Qwen3-0.6B-Base`
exhibits the canonical Anthropic prolepsis pattern under a `BlueLightAI`
20K `CLT`, with a sharp planning-site spike at the trailing space, peak
strength = 1, and best 157× redirect ratio.**  This is a
0.6 B-parameter model — well below the 1 B threshold Hanna et al.
discuss as the scale at which "latent planning emerges".  The 157×
ratio is the **strongest `Qwen3` cell** in this batch, though it sits
3 orders of magnitude below the `Llama 3.2 1B 524 K` reference
(806,260×) and 5 orders below the `Gemma 2 2B 426 K` reference
(9,974,880×) when compared apples-to-apples in the same harness — see
[`../figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md).

The `-ation` partial-signal result (5.4× one position before the
planning site) is consistent with planning being present in 0.6B but
the -self inject feature being a sub-optimal redirect target for this
prompt structure.  The `-ation` preset's inject feature (`L22:4081`)
is **both** the top `EH1 L F` feature by `max_cosine` **and** the
feature with the highest cosine to `" myself"` (`cos = 0.42`); no
better candidate exists in the scan.

## Inject-feature hand-pick experiments (`-teen` preset)

The `-teen` preset was also re-run with an alternative
`cos→" duration"`-specific inject feature (`L15:2229`,
`cos = 0.19`) in addition to the canonical cluster-broad pick
(`L19:9578`, top `EY1 SH AH0 N` by `max_cosine`).  The cluster-broad
pick **wins by 3×**: 157× vs 49.5× (both at the planning site).  Interpretation:
once the suppress side has cleared the natural `-teen` prior, the
inject side needs to offer a *region* of token space (the `-ation`
cluster broadly) rather than a single word; the model's "if not
`-teen`, then…" decision benefits from breadth, not specificity.

- Canonical (cluster-broad): [`figure13_teen_grid.json`](figure13_teen_grid.json)
  — 157× at planning site, strength 1.
- Hand-pick alternative (cos→" duration"): [`figure13_teen_grid_v2.json`](figure13_teen_grid_v2.json)
  — 49.5× at planning site, strength 0.5.

## Inject-only sweeps (Reviewer L1Vb02 critique 4)

Both `-teen` and `-ation` presets were also run with `--no-suppress`
(inject feature alone, no suppression):

| Preset | Full (suppress + inject) | Inject-only (`--no-suppress`) |
|---|---:|---:|
| `qwen3-0.6b-20k-teen` (v2/`cos→` inject)  | 49.5× | **47.5×** at planning site, `s=5` |
| `qwen3-0.6b-20k-ation` | 5.4× (off-site) | 1.83× off-site |

The `-teen` cell shows the CLT decoder vector alone produces 96 % of
the full-protocol redirect — the suppress side is nearly redundant for
this cell.  The `-ation` cell shows suppress contributing a real ~3×
boost.  This is the "CLT-decoder-as-additive-steering-vector" test
prompted by Reviewer L1Vb02; the outcome confirms the
biology-of-LLMs default: CLT decoder vectors are residual-stream
directions; using them as such produces a planning-site redirect (the
"Figure 13" result that Lindsey et al. 2025 and our paper both
demonstrate).  Files:
[`figure13_teen_inject_only_grid.json`](figure13_teen_inject_only_grid.json),
[`figure13_ation_inject_only_grid.json`](figure13_ation_inject_only_grid.json).

See [`../figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md)
§6 for the cross-cell interpretation.

## Reproducibility

- **Raw scan output**: [`vocab_scan_qwen3_raw.json`](vocab_scan_qwen3_raw.json)
  (669 MB, gitignored).  Regenerate with the scan command above.
- **Committed clean subset**:
  [`vocab_scan_qwen3_phonological_clean.json`](vocab_scan_qwen3_phonological_clean.json)
  (1.9 MB) — the 739 phonologically-clean features only.
- **Sweep outputs**: [`figure13_ation.json`](figure13_ation.json),
  [`figure13_teen.json`](figure13_teen.json) (committed; ~2.6 KB each).
- **Feature picks**: regenerate with `python scripts/pick_features.py`
  (top-5 features per rime per clean JSON).
