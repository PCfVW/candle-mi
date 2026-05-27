# `Qwen3-0.6B-Base` × `BlueLightAI`-dev 16K — Appendix B vocabulary scan + Figure 13 sweep

**Date**: 2026-05-27
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, Rust 1.95
**Vocab-scan outcome**: **N = 648 phonologically-clean features** (vs 739 at
the same base model + 20K width; the narrower `CLT` loses ~12 % of the clean
features and ~62 % of the `-ation` group specifically).
**Sweep outcome**: `-ation → -self` redirect ratio 3.6× at position 0 — in
noise; consistent with the same tokenizer + inject-feature limitations
documented for the 20K `-ation` run.  *No `-teen` sweep was run for this
variant; it would require pre-caching the 23.68 GiB 20K `CLT` for the same
base model, which is already exercised in the sibling experiment.*

## Pipeline

```powershell
# 1. Pre-cache the dev CLT (16.79 GiB; ~17 min on a 17 MiB/s link).
hf-fm bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test --timeout-per-file-secs 1800

# 2. Run the vocabulary scan (28 layers × 16 384 features × 151 936 tokens).
cargo run --release --features clt,transformer,mmap --example vocab_scan -- `
    --model Qwen/Qwen3-0.6B-Base `
    --clt-repo bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test `
    --output docs/experiments/figure13-qwen3-0.6b-16k/vocab_scan_qwen3_raw.json

# 3. Filter through CMUdict + dump the commit-friendly clean-only subset.
python scripts/vocab_scan_cmudict_filter.py `
    docs/experiments/figure13-qwen3-0.6b-16k/vocab_scan_qwen3_raw.json `
    --clean-only-output `
    --output docs/experiments/figure13-qwen3-0.6b-16k/vocab_scan_qwen3_phonological_clean.json

# 4. Run the -ation figure13 sweep.
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- `
    --preset qwen3-0.6b-16k-ation `
    --output docs/experiments/figure13-qwen3-0.6b-16k/figure13_ation.json
```

## Runtime

| Stage | Wall-clock |
|---|---|
| Decoder pre-cache (16.79 GiB) | ~17 min (one-time) |
| Vocab scan (28 × 16 384 × 151 936, GPU `F32`) | **176 s (2.9 min)**, ~6.3 s/layer |
| `CMUdict` filter + dedup | <1 min on CPU |
| Figure 13 sweep (20 positions × strength 10) | ~4 s after model load |

## Phonological scan: top rhyme groups (after unique-word dedup)

After deduplicating top-K tokens by normalised English word and applying the
`share ≥ 0.50, count ≥ 3` heuristic, the filter flags **N = 648** clean
features.  Top 10 rimes by feature count:

| Rime (`ARPABET`) | Features | Sample words |
|---|---:|---|
| `IY1`            | 252 | be, de, me, ne, te, je |
| `AA1`            | 106 | da, ga, ha, ja, ma, pa |
| `OW1`            |  62 | au, ow, ro, ho, low, mo |
| `AY1`            |  34 | cy, ly, sy, ty, sky, vy |
| `EY1 SH AH0 N`   |  27 | creation, differentiation, exploitation, formation, illumination |
| `IY1 N`          |  24 | eighteen, fifteen, fourteen, nineteen, seventeen, thirteen |
| `UW1`            |  20 | do, qu, to, deux, two, fu |
| `EH1 L`          |  12 | nel, nell, nelle, bell, cell, tel |
| `EY1 SH AH0 N Z` |   7 | explanations, indications, investigations, revelations |
| `UH1 D`          |   6 | could, should, would |

The 16K width compresses the productive English families:
- `-ation`: 27 features here vs 71 in the 20K sibling (−62 %).
- `-teen`: 24 here vs 33 (−27 %).
- `-self`: 3 here vs 8 (−63 %).
- `-ould` actually *gains*: 6 features vs 5 in the 20K (the modal-verb
  cluster is tight enough that even a narrow `CLT` keeps it intact).

The total clean count drops only ~12 % (648 vs 739), but the
*per-cluster* loss is uneven — `CLT` width disproportionately affects
the morphologically rich families.

## Figure 13 sweep results

One preset, swept as a 2D **position × strength** grid (strengths
`{0.5, 1, 2.5, 5, 10, 25, 50, 100}`).  The earlier single-strength run
(`figure13_ation.json`) is retained for the audit trail; the grid run
(`figure13_ation_grid.json`) is the load-bearing artefact.

A `find_token_id` fix (`src/tokenizer/mod.rs`, same release) was required
before the grid run: the previous implementation assumed `BOS` was always
prepended (`Llama` / `Gemma` convention), silently fell through on `Qwen3`
(`add_bos_token = false`), and returned the wrong sub-token for multi-token
words.  The grid run uses the corrected lookup, so `inject_word = "myself"`
now correctly resolves to token id `7037` (`" myself"`) instead of the bare
`"self"` sub-token (id `721`).  The baseline P(`" myself"`) is
`2.69 × 10⁻⁷` — about 100× the previously measured P(`"self"`)
`2.26 × 10⁻⁹`.

### `qwen3-0.6b-16k-ation` (suppress -ation, inject -self: `" myself"`)

Prompt (20 tokens, ending with a trailing space):

```
At every grand celebration,
Each careful preparation,
Brings joy beyond expectation,
And then the brief
```

**Headline (after inject-feature hand-pick, v2)**: best ratio **33,860×**
at **position 19 (the trailing-space planning site)** at **strength 25**,
with the original `L15:6772` top-`EH1 L F` inject swapped for the
`cos→" myself"`-specific pick `L22:8011` (`cos = 0.30` to `" myself"`,
top tokens all `" my"` variants).  Compared to the first grid run with
`L15:6772` (max ratio 3.48× at strength 10), the hand-pick:

- **multiplies the ratio by ~10,000×** (3.48× → 33,860×)
- **multiplies the absolute P by ~10,000×** (9.4 × 10⁻⁷ → 9.12 × 10⁻³)
- Keeps the spike *at* the planning site (pos 19, unchanged)

Per-strength profile at the planning site (v2, `L22:8011` inject):

| Strength | P at pos 19 | Ratio |
|---:|---|---:|
| 0.5  | 1.38 × 10⁻⁶ |       5.1× |
| 1    | 5.27 × 10⁻⁶ |      19.6× |
| 2.5  | 5.55 × 10⁻⁵ |     206.1× |
| 5    | 2.12 × 10⁻³ |   7,888.4× |
| 10   | 5.32 × 10⁻³ |  19,762.7× |
| **25** | **9.12 × 10⁻³** | **33,859.6×** |
| 50   | 8.28 × 10⁻³ |  30,727.4× |
| 100  | 6.25 × 10⁻³ |  23,206.4× |

The profile is **smooth and monotone-increasing** through `s = 25` then
plateau — the canonical "well-behaved sweep" shape, in stark contrast
to the original `L15:6772` non-monotonic profile.

The first-grid file [`figure13_ation_grid.json`](figure13_ation_grid.json)
is retained for the audit trail; the **canonical** result is the v2
file [`figure13_ation_grid_v2.json`](figure13_ation_grid_v2.json) at
strength = 25, position 19, `P(" myself") = 9.12 × 10⁻³`.

The inject-feature swap is documented in the `QWEN3_0_6B_16K_ATION`
preset constant in [`examples/figure13_planning_poems.rs`](../../../examples/figure13_planning_poems.rs).

## Interpretation

The 16K dev `CLT` retains the *qualitative* phonological structure that
the 20K production `CLT` carries — same dominant rimes, same broad rank
ordering — but with a thinner per-cluster feature density.  For the
productive English suffix families that the figure13 protocol targets,
the 16K width loses about two-thirds of the per-cluster features
(`-ation`, `-self`).

**Surprise of the 33,860× result**: this narrower 16K dev `CLT`
**outperforms** the wider 20K production `CLT` by ~6,300× on the
exact same prompt and protocol (33,860× vs 5.4× for the 20K sibling).
The explanation is in the inject feature: `L22:8011` in the 16K dev
`CLT` happens to be an unusually strong steering vector for `" myself"`
(`cos = 0.30`), while the 20K production `CLT`'s best `EH1 L F`
feature (`L22:4081`) is broader (its top tokens are
`自己的 / itself / 自己 / own / themselves`).  Per-feature target-word
specificity beats per-`CLT` feature count on this cell.

Open hypothesis (recorded as future work in
[`../figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md)
"Future work" section): does the 16K dev `CLT` consistently produce
more word-specific decoder vectors than the 20K production `CLT`?  A
broader scan across multiple (target-word, inject feature) pairs would
either confirm this or localise it to the `-self` cluster specifically.

## Reproducibility

- **Raw scan output**: [`vocab_scan_qwen3_raw.json`](vocab_scan_qwen3_raw.json)
  (531 MB, gitignored).
- **Committed clean subset**:
  [`vocab_scan_qwen3_phonological_clean.json`](vocab_scan_qwen3_phonological_clean.json)
  (1.7 MB) — the 648 phonologically-clean features only.
- **Sweep output**: [`figure13_ation.json`](figure13_ation.json) (committed; ~2.6 KB).
