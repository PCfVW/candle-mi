# `Gemma 2 2B` × `mntss/clt-gemma-2-2b-426k` — `figure13_planning_poems` reference cell

**Date**: 2026-05-27
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, Rust 1.95
**Role**: apples-to-apples reference for the
[cross-size sweep](../figure13-qwen3-cross-size.md).  Same harness as the
five `Qwen3` cells; same 2D **position × strength** grid sweep
(`strengths = {0.5, 1, 2.5, 5, 10, 25, 50, 100}`).

## Headline

`P(" around")` = **0.4824** at the **trailing-space planning site**
(position 31) at **strength = 25**.  Ratio: **9,974,880×** above baseline
(`baseline P(" around") = 4.836 × 10⁻⁸`).

This matches the paper's reported best-of-136-pair number `P(" around")
= 0.483` (paper §5).  The grid sweep places the optimum at `s = 25`,
slightly above the paper's `s = 10` (which gives `0.457`).

## Pipeline

```powershell
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- `
    --preset gemma2-2b-426k `
    --strength-grid 0.5,1,2.5,5,10,25,50,100 `
    --output docs/experiments/figure13-gemma-426k/figure13_out_grid.json
```

Preset `GEMMA` defined in [`examples/figure13_planning_poems.rs`](../../../examples/figure13_planning_poems.rs):
suppress = `[L16:13725, L25:9385]` (top -out features),
inject = `L22:10243` (`" around"` feature).

## Per-strength profile

| Strength | Max P (at pos 31) | Ratio |
|---:|---|---:|
| 0.5  | 1.46 × 10⁻⁷ |        3.0× |
| 1    | 4.38 × 10⁻⁷ |        9.1× |
| 2.5  | 1.09 × 10⁻⁵ |      225.7× |
| 5    | 1.51 × 10⁻³ |   31,176.5× |
| 10   | 4.57 × 10⁻¹ | 9,443,600.2× |
| **25** | **4.82 × 10⁻¹** | **9,974,879.5×** |
| 50   | 2.14 × 10⁻¹ | 4,435,070.9× |
| 100  | 1.51 × 10⁻¹ | 3,117,756.3× |

Profile shape: smooth, monotone-increasing through `s = 25` then
**decreasing** at higher strengths (the residual stream is driven
off-manifold and the soft-capped output distribution flattens).
The `s = 25` peak at `0.4824` matches the paper's
`best of 136 pairs = 0.483`.

## Comparison to the paper's reported numbers

| Source | Strength | Best position | Best `P(" around")` | Ratio |
|---|---:|---:|---|---:|
| Paper (Figure 1)               | 10 | 31 | 0.457 |   10⁷× ("ten-million-fold") |
| Paper (best of 136 pairs)      | 10 | 31 | **0.483** | — |
| Prior `clt_step_a_gemma.json` (v0.1.10, this repo) | 10 | 31 | 0.4567 | 9,443,600× |
| **This grid sweep (canonical)**  | **25** | **31** | **0.4824** | **9,974,880×** |

`0.4824` at `s = 25` lands within `0.001` of the paper's `best-of-136
= 0.483` single-prompt number, on this single-prompt run.  This confirms
the paper's protocol reproduces in this candle 0.9 stack to within rounding.

## Cross-cell context

See [`../figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md)
for the full 7-cell comparison.  This row is row 2 of the headline table
and is the **strongest single redirect** observed across the entire
sweep — the mntss 426 K `CLT` on `Gemma 2 2B` delivers a `9.97 × 10⁶`
ratio, three orders of magnitude above the `Llama 524 K` reference and
five orders above the strongest `BlueLightAI` `Qwen3` cell.  This is a
direct quality difference between the two `CLT` training pipelines
(`mntss` plain `ReLU` at much wider feature counts vs `BlueLightAI`
`JumpReLU` at 20 K width).

## Reproducibility

- **Grid output**: [`figure13_out_grid.json`](figure13_out_grid.json)
  (committed, ~22 KB) — full position × strength sweep.
- **Quick-look**: `python scripts/inspect_grid.py figure13_out_grid.json`
