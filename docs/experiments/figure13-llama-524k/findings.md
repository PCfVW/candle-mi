# `Llama 3.2 1B` × `mntss/clt-llama-3.2-1b-524k` — `figure13_planning_poems` reference cell

**Date**: 2026-05-27
**Hardware**: RTX 5060 Ti 16 GB, Windows 11, Rust 1.95
**Role**: apples-to-apples reference for the
[cross-size sweep](../figure13-qwen3-cross-size.md).  Same harness as the
five `Qwen3` cells; same 2D **position × strength** grid sweep
(`strengths = {0.5, 1, 2.5, 5, 10, 25, 50, 100}`).

## Headline

`P(" that")` = **0.8525** at the **trailing-space planning site** (position 30)
at **strength = 25**.  Ratio: **806,260×** above baseline
(`baseline P(" that") = 1.057 × 10⁻⁶`).

This **modestly exceeds** the paper's reported `s = 10` number
(`P(" that") = 0.777`, ratio 133,879×).  The grid sweep confirms that
`s = 10` was a strong but not-quite-optimal choice; the optimum for this
cell sits at `s = 25`.

## Pipeline

```powershell
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- `
    --preset llama3.2-1b-524k `
    --strength-grid 0.5,1,2.5,5,10,25,50,100 `
    --output docs/experiments/figure13-llama-524k/figure13_ee_grid.json
```

Preset `LLAMA` defined in [`examples/figure13_planning_poems.rs`](../../../examples/figure13_planning_poems.rs):
suppress = `[L13:30985, L9:5488, L14:27874, L13:32049]` (top -ee features),
inject = `L14:13043` (`" that"` feature).

## Per-strength profile

| Strength | Max P (at pos 30) | Ratio |
|---:|---|---:|
| 0.5  | 2.85 × 10⁻⁶ |        2.7× |
| 1    | 7.83 × 10⁻⁶ |        7.4× |
| 2.5  | 1.75 × 10⁻⁴ |      165.9× |
| 5    | 3.01 × 10⁻² |   28,470.1× |
| 10   | 6.87 × 10⁻¹ |  649,327.6× |
| **25** | **8.53 × 10⁻¹** | **806,260.3×** |
| 50   | 8.18 × 10⁻¹ |  773,649.5× |
| 100  | 8.36 × 10⁻¹ |  790,615.7× |

Profile shape: smooth, monotone-increasing through `s = 25` then plateau
in the 0.78–0.84 absolute-`P` range.  The plateau at high strength is
the natural saturation of a single-token probability — there's a ceiling
near 0.85.

## Comparison to the paper's reported numbers

| Source | Strength | Best position | Best `P(" that")` | Ratio |
|---|---:|---:|---|---:|
| Paper (Q2, §5)                 | 10 | 30 | 0.777 | 133,879× |
| Prior `clt_step_a_llama.json` (v0.1.9, this repo) | 10 | 30 | **0.6866** | 649,327× |
| **This grid sweep (canonical)** | **25** | **30** | **0.8525** | **806,260×** |

The paper's `133,879×` ratio is computed against a different baseline (the
paper's prompt set has 44 prompts × 11 alt-groups; this single-prompt
run uses one prompt × one alt-group).  Both yield the same
order-of-magnitude effect.  Per-prompt drift between the paper's
documented `0.777` and our `0.6866` (at `s = 10`) is the candle 0.9
stack vs paper-stack drift documented in
[`v0.1.11-plan-and-h-and-a-discussion.md`](../../v0.1.11-plan-and-h-and-a-discussion.md).

## Cross-cell context

See [`../figure13-qwen3-cross-size.md`](../figure13-qwen3-cross-size.md)
for the full 7-cell comparison.  This row is row 1 of the headline table
and serves as the apples-to-apples upper-bound reference for the
`Qwen3` × `BlueLightAI` cells (which sit 3–5 orders of magnitude below).

## Reproducibility

- **Grid output**: [`figure13_ee_grid.json`](figure13_ee_grid.json)
  (committed, ~21 KB) — full position × strength sweep.
- **Quick-look**: `python scripts/inspect_grid.py figure13_ee_grid.json`
