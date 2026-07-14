# Null-model probability for the localization claim (controls-spec item 1)

**Date**: 2026-07-14
**Hardware**: none — pure analysis over the committed grid JSONs (no GPU).
**Spec**: `controls-and-breadth-spec.md` item 1 ("exact localization
probability"), feeding paper §4.1 + appendix.
**Script**: [`scripts/newline_localization_null.py`](../../../scripts/newline_localization_null.py)
(run from repo root: `python scripts/newline_localization_null.py`).

## Headline

Under the null that each cell's spike position is uniform over its `n_i` swept
positions, and cells are independent, the probability that **all seven cells
spike within their last two positions** (the observed pattern — six exactly at
the final token, one adjacent) is

> **p = Πᵢ (2 / nᵢ) = 3.33 × 10⁻⁸  (≈ 10⁻⁷·⁵).**

The "last two positions" event is used (rather than strict last-token) precisely
because of the one adjacent-position cell (Qwen3-0.6B, 20K, -ation). The strict
and appendix variants are reported below.

## Inputs (nᵢ from the committed Table-2 grids)

`nᵢ = len(sweep)` in each cell's grid JSON — the token/position count of the
prompt, invariant to steering strength and grid version. The authoritative grid
per cell is the one at that cell's **Best s** in `tab:cells`; each grid's spike
site is asserted by the script to match the paper's stated pattern (all match).

| Cell | grid (Best s) | nᵢ | spike pos | site |
|---|---|---:|---:|---|
| Gemma 2 2B × mntss 426K (-out)    | `figure13_out_grid.json` (s=25)      | 32 | 31 | final |
| Llama 3.2 1B × mntss 524K (-ee)   | `figure13_ee_grid.json` (s=25)       | 31 | 30 | final |
| Qwen3-0.6B × BLA-dev 16K (-ation) | `figure13_ation_grid_v2.json` (s=25) | 20 | 19 | final |
| Qwen3-0.6B × BLA 20K (-teen)      | `figure13_teen_grid.json` (s=1)      | 22 | 21 | final |
| Qwen3-1.7B × BLA 20K (-teen)      | `figure13_teen_grid.json` (s=5)      | 22 | 21 | final |
| Qwen3-0.6B × BLA 20K (-ation)     | `figure13_ation_grid.json` (s=10)    | 20 | 18 | **adjacent** |
| Qwen3-1.7B × BLA 20K (-ation)     | `figure13_ation_grid_v2.json` (s=2.5)| 20 | 19 | final |

`nᵢ = [32, 31, 20, 22, 22, 20, 20]`, `Σ nᵢ = 167`, all `nᵢ ∈ [20, 32]`.

## The three numbers

| Event | Formula | Value |
|---|---|---|
| **Headline** — all 7 within last two positions | `Πᵢ (2/nᵢ)` | **3.33 × 10⁻⁸** (≈ 10⁻⁷·⁵) |
| **Strict** — all 7 exactly at final token | `Πᵢ (1/nᵢ)` | 2.60 × 10⁻¹⁰ (≈ 10⁻⁹·⁶) |
| **Appendix** — ≥ 6 of 7 at final token, rate 1/nᵢ | exact Poisson-binomial | 4.19 × 10⁻⁸ (≈ 10⁻⁷·⁴) |
| &nbsp;&nbsp;└ sanity check | simple binomial at mean rate p̄ = 0.0435 | 4.56 × 10⁻⁸ |

The appendix figure is the exact ≥ 6-of-7 tail of a Poisson-binomial with
heterogeneous success rates `pᵢ = 1/nᵢ` (the observed pattern is 6-of-7 at the
final token, not 7-of-7, so the strict all-final product understates it). The
simple-binomial number at the mean rate agrees to within 10 %, confirming the
tail is not sensitive to the rate spread.

## Sentence for §4.1

> Under a uniform-position null (each cell's spike equally likely at any of its
> `nᵢ ∈ [20, 32]` swept positions, cells independent), the probability that all
> seven cells localize within their last two positions is
> `Πᵢ (2/nᵢ) = 3.3 × 10⁻⁸`; the strict all-final-token event has probability
> `Πᵢ (1/nᵢ) = 2.6 × 10⁻¹⁰`, and the observed six-of-seven-at-the-final-token
> pattern has probability `4.2 × 10⁻⁸` (Poisson-binomial, appendix).

## Per-prompt binomial (Exp 4 in hand)

The spec's follow-on: with Exp 4 (prompt breadth) run, add the per-prompt
binomial — k of n prompts localizing at the final token against a `1/nᵢ` chance
rate. Both reference cells localize **4/4** prompts at the final token
([`breadth_gemma-426k.json`](breadth_gemma-426k.json),
[`breadth_llama-524k.json`](breadth_llama-524k.json)); under the uniform-position
null the probability of all four landing on the final token is `Πᵢ (1/nᵢ)`:

| Cell | prompts at final | `nᵢ` | null `Πᵢ(1/nᵢ)` |
|---|---|---|---|
| Gemma 426K | 4/4 | 32, 34, 35, 36 | 7.3 × 10⁻⁷ |
| Llama 524K | 4/4 | 31, 31, 33, 32 | 9.9 × 10⁻⁷ |
| **Combined** | **8/8** | — | **7.2 × 10⁻¹³** |

Computed by [`scripts/breadth_aggregate.py`](../../../scripts/breadth_aggregate.py);
see [`breadth.md`](breadth.md) for the full Exp-4 write-up.
