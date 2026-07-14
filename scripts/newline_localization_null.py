#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Null-model probability for the Figure-13 localization claim (Exp analysis item).

The paper's localization claim (main.tex, tab:cells / sec:sweep) is that across
all seven cells the steering spike lands within the last two prompt positions --
six cells exactly at the final token, the seventh (Qwen3-0.6B, 20K, -ation) one
position earlier. This script attaches an exact null-model probability to that
claim: under the null that each cell's spike position is uniform over its n_i
swept positions, and cells are independent,

    p_last_two = prod_i (2 / n_i)                    (the "last two positions" event)
    p_all_final = prod_i (1 / n_i)                   (strict: every cell at final token)

and, because the observed pattern is 6-of-7 exactly at the final token, the
appendix "6-of-7" number is the probability of at least six cells landing on the
final token under per-cell success rate p_i = 1 / n_i (an exact Poisson-binomial,
with a mean-rate simple-binomial reported alongside as a sanity check).

n_i is read from each cell's committed grid JSON as ``len(sweep)`` -- it is the
token/position count of the cell's prompt and is therefore invariant to steering
strength and grid version. The authoritative grid per cell is the one at that
cell's Table-2 "Best s"; the mapping below is transcribed from tab:cells and the
spike position each grid reports is asserted to match the paper's stated pattern.

Run from the repo root:  python scripts/newline_localization_null.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# The seven Table-2 cells, each mapped to the authoritative grid JSON at that
# cell's "Best s" (main.tex, tab:cells) and the spike site the paper reports.
# ``site`` is "final" (spike at position n-1) or "adjacent" (position n-2).
CELLS = [
    ("Gemma 2 2B x mntss 426K (-out)",      "figure13-gemma-426k/figure13_out_grid.json",        25.0,  "final"),
    ("Llama 3.2 1B x mntss 524K (-ee)",     "figure13-llama-524k/figure13_ee_grid.json",         25.0,  "final"),
    ("Qwen3-0.6B x BLA-dev 16K (-ation)",   "figure13-qwen3-0.6b-16k/figure13_ation_grid_v2.json", 25.0, "final"),
    ("Qwen3-0.6B x BLA 20K (-teen)",        "figure13-qwen3-0.6b-20k/figure13_teen_grid.json",    1.0,  "final"),
    ("Qwen3-1.7B x BLA 20K (-teen)",        "figure13-qwen3-1.7b-20k/figure13_teen_grid.json",    5.0,  "final"),
    ("Qwen3-0.6B x BLA 20K (-ation)",       "figure13-qwen3-0.6b-20k/figure13_ation_grid.json",   10.0, "adjacent"),
    ("Qwen3-1.7B x BLA 20K (-ation)",       "figure13-qwen3-1.7b-20k/figure13_ation_grid_v2.json", 2.5, "final"),
]

EXPERIMENTS = Path(__file__).resolve().parent.parent / "docs" / "experiments"


def load_cell(rel_path: str):
    """Return (n, spike_position, best_prob, strength) for one grid JSON."""
    grid = json.loads((EXPERIMENTS / rel_path).read_text(encoding="utf-8"))
    sweep = grid["sweep"]
    n = len(sweep)
    best = max(sweep, key=lambda s: s["prob"])
    return n, best["position"], best["prob"], grid.get("strength")


def poisson_binomial_at_least(k: int, probs: list[float]) -> float:
    """Exact P(>= k successes) for independent Bernoulli trials with heterogeneous
    ``probs`` (the Poisson-binomial distribution), via the standard DP over the
    success-count PMF."""
    # pmf[j] = probability of exactly j successes among the trials seen so far.
    pmf = [1.0]
    for p in probs:
        nxt = [0.0] * (len(pmf) + 1)
        for j, mass in enumerate(pmf):
            nxt[j] += mass * (1.0 - p)      # this trial fails
            nxt[j + 1] += mass * p          # this trial succeeds
        pmf = nxt
    return sum(pmf[k:])


def main() -> int:
    ns: list[int] = []
    rates: list[float] = []
    n_final = 0
    print(f"Reading committed grids from {EXPERIMENTS}\n")
    print(f"{'cell':38s} {'n_i':>4} {'spike':>6} {'site':>9} {'best P':>10} {'s':>5}")
    print("-" * 78)
    ok = True
    for name, rel, s_expected, site in CELLS:
        n, spike, best_p, strength = load_cell(rel)
        observed_site = "final" if spike == n - 1 else ("adjacent" if spike == n - 2 else f"other({spike})")
        flag = "" if observed_site == site else "  <-- MISMATCH vs tab:cells"
        if observed_site != site:
            ok = False
        if strength != s_expected:
            flag += f"  <-- strength {strength} != Best s {s_expected}"
            ok = False
        print(f"{name:38s} {n:>4} {spike:>6} {observed_site:>9} {best_p:>10.2e} {strength:>5}{flag}")
        ns.append(n)
        rates.append(1.0 / n)
        if observed_site == "final":
            n_final += 1
    print("-" * 78)

    k = len(ns)
    sum_ns = sum(ns)
    p_last_two = math.prod(2.0 / n for n in ns)
    p_all_final = math.prod(1.0 / n for n in ns)

    # Exact appendix number: P(>= 6 of 7 cells at the final token), rates 1/n_i.
    p_at_least_6_exact = poisson_binomial_at_least(6, rates)
    # Simple-binomial sanity check at the mean rate.
    p_bar = sum(rates) / k
    p_at_least_6_binom = (
        math.comb(k, 6) * p_bar**6 * (1.0 - p_bar) + p_bar**7
    )

    print(f"\nCells: {k}    n_i = {ns}    sum n_i = {sum_ns}")
    print(f"Observed pattern: {n_final} of {k} spikes exactly at the final token, "
          f"{k - n_final} adjacent (last-two elsewhere)\n")

    print("Headline (last-two-positions event, all seven cells):")
    print(f"    p_last_two = prod_i (2 / n_i) = {p_last_two:.3e}   (~10^{math.log10(p_last_two):.1f})\n")

    print("Strict (every cell exactly at the final token):")
    print(f"    p_all_final = prod_i (1 / n_i) = {p_all_final:.3e}   (~10^{math.log10(p_all_final):.1f})\n")

    print("Appendix (>= 6 of 7 cells at the final token, per-cell rate 1/n_i):")
    print(f"    exact Poisson-binomial      = {p_at_least_6_exact:.3e}   (~10^{math.log10(p_at_least_6_exact):.1f})")
    print(f"    simple binomial at mean p   = {p_at_least_6_binom:.3e}   (p_bar = {p_bar:.4f})")

    if not ok:
        print("\nWARNING: an observed spike/strength did not match tab:cells; "
              "re-check the authoritative grid mapping before quoting these numbers.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
