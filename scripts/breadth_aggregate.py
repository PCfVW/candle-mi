#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Aggregate Exp 4 (prompt breadth) sweeps into per-cell breadth_<cell>.json.

Exp 4 reruns the three other COLM-validated prompts per reference cell through
``figure13_planning_poems`` at the cell's best strength (s = 25), keeping the
cell's preset suppress/inject features and inject word fixed (only the prompt
varies). This script collects those three fresh sweeps plus prompt #1 (read from
the committed Table-2 grid, which is already at s = 25), and reports, per the
controls-and-breadth spec:

* per prompt: spike position, whether it localizes at the final token, best P,
  and best ratio (P / baseline);
* per cell: the localization count k/4 at the final token with an exact
  Clopper-Pearson 95% CI, plus the median and range of the best ratio.

Run after ``run_breadth.sh``:
    python scripts/breadth_aggregate.py

Reads fresh sweeps from ``docs/experiments/figure13-controls/_runs/`` and prompt
#1 from the committed grids; writes ``breadth_<cell>.json`` into
``docs/experiments/figure13-controls/``.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTROLS = REPO / "docs" / "experiments" / "figure13-controls"
RUNS = CONTROLS / "_runs"

# Per cell: (preset, committed-grid prompt-#1 path, the three rerun labels, the
# rime group each prompt naturally primes -- prompt #1 first).
CELLS = {
    "gemma-426k": {
        "preset": "gemma2-2b-426k",
        "prompt1_grid": REPO / "docs/experiments/figure13-gemma-426k/figure13_out_grid.json",
        "prompt1_group": "-out (about)",
        "reruns": [("so", "-ow"), ("shout", "-out"), ("who", "-oo")],
    },
    "llama-524k": {
        "preset": "llama3.2-1b-524k",
        "prompt1_grid": REPO / "docs/experiments/figure13-llama-524k/figure13_ee_grid.json",
        "prompt1_group": "-ee (free)",
        "reruns": [("new", "-oo"), ("sat", "-at"), ("more", "-ore")],
    },
}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson (Beta) two-sided CI for a binomial proportion.

    The interval endpoints are Beta quantiles: lower = B(alpha/2; k, n-k+1),
    upper = B(1-alpha/2; k+1, n-k), with the degenerate ends (k=0 -> lower 0,
    k=n -> upper 1) handled explicitly."""
    from scipy.stats import beta  # local import: only needed for the CI

    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def spike_of(sweep: list[dict]) -> tuple[int, float]:
    """Return (argmax position, max prob) of a sweep."""
    best = max(sweep, key=lambda s: s["prob"])
    return best["position"], best["prob"]


def prompt_record(grid: dict, group: str, is_reference: bool) -> dict:
    sweep = grid["sweep"]
    n = len(sweep)
    pos, best_p = spike_of(sweep)
    baseline = grid["baseline_prob"]
    ratio = best_p / baseline if baseline > 0 else float("inf")
    site = "final" if pos == n - 1 else ("adjacent" if pos == n - 2 else "other")
    return {
        "natural_group": group,
        "prompt": grid["prompt"],
        "n_positions": n,
        "spike_position": pos,
        "spike_token": sweep[pos]["token"],
        "spike_site": site,
        "localizes_at_final": pos == n - 1,
        "baseline_prob": baseline,
        "best_prob": best_p,
        "best_ratio": ratio,
        "strength": grid["strength"],
        "is_reference_prompt": is_reference,
    }


def main() -> int:
    combined_null = 1.0
    combined_k = combined_n = 0
    for cell, spec in CELLS.items():
        prompts = []
        # Prompt #1 from the committed grid (already s=25).
        grid1 = json.loads(Path(spec["prompt1_grid"]).read_text(encoding="utf-8"))
        prompts.append(prompt_record(grid1, spec["prompt1_group"], is_reference=True))
        # Prompts #2-#4 from the fresh reruns.
        for label, group in spec["reruns"]:
            path = RUNS / f"breadth_{spec['preset']}_{label}.json"
            grid = json.loads(path.read_text(encoding="utf-8"))
            prompts.append(prompt_record(grid, group, is_reference=False))

        n = len(prompts)
        k = sum(1 for p in prompts if p["localizes_at_final"])
        lo, hi = clopper_pearson(k, n)
        ratios = [p["best_ratio"] for p in prompts]

        # Per-prompt null binomial (spec item 1 extension): under the null that
        # each prompt's spike is uniform over its n_i positions, the probability
        # that all k observed final-token localizations occur is prod (1/n_i)
        # over exactly those prompts (an exact Poisson-binomial all-success term,
        # since here k == n).
        null_final = math.prod(1.0 / p["n_positions"] for p in prompts if p["localizes_at_final"])

        out = {
            "cell": cell,
            "preset": spec["preset"],
            "strength": 25.0,
            "note": (
                "Exp 4 prompt breadth: the cell's preset suppress/inject features "
                "and inject word are held fixed; only the prompt varies. Prompt #1 "
                "is the reference cell (from the committed Table-2 grid); prompts "
                "#2-#4 are the other COLM-validated prompts, rerun at s=25."
            ),
            "n_prompts": n,
            "k_localize_final_token": k,
            "localization_rate": k / n,
            "clopper_pearson_95ci": [lo, hi],
            "null_all_final_prob": null_final,
            "best_ratio_median": statistics.median(ratios),
            "best_ratio_min": min(ratios),
            "best_ratio_max": max(ratios),
            "prompts": prompts,
        }
        dest = CONTROLS / f"breadth_{cell}.json"
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

        print(f"\n=== {cell} ({spec['preset']}) ===")
        print(f"{'group':14s} {'n':>3} {'spike':>6} {'site':>9} {'best P':>10} {'ratio':>13}  ref")
        print("-" * 70)
        for p in prompts:
            print(f"{p['natural_group']:14s} {p['n_positions']:>3} {p['spike_position']:>6} "
                  f"{p['spike_site']:>9} {p['best_prob']:>10.2e} {p['best_ratio']:>13.1f}"
                  f"  {'#1' if p['is_reference_prompt'] else ''}")
        print("-" * 70)
        print(f"localization at final token: {k}/{n} = {k/n:.0%}  "
              f"(Clopper-Pearson 95% CI [{lo:.3f}, {hi:.3f}])")
        print(f"null P(all {k} at final token) = prod(1/n_i) = {null_final:.2e}")
        print(f"best ratio: median {statistics.median(ratios):.1f}x, "
              f"range [{min(ratios):.1f}x, {max(ratios):.1f}x]")
        print(f"wrote {dest.relative_to(REPO)}")
        combined_null *= null_final
        combined_k += k
        combined_n += n

    print(f"\n=== combined ===")
    print(f"final-token localization: {combined_k}/{combined_n} prompts across both cells")
    print(f"null P(all {combined_k} at final token) = prod(1/n_i) = {combined_null:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
