#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Summarise Exp 3a random-feature / random-direction inject controls.

Reads the ``random_inject_<cell>.json`` files written by
``figure13_planning_poems --random-inject/--random-direction`` and reports, per
cell and against the registered decision criterion:

* the real inject feature's max ratio (P(target) / baseline), for scale;
* the worst-case (max over draws) ratio under the random-feature and
  random-direction controls -- the claim holds iff this stays within 10x of
  baseline (i.e. ratio <= 10), against 1e5-1e7x for the real feature;
* readout (ii): how many random features spike their OWN top decoder token at
  the final token (the decoder-only-regime signature), with the median own P.

Run after ``run_random_controls.sh``:
    python scripts/random_controls_aggregate.py
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTROLS = REPO / "docs" / "experiments" / "figure13-controls"

CELLS = [
    ("gemma-426k", "Gemma 2 2B x mntss 426K (-out -> around)"),
    ("llama-524k", "Llama 3.2 1B x mntss 524K (-ee -> that)"),
    ("qwen3-0.6b-16k", "Qwen3-0.6B x BLA-dev 16K (-ation -> myself)"),
]

# The final-token position is n_tokens - 1 (the trailing-space planning site).
def is_final(pos: int, n_tokens: int) -> bool:
    return pos == n_tokens - 1


def main() -> int:
    summary_rows = []
    for cell, label in CELLS:
        path = CONTROLS / f"random_inject_{cell}.json"
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue
        j = json.loads(path.read_text(encoding="utf-8"))
        n_tokens = len(j["tokens"])
        baseline = j["baseline_prob"]
        real_ratio = j["real_inject"]["max_ratio_target"]
        real_pos = j["real_inject"]["max_position"]

        ri = j["random_inject"]
        rd = j["random_direction"]
        ri_ratios = [d["max_ratio_target"] for d in ri]
        rd_ratios = [d["max_ratio_target"] for d in rd]

        # Decision criterion: worst-case random ratio within 10x baseline.
        ri_worst = max(ri_ratios) if ri_ratios else 0.0
        rd_worst = max(rd_ratios) if rd_ratios else 0.0
        passes = ri_worst <= 10.0 and rd_worst <= 10.0

        # Readout (ii): own-token spikes at the final token.
        own_final = [d for d in ri if is_final(d["max_position_own"], n_tokens)]
        own_final_ps = [d["max_p_own"] for d in own_final]

        print(f"\n=== {cell} — {label} ===")
        print(f"baseline P(target) = {baseline:.3e}   (n_tokens={n_tokens})")
        print(f"REAL inject:      max ratio {real_ratio:,.0f}x at pos {real_pos} "
              f"({'final' if is_final(real_pos, n_tokens) else 'pos '+str(real_pos)})")
        print(f"random-inject:    worst {ri_worst:.2f}x, median {statistics.median(ri_ratios):.2f}x "
              f"over {len(ri_ratios)} draws  -> "
              f"{sum(r > 10 for r in ri_ratios)} exceed 10x")
        print(f"random-direction: worst {rd_worst:.2f}x, median {statistics.median(rd_ratios):.2f}x "
              f"over {len(rd_ratios)} draws  -> "
              f"{sum(r > 10 for r in rd_ratios)} exceed 10x")
        print(f"decision (both controls <=10x): {'PASS' if passes else 'FAIL'}  "
              f"(real is {real_ratio/max(ri_worst,rd_worst,1e-9):,.0f}x above the worst random draw)")
        print(f"readout (ii) own-token @ final: {len(own_final)}/{len(ri)} random features"
              + (f"; median own P = {statistics.median(own_final_ps):.2e}" if own_final_ps else ""))

        summary_rows.append({
            "cell": cell,
            "label": label,
            "baseline_prob": baseline,
            "n_tokens": n_tokens,
            "real_max_ratio": real_ratio,
            "real_max_position": real_pos,
            "real_at_final": is_final(real_pos, n_tokens),
            "random_inject_worst_ratio": ri_worst,
            "random_inject_median_ratio": statistics.median(ri_ratios),
            "random_inject_n_exceed_10x": sum(r > 10 for r in ri_ratios),
            "random_direction_worst_ratio": rd_worst,
            "random_direction_median_ratio": statistics.median(rd_ratios),
            "random_direction_n_exceed_10x": sum(r > 10 for r in rd_ratios),
            "decision_pass": passes,
            "own_token_final_count": len(own_final),
            "own_token_total": len(ri),
            "own_token_final_median_p": statistics.median(own_final_ps) if own_final_ps else None,
            "seed": j.get("seed"),
        })

    # --- Exp 3b: random-model (dead-salmon) controls, Gemma 426K, 3 seeds ---
    init_rows = summarize_random_model(
        sorted(CONTROLS.glob("random_model_seed*.json")),
        "random-init (fresh N(0,0.02) weights)",
    )
    shuffle_rows = summarize_random_model(
        sorted(CONTROLS.glob("random_model_shuffle_seed*.json")),
        "weight-shuffle (norm-preserving; rules out 'just the scales')",
    )

    dest = CONTROLS / "random_controls_summary.json"
    dest.write_text(
        json.dumps(
            {
                "exp3a_inject": summary_rows,
                "exp3b_random_init": init_rows,
                "exp3b_weight_shuffle": shuffle_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {dest.relative_to(REPO)}")
    return 0


def summarize_random_model(seeds: list[Path], label: str) -> list[dict]:
    """Exp 3b: for each random-model seed, the max P(target) ratio and its
    position (should be no spike anywhere; unstable across seeds)."""
    if not seeds:
        return []
    print(f"\n=== Exp 3b — {label}, Gemma 2 2B x mntss 426K (-out -> around) ===")
    print(f"{'seed':>4} {'baseline':>10} {'max ratio':>10} {'max pos':>8} {'spike?':>8}")
    print("-" * 46)
    rows = []
    positions = set()
    for path in seeds:
        j = json.loads(path.read_text(encoding="utf-8"))
        # Seed is the trailing integer of the filename (schema-independent).
        m = re.search(r"seed(\d+)", path.stem)
        seed = int(m.group(1)) if m else None
        base = j["baseline_prob"]
        sweep = j["sweep"]
        n = len(sweep)
        best = max(sweep, key=lambda s: s["prob"])
        ratio = best["prob"] / base if base > 0 else float("inf")
        at_final = best["position"] == n - 1
        positions.add(best["position"])
        print(f"{seed!s:>4} {base:>10.2e} {ratio:>9.2f}x {best['position']:>8} "
              f"{'final' if at_final else 'pos '+str(best['position']):>8}")
        rows.append({
            "seed": seed,
            "baseline_prob": base,
            "max_ratio": ratio,
            "max_position": best["position"],
            "max_at_final": at_final,
            "n_tokens": n,
        })
    worst = max(r["max_ratio"] for r in rows)
    print("-" * 46)
    print(f"worst-case ratio across seeds: {worst:.2f}x  (real trained model: 9,974,880x)")
    print(f"argmax position varies across seeds: {sorted(positions)} "
          f"({'stable' if len(positions) == 1 else 'unstable — no consistent site'})")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
