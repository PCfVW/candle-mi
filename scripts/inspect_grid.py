#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pretty-print the figure13 strength-grid output for a quick sanity look.

Helper for the v0.1.11 Wednesday handoff: dump the per-strength max-ratio
profile plus the planning-site (trailing-space) probability per strength.
"""

import json
import sys


def show(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    print(f"=== {path} ===")
    print(f"baseline_prob: {d['baseline_prob']:.3e}")
    best_r = d.get("best_ratio")
    best_pos = d.get("best_position")
    print(
        f"best_ratio: {best_r:.2f}x at pos {best_pos}, strength {d['strength']}"
    )
    print()
    print("--- Per-strength: max P over all positions, plus planning-site (last position) ---")
    grid = d.get("sweep_grid") or []
    n_pos = len(grid[0]["sweep"]) if grid else 0
    spike_pos = n_pos - 1  # the trailing-space position
    for row in grid:
        s = row["strength"]
        probs = [p["prob"] for p in row["sweep"]]
        max_p = max(probs)
        max_pos = probs.index(max_p)
        spike_p = probs[spike_pos]
        baseline = d["baseline_prob"]
        max_ratio = max_p / baseline if baseline > 0 else 0.0
        spike_ratio = spike_p / baseline if baseline > 0 else 0.0
        print(
            f"  s={s:>6g}: max P={max_p:.3e} at pos {max_pos:>2} "
            f"(ratio {max_ratio:>6.2f}x); spike(pos {spike_pos}) P={spike_p:.3e} "
            f"(ratio {spike_ratio:>6.2f}x)"
        )
    print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for path in sys.argv[1:]:
        show(path)


if __name__ == "__main__":
    main()
