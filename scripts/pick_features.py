#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Quick picker: extract top features per rime from a CMUdict-clean JSON.

Helper for the v0.1.11 Wednesday handoff, Step 2.4.  Reads one of the
``vocab_scan_qwen3_phonological_clean.json`` files and prints, per rime,
the top features sorted by ``max_cosine``.
"""

import json
import sys
from collections import defaultdict


def pick(path, rimes_wanted, topn=5):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"]
    by_rime = defaultdict(list)
    for ft in feats:
        r = ft.get("cmudict_rime")
        if r in rimes_wanted:
            by_rime[r].append(ft)
    for r in rimes_wanted:
        lst = sorted(by_rime[r], key=lambda x: -x["max_cosine"])
        print(f"--- {path} : {r} (top {topn} by max_cosine) ---")
        for ft in lst[:topn]:
            tops = ", ".join(
                t["text"].strip() + f"({t['cosine']:.3f})" for t in ft["top_tokens"][:5]
            )
            samples = ", ".join(ft["cmudict_rime_sample_words"][:5])
            print(
                f"  L{ft['feature']['layer']}:{ft['feature']['index']}  "
                f"max={ft['max_cosine']:.4f}  "
                f"share={ft['cmudict_rime_share']:.2f}  "
                f"[{tops}]  samples=[{samples}]"
            )
        print()


def main():
    rimes = ["EY1 SH AH0 N", "IY1 N", "EH1 L F", "UH1 D"]
    paths = [
        "docs/experiments/figure13-qwen3-1.7b-20k/vocab_scan_qwen3_phonological_clean.json",
        "docs/experiments/figure13-qwen3-0.6b-20k/vocab_scan_qwen3_phonological_clean.json",
        "docs/experiments/figure13-qwen3-0.6b-16k/vocab_scan_qwen3_phonological_clean.json",
    ]
    for p in paths:
        print(f"=========== {p} ===========")
        pick(p, rimes)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
