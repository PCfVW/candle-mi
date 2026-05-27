#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Find the best inject feature for a target word in a raw vocab-scan JSON.

The default `pick_features.py` picks suppress features by max `cosine` of
their #1 token within a rime cluster.  That's the right metric for the
suppress side (we want features that broadly cover the rime).

For the inject side, we want a feature whose decoder vector specifically
encodes the *target inject word's* embedding direction — not a feature
whose #1 token happens to be a different member of the contrast cluster.

This script scans the raw `vocab_scan_*.json` (gitignored, ~500 MB to
1.5 GB) and, for each requested target word, ranks all features by the
cosine they assign to that word in their top-K.  The script reports
top-5 candidates per target.
"""

import json
import sys
from collections import defaultdict


def normalize(text):
    """Match the filter's normalisation: strip leading underscore + space,
    case-fold.  Returns the cleaned form for word equality testing."""
    t = text
    # Common BPE leading-space marker variants.
    while t and t[0] in (" ", "_", "▁"):
        t = t[1:]
    return t.casefold()


def find_for_target(features, target):
    """Return list of (max_cosine_for_target, feature_dict) sorted desc."""
    target_norm = normalize(target)
    hits = []
    for ft in features:
        best = None
        for tok in ft["top_tokens"]:
            if normalize(tok["text"]) == target_norm:
                cos = tok["cosine"]
                if best is None or cos > best:
                    best = cos
        if best is not None:
            hits.append((best, ft))
    hits.sort(key=lambda x: -x[0])
    return hits


def show(raw_path, targets, topn=5):
    with open(raw_path, encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"]
    print(f"=== {raw_path} ({len(feats)} features scanned) ===")
    for target in targets:
        hits = find_for_target(feats, target)
        print(f"\n  Top {topn} features for inject_word \"{target}\":")
        if not hits:
            print(f"    (no feature has \"{target}\" in its top-{data.get('top_k','?')})")
            continue
        for cos, ft in hits[:topn]:
            top5 = ", ".join(
                t["text"].strip() + f"({t['cosine']:.3f})"
                for t in ft["top_tokens"][:5]
            )
            print(
                f"    L{ft['feature']['layer']:>2}:{ft['feature']['index']:>5}  "
                f"cos_to_target={cos:.4f}  max_cosine={ft['max_cosine']:.4f}  "
                f"[{top5}]"
            )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        print(
            "Usage: pick_inject_feature.py <raw_scan_json> "
            "<target_word> [<target_word> ...]"
        )
        sys.exit(1)
    raw_path = sys.argv[1]
    targets = sys.argv[2:]
    show(raw_path, targets)


if __name__ == "__main__":
    main()
