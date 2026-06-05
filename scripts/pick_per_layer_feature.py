#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Per-layer best CLT feature for each target word, from a raw vocab-scan JSON.

Companion to ``pick_inject_feature.py`` (which finds the single globally best
inject feature). For the commitment-onset measurement we instead need, **for
every layer**, the feature at that layer whose decoder vector most encodes the
target word's embedding direction — so we can read that feature's activation at
each layer and locate where the planned-token representation switches on.

Emits a compact JSON consumed by ``examples/commitment_onset.rs``:

```json
{ " on": {"0": {"index": 1234, "cosine": 0.11}, "1": {...}, ...}, " off": {...} }
```

Usage:
    python scripts/pick_per_layer_feature.py <raw_scan_json> " on" " off" \
        --output docs/.../per_layer_features_<clt>.json
"""

import argparse
import json
import sys


def normalize(text):
    """Strip leading space/underscore/metaspace markers and case-fold."""
    t = text
    while t and t[0] in (" ", "_", "▁"):
        t = t[1:]
    return t.casefold()


def best_cosine_for_target(feature, target_norm):
    """Max cosine this feature assigns to any case/space variant of the target."""
    best = None
    for tok in feature["top_tokens"]:
        if normalize(tok["text"]) == target_norm:
            c = tok["cosine"]
            if best is None or c > best:
                best = c
    return best


def per_layer_best(features, target):
    """layer (int) -> {'index': int, 'cosine': float} keeping the max-cosine
    feature per layer for `target`."""
    target_norm = normalize(target)
    by_layer = {}
    for ft in features:
        c = best_cosine_for_target(ft, target_norm)
        if c is None:
            continue
        layer = ft["feature"]["layer"]
        prev = by_layer.get(layer)
        if prev is None or c > prev["cosine"]:
            by_layer[layer] = {"index": ft["feature"]["index"], "cosine": c}
    return by_layer


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_scan_json")
    parser.add_argument("targets", nargs="+", help="target words, e.g. ' on' ' off'")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.raw_scan_json, encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"]

    out = {}
    for target in args.targets:
        by_layer = per_layer_best(feats, target)
        # JSON object keys must be strings.
        out[target] = {str(k): v for k, v in sorted(by_layer.items())}
        covered = len(by_layer)
        print(
            f"{target!r}: {covered} layers covered "
            f"(of {data.get('n_features_scanned', '?')} features)",
            file=sys.stderr,
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
