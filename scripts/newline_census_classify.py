#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Stage 2 of the newline feature census (Experiment 1).

Augments a stage-1 ``census_<cell>.json`` (written by
``examples/figure13_newline_census.rs``) with the CMUdict-dependent
classification fields and the registered ``plan_like`` decision, then
prints a per-role summary.

The stage-1 file records, per selected position, each collected CLT
feature's activation and **c3** (decoder cosine to the natural target
word's token embedding). This script adds, per feature:

* **c2 ``decoder_top``** — the feature's decoder-to-vocabulary top tokens,
  joined by ``(layer, index)`` from the cell's *raw* ``vocab_scan`` JSON
  (the spec's "available in the raw vocab-scan JSONs"). No recomputation.
* **c1 ``in_census`` / ``census_rime``** — whether the feature is in the
  cell's phonologically-clean vocab-scan set, and its dominant rime.
  ``null`` when no ``--clean-scan`` is supplied.
* **c2 ``contains_target`` / ``group_hits`` / ``contains_inject``** — from
  CMUdict rime grouping of ``decoder_top`` against the natural target word
  and its rime group (and, optionally, the inject word).
* **``plan_like``** — the registered criterion: active at the position AND
  (``group_hits >= 2`` OR ``cos_to_target >= --cos-threshold``).

The CMUdict token-cleaning and rime-extraction are imported verbatim from
``vocab_scan_cmudict_filter.py`` so the phonology matches the pipeline that
produced the vocab scans.

Usage:
    python scripts/newline_census_classify.py \\
        docs/experiments/figure13-newline/census_gemma2-2b-426k.json \\
        --raw-scan docs/experiments/means-ends-prolepsis/vocab_scan_gemma2_2b_426k_raw.json \\
        [--clean-scan <phonological_clean.json>] \\
        [--inject-word around] [--cos-threshold 0.3] \\
        [--output <path>]   # defaults to overwriting the input in place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the exact CMUdict machinery that produced the vocab scans.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab_scan_cmudict_filter import (  # noqa: E402
    cmudict,
    extract_rime,
    normalise_token,
)

# The registered c3 threshold from the spec's decision criterion.
DEFAULT_COS_THRESHOLD = 0.3


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON file (mandatory encoding for multilingual vocabs)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_decoder_top_index(raw_scan: dict) -> dict[tuple[int, int], list[str]]:
    """Map ``(layer, index) -> [decoder-top token texts]`` from a raw scan."""
    index: dict[tuple[int, int], list[str]] = {}
    for feat in raw_scan.get("features", []):
        fid = feat.get("feature", {})
        key = (fid.get("layer"), fid.get("index"))
        texts = [t.get("text", "") for t in feat.get("top_tokens", [])]
        index[key] = texts
    return index


def build_clean_index(clean_scan: dict) -> dict[tuple[int, int], str | None]:
    """Map ``(layer, index) -> dominant rime`` for phonologically-clean
    features. Membership of the returned dict is the c1 census set."""
    index: dict[tuple[int, int], str | None] = {}
    for feat in clean_scan.get("features", []):
        fid = feat.get("feature", {})
        key = (fid.get("layer"), fid.get("index"))
        index[key] = feat.get("cmudict_rime")
    return index


def word_rime(word: str, cmu: dict[str, list[list[str]]]) -> str | None:
    """Rime of a single (already-normalised) word, or ``None`` if unresolved."""
    variants = cmu.get(word)
    if not variants:
        return None
    return extract_rime(variants[0])


def count_group_hits(
    decoder_top: list[str],
    cmu: dict[str, list[list[str]]],
    target_rime: str | None,
) -> int:
    """Return the number of **unique** CMU-resolvable words in ``decoder_top``
    that share ``target_rime`` (the spec's "``>= 2`` words of the natural rhyme
    group").

    BPE variants of one word collapse to a single hit, mirroring
    ``feature_dominant_rime``. Returns ``0`` when ``target_rime`` is ``None``.
    """
    if target_rime is None:
        return 0
    unique_words = {
        w for w in (normalise_token(t) for t in decoder_top) if w is not None
    }
    return sum(1 for w in unique_words if word_rime(w, cmu) == target_rime)


def contains_word(decoder_top: list[str], word: str | None) -> bool:
    """Whether ``word`` (normalised) appears among the decoder-top tokens."""
    if not word:
        return False
    target = normalise_token(word) or word.lower().strip()
    return any(normalise_token(t) == target for t in decoder_top)


def classify(
    census: dict,
    decoder_index: dict[tuple[int, int], list[str]],
    clean_index: dict[tuple[int, int], str | None] | None,
    cmu: dict[str, list[list[str]]],
    inject_word: str | None,
    cos_threshold: float,
) -> dict[str, dict[str, int]]:
    """Augment ``census`` in place; return per-role feature/plan-like counts."""
    natural_word = census.get("natural_word", "")
    natural_norm = normalise_token(natural_word) or natural_word.lower().strip()
    natural_rime = word_rime(natural_norm, cmu)

    inject_norm = None
    inject_rime = None
    if inject_word:
        inject_norm = normalise_token(inject_word) or inject_word.lower().strip()
        inject_rime = word_rime(inject_norm, cmu)

    counts: dict[str, dict[str, int]] = {}
    for pos in census.get("positions", []):
        role = pos.get("role", "?")
        role_counts = counts.setdefault(role, {"features": 0, "plan_like": 0})
        for feat in pos.get("features", []):
            key = (feat.get("layer"), feat.get("index"))

            decoder_top = decoder_index.get(key)
            feat["decoder_top"] = decoder_top  # None if not in the raw scan

            if clean_index is not None:
                feat["in_census"] = key in clean_index
                feat["census_rime"] = clean_index.get(key)
            else:
                feat["in_census"] = None
                feat["census_rime"] = None

            top = decoder_top or []
            feat["contains_target"] = contains_word(top, natural_word)
            group_hits = count_group_hits(top, cmu, natural_rime)
            feat["group_hits"] = group_hits
            if inject_word:
                inject_hits = count_group_hits(top, cmu, inject_rime)
                feat["contains_inject"] = (
                    contains_word(top, inject_word) or inject_hits >= 2
                )
            else:
                feat["contains_inject"] = None

            active = feat.get("activation", 0.0) > 0.0
            cos = feat.get("cos_to_target", 0.0)
            plan_like = active and (group_hits >= 2 or cos >= cos_threshold)
            feat["plan_like"] = plan_like

            role_counts["features"] += 1
            if plan_like:
                role_counts["plan_like"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: classify a newline feature census (c1/c2/plan_like)."
    )
    parser.add_argument("census", type=Path, help="stage-1 census_<cell>.json")
    parser.add_argument(
        "--raw-scan",
        type=Path,
        required=True,
        help="cell's raw vocab_scan JSON (for the c2 decoder_top join)",
    )
    parser.add_argument(
        "--clean-scan",
        type=Path,
        default=None,
        help="cell's phonological-clean vocab_scan JSON (for c1 in_census/rime)",
    )
    parser.add_argument(
        "--inject-word",
        type=str,
        default=None,
        help="alternative-group inject word (for the informational c2.iii field)",
    )
    parser.add_argument(
        "--cos-threshold",
        type=float,
        default=DEFAULT_COS_THRESHOLD,
        help=f"c3 plan-like cosine threshold (default: {DEFAULT_COS_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path (default: overwrite the input census in place)",
    )
    args = parser.parse_args()

    census = load_json(args.census)
    if census.get("stage") != "rust-census-v1":
        print(
            f"warning: {args.census} has stage={census.get('stage')!r} "
            "(expected 'rust-census-v1'); classifying anyway.",
            file=sys.stderr,
        )

    print(f"Loading raw scan {args.raw_scan} ...", file=sys.stderr)
    decoder_index = build_decoder_top_index(load_json(args.raw_scan))
    print(f"  {len(decoder_index)} features in raw scan", file=sys.stderr)

    clean_index = None
    if args.clean_scan is not None:
        print(f"Loading clean scan {args.clean_scan} ...", file=sys.stderr)
        clean_index = build_clean_index(load_json(args.clean_scan))
        print(f"  {len(clean_index)} phonologically-clean features", file=sys.stderr)

    cmu = cmudict.dict()
    print(f"CMUdict loaded: {len(cmu)} words", file=sys.stderr)

    counts = classify(
        census,
        decoder_index,
        clean_index,
        cmu,
        args.inject_word,
        args.cos_threshold,
    )

    census["classified"] = True
    census["classify_cos_threshold"] = args.cos_threshold
    census["classify_inject_word"] = args.inject_word
    census["classify_raw_scan"] = str(args.raw_scan)
    census["classify_clean_scan"] = (
        str(args.clean_scan) if args.clean_scan is not None else None
    )

    out_path = args.output or args.census
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(census, f, ensure_ascii=False, indent=2)
    print(f"\nWrote classified census to {out_path}", file=sys.stderr)

    # --- Summary: the registered decision looks at these counts ---
    print("\n=== plan-like features per role ===")
    print(f"{'role':<10}{'features':>10}{'plan_like':>12}")
    for role in ("newline", "control", "final"):
        rc = counts.get(role, {"features": 0, "plan_like": 0})
        print(f"{role:<10}{rc['features']:>10}{rc['plan_like']:>12}")
    newline_hits = counts.get("newline", {}).get("plan_like", 0)
    final_hits = counts.get("final", {}).get("plan_like", 0)
    print(
        f"\nCell verdict: newline plan-like = {newline_hits}, "
        f"positive-control (final) plan-like = {final_hits}."
    )
    if newline_hits == 0:
        print("  -> consistent with 'no plan content at the newline' for this cell.")
    else:
        print("  -> plan-like features FOUND at a newline (Exp 1.5 bridge applies).")


if __name__ == "__main__":
    main()
