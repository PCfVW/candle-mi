#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Phonological clustering filter for ``vocab_scan`` JSON output.

Reads the JSON produced by ``examples/vocab_scan.rs``, looks up each
top-K token's pronunciation in `CMUdict` (via NLTK), groups tokens by
**rime** (last stressed vowel + everything after, per ARPABET; e.g.
``IY1 T`` for "feet" / "meet" / "street"), and flags features that are
"phonologically clean" — defined as having at least ``--rime-threshold``
fraction of their CMU-resolvable top-K tokens sharing the same rime.

The output is a labelled JSON file plus a stdout summary of the count
per rhyme group, suitable for the May 29 afternoon decision point
(``N ≥ 10 phonologically-clean features`` → figure13 sweep;
``N < 10`` → resolution-boundary writeup).

Methodology:

1. Load ``vocab_scan`` JSON.
2. For each feature, take its ``top_K`` tokens (after the optional
   ``--top-k`` cap).
3. Decode each token's text, strip BPE whitespace markers, lowercase,
   normalise common boundaries (leading underscore from ``Ġ`` decode
   variants, etc.).
4. Resolve each cleaned token to a `CMUdict` entry; skip tokens not
   found in `CMUdict` (e.g. punctuation, non-English subwords, code
   identifiers).
5. Extract the **rime** of each resolved word: the substring of the
   ARPABET pronunciation starting at the last primary-stress vowel
   (`*1`) and continuing to the end.  Tokens without a primary stress
   fall back to "last vowel onwards".
6. Compute the **dominant rime** of each feature: the rime with the
   most CMU-resolvable top-K tokens sharing it.  ``share = count(dominant)
   / count(resolvable)``.
7. Feature is "phonologically clean" iff ``share >= rime_threshold``
   AND ``count(dominant) >= --min-cluster-size`` (default 3).

Dependencies: ``nltk`` (with ``cmudict`` corpus downloaded; if not,
the script will fetch it via ``nltk.download('cmudict')``).

Usage:
    python scripts/vocab_scan_cmudict_filter.py <input.json> \\
        [--top-k N] [--rime-threshold T] [--min-cluster-size N] \\
        [--output out.json]

Output:
    stdout: summary table (top rhyme groups + total N count).
    --output (optional): annotated JSON with ``cmudict_label`` and
    ``cmudict_rime`` fields added to each feature.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from nltk.corpus import cmudict
    # Try to access the corpus; trigger a download if missing.
    _ = cmudict.dict()
except LookupError:
    import nltk  # type: ignore[import-untyped]

    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict  # noqa: E402


# Tokens that are common BPE artefacts; we strip them before CMU lookup.
# Most modern tokenizers (Llama 3, Qwen3, Gemma 2) prefix new-word tokens
# with a leading space or a metaspace marker; the rust-side decoder
# usually emits a literal leading space.
_LEADING_SPACE_RE = re.compile(r"^[ ▁ĠĠ]+")
_TRAILING_PUNCT_RE = re.compile(r"[^a-zA-Z']+$")
_NON_LETTERS_RE = re.compile(r"[^a-zA-Z']")


def normalise_token(text: str) -> str | None:
    """Return a lowercase, whitespace-stripped word suitable for CMUdict
    lookup, or `None` if the token is non-alphabetic / not word-like.

    Examples:
        " Paris" -> "paris"
        "Ġfeet" -> "feet"
        "▁meet" -> "meet"
        "." -> None
        "123" -> None
        "rpc" -> "rpc"  (kept; CMUdict lookup will then fail and the
                         token is filtered out at lookup time)
    """
    cleaned = _LEADING_SPACE_RE.sub("", text)
    cleaned = _TRAILING_PUNCT_RE.sub("", cleaned)
    cleaned = cleaned.lower().strip()
    if not cleaned:
        return None
    if _NON_LETTERS_RE.search(cleaned):
        return None
    if len(cleaned) < 2:
        return None
    return cleaned


def extract_rime(pron: list[str]) -> str | None:
    """Extract the rime from an ARPABET pronunciation: from the last
    primary-stress (`*1`) vowel onwards.  Falls back to the last vowel
    if no primary stress is marked (rare in CMUdict).

    Examples:
        ['F', 'IY1', 'T']           -> 'IY1 T'
        ['S', 'T', 'R', 'IY1', 'T'] -> 'IY1 T'
        ['M', 'EH1', 'T']           -> 'EH1 T'
        ['ER0']                     -> 'ER0' (fallback path)
    """
    last_stressed = -1
    last_vowel = -1
    for i, ph in enumerate(pron):
        # ARPABET vowels end with a digit (0/1/2 for stress).
        if ph and ph[-1].isdigit():
            last_vowel = i
            if ph.endswith("1"):
                last_stressed = i
    pivot = last_stressed if last_stressed >= 0 else last_vowel
    if pivot < 0:
        return None
    return " ".join(pron[pivot:])


def feature_dominant_rime(
    top_tokens: list[dict],
    cmu: dict[str, list[list[str]]],
    top_k: int,
) -> tuple[str | None, int, int, list[str]]:
    """Return ``(dominant_rime, unique_word_count, n_unique_resolved,
    sample_words)`` for one feature's top-K tokens.

    BPE variants of the same word (` the`, `The`, `_the`, `.the`, etc.)
    collapse to one normalised entry — phonological clusters should be
    distinct words sharing a rime, not the same word in multiple
    tokenisations.

    `dominant_rime` is the rime with the most *unique normalised words*
    among CMU-resolvable top-K tokens.  `unique_word_count` is the
    number of unique words sharing that rime.  `n_unique_resolved` is
    the total number of unique CMU-resolvable words in the top-K
    (denominator for the rime-share fraction).  `sample_words` lists
    up to 5 of the dominant-rime words for downstream display.

    Returns ``(None, 0, 0, [])`` when no tokens resolve.
    """
    # Group resolvable top-K tokens by rime, keeping unique words only.
    rime_to_words: dict[str, set[str]] = defaultdict(set)
    seen_words: set[str] = set()
    for entry in top_tokens[:top_k]:
        text = entry.get("text", "")
        word = normalise_token(text)
        if word is None:
            continue
        if word in seen_words:
            continue
        pron_variants = cmu.get(word)
        if not pron_variants:
            seen_words.add(word)
            continue
        # Use the first pronunciation variant (CMUdict orders by frequency).
        rime = extract_rime(pron_variants[0])
        if rime is None:
            seen_words.add(word)
            continue
        rime_to_words[rime].add(word)
        seen_words.add(word)
    n_unique_resolved = sum(len(w) for w in rime_to_words.values())
    if not rime_to_words:
        return (None, 0, n_unique_resolved, [])
    dominant_rime, dominant_words = max(
        rime_to_words.items(), key=lambda kv: len(kv[1])
    )
    sample = sorted(dominant_words)[:5]
    return (dominant_rime, len(dominant_words), n_unique_resolved, sample)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter vocab_scan output by CMUdict phonological clusters"
    )
    parser.add_argument("input", type=Path, help="vocab_scan JSON output")
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top tokens per feature to consider (default: 20)",
    )
    parser.add_argument(
        "--rime-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum share of CMU-resolved top-K tokens that must agree "
            "on the dominant rime for the feature to be flagged "
            "phonologically clean (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help=(
            "Minimum absolute count of top-K tokens sharing the dominant "
            "rime (default: 3)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional annotated JSON output path",
    )
    parser.add_argument(
        "--clean-only-output",
        action="store_true",
        help=(
            "When set, the --output JSON contains ONLY the phonologically-"
            "clean features (commit-friendly subset, typically a few MB) "
            "instead of every feature in the scan (typically 100s of MB to "
            "1+ GB).  Rhyme-group counts + scan parameters are always "
            "included in the header regardless of this flag."
        ),
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    # `encoding="utf-8"` is mandatory on Windows — the default cp1252 cannot
    # decode multilingual subword tokens (CJK, emoji, etc.) that appear in
    # Qwen3 / Llama 3 / Gemma 2 vocab scans.
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    cmu = cmudict.dict()
    print(f"CMUdict loaded: {len(cmu)} words", file=sys.stderr)

    features = data.get("features", [])
    print(f"Scanning {len(features)} features (top_k={args.top_k}) ...", file=sys.stderr)

    # Annotate every feature with its dominant rime + share; flag clean.
    # All counts are over UNIQUE NORMALISED WORDS (not raw tokens) — BPE
    # variants of the same word do not inflate a rime cluster.
    clean_features: list[dict] = []
    rime_to_features: dict[str, list[dict]] = defaultdict(list)
    for feat in features:
        top_tokens = feat.get("top_tokens", [])
        dominant, count, n_unique, sample = feature_dominant_rime(
            top_tokens, cmu, args.top_k
        )
        share = count / n_unique if n_unique > 0 else 0.0
        feat["cmudict_rime"] = dominant
        feat["cmudict_rime_unique_word_count"] = count
        feat["cmudict_unique_resolved_count"] = n_unique
        feat["cmudict_rime_share"] = round(share, 4)
        feat["cmudict_rime_sample_words"] = sample
        is_clean = (
            dominant is not None
            and count >= args.min_cluster_size
            and share >= args.rime_threshold
        )
        feat["cmudict_clean"] = is_clean
        if is_clean and dominant is not None:
            clean_features.append(feat)
            rime_to_features[dominant].append(feat)

    # Summary table.
    n_clean = len(clean_features)
    print(file=sys.stderr)
    print(
        f"Phonologically-clean features: N = {n_clean} "
        f"(threshold: share >= {args.rime_threshold}, count >= {args.min_cluster_size})",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print("Top rhyme groups by feature count:", file=sys.stderr)
    sorted_rimes = sorted(
        rime_to_features.items(), key=lambda kv: -len(kv[1])
    )
    for rime, feats in sorted_rimes[:30]:
        # Show diverse sample words (drawn from the dominant-rime samples
        # already computed per feature).
        seen: set[str] = set()
        diverse: list[str] = []
        for f in feats:
            for w in f.get("cmudict_rime_sample_words", []):
                if w not in seen:
                    seen.add(w)
                    diverse.append(w)
                if len(diverse) >= 6:
                    break
            if len(diverse) >= 6:
                break
        sample_str = ", ".join(diverse)
        print(
            f"  {rime:<14}  {len(feats):4d} features  words: {sample_str}",
            file=sys.stderr,
        )

    # JSON output.
    output_features = clean_features if args.clean_only_output else features
    output_data = {
        "input": str(args.input),
        "top_k": args.top_k,
        "rime_threshold": args.rime_threshold,
        "min_cluster_size": args.min_cluster_size,
        "n_features_scanned": len(features),
        "n_phonologically_clean": n_clean,
        "clean_only_output": args.clean_only_output,
        "rhyme_group_counts": {rime: len(feats) for rime, feats in sorted_rimes},
        "features": output_features,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        size_kb = args.output.stat().st_size / 1024
        print(
            f"\nAnnotated JSON written to {args.output} ({size_kb:.1f} KB)",
            file=sys.stderr,
        )

    # Final line on stdout for shell consumption.
    print(n_clean)


if __name__ == "__main__":
    main()
