#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""m1 classifier for Experiment 2 (full-line steering at the newline).

Augments a ``fullline_<cell>.json`` (written by
``examples/figure13_newline_steering.rs``) with the **m1** behavioural metric:
for each condition, classify the final word of each sampled composed line by
CMUdict rime into the **natural** rhyme group, the **inject** rhyme group, or
**other**, and report exact Clopper-Pearson 95% confidence intervals on each
fraction.

The registered Exp-2 criterion (spec): newline planning is *recovered* iff
suppress+inject moves the inject-group fraction above baseline with
non-overlapping 95% CIs (and the greedy line ends in the inject group);
*improvisation* is supported iff no newline condition moves the distribution
beyond CI overlap.

CMUdict token cleaning + rime extraction are imported verbatim from
``vocab_scan_cmudict_filter.py`` (same phonology as the census / vocab scans).

Usage:
    python scripts/newline_steering_classify.py \\
        docs/experiments/figure13-newline/fullline_gemma2-2b-426k.json [more.json ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scipy.stats import beta  # exact Clopper-Pearson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab_scan_cmudict_filter import (  # noqa: E402
    cmudict,
    extract_rime,
    normalise_token,
)

_TRAILING_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+$")


def word_rime(word: str, cmu: dict[str, list[list[str]]]) -> str | None:
    """Rime of an already-normalised word, or ``None`` if unresolved."""
    variants = cmu.get(word)
    if not variants:
        return None
    return extract_rime(variants[0])


def final_word(line: str) -> str:
    """Extract the final word of a composed line: first physical line, strip
    right-side non-alphanumerics, split on whitespace, take the last, lowercase."""
    first = line.split("\n", 1)[0]
    trimmed = _TRAILING_NON_ALNUM.sub("", first)
    parts = trimmed.split()
    return parts[-1].lower() if parts else ""


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson interval for ``k`` successes in ``n`` trials."""
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (round(lo, 4), round(hi, 4))


def classify_group(
    word: str,
    cmu: dict[str, list[list[str]]],
    natural_rime: str | None,
    inject_rime: str | None,
) -> str:
    """Return ``"natural"`` / ``"inject"`` / ``"other"`` for a final word."""
    norm = normalise_token(word) or word
    rime = word_rime(norm, cmu)
    if rime is None:
        return "other"
    # A word can in principle share both rimes only if they are equal; the
    # cells are chosen so natural_rime != inject_rime, so order does not matter.
    if natural_rime is not None and rime == natural_rime:
        return "natural"
    if inject_rime is not None and rime == inject_rime:
        return "inject"
    return "other"


def m1_for_lines(
    lines: list[str],
    cmu: dict[str, list[list[str]]],
    natural_rime: str | None,
    inject_rime: str | None,
) -> dict:
    """Compute the m1 breakdown for one condition's sampled lines."""
    words = [final_word(ln) for ln in lines]
    groups = [classify_group(w, cmu, natural_rime, inject_rime) for w in words]
    n = len(groups)
    out: dict = {"n": n, "final_words": words}
    for grp in ("natural", "inject", "other"):
        k = groups.count(grp)
        lo, hi = clopper_pearson(k, n)
        out[grp] = {
            "count": k,
            "fraction": round(k / n, 4) if n else 0.0,
            "ci95": [lo, hi],
        }
    return out


def process(path: Path, cmu: dict[str, list[list[str]]]) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    natural_word = data.get("suppress_word", "")
    inject_word = data.get("inject_word", "")
    natural_rime = word_rime(normalise_token(natural_word) or natural_word, cmu)
    inject_rime = word_rime(normalise_token(inject_word) or inject_word, cmu)

    print(f"\n=== {path.name} ===")
    print(f"natural: {natural_word!r} (rime {natural_rime})   "
          f"inject: {inject_word!r} (rime {inject_rime})")
    print(f"{'condition':<16}{'natural %':>22}{'inject %':>22}{'other %':>12}")

    for cond in data.get("conditions", []):
        m1 = m1_for_lines(
            cond.get("sampled_lines", []), cmu, natural_rime, inject_rime
        )
        cond["m1"] = m1
        nat, inj, oth = m1["natural"], m1["inject"], m1["other"]
        print(
            f"{cond.get('condition', '?'):<16}"
            f"{nat['fraction'] * 100:>7.1f} "
            f"[{nat['ci95'][0] * 100:>4.0f},{nat['ci95'][1] * 100:>4.0f}]"
            f"{inj['fraction'] * 100:>10.1f} "
            f"[{inj['ci95'][0] * 100:>4.0f},{inj['ci95'][1] * 100:>4.0f}]"
            f"{oth['fraction'] * 100:>11.1f}"
        )

    data["m1_classified"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Registered-criterion hint: compare suppress+inject inject-fraction vs baseline.
    by_name = {c.get("condition"): c for c in data.get("conditions", [])}
    base = by_name.get("baseline", {}).get("m1", {}).get("inject", {})
    si = by_name.get("suppress+inject", {}).get("m1", {}).get("inject", {})
    if base and si:
        base_ci = base["ci95"]
        si_ci = si["ci95"]
        redirected = si_ci[0] > base_ci[1]  # non-overlapping, si above baseline
        verdict = (
            "newline planning RECOVERED (inject fraction up, non-overlapping CIs)"
            if redirected
            else "no newline redirect (CIs overlap) -> improvisation-consistent"
        )
        print(f"  -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="m1: classify Exp-2 sampled composed lines by final-word rime."
    )
    parser.add_argument("fullline", type=Path, nargs="+", help="fullline_<cell>.json file(s)")
    args = parser.parse_args()

    cmu = cmudict.dict()
    print(f"CMUdict loaded: {len(cmu)} words", file=sys.stderr)
    for path in args.fullline:
        process(path, cmu)


if __name__ == "__main__":
    main()
