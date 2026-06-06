#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Classify versified-couplet completions: rhyme vs goal, per base model.

Reads the input couplets (`versified_couplets.json`, for the line-1 `anchor`) and
the per-model `means_ends_prolepsis` outputs, joins them by `prompt`, and for each
greedy completion ``C = top1_token_trimmed`` computes:

- **rhyme**     : ``C`` rhymes with the ``anchor`` (CMUdict *rime* equality);
- **goal-lean** : ``P(W) > P(W')`` (``forced_choice_correct``);
- **dual-hit**  : ``C == W`` (``full_vocab_top1_correct``).

It reports, per model, dual-hit / rhyme / goal-lean rates, the 4-way
{both / rhyme-only / goal-only / neither} split (rhyme x goal-lean), the most common
completions, and a per-family dual-hit breakdown.

Usage:
    python scripts/classify_versified.py \
        docs/experiments/means-ends-prolepsis/versified_couplets.json \
        gemma2_2b=docs/.../versified_gemma2_2b.json \
        llama32_1b=docs/.../versified_llama32_1b.json ...
"""

import json
import sys
from collections import Counter, defaultdict

try:
    from nltk.corpus import cmudict

    _CMU = cmudict.dict()
except (ImportError, LookupError):
    import nltk

    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict

    _CMU = cmudict.dict()


def extract_rime(pron):
    """ARPABET rime = from the last primary-stress vowel onward (fallback: last
    vowel). Mirrors `vocab_scan_cmudict_filter.extract_rime`."""
    last_stressed = -1
    last_vowel = -1
    for i, ph in enumerate(pron):
        if ph and ph[-1].isdigit():
            last_vowel = i
            if ph.endswith("1"):
                last_stressed = i
    pivot = last_stressed if last_stressed >= 0 else last_vowel
    if pivot < 0:
        return None
    return " ".join(pron[pivot:])


def word_rimes(word):
    prons = _CMU.get(word.strip().lower())
    if not prons:
        return set()
    return {r for r in (extract_rime(p) for p in prons) if r}


def rhymes(a, b):
    """True / False / None(=out-of-vocabulary, can't tell)."""
    ra, rb = word_rimes(a), word_rimes(b)
    if not ra or not rb:
        return None
    return len(ra & rb) > 0


def pct(n, d):
    return f"{100.0 * n / d:.0f}%" if d else "—"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        print("Usage: classify_versified.py <couplets.json> <tag=output.json> ...")
        sys.exit(1)

    couplets = json.load(open(sys.argv[1], encoding="utf-8"))
    anchor_by_prompt = {it["prompt"]: it["anchor"] for it in couplets}
    fam_by_prompt = {it["prompt"]: it["family"] for it in couplets}

    summary = {}
    print(f"\nVersified couplets: {len(couplets)} prompts\n")
    header = (
        f"{'model':<12} {'dual-hit':>9} {'rhyme':>7} {'goal-lean':>10} "
        f"{'both':>6} {'rhymeO':>7} {'goalO':>6} {'neither':>8} {'oov':>4}"
    )
    print(header)
    print("-" * len(header))

    for spec in sys.argv[2:]:
        tag, path = spec.split("=", 1)
        data = json.load(open(path, encoding="utf-8"))
        items = data["items"]
        n = len(items)
        dual = rhyme_yes = goal = 0
        both = rhyme_only = goal_only = neither = oov = 0
        comp = Counter()
        fam_dual = defaultdict(lambda: [0, 0])  # family -> [dual_hits, total]
        for it in items:
            c = it["top1_token_trimmed"].strip()
            anchor = anchor_by_prompt.get(it["prompt"], "")
            rh = rhymes(c, anchor)
            g = bool(it["forced_choice_correct"])
            d = bool(it["full_vocab_top1_correct"])
            comp[c] += 1
            fam = fam_by_prompt.get(it["prompt"], "?")
            fam_dual[fam][1] += 1
            if d:
                dual += 1
                fam_dual[fam][0] += 1
            if rh is None:
                oov += 1
            elif rh:
                rhyme_yes += 1
            if g:
                goal += 1
            r_yes = rh is True
            if r_yes and g:
                both += 1
            elif r_yes and not g:
                rhyme_only += 1
            elif (not r_yes) and g:
                goal_only += 1
            else:
                neither += 1
        print(
            f"{tag:<12} {pct(dual, n):>9} {pct(rhyme_yes, n):>7} {pct(goal, n):>10} "
            f"{pct(both, n):>6} {pct(rhyme_only, n):>7} {pct(goal_only, n):>6} "
            f"{pct(neither, n):>8} {oov:>4}"
        )
        summary[tag] = {
            "n": n,
            "dual_hit": dual,
            "rhyme": rhyme_yes,
            "goal_lean": goal,
            "four_way": {"both": both, "rhyme_only": rhyme_only, "goal_only": goal_only, "neither": neither},
            "oov": oov,
            "top_completions": comp.most_common(8),
            "per_family_dual_hit": {f: v for f, v in sorted(fam_dual.items())},
        }

    print("\nTop greedy completions per model:")
    for tag, s in summary.items():
        tops = ", ".join(f"{w!r}:{c}" for w, c in s["top_completions"])
        print(f"  {tag:<12} {tops}")

    # Derive the summary name from the couplets file so v1/v2 don't clobber.
    out_path = sys.argv[1].replace("_couplets.json", "_classified.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
