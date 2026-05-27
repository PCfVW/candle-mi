#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate rhyme-family prompt sets for the Maar contrastive-steering example.

Step B fallback (per the v0.1.12 plan): when Maar et al.'s supplementary
.zip is unavailable, this script generates candle-mi-authored prompt sets
following Maar's *described* structure:

- 85 positive prompts per rhyme family: first-line couplet starts whose
  natural completion is a word from the target rhyme family.
- 85 negative prompts: first lines whose natural completion is a word
  from a different ("contrast") rhyme family.
- 20 held-out evaluation prompts: same structure as positive, with an
  explicit `target_token` (the BPE encoding of the natural completion)
  and `target_rhyme_words` (the full list of rhyme-family words used to
  score `is_hit`).

All prompts use Maar's template: `"A rhyming couplet:\n{line}"`.

Rhyme families are looked up via `nltk.corpus.cmudict` (must be
pre-installed; on this machine it's at
`~/AppData/Roaming/nltk_data/corpora/cmudict/cmudict`).

## Output schema

A JSON file matching `examples/maar_contrastive_steering`'s expected
schema:

```json
{
  "family": "-ee",
  "template": "A rhyming couplet:\\n{line}",
  "positive": [...85 strings...],
  "negative": [...85 strings...],
  "eval":     [{"prompt": ..., "target_token": ..., "target_rhyme_words": [...]}, ...20...],
  "source": "candle-mi-authored",
  "source_url": null
}
```

## Usage

```bash
python scripts/regenerate_rhyme_prompts.py \
    --family -ee \
    --contrast -out \
    --model-family llama32 \
    --output examples/results/maar_contrastive_steering/prompts/llama32_1b_rhyme_ee.json
```

The `--model-family` argument controls the leading-space convention for
the `target_token` field (Llama/Gemma tokenisers prepend a leading space
to standalone words).
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Rhyme-family seeds: maps a human-readable family tag to the CMUdict
# `ARPABET` rime (last stressed vowel onward).  Adding a family means
# adding a row here PLUS hand-writing couplet-stem templates in
# `RHYME_STEMS_BY_FAMILY` below.
RHYME_FAMILIES = {
    "-ee": "IY1",
    "-out": "AW1 T",
    "-ate": "EY1 T",
    "-ight": "AY1 T",
    "-ay": "EY1",
    "-ound": "AW1 N D",
}

# Hand-curated rhyme-family word lists (the set used to score
# `is_hit` on each eval prompt).  Each list intentionally short and
# common so the model's natural top-1 is plausibly in-set even without
# steering.
RHYME_FAMILY_WORDS = {
    "-ee": [" tree", " sea", " free", " three", " bee", " he", " she", " me",
            " key", " agree", " degree", " knee"],
    "-out": [" out", " about", " shout", " doubt", " route", " trout", " scout",
             " spout", " sprout", " stout"],
    "-ate": [" gate", " plate", " state", " fate", " mate", " late", " rate",
             " date", " hate", " skate"],
    "-ight": [" light", " night", " right", " bright", " fight", " sight",
              " white", " might", " flight", " height"],
    "-ay": [" day", " way", " say", " play", " stay", " bay", " gray",
            " ray", " spray", " tray"],
    "-ound": [" sound", " round", " ground", " found", " bound", " hound",
              " mound", " pound", " around", " astound"],
}

# Hand-curated couplet-stem templates per rhyme family.  Each stem is a
# first-line ending that naturally invites the rhyme word as the
# completion of the second line.  Maar uses ~105 per family in their
# paper; for the v0.1.12 fallback we use 85 (their training split size)
# expanded by light templating.
RHYME_STEMS_BY_FAMILY = {
    "-ee": [
        "I saw a robin in the apple {w}",
        "The branches whispered through the tall oak {w}",
        "She climbed the highest, most majestic {w}",
        "We rested in the shade of one wide {w}",
        "He pruned the lowest limb upon the {w}",
        "A heron stood beside the silver {w}",
        "The sailors set their course out toward the {w}",
        "We watched the waves crash on the open {w}",
        "She wrote a letter from across the {w}",
        "The little boat sailed gently on the {w}",
        "He longed to roam the world and to be {w}",
        "The horses ran across the wild and {w}",
        "We danced beneath the stars, our spirits {w}",
        "She felt at last completely safe and {w}",
        "The captive bird at long last was set {w}",
        "I counted one and two and then said {w}",
        "We waited until I had finished counting to {w}",
        "She turned the corner, then she counted {w}",
        "He had not slept for two whole nights, but {w}",
        "The chorus had to sing the part in {w}",
    ],
    "-out": [
        "The lantern light went suddenly clean {w}",
        "She held the secret tight, and would not let it {w}",
        "He raised his voice and tried his best to {w}",
        "We stood and waited for the truth to come {w}",
        "The signal sounded from the dark redoubt and we cried {w}",
        "She asked the simple question, then expressed her {w}",
        "We argued late into the night with little {w}",
        "He paused, considered, and then voiced his {w}",
        "Without a map, the path was very much in {w}",
        "She circled twice, then chose a different {w}",
        "The bus took the long and winding {w}",
        "He fished a small but pretty silver {w}",
        "The scoutmaster called over the nearest {w}",
        "She lifted up the kettle from the {w}",
        "He cut the seedling, then he watched it {w}",
        "The boxer staggered briefly, but stayed {w}",
        "The trees endured, they never seemed to {w}",
    ],
    "-ate": [
        "She arrived at exactly half-past {w}",
        "We waited at the rusty iron {w}",
        "He served the soup upon the silver {w}",
        "The senator addressed the upper {w}",
        "I cannot now control my final {w}",
        "She was his constant friend and trusted {w}",
        "He came in early, never very {w}",
        "We measured at a slow and steady {w}",
        "He set the meeting for a future {w}",
        "She bore no man on earth a moment's {w}",
        "We laced our boots, then went to learn to {w}",
    ],
    "-ight": [
        "The lamp went out, but suddenly we saw a {w}",
        "I closed my eyes and slept the whole long {w}",
        "She knew the answer, knew it had to be {w}",
        "The flag was hoisted, gleaming pure and {w}",
        "He squared up, ready willing now to {w}",
        "I took the binoculars to improve my {w}",
        "The blank page lay before me, blinding {w}",
        "He hoped his answer was indeed the one and only {w}",
        "She measured up to her companion's {w}",
        "We watched the airplane vanish into {w}",
    ],
    "-ay": [
        "The sun came out on this most lovely {w}",
        "She showed me only one improved {w}",
        "He had so very many things to {w}",
        "We watched the children laugh and {w}",
        "I asked her if she would consent to {w}",
        "The water on the shore was steel-grey, in the {w}",
        "He looked up at the colour of the {w}",
        "A single beam of light came as a {w}",
        "The leaves were dappled, sun, then mottled {w}",
        "We saw it on a bright and breezy {w}",
    ],
    "-ound": [
        "The footsteps echoed loudly underground without a {w}",
        "We searched the woods until at last the boy was {w}",
        "She traced the perfect circle, smooth and {w}",
        "He turned the spinning wheel and watched it go {w}",
        "We searched the cave and then at last we {w}",
        "The captive panther leapt up with a {w}",
        "We dug a foot deep into the soft {w}",
        "She left the borders of the camp she'd been {w}",
        "He could not find his way; he was lost and {w}",
        "The dog gave one expectant little {w}",
    ],
}


def collect_words_for_family(family: str, cmudict_entries: list) -> list[str]:
    """Return CMUdict entries whose primary pronunciation ends with the
    family's rime; used only as a sanity check (we curate the actual
    target lists in `RHYME_FAMILY_WORDS`)."""
    rime = RHYME_FAMILIES[family]
    rime_tokens = rime.split()
    rime_len = len(rime_tokens)
    matches = []
    for word, pron in cmudict_entries:
        if len(pron) < rime_len:
            continue
        if pron[-rime_len:] == rime_tokens:
            matches.append(word.lower())
    return sorted(set(matches))


def generate_prompt_set(
    family: str, contrast: str, n_positive: int, n_negative: int, n_eval: int
) -> dict:
    """Generate the {family, template, positive, negative, eval, source}
    JSON dict for the requested family.

    Positive prompts: pick a rhyme word from `family` and substitute into a
    stem template from `RHYME_STEMS_BY_FAMILY[family]` (cycling through
    stems and words to reach `n_positive`).  Negative prompts: same
    process but using stems from `contrast` family.  Eval prompts: held
    out (random subset of family stems not used in positive)."""
    pos_words = [w.strip() for w in RHYME_FAMILY_WORDS[family]]
    neg_words = [w.strip() for w in RHYME_FAMILY_WORDS[contrast]]
    pos_stems = RHYME_STEMS_BY_FAMILY[family]
    neg_stems = RHYME_STEMS_BY_FAMILY[contrast]

    rng = random.Random(20260528)

    def expand(stems: list[str], words: list[str], n: int) -> list[str]:
        out = []
        for i in range(n):
            stem = stems[i % len(stems)]
            word = words[(i * 3 + 1) % len(words)]
            out.append(f"A rhyming couplet:\n{stem.format(w=word)}")
        return out

    positive = expand(pos_stems, pos_words, n_positive)
    negative = expand(neg_stems, neg_words, n_negative)

    # Held-out eval: re-use one canonical stem per rhyme word, with the
    # rhyme word swapped out for an "open" placeholder that the model
    # is supposed to complete.
    eval_stems = pos_stems[: min(n_eval, len(pos_stems))]
    eval_words = pos_words[: min(n_eval, len(pos_words))]
    rng.shuffle(eval_stems)
    rng.shuffle(eval_words)
    eval_prompts = []
    for i, stem in enumerate(eval_stems[:n_eval]):
        # Replace "{w}" with empty; the prompt ends mid-line and the model
        # must produce the rhyme word as its next token.
        target_word = eval_words[i % len(eval_words)]
        prompt_body = stem.replace(" {w}", "").rstrip()
        prompt = f"A rhyming couplet:\n{prompt_body}"
        eval_prompts.append(
            {
                "prompt": prompt,
                "target_token": target_word if target_word.startswith(" ") else f" {target_word}",
                "target_rhyme_words": RHYME_FAMILY_WORDS[family],
            }
        )

    return {
        "family": family,
        "template": "A rhyming couplet:\n{line}",
        "positive": positive,
        "negative": negative,
        "eval": eval_prompts,
        "source": "candle-mi-authored",
        "source_url": None,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Generate Maar-style rhyme-family prompt sets (fallback when "
        "Maar supplementary unavailable)."
    )
    parser.add_argument(
        "--family",
        choices=list(RHYME_FAMILIES.keys()),
        required=True,
        help="Target rhyme family (e.g. -ee, -out, -ate, -ight, -ay, -ound).",
    )
    parser.add_argument(
        "--contrast",
        choices=list(RHYME_FAMILIES.keys()),
        default=None,
        help="Contrast rhyme family for negative prompts.  Defaults to "
        "a sensible non-overlapping choice per --family.",
    )
    parser.add_argument(
        "--n-positive",
        type=int,
        default=85,
        help="Number of positive prompts (Maar's training set size).",
    )
    parser.add_argument(
        "--n-negative",
        type=int,
        default=85,
        help="Number of negative prompts (Maar's training set size).",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=20,
        help="Number of held-out eval prompts (Maar's test split size).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--validate-cmudict",
        action="store_true",
        help="Sanity-check rhyme-family word lists against CMUdict.",
    )
    args = parser.parse_args()

    contrast_defaults = {
        "-ee": "-out",
        "-out": "-ee",
        "-ate": "-ound",
        "-ight": "-ay",
        "-ay": "-ight",
        "-ound": "-ate",
    }
    contrast = args.contrast or contrast_defaults[args.family]

    if args.validate_cmudict:
        try:
            from nltk.corpus import cmudict
            entries = cmudict.entries()
        except ImportError:
            sys.stderr.write(
                "ERROR: nltk not installed; cannot validate CMUdict.\n"
            )
            return 2
        family_words = collect_words_for_family(args.family, entries)
        print(f"CMUdict words ending with rime {RHYME_FAMILIES[args.family]}: "
              f"{len(family_words)} (sample: {family_words[:8]})")

    out = generate_prompt_set(
        args.family, contrast, args.n_positive, args.n_negative, args.n_eval
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(
        f"Wrote {args.output} "
        f"({len(out['positive'])} positive, {len(out['negative'])} negative, "
        f"{len(out['eval'])} eval) — family {args.family} vs contrast {contrast}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
