#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Convert Maar's supplementary `rhyme_family_lines.json` to candle-mi's
prompts schema for `examples/maar_contrastive_steering`.

Maar et al. (2026) "What's the plan?" (ICLR 2026, arXiv 2601.20164,
OpenReview Z10pxu0Q7X) ship their experimental data inside the paper's
supplementary `.zip`.  Specifically, the supplementary contains:

- `paper_experiments/data/train/rhyme_family_lines.json` — 85 lines per
  rhyme family used to build the contrastive direction (positive set for
  the family of interest; negative set for any contrasting family).
- `paper_experiments/data/test/rhyme_family_lines.json` — 20 held-out
  lines per family used as eval prompts.
- `paper_experiments/shared_utils.py` — contains the `rhyme_family_words`
  dictionary mapping each family code (`ee`, `oat`, `air`, ...) to a list
  of canonical rhyme words used in the `is_hit` family-membership check.

This script converts those three inputs into the four `*_maar.json` files
the `maar_contrastive_steering` example consumes (one per cell):

- `prompts/llama32_3b_rhyme_ee_maar.json`  — `-ee` vs `-oat` contrast
- `prompts/llama32_1b_rhyme_ee_maar.json`  — `-ee` vs `-oat` contrast
- `prompts/gemma2_rhyme_ee_maar.json`      — `-ee` vs `-oat` contrast
- `prompts/gemma2_rhyme_oat_maar.json`     — `-oat` vs `-ee` contrast

Each output JSON carries `"source": "maar-supplementary"` and a
`"source_url"` pointing at OpenReview, so reviewers can audit
provenance without us committing Maar's 60 MB code drop (which the
`.gitignore` excludes via the `maar_supp/` entry added in Commit 4).

Note: Maar's family taxonomy has no `-out` family (despite our
candle-mi-authored `gemma2_rhyme_out.json` using that label).  The
10 Maar families are `air ake ee ight ing ip ird it oat ow`.  We use
`-oat` as the `-ee` contrast (long-o vs long-e, phonologically distant
and well-populated in their word lists).

## Usage

```bash
# Default: reads from examples/results/maar_contrastive_steering/maar_supp/
# and writes to examples/results/maar_contrastive_steering/prompts/.
python scripts/convert_maar_prompts.py

# Override paths:
python scripts/convert_maar_prompts.py \
    --base-dir <path-to-supplementary_material> \
    --out-dir <path-to-output-directory>

# Dry-run (print what would be written, do not touch disk):
python scripts/convert_maar_prompts.py --dry-run
```

## Implementation notes

- `rhyme_family_words` is extracted from `shared_utils.py` by regex
  (the source uses `set([...])` literals) and parsed via
  `ast.literal_eval` after rewriting `set([...])` → `[...]`.  This
  avoids using `eval` on untrusted input.
- All file IO uses UTF-8 with `ensure_ascii=False` so apostrophes and
  diacritics in Maar's word lists (e.g. `"'e"`) round-trip correctly.
- The output schema matches `examples/results/maar_contrastive_steering/README.md`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

SOURCE_URL = (
    "https://openreview.net/attachment?id=Z10pxu0Q7X&name=supplementary_material"
)

# (family, contrast, output_filename) triples for the four committed cells.
# Cell list is intentionally hard-coded: it matches the four presets in
# examples/maar_contrastive_steering.rs's `select_preset` table.
CELLS = [
    ("ee",  "oat", "llama32_3b_rhyme_ee_maar.json"),
    ("ee",  "oat", "llama32_1b_rhyme_ee_maar.json"),
    ("ee",  "oat", "gemma2_rhyme_ee_maar.json"),
    ("oat", "ee",  "gemma2_rhyme_oat_maar.json"),
]


def parse_rhyme_family_words(shared_utils_src: str) -> dict[str, list[str]]:
    """Extract `rhyme_family_words = {...}` from `shared_utils.py` source.

    Maar's `rhyme_family_words` is a dict mapping family code to
    `set([...])`.  We rewrite the `set([...])` calls to plain list
    literals, then parse via `ast.literal_eval` (safe: refuses anything
    other than Python literal constants).
    """
    m = re.search(
        r"rhyme_family_words\s*=\s*(\{.*?^\})",
        shared_utils_src,
        flags=re.DOTALL | re.MULTILINE,
    )
    if m is None:
        raise ValueError(
            "convert_maar_prompts: rhyme_family_words dict not found in shared_utils.py "
            "(expected `rhyme_family_words = {...}` at module scope)"
        )
    literal = m.group(1)
    # Rewrite `set([...])` and `set( [ ... ] )` → `[...]` (literal list).
    literal = re.sub(r"set\(\s*\[", "[", literal)
    literal = re.sub(r"\]\s*\)", "]", literal)
    parsed = ast.literal_eval(literal)
    if not isinstance(parsed, dict):
        raise TypeError(
            f"convert_maar_prompts: expected dict, got {type(parsed).__name__}"
        )
    return {k: sorted(v) for k, v in parsed.items()}


def build_cell(
    *,
    family: str,
    contrast: str,
    train: dict[str, list[str]],
    test: dict[str, list[str]],
    rhyme_family_words: dict[str, list[str]],
) -> dict:
    """Construct one prompts-JSON dict for a (family, contrast) pair."""
    for required in (family, contrast):
        if required not in train:
            raise KeyError(f"convert_maar_prompts: '{required}' missing from train")
        if required not in test:
            raise KeyError(f"convert_maar_prompts: '{required}' missing from test")
        if required not in rhyme_family_words:
            raise KeyError(
                f"convert_maar_prompts: '{required}' missing from rhyme_family_words"
            )
    words = rhyme_family_words[family]
    return {
        "family": f"-{family}",
        "template": "{line}\n",
        "positive": train[family],
        "negative": train[contrast],
        "eval": [
            {
                "prompt": line,
                "target_token": f" {words[0]}",
                "target_rhyme_words": [f" {w}" for w in words],
            }
            for line in test[family]
        ],
        "source": "maar-supplementary",
        "source_url": SOURCE_URL,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Maar's supplementary prompts to candle-mi's schema."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            "examples/results/maar_contrastive_steering/maar_supp/supplementary_material"
        ),
        help="Path to the extracted supplementary_material directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/results/maar_contrastive_steering/prompts"),
        help="Output directory for the converted prompts JSONs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written; do not touch disk.",
    )
    args = parser.parse_args()

    base: Path = args.base_dir
    out_dir: Path = args.out_dir

    train_path = base / "paper_experiments" / "data" / "train" / "rhyme_family_lines.json"
    test_path = base / "paper_experiments" / "data" / "test" / "rhyme_family_lines.json"
    shared_path = base / "paper_experiments" / "shared_utils.py"

    for p in (train_path, test_path, shared_path):
        if not p.is_file():
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
            return 2

    with train_path.open(encoding="utf-8") as f:
        train = json.load(f)
    with test_path.open(encoding="utf-8") as f:
        test = json.load(f)
    with shared_path.open(encoding="utf-8") as f:
        shared_src = f.read()

    rhyme_family_words = parse_rhyme_family_words(shared_src)
    print(
        f"Parsed rhyme_family_words: {len(rhyme_family_words)} families "
        f"({sorted(rhyme_family_words.keys())})"
    )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for family, contrast, filename in CELLS:
        cell = build_cell(
            family=family,
            contrast=contrast,
            train=train,
            test=test,
            rhyme_family_words=rhyme_family_words,
        )
        target = out_dir / filename
        size_summary = (
            f"{len(cell['positive'])} positive + "
            f"{len(cell['negative'])} negative + "
            f"{len(cell['eval'])} eval"
        )
        if args.dry_run:
            print(f"[dry-run] would write {target}  ({size_summary})")
        else:
            with target.open("w", encoding="utf-8") as f:
                json.dump(cell, f, indent=2, ensure_ascii=False)
            print(f"wrote {target}  ({size_summary})")

    return 0


if __name__ == "__main__":
    # Windows console default is cp1252; force UTF-8 so apostrophes don't
    # crash when we print Maar's word lists.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    sys.exit(main())
