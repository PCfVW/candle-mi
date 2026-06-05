#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate goal-contrastive means-ends planning items.

Step A (linguistic cell) of the prolepsis-in-planning experiment: emit a JSON
list of means-ends operator-selection items where each prompt states a current
state and a goal, then ends at the planning site (no trailing space). The
model's next token should be the goal-correct single-token action.

## Goal-contrastive design

Every device appears in **both directions** of its action family, e.g. for the
`on/off` family the lamp yields one item whose goal calls for `on` and one whose
goal calls for `off`. The set is **balanced** across the six (family, token)
groups, so the "override the lexical default" sides (`off`, `shut`, `down`) carry
equal weight — those are the cases that distinguish genuine goal-conditioning
from emitting the high-frequency collocation ("turn the lamp on").

## Planning-site convention (must match the Rust scorer)

Each prompt ends at a content token with **no trailing space**. A trailing space
tokenizes as a standalone metaspace token, after which the space-prefixed answer
token is unreachable. The model's next token is the space-prefixed action token
(scored via `find_token_id`).

## Action families

Three families, each a contrastive single-token pair and a stem ending at the
planning site:

- `on/off`     — `"... Turn the {device}"`       -> `on` | `off`
- `open/shut`  — `"... The {device} should be"`  -> `open` | `shut`
- `up/down`    — `"... The {device} should be"`  -> `up` | `down`

(Short, high-frequency tokens chosen so both sides are single tokens; the Rust
scorer validates this and reports any that are not.)

## Output schema (flat list)

```json
[{"prompt": str, "correct": str, "alternative": str, "family": str}, ...]
```

## Usage

```bash
python scripts/means_ends_generator.py --num-instances 200 --seed 0 \
    --output docs/experiments/means-ends-prolepsis/means_ends_items.json
```
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OUTPUT = "docs/experiments/means-ends-prolepsis/means_ends_items.json"

# Each family: token pair (a, b), the stem ending at the planning site, and a
# device -> {a_goal, b_goal} table. The state clause shows the OPPOSITE token
# (you act to change it): for the `a` direction the device currently shows `b`.
FAMILIES = {
    "on_off": {
        "tokens": ("on", "off"),
        "stem": "Turn the {device}",
        "devices": {
            "lamp": ("We want the room to be bright", "We want the room to be dark"),
            "heater": ("We want the room to be warm", "We want the room to be cool"),
            "fan": ("We want the air to keep moving", "We want the air to stay still"),
            "radio": ("We want to hear some music", "We want complete silence"),
            "television": ("We want to watch the news", "We want some quiet"),
            "oven": ("We want to bake the bread", "We want to stop the heat"),
            "kettle": ("We want to boil the water", "We want to stop boiling"),
            "computer": ("We want to start working", "We want to save power"),
            "printer": ("We want to print a page", "We want to save power"),
            "speaker": ("We want to play the song", "We want it to be silent"),
            "projector": ("We want to show the slides", "We want a dark screen"),
            "stove": ("We want to cook dinner", "We want to stop cooking"),
        },
    },
    "open_closed": {
        "tokens": ("open", "closed"),
        "stem": "The {device} should be",
        "devices": {
            "door": ("We want to walk through", "We want some privacy"),
            "window": ("We want some fresh air", "We want to keep out the cold"),
            "gate": ("We want to let the car in", "We want to keep the dog inside"),
            "valve": ("We want to let the water flow", "We want to stop the leak"),
            "lid": ("We want to reach inside", "We want to keep it sealed"),
            "curtain": ("We want to let the sunlight in", "We want to block the glare"),
            "drawer": ("We want to get a spoon", "We want to tidy the desk"),
            "cabinet": ("We want to take a plate", "We want to hide the clutter"),
            "jar": ("We want to take a cookie", "We want to keep it fresh"),
            "trunk": ("We want to load the bags", "We want to drive away"),
            "hatch": ("We want to climb out", "We want to stay warm"),
            "box": ("We want to see what is inside", "We want to keep it safe"),
        },
    },
    "up_down": {
        "tokens": ("up", "down"),
        "stem": "The {device} should be",
        "devices": {
            "shade": ("We want to see the view", "We want to keep the room cool"),
            "roller blind": ("We want to let the sunlight in", "We want to block the bright sun"),
            "volume": ("We want to hear the song clearly", "We want to avoid waking the baby"),
            "thermostat": ("We want to make the room warmer", "We want to make the room cooler"),
            "car window": ("We want to keep the rain out", "We want to get some fresh air"),
            "garage door": ("We want to drive the car out", "We want to secure the garage"),
            "window blind": ("We want to let the morning light in", "We want to darken the room for sleep"),
            "projector screen": ("We want to use the whiteboard behind it", "We want to show the movie"),
            "hospital bed": ("We want to sit up and eat", "We want to lie flat and rest"),
            "recliner": ("We want to sit upright to work", "We want to lie back and relax"),
            "tray table": ("We want to get ready for landing", "We want to eat the meal"),
            "seat back": ("We want to prepare for takeoff", "We want to recline and rest"),
        },
    },
}


def render(device, state_token, goal, stem, template_idx):
    """Render one prompt; ends at the planning site (no trailing space)."""
    state = f"The {device} is {state_token}."
    stem_text = stem.format(device=device)
    if template_idx == 0:
        return f"{state} {goal}. {stem_text}"
    if template_idx == 1:
        return f"{goal}. {state} {stem_text}"
    return f"Right now, the {device} is {state_token}. {goal}. {stem_text}"


def build_all():
    """Build every candidate item, grouped by (family, correct_token)."""
    by_group = {}
    for family, spec in FAMILIES.items():
        tok_a, tok_b = spec["tokens"]
        stem = spec["stem"]
        for device, (goal_a, goal_b) in spec["devices"].items():
            # (correct, alternative, opposite-state token, goal text)
            for correct, alternative, goal in (
                (tok_a, tok_b, goal_a),
                (tok_b, tok_a, goal_b),
            ):
                for template_idx in range(3):
                    prompt = render(device, alternative, goal, stem, template_idx)
                    item = {
                        "prompt": prompt,
                        "correct": correct,
                        "alternative": alternative,
                        "family": family,
                    }
                    by_group.setdefault((family, correct), []).append(item)
    return by_group


def balanced_sample(by_group, num_instances, rng):
    """Draw a (family, token)-balanced sample, remainder distributed round-robin."""
    groups = sorted(by_group)
    base, remainder = divmod(num_instances, len(groups))
    chosen = []
    for i, key in enumerate(groups):
        want = base + (1 if i < remainder else 0)
        pool = by_group[key]
        if want > len(pool):
            raise ValueError(
                f"group {key} has only {len(pool)} candidates but {want} requested "
                f"(add devices/templates or lower --num-instances)"
            )
        chosen.extend(rng.sample(pool, want))
    rng.shuffle(chosen)
    return chosen


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()

    if args.num_instances < 1:
        parser.error("--num-instances must be >= 1")

    rng = random.Random(args.seed)
    by_group = build_all()
    items = balanced_sample(by_group, args.num_instances, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
        f.write("\n")

    fam_counts = Counter(i["family"] for i in items)
    tok_counts = Counter((i["family"], i["correct"]) for i in items)
    print(
        f"Wrote {len(items)} items (seed {args.seed}) to {args.output}",
        file=sys.stderr,
    )
    print(f"Per-family: {dict(sorted(fam_counts.items()))}", file=sys.stderr)
    summary = ", ".join(f"{fam}:{tok}={n}" for (fam, tok), n in sorted(tok_counts.items()))
    print(f"Per-(family,token): {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
