#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate gridworld instances with a single unambiguous correct first move.

Step 0 (Gap A) of `docs/roadmaps/PLAN-GRIDWORLD-PROLEPSIS.md`: emit a JSON list
of simple 2D-gridworld planning instances, each with a unique ground-truth
first action, for the gridworld-prolepsis transfer experiment.

## Coordinate / action convention

A cell is `(x, y)` with `0 <= x, y < N`.  With `dx = goal_x - agent_x` and
`dy = goal_y - agent_y`:

- `+x -> Right`, `-x -> Left`, `+y -> Up`, `-y -> Down`.
- `|dx| > |dy|`  -> a horizontal move (`Right` if `dx > 0` else `Left`).
- `|dy| > |dx|`  -> a vertical move (`Up` if `dy > 0` else `Down`).
- `|dx| == |dy|` (which includes `agent == goal`) -> the dominant action is
  **not unique**, so the instance is dropped.

This is the Manhattan-distance-dominant-action rule from the plan: agent
`(2, 3)` to goal `(4, 4)` has `dx = 2, dy = 1` -> `Right` (kept); agent
`(3, 3)` to goal `(4, 4)` has `dx = 1, dy = 1` -> dropped.

## Output schema (flat list, exactly as the plan specifies)

```json
[
  {"agent": [x, y], "goal": [x, y], "correct_action": "Up", "instance_id": 0},
  ...
]
```

The grid size is NOT stored in the file; the Rust prompt formatter takes its
own `--grid-size`.  Keep the two `--grid-size` values in sync.

## Balancing

By default the sample is **balanced**: an equal number of instances per
cardinal action (any remainder is distributed round-robin), so the Step C
per-action-class comparison has even support.  `--no-balanced` falls back to a
plain shuffled draw from the unique-action pool.

## Usage

```bash
python scripts/gridworld_generator.py \
    --grid-size 5 --num-instances 100 --seed 0 \
    --output docs/experiments/gridworld-prolepsis/gridworld_instances.json
```
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Cardinal actions in a fixed order (matches the Rust `CardinalAction` enum).
ACTIONS = ["Up", "Down", "Left", "Right"]

DEFAULT_OUTPUT = "docs/experiments/gridworld-prolepsis/gridworld_instances.json"


def dominant_action(agent, goal):
    """Return the unique dominant cardinal action, or None if ambiguous.

    `agent` and `goal` are `(x, y)` tuples.  Returns one of "Up", "Down",
    "Left", "Right", or None when `|dx| == |dy|` (no strict dominance).
    """
    dx = goal[0] - agent[0]
    dy = goal[1] - agent[1]
    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"
    if abs(dy) > abs(dx):
        return "Up" if dy > 0 else "Down"
    return None


def build_pool(grid_size):
    """Build the pool of unique-dominant-action instances on an N x N grid.

    Returns a dict mapping each action label to a list of `(agent, goal)`
    tuples whose strictly dominant action is that label.
    """
    by_action = defaultdict(list)
    for ax in range(grid_size):
        for ay in range(grid_size):
            for gx in range(grid_size):
                for gy in range(grid_size):
                    agent = (ax, ay)
                    goal = (gx, gy)
                    action = dominant_action(agent, goal)
                    if action is not None:
                        by_action[action].append((agent, goal))
    return by_action


def balanced_sample(by_action, num_instances, rng):
    """Draw a per-action-balanced sample of `(agent, goal, action)` triples.

    Splits `num_instances` as evenly as possible across the four actions,
    distributing any remainder round-robin in the fixed `ACTIONS` order.
    Raises `ValueError` if a class lacks enough distinct pairs.
    """
    base, remainder = divmod(num_instances, len(ACTIONS))
    chosen = []
    for i, action in enumerate(ACTIONS):
        want = base + (1 if i < remainder else 0)
        pool = by_action.get(action, [])
        if want > len(pool):
            raise ValueError(
                f"action {action!r} has only {len(pool)} unique-dominant "
                f"instances but {want} were requested (raise --grid-size or "
                f"lower --num-instances)"
            )
        for agent, goal in rng.sample(pool, want):
            chosen.append((agent, goal, action))
    rng.shuffle(chosen)
    return chosen


def unbalanced_sample(by_action, num_instances, rng):
    """Draw a plain shuffled sample of `(agent, goal, action)` triples."""
    flat = [
        (agent, goal, action)
        for action, pairs in by_action.items()
        for agent, goal in pairs
    ]
    if num_instances > len(flat):
        raise ValueError(
            f"requested {num_instances} instances but only {len(flat)} "
            f"unique-dominant pairs exist on a {len(by_action)}-action grid"
        )
    return rng.sample(flat, num_instances)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="equal instances per cardinal action (default: balanced)",
    )
    args = parser.parse_args()

    if args.grid_size < 2:
        parser.error("--grid-size must be >= 2")
    if args.num_instances < 1:
        parser.error("--num-instances must be >= 1")

    rng = random.Random(args.seed)
    by_action = build_pool(args.grid_size)

    if args.balanced:
        chosen = balanced_sample(by_action, args.num_instances, rng)
    else:
        chosen = unbalanced_sample(by_action, args.num_instances, rng)

    instances = [
        {
            "agent": [agent[0], agent[1]],
            "goal": [goal[0], goal[1]],
            "correct_action": action,
            "instance_id": instance_id,
        }
        for instance_id, (agent, goal, action) in enumerate(chosen)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2)
        f.write("\n")

    counts = Counter(inst["correct_action"] for inst in instances)
    summary = ", ".join(f"{a}={counts.get(a, 0)}" for a in ACTIONS)
    print(
        f"Wrote {len(instances)} instances "
        f"(grid {args.grid_size}x{args.grid_size}, seed {args.seed}, "
        f"{'balanced' if args.balanced else 'unbalanced'}) to {args.output}",
        file=sys.stderr,
    )
    print(f"Per-action counts: {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
