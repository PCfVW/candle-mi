#!/usr/bin/env bash
# Exp 3b random-model control (the "dead-salmon" test): build Gemma 2 2B from
# config with seeded Gaussian-random weights (no trained values read) and run
# the standard suppress+inject sweep with the REAL mntss 426K CLT features, at
# the cell's best strength (s=25), across 3 seeds. Registered prediction: no
# target spike at any position; effects position- and target-nonspecific and
# unstable across seeds.
#
# Run from the repo root:  bash scripts/run_random_model.sh
# Then aggregate:          python scripts/random_controls_aggregate.py  (3b section)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

OUT="docs/experiments/figure13-controls"
BIN="target/release/examples/figure13_planning_poems"

# Primary control: fresh random init (N(0, 0.02) weights).
for seed in 0 1 2; do
  echo "############ random-init Gemma 426K seed $seed ############"
  "$BIN" \
    --preset gemma2-2b-426k \
    --strength 25 \
    --random-init \
    --seed "$seed" \
    --output "$OUT/random_model_seed${seed}.json"
done

# Stricter control: per-tensor weight shuffle (preserves norm/scale statistics).
for seed in 0 1 2; do
  echo "############ weight-shuffle Gemma 426K seed $seed ############"
  "$BIN" \
    --preset gemma2-2b-426k \
    --strength 25 \
    --shuffle-weights \
    --seed "$seed" \
    --output "$OUT/random_model_shuffle_seed${seed}.json"
done

echo "############ ALL RANDOM-MODEL RUNS DONE ############"
