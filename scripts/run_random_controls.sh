#!/usr/bin/env bash
# Exp 3a random-feature / random-direction inject controls.
# For each of the three P>0.009 cells, at its Table-2 best strength (s=25), run
# N=10 random-inject draws (layer-matched to the real inject feature) + N=10
# random-direction draws (Gaussian, norm-matched per downstream layer). The
# suppress side and strength are held fixed; only the inject varies.
#
# Run from the repo root:  bash scripts/run_random_controls.sh
# Then aggregate:          python scripts/random_controls_aggregate.py
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

OUT="docs/experiments/figure13-controls"
BIN="target/release/examples/figure13_planning_poems"
SEED=42

run() {  # preset  cell-label
  local preset="$1" cell="$2"
  echo "############ $cell ($preset) ############"
  "$BIN" \
    --preset "$preset" \
    --strength 25 \
    --random-inject 10 \
    --random-direction 10 \
    --seed "$SEED" \
    --output "$OUT/random_inject_${cell}.json"
}

run gemma2-2b-426k      gemma-426k
run llama3.2-1b-524k    llama-524k
run qwen3-0.6b-16k-ation qwen3-0.6b-16k

echo "############ ALL RANDOM-CONTROL RUNS DONE ############"
echo "next: python scripts/random_controls_aggregate.py"
