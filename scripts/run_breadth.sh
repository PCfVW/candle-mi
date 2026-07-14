#!/usr/bin/env bash
# Exp 4 (prompt breadth): rerun prompts #2-#4 per reference cell through the
# figure13_planning_poems grid harness at s=25, keeping each cell's preset
# suppress/inject features + inject word unchanged (only --prompt varies).
#
# Run from the repo root:  bash scripts/run_breadth.sh
# Then aggregate:          python scripts/breadth_aggregate.py
set -euo pipefail
cd "$(dirname "$0")/.."

# HF token (models are cached, but set it in case the loader needs auth).
if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

OUT="docs/experiments/figure13-controls/_runs"
mkdir -p "$OUT"

FEATURES="clt,transformer,mmap"

run() {  # preset  label  prompt
  local preset="$1" label="$2" prompt="$3"
  echo "### RUN $preset / $label ###"
  cargo run --release --features "$FEATURES" --example figure13_planning_poems -- \
    --preset "$preset" \
    --prompt "$prompt" \
    --strength-grid 25 \
    --output "$OUT/breadth_${preset}_${label}.json"
}

# ── Gemma 2 2B x mntss 426K (-out suppress, inject "around") ──────────────
run gemma2-2b-426k so $'A sailor sailed across the bay,\nAnd dreamed of home throughout the day.\nThe world keeps spinning even so,\nThere is so much we do not'
run gemma2-2b-426k shout $'A sailor sailed across the bay,\nAnd dreamed of home throughout the day.\nHe raised his voice and gave a shout,\nThe truth was struggling to come'
run gemma2-2b-426k who $'The sun goes up, the sun goes down,\nThe moon shines bright above the town.\nNobody knows or remembers who,\nWould come to find a way back'

# ── Llama 3.2 1B x mntss 524K (-ee suppress, inject "that") ───────────────
run llama3.2-1b-524k new $'The morning sky was painted blue,\nThe garden sparkled bright with dew.\nThe world had started fresh and new,\nAnd there was nothing left to'
run llama3.2-1b-524k sat $'The old man wore a tattered hat,\nUpon the porch he always sat.\nHe told the tale of this and that,\nAnd in the corner slept the'
run llama3.2-1b-524k more $'The waves came crashing on the shore,\nThe wind was howling more and more.\nShe asked what all the fuss was for,\nAnd opened up the'

echo "### ALL BREADTH RUNS DONE ###  now run: python scripts/breadth_aggregate.py"
