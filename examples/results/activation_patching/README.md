# Activation Patching (Causal Tracing) — Experiment Results

## What is this?

This folder contains the results of a **causal tracing** experiment run with
[candle-mi](https://crates.io/crates/candle-mi), replicating the core
technique from:

> **Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022).** *Locating and
> Editing Factual Associations in GPT.* Advances in Neural Information
> Processing Systems (NeurIPS).
> <https://arxiv.org/abs/2202.05262>
> (Section 2.1 "Causal Tracing of Factual Associations", Figure 1e)

The experiment patches clean residual stream activations into a corrupted
forward pass at every (layer × token position) combination, measuring how
much the correct prediction recovers. The result is a 2D heatmap — the
"causal trace" — that reveals where in the network factual associations
are stored and recalled.

## Why does this matter?

Meng et al. discovered that factual recall in GPT-2-XL involves two
distinct sites:

- An **early site** at mid-layers, concentrated at the last subject token —
  where MLP modules recall stored facts.
- A **late site** at the last token in the final layers — where attention
  copies the recalled information to the output position.

This two-site structure has implications for model editing (ROME, MEMIT)
and for understanding how transformers store knowledge.

## Experiment setup

| Parameter            | Value                                          |
|----------------------|-------------------------------------------------|
| **candle-mi version** | v0.1.2 + unreleased commits                   |
| **Model**            | `google/gemma-2-2b` (26 layers, hidden=2304)   |
| **Precision**        | F32 (research-grade, matches Python/PyTorch)    |
| **GPU**              | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)         |

### Prompt pairs

| Label | Clean prompt | Corrupted prompt | Expected answer |
|-------|-------------|-----------------|-----------------|
| France→Paris | "The capital of France is" | "The capital of Poland is" | Paris |
| Space Needle→Seattle | "The Space Needle is in downtown" | "The Eiffel Tower is in downtown" | Seattle |

The corrupted prompt swaps the subject entity while keeping the same
sentence structure and token count. For each (layer, position) pair, the
clean residual at that position is patched into the corrupted forward pass,
and recovery is measured as `1 - KL_patched / KL_corrupted`.

## Folder contents

```
activation_patching/
├── README.md                      ← this file
├── causal_trace_plot.wl           ← Wolfram Mathematica plotting script
├── gemma-2-2b_france.json         ← JSON results (France→Paris)
├── gemma-2-2b_space needle.json   ← JSON results (Space Needle→Seattle)
└── plots/                         ← generated PNG visualizations
    ├── google_gemma-2-2b_causal_trace_heatmap.png  ← Figure 1e heatmap
    └── google_gemma-2-2b_subject_recovery.png      ← recovery curve
```

## Key findings (Gemma 2 2B)

- **Late site confirmed** (layers 22–25, last token "downtown"): bright
  recovery — attention at the last token copies factual information to the
  output, matching Meng et al.'s finding in GPT-2-XL (layers 25–45).
- **Subject tokens carry the signal** ("Space", "Needle" rows): strong
  recovery across most layers. In GPT-2-XL, the "early site" was sharply
  concentrated at the last subject token in mid-layers; in Gemma 2 2B the
  signal is more diffuse, likely due to GQA and logit softcapping
  distributing information more broadly.
- **Non-subject tokens are inert** ("<bos>", "The", "is"): patching at
  these positions has negligible effect, confirming that factual recall
  flows through the subject tokens.

## Reproducing this experiment

```bash
# Run on Gemma 2 2B with JSON output
cargo run --release --features transformer,mmap,memory \
    --example activation_patching -- \
    "google/gemma-2-2b" \
    --output examples/results/activation_patching/gemma-2-2b.json

# Run on Llama 3.2 1B
cargo run --release --features transformer,memory \
    --example activation_patching -- \
    "meta-llama/Llama-3.2-1B" \
    --output examples/results/activation_patching/llama-3.2-1b.json

# Run on all cached models (no JSON output)
cargo run --release --features transformer,mmap --example activation_patching
```

Both prompt pairs (France→Paris and Space Needle→Seattle) run automatically.
With `--output`, each pair produces a separate JSON file (e.g.,
`gemma-2-2b_france.json` and `gemma-2-2b_space needle.json`).

## Plotting

Open `causal_trace_plot.wl` in Wolfram Mathematica (v13+ recommended). Set
the `jsonFile` variable at the top to the JSON file you want to plot. The
script generates two plots:

1. **Causal trace heatmap** — tokens on Y-axis, layers on X-axis, matching
   the paper's Figure 1(e) orientation.
2. **Subject-position recovery curve** — layer sweep at the subject token
   only, with the best layer highlighted.

## References

1. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and
   Editing Factual Associations in GPT.* NeurIPS.
   <https://arxiv.org/abs/2202.05262>

2. candle-mi documentation: <https://docs.rs/candle-mi>
