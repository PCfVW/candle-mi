# Steering Convergence — Attractor Dynamics in LLM Residual Streams

## What is this?

This folder contains the results of a **steering convergence** experiment run
with [candle-mi](https://crates.io/crates/candle-mi). The experiment answers:

> **When we externally steer a model toward a target output via residual stream
> injection, does the model's internal activation pattern converge to what it
> naturally computes, or does it take a different internal path?**

This combines controllable generation with mechanistic interpretability —
systematic external control with full internal observation.

## Method

1. **Baseline run** — forward pass on "The capital of France is", capturing
   `ResidPost` at every layer.
2. **Contrastive run** — forward pass on "The capital of Germany is". Per-layer
   steering vectors are computed as `ResidPost_france − ResidPost_germany` at
   the last token position.
3. **Injection sweep** — for each layer L, the steering vector is injected via
   `Intervention::Add` at `ResidPost(L)` (last token only). All layers are
   captured in the steered forward pass.
4. **Convergence matrix** — cosine similarity between steered and natural
   activations at every (injection_layer, observation_layer) pair. Values
   below the diagonal are 1.0 (unaffected). Values on/above the diagonal
   reveal how quickly the model absorbs the perturbation.
5. **Absorption boundary** — the earliest layer after injection where cosine
   similarity exceeds a threshold (default 0.95).
6. **Strength sweep** — at the deepest absorbing layer, sweep strength from
   0.5× to 6.0× to find the critical threshold where absorption breaks.

## Results

### Llama 3.2 1B (16 layers)

| Metric | Value |
|--------|-------|
| Absorption rate | 81% of layers (13/16) |
| Avg absorption depth | 1.5 layers |
| Critical strength | ~1.2× contrastive distance |
| Boundary layer | 12 (absorption at layer 15) |
| Baseline P("Paris") | 39.3% |
| Pattern | **ATTRACTOR** |

The convergence matrix shows near-perfect similarity (>0.99) for early-layer
injections, with a sharp drop in the last 3 layers. The strength sweep at
layer 12 reveals a critical threshold between 1.0× and 1.5×: below it, the
model absorbs the perturbation within 3 layers; above it, convergence breaks
permanently.

![Llama 3.2 1B convergence matrix](plots/meta-llama_Llama-3.2-1B_convergence_matrix.png)

![Llama 3.2 1B P(target) by injection layer](plots/meta-llama_Llama-3.2-1B_p_target_by_layer.png)

![Llama 3.2 1B strength sweep](plots/meta-llama_Llama-3.2-1B_strength_sweep.png)

### Gemma 2 2B (26 layers)

| Metric | Value |
|--------|-------|
| Absorption rate | 92% of layers (24/26) |
| Avg absorption depth | 1.0 layers |
| Critical strength | ~1.2× contrastive distance |
| Boundary layer | 23 (absorption at layer 24) |
| Baseline P("Paris") | 3.9% (softcapping flattens distribution) |
| Pattern | **ATTRACTOR** |

Gemma shows an even stronger attractor — perturbations are absorbed within a
single layer. The critical strength threshold is identical (~1.2×), suggesting
this is a universal property of pre-norm transformer residual streams.

![Gemma 2 2B convergence matrix](plots/google_gemma-2-2b_convergence_matrix.png)

![Gemma 2 2B P(target) by injection layer](plots/google_gemma-2-2b_p_target_by_layer.png)

![Gemma 2 2B strength sweep](plots/google_gemma-2-2b_strength_sweep.png)

### Cross-model comparison

| Property | Llama 3.2 1B | Gemma 2 2B |
|----------|-------------|------------|
| Absorption rate | 81% | 92% |
| Avg depth | 1.5 layers | 1.0 layers |
| Critical strength | ~1.2× | ~1.2× |
| Boundary site (% depth) | 75% | 88% |
| Strength response | U-shaped | Monotonic decline |

**Key finding:** The attractor basin has a consistent radius (~1.2× the
contrastive direction) across architectures, scales, and normalization
strategies. This suggests that the residual stream's self-correcting behavior
is a structural property of pre-norm transformers, not a model-specific
artifact.

## Why does this matter?

1. **MI circuit validity.** If the model always converges to the same attractor,
   then circuits discovered via intervention (e.g., CLT suppress/inject) are
   likely the same circuits used during natural computation. The internal path
   is unique, not one of many degenerate solutions.

2. **Steering safety.** The sharp critical threshold means moderate steering
   is absorbed harmlessly, but exceeding ~1.2× the contrastive distance
   causes permanent divergence. This quantifies the "safe steering range."

3. **Attractor depth.** The ~1-2 layer absorption depth means the model's
   computation is highly local: each layer mostly corrects toward the
   attractor independently, rather than requiring multi-layer coordination.

## Experiment setup

| Parameter | Value |
|-----------|-------|
| **candle-mi version** | v0.1.3 + unreleased commits |
| **Hardware** | NVIDIA RTX 5060 Ti (16 GB VRAM) |
| **Precision** | F32 |
| **Prompt** | "The capital of France is" |
| **Contrastive** | "The capital of Germany is" |
| **Threshold** | 0.95 (cosine similarity) |
| **Strength sweep** | 0.5 to 6.0 in 12 steps |

## Reproducing

```bash
# Generate JSON output
cargo run --features transformer,mmap,memory --release --example steering_convergence -- "meta-llama/Llama-3.2-1B" --output examples/results/steering_convergence/llama-3.2-1b.json
cargo run --features transformer,mmap,memory --release --example steering_convergence -- "google/gemma-2-2b" --output examples/results/steering_convergence/gemma-2-2b.json

# Plot with Mathematica
# Open convergence_plot.wl and evaluate all — it auto-iterates over all JSON files.
# Plots are exported to the plots/ subfolder.
```

## Files

| File | Description |
|------|-------------|
| `llama-3.2-1b.json` | Full output for Llama 3.2 1B |
| `gemma-2-2b.json` | Full output for Gemma 2 2B |
| `convergence_plot.wl` | Mathematica plotting script (auto-iterates all JSON files) |
| `plots/` | Generated PNG plots (convergence matrix, P(target), strength sweep) |
| `README.md` | This file |
