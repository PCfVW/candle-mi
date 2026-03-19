# Attention Routing — How Planning Decisions Travel Through a Transformer

## The question

When a language model plans a rhyme, the decision happens at an early position
in the sequence (the "planning site") — not at the position where the rhyme
word is ultimately predicted. But how does that decision travel from where it's
made to where it's used?

This experiment answers that question by measuring **attention pattern changes**
when we intervene on planning circuits using CLT (Cross-Layer Transcoder) features.

## Background: three computation regimes

Previous work with [steering_convergence](../steering_convergence/) revealed
that planning and factual recall operate in fundamentally different ways:

| Regime | Where it lives | Example |
|--------|---------------|---------|
| **Factual recall** | Last-token residual stream | "The capital of France is ___" → "Paris" |
| **Planning (local)** | Planning-site residual stream | CLT features fire at the rhyme word position |
| **Planning (output)** | Attention routing | **This experiment** |

Factual recall is a direction in the residual stream — you can steer it with
a simple vector and measure convergence. Planning is different: the CLT
features modify the residual stream at position 23 ("about"), but the model
predicts the next word at position 30 (end of "passage"). Something must
connect these two positions. That something is **attention**.

## Method

We use the exact same intervention as Anthropic's Figure 13 replication
(`figure13_planning_poems`):

1. **Suppress** natural rhyme features (e.g., "-out" group) with negative steering
2. **Inject** an alternative rhyme feature (e.g., "around") with positive steering
3. Both interventions target the **planning site** (position 23, the word "about")

But instead of measuring the output probability (which Figure 13 already does),
we capture the **attention patterns** at every layer and ask:

> Which attention heads change how much they attend from the **last token**
> (position 30) back to the **planning site** (position 23)?

If a head increases its attention weight on that path, it's a **routing head** —
it carries the planning decision from where it's made to where it's needed.

### What we capture

For each of 208 attention heads (26 layers × 8 heads per layer in Gemma 2 2B):

```
baseline_attn[layer][head] = attention_weight(pos 30 → pos 23)  # no intervention
steered_attn[layer][head]  = attention_weight(pos 30 → pos 23)  # with suppress+inject
delta[layer][head]         = steered - baseline
```

A positive delta means the head attends **more** to the planning site after
intervention. A negative delta means it attends **less**.

## Results

### 426K CLT (feature L22:10243, "around")

Suppress: L16:13725 ("about") + L25:9385 ("out"). Inject: L22:10243 ("around").
Strength: 10.0 (same as Figure 13).

![Top 10 routing heads, 426K CLT](plots/top10_routing_heads.png)

**Three heads dominate the routing:**

| Head | Delta | Direction | Interpretation |
|------|-------|-----------|----------------|
| **L21:H5** | -0.046 | Decreased | Was reading "-out" plan; suppression reduces its need to attend |
| **L20:H6** | +0.027 | Increased | Picks up the new "around" signal at the planning site |
| **L17:H4** | +0.014 | Increased | Earlier routing head, also responds to the injected feature |

The pattern is a **push-pull redistribution**: some heads disengage from the
planning site (red bars) while others engage (green bars). The model doesn't
just add attention — it reorganizes its information flow.

### 2.5M CLT (feature L25:82839, "can")

Suppress: L25:57092 ("about") + L23:49923 ("out") + L20:77102. Inject: L25:82839 ("can").
Strength: 10.0.

![Top 10 routing heads, 2.5M CLT](plots/top10_routing_heads_2.5m.png)

**Same top head (L21:H5), same direction, 5× weaker signal.** The 2.5M CLT's
inject feature is at layer 25 (only 1 downstream layer), so most of the
routing effect comes from the suppress features. The 426K CLT's inject at
layer 22 has 4 downstream layers, giving attention more room to reorganize.

**Head H5 appears at multiple layers** (L21, L23, L24, L25) in both CLTs —
suggesting that head 5 is a dedicated planning routing channel in Gemma 2 2B.

### The planning attractor boundary

We swept the steering strength from 0 to 20 and measured how the top head's
attention delta scales:

![Strength sweep, top head delta](plots/strength_sweep_top_head.png)

**Key observation:** The 426K curve (blue) follows a linear trajectory up to
strength ~12, then **bends below** the dashed linear extrapolation. This is
the **saturation onset** — the point where the attention routing starts
resisting further perturbation. The highlighted region shows where the
attractor's "soft boundary" begins.

The 2.5M curve (red) stays perfectly linear — no saturation, because the
inject feature at L25 bypasses attention entirely and the suppress features
produce a weaker signal.

**Comparison with factual recall:**

| Property | Factual recall | Planning (attention) |
|----------|---------------|---------------------|
| Boundary type | **Hard** — sharp threshold at ~1.2× | **Soft** — gradual saturation at ~15× |
| Response shape | Linear then divergence | Linear then bending |
| Recovery | Full absorption within 1-2 layers | No absorption (different mechanism) |
| Measurement space | Residual stream cosine similarity | Attention weight delta |

Factual recall has a hard basin: below the threshold, perturbations are fully
absorbed; above it, the model diverges. Planning has a soft boundary: the
response is proportional until the routing heads start saturating, then the
effect grows sub-linearly. There's no catastrophic divergence — just diminishing
returns.

### Total attention redistribution

![Total routing shift](plots/strength_sweep_total_routing.png)

The total routing shift (sum of absolute deltas across all 208 heads) shows
the same pattern: 426K produces 3.3× more redistribution than 2.5M at every
strength level, and both scale nearly linearly. The 426K shows slight
concavity at high strength, consistent with the saturation onset.

## Why suppress matters

A critical finding from this work: **inject-only produces 13× weaker attention
deltas than suppress+inject.** Without suppressing the natural "-out" features,
the injection fights against the model's existing rhyme computation and barely
moves the attention patterns.

| Mode | Top head delta (L21:H5) | Total routing |
|------|------------------------|---------------|
| Inject only (426K) | -0.0035 | 0.017 |
| **Suppress + inject (426K)** | **-0.046** | **0.191** |

This parallels Figure 13's finding: the probability spike at the planning site
only appears with both suppress and inject. The suppress clears the path; the
inject redirects it.

## What this means for mechanistic interpretability

1. **Planning circuits use attention routing.** The planning decision at position
   23 reaches the output at position 30 through specific attention heads, not
   through residual stream propagation. This is fundamentally different from
   factual recall, which operates within the residual stream at a single position.

2. **Head 5 is a planning routing channel.** Across both CLT granularities,
   head 5 at layers 17-25 consistently shows the largest attention changes.
   This suggests a structurally specialized role for this head family in
   Gemma 2 2B's planning computation.

3. **Planning attractors are soft, not hard.** Unlike factual recall's sharp
   basin boundary (~1.2× critical strength), planning routing scales linearly
   with eventual saturation. This means planning interventions are more
   forgiving — there's no "cliff" where the model suddenly diverges.

4. **CLT source layer determines routing impact.** Features at earlier layers
   (L22 for 426K) produce stronger attention routing because they have more
   downstream layers to propagate through. Features at late layers (L25 for
   2.5M) mainly affect the residual stream directly, bypassing attention.

## Reproducing

```bash
# 426K CLT (strongest routing signal)
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --suppress L16:13725 --suppress L25:9385 \
     --output examples/results/attention_routing/gemma-2-2b-426k.json

# 2.5M CLT (word-level features)
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --clt-repo mntss/clt-gemma-2-2b-2.5m --feature L25:82839 \
     --suppress L25:57092 --suppress L23:49923 --suppress L20:77102 \
     --output examples/results/attention_routing/gemma-2-2b-2.5m.json

# Inject only (for comparison — 13x weaker)
cargo run --release --features clt,transformer,mmap --example attention_routing

# Plot with Mathematica
# Open attention_routing_plot.wl and evaluate all cells.
```

## Files

| File | Description |
|------|-------------|
| `gemma-2-2b-426k.json` | Full output: 426K CLT, suppress+inject |
| `gemma-2-2b-2.5m.json` | Full output: 2.5M CLT, suppress+inject |
| `attention_routing_plot.wl` | Mathematica plotting script |
| `plots/` | Generated PNG plots |
| `README.md` | This file |

## Experiment setup

| Parameter | Value |
|-----------|-------|
| **Model** | Gemma 2 2B (`google/gemma-2-2b`) |
| **candle-mi version** | v0.1.3 + unreleased commits |
| **Hardware** | NVIDIA RTX 5060 Ti (16 GB VRAM) |
| **Precision** | F32 |
| **Prompt** | Figure 13 Gemma preset (4-line couplet, "about" rhyme) |
| **Planning site** | Position 23 (token "about") |
| **Output position** | Position 30 (last token) |
| **Strength** | 10.0 (Figure 13 default), sweep 0–20 |
