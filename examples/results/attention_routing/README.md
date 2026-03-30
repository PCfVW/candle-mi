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

## Comparison with Anthropic's "Planning in Poems"

Our attention routing experiment relates directly to Anthropic's investigation
of rhyme planning in their attribution graphs paper:

> Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N.,
> Citro, C., Olah, C., & Templeton, A. (2025). *On the Biology of a Large
> Language Model.* Transformer Circuits Thread.
> [Section: Planning in Poems](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems)

### What Anthropic found

Using 30M-feature Cross-Layer Transcoders on Claude, Anthropic discovered that
LLMs plan rhymes through a **forward planning strategy**:

1. At newline tokens (before a line begins), features for candidate rhyming
   words activate — the model pre-computes where the line needs to end
2. These "planned word features" then influence how the model constructs the
   entire line **backward** — writing text that naturally leads to the target word
3. The model holds **multiple candidate rhymes** simultaneously (e.g., "rabbit"
   and "habit" both active) before committing to one

They built full attribution graphs showing how rhyming constraint features
connect to candidate completion features, which connect to transition features,
which finally determine output tokens. This is a layered causal hierarchy.

### What Anthropic explicitly could NOT measure

In their paper, Anthropic makes a striking admission about a gap in their
methodology:

> *"One crucial interaction (seems) to be mediated by changing where attention
> heads attend, by participating in their QK circuits. **This is invisible to
> our current approach.**"*

Their attribution graph method traces information flow through CLT features
(which replace MLPs), but it cannot see how attention heads **re-route** their
queries and keys in response to planning features. The QK circuit — the
mechanism that decides "which positions should attend to which other positions"
— is a blind spot.

### What our experiment measures

This is exactly the gap we fill. Our `attention_routing` example directly
measures how attention patterns change when planning features are activated:

| Aspect | Anthropic | candle-mi |
|--------|-----------|-----------|
| **What they see** | Features activate at the planning site | Same (we use their published CLT features) |
| **What they trace** | Feature → feature causal flow through MLPs | Feature → attention head routing changes |
| **Attention heads** | "Invisible to our current approach" | **L21:H5 dominant; H5 family spans layers 17-25** |
| **Attractor dynamics** | Not measured | Soft boundary at ~15×, linear regime below |
| **Scale** | Claude (large model, internal infrastructure) | Gemma 2 2B (consumer GPU, open weights) |

### How the two approaches complement each other

Think of it this way. Anthropic built a map of the **roads** (which features
connect to which other features). We measured the **traffic signals** (which
attention heads change their routing when those features are active). You need
both to understand how information actually flows through the network:

- **Anthropic's attribution graphs** answer: *"What features are causally
  upstream of the rhyme prediction?"* They identify the nodes in the circuit.
- **Our attention routing** answers: *"How does the planning decision at
  position 23 physically reach position 30?"* We identify the edges — the
  specific attention heads that carry the signal.

Together, these give a more complete picture: features at the planning site
activate (Anthropic's finding), specific attention heads then re-route to read
those features (our finding), and the information flows through the identified
causal chain to produce the output (Anthropic's graph structure).

### What we add beyond Anthropic

1. **Direct attention head identification.** L21:H5 is the dominant planning
   routing head in Gemma 2 2B, with head 5 appearing across multiple layers
   (17, 21, 23, 24, 25). This is the first identification of specific attention
   heads involved in rhyme planning — the exact measurement Anthropic said was
   invisible to their approach.

2. **Quantitative attractor characterization.** We measured how the attention
   routing scales with intervention strength and found a soft boundary (gradual
   saturation) rather than a hard threshold. This is a dynamical systems
   property that attribution graphs cannot capture.

3. **Suppress amplification.** We quantified that suppress+inject produces 13×
   stronger attention routing than inject alone. This has methodological
   implications: single-feature ablation studies may underestimate the true
   routing effect because they don't account for competing features.

4. **Cross-granularity validation.** The same head (L21:H5) dominates at both
   426K and 2.5M CLT granularities, confirming this is a structural property
   of the model, not an artifact of any particular CLT decomposition.

### What Anthropic has that we don't (yet)

1. **Full attribution graphs** — feature-to-feature causal graphs with error
   nodes quantifying unexplained computation. This is planned for candle-mi
   v0.2 (the `build_attribution_graph` API already exists but hasn't been
   integrated into the convergence/routing experiments).

2. **30M-feature CLTs** — much finer granularity than our 2.5M. Finer features
   could reveal more specific routing patterns.

3. **Multi-candidate analysis** — Anthropic observes simultaneous activation of
   competing rhyme options. Our experiment could test this by injecting competing
   features and measuring whether different attention heads route to different
   candidates.

4. **Scale** — Anthropic works on Claude (a large, proprietary model). Our
   results on Gemma 2 2B (2.6B parameters) demonstrate the same planning
   mechanism exists in smaller, open models on consumer hardware.

## Cross-model comparison: Llama 3.2 1B (N=4)

We ran attention routing on Llama 3.2 1B across **4 prompts from 3 rhyme
groups** (-ee, -ore, -oo, -at), using validated features from
`rhyme_pairs_llama.json` (found by the systematic decoder-projection
vocabulary scan). All suppress and inject features are traceable to
the plip-rs vocabulary scan.

### Llama 3.2 1B planning routing heads (N=4)

![Per-prompt routing heads](plots/top10_routing_heads_llama.png)

**Planning routing is prompt-specific**: each rhyme group recruits a
different dominant head (L15:H8 for -ee, L6:H12 for -ore, L14:H22 for
-oo, L9:H13 for -at). The heads that recur across 2+ prompts (L6:H12,
L9:H13, L5:H17, L11:H0, L6:H24) are concentrated in mid-layers (5-11).

### Cross-model strength sweep (N=4)

![Cross-model top head sweep](plots/cross_model_top_head.png)

The spread across prompts reveals the prompt-specificity: -oo (L14:H22)
exceeds Gemma at high strengths, while -ee (L15:H8) is much weaker. All
curves show soft, roughly linear responses with no hard threshold —
the same attractor dynamics as Gemma.

![Cross-model total routing](plots/cross_model_total_routing.png)

Total attention redistribution: **all 4 Llama prompts exceed Gemma**
despite having no single dominant head. Llama compensates with
distributed routing across many heads, while Gemma concentrates the
signal through L21:H5. This distributed routing may explain why Llama
searches but cannot commit.

### Cross-model comparison table

| Property | Gemma 2 2B (426K) | Llama 3.2 1B (N=4) |
|----------|-------------------|---------------------|
| **Top head** | L21:H5 (-0.046) | Prompt-specific |
| **Recurring heads (2+)** | H5 family (L21-L25) | L5:H17, L6:H12, L9:H13 |
| **Mean layer (recurring)** | 22.2 (85% depth) | 7.4 (46% depth) |
| **Direction** | Push-pull | Push-pull |
| **Total routing (str 10)** | 0.19 | 0.05-1.27 (varies) |
| **Saturation** | Soft, ~15× | Soft, linear |

## Connection to factual routing (prolepsis)

The `factual_routing` experiment measured attention routing during
CounterFact activation patching on the **same Llama 3.2 1B model**. The
recurring planning heads and factual routing heads occupy **different
network depths**:

| | Planning routing (N=4) | Factual routing (89 pairs) |
|---|---|---|
| **Dominant head** | Prompt-specific | L15:H8 (48.6%) |
| **Recurring heads (2+)** | L5:H17, L6:H12, L6:H24, L9:H13, L11:H0 | L15:H8, L14:H27, L15:H16 |
| **Mean layer (recurring)** | 7.4 (46% depth) | 12.8 (80% depth) |
| **Recurring ∩ factual top-10** | **0 / 5** | — |

Zero overlap at the aggregate level — the recurring planning heads are
disjoint from the factual top-10. We call this **prolepsis**
(πρόληψις, "anticipation"): the model commits early and sustains the
commitment through attention routing at task-dependent network depths.
Planning routes through mid-layer heads, factual recall through late-layer
heads, but the structural template is shared:

- **Early commitment** — the decision is made in the first half of the network
- **Sustained propagation** — routing heads maintain the commitment
- **Irrevocability** — no layer reverses the committed answer

See the [factual routing results](../factual_routing/) and the
[counterfact patching results](../counterfact_patching/) for the full
factual recall data.

## Reproducing

```bash
# Gemma 2 2B — 426K CLT (strongest routing signal)
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --suppress L16:13725 --suppress L25:9385 \
     --output examples/results/attention_routing/gemma-2-2b-426k.json

# Gemma 2 2B — 2.5M CLT (word-level features)
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --clt-repo mntss/clt-gemma-2-2b-2.5m --feature L25:82839 \
     --suppress L25:57092 --suppress L23:49923 --suppress L20:77102 \
     --output examples/results/attention_routing/gemma-2-2b-2.5m.json

# Llama 3.2 1B — 524K CLT (N=4, validated features)
# -ee prompt (suppress -ee group, inject "that")
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --model meta-llama/Llama-3.2-1B --clt-repo mntss/clt-llama-3.2-1b-524k \
     --feature L14:13043 --suppress L13:30985 --suppress L9:5488 \
     --suppress L14:27874 --suppress L13:32049 --strength 10.0 --planning-site 14 \
     --prompt "The birds were singing in the tree,\nAnd everything was wild and free.\nThe river ran down to the sea,\nThere is so much we cannot" \
     --output examples/results/attention_routing/llama-3.2-1b-524k-ee.json

# -ore prompt
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --model meta-llama/Llama-3.2-1B --clt-repo mntss/clt-llama-3.2-1b-524k \
     --feature L14:13043 --suppress L1:5297 --suppress L3:22663 \
     --suppress L10:18203 --strength 10.0 --planning-site 16 \
     --prompt "The waves came crashing on the shore,\nThe wind was howling more and more.\nShe asked what all the fuss was for,\nAnd opened up the" \
     --output examples/results/attention_routing/llama-3.2-1b-524k-ore.json

# -oo prompt
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --model meta-llama/Llama-3.2-1B --clt-repo mntss/clt-llama-3.2-1b-524k \
     --feature L14:13043 --suppress L14:18284 --suppress L15:8165 \
     --suppress L11:20779 --strength 10.0 --planning-site 22 \
     --prompt "The morning sky was painted blue,\nThe garden sparkled bright with dew.\nThe world had started fresh and new,\nAnd there was nothing left to" \
     --output examples/results/attention_routing/llama-3.2-1b-524k-oo.json

# -at prompt
cargo run --release --features clt,transformer,mmap --example attention_routing \
  -- --model meta-llama/Llama-3.2-1B --clt-repo mntss/clt-llama-3.2-1b-524k \
     --feature L1:5297 --suppress L14:6132 --strength 10.0 --planning-site 15 \
     --prompt "The old man wore a tattered hat,\nUpon the porch he always sat.\nHe told the tale of this and that,\nAnd in the corner slept the" \
     --output examples/results/attention_routing/llama-3.2-1b-524k-at.json

# Inject only (for comparison — 13x weaker on Gemma)
cargo run --release --features clt,transformer,mmap --example attention_routing

# Plot with Mathematica
# Open attention_routing_plot.wl and evaluate all cells.
# Generates 7 plots: 4 original + 3 cross-model comparisons.
```

## Files

| File | Description |
|------|-------------|
| `gemma-2-2b-426k.json` | Full output: Gemma 2 2B, 426K CLT, suppress+inject |
| `gemma-2-2b-2.5m.json` | Full output: Gemma 2 2B, 2.5M CLT, suppress+inject |
| `llama-3.2-1b-524k.json` | Full output: Llama 3.2 1B, 524K CLT, suppress+inject |
| `attention_routing_plot.wl` | Mathematica plotting script (7 plots) |
| `plots/` | Generated PNG plots (7 files) |
| `README.md` | This file |

## Experiment setup

| Parameter | Gemma 2 2B | Llama 3.2 1B |
|-----------|------------|--------------|
| **Model** | `google/gemma-2-2b` | `meta-llama/Llama-3.2-1B` |
| **CLT** | 426K / 2.5M | 524K |
| **candle-mi version** | v0.1.3+ | v0.1.5+ |
| **Hardware** | NVIDIA RTX 5060 Ti (16 GB) | NVIDIA RTX 5060 Ti (16 GB) |
| **Precision** | F32 | F32 |
| **Prompt** | 4-line couplet ("about" rhyme) | 4-line couplet ("cat" rhyme) |
| **Planning site** | Position 23 ("about") | Position 23 ("cat") |
| **Output position** | Position 30 (last token) | Position 29 (last token) |
| **Strength** | 10.0 (default), sweep 0–20 | 15.0, sweep 0–20 |
