# Proposal: Beginner MI Examples for candle-mi

**Date:** 2026-03-09
**Source:** [Neel Nanda's MI Quickstart Guide](https://www.neelnanda.io/mechanistic-interpretability/quickstart-old)
**Goal:** Five new examples targeting MI newcomers who want to learn the
first steps of mechanistic interpretability in Rust.

---

## Existing examples (for reference)

| Example | MI technique |
|---------|-------------|
| `quick_start_transformer.rs` | Load model, run forward pass |
| `logit_lens.rs` | Per-layer prediction tracking |
| `attention_patterns.rs` | Attention weight visualization |
| `activation_patching.rs` | Causal intervention via patching |
| `attention_knockout.rs` | Zero-out specific attention heads |
| `steering_dose_response.rs` | Attention steering with dose curves |
| `figure13_planning_poems.rs` | CLT-based circuit analysis |

---

## Proposed examples

### 1. `inspect_weights.rs` -- "What does this model look like inside?"

**Nanda mapping:** TransformerLens Main Demo -- "loading in a model,
looking at its weights"

**What it does:**
- Load a model (LLaMA 3.2 1B or Qwen2.5-Coder-3B)
- Print the layer count, hidden size, head count, vocab size
- For each layer, print weight matrix shapes and Frobenius norms
  (Q, K, V, O, gate, up, down projections)
- Print embedding matrix stats (mean, std, norm)
- Show how weight norms change across layers (do later layers have
  larger norms?)

**What it teaches:**
- Model anatomy: what each weight matrix is for
- How `candle-mi` organizes model internals
- First intuition for model structure before doing any MI

**Hooks used:** None (direct weight inspection via `VarBuilder` tensors)

**Difficulty:** Beginner (first example to run)

---

### 2. `neuron_activations.rs` -- "What does this neuron respond to?"

**Nanda mapping:** "Looking for Learned Features" / Neuroscope exercise

**What it does:**
- Define a diverse set of prompts (code, math, poetry, factual,
  conversational)
- Run each prompt through the model, capturing `HookPoint::MlpOut` at
  every layer
- For each layer, find the top-5 most active neurons (highest absolute
  activation)
- For each top neuron, show which tokens in which prompts activated it
  most
- Print a summary table: neuron ID, layer, top-activating tokens,
  hypothesized feature

**What it teaches:**
- Hook capture for MLP activations
- That individual neurons often respond to interpretable features
  (punctuation, numbers, code keywords, etc.)
- The "form hypothesis, test hypothesis" workflow of MI

**Hooks used:** `HookPoint::MlpOut(layer)` for all layers

**Difficulty:** Beginner-Intermediate

---

### 3. `logit_attribution.rs` -- "Which components drive the prediction?"

**Nanda mapping:** Direct logit attribution from TransformerLens demo

**What it does:**
- For a prompt like "The Eiffel Tower is in", run the full forward pass
  capturing residual stream at each layer
- Compute the "direct effect" of each layer on the final logit: project
  each layer's residual contribution through the unembedding matrix
- Show a per-layer bar chart (text-based) of how much each attention
  layer and MLP layer contributes to the top prediction ("Paris")
- Contrast with a control prompt where the answer is less localized

**What it teaches:**
- The residual stream is additive: each layer adds its contribution
- Some layers matter much more than others for a given prediction
- Direct logit attribution as the foundation of circuit discovery

**Hooks used:** `HookPoint::Resid(layer)` for all layers, plus
`project_to_vocab()` for logit projection

**Difficulty:** Intermediate

---

### 4. `token_prediction_sweep.rs` -- "How does the model build up its answer?"

**Nanda mapping:** Toy model analysis -- "visualize a bunch of model
internals"

**What it does:**
- Define a batch of factual prompts:
  - "The capital of France is"
  - "Barack Obama was born in"
  - "The largest planet in the solar system is"
  - "Water freezes at zero degrees"
  - "def fibonacci(n):"
- Run logit lens at every layer for each prompt
- Print a matrix: rows = layers, columns = prompts, cells = top-1
  predicted token
- Highlight where the correct answer first appears and where it
  stabilizes
- Show that different facts "crystallize" at different layers

**What it teaches:**
- Layer-by-layer prediction formation (the core logit lens insight)
- That factual knowledge is stored at specific layers
- How to use logit lens as a discovery tool across many prompts

**Hooks used:** Reuses `LogitLensAnalysis` from `candle_mi::interp`

**Difficulty:** Beginner (builds on `logit_lens.rs` with more prompts
and better visualization)

---

### 5. `ioi_circuit.rs` -- "Finding a circuit in the wild"

**Nanda mapping:** "Looking for Circuits in the Wild" / IOI paper

**What it does:**
- Implement the Indirect Object Identification (IOI) task:
  - "When Mary and John went to the store, John gave a drink to" -> "Mary"
  - Generate several IOI variants with swapped names
- Run the model on IOI prompts, measure baseline logit difference
  between correct (indirect object) and incorrect (subject) name
- Systematically knock out attention heads one at a time using
  `HookPoint::AttnPattern` interventions
- Identify which heads are critical for IOI (large logit diff drop
  when knocked out)
- Classify heads as "name mover heads", "backup name movers", or
  "inhibition heads" based on their knockout effect sign
- Print a summary of the discovered IOI circuit

**What it teaches:**
- The full circuit discovery workflow: task, metric, ablation, classify
- That specific heads have specific roles in a computation
- How to use candle-mi's knockout API for systematic head ablation
- Connects to the seminal IOI paper (Wang et al., 2022)

**Hooks used:** `HookPoint::AttnPattern(layer)` with knockout
interventions, `extract_token_prob()` for logit differences

**Difficulty:** Intermediate-Advanced

---

## Implementation priority

| Priority | Example | Rationale |
|----------|---------|-----------|
| 1 | `inspect_weights.rs` | Lowest barrier to entry; no MI knowledge needed |
| 2 | `neuron_activations.rs` | First real MI exercise; builds hook intuition |
| 3 | `logit_attribution.rs` | Core MI technique; bridges to circuit discovery |
| 4 | `token_prediction_sweep.rs` | Extends existing logit_lens.rs for exploration |
| 5 | `ioi_circuit.rs` | Capstone; ties everything together |

## Model recommendations

- **Primary:** LLaMA 3.2 1B (smallest validated transformer, fast on CPU)
- **Alternative:** Qwen2.5-Coder-3B (for code-oriented prompts in
  neuron_activations)
- All examples should work on CPU with F32 (no GPU required for
  beginners)
