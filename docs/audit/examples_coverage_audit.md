# Examples Coverage Audit

**Date:** 2026-03-08 (initial) · 2026-03-09 (updated with medium-impact examples)
**Scope:** candle-mi public API vs. existing `examples/` folder
**Goal:** Identify missing crate-level examples for Phase 5 documentation.

---

## 1. All Implemented Examples

| Example | Status | API Features Demonstrated |
|---|---|---|
| `quick_start_transformer.rs` | Shipped | `MIModel::from_pretrained`, `HookSpec` (empty), `encode()`, forward pass, top-K from logits |
| `quick_start_sae.rs` | Shipped | `SparseAutoencoder::from_pretrained_npz`, `HookPoint::ResidPost` capture, `encode()`, feature inspection |
| `auto_config_dogfood.rs` | Shipped | `download_model_blocking`, `TransformerConfig::from_hf_config_auto`, `CompatibilityReport` |
| `fast_download.rs` | Shipped | `download_model_blocking`, tracing-based progress bars |
| `figure13_planning_poems.rs` | Shipped | `CrossLayerTranscoder`, CLT feature injection via `HookSpec`, position sweep, `extract_token_prob`, JSON export |
| `generate.rs` | **NEW — Implemented** | `sample_token`, `GenerationResult`, `HookSpec`, autoregressive loop, multi-model sweep |
| `logit_lens.rs` | **NEW — Implemented** | `LogitLensAnalysis`, `LogitLensResult`, `TokenPrediction`, `HookPoint::ResidPost`, `project_to_vocab`, `decode_predictions_with` |
| `attention_knockout.rs` | **NEW — Implemented** | `KnockoutSpec`, `AblationResult`, `Intervention::Knockout`, `create_knockout_mask`, `HookPoint::AttnScores`, `format_probability` |
| `steering_dose_response.rs` | **NEW — Implemented** | `SteeringCalibration`, `DoseResponseCurve`, `SteeringSpec`, `SteeringResult`, `apply_steering`, `measure_attention_to_targets`, `DOSE_LEVELS`, `Intervention::Replace` |
| `attention_patterns.rs` | **NEW — Implemented** | `AttentionCache`, `HookPoint::AttnPattern`, `attention_from_position`, `attention_to_position`, `top_attended_positions` |
| `activation_patching.rs` | **NEW — Implemented** | `FullActivationCache`, `Intervention::Replace`, `Intervention::Add`, `HookPoint::ResidPost`, `HookPoint::Embed`, `kl_divergence`, position-specific patching |
| `token_positions.rs` | **NEW — Implemented** | `EncodingWithOffsets`, `convert_positions`, `TokenWithOffset`, `PositionConversion`, `encode_with_offsets`, `char_to_token`, `token_to_char_range`, `char_range_to_tokens` |

**Coverage estimate:** ~75 % of the public API surface (up from ~65 %).

---

## 2. Public API with Zero Example Coverage

### Steering & Intervention Specs
- ~~`SteeringCalibration`, `DoseResponseCurve`, `DoseResponsePoint`, `DOSE_LEVELS`~~ → covered by `steering_dose_response.rs`
- ~~`SteeringSpec`, `SteeringResult`~~ → covered by `steering_dose_response.rs`
- `HeadSpec`, `LayerSpec`, `AttentionEdge` (structural specs for targeting
  specific heads, layers, and positional edges — `attention_knockout.rs` uses
  `KnockoutSpec` directly but does not demonstrate these building blocks)
- `InterventionType` (enum classifying intervention kinds)

### Attention Analysis (full capture)
- ~~`AttentionCache`, `AttnPattern`~~ → covered by `attention_patterns.rs`
- `AttnScores` capture for analysis (knockout uses it for intervention only)

### Caching
- ~~`FullActivationCache`~~ → covered by `activation_patching.rs`
- `ActivationCache` (last-token-only cache; not yet demonstrated separately)
- `KVCache` (`generate.rs` uses full-sequence recompute; KV-cached
  generation remains undemonstrated)

### Positioning Utilities
- ~~`EncodingWithOffsets`, `convert_positions`, `TokenWithOffset`~~ → covered by `token_positions.rs`

### Mask Utilities
- `create_causal_mask`, `create_generation_mask`

### RWKV Backend
- `GenericRwkv`, `RwkvConfig`, `RwkvVersion`, `RwkvLoraDims`
- `StateKnockoutSpec`, `StateSteeringSpec`, `StateAblationResult`,
  `StateSteeringResult`

### Recurrent Feedback
- `RecurrentFeedbackEntry`, `RecurrentPassSpec`

### Intervention Primitives (direct use)
- ~~`Intervention::Replace`~~ → covered by `steering_dose_response.rs`, `activation_patching.rs`
- ~~`Intervention::Add`~~ → covered by `activation_patching.rs`
- `Intervention::Scale`,
  `Intervention::Zero` (knockout uses `Intervention::Knockout`; these
  two remain uncovered)

### CLT Attribution
- `AttributionGraph`, `AttributionEdge` (injection is shown in figure13, but
  graph scoring is not)

### Memory Reporting
- ~~`MemorySnapshot`, `MemoryReport`~~ → opt-in in all 5 high-impact examples
  via `#[cfg(feature = "memory")]`

---

## 3. Implemented Examples — Design Decisions & Test Results

### 3.1 Design Conventions (decided during brainstorming)

The following conventions apply to all 5 implemented examples and should be
followed by future examples.

#### Input format: CLI arguments (not JSON)

Examples accept model IDs and parameters as CLI arguments, not input JSON files.
Rationale: CLI args are the most natural interface for Rust developers; JSON
input adds friction without adding value for single-prompt examples.

- When no model is specified, examples auto-discover all cached models and run
  on each (multi-model sweep).
- A single model can be passed as a positional argument to avoid OOM on large
  models or to focus on one result.

#### Output format: CLI printing + opt-in JSON

Examples print human-readable results to stdout by default. JSON output for
programmatic consumption is opt-in (planned for future work via `--json` flag).

- CLI output follows a consistent format: model name header, indented results,
  summary tables.
- JSON output, when added, will be written to `examples/results/<example>/<model>.json`.

#### CLI argument parsing

The 5 implemented examples use `std::env::args()` for simplicity (single
optional model ID, no flags). For future examples with richer CLI (e.g.,
`--json`, `--layer`, `--top-k`), the agreed convention is **Clap isolation**:
separate CLI parsing from MI logic with clear section comments:

```rust
// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(clap::Parser)]
#[command(name = "example_name", about = "...")]
struct Cli {
    /// Optional model ID
    model: Option<String>,
}

// ---------------------------------------------------------------------------
// MI logic
// ---------------------------------------------------------------------------

fn run() -> candle_mi::Result<()> {
    let cli = Cli::parse();
    // ... MI code here ...
}
```

This pattern keeps the Clap dependency localized and makes the MI logic
readable independently of CLI parsing.

#### Cross-model comparison tables

Results from multiple models are documented in `examples/README.md` as
comparison tables. This is the primary mechanism for showing how MI techniques
behave differently across architectures (GQA, soft-capping, code vs. language
models, etc.).

Each example section in `examples/README.md` includes:
1. A prose summary of the key finding per model.
2. A comparison table with quantitative results.

#### Probability formatting (mandatory)

All examples that print probabilities (0.0–1.0 values representing token or
attention probabilities) must use `format_probability()` from
`candle_mi::interp::logit_lens` for adaptive precision:

- ≥1%: 1 decimal place (e.g., "39.3%")
- ≥0.01%: 3 decimal places (e.g., "0.012%")
- <0.01%: scientific notation (e.g., "3.2e-4%")

This ensures consistent, readable output across models with very different
probability distributions (e.g., Gemma 2's soft-capped logits produce much
smaller probabilities than Llama's).

**Note:** Attention weights (0.0–1.0 values from softmax over positions) and
KL divergence values are **not** vocabulary probabilities — they use fixed
precision (`.3` for attention weights, `.6` for KL/attention measurements).

#### Memory reporting (implemented)

The `memory` feature (`src/memory.rs`) provides `MemorySnapshot` and
`MemoryReport` for exact per-process RAM measurement (Windows FFI / Linux
procfs) and device-wide VRAM (nvidia-smi). All 5 high-impact examples include
opt-in memory reporting via `#[cfg(feature = "memory")]` guards, activated
with `--features memory`.

### 3.2 `generate.rs` — Autoregressive Text Generation ✅

**Status:** Implemented and tested on 3 models.

**API surface covered:**
- `sample_token()`, `GenerationResult`, `HookSpec`, `MIModel::forward`
- `MITokenizer::encode()`, `MITokenizer::decode()`
- Multi-model auto-discovery via HuggingFace cache scanning

**Design:**
- Greedy generation (temperature 0.0), max 20 tokens
- Full-sequence recompute each step (no KV cache — MI-first design)
- Stop early on EOS token
- `main()` → `run() -> candle_mi::Result<()>` (standard pattern)

**Test results:** Generates coherent continuations on Llama 3.2 1B, Gemma 2 2B,
StarCoder2 3B. Each model produces output characteristic of its training data
(prose, Python code, etc.).

### 3.3 `logit_lens.rs` — Layer-by-Layer Prediction Tracking ✅

**Status:** Implemented and tested on 3 models.

**API surface covered:**
- `LogitLensAnalysis`, `LogitLensResult`, `TokenPrediction`
- `HookPoint::ResidPost` (all layers captured)
- `project_to_vocab`, `decode_predictions_with`
- `HookSpec::capture()` at multiple hook points

**Design:**
- Captures `ResidPost` at all layers in a single forward pass
- For each layer: extract last-position hidden state, project to vocabulary,
  softmax, top-5 predictions
- Builds `LogitLensAnalysis`, prints summary and detailed per-layer results
- Reports `first_appearance("Paris", top_k)` for convergence tracking

**Test results:**

| Model | Layers | "Paris" first appears | Final prediction |
|-------|--------|-----------------------|------------------|
| Llama 3.2 1B | 16 | Layer 11 (top-5) | "Paris" (top-1) |
| Gemma 2 2B | 26 | Never in top-10 | " a" (soft-capped logits flatten distribution) |
| StarCoder2 3B | 30 | Never in top-10 | Code tokens dominate |

### 3.4 `attention_knockout.rs` — Head / Edge Ablation ✅

**Status:** Implemented and tested on 3 models.

**API surface covered:**
- `KnockoutSpec`, `AblationResult`, `Intervention::Knockout`, `create_knockout_mask`
- `HookPoint::AttnScores` (intervention target)
- Baseline vs. ablated comparison, KL divergence

**Not covered** (exported but not demonstrated by this example):
`HeadSpec`, `LayerSpec`, `AttentionEdge`, `InterventionType` — these
structural building blocks are available for fine-grained knockout
specifications but the example constructs a mask directly.

**Design:**
- Knocks out a single attention edge (last → first token) at the middle layer
- Compares baseline and ablated logit distributions
- Reports KL divergence, target token probability change, top changed tokens
- Single-edge knockout (not full-row) to avoid NaN in softmax

**Test results:**

| Model | Layer | Heads | KL div | "Paris" baseline | "Paris" ablated | Logit diff |
|-------|-------|-------|--------|-----------------|----------------|------------|
| Llama 3.2 1B | 8 | 32 | 0.056 | 39.3% | 26.0% | −1.06 |
| Gemma 2 2B | 13 | 8 | 0.017 | 3.9% | 6.7% | −0.33 |
| StarCoder2 3B | 15 | 24 | 0.029 | — | — | −1.08 |

**Key finding:** Llama 3.2 1B shows the strongest effect — factual recall
relies on early-position attention at mid-depth. Gemma 2's GQA + soft-capping
redistributes probability mass differently after knockout.

**Bug fixed during implementation:** Full-row knockout (zeroing all attention
from a position) caused NaN in softmax. Fixed by using single-edge knockout,
which is also more interpretable.

### 3.5 `steering_dose_response.rs` — Calibrating Intervention Strength ✅

**Status:** Implemented and tested on 3 models.

**API surface covered:**
- `SteeringCalibration`, `DoseResponseCurve`, `DoseResponsePoint`, `DOSE_LEVELS`
- `SteeringSpec`, `SteeringResult`, `InterventionType`
- `apply_steering()`, `measure_attention_to_targets()`
- `Intervention::Replace`, `HookPoint::AttnPattern`
- Multi-model auto-discovery via HuggingFace cache scanning

**Design:**
- Measures baseline attention from last token to position 0 at a target layer
- Creates `SteeringCalibration` from measured baseline and a 2× reference
- Sweeps `DOSE_LEVELS` (0.5–6.0): builds `SteeringSpec`, applies
  `apply_steering()` with renormalization, injects via `Intervention::Replace`
- Builds `SteeringResult` for KL divergence and logit diff analysis
- Records in `DoseResponseCurve`, queries `scale_for_target()` for interpolation
- Reports calibration info, dose-response table, and dose levels as absolute
  attention targets

**Test results:**

| Model | Layer | Baseline attn | Dose 0.5 KL | Dose 4.0 KL | Dose 6.0 KL |
|-------|-------|--------------|-------------|-------------|-------------|
| Llama 3.2 1B | 8 | 0.630 | 0.006 | 0.029 | 0.043 |
| Gemma 2 2B | 13 | 0.589 | 0.001 | 0.002 | 0.003 |
| StarCoder2 3B | 15 | 0.673 | 0.002 | 0.004 | 0.005 |

**Key finding:** Llama 3.2 1B shows the strongest dose-response: KL divergence
grows from 0.006 at half-dose to 0.043 at 6× dose. Gemma 2 2B is robust to
single-edge steering (KL stays below 0.004 even at 6× dose) due to GQA and
soft-capped logits.

### 3.6 `attention_patterns.rs` — Inspecting Head-Level Attention ✅

**Status:** Implemented and tested on 3 models.

**API surface covered:**
- `AttentionCache`, `HookPoint::AttnPattern`
- `attention_from_position()`, `attention_to_position()`,
  `top_attended_positions()`
- `HookSpec::capture()` at all layers in a single forward pass
- Multi-model auto-discovery via HuggingFace cache scanning

**Design:**
- Captures `AttnPattern` at all layers in a single forward pass
- Builds `AttentionCache`, queries per-layer attention distributions
- Per-layer top-5 attended positions for the last token
- Tracks incoming attention to position 0 across all layers
- Identifies peak last→first attention layer (connects to knockout experiment)

**Test results:**

| Model | Peak layer (last→first) | Peak attention | Top-1 at most layers |
|-------|------------------------|----------------|---------------------|
| Llama 3.2 1B | 2 | 0.847 | BOS |
| Gemma 2 2B | 22 | 0.845 | BOS |
| StarCoder2 3B | 26 | 0.866 | "The" (no BOS) |

**Key finding:** All three models show strong attention to the first token
across most layers (the "BOS sink" pattern). StarCoder2 3B lacks a BOS token
so the first real token ("The") serves as the attention sink. Llama 3.2 1B
peaks early (layer 2), while Gemma 2 2B peaks late (layer 22).

---

## 4. Proposed Future Examples

### Medium Impact

#### ~~`activation_patching.rs` — Causal Tracing via Patching~~ ✅ Implemented

~~Run two forward passes (clean and corrupted), capture `FullActivationCache` for
both, then patch clean activations into the corrupted run layer-by-layer using
`Intervention::Replace` to localize the causal site.~~

**Implemented** with position-specific patching (Meng et al., 2022): clean vs.
corrupted prompt ("France" → "Poland"/"Canada"), restore subject token residual
at each layer. Tested on Llama 3.2 1B, Gemma 2 2B, StarCoder2 3B.

---

#### `clt_attribution.rs` — Feature Attribution Scoring

Use `build_attribution_graph()` to score CLT features against a target logit
direction, then inspect `AttributionEdge` scores. Complements the existing
injection-only figure13 example.

**Note:** `CrossLayerTranscoder::build_attribution_graph` is not yet
implemented — this example depends on its addition (tracked in ROADMAP).

**Key API surface:**
`AttributionGraph`, `AttributionEdge`, `CrossLayerTranscoder::build_attribution_graph`

---

#### ~~`token_positions.rs` — Character-to-Token Mapping~~ ✅ Implemented

~~Demonstrate `EncodingWithOffsets`, `convert_positions()`, and the various
`char_to_token` / `token_to_char_range` helpers.~~

**Implemented** as a pure utility example (no GPU, no `transformer` feature).
Also added `encode_with_offsets()` and `encode_raw_with_offsets()` to
`MITokenizer`. Tested on Llama 3.2 1B, Gemma 2 2B, StarCoder2 3B.

---

### Lower Priority

#### `rwkv_inference.rs` — RWKV-7 Linear RNN Inference

Forward pass with `GenericRwkv`, demonstrating the recurrent state handling that
differs from transformer attention.

**Key API surface:**
`GenericRwkv`, `RwkvConfig`, `RwkvVersion`, `RwkvLoraDims`,
`StateKnockoutSpec`, `StateSteeringSpec`, `StateAblationResult`,
`StateSteeringResult`

---

#### `recurrent_feedback.rs` — Anacrousis / Recurrent Passes

Demonstrate `RecurrentPassSpec` and `RecurrentFeedbackEntry` for multi-pass
generation with recurrent feedback.

**Key API surface:**
`RecurrentFeedbackEntry`, `RecurrentPassSpec`

---

## 5. Coverage After All Proposed Examples

| Category | Before audit | After 5 implemented | After all proposed |
|---|---|---|---|
| Basic inference & loading | Covered | Covered | Covered |
| Download & auto-config | Covered | Covered | Covered |
| SAE encode / decode | Covered | Covered | Covered |
| CLT injection | Covered | Covered | Covered |
| **Text generation** | Missing | ✅ `generate.rs` | ✅ |
| **Logit lens** | Missing | ✅ `logit_lens.rs` | ✅ |
| **Attention knockout** | Missing | ✅ `attention_knockout.rs` | ✅ |
| **Steering calibration** | Missing | ✅ `steering_dose_response.rs` | ✅ |
| **Attention patterns** | Missing | ✅ `attention_patterns.rs` | ✅ |
| **Memory reporting** | N/A | ✅ Opt-in in all 5 examples | ✅ |
| **Activation patching** | Missing | ✅ `activation_patching.rs` | ✅ |
| **CLT attribution** | Missing | Missing | `clt_attribution.rs` |
| **Token positioning** | Missing | ✅ `token_positions.rs` | ✅ |
| **RWKV backend** | Missing | Missing | `rwkv_inference.rs` |
| **Recurrent feedback** | Missing | Missing | `recurrent_feedback.rs` |
| **KV-cached generation** | Missing | Missing | No proposed example yet |
| **PCA / dimensionality reduction** | Missing | Missing | `character_count_helix.rs` (§7) |
| **Paper replication** | Missing | Missing | `character_count_helix.rs` (§7) |

**Estimated coverage:** ~75 % now → ~95 % after all proposed examples.

---

## 6. Folder Structure (Option B — Flat Files + Results Directory)

After brainstorming, **Option B** was chosen: flat `.rs` files in `examples/`
with a shared `examples/results/` directory for golden JSON outputs. This was
preferred over subdirectory-per-example because:

1. Cargo auto-discovers flat `examples/*.rs` files without `path` overrides.
2. `cargo run --example <tab>` completion works naturally.
3. JSON results are co-located but separated from source code.

```
examples/
├── README.md                          # index + cross-model comparison tables

│   # ── Getting Started ──────────────────────────────────────
├── quick_start_transformer.rs         # (existing) basic inference
├── quick_start_sae.rs                 # (existing) SAE encode/decode
├── generate.rs                        # ✅ autoregressive generation loop

│   # ── Interpretability ─────────────────────────────────────
├── logit_lens.rs                      # ✅ layer-by-layer predictions
├── attention_knockout.rs              # ✅ head/edge ablation
├── attention_patterns.rs              # ✅ head-level attention
├── activation_patching.rs             # ✅ causal tracing (Meng et al., 2022)
├── steering_dose_response.rs          # ✅ intervention calibration
├── character_count_helix.rs           # (proposed) helix replication (§7)

│   # ── Sparse Features ──────────────────────────────────────
├── clt_attribution.rs                 # (proposed) CLT feature attribution
├── figure13_planning_poems.rs         # (existing) CLT injection + sweep
├── figure13/                          # (existing) figure13 assets

│   # ── Utilities ────────────────────────────────────────────
├── token_positions.rs                 # ✅ character-to-token mapping
├── fast_download.rs                   # (existing) parallel download
├── auto_config_dogfood.rs             # (existing) auto-config + compat check
├── screenshots/                       # (existing) auto-config screenshots

│   # ── Alternative Backends ─────────────────────────────────
├── rwkv_inference.rs                  # (proposed) RWKV-7 forward pass
├── recurrent_feedback.rs              # (proposed) anacrousis

│   # ── Golden Outputs ───────────────────────────────────────
└── results/                           # checked-in reference JSON
    ├── logit_lens/
    │   ├── llama-3.2-1b.json
    │   ├── gemma-2-2b.json
    │   └── starcoder2-3b.json
    ├── attention_knockout/
    │   ├── llama-3.2-1b.json
    │   ├── gemma-2-2b.json
    │   └── starcoder2-3b.json
    └── character_count_helix/         # (future)
        └── helix_plot.wl
```

### Golden outputs

Reference JSON files for key models are checked into `examples/results/`. These
serve as regression baselines and documentation of expected behavior. They are
**not** tested automatically — they document "what a real run produces" for
users who don't have the model weights cached.

### `Cargo.toml` `[[example]]` entries

Examples that import feature-gated types at compile time need explicit
`[[example]]` entries with `required-features` in `Cargo.toml`. Examples that
only need features at runtime (e.g., `quick_start_transformer`,
`auto_config_dogfood`) are auto-discovered by Cargo and compile without the
feature flag — they fail gracefully at runtime if the backend isn't available.

| Example | `required-features` | Status |
|---|---|---|
| `quick_start_transformer` | — (runtime only) | Existing, auto-discovered |
| `quick_start_sae` | `["sae", "transformer"]` | Existing |
| `figure13_planning_poems` | `["clt", "transformer"]` | Existing |
| `generate` | `["transformer"]` | ✅ Implemented |
| `logit_lens` | `["transformer"]` | ✅ Implemented |
| `attention_knockout` | `["transformer"]` | ✅ Implemented |
| `attention_patterns` | `["transformer"]` | ✅ Implemented |
| `activation_patching` | `["transformer"]` | ✅ Implemented |
| `steering_dose_response` | `["transformer"]` | ✅ Implemented |
| `character_count_helix` | `["transformer"]` | Proposed |
| `clt_attribution` | `["clt", "transformer"]` | Proposed |
| `token_positions` | — (no feature gate) | ✅ Implemented |
| `rwkv_inference` | `["rwkv"]` | Proposed |
| `recurrent_feedback` | `["transformer"]` | Proposed |
| `auto_config_dogfood` | — (runtime only) | Existing, auto-discovered |
| `fast_download` | — (no feature gate) | Existing, auto-discovered |

---

## 7. Paper Replication Study: Character Count Helix

**Paper:** Gurnee et al., "When Models Manipulate Manifolds: The Geometry of
a Counting Task", Transformer Circuits, October 2025.
https://transformer-circuits.pub/2025/linebreaks/index.html

### 7.1 Paper Summary

The paper studies how **Claude 3.5 Haiku** predicts newlines in fixed-width
line-wrapped text. The core finding: the model represents **line character
count** (characters since the last `\n`) as a 1-dimensional manifold (a
**helix**) embedded in a ~6-dimensional subspace of the residual stream.

Key results:

1. **Character count helix.** Average residual stream vectors at layer 2,
   grouped by character count (1–150), form a twisting curve in PCA space.
   The top 6 PCs capture 95% of the variance. Viewed in PC1–3, the curve
   resembles a helix; in PC4–6, a more complex twist. This "rippled"
   geometry is an optimal tradeoff between dimensionality and resolution.

2. **Ringing pattern.** Cosine similarities between mean activation vectors
   show off-diagonal bands — features nearby are positively correlated,
   those further away are negatively correlated, then positive again. This
   is the Gibbs-phenomenon-like "ringing" from projecting a high-curvature
   curve into a low-dimensional space.

3. **Causal validation.** Ablating the 6D character count subspace
   selectively disrupts newline predictions. Surgically patching mean
   activations for a different character count shifts the model's
   line-breaking behavior.

4. **Boundary head twist.** Dedicated attention heads "twist" the character
   count manifold via their QK circuit so that character count `i` aligns
   with line width `k = i + ε`, triggering attention to the previous newline
   when the line is nearly full.

5. **Distributed counting algorithm.** Multiple Layer 0 heads each output
   near-1D contributions; their *sum* forms the curved manifold. Layer 1
   heads further refine curvature.

### 7.2 The Helix Replication Experiment

The specific experiment to replicate the "character count subspace helix":

#### Inputs

- Take diverse prose passages, strip all newlines.
- Re-wrap at fixed width `k` characters (to the nearest word boundary ≤ k),
  for `k = 15, 20, 25, ..., 150`.
- This produces prompts where the ground-truth character count at each token
  position is known.

#### Activations to Capture

- Residual stream after an early layer, for **all token positions** (not
  just the last token). The paper uses layer 2 of Claude 3.5 Haiku (26
  layers total). For Gemma 2 2B (26 layers), the analogous hook is
  `HookPoint::ResidPost(1)` (0-indexed). The example should try layers
  0–3 and report which captures the most variance.
- Each captured tensor has shape `[seq_len, d_model]`.

#### Analysis

1. For each token position, compute its **line character count** = total
   characters since the last `\n`, including the current token's characters.
2. **Average** the residual stream vectors across all tokens sharing the
   same character count → 150 mean vectors of shape `[d_model]`.
3. **PCA** on the 150 × d_model matrix. The top 6 components should capture
   ~95% of the variance.
4. **Visualize** the 150 points projected into PC1–3 → expect the helix.
5. **Cosine similarity matrix** of the 150 mean vectors → expect the
   ringing pattern (diagonal band + off-diagonal oscillation).

### 7.3 Feasibility Assessment for candle-mi

#### What candle-mi already provides

| Requirement | Status | How |
|---|---|---|
| Load open-weight model | Yes | `MIModel::from_pretrained("model-id")` |
| Tokenize text | Yes | `MITokenizer::encode()` / `encode_raw()` |
| Capture residual stream at layer N, all positions | Yes | `HookPoint::ResidPost(n)` stores full `[seq_len, d_model]` tensor |
| Map tokens to character offsets | Yes | `EncodingWithOffsets`, `token_to_char_range()` |
| Extract per-position activations | Yes | `HookCache::get(&ResidPost(n))` then `narrow(0, pos, 1)` |
| Cosine similarity between tensors | Yes | Candle tensor ops (dot, div by norms) |
| Causal intervention (ablation) | Yes | `Intervention::Zero`, `Intervention::Replace` on the 6D subspace |

#### What is missing

| Requirement | Status | Notes |
|---|---|---|
| **PCA** | **Missing** | candle-mi has no PCA utility. Candle-core 0.9 has no SVD. Recommended approach: power iteration with deflation on the kernel matrix, using only candle `matmul` + arithmetic — runs on CPU or GPU with zero transfers. See "PCA approach" below. |
| **3D visualization** | **Out of scope** | The example should output data (CSV/JSON) for external plotting (Python matplotlib, Mathematica, etc.). This is the same pattern as `figure13_planning_poems.rs`. |
| **Claude 3.5 Haiku weights** | **Unavailable** | The paper uses a proprietary model. We must use an open-weight model. |

#### Which open model to use

The paper itself mentions attribution graphs for **Gemma 2 2B** and
**Qwen 3 4B** on the same linebreaking task, confirming the phenomenon
is not unique to Claude. Both models are validated in candle-mi:

- **Gemma 2 2B** (`google/gemma-2-2b`): fully validated, known to work
  well. ~8 GB VRAM at F32. Best candidate.
- **Qwen 2.5 Coder 3B** (`Qwen/Qwen2.5-Coder-3B`): validated, but the
  paper references Qwen 3 4B which differs.

**Recommendation:** Use **Gemma 2 2B**. The paper explicitly cites it as
exhibiting the same linebreaking circuitry.

#### PCA approach: GPU-native power iteration

Candle-core 0.9 has **no SVD**. However, PCA only requires `matmul` and
basic arithmetic — all GPU-accelerated in candle. The recommended approach
is **power iteration with deflation on the kernel matrix**.

**Algorithm.** Given the centered data matrix `X` of shape `[n, d]`
(here `n = 150`, `d = 2304`):

```
// Step 1: Kernel trick — work in sample space (150×150), not feature
//         space (2304×2304). One GPU matmul.
K = X @ X.T                          // shape [150, 150]

// Step 2: Power iteration for top eigenvector of K
v = random_unit_vector(150)
repeat ~50 times:
    v = K @ v                         // 150×150 matmul — microseconds
    v = v / ||v||                     // normalize

eigenvalue = v.T @ K @ v

// Step 3: Deflate and repeat for next eigenvector
K = K - eigenvalue * (v @ v.T)        // outer product via matmul
// → repeat Steps 2–3 for k = 6 components total

// Step 4: Recover principal directions in original d_model space
pc_i = X.T @ v_i / ||X.T @ v_i||     // shape [d_model]
```

**Every operation** — `matmul`, `div`, `sum`, `sqrt`, `sub`, outer product
via reshaped `matmul` — is a candle-core op that runs on whatever `Device`
the tensors live on. If the model is on CUDA, the PCA stays on CUDA with
zero host↔device transfers.

| Property | Value | Implication |
|---|---|---|
| Kernel matrix size | 150 × 150 | Power iteration converges in ~30 iters |
| Components needed | 6 | Only 6 deflation rounds |
| Total matmuls | ~180 tiny (150²) + 6 medium (150×2304) | Sub-millisecond on GPU |
| Accuracy | Exact for well-separated eigenvalues | Character count eigenvalues are well-separated per the paper |
| New dependencies | **None** | Pure candle tensor ops |

**Recommendation:** Add a `pub fn pca_top_k(matrix: &Tensor, k: usize)`
helper in `src/util/pca.rs`, ~40 lines of candle tensor ops, returning
`(components: Tensor, eigenvalues: Tensor)`. This runs transparently on
CPU or GPU and serves the helix example, logit lens, activation patching,
and any future analysis needing dimensionality reduction. No external
crate required.

### 7.4 Proposed Example: `character_count_helix.rs`

**Category:** Interpretability (high impact — replicates a landmark paper)

**Required features:** `["transformer"]`

**Outline:**

```
1. Load Gemma 2 2B via MIModel::from_pretrained
2. Build synthetic dataset:
   - Take a hardcoded prose passage (~500 words)
   - Strip newlines
   - Re-wrap at widths k = 20, 30, 40, ..., 100
3. For each wrapped passage:
   a. Tokenize
   b. Forward pass with HookSpec capturing ResidPost(0)..ResidPost(3)
   c. For each token position, compute line character count
   d. Accumulate (character_count → Vec<residual_vector>) mapping per layer
4. Average residual vectors per character count → 150 × d_model matrix
5. Center the matrix (subtract grand mean)
6. PCA via pca_top_k() — power iteration on the kernel matrix, on GPU
7. Project the 150 mean vectors into the 6D subspace
8. Output:
   - JSON of the 150 points in PC1–6 (for 3D plotting)
   - Cosine similarity matrix as JSON (for heatmap)
   - Explained variance per component
9. Print summary: variance captured, helix visual check
```

**What this demonstrates:**
- `MIModel::from_pretrained`, `HookSpec::capture`, `HookPoint::ResidPost`
- Full-sequence activation capture (not just last token)
- `EncodingWithOffsets` for character-to-token mapping
- `pca_top_k()` — GPU-native PCA via power iteration (new utility)
- Candle tensor ops: matmul, normalization, cosine similarity
- Real MI research workflow: synthetic data → activation capture → analysis
- JSON output + Mathematica `.wl` companion script (same pattern as figure13)

**What it does NOT attempt** (out of scope for one example):
- Boundary head QK twist analysis (would need weight extraction)
- Attribution graphs (would need crosscoders / CLT)
- Interactive 3D visualization (output JSON for Mathematica / matplotlib)

### 7.5 Verdict

**Feasible: YES.** The PCA gap is closed by a ~40-line `pca_top_k()`
utility using pure candle tensor ops (power iteration + deflation on the
kernel matrix). This runs on GPU with zero host↔device transfers, needs
no external dependency, and is reusable across multiple examples (logit
lens, activation patching, etc.).

The example replicates a key finding from a landmark Anthropic paper
using an open-weight model (Gemma 2 2B), exercises multiple uncovered API
surfaces, and follows the existing `figure13` pattern of producing data
for external visualization. It slots into the **Interpretability** group
in §6.

**Implementation order:**
1. Add `src/util/pca.rs` with `pca_top_k()` (power iteration, ~40 lines)
2. Write `character_count_helix.rs` example
3. Write companion `character_count_helix/helix_plot.wl` Mathematica script

---

## 8. Architecture: `memory` Feature for Real Measurements

During brainstorming, we decided that examples should report **real** RAM and
VRAM measurements, not estimates. This led to the implementation of the
`memory` feature (`src/memory.rs`), which provides:

- **`MemorySnapshot::now(device)`** — captures process RSS + optional VRAM
- **`MemoryReport::new(before, after)`** — computes deltas with `print_delta()`
  and `print_before_after()` formatting

### Platform support

| Metric | Windows | Linux |
|--------|---------|-------|
| RAM (RSS) | `K32GetProcessMemoryInfo` (per-process, exact) | `/proc/self/status` `VmRSS` (per-process, exact) |
| VRAM | `nvidia-smi` (device-wide) | `nvidia-smi` (device-wide) |

### Feature gate design

The `memory` feature relaxes `#![forbid(unsafe_code)]` to
`#![deny(unsafe_code)]` for the Windows FFI call, following the same pattern
as the existing `mmap` feature. This is documented in `CONVENTIONS.md` under
the `// SAFETY:` section with a policy table.

### Integration status

All 5 high-impact examples wrap model loading with `MemorySnapshot::now(device)`
before and after, then print the delta via `MemoryReport::print_before_after()`.
This gives users concrete numbers for their hardware. Activated with
`--features memory` (e.g., `cargo run --release --features transformer,memory
--example generate -- "meta-llama/Llama-3.2-1B"`).
