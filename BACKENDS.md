# How to Add a New Model Architecture

> This guide explains how candle-mi supports new model families.  There
> are three paths — from easiest (auto-config) to most flexible (custom
> backend) — depending on how close the new model is to a standard
> decoder-only transformer.

---

## Table of Contents

- [Overview: Three Paths](#overview-three-paths)
- [Path 1: Auto-Config (Zero Code)](#path-1-auto-config-zero-code)
  - [What Auto-Config Detects](#what-auto-config-detects)
  - [Compatibility Check](#compatibility-check)
  - [Limitations](#limitations)
- [Path 2: Config Parser (Recommended for Known Families)](#path-2-config-parser-recommended-for-known-families)
  - [Step 1: Identify Configuration Axes](#step-1-identify-configuration-axes)
  - [Step 2: Write the Parser](#step-2-write-the-parser)
  - [Step 3: Register the Model Type](#step-3-register-the-model-type)
  - [Step 4: Validate Against Python](#step-4-validate-against-python)
- [Path 3: Custom MIBackend (Non-Transformer Architectures)](#path-3-custom-mibackend-non-transformer-architectures)
  - [The MIBackend Trait](#the-mibackend-trait)
  - [Implementing forward()](#implementing-forward)
  - [Hook Integration Checklist](#hook-integration-checklist)
  - [Registering with MIModel::from_pretrained()](#registering-with-mimodelfrom_pretrained)
- [TransformerConfig Reference](#transformerconfig-reference)
  - [Configuration Axes](#configuration-axes)
  - [Existing Parsers as Templates](#existing-parsers-as-templates)
- [Testing Checklist](#testing-checklist)
- [Weight Naming Convention](#weight-naming-convention)

---

## Overview: Three Paths

| Path | When to use | Effort | Hook support |
|------|-------------|--------|--------------|
| **Auto-config** | Model uses HuggingFace-standard weight naming and is a standard decoder-only transformer | None | Full (automatic) |
| **Config parser** | Model is a decoder-only transformer with quirks (special norm, bias pattern, etc.) | ~30 lines | Full (automatic) |
| **Custom `MIBackend`** | Model is not a decoder-only transformer (e.g., RWKV, Mamba, encoder-decoder) | ~500+ lines | Manual (you place hooks) |

Most HuggingFace transformer models work out of the box via auto-config.
If yours doesn't, a config parser is usually enough.  A custom backend is
only needed for fundamentally different architectures.

---

## Path 1: Auto-Config (Zero Code)

When `MIModel::from_pretrained()` encounters an unknown `model_type` in
`config.json`, it attempts to infer a `TransformerConfig` automatically:

```rust
// Works for any model with standard HF weight naming
let model = MIModel::from_pretrained("some-org/novel-transformer-3B")?;
```

### What Auto-Config Detects

Auto-config reads configuration from two sources:

**Tier 1–2: `config.json` scalars** (same fields as known families):
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `vocab_size`
- `num_key_value_heads`, `head_dim`, `rms_norm_eps`, `rope_theta`, `max_position_embeddings`
- `tie_word_embeddings`, `attention_bias`

**Tier 3: safetensors tensor names** (structural inference):
- **QKV layout**: `qkv_proj` → `Fused`; `q_proj` + `k_proj` + `v_proj` → `Separate`
- **MLP layout**: `gate_up_proj` → `GatedFused`; `gate_proj` + `up_proj` → `GatedSeparate`; `fc1` + `fc2` → `Plain`
- **Bias flags**: presence of `.bias` tensors in attention/MLP projections
- **Norm type**: `input_layernorm.bias` → `LayerNorm`; otherwise `RmsNorm`

**Tier 4: `model_type` fixups** (architecture-specific overrides):
- Gemma/Gemma2 → `GemmaRmsNorm`, `embedding_scale`, `alternating_sliding_window`
- Any model with `attn_logit_softcapping` → soft-capping enabled

For a visual overview of how these config fields map to transformer blocks, see Raschka's
[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison).

### Compatibility Check

Before loading weights, `check_auto_compatibility()` runs a preflight
check that detects incompatible models:

```rust
use candle_mi::{CompatibilityReport, TransformerConfig};

let report = TransformerConfig::check_auto_compatibility(&config_json, &tensor_names);
if !report.compatible {
    for issue in &report.issues {
        eprintln!("  - {issue}");
    }
}
```

The check verifies:
- Required `config.json` fields are present
- Expected weight tensors exist (`embed_tokens`, `input_layernorm`, `lm_head`, etc.)
- When tensors are missing, it suggests the closest match ("did you mean?")
- Detects non-HF naming conventions (e.g., GGUF, custom prefixes)

### Limitations

Auto-config does **not** handle:
- Non-standard weight naming (e.g., `transformer.h.0.attn` instead of `model.layers.0.self_attn`)
- Architectures that aren't decoder-only transformers (RWKV, Mamba, encoder-decoder)
- Novel attention mechanisms (linear attention, local-global hybrid)
- Custom activations not in the standard set (`SiLU`, `GELU`, `GELU tanh`)

If auto-config fails, write a config parser (Path 2) or a custom backend
(Path 3).

### What failure looks like

The `auto_config_dogfood` example demonstrates both success and failure modes:

```bash
# Success — known family, uses manual parser
cargo run --release --features transformer --example auto_config_dogfood -- "meta-llama/Llama-3.2-1B"

# Failure — unsupported architecture (weight name mismatch)
cargo run --release --features transformer --example auto_config_dogfood -- "allenai/OLMo-1B-hf"

# Failure — actionable diagnostics (non-standard naming convention)
cargo run --release --features transformer --example auto_config_dogfood -- "EleutherAI/pythia-1.4b"
```

**OLMo-1B** fails the compatibility check because its weight names
(`model.layers.*.input_layernorm.weight`, `model.final_norm.weight`) do not
match the normalisation tensor patterns that `GenericTransformer` expects.

**Pythia 1.4B** uses the `gpt_neox.layers.{i}` weight prefix instead of the
HF-standard `model.layers.{i}`. The error message shows which tensors
*were* found for each expected category (embedding, norm, attention, MLP),
detects the GPT-NeoX / Pythia naming convention, and points to Phase 9
(tensor name remapping) for planned support. This is the diagnostic output
that tells contributors exactly where to look when adding a new model family.

---

## Path 2: Config Parser (Recommended for Known Families)

If the model is a decoder-only transformer but auto-config fails (or you
want guaranteed correctness), write a dedicated config parser.  This is
typically ~30 lines of code.

### Step 1: Identify Configuration Axes

Compare the new model to the closest existing family.  The axes to check:

| Axis | Question | Where to look |
|------|----------|---------------|
| Norm type | RmsNorm, LayerNorm, or GemmaRmsNorm? | `config.json` or paper |
| Activation | SiLU, GELU, or GELU tanh? | `hidden_act` field |
| QKV layout | Separate or fused? | Weight tensor names |
| MLP layout | GatedSeparate, GatedFused, or Plain? | Weight tensor names |
| QKV bias | Do Q, K, V have bias terms? | Weight tensor names |
| Use bias | Does every projection have bias? | Weight tensor names |
| Post-norms | 2 or 4 norms per layer? | Weight tensor names |
| Embedding scale | Multiply embeddings by `sqrt(hidden_size)`? | Paper or code |
| Soft-capping | Pre-softmax logit capping? | `attn_logit_softcapping` |
| Tied embeddings | `lm_head.weight` shared with `embed_tokens`? | `tie_word_embeddings` |
| Sliding window | Alternating full/sliding attention? | `sliding_window` |

### Step 2: Write the Parser

Add a `parse_*` function in `src/config.rs`.  Use an existing parser as a
template — `parse_llama` is the simplest baseline:

```rust
/// Parse a Llama-family `config.json`.
fn parse_llama(config: &Value) -> Result<TransformerConfig> {
    let base = parse_common_fields(config)?;
    Ok(TransformerConfig {
        norm_type: NormType::RmsNorm,
        activation: Activation::Silu,
        qkv_layout: QkvLayout::Separate,
        mlp_layout: MlpLayout::GatedSeparate,
        qkv_bias: false,
        use_bias: false,
        use_post_norms: false,
        embedding_scale: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: None,
        alternating_sliding_window: false,
        ..base
    })
}
```

For a model with quirks, override only what differs:

```rust
fn parse_qwen2(config: &Value) -> Result<TransformerConfig> {
    let base = parse_common_fields(config)?;
    // Qwen2: attention_bias applies to Q, K, V (NOT o_proj)
    let qkv_bias = config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    Ok(TransformerConfig {
        norm_type: NormType::RmsNorm,
        activation: Activation::Silu,
        qkv_layout: QkvLayout::Separate,
        mlp_layout: MlpLayout::GatedSeparate,
        qkv_bias,
        use_bias: false,
        use_post_norms: false,
        embedding_scale: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: None,
        alternating_sliding_window: false,
        ..base
    })
}
```

### Step 3: Register the Model Type

1. Add the `model_type` string to `SUPPORTED_MODEL_TYPES` in `src/config.rs`:

```rust
pub const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "gemma", "gemma2", "llama", "mistral",
    "my_new_model",  // ← add here
    "phi3", "qwen2", "starcoder2",
];
```

2. Add a match arm in `from_hf_config()`:

```rust
"my_new_model" => parse_my_new_model(config),
```

### Step 4: Validate Against Python

Run the model in both Python (HuggingFace Transformers) and Rust, and
compare logits:

```python
# Python reference
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("org/model", torch_dtype=torch.float32)
tokens = tokenizer("The capital of France is", return_tensors="pt")
logits = model(**tokens).logits[0, -1, :]  # last position
top5 = torch.topk(logits, 5)
```

```rust
// Rust — should match Python's top-5 exactly
let model = MIModel::from_pretrained("org/model")?;
let tokens = tokenizer.encode("The capital of France is")?;
// ... forward pass, extract last-position logits, compare top-5
```

With F32 on both sides, logits should match to ~6 decimal places.

---

## Path 3: Custom MIBackend (Non-Transformer Architectures)

For architectures that aren't decoder-only transformers (RWKV, Mamba,
encoder-decoder, etc.), implement the `MIBackend` trait directly.

### The MIBackend Trait

```rust
pub trait MIBackend: Send + Sync {
    // --- Required: metadata ---
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn num_heads(&self) -> usize;

    // --- Required: forward pass ---
    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache>;

    // --- Required: logit projection ---
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>;

    // --- Optional ---
    fn chat_template(&self, _prompt: &str, _system_prompt: Option<&str>) -> Option<String> {
        None
    }
    fn embedding_vector(&self, _token_id: u32) -> Result<Tensor> {
        Err(MIError::Hook("embedding_vector not supported".into()))
    }
}
```

**Contract:**
- `forward()` must return a `HookCache` with logits at shape `[batch, seq, vocab_size]`
- When `hooks` is empty, the forward pass must produce **zero extra allocations**
- `project_to_vocab()` applies the final norm + unembedding projection (for logit lens)
- The trait is `Send + Sync` because `MIModel` may be shared across threads

### Implementing forward()

The forward pass must integrate with the hook system.  Here is the
pattern:

```rust
fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
    // 1. Create a placeholder HookCache
    let placeholder = Tensor::zeros((1,), DType::F32, input_ids.device())?;
    let mut cache = HookCache::new(placeholder);

    // 2. Embedding
    let mut hidden = self.embed(input_ids)?;

    // 3. Hook: Embed (capture + intervene)
    if hooks.is_captured(&HookPoint::Embed) {
        cache.store(HookPoint::Embed, hidden.clone());
    }
    for intervention in hooks.interventions_at(&HookPoint::Embed) {
        hidden = apply_intervention(&hidden, intervention)?;
    }

    // 4. Layer loop
    for i in 0..self.num_layers {
        // ResidPre capture + intervene
        if hooks.is_captured(&HookPoint::ResidPre(i)) {
            cache.store(HookPoint::ResidPre(i), hidden.clone());
        }
        for intervention in hooks.interventions_at(&HookPoint::ResidPre(i)) {
            hidden = apply_intervention(&hidden, intervention)?;
        }

        // ... your layer logic with hook points at each stage ...

        // ResidPost capture + intervene
        if hooks.is_captured(&HookPoint::ResidPost(i)) {
            cache.store(HookPoint::ResidPost(i), hidden.clone());
        }
        for intervention in hooks.interventions_at(&HookPoint::ResidPost(i)) {
            hidden = apply_intervention(&hidden, intervention)?;
        }
    }

    // 5. Final logits
    let logits = self.compute_logits(&hidden)?;
    cache.set_output(logits);
    Ok(cache)
}
```

**Key points:**
- `apply_intervention()` is a crate-internal helper (`pub(crate)` in
  `src/hooks.rs`) — if you're implementing a backend inside the crate, use
  it directly; if outside, implement the intervention logic yourself.
- Capture checks (`is_captured`) are cheap `HashSet` lookups — when false,
  the `.clone()` is skipped entirely.
- Intervention iteration (`interventions_at`) returns an empty iterator
  when no interventions target that hook point.

### Hook Integration Checklist

For a new backend, decide which hook points to support.  At minimum:

| Hook Point | Priority | Purpose |
|------------|----------|---------|
| `Embed` | Required | Token embedding output |
| `ResidPre(i)` | Required | Residual stream before each layer |
| `ResidPost(i)` | Required | Residual stream after each layer (logit lens) |
| `FinalNorm` | Required | After final norm (logit lens) |
| `AttnPattern(i)` | Recommended | Attention visualization |
| `AttnScores(i)` | Recommended | Knockout interventions |
| `AttnQ/K/V(i)` | Optional | Q, K, V inspection |
| `MlpPre/Post/Out(i)` | Optional | MLP analysis |

Backend-specific hook points should use `HookPoint::Custom("name")` or
propose a new enum variant.

### Registering with MIModel::from_pretrained()

To make your backend discoverable via `from_pretrained()`, add a match arm
in `src/backend.rs` inside `MIModel::from_pretrained()`:

```rust
#[cfg(feature = "my_backend")]
"my_model_type" => {
    let config = MyConfig::from_hf_config(&json)?;
    let model = MyBackend::load(config, &device, dtype, vb)?;
    Ok(Self::with_tokenizer(Box::new(model), device, tokenizer))
}
```

The backend is gated behind a feature flag, so users only compile what they
need.

### Stoicheia: A Worked Example

The `stoicheia` module (`src/stoicheia/`) implements two custom `MIBackend`s
for ARC's [AlgZoo](https://github.com/alignment-research-center/alg-zoo)
tiny models (8–1,408 parameters):

- **`StoicheiaRnn`** — single-layer ReLU RNN (continuous input). Uses
  `HookPoint::Custom("rnn.hook_hidden.{t}")` for per-timestep activation
  capture.
- **`StoicheiaTransformer`** — attention-only transformer (discrete input,
  no MLP, no causal mask). Reuses standard `HookPoint` variants (`Embed`,
  `AttnScores`, `AttnPattern`, `ResidPre/Post`).

These load from `.safetensors` or `.pth` files (auto-detected by extension),
use explicit `load(config, path, device)` constructors instead of
`from_pretrained()`, and demonstrate how a non-language-model backend fits
into the `MIBackend` trait. See `src/stoicheia/mod.rs` for the full
implementation (~500 lines for both backends).

---

## TransformerConfig Reference

### Configuration Axes

The `GenericTransformer` is parameterized by these fields:

| Field | Type | Description |
|-------|------|-------------|
| `hidden_size` | `usize` | Model dimension (`d_model`) |
| `num_layers` | `usize` | Number of decoder blocks |
| `num_attention_heads` | `usize` | Number of query heads |
| `num_kv_heads` | `usize` | Number of key/value heads (GQA) |
| `head_dim` | `usize` | Per-head dimension |
| `intermediate_size` | `usize` | MLP hidden dimension |
| `vocab_size` | `usize` | Vocabulary size |
| `norm_type` | `NormType` | `RmsNorm`, `LayerNorm`, or `GemmaRmsNorm` |
| `norm_eps` | `f64` | Normalization epsilon |
| `activation` | `Activation` | `Silu`, `Gelu`, or `GeluApprox` |
| `qkv_layout` | `QkvLayout` | `Separate` or `Fused` |
| `mlp_layout` | `MlpLayout` | `GatedSeparate`, `GatedFused`, or `Plain` |
| `qkv_bias` | `bool` | Bias on Q, K, V projections |
| `use_bias` | `bool` | Bias on all projections |
| `use_post_norms` | `bool` | 4-norm layers (Gemma 2) |
| `embedding_scale` | `Option<f64>` | Multiply embeddings by this value |
| `attn_logit_softcapping` | `Option<f64>` | Pre-softmax logit soft-capping threshold |
| `query_pre_attn_scalar` | `Option<f64>` | Custom Q scaling (overrides `1/sqrt(head_dim)`) |
| `rope_theta` | `f64` | RoPE base frequency |
| `max_position_embeddings` | `usize` | Maximum sequence length |
| `tie_word_embeddings` | `bool` | Share `embed_tokens` and `lm_head` weights |
| `alternating_sliding_window` | `bool` | Alternating full/sliding attention (Gemma 2) |
| `sliding_window` | `Option<usize>` | Sliding window size |

### Existing Parsers as Templates

| Family | Parser | Key differences from LLaMA baseline |
|--------|--------|-------------------------------------|
| LLaMA | `parse_llama` | Baseline: GQA, SiLU, RmsNorm, separate QKV, gated MLP |
| Qwen2 | `parse_qwen2` | + `qkv_bias: true` (Q, K, V only, not `o_proj`) |
| Gemma | `parse_gemma` | + `GemmaRmsNorm`, `embedding_scale`, `GeluApprox` |
| Gemma 2 | `parse_gemma2` | + post-norms, soft-capping, `query_pre_attn_scalar`, alternating sliding window |
| Phi-3 | `parse_phi3` | + fused QKV, fused gate+up MLP |
| StarCoder2 | `parse_starcoder2` | + `LayerNorm`, `Plain` MLP, `use_bias: true`, `GeluApprox` |
| Mistral | `parse_mistral` | + sliding window attention |

---

## Testing Checklist

When adding a new model family, verify:

- [ ] **Config parsing**: `TransformerConfig::from_hf_config(&json)` produces the correct config for a known model
- [ ] **Forward pass**: top-5 predictions match Python HuggingFace Transformers (F32 on both sides)
- [ ] **Hook capture**: all hook points produce tensors with the expected shapes
- [ ] **Intervention**: `Intervention::Zero` at `ResidPost(0)` changes the output (proves hooks are wired)
- [ ] **Logit lens**: `project_to_vocab()` produces meaningful predictions at intermediate layers
- [ ] **No regression**: existing model tests still pass (`cargo test --features transformer`)

For GPU testing, run with `--test-threads=1` to avoid OOM on 3B+ models.

---

## Weight Naming Convention

The `GenericTransformer` expects HuggingFace-standard weight names under
the `model.` prefix:

```
model.embed_tokens.weight                         # [vocab, hidden]
model.layers.{i}.input_layernorm.weight           # [hidden]
model.layers.{i}.self_attn.q_proj.weight          # [n_heads * head_dim, hidden]
model.layers.{i}.self_attn.k_proj.weight          # [n_kv_heads * head_dim, hidden]
model.layers.{i}.self_attn.v_proj.weight          # [n_kv_heads * head_dim, hidden]
model.layers.{i}.self_attn.o_proj.weight          # [hidden, n_heads * head_dim]
model.layers.{i}.post_attention_layernorm.weight  # [hidden]
model.layers.{i}.mlp.gate_proj.weight             # [intermediate, hidden]
model.layers.{i}.mlp.up_proj.weight               # [intermediate, hidden]
model.layers.{i}.mlp.down_proj.weight             # [hidden, intermediate]
model.norm.weight                                 # [hidden]
lm_head.weight                                    # [vocab, hidden]
```

Variants by family:

| Family | Difference |
|--------|-----------|
| Phi-3 | `qkv_proj` instead of `q/k/v_proj`; `gate_up_proj` instead of `gate_proj` + `up_proj` |
| StarCoder2 | `fc1`/`fc2` instead of `gate_proj`/`up_proj`/`down_proj`; `.bias` on all projections |
| Gemma 2 | + `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `post_attention_layernorm` (4 norms) |
| Tied embeddings | No `lm_head.weight` — reuses `embed_tokens.weight` |

Models with non-standard naming (e.g., `transformer.h.{i}.attn` instead
of `model.layers.{i}.self_attn`) require a custom `MIBackend` (Path 3).
