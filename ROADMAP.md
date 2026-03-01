# Roadmap: A Rust Crate for Mechanistic Interpretability

> *MI for the Rust of us*

**Date:** February 19, 2026 (last updated: March 1, 2026)
**Status:** Phase 0 + Phase 1 + Phase 2 complete. Published on [crates.io](https://crates.io/crates/candle-mi) as v0.0.3.
**Context:** Building on plip-rs experience (7 model backends incl. Gemma 2, attention knockout, state knockout, effective attention, steering, logit lens, CLT encoding/injection). Two successful replications of Anthropic's "Planning in Poems" Figure 13 validate the approach: Gemma 2 2B with 426K CLTs (melometis branch) and Llama 3.2 1B with 524K CLTs (tragos branch). Target: a publishable, generic Rust MI crate endorsed by HuggingFace.

---

## Table of Contents

- [0. Naming & License](#0-naming--license)
- [1. What Exists Today (State of the Art)](#1-what-exists-today-state-of-the-art)
  - [1.1 Python MI Ecosystem](#11-python-mi-ecosystem)
  - [1.2 Rust MI Ecosystem](#12-rust-mi-ecosystem)
  - [1.3 candle-transformers](#13-candle-transformers)
  - [1.4 Open Interpretability Dictionaries (CLTs and SAEs)](#14-open-interpretability-dictionaries-clts-and-saes)
- [2. Architecture: What We Build](#2-architecture-what-we-build)
  - [2.1 Design Philosophy](#21-design-philosophy)
  - [2.2 What We Reuse from plip-rs](#22-what-we-reuse-from-plip-rs)
  - [2.3 What We Change from plip-rs](#23-what-we-change-from-plip-rs)
- [3. The Generic Transformer](#3-the-generic-transformer)
  - [3.1 Configuration Axes](#31-configuration-axes)
  - [3.2 Config Struct](#32-config-struct)
  - [3.3 Coverage](#33-coverage)
  - [3.4 What It Does NOT Cover](#34-what-it-does-not-cover)
  - [3.5 Config Parsing from HuggingFace `config.json`](#35-config-parsing-from-huggingface-configjson)
  - [3.6 Weight Name Mapping](#36-weight-name-mapping)
- [4. Generic RWKV / Linear RNN Support](#4-generic-rwkv--linear-rnn-support)
  - [4.1 The RWKV Family](#41-the-rwkv-family)
  - [4.2 Available Models (HuggingFace, safetensors)](#42-available-models-huggingface-safetensors)
  - [4.3 Architectural Comparison](#43-architectural-comparison)
  - [4.4 Generic RWKV Design](#44-generic-rwkv-design)
  - [4.5 Related Linear RNN / SSM Architectures](#45-related-linear-rnn--ssm-architectures)
- [5. MI Capabilities](#5-mi-capabilities)
  - [5.1 Core (from plip-rs, reusable)](#51-core-from-plip-rs-reusable)
  - [5.2 New (required for Melomētis, general-purpose)](#52-new-required-for-melomētis-general-purpose)
  - [5.3 Future (not required now)](#53-future-not-required-now)
- [6. Crate Structure](#6-crate-structure)
  - [6.1 Feature Gates](#61-feature-gates)
  - [6.2 Documentation](#62-documentation)
- [7. Phased Development Plan](#7-phased-development-plan)
  - [7.0 Git Workflow](#70-git-workflow)
  - [Phase 0: Foundation](#phase-0-foundation) ✅
  - [Phase 1: Generic Transformer](#phase-1-generic-transformer) ✅
  - [Phase 2: RWKV-6 + RWKV-7](#phase-2-rwkv-6--rwkv-7)
  - [Phase 3: CLT Support](#phase-3-clt-support)
  - [Phase 3b: SAE Support](#phase-3b-sae-support)
  - [Phase 4: Polish + Publish](#phase-4-polish--publish)
  - [Phase 5+: Extensions (Future)](#phase-5-extensions-future)
- [8. Key Design Decisions](#8-key-design-decisions)
- [9. Relationship to Existing Projects](#9-relationship-to-existing-projects)
  - [9.1 plip-rs (AIware 2026)](#91-plip-rs-aiware-2026)
  - [9.2 Melomētis + Tragos — Planning in Poems](#92-melomētis--tragos--planning-in-poems-plip-rs-melometis-branch-v140)
  - [9.3 Deloson (Visualization)](#93-deloson-visualization)
  - [9.4 candle-transformers](#94-candle-transformers)
- [10. Risk Assessment](#10-risk-assessment)
- [References](#references)

---

## 0. Naming & License

### Name: `candle-mi`

The initial concern was that all `candle-*` crates on crates.io are published by HuggingFace, and using the prefix might imply official affiliation. This is now resolved: **Eric Buehler** (HuggingFace candle collaborator) [endorsed the `candle-mi` name](https://github.com/huggingface/candle/discussions/3368) and invited a pull request to add the crate to candle's "Useful External Resources" section.

The name is short, unambiguous, and signals both the candle foundation and the MI purpose. All type names follow: `MIBackend`, `MIModel`, `MITokenizer`, etc.

### License: MIT OR Apache-2.0

Dual-licensed under MIT and Apache-2.0, the standard for the Rust ecosystem and consistent with candle itself. Each source file carries the SPDX header:

```
// SPDX-License-Identifier: MIT OR Apache-2.0
```

`Cargo.toml`:
```toml
license = "MIT OR Apache-2.0"
```

---

## 1. What Exists Today (State of the Art)

### 1.1 Python MI Ecosystem

| Library | Approach | Model Support | Key Insight |
|---------|----------|---------------|-------------|
| **TransformerLens** | Re-implements models with hook points | 50+ architectures (canonical format) | Gold standard; `run_with_cache`, `run_with_hooks` |
| **nnsight** | Wraps existing HF models via tracing proxy | Any PyTorch model | No re-implementation; remote execution via NDIF |
| **nnterp** | Standardized names on top of nnsight | 50+ variants across 16 families | Best of both: wrap existing model, uniform naming |
| **pyvene** | Configuration-based interventions | Any PyTorch model (incl. RNNs, CNNs) | Trainable interventions; declarative API |
| **SAELens** | SAE training + analysis | GPT-2, Gemma 2, LLaMA 3 | HookedSAETransformer for integrated MI+SAE |
| **circuit-tracer** | CLT attribution graphs | Gemma 2 2B, LLaMA 3.2 1B, Qwen-3 | Anthropic's open-source tool for circuit discovery |
| **sparsify** | SAE + transcoder training | Any HF model | EleutherAI; TopK SAEs |

### 1.2 Rust MI Ecosystem

**No published crates.** plip-rs is the only Rust project performing MI operations on language models (attention knockout, state knockout, effective attention, steering, logit lens, CLT encoding/injection). It has produced two independent replications of Anthropic's "Planning in Poems" Figure 13 ([Lindsey et al. 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location)):

| Replication | Model | CLT | Branch | Key Result |
|------------|-------|-----|--------|------------|
| **Melomētis** | Gemma 2 2B | 426K features | `melometis` | 48.3% probability redirect at planning site; 70% of 136 suppress+inject pairs peak at planning site |
| **Tragos** | Llama 3.2 1B | 524K features | `tragos` | Second independent replication confirming the phenomenon generalises across architectures |

These results validate the plip-rs infrastructure that candle-mi will reuse and extend.

### 1.3 candle-transformers

Over 100 model implementations, all separate files, no unified hook wrapper, no MI features. Same architectural pattern as plip-rs (separate forward files per model). No generic/config-driven transformer exists.

### 1.4 Open Interpretability Dictionaries (CLTs and SAEs)

| Model | CLT Weights | SAE Weights |
|-------|-------------|-------------|
| **Gemma 2 2B** | circuit-tracer (426K, 2.5M features) | Gemma Scope (JumpReLU, every layer/sublayer) |
| **Gemma 3 270M/1B** | Gemma Scope 2 (transcoders) | Gemma Scope 2 |
| **LLaMA 3.2 1B** | circuit-tracer (524K features) | Llama Scope |
| **Qwen-3 (0.6B-14B)** | circuit-tracer | — |
| **GPT-2 Small** | openCLT (incomplete) | SAELens (MLP transcoders) |
| **Claude 3.5 Haiku** | Internal only (not released) | Internal only |

---

## 2. Architecture: What We Build

### 2.1 Design Philosophy

**TransformerLens approach** (re-implement models with hooks built in), NOT nnsight approach (wrap existing models). Rationale:
- Rust has no equivalent of PyTorch's `register_forward_hook` — we can't wrap arbitrary code
- Re-implementation gives us full control over hook placement and data extraction
- plip-rs already validates this approach across 7 architectures (incl. Gemma 2)
- The generic transformer (see §3) amortizes the re-implementation cost: one implementation covers ~80% of modern LLMs

**TransformerLens hook points as the reference API:**

| Hook Point | Location | Shape |
|------------|----------|-------|
| `hook_embed` | After token embedding | `[batch, seq, d_model]` |
| `blocks.{i}.hook_resid_pre` | Before layer i | `[batch, seq, d_model]` |
| `blocks.{i}.attn.hook_q` | Query vectors | `[batch, seq, n_heads, d_head]` |
| `blocks.{i}.attn.hook_k` | Key vectors | `[batch, seq, n_heads, d_head]` |
| `blocks.{i}.attn.hook_v` | Value vectors | `[batch, seq, n_heads, d_head]` |
| `blocks.{i}.attn.hook_scores` | Pre-softmax attention | `[batch, n_heads, seq_q, seq_k]` |
| `blocks.{i}.attn.hook_pattern` | Post-softmax attention | `[batch, n_heads, seq_q, seq_k]` |
| `blocks.{i}.hook_attn_out` | Attention output | `[batch, seq, d_model]` |
| `blocks.{i}.hook_resid_mid` | Between attention and MLP | `[batch, seq, d_model]` |
| `blocks.{i}.mlp.hook_pre` | MLP pre-activation | `[batch, seq, d_mlp]` |
| `blocks.{i}.mlp.hook_post` | MLP post-activation | `[batch, seq, d_mlp]` |
| `blocks.{i}.hook_mlp_out` | MLP output | `[batch, seq, d_model]` |
| `blocks.{i}.hook_resid_post` | After full layer | `[batch, seq, d_model]` |
| `hook_final_norm` | After final norm | `[batch, seq, d_model]` |

Not all hooks need to be captured on every forward pass. A `HookSpec` enum selects which hooks are active, so the default path has zero overhead.

### 2.2 What We Reuse from plip-rs

**Directly reusable (~3500 lines, ~30% of plip-rs):**

| Module | Lines | Status |
|--------|-------|--------|
| `model.rs` → `backend.rs` | ~700 | Rename `PlipBackend` → `MIBackend`; keep trait methods, `PlipModel` → `MIModel`, `PlipTokenizer` → `MITokenizer` |
| `intervention.rs` | ~1700 | Copy as-is; attention knockout/steering are 100% model-agnostic; state interventions stay as optional trait methods |
| `kv_cache.rs` | ~200 | Copy as-is; pure tensor storage |
| `cache.rs` | ~100 | Copy as-is; per-layer activation storage |
| `masks.rs` | ~150 | Copy as-is; causal/generation masks with caching |
| `logit_lens.rs` | ~200 | Copy as-is; architecture-agnostic hidden→vocab projection |
| `positioning.rs` | ~200 | Copy as-is; character↔token mapping |
| `steering.rs` | ~300 | Refactor: remove Python/Rust labels, keep calibration+dose-response logic |
| `tokenizer_rwkv.rs` | ~300 | Keep as optional feature (`feature = "rwkv-tokenizer"`) |

**Not reused directly (architecture-specific, refactored into generic backends, ~8000 lines):**

| Module | Reason |
|--------|--------|
| `forward.rs` (StarCoder2) | Replaced by generic transformer |
| `forward_qwen2.rs` (Qwen2) | Replaced by generic transformer |
| `forward_gemma.rs` (Gemma) | Replaced by generic transformer |
| `forward_gemma2.rs` (Gemma 2) | Replaced by generic transformer |
| `forward_llama.rs` (LLaMA) | Replaced by generic transformer |
| `forward_phi3.rs` (Phi-3) | Replaced by generic transformer |
| `forward_rwkv6.rs` (RWKV-6) | Replaced by generic RWKV (see §4) |
| `clt.rs` | Refactor into `interp/clt.rs`; validated against circuit-tracer Python (90/90 top-10 matches) |
| `corpus.rs` | PLIP-paper-specific corpus format |
| `experiment.rs` | PLIP-paper-specific experiment runner |
| `probe.rs` | Move to optional `probing` feature or separate crate |

### 2.3 What We Change from plip-rs

1. **Trait redesign.** `PlipBackend` becomes `MIBackend` with a single hook-aware forward method:
   - `forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache>` — captures and intervenes via `HookSpec`; replaces plip-rs's `forward_with_cache`, `forward_with_attention`, `forward_with_kv_cache`
   - `project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>` — logit projection (used by logit lens)
   - Metadata: `num_layers()`, `hidden_size()`, `vocab_size()`, `num_heads()`
   - Optional: `chat_template()`, `embedding_vector()`

2. **Hook-based architecture.** Instead of plip-rs's separate methods for each extraction type, use a single `HookSpec` that declares what to capture:
   ```rust
   let mut hooks = HookSpec::new();
   hooks.capture(HookPoint::AttnPattern(5));
   hooks.capture(HookPoint::ResidPost(5));
   hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
   let cache = model.forward(&tokens, &hooks)?;
   let attn = cache.require(&HookPoint::AttnPattern(5))?;
   ```

3. **Config-driven model loading.** Replace `ModelArchitecture::from_model_id()` string matching with `config.json` parsing that reads `model_type` and `architectures` fields (like HuggingFace does).

---

## 3. The Generic Transformer

### 3.1 Configuration Axes

Analysis of plip-rs's 6 transformer backends (StarCoder2, Qwen2, Gemma, Gemma 2, LLaMA, Phi-3) reveals ~12 configuration axes:

| Axis | Variants | Models |
|------|----------|--------|
| **Normalization** | `RmsNorm` / `LayerNorm` / `GemmaRmsNorm` (+1 to weight) | Most use RmsNorm; Gemma has custom variant |
| **Activation** | `Silu` / `Gelu` / `GeluApprox` (PyTorch tanh approx) | LLaMA/Qwen/Phi=Silu; StarCoder2/Gemma 2=GeluApprox; Gemma=Gelu |
| **QKV layout** | Separate Q,K,V projections / Fused single QKV | All separate except Phi-3 (fused) |
| **MLP layout** | Gated (gate+up→down) / Fused gated / Plain (fc→proj) | LLaMA/Qwen/Gemma=gated separate; Phi-3=gated fused; StarCoder2=plain |
| **QKV bias** | Yes / No | Qwen2 on Q,K,V; StarCoder2 on all projections |
| **Bias granularity** | `o_proj_bias`, `mlp_bias` (separate from QKV) | StarCoder2 has bias everywhere; Qwen2 only on Q,K,V |
| **Embedding** | Standard / Scaled (`* sqrt(hidden)`) | Only Gemma scales |
| **LM head** | Tied to embeddings / Separate | Controlled by `tie_word_embeddings` |
| **Gemma 2 extensions** | Attention logit soft-capping, final logit soft-capping, 4-norm per layer, custom attention scalar, alternating sliding window | Gemma 2 only |
| **Sliding window** | Global / Fixed window / Alternating (per-layer) | Mistral=fixed; Gemma 2=alternating |

### 3.2 Config Struct

```rust
pub struct TransformerConfig {
    // Dimensions
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,              // usually hidden_size / num_attention_heads
    pub intermediate_size: usize,
    pub vocab_size: usize,

    // Architecture axes
    pub norm_type: NormType,           // RmsNorm | LayerNorm | GemmaRmsNorm
    pub norm_eps: f64,
    pub activation: Activation,        // Silu | Gelu | GeluApprox
    pub qkv_layout: QkvLayout,        // Separate | Fused
    pub mlp_layout: MlpLayout,        // GatedSeparate | GatedFused | Plain
    pub qkv_bias: bool,
    pub o_proj_bias: bool,             // StarCoder2
    pub mlp_bias: bool,                // StarCoder2
    pub embedding_scale: Option<f64>,  // None or Some(sqrt(hidden_size))
    pub tie_word_embeddings: bool,

    // Positional encoding
    pub rope_theta: f64,
    pub max_position_embeddings: usize,

    // Gemma 2 extensions
    pub attn_logit_softcapping: Option<f64>,   // Some(50.0) for Gemma 2
    pub final_logit_softcapping: Option<f64>,  // Some(30.0) for Gemma 2
    pub query_pre_attn_scalar: Option<f64>,    // Some(256.0) for Gemma 2
    pub use_post_norms: bool,                  // 4-norm per layer (Gemma 2)

    // Sliding window attention
    pub sliding_window: Option<usize>,         // Mistral, Gemma 2
    pub alternating_sliding_window: bool,      // Gemma 2
}
```

### 3.3 Coverage

With this config, **one implementation** covers:

| Model Family | Config Notes | Validated |
|--------------|-------------|-----------|
| **LLaMA 1/2/3 / Code-LLaMA** | GQA, SiLU, RmsNorm, separate lm_head | LLaMA 3.2 1B |
| **Qwen 2 / 2.5** | GQA, SiLU, RmsNorm, QKV bias, conditional tied embeddings | Qwen2.5-Coder-3B |
| **Gemma 1 / CodeGemma** | GQA, GELU, GemmaRmsNorm, sqrt embedding scale, tied lm_head | — |
| **Gemma 2** | + GeluApprox, soft-capping, 4-norm, custom attn scalar, alternating sliding window | Gemma 2 2B |
| **Phi-3 / Phi-4** | GQA, SiLU, RmsNorm, fused QKV, fused MLP | Phi-3 Mini 4K |
| **StarCoder2** | GQA, GeluApprox, RmsNorm, plain MLP, bias everywhere, tied lm_head | StarCoder2 3B |
| **Mistral / Mixtral** (dense layers) | GQA, SiLU, RmsNorm, sliding window | Mistral 7B v0.1 |
| **DeepSeek** (dense layers) | GQA, SiLU, RmsNorm | — |
| **Yi** | GQA, SiLU, RmsNorm (LLaMA-like) | — |
| **InternLM 2** | GQA, SiLU, RmsNorm (LLaMA-like) | — |

**VRAM budget (BF16, RTX 5060 Ti 16 GB).** Non-validated families have at least one size that fits on 16 GB:

| Model Family | Smallest Size | BF16 VRAM est. |
|--------------|---------------|----------------|
| Gemma 1 / CodeGemma | 2B (~2.5 B actual) | ~6 GB |
| DeepSeek-Coder (dense) | 1.3B | ~3.6 GB |
| Yi 1.5 | 6B | ~13 GB |
| InternLM 2 | 1.8B | ~4.6 GB |
| Phi-4-mini | 3.8B | ~8.6 GB |

The 7 B variants (DeepSeek-LLM 7B, InternLM 2 7B) are borderline at ~15 GB and require short sequences.

### 3.4 What It Does NOT Cover

| Architecture | Why | Solution |
|-------------|-----|----------|
| **Mixture-of-Experts** (Mixtral, DeepSeek-V2/V3) | Router + expert selection | Separate `MoETransformer` implementation |
| **RWKV / Linear RNN** | Fundamentally different (no attention matrix) | Separate generic RWKV backend (see §4) |
| **Mamba / SSM** | Selective state space; different recurrence | Separate backend |
| **Encoder-only** (BERT, etc.) | Bidirectional attention, [MASK] token | Different forward pass structure |
| **Encoder-decoder** (T5, etc.) | Cross-attention between encoder and decoder | Different forward pass structure |
| **Very old architectures** (GPT-2, GPT-J) | Absolute position embeddings, post-norm | Could be added as config variants if needed |
| **Codestral** (Mistral) | Only available at 22 B; exceeds 16 GB MI GPU budgets | Would need ≥48 GB VRAM (A6000, etc.) |

### 3.5 Config Parsing from HuggingFace `config.json`

The generic transformer reads `model_type` from `config.json` and maps it to a `TransformerConfig`:

```rust
impl TransformerConfig {
    pub fn from_hf_config(config: &serde_json::Value) -> Result<Self> {
        let model_type = config["model_type"].as_str()
            .ok_or_else(|| MIError::Config("missing 'model_type'".into()))?;
        match model_type {
            "llama" => Self::parse_llama(config),
            "qwen2" => Self::parse_qwen2(config),
            "gemma" => Self::parse_gemma(config),
            "gemma2" => Self::parse_gemma2(config),
            "phi3" => Self::parse_phi3(config),
            "starcoder2" => Self::parse_starcoder2(config),
            "mistral" => Self::parse_mistral(config),
            other => Err(MIError::Config(format!("unsupported model_type: '{other}'"))),
        }
    }
}
```

Each `parse_*` function reads the relevant fields and sets the config knobs. Adding a new model family = adding one `parse_*` function (~30 lines), not a new forward pass implementation.

### 3.6 Weight Name Mapping

Different model families use different weight naming conventions. All 7 validated families happen to share the same `model.layers.{i}.self_attn.*` / `model.layers.{i}.mlp.*` naming scheme (LLaMA-style), so weight loading is handled directly in `GenericTransformer::load()` using `VarBuilder` path prefixes. If a future model family requires different weight names, a `WeightMap` trait can be introduced at that point.

---

## 4. Generic RWKV / Linear RNN Support

### 4.1 The RWKV Family

| Version | Name | Year | Key Innovation | State Shape |
|---------|------|------|----------------|-------------|
| **RWKV-4** | Dove | 2023 | Scalar WKV with normalization denominator | Vector (scalar per channel) |
| **RWKV-5** | Eagle | 2024 | Matrix-valued states, multi-head, gating | `[n_heads, head_dim, head_dim]` |
| **RWKV-6** | Finch | 2024 | Data-dependent decay + token shift (LoRA) | `[n_heads, head_dim, head_dim]` |
| **RWKV-7** | Goose | 2025 | Generalized delta rule (diag + rank-1 state transition) | `[n_heads, head_dim, head_dim]` |

**Status:** RWKV team has declared v4-v6 archived. v7 is the active version. plip-rs implements v6.

### 4.2 Available Models (HuggingFace, safetensors)

| Version | Sizes | HF Repos |
|---------|-------|----------|
| **RWKV-4** | 169M, 430M, 1.5B, 3B, 7B, 14B | `RWKV/rwkv-4-*-pile`, `RWKV/rwkv-raven-*` |
| **RWKV-5** | 7B | `RWKV/v5-Eagle-7B-HF` |
| **RWKV-6** | 1.6B, 3B, 7B, 14B | `RWKV/v6-Finch-*-HF` |
| **RWKV-7** | 0.1B, 0.4B, 1.5B, 2.9B | `RWKV/RWKV7-Goose-*-HF` (fla format) |

### 4.3 Architectural Comparison

All RWKV versions share the same block structure:

```
Input → Embedding → [Pre-LN (layer 0 only)]
  → for each layer:
      → LN1 → TimeMix (recurrence) → residual add
      → LN2 → ChannelMix (FFN) → residual add
  → Final LN → LM Head → logits
```

The differences are **inside TimeMix** (the recurrence formula) and **inside ChannelMix** (the FFN):

#### TimeMix Recurrence

| Component | RWKV-4 | RWKV-5 | RWKV-6 | RWKV-7 |
|-----------|--------|--------|--------|--------|
| **Token shift** | Static lerp (`mu * x + (1-mu) * prev`) | Static lerp | Data-dependent lerp (LoRA: `A[D,32] @ B[32,D]`) | Static lerp (reverted from v6) |
| **State transition** | `N/D` ratio (scalar state with denominator) | `diag(w) * S + k^T @ v` (matrix state) | `diag(w_t) * S + k^T @ v` (w_t data-dependent via LoRA) | `(diag(w_t) + a^T @ b) * S + v^T @ k` (diag + rank-1) |
| **Output** | `sigmoid(r) * N/D` | `r @ (diag(u) * k^T@v + S_{t-1})` | Same as v5 | `r @ S_t` (uses S_t not S_{t-1}) |
| **Gate** | None | SiLU gate | SiLU gate | LoRA-based gate |
| **GroupNorm** | None | After WKV output | After WKV output | After WKV output |
| **Bonus** | `time_first` (scalar) | `time_faaaa` (vector) | `time_faaaa` (vector) | `bonus` (vector) |

#### ChannelMix (FFN)

| Component | RWKV-4/5/6 | RWKV-7 |
|-----------|------------|--------|
| **Structure** | Receptance-gated: `sigmoid(r) * (W_v @ sqrelu(W_k @ x))` | Plain 2-layer MLP: `W_v @ sqrelu(W_k @ x)` |
| **Activation** | Squared ReLU | Squared ReLU |
| **Token shift** | Static lerp (v4/v5) or data-dependent (v6) | Static lerp |
| **Hidden ratio** | `(hidden * 7/2) / 32 * 32` (implicit) | `hidden * hidden_ratio` (explicit in config, typically 4.0) |

#### Config Fields

| Field | v4 | v5 | v6 | v7 |
|-------|----|----|----|----|
| `model_type` | `rwkv` | `rwkv5` | `rwkv6` | `rwkv7` |
| Head size | N/A | `head_size: 64` | `head_size: 64` + misleading `num_attention_heads` | `head_dim: 64` |
| Intermediate | Explicit | Null (computed) | Null (computed) | Explicit |
| LoRA dims | N/A | N/A | Hardcoded (32, 64) | Explicit in config (`a_low_rank_dim`, `decay_low_rank_dim`, etc.) |
| Norm epsilon | `layer_norm_epsilon` | `layer_norm_epsilon` | `layer_norm_epsilon` | `norm_eps` |
| Rescale | `rescale_every: 6` | `rescale_every: 6` | `rescale_every: 6` | Removed |

#### Weight Name Prefixes

| Component | v4 (HF) | v5 (HF) | v6 (HF) | v7 (fla) |
|-----------|---------|---------|---------|---------|
| Block prefix | `rwkv.blocks.{i}` | `blocks.{i}` | `rwkv.blocks.{i}` | `model.layers.{i}` |
| Attn module | `.attention` | `.attention` | `.attention` | `.attn` |
| FFN module | `.feed_forward` | `.feed_forward` | `.feed_forward` | `.ffn` |
| Projections | `key/value/receptance/output` | Same + `gate` | Same + `gate` | `r_proj/k_proj/v_proj/o_proj` |
| Embedding | `rwkv.embeddings` | `embeddings` | `rwkv.embeddings` | `model.embeddings` |
| Final norm | `rwkv.ln_out` | `ln_out` | `rwkv.ln_out` | `model.norm` |
| LM head | `head` | `head` | `head` | `model.lm_head` |

### 4.4 Generic RWKV Design

Despite the differences, a **partially generic implementation** is feasible:

```rust
pub struct RwkvConfig {
    pub version: RwkvVersion,          // V6 | V7 (v4/v5 archived; add if community demand)
    pub hidden_size: usize,
    pub num_layers: usize,
    pub head_dim: usize,               // 64 for all current models
    pub num_heads: usize,              // hidden_size / head_dim
    pub vocab_size: usize,
    pub norm_eps: f64,
    pub intermediate_size: usize,      // explicit for v7, computed for v6

    // Version-specific
    pub rescale_every: Option<usize>,  // v6: Some(6), v7: None
    pub head_size_divisor: Option<f64>, // v6: Some(8.0), v7: None
    pub lora_dims: Option<RwkvLoraDims>, // v6: hardcoded, v7: from config
    pub hidden_ratio: Option<f64>,     // v7: Some(4.0), v6: None
}

pub enum RwkvVersion { V6, V7 }
```

**Shared code (v6 + v7):**
- Block structure (LN → TimeMix → residual → LN → ChannelMix → residual)
- LayerNorm (with bias)
- GroupNorm after WKV output
- Embedding → blocks → final LN → LM head
- KV state encoding in cache
- Weight loading framework (version-aware name mapping)

**Version-specific code (must be separate):**
- WKV recurrence formula (the core ~50 lines per version)
- Token shift mechanism (v6: LoRA-based ddlerp; v7: static lerp)
- ChannelMix (v6: receptance-gated; v7: plain MLP)
- Config parsing (different field names per version)
- Weight name mapping

**Recommended approach:** A `GenericRwkv` struct with version-specific `WkvKernel` and `ChannelMixKernel` trait objects:

```rust
trait WkvKernel {
    fn step(&self, r: &Tensor, k: &Tensor, v: &Tensor, w: &Tensor,
            state: &mut Tensor, bonus: &Tensor) -> Result<Tensor>;
}

trait ChannelMixKernel {
    fn forward(&self, x: &Tensor, prev_x: &Tensor) -> Result<Tensor>;
}
```

Two implementations of `WkvKernel` (v6 diagonal decay, v7 diag+rank-1), two implementations of `ChannelMixKernel` (receptance-gated for v6, plain MLP for v7). The outer `GenericRwkv` struct handles the shared block structure and delegates to the kernels. V4/V5 can be added later as additional kernel implementations if community demand materializes.

### 4.5 Related Linear RNN / SSM Architectures

The `fla` (flash-linear-attention) library unifies RWKV-7, GLA, RetNet, HGRN2, DeltaNet, and Mamba-2 under a common framework in Python. All share the recurrence pattern:

```
S_t = A_t * S_{t-1} + B_t    (state transition + input injection)
y_t = C_t * S_t               (output extraction)
```

Where `A_t` (transition), `B_t` (input), and `C_t` (output) vary by architecture:

| Architecture | A_t (transition) | B_t (input) | C_t (output) |
|-------------|-----------------|-------------|-------------|
| **RetNet** | `gamma * I` (fixed scalar) | `k^T @ v` | `q` |
| **GLA** | `diag(sigmoid(g))` (full gate) | `k^T @ v` | `q` |
| **RWKV-5** | `diag(w)` (diagonal decay) | `k^T @ v` | `r` |
| **RWKV-6** | `diag(w_t)` (data-dependent diagonal) | `k^T @ v` | `r` |
| **RWKV-7** | `diag(w_t) + a^T @ b` (diag + rank-1) | `v^T @ k` | `r` |
| **DeltaNet** | `I - eta * k^T @ k` (delta rule) | `eta * v^T @ k` | `q` |
| **Mamba-2** | `alpha * I` (scalar, discretized SSM) | `B * x` | `C` |

**Implication for the crate:** A truly generic linear RNN backend is possible using the same `WkvKernel` trait pattern. Each architecture provides its own `step()` implementation. The outer block structure (norm → recurrence → residual → norm → FFN → residual) is shared.

**Prioritization:** RWKV-6 first (validated by plip-rs), then RWKV-7 (available on HF). Mamba/GLA/RetNet as future extensions.

---

## 5. MI Capabilities

### 5.1 Core (from plip-rs, reusable)

| Capability | Status | Notes |
|-----------|--------|-------|
| **Attention extraction** | ✅ Working (transformers) | Per-layer, per-head attention patterns via `HookPoint::AttnPattern` |
| **Attention knockout** | ✅ Spec types ready | Pre-softmax `-inf` masking via `Intervention::Knockout`; measures KL divergence |
| **Attention steering** | ✅ Spec types ready | Post-softmax scaling/setting + renormalization via `Intervention::Steer` |
| **Steering calibration** | ✅ Infrastructure ready | Baseline measurement, dose-response curves |
| **State knockout** (RWKV) | ✅ Working (Phase 2) | Zero specific head states via `HookSpec::set_state_knockout()`; skips kv write (`state = decay * state`). V6 + V7 |
| **State steering** (RWKV) | ✅ Working (Phase 2) | Scale kv write via `HookSpec::set_state_steering()` (`state = scale * kv + decay * state`). V6 + V7 |
| **Effective attention** (RWKV) | ✅ Working (Phase 2) | `[b,h,t,t]` attention-equivalent from WKV recurrence via `HookPoint::RwkvEffectiveAttn`. V6 (prefix-sum) + V7 (backward linear functional) |
| **Logit lens** | ✅ Infrastructure ready | Per-layer vocab projection via `project_to_vocab` |
| **Activation caching** | ✅ Working | Per-layer hidden state storage (`ActivationCache`, `FullActivationCache`) |
| **KV cache** | ✅ Infrastructure ready | Autoregressive generation with intervention (`KVCache`) |
| **Position mapping** | ✅ Working | Character offset ↔ token index conversion (`PositionConversion`) |

### 5.2 New (required for Melomētis, general-purpose)

| Capability | Priority | Notes |
|-----------|----------|-------|
| **CLT loading + encoding** | High | Validated in plip-rs (`src/clt.rs`, 1,640 lines); lazy HF loading, sparse encoding, 90/90 top-10 match vs Python |
| **CLT feature injection** | High | Validated in plip-rs (suppress+inject protocol); port and generalise |
| **Attribution graphs** | High | Record edge weights through CLT features; prune to circuits |
| **SAE loading + encoding** | Medium | Load pre-trained SAE weights; encode activations |
| **SAE feature injection** | Medium | Same as CLT but for SAEs |
| **Activation patching** | Medium | Swap activations between clean/corrupted runs at specific hook points |
| **Residual stream decomposition** | Medium | Decompose residual stream into per-layer, per-component contributions |

### 5.3 Future (not required now)

| Capability | Notes |
|-----------|-------|
| **Probing** | Linear probes on activations (move from plip-rs to optional feature) |
| **Causal scrubbing** | Systematic causal intervention framework |
| **Indirect object identification** | IOI-style circuit analysis |
| **Induction head detection** | Automated induction head finding |
| **Feature visualization** | Export attention/activation data for visualization tools; deloson ([live demo](https://PCfVW.github.io/deloson/)) already consumes plip-rs layer scan JSON — candle-mi should preserve this output format |

---

## 6. Crate Structure

```
candle-mi/
├── Cargo.toml
├── LICENSE-MIT
├── LICENSE-APACHE
├── README.md
├── ROADMAP.md
├── CHANGELOG.md
├── CONVENTIONS.md                — Code conventions (unsafe policy, annotations, etc.)
├── design/                       — Design proposals (one Markdown file per decision, see §8)
├── src/
│   ├── lib.rs                      — Public API, feature gates, re-exports
│   │
│   ├── backend.rs                  — MIBackend trait, MIModel wrapper, from_pretrained, sampling
│   ├── hooks.rs                    — HookPoint, HookSpec, HookCache, Intervention enum
│   ├── config.rs                   — TransformerConfig, enums, per-family config parsers
│   ├── error.rs                    — MIError enum (thiserror)
│   │
│   ├── transformer/                — Generic transformer (feature: "transformer") ✅
│   │   ├── mod.rs                  — GenericTransformer, MIBackend impl, load()
│   │   ├── attention.rs            — Multi-head attention (GQA/MHA/MQA, separate + fused QKV)
│   │   ├── mlp.rs                  — MLP variants (gated separate, gated fused, plain)
│   │   ├── norm.rs                 — RmsNorm, LayerNorm, GemmaRmsNorm
│   │   └── rope.rs                 — Rotary position embeddings (pre-computed cos/sin)
│   │
│   ├── rwkv/                       — Generic RWKV (feature: "rwkv") — Phase 2
│   │   ├── mod.rs                  — GenericRwkv struct, version dispatch
│   │   ├── wkv_v6.rs              — RWKV-6 WKV kernel (from plip-rs)
│   │   ├── wkv_v7.rs              — RWKV-7 WKV kernel
│   │   ├── channel_mix.rs         — Receptance-gated (v6) + plain MLP (v7)
│   │   └── token_shift.rs         — Static lerp + data-dependent lerp
│   │
│   ├── interp/                     — Interpretability tools ✅ (core)
│   │   ├── mod.rs
│   │   ├── intervention.rs         — Knockout, steering, ablation spec types
│   │   ├── steering.rs             — Calibration, dose-response
│   │   ├── logit_lens.rs           — Per-layer vocab projection
│   │   ├── clt.rs                  — Cross-layer transcoder — Phase 3a
│   │   ├── sae.rs                  — Sparse autoencoder — Phase 3b
│   │   └── attribution.rs          — Attribution graphs — Phase 3a
│   │
│   ├── cache/                      — Caching infrastructure ✅
│   │   ├── mod.rs
│   │   ├── activation.rs           — ActivationCache, FullActivationCache
│   │   ├── kv.rs                   — KVCache
│   │   └── attention.rs            — AttentionCache
│   │
│   ├── tokenizer/                  — Tokenizer abstraction ✅
│   │   ├── mod.rs                  — MITokenizer (HuggingFace tokenizers wrapper)
│   │   └── rwkv.rs                 — RWKV World tokenizer (feature: "rwkv-tokenizer")
│   │
│   └── util/                       — Shared utilities ✅
│       ├── mod.rs
│       ├── masks.rs                — Causal/generation masks with caching
│       └── positioning.rs          — Character ↔ token mapping
│
├── examples/                       — Quick start + capability examples — Phase 4
│   ├── quick_start_transformer.rs  — Load model, forward pass, print top tokens
│   ├── logit_lens.rs               — Run logit lens on any supported model
│   ├── attention_patterns.rs       — Extract and print attention patterns
│   ├── knockout.rs                 — Attention knockout experiment
│   ├── steering.rs                 — Attention steering experiment
│   └── clt_scan.rs                 — CLT feature scan
│
└── tests/
    ├── validate_models.rs          — Per-family validation against Python HF reference ✅
    └── bench_hook_overhead.rs      — Hook overhead benchmark ✅
```

### 6.1 Feature Gates

```toml
[features]
default = ["cuda", "transformer"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
transformer = []           # Generic transformer backend
mmap = []                  # Memory-mapped weight loading (unsafe, for 7B+ models)
rwkv = []                  # RWKV v6-v7 backends
rwkv-tokenizer = []        # RWKV World tokenizer
clt = []                   # Cross-layer transcoder support
sae = []                   # Sparse autoencoder support
probing = ["linfa", "linfa-logistic", "ndarray"]  # Linear probing
```

### 6.2 Documentation

| Deliverable | Format | Contents |
|-------------|--------|----------|
| **README.md** | Markdown | Quick start (load a model, run logit lens in 10 lines), feature overview, supported models table, hardware requirements, link to docs.rs |
| **Rustdoc** (`cargo doc`) | Inline | Crate-level overview in `lib.rs`; module-level docs for every public module; doc-tests on all public functions and key types |
| **BACKENDS.md** | Markdown | Step-by-step guide to adding a new model architecture: config parser, weight map, validation protocol |
| **HOOKS.md** | Markdown | Hook point reference table (mirroring §2.1), intervention API walkthrough, worked examples (capture attention, run knockout, steer residual stream) |
| **CHANGELOG.md** | Markdown | [Keep a Changelog](https://keepachangelog.com/) format from v0.0.1 onwards |
| **Examples** | Rust (`examples/`) | Quick-start per backend (`quick_start_transformer.rs`, `quick_start_rwkv.rs`, etc.) + one per major capability (`logit_lens.rs`, `knockout.rs`, `steering.rs`, `clt_scan.rs`) — each self-contained with inline comments |

**Rustdoc policy:** Every `pub` item must have a doc comment. Types include a one-line summary + "# Examples" section with a runnable doc-test. `#![warn(missing_docs)]` enforced at crate level.

---

## 7. Phased Development Plan

### 7.0 Git Workflow

**Code quality — zero tolerance for warnings from day one.** Every commit must satisfy all three checks:

1. `cargo build --release` — zero errors, zero warnings (set `#![deny(warnings)]` in `lib.rs`)
2. `cargo clippy -- -W clippy::pedantic` — zero warnings (suppress individual false positives with targeted `#[allow(...)]` + a comment explaining why, never blanket `#[allow(clippy::pedantic)]`)
3. `cargo fmt --check` — zero formatting diffs (run `cargo fmt` before every commit)

CI enforces the same three checks on every push. A red CI is treated as a blocking bug, not a "fix later" item. This policy applies from the very first commit (Phase 0 repo scaffold) and at every subsequent step — there is no grace period.

**Commit granularity:** One commit per logical unit — a single module port, a new config parser, a passing validation, a bug fix. Each commit must pass the three code-quality checks above. Avoid monolithic "implement phase N" commits.

**Push cadence:** Push at the end of each working session (backup) and always at the milestones marked with **`PUSH`** below. Every push to `main` must pass CI (build + clippy + fmt + tests).

**Branch strategy:** Work directly on `main` during solo development. Use short-lived feature branches only when a change spans multiple sessions and may leave `main` broken in between; merge back when green.

**Tag convention:** Tag at each phase completion: `v0.0.1-phase0`, `v0.0.2-phase1`, etc. Tag `v0.1.0` at publication (Phase 4). Tags matching `v*` trigger `publish.yml`, which runs full CI then `cargo publish` automatically. **Always wait for `ci.yml` green before tagging** — bump `Cargo.toml` version + commit `Cargo.lock` first.

### Phase 0: Foundation

**Goal:** Core trait, hook system, infrastructure — no model backends yet.

- [x] Create repository, Cargo.toml, CI, ROADMAP.md (this document), CHANGELOG.md (empty "Unreleased" section), LICENSE-MIT, LICENSE-APACHE, `design/` directory (design proposals from §8) — **commit: `init: scaffold repo with CI, roadmap, design docs, and dual license`** — **PUSH**
- [x] Port `backend.rs` (MIBackend trait) from plip-rs `model.rs` — redesign with hook-based API — **commit**
- [x] Implement `HookSpec` and `HookCache` types — **commit**
- [x] Port `intervention.rs` (knockout, steering, ablation spec types and trait method signatures — attention-specific implementations are model-agnostic and land here; RWKV-specific state intervention *implementations* land in Phase 2) — **commit**
- [x] Port `cache/` modules (activation, KV, attention) — **commit**
- [x] Port `util/` modules (masks, positioning) — **commit** — **PUSH** (mid-phase checkpoint)
- [x] Port `interp/logit_lens.rs` — **commit**
- [x] Port `interp/steering.rs` (calibration, dose-response) — **commit**
- [x] Port `tokenizer/` (MITokenizer enum + RWKV tokenizer) — **commit**
- [x] Write comprehensive tests for all ported modules — **commit**

**Deliverable:** Compiling crate with full MI infrastructure, no backends. — **PUSH + tag `v0.0.1-phase0`** ✅ Published to [crates.io](https://crates.io/crates/candle-mi) on 2026-02-23.

### Phase 1: Generic Transformer

**Goal:** One forward pass implementation that covers LLaMA, Qwen2, Gemma, Gemma 2, Phi-3, StarCoder2, Mistral.

- [x] Implement `TransformerConfig` with all 7+ axes (~12 config fields) — **commit**
- [x] Implement config parsers for `llama`, `qwen2`, `gemma`, `gemma2`, `phi3`, `starcoder2`, `mistral` — **commit**
- [x] Implement generic forward pass (one commit per component):
  - [x] Embedding (with optional scaling) — **commit**
  - [x] RoPE (via `candle_nn::rotary_emb::rope()`) — **commit**
  - [x] Multi-head attention (GQA/MHA/MQA via `num_kv_heads`) — **commit**
  - [x] QKV projection (separate and fused) — **commit**
  - [x] MLP (gated, fused gated, plain) — **commit**
  - [x] Normalization (RmsNorm, LayerNorm, GemmaRmsNorm) — **commit**
  - [x] LM head (tied, separate, conditional) — **commit**
- [x] Forward pass compiles end-to-end — **PUSH**
- [x] Integrate hook points at all TransformerLens-equivalent locations — **commit**
- [x] Implement `MIBackend` for `GenericTransformer` — **commit**
- [x] Validate incrementally, one model family at a time (each adds 1–2 config axes):
  1. [x] **LLaMA** 3.2 1B — "Paris" #1 (CPU/GPU, exact match with Python HF) — **commit** — **PUSH**
  2. [x] **Qwen2** 2.5-Coder-3B — "Paris" #1 (CPU/GPU) — **commit** — **PUSH**
  3. [x] **Gemma 2** 2B — "Paris" #8 (correct: logit softcapping flattens distribution) — **commit** — **PUSH**
  4. [x] **Phi-3** Mini 4K — "Paris" #1 (CPU/GPU) — **commit** — **PUSH**
  5. [x] **StarCoder2** 3B — "Hello" #1 (CPU/GPU) — **commit** — **PUSH**
  6. [x] **Mistral** 7B v0.1 — "Paris" #4 (CPU/GPU, exact match with Python HF) — **commit** — **PUSH**
- [x] Benchmark hook overhead on LLaMA 3.2 1B (16 layers × 12 hooks = 194 hook points):
  - GPU (CUDA BF16): +11.5% overhead with full capture (25.5ms → 28.5ms)
  - CPU (F32): within noise (zero overhead when inactive) — **commit**

**Deliverable:** `MIModel::from_pretrained("meta-llama/Llama-3.2-1B")` works with full MI support. — **PUSH + tag `v0.0.2-phase1`** ✅ Completed on 2026-02-25.

**Validation protocol:** For each model family, compare top-10 logits for 5 test prompts against Python HuggingFace outputs. Tolerance: abs < 1e-4 (F32). Validate in the incremental order above — if LLaMA works and Qwen2 breaks, the bug is in QKV bias or tied embeddings.

### Phase 2: RWKV-6 + RWKV-7

**Goal:** Port plip-rs RWKV-6 backend, add RWKV-7, validate both.

- [x] Implement `RwkvConfig` with version dispatch — **commit**
- [x] Implement shared RWKV block structure — **commit**
- [x] Implement version-specific token shift (static lerp + ddlerp) — **commit**
- [x] Implement version-specific channel mix (receptance-gated + plain) — **commit**
- [x] Implement RWKV weight name mapping per version — **commit**
- [x] Port RWKV-6 WKV kernel from plip-rs `forward_rwkv6.rs` — **commit**
- [x] Validate RWKV-6: compare against plip-rs reference outputs — **commit** — **PUSH** (RWKV-6 green)
- [x] Implement RWKV-7 WKV kernel (generalized delta rule) — **commit**
- [x] Validate RWKV-7: compare against fla Python reference — **commit** — **PUSH** (RWKV-7 green)
- [x] Port effective attention computation for RWKV-6 — **commit**
- [x] Derive effective attention for RWKV-7 (new: diag+rank-1 complicates the formula) — **commit**
- [x] Implement RWKV-specific state knockout + state steering (the `MIBackend` trait methods and spec types were ported in Phase 0; this provides the concrete implementations extracted from plip-rs `forward_rwkv6.rs`) — **commit**

**Deliverable:** `MIModel::from_pretrained("RWKV/RWKV7-Goose-World3-1.5B-HF")` works. ✅ — **PUSH + tag `v0.0.3`**

### Phase 3: CLT Support

**Goal:** Load and use pre-trained cross-layer transcoders. CLT infrastructure already validated in plip-rs (`src/clt.rs`, 1,640 lines) — port and generalise.

- [ ] Port CLT weight loading from plip-rs (circuit-tracer format, lazy HF download) — **commit**
- [ ] Port CLT encoding (activations → feature activations, sparse top-k) — **commit**
- [ ] Port CLT feature injection (suppress+inject protocol from melometis/tragos) — **commit**
- [ ] Validate: load Gemma 2 2B CLT, reproduce melometis position-sweep results — **commit** — **PUSH** (CLT pipeline green on Gemma 2)
- [ ] Validate: load Llama 3.2 1B CLT, reproduce tragos position-sweep results — **commit** — **PUSH** (CLT pipeline green on Llama 3.2)
- [ ] Implement attribution graph construction — **commit**

**Deliverable:** Full CLT pipeline working on both Gemma 2 2B and Llama 3.2 1B. — **PUSH + tag `v0.0.4-phase3`**

### Phase 3b: SAE Support

**Goal:** Load and use pre-trained sparse autoencoders.

- [ ] Implement SAE weight loading (SAELens / Gemma Scope format) — **commit**
- [ ] Implement SAE encoding and feature injection — **commit**
- [ ] Validate: load Gemma Scope SAE, encode activations, verify reconstruction — **commit** — **PUSH**

**Deliverable:** SAE pipeline working alongside CLTs. — **PUSH + tag `v0.0.5-phase3b`**

### Phase 4: Polish + Publish

**Goal:** Documentation, examples, crates.io publication.

- [ ] Write crate-level documentation with examples — **commit**
- [ ] Write `BACKENDS.md` — how to add a new model architecture — **commit**
- [ ] Write `HOOKS.md` — hook point reference and intervention walkthrough — **commit**
- [ ] Write example programs (logit lens, knockout, steering, CLT scan) — **commit per example**
- [ ] Audit public API surface (`pub` vs `pub(crate)`) — **commit**
- [ ] Populate CHANGELOG.md with all notable changes from Phases 0–3b — **commit** — **PUSH** (release candidate)
- [ ] **Release workflow** (publish v0.1.0 to crates.io — automated via `publish.yml`):
  1. Ensure `main` is clean: `git status` shows no uncommitted changes
  2. Bump version in `Cargo.toml` to `0.1.0` + update `Cargo.lock` — **commit: `release: v0.1.0`** — **PUSH**
  3. Wait for `ci.yml` to pass on the version-bump commit (build + clippy + fmt + tests)
  4. `git tag v0.1.0` — tag the exact commit that CI validated
  5. `git push origin v0.1.0` — push the tag (triggers `publish.yml`)
  6. `publish.yml` runs full CI checks then `cargo publish` automatically
  7. Verify the published crate: `cargo install candle-mi --version 0.1.0` in a fresh directory
- [ ] Submit PR to candle repo adding candle-mi to "Useful External Resources" (per Eric Buehler's invitation)
- [ ] Announce (Rust ML community, MI community)

### Phase 5+: Extensions (Future)

- [ ] RWKV-4/5 backends (if community demand)
- [ ] Mamba / Mamba-2 backend
- [ ] GLA, RetNet backends (via generic linear RNN trait)
- [ ] Activation patching
- [ ] Probing module (optional feature)
- [ ] Feature visualization export (JSON for web UIs)
- [ ] MoE transformer variant
- [ ] Quantized model support (GGUF, GPTQ, AWQ)

---

## 8. Key Design Decisions

Detailed design proposals live in the [`design/`](design/) directory. Summary:

1. **Crate name.** ~~Must be decided first.~~ **Decided: `candle-mi`** — endorsed by HuggingFace (see §0).
2. **Hook system** — enum primary with `Display`/`FromStr` string conversion. See [`design/hook-system.md`](design/hook-system.md).
3. **Hook overhead** — zero-cost when inactive (conditional clone at each hook point). See [`design/hook-overhead.md`](design/hook-overhead.md).
4. **Intervention API** — unified `forward(tokens, config)` (pyvene-style). See [`design/intervention-api.md`](design/intervention-api.md).
5. **Error handling** — typed `MIError` enum with `thiserror`. See [`design/error-handling.md`](design/error-handling.md).
6. **Candle version** — pin to `=0.9`, update incrementally. See [`design/candle-version.md`](design/candle-version.md).
7. **RWKV-7 effective attention** — diag+rank-1 state transition; needs mathematical work. See [`design/rwkv7-effective-attention.md`](design/rwkv7-effective-attention.md).

---

## 9. Relationship to Existing Projects

### 9.1 plip-rs (AIware 2026)

**Frozen at v1.4.0** on the `melometis` branch (the `tragos` branch has been merged into `melometis`). The crate starts fresh but reuses ~3500 lines of infrastructure. plip-rs remains the supplementary material for the AIware paper.

### 9.2 Melomētis + Tragos — Planning in Poems (plip-rs `melometis` branch, v1.4.0)

Two independent replications of Anthropic's "Planning in Poems" [Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location), now unified on the `melometis` branch:

**Gemma 2 2B** (426K and 2.5M CLTs): suppress all `-out` rhyme-group features, inject a single "around" feature at L22, sweep injection position across 31 tokens — P("around") is flat at ~4.5e-8 then **jumps to 0.483 at the planning site** (155-million-fold ratio). 70% of 136 suppress+inject pairs peak at the planning site. The 2.5M CLT upgrade achieves word-level resolution (209 words, 52.2% best redirect, 3.78-trillion-fold spike ratio).

**Llama 3.2 1B** (524K CLT): second independent replication confirming the phenomenon generalises across architectures. Cross-model comparison reveals the **search vs. commitment distinction**: both models have phonological search (Jaccard 0.24), but Llama lacks the commitment circuit that sustains the rhyme signal through the final layers (82% of planning features crammed into L15, the last layer). Layer suppression on Gemma confirms L22–25 are causally necessary for rhyming (0/10 without them).

Documentation: `docs/planning-in-poems/` (Gemma 2 2B) and `docs/planning-circuit-hunt/` (cross-model investigation). Full pipeline reproducible in ~3 hours: `cargo run --release --example reproduce_pipeline`.

Future: becomes a **consumer of candle-mi** — a separate repository that depends on `candle-mi` for model loading, CLT injection, and intervention infrastructure. Melomētis-specific code (GTPyhop integration, poetry corpus, HTN planning) lives in its own repo.

### 9.3 Deloson (Visualization)

Interactive web visualizer for plip-rs layer scan results. Built with React 19, React Flow, and Recharts; deployed on [GitHub Pages](https://PCfVW.github.io/deloson/). Renders per-layer attention statistics (Cohen's d, Welch's t, p-values, Python/Rust ratios) as node-based flow diagrams and cross-model comparison charts across 6 code LLMs (StarCoder2, Qwen2.5-Coder 3B/7B, CodeGemma, Code-LLaMA, Phi-3). Supports drag-and-drop loading of new `layer_scan_universal_*.json` files at runtime.

Deloson demonstrates the visualization half of the MI pipeline: candle-mi produces the numerical results, deloson makes them explorable. The JSON output format used by plip-rs's `layer_scan_universal` example should be preserved (or extended) in candle-mi to maintain compatibility.

### 9.4 candle-transformers

The crate does NOT depend on candle-transformers. It provides its own model implementations with MI hooks built in. This is intentional: candle-transformers' models don't expose internals.

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Generic transformer has subtle bugs vs HF reference | High | High | Rigorous per-model validation protocol (§7 Phase 1) |
| RWKV-7 HF format changes (fla library is young) | Medium | Medium | Pin to specific HF model revisions; version-aware config |
| Hook system adds unacceptable overhead | Low | High | Benchmarked in Phase 1 (after LLaMA validates); zero-cost when inactive |
| No users | Medium | Low | HuggingFace endorsement + two published replications lower the barrier; the crate serves Melomētis regardless |
| candle breaking changes | Medium | Medium | Pin candle version; update incrementally |
| CLT/SAE weight format changes | Medium | Medium | Support multiple formats; version-aware loading |

---

## References

1. Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Anthropic, 2021.
2. Lindsey, J. et al. "On the Biology of a Large Language Model." Anthropic, March 2025.
3. Nanda, N. & Bloom, J. "TransformerLens." GitHub, 2022-2026.
4. Fiotto-Kaufman, J. et al. "nnsight: Democratizing Access to Neural Network Internals." 2024.
5. Wu, Z. et al. "pyvene: A Library for Understanding and Improving PyTorch Models via Interventions." NeurIPS 2024.
6. Peng, B. et al. "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence." ICLR 2025.
7. Peng, B. et al. "RWKV-7 Goose with Expressive Dynamic State Evolution." arXiv 2503.14456, March 2025.
8. Dao, T. & Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024.
9. Yang, S. et al. "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024.
10. Anthropic. "circuit-tracer." GitHub, May 2025.
11. Lieberum, T. et al. "Gemma Scope." Google DeepMind, 2024.
12. Bloom, J. et al. "SAELens." GitHub, 2024-2026.
