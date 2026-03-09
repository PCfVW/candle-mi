# Examples

![CUDA](https://img.shields.io/badge/CUDA-13.1-76B900?logo=nvidia)
![GPU](https://img.shields.io/badge/GPU-RTX%205060%20Ti%2016GB-76B900?logo=nvidia)
![RAM](https://img.shields.io/badge/RAM-64%20GB-blue)

Runnable examples demonstrating candle-mi features.

## Available Examples

| Example | Features | Description |
|---------|----------|-------------|
| `quick_start_transformer` | `transformer` | Discover cached transformers, run inference, print top-5 predictions |
| `fast_download` | *(default)* | Download a model from `HuggingFace` Hub with parallel chunked transfers |
| `quick_start_sae` | `sae`, `transformer` | Load an SAE, encode model activations, print top features and reconstruction error |
| `auto_config_dogfood` | `transformer` | Download a model and test auto-config loading with compatibility check |
| `generate` | `transformer` | Greedy autoregressive text generation on all cached models |
| `logit_lens` | `transformer` | Layer-by-layer prediction tracking via residual stream projection |
| `attention_knockout` | `transformer` | Knock out a specific attention edge (last→first token), measure KL divergence and top changed tokens |
| `steering_dose_response` | `transformer` | Sweep steering dose levels, build a dose-response curve, and interpolate target attention |
| `attention_patterns` | `transformer` | Capture and analyze per-head attention patterns at every layer |
| `activation_patching` | `transformer` | Causal tracing via position-specific activation patching (Meng et al., 2022) |
| `token_positions` | *(default)* | Character-to-token mapping with `EncodingWithOffsets` and `convert_positions` |
| `rwkv_inference` | `rwkv` | RWKV-7 linear RNN inference with state hook capture and state knockout |
| `recurrent_feedback` | `transformer` | Anacrousis / recurrent passes for rhyme completion (Taufeeque et al., 2024) |
| `figure13_planning_poems` | `clt`, `transformer` | Replication of [Anthropic's Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location) (suppress + inject position sweep) |

## Running

```bash
# Transformer inference on all cached models
cargo run --release --example quick_start_transformer

# Download a model (defaults to a tiny test repo)
cargo run --example fast_download -- meta-llama/Llama-3.2-1B

# SAE encoding on Gemma 2 2B
cargo run --release --features sae,transformer --example quick_start_sae

# Auto-config dogfooding — success (known model family, manual parser)
cargo run --release --features transformer --example auto_config_dogfood -- "meta-llama/Llama-3.2-1B"

# Auto-config dogfooding — failure (unsupported architecture)
cargo run --release --features transformer --example auto_config_dogfood -- "allenai/OLMo-1B-hf"

# Greedy text generation — single model (recommended for 7B+ to avoid OOM)
cargo run --release --features transformer --example generate -- "meta-llama/Llama-3.2-1B"

# Greedy text generation — all cached models (add mmap for sharded weights)
cargo run --release --features transformer,mmap --example generate

# Logit lens — single model
cargo run --release --features transformer --example logit_lens -- "meta-llama/Llama-3.2-1B"

# Logit lens — all cached models
cargo run --release --features transformer,mmap --example logit_lens

# Attention knockout — single model
cargo run --release --features transformer --example attention_knockout -- "meta-llama/Llama-3.2-1B"

# Attention knockout — all cached models
cargo run --release --features transformer,mmap --example attention_knockout

# Steering dose-response — single model
cargo run --release --features transformer --example steering_dose_response -- "meta-llama/Llama-3.2-1B"

# Steering dose-response — all cached models
cargo run --release --features transformer,mmap --example steering_dose_response

# Attention patterns — single model
cargo run --release --features transformer --example attention_patterns -- "meta-llama/Llama-3.2-1B"

# Attention patterns — all cached models
cargo run --release --features transformer,mmap --example attention_patterns

# Activation patching (causal tracing) — single model
cargo run --release --features transformer --example activation_patching -- "meta-llama/Llama-3.2-1B"

# Activation patching — all cached models
cargo run --release --features transformer,mmap --example activation_patching

# Token positions — single model (tokenizer only, no GPU)
cargo run --example token_positions -- "meta-llama/Llama-3.2-1B"

# Token positions — all cached models
cargo run --example token_positions

# RWKV inference — auto-discover cached RWKV models
cargo run --release --features rwkv --example rwkv_inference

# RWKV inference — specific model
cargo run --release --features rwkv --example rwkv_inference -- "RWKV/RWKV7-Goose-World3-1.5B-HF"

# RWKV inference — RWKV-6 model (requires rwkv-tokenizer feature)
cargo run --release --features rwkv,rwkv-tokenizer --example rwkv_inference -- "RWKV/v6-Finch-1B6-HF"

# Recurrent feedback — default (Llama 3.2 1B, unembed layers 8-15, strength 2.0)
cargo run --release --features transformer --example recurrent_feedback

# Recurrent feedback — sustained mode with lower strength
cargo run --release --features transformer --example recurrent_feedback -- --sustained --strength 1.0

# Recurrent feedback — custom layer range and couplet limit
cargo run --release --features transformer --example recurrent_feedback -- --loop-start 14 --loop-end 15 --max-couplets 5

# Figure 13 replication — Llama 3.2 1B (default)
cargo run --release --features clt,transformer --example figure13_planning_poems

# Figure 13 replication — Gemma 2 2B, 426K CLT (requires mmap for sharded weights)
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-426k

# Figure 13 replication — Gemma 2 2B, 2.5M CLT (word-level features)
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-2.5m
```

### Example output: `logit_lens`

Prompt: *"The capital of France is"* — tracking when "Paris" first enters the
top predictions across layers.

**Llama 3.2 1B** (16 layers): "Paris" first appears at layer 11 (top-5).
Early layers predict generic tokens; convergence happens in the final third.

**Gemma 2 2B** (26 layers): "Paris" never reaches top-10. The model predicts
" a" at the final layer — consistent with Gemma 2's soft-capped logit
distribution which flattens probabilities across the vocabulary.

**StarCoder2 3B** (30 layers): "Paris" never reaches top-10. As a code model,
predictions are dominated by code tokens (`\_\-`, `selecione`, `Par`). Natural
language knowledge is minimal.

### Example output: `attention_knockout`

Prompt: *"The capital of France is"* — knock out the attention edge from the
last token to position 0 (first token) across all heads at the middle layer.

| Model | Layer | Heads | KL div | "Paris" baseline | "Paris" ablated | Logit diff |
|-------|-------|-------|--------|-----------------|----------------|------------|
| Llama 3.2 1B | 8 | 32 | 0.056 | 39.3% | 26.0% | −1.06 |
| Gemma 2 2B | 13 | 8 | 0.017 | 3.9% | 6.7% | −0.33 |
| StarCoder2 3B | 15 | 24 | 0.029 | — | — | −1.08 |

**Llama 3.2 1B** shows the strongest effect: "Paris" drops from 39.3% to
26.0% when the last token can't attend to the first token at layer 8.
The model relies on early-position attention at mid-depth for factual recall.

**Gemma 2 2B** shows a weaker and inverted effect: "Paris" *increases* from
3.9% to 6.7%. With only 8 KV heads (GQA) and soft-capped logits, the
knockout redistributes probability mass differently.

**StarCoder2 3B** predicts "Par" (a code subword) rather than "Paris", so the
logit diff applies to " Paris" (token 316) which has negligible baseline
probability.

### Example output: `steering_dose_response`

Prompt: *"The capital of France is"* — steer the attention edge from the last
token to position 0 at the middle layer, sweeping 6 dose levels.

| Model | Layer | Baseline attn | Dose 0.5 KL | Dose 4.0 KL | Dose 6.0 KL |
|-------|-------|--------------|-------------|-------------|-------------|
| Llama 3.2 1B | 8 | 0.630 | 0.006 | 0.029 | 0.043 |
| Gemma 2 2B | 13 | 0.589 | 0.001 | 0.002 | 0.003 |
| StarCoder2 3B | 15 | 0.673 | 0.002 | 0.004 | 0.005 |

**Llama 3.2 1B** shows the strongest dose-response: KL divergence grows from
0.006 at half-dose to 0.043 at 6× dose, with "Paris" logit diff reaching
−0.31. The model's factual recall is sensitive to attention steering at
mid-depth.

**Gemma 2 2B** shows much weaker sensitivity: KL stays below 0.004 even at 6×
dose. With GQA (8 KV heads) and soft-capped logits, the prediction
distribution is robust to single-edge steering.

### Example output: `attention_patterns`

Prompt: *"The capital of France is"* — capture attention at every layer and
analyze what the last token attends to.

| Model | Peak layer (last→first) | Peak attention | Top-1 at most layers |
|-------|------------------------|----------------|---------------------|
| Llama 3.2 1B | 2 | 0.847 | `<\|begin_of_text\|>` (BOS) |
| Gemma 2 2B | 22 | 0.845 | `<bos>` (BOS) |
| StarCoder2 3B | 26 | 0.866 | `The` (first real token, no BOS) |

All three models show strong attention to the first token across most layers
(the "BOS sink" pattern). **StarCoder2 3B** lacks a BOS token so the first
real token ("The") serves as the attention sink. **Llama 3.2 1B** peaks early
(layer 2), while **Gemma 2 2B** peaks late (layer 22).

### Example output: `activation_patching`

Clean prompt: *"The capital of France is"* vs. corrupted: *"The capital of Poland
is"*. For each layer, the clean residual at the subject position ("France") is
patched into the corrupted forward pass. Recovery measures how much the clean
"Paris" prediction is restored.

| Model | Subject pos | Corrupted KL | Best layer | Best recovery | Sharp cliff |
|-------|------------|-------------|------------|--------------|-------------|
| Llama 3.2 1B | 4 | 3.78 | 1 (100%) | Layers 0-8: >99% | Layer 9-15: 92%→0% |
| Gemma 2 2B | 4 | 0.50 | 1 (100%) | Layers 0-17: >89% | Layer 18-25: 74%→0% |
| StarCoder2 3B | 3 | 4.16 | 9 (99.9%) | Layers 0-20: >94% | Layer 21: 5% cliff |

**Llama 3.2 1B** shows a gradual decline: recovery drops from 100% at early
layers to 72% at layer 11, reaching 0% by the final layer. The factual
association "France → Paris" forms in the middle layers (8-13).

**Gemma 2 2B** maintains high recovery through layer 17 (89%), then drops
sharply. The factual lookup happens later in the network, consistent with
its deeper architecture.

**StarCoder2 3B** shows an abrupt cliff at layer 21: recovery drops from
94% to 5% in a single layer. As a code model, it stores factual knowledge
in a concentrated layer band.

### Example output: `token_positions`

Text: *"The Eiffel Tower is located in Paris, France."* — mapping character
annotations to token positions across different tokenizers.

| Entity | Char range | Llama 3.2 1B tokens | Gemma 2 2B tokens | StarCoder2 3B tokens |
|--------|-----------|--------------------|--------------------|---------------------|
| "Eiffel Tower" | 4-16 | 4 tokens (E+iff+el+Tower) | 2 tokens (Eiffel+Tower) | 5 tokens (E+iff+el+T+ower) |
| "Paris" | 31-36 | 1 token | 1 token | 2 tokens (Par+is) |
| "France" | 38-44 | 1 token | 1 token | 1 token |

The example shows how the same character span maps to different numbers of
tokens across models. `char_range_to_tokens()` handles this automatically,
and `convert_positions()` provides exact-vs-fuzzy matching for positions
between or beyond token boundaries.

### Example output: `rwkv_inference`

Prompt: *"The capital of France is"* — RWKV-7 linear RNN inference with state
hooks and state knockout.

**RWKV-7 Goose 1.5B**: Top-1 prediction is "Paris" at high probability. The
example captures RWKV-specific hook points — `RwkvState` (recurrent state
matrix, shape `[1, heads, head_dim, head_dim]`), `RwkvDecay` (data-dependent
decay), and `ResidPost` (residual stream) — demonstrating the structural
differences between recurrent and attention-based architectures.

State knockout at position 0 (making the first token invisible to future tokens)
shows the impact on factual recall via KL divergence and top changed tokens.

### Example output: `recurrent_feedback`

15 canonical couplets from Taufeeque et al. (2024) — baseline generation vs.
recurrent feedback with averaged rhyme direction injection.

| Mode | Settings | Rhymes | Rescued |
|------|----------|--------|---------|
| Baseline | — | 9/15 | — |
| Recurrent (prefill) | unembed L8–15, s=2.0 | 11/15 | +2 |
| Recurrent (sustained) | unembed L14–15, s=1.0 | 9/15 | +0 |

**Prefill mode** (default) injects the rhyme direction during the original
prompt positions only. **Sustained mode** (`--sustained`) also injects at the
current last token during each generation step. The prefill mode with layers
8–15 and strength 2.0 shows the best improvement (+2 rescued couplets).

Note: candle-mi recomputes the full sequence at each generation step (no KV
cache), so every step gets the double-pass benefit. This differs from plip-rs
where only the prefill step benefits from the recurrent pass.

**References:**
- Taufeeque et al., "Planning in Poems", arXiv:2407.15421, 2024
- Lindsey et al., "On the Biology of a Large Language Model", 2025
- Eric Jacopin, "Replicating 'Planning in Poems' with Open Tools" (plip-rs)

### Example output: `auto_config_dogfood`

**Success** on Llama 3.2 1B (known family, uses manual parser):

![auto_config_dogfood success on Llama-3.2-1B](screenshots/auto_config_llama.png)

**Failure** on OLMo-1B (unsupported architecture):

![auto_config_dogfood failure on OLMo-1B-hf](screenshots/auto_config_olmo.png)

OLMo-1B fails the compatibility check because its weight names
(`model.layers.*.input_layernorm.weight`, `model.final_norm.weight`) do not
match the normalisation tensor patterns that `GenericTransformer` expects.
candle-mi currently supports 6 model families: LLaMA, Qwen2, Gemma/Gemma 2,
Phi-3, Mistral, and StarCoder2.

### Example output: `figure13_planning_poems`

Replicates [Anthropic's Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location)
from "On the Biology of a Large Language Model": suppress natural rhyme
features and inject an alternative, sweeping injection position across all
tokens.  Three presets are available: `llama3.2-1b-524k` (Llama 3.2 1B),
`gemma2-2b-426k` (Gemma 2 2B, 426K CLT), and `gemma2-2b-2.5m` (Gemma 2 2B,
2.5M CLT with word-level feature granularity).

Output JSON and Mathematica plotting script are in
[`examples/figure13/`](figure13/).

## Prerequisites

- **quick_start_transformer** and **quick_start_sae** require models cached
  in `~/.cache/huggingface/hub/`. Download them first with `fast_download`
  or via Python (`huggingface_hub.snapshot_download()`).
- **quick_start_sae** downloads the Gemma Scope SAE (`google/gemma-scope-2b-pt-res`)
  automatically via `hf-fetch-model`.
- **figure13_planning_poems** requires a CLT from `HuggingFace` (downloaded
  automatically on first run). Gemma 2 2B preset requires `--features mmap`.
- **rwkv_inference** requires an RWKV model cached locally. RWKV-7 models
  include `tokenizer.json`; RWKV-6 models require `--features rwkv-tokenizer`.
- **recurrent_feedback** requires `meta-llama/Llama-3.2-1B` (default) cached
  locally.
- **GPU recommended** for models larger than 1B parameters. candle-mi is
  developed on an RTX 5060 Ti (16 GB VRAM) with 64 GB RAM and CUDA 13.1.
