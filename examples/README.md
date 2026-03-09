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
- **GPU recommended** for models larger than 1B parameters. candle-mi is
  developed on an RTX 5060 Ti (16 GB VRAM) with 64 GB RAM and CUDA 13.1.
