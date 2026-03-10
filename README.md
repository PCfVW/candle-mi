# candle-mi

[![CI](https://github.com/PCfVW/candle-mi/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/candle-mi/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/candle-mi)](https://crates.io/crates/candle-mi)
[![docs.rs](https://img.shields.io/docsrs/candle-mi)](https://docs.rs/candle-mi)
[![Rust 1.87+](https://img.shields.io/badge/rust-1.87%2B-orange)](https://www.rust-lang.org)
[![Edition 2024](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/PCfVW/candle-mi)](https://github.com/PCfVW/candle-mi/commits/main)

*Mechanistic Interpretability for the Rust of us.*

> **Note:** v0.1.0 — the API may change between minor versions. See the [CHANGELOG](CHANGELOG.md).

## Table of Contents

- [What is this?](#what-is-this)
- [What can you do with it?](#what-can-you-do-with-it)
- [Quick start](#quick-start)
- [Supported models](#supported-models)
- [Feature flags](#feature-flags)
- [Documentation](#documentation)
- [License](#license)
- [Development](#development)

## What is this?

**Mechanistic interpretability** (MI) is the study of *how* a language model arrives at its predictions — not just what it outputs, but what happens inside. By inspecting and manipulating the model's internal activations (attention patterns, residual streams, MLP outputs), researchers can understand which components drive specific behaviors.

**candle-mi** is a Rust library that makes this possible. It re-implements model forward passes with built-in **hook points** — named locations where you can:

- **Capture** activations (e.g., "what does the attention pattern look like at layer 5?")
- **Intervene** on activations mid-forward-pass (e.g., "what happens if I knock out this attention edge?" or "what if I steer the residual stream toward a concept?")

This is the Rust equivalent of Python's [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), built on [candle](https://github.com/huggingface/candle) for GPU acceleration. The hook system is type-safe (typos are caught at compile time, not silently ignored at runtime) and zero-overhead (an empty hook spec adds no allocations or clones to the forward pass).

**Why Rust?** Running published MI experiments — such as Anthropic's [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/) or [*Planning in poems*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems) — quickly hits the limits of CPU-only Python. Cloud GPUs are always an option, but not a frugal one. With a consumer-grade GPU, memory and runtime become the real bottleneck. [candle](https://github.com/huggingface/candle) solves both: Rust's zero-cost abstractions minimize memory overhead, compiled code runs faster, and candle provides direct CUDA/Metal access without Python's runtime tax. That's how candle-mi started: let's bring MI to local hardware. (See the [Figure 13 replication](examples/README.md#example-output-figure13_planning_poems) for a concrete example — *Planning in poems* reproduced on a consumer GPU.)

## What can you do with it?

| Technique | What it does | Example |
|-----------|-------------|---------|
| **Logit lens** | See what the model "thinks" at each layer by projecting intermediate residual streams to vocabulary space | [`logit_lens`](examples/README.md#example-output-logit_lens) |
| **Attention knockout** | Block specific attention edges (e.g., "token 5 cannot attend to token 0") and measure how predictions change | [`attention_knockout`](examples/README.md#example-output-attention_knockout) |
| **Activation steering** | Add a direction vector to the residual stream to shift model behavior (e.g., make it more positive or more formal) | [`steering_dose_response`](examples/README.md#example-output-steering_dose_response) |
| **Activation patching** | Swap activations between a clean and corrupted run to identify which components causally drive a prediction | [`activation_patching`](examples/README.md#example-output-activation_patching) |
| **Attention patterns** | Visualize where each attention head attends across the sequence | [`attention_patterns`](examples/README.md#example-output-attention_patterns) |
| **RWKV state analysis** | Inspect and intervene on recurrent state — not just transformers | [`rwkv_inference`](examples/README.md#example-output-rwkv_inference) |

## Quick start

```rust
use candle_mi::{HookSpec, MIModel};

fn main() -> candle_mi::Result<()> {
    // 1. Load a model (auto-detects architecture from HuggingFace config)
    let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
    let tokenizer = model.tokenizer().unwrap();

    // 2. Tokenize a prompt
    let tokens = tokenizer.encode("The capital of France is")?;
    let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;

    // 3. Run a forward pass (HookSpec::new() = no hooks, zero overhead)
    let cache = model.forward(&input, &HookSpec::new())?;
    let logits = cache.output();  // [1, seq_len, vocab_size]

    // 4. Decode the top prediction
    let last_logits = logits.get(0)?.get(tokens.len() - 1)?;
    let token_id = candle_mi::sample_token(&last_logits, 0.0)?;  // greedy
    println!("{}", tokenizer.decode(&[token_id])?);  // " Paris"
    Ok(())
}
```

Here is what an end-to-end run looks like (auto-config loading LLaMA 3.2 1B — config detection, forward pass, and top-5 predictions):

<p align="center">
  <img src="examples/screenshots/auto_config_llama.png" alt="Auto-config loading LLaMA 3.2 1B" width="700">
</p>

## Supported models

| Backend | Model families | Validated models | Feature flag |
|---------|---------------|-----------------|-------------|
| `GenericTransformer` | LLaMA 1/2/3, Qwen 2/2.5, Gemma, Gemma 2, Phi-3/4, StarCoder2, Mistral | LLaMA 3.2 1B, Qwen2.5-Coder-3B, Gemma 2 2B, Phi-3 Mini, StarCoder2 3B, Mistral 7B v0.1 | `transformer` (default) |
| `GenericRwkv` | RWKV-6 (Finch), RWKV-7 (Goose) | RWKV-7 1.6B | `rwkv` |

**Model families** are what the config parser accepts (any model reporting that `model_type` in its HuggingFace `config.json`). **Validated models** are those verified to match Python/PyTorch reference output. Most other HuggingFace transformer models work out of the box via **auto-config** — no code changes needed. See [BACKENDS.md](BACKENDS.md) for details.

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `transformer` | yes | Generic transformer backend (decoder-only) |
| `cuda` | yes | CUDA GPU acceleration |
| `rwkv` | no | RWKV-6/7 linear RNN backend |
| `rwkv-tokenizer` | no | RWKV world tokenizer (required for RWKV inference) |
| `clt` | no | Cross-Layer Transcoder support |
| `sae` | no | Sparse Autoencoder support |
| `mmap` | no | Memory-mapped weight loading (required for sharded models) |
| `memory` | no | RAM/VRAM memory reporting |
| `probing` | no | Linear probing via linfa (experimental) |
| `metal` | no | Apple Metal GPU acceleration |

## Documentation

| Document | Description |
|----------|-------------|
| [API docs (docs.rs)](https://docs.rs/candle-mi) | Crate-level documentation with quick start and examples |
| [HOOKS.md](HOOKS.md) | Hook point reference, intervention API walkthrough, and worked examples |
| [BACKENDS.md](BACKENDS.md) | How to add a new model architecture (auto-config, config parser, custom backend) |
| [examples/README.md](examples/README.md) | 15 runnable examples covering inference, logit lens, knockout, steering, and more |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [ROADMAP.md](ROADMAP.md) | Development roadmap and architecture decisions |

## License

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)

## Development

Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev) and [Augment Code](https://www.augmentcode.com/) (review). Git workflow managed with [Fork](https://fork.dev/).
