# candle-mi

[![CI](https://github.com/PCfVW/candle-mi/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/candle-mi/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/candle-mi)](https://crates.io/crates/candle-mi)
[![docs.rs](https://img.shields.io/docsrs/candle-mi)](https://docs.rs/candle-mi)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange)](https://www.rust-lang.org)
[![Edition 2024](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/PCfVW/candle-mi)](https://github.com/PCfVW/candle-mi/commits/main)

*Mechanistic Interpretability for the Rust of us.*

> **Note:** v0.1.5 — the API may change between minor versions. See the [CHANGELOG](CHANGELOG.md).

## Supported model families

| Architecture | Families | Validated models | Feature |
|---|---|---|---|
| Decoder-only transformer | LLaMA 1/2/3, Mistral, Qwen 2/2.5, Phi-3/4, Gemma, Gemma 2, StarCoder2 | LLaMA 3.2 1B, Qwen2.5-Coder-3B, Gemma 2 2B, Phi-3 Mini, StarCoder2 3B, Mistral 7B | `transformer` |
| Linear RNN | RWKV-6 (Finch), RWKV-7 (Goose) | RWKV-7 1.6B | `rwkv` |

Most HuggingFace transformer models work out of the box via **auto-config** — no code changes needed. See [BACKENDS.md](BACKENDS.md) for details and how to add new architectures.

**Hardware:** candle-mi runs on a single consumer GPU (developed on an RTX 5060 Ti, 16 GB VRAM). Models up to ~7B fit in 16 GB at F32 precision — no H100 cluster required. CPU-only works for small models and tokenizer-only workflows.

## Table of Contents

- [What is this?](#what-is-this)
- [What can you do with it?](#what-can-you-do-with-it)
- [See it in action](#see-it-in-action)
- [Quick start](#quick-start)
- [Design philosophy](#design-philosophy)
- [Paper replications](#paper-replications)
- [Feature flags](#feature-flags)
- [Documentation](#documentation)
- [License](#license)
- [Development](#development)

## What is this?

**Mechanistic interpretability** (MI) is the study of *how* a language model arrives at its predictions — not just what it outputs, but what happens inside. By inspecting and manipulating the model's internal activations (attention patterns, residual streams, MLP outputs), researchers can understand which components drive specific behaviors.

**candle-mi** is a Rust library that makes this possible. It re-implements model forward passes with built-in **hook points** — type-safe, named locations in the computation graph (e.g., `HookPoint::AttnPattern(5)` for the post-softmax attention at layer 5) where you can:

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

candle-mi is (to our knowledge) the only MI toolkit with hook points for recurrent architectures — `RwkvState`, `RwkvDecay`, and `RwkvEffectiveAttn` enable mechanistic analysis of RWKV-6/7 models, a frontier that most MI tooling ignores entirely.

## See it in action

### The logit lens — what does the model "think" at each layer?

```bash
cargo run --release --features transformer --example logit_lens -- "meta-llama/Llama-3.2-1B"
```

This loads in ~2 seconds and runs in ~112ms on an RTX 5060 Ti (16 GB VRAM) or ~3 seconds on CPU, revealing how factual recall emerges across layers. The same prompt (*"The capital of France is"*) tells three different stories:

- **Llama 3.2 1B**: "Paris" appears at layer 11 (69% depth) — typical early factual resolution.
- **Gemma 2 2B**: "Paris" appears at layer 25 (the very last layer, rank 8) — the model hedges until the end.
- **StarCoder2 3B**: "Paris" never appears as a single token — its BPE tokenizer splits it into " Par", which dominates from layer 22 at 33%→74%. The model *knows* the answer but its code-oriented vocabulary hides it.

See the [full comparison](examples/README.md#example-output-logit_lens) with per-layer tables.

### The flagship — Anthropic's circuit tracing on consumer hardware

This library was built to replicate Anthropic's [circuit-tracing work](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) on consumer hardware. Here is [Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location) from *"On the Biology of a Large Language Model"*, running on a single GPU:

```bash
cargo run --release --features clt,transformer --example figure13_planning_poems
```

This uses Llama 3.2 1B with a 524K-feature Cross-Layer Transcoder to suppress natural rhyme features and inject an alternative ("that" → P=0.98), sweeping injection position across all prompt tokens. See the [full output](examples/README.md#example-output-figure13_planning_poems) and the [17 examples](examples/README.md) covering logit lens, attention knockout, steering, activation patching, CLT circuits, SAE encoding, RWKV inference, and more.

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

### With hooks — capture attention patterns

```rust
use candle_mi::{HookPoint, HookSpec, MIModel};

fn main() -> candle_mi::Result<()> {
    let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
    let tokenizer = model.tokenizer().unwrap();

    let tokens = tokenizer.encode("The capital of France is")?;
    let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;

    // Capture the post-softmax attention pattern at layer 5
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::AttnPattern(5));

    let cache = model.forward(&input, &hooks)?;

    // Retrieve the captured tensor — [1, heads, seq, seq]
    let attn = cache.require(&HookPoint::AttnPattern(5))?;
    println!("Layer 5 attention shape: {:?}", attn.shape());
    Ok(())
}
```

Here is what an end-to-end run looks like (auto-config loading LLaMA 3.2 1B — config detection, forward pass, and top-5 predictions):

<p align="center">
  <img src="examples/screenshots/auto_config_llama.png" alt="Auto-config loading LLaMA 3.2 1B" width="700">
</p>

## Design philosophy

candle-mi makes a deliberate trade-off: **full-sequence recompute at every generation step** (no KV cache). This is slower than production inference engines, but it means:

- **Maximum observability.** Hooks can re-observe how earlier positions change under intervention at every step. Interventions "just work" without KV cache invalidation.
- **Interventions compound.** When steering the residual stream during autoregressive generation, each new token is generated with the intervention re-applied across the full context. This is why candle-mi's [recurrent feedback](examples/README.md#example-output-recurrent_feedback) rescues +2 rhyming couplets (out of 15) where a KV-cached approach gets +1 — the intervention is observed at every step, not just once during prefill.

This is a research-first design: MI analyses need to see everything, and the performance cost is acceptable when the alternative is missing causal effects. candle-mi is not an inference engine — for production serving, see [candle-vllm](https://github.com/EricLBuehler/candle-vllm), [vllm.rs](https://github.com/guoqingbao/vllm.rs), or [vLLM](https://github.com/vllm-project/vllm) (Python). It is optimized for observability, not throughput.

## Paper replications

| Paper | What we replicate | Example |
|-------|------------------|---------|
| Anthropic, [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (2025) | Figure 13 — suppress natural rhyme features and inject an alternative via CLT, sweeping injection position | [`figure13_planning_poems`](examples/README.md#example-output-figure13_planning_poems) |
| Anthropic, [*When Models Manipulate Manifolds*](https://transformer-circuits.pub/2025/linebreaks/index.html) (2025) | Character count helix — residual stream encodes line position as a helical manifold; reproduced on Gemma 2 2B with 30 Dickens chapters | [`character_count_helix`](examples/README.md#example-output-character_count_helix) |
| Meng et al., [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262) (2022) | Causal tracing via position-specific activation patching | [`activation_patching`](examples/README.md#example-output-activation_patching) |
| Taufeeque et al., [*Recurrent Feedback*](https://arxiv.org/abs/2407.15421) (2024) | Anacrousis — recurrent steering passes for rhyme completion | [`recurrent_feedback`](examples/README.md#example-output-recurrent_feedback) |


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
| `memory-debug` | no | Raw DXGI/NVML values and per-chunk VRAM on stderr (implies `memory`) |
| `probing` | no | Linear probing via linfa (experimental) |
| `metal` | no | Apple Metal GPU acceleration |

## Documentation

| Document | Description |
|----------|-------------|
| [API docs (docs.rs)](https://docs.rs/candle-mi) | Crate-level documentation with quick start and examples |
| [HOOKS.md](HOOKS.md) | Hook point reference, intervention API walkthrough, and worked examples |
| [BACKENDS.md](BACKENDS.md) | How to add a new model architecture (auto-config, config parser, custom backend) |
| [examples/README.md](examples/README.md) | 17 runnable examples covering inference, logit lens, knockout, steering, and more |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [ROADMAP.md](ROADMAP.md) | Development roadmap and architecture decisions |

## License

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev) and [Augment Code](https://www.augmentcode.com/) (review)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.

