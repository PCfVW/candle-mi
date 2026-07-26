# Backbones should be trainable — and today `backward()` says nothing when they are not

**Date:** July 26, 2026
**Source:** askesis `canvas` leg — B1 (the candle training loop; askesis's deferred V9)
**Affected area:** `src/diffusion/othello.rs`, `src/diffusion/mdlm.rs`, `src/transformer/attention.rs`,
`src/transformer/norm.rs`, `src/stoicheia/mod.rs`; plus a new initialization surface
**Severity:** Feature — with a **silent-wrong-answer** component (§2), which is the part that matters
**Measured against:** candle-mi v0.1.19, `candle-core`/`candle-nn` 0.11.0, `rustc` 1.97.1, Windows 11,
`CPU` `F32`

---

## 1. The use case

`canvas` is askesis's fourth leg: a masked-diffusion model over blocksworld plans, testing whether
`MDLM` decoding *is* plan-space refinement. Phase A is complete — a 17-token grid vocabulary, a
203,471-row dataset, and a symbolic referee (the `pweak` crate, `PWEAK`'s Theorem 1). Phase B trains
a 6-layer / 384-d bidirectional denoiser from scratch, and Phase D probes it against the referee's
ground truth: causal links, threats, commitment order. The leg is **Rust-first** by decision, so
both halves — training and probing — should run on candle.

`OthelloGpt` is already exactly the right architecture, and it is already parity-checked against the
fp32 `PyTorch` oracle (`tests/validate_othello_forward.rs`, max-abs logit diff `< 1e-3`). So the
obvious move is to train **that** module, over a `VarMap`, and hand the resulting checkpoint straight
back to candle-mi's hook surface for Phase D.

The reason to want this is scientific, not merely convenient. If `canvas` trains its own copy of the
architecture and probes candle-mi's, then any divergence between the two — a `GELU` variant, a
`LayerNorm` bias, an attention scale — becomes a confound that **no downstream measurement can
detect**. Interpretability results are claims about a specific set of weights running a specific
forward pass. Training and probing the same object makes that identity hold by construction rather
than by vigilance. That is the same argument the crate's own design philosophy makes for
re-implementing models with hooks built in (`ROADMAP.md` §2.1) — the backbone is candle-mi's, so it
should be candle-mi's for the whole life cycle of the weights.

## 2. The finding that matters: `backward()` succeeds and trains 1 parameter in 29

Building `OthelloGpt` over a `VarMap` and calling `backward()` **does not fail**:

```text
=== end-to-end: OthelloGpt built over a VarMap, then backward ===
vars created: 29
  init blocks.0.attn.qkv.weight   max|w| = 0.923699
  init pos_emb.weight             max|w| = 0.000000
  init tok_emb.weight             max|w| = 0.000000
forward ok, logits [1, 8, 17]
backward returned Ok — gradients for 1/29 vars: ["head.weight"]
```

`head.weight` is the only parameter that receives a gradient. Everything upstream of the final
`LayerNorm` is cut off.

**The mechanism.** `candle_nn::ops::softmax_last_dim` and the fused `layer_norm` / `rms_norm` are
built with `apply_op*_no_bwd` (`candle-nn` 0.11 `src/ops.rs:438`, `:944`, `:684`; `sdpa` at `:1317`
is a fourth, which candle-mi does not use). That constructor produces a tensor with **no recorded
op**, so it is not an error node in the graph — it is a *leaf*. `backward()` reaches it, finds
nothing above it, and stops.
`GradStore::get` then returns `None` for every parameter upstream, which an optimizer loop reads as
"this variable has no gradient" and skips.

**Why this is worse than a compile error or a panic.** A user who writes the obvious training loop
gets a loss that *decreases*, a checkpoint that saves cleanly, a model that loads and runs, and
evaluation numbers that are simply wrong. There is no error, no warning, and no lint. Measured — 300
`AdamW` steps at `lr = 1e-3` on a memorise-one-batch task, run on the same architecture built two
ways:

```text
  OthelloGpt over a VarMap (29 vars, 1 receiving a gradient):
    step   0  loss 3.8872        step 150  loss 2.1615
    step  50  loss 2.7870        step 200  loss 2.1255
    step 100  loss 2.2955        step 300  loss 2.1019

  the same architecture, backward-safe ops (29 vars, 29 receiving a gradient):
    step   0  loss 2.8332        step 150  loss 0.7033
    step  50  loss 2.3479        step 200  loss 0.3251
    step 100  loss 1.4315        step 300  loss 0.0996
```

The broken loop's curve is the shape everyone recognizes as learning: a sharp early drop, then a
slow approach to a floor. The floor is the tell, and it is exactly identifiable — the eight target
tokens are distinct, so the marginal entropy is `ln(8) = 2.079`, and the run plateaus at **2.1019**.
With `tok_emb` and `pos_emb` initialized to zeros (§5.1) the hidden state is constant across every
position and every input, so the only thing `head.weight` *can* fit is the marginal token
distribution — and it fits it, precisely, and stops. Nothing in that trajectory looks like a bug. It
looks like a model that needs more data, a bigger width, or a better learning rate, which is the
diagnosis a user would reach for first and the one that leads nowhere.

*Honest caveat on the comparison:* the two runs differ in the ops of §6 **and**, unavoidably, in the
embedding init — `candle_nn::embedding` supplies a `Randn` hint where `VarBuilder::get` supplies
`Const(0.)`, which is gap §5.1 showing up a second time. That difference explains the different
step-0 losses (3.8872 vs 2.8332); it does not touch the finding, since the gradient counts (1/29 and
29/29) are independent of initialization.

This is the failure mode the askesis house discipline is organized around, and it is worth naming as
such in the fix: **silence is the failure mode that matters**.

## 3. Op-level measurements

Each op below was applied to a fresh leaf `Var`, summed, and back-propagated; the test is whether
the gradient reaches the leaf.

| op | backward? |
|---|---|
| `ops::softmax_last_dim` | ❌ no gradient |
| `ops::softmax(dim)` | ✅ ok |
| `ops::log_softmax(dim)` | ✅ ok |
| `LayerNorm` (weight **+ bias**) | ❌ no gradient |
| `LayerNorm` (weight only) | ✅ ok |
| `RmsNorm` | ❌ no gradient |
| `gelu_erf` | ✅ ok |
| `matmul` + `narrow` | ✅ ok |

The `LayerNorm` split is not a quirk of the test — it is the dispatch inside
`candle_nn::LayerNorm::forward`, which takes the fused kernel only when
`x.is_contiguous() && remove_mean && bias.is_some()`. **Weight-only `LayerNorm` is accidentally
differentiable; with-bias `LayerNorm` is not.** Worth knowing, because it means `GenericMdlm`'s norms
already carry a gradient while `OthelloGpt`'s do not — for a reason invisible at either call site,
both of which just say `LayerNorm`. (Both models remain blocked at attention regardless.)

## 4. Where this bites in candle-mi

### Gradient barriers (all four attention implementations, and two norm families)

| site | op | consequence |
|---|---|---|
| `src/diffusion/othello.rs:332` | `ops::softmax_last_dim` | every block's attention is a barrier |
| `src/diffusion/mdlm.rs:251` | `ops::softmax_last_dim` | idem |
| `src/transformer/attention.rs:304` | `ops::softmax_last_dim` | idem |
| `src/stoicheia/mod.rs:393` | `ops::softmax_last_dim` | idem |
| `src/diffusion/othello.rs` — `ln1`/`ln2`/`ln_f` via `candle_nn::layer_norm` | fused `layer_norm` (with bias) | barrier at every norm site |
| `src/transformer/norm.rs` — `Norm::Layer`, `Norm::Rms` | fused `layer_norm` / `rms_norm` | idem |

### Already differentiable (no change needed)

- `src/diffusion/mdlm.rs` `load_layer_norm` → `LayerNorm::new_no_bias` (weight-only ⇒ composed path).
- `src/rwkv/norm.rs` — `LayerNorm` written out by hand.
- `GemmaRmsNorm` in `src/transformer/norm.rs` — likewise written out by hand.
- `gelu_erf`, `matmul`, `narrow`, `transpose`, `contiguous`, `broadcast_*`, `Embedding`, `Linear`.

### Inference-only, no gradient wanted

`src/backend.rs:746` (sampling) and `src/interp/intervention.rs:81` (probability read-out) also call
`softmax_last_dim`. Both are terminal read-outs; leaving them untouched is correct, and the dispatch
in §6 leaves them untouched automatically.

## 5. Two smaller gaps found on the way

### 5.1 Initialization is not part of the loader's contract

`Init::default()` is `Const(0.)` (`candle-nn` 0.11 `src/init.rs:143`), and `VarBuilder::get` uses it.
So over a `VarMap`, `OthelloGpt::load` creates `tok_emb.weight` and `pos_emb.weight` as **exact
zeros** — measured above, `max|w| = 0.000000`. A from-scratch model therefore starts with no token
identity and no position information at all: every input maps to the same hidden state. Once §6 is
fixed the embeddings do begin to receive gradients, but the starting point is not GPT-2's recipe and
nobody chose it.

The `Linear` layers fare better only by accident: `candle_nn::linear` supplies its own
Kaiming-uniform init, which is *a* sane init but is not GPT-2's `N(0, 0.02)` either.

Two consequences worth separating:

1. **No init recipe.** Loading and initializing are different operations sharing one entry point,
   and the default for the shared one is `Const(0.)`.
2. **No seed policy.** Whatever init happens draws from the device RNG, so two runs of the same
   program give different weights unless the caller remembers `Device::set_seed`. This is visible in
   the measurements above: across four runs of the identical probe, `blocks.0.attn.qkv.weight` came
   out with `max|w|` of 1.012280, 0.911329, 0.850538 and 0.923699. For a crate whose validation
   culture is exact differential testing, a from-scratch model should be reproducible from
   `(config, seed)` alone.

### 5.2 There is no training surface at all

`src/` contains no `VarMap`, no optimizer, no learning-rate schedule, and no checkpoint writer:
`VarMap|AdamW|Optimizer|SGD::` matches **0** lines, and all four matches for "backward" are doc prose
in `src/rwkv/mod.rs` about backward *recurrences* and linear functionals. That is a reasonable state
for an inference-and-probing crate, and §7 argues it should largely stay that way.

## 6. The fix, and it is validated

Replace the three fused ops with their composed equivalents **only when a gradient is actually
needed**, dispatching on `Tensor::track_op` (`candle-core` 0.11 `src/tensor.rs:592`).

This predicate is exactly right, and not by luck: candle records a `BackpropOp` only when an input
already tracks, so a forward with no `Var` anywhere upstream never starts tracking. Measured:

```text
=== is `track_op()` a sound dispatch predicate? ===
  varmap-backed output.track_op() = true
  inference-only output.track_op() = false
```

So the inference path is not merely fast — it is **byte-identical to today's**, and every existing
parity test keeps its meaning without re-baselining.

A small internal module (say `src/nn_ops.rs`), called from the sites in §4:

```rust
/// Softmax over the last dimension, differentiable when the graph is being tracked.
///
/// Dispatches on `Tensor::track_op`: an inference forward (no `Var` upstream) takes candle's
/// fused kernel, unchanged; a forward under a `VarMap` takes the composed form, which carries a
/// backward. Both subtract the row maximum before exponentiating, so the two paths agree to `F32`
/// rounding.
///
/// # Shapes
/// - `xs`: `[.., n]` — softmax is taken over the final axis
/// - returns: `[.., n]`
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor failures.
pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    if xs.track_op() {
        Ok(candle_nn::ops::softmax(xs, D::Minus1)?)
    } else {
        Ok(candle_nn::ops::softmax_last_dim(xs)?)
    }
}
```

…and the same shape of helper for `layer_norm` (weight + bias) and `rms_norm`, whose composed forms
are the formulas already present in `candle_nn`'s own non-fused path and in `src/rwkv/norm.rs`.

**Validated end to end.** Rebuilding the identical `OthelloGpt` architecture — same layer count,
same fused QKV, same `gelu_erf` MLP, same untied head — with only these substitutions:

```text
=== the same architecture, backward-safe ops only ===
gradients for 29/29 vars; missing: []
```

29 of 29, from 1 of 29. The swap is sufficient; nothing else in the backbone blocks a gradient.

### Why dispatch rather than a feature flag

A `training` feature would keep two code paths alive and let a user train under one and probe under
the other — numerically close, but the crate's whole discipline is that "close" is a thing you
*measure*, not assume. Runtime dispatch on `track_op` has one code path, no feature matrix, no
config mode to forget, and provably zero effect on inference. The only cost is one boolean test per
call site.

## 7. Scope recommendation: differentiability, not a training loop

The concrete ask is that candle-mi's backbones **carry a gradient**, plus a seeded initializer. It
is *not* that candle-mi grow an optimizer, an `EMA`, a learning-rate schedule, or a data loader —
those are experiment-shaped, they vary per study, and they would pull an interpretability crate into
a second identity it does not need. `canvas` will keep its own loop and is happy to.

**Sequencing note.** `canvas` B1 (the training loop) is blocked on item 1 below; items 2 and 3 are
what make the result reproducible and keep it from regressing. Everything else in this report can
wait for a later release without holding the askesis leg up.

Minimum for v0.1.20, in dependency order:

1. **`src/nn_ops.rs`** with the three dispatching helpers; the call sites in §4 switched over.
2. **A seeded initializer** — e.g. `OthelloGpt::init(config, &VarMap, device, seed)` applying the
   GPT-2 recipe (`N(0, 0.02)` for embeddings and linear weights, zero biases, `LayerNorm` weight 1 /
   bias 0), drawn from an explicitly seeded generator so a from-scratch model is reproducible from
   `(config, seed)` alone. `rand 0.8` is already a dependency.
3. **A regression test** that is the §2 measurement itself: build over a `VarMap`, backward, assert
   **every** variable receives a gradient. This is the test whose absence let the gap exist — and,
   being a count, it fails loudly rather than silently.

Two optional follow-ups, neither needed by `canvas`:

- **`Dropout`.** No backbone has it (they were built for inference). Training recipes that want
  `p > 0` would need it; `canvas` uses `0.0`, per the proven rhyme-leg recipe.
- **A lean logits path.** `MIBackend::forward` returns a `HookCache`, allocating a `HashMap` per
  step and taking a `HookSpec` that means nothing during training. It works — the measurements above
  went through it — but a `logits(&self, input_ids)` shortcut would be the honest signature for a
  training step.

## 8. What candle-mi gets out of it

- **A trainable reference model in-crate.** Small models trained from scratch are the cheapest
  possible fixtures for hook, intervention, and `CLT` tests — no download, no external checkpoint,
  and full control over what the model knows, which is precisely what a probe test wants.
- **The `canvas` checkpoint becomes candle-mi-native.** Phase D's probes (causal links, threats,
  commitment order against `pweak`'s ground truth) run on the hook surface with no conversion step
  and no second implementation to keep in parity.
- **`ROADMAP.md` §2.1 carried to its conclusion.** The crate already argues for owning model
  definitions rather than wrapping them. Owning them means owning them for training too — otherwise
  every study that needs a trained model must fork the definition, which is the one outcome the
  design philosophy exists to prevent.

## 9. How to reproduce

Throwaway crate, `CPU` only (no `CUDA` needed), ~70 s cold build:

```toml
[dependencies]
candle-core = "0.11"
candle-nn = "0.11"
anyhow = "1"
candle-mi = { path = "…/candle-mi", default-features = false, features = ["diffusion"] }
```

```rust
// Op-level: does the gradient reach a fresh leaf Var?
let v = Var::randn(0f32, 1f32, (2, 8), &Device::Cpu)?;
let out = candle_nn::ops::softmax_last_dim(v.as_tensor())?;
let grads = out.sum_all()?.backward()?;
assert!(grads.get(&v).is_none(), "softmax_last_dim is a gradient barrier");

// End to end: how many of OthelloGpt's parameters actually train?
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
let model = OthelloGpt::load(OthelloGptConfig::new(17, 16, 2, 2, 32, false)?, vb)?;
let cache = MIBackend::forward(&model, &ids, &HookSpec::new())?;
let grads = cache.output().sum_all()?.backward()?;
// -> 1 of 29: head.weight

// And what it looks like from inside a training loop (the §2 curves):
let mut adam = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;
for step in 0..=300 {
    let cache = MIBackend::forward(&model, &ids, &HookSpec::new())?;
    let logits = cache.output().reshape((8, 17))?;
    let loss = candle_nn::loss::cross_entropy(&logits, &flat_targets)?;
    candle_nn::Optimizer::backward_step(&mut adam, &loss)?;   // no error, ever
}
// -> 3.8872 -> 2.1019, plateauing at ln(8), the marginal entropy of the targets
```

The full probe (op table, end-to-end gradient count, `track_op` check, the backward-safe rebuild that
reaches 29/29, and both training curves) was run from `%TEMP%\train-probe`; it is disposable and
touches no repository.
