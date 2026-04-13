# Hook Point Reference and Intervention Walkthrough

> Hook points are named locations in a model's forward pass where activations
> can be **captured** (read) or **intervened on** (modified).  candle-mi
> follows the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
> naming convention, extended with RWKV-specific hook points.

---

## Table of Contents

- [Overview](#overview)
- [Hook Points](#hook-points)
  - [Naming Convention](#naming-convention)
  - [Transformer Hook Points](#transformer-hook-points)
  - [RWKV Hook Points](#rwkv-hook-points)
- [HookSpec: Declaring Captures and Interventions](#hookspec-declaring-captures-and-interventions)
  - [Captures](#captures)
  - [Interventions](#interventions)
  - [Combining Captures and Interventions](#combining-captures-and-interventions)
  - [Merging Specs](#merging-specs)
- [HookCache: Retrieving Results](#hookcache-retrieving-results)
- [Intervention Types](#intervention-types)
  - [Replace](#replace)
  - [Add (Steering)](#add-steering)
  - [Knockout](#knockout)
  - [Scale](#scale)
  - [Zero](#zero)
- [RWKV State Interventions](#rwkv-state-interventions)
  - [State Knockout](#state-knockout)
  - [State Steering](#state-steering)
- [Zero-Overhead Guarantee](#zero-overhead-guarantee)
- [Worked Examples](#worked-examples)
  - [1. Capture Attention Patterns](#1-capture-attention-patterns)
  - [2. Logit Lens via Residual Stream](#2-logit-lens-via-residual-stream)
  - [3. Attention Knockout](#3-attention-knockout)
  - [4. Activation Patching](#4-activation-patching)
  - [5. RWKV State Knockout](#5-rwkv-state-knockout)

---

## Overview

The hook system has three components:

| Type | Role |
|------|------|
| [`HookPoint`](#hook-points) | Identifies **where** in the forward pass to act |
| [`HookSpec`](#hookspec-declaring-captures-and-interventions) | Declares **what** to capture and **which** interventions to apply |
| [`HookCache`](#hookcache-retrieving-results) | Stores the **results**: output logits + any captured tensors |

The flow is always:

```rust
// 1. Declare what you want
let mut hooks = HookSpec::new();
hooks.capture(HookPoint::AttnPattern(5));

// 2. Run the forward pass
let cache = model.forward(&input, &hooks)?;

// 3. Retrieve results
let logits = cache.output();                              // always present
let attn = cache.require(&HookPoint::AttnPattern(5))?;   // captured tensor
```

When `hooks` is empty, the forward pass has **zero overhead** — no extra
clones, no allocations.  See [Zero-Overhead Guarantee](#zero-overhead-guarantee).

---

## Hook Points

### Naming Convention

Every `HookPoint` variant maps to a TransformerLens-style string via
`Display` and `FromStr`:

```rust
use candle_mi::HookPoint;

let hook = HookPoint::AttnPattern(5);
assert_eq!(hook.to_string(), "blocks.5.attn.hook_pattern");

let parsed: HookPoint = "blocks.5.attn.hook_pattern".parse().unwrap();
assert_eq!(parsed, hook);
```

API methods accept `Into<HookPoint>`, so both styles work interchangeably:

```rust
hooks.capture(HookPoint::AttnPattern(5));      // enum — compile-time checked
hooks.capture("blocks.5.attn.hook_pattern");   // string — TransformerLens style
```

Unknown strings parse as `HookPoint::Custom(s)`, providing an escape hatch
for backend-specific hook points.

### Transformer Hook Points

The table below lists all hook points in the `GenericTransformer` forward
pass, in execution order.  All hook points support both **capture** and
**intervention**.

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Embed` | `hook_embed` | `[batch, seq, hidden]` | After token embedding (and optional embedding scale) |
| `ResidPre(i)` | `blocks.{i}.hook_resid_pre` | `[batch, seq, hidden]` | Residual stream before layer `i` |
| `AttnQ(i)` | `blocks.{i}.attn.hook_q` | `[batch, n_heads, seq, head_dim]` | Query vectors (before RoPE) |
| `AttnK(i)` | `blocks.{i}.attn.hook_k` | `[batch, n_kv_heads, seq, head_dim]` | Key vectors (before RoPE) |
| `AttnV(i)` | `blocks.{i}.attn.hook_v` | `[batch, n_kv_heads, seq, head_dim]` | Value vectors |
| `AttnScores(i)` | `blocks.{i}.attn.hook_scores` | `[batch, n_heads, seq_q, seq_k]` | Pre-softmax attention logits |
| `AttnPattern(i)` | `blocks.{i}.attn.hook_pattern` | `[batch, n_heads, seq_q, seq_k]` | Post-softmax attention probabilities |
| `AttnOut(i)` | `blocks.{i}.hook_attn_out` | `[batch, seq, hidden]` | Attention output (after `o_proj`) |
| `ResidMid(i)` | `blocks.{i}.hook_resid_mid` | `[batch, seq, hidden]` | Residual stream after attention, before MLP |
| `MlpPre(i)` | `blocks.{i}.mlp.hook_pre` | `[batch, seq, hidden]` | After mid-layer norm (MLP input) |
| `MlpPost(i)` | `blocks.{i}.mlp.hook_post` | `[batch, seq, hidden]` | MLP output (before optional post-feedforward norm) |
| `MlpOut(i)` | `blocks.{i}.hook_mlp_out` | `[batch, seq, hidden]` | After optional post-feedforward norm (Gemma 2 only; otherwise same as `MlpPost`) |
| `ResidPost(i)` | `blocks.{i}.hook_resid_post` | `[batch, seq, hidden]` | Residual stream after full layer `i` |
| `FinalNorm` | `hook_final_norm` | `[batch, seq, hidden]` | After final layer norm (before logit projection) |

**Notes:**

- `n_kv_heads` may differ from `n_heads` in grouped-query attention (GQA).
  LLaMA 3, Qwen2, Gemma, Phi-3, and Mistral all use GQA.
- Q and K are captured **before** rotary position embedding (RoPE).  This
  matches TransformerLens behavior and is the right level for
  interventions: if you replace Q or K, RoPE is applied to the replacement.
- `AttnScores` is the standard point for knockout masks (`Intervention::Knockout`).
  The mask is added pre-softmax, so `-inf` entries become zero probability
  after softmax.
- `MlpOut` differs from `MlpPost` only for Gemma 2, which has a
  post-feedforward layernorm.  For all other architectures the tensors are
  identical.

### RWKV Hook Points

The `GenericRwkv` backend (RWKV-6 Finch and RWKV-7 Goose) exposes these
hook points.  RWKV hooks are **capture-only** except for `Embed`, which
also supports interventions.  State modifications use the dedicated
[State Intervention](#rwkv-state-interventions) API.

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Embed` | `hook_embed` | `[batch, seq, hidden]` | After token embedding |
| `ResidPre(i)` | `blocks.{i}.hook_resid_pre` | `[batch, seq, hidden]` | Residual stream before layer `i` |
| `RwkvState(i)` | `blocks.{i}.rwkv.hook_state` | `[batch, n_heads, head_dim, head_dim]` | Accumulated WKV recurrent state after layer `i` |
| `RwkvDecay(i)` | `blocks.{i}.rwkv.hook_decay` | `[batch, seq, n_heads, head_dim]` | Per-timestep decay weights |
| `RwkvEffectiveAttn(i)` | `blocks.{i}.rwkv.hook_effective_attn` | `[batch, n_heads, seq_q, seq_src]` | Effective attention derived from WKV recurrence (ReLU + L1 normalized) |
| `ResidPost(i)` | `blocks.{i}.hook_resid_post` | `[batch, seq, hidden]` | Residual stream after full layer `i` |
| `FinalNorm` | `hook_final_norm` | `[batch, seq, hidden]` | After final layer norm |

**Notes:**

- `RwkvEffectiveAttn` is **computed on demand**: it is only calculated when
  you request capture of that hook point.  This avoids the overhead of
  deriving the attention matrix from the WKV recurrence when not needed.
- `RwkvState` contains the accumulated key-value outer product state from
  the WKV recurrence — the RWKV equivalent of a KV cache, but compressed
  into a fixed-size matrix per head.

### Stoicheia Hook Points

The stoicheia backends (`StoicheiaRnn`, `StoicheiaTransformer`) expose hook
points for ARC's [AlgZoo](https://github.com/alignment-research-center/alg-zoo)
tiny models.

**`StoicheiaRnn`** — uses `HookPoint::Custom(String)` for per-timestep
capture (the RNN cell has no natural correspondence to transformer layers):

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Custom` | `rnn.hook_pre_activation.{t}` | `[batch, H]` | Before ReLU at timestep `t` |
| `Custom` | `rnn.hook_hidden.{t}` | `[batch, H]` | Hidden state after timestep `t` |
| `Custom` | `rnn.hook_final_state` | `[batch, H]` | Final hidden state `h_n` |
| `Custom` | `rnn.hook_output` | `[batch, output_size]` | After output projection |

Per-timestep hooks (`{t}`) are only captured when explicitly requested —
the backend pre-scans which timesteps are in the `HookSpec` to avoid
per-step string allocation (zero overhead when no hooks are active).

**`StoicheiaTransformer`** — reuses standard `HookPoint` variants since
the attention-only architecture maps naturally:

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Embed` | `hook_embed` | `[batch, seq, H]` | After token + positional embedding |
| `ResidPre(i)` | `blocks.{i}.hook_resid_pre` | `[batch, seq, H]` | Before attention layer `i` |
| `AttnScores(i)` | `blocks.{i}.attn.hook_scores` | `[batch, 1, seq, seq]` | Pre-softmax attention |
| `AttnPattern(i)` | `blocks.{i}.attn.hook_pattern` | `[batch, 1, seq, seq]` | Post-softmax attention |
| `AttnOut(i)` | `blocks.{i}.hook_attn_out` | `[batch, seq, H]` | Attention output |
| `ResidPost(i)` | `blocks.{i}.hook_resid_post` | `[batch, seq, H]` | After residual add |

No `MlpPre/Post/Out` (no MLP blocks), no `FinalNorm` (no normalization),
no `ResidMid` (no MLP means `ResidPost` = after attention). Attention is
full bidirectional (no causal mask), single-head.

---

## HookSpec: Declaring Captures and Interventions

`HookSpec` is the single configuration object passed to every `forward()`
call.  It declares both what to capture and where to intervene.

### Captures

Request a tensor snapshot at a hook point:

```rust
use candle_mi::{HookPoint, HookSpec};

let mut hooks = HookSpec::new();
hooks.capture(HookPoint::AttnPattern(5))
     .capture(HookPoint::ResidPost(5))
     .capture("blocks.5.hook_resid_pre");  // string form works too
```

### Interventions

Register a modification at a hook point:

```rust
use candle_mi::{HookPoint, HookSpec, Intervention};

let mut hooks = HookSpec::new();
hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
```

Multiple interventions can target the same hook point — they are applied
in registration order:

```rust
hooks.intervene(HookPoint::AttnScores(5), Intervention::Scale(0.5));
hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
```

### Combining Captures and Interventions

Captures and interventions are independent.  You can capture a hook point
without intervening, intervene without capturing, or both:

```rust
let mut hooks = HookSpec::new();
// Capture attention patterns AND knock out an edge
hooks.capture(HookPoint::AttnPattern(5));
hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
```

### Merging Specs

Use `extend()` to merge two `HookSpec` instances.  This is useful when
combining intervention sources (e.g., CLT suppress + inject):

```rust
let mut combined = HookSpec::new();
combined.extend(&capture_spec);
combined.extend(&intervention_spec);
```

### Query Methods

| Method | Returns |
|--------|---------|
| `is_empty()` | `true` if no captures, interventions, or state specs |
| `num_captures()` | Number of requested captures |
| `num_interventions()` | Number of registered interventions |
| `is_captured(&hook)` | Whether a specific hook point will be captured |
| `has_intervention_at(&hook)` | Whether any intervention targets a hook point |

---

## HookCache: Retrieving Results

`HookCache` is returned by `forward()`.  It always contains the output
logits and any tensors captured via `HookSpec::capture()`.

```rust
let cache = model.forward(&input, &hooks)?;

// Output logits — always present
let logits = cache.output();               // &Tensor, shape [batch, seq, vocab]
let logits = cache.into_output();          // Tensor (consumes cache)

// Captured tensors
let attn = cache.get(&HookPoint::AttnPattern(5));          // Option<&Tensor>
let attn = cache.require(&HookPoint::AttnPattern(5))?;     // Result<&Tensor>

// Metadata
let n = cache.num_captures();
```

`get()` returns `None` if the hook point was not captured; `require()`
returns `MIError::Hook` with a descriptive message.

---

## Intervention Types

The `Intervention` enum provides five modification primitives.  All five
work at any transformer hook point that supports interventions.

### Replace

Replace the tensor entirely with a provided value:

```rust
Intervention::Replace(new_tensor)
```

**Use cases:** activation patching (swap clean activations into a corrupted
run), attention pattern replacement for steering experiments.

**Shape requirement:** `new_tensor` must match the original tensor's shape.

### Add (Steering)

Add a vector to the activation via broadcasting:

```rust
Intervention::Add(direction_vector)
```

**Use cases:** residual stream steering (add a direction vector scaled by
a coefficient), feature injection.

**Shape requirement:** `direction_vector` must be broadcastable to the
tensor's shape.  Dtype conversion is automatic — if you inject an F32
vector into a BF16 forward pass, the vector is cast to BF16 before
addition.

### Knockout

Add a pre-softmax mask to attention scores:

```rust
Intervention::Knockout(mask)
```

The mask contains `0.0` for positions to keep and `-inf` for positions to
knock out.  After softmax, `-inf` entries become zero probability.

**Use cases:** ablating specific attention edges (e.g., "what happens if
head 3 cannot attend from position 7 to position 0?").

**Shape requirement:** `mask` must be broadcastable to
`[batch, n_heads, seq_q, seq_k]`.

Use `create_knockout_mask()` to build masks from a `KnockoutSpec`:

```rust
use candle_mi::{KnockoutSpec, HookPoint, HookSpec, Intervention, create_knockout_mask};

let spec = KnockoutSpec::new()
    .layer(target_layer)
    .edge(query_pos, key_pos);    // knock out one edge

let mask = create_knockout_mask(
    &spec, n_heads, seq_len, device, candle_core::DType::F32,
)?;

let mut hooks = HookSpec::new();
hooks.intervene(HookPoint::AttnScores(target_layer), Intervention::Knockout(mask));
```

### Scale

Multiply all attention weights by a constant factor:

```rust
Intervention::Scale(2.0)
```

**Use cases:** amplifying or dampening attention at a layer, probing
attention sensitivity.

### Zero

Zero the tensor entirely:

```rust
Intervention::Zero
```

**Use cases:** complete ablation of a component (e.g., zero the MLP output
at a layer to measure its contribution).

---

## RWKV State Interventions

RWKV models have a recurrent state that accumulates information across
token positions.  Standard `Intervention` variants operate on tensors at
hook points; state interventions operate on the WKV recurrence loop itself.

### State Knockout

Skip the key-value write at specified token positions, making those tokens
invisible to all future positions:

```rust
use candle_mi::StateKnockoutSpec;

let spec = StateKnockoutSpec::new()
    .position(0)              // knock out position 0
    .position(3)              // and position 3
    .layer(12);               // only at layer 12

let mut hooks = HookSpec::new();
hooks.set_state_knockout(spec);
```

**Layer targeting:**

| Method | Effect |
|--------|--------|
| (default) | All layers |
| `.layer(i)` | Single layer |
| `.layers(&[2, 5, 8])` | Specific layers |
| `.layer_range(5, 10)` | Inclusive range |

### State Steering

Scale the key-value write at specified positions by a factor:

```rust
use candle_mi::StateSteeringSpec;

let spec = StateSteeringSpec::new(2.0)   // amplify 2x
    .position(0)
    .layer_range(0, 11);

let mut hooks = HookSpec::new();
hooks.set_state_steering(spec);
```

**Scale semantics:**

| Scale | Effect |
|-------|--------|
| `0.0` | Knockout (equivalent to `StateKnockoutSpec`) |
| `1.0` | No-op (normal forward pass) |
| `< 1.0` | Dampen the token's state contribution |
| `> 1.0` | Amplify the token's state contribution |

**Priority:** if both `state_knockout` and `state_steering` are set,
knockout takes priority at positions where both apply.

---

## Zero-Overhead Guarantee

When `HookSpec` is empty (no captures, no interventions, no state specs),
the forward pass is identical to a plain forward pass:

- **No tensor clones** — capture checks are `HashSet::contains()` returning
  `false`, which skips the `.clone()`.
- **No extra allocations** — intervention lists are empty; iteration is
  a no-op.
- **Minimal branch cost** — each hook point is a single `if` check.

This guarantee is verified by benchmarks (see `design/hook-overhead.md`):
+11.5% GPU overhead with full capture of all 194 hook points, within noise
on CPU.

---

## Worked Examples

### 1. Capture Attention Patterns

Capture the post-softmax attention pattern at layer 5 and inspect its shape:

```rust
use candle_mi::{HookPoint, HookSpec, MIModel};

let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
let tokenizer = model.tokenizer().unwrap();

let tokens = tokenizer.encode("The capital of France is")?;
let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;

let mut hooks = HookSpec::new();
hooks.capture(HookPoint::AttnPattern(5));

let cache = model.forward(&input, &hooks)?;
let attn = cache.require(&HookPoint::AttnPattern(5))?;
// Shape: [1, n_heads, seq_len, seq_len]
println!("Attention shape: {:?}", attn.dims());
```

### 2. Logit Lens via Residual Stream

Capture residual streams at every layer and project each to vocabulary
logits:

```rust
use candle_mi::{HookPoint, HookSpec, MIModel};

let mut hooks = HookSpec::new();
for layer in 0..model.num_layers() {
    hooks.capture(HookPoint::ResidPost(layer));
}

let cache = model.forward(&input, &hooks)?;

for layer in 0..model.num_layers() {
    let resid = cache.require(&HookPoint::ResidPost(layer))?;
    // Extract last position: [1, seq, hidden] → [1, hidden]
    let last = resid.get(0)?.get(seq_len - 1)?.unsqueeze(0)?;
    let logits = model.project_to_vocab(&last)?;
    let token_id = candle_mi::sample_token(&logits.flatten_all()?, 0.0)?;
    let token_text = tokenizer.decode(&[token_id])?;
    println!("Layer {layer:>2}: {token_text}");
}
```

### 3. Attention Knockout

Knock out the attention edge from the last token to position 0 at a middle
layer:

```rust
use candle_mi::{HookPoint, HookSpec, Intervention, KnockoutSpec, create_knockout_mask};

let target_layer = model.num_layers() / 2;
let spec = KnockoutSpec::new()
    .layer(target_layer)
    .edge(seq_len - 1, 0);   // last token cannot attend to position 0

let mask = create_knockout_mask(
    &spec, model.num_heads(), seq_len, model.device(), candle_core::DType::F32,
)?;

// Baseline (no intervention)
let baseline = model.forward(&input, &HookSpec::new())?;

// Ablated (with knockout)
let mut hooks = HookSpec::new();
hooks.intervene(HookPoint::AttnScores(target_layer), Intervention::Knockout(mask));
let ablated = model.forward(&input, &hooks)?;

// Compare via KL divergence
let result = candle_mi::AblationResult::new(
    baseline.output().get(0)?.get(seq_len - 1)?,
    ablated.output().get(0)?.get(seq_len - 1)?,
    spec,
);
println!("KL divergence: {:.6}", result.kl_divergence()?);
```

### 4. Activation Patching

Patch clean activations into a corrupted forward pass at a specific layer:

```rust
use candle_mi::{HookPoint, HookSpec, Intervention};

// 1. Run clean forward, capturing residuals
let mut capture_hooks = HookSpec::new();
capture_hooks.capture(HookPoint::ResidPost(target_layer));
let clean_cache = model.forward(&clean_input, &capture_hooks)?;
let clean_resid = clean_cache.require(&HookPoint::ResidPost(target_layer))?.clone();

// 2. Run corrupted forward, replacing the residual with clean
let mut patch_hooks = HookSpec::new();
patch_hooks.intervene(
    HookPoint::ResidPost(target_layer),
    Intervention::Replace(clean_resid),
);
let patched_cache = model.forward(&corrupted_input, &patch_hooks)?;

// 3. Compare patched output to clean and corrupted baselines
```

### 5. RWKV State Knockout

Make a token invisible in the RWKV recurrent state at a specific layer:

```rust
use candle_mi::{HookSpec, MIModel, StateAblationResult, StateKnockoutSpec};

let model = MIModel::from_pretrained("RWKV/v6-Finch-1B6-HF")?;

// Knock out position 0 at the middle layer
let spec = StateKnockoutSpec::new()
    .position(0)
    .layer(model.num_layers() / 2);

let baseline = model.forward(&input, &HookSpec::new())?;

let mut hooks = HookSpec::new();
hooks.set_state_knockout(spec.clone());
let ablated = model.forward(&input, &hooks)?;

let result = StateAblationResult::new(
    baseline.output().get(0)?.get(seq_len - 1)?,
    ablated.output().get(0)?.get(seq_len - 1)?,
    spec,
);
println!("KL divergence: {:.6}", result.kl_divergence()?);
```
