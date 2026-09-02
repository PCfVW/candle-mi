# Design: Intervention API

**Status:** Implemented — the unified-config recommendation shipped, but as
`HookSpec` (`src/hooks.rs`), **not** `ForwardConfig` (that name was never built).
See "As implemented" below.
**Relates to:** Roadmap §8 item 4

## Question

Should interventions use separate methods (plip-rs style) or a unified configuration object (pyvene style)?

## Context

- **plip-rs** uses separate methods: `forward_with_intervention`, `forward_with_steering`, `forward_with_attention`, etc. Simple but proliferates the API surface.
- **pyvene** uses a declarative configuration: one `forward` call with a config that specifies both what to capture and what to intervene on.

## Recommendation (as originally proposed)

Unified `forward(tokens, config)` where the config includes both hooks and
interventions, collapsing plip-rs's 5+ forward methods into one declarative,
composable call.

## As implemented

The recommendation shipped, but the unified type is named **`HookSpec`**, not
`ForwardConfig`. Its builder methods take `&mut self` and return `&mut Self`, and
`forward` returns a `HookCache` of the requested captures:

```rust
use candle_mi::{HookPoint, HookSpec, Intervention};

let mut hooks = HookSpec::new();
hooks
    .capture(HookPoint::AttnPattern(5))
    .intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask))
    .intervene(HookPoint::ResidPost(10), Intervention::Add(steering_vector));

let cache = model.forward(&input_ids, &hooks)?;
```

- The `Intervention` enum (`src/hooks.rs`) provides `Replace(Tensor)`,
  `PatchAt { position, value }` (one sequence position; see
  [`patch-at-position.md`](patch-at-position.md)), `Add(Tensor)`
  (residual/steering), `Knockout(Tensor)` (pre-softmax `-inf` mask),
  `Scale(f64)`, and `Zero` — there is no `Steer(vector, strength)`
  variant; steering is `Add` of a pre-scaled vector.
- `apply_intervention` takes the `HookPoint` alongside the tensor. Only
  `PatchAt` reads it, but it has to: dim 1 is the sequence in a
  `[batch, seq_len, hidden]` activation and a head in a
  `[batch, n_heads, seq_len, head_dim]` one, so a positional intervention
  cannot be applied blind.
- Richer attention-edge interventions live in `src/interp/intervention.rs`
  (`KnockoutSpec`, steering specs, `InterventionType`); RWKV state interventions
  are separate fields on `HookSpec` (`StateKnockoutSpec` / `StateSteeringSpec`).

### Resolved open questions

- **Reusable across calls?** Yes — `HookSpec` is `Clone + Default`; build once and
  pass by `&` to each `forward`.
- **Interventions needing state from a previous pass (activation patching)?**
  Captured activations are returned in the `HookCache`; feed a captured tensor
  into a later `forward` via `Intervention::Replace`/`Add` at the target hook.

## See also

- [HOOKS.md](../HOOKS.md) — user-facing reference for the intervention types and worked examples.
