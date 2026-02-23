# Design: Intervention API

**Status:** Proposed
**Relates to:** Roadmap §8 item 4

## Question

Should interventions use separate methods (plip-rs style) or a unified configuration object (pyvene style)?

## Context

- **plip-rs** uses separate methods: `forward_with_intervention`, `forward_with_steering`, `forward_with_attention`, etc. Simple but proliferates the API surface.
- **pyvene** uses a declarative configuration: one `forward` call with a config that specifies both what to capture and what to intervene on.

## Recommendation

Unified `forward(tokens, config)` where the config includes both hooks and interventions:

```rust
let config = ForwardConfig::new()
    .capture(HookPoint::AttnPattern(5))
    .intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask))
    .intervene(HookPoint::ResidPost(10), Intervention::Steer(vector, strength));

let result = model.forward(tokens, &config)?;
```

This collapses plip-rs's 5+ forward methods into one, while keeping the API declarative and composable.

## Open questions

- Should `ForwardConfig` be reusable across calls (amortize allocation) or built fresh each time?
- How to handle interventions that need state from a previous forward pass (e.g., activation patching)?
