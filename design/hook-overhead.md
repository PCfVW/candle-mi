# Design: Hook Overhead Model

**Status:** Proposed
**Relates to:** Roadmap §8 item 3

## Question

How do we ensure zero overhead when no hooks are active?

## Constraint

When no hooks are captured and no interventions are applied, the forward pass must have **zero extra allocations and zero extra tensor copies** compared to a plain forward pass without hook infrastructure.

## Approach

Conditional logic at each hook point: check the `HookSpec` before cloning any tensor. When the spec is empty (the common case), each check is a single branch on an empty set — effectively free.

```rust
// Inside the forward pass:
let attn_pattern = softmax(&scores)?;
if hooks.is_captured(HookPoint::AttnPattern(layer)) {
    cache.store(HookPoint::AttnPattern(layer), attn_pattern.clone());
}
// Continue with attn_pattern (no clone on the hot path)
```

## Open questions

- Should `HookSpec` use a `HashSet<HookPoint>` or a fixed-size bitfield for the standard hooks?
- What is the measured overhead of the branch checks? (Benchmark in Phase 1)
