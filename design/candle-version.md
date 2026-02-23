# Design: Minimum Supported Candle Version

**Status:** Proposed
**Relates to:** Roadmap §8 item 6

## Question

What candle version should candle-mi target, and how should we pin it?

## Context

candle is pre-1.0 (currently 0.9.x). Breaking changes between minor versions are possible.

## Recommendation

Pin to a specific minor version in `Cargo.toml` and update incrementally:

```toml
[dependencies]
candle-core = "=0.9"
candle-nn = "=0.9"
```

Test against the pinned version in CI. Update the pin only after verifying compatibility.

## Open questions

- Should we support multiple candle versions via feature flags, or just track the latest?
- When candle reaches 1.0, switch to standard semver ranges.
