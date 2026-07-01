# Design: Minimum Supported Candle Version

**Status:** Implemented
**Relates to:** Roadmap §8 item 6

## Question

What candle version should candle-mi target, and how should we pin it?

## Context

candle is pre-1.0 (currently 0.9.x). Breaking changes between minor versions are possible.

## Recommendation

Track a minor version in `Cargo.toml` and update incrementally:

```toml
[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
```

> **As implemented:** the shipped `Cargo.toml` uses the caret range `"0.9"`
> (`>=0.9.0, <0.10.0`), not the exact pin `"=0.9"` this note originally proposed.
> The caret admits compatible `0.9.x` patch upgrades automatically; CI's rolling
> matrix plus the pinned `Cargo.lock` provide the compatibility check the exact
> pin was meant to guarantee.

Test against the resolved version in CI. Widen the range only after verifying compatibility.

## Open questions

- Should we support multiple candle versions via feature flags, or just track the latest?
- When candle reaches 1.0, switch to standard semver ranges.
