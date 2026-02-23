# Design: Hook System

**Status:** Proposed
**Relates to:** Roadmap §2.1 (hook point table), §8 item 2

## Question

How should hook points be identified — enum, string, or both?

## Context

TransformerLens uses string names (`"blocks.5.attn.hook_pattern"`). Rust's type system offers an alternative: an enum with compile-time safety. The two are not mutually exclusive.

## Decision: Enum primary, string conversion, `Custom` escape hatch

### The enum

```rust
pub enum HookPoint {
    // Embedding
    Embed,
    // Per-layer (transformer)
    ResidPre(usize),
    AttnQ(usize),
    AttnK(usize),
    AttnV(usize),
    AttnScores(usize),
    AttnPattern(usize),
    AttnOut(usize),
    ResidMid(usize),
    MlpPre(usize),
    MlpPost(usize),
    MlpOut(usize),
    ResidPost(usize),
    // Final
    FinalNorm,
    // RWKV-specific
    RwkvState(usize),
    RwkvDecay(usize),
    // Escape hatch
    Custom(String),
}
```

### String conversion

`Display` and `FromStr` give TransformerLens-compatible serialization:

```rust
// HookPoint::AttnPattern(5).to_string() → "blocks.5.attn.hook_pattern"
// "blocks.5.attn.hook_pattern".parse::<HookPoint>() → Ok(AttnPattern(5))
```

### Accept both at the API level

API methods accept `Into<HookPoint>`, so callers can use either style:

```rust
impl HookSpec {
    pub fn capture<H: Into<HookPoint>>(&mut self, hook: H) -> &mut Self;
}

// Both work:
hooks.capture(HookPoint::AttnPattern(5));       // enum — compiler-checked
hooks.capture("blocks.5.attn.hook_pattern");    // string — TransformerLens-style
```

This requires `impl From<&str> for HookPoint` which parses the string and falls back to `Custom(s.to_string())` if unrecognised.

## Trade-offs

| Aspect | Enum | String | Both |
|--------|------|--------|------|
| Typo safety | Compile-time error | Silent runtime failure | Compile-time for known hooks |
| IDE autocomplete | Yes | No | Yes for enum path |
| Exhaustive matching | Yes (`match` coverage) | No | Weakened by `Custom(_)` arm |
| Extensibility | Requires crate release | Fully open | Open via `Custom` |
| TransformerLens compat | Needs `Display` | Native | Via `Display`/`FromStr` |
| Serialization (JSON) | Needs `Serialize` impl | Free | Via string conversion |

## Alternatives considered

1. **Enum-only** — maximum safety but no extensibility; RWKV/CLT hooks would force frequent releases.
2. **String-only** — maximum flexibility but `"blocks.5.atn.hook_pattern"` (typo) compiles and fails silently at runtime.
3. **Trait-based** (`trait HookName`) — over-engineered for the use case; adds indirection without clear benefit.

## Open questions

- Should `Custom` hooks participate in the zero-overhead guarantee (§8 item 3), or are they allowed a small allocation cost?
- Should the enum carry architecture tags (`Transformer(_)` vs `Rwkv(_)`) or stay flat?
