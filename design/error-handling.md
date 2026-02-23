# Design: Error Handling

**Status:** Proposed
**Relates to:** Roadmap §8 item 5

## Question

Should candle-mi use `anyhow::Result` (like plip-rs) or typed error enums?

## Context

plip-rs uses `anyhow` throughout — convenient for a private project but poor for downstream consumers who need to match on specific error kinds.

## Recommendation

Typed error enums wrapping candle errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum MIError {
    #[error("model error: {0}")]
    Model(#[from] candle_core::Error),
    #[error("hook error: {0}")]
    Hook(String),
    #[error("intervention error: {0}")]
    Intervention(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MIError>;
```

## Open questions

- Should `Hook` and `Intervention` errors carry structured data (e.g., the offending `HookPoint`) or just a message string?
- Should we re-export `candle_core::Error` for convenience?
