# candle-mi Coding Conventions (Grit + Grit-MI Extensions)

This document describes the [Amphigraphic coding](https://github.com/PCfVW/Amphigraphic-Strict) conventions used in candle-mi. It is a superset of
the [Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit).

## Annotation Patterns

Every annotation below is mandatory when the corresponding situation applies.

### `// TRAIT_OBJECT: <reason>`
Required on every `Box<dyn Trait>` or `&dyn Trait` usage.
> Example: `// TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch`

### `// EXHAUSTIVE: <reason>`
Required on `#[allow(clippy::exhaustive_enums)]`.
> Example: `// EXHAUSTIVE: internal dispatch enum; crate owns and matches all variants`

### `// EXPLICIT: <reason>`
Required when a match arm is intentionally a no-op, or when an imperative
loop is used instead of an iterator chain for a stateful computation.
> Example: `// EXPLICIT: WKV recurrence is stateful; .map() would hide the state update`

### `// PROMOTE: <reason>`
Required immediately before any `.to_dtype(DType::F32)?` call that precedes
a numerically sensitive operation (softmax, log, exp, norm).
> Example: `// PROMOTE: softmax over F16 produces NaN; compute in F32`

### `// CONTIGUOUS: <reason>`
Required immediately before any `.contiguous()?` call that precedes a matmul.
> Example: `// CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout`

### `// BORROW: <what is converted>`
Required on explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions (Grit Rule 2).
> Example: `// BORROW: explicit .as_str() instead of Deref coercion`

### `// SAFETY: <invariants>`
Required on every `unsafe` block or function (inline comment, not a doc comment).
Not expected in candle-mi (`#![forbid(unsafe_code)]`); included for completeness.

## Shape Documentation Format (Rule 12)

All public functions that accept or return `Tensor` must document shapes in
their doc comment using the following format:

    /// # Shapes
    /// - `q`: `[batch, n_heads, seq_q, head_dim]` -- query tensor
    /// - `k`: `[batch, n_kv_heads, seq_k, head_dim]` -- key tensor
    /// - returns: `[batch, n_heads, seq_q, head_dim]`

Rules:
- Use concrete dimension names, never `d0`/`d1`.
- Batch dimension is always first.
- Document every tensor argument and the return tensor.

## #[non_exhaustive] Policy (Rule 11)

- Public enums that may gain new variants: `#[non_exhaustive]`.
- Internal dispatch enums matched exhaustively by this crate:
  `#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: <reason>`.

## Hook Purity Contract (Rule 16)

- `HookSpec::capture()` takes only a hook point name -- no callback. The
  captured tensor is stored in `HookCache` and retrieved after `forward()`.
  The absence of a mutation mechanism is the enforcement.
- `HookSpec::intervene()` takes a typed `Intervention` value. All mutations
  go through this path and are visible at the call site.

## `#[must_use]` Policy (Rule 17)

All public functions and methods that return a value and have no side effects
must be annotated `#[must_use]`.  This includes constructors (`new`,
`with_capacity`), accessors (`len`, `is_empty`, `get_*`), and pure queries.
Without the annotation, a caller can silently discard the return value — which
for these functions is always a bug, since the call has no other effect.

The `clippy::must_use_candidate` lint enforces this at `warn` level
(promoted to error by `#![deny(warnings)]`).
