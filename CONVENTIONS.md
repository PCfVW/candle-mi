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
Required immediately before any `.to_dtype(DType::F32)?` call that promotes
a tensor from a lower-precision dtype (F16, BF16) to F32 for numerical
correctness. Common reasons include:

- **Numerical functions**: softmax, log, exp, norm, sqrt
- **Matmul precision**: decoder weights stored as BF16 on disk
- **Accumulation**: running sums, averages, WKV recurrence
- **Dot-product precision**: matching a Python reference implementation
- **DType extraction**: `to_vec1::<f32>()` from BF16 safetensors

The reason must be specific to the call site, not a generic "numerical stability".

> Example: `// PROMOTE: softmax over F16 produces NaN; compute in F32`
> Example: `// PROMOTE: decoder weights are BF16 on disk; F32 for matmul precision`
> Example: `// PROMOTE: WKV recurrence must be in F32 for numerical stability`

### `// CONTIGUOUS: <reason>`
Required immediately before any `.contiguous()?` call that precedes a matmul.
> Example: `// CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout`

### `// BORROW: <what is converted>`
Required on explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions (Grit Rule 2).
> Example: `// BORROW: explicit .as_str() instead of Deref coercion`

### `// SAFETY: <invariants>`
Required on every `unsafe` block or function (inline comment, not a doc comment).

candle-mi is `#![forbid(unsafe_code)]` **by default**. Specific feature flags
relax this to `#![deny(unsafe_code)]` for narrowly scoped platform FFI:

| Feature | Accepted `unsafe` scope |
|---------|------------------------|
| `mmap` | Memory-mapped file I/O for sharded safetensors |
| `memory` | OS/GPU memory queries (`GetProcessMemoryInfo`, `cuMemGetInfo`) |

Each accepted use must satisfy all of:
1. The `unsafe` block is in a **single, dedicated module** (e.g., `src/mmap.rs`,
   `src/memory.rs`) — never scattered across the codebase.
2. Every `unsafe` block carries a `// SAFETY:` comment documenting the invariants.
3. The module is gated behind `#[cfg(feature = "...")]` — users who don't
   enable the feature get `forbid(unsafe_code)` with zero exceptions.

Adding a new accepted use requires updating this table and the `cfg_attr` lines
in `lib.rs`.

### `// INDEX: <reason>`
Required on every direct slice index (`slice[i]`, `slice[a..b]`) that cannot
be replaced by an iterator. Direct indexing panics on out-of-bounds; prefer
`.get(i)` with `?` or explicit error handling. Use direct indexing only when
the bound is provably valid and an iterator idiom would be significantly less
readable.
> Example: `// INDEX: i is bounded by dims.len() checked two lines above`

### `// CAST: <from> → <to>, <reason>`
Required on every `as` cast between numeric types. Prefer `From`/`Into` for
lossless conversions and `TryFrom`/`TryInto` with `?` for fallible ones.
Use `as` only when truncation or wrapping is the deliberate intent, or when
interfacing with a C-style API that mandates it.
> Example: `// CAST: usize → u32, tensor dim fits in u32 (checked at construction)`
> Example: `// CAST: f64 → f32, precision loss acceptable; value is a display scalar`

---

## Doc-Comment Rules

### Backtick Hygiene (`doc_markdown`)

All identifiers, types, trait names, field names, crate names, and
file-format names in doc comments must be wrapped in backticks so that
rustdoc renders them as inline code and Clippy's `doc_markdown` lint passes.

Applies to: struct/enum/field names, method names (`fn foo`), types
(`Vec<T>`, `Option<f32>`), crate names (`candle_core`, `safetensors`),
file extensions (`.npy`, `.npz`, `.safetensors`), and acronyms that double
as types (`DType`, `NaN`, `CPU`, `GPU`).

> ✅ `/// Loads weights from a [`.safetensors`] file into a [`Tensor`].`
> ❌ `/// Loads weights from a .safetensors file into a Tensor.`

### Field-Level Docs

Every field of every `pub` struct must carry a `///` doc comment describing:
1. what the field represents,
2. its unit or valid range where applicable.

Fields of `pub(crate)` structs follow the same rule. Private fields inside a
`pub(crate)` or `pub` struct must have at minimum a `//` comment if their
purpose is not self-evident from the name alone.

> Example:
> ```rust
> pub struct AttentionConfig {
>     /// Number of query heads. Must be a multiple of `n_kv_heads`.
>     pub n_heads: usize,
>     /// Number of key/value heads (grouped-query attention).
>     pub n_kv_heads: usize,
>     /// Per-head dimension. Total hidden size = `n_heads * head_dim`.
>     pub head_dim: usize,
> }
> ```

---

## Control-Flow Rules

### `if let` vs `match` (`match_like_matches_macro`, `single_match`)

Use the most specific construct for the pattern at hand:

| Situation | Preferred form |
|---|---|
| Testing a single variant, no binding needed | `matches!(expr, Pat)` |
| Testing a single variant, binding needed | `if let Pat(x) = expr { … }` |
| Two or more variants with different bodies | `match expr { … }` |
| Exhaustive dispatch over an enum | `match expr { … }` (never `if let` chains) |

Never use a `match` with a single non-`_` arm and a no-op `_ => {}` where
`if let` or `matches!` would be clearer. Conversely, never chain three or
more `if let … else if let …` arms where a `match` would be exhaustive.

> ✅ `if let Some(w) = weight { apply(w); }`
> ✅ `matches!(dtype, DType::F16 | DType::BF16)`
> ❌ `match weight { Some(w) => apply(w), None => {} }`

---

## Function Signature Rules

### `const fn`

Declare a function `const fn` when **all** of the following hold:
1. The body contains no heap allocation, I/O, or `dyn` dispatch.
2. All called functions are themselves `const fn`.
3. There are no trait-method calls that are not yet `const`.

This applies to constructors, accessors, and pure arithmetic helpers.
When in doubt, annotate and let the compiler reject it — do not omit `const`
preemptively.

> ✅ `pub const fn head_dim(&self) -> usize { self.hidden_size / self.n_heads }`
> ❌ `pub fn head_dim(&self) -> usize { self.hidden_size / self.n_heads }`

### Pass by Value vs Reference (`needless_pass_by_ref_mut`, `trivially_copy_pass_by_ref`)

Follow these rules for function parameters:

| Type | Rule |
|---|---|
| `Copy` type ≤ 2 words (`usize`, `f32`, `bool`, small `enum`) | Pass by value |
| `Copy` type > 2 words | Pass by reference |
| Non-`Copy`, not mutated | Pass by `&T` or `&[T]` |
| Non-`Copy`, mutated | Pass by `&mut T` |
| Owned, consumed by callee | Pass by value (move semantics) |
| `&mut T` not actually mutated in body | Change to `&T` |

Never accept `&mut T` when the function body never writes through the reference;
Clippy's `needless_pass_by_ref_mut` will flag it and callers lose the ability
to pass shared references.

> ✅ `fn scale(x: f32, factor: f32) -> f32`
> ❌ `fn scale(x: &f32, factor: &f32) -> f32`
> ✅ `fn apply_mask(tensor: &mut Tensor, mask: &Tensor)`
> ❌ `fn apply_mask(tensor: &mut Tensor, mask: &mut Tensor)` ← if mask is only read

---

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

## `# Errors` Doc Section

All public fallible methods (`-> Result<T>`) must include an `# Errors` section
in their doc comment. Each bullet uses the format:

    /// # Errors
    /// Returns [`MIError::Config`] if the model type is unsupported.
    /// Returns [`MIError::Model`] on weight loading failure.

Rules:
- Start each bullet with `Returns` followed by the variant in rustdoc link
  syntax, e.g., `` [`MIError::Config`] ``.
- Follow with `if` (condition), `on` (event), or `when` (circumstance).
- Use the concrete variant name, not the generic `MIError`.
- One bullet per distinct error path.

## Error Message Wording

Error strings passed to `MIError` variants follow two patterns:

- **External failures** (I/O, serde, network): `"failed to <verb>: {e}"`
  > Example: `MIError::Config(format!("failed to parse config: {e}"))`
- **Validation failures** (range, shape, lookup): `"<noun> <problem> (<context>)"`
  > Example: `MIError::Config(format!("source_layer {src} out of range (max {max})"))`
  > Example: `MIError::Hook(format!("hook point {point:?} not captured"))`

Rules:
- Use lowercase, no trailing period.
- Include the offending value and the valid range or constraint when applicable.
- Wrap external errors with `: {e}`, not `.to_string()`.

## `# Memory` Doc Section

Public methods that load large files (>100 MB, typically safetensors decoder
files) must include a `# Memory` section documenting:

1. **Peak allocation** — how much memory the method allocates at its peak.
2. **Residency** — whether the large allocation lives on CPU, GPU, or both.
3. **Lifetime** — whether the allocation is dropped before the method returns
   or persists in the returned value.

Format:

    /// # Memory
    /// Loads one decoder file (~2 GB) to CPU per source layer. Each file is
    /// dropped before loading the next. Peak: ~2 GB CPU.

## OOM-safe Decoder Loading Pattern

When loading large safetensors files (decoder weights, encoder weights),
follow the 7-step pattern to bound peak memory:

1. `ensure_path()` — resolve the file path (may trigger download).
2. `fs::read(&path)` — read the entire file into a `Vec<u8>` on CPU.
3. `SafeTensors::deserialize(&bytes)` — zero-copy parse of the byte buffer.
4. Extract the tensor view and build a candle `Tensor` on CPU.
5. Slice or narrow to the needed subset.
6. `drop(bytes)` (or let it go out of scope) — free the raw file buffer
   **before** loading the next file.
7. Optionally `.to_device(device)` if GPU computation follows.

The key invariant is: **at most one large file buffer is alive at any time**.
This bounds peak memory to roughly 1× the largest decoder file (~2 GB for
Gemma 2 2B CLTs).

> Example location: `CrossLayerTranscoder::score_features_by_decoder_projection`

## HashMap Grouping Idiom

When operations must be batched by a key (e.g., grouping features by source
layer to load each decoder file only once), use the `Entry` API:

```rust
let mut by_source: HashMap<usize, Vec<Item>> = HashMap::new();
for item in items {
    by_source.entry(item.key()).or_default().push(item);
}
```

Rules:
- Name the map `by_<grouping_key>` (e.g., `by_source`, `by_layer`).
- Use `.entry(key).or_default().push()` — never `if let Some` + `else insert`.
- Iterate the map to perform the batched operation (one file load per key).
