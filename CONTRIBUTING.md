# Contributing to candle-mi

Thank you very much for your interest in contributing to candle-mi! This document
explains what we expect from contributions so that your PR has the best
chance of a smooth review.

## candle-mi is a library, not an application

candle-mi is a published crate with a deliberate public API surface.
Every `pub` item is a commitment to downstream users. Before adding or
changing public API:

1. **Read the existing API.** Check whether the operation you need
   already exists (or nearly exists) in the crate. Duplicating logic
   that lives elsewhere in the codebase will be caught in review.
2. **Understand the design.** The hook system, intervention enum,
   backend trait, and config-driven transformer are all intentional
   abstractions. Read [`HOOKS.md`](HOOKS.md),
   [`BACKENDS.md`](BACKENDS.md), and the relevant source files before
   proposing changes to them.
3. **Minimise the public surface.** Prefer `pub(crate)` over `pub`
   unless external callers genuinely need the item. Making an internal
   function `pub` requires the same API contract as any other public
   item: doc comments, `# Errors` / `# Shapes` sections, `#[must_use]`
   where applicable.
4. **Be consistent.** If an existing variant uses `f64`, a new variant
   in the same enum should use `f64` unless there is a documented
   reason not to. Consistency across the API matters more than local
   convenience.
5. **Check the design docs.** The [`design/`](design/) folder contains
   design proposals and decisions for major features. If a design doc
   already exists for the area you want to change, read it first. For
   non-trivial API additions, a design doc may be requested before
   implementation begins.

## Before you open a PR

### Read the conventions

All code in candle-mi follows [`CONVENTIONS.md`](CONVENTIONS.md). This
is not optional. The conventions include mandatory annotations
(`// CAST:`, `// PROMOTE:`, `// BORROW:`, `// INDEX:`,
`// CONTIGUOUS:`, `// SAFETY:`, `// EXPLICIT:`, `// TRAIT_OBJECT:`),
doc-comment rules, shape documentation, `#[must_use]` policy, and more.

Read the file. Apply every applicable rule **as you write each line**,
not as a review pass after the fact. In particular, every new `.rs`
file must carry the SPDX license header:
`// SPDX-License-Identifier: MIT OR Apache-2.0`

### Run the CI checks locally

Your PR will be tested by GitHub Actions. Save yourself (and us) a
round-trip by running these before pushing:

```powershell
cargo fmt --check
cargo clippy --features transformer -- -W clippy::pedantic
cargo test
```

If your change touches RWKV code, also run with `--features rwkv`.
If it touches CLT/SAE code, add `--features clt` or `--features sae`.

A PR that fails `cargo fmt --check` will be rejected by CI immediately.

### Check for existing helpers

Before writing a new utility function, search the codebase for similar
logic. candle-mi already has internal helpers for common tensor
operations (sparse delta construction, dtype promotion, position-based
injection, etc.). Duplicating existing code will be flagged in review
and asked to be refactored.

### Write tests

- Cover the happy path and at least one error path.
- Test edge cases (e.g., multiple positions, scale factors != 1.0,
  dtype mismatches, out-of-bounds inputs).
- Gate tests behind the appropriate feature flags with
  `#[cfg(any(feature = "transformer", feature = "rwkv"))]` when they
  use backend-specific code.

## PR expectations

### Keep it focused

One feature or fix per PR. If your change touches unrelated code,
split it into separate PRs.

### Explain the "why"

The PR description should explain **why** this change is needed, not
just what it does. Link to papers, issues, or use cases. The code
shows the "what"; the description provides the context reviewers need.

### Performance matters

candle-mi runs on consumer GPUs with limited VRAM. If your change adds
per-element loops, per-position allocations, or host-device round-trips
inside a hot path, justify it or batch the work. One tensor operation
is almost always better than N operations in a loop.

### No AI-generated code without review

If you use an AI coding assistant to help write your contribution,
**you are still responsible for the result.** That means:

- Every convention annotation is present and correct.
- No code is duplicated from elsewhere in the crate.
- The public API is consistent with existing patterns.
- You have read and understood the code you are submitting.

AI assistants do not read `CONVENTIONS.md` or check for existing
helpers unless explicitly told to. The output needs your review before
it becomes a PR.

## What goes in the changelog

If your PR adds, changes, or fixes user-visible behaviour, add an
entry to the `## [Unreleased]` section of
[`CHANGELOG.md`](CHANGELOG.md) following the existing format.

## License

candle-mi is dual-licensed under MIT and Apache 2.0. By submitting a
PR, you agree that your contribution is licensed under the same terms.
