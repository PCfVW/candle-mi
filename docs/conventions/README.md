# Convention Documents for LLM-Assisted Development

## The Problem

When an LLM (Claude Code, in our case) generates code against a coding
standards document, it consistently misses rules on first pass — then fixes
them iteratively after linter feedback. Each iteration burns context, looks
sloppy, and sometimes introduces new issues while fixing old ones.

We observed this repeatedly on [candle-mi](https://github.com/PCfVW/candle-mi),
a mechanistic interpretability crate with strict conventions (annotation
patterns, doc-comment rules, function signature policies). The conventions
were clear and complete. The LLM read them every conversation. It still
missed rules — especially doc-comment backtick hygiene, `const fn`, and
`#[must_use]` — requiring 3–5 clippy fix cycles per file.

## The Root Cause

**Convention documents are written for humans, not LLMs.**

A human reads CONVENTIONS.md once, internalizes the rules, and applies them
from muscle memory. An LLM re-reads the document each conversation but does
not internalize — it pattern-matches against what is in context. If the
document is organized as a reference manual ("here are all the annotation
rules"), the LLM must mentally reorganize it into action triggers while
simultaneously writing code. That reorganization competes with the algorithm
being implemented, and the algorithm wins.

The standard industry workflow — write code, run linter, fix — works for
humans because fixing is fast and instructive. For an LLM, this loop is
wasteful: each fix iteration consumes context window, and corrections
occasionally introduce new issues.

## The Solution: Trigger-Based Organization

We restructured our CONVENTIONS.md from **rule-type grouping** (all
annotations together, all doc-comment rules together) to **trigger-moment
grouping** (all rules that apply "when writing a doc comment" together,
all rules that apply "when writing an `as` cast" together).

The key addition is a **trigger checklist** at the top of the document — a
table mapping "you are about to..." → "check these rules":

```markdown
| You are about to...              | Check these rules                    |
|----------------------------------|--------------------------------------|
| Write a `///` or `//!` comment   | Backtick hygiene, field-level docs   |
| Write a `pub fn`                 | `const fn`, `#[must_use]`, pass-by   |
| Write an `as` cast               | `// CAST:` annotation                |
| Write `slice[i]`                 | `// INDEX:` annotation               |
| ...                              | ...                                  |
```

This works because it matches how code generation actually proceeds: the LLM
is about to write a specific kind of construct, and the checklist tells it
what rules fire at that moment — before the line is written, not after the
linter catches it.

## Files

| File | Description |
|------|-------------|
| `CONVENTIONS-v1-reference-based.md` | Original: rules grouped by type (annotations, doc-comments, signatures). Clear and complete, but LLM misses rules on first pass. |
| `CONVENTIONS-v2-trigger-based.md` | Refactored: rules grouped by trigger moment ("when writing X, check Y"). Trigger checklist at top. Same rules, different organization. |

The root `CONVENTIONS.md` is always the active version (currently v2).

## Observations

- The annotation rules (CAST, INDEX, SAFETY) worked well even in v1 because
  they already had implicit triggers: each rule starts with "Required on
  every X". The trigger is the code pattern itself.
- Doc-comment rules (backtick hygiene, `# Errors` sections) were the most
  frequently missed because they have no code-pattern trigger — the LLM is
  thinking about *what to say*, not *how to format it*.
- Function signature rules (`const fn`, `#[must_use]`) were missed because
  they are declaration-time checks, not expression-time checks — they fire
  when writing a signature, a moment when attention is on types and names.
- The trigger checklist at the top interrupts the LLM's default priority
  ordering (correctness > compilation > conventions) by surfacing convention
  checks *before* code is written.

## Broader Applicability

This is not specific to Rust, candle-mi, or Claude. Any team using LLM-
assisted development with a coding standards document could benefit from
restructuring it as a trigger checklist. The principle:

> **Organize conventions by when they fire, not by what kind of rule they are.**

The cost is a one-time refactor of the document. The rules themselves don't
change — only their presentation.
