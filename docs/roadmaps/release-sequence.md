# candle-mi release sequence

Canonical checklist for a `vX.Y.Z` release on crates.io. Distilled from the
v0.1.9 release prep (2026-04-19). Replaces any ad-hoc derivation of the
release steps on future version bumps (v0.1.10, v0.1.11, …).

## When to use this doc

On a commit that is ready to become a release tag. Before starting:

- [`CLAUDE.md`](../../CLAUDE.md) pre-commit gate has already passed
  (`cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test`).
- `CHANGELOG.md` has bullets under `[Unreleased]` for every user-visible
  change since the last tag.
- All feature work for this release has landed on `main`.
- The next release artefact (e.g. `findings.md`, README rows) is committed.

## The sequence

### 1. Bump version

- `Cargo.toml` → `version = "X.Y.Z"`.
- Refresh `Cargo.lock` so it matches: `cargo update -p candle-mi`.
  Also good release hygiene: `cargo update -p hf-fetch-model` to pick up any
  patch release of the download crate.

### 2. Consolidate CHANGELOG

- Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (use today's UTC date).
- Insert a fresh empty `## [Unreleased]` section above it.
- Preserve the existing `### Added` / `### Changed` / `### Fixed` / `### Tests`
  subsections under the new `[X.Y.Z]` header.

### 3. Commit the release bump

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z and update CHANGELOG"
```

**CLAUDE.md rule:** `Cargo.toml` and `Cargo.lock` go in the **same commit**.
`publish.yml` fails on a dirty `Cargo.lock`.

### 4. Dry-run the CI workflow locally

Mirrors every step in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
(stable-Rust lane; MSRV 1.88 lane differs only in toolchain, not commands).
`&&` chaining stops on first failure:

```bash
cargo fmt --check \
  && cargo build --no-default-features --features transformer \
  && cargo clippy --no-default-features --features transformer -- -W clippy::pedantic \
  && cargo test --no-default-features --features transformer \
  && cargo build --no-default-features --features "rwkv,rwkv-tokenizer" \
  && cargo clippy --no-default-features --features "rwkv,rwkv-tokenizer" -- -W clippy::pedantic \
  && cargo test --no-default-features --features "rwkv,rwkv-tokenizer" \
  && cargo build --no-default-features --features stoicheia \
  && cargo clippy --no-default-features --features stoicheia -- -W clippy::pedantic \
  && cargo test --no-default-features --features stoicheia --lib --test stoicheia_analysis --test validate_stoicheia \
  && cargo build --no-default-features --features "clt,transformer" \
  && cargo clippy --no-default-features --features "clt,transformer" -- -W clippy::pedantic \
  && cargo test --no-default-features --features "clt,transformer" --lib \
  && cargo build --no-default-features \
  && cargo build --no-default-features --features "transformer,rwkv,rwkv-tokenizer,clt,sae,stoicheia,probing"
```

First run: ~15–25 min cold compile. Incremental reruns much faster.

### 5. Push; wait for remote CI green

```bash
git push
```

Wait for both matrix entries (MSRV 1.88 and Stable) to report green in the
GitHub Actions UI. Do not proceed until both are ✓.

### 6. Dry-run the publish workflow locally

Mirrors [`.github/workflows/publish.yml`](../../.github/workflows/publish.yml),
ending with `cargo publish --dry-run` instead of the real publish. Differs
from step 4 by dropping the CLT lane (publish.yml doesn't have one) and adding
the dry-run publish command at the end:

```bash
cargo fmt --check \
  && cargo build --no-default-features --features transformer \
  && cargo clippy --no-default-features --features transformer -- -W clippy::pedantic \
  && cargo test --no-default-features --features transformer \
  && cargo build --no-default-features --features "rwkv,rwkv-tokenizer" \
  && cargo clippy --no-default-features --features "rwkv,rwkv-tokenizer" -- -W clippy::pedantic \
  && cargo test --no-default-features --features "rwkv,rwkv-tokenizer" \
  && cargo build --no-default-features --features stoicheia \
  && cargo clippy --no-default-features --features stoicheia -- -W clippy::pedantic \
  && cargo test --no-default-features --features stoicheia --lib --test stoicheia_analysis --test validate_stoicheia \
  && cargo build --no-default-features \
  && cargo build --no-default-features --features "transformer,rwkv,rwkv-tokenizer,clt,sae,stoicheia,probing" \
  && cargo publish --no-default-features --features transformer --dry-run
```

The final `cargo publish --dry-run` is what catches packaging issues the
lane checks cannot see: missing files in `[package] exclude` rules, metadata
validation, licence headers, simulated crates.io upload.

### 7. Tag and push

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

`publish.yml` fires on the tag match (`tags: ["v*", "!v*-*"]` — hyphenated
tags like `v0.1.9-plt` do **not** publish; they are git-level milestones).
The workflow runs the full lane gauntlet again on Ubuntu, then the real
`cargo publish` step — publishes to crates.io.

Monitor the workflow run. On success, the crate is live at
`https://crates.io/crates/candle-mi/X.Y.Z`.

## Why both dry-runs matter

- **Step 4 (pre-push CI dry-run)** is about not pushing a broken
  release-prep commit to `origin` at all. Remote CI is ~10 min of feedback
  latency; local is ~15–25 min of cold compile but gives immediate termination
  on first failure. On the release commit specifically (what the tag will
  point at) this matters — you do not want a "fix CI for vX.Y.Z" commit
  polluting `git log` right before the tag.
- **Step 6 (pre-tag publish dry-run)** catches things neither step 4 nor
  remote CI sees: `cargo publish --dry-run` builds a crate package under
  `[package] exclude` rules, validates metadata completeness, verifies
  licences, and simulates the crates.io upload. The standing rule that this
  step is non-skippable is recorded as
  [`feedback_dry_run_before_tag.md`](../../../../.claude/projects/c--Users-Eric-JACOPIN-Documents-Code-Source-candle-mi/memory/feedback_dry_run_before_tag.md).

## Drift check

The bash blocks in steps 4 and 6 are transcribed from the workflow YAML.
**Before running them, diff against the current workflow files** in case
steps have been added/removed since this doc was last updated:

```bash
grep -E "^\s+run: cargo" .github/workflows/ci.yml
grep -E "^\s+run: cargo" .github/workflows/publish.yml
```

If the output does not match the commands above verbatim, update this doc
**or** regenerate the bash blocks from the current YAML before running.

## Alternative: `act`

[`nektos/act`](https://github.com/nektos/act) runs `.github/workflows/*.yml`
literally in Docker, using the same `ubuntu-latest` image as GitHub Actions.
Highest-fidelity dry-run possible — catches Linux-specific issues that
Windows-local transcription misses. Downsides: requires Docker Desktop, has
quirks with `Swatinem/rust-cache@v2` and some GitHub-context-dependent
actions. Optional; the shell-transcription approach catches ~95% of issues.

## Cross-reference

- Pre-commit hygiene: [`CLAUDE.md`](../../CLAUDE.md) §Pre-commit Checks.
- Publish trigger rules (which tag names publish vs git-only milestones):
  [`.github/workflows/publish.yml`](../../.github/workflows/publish.yml)
  header comment. Tags `v0.1.9-plt`, `v1.0.0-rc.1`, etc. are excluded by the
  `!v*-*` filter.
- Dry-run rule origin: `feedback_dry_run_before_tag.md` (user memory).
