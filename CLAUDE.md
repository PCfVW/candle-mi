# Claude Code Instructions

## Coding Conventions

Always apply the rules in `CONVENTIONS.md` to all code changes. Every annotation pattern, doc-comment rule, and style rule in that file is mandatory.

Every `.rs` file must start with `// SPDX-License-Identifier: MIT OR Apache-2.0` as its first line.

## Version Control

- **Commit directly to the current branch (normally `main`). Do NOT create a new branch before committing.** This explicitly overrides the default "if on the default branch, branch first" behavior — in this repo, committing onto `main` is correct and expected. Create a branch only if the user explicitly asks for one (e.g. for a PR).
- Commit and push **only when the user asks** — never preemptively. (This part of the default is correct; keep it.)

## Pre-commit Checks

Before every commit, run and fix any issues from:
1. `cargo update -p hf-fetch-model` — pick up the latest compatible patch release
2. `cargo fmt`
3. `cargo clippy --all-targets --all-features -- -D warnings`
4. `cargo test`
5. Update `CHANGELOG.md` — add a bullet under the `[Unreleased]` section for any user-visible change (new feature, fix, breaking change). Follow [Keep a Changelog](https://keepachangelog.com/) categories: Added, Changed, Fixed, Removed.

CI runs clippy separately for each backend feature. Before pushing, also run clippy with each feature flag individually:
- `cargo clippy --features transformer -- -W clippy::pedantic`
- `cargo clippy --features rwkv -- -W clippy::pedantic`

Checking only `--all-features` will miss lint errors that appear under a single feature flag.

Before every push, run `./scripts/preflight.ps1`. It freshens the toolchains (`rustup update stable`, and ensures the MSRV `1.88` toolchain) so local lints match CI's rolling stable — a dry-run on a stale compiler can pass while CI fails on a newer lint (this is how `clippy::suboptimal_flops` from Rust 1.96 broke a clean `main`; the same lint, on 1.88, also broke a push when preflight didn't yet run the MSRV lane).

Preflight is **tiered** — CI runs a `1.88` (MSRV) + `stable` matrix, and the default fast path does not fully mirror both:
- **Default** (`./scripts/preflight.ps1`): full **stable** mirror (every CI lane on stable) **plus** MSRV `1.88` fmt + clippy. The MSRV clippy lanes catch version-specific lints/compile errors (the gap that bit us) cheaply.
- **`-Ci`** (`./scripts/preflight.ps1 -Ci`): the **full** both-toolchain mirror — every CI step on **both** `1.88` and `stable`. This is the run that literally means "green preflight = green CI"; use it before important pushes and after any MSRV-sensitive change.
- **`-Full`**: also runs the `bench_hook_*` CPU benches (composes with `-Ci`). These **skip on CI** (the gated `Llama-3.2-1B` isn't cached on runners), so they are not part of "green CI"; run `-Full` only when adding a new model family — the change that can shift the benchmarked forward/hook paths.

## Releasing

Cutting a release (a `vMAJOR.MINOR.PATCH` tag → crates.io publish):

1. **Verify green first.** Run `./scripts/preflight.ps1 -Ci` (both toolchains) and `cargo publish --dry-run`; re-run `./scripts/resurrect.ps1` after any numeric-path change. Bump `version` in **both** `Cargo.toml` and `Cargo.lock` in the same commit (`publish.yml` fails on a dirty lockfile), promote the CHANGELOG `[Unreleased]` section to `## [X.Y.Z] - <date>`, and bump the README version banner.
2. **Push, wait for green CI, then tag.** The tag is what fires `publish.yml` → crates.io (irreversible) — tag only after CI is green on the release commit. Real release tags are `vMAJOR.MINOR.PATCH` only; hyphenated tags (`v0.1.9-plt`) are milestones that do NOT publish.
3. **Cut the GitHub Release — AFTER the crates.io Publish workflow is green**, never before (the crate must actually be live). Use a hand-authored narrative body (title + "In the crate" / "Experiments" / "Verified before tagging" — the v0.1.18/v0.1.19 house style), NOT a raw changelog dump. `scripts/release-notes.ps1 -Version X.Y.Z -Theme "..."` scaffolds it from the CHANGELOG section; edit into prose, then `gh release create vX.Y.Z --title "..." --notes-file <f> --verify-tag --latest`. GitHub Releases are for real `vMAJOR.MINOR.PATCH` tags only (mirrors the publish trigger).

## Oracle-suite resurrection

Every oracle/parity test is `#[ignore]`d (needs cached — often gated — models, and many need a ≥16 GiB GPU), so CI never runs them and "green CI" says nothing about numeric parity. `scripts/resurrect.ps1` runs that suite **locally** (where the models are cached) and stamps `RESURRECTION.md` with a per-test "last verified" record. Tiers: `-Quick` (~5 min ungated-CPU smoke), default (~40–50 min, all but the Mistral-7B CPU forward + anacrousis), `-Full` (~1.5–3 h, everything). Run it periodically and after any change to a forward/CLT/SAE/quantized numeric path; commit the refreshed `RESURRECTION.md`.

Selecting entries, so a single refresh costs a minute rather than the full tier:
- `-List` prints the number/slug map plus each entry's tier and last-verified date. Start here.
- `-Only <tokens>` / `-Skip <tokens>` take 1-based numbers **or** stable slugs (`-Only longrope`, `-Only 12`, `-Only clt,sae`, `-Skip longrope`). Prefer slugs in anything you write down: inserting an entry renumbers everything after it.
- `-Skip longrope` is the common shortcut: that one entry is ~15 min of the ~44 min default tier (Phi-3.5-mini at F32 spills ~8.8 GiB), so skipping it gives ~28 min whenever the VRAM-spill path is not what changed.
- A partial run stamps `partial (N of 20: …)`, never a tier name, so a three-entry run can never later read as full coverage.

**A default-off feature that no entry enables cannot invalidate the suite.** `training` gates `src/optim.rs` entirely and appears nowhere in `resurrect.ps1`, so adding it after a green run needs no re-run. Check the same way: grep the feature name in `resurrect.ps1` and confirm the module is fully `cfg`-gated.

## Shell Environment

The user runs PowerShell on Windows. Use PowerShell syntax for all suggested commands:
- Use `$env:VAR="value";` instead of `VAR=value` for environment variables
- Use semicolons to chain commands, not `&&`
- Use forward slashes in paths when running Rust/cargo commands
