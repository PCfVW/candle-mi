# Validation, Testing & Documentation Audit — candle-mi v0.1.17

**Date:** 2026-07-01
**Scope:** All test validation (`tests/*.rs`, `src/**/#[cfg(test)]`), CI/preflight wiring (`.github/workflows/*.yml`, `scripts/preflight.ps1`), coding-convention compliance (`CONVENTIONS.md` vs `src/`), and documentation staleness (`README.md`, `ROADMAP.md`, `BACKENDS.md`, `HOOKS.md`, `CHANGELOG.md`, `docs/`, `design/`, `scripts/README.md`).
**Method:** Six parallel deep-read audits (one per area), each independently cross-checking claims against source, several with **empirical verification** — actually running `cargo test -- --ignored`, `cargo test --no-run`, and `cargo clippy` locally against real feature combinations rather than inferring from static reading alone. Total: ~40 tool-using sub-agent passes across the six areas.
**Relationship to prior audits:** `docs/audit/CONSISTENCY_AUDIT.md` (2026-03-06) and `docs/audit/examples_coverage_audit.md` (2026-03-08/09) predate ~9 releases of work (diffusion/MDLM/OthelloGpt, RWKV, CLT/PLT, SAE, stoicheia, quantized loading, the hypomnesis memory migration) and are treated here as **historical only** — none of their findings were re-verified or relied upon; this is a ground-up fresh audit.

## Executive summary

The engineering discipline in this codebase is genuinely strong where it's visible to the compiler: `CONVENTIONS.md` compliance is close to perfect (verified live with `cargo clippy --features transformer -- -W clippy::pedantic` → zero warnings), the `#[non_exhaustive]` policy is applied to 22 of 23 public enums (the one exception is explicitly and correctly justified), and no production-code `unwrap`/`expect`/`panic` escapes the deny-level lints anywhere.

The risk is concentrated **one layer up**: in what CI and the test suite actually *execute* versus what merely *exists and compiles*. The single most important finding, confirmed independently by three of the six audits and empirically reproduced today, is:

> **Nearly every test that compares candle-mi's output against an independent oracle (Python/PyTorch) is `#[ignore]`d, and there is no automated resurrection path anywhere in this repository** — no CI `--ignored` invocation, no cron, no `workflow_dispatch` step, no preflight tier. `grep -r "\-\-ignored"` across `.github/` and `scripts/` returns **zero matches**. These tests exist only to be run manually by a developer who remembers to do so before trusting a change.

Compounding this, several tests that are *not* marked `#[ignore]` (so they look like they run in CI) were **empirically shown today to vacuously pass** — they detect the absence of cached model weights, print `SKIP`, and `return Ok(())`, which `cargo test` reports as a pass. This means "the test suite is green" currently carries much less information than it appears to for the numeric-parity claims that back this crate's core value proposition (research-grade F32 fidelity to Python/PyTorch).

Documentation staleness is a secondary but real issue: `README.md`'s version banner is 5 patch releases behind, `ROADMAP.md` is ~9 releases and contradicts itself internally, and two `docs/roadmaps/PLAN-*.md` files show a stale "not started" status for work that (in one case) has actually shipped.

None of this indicates fabricated results, mocked oracles, or dishonesty — every genuine oracle-backed test that *was* found used real, independently-generated Python data (verified: real non-round F32 logits, real library imports for bnb/AWQ/GPTQ, real HF `transformers` forward passes for several files). The issue is entirely **process**: valuable checks exist but are disconnected from the safety net that would keep them true over time.

---

## 1. Test validation holes (highest priority)

### 1.1 Systemic: `#[ignore]`d oracle tests have no automated resurrection path — **High**

Every dedicated forward-parity oracle test in the transformer family (`validate_llama32_forward.rs`, `validate_gemma2_forward.rs`, `validate_phi3_mini_forward.rs`, `validate_mistral_7b_forward.rs`, `validate_qwen3_forward.rs`, `validate_qwen25_coder_forward.rs`, `validate_starcoder2_forward.rs`, `validate_deepseek_forward.rs`, `validate_longrope.rs`, `validate_bidirectional_forward.rs`, `validate_quantized_loading.rs`, plus `validate_clt.rs`, `validate_anacrousis.rs`, the live-CUDA cases in `validate_memory.rs`) is `#[test] #[ignore]`. A repo-wide grep for `--ignored` across `.github/workflows/*.yml` and `scripts/preflight.ps1` returns **zero hits**. The weekly Monday cron (`ci.yml:8-11`) reruns the identical `check` job — still no `--ignored`.

This is disclosed behavior (each file's doc comment says "Run: ... -- --ignored"), not a hidden bug, but it means the tests that actually verify candle-mi's central claim — F32 parity with Python/PyTorch to within a justified tolerance — are **100% manual**, with no CI, cron, or scheduled safety net if developer discipline lapses. `cargo test` reporting green tells you nothing about whether any of these ever ran on a given machine.

**Recommendation:** add a `workflow_dispatch`-triggered or scheduled (e.g. monthly) CI job that runs `--ignored` against cached/downloadable gated models, even if it can't run on every push. At minimum, add a `RESURRECTION.md`/CI comment tracking "last verified" dates per oracle test so staleness is visible.

### 1.2 `validate_models.rs` is the *only* transformer test that runs by default — and it's shape/top-k only, not numeric — **High**

Full read of `tests/validate_models.rs` (661 lines) confirms it loads **no reference JSON anywhere**. Every assertion is either `assert_in_top_k(&top5, "Paris", ...)` or a shape check (`dims3()`). The file's own docstring is honest that this is a smoke test; sibling per-family files' docstrings explicitly note that "Paris in top-5" could pass with "a subtly-wrong RoPE" or "subtly-wrong soft-capping" — the project's own authors flagged this class of test as inadequate and built the dedicated `#[ignore]`d replacements. But `validate_models.rs` remains the **only** test that CI actually executes for Llama/Gemma2/Mistral/StarCoder2/Qwen2.5-Coder/Phi-3-mini (per §1.1, the numeric siblings never run).

Additionally, every test in this file does `find_snapshot(...) → None ⇒ eprintln!("SKIP..."); return;` — on a fresh GitHub-hosted `ubuntu-latest` runner with no HF cache and no `HF_TOKEN`, the gated models (Llama-3.2-1B, Gemma-2-2b, Mistral-7B) silently skip even in this "always-on" file. Only ungated, cached-locally models exercise any assertions at all on CI infrastructure — and CI has no cache-seeding step (confirmed: no `hf-fetch-model` invocation anywhere in `ci.yml`).

**Recommendation:** either seed a small ungated model in CI (there's precedent — MDLM-owt/Othello are small enough) so at least one model gets non-trivial live coverage, or make the CI job print an explicit end-of-run summary of which tests SKIPped vs ran, so "green" is legible.

### 1.3 RWKV: final-token-only comparison, loose tolerance, and empirically-confirmed vacuous pass — **High**

`tests/validate_rwkv6.rs` / `validate_rwkv7.rs` are **not** `#[ignore]`d (they look mandatory) and CI does invoke `cargo test --features "rwkv,rwkv-tokenizer"` (`ci.yml:63-64`). But:
- The oracle sequence is only **7 tokens**; only the **last-position** logits are checked. `rwkv{6,7}_hook_capture_state` assert **shapes only** on `RwkvState`/`RwkvDecay`, never values — no intermediate-timestep numeric check of the WKV recurrence/decay exists anywhere.
- Tolerance is `logit_diff < 1.0` absolute against a reference logit of ~4.8–7.6 (≈15–20% relative), loosening further to `< 2.0` on GPU/BF16 — loose enough to really only confirm the argmax token, not the underlying math.
- **Empirically confirmed today**: running `cargo test --features "rwkv,rwkv-tokenizer" --test validate_rwkv6` locally (no cached `RWKV/v6-Finch-1B6-HF`) reports **5 passed in 8.73s** — far too fast for a real 1.6B-parameter forward pass, proving every test (including the non-ignored ones) took the `SKIP`/`return` branch. Since CI has no cache-seeding step either, CI almost certainly passes this lane vacuously too.

A bug in the WKV recurrence's accumulation order, decay sign, or LoRA-decay computation could still produce a coincidentally-close final logit at 7 tokens and pass — and currently nothing would even reach that check on CI.

### 1.4 Bidirectional-attention forward-parity test is completely unwired from CI/preflight — **High**

`tests/validate_bidirectional_forward.rs` is the only test validating that "bidirectional attention provably differs from causal at early positions" — the core Stage-3 (masked-diffusion-via-decoder) correctness claim (Dream/`a2d-qwen2`). It is **absent from both `ci.yml` and `preflight.ps1`** — not even a `--no-run` compile check, unlike its sibling `validate_bidirectional_sampler.rs` which does get a compile-check (`ci.yml:104-105`). Verified this compiles cleanly today (`cargo test --features "transformer,diffusion" --test validate_bidirectional_forward --no-run`, 25s, no errors) — the gap is pure CI wiring, not a build break. `validate_bidirectional_sampler.rs` itself has no oracle logit comparison (structural/determinism checks only), so **no CI-exercised numeric check of bidirectional attention correctness exists at all** currently.

Also flagged: `a2d-qwen3` is claimed in project memory as validated ("external fp32 oracle parity 2.61e-4") but only `a2d-qwen2` appears in any test file or `scripts/` reference JSON — no qwen3 test or fixture exists.

### 1.5 `validate_clt.rs` has zero comparison to any external oracle — **High**

All 10 tests in `tests/validate_clt.rs` (1479 lines) are self-consistency checks against the model's own live forward pass: sorted-descending activations, arbitrary thresholds (`l2_dist > 0.01`/`> 1.0`, `jaccard < 0.8`, `concentration_ratio > 1.2`, `rank < 3`) with no external derivation. `scripts/clt_position_sweep_reference.json` exists on disk (built from a real HF forward pass) but is **never read by any Rust test** — confirmed via grep, zero matches; it is an orphaned oracle. A sign error or wrong-layer indexing bug in CLT feature injection could pass every check in this file, and this file underlies the crate's headline "attribution graph" / planning-signal MI claims (Figure 13 replication, prolepsis work).

By contrast, `validate_plt.rs`, `validate_plt_gemma.rs`, and `validate_clt_qwen3.rs` **do** load checked-in oracle JSON and assert `abs diff < 1e-4` + exact top-k index match — real, if narrower, validation (see caveat in §1.10).

### 1.6 Bench files assert no correctness or performance thresholds — **High**

`tests/bench_hook_overhead.rs` and `tests/bench_hook_diagnostic.rs` are registered as `[[test]]`s and do run in CI's `transformer` lane (`ci.yml:54-55`, no `--skip`), but contain only presence checks (`result.get(&HookPoint::Embed).is_some()`) — never a timing threshold or correctness bound. A hook-overhead regression (e.g. +50% GPU overhead from a refactor) would produce **zero test failures**, only a human reading `--nocapture` output would notice. Both tests also silently `return` with no assertion at all if the model isn't cached — on a fresh CI checkout this is a pure no-op reported as green.

### 1.7 `src/stoicheia/ablation.rs` and `tasks.rs` — core math with weak/absent unit coverage — **High**

- `ablation.rs::full_ablation_near_chance` (despite its name) only asserts `results.len() == 2` — no accuracy or delta assertion at all. A sign flip in the ablation delta/interaction-score computation would pass silently.
- `tasks.rs` (`longest_cycle` graph-traversal logic, used by the stoicheia surprise/task scoring) has **no `#[cfg(test)]` block whatsoever** — zero unit coverage for nontrivial logic.
- `probing.rs::probe_runs_on_tiny_model` only checks `correlation >= 0.0`, never which `NeuronRole` is returned, despite the tiny model having a fully known functional signature — the core `best_probe_match` selection logic is unverified.

These are the modules computing the actual "surprise"/ablation/probing statistics that stoicheia's MI claims rest on.

### 1.8 SAE test silently reports "passed" when its reference fixture is missing — **Medium**

`tests/validate_sae.rs::sae_vs_python_reference` (the one test in the file with a real external oracle — genuine HF `transformers.AutoModelForCausalLM` forward pass) is itself `#[test] #[ignore]`-gated (like every other oracle test in §1.1), so it does not run in CI or on a plain `cargo test` either way. But *when a developer does run it* with `--ignored`, it prints `"SKIP..."` and `return`s if `scripts/sae_reference.json` is absent (lines 392–398) — **reported by `cargo test` as passed**, not skipped or failed. This is inconsistent with the sibling PLT/CLT-Qwen3 tests, which `.expect()`/`panic!` (hard fail) on a missing reference JSON — the stricter, more honest gate. A developer who runs `--ignored` locally without first running `scripts/sae_validation.py` gets a false-green on this specific test with zero validation performed, while the PLT/CLT-Qwen3 siblings would correctly crash and tell them what's missing. (Severity lowered from the original Medium-High pass since the test's `#[ignore]` gating means this only misleads a developer actively running the ignored suite, not CI.)

Also: SAE's magnitude tolerances are notably loose relative to CLT/PLT's `1e-4` bar — active-count diff `≤10`, top-feature match `≥50%`, MSE ratio within **10×** — a materially broken JumpReLU gate or bias-handling bug could hide inside these bounds.

### 1.9 Feature-combination CI gaps — **High/Medium**

Cross-referencing `Cargo.toml`'s `[features]`/`[[test]]` table against every `cargo build`/`clippy`/`test` invocation in `ci.yml`:

- **`quantized` has zero CI presence** — no build, no clippy, no test, not even in the combined "all software features" build (`ci.yml:134`, which lists `transformer,rwkv,rwkv-tokenizer,diffusion,clt,sae,stoicheia,probing` — `quantized` is conspicuously absent). `tests/validate_quantized_loading.rs` is never compiled by CI at all. Regressions in the anamnesis bnb/NF4/AWQ/GPTQ dequant path (a path candle-mi's own project memory already flags as having broken once from an upstream signature change) can merge to `main` completely undetected. **Severity: High.**
- **No standalone clippy lane for `sae`** — only appears bundled into the combined build (no clippy on that combo either). `validate_sae.rs` and `quick_start_sae` are never clippy'd in isolation. **Medium.**
- **No standalone clippy/build lane for `mmap`, `memory-debug`, `probing`** — `probing` only appears in the bundled build (no clippy); `mmap` (a feature CLAUDE.md tells users to "always" enable for examples) and `memory-debug` don't appear anywhere in `ci.yml`. **Medium.**
- **`clt,sae,transformer` combo** (needed by `validate_plt_gemma`, which requires all three) is **never compiled in any CI test-building step at all**, let alone run. The only place `sae` appears in `ci.yml` is the line-134 combined step, which is a plain `cargo build` (not `cargo test`/`--tests`) — that doesn't compile test binaries, so `validate_plt_gemma` never even reaches "compiles but unused"; it simply never builds on CI. **Medium.**

### 1.10 Doctest coverage gap for `clt`/`sae`/`rwkv` — **High**

Exactly one `--doc` lane exists in all of CI: `cargo test --no-default-features --features "transformer,memory" --doc` (`ci.yml:124-125`, added specifically for the memory module per commit `60bdb11`). Runnable (`no_run`/bare-fence) doc examples requiring other features are **not** covered by any lane:
- `src/clt/mod.rs:315-328` (needs `clt`) — uncovered.
- `src/sae/mod.rs:274-287` (needs `sae`) — uncovered.
- `src/rwkv/config.rs:11-21` (needs `rwkv`) — uncovered.

A `clt`- or `sae`-only doctest break (e.g. a public API signature change on `CrossLayerTranscoder::open`) is invisible to CI.

### 1.11 PLT/CLT-Qwen3 Python oracles share the algorithm under test, not just the data — **Medium**

`validate_plt.rs`, `validate_plt_gemma.rs`, and `validate_clt_qwen3.rs` all load a real oracle JSON generated from **real downloaded pretrained weights**, which is good — but each generator script's own docstring discloses "from-first-principles... NO circuit-tracer," meaning the Python side re-derives the *same* encoder formula (`ReLU(W_enc @ x + b_enc)` / thresholded variant) that `src/clt/mod.rs` implements, rather than using an independent reference library (`circuit-tracer`, `sae_lens`). This catches cross-language transcription bugs (wrong axis, wrong bias-add order, transpose errors) but **not** a shared conceptual misunderstanding of the formula itself, since both sides would encode the same mistake identically. `validate_sae.rs`'s oracle is comparatively stronger here — its Python side drives real HF `transformers.AutoModelForCausalLM` for the input activations, though the SAE encode/decode math itself is still hand-rolled to match.

### 1.12 Memory module: Metal path entirely untested — **Medium**

`tests/validate_memory.rs` has exactly 3 tests as project memory claims, confirmed accurate. `cpu_snapshot_is_ram_only` (always runs) validates only the *absence* of VRAM fields on CPU. The two tests that do real ground-truth validation (`cuda_snapshot_is_sane`, `cuda_allocation_is_visible_in_vram` — the "512 MiB alloc → exact delta" check) are `#[ignore]`+CUDA-gated and silently no-op (reported as passed) without a GPU. Despite `src/memory.rs` explicitly supporting a Metal code path (`device.is_cuda() || device.is_metal()`) and `hypomnesis`'s `metal` feature being enabled, **no live test exercises the Metal path at all** — a macOS-specific bug in the flatten logic would never be caught, especially since CI runs on `ubuntu-latest` only.

### 1.13 Minor: asymmetric rigor across the per-family transformer tests — **Low**

`validate_llama32_forward.rs`, `validate_deepseek_forward.rs`, and `validate_longrope.rs` cross-check family-specific config fields (`rope_scaling`, `attention_factor`) against the independently-recorded value in the oracle JSON. `validate_gemma2_forward.rs`, `validate_phi3_mini_forward.rs`, `validate_qwen3_forward.rs`, `validate_qwen25_coder_forward.rs`, and `validate_starcoder2_forward.rs` only assert the Rust-parsed config against a hardcoded expected value (self-consistency, not cross-checked against the independently-recorded oracle field). Not wrong, just inconsistent rigor within the same test family — low-cost to fix by porting the stronger pattern.

### What's genuinely solid (for balance)

- Oracle independence was spot-checked and confirmed genuine everywhere a real oracle exists: real non-round F32 logits in reference JSONs, real `torch`/`transformers` imports with recorded library versions, real `bitsandbytes`/`AutoAWQ`/`gptqmodel` imports for the three quantization schemes, deterministic seeding (`torch.manual_seed(0)`, `use_deterministic_algorithms(True)`).
- Every tolerance found — even the loosest (BF16 `0.1`, bnb `1.5`) — carries an explicit, checkable rationale tied to known precision limits (BF16 mantissa width, F32-vs-BF16 dequant gap), not an arbitrary loosening to make a flaky test pass.
- `tests/validate_stoicheia.rs` (empirically run: 4/4 pass, no `#[ignore]`, checked-in fixtures) is the most rigorously validated integration test in the repo — genuine oracle comparison at `1e-4`/`1e-2` tolerance, always runs.
- The `#[non_exhaustive]` policy, `unwrap`/`expect`/`panic` containment, and `unsafe`-scoping conventions are essentially perfectly followed (see §3).
- `validate_quantized_loading.rs`'s tolerance design is a standout: the loose magnitude bar is explicitly demoted to a secondary check, with **exact top-1 token match** as the real, unconditionally-enforced correctness gate — an intellectually honest design.

---

## 2. CI / workflow consistency

| Gap | Where | Severity |
|---|---|---|
| `quantized` feature: no build/clippy/test anywhere in CI | `ci.yml` (absent from line 134's bundle) | High |
| `clt`+`sae` doc examples never doctested | only `transformer,memory --doc` lane exists (`ci.yml:124-125`) | High |
| `sae`, `probing`, `mmap`, `memory-debug`: no standalone clippy lane | `ci.yml` | Medium |
| `validate_plt_gemma` (needs `clt,sae,transformer`) never compiles in any CI test-building step — `sae` only appears in a plain `cargo build`, not `cargo test` | `ci.yml` | Medium |
| `publish.yml` has no explicit dependency (`needs`/`workflow_run`) on `ci.yml`'s success | `publish.yml` | Medium |
| `publish.yml` has no `cargo publish --dry-run` step (relies on a documented manual pre-tag step) | `publish.yml` | Low |
| `rwkv/config.rs` doctest (needs `rwkv`) uncovered by any `--doc` lane | `ci.yml` | Medium |

Confirmed correct / not stale: the MSRV(1.88)+stable matrix genuinely runs the full step list on both toolchains (`ci.yml:21-27`, not lint-only on MSRV as some projects do); `scripts/preflight.ps1`'s three tiers (default/`-Ci`/`-Full`) match CLAUDE.md's description exactly, parameter names and logic both verified; `publish.yml`'s tag trigger is the correct single-list-with-`!`-exclusion form (`tags: ["v*", "!v*-*"]`), not the buggy `tags-ignore` variant that caused a prior incident; `Cargo.toml`'s version (`0.1.17`) and `CHANGELOG.md`'s latest dated entry agree exactly, with an empty (genuinely up-to-date) `[Unreleased]` section.

`publish.yml` does re-run its own full check sequence independently before publishing (not a bare `cargo publish`), which mitigates most of the risk from the gaps above — but since it's a near-verbatim copy of `ci.yml`'s steps, it inherits every gap listed in the table (a `quantized`-only regression could publish even though `ci.yml` never would have caught it either).

---

## 3. CONVENTIONS.md compliance

**Overall: excellent.** Live-verified with `cargo clippy --features transformer -- -W clippy::pedantic` → zero warnings, confirming annotations correspond to real suppressed lints rather than decoration. Across ~185 `#[allow(clippy::...)]` sites and ~143 `as`-cast sites sampled, essentially all carry the required `// CAST:`, `// INDEX:`, `// CONTIGUOUS:`, `// EXHAUSTIVE:`, or `// SAFETY:` annotation (inline or via a function-level allow covering the whole body). Only three Low-severity nits surfaced:

1. **Two uncommented casts**: `src/interp/intervention.rs:997` and `:1008` (`.len() as f32`) have no `// CAST:` comment and no `#[allow]` (clippy doesn't flag these specific small-`.len()` sites, so it's a documentation gap, not a lint bypass).
2. **Two `.contiguous()` sites with prose-only, untagged comments**: `src/transformer/rope.rs:322` and `src/diffusion/rope.rs:95` — substance is present ("rope() expects contiguous input"), just not the literal `// CONTIGUOUS:` tag. (Two other sites flagged in an earlier pass — `src/clt/mod.rs:978` and `src/util/pca.rs:79` — were rechecked and found to already carry the literal `// CONTIGUOUS:` tag; they are compliant and are not nits.)
3. **`SAFETY:` comment placement**: in `src/memory.rs`, the `// SAFETY:`-equivalent explanatory comment sits one block above the outer `#[allow(unsafe_code)] { ... }` scope rather than immediately above the inner `unsafe { }` — unambiguous given the file's small size, but looser than CONVENTIONS.md's "immediately before" wording.

Confirmed fully compliant: both `unsafe` sites (`src/backend.rs:906` mmap loader, `src/memory.rs:248` CUDA pool-trim) match CLAUDE.md's documented inventory exactly, each has a `// SAFETY:` comment; 22 of 23 public enums carry `#[non_exhaustive]`, and the sole exception (`Norm` in `src/transformer/norm.rs`) is explicitly and correctly justified via `#[allow(clippy::exhaustive_enums)]` + an `// EXHAUSTIVE:` comment, exactly matching the documented alternative path in `CONVENTIONS.md`; zero production-code `unwrap`/`expect`/`panic!` — every hit resolves to a `#[cfg(test)] mod tests` block, a whole-file `#[cfg(test)]` module (`registration_guard.rs`), or a deterministic (non-`no_run`) doctest on literal inputs that cannot actually panic.

---

## 4. Documentation staleness

| Document | Finding | Severity |
|---|---|---|
| `README.md:13` | Version banner reads "v0.1.12" — actual is **v0.1.17** (5 patch releases stale), while the rest of the same file already documents v0.1.14+ features (MDLM, OthelloGpt, hypomnesis memory) | High (visibility) |
| `ROADMAP.md` (header) | Dated Feb 19 / last-updated Apr 13, 2026; says "published ... as v0.1.8" — ~9 releases stale | High |
| `ROADMAP.md` §6.1 | Feature-gate list omits `diffusion`, `stoicheia`, `quantized` (all real `Cargo.toml` features) | Medium |
| `ROADMAP.md` Phase 6 checklist | Unchecked `[ ] Add qwen3 auto-config support`, while the **same document's** §3.3 table already lists Qwen3 as validated/exact-parity — internal self-contradiction | Medium |
| `docs/roadmaps/` (whole directory) | 11 files, **no index/README** distinguishing current vs. historical; root `README.md` links only 1 of the 11 (`diffusion-lm-roadmap.md`; its other roadmap link is the separate top-level `ROADMAP.md`, not a member of this directory) | Medium |
| `docs/roadmaps/PLAN-GRIDWORLD-PROLEPSIS.md` | Status table says "⏳ Not started" (dated 2026-06-04) — but CHANGELOG.md and `docs/experiments/gridworld-prolepsis/` confirm this experiment **actually ran and completed** (negative result, correctly attributed elsewhere) — actively misleading to a new reader | High |
| `docs/roadmaps/PLAN-GEOMETRIC-CALCULATOR.md` | Status table says "⏳ Not started" targeting v0.1.13; 3 releases later, no `docs/experiments/geometric-calculator/`, no CHANGELOG mention — plan appears abandoned with no signal of that | Medium |
| `docs/roadmaps/candle_mi_v019_roadmap{,_V2,_V3}.md` | Three superseding variants, none marked "superseded by," none point to what actually shipped (v0.1.9–v0.1.11) | Medium |
| `design/intervention-api.md` | Marked "Status: Implemented" but describes a unified `ForwardConfig::new().capture(...).intervene(...)` API that **does not exist** (`ForwardConfig` — zero hits in `src/`). Actual shipped API is `forward(&self, input_ids, hooks: &HookSpec)` + separate intervention specs in `src/interp/intervention.rs` | High |
| `design/rwkv7-effective-attention.md` | Marked "Status: Implemented" but body still reads as an open research question ("is there a closed-form...", "defer: ship without it") — doesn't record that `compute_effective_attention_v7` was built and shipped (validated to 6 decimal places per project memory) | Medium |
| `design/hook-system.md` | `HookPoint` enum listing omits `RwkvEffectiveAttn(usize)` (exists in `src/hooks.rs`, fully wired) | Low |
| `design/candle-version.md` | Recommends exact-pin `candle-core = "=0.9"`; actual `Cargo.toml` uses caret range `"0.9"` — implemented policy looser than documented | Low |
| `scripts/README.md` | Accurate for everything through ~v0.1.12; silently missing ~20 newer scripts (quantized validation ×3, bidirectional/MDLM/Othello validation, clt_qwen3, qwen3, deepseek, longrope, gridworld/means-ends generators) — omission, not a false claim | Low |
| `CHANGELOG.md` | Content is accurate and complete for every version 0.1.9–0.1.17 (verified); only defect is a dangling version-link footer stuck at `[0.1.8]` — cosmetic | Low |

Confirmed **not** stale (checked, no issues): `BACKENDS.md`, `HOOKS.md`, `CONTRIBUTING.md`, `CONVENTIONS.md`, `docs/adding-a-model.md`, `docs/hook-architecture-diagnostic.md`, `design/error-handling.md` (aside from a snippet needing the two newer `MIError` variants and `#[non_exhaustive]` shown), `design/migrate-npz-to-anamnesis.md`, `docs/roadmaps/diffusion-lm-roadmap.md` (the model example — explicit DONE/DEFERRED markers matching ground truth exactly), `docs/roadmaps/release-sequence.md`, all three `docs/v0.1.1{1,2}-*.md` handoff notes (correctly self-described as historical, consistent with what CHANGELOG confirms shipped).

---

## Prioritized recommendations

1. **Give the `#[ignore]`d oracle suite a resurrection path.** Even a monthly `workflow_dispatch`/scheduled job running `--ignored` against whatever's cache-able (ungated models, or gated ones via a repo secret token) would convert "trust the developer remembered" into an actual safety net. This is the single highest-leverage fix (§1.1–1.4).
2. **Close the `quantized` CI gap** — add build+clippy+test lanes; this is the only feature with literally zero CI presence (§1.9, §2).
3. **Strengthen `validate_clt.rs`** with at least one real oracle comparison (the orphaned `clt_position_sweep_reference.json` is sitting right there, already computed) — this is the test suite backing the crate's flagship attribution-graph/planning-signal claims (§1.5).
4. **Add assertions to the bench files**, or rename/relocate them out of `[[test]]` so their lack of correctness checks isn't mistaken for coverage (§1.6).
5. **Fix the `validate_sae.rs` silent-pass-on-missing-fixture** to match the hard-fail pattern already used by its PLT/CLT-Qwen3 siblings (§1.8).
6. **Add unit tests for `stoicheia::ablation.rs`'s delta/interaction-score math and `tasks.rs::longest_cycle`** — currently the least-tested non-trivial logic in the crate (§1.7).
7. **One-line fixes, high visibility-to-effort ratio:** bump `README.md`'s version banner; add a short "supersedes/superseded by" or DONE/STALE banner to the `docs/roadmaps/PLAN-*.md` files (especially `PLAN-GRIDWORLD-PROLEPSIS.md`, which is actively misleading); refresh `ROADMAP.md`'s header/feature list or mark it explicitly historical in favor of CHANGELOG.md as the status source of truth.
8. **Correct or re-scope `design/intervention-api.md` and `design/rwkv7-effective-attention.md`** to describe what was actually shipped, since both currently mislead a reader into thinking a different (unbuilt or unresolved) design is current.

No High-severity CONVENTIONS.md violations were found — that discipline is worth preserving as-is; the effort is better spent on the CI/test-resurrection items above.

---

## Verification addendum (2026-07-01, same-day)

This report was independently fact-checked section-by-section against the live repository: every file:line reference, quoted tolerance/count, and behavioral claim (SKIP branches, `#[ignore]` gating, grep results) was re-derived from source rather than taken on trust, and the CONVENTIONS.md clippy claim was re-run live (`cargo clippy --features transformer -- -W clippy::pedantic` → confirmed zero warnings today).

**Corrections applied as a result** (all reflected in the sections above):
- §1.2: `validate_models.rs` is 661 lines, not 598.
- §1.5: `validate_clt.rs` has 10 `#[test]` functions (1479 lines), not 8/1480.
- §1.8: `sae_vs_python_reference` is itself `#[ignore]`-gated — the silent-pass-on-missing-fixture bug only misleads a developer who manually runs `--ignored` without first generating the fixture, not a default CI run. Severity note added; the underlying inconsistency-with-siblings finding still holds.
- §2 table: the `validate_plt_gemma` finding was corrected from "compiles but never invoked as a test" to "never compiles in any CI test-building step at all" — `sae` (one of its three required features) only appears in `ci.yml` inside a plain `cargo build`, which doesn't build test binaries.
- §2 table: the `docs/roadmaps/` index finding was corrected from "README.md links only 2 of the 11" to "links only 1 of the 11" (the second cited link is the separate top-level `ROADMAP.md`, not a member of that directory).
- §3: the enum ratio was corrected from "23/24" to "22 of 23" (a clean recursive grep of `pub enum` in `src/` returns exactly 23 declarations).
- §3: two of the four `.contiguous()` sites originally flagged as missing the `// CONTIGUOUS:` tag — `src/clt/mod.rs:978` and `src/util/pca.rs:79` — were rechecked and found to already carry the tag; they were removed from the nit list. The other two (`src/transformer/rope.rs:322`, `src/diffusion/rope.rs:95`) were confirmed genuinely untagged.

**Everything else held up.** All remaining file:line citations, quoted tolerances, "confirmed correct" claims (MSRV/stable CI matrix, preflight tiers, publish.yml tag trigger, Cargo.toml/CHANGELOG version match, `ForwardConfig` non-existence, the `PLAN-GRIDWORLD-PROLEPSIS.md` vs. CHANGELOG contradiction, the ROADMAP.md Qwen3 self-contradiction, and every other doc-staleness row) were independently confirmed against source. No finding's substance or severity rating changed as a result of this pass — only line/count precision and two framing nuances (§1.8, §2) were sharpened.
