# Validation of the 2026-07-01 Audit — Confirmed Findings

**Companion to:** [`VALIDATION_AUDIT_2026-07-01.md`](VALIDATION_AUDIT_2026-07-01.md)
**Verification date:** 2026-07-01
**Purpose:** An independent, adversarial re-derivation of every claim in the source audit — file:line references, grep results, counts, tolerances, `#[ignore]` gating, SKIP-branch behavior, and doc-staleness rows — checked against the live repository from scratch (no trust in the audit's own text or its self-described "verification addendum").
**Method:** Five parallel deep-read verification passes (one per audit cluster: §1.1–1.4, §1.5–1.9, §1.10–1.13 + §2, §3, §4), each instructed to return CONFIRMED / REFUTED / PARTIAL with quoted evidence, plus direct spot-checks by the coordinator on the highest-stakes and every disputed claim.

## Bottom line

The audit is **substantially accurate and can be trusted as a work list.** Of ~50 discrete verifiable claims, exactly **three needed correction, and all three are narrow**: one Low-severity finding (§1.13) is wrong as stated and should be dropped or rewritten; one Low nit (§3 `.contiguous()`) is over-counted (two sites → one; the finding itself still stands for the one real site); one count ("~143 as-casts") is an imprecise ballpark. Every other finding survived intact. **None of the High/Medium findings — the ones that drive the recommendations — were weakened.** The central thesis (oracle-parity tests are `#[ignore]`d with no automated resurrection path, and several non-ignored tests pass vacuously via silent cache-miss SKIP branches) is fully confirmed.

---

## Corrections to the source audit (apply these)

### C1. §1.13 — REFUTED as stated (Low severity). The per-family taxonomy is incorrect.

The audit splits the eight per-family forward tests into two groups:
- "Group A" (llama32, deepseek, longrope) — *claimed* to cross-check a family-specific rope field against the **oracle JSON's** independently-recorded value.
- "Group B" (gemma2, phi3, qwen3, qwen25, starcoder2) — *claimed* to "only assert the Rust-parsed config against a **hardcoded** expected value (self-consistency, not cross-checked against the oracle field)."

Both halves are wrong, verified directly against source:

- **"Group B" tests DO cross-check core config against the oracle.** e.g. `tests/validate_gemma2_forward.rs:142-145` reads `reference["hidden_size"]`, `reference["num_layers"]`, `reference["vocab_size"]`, `reference["head_dim"]` from the oracle JSON and asserts the Rust-parsed config against those oracle values (`:167-170`). The same pattern holds for phi3 (`:140-143/165-168`), qwen3 (`:126-129/161-164`), qwen25 (`:139-142/163-166`), starcoder2 (`:138-141/163-166`). They are **not** hardcoded-only.
- **"Group A" llama32/deepseek assert their rope field against a HARDCODED literal, not the oracle.** `tests/validate_llama32_forward.rs:148-157` asserts `config.rope_scaling == Some(RopeScaling::Llama3 { factor: 32.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_position_embeddings: 8192 })` — a hardcoded literal, not a value read from the oracle JSON. `tests/validate_deepseek_forward.rs:160-163` likewise hardcodes `Some(RopeScaling::Linear { factor: 4.0 })`.
- **The only test that genuinely cross-checks a *family-specific* field against an oracle value is `validate_longrope.rs`** (`attention_factor` vs `reference["attention_scaling"]`, `:98`/`:117`, `< 1e-4`). qwen3 partially touches an oracle family field (`reference["use_qk_norm"]` asserted true, `:129`/`:134`) but asserts the Rust side against a hardcoded `true`.

**Disposition:** The *spirit* of §1.13 ("asymmetric/inconsistent rigor across the family tests") has a grain of truth — only longrope cross-checks a family-specific field against the oracle — but the specific two-group classification the finding is built on does not match the code. Drop the finding or rewrite it as: "only `validate_longrope.rs` cross-checks a family-specific config field against the oracle; llama32/deepseek assert their rope field against hardcoded literals; all eight cross-check the four *core* fields against the oracle." Severity remains Low.

### C2. §3 nit #2 — half-refuted (Low). Only **one** untagged `.contiguous()` site, not two.

The audit (and its verification addendum, which explicitly claims to have "confirmed genuinely untagged" both) flags `src/transformer/rope.rs:322` **and** `src/diffusion/rope.rs:95` as prose-only, missing the literal `// CONTIGUOUS:` tag.

- `src/transformer/rope.rs:321-322` — genuinely prose-only (`// candle_nn::rotary_emb::rope() expects contiguous input`), no literal tag. **Nit stands.**
- `src/diffusion/rope.rs:94-95` — **carries the literal tag**: line 94 reads `// CONTIGUOUS: candle_nn::rotary_emb::rope requires a contiguous input`. **This site is compliant; the nit is wrong here.** (Verified directly by the coordinator.)

**Disposition:** Reduce the §3 nit to a single site (`src/transformer/rope.rs:322`).

### C3. §3 "~143 as-cast sites" — imprecise ballpark (no severity; documentation figure only).

`#[allow(clippy::` count reproduces exactly (**185**, matching "~185"). The "~143 as-cast sites" figure reproduces no clean count: strict typed-numeric casts ≈ 62–73, `// CAST:` annotation tags present = 69, broadest `X as Y` heuristic ≈ 175. The real annotated-cast population is ≈ 69–73. Treat "~143" as loose; it does not affect the compliance conclusion (which is CONFIRMED — see below).

---

## Precision caveats (findings hold; wording could be tightened)

These do **not** change any finding's substance or severity — noted for accuracy only.

- **§1.2** — The "this is a smoke test" admission lives in the *sibling* per-family files' docstrings (e.g. `validate_llama32_forward.rs:10-12`, "a subtly-wrong `RoPE` still passes"; `validate_gemma2_forward.rs:10-12`, "subtly-wrong soft-capping/4-norm"), **not** in `validate_models.rs`'s own docstring (which says "validate forward-pass outputs"). The quoted phrases are verbatim-correct; only their attribution is loose. 661 lines confirmed via `wc -l`.
- **§1.4** — A single a2d-qwen3 artifact *does* exist — `#[test] fn parse_a2d_qwen3_bidirectional()` at `src/config.rs:2019` — but it is a synthetic-JSON **config-parse** unit test in `src/`, not a forward/oracle fixture in `tests/`. The audit's claim was scoped to `tests/` and `scripts/`, where it holds exactly (no a2d-qwen3 forward test or reference fixture).
- **§1.5** — "grep `clt_position_sweep` → zero matches" is imprecise: that string matches four *function names* in `validate_clt.rs` (`:445,603,1122,1280`). The substantive claim is nonetheless correct — grepping `clt_position_sweep_reference` (the JSON filename) is genuinely zero, and no test reads that oracle file (verified: `validate_clt.rs` only ever `read_to_string`s `config.json`/`index.json`). The oracle is orphaned.
- **§1.10** — The clt (`src/clt/mod.rs:315`) and sae (`src/sae/mod.rs:274`) doc examples are ` ```no_run ` (compiled, not executed); only the rwkv example (`src/rwkv/config.rs:11`) is a fully runnable doctest. All three are still uncovered by the sole `transformer,memory --doc` lane, so the coverage-gap finding is unaffected.
- **§4 row 13** — There is no `quantized` validation *Python* script (quantized parity is Rust-only in `tests/validate_quantized_loading.rs`), so that one cited example is vacuous; every other named missing script (`clt_qwen3_validation.py`, `qwen3_forward_validation.py`, `deepseek_coder_validation.py`, `phi35_longrope_validation.py`, `bidirectional_forward_validation.py`, `mdlm_forward_validation.py`, `convert_othello_mdlm.py`, `gridworld_generator.py`, `means_ends_generator.py`) is confirmed present-in-`scripts/`-but-absent-from-`scripts/README.md`.

---

## Confirmed findings (the validated work list)

Every item below was independently re-derived from source and **CONFIRMED**. Severities are the audit's own, retained where verification upheld them.

### Test validation

- **§1.1 — `#[ignore]`d oracle tests have no automated resurrection path. (High)** ✅ Every listed forward/parity test is individually `#[ignore = "..."]`-gated (13 files + the two live-CUDA cases in `validate_memory.rs`; per-file `#[test]`→`#[ignore]` line pairs verified). `grep -r "\-\-ignored"` across `.github/workflows/*.yml` and `scripts/preflight.ps1` returns **zero** hits (the only `--ignored` strings in the repo are in docs: `scripts/README.md`, `scripts/clt_position_sweep_comparison.md`). The weekly Monday cron (`ci.yml:8-11`, `cron: "0 6 * * 1"`) reruns the same `check` matrix with no `--ignored`. Central thesis: **upheld.**
- **§1.2 — `validate_models.rs` is the only default-run transformer test, and it's shape/top-k only. (High)** ✅ 661 lines; loads no oracle reference JSON (all `.json` reads are `config.json`/`tokenizer.json`/`.index.json` model files); every assertion is `assert_in_top_k(...,"Paris",...)` or a `dims3()` shape check; every test SKIP/returns on cache miss; no `hf-fetch-model` cache-seeding invocation anywhere in `ci.yml` (the two matches are inside an explanatory comment).
- **§1.3 — RWKV: final-token-only, loose tolerance, empirically-confirmed vacuous pass. (High)** ✅ `validate_rwkv6/7.rs` are not `#[ignore]`d; CI runs `cargo test --no-default-features --features "rwkv,rwkv-tokenizer"` (`ci.yml:63-64`); oracle is 7 tokens (`rwkv7_reference.json` token_ids has 7 entries); only last-position logits checked (`.i((0, seq_len - 1))`); `hook_capture_state` asserts shapes only; tolerance `logit_diff < 1.0` (CPU + GPU-F32), loosening to `< 2.0` on BF16; reference top logits 4.7979 (v6) / 7.5585 (v7); SKIP/return on cache miss present. The vacuous-pass mechanism is real.
- **§1.4 — Bidirectional forward-parity test unwired from CI/preflight. (High)** ✅ `validate_bidirectional_forward.rs` appears in neither `ci.yml` nor `preflight.ps1` (zero grep hits); sibling `validate_bidirectional_sampler.rs` gets a `--no-run` compile check (`ci.yml:105`, mirrored `preflight.ps1:131`) but has no oracle logit comparison (determinism/structure only; its own docstring says token values can't be matched against PyTorch). No CI-exercised numeric check of bidirectional attention exists. No a2d-qwen3 forward test/fixture in `tests/`+`scripts/` (see caveat above re: the src/ config-parse unit test).
- **§1.5 — `validate_clt.rs` has zero external-oracle comparison. (High)** ✅ 10 `#[test]` fns, 1479 lines, all self-consistency; thresholds `l2_dist > 0.01`/`> 1.0`, `jaccard < 0.8`, `concentration_ratio > 1.2`, `last_rank < 3` all present; `scripts/clt_position_sweep_reference.json` exists but is never read by any test. `validate_plt.rs`/`validate_plt_gemma.rs`/`validate_clt_qwen3.rs` do load oracle JSON and assert `abs diff < 1e-4` + exact top-k index match (see §1.11 caveat).
- **§1.6 — Bench files assert no correctness/perf thresholds. (High)** ✅ `bench_hook_overhead.rs`/`bench_hook_diagnostic.rs` are `[[test]]` (`Cargo.toml:138-144`, `required-features=["transformer"]`), run in the transformer lane (`ci.yml:55`, no `--skip`), assert only `result.get(&HookPoint::...).is_some()`; `overhead_pct`/timings are printed, never asserted; both silently `return` on cache miss.
- **§1.7 — `stoicheia` core math with weak/absent unit coverage. (High)** ✅ `ablation.rs::full_ablation_near_chance` asserts only `results.len() == 2` (`:294`); `tasks.rs::longest_cycle` (`:77`) has no `#[cfg(test)]` block anywhere in the file; `probing.rs::probe_runs_on_tiny_model` (`:370`) asserts only `correlation >= 0.0`, never the returned `NeuronRole`, leaving `best_probe_match` selection logic unverified.
- **§1.8 — SAE test silently passes on missing fixture; loose tolerances. (Medium)** ✅ `sae_vs_python_reference` is `#[test] #[ignore] #[serial]` (`:388-390`); on missing `scripts/sae_reference.json` it prints "SKIP…" and returns (`:393-398`) → reported as passed, in contrast to its PLT/CLT-Qwen3 siblings which `.expect()`-hard-fail (`validate_plt.rs:59`, `validate_plt_gemma.rs:60`, `validate_clt_qwen3.rs:60`). Tolerances confirmed loose: active-count diff `≤10` (`:469`), top-feature match `≥50%` (`:503-505`), MSE ratio within 10× (`:514-516`).
- **§1.9 — Feature-combination CI gaps. (High/Medium)** ✅ `quantized` has **zero** CI presence (no build/clippy/test; absent from the `ci.yml:134` combined build; `validate_quantized_loading.rs` never compiled) — **High**. No standalone clippy lane for `sae` (only in the line-134 `cargo build`). No standalone clippy/build lane for `mmap`, `memory-debug`, `probing`. `clt,sae,transformer` (needed by `validate_plt_gemma`) never compiled in any test-building step (`sae` appears only in a plain `cargo build`, which doesn't build test binaries). **All confirmed in both `ci.yml` and — additionally — `publish.yml`, which mirrors the same lanes and gaps.**
- **§1.10 — Doctest coverage gap for `clt`/`sae`/`rwkv`. (High)** ✅ Exactly one `--doc` lane exists: `cargo test --no-default-features --features "transformer,memory" --doc` (`ci.yml:125`). Doc examples at `src/clt/mod.rs:315-328` (needs `clt`), `src/sae/mod.rs:274-287` (needs `sae`), `src/rwkv/config.rs:11-21` (needs `rwkv`) are all uncovered (see caveat: first two are `no_run`).
- **§1.11 — PLT/CLT-Qwen3 oracles share the algorithm under test, not just data. (Medium)** ✅ Each generator script's docstring discloses "from-first-principles… NO circuit-tracer" and re-implements the same `ReLU(W_enc @ x + b_enc)` encoder formula that `src/clt/mod.rs:18` implements (`plt_llama_validation.py:6-9`, `plt_gemma_validation.py:5-11`, `clt_qwen3_validation.py:5-11`). They catch transcription bugs but not a shared conceptual error. `validate_sae.rs`'s oracle is comparatively stronger — its Python side drives real `AutoModelForCausalLM` (`sae_validation.py:30,130`).
- **§1.12 — Memory module: Metal path entirely untested. (Medium)** ✅ Exactly 3 tests; `cpu_snapshot_is_ram_only` always runs (asserts VRAM fields `None`); the two CUDA ground-truth tests (512 MiB alloc → delta `>256` MB) are `#[ignore]`+CUDA-gated and silent-no-op without a GPU. `src/memory.rs:125` has `device.is_cuda() || device.is_metal()`; hypomnesis `metal` feature enabled (`Cargo.toml`); no live test constructs a Metal device; CI is `ubuntu-latest` only.

### CI / workflow (§2)

- **`quantized`: no build/clippy/test anywhere (High); `clt`+`sae` doc examples never doctested (High); `sae`/`probing`/`mmap`/`memory-debug` no standalone clippy lane (Medium); `validate_plt_gemma` never compiles in any CI test-building step (Medium); `rwkv/config.rs` doctest uncovered (Medium)** ✅ All confirmed (see §1.9/§1.10).
- **`publish.yml` has no `needs`/`workflow_run` dependency on `ci.yml` (Medium); no `cargo publish --dry-run` step (Low).** ✅ Confirmed — but `publish.yml` re-runs its own full check sequence before publishing (`:42-120`, a near-verbatim copy of `ci.yml`), which mitigates most risk while inheriting every gap above.
- **Confirmed correct / not stale:** MSRV(1.88)+stable matrix runs the full step list on **both** toolchains (`ci.yml:21-27`, no toolchain conditionals), not lint-only on MSRV; `preflight.ps1`'s three tiers match CLAUDE.md; `publish.yml` tag trigger is the correct single-list `tags: ["v*", "!v*-*"]` form (`:20-22`), not `tags-ignore`; `Cargo.toml` version `0.1.17` matches CHANGELOG's latest dated entry with an empty `[Unreleased]`.

### CONVENTIONS.md compliance (§3)

- **Overall: excellent. (Confirmed.)** `#[allow(clippy::` count = **185** (exact); `#[non_exhaustive]` on **22 of 23** public enums, the sole exception `Norm` (`src/transformer/norm.rs:23`) correctly justified via `// EXHAUSTIVE:` + `#[allow(clippy::exhaustive_enums)]` (`:21-22`). Both `unsafe` sites (`src/backend.rs:906` mmap loader, `src/memory.rs` CUDA pool-trim) carry `// SAFETY:` comments and match the documented inventory. Zero production-code `unwrap`/`expect`/`panic!` — all 424 hits resolve to `#[cfg(test)]`/`#[cfg(all(test,...))]` blocks, the whole-file test module `registration_guard.rs`, or deterministic module-doc doctests on literal inputs.
- **✅ RESOLVED (commit `47ddab3`) — all three Low nits fixed.** The two `.len() as f32` casts in `src/interp/intervention.rs` now carry `// CAST: usize → f32` tags; the prose-only `.contiguous()` comment in `src/transformer/rope.rs` was retagged `// CONTIGUOUS:` (the `src/diffusion/rope.rs` site per correction C2 was already compliant); and the `src/memory.rs` `// SAFETY:` block was moved to sit immediately above the inner `unsafe {`. Verified clean under `cargo clippy` (transformer + memory, pedantic) and a `cuda,memory` build.

### Documentation staleness (§4) — all 14 table rows confirmed (plus the "not stale" set)

- **`README.md:13`** version banner reads "v0.1.12", actual v0.1.17 (`Cargo.toml:3`); rest of README is v0.1.14+ current. **High (visibility).** — **✅ RESOLVED:** banner bumped to v0.1.17.
- **`ROADMAP.md`** header dated Feb 19 / updated Apr 13, "published… as v0.1.8" (`:5-6`) — ~9 releases stale. **High.** — **✅ RESOLVED:** added a "historical planning document" banner pointing to `CHANGELOG.md` as the authoritative status source.
- **`ROADMAP.md §6.1`** feature-gate list (`:688-704`) omits `diffusion`/`stoicheia`/`quantized` (all real `Cargo.toml` features). **Medium.**
- **`ROADMAP.md` self-contradiction:** Phase-6 checklist `[ ] Add qwen3 auto-config support` (`:880`) unchecked while §3.3 (`:300`) lists Qwen3 as exact-parity validated. **Medium.**
- **`docs/roadmaps/`** — 11 files, no index/README; root `README.md` links only 1 of the 11 (`diffusion-lm-roadmap.md`, `README.md:203`; the other roadmap link `:205` is the top-level `ROADMAP.md`). **Medium.**
- **`PLAN-GRIDWORLD-PROLEPSIS.md`** status "⏳ Not started" (dated 2026-06-04, `:22-34`) while `CHANGELOG.md:288-321` and `docs/experiments/gridworld-prolepsis/` (11 result JSONs) confirm it ran to a negative result — actively misleading. **High.** — **✅ RESOLVED:** status section rewritten — Steps 0/A marked Done (A negative, a modality finding), Steps B–E marked "not pursued (gated by Step A)," with a one-sentence result and a pointer to the means-ends linguistic cell.
- **`PLAN-GEOMETRIC-CALCULATOR.md`** status "⏳ Not started" targeting v0.1.13 (`:20-31`); no `docs/experiments/geometric-calculator/`, zero CHANGELOG mention, 4 releases later — abandoned with no signal. **Medium.**
- **`candle_mi_v019_roadmap{,_V2,_V3}.md`** — three superseding variants, zero "supersed*" markers, none point to what shipped (v0.1.9–v0.1.11). **Medium.**
- **`design/intervention-api.md`** marked "Status: Implemented" (`:3`) but describes a `ForwardConfig::new().capture(...).intervene(...)` API that does not exist (`grep ForwardConfig src/` → zero); actual API is `HookSpec`-based. **High.** — **✅ RESOLVED:** rewrote the Status line and added an "As implemented" section documenting the real `HookSpec` API (`.new()/.capture()/.intervene()` → `forward` returns `HookCache`), the actual `Intervention` variants (`Replace/Add/Knockout/Scale/Zero`, no `Steer`), and resolved the two open questions.
- **`design/rwkv7-effective-attention.md`** marked "Status: Implemented" (`:3`) but body (`:26-37`) still reads as an open research question ("Defer: ship without it"); `compute_effective_attention_v7` was in fact built (`src/rwkv/mod.rs:1181`). **Medium.** — **✅ RESOLVED:** Status line updated and a "Resolution (as implemented)" section added — approach 1 (numerical, row-by-row, exact, validated to 6 decimals) shipped as `compute_effective_attention_v7`, exposed via `HookPoint::RwkvEffectiveAttn`; retrospective notes replace the open questions.
- **`design/hook-system.md`** HookPoint listing (`:19-42`) omits `RwkvEffectiveAttn(usize)`, which exists and is fully wired (`src/hooks.rs:93,119,171,510`). **Low.** — **✅ RESOLVED:** added `RwkvEffectiveAttn(usize)` to the enum listing, immediately after `RwkvDecay(usize)`, mirroring source order.
- **`design/candle-version.md`** recommends exact-pin `candle-core = "=0.9"` (`:19-22`); actual `Cargo.toml:32-33` uses caret `"0.9"`. **Low.**
- **`scripts/README.md`** accurate through ~v0.1.12; ~20 newer scripts undocumented (omission, not false claim; see §4-row-13 caveat re: no quantized *Python* script). **Low.**
- **`CHANGELOG.md`** body accurate/complete for 0.1.9–0.1.17; sole defect is a dangling version-link footer stuck at `[0.1.8]`. **Low.**
- **Confirmed not stale:** `BACKENDS.md`, `HOOKS.md`, `CONTRIBUTING.md`, `CONVENTIONS.md` all present.

---

## What the audit got right about its own limits

The source audit's "What's genuinely solid" and "Verification addendum" sections were themselves spot-checked. They hold — with the **single exception** noted in C2 (the addendum wrongly claims to have re-confirmed `src/diffusion/rope.rs:95` as untagged; it carries the tag). The audit's own numeric self-corrections (661 lines, 10 CLT tests, 22/23 enums, the `validate_plt_gemma` "never compiles" sharpening, the "1 of 11" roadmap-link count) were all independently reproduced and are correct.

## Recommendations — unchanged from the source audit

Because every High/Medium finding survived verification, the source audit's prioritized recommendations stand as written (resurrection path for the `#[ignore]`d suite; close the `quantized` CI gap; give `validate_clt.rs` a real oracle via the orphaned `clt_position_sweep_reference.json`; add assertions to the bench files; fix the `validate_sae.rs` silent-pass; unit-test `stoicheia` ablation/`longest_cycle`; the one-line doc-version/roadmap-banner fixes; correct the two `design/*.md` status headers). The only edits to the audit itself are: **drop or rewrite §1.13**, and **halve the §3 `.contiguous()` nit to one site.**
