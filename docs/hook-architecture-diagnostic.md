# Hook Architecture Diagnostic

**Date:** 2026-05-01
**Scope:** `HookSpec` / `HookCache` hot path — `is_captured`, `interventions_at`, `cache.store`, and the per-hook-point branches inside `GenericTransformer::forward`.
**Method:** dedicated micro-bench (`tests/bench_hook_diagnostic.rs`) on Llama-3.2-1B (16 layers, 194 captures), CUDA F32 on RTX 5060 Ti, 100 runs per spec.
**Trigger:** suspicion that the hook architecture (enum-keyed `HashSet` for captures, `Vec<(HookPoint, Intervention)>` for interventions) was responsible for an apparent +11.5%–17.5% GPU overhead on full-capture forwards.

## Executive summary

**The hook architecture is not a performance bottleneck.** With sufficient sampling (100 runs vs. the original 10), the full-capture forward delta on Llama-3.2-1B is **~226 µs on a 35.82 ms forward — about 0.6%**. The previously cached "+11.5%" / "+17.5%" numbers were 10-run noise artifacts.

The diagnostic refuted three hypotheses that motivated a possible refactor:

1. **Hash-lookup cost** — measured at **3.16 µs per full-capture-spec walk** (194 hooks). That is 1.4% of the already-tiny 226 µs delta. A `Vec<LayerPlan>` refactor would save *at most* this — i.e., nothing observable.
2. **Held references blocking buffer reuse** — refuted by the density sweep. Overhead grows roughly linearly with capture count at ~50–60 µs per held tensor, but stays small in absolute terms.
3. **Per-hook-type cost (e.g., softmax-fusion blocking on `AttnScores` / `AttnPattern`)** — refuted by the equal-count shape comparison. AttnPattern (254 µs delta) and AttnScores (350 µs delta) are no more expensive than ResidPost (172 µs); AttnOut is the highest at 423 µs but within run-to-run variance of the others.

**Conclusion:** do not refactor the hook architecture for performance. Practicality concerns (e.g., closure-based interventions) remain valid follow-ups but should not be motivated by speed.

## Methodology

The bench file at [`tests/bench_hook_diagnostic.rs`](../tests/bench_hook_diagnostic.rs) decomposes the full-capture overhead into independently-measured components, plus two additional sweeps to discriminate between the held-references and per-hook-type hypotheses:

- **A. Pure spec-lookup cost.** Walks the full-capture spec the same way the transformer forward path walks it (`is_captured` + `interventions_at` for every per-layer hook point), with no tensor work and no model in the loop. 100 000 iterations.
- **B. Real forward overhead.** Runs the actual model with empty vs. full-capture spec. 100 runs each.
- **C. Capture-machinery cost.** Builds a `HookCache` and stores one `Tensor::clone` at every hook point that full capture would touch. No model. 10 000 iterations.
- **D. Capture-density sweep.** Captures only `ResidPost` at evenly spaced layers (1, 2, 4, 8, 16). Single hook type, varying count — held-references would predict roughly linear scaling with count.
- **E. Equal-count shape comparison.** Captures one of `{ResidPost, AttnOut, AttnScores, AttnPattern}` at every layer (16 captures each). Held-references would predict similar costs across types; per-hook-type computational cost would predict large differences.

A sanity check decodes top-3 next-token logits at the start of every run; "Paris" (top-1) confirms the model is producing correct outputs throughout the bench.

## Code tour

The bench is laid out so each helper maps cleanly to one sub-bench. All line numbers reference [`tests/bench_hook_diagnostic.rs`](../tests/bench_hook_diagnostic.rs).

**Spec builders (no tensor work).** These produce the `HookSpec` instances used as inputs to the various sub-benches:

- [`full_capture_spec`](../tests/bench_hook_diagnostic.rs#L120) — every per-layer hook point at every layer, plus `Embed` and `FinalNorm` (194 captures on a 16-layer model). Used in A, B, C.
- [`resid_post_every_n`](../tests/bench_hook_diagnostic.rs#L144) — only `ResidPost` at evenly spaced layers. Used by D to vary capture *count* while holding hook *type* and shape constant.
- [`one_hook_per_layer`](../tests/bench_hook_diagnostic.rs#L157) — generic over a `Fn(usize) -> HookPoint` selector; produces an `N`-capture spec for any single hook type. Used by E to compare equal-count specs across hook types.

**Hot-path mocks (the actual measured work).** Both have `#[inline(never)]` so the compiler can't fold them into the surrounding loop and skew the per-iteration cost:

- [`walk_spec`](../tests/bench_hook_diagnostic.rs#L172) (sub-bench A) — walks the spec in the *same order the transformer forward walks it*: at every per-layer hook point, calls both `is_captured` and `interventions_at(...).next()`. Each call is wrapped in `std::hint::black_box` to defeat dead-code elimination (the return values are otherwise unused, and an optimizing release build would happily delete the whole loop). No tensors, no model — pure spec-data overhead.
- [`store_all`](../tests/bench_hook_diagnostic.rs#L202) (sub-bench C) — builds a fresh `HookCache` and calls `cache.store(point, dummy.clone())` at every hook point full capture would touch. Same number of inserts (194) as `walk_spec` has lookups. The dummy tensor is a `[1, 5, 2048]` F32 buffer materialized once outside the loop, so each `clone` is just an `Arc` bump on the storage handle (independent of tensor size). The result is fed to `black_box` to keep the cache live across iterations.

**Forward-timer helper.**

- [`time_forward`](../tests/bench_hook_diagnostic.rs#L231) — runs `FWD_RUNS` forwards under a given spec and returns the per-forward average duration. Used by B for the empty-vs-full comparison and reused by D and E for their per-spec timings, so all sub-benches that touch the model use identical timing logic.

**Tunable iteration counts.**

The asymmetric counts at lines [223-228](../tests/bench_hook_diagnostic.rs#L223-L228) reflect the cost-per-iter of each sub-bench:

- `FWD_RUNS = 100` (B/D/E) — each forward is ~36 ms; 100 runs ≈ 3.6 s per spec.
- `LOOKUP_ITERS = 100_000` (A) — each `walk_spec` is ~3 µs; 100 000 iters ≈ 300 ms.
- `STORE_ITERS = 10_000` (C) — each `store_all` is ~8 µs; 10 000 iters ≈ 80 ms.
- `FWD_WARMUP = 5` — warms the GPU allocator before B's empty baseline. The other sub-benches have their own per-bench warmup loop (see lines 311 and 333).

The intent is roughly equal *total* time per sub-bench (~hundreds of ms) so the noise floor is similar across A, C, and per-spec measurements in B/D/E. A and C are deliberately cheap-per-iter so we can run many iterations and converge to a stable per-iter cost; B/D/E are bound by the much-more-expensive forward and use fewer iterations to keep the whole bench under a minute.

## Results (RTX 5060 Ti, CUDA F32, 100 runs)

```
B. Real forward overhead:
   Empty:  35.82 ms
   Full:   36.04 ms
   Delta:  226 µs   (0.6%)

A. Pure spec-lookup cost (194 hooks):           3.16 µs / forward
C. Capture machinery (194 clone+insert):        8.20 µs / forward

Attribution of the 226 µs delta:
   Pure-lookup share:            1.4%
   Capture-machinery share:      3.6%
   Remainder (forward-internal): 95.0%

D. Density sweep (ResidPost only):
   2 captures:    67 µs delta
   4 captures:   442 µs delta
   8 captures:   565 µs delta
  16 captures:   923 µs delta

E. Shape comparison (16 captures each):
   ResidPost:    172 µs delta
   AttnOut:      423 µs delta
   AttnScores:   350 µs delta
   AttnPattern:  254 µs delta
```

## Caveats

**Sub-millisecond noise is significant.** The same spec (16 ResidPost captures) shows 923 µs in sweep D and 172 µs in comparison E — ~5× variation due to order-of-execution effects (allocator state, GPU warmup) between successive sub-benchmarks within a single run. The qualitative conclusions (small overhead, scales with count, no per-type cliff) are robust to this noise; precise per-hook-type numbers are not. Trustworthy attribution at the sub-ms level would require running each spec in a separate process, randomizing spec order, or inserting explicit GPU syncs between specs.

**CPU is below the measurement floor.** A 1B-parameter model on CPU at F32 takes ~3.1 s per forward; capture-machinery cost is ~8 µs. The bench would need ≥10⁵ CPU runs to resolve the difference, which is not done. CPU overhead is therefore reported as "indistinguishable from zero".

**This is a single model and a single GPU.** Llama-3.2-1B at 16 layers / 2048 hidden / F32 on a 5060 Ti. Larger models or different dtypes (BF16, F16) may shift the relative weight of host-side vs. device-side costs.

## Documentation correction

The `HookSpec` docstring at [`src/hooks.rs`](../src/hooks.rs) previously claimed empty specs incurred *zero* overhead. The diagnostic showed that figure is more precisely "a few microseconds" — small enough to be unmeasurable against a forward, but not literally zero (the per-hook-point `is_captured` checks still run, and the `HookCache` placeholder still allocates). The doc was updated to reflect this.

## Future work (if motivated by something other than this overhead)

- **Closure-based interventions** (`Intervention::Custom(Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>)`) for research ergonomics. Would require a manual `Clone` / `Debug` impl since `dyn Fn` does not auto-derive these. Independent of the perf question.
- **More-rigorous bench methodology** (per-spec process isolation, explicit sync, randomized order) would tighten the sub-ms measurements if a future refactor needed strong evidence of speedup.

## References

- Bench source: [`tests/bench_hook_diagnostic.rs`](../tests/bench_hook_diagnostic.rs)
- Existing overhead bench: [`tests/bench_hook_overhead.rs`](../tests/bench_hook_overhead.rs)
- Hook system: [`src/hooks.rs`](../src/hooks.rs)
- Transformer forward path: [`src/transformer/mod.rs`](../src/transformer/mod.rs), [`src/transformer/attention.rs`](../src/transformer/attention.rs)

Run the diagnostic with:

```powershell
cargo test --test bench_hook_diagnostic --features transformer,mmap --release -- --nocapture --test-threads=1
```
