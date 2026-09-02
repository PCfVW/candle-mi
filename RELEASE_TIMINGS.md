# Hook-overhead release timings

`tests/bench_hook_overhead.rs` and `tests/bench_hook_diagnostic.rs` measure
the runtime cost of the hook architecture (`HookSpec`/`HookCache`) against a
plain forward pass. Unlike [`RESURRECTION.md`](RESURRECTION.md), which is a
**correctness** ledger (did an oracle test's output still match the
reference?), this file is a **performance** ledger: did hook overhead drift
between releases? Neither test does an oracle comparison, and neither is
`#[ignore]`d, so they never run under plain `cargo test` numbers you'd want
to compare release-over-release unless invoked exactly as below.

**Refresh it** at release time (see `CLAUDE.md` `## Releasing`), and whenever
a change could plausibly move the benchmarked forward/hook paths (new model
family, `HookCache`/`HookSpec` refactor):

```
cargo test --test bench_hook_overhead   --features transformer,mmap --release -- --nocapture
cargo test --test bench_hook_diagnostic --features transformer,mmap --release -- --nocapture
```

**Must be `--release`.** `scripts/preflight.ps1 -Full` also runs these two
tests, but in `dev` profile (no `--release`) — that pass is a "does it still
run" smoke check folded into the CI mirror, not a timing sample. This repo's
`[profile.dev]` sets `opt-level = 1` (not Cargo's default `0`), so a `dev`
run is not the worst case, but `[profile.release]` adds `opt-level = 3` +
LTO + `codegen-units = 1` on top — still not the same number. A `-Full` run
is not comparable to a row below; only a standalone `--release` run belongs
here.

Both benches load a single model, `meta-llama/Llama-3.2-1B` (cached locally,
gated), no other model in the roster — the cost is iteration count, not
model count. `bench_hook_diagnostic` alone runs roughly 1,100 forward passes
across its five sections (A-E) against that one model. For the CUDA rows, a
CUDA device is required too.

**Headline metrics:**
- `bench_hook_overhead`: forward-pass average with no hooks vs. full capture,
  and the overhead as a percentage, measured separately for CPU F32 and CUDA
  BF16.
- `bench_hook_diagnostic`, section B ("real forward overhead"): the same
  empty-vs-full-spec delta as `bench_hook_overhead`'s CPU row, plus the
  attribution split — how much of that delta is pure spec-lookup cost (A)
  vs. capture-machinery cost (C) vs. forward-internal remainder.

**Why CPU and thread count are their own columns, not a header note** (unlike
`RESURRECTION.md`'s single "Last run" block): candle's CPU gemm path
parallelizes across cores via `rayon`, so these numbers are not portable
across machines, or even across a core-count change on the same machine.
Every row needs its own hardware context to stay interpretable once this
table has more than one entry.

| Version | Date | CPU (cores/threads) | Toolchain | GPU | CPU F32 overhead | CUDA BF16 overhead | Diagnostic: lookup / capture / remainder |
|---|---|---|---|---|---|---|---|
| 0.1.23 | 2026-08-24 | AMD Ryzen 9 5950X (16/32) | rustc 1.98.0 | RTX 5060 Ti 16 GiB | -0.9% (3.11s → 3.08s, 10 runs) | +8.5% (32.96ms → 35.76ms, 10 runs) | 0.0% / 0.0% / 100.0% |
| 0.1.24 | 2026-09-02 | AMD Ryzen 9 5950X (16/32) | rustc 1.98.0 | RTX 5060 Ti 16 GiB | +0.3% / +0.1% (3.13s → 3.14s; 3.01s → 3.01s, two 10-run samples) | +11.1% / +7.8% (32.14ms → 35.69ms; 33.43ms → 36.05ms, two 10-run samples) | 2.6% / 5.3% / 92.2% (GPU); degenerate on CPU |

CPU overhead is within noise (negative, i.e. "full capture" measured
*faster* than "no hooks" — expected at ~3s/forward where hook bookkeeping is
a rounding error). GPU shows a real, small overhead: capture-machinery cost
becomes visible as a percentage once the forward itself only takes ~33ms.
The diagnostic attribution (0/0/100) is degenerate at this scale — section
B's delta was ≈0.00ns on both devices, so dividing A and C by ~zero isn't a
meaningful split; treat this row's diagnostic column as "no measurable
overhead to attribute" rather than a real 0/0/100 breakdown. A future run
where overhead is large enough to attribute meaningfully will make that
column more informative.

**0.1.24 refresh, and what it says about the headline number.** Run because the
hook path changed shape: `apply_intervention` now takes the `HookPoint` at all
17 call sites, and seven blocks in `forward_layer_range` bind the point once and
`clone()` it into `cache.store` instead of constructing it fresh. Two 10-run
samples were taken back to back on identical code, and they **straddle** the
0.1.23 figure: +11.1% then +7.8% on CUDA, against 8.5% before. A 3.3-point
spread between consecutive runs is the measurement's own noise, so read the
CUDA column as "roughly 8 to 11 percent, and do not compare two rows to fewer
significant figures than that".

The better estimate is in the diagnostic, not here. Its section B measures the
same empty-vs-full delta over **100** runs rather than 10, and reports
163.15us on the GPU against a ~35ms forward: about **0.5%**, some twenty times
smaller than the 10-run headline. Where the two disagree by that much, trust the
100-run figure and treat this table's percentage as a coarse regression tripwire
rather than a measurement.

**The GPU attribution is finally non-degenerate**, which the 0.1.23 note
predicted would eventually happen. Of the 163.15us of real capture overhead,
2.6% is pure spec lookup and 5.3% is capture machinery, leaving 92.2%
forward-internal. So roughly **8% of the cost is the hook architecture and 92%
is not**, which is the first direct measurement of the question
`docs/hook-architecture-diagnostic.md` was written to settle, and it corroborates
that document's "do not refactor the hook architecture for performance". CPU
stays degenerate at 0/0/100, as expected at ~3s per forward.

**Known quirk, not a hang to panic over:** both test binaries print their
final `test result: ok` line and then take a long time to actually exit —
suspected CUDA context teardown on Windows/WDDM, roughly proportional to how
many device allocations the run made. `bench_hook_diagnostic` (~2,200 total
forwards, each briefly touching the GPU via `HookCache`) took nearly 50
minutes to exit after finishing its measured work; the much lighter
`bench_hook_overhead` (48 forwards) still didn't exit within 2 minutes of
printing its result. The data is complete and correct by the time
`test result: ok` prints — kill the process at that point rather than
waiting for it to exit on its own.
