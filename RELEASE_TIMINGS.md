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
| _(unpopulated)_ | — | — | — | — | — | — | — |

`_(unpopulated)_` = no `--release` run has been recorded yet. Populate the
first real row at the next release that touches the hook path. Current dev
machine, for reference: AMD Ryzen 9 5950X (16 cores / 32 threads), RTX 5060
Ti 16 GiB.
