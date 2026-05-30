# Preflight checks — run before every push.
#
# Mirrors the per-feature lanes in .github/workflows/ci.yml so that a clean run
# here means a clean run on CI. The `rustup update stable` at the top is the
# whole point: CI tracks rolling stable, so a dry-run on a stale local toolchain
# can pass while CI fails on a lint that only the newer compiler knows about
# (e.g. clippy::suboptimal_flops, added in Rust 1.96). Freshen first, then lint.
#
# Usage:  ./scripts/preflight.ps1          # fast: skips the ~90 min hook benches
#         ./scripts/preflight.ps1 -Full    # full: includes bench_hook_* CPU benches
#
# Run the fast path before every push. Run -Full only when adding a new model
# family — that is the change that can affect the benchmarked forward/hook paths,
# so the heavy `bench_hook_*` CPU benches (≈90 min) are worth paying for then.
#
# Bypass: not recommended — if you must, just don't run it (it is convention,
#         not an enforced git hook).

param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Freshen the toolchain so local clippy == CI's rolling stable.
Invoke-Step "Update stable toolchain" { rustup update stable }
Write-Host "Using:" -ForegroundColor Yellow
rustc --version
cargo clippy --version

Invoke-Step "Pick up latest hf-fetch-model patch" { cargo update -p hf-fetch-model }

Invoke-Step "Formatting" { cargo fmt --check }

# Per-feature clippy lanes — must match ci.yml. Checking only --all-features
# misses lints that appear under a single feature flag.
Invoke-Step "Clippy (transformer)" {
    cargo clippy --no-default-features --features transformer -- -W clippy::pedantic
}
Invoke-Step "Clippy (rwkv)" {
    cargo clippy --no-default-features --features "rwkv,rwkv-tokenizer" -- -W clippy::pedantic
}
Invoke-Step "Clippy (stoicheia)" {
    cargo clippy --no-default-features --features stoicheia -- -W clippy::pedantic
}
Invoke-Step "Clippy (clt)" {
    cargo clippy --no-default-features --features "clt,transformer" -- -W clippy::pedantic
}

# The bench_hook_* tests are timing benchmarks (~90 min on CPU), not correctness
# checks. Skip them on the fast path; -Full runs them when a new model family
# could have shifted the benchmarked forward/hook paths.
$benchArgs = if ($Full) { @() } else { @("--", "--skip", "bench_hook") }
Invoke-Step "Tests (transformer)" {
    cargo test --no-default-features --features transformer @benchArgs
}
Invoke-Step "Tests (stoicheia)" {
    cargo test --no-default-features --features stoicheia --lib --test stoicheia_analysis --test validate_stoicheia
}

Write-Host "`nAll preflight checks passed — safe to push." -ForegroundColor Green
