# Preflight checks — run before every push.
#
# Mirrors the per-feature lanes in .github/workflows/ci.yml so that a clean run
# here predicts a clean run on CI. CI runs a matrix of two toolchains:
# MSRV (1.88) and stable, each executing fmt + build/clippy/test for the
# transformer, rwkv, stoicheia, clt and diffusion feature sets, plus bare
# and all-software-features builds.
#
# Tiers:
#   ./scripts/preflight.ps1        # FAST (default): full STABLE mirror + MSRV
#                                   # (1.88) fmt + clippy. MSRV clippy compiles on
#                                   # 1.88, so it catches version-specific lints
#                                   # (e.g. clippy::suboptimal_flops, which fired
#                                   # only on 1.88) without paying for MSRV
#                                   # build/test.
#   ./scripts/preflight.ps1 -Ci    # FULL mirror: every ci.yml step on BOTH 1.88
#                                   # and stable. Literal "green preflight =
#                                   # green CI"; use before important pushes or
#                                   # after MSRV-sensitive changes.
#   ./scripts/preflight.ps1 -Full   # also run the bench_hook_* CPU benches
#                                   # (composes with -Ci). See note below.
#
# The `rustup update stable` at the top is the whole point: CI tracks rolling
# stable, so a dry-run on a stale local toolchain can pass while CI fails on a
# lint only the newer compiler knows (this is how clippy::suboptimal_flops from
# Rust 1.96 once broke a clean main). Freshen first, then lint.
#
# bench_hook_* note: these are timing benchmarks, not correctness checks. They
# SKIP on CI because the gated meta-llama/Llama-3.2-1B isn't cached on the
# runners — so they are not part of "green CI", and the default/-Ci paths skip
# them (`--skip bench_hook`) for parity. Locally the model IS cached, so -Full
# runs them (~slow); do that only when adding a new model family, the change
# that can shift the benchmarked forward/hook paths.
#
# Bypass: not recommended — if you must, just don't run it (convention, not an
#         enforced git hook).

param(
    [switch]$Full,
    [switch]$Ci
)

$ErrorActionPreference = "Stop"

# Run a scriptblock step (used for the toolchain-setup commands that take no
# per-lane parameters), halting the whole preflight on a non-zero exit.
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

# Run `cargo +<toolchain> <args...>` as a named step. Arguments are passed as an
# explicit array (not captured in a scriptblock) so toolchain/feature values are
# bound by value — no PowerShell closure surprises.
function Invoke-Cargo {
    param(
        [string]$Name,
        [string]$Tc,
        [string[]]$CargoArgs
    )
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & cargo "+$Tc" @CargoArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Feature sets and the all-software set — keep in sync with ci.yml.
$featureSets = @("transformer", "rwkv,rwkv-tokenizer", "stoicheia", "clt,transformer", "diffusion")
$allSoftware = "transformer,rwkv,rwkv-tokenizer,diffusion,clt,sae,stoicheia,probing"

# Skip the bench_hook_* benches unless -Full (see header note).
$benchArgs = if ($Full) { @() } else { @("--", "--skip", "bench_hook") }

# Run CI's lanes for one toolchain. With -LintOnly, run only fmt + the clippy
# lanes (clippy compiles + lints on that toolchain) and skip the build/test
# steps — used for the MSRV toolchain on the fast default path.
function Invoke-Lanes {
    param(
        [string]$Tc,
        [bool]$LintOnly
    )

    Invoke-Cargo "[$Tc] Formatting" $Tc @("fmt", "--check")

    foreach ($f in $featureSets) {
        Invoke-Cargo "[$Tc] Clippy ($f)" $Tc `
            @("clippy", "--no-default-features", "--features", $f, "--", "-W", "clippy::pedantic")
    }

    if ($LintOnly) { return }

    # Builds + tests, mirroring ci.yml step-for-step.
    Invoke-Cargo "[$Tc] Build (transformer)" $Tc `
        @("build", "--no-default-features", "--features", "transformer")
    Invoke-Cargo "[$Tc] Tests (transformer)" $Tc `
        (@("test", "--no-default-features", "--features", "transformer") + $benchArgs)

    Invoke-Cargo "[$Tc] Build (RWKV)" $Tc `
        @("build", "--no-default-features", "--features", "rwkv,rwkv-tokenizer")
    Invoke-Cargo "[$Tc] Tests (RWKV)" $Tc `
        @("test", "--no-default-features", "--features", "rwkv,rwkv-tokenizer")

    Invoke-Cargo "[$Tc] Build (Stoicheia)" $Tc `
        @("build", "--no-default-features", "--features", "stoicheia")
    Invoke-Cargo "[$Tc] Tests (Stoicheia)" $Tc `
        @("test", "--no-default-features", "--features", "stoicheia", "--lib", "--test", "stoicheia_analysis", "--test", "validate_stoicheia")

    Invoke-Cargo "[$Tc] Build (CLT)" $Tc `
        @("build", "--no-default-features", "--features", "clt,transformer")
    Invoke-Cargo "[$Tc] Tests (CLT)" $Tc `
        @("test", "--no-default-features", "--features", "clt,transformer", "--lib")

    Invoke-Cargo "[$Tc] Build (Diffusion + examples)" $Tc `
        @("build", "--no-default-features", "--features", "diffusion", "--examples")
    Invoke-Cargo "[$Tc] Tests (Diffusion)" $Tc `
        @("test", "--no-default-features", "--features", "diffusion", "--lib", "--test", "validate_mdlm_forward", "--test", "validate_othello_forward")

    # validate_bidirectional_sampler needs transformer + diffusion together; no
    # single-feature lane builds it, so compile-check it explicitly.
    Invoke-Cargo "[$Tc] Tests compile (transformer+diffusion)" $Tc `
        @("test", "--no-default-features", "--features", "transformer,diffusion", "--test", "validate_bidirectional_sampler", "--no-run")

    Invoke-Cargo "[$Tc] Build (no default features)" $Tc `
        @("build", "--no-default-features")
    Invoke-Cargo "[$Tc] Build (all software features)" $Tc `
        @("build", "--no-default-features", "--features", $allSoftware)
}

# --- Freshen toolchains (match CI's rolling stable; ensure MSRV present) ---
Invoke-Step "Update stable toolchain" { rustup update stable }
Invoke-Step "Ensure MSRV 1.88 toolchain" { rustup toolchain install 1.88 }
Invoke-Step "Ensure 1.88 components (clippy, rustfmt)" { rustup component add clippy rustfmt --toolchain 1.88 }
Write-Host "Using:" -ForegroundColor Yellow
& rustc "+stable" --version
& rustc "+1.88" --version

Invoke-Step "Pick up latest hf-fetch-model patch" { cargo update -p hf-fetch-model }

# Stable: always the full lane set. MSRV 1.88: fmt + clippy by default; the full
# lane set only under -Ci.
Invoke-Lanes -Tc "stable" -LintOnly $false
Invoke-Lanes -Tc "1.88" -LintOnly (-not $Ci)

if ($Ci) {
    Write-Host "`nAll preflight checks passed (full 1.88 + stable mirror) — safe to push." -ForegroundColor Green
}
else {
    Write-Host "`nAll preflight checks passed (full stable mirror + MSRV clippy)." -ForegroundColor Green
    Write-Host "Run './scripts/preflight.ps1 -Ci' for the full both-toolchain mirror before important pushes." -ForegroundColor Yellow
}
