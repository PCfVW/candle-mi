# resurrect.ps1 — run the #[ignore]d oracle/parity suite locally and stamp
# RESURRECTION.md with a PER-TEST "last verified" date.
#
# The oracle/parity tests are #[ignore]d because they need cached HuggingFace
# models (some gated) and, for many, a CUDA GPU with >= 16 GiB VRAM — so they
# never run in GitHub CI (CPU-only runners, no cached models). This runner
# exercises them on a workstation where the models are cached, giving the
# #[ignore]d suite the local safety net the 2026-07-01 audit (§1.1) flagged as
# missing, and writes a committed staleness log.
#
# Staleness is tracked PER ENTRY, not per run: each test carries its own "last
# verified" date (advanced only on a real PASS). So a -Quick run refreshes only
# the two entries it ran; everything else keeps its true (older) date. This keeps
# the report honest and decouples staleness from the tier you happened to run.
#
# Tiers (how much to run now — nothing more):
#   -Quick    cheap ungated CPU encoder-parity only (clt_qwen3, plt_gemma) — ~5 min
#   (default) everything EXCEPT the two slow outliers (Mistral-7B CPU forward and
#             the anacrousis 28x15 matrix) — ~40-50 min
#   -Full     literally everything, incl. the two slow outliers — ~1.5-3 h
#
# Read-only:
#   -Status   print how stale the suite is (oldest entry, entries past the
#             threshold, any failing) and exit — runs NOTHING, downloads NOTHING.
#             `-StaleDays N` sets the threshold (default 50).
#
# Device policy: GPU tests run on the GPU; CPU-parity tests (*_cpu, plt,
# clt_qwen3) run on CPU by design. Runs with the default feature set so `cuda`
# stays ON; each entry adds its extra feature(s).

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$Status,
    [int]$StaleDays = 50
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$mdPath = Join-Path $repoRoot 'RESURRECTION.md'

# Canonical entry list. `Features` are ADDED to the default set (cuda,transformer).
# Quick=$true -> included in -Quick; FullOnly=$true -> only in -Full;
# SlowSkip -> a `--skip <pat>` applied in the default tier (dropped in -Full).
$entries = @(
    @{ Name = 'clt_qwen3 (encoder parity)'; Features = 'clt';                         Tests = @('validate_clt_qwen3');                             Device = 'CPU';     Models = 'bluelightai/clt-qwen3-1.7b-base-20k (~240 MiB)';        Quick = $true }
    @{ Name = 'plt_gemma (encoder parity)'; Features = 'clt,sae';                     Tests = @('validate_plt_gemma');                             Device = 'CPU';     Models = 'google/gemma-scope-2b-pt-transcoders (~864 MiB, gated)'; Quick = $true }
    @{ Name = 'plt_llama (encoder parity)'; Features = 'clt';                         Tests = @('validate_plt');                                   Device = 'CPU';     Models = 'mntss/transcoder-Llama-3.2-1B (~16 GiB)' }
    @{ Name = 'llama32 forward';            Features = 'mmap';                        Tests = @('validate_llama32_forward');                       Device = 'CPU+GPU'; Models = 'meta-llama/Llama-3.2-1B (gated)' }
    @{ Name = 'gemma2 forward';             Features = 'mmap';                        Tests = @('validate_gemma2_forward');                        Device = 'CPU+GPU'; Models = 'google/gemma-2-2b (gated)' }
    @{ Name = 'phi3-mini forward';          Features = 'mmap';                        Tests = @('validate_phi3_mini_forward');                     Device = 'CPU+GPU'; Models = 'microsoft/Phi-3-mini-4k-instruct' }
    @{ Name = 'mistral-7b forward';         Features = 'mmap';                        Tests = @('validate_mistral_7b_forward');                    Device = 'CPU+GPU'; Models = 'mistralai/Mistral-7B-v0.1 (gated)'; SlowSkip = '_cpu' }
    @{ Name = 'qwen3 forward';              Features = 'mmap';                        Tests = @('validate_qwen3_forward');                         Device = 'CPU+GPU'; Models = 'Qwen/Qwen3-1.7B-Base' }
    @{ Name = 'qwen2.5-coder forward';      Features = 'mmap';                        Tests = @('validate_qwen25_coder_forward');                  Device = 'CPU+GPU'; Models = 'Qwen/Qwen2.5-Coder-3B-Instruct' }
    @{ Name = 'starcoder2 forward';         Features = 'mmap';                        Tests = @('validate_starcoder2_forward');                    Device = 'CPU+GPU'; Models = 'bigcode/starcoder2-3b (gated)' }
    @{ Name = 'deepseek forward';           Features = 'mmap';                        Tests = @('validate_deepseek_forward');                      Device = 'CPU+GPU'; Models = 'deepseek-ai/deepseek-coder-1.3b-base' }
    @{ Name = 'longrope forward';           Features = 'mmap';                        Tests = @('validate_longrope');                              Device = 'GPU';     Models = 'microsoft/Phi-3.5-mini-instruct (~15 GiB)' }
    @{ Name = 'bidirectional (a2d-qwen2)';  Features = 'diffusion,mmap';              Tests = @('validate_bidirectional_forward');                 Device = 'CPU+GPU'; Models = 'dllm-hub/Qwen2.5-Coder-0.5B-...-mdlm (~1.2 GiB)' }
    @{ Name = 'mdlm + othello';             Features = 'diffusion,mmap';              Tests = @('validate_mdlm_forward', 'validate_othello_forward'); Device = 'CPU+GPU'; Models = 'kuleshov-group/mdlm-owt; Othello fixtures (OTHELLO_MDLM_FIXTURES)' }
    @{ Name = 'quantized (bnb/AWQ/GPTQ)';   Features = 'quantized,mmap';              Tests = @('validate_quantized_loading');                     Device = 'GPU';     Models = 'medmekk/...-bnb-nf4; casperhansen/...-awq; shuyuej/...-GPTQ' }
    @{ Name = 'clt (encode/inject/sweep)';  Features = 'clt,mmap';                    Tests = @('validate_clt');                                   Device = 'GPU';     Models = 'gemma-2-2b + llama-3.2-1b + mntss CLTs (>=16 GiB VRAM)' }
    @{ Name = 'sae (encode/inject/parity)'; Features = 'sae,mmap';                    Tests = @('validate_sae');                                   Device = 'GPU';     Models = 'gemma-2-2b + gemma-scope-2b-pt-res' }
    @{ Name = 'memory (VRAM probe)';        Features = 'memory';                      Tests = @('validate_memory');                                Device = 'GPU';     Models = '(none - allocates a GPU tensor)' }
    @{ Name = 'rwkv6 + rwkv7';              Features = 'rwkv,rwkv-tokenizer,mmap';    Tests = @('validate_rwkv6', 'validate_rwkv7');               Device = 'CPU+GPU'; Models = 'RWKV v6-Finch-1B6; RWKV7-Goose-1.5B' }
    @{ Name = 'anacrousis (28x15 matrix)';  Features = 'mmap';                        Tests = @('validate_anacrousis');                            Device = 'GPU';     Models = 'meta-llama/Llama-3.2-1B (gated)'; FullOnly = $true }
)

# --- Read prior per-entry state back out of RESURRECTION.md (Name -> {Date, Outcome}) ---
function Read-PriorState {
    param([string]$Path)
    $state = @{}
    if (-not (Test-Path -LiteralPath $Path)) { return $state }
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -notmatch '^\|') { continue }
        $cells = ($line.Trim().Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
        if ($cells.Count -lt 5) { continue }
        $name = $cells[0]
        if ($name -eq 'Test' -or $name -match '^-+$') { continue }
        $state[$name] = @{ Date = $cells[3]; Outcome = $cells[4] }
    }
    return $state
}

function Format-Age {
    param([int]$Days)
    if ($Days -ge 60) { return "~$([math]::Round($Days / 30)) months" }
    return "$Days days"
}

# --- -Status: report staleness and exit; runs nothing ---
if ($Status) {
    $prior = Read-PriorState $mdPath
    $today = Get-Date
    $stale = @()
    $failing = @()
    $oldestDays = -1
    $oldestName = ''
    foreach ($e in $entries) {
        $s = $prior[$e.Name]
        $date = if ($s) { $s.Date } else { 'never' }
        $parsed = $date -as [datetime]
        if ($null -eq $parsed) {
            $stale += "$($e.Name): never"
            $oldestDays = [int]::MaxValue; $oldestName = $e.Name
        }
        else {
            $days = ($today - $parsed).Days
            if ($days -gt $oldestDays) { $oldestDays = $days; $oldestName = $e.Name }
            if ($days -gt $StaleDays) { $stale += "$($e.Name): $(Format-Age $days)" }
        }
        if ($s -and $s.Outcome -match 'FAIL') { $failing += $e.Name }
    }

    Write-Host 'Oracle parity suite (scripts/resurrect.ps1):' -ForegroundColor Cyan
    if ($oldestDays -eq [int]::MaxValue) {
        Write-Host "  Oldest: never verified ($oldestName)" -ForegroundColor DarkYellow
    }
    else {
        Write-Host "  Oldest verified: $(Format-Age $oldestDays) ago ($oldestName)"
    }
    if ($failing.Count -gt 0) {
        Write-Host "  FAILING: $($failing -join ', ')" -ForegroundColor Red
    }
    if ($stale.Count -eq 0) {
        Write-Host "  All $($entries.Count) oracles verified within $StaleDays days." -ForegroundColor Green
    }
    else {
        Write-Host "  Stale (> $StaleDays days) or never run: $($stale.Count) of $($entries.Count) — run scripts/resurrect.ps1 [-Full] to refresh" -ForegroundColor DarkYellow
        foreach ($item in $stale) { Write-Host "    - $item" -ForegroundColor DarkYellow }
    }
    return
}

# --- Run path ---
Push-Location $repoRoot
try {
    if ($Quick) { $selected = @($entries | Where-Object { $_.Quick }); $tier = 'Quick' }
    elseif ($Full) { $selected = $entries; $tier = 'Full' }
    else { $selected = @($entries | Where-Object { -not $_.FullOnly }); $tier = 'Default' }

    Write-Host "Resurrecting oracle suite - tier: $tier ($($selected.Count) of $($entries.Count) entries)" -ForegroundColor Cyan
    $rustc = (rustc --version)

    $resultByName = @{}
    foreach ($e in $selected) {
        $cargoArgs = @('test', '--features', $e.Features, '--release')
        foreach ($t in $e.Tests) { $cargoArgs += @('--test', $t) }
        $cargoArgs += '--'
        $cargoArgs += @('--include-ignored', '--test-threads=1', '--nocapture')
        $skipNote = ''
        if ($e.SlowSkip -and -not $Full) {
            $cargoArgs += @('--skip', $e.SlowSkip)
            $skipNote = " (skipping '$($e.SlowSkip)')"
        }

        Write-Host "`n=== $($e.Name)$skipNote ===" -ForegroundColor Yellow
        Write-Host "cargo $($cargoArgs -join ' ')" -ForegroundColor DarkGray

        cargo @cargoArgs 2>&1 | Tee-Object -Variable teeOut
        $code = $LASTEXITCODE

        $skipped = @($teeOut | Select-String -SimpleMatch 'SKIP').Count -gt 0
        if ($code -ne 0) { $entryStatus = 'FAIL'; $color = 'Red' }
        elseif ($skipped) { $entryStatus = 'SKIP'; $color = 'DarkYellow' }
        else { $entryStatus = 'PASS'; $color = 'Green' }
        Write-Host "  -> $entryStatus" -ForegroundColor $color

        $resultByName[$e.Name] = $entryStatus
    }

    # --- Rewrite RESURRECTION.md, preserving per-entry dates for what we didn't
    #     re-verify. "Last verified" advances ONLY on a real PASS: a SKIP (model
    #     uncached) or FAIL is not a fresh verification, so it keeps the old date. ---
    $prior = Read-PriorState $mdPath
    $today = Get-Date -Format 'yyyy-MM-dd'
    $rowLines = $entries | ForEach-Object {
        $s = $resultByName[$_.Name]
        if ($null -ne $s) {
            $outcome = switch ($s) {
                'PASS' { '✅ PASS' }
                'SKIP' { '⏭️ SKIP (uncached)' }
                'FAIL' { '❌ FAIL' }
                default { $s }
            }
            if ($s -eq 'PASS') { $verified = $today }
            else { $verified = if ($prior[$_.Name]) { $prior[$_.Name].Date } else { 'never' } }
        }
        else {
            $p = $prior[$_.Name]
            if ($p) { $verified = $p.Date; $outcome = $p.Outcome }
            else { $verified = 'never'; $outcome = '— never' }
        }
        "| $($_.Name) | $($_.Models) | $($_.Device) | $verified | $outcome |"
    }

    $stampNow = Get-Date -Format 'yyyy-MM-dd HH:mm'
    $header = @(
        '# Oracle-suite resurrection log'
        ''
        'The oracle/parity tests below are `#[ignore]`d — they need cached'
        'HuggingFace models (some gated) and, for many, a CUDA GPU with ≥ 16 GiB'
        'VRAM — so GitHub CI never runs them. This file records, **per test**, when'
        'each last **passed** its oracle comparison locally.'
        ''
        '**Refresh it** with [`scripts/resurrect.ps1`](scripts/resurrect.ps1):'
        ''
        '```'
        'scripts/resurrect.ps1          # default: all but the two slow outliers (~40-50 min)'
        'scripts/resurrect.ps1 -Quick   # cheap ungated CPU encoder-parity smoke (~5 min)'
        'scripts/resurrect.ps1 -Full    # + Mistral-7B CPU forward + anacrousis 28x15 (~1.5-3 h)'
        'scripts/resurrect.ps1 -Status  # report staleness (runs nothing); -StaleDays N sets the threshold'
        '```'
        ''
        '"Last verified" = the last date this entry **passed** (a `⏭️ SKIP` /'
        '`❌ FAIL` does not advance it). `never` = not yet verified on this machine.'
        'Staleness is per-entry, so a `-Quick` run only refreshes its two rows.'
        ''
        "- **Last run:** $stampNow — tier **$tier**"
        "- **Toolchain:** $rustc"
        ''
        '| Test | Models | Device(s) | Last verified | Outcome |'
        '|---|---|---|---|---|'
    )
    $md = ($header + $rowLines) -join "`n"
    Set-Content -Path $mdPath -Value $md -Encoding utf8
    Write-Host "`nStamped RESURRECTION.md ($tier, $stampNow)" -ForegroundColor Cyan

    $failed = @($resultByName.Values | Where-Object { $_ -eq 'FAIL' }).Count
    if ($failed -gt 0) {
        Write-Host "$failed entr$(if ($failed -eq 1) {'y'} else {'ies'}) FAILED — see output above." -ForegroundColor Red
        exit 1
    }
    Write-Host 'Done.' -ForegroundColor Green
}
finally {
    Pop-Location
}
