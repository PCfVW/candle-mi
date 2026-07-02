# resurrect.ps1 — run the #[ignore]d oracle/parity suite locally and stamp
# RESURRECTION.md with a per-test "last verified" record.
#
# The oracle/parity tests are #[ignore]d because they need cached HuggingFace
# models (some gated) and, for many, a CUDA GPU with >= 16 GiB VRAM — so they
# never run in GitHub CI (CPU-only runners, no cached models). This runner
# exercises them on a workstation where the models are cached, giving the
# #[ignore]d suite the automated safety net the 2026-07-01 audit (§1.1) flagged
# as missing, and writes a committed staleness log.
#
# Tiers:
#   -Quick    cheap ungated CPU encoder-parity only (clt_qwen3, plt_gemma) — ~5 min
#   (default) everything EXCEPT the two slow outliers (Mistral-7B CPU forward and
#             the anacrousis 28x15 matrix) — ~40-50 min; hits every model + GPU
#             path once
#   -Full     literally everything, incl. the two slow outliers — ~1.5-3 h
#
# Device policy: GPU tests run on the GPU; CPU-parity tests (*_cpu, plt,
# clt_qwen3) run on CPU by design (they verify the CPU F32 path). Runs with the
# default feature set so `cuda` stays ON; each entry adds its extra feature(s).

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Full
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    # Entry list. `Features` are ADDED to the default set (cuda,transformer).
    # Quick=$true  -> included in -Quick; FullOnly=$true -> only in -Full;
    # SlowSkip     -> a `--skip <pat>` applied in the default tier (dropped in -Full).
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

    if ($Quick) {
        $selected = @($entries | Where-Object { $_.Quick })
        $tier = 'Quick'
    }
    elseif ($Full) {
        $selected = $entries
        $tier = 'Full'
    }
    else {
        $selected = @($entries | Where-Object { -not $_.FullOnly })
        $tier = 'Default'
    }

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
        if ($code -ne 0) { $status = 'FAIL'; $color = 'Red' }
        elseif ($skipped) { $status = 'SKIP'; $color = 'DarkYellow' }
        else { $status = 'PASS'; $color = 'Green' }
        Write-Host "  -> $status" -ForegroundColor $color

        $resultByName[$e.Name] = $status
    }

    # --- Rewrite RESURRECTION.md from scratch (avoids fragile in-place edits) ---
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
    $rowLines = $entries | ForEach-Object {
        $r = $resultByName[$_.Name]
        $outcome = if ($null -ne $r) {
            switch ($r) {
                'PASS' { '✅ PASS' }
                'SKIP' { '⏭️ SKIP (uncached)' }
                'FAIL' { '❌ FAIL' }
                default { $r }
            }
        }
        else { "— not run ($tier)" }
        "| $($_.Name) | $($_.Models) | $($_.Device) | $outcome |"
    }

    $header = @(
        '# Oracle-suite resurrection log'
        ''
        'The oracle/parity tests below are `#[ignore]`d — they need cached'
        'HuggingFace models (some gated) and, for many, a CUDA GPU with ≥ 16 GiB'
        'VRAM — so GitHub CI never runs them. This file records when each was last'
        'exercised locally.'
        ''
        '**Refresh it** with [`scripts/resurrect.ps1`](scripts/resurrect.ps1):'
        ''
        '```'
        'scripts/resurrect.ps1          # default: all but the two slow outliers (~40-50 min)'
        'scripts/resurrect.ps1 -Quick   # cheap ungated CPU encoder-parity smoke (~5 min)'
        'scripts/resurrect.ps1 -Full    # + Mistral-7B CPU forward + anacrousis 28x15 (~1.5-3 h)'
        '```'
        ''
        '`✅ PASS` = ran and matched its oracle; `⏭️ SKIP` = model/GPU not available'
        '(the test printed SKIP and returned); `❌ FAIL` = a real mismatch —'
        'investigate. `— not run` = outside the tier of the last run.'
        ''
        "- **Last run:** $stamp — tier **$tier**"
        "- **Toolchain:** $rustc"
        ''
        '| Test | Models | Device(s) | Outcome |'
        '|---|---|---|---|'
    )
    $md = ($header + $rowLines) -join "`n"
    Set-Content -Path (Join-Path $repoRoot 'RESURRECTION.md') -Value $md -Encoding utf8
    Write-Host "`nStamped RESURRECTION.md ($tier, $stamp)" -ForegroundColor Cyan

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
