# release-notes.ps1 — scaffold a GitHub Release body for a candle-mi version from
# its CHANGELOG.md section, then print the `gh release create` command to run.
#
# The house style (v0.1.18 / v0.1.19) is a hand-authored narrative — title plus
# "In the crate" / "Experiments" / "Verified before tagging" sections — NOT a raw
# changelog dump. This helper only does the mechanical part: it extracts the raw
# `## [X.Y.Z]` CHANGELOG entry as starting material and wires up the gh command,
# so the prose is still yours to write.
#
# Usage:
#   scripts/release-notes.ps1 -Version 0.1.19 -Theme "candle 0.11 + the planning-floor toolkit"
#   # edit the scaffolded file into narrative, then — AFTER the crates.io Publish
#   # workflow is green — run the printed `gh release create ...` command.
#
# Real releases only: this mirrors the publish trigger (`vMAJOR.MINOR.PATCH`).
# Hyphenated milestone tags (v0.1.9-plt) neither publish nor get a GitHub Release.

[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Version,          # e.g. 0.1.19 (no leading 'v')
    [string]$Theme = '<one-line theme>',
    [string]$OutFile
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$changelog = Join-Path $repoRoot 'CHANGELOG.md'
if (-not (Test-Path -LiteralPath $changelog)) { throw "CHANGELOG.md not found at $changelog" }

# Extract the `## [X.Y.Z] ...` section body, up to the next `## [` heading.
$lines = Get-Content -LiteralPath $changelog
$start = -1
$end = $lines.Count
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "^##\s+\[$([regex]::Escape($Version))\]") { $start = $i; continue }
    if ($start -ge 0 -and $lines[$i] -match '^##\s+\[') { $end = $i; break }
}
if ($start -lt 0) { throw "No ``## [$Version]`` section found in CHANGELOG.md — is the version promoted out of [Unreleased]?" }
$section = ($lines[($start + 1)..($end - 1)] -join "`n").Trim()

if (-not $OutFile) { $OutFile = Join-Path $repoRoot "target/release-notes-v$Version.md" }
$outDir = Split-Path -Parent $OutFile
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$scaffold = @"
## candle-mi v$Version — $Theme

<!-- Rewrite the raw CHANGELOG entry below into reader-facing narrative: title +
     "In the crate" / "Experiments" / "Verified before tagging" sections (the
     v0.1.18 / v0.1.19 house style). Delete this comment and the raw block when done. -->

### Raw CHANGELOG [$Version] — starting material, rewrite then delete
$section
"@

Set-Content -Path $OutFile -Value $scaffold -Encoding utf8
Write-Host "Scaffolded release notes -> $OutFile" -ForegroundColor Green
Write-Host "`nEdit it into narrative, then — AFTER the crates.io Publish workflow is green — run:" -ForegroundColor Cyan
Write-Host "  gh release create v$Version --title `"v$Version — $Theme`" --notes-file `"$OutFile`" --verify-tag --latest" -ForegroundColor Yellow
