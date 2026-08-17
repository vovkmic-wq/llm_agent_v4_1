param(
    [switch]$ValidateOnly,
    [string]$TargetPath = "C:\script\20260808-LLM_Agent_file\llm_agent_v4_6_production_fixed"
)

$ErrorActionPreference = "Stop"

$source = Join-Path $PSScriptRoot "llm_agent_v4_6_production_fixed_release_20260817"
$target = [IO.Path]::GetFullPath($TargetPath)
$expectedTarget = [IO.Path]::GetFullPath(
    "C:\script\20260808-LLM_Agent_file\llm_agent_v4_6_production_fixed"
)

if ($target -ne $expectedTarget) {
    throw "Unexpected target path: $target"
}
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Release source not found: $source"
}

$forbidden = Get-ChildItem -LiteralPath $source -Force -Recurse | Where-Object {
    $_.Name -eq ".env" -or
    $_.Name -eq "__pycache__" -or
    $_.Name -in @(".pytest_cache", ".mypy_cache", ".ruff_cache", ".git") -or
    $_.Extension -eq ".pyc"
}
if ($forbidden.Count -gt 0) {
    throw "Release source contains forbidden runtime/cache files."
}

if (-not (Test-Path -LiteralPath $target -PathType Container)) {
    if ($ValidateOnly) {
        throw "Target project not found: $target"
    }
    New-Item -ItemType Directory -Path $target | Out-Null
}

if (-not $ValidateOnly) {
    Get-ChildItem -LiteralPath $source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force
    }
}

$mismatches = @()
$releaseFiles = Get-ChildItem -LiteralPath $source -File -Recurse
foreach ($sourceFile in $releaseFiles) {
    $relativePath = $sourceFile.FullName.Substring($source.Length + 1)
    $targetFile = Join-Path $target $relativePath
    if (-not (Test-Path -LiteralPath $targetFile -PathType Leaf)) {
        $mismatches += "$relativePath (missing target)"
        continue
    }
    $sourceHash = (Get-FileHash -LiteralPath $sourceFile.FullName -Algorithm SHA256).Hash
    $targetHash = (Get-FileHash -LiteralPath $targetFile -Algorithm SHA256).Hash
    if ($sourceHash -ne $targetHash) {
        $mismatches += "$relativePath (hash mismatch)"
    }
}
if ($mismatches.Count -gt 0) {
    throw "Deployment validation failed: $($mismatches -join '; ')"
}

$envFile = Join-Path $target ".env"
if (-not $ValidateOnly -and -not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    Copy-Item -LiteralPath (Join-Path $target ".env.example") -Destination $envFile
}
if (-not $ValidateOnly) {
    $lines = Get-Content -LiteralPath $envFile
    $modelFound = $false
    $workspaceFound = $false
    $updated = foreach ($line in $lines) {
        if ($line -match '^\s*LMSTUDIO_MODEL\s*=') {
            $modelFound = $true
            "LMSTUDIO_MODEL=qwen2.5-7b-instruct"
        }
        elseif ($line -match '^\s*AGENT_WORKSPACE_DIR\s*=') {
            $workspaceFound = $true
            "AGENT_WORKSPACE_DIR=C:/script/20260808-LLM_Agent_file/AGENT_WORKSPACE"
        }
        else {
            $line
        }
    }
    if (-not $modelFound) {
        $updated += "LMSTUDIO_MODEL=qwen2.5-7b-instruct"
    }
    if (-not $workspaceFound) {
        $updated += "AGENT_WORKSPACE_DIR=C:/script/20260808-LLM_Agent_file/AGENT_WORKSPACE"
    }
    Set-Content -LiteralPath $envFile -Value $updated -Encoding utf8
}

if ($ValidateOnly) {
    Write-Host "Validation PASS: $($releaseFiles.Count) release files match the target."
}
else {
    Write-Host "Deployment PASS: $($releaseFiles.Count) files copied and SHA-256 verified."
    Write-Host "Exact LM Studio model: qwen2.5-7b-instruct"
    Write-Host "Start: & '$target\.venv\Scripts\python.exe' '$target\main.py'"
}
