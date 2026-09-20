# IBVAP Rust Command Center Launcher (Cross-Device / Portable)
param(
    [Alias("r", "Requirements", "reqs")]
    [switch]$InstallReqs
)

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  IBVAP Command Center - Multi-Device Launcher   " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# 1. Ensure Cargo / Rust is in PATH
$cargoFound = $false
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $cargoFound = $true
} else {
    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if (Test-Path (Join-Path $cargoBin "cargo.exe")) {
        $env:PATH = "$cargoBin;" + $env:PATH
        $cargoFound = $true
        Write-Host "[INFO] Added Cargo to PATH from: $cargoBin" -ForegroundColor Gray
    }
}

if (-not $cargoFound) {
    Write-Host "[ERROR] Rust toolchain (cargo) is not installed or not in PATH." -ForegroundColor Red
    Write-Host "To install Rust, run:" -ForegroundColor Yellow
    Write-Host "    winget install Rustlang.Rustup" -ForegroundColor White
    Write-Host "Or download the official installer from: https://rustup.rs" -ForegroundColor White
    Write-Host "After installation, close and reopen your PowerShell window." -ForegroundColor Yellow
    exit 1
}

# 2. Dynamically discover Python environment
$detectedPython = $null
$detectedScripts = $null

# Check virtual environments first
$candidateVenvs = @(
    (Join-Path $PSScriptRoot ".venv"),
    (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv")
)

foreach ($venv in $candidateVenvs) {
    $pyPath = Join-Path $venv "Scripts\python.exe"
    if (Test-Path $pyPath) {
        $detectedPython = $pyPath
        $detectedScripts = Join-Path $venv "Scripts"
        $env:VIRTUAL_ENV = $venv
        Write-Host "[INFO] Detected Virtualenv: $venv" -ForegroundColor Green
        break
    }
}

# If no venv, check system Python
if (-not $detectedPython) {
    $cmdPy = Get-Command python -ErrorAction SilentlyContinue
    if ($cmdPy -and (Test-Path $cmdPy.Source)) {
        $detectedPython = $cmdPy.Source
        $pyHome = Split-Path $detectedPython -Parent
        $detectedScripts = Join-Path $pyHome "Scripts"
        Write-Host "[INFO] Detected System Python: $detectedPython" -ForegroundColor Green
    }
}

# Fallback: scan standard Windows user/system installation folders
if (-not $detectedPython) {
    $searchDirs = @(
        "$env:LOCALAPPDATA\Programs\Python\Python*",
        "$env:ProgramFiles\Python*",
        "$env:SystemDrive\Python*"
    )
    foreach ($pattern in $searchDirs) {
        $matched = Get-Item -Path $pattern -ErrorAction SilentlyContinue | Sort-Object -Descending -Property Name
        foreach ($item in $matched) {
            $testExe = Join-Path $item.FullName "python.exe"
            if (Test-Path $testExe) {
                $detectedPython = $testExe
                $detectedScripts = Join-Path $item.FullName "Scripts"
                Write-Host "[INFO] Located Python installation: $detectedPython" -ForegroundColor Green
                break
            }
        }
        if ($detectedPython) { break }
    }
}

# Prepend Python to PATH so PyO3 and C-extensions find the matching DLLs
if ($detectedPython) {
    $pyHome = Split-Path $detectedPython -Parent
    $pathAdditions = @($pyHome)
    if ($detectedScripts -and (Test-Path $detectedScripts)) {
        $pathAdditions += $detectedScripts
    }
    $env:PATH = ($pathAdditions -join ";") + ";" + $env:PATH
    $env:PYO3_PYTHON = $detectedPython
    $env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"
    Write-Host "[INFO] PyO3 target Python set to: $detectedPython" -ForegroundColor Gray
} else {
    Write-Host "[WARN] No Python installation detected automatically; relying on default environment." -ForegroundColor Yellow
}

# 3. Optional automatic requirements installation
if ($InstallReqs) {
    $reqFile = if (Test-Path "$PSScriptRoot\requirements.txt") {
        "$PSScriptRoot\requirements.txt"
    } elseif (Test-Path "$PSScriptRoot\..\requirements.txt") {
        "$PSScriptRoot\..\requirements.txt"
    } else {
        $null
    }

    if ($reqFile -and $detectedPython) {
        Write-Host "[INFO] Installing/updating requirements into environment..." -ForegroundColor Cyan
        & "$detectedPython" -m pip install -r "$reqFile"
    }
}

# 4. Launch Cargo
Set-Location -Path $PSScriptRoot
Write-Host "[INFO] Starting IBVAP Rust Command Center (cargo run --release)..." -ForegroundColor Cyan
cargo run --release
