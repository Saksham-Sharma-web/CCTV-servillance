# IBVAP Rust Command Center Launcher (Windows)
param(
    [switch]$InstallReqs
)

# 1. Ensures Python 3.12 is in PATH so PyO3 matches the .venv C-extensions
$env:PATH = "C:\Users\Saksham\AppData\Local\Programs\Python\Python312;C:\Users\Saksham\AppData\Local\Programs\Python\Python312\Scripts;" + $env:PATH

# 2. Optional automatic requirements installation
if ($InstallReqs) {
    Write-Host "[INFO] Installing/updating requirements into .venv..." -ForegroundColor Cyan
    & "$PSScriptRoot\..\.venv\Scripts\python.exe" -m pip install -r "$PSScriptRoot\..\requirements.txt"
}

# 3. Launch
cargo run --release
