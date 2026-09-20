# IBVAP Rust Command Center Launcher (Windows)
# Ensures Python 3.12 is prioritized in PATH so PyO3 matches the .venv C-extensions
$env:PATH = "C:\Users\Saksham\AppData\Local\Programs\Python\Python312;C:\Users\Saksham\AppData\Local\Programs\Python\Python312\Scripts;" + $env:PATH
cargo run --release
