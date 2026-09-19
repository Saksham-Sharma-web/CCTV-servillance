#!/bin/bash
set -e

echo "==========================================="
echo "   IBVAP Command Center Installer/Runner   "
echo "==========================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed."
    read -p "Do you want to install it now via apt? (y/n): " choice
    if [ "$choice" == "y" ]; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv python3-pip
    else
        echo "Please install Python 3 and try again."
        exit 1
    fi
fi

# Ensure python3-venv is available
if ! python3 -m venv --help &> /dev/null; then
    echo "python3-venv is missing."
    read -p "Do you want to install it now via apt? (y/n): " choice
    if [ "$choice" == "y" ]; then
        sudo apt-get update
        sudo apt-get install -y python3-venv
    else
        echo "Please install python3-venv and try again."
        exit 1
    fi
fi

# Set up virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating isolated Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Install required Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install opencv-python ultralytics onvif-zeep numpy WSDiscovery

# Build and run Rust Application
echo "Building and launching the Rust Command Center..."
cargo run --release
