#!/bin/bash

# Audio Pattern Recognition Engine (APRE) - Environment Setup Script
# This script creates a virtual environment and installs all required dependencies

echo "Setting up Audio Pattern Recognition Engine (APRE) Environment..."
echo "================================================================="

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher and try again"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python version: $PYTHON_VERSION"

# Check if the version is 3.8 or higher
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✓ Python version is compatible"
else
    echo "✗ Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "apre_env" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf apre_env
fi

python3 -m venv apre_env

# Activate virtual environment
echo "Activating virtual environment..."
source apre_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo ""
echo "Installing required packages..."
pip install -r requirements.txt

# Install the package in development mode
echo ""
echo "Installing APRE package in development mode..."
pip install -e .

# Create directories for outputs
echo ""
echo "Creating output directories..."
mkdir -p outputs
mkdir -p logs

echo ""
echo "================================================================="
echo "Environment setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source apre_env/bin/activate"
echo ""
echo "To run the APRE system, use:"
echo "  python main.py"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo "================================================================="