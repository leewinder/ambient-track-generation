#!/bin/bash
set -e  # Exit on any error

echo "Installing Image Generation Script"
echo "=================================="

# Check if pyenv is installed (REQUIRED)
if ! command -v pyenv &> /dev/null; then
    echo "ERROR: pyenv is required but not found!"
    echo "   Install pyenv from: https://github.com/pyenv/pyenv"
    echo "   This script cannot continue without pyenv."
    exit 1
fi

echo "pyenv detected"

# Check if the required Python version is installed
required_version=$(cat .python-version)
echo "Required Python version: $required_version"

if pyenv versions --bare | grep -q "^$required_version$"; then
    echo "Python $required_version is already installed"
else
    echo "Installing Python $required_version..."
    pyenv install $required_version
fi

# Set local Python version
echo "Setting local Python version to $required_version..."
pyenv local $required_version

# Refresh shell to use new Python version
eval "$(pyenv init -)"

# Verify Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Current Python version: $python_version"

if [[ ! "$python_version" =~ ^3\.11\. ]]; then
    echo "Wrong Python version. Expected 3.11.x, got $python_version"
    exit 1
fi

# Navigate to image generation folder
cd "scripts/01 - Generate Images"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (use pinned versions for this stage)
echo "Installing PyTorch..."
pip install -r requirements/requirements-torch.txt

# Install other requirements for this stage
echo "Installing other dependencies..."
pip install -r requirements/requirements-all.txt

echo ""
echo "Installation complete!"
echo "Activate the virtual environment with: source venv/bin/activate"
echo "Then run: python generate_image.py"