#!/bin/bash
set -e  # Exit on any error

echo ""
echo "==================================================="
echo "= Starting ambient track generation pipeline     ="
echo "==================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/generate.py" ]; then
    echo "ERROR: generate.py not found in src/ directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "src/venv" ]; then
    echo "ERROR: Virtual environment not found in src/venv"
    echo "Please run install.sh first to set up the environment"
    exit 1
fi

echo "Activating virtual environment..."
cd src
source venv/bin/activate

echo "Starting generation pipeline..."
echo ""

# Run the generation script with all arguments passed through
python generate.py "$@"

echo ""
echo "Generation pipeline completed!"
