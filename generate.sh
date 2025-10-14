#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define paths
PYTHON_BIN="$SCRIPT_DIR/src/modules/generation_runner/venv/bin/python"
RUNNER_SCRIPT="$SCRIPT_DIR/src/modules/generation_runner/main.py"
LOG_FILE="$SCRIPT_DIR/pipeline.log"

# Validate required files exist
if [[ ! -f "$PYTHON_BIN" ]]; then
    echo "ERROR: Python interpreter not found at $PYTHON_BIN"
    echo "Please ensure the generation_runner module venv is properly set up"
    exit 1
fi

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
    echo "ERROR: Runner script not found at $RUNNER_SCRIPT"
    exit 1
fi

# Start pipeline generation
echo "Starting pipeline generation..."
echo "Using Python: $PYTHON_BIN"
echo "Runner script: $RUNNER_SCRIPT"
echo "Log file: $LOG_FILE"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Delete existing log file to start fresh
if [[ -f "$LOG_FILE" ]]; then
    echo "Deleting existing log file: $LOG_FILE"
    rm "$LOG_FILE"
fi

# Change to project root directory and execute the runner
cd "$SCRIPT_DIR"
"$PYTHON_BIN" "$RUNNER_SCRIPT" --step all --log-file "$LOG_FILE" || exit 1

echo ""
echo "Pipeline generation complete. Log: $LOG_FILE"
