#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define paths
PYTHON_BIN="$SCRIPT_DIR/src/modules/generation_runner/venv/bin/python"
RUNNER_SCRIPT="$SCRIPT_DIR/src/modules/generation_runner/main.py"
LOG_FILE="$SCRIPT_DIR/pipeline.log"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Flags
CLEAN_UP=false
STEPS_ARG="all"

# Help
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        echo "Usage: $(basename "$0") [--steps STEPS] [--clean-up]"
        echo "  --steps      Steps to execute (default: all)"
        echo "  --clean-up   Delete the root output directory after a successful run"
        exit 0
    fi
done

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            STEPS_ARG="$2"
            shift 2
            ;;
        --clean-up)
            CLEAN_UP=true
            shift
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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
"$PYTHON_BIN" "$RUNNER_SCRIPT" --step "$STEPS_ARG" --log-file "$LOG_FILE" || true
RUN_STATUS=$?

# Conditional cleanup (only on success if requested)
if [[ $RUN_STATUS -eq 0 && "$CLEAN_UP" == "true" ]]; then
    echo "Cleanup requested: deleting output directory at $OUTPUT_DIR"
    if [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" == "$SCRIPT_DIR/output" && -d "$OUTPUT_DIR" ]]; then
        rm -rf "$OUTPUT_DIR" || exit 1
        echo "Output directory deleted: $OUTPUT_DIR"
    else
        echo "Skip cleanup: invalid or missing output directory ($OUTPUT_DIR)"
    fi
else
    echo "Skipping cleanup (status: $RUN_STATUS, requested: $CLEAN_UP)"
fi

echo ""
echo "Pipeline generation complete. Log: $LOG_FILE"
exit "$RUN_STATUS"
