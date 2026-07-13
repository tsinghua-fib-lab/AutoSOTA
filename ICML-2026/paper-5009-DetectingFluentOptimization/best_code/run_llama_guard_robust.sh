#!/usr/bin/env bash
set -euo pipefail

# Robust wrapper for llama_guard_baseline.py with automatic retry on failure
# Usage: ./run_llama_guard_robust.sh [python script args...]
#
# Example:
#   ./run_llama_guard_robust.sh --input-csv data/llama-7b_dataset.csv \
#                                --output-csv results/lg_results.csv \
#                                --models lg1 lg2 lg3 \
#                                --batch-size 2

MAX_RETRIES=10
RETRY_DELAY=30  # seconds to wait between retries

# Get script directory
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Default batch size (can be overridden in args)
INITIAL_BATCH_SIZE=4

echo "=========================================="
echo "Robust LlamaGuard Runner"
echo "Max retries: $MAX_RETRIES"
echo "Retry delay: ${RETRY_DELAY}s"
echo "=========================================="
echo ""

# Pass all arguments to the Python script
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "[Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES] Starting llama_guard_baseline.py..."
    echo "Args: $@"
    echo ""

    if python compute/llama_guard_baseline.py "$@"; then
        echo ""
        echo "=========================================="
        echo "SUCCESS! All models completed."
        echo "=========================================="
        exit 0
    else
        EXIT_CODE=$?
        RETRY_COUNT=$((RETRY_COUNT + 1))

        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo ""
            echo "=========================================="
            echo "FAILURE (exit code: $EXIT_CODE)"
            echo "Retrying in ${RETRY_DELAY}s..."
            echo "Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
            echo "=========================================="
            echo ""

            # Note: GPU reset requires root privileges, so we skip it
            # The Python script will auto-select an available GPU instead
            sleep $RETRY_DELAY
        else
            echo ""
            echo "=========================================="
            echo "FATAL: Max retries ($MAX_RETRIES) exceeded"
            echo "Check logs and checkpoints for recovery"
            echo "=========================================="
            exit $EXIT_CODE
        fi
    fi
done
