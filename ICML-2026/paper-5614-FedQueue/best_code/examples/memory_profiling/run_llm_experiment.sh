#!/bin/bash
# Script to run LLM memory profiling experiment with dummy data

set -e

NUM_CLIENTS=3
MODEL_NAME="meta-llama/Llama-3.1-8B"

# Create output directory
OUTPUT_DIR="./memory_profiles/llm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting LLM memory profiling experiment with dummy data..."
echo "Output directory: $OUTPUT_DIR"
echo "Number of clients: $NUM_CLIENTS"

run_llm_experiment() {
    local VERSION=$1
    local USE_OPTIMIZED=$2
    local VERSION_FLAG=""

    if [ "$USE_OPTIMIZED" = true ]; then
        VERSION_FLAG="--use_optimized_version"
    fi

    echo "============================================"
    echo "Running $VERSION version LLM experiment..."
    echo "============================================"

    # Start server
    echo "Starting $VERSION server..."
    python memory_profiling/run_server_memray.py \
        --config ./memory_profiling/configs/server_llm_dummy.yaml \
        --output-dir "$OUTPUT_DIR" \
        --num_clients "$NUM_CLIENTS" \
        --model_name "$MODEL_NAME" \
        $VERSION_FLAG &
    SERVER_PID=$!

    sleep 5

    # Start clients dynamically
    CLIENT_PIDS=()

    for ((i=0; i<NUM_CLIENTS; i++)); do
        echo "Starting $VERSION client $i..."
        python memory_profiling/run_client_memray.py \
            --config ./memory_profiling/configs/client_1_llm_dummy.yaml \
            --output-dir "$OUTPUT_DIR" \
            --num_clients "$NUM_CLIENTS" \
            --client_idx "$i" \
            $VERSION_FLAG &
        CLIENT_PIDS+=($!)
    done

    echo "Waiting for $VERSION clients to complete..."

    for PID in "${CLIENT_PIDS[@]}"; do
        wait "$PID"
    done

    sleep 30 # Ensure server has time to finish processing

    echo "Stopping $VERSION server..."
    kill "$SERVER_PID" 2>/dev/null || true

    echo "$VERSION version LLM experiment completed!"
    echo ""
}

# Run optimized version
run_llm_experiment "OPTIMIZED" true

echo "============================================"
echo "LLM experiment completed!"
echo "============================================"

echo "Running memory profile analysis..."
python memory_profiling/analyze_profiles.py "$OUTPUT_DIR"
