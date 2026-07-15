#!/bin/bash
# Script to run ResNet memory profiling experiment with dummy CIFAR-10 data

set -e

NUM_CLIENTS=3

# Create output directory
OUTPUT_DIR="./memory_profiles/resnet_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting ResNet memory profiling experiment with dummy CIFAR-10 data..."
echo "Output directory: $OUTPUT_DIR"
echo "Number of clients: $NUM_CLIENTS"

run_resnet_experiment() {
    local VERSION=$1
    local USE_OPTIMIZED=$2
    local VERSION_FLAG=""

    if [ "$USE_OPTIMIZED" = true ]; then
        VERSION_FLAG="--use_optimized_version"
    fi

    echo "============================================"
    echo "Running $VERSION version ResNet experiment..."
    echo "============================================"

    # Start server
    echo "Starting $VERSION server..."
    python memory_profiling/run_server_memray.py \
        --config ./memory_profiling/configs/server_resnet_dummy.yaml \
        --output-dir "$OUTPUT_DIR" \
        --num_clients "$NUM_CLIENTS" \
        $VERSION_FLAG &
    SERVER_PID=$!

    sleep 5

    # Start clients dynamically
    CLIENT_PIDS=()

    for ((i=0; i<NUM_CLIENTS; i++)); do
        echo "Starting $VERSION client $i..."
        python memory_profiling/run_client_memray.py \
            --config ./memory_profiling/configs/client_1_resnet_dummy.yaml \
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

    echo "Stopping $VERSION server..."
    kill "$SERVER_PID" 2>/dev/null || true

    echo "$VERSION version ResNet experiment completed!"
    echo ""
}

# Run optimized version
run_resnet_experiment "OPTIMIZED" true

echo "============================================"
echo "ResNet experiment completed!"
echo "============================================"

echo "Running memory profile analysis..."
python memory_profiling/analyze_profiles.py "$OUTPUT_DIR"
