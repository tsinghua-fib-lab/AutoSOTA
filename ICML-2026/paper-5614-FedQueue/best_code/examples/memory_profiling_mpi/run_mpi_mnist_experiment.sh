#!/bin/bash

# MPI MNIST Memory Profiling Experiment for APPFL
# This script runs memory profiling experiments using CNN on real MNIST dataset
# comparing original vs optimized MPI implementations

set -e

echo "=========================================="
echo "APPFL MPI MNIST Memory Profiling"
echo "=========================================="

# Configuration
NUM_PROCESSES=3  # 1 server + 2 clients
SERVER_CONFIG_FILE="./resources/configs/mnist/server_fedavg.yaml"
CLIENT_CONFIG_FILE="./resources/configs/mnist/client_1.yaml"
BASE_OUTPUT_DIR="./memory_profiles"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/mpi_mnist_${TIMESTAMP}"

# Check if config files exist
if [ ! -f "$SERVER_CONFIG_FILE" ]; then
    echo "Error: Server config file not found: $SERVER_CONFIG_FILE"
    echo "Please ensure you're running from the examples/ directory"
    exit 1
fi

if [ ! -f "$CLIENT_CONFIG_FILE" ]; then
    echo "Error: Client config file not found: $CLIENT_CONFIG_FILE"
    echo "Please ensure you're running from the examples/ directory"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Server configuration: $SERVER_CONFIG_FILE"
echo "Client configuration: $CLIENT_CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Number of MPI processes: $NUM_PROCESSES"
echo "Experiment timestamp: $TIMESTAMP"
echo ""
echo "This experiment uses:"
echo "  - CNN model (lightweight convolutional neural network)"
echo "  - Real MNIST dataset with class non-IID partitioning"
echo "  - CPU-based training for memory profiling clarity"
echo "  - Focus on lightweight federated learning memory patterns"
echo ""

# Function to run experiment and generate report
run_experiment() {
    local use_optimized=$1
    local experiment_name=$2

    echo "----------------------------------------"
    echo "Running $experiment_name experiment..."
    echo "----------------------------------------"

    # Run the MPI experiment
    if [ "$use_optimized" = true ]; then
        echo "Starting MPI experiment WITH memory optimizations..."
        mpiexec -n $NUM_PROCESSES python memory_profiling_mpi/run_mpi_memray.py \
            --server_config "$SERVER_CONFIG_FILE" \
            --client_config "$CLIENT_CONFIG_FILE" \
            --output-dir "$OUTPUT_DIR" \
            --use_optimized_version
    else
        echo "Starting MPI experiment WITHOUT memory optimizations..."
        mpiexec -n $NUM_PROCESSES python memory_profiling_mpi/run_mpi_memray.py \
            --server_config "$SERVER_CONFIG_FILE" \
            --client_config "$CLIENT_CONFIG_FILE" \
            --output-dir "$OUTPUT_DIR"
    fi

    echo "$experiment_name experiment completed."
    echo ""
}

# Function to generate analysis reports
generate_analysis() {
    echo "=========================================="
    echo "Generating Memory Analysis Reports"
    echo "=========================================="

    # Check if we have profile files
    PROFILE_FILES=$(find "$OUTPUT_DIR" -name "*.bin" | wc -l)
    if [ "$PROFILE_FILES" -eq 0 ]; then
        echo "No profile files found in $OUTPUT_DIR"
        return 1
    fi

    echo "Found $PROFILE_FILES profile files"
    echo ""

    # Generate flamegraphs for each profile
    for profile in "$OUTPUT_DIR"/*.bin; do
        if [ -f "$profile" ]; then
            base_name=$(basename "$profile" .bin)
            html_file="$OUTPUT_DIR/${base_name}.html"

            echo "Generating flamegraph for $(basename "$profile")..."

            # Check if profile file is valid (not empty)
            if [ ! -s "$profile" ]; then
                echo "  ✗ Profile file is empty: $profile"
                continue
            fi

            # Check if profile file is readable by memray
            if ! python -m memray stats "$profile" >/dev/null 2>&1; then
                echo "  ✗ Profile file is corrupted or invalid: $profile"
                continue
            fi

            # Generate flamegraph
            if python -m memray flamegraph "$profile" -o "$html_file" 2>/tmp/memray_error_$$; then
                echo "  ✓ Generated: $html_file"
            else
                echo "  ✗ Failed to generate flamegraph for $profile"
                echo "     Error details:"
                cat /tmp/memray_error_$$ | head -3 | sed 's/^/     /'
                rm -f /tmp/memray_error_$$
            fi
        fi
    done

    echo ""
    echo "Memory analysis report generation completed!"
    echo ""

    # Show summary of generated files
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/*.html 2>/dev/null || echo "No HTML files generated"
    echo ""
}

# Function to show comparison results
show_comparison() {
    echo "=========================================="
    echo "Memory Usage Comparison"
    echo "=========================================="

    echo "Comparing memory profiles..."
    echo ""

    # Find matching original and optimized profiles
    for rank in 0 1 2; do
        original_profile="$OUTPUT_DIR/mpi_rank_${rank}_original_memory_profile.bin"
        optimized_profile="$OUTPUT_DIR/mpi_rank_${rank}_optimized_memory_profile.bin"

        if [ -f "$original_profile" ] && [ -f "$optimized_profile" ]; then
            echo "Rank $rank Memory Statistics:"
            echo "  Original version:"
            python -m memray stats "$original_profile" 2>/dev/null | head -10 2>/dev/null || echo "    (stats extraction failed - check flamegraph instead)"
            echo ""
            echo "  Optimized version:"
            python -m memray stats "$optimized_profile" 2>/dev/null | head -10 2>/dev/null || echo "    (stats extraction failed - check flamegraph instead)"
            echo ""
            echo "----------------------------------------"
        fi
    done
}

# Main execution
main() {
    # Check dependencies
    if ! command -v mpiexec &> /dev/null; then
        echo "Error: mpiexec not found. Please install MPI (e.g., OpenMPI or MPICH)"
        exit 1
    fi

    if ! python -c "import memray" &> /dev/null; then
        echo "Error: memray not found. Please install: pip install memray"
        exit 1
    fi

    if ! python -c "import mpi4py" &> /dev/null; then
        echo "Error: mpi4py not found. Please install: pip install mpi4py"
        exit 1
    fi

    echo "All dependencies found ✓"
    echo ""

    # Run experiments
    run_experiment false "Original MPI"
    sleep 2  # Brief pause between experiments
    run_experiment true "Optimized MPI"

    # Generate comprehensive analysis (similar to gRPC memory profiling)
    echo "=========================================="
    echo "Generating Comprehensive Analysis Report"
    echo "=========================================="

    if python memory_profiling_mpi/generate_comprehensive_results.py "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/analysis" 2>/dev/null; then
        echo "  ✓ Comprehensive analysis completed successfully!"
        echo "  📊 Check $OUTPUT_DIR/analysis/ for detailed results"
    else
        echo "  ⚠ Comprehensive analysis failed, falling back to basic analysis..."
        # Fallback to basic analysis
        generate_analysis
    fi

    # Show comparison if both profiles exist
    show_comparison

    echo "=========================================="
    echo "MNIST Experiment Complete!"
    echo "=========================================="
    echo ""
    echo "Experiment timestamp: $TIMESTAMP"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "No files generated"
    echo ""
    echo "To view detailed memory profiles:"
    echo "  🔥 Interactive flamegraphs: Open .html files in browser"
    echo "  📊 Comparison plots: Check $OUTPUT_DIR/analysis/*.png"
    echo "  📈 Statistical data: $OUTPUT_DIR/analysis/*.csv"
    echo "  📝 Comprehensive report: $OUTPUT_DIR/analysis/mpi_memory_optimization_report.txt"
    echo ""
    echo "Generated comprehensive analysis similar to gRPC memory profiling!"
    echo ""
    echo "Key files:"
    echo "  - mpi_rank_0_*.html: Server memory flamegraphs"
    echo "  - mpi_rank_1_*.html: Client 1 memory flamegraphs"
    echo "  - mpi_rank_2_*.html: Client 2 memory flamegraphs"
    echo "  - analysis/: Complete analysis results with plots and reports"
    echo ""
    echo "Compare original vs optimized versions to see memory improvements!"
    echo "For advanced analysis:"
    echo "  python memory_profiling_mpi/generate_comprehensive_results.py $OUTPUT_DIR"
    echo ""
    echo "MNIST-specific insights to look for:"
    echo "  • CNN model has different memory patterns than ResNet"
    echo "  • CPU-based training shows clearer memory allocation patterns"
    echo "  • Smaller model size highlights optimization impact better"
    echo "  • MNIST data loading patterns vs CIFAR-10 differences"
}

# Run main function
main
