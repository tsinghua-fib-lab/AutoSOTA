#!/bin/bash

# MPI ResNet Memory Profiling Experiment for APPFL
# This script runs memory profiling experiments focusing on training memory with dummy data

set -e

echo "=========================================="
echo "APPFL MPI ResNet Memory Profiling"
echo "=========================================="

# Configuration
NUM_PROCESSES=4  # 1 server + 3 clients for more realistic scenario
SERVER_CONFIG_FILE="./memory_profiling/configs/server_resnet_dummy.yaml"
CLIENT_CONFIG_FILE="./memory_profiling/configs/client_1_resnet_dummy.yaml"
BASE_OUTPUT_DIR="./memory_profiles"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/mpi_resnet_${TIMESTAMP}"

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

# Function to run experiment with specific parameters
run_experiment() {
    local use_optimized=$1
    local experiment_name=$2

    echo "----------------------------------------"
    echo "Running $experiment_name experiment..."
    echo "----------------------------------------"

    # Run the MPI experiment with timeout to prevent hanging
    if [ "$use_optimized" = true ]; then
        echo "Starting MPI ResNet experiment WITH memory optimizations..."
        if mpiexec -n $NUM_PROCESSES python memory_profiling_mpi/run_mpi_memray.py \
            --server_config "$SERVER_CONFIG_FILE" \
            --client_config "$CLIENT_CONFIG_FILE" \
            --output-dir "$OUTPUT_DIR" \
            --use_optimized_version; then
            echo "  ✓ Experiment completed successfully"
        else
            echo "  ⚠ Experiment timed out or failed"
        fi
    else
        echo "Starting MPI ResNet experiment WITHOUT memory optimizations..."
        if mpiexec -n $NUM_PROCESSES python memory_profiling_mpi/run_mpi_memray.py \
            --server_config "$SERVER_CONFIG_FILE" \
            --client_config "$CLIENT_CONFIG_FILE" \
            --output-dir "$OUTPUT_DIR"; then
            echo "  ✓ Experiment completed successfully"
        else
            echo "  ⚠ Experiment timed out or failed"
        fi
    fi

    # Check if any profile files were generated
    profile_count=$(find "$OUTPUT_DIR" -name "*.bin" 2>/dev/null | wc -l)
    echo "Generated $profile_count profile files"

    echo "$experiment_name experiment completed."
    echo ""
}

# Function to generate detailed analysis
generate_detailed_analysis() {
    echo "=========================================="
    echo "Generating Detailed Memory Analysis"
    echo "=========================================="

    # Create analysis script on the fly
    cat > "$OUTPUT_DIR/mpi_analysis.py" << 'EOF'
#!/usr/bin/env python3
import os
import subprocess
import glob

def analyze_profiles(output_dir):
    """Generate detailed analysis of MPI memory profiles"""
    profile_files = glob.glob(os.path.join(output_dir, "*.bin"))

    if not profile_files:
        print("No profile files found")
        return

    print(f"Found {len(profile_files)} profile files")

    # Group profiles by rank and type
    profiles = {}
    for profile in profile_files:
        filename = os.path.basename(profile)
        parts = filename.replace('.bin', '').split('_')

        if len(parts) >= 4:  # mpi_rank_X_type
            rank = parts[2]
            profile_type = parts[3]

            if rank not in profiles:
                profiles[rank] = {}
            profiles[rank][profile_type] = profile

    # Generate comparison for each rank
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON BY RANK")
    print("="*60)

    for rank in sorted(profiles.keys()):
        print(f"\nRank {rank} ({'Server' if rank == '0' else f'Client {rank}'}):")
        print("-" * 40)

        rank_profiles = profiles[rank]

        for profile_type in ['original', 'optimized']:
            if profile_type in rank_profiles:
                print(f"\n{profile_type.capitalize()} Version:")
                try:
                    result = subprocess.run([
                        'python', '-m', 'memray', 'stats', rank_profiles[profile_type]
                    ], capture_output=True, text=True, timeout=30)

                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        # Show summary stats
                        for line in lines[:8]:
                            if line.strip():
                                print(f"  {line}")
                    else:
                        print(f"  Error analyzing profile: {result.stderr}")
                except subprocess.TimeoutExpired:
                    print("  Analysis timed out")
                except Exception as e:
                    print(f"  Error: {e}")

        print()

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./memory_profiles"
    analyze_profiles(output_dir)
EOF

    chmod +x "$OUTPUT_DIR/mpi_analysis.py"
    python "$OUTPUT_DIR/mpi_analysis.py" "$OUTPUT_DIR"
}

# Function to generate flamegraphs with error handling
# Function to diagnose profile files
diagnose_profiles() {
    echo "=========================================="
    echo "Profile File Diagnostics"
    echo "=========================================="

    for profile in "$OUTPUT_DIR"/*.bin; do
        if [ -f "$profile" ]; then
            echo "Profile: $(basename "$profile")"
            echo "  Size: $(ls -lh "$profile" | awk '{print $5}')"
            echo "  Permissions: $(ls -l "$profile" | awk '{print $1}')"

            # Try to get basic stats
            if python -m memray stats "$profile" >/dev/null 2>&1; then
                echo "  Status: ✓ Valid memray profile"
            else
                echo "  Status: ✗ Invalid or corrupted profile"
            fi
            echo ""
        fi
    done
}

generate_flamegraphs() {
    echo "=========================================="
    echo "Generating Flamegraph Reports"
    echo "=========================================="

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

            # Generate flamegraph with timeout and error capture
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
    echo "Flamegraph generation completed!"
}

# Function to show memory optimization summary
show_optimization_summary() {
    echo "=========================================="
    echo "MPI Memory Optimization Summary"
    echo "=========================================="
    echo ""
    echo "This experiment tested the following MPI-specific optimizations:"
    echo ""
    echo "1. MPIServerCommunicator optimizations:"
    echo "   • Memory-efficient model serialization using context managers"
    echo "   • Strategic cleanup of request buffers and response data"
    echo "   • Periodic garbage collection during message processing"
    echo "   • CPU-first loading to reduce memory pressure"
    echo ""
    echo "2. MPIClientCommunicator optimizations:"
    echo "   • Optimized model transfer before sending to server"
    echo "   • Resource cleanup of communication buffers"
    echo "   • Memory-aware deserialization with immediate cleanup"
    echo ""
    echo "3. MPI Serializer optimizations:"
    echo "   • Memory-efficient BytesIO handling with context managers"
    echo "   • CPU loading for models to reduce memory pressure"
    echo "   • Automatic garbage collection after operations"
    echo ""
    echo "Key differences from gRPC optimizations:"
    echo "   • MPI-specific message passing optimization"
    echo "   • Multi-process memory profiling (one profile per rank)"
    echo "   • Optimized for MPI communication patterns"
    echo ""
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
    echo "Running ResNet memory profiling experiments..."
    run_experiment false "Original MPI ResNet"
    sleep 3  # Pause between experiments
    run_experiment true "Optimized MPI ResNet"

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
        diagnose_profiles
        generate_flamegraphs
        generate_detailed_analysis
    fi

    show_optimization_summary

    echo "=========================================="
    echo "ResNet Experiment Complete!"
    echo "=========================================="
    echo ""
    echo "Experiment timestamp: $TIMESTAMP"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "No files generated"
    echo ""
    echo "To view results:"
    echo "  🔥 Interactive flamegraphs: Open .html files in browser"
    echo "  📊 Comparison plots: Check $OUTPUT_DIR/analysis/*.png"
    echo "  📈 Statistical data: $OUTPUT_DIR/analysis/*.csv"
    echo "  📝 Comprehensive report: $OUTPUT_DIR/analysis/mpi_memory_optimization_report.txt"
    echo ""
    echo "Generated comprehensive analysis similar to gRPC memory profiling!"
    echo "Compare:"
    echo "  • Server (rank 0) vs Client (rank 1+) memory usage"
    echo "  • Original vs Optimized versions impact"
    echo "  • Memory allocation patterns and optimization effectiveness"
    echo ""
    echo "For advanced analysis:"
    echo "  python memory_profiling_mpi/generate_comprehensive_results.py $OUTPUT_DIR"
    echo "  python memory_profiling_mpi/result_profile_analysis.py --profiles-dir $OUTPUT_DIR"
}

# Run main function
main
