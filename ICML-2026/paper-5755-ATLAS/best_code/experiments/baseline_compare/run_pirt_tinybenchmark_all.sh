#!/bin/bash
# Run p-IRT estimation for all TinyBenchmarks datasets

BASE_DIR="experiments/baseline_compare"

# Load R module
module load R 2>/dev/null || true

echo "======================================================================"
echo "Running p-IRT Estimation for All TinyBenchmarks Datasets"
echo "======================================================================"
echo ""

# TinyBenchmarks datasets
BENCHMARKS=("arc" "gsm8k" "hellaswag" "truthfulqa" "winogrande")

for benchmark in "${BENCHMARKS[@]}"; do
  echo "----------------------------------------------------------------------"
  echo "Benchmark: $benchmark"
  echo "----------------------------------------------------------------------"
  
  # Check if selected items file exists
  SELECTED_FILE="${BASE_DIR}/tiny${benchmark}_item_ids.csv"
  
  if [ ! -f "$SELECTED_FILE" ]; then
    echo "⚠ Skipping: Selected items file not found"
    echo "  Expected: $SELECTED_FILE"
    echo "  Please run generate_all_tinybenchmark_ids.py to generate this file."
    echo ""
    continue
  fi
  
  # Check if item parameters file exists
  ITEM_PARAMS="${benchmark}/irt_item_parameters_combined.csv"
  if [ ! -f "$ITEM_PARAMS" ]; then
    echo "⚠ Skipping: Item parameters file not found"
    echo "  Expected: $ITEM_PARAMS"
    echo ""
    continue
  fi
  
  # Check if response matrix exists
  RESPONSE_FILE="data/gaussian_sampled_${benchmark}_response_matrix_test.csv"
  if [ ! -f "$RESPONSE_FILE" ]; then
    echo "⚠ Skipping: Response matrix not found"
    echo "  Expected: $RESPONSE_FILE"
    echo ""
    continue
  fi
  
  echo "✓ Found required files"
  
  # Step 1: Compute p-IRT estimates
  echo ""
  echo "[1/2] Computing p-IRT accuracy estimates..."
  Rscript ${BASE_DIR}/pirt_tinybenchmark.r --benchmark=$benchmark
  
  if [ $? -ne 0 ]; then
    echo "✗ Error computing p-IRT estimates for $benchmark"
    echo ""
    continue
  fi
  
  # Step 2: Compare with actual
  echo ""
  echo "[2/2] Comparing p-IRT estimates with actual accuracy..."
  Rscript ${BASE_DIR}/compare_pirt_tinybenchmark.r --benchmark=$benchmark
  
  if [ $? -ne 0 ]; then
    echo "✗ Error comparing results for $benchmark"
    echo ""
    continue
  fi
  
  echo "✓ Complete: $benchmark"
  echo ""
done

echo "======================================================================"
echo "All benchmarks processed!"
echo "======================================================================"
echo ""
echo "Summary of outputs:"
echo "  - baseline_compare/pirt_tiny{benchmark}.csv"
echo "  - baseline_compare/pirt_tiny{benchmark}_vs_actual.csv"
echo "  - baseline_compare/pirt_tiny{benchmark}_vs_actual.png"
echo ""
