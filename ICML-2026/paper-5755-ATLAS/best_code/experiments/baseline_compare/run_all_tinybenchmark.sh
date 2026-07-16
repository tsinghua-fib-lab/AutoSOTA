#!/bin/bash
# Run TinyBenchmark p-IRT estimation for all datasets (except MMLU)

module load R/4.4.0/gcc/11.4.1

BASE_DIR="experiments/baseline_compare"

# Datasets to process
DATASETS="arc gsm8k hellaswag truthfulqa winogrande"

echo "========================================================================"
echo "TinyBenchmark p-IRT Estimation - All Datasets"
echo "========================================================================"
echo ""

for benchmark in $DATASETS; do
  echo "------------------------------------------------------------------------"
  echo "Processing: $benchmark"
  echo "------------------------------------------------------------------------"
  
  # Step 1: Create TinyBenchmark response matrix (if not exists)
  output_matrix="${BASE_DIR}/gaussian_sampled_${benchmark}_response_matrix_test_tinybenchmark.csv"

  if [ -f "$output_matrix" ]; then
    echo "  ✓ Response matrix already exists: $output_matrix"
  else
    echo "  Creating TinyBenchmark response matrix..."
    Rscript ${BASE_DIR}/create_tinybenchmark_response_matrix_test.r --benchmark=$benchmark
    
    if [ $? -ne 0 ]; then
      echo "  ✗ ERROR: Failed to create response matrix for $benchmark"
      continue
    fi
  fi
  
  # Step 2: Run p-IRT estimation
  echo "  Running p-IRT estimation..."
  Rscript ${BASE_DIR}/pirt_tinybenchmark.r --benchmark=$benchmark
  
  if [ $? -eq 0 ]; then
    echo "  ✓ Completed: $benchmark"
  else
    echo "  ✗ ERROR: p-IRT estimation failed for $benchmark"
  fi
  
  echo ""
done

echo "========================================================================"
echo "All TinyBenchmark p-IRT estimations complete!"
echo "========================================================================"
echo ""
echo "Results:"
ls -lh ${BASE_DIR}/pirt_tiny*.csv
