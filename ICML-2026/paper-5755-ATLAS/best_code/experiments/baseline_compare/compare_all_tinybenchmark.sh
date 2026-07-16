#!/bin/bash
# Compare TinyBenchmark p-IRT estimates with actual accuracy for all datasets

module load R/4.4.0/gcc/11.4.1

# Run from ATLAS repo root

# Datasets to process
DATASETS="arc gsm8k hellaswag truthfulqa winogrande"

echo "========================================================================"
echo "TinyBenchmark p-IRT vs Actual Accuracy - All Datasets"
echo "========================================================================"
echo ""

for benchmark in $DATASETS; do
  echo "------------------------------------------------------------------------"
  echo "Comparing: $benchmark"
  echo "------------------------------------------------------------------------"
  
  pirt_file="experiments/baseline_compare/pirt_tiny${benchmark}.csv"

  if [ ! -f "$pirt_file" ]; then
    echo "  ✗ ERROR: p-IRT file not found: $pirt_file"
    echo "  Skipping $benchmark"
    echo ""
    continue
  fi
  
  echo "  Running comparison..."
  Rscript experiments/baseline_compare/compare_pirt_tinybenchmark.r --benchmark=$benchmark
  
  if [ $? -eq 0 ]; then
    echo "  ✓ Completed: $benchmark"
  else
    echo "  ✗ ERROR: Comparison failed for $benchmark"
  fi
  
  echo ""
done

echo "========================================================================"
echo "All comparisons complete!"
echo "========================================================================"
echo ""
echo "Results:"
ls -lh experiments/baseline_compare/pirt_tiny*_vs_actual.csv 2>/dev/null
