#!/bin/bash
# Run p-IRT estimation for all Random 100 baselines

BASE_DIR="experiments/baseline_compare"

# Load R module
module load R 2>/dev/null || true

echo "======================================================================"
echo "Running p-IRT Estimation for All Random 100 Baselines"
echo "======================================================================"
echo ""

# Find all random_100_*_selected_items.csv files
for file in ${BASE_DIR}/random_100_*_selected_items.csv; do
  if [ -f "$file" ]; then
    # Extract benchmark name from filename
    # e.g., random_100_hellaswag_selected_items.csv -> hellaswag
    filename=$(basename "$file")
    benchmark=$(echo "$filename" | sed 's/random_100_//' | sed 's/_selected_items.csv//')
    
    echo "----------------------------------------------------------------------"
    echo "Benchmark: $benchmark"
    echo "----------------------------------------------------------------------"
    
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
    Rscript ${BASE_DIR}/pirt_random_100.r --benchmark=$benchmark
    
    if [ $? -ne 0 ]; then
      echo "✗ Error computing p-IRT estimates for $benchmark"
      echo ""
      continue
    fi
    
    # Step 2: Compare with actual
    echo ""
    echo "[2/2] Comparing p-IRT estimates with actual accuracy..."
    Rscript ${BASE_DIR}/compare_pirt_random_100.r --benchmark=$benchmark
    
    if [ $? -ne 0 ]; then
      echo "✗ Error comparing results for $benchmark"
      echo ""
      continue
    fi
    
    echo "✓ Complete: $benchmark"
    echo ""
  fi
done

echo "======================================================================"
echo "All benchmarks processed!"
echo "======================================================================"
echo ""
echo "Summary of outputs:"
echo "  - baseline_compare/pirt_random_100_{benchmark}.csv"
echo "  - baseline_compare/pirt_random_100_vs_actual_{benchmark}.csv"
echo "  - baseline_compare/pirt_random_100_vs_actual_{benchmark}.png"
echo ""
