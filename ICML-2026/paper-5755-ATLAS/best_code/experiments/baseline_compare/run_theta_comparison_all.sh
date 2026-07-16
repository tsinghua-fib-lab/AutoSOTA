#!/bin/bash
# Run theta_whole vs theta_reduced comparison for all benchmarks

module load R

BASE_DIR="experiments/baseline_compare"

# List of benchmarks to process
BENCHMARKS=("arc" "gsm8k" "hellaswag" "truthfulqa" "winogrande")

echo "========================================================================"
echo "Running Theta Comparison for All Benchmarks"
echo "========================================================================"
echo ""

# Run for each benchmark
for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "------------------------------------------------------------------------"
    echo "Processing: $BENCHMARK"
    echo "------------------------------------------------------------------------"
    
    # Check if required files exist
    ITEM_PARAMS="${BENCHMARK}/irt_item_parameters_combined.csv"
    THETA_WHOLE="${BENCHMARK}/irt_person_scores_WLE_SE_test.csv"
    RESPONSE_MATRIX="data/gaussian_sampled_${BENCHMARK}_response_matrix_train.csv"
    
    if [ ! -f "$ITEM_PARAMS" ]; then
        echo "WARNING: Item parameters not found for $BENCHMARK, skipping..."
        echo "  Expected: $ITEM_PARAMS"
        continue
    fi
    
    if [ ! -f "$THETA_WHOLE" ]; then
        echo "WARNING: Theta_whole not found for $BENCHMARK, skipping..."
        echo "  Expected: $THETA_WHOLE"
        continue
    fi
    
    if [ ! -f "$RESPONSE_MATRIX" ]; then
        echo "WARNING: Response matrix not found for $BENCHMARK, skipping..."
        echo "  Expected: $RESPONSE_MATRIX"
        continue
    fi
    
    # Run the comparison script
    Rscript ${BASE_DIR}/compare_theta_whole_vs_reduced.r --benchmark=${BENCHMARK}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed $BENCHMARK"
    else
        echo "✗ Error processing $BENCHMARK"
    fi
    
    echo ""
done

echo "========================================================================"
echo "Creating Summary Table"
echo "========================================================================"

# Create a summary table combining all benchmarks
Rscript -e "
library(dplyr)

base_dir <- 'experiments/baseline_compare'
benchmarks <- c('arc', 'gsm8k', 'hellaswag', 'truthfulqa', 'winogrande')

summary_list <- list()

for (benchmark in benchmarks) {
  file_path <- paste0(base_dir, '/theta_comparison_', benchmark, '.csv')
  
  if (file.exists(file_path)) {
    data <- read.csv(file_path, stringsAsFactors = FALSE)
    
    # Calculate summary statistics
    summary_row <- data.frame(
      benchmark = benchmark,
      total_models = nrow(data),
      stringsAsFactors = FALSE
    )
    
    # TinyBenchmark
    if ('mae_tinybenchmark' %in% names(data)) {
      summary_row\$tiny_items <- unique(na.omit(data\$n_items_tinybenchmark))[1]
      summary_row\$tiny_mae_mean <- mean(data\$mae_tinybenchmark, na.rm = TRUE)
      summary_row\$tiny_mae_sd <- sd(data\$mae_tinybenchmark, na.rm = TRUE)
    }
    
    # Random100
    if ('mae_random100' %in% names(data)) {
      summary_row\$random100_items <- unique(na.omit(data\$n_items_random100))[1]
      summary_row\$random100_mae_mean <- mean(data\$mae_random100, na.rm = TRUE)
      summary_row\$random100_mae_sd <- sd(data\$mae_random100, na.rm = TRUE)
    }
    
    # Metabench Primary
    if ('mae_metabench_primary' %in% names(data)) {
      summary_row\$metabench_primary_items <- unique(na.omit(data\$n_items_metabench_primary))[1]
      summary_row\$metabench_primary_mae_mean <- mean(data\$mae_metabench_primary, na.rm = TRUE)
      summary_row\$metabench_primary_mae_sd <- sd(data\$mae_metabench_primary, na.rm = TRUE)
    }
    
    # Metabench Secondary
    if ('mae_metabench_secondary' %in% names(data)) {
      summary_row\$metabench_secondary_items <- unique(na.omit(data\$n_items_metabench_secondary))[1]
      summary_row\$metabench_secondary_mae_mean <- mean(data\$mae_metabench_secondary, na.rm = TRUE)
      summary_row\$metabench_secondary_mae_sd <- sd(data\$mae_metabench_secondary, na.rm = TRUE)
    }
    
    summary_list[[length(summary_list) + 1]] <- summary_row
  }
}

if (length(summary_list) > 0) {
  summary_df <- do.call(rbind, summary_list)
  
  # Save summary
  output_file <- paste0(base_dir, '/theta_comparison_summary.csv')
  write.csv(summary_df, output_file, row.names = FALSE)
  
  cat('\n')
  cat('======================================================================\n')
  cat('THETA COMPARISON SUMMARY - ALL BENCHMARKS\n')
  cat('======================================================================\n')
  print(summary_df)
  cat('\n')
  cat('Summary saved to:', output_file, '\n')
  cat('======================================================================\n')
}
"

echo ""
echo "========================================================================"
echo "✓ All benchmarks processed!"
echo "========================================================================"
