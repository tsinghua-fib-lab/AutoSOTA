#!/usr/bin/env Rscript
# Summarize TinyBenchmark p-IRT results across all datasets

cat("\n======================================================================\n")
cat("TinyBenchmark p-IRT Performance Summary\n")
cat("======================================================================\n\n")

BASE_DIR <- "experiments/baseline_compare"
DATASETS <- c("arc", "gsm8k", "hellaswag", "truthfulqa", "winogrande")

# Initialize results data frame
results <- data.frame(
  Dataset = character(),
  Models = integer(),
  Subset_Size = integer(),
  Full_Size = integer(),
  Reduction = numeric(),
  MAE = numeric(),
  RMSE = numeric(),
  Correlation = numeric(),
  R_squared = numeric(),
  stringsAsFactors = FALSE
)

# Read results for each dataset
for (benchmark in DATASETS) {
  file_path <- paste0(BASE_DIR, "/pirt_tiny", benchmark, "_vs_actual.csv")
  
  if (file.exists(file_path)) {
    data <- read.csv(file_path, stringsAsFactors = FALSE)
    
    # Calculate metrics
    n_models <- nrow(data)
    mae <- mean(abs(data$error), na.rm = TRUE)
    rmse <- sqrt(mean(data$squared_error, na.rm = TRUE))
    correlation <- cor(data$pirt_accuracy, data$actual_accuracy, use = "complete.obs")
    r_squared <- correlation^2
    
    # Get configuration from first row
    subset_size <- data$n_subset_items[1]
    full_size <- data$n_all_items[1]
    reduction <- subset_size / full_size
    
    # Add to results
    results <- rbind(results, data.frame(
      Dataset = benchmark,
      Models = n_models,
      Subset_Size = subset_size,
      Full_Size = full_size,
      Reduction = reduction,
      MAE = mae,
      RMSE = rmse,
      Correlation = correlation,
      R_squared = r_squared,
      stringsAsFactors = FALSE
    ))
  }
}

# Print summary table
cat("Performance Metrics:\n")
cat("----------------------------------------------------------------------\n")
print(results, row.names = FALSE, digits = 4)

cat("\n======================================================================\n")
cat("Summary Statistics:\n")
cat("======================================================================\n")
cat("Mean MAE across datasets:", round(mean(results$MAE), 4), "\n")
cat("Mean RMSE across datasets:", round(mean(results$RMSE), 4), "\n")
cat("Mean Correlation across datasets:", round(mean(results$Correlation), 4), "\n")
cat("Mean R-squared across datasets:", round(mean(results$R_squared), 4), "\n")
cat("Mean Reduction ratio:", round(mean(results$Reduction), 4), "\n")
cat("Total models evaluated:", sum(results$Models), "\n")
cat("======================================================================\n\n")

# Save summary to CSV
output_file <- paste0(BASE_DIR, "/tinybenchmark_summary.csv")
write.csv(results, output_file, row.names = FALSE)
cat("Summary saved to:", output_file, "\n\n")
