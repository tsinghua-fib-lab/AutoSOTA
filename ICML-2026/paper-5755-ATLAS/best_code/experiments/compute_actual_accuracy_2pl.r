#!/usr/bin/env Rscript
# Compute Actual Accuracy from Full Response Matrix
# This script computes the actual accuracy for each model from the full response matrix
# to compare against p-IRT estimates

# Parse CLI arguments
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (a in args) {
    if (grepl("^--benchmark=", a)) {
      out$benchmark <- sub("^--benchmark=", "", a)
    }
  }
  return(out)
}

.cli_args <- parse_args()
BENCHMARK <- if (!is.null(.cli_args$benchmark)) .cli_args$benchmark else "arc"

cat("\n", rep("=", 70), "\n", sep = "")
cat("Computing Actual Accuracy from Full Response Matrix\n")
cat("Benchmark:", BENCHMARK, "\n")
cat(rep("=", 70), "\n\n", sep = "")

# File paths
RESPONSE_FILE <- paste0("data/gaussian_sampled_", BENCHMARK, "_response_matrix_test.csv")
OUTPUT_FILE <- paste0("experiments/", BENCHMARK, "_2pl/actual_accuracy.csv")

# Read response matrix
cat("Reading response matrix from:", RESPONSE_FILE, "\n")
response_matrix <- read.csv(RESPONSE_FILE, stringsAsFactors = FALSE, check.names = FALSE)

# First column is model names
model_names <- response_matrix[[1]]
n_models <- nrow(response_matrix)

# Remaining columns are item responses
item_cols <- 2:ncol(response_matrix)
n_items <- length(item_cols)

cat("  Number of models:", n_models, "\n")
cat("  Number of items:", n_items, "\n\n")

# Compute accuracy for each model
cat("Computing accuracies...\n")
results <- data.frame(
  model = model_names,
  actual_accuracy = numeric(n_models),
  n_items = n_items,
  stringsAsFactors = FALSE
)

for (i in 1:n_models) {
  responses <- as.numeric(response_matrix[i, item_cols])
  
  # Remove NA values if any
  valid_responses <- responses[!is.na(responses)]
  
  # Compute accuracy as mean of responses
  results$actual_accuracy[i] <- mean(valid_responses)
  results$n_items[i] <- length(valid_responses)
  
  if (i %% 100 == 0 || i == 1) {
    cat("  Progress:", i, "/", n_models, "\n")
  }
}

# Create output directory if needed
dir.create(dirname(OUTPUT_FILE), recursive = TRUE, showWarnings = FALSE)

# Save results
cat("\nSaving results to:", OUTPUT_FILE, "\n")
write.csv(results, OUTPUT_FILE, row.names = FALSE)

# Summary statistics
cat("\n", rep("=", 70), "\n", sep = "")
cat("Summary Statistics:\n")
cat(rep("=", 70), "\n", sep = "")
cat("  Models processed:", nrow(results), "\n")
cat("  Mean accuracy:", round(mean(results$actual_accuracy, na.rm = TRUE), 4), "\n")
cat("  SD accuracy:", round(sd(results$actual_accuracy, na.rm = TRUE), 4), "\n")
cat("  Min/Max accuracy:", round(min(results$actual_accuracy, na.rm = TRUE), 4), "/",
    round(max(results$actual_accuracy, na.rm = TRUE), 4), "\n")
cat(rep("=", 70), "\n\n", sep = "")

cat("✓ Actual accuracy computation complete!\n\n")
