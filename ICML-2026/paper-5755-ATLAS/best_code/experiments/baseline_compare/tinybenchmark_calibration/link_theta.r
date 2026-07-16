#!/usr/bin/env Rscript

# IRT Linking Script: Link theta estimates from tiny benchmark to whole dataset
# Using common items and Mean-Sigma linking method

library(mirt)

# Function to perform Mean-Sigma linking
mean_sigma_linking <- function(params_tiny, params_whole, common_items) {
  # Extract a and d parameters for common items
  a_tiny <- params_tiny[common_items, "a1"]
  d_tiny <- params_tiny[common_items, "d"]
  
  a_whole <- params_whole[common_items, "a1"]
  d_whole <- params_whole[common_items, "d"]
  
  # Calculate linking constants using Mean-Sigma method
  # a_whole = A * a_tiny (slope)
  # d_whole = A * d_tiny + B (intercept)
  
  A <- mean(a_whole) / mean(a_tiny)
  b_whole <- -d_whole/a_whole
  b_tiny <- -d_tiny/a_tiny
  B <- mean(b_whole - b_tiny)
  
  cat("Linking constants:\n")
  cat("  A (slope) =", A, "\n")
  cat("  B (intercept) =", B, "\n")
  
  return(list(A = A, B = B))
}

# Function to transform theta from tiny scale to whole scale
transform_theta <- function(theta_tiny, A, B) {
  # theta_whole = (theta_tiny - B) / A
  theta_linked <- (theta_tiny - B) / A
  return(theta_linked)
}

# Main processing
BENCHMARK <- "truthfulqa"

cat("\n=== IRT Linking for", BENCHMARK, "===\n\n")

# Read item parameters
cat("Reading item parameters...\n")
items_tiny_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/irt_item_parameters_tiny", BENCHMARK, ".csv")
items_whole_path <- paste0(BENCHMARK, "/irt_item_parameters_combined.csv")

items_tiny <- read.csv(items_tiny_path, stringsAsFactors = FALSE, row.names = 1)
items_whole <- read.csv(items_whole_path, stringsAsFactors = FALSE, row.names = 1)

cat("  Tiny benchmark items:", nrow(items_tiny), "\n")
cat("  Whole dataset items:", nrow(items_whole), "\n")

# Find common items
# Tiny items have names like "11", "25", etc.
# Whole items have names like "X1", "X2", "X11", "X25", etc.
tiny_items_names <- rownames(items_tiny)
whole_items_names <- rownames(items_whole)

# Create X-prefixed versions of tiny item names
tiny_items_with_X <- paste0("X", tiny_items_names)

# Find common items
common_items <- intersect(tiny_items_with_X, whole_items_names)
cat("\nCommon items found:", length(common_items), "\n")

if (length(common_items) < 2) {
  cat("ERROR: Need at least 2 common items for linking\n")
  quit(status = 1)
}

# Reindex for matching
# Subset tiny items where the X-prefixed name is in common items
tiny_indices <- tiny_items_names[tiny_items_with_X %in% common_items]
items_tiny_common <- items_tiny[tiny_indices, ]
rownames(items_tiny_common) <- paste0("X", tiny_indices)

items_whole_common <- items_whole[common_items, ]

# Perform linking
cat("\nPerforming Mean-Sigma linking...\n")
linking_constants <- mean_sigma_linking(items_tiny_common, items_whole_common, common_items)

# Read person parameters (theta estimates)
cat("\nReading person parameters...\n")
theta_tiny_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/irt_person_scores_tiny", BENCHMARK, ".csv")
theta_whole_path <- paste0(BENCHMARK, "/irt_person_scores_WLE_SE_test.csv")

theta_tiny <- read.csv(theta_tiny_path, stringsAsFactors = FALSE)
theta_whole <- read.csv(theta_whole_path, stringsAsFactors = FALSE)

cat("  Tiny benchmark persons:", nrow(theta_tiny), "\n")
cat("  Whole dataset persons:", nrow(theta_whole), "\n")

# Transform theta from tiny scale to whole scale
cat("\nTransforming theta estimates...\n")
theta_tiny$Theta_Linked <- transform_theta(theta_tiny$F1, linking_constants$A, linking_constants$B)

# For comparison, match by row order (assuming same order)
n_compare <- min(nrow(theta_tiny), nrow(theta_whole))
theta_comparison <- data.frame(
  Theta_Tiny = theta_tiny$F1[1:n_compare],
  Theta_Linked = theta_tiny$Theta_Linked[1:n_compare],
  Theta_Whole = theta_whole$Theta_WLE[1:n_compare]
)

# Calculate metrics comparing Theta_Linked vs Theta_Whole
rmse_unlinked <- sqrt(mean((theta_comparison$Theta_Tiny - theta_comparison$Theta_Whole)^2, na.rm = TRUE))
mae_unlinked <- mean(abs(theta_comparison$Theta_Tiny - theta_comparison$Theta_Whole), na.rm = TRUE)

rmse_linked <- sqrt(mean((theta_comparison$Theta_Linked - theta_comparison$Theta_Whole)^2, na.rm = TRUE))
mae_linked <- mean(abs(theta_comparison$Theta_Linked - theta_comparison$Theta_Whole), na.rm = TRUE)

# Print results
cat("\n=== Comparison Results ===\n")
cat("\nBefore Linking (Tiny vs Whole):\n")
cat("  RMSE:", rmse_unlinked, "\n")
cat("  MAE:", mae_unlinked, "\n")

cat("\nAfter Linking (Linked vs Whole):\n")
cat("  RMSE:", rmse_linked, "\n")
cat("  MAE:", mae_linked, "\n")

cat("\nImprovement:\n")
cat("  RMSE reduction:", rmse_unlinked - rmse_linked, "(", round((rmse_unlinked - rmse_linked) / rmse_unlinked * 100, 2), "%)\n")
cat("  MAE reduction:", mae_unlinked - mae_linked, "(", round((mae_unlinked - mae_linked) / mae_unlinked * 100, 2), "%)\n")

# Save linked theta estimates
output_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/irt_person_scores_tiny", BENCHMARK, "_linked.csv")
write.csv(theta_tiny, output_path, row.names = FALSE)
cat("\nLinked theta estimates saved to:", output_path, "\n")

# Save comparison results
comparison_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/theta_comparison_", BENCHMARK, ".csv")
write.csv(theta_comparison, comparison_path, row.names = FALSE)
cat("Comparison results saved to:", comparison_path, "\n")
