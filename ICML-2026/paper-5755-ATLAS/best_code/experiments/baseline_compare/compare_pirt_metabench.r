#!/usr/bin/env Rscript
# Compare p-IRT Metabench Estimates with Actual Accuracy

# Parse CLI arguments
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (a in args) {
    if (grepl("^--benchmark=", a)) {
      out$benchmark <- sub("^--benchmark=", "", a)
    } else if (grepl("^--version=", a)) {
      out$version <- sub("^--version=", "", a)
    }
  }
  return(out)
}

.cli_args <- parse_args()
BENCHMARK <- if (!is.null(.cli_args$benchmark)) .cli_args$benchmark else "truthfulqa"
VERSION <- if (!is.null(.cli_args$version)) .cli_args$version else "secondary"

cat("\n", rep("=", 70), "\n", sep = "")
cat("Comparing p-IRT Metabench with Actual Accuracy\n")
cat("Benchmark:", BENCHMARK, "\n")
cat("Version:", VERSION, "\n")
cat(rep("=", 70), "\n\n", sep = "")

# File paths
BASE_DIR <- "experiments/baseline_compare"
PIRT_FILE <- paste0(BASE_DIR, "/pirt_metabench_", BENCHMARK, "_", VERSION, ".csv")
ACTUAL_FILE <- paste0(BENCHMARK, "/actual_accuracy.csv")
OUTPUT_FILE <- paste0(BASE_DIR, "/pirt_metabench_vs_actual_", BENCHMARK, "_", VERSION, ".csv")
PLOT_FILE <- paste0(BASE_DIR, "/pirt_metabench_vs_actual_", BENCHMARK, "_", VERSION, ".png")

# Check if actual accuracy file exists, if not compute it
if (!file.exists(ACTUAL_FILE)) {
  cat("Actual accuracy file not found. Computing it now...\n")
  system(paste0("Rscript scripts/05_compute_actual_acc.r --benchmark=", BENCHMARK))
}

# Read data
cat("Reading p-IRT estimates from:", PIRT_FILE, "\n")
pirt_data <- read.csv(PIRT_FILE, stringsAsFactors = FALSE)

cat("Reading actual accuracy from:", ACTUAL_FILE, "\n")
actual_data <- read.csv(ACTUAL_FILE, stringsAsFactors = FALSE)

# Merge datasets
cat("\nMerging datasets...\n")
merged <- merge(pirt_data, actual_data, by = "model", all = TRUE)
n_models <- nrow(merged)
cat("  Models matched:", n_models, "\n")

# Compute error metrics
merged$error <- merged$pirt_accuracy - merged$actual_accuracy
merged$abs_error <- abs(merged$error)
merged$squared_error <- merged$error^2

# Save merged results
cat("\nSaving comparison results to:", OUTPUT_FILE, "\n")
write.csv(merged, OUTPUT_FILE, row.names = FALSE)

# Compute summary statistics
cat("\n", rep("=", 70), "\n", sep = "")
cat("Error Metrics:\n")
cat(rep("=", 70), "\n", sep = "")
cat("  Mean Error (ME):", round(mean(merged$error, na.rm = TRUE), 6), "\n")
cat("  Mean Absolute Error (MAE):", round(mean(merged$abs_error, na.rm = TRUE), 6), "\n")
cat("  Root Mean Squared Error (RMSE):", round(sqrt(mean(merged$squared_error, na.rm = TRUE)), 6), "\n")
cat("  Correlation (r):", round(cor(merged$pirt_accuracy, merged$actual_accuracy, use = "complete.obs"), 6), "\n")
cat("  R-squared:", round(cor(merged$pirt_accuracy, merged$actual_accuracy, use = "complete.obs")^2, 6), "\n")
cat(rep("=", 70), "\n", sep = "")

# Additional statistics
cat("\np-IRT Statistics:\n")
cat("  Mean:", round(mean(merged$pirt_accuracy, na.rm = TRUE), 4), "\n")
cat("  SD:", round(sd(merged$pirt_accuracy, na.rm = TRUE), 4), "\n")
cat("  Range:", round(min(merged$pirt_accuracy, na.rm = TRUE), 4), "to",
    round(max(merged$pirt_accuracy, na.rm = TRUE), 4), "\n")

cat("\nActual Accuracy Statistics:\n")
cat("  Mean:", round(mean(merged$actual_accuracy, na.rm = TRUE), 4), "\n")
cat("  SD:", round(sd(merged$actual_accuracy, na.rm = TRUE), 4), "\n")
cat("  Range:", round(min(merged$actual_accuracy, na.rm = TRUE), 4), "to",
    round(max(merged$actual_accuracy, na.rm = TRUE), 4), "\n")

cat("\nMetabench Configuration:\n")
cat("  Version:", VERSION, "\n")
cat("  Subset size:", merged$n_subset_items[1], "\n")
cat("  Full dataset size:", merged$n_all_items[1], "\n")
cat("  Reduction ratio:", round(merged$n_subset_items[1] / merged$n_all_items[1], 4), "\n")

# Create visualization
cat("\nCreating visualization...\n")
png(PLOT_FILE, width = 1200, height = 1200, res = 150)
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 2.5, 1))

# 1. Scatter plot: p-IRT vs Actual
plot(merged$actual_accuracy, merged$pirt_accuracy,
     xlab = "Actual Accuracy", ylab = paste0("p-IRT Accuracy (Metabench-", VERSION, ")"),
     main = paste0("p-IRT Metabench vs Actual Accuracy\n", BENCHMARK, " (", VERSION, ")"),
     pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.8)
abline(0, 1, col = "red", lwd = 2, lty = 2)
abline(lm(pirt_accuracy ~ actual_accuracy, data = merged), col = "blue", lwd = 2)
legend("topleft", 
       legend = c("Perfect prediction", "Linear fit",
                 paste0("r = ", round(cor(merged$pirt_accuracy, merged$actual_accuracy, use = "complete.obs"), 4)),
                 paste0("RMSE = ", round(sqrt(mean(merged$squared_error, na.rm = TRUE)), 4))),
       col = c("red", "blue", NA, NA), lty = c(2, 1, NA, NA), lwd = 2, bty = "n")

# 2. Error distribution
hist(merged$error, breaks = 30, col = "lightblue", border = "white",
     xlab = "Error (p-IRT - Actual)", main = "Error Distribution",
     xlim = c(-0.3, 0.3))
abline(v = 0, col = "red", lwd = 2, lty = 2)
abline(v = mean(merged$error, na.rm = TRUE), col = "blue", lwd = 2)
legend("topright",
       legend = c(paste0("Mean = ", round(mean(merged$error, na.rm = TRUE), 5)),
                 paste0("MAE = ", round(mean(merged$abs_error, na.rm = TRUE), 5))),
       bty = "n")

# 3. Absolute error vs theta
plot(merged$theta, merged$abs_error,
     xlab = "Theta (Ability)", ylab = "Absolute Error",
     main = "Absolute Error vs Estimated Ability",
     pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.8)
# Add smoothing line
if (nrow(merged) > 10) {
  smooth_fit <- loess(abs_error ~ theta, data = merged)
  pred_order <- order(merged$theta)
  lines(merged$theta[pred_order], 
        predict(smooth_fit)[pred_order], 
        col = "blue", lwd = 2)
}

# 4. Error vs actual accuracy
plot(merged$actual_accuracy, merged$error,
     xlab = "Actual Accuracy", ylab = "Error (p-IRT - Actual)",
     main = "Error vs Actual Accuracy",
     pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.8)
abline(h = 0, col = "red", lwd = 2, lty = 2)
# Add smoothing line
if (nrow(merged) > 10) {
  smooth_fit <- loess(error ~ actual_accuracy, data = merged)
  pred_order <- order(merged$actual_accuracy)
  lines(merged$actual_accuracy[pred_order], 
        predict(smooth_fit)[pred_order], 
        col = "blue", lwd = 2)
}

dev.off()
cat("  Plot saved to:", PLOT_FILE, "\n")

cat("\n", rep("=", 70), "\n", sep = "")
cat("✓ Comparison complete!\n")
cat("  Results saved to:", OUTPUT_FILE, "\n")
cat("  Visualization saved to:", PLOT_FILE, "\n")
cat(rep("=", 70), "\n\n", sep = "")
