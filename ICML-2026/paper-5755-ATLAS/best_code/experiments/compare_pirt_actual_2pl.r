#!/usr/bin/env Rscript
# Compare p-IRT Estimates with Actual Accuracy
# This script compares p-IRT accuracy estimates against actual accuracy

# Parse CLI arguments
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (a in args) {
    if (grepl("^--benchmark=", a)) {
      out$benchmark <- sub("^--benchmark=", "", a)
    } else if (grepl("^--se_theta_stop=", a)) {
      out$se_theta_stop <- sub("^--se_theta_stop=", "", a)
    }
  }
  return(out)
}

.cli_args <- parse_args()
BENCHMARK <- if (!is.null(.cli_args$benchmark)) .cli_args$benchmark else "arc"
SE_THETA_STOP <- if (!is.null(.cli_args$se_theta_stop)) .cli_args$se_theta_stop else "0.1"

cat("\n", rep("=", 70), "\n", sep = "")
cat("Comparing p-IRT Estimates with Actual Accuracy\n")
cat("Benchmark:", BENCHMARK, "\n")
cat("SE threshold:", SE_THETA_STOP, "\n")
cat(rep("=", 70), "\n\n", sep = "")

# File paths
PIRT_FILE <- paste0("experiments/", BENCHMARK, "_2pl/pirt_accuracy_se_", SE_THETA_STOP, ".csv")
ACTUAL_FILE <- paste0("experiments/", BENCHMARK, "_2pl/actual_accuracy.csv")
OUTPUT_FILE <- paste0("experiments/", BENCHMARK, "_2pl/pirt_vs_actual_se_", SE_THETA_STOP, ".csv")
PLOT_FILE <- paste0("experiments/", BENCHMARK, "_2pl/pirt_vs_actual_se_", SE_THETA_STOP, ".png")

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

# Additional statist-benics
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

cat("\nSubset Size Statistics:\n")
cat("  Mean subset size:", round(mean(merged$n_subset_items, na.rm = TRUE), 1), "\n")
cat("  SD:", round(sd(merged$n_subset_items, na.rm = TRUE), 1), "\n")
cat("  Full dataset size:", merged$n_all_items[1], "\n")
cat("  Reduction ratio:", round(mean(merged$n_subset_items / merged$n_all_items, na.rm = TRUE), 4), "\n")

# Create visualization
cat("\nCreating visualization...\n")
png(PLOT_FILE, width = 1200, height = 1200, res = 150)
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 2.5, 1))

# 1. Scatter plot: p-IRT vs Actual
plot(merged$actual_accuracy, merged$pirt_accuracy,
     xlab = "Actual Accuracy", ylab = "p-IRT Accuracy",
     main = paste0("p-IRT vs Actual Accuracy\n", BENCHMARK, " (SE=", SE_THETA_STOP, ")"),
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

# 3. Absolute error vs subset size
plot(merged$n_subset_items, merged$abs_error,
     xlab = "Subset Size", ylab = "Absolute Error",
     main = "Absolute Error vs Subset Size",
     pch = 16, col = rgb(0, 0, 0, 0.3), cex = 0.8)
# Add smoothing line
if (nrow(merged) > 10) {
  smooth_fit <- loess(abs_error ~ n_subset_items, data = merged)
  pred_order <- order(merged$n_subset_items)
  lines(merged$n_subset_items[pred_order], 
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
