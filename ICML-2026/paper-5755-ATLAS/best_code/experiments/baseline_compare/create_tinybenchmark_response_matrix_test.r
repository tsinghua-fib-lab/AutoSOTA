#!/usr/bin/env Rscript
# Create TinyBenchmark-specific response matrix for test models
# Combines test response matrix with missing TinyBenchmark items from full dataset

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
BENCHMARK <- "arc"
if (length(args) > 0) {
  for (arg in args) {
    if (grepl("^--benchmark=", arg)) {
      BENCHMARK <- sub("^--benchmark=", "", arg)
    }
  }
}

cat("\n=======================================================================\n")
cat("Creating TinyBenchmark Response Matrix for Test Models\n")
cat("Benchmark:", BENCHMARK, "\n")
cat("=======================================================================\n\n")

# File paths
BASE_DIR <- "experiments/baseline_compare"
TINY_INDICES_FILE <- paste0(BASE_DIR, "/tiny", BENCHMARK, "_numeric_indices.csv")
TEST_RESPONSE_FILE <- paste0("data/gaussian_sampled_", BENCHMARK, "_response_matrix_test.csv")
FULL_RESPONSE_FILE <- paste0(BASE_DIR, "/tinybenchmark_calibration/response_matrix_tiny", 
                             BENCHMARK, ".csv")
OUTPUT_FILE <- paste0(BASE_DIR, "/gaussian_sampled_", BENCHMARK, 
                      "_response_matrix_test_tinybenchmark.csv")

# Step 1: Load TinyBenchmark items
cat("1. Loading TinyBenchmark items...\n")
tiny_items <- read.csv(TINY_INDICES_FILE, stringsAsFactors = FALSE)
tiny_item_ids <- as.character(tiny_items$item_index)
cat("  TinyBenchmark items:", length(tiny_item_ids), "\n")

# Step 2: Load test response matrix
cat("\n2. Loading test response matrix...\n")
test_matrix <- read.csv(TEST_RESPONSE_FILE, stringsAsFactors = FALSE, check.names = FALSE)
test_model_names <- test_matrix[[1]]
test_item_ids <- colnames(test_matrix)[-1]
cat("  Test models:", length(test_model_names), "\n")
cat("  Test items:", length(test_item_ids), "\n")

# Step 3: Find missing TinyBenchmark items
cat("\n3. Finding missing TinyBenchmark items...\n")
missing_items <- setdiff(tiny_item_ids, test_item_ids)
cat("  Missing items:", length(missing_items), "\n")
if (length(missing_items) > 0) {
  cat("  Missing item IDs:", paste(head(missing_items, 20), collapse=", "), "\n")
}

if (length(missing_items) == 0) {
  cat("\n✓ No missing items! Test matrix already contains all TinyBenchmark items.\n")
  cat("  Using existing test matrix.\n")
  file.copy(TEST_RESPONSE_FILE, OUTPUT_FILE, overwrite = TRUE)
  quit(status = 0)
}

# Step 4: Load TinyBenchmark calibration matrix (has all 99 items)
cat("\n4. Loading TinyBenchmark calibration matrix to extract missing items...\n")
# Read header first to check columns
tiny_calib_header <- read.csv(FULL_RESPONSE_FILE, stringsAsFactors = FALSE, 
                              check.names = FALSE, nrows = 1)
tiny_calib_item_ids <- colnames(tiny_calib_header)[-1]
cat("  TinyBenchmark calibration items:", length(tiny_calib_item_ids), "\n")

# Check which missing items exist in TinyBenchmark calibration
available_missing <- intersect(missing_items, tiny_calib_item_ids)
cat("  Missing items available in TinyBenchmark calibration:", length(available_missing), "\n")

if (length(available_missing) == 0) {
  cat("\n✗ ERROR: Missing items not found in TinyBenchmark calibration!\n")
  quit(status = 1)
}

# Find column indices for missing items in calibration matrix
missing_col_indices <- which(tiny_calib_item_ids %in% available_missing)
cat("  Column indices to extract:", paste(head(missing_col_indices + 1, 10), collapse=", "), "...\n")

# Step 5: Load TinyBenchmark calibration matrix and extract test models + missing columns
cat("\n5. Extracting missing items for test models from TinyBenchmark calibration...\n")
tiny_calib_matrix <- read.csv(FULL_RESPONSE_FILE, stringsAsFactors = FALSE, check.names = FALSE)
tiny_calib_model_names <- tiny_calib_matrix[[1]]

# Find test model indices in TinyBenchmark calibration matrix
test_model_indices <- which(tiny_calib_model_names %in% test_model_names)
cat("  Test models found in TinyBenchmark calibration:", length(test_model_indices), "\n")

if (length(test_model_indices) != length(test_model_names)) {
  cat("  WARNING: Not all test models found in TinyBenchmark calibration!\n")
  cat("  Available:", length(test_model_indices), "/ Expected:", length(test_model_names), "\n")
}

# Extract missing columns for test models
# Column indices need +1 because first column is model names
missing_data <- tiny_calib_matrix[test_model_indices, c(1, missing_col_indices + 1), drop = FALSE]
cat("  Extracted matrix:", nrow(missing_data), "models ×", ncol(missing_data) - 1, "items\n")

# Step 6: Merge test matrix with missing items
cat("\n6. Merging test matrix with missing items...\n")

# Reorder missing_data to match test_model_names order
missing_data_ordered <- missing_data[match(test_model_names, missing_data[[1]]), ]

# Combine: test matrix + missing columns (excluding duplicate model name column)
combined_matrix <- cbind(test_matrix, missing_data_ordered[, -1, drop = FALSE])
cat("  Combined matrix:", nrow(combined_matrix), "models ×", ncol(combined_matrix) - 1, "items\n")

# Step 7: Verify all TinyBenchmark items are now present
combined_item_ids <- colnames(combined_matrix)[-1]
all_tiny_present <- all(tiny_item_ids %in% combined_item_ids)
cat("  All TinyBenchmark items present:", all_tiny_present, "\n")

if (!all_tiny_present) {
  still_missing <- setdiff(tiny_item_ids, combined_item_ids)
  cat("  WARNING: Still missing items:", paste(still_missing, collapse=", "), "\n")
}

# Step 8: Save combined matrix
cat("\n7. Saving TinyBenchmark response matrix...\n")
write.csv(combined_matrix, OUTPUT_FILE, row.names = FALSE)
cat("  Saved to:", OUTPUT_FILE, "\n")

# Summary
cat("\n", rep("=", 70), "\n", sep = "")
cat("Summary:\n")
cat(rep("=", 70), "\n", sep = "")
cat("  Models:", nrow(combined_matrix), "\n")
cat("  Items:", ncol(combined_matrix) - 1, "\n")
cat("  TinyBenchmark items:", length(tiny_item_ids), "\n")
cat("  TinyBenchmark items available:", sum(tiny_item_ids %in% combined_item_ids), "\n")
cat("  Additional items from test set:", length(test_item_ids), "\n")
cat("  Total unique items:", length(unique(combined_item_ids)), "\n")
cat(rep("=", 70), "\n", sep = "")
cat("\n✓ TinyBenchmark response matrix created successfully!\n\n")
