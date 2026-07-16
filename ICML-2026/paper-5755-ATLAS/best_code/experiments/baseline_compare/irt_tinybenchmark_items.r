#!/usr/bin/env Rscript
# 3PL IRT on TinyBenchmarks Items using MetaBench Data
# This script performs IRT calibration on tinybenchmarks items using model performance
# from metabench data, filtered to models present in the training datasets

library(mirt)

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
BENCHMARK <- if (!is.null(.cli_args$benchmark)) .cli_args$benchmark else "truthfulqa"

cat("\n", rep("=", 70), "\n", sep = "")
cat("3PL IRT on TinyBenchmarks Items - MetaBench Data\n")
cat("Benchmark:", BENCHMARK, "\n")
cat(rep("=", 70), "\n\n", sep = "")

# File paths
BASE_DIR <- "experiments/baseline_compare"
METABENCH_DATA_FILE <- paste0("data/metabench_data/", BENCHMARK, ".csv")
TRAINING_MODELS_FILE <- paste0("data/gaussian_sampled_", BENCHMARK, "_response_matrix_train.csv")
TINY_ITEMS_FILE <- paste0(BASE_DIR, "/tiny", BENCHMARK, "_item_ids.csv")
OUTPUT_DIR <- paste0(BASE_DIR, "/tinybenchmark_calibration")
OUTPUT_FILE <- paste0(OUTPUT_DIR, "/irt_item_parameters_tiny", BENCHMARK, ".csv")
THETA_FILE <- paste0(OUTPUT_DIR, "/irt_person_scores_tiny", BENCHMARK, ".csv")
RESPONSE_MATRIX_FILE <- paste0(OUTPUT_DIR, "/response_matrix_tiny", BENCHMARK, ".csv")

# Create output directory if needed
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
  cat("Created output directory:", OUTPUT_DIR, "\n")
}

# Step 1: Load tinybenchmark item IDs
cat("\n1. Loading tinybenchmark item IDs...\n")
if (!file.exists(TINY_ITEMS_FILE)) {
  cat("ERROR: Tinybenchmark items file not found:", TINY_ITEMS_FILE, "\n")
  quit(status = 1)
}

tiny_items_df <- read.csv(TINY_ITEMS_FILE, stringsAsFactors = FALSE)
tiny_item_ids <- as.character(tiny_items_df$item_id)
tiny_item_ids <- tiny_item_ids[!is.na(tiny_item_ids) & tiny_item_ids != ""]
cat("  Number of tinybenchmark items:", length(tiny_item_ids), "\n")
cat("  Sample items:", paste(head(tiny_item_ids, 5), collapse=", "), "\n")

# Step 2: Load metabench data
cat("\n2. Loading metabench data...\n")
if (!file.exists(METABENCH_DATA_FILE)) {
  cat("ERROR: Metabench data file not found:", METABENCH_DATA_FILE, "\n")
  quit(status = 1)
}

metabench_data <- read.csv(METABENCH_DATA_FILE, stringsAsFactors = FALSE)
cat("  Total rows in metabench data:", nrow(metabench_data), "\n")
cat("  Columns:", paste(colnames(metabench_data), collapse=", "), "\n")

# Convert correct column to 0/1 if it's True/False
if (is.character(metabench_data$correct) || is.logical(metabench_data$correct)) {
  metabench_data$correct <- as.integer(metabench_data$correct == "True" | metabench_data$correct == TRUE)
}

# Step 3: Filter to tinybenchmark items only
cat("\n3. Filtering to tinybenchmark items...\n")
metabench_data$item <- as.character(metabench_data$item)
filtered_data <- metabench_data[metabench_data$item %in% tiny_item_ids, ]
cat("  Rows after filtering:", nrow(filtered_data), "\n")
cat("  Unique models:", length(unique(filtered_data$source)), "\n")
cat("  Unique items:", length(unique(filtered_data$item)), "\n")

if (nrow(filtered_data) == 0) {
  cat("ERROR: No matching data found for tinybenchmark items\n")
  quit(status = 1)
}

# Step 4: Load training models list
cat("\n4. Loading training models...\n")
if (!file.exists(TRAINING_MODELS_FILE)) {
  cat("ERROR: Training models file not found:", TRAINING_MODELS_FILE, "\n")
  quit(status = 1)
}

# Read first column (model names) from training file
training_models_df <- read.csv(TRAINING_MODELS_FILE, stringsAsFactors = FALSE, nrows = 1)
# Get model names from actual data
training_models_full <- read.csv(TRAINING_MODELS_FILE, stringsAsFactors = FALSE)
training_models <- training_models_full[[1]]  # First column contains model names
cat("  Number of training models:", length(training_models), "\n")
cat("  Sample training models:", paste(head(training_models, 3), collapse=", "), "\n")

# Step 5: Filter to training models only
cat("\n5. Filtering to training models...\n")
filtered_data <- filtered_data[filtered_data$source %in% training_models, ]
cat("  Rows after model filtering:", nrow(filtered_data), "\n")
cat("  Unique models remaining:", length(unique(filtered_data$source)), "\n")

if (nrow(filtered_data) == 0) {
  cat("ERROR: No data remains after filtering to training models\n")
  quit(status = 1)
}

# Step 6: Create response matrix (rows = models, columns = items)
cat("\n6. Creating response matrix...\n")
response_wide <- reshape(filtered_data[, c("source", "item", "correct")], 
                         idvar = "source", 
                         timevar = "item", 
                         direction = "wide")

# Clean up column names (remove "correct." prefix)
colnames(response_wide) <- gsub("^correct\\.", "", colnames(response_wide))

# Set row names to model names and remove the source column
rownames(response_wide) <- response_wide$source
response_matrix <- response_wide[, -1]

cat("  Response matrix dimensions:", nrow(response_matrix), "models x", ncol(response_matrix), "items\n")

# Step 7: Clean the data
cat("\n7. Cleaning data...\n")
cat("  Original dimensions:", dim(response_matrix), "\n")

# Remove rows with all NAs
all_na_rows <- apply(response_matrix, 1, function(x) all(is.na(x)))
response_matrix <- response_matrix[!all_na_rows, ]
cat("  After removing all-NA rows:", dim(response_matrix), "\n")

# Remove columns with all NAs
all_na_cols <- apply(response_matrix, 2, function(x) all(is.na(x)))
response_matrix <- response_matrix[, !all_na_cols]
cat("  After removing all-NA columns:", dim(response_matrix), "\n")

# Remove constant columns (no variance)
constant_cols <- apply(response_matrix, 2, function(x) {
  x_clean <- x[!is.na(x)]
  if(length(x_clean) == 0) return(TRUE)
  length(unique(x_clean)) == 1
})
response_matrix <- response_matrix[, !constant_cols]
cat("  After removing constant columns:", dim(response_matrix), "\n")

# Remove constant rows
constant_rows <- apply(response_matrix, 1, function(x) {
  x_clean <- x[!is.na(x)]
  if(length(x_clean) == 0) return(TRUE)
  length(unique(x_clean)) == 1
})
response_matrix <- response_matrix[!constant_rows, ]
cat("  Final dimensions:", dim(response_matrix), "\n")

# Check if we have enough data
if (nrow(response_matrix) < 10) {
  cat("ERROR: Not enough models after cleaning (", nrow(response_matrix), ")\n")
  quit(status = 1)
}

if (ncol(response_matrix) < 5) {
  cat("ERROR: Not enough items after cleaning (", ncol(response_matrix), ")\n")
  quit(status = 1)
}

# Save response matrix for inspection
cat("  Saving response matrix to:", RESPONSE_MATRIX_FILE, "\n")
write.csv(response_matrix, RESPONSE_MATRIX_FILE, row.names = TRUE)

# Step 8: Fit 3PL IRT model
cat("\n8. Fitting 3PL IRT model...\n")
cat("  This may take several minutes...\n")

model <- mirt(response_matrix, 
              model = 1, 
              itemtype = "3PL", 
              method = "EM",
              technical = list(NCYCLES = 100000))

cat("  Model fitting complete!\n")
print(model)

# Step 9: Extract and save item parameters
cat("\n9. Extracting item parameters...\n")
item_params <- coef(model, simplify = TRUE)$items
print(head(item_params))

cat("  Saving item parameters to:", OUTPUT_FILE, "\n")
write.csv(item_params, OUTPUT_FILE, row.names = TRUE)

# Step 10: Estimate model theta (person scores)
cat("\n10. Estimating model theta scores...\n")
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE, full.scores.SE = TRUE, quadpts = 61)
print(head(theta_scores))

cat("  Saving theta scores to:", THETA_FILE, "\n")
write.csv(theta_scores, THETA_FILE, row.names = TRUE)

# Summary
cat("\n", rep("=", 70), "\n", sep = "")
cat("Summary:\n")
cat(rep("=", 70), "\n", sep = "")
cat("  Benchmark:", BENCHMARK, "\n")
cat("  Number of items calibrated:", ncol(response_matrix), "\n")
cat("  Number of models used:", nrow(response_matrix), "\n")
cat("  Item parameters saved to:", OUTPUT_FILE, "\n")
cat("  Theta scores saved to:", THETA_FILE, "\n")
cat("  Response matrix saved to:", RESPONSE_MATRIX_FILE, "\n")
cat(rep("=", 70), "\n\n", sep = "")

cat("✓ IRT calibration complete!\n\n")
