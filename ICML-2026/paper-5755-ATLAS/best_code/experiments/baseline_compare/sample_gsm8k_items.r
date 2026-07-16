#!/usr/bin/env Rscript

# Script to randomly sample 100 items from GSM8K origin files and save to new files
# Created: 2025-09-23

set.seed(42) # Setting seed for reproducibility

# Define file paths
primary_origin_file <- "experiments/baseline_compare/metabench_gsm8k_item_ids_primary_origin.csv"
secondary_origin_file <- "experiments/baseline_compare/metabench_gsm8k_item_ids_secondary_origin.csv"
primary_output_file <- "experiments/baseline_compare/metabench_gsm8k_item_ids_primary.csv"
origin_output_file <- "experiments/baseline_compare/metabench_gsm8k_item_ids_secondary.csv"

# Read the origin files
cat("Reading input files...\n")
primary_data <- read.csv(primary_origin_file, stringsAsFactors = FALSE)
secondary_data <- read.csv(secondary_origin_file, stringsAsFactors = FALSE)

# Print counts of items in original files
cat("Original primary items count:", nrow(primary_data), "\n")
cat("Original secondary items count:", nrow(secondary_data), "\n")

# Randomly sample 100 items from each file
cat("Sampling 100 items from each file...\n")
if(nrow(primary_data) >= 100) {
  sampled_primary <- primary_data[sample(1:nrow(primary_data), 100), , drop=FALSE]
} else {
  cat("WARNING: Primary file has fewer than 100 items. Using all available items.\n")
  sampled_primary <- primary_data
}

if(nrow(secondary_data) >= 100) {
  sampled_secondary <- secondary_data[sample(1:nrow(secondary_data), 100), , drop=FALSE]
} else {
  cat("WARNING: Secondary file has fewer than 100 items. Using all available items.\n")
  sampled_secondary <- secondary_data
}

# Save sampled data to output files
cat("Saving sampled items to output files...\n")
write.csv(sampled_primary, file = primary_output_file, row.names = FALSE)
write.csv(sampled_secondary, file = origin_output_file, row.names = FALSE)

cat("Done!\n")
cat("Primary items saved to:", primary_output_file, "\n")
cat("Secondary items saved to:", origin_output_file, "\n")
