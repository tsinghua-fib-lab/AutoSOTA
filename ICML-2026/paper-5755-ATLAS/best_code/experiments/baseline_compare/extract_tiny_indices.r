# Extract numeric indices from TinyBenchmark calibration response matrices
# These are the actual column indices used in the response matrices

benchmarks <- c("arc", "gsm8k", "hellaswag", "truthfulqa", "winogrande")

for (benchmark in benchmarks) {
  # Read TinyBenchmark calibration response matrix
  tiny_file <- paste0("tinybenchmark_calibration/response_matrix_tiny", benchmark, ".csv")
  
  if (!file.exists(tiny_file)) {
    cat("Skipping", benchmark, "- file not found\n")
    next
  }
  
  # Read just the header
  tiny_matrix <- read.csv(tiny_file, stringsAsFactors = FALSE, check.names = FALSE, nrows = 1)
  
  # Get column names (excluding first column which is model names)
  item_indices <- colnames(tiny_matrix)[-1]
  
  # Create data frame
  indices_df <- data.frame(
    item_index = item_indices,
    stringsAsFactors = FALSE
  )
  
  # Save to CSV
  output_file <- paste0("tiny", benchmark, "_numeric_indices.csv")
  write.csv(indices_df, output_file, row.names = FALSE)
  
  cat("Saved", nrow(indices_df), "indices for", benchmark, "to", output_file, "\n")
}

cat("\nDone! Use these numeric index files instead of the Mercury ID files.\n")
