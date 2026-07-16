#!/usr/bin/env Rscript
# p-IRT Accuracy Estimation for TinyBenchmarks
# Estimates full dataset accuracy based on TinyBenchmark selected items using 3PL IRT model

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
cat("p-IRT Accuracy Estimation - TinyBenchmarks\n")
cat("Benchmark:", BENCHMARK, "\n")
cat(rep("=", 70), "\n\n", sep = "")

# File paths
BASE_DIR <- "experiments/baseline_compare"
ITEM_PARAMS_PRIMARY <- paste0(BENCHMARK, "/irt_item_parameters_combined.csv")
ITEM_PARAMS_FALLBACK <- paste0(BASE_DIR, "/tinybenchmark_calibration/irt_item_parameters_tiny", BENCHMARK, ".csv")
RESPONSE_FILE <- paste0(BASE_DIR, "/gaussian_sampled_", BENCHMARK, "_response_matrix_train_tinybenchmark.csv")
SELECTED_ITEMS_FILE <- paste0(BASE_DIR, "/tiny", BENCHMARK, "_numeric_indices.csv")
OUTPUT_FILE <- paste0(BASE_DIR, "/tiny", BENCHMARK, "_theta_eap.csv")

# Load theta estimation utilities
source(paste0(BASE_DIR, "/theta_estimation_utils.r"))

# Convert item parameters from (a1, d, g, u) to (a, b, c)
prepare_item_parameters_single <- function(file_path) {
  params <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Handle different ID column formats
  if ("X" %in% names(params)) {
    item_ids <- params$X
  } else if ("" %in% names(params)) {
    item_ids <- params[[1]]
  } else {
    item_ids <- rownames(params)
  }
  
  # Normalize item IDs: remove "X" prefix if present (e.g., "X1" -> "1")
  item_ids <- gsub("^X", "", item_ids)
  
  # Create item parameter matrix with converted values
  item_params <- data.frame(
    item_id = item_ids,
    a = params$a1,
    b = -params$d / params$a1,
    c = params$g,
    stringsAsFactors = FALSE
  )
  
  return(item_params)
}

# Load item parameters with fallback strategy
prepare_item_parameters <- function(primary_path, fallback_path) {
  cat("Loading item parameters with fallback strategy:\n")
  cat("  Primary (full benchmark):", primary_path, "\n")
  cat("  Fallback (TinyBenchmark):", fallback_path, "\n\n")
  
  # Load primary parameters (full benchmark - e.g., 844 items)
  cat("Reading primary item parameters...\n")
  primary_params <- prepare_item_parameters_single(primary_path)
  cat("  Items from primary source:", nrow(primary_params), "\n")
  
  # Load fallback parameters (TinyBenchmark 100 items)
  cat("Reading fallback item parameters...\n")
  fallback_params <- prepare_item_parameters_single(fallback_path)
  cat("  Items from fallback source:", nrow(fallback_params), "\n")
  
  # Merge: prioritize primary, use fallback for missing items
  item_params <- primary_params
  
  # Add fallback items that are not in primary
  missing_items <- fallback_params[!fallback_params$item_id %in% primary_params$item_id, ]
  if (nrow(missing_items) > 0) {
    item_params <- rbind(item_params, missing_items)
    cat("  Items added from fallback:", nrow(missing_items), "\n")
  }
  
  cat("  Total items available:", nrow(item_params), "\n\n")
  
  # Handle edge cases
  invalid_items <- which(item_params$a <= 0 | is.na(item_params$b) | is.infinite(item_params$b))
  if (length(invalid_items) > 0) {
    cat("  Warning:", length(invalid_items), "items have invalid parameters\n")
  }
  
  cat("  Parameter ranges:\n")
  cat("    a:", round(min(item_params$a, na.rm = TRUE), 3), "to", 
      round(max(item_params$a, na.rm = TRUE), 3), "\n")
  cat("    b:", round(min(item_params$b, na.rm = TRUE), 3), "to", 
      round(max(item_params$b, na.rm = TRUE), 3), "\n")
  cat("    c:", round(min(item_params$c, na.rm = TRUE), 3), "to", 
      round(max(item_params$c, na.rm = TRUE), 3), "\n")
  
  return(item_params)
}


# Compute p-IRT accuracy for a model  
# For TinyBenchmark: 100 observed items, predict on full benchmark
compute_theta_eap <- function(model_name, item_params, response_matrix, selected_item_ids) {
  # Get model responses
  model_idx <- which(response_matrix[[1]] == model_name)
  if (length(model_idx) == 0) {
    return(NULL)
  }
  
  # Get all item IDs from response matrix
  all_response_item_ids <- colnames(response_matrix)[-1]
  
  # Match selected TinyBenchmark numeric indices in the response matrix
  # Both should now be numeric strings (e.g., "1", "6", "24")
  subset_cols <- which(all_response_item_ids %in% selected_item_ids)
  
  if (length(subset_cols) == 0) {
    warning(paste("No matching TinyBenchmark items found for model", model_name))
    return(NULL)
  }
  
  subset_item_ids <- all_response_item_ids[subset_cols]
  n_subset_items <- length(subset_item_ids)
  
  # Full benchmark: ALL items in item parameters
  all_item_ids <- item_params$item_id
  n_all_items <- length(all_item_ids)
  
  # Get responses for the selected TinyBenchmark items only
  subset_responses <- as.numeric(response_matrix[model_idx, subset_cols + 1])
  
  # Match item parameters for the observed subset
  item_params_subset <- item_params[item_params$item_id %in% subset_item_ids, ]
  
  # Convert (a, b, c) back to (a1, d, g) for mirt
  item_params_mirt <- data.frame(
    item_id = item_params_subset$item_id,
    a1 = item_params_subset$a,
    d = -item_params_subset$a * item_params_subset$b,  # d = -a*b
    g = item_params_subset$c
  )
  
  # Estimate theta from the observed 100 items using mirt EAP
  theta_result <- estimate_theta_mirt(subset_responses, item_params_mirt)
  
  return(list(
    Model_Name = model_name,
    Theta_WLE = theta_result$theta,
    SE = theta_result$se
  ))
}



# Main execution
cat("\n1. Loading TinyBenchmark selected items...\n")
if (!file.exists(SELECTED_ITEMS_FILE)) {
  cat("ERROR: Selected items file not found:", SELECTED_ITEMS_FILE, "\n")
  cat("Please generate TinyBenchmark item IDs first.\n")
  quit(status = 1)
}
selected_items_df <- read.csv(SELECTED_ITEMS_FILE, stringsAsFactors = FALSE)
# Use numeric indices (e.g., "1", "6", "24") not Mercury IDs
selected_item_ids <- as.character(selected_items_df$item_index)
cat("  Number of selected items:", length(selected_item_ids), "\n")
cat("  First few items:", paste(head(selected_item_ids, 10), collapse=", "), "\n")

cat("\n2. Preparing item parameters...\n")
item_params <- prepare_item_parameters(ITEM_PARAMS_PRIMARY, ITEM_PARAMS_FALLBACK)

cat("\n3. Reading response matrix...\n")
response_matrix <- read.csv(RESPONSE_FILE, stringsAsFactors = FALSE, check.names = FALSE)
model_names <- response_matrix[[1]]
n_models <- length(model_names)
cat("  Number of models:", n_models, "\n")
cat("  Number of items in matrix:", ncol(response_matrix) - 1, "\n")

cat("\n4. Computing p-IRT accuracy estimates...\n")
results <- list()

for (i in 1:n_models) {
  model_name <- model_names[i]
  
  if (i %% 50 == 0 || i == 1) {
    cat("  Progress:", i, "/", n_models, "\n")
  }
  
  result <- tryCatch({
    compute_theta_eap(model_name, item_params, response_matrix, selected_item_ids)
  }, error = function(e) {
    cat("    Error processing", model_name, ":", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(result)) {
    results[[i]] <- data.frame(
      model = model_name,
      Theta_WLE = result$Theta_WLE,
      SE = result$SE,
      stringsAsFactors = FALSE
    )
  }
}

# Combine results
cat("\n5. Saving results...\n")
results_df <- do.call(rbind, results[!sapply(results, is.null)])

write.csv(results_df, OUTPUT_FILE, row.names = FALSE)
cat("  Results saved to:", OUTPUT_FILE, "\n")
