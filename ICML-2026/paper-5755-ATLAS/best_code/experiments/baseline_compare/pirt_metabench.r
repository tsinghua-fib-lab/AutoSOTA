#!/usr/bin/env Rscript
# p-IRT Accuracy Estimation for Metabench
# Estimates full dataset accuracy based on metabench selected items using 3PL IRT model
# Supports both primary and secondary versions

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
cat("p-IRT Accuracy Estimation - Metabench\n")
cat("Benchmark:", BENCHMARK, "\n")
cat("Version:", VERSION, "\n")
cat(rep("=", 70), "\n\n", sep = "")

# File paths
BASE_DIR <- "experiments/baseline_compare"
ITEM_PARAMS_FILE <- paste0(BENCHMARK, "/irt_item_parameters_combined.csv")
RESPONSE_FILE <- paste0("data/gaussian_sampled_", BENCHMARK, "_response_matrix_test.csv")
SELECTED_ITEMS_FILE <- paste0(BASE_DIR, "/metabench_", BENCHMARK, "_item_ids_", VERSION, ".csv")
OUTPUT_FILE <- paste0(BASE_DIR, "/pirt_metabench_", BENCHMARK, "_", VERSION, ".csv")

# Load theta estimation utilities
source(paste0(BASE_DIR, "/theta_estimation_utils.r"))

# Convert item parameters from (a1, d, g, u) to (a, b, c)
prepare_item_parameters <- function(file_path) {
  cat("Reading item parameters from:", file_path, "\n")
  params <- read.csv(file_path, stringsAsFactors = FALSE)
  
  n_items <- nrow(params)
  cat("  Number of items:", n_items, "\n")
  
  # Normalize item IDs: remove "X" prefix if present
  item_ids <- params$X
  item_ids <- gsub("^X", "", item_ids)
  
  # Create item parameter matrix with converted values
  item_params <- data.frame(
    item_id = item_ids,
    a = params$a1,
    b = -params$d / params$a1,
    c = params$g,
    stringsAsFactors = FALSE
  )
  
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

# Compute 3PL probability
compute_3pl_prob <- function(theta, a, b, c) {
  if (is.na(a) || is.na(b) || is.na(c) || a <= 0) {
    return(c)
  }
  
  logit <- a * (theta - b)
  sigma <- 1 / (1 + exp(-logit))
  p <- c + (1 - c) * sigma
  
  return(p)
}

# Estimate theta using simplified approach
estimate_theta_simple <- function(responses, item_params_subset) {
  # Filter to valid items and their responses
  valid_items <- !is.na(item_params_subset$a) & 
                 !is.na(item_params_subset$b) & 
                 !is.na(item_params_subset$c) &
                 !is.infinite(item_params_subset$b) &
                 item_params_subset$a > 0.01
  
  n_valid <- sum(valid_items)
  
  if (n_valid == 0) {
    # No valid items, use simple logit transformation
    p_correct <- mean(responses, na.rm = TRUE)
    if (p_correct <= 0.01) p_correct <- 0.01
    if (p_correct >= 0.99) p_correct <- 0.99
    return(log(p_correct / (1 - p_correct)))
  }
  
  # Use only valid items
  valid_responses <- responses[valid_items]
  valid_params <- item_params_subset[valid_items, ]
  
  # Adjust observed proportion for guessing
  avg_c <- mean(valid_params$c, na.rm = TRUE)
  p_obs <- mean(valid_responses, na.rm = TRUE)
  
  # Adjust for guessing: p_obs = c + (1-c)*p_true
  if (avg_c < 0.99) {
    p_adj <- (p_obs - avg_c) / (1 - avg_c)
  } else {
    p_adj <- p_obs
  }
  
  # Bound the adjusted probability
  if (p_adj <= 0.01) p_adj <- 0.01
  if (p_adj >= 0.99) p_adj <- 0.99
  
  # Simple logit transformation
  theta <- log(p_adj / (1 - p_adj))
  
  # Optionally adjust by average item difficulty
  avg_b <- mean(valid_params$b, na.rm = TRUE)
  if (!is.na(avg_b) && !is.infinite(avg_b) && abs(avg_b) < 10) {
    theta <- theta + avg_b * 0.1
  }
  
  # Bound theta to reasonable range
  theta <- max(-3, min(3, theta))
  
  return(theta)
}

# Compute p-IRT accuracy for a model
compute_pirt_accuracy <- function(model_name, item_params, response_matrix, selected_item_ids) {
  # Get model responses
  model_idx <- which(response_matrix[[1]] == model_name)
  if (length(model_idx) == 0) {
    cat("    Warning: Model", model_name, "not found in response matrix\n")
    return(NULL)
  }
  
  # Get all item IDs from response matrix
  all_item_ids <- colnames(response_matrix)[-1]
  n_all_items <- length(all_item_ids)
  
  # Normalize selected item IDs to match response matrix format
  selected_item_ids_normalized <- selected_item_ids
  selected_item_ids_normalized <- gsub('^"|"$', '', selected_item_ids_normalized)
  
  # Match selected items in response matrix
  subset_cols <- which(all_item_ids %in% selected_item_ids_normalized)
  
  if (length(subset_cols) == 0) {
    cat("    Warning: No matching items found for model", model_name, "\n")
    return(NULL)
  }
  
  n_subset_items <- length(subset_cols)
  
  # Get responses for subset
  subset_responses <- as.numeric(response_matrix[model_idx, subset_cols + 1])
  
  # Match item parameters for subset
  item_params_subset <- item_params[item_params$item_id %in% all_item_ids[subset_cols], ]
  
  # Convert (a, b, c) back to (a1, d, g) for mirt
  item_params_mirt <- data.frame(
    item_id = item_params_subset$item_id,
    a1 = item_params_subset$a,
    d = -item_params_subset$a * item_params_subset$b,  # d = -a*b
    g = item_params_subset$c
  )
  
  # Estimate theta from subset using mirt EAP
  theta_result <- estimate_theta_mirt(subset_responses, item_params_mirt)
  theta_l <- theta_result$theta
  
  # Part 1: Average of observed responses in subset
  avg_observed <- mean(subset_responses, na.rm = TRUE)
  weight_observed <- n_subset_items / n_all_items
  
  # Part 2: Average of predicted probabilities for unobserved items
  unobserved_cols <- setdiff(1:n_all_items, subset_cols)
  n_unobserved_items <- length(unobserved_cols)
  
  predicted_probs <- numeric(n_unobserved_items)
  
  for (i in 1:n_unobserved_items) {
    item_id <- all_item_ids[unobserved_cols[i]]
    item_idx <- which(item_params$item_id == item_id)
    
    if (length(item_idx) > 0) {
      a_i <- item_params$a[item_idx]
      b_i <- item_params$b[item_idx]
      c_i <- item_params$c[item_idx]
      
      predicted_probs[i] <- compute_3pl_prob(theta_l, a_i, b_i, c_i)
    } else {
      predicted_probs[i] <- 0.5
    }
  }
  
  avg_predicted <- if (n_unobserved_items > 0) mean(predicted_probs) else 0
  weight_predicted <- n_unobserved_items / n_all_items
  
  # Combine both parts
  pirt_accuracy <- weight_observed * avg_observed + weight_predicted * avg_predicted
  
  return(list(
    pirt_accuracy = pirt_accuracy,
    theta = theta_l,
    n_subset_items = n_subset_items,
    n_all_items = n_all_items,
    avg_observed = avg_observed,
    avg_predicted = avg_predicted
  ))
}

# Main execution
cat("\n1. Loading selected items...\n")
if (!file.exists(SELECTED_ITEMS_FILE)) {
  cat("ERROR: Selected items file not found:", SELECTED_ITEMS_FILE, "\n")
  cat("Please run get_metabench.py first to generate the item IDs.\n")
  quit(status = 1)
}

selected_items_df <- read.csv(SELECTED_ITEMS_FILE, stringsAsFactors = FALSE)
selected_item_ids <- selected_items_df$item_id
cat("  Number of selected items:", length(selected_item_ids), "\n")
cat("  First few items:", paste(head(selected_item_ids, 10), collapse=", "), "\n")

cat("\n2. Preparing item parameters...\n")
item_params <- prepare_item_parameters(ITEM_PARAMS_FILE)

cat("\n3. Reading response matrix...\n")
response_matrix <- read.csv(RESPONSE_FILE, stringsAsFactors = FALSE, check.names = FALSE)
model_names <- response_matrix[[1]]
n_models <- length(model_names)
cat("  Number of models:", n_models, "\n")

cat("\n4. Computing p-IRT accuracy estimates...\n")
results <- list()

for (i in 1:n_models) {
  model_name <- model_names[i]
  
  if (i %% 50 == 0 || i == 1) {
    cat("  Progress:", i, "/", n_models, "\n")
  }
  
  result <- tryCatch({
    compute_pirt_accuracy(model_name, item_params, response_matrix, selected_item_ids)
  }, error = function(e) {
    cat("    Error processing", model_name, ":", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(result)) {
    results[[i]] <- data.frame(
      model = model_name,
      pirt_accuracy = result$pirt_accuracy,
      theta = result$theta,
      n_subset_items = result$n_subset_items,
      n_all_items = result$n_all_items,
      avg_observed = result$avg_observed,
      avg_predicted = result$avg_predicted,
      stringsAsFactors = FALSE
    )
  }
}

# Combine results
cat("\n5. Saving results...\n")
results_df <- do.call(rbind, results[!sapply(results, is.null)])

write.csv(results_df, OUTPUT_FILE, row.names = FALSE)
cat("  Results saved to:", OUTPUT_FILE, "\n")

# Summary statistics
cat("\n", rep("=", 70), "\n", sep = "")
cat("Summary Statistics:\n")
cat(rep("=", 70), "\n", sep = "")
cat("  Models processed:", nrow(results_df), "\n")
cat("  Mean p-IRT accuracy:", round(mean(results_df$pirt_accuracy, na.rm = TRUE), 4), "\n")
cat("  SD p-IRT accuracy:", round(sd(results_df$pirt_accuracy, na.rm = TRUE), 4), "\n")
cat("  Min/Max p-IRT accuracy:", round(min(results_df$pirt_accuracy, na.rm = TRUE), 4), "/",
    round(max(results_df$pirt_accuracy, na.rm = TRUE), 4), "\n")
cat("  Subset size:", results_df$n_subset_items[1], "\n")
cat("  Full dataset size:", results_df$n_all_items[1], "\n")
cat("  Reduction ratio:", round(results_df$n_subset_items[1] / results_df$n_all_items[1], 4), "\n")
cat("  Mean observed accuracy:", round(mean(results_df$avg_observed, na.rm = TRUE), 4), "\n")
cat("  Mean predicted accuracy:", round(mean(results_df$avg_predicted, na.rm = TRUE), 4), "\n")
cat(rep("=", 70), "\n\n", sep = "")

cat("✓ p-IRT estimation complete!\n\n")
