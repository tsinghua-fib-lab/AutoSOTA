#!/usr/bin/env Rscript
# p-IRT Accuracy Estimation Script
# Estimates full dataset accuracy based on subset responses using 3PL IRT model

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
SE_THETA_STOP <- if (!is.null(.cli_args$se_theta_stop)) .cli_args$se_theta_stop else "0.1"
BENCHMARK <- if (!is.null(.cli_args$benchmark)) .cli_args$benchmark else "arc"
cat("\n", rep("=", 70), "\n", sep = "")
cat("p-IRT Accuracy Estimation\n")
cat("SE threshold:", SE_THETA_STOP, "\n")
cat(rep("=", 70), "\n\n", sep = "")


ITEM_PARAMS_FILE <- paste0(BENCHMARK, "/irt_item_parameters_combined.csv")
RESPONSE_FILE <- paste0("data/gaussian_sampled_", BENCHMARK, "_response_matrix_test.csv")
SELECTED_ITEMS_DIR <- paste0(BENCHMARK,"/atlas_", BENCHMARK, "_random/selected_items_",SE_THETA_STOP,"/")
OUTPUT_FILE <- paste0(BENCHMARK, "/pirt_accuracy_se_", SE_THETA_STOP, ".csv")

# Convert item parameters from (a1, d, g, u) to (a, b, c)
prepare_item_parameters <- function(file_path) {
  cat("Reading item parameters from:", file_path, "\n")
  params <- read.csv(file_path, stringsAsFactors = FALSE)
  
  n_items <- nrow(params)
  cat("  Number of items:", n_items, "\n")
  
  # Create item parameter matrix with converted values
  # a1 -> a (discrimination)
  # d -> b (difficulty), where b = -d/a1
  # g -> c (pseudo-guessing)
  
  item_params <- data.frame(
    item_id = params$X,  # Item ID
    a = params$a1,       # Discrimination (a1 -> a)
    b = -params$d / params$a1,  # Difficulty (convert d to b)
    c = params$g,        # Pseudo-guessing (g -> c)
    stringsAsFactors = FALSE
  )
  
  
  return(item_params)
}

# Compute 3PL probability
# p_il = c_i + (1 - c_i) * sigma(a_i * (theta_l - b_i))
# where sigma(x) = 1 / (1 + exp(-x))
compute_3pl_prob <- function(theta, a, b, c) {
  # Handle invalid parameters
  if (is.na(a) || is.na(b) || is.na(c) || a <= 0) {
    return(c)  # If parameters invalid, return guessing parameter
  }
  
  # 3PL model
  logit <- a * (theta - b)
  sigma <- 1 / (1 + exp(-logit))
  p <- c + (1 - c) * sigma
  
  return(p)
}

# Compute p-IRT accuracy estimate for a model
compute_pirt_accuracy <- function(model_name, item_params, response_matrix, selected_items_file) {
  # Read selected items for this model
  if (!file.exists(selected_items_file)) {
    cat("    Warning: Selected items file not found for", model_name, "\n")
    return(NULL)
  }
  
  selected_items <- read.csv(selected_items_file, stringsAsFactors = FALSE)
  
  # Get the item IDs in the subset (I_bj)
  subset_item_ids <- selected_items$item_id
  
  # Get the model's final theta estimate (use the last theta value)
  theta_l <- selected_items$theta[nrow(selected_items)]
  
  # Get all item IDs in the full dataset (I_j)
  all_item_ids <- item_params$item_id
  n_all_items <- length(all_item_ids)
  n_subset_items <- length(subset_item_ids)
  
  # Identify items NOT in the subset (I_j \ I_bj)
  unobserved_item_ids <- setdiff(all_item_ids, subset_item_ids)
  n_unobserved_items <- length(unobserved_item_ids)
  
  # Part 1: Average of observed responses in subset
  # (|I_bj| / |I_j|) * (1 / |I_bj|) * sum(Y_il for i in I_bj)
  observed_scores <- selected_items$score
  avg_observed <- mean(observed_scores)
  weight_observed <- n_subset_items / n_all_items
  
  # Part 2: Average of predicted probabilities for unobserved items
  # (|I_j \ I_bj| / |I_j|) * (1 / |I_j \ I_bj|) * sum(p_il for i in I_j \ I_bj)
  predicted_probs <- numeric(n_unobserved_items)
  
  for (i in 1:n_unobserved_items) {
    item_id <- unobserved_item_ids[i]
    item_idx <- which(item_params$item_id == item_id)
    
    if (length(item_idx) > 0) {
      a_i <- item_params$a[item_idx]
      b_i <- item_params$b[item_idx]
      c_i <- item_params$c[item_idx]
      
      predicted_probs[i] <- compute_3pl_prob(theta_l, a_i, b_i, c_i)
    } else {
      # Item not found in parameters (shouldn't happen)
      predicted_probs[i] <- 0.5
    }
  }
  
  avg_predicted <- if (n_unobserved_items > 0) mean(predicted_probs) else 0
  weight_predicted <- n_unobserved_items / n_all_items
  
  # Combine both parts for final p-IRT estimate
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
cat("\n1. Preparing item parameters...\n")
item_params <- prepare_item_parameters(ITEM_PARAMS_FILE)

cat("\n2. Reading response matrix...\n")
response_matrix <- read.csv(RESPONSE_FILE, stringsAsFactors = FALSE, check.names = FALSE)
model_names <- response_matrix[[1]]
n_models <- length(model_names)
cat("  Number of models:", n_models, "\n")

cat("\n3. Computing p-IRT accuracy estimates...\n")
results <- list()

for (i in 1:n_models) {
  model_name <- model_names[i]
  
  if (i %% 50 == 0 || i == 1) {
    cat("  Progress:", i, "/", n_models, "\n")
  }
  
  # Create safe filename for this model
  safe_model_name <- gsub("/|\\\\|:|\\*|\\?|\"|<|>|\\||[[:space:]]", "_", model_name)
  selected_items_file <- paste0(SELECTED_ITEMS_DIR, safe_model_name, "_items.csv")
  
  # Compute p-IRT accuracy
  result <- tryCatch({
    compute_pirt_accuracy(model_name, item_params, response_matrix, selected_items_file)
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
cat("\n4. Saving results...\n")
results_df <- do.call(rbind, results[!sapply(results, is.null)])

# Create output directory if needed
dir.create(dirname(OUTPUT_FILE), recursive = TRUE, showWarnings = FALSE)

# Save results
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
cat("  Mean subset size:", round(mean(results_df$n_subset_items, na.rm = TRUE), 1), "\n")
cat("  Mean observed accuracy:", round(mean(results_df$avg_observed, na.rm = TRUE), 4), "\n")
cat("  Mean predicted accuracy:", round(mean(results_df$avg_predicted, na.rm = TRUE), 4), "\n")
cat(rep("=", 70), "\n\n", sep = "")

cat("✓ p-IRT estimation complete!\n\n")
