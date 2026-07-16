#!/usr/bin/env Rscript
# Compare theta_whole with theta_reduced for Metabench, Random100, and TinyBenchmark
# Parameterized for any benchmark

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
BENCHMARK <- if (!is.null(.cli_args$benchmark)) .cli_args$benchmark else "arc"

cat("\n", rep("=", 70), "\n", sep = "")
cat("Comparing Theta_Whole vs Theta_Reduced\n")
cat("Benchmark:", BENCHMARK, "\n")
cat(rep("=", 70), "\n\n", sep = "")

# Base directories
BASE_DIR <- "baseline_compare"
BENCHMARK_DIR <- paste0(BENCHMARK)

# Load theta estimation utilities
source(paste0(BASE_DIR, "/theta_estimation_utils.r"))

# File paths
ITEM_PARAMS_FILE <- paste0(BENCHMARK_DIR, "/irt_item_parameters_combined.csv")
THETA_WHOLE_FILE <- paste0(BENCHMARK_DIR, "/irt_person_scores_WLE_SE.csv")
RESPONSE_FILE <- paste0("data/gaussian_sampled_", BENCHMARK, "_response_matrix_train.csv")

# Output file
OUTPUT_FILE <- paste0(BASE_DIR, "/theta_comparison_", BENCHMARK, ".csv")

#########################################
# Check if required files exist
#########################################
if (!file.exists(ITEM_PARAMS_FILE)) {
  cat("ERROR: Item parameters file not found:", ITEM_PARAMS_FILE, "\n")
  quit(status = 1)
}
if (!file.exists(THETA_WHOLE_FILE)) {
  cat("ERROR: Theta_whole file not found:", THETA_WHOLE_FILE, "\n")
  quit(status = 1)
}
if (!file.exists(RESPONSE_FILE)) {
  cat("ERROR: Response matrix file not found:", RESPONSE_FILE, "\n")
  quit(status = 1)
}

#########################################
# Load item parameters and theta_whole
#########################################
cat("1. Loading item parameters and theta_whole...\n")

# Load item parameters
item_params_raw <- read.csv(ITEM_PARAMS_FILE, stringsAsFactors = FALSE)
cat("  Item parameters loaded:", nrow(item_params_raw), "items\n")

# Prepare item parameters (keep in a1, d, g format for mirt)
if ("X" %in% names(item_params_raw)) {
  item_ids <- gsub("^X", "", item_params_raw$X)
} else {
  item_ids <- gsub("^X", "", rownames(item_params_raw))
}

item_params <- data.frame(
  item_id = item_ids,
  a1 = item_params_raw$a1,
  d = item_params_raw$d,
  g = item_params_raw$g,
  stringsAsFactors = FALSE
)

cat("  Item parameter ranges:\n")
cat("    a1:", round(min(item_params$a1, na.rm = TRUE), 3), "to", 
    round(max(item_params$a1, na.rm = TRUE), 3), "\n")
cat("    d:", round(min(item_params$d, na.rm = TRUE), 3), "to", 
    round(max(item_params$d, na.rm = TRUE), 3), "\n")
cat("    g:", round(min(item_params$g, na.rm = TRUE), 3), "to", 
    round(max(item_params$g, na.rm = TRUE), 3), "\n")

# Load theta_whole
theta_whole_df <- read.csv(THETA_WHOLE_FILE, stringsAsFactors = FALSE)
cat("  Theta_whole loaded:", nrow(theta_whole_df), "models\n")
cat("  Columns:", paste(names(theta_whole_df), collapse=", "), "\n")

# Load response matrix
response_matrix <- read.csv(RESPONSE_FILE, stringsAsFactors = FALSE, check.names = FALSE)
cat("  Response matrix loaded:", nrow(response_matrix), "models,", 
    ncol(response_matrix) - 1, "items\n")

# Get model names
model_names <- response_matrix[[1]]
all_item_ids <- colnames(response_matrix)[-1]

#########################################
# Function to estimate theta from subset
#########################################
estimate_theta_from_subset <- function(model_name, subset_item_ids, method_name) {
  # Get model index
  model_idx <- which(response_matrix[[1]] == model_name)
  if (length(model_idx) == 0) {
    return(NULL)
  }
  
  # Find matching item columns
  subset_cols <- which(all_item_ids %in% subset_item_ids)
  if (length(subset_cols) == 0) {
    warning(paste("No matching items found for", method_name, "- model", model_name))
    return(NULL)
  }
  
  # Get responses for subset
  subset_responses <- as.numeric(response_matrix[model_idx, subset_cols + 1])
  
  # Get item parameters for subset
  item_params_subset <- item_params[item_params$item_id %in% subset_item_ids, ]
  
  if (nrow(item_params_subset) == 0) {
    warning(paste("No item parameters found for", method_name, "- model", model_name))
    return(NULL)
  }
  
  # Estimate theta using mirt EAP
  theta_result <- estimate_theta_mirt(subset_responses, item_params_subset)
  
  return(list(
    theta = theta_result$theta,
    se = theta_result$se,
    n_items = length(subset_item_ids)
  ))
}

#########################################
# Load item selections for each method
#########################################
cat("\n2. Loading item selections...\n")

# TinyBenchmark
tiny_file <- paste0(BASE_DIR, "/tiny", BENCHMARK, "_numeric_indices.csv")
if (file.exists(tiny_file)) {
  tiny_items <- read.csv(tiny_file, stringsAsFactors = FALSE)
  tiny_item_ids <- as.character(tiny_items$item_index)
  cat("  TinyBenchmark items loaded:", length(tiny_item_ids), "items\n")
} else {
  cat("  TinyBenchmark file not found:", tiny_file, "\n")
  tiny_item_ids <- NULL
}

# Random100
random100_file <- paste0(BASE_DIR, "/random_100_", BENCHMARK, "_selected_items.csv")
if (file.exists(random100_file)) {
  random100_items <- read.csv(random100_file, stringsAsFactors = FALSE)
  # Check column name
  if ("item_id" %in% names(random100_items)) {
    random100_item_ids <- gsub("^X", "", as.character(random100_items$item_id))
  } else if ("item_index" %in% names(random100_items)) {
    random100_item_ids <- as.character(random100_items$item_index)
  } else {
    random100_item_ids <- gsub("^X", "", as.character(random100_items[[1]]))
  }
  cat("  Random100 items loaded:", length(random100_item_ids), "items\n")
} else {
  cat("  Random100 file not found:", random100_file, "\n")
  random100_item_ids <- NULL
}

# Metabench - Primary
metabench_primary_file <- paste0(BASE_DIR, "/metabench_", BENCHMARK, "_item_ids_primary.csv")
if (file.exists(metabench_primary_file)) {
  metabench_primary_items <- read.csv(metabench_primary_file, stringsAsFactors = FALSE)
  # Extract numeric IDs
  if ("item_id" %in% names(metabench_primary_items)) {
    metabench_primary_item_ids <- gsub("^X", "", as.character(metabench_primary_items$item_id))
  } else {
    metabench_primary_item_ids <- gsub("^X", "", as.character(metabench_primary_items[[1]]))
  }
  cat("  Metabench Primary items loaded:", length(metabench_primary_item_ids), "items\n")
} else {
  cat("  Metabench Primary file not found:", metabench_primary_file, "\n")
  metabench_primary_item_ids <- NULL
}

# Metabench - Secondary
metabench_secondary_file <- paste0(BASE_DIR, "/metabench_", BENCHMARK, "_item_ids_secondary.csv")
if (file.exists(metabench_secondary_file)) {
  metabench_secondary_items <- read.csv(metabench_secondary_file, stringsAsFactors = FALSE)
  # Extract numeric IDs
  if ("item_id" %in% names(metabench_secondary_items)) {
    metabench_secondary_item_ids <- gsub("^X", "", as.character(metabench_secondary_items$item_id))
  } else {
    metabench_secondary_item_ids <- gsub("^X", "", as.character(metabench_secondary_items[[1]]))
  }
  cat("  Metabench Secondary items loaded:", length(metabench_secondary_item_ids), "items\n")
} else {
  cat("  Metabench Secondary file not found:", metabench_secondary_file, "\n")
  metabench_secondary_item_ids <- NULL
}

#########################################
# Process all models
#########################################
cat("\n3. Calculating theta_reduced for all models...\n")

results_list <- list()

for (i in 1:nrow(theta_whole_df)) {
  model_name <- theta_whole_df$Model_Name[i]
  theta_whole <- theta_whole_df$Theta_WLE[i]
  
  if (i %% 100 == 0 || i == 1) {
    cat("  Progress:", i, "/", nrow(theta_whole_df), "\n")
  }
  
  result_row <- data.frame(
    model = model_name,
    theta_whole = theta_whole,
    stringsAsFactors = FALSE
  )
  
  # TinyBenchmark
  if (!is.null(tiny_item_ids)) {
    tiny_result <- tryCatch({
      estimate_theta_from_subset(model_name, tiny_item_ids, "TinyBenchmark")
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(tiny_result)) {
      result_row$theta_tinybenchmark <- tiny_result$theta
      result_row$n_items_tinybenchmark <- tiny_result$n_items
      result_row$mae_tinybenchmark <- abs(tiny_result$theta - theta_whole)
    } else {
      result_row$theta_tinybenchmark <- NA
      result_row$n_items_tinybenchmark <- NA
      result_row$mae_tinybenchmark <- NA
    }
  }
  
  # Random100
  if (!is.null(random100_item_ids)) {
    random_result <- tryCatch({
      estimate_theta_from_subset(model_name, random100_item_ids, "Random100")
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(random_result)) {
      result_row$theta_random100 <- random_result$theta
      result_row$n_items_random100 <- random_result$n_items
      result_row$mae_random100 <- abs(random_result$theta - theta_whole)
    } else {
      result_row$theta_random100 <- NA
      result_row$n_items_random100 <- NA
      result_row$mae_random100 <- NA
    }
  }
  
  # Metabench Primary
  if (!is.null(metabench_primary_item_ids)) {
    metabench_primary_result <- tryCatch({
      estimate_theta_from_subset(model_name, metabench_primary_item_ids, "Metabench_Primary")
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(metabench_primary_result)) {
      result_row$theta_metabench_primary <- metabench_primary_result$theta
      result_row$n_items_metabench_primary <- metabench_primary_result$n_items
      result_row$mae_metabench_primary <- abs(metabench_primary_result$theta - theta_whole)
    } else {
      result_row$theta_metabench_primary <- NA
      result_row$n_items_metabench_primary <- NA
      result_row$mae_metabench_primary <- NA
    }
  }
  
  # Metabench Secondary
  if (!is.null(metabench_secondary_item_ids)) {
    metabench_secondary_result <- tryCatch({
      estimate_theta_from_subset(model_name, metabench_secondary_item_ids, "Metabench_Secondary")
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(metabench_secondary_result)) {
      result_row$theta_metabench_secondary <- metabench_secondary_result$theta
      result_row$n_items_metabench_secondary <- metabench_secondary_result$n_items
      result_row$mae_metabench_secondary <- abs(metabench_secondary_result$theta - theta_whole)
    } else {
      result_row$theta_metabench_secondary <- NA
      result_row$n_items_metabench_secondary <- NA
      result_row$mae_metabench_secondary <- NA
    }
  }
  
  results_list[[i]] <- result_row
}

# Combine results
results_df <- do.call(rbind, results_list)

#########################################
# Save results
#########################################
cat("\n4. Saving results...\n")
write.csv(results_df, OUTPUT_FILE, row.names = FALSE)
cat("  Results saved to:", OUTPUT_FILE, "\n")

#########################################
# Calculate and display summary statistics
#########################################
cat("\n", rep("=", 70), "\n", sep = "")
cat("SUMMARY STATISTICS - ", toupper(BENCHMARK), "\n", sep = "")
cat(rep("=", 70), "\n\n", sep = "")

# Total items
cat("Total items in full benchmark:", nrow(item_params), "\n\n")

# TinyBenchmark
if ("mae_tinybenchmark" %in% names(results_df)) {
  cat("TinyBenchmark:\n")
  cat("  Items used:", unique(na.omit(results_df$n_items_tinybenchmark))[1], "\n")
  cat("  MAE (mean):", round(mean(results_df$mae_tinybenchmark, na.rm = TRUE), 6), "\n")
  cat("  MAE (SD):", round(sd(results_df$mae_tinybenchmark, na.rm = TRUE), 6), "\n")
  cat("  Models processed:", sum(!is.na(results_df$mae_tinybenchmark)), "\n\n")
}

# Random100
if ("mae_random100" %in% names(results_df)) {
  cat("Random100:\n")
  cat("  Items used:", unique(na.omit(results_df$n_items_random100))[1], "\n")
  cat("  MAE (mean):", round(mean(results_df$mae_random100, na.rm = TRUE), 6), "\n")
  cat("  MAE (SD):", round(sd(results_df$mae_random100, na.rm = TRUE), 6), "\n")
  cat("  Models processed:", sum(!is.na(results_df$mae_random100)), "\n\n")
}

# Metabench Primary
if ("mae_metabench_primary" %in% names(results_df)) {
  cat("Metabench Primary:\n")
  cat("  Items used:", unique(na.omit(results_df$n_items_metabench_primary))[1], "\n")
  cat("  MAE (mean):", round(mean(results_df$mae_metabench_primary, na.rm = TRUE), 6), "\n")
  cat("  MAE (SD):", round(sd(results_df$mae_metabench_primary, na.rm = TRUE), 6), "\n")
  cat("  Models processed:", sum(!is.na(results_df$mae_metabench_primary)), "\n\n")
}

# Metabench Secondary
if ("mae_metabench_secondary" %in% names(results_df)) {
  cat("Metabench Secondary:\n")
  cat("  Items used:", unique(na.omit(results_df$n_items_metabench_secondary))[1], "\n")
  cat("  MAE (mean):", round(mean(results_df$mae_metabench_secondary, na.rm = TRUE), 6), "\n")
  cat("  MAE (SD):", round(sd(results_df$mae_metabench_secondary, na.rm = TRUE), 6), "\n")
  cat("  Models processed:", sum(!is.na(results_df$mae_metabench_secondary)), "\n\n")
}

cat(rep("=", 70), "\n", sep = "")
cat("✓ Theta comparison complete for", BENCHMARK, "!\n")
cat("  Results saved to:", OUTPUT_FILE, "\n")
cat(rep("=", 70), "\n\n", sep = "")
