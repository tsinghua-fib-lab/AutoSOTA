#!/usr/bin/env Rscript

# Script to calculate the average of RMSE_single from CAT files

#############################################
# Function to compare Theta whole and Theta reduced
#############################################



read_and_process <- function(theta_whole_file_path, theta_reduced_file_path) {
  cat("Processing file:", theta_reduced_file_path, "\n")
  
  data_whole <- read.csv(theta_whole_file_path, stringsAsFactors = FALSE)
  data_reduced <- read.csv(theta_reduced_file_path, stringsAsFactors = FALSE)
  
  if ("F1" %in% colnames(data_reduced) && "Theta_WLE" %in% colnames(data_whole)) {
    cat("  Theta_WLE columns found\n")
    
    # Check if Model_Name column exists in reduced data
    if ("Model_Name" %in% colnames(data_reduced)) {
      # Match by Model_Name
      cat("  Matching by Model_Name\n")
      common_models <- intersect(data_reduced$Model_Name, data_whole$Model_Name)
      
      if(length(common_models) == 0) {
        cat("  No common models found between the datasets\n")
        return(NULL)
      }
      
      # Filter both datasets to include only common models
      data_reduced_filtered <- data_reduced[data_reduced$Model_Name %in% common_models, ]
      data_whole_filtered <- data_whole[data_whole$Model_Name %in% common_models, ]
      
      # Ensure they're in the same order
      data_reduced_filtered <- data_reduced_filtered[order(data_reduced_filtered$Model_Name), ]
      data_whole_filtered <- data_whole_filtered[order(data_whole_filtered$Model_Name), ]
      
      theta_reduced <- data_reduced_filtered$F1
      theta_whole <- data_whole_filtered$Theta_WLE
      n_models <- length(common_models)
    } else {
      # Match by row order (no Model_Name in reduced data)
      cat("  Matching by row order (no Model_Name in reduced data)\n")
      
      # Take the minimum number of rows to compare
      n_models <- min(nrow(data_reduced), nrow(data_whole))
      
      if(n_models == 0) {
        cat("  No data to compare\n")
        return(NULL)
      }
      
      theta_reduced <- data_reduced$F1[1:n_models]
      theta_whole <- data_whole$Theta_WLE[1:n_models]
    }
    
    # Calculate RMSE and MAE
    rmse <- sqrt(mean((theta_reduced - theta_whole)^2, na.rm = TRUE))
    mae <- mean(abs(theta_reduced - theta_whole), na.rm = TRUE)
    
    # Return results
    return(data.frame(
      file = theta_reduced_file_path,
      model_count = nrow(data_reduced),
      compared_models = n_models,
      rmse = rmse,
      mae = mae
    ))
  } else {
    cat("  Theta_WLE or F1 column not found\n")
    return(NULL)
  }
}

# Process each file and collect results
results <- list()
BENCHMARK <- "arc"
# List all files that might contain RMSE_single column
theta_whole_file_path <- paste0(BENCHMARK, "/irt_person_scores_WLE_SE_test.csv")
theta_reduced_file_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/irt_person_scores_tiny", BENCHMARK, ".csv")

# Compare theta_whole and theta_reduced using the specified file paths
result <- read_and_process(theta_whole_file_path, theta_reduced_file_path)
if (!is.null(result)) {
  results[[length(results) + 1]] <- result
  cat("\n=== Comparison for", BENCHMARK, "===\n")
  cat("RMSE:", result$rmse, "\n")
  cat("MAE:", result$mae, "\n")
  cat("Compared models:", result$compared_models, "\n")
}

