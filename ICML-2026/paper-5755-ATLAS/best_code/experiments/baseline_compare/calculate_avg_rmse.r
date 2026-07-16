#!/usr/bin/env Rscript

# Script to calculate the average of RMSE_single from CAT files

# Function to check and install packages if needed
install_required_packages <- function(packages) {
  # Set a CRAN mirror
  repos <- "https://cloud.r-project.org"
  
  # Check for each package
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat(paste("Installing package:", pkg, "\n"))
      install.packages(pkg, repos = repos)
      if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat(paste("Failed to install package:", pkg, "\n"))
      }
    } else {
      cat(paste("Package already installed:", pkg, "\n"))
    }
  }
}

# Required packages
required_packages <- c("dplyr")

# Install/load required packages
install_required_packages(required_packages)

#############################################
# Function to compare Theta whole and Theta reduced
#############################################



read_and_process <- function(theta_whole_file_path, theta_reduced_file_path) {
  cat("Processing file:", theta_reduced_file_path, "\n")
  
  data_whole <- read.csv(theta_whole_file_path, stringsAsFactors = FALSE)
  data_reduced <- read.csv(theta_reduced_file_path, stringsAsFactors = FALSE)
  
  if ("Theta_WLE" %in% colnames(data_reduced) && "Theta_WLE" %in% colnames(data_whole)) {
    cat("  Theta_WLE columns found\n")
    
    # Find common model names to ensure proper matching
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
    
    # Calculate RMSE and MAE
    rmse <- sqrt(mean((data_reduced_filtered$Theta_WLE - data_whole_filtered$Theta_WLE)^2, na.rm = TRUE))
    mae <- mean(abs(data_reduced_filtered$Theta_WLE - data_whole_filtered$Theta_WLE), na.rm = TRUE)
    
    # Return results
    return(data.frame(
      file = theta_reduced_file_path,
      model_count = nrow(data_reduced),
      common_models = length(common_models),
      rmse = rmse,
      mae = mae
    ))
  } else {
    cat("  Theta_WLE column not found\n")
    return(NULL)
  }
}

# Process each file and collect results
results <- list()

# List all files that might contain RMSE_single column
theta_whole_file_path <- "winogrande/irt_person_scores_WLE_SE_test.csv"
# theta_reduced_file_path <- "/store01/nchawla/pli9/llmbenchmark/baseline_compare/random_100_winogrande_theta_results.csv"
PATH <- "experiments/baseline_compare/"

# Use ignore.case=TRUE to match both Winogrande and winogrande
winogrande_files <- list.files(path = PATH, pattern = ".*[Ww]inogrande.*theta_results.*\\.csv$", full.names = TRUE, ignore.case = TRUE)
# theta_reduced_file_path <- unique(c(winogrande_files))
cat("Found", length(winogrande_files), "Theta reduced result files\n")


for (file in winogrande_files) {
  result <- read_and_process(theta_whole_file_path, file)
  if (!is.null(result)) {
    results[[length(results) + 1]] <- result
  }
}



theta_whole_file_path <- "truthfulqa/irt_person_scores_WLE_SE_test.csv"
PATH <- "experiments/baseline_compare/"

truthfulqa_files <- list.files(path = PATH, pattern = ".*[Tt]ruthfulqa.*theta_results.*\\.csv$", full.names = TRUE, ignore.case = TRUE)
cat("Found", length(truthfulqa_files), "Theta reduced result files\n")


for (file in truthfulqa_files) {
  result <- read_and_process(theta_whole_file_path, file)
  if (!is.null(result)) {
    results[[length(results) + 1]] <- result
  }
}


theta_whole_file_path <- "hellaswag/irt_person_scores_WLE_SE_test.csv"
PATH <- "experiments/baseline_compare/"

hellaswag_files <- list.files(path = PATH, pattern = ".*[Hh]ellaswag.*theta_results.*\\.csv$", full.names = TRUE, ignore.case = TRUE)
cat("Found", length(hellaswag_files), "Theta reduced result files\n")


for (file in hellaswag_files) {
  result <- read_and_process(theta_whole_file_path, file)
  if (!is.null(result)) {
    results[[length(results) + 1]] <- result
  }
}


theta_whole_file_path <- "gsm8k/irt_person_scores_WLE_SE_test.csv"
PATH <- "experiments/baseline_compare/"

gsm8k_files <- list.files(path = PATH, pattern = ".*[Gg]sm8k.*theta_results.*\\.csv$", full.names = TRUE, ignore.case = TRUE)
cat("Found", length(gsm8k_files), "Theta reduced result files\n")


for (file in gsm8k_files) {
  result <- read_and_process(theta_whole_file_path, file)
  if (!is.null(result)) {
    results[[length(results) + 1]] <- result
  }
}


theta_whole_file_path <- "arc/irt_person_scores_WLE_SE_test.csv"
PATH <- "experiments/baseline_compare/"

arc_files <- list.files(path = PATH, pattern = ".*[Aa]rc.*theta_results.*\\.csv$", full.names = TRUE, ignore.case = TRUE)
cat("Found", length(arc_files), "Theta reduced result files\n")


for (file in arc_files) {
  result <- read_and_process(theta_whole_file_path, file)
  if (!is.null(result)) {
    results[[length(results) + 1]] <- result
  }
}

# Combine results
if (length(results) > 0) {
  all_results <- do.call(rbind, results)
  
  # Print summary
  cat("\n=== Summary of RMSE_CAT Results ===\n")
  print(all_results)
  
  # Save results to a CSV file
  write.csv(all_results, paste0(PATH, "rmse_summary.csv"), row.names = FALSE)
  cat("\nResults saved to rmse_summary.csv\n")
} else {
  cat("\nNo files with RMSE_CAT column were found\n")
}
