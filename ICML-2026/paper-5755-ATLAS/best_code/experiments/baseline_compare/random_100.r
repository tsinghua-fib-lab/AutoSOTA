# Random Selection of 100 Items from TruthfulQA and Theta Calculation
# Script with separate functions for item selection, mapping, and theta calculation

# 1. Install and Load Required Packages
if (!require(mirt)) install.packages("mirt")
library(mirt)
library(stats4)
DATASET_NAME <- "gsm8k"
# 2. Define output directory for results
DIRECTORY <- "./"
dir.create(DIRECTORY, recursive = TRUE, showWarnings = FALSE)

# 3. Load response matrix function
load_response_matrix <- function() {
if (DATASET_NAME == "truthfulqa" || DATASET_NAME == "hellaswag" || DATASET_NAME == "winogrande") {
  response_data <- read.csv(paste0("../select_model/gaussian_sampled_", DATASET_NAME, "_response_matrix_train.csv"), 
                          stringsAsFactors = FALSE, check.names = FALSE)
  return(response_data)
} else {
  response_data <- read.csv(paste0("../select_model/gaussian_sampled_", DATASET_NAME, "_response_matrix_train.csv"), 
                          stringsAsFactors = FALSE, check.names = FALSE)
  return(response_data)
}
}

# Function to load item parameters
load_item_parameters <- function() {
  item_params <- read.csv(paste0("../", DATASET_NAME, "/irt_item_parameters_combined.csv"), stringsAsFactors = FALSE)
  cat("Loaded", nrow(item_params), "items from parameter file\n")
  cat("First few parameter IDs:", paste(head(item_params$X, 10), collapse=", "), "...\n")
  return(item_params)
}

# 4. Function 1: Randomly select items from TruthfulQA
randomly_select_items <- function(n_items = 100) {
  # Load response matrix
  response_data <- load_response_matrix()
  
  # Item columns (skip first column which is model names)
  item_columns <- colnames(response_data)[-1]
  
  # Check for constant columns (no variance)
  constant_columns <- c()
  for (col in item_columns) {
    # Check if the column has only one unique value
    if (length(unique(response_data[[col]])) == 1) {
      constant_columns <- c(constant_columns, col)
      cat("Item", col, "has only one response category and will be excluded\n")
    }
  }
  
  # Filter out constant columns
  if (length(constant_columns) > 0) {
    item_columns <- setdiff(item_columns, constant_columns)
    cat("Excluded", length(constant_columns), "items with no variance\n")
  }
  
  # Ensure we have enough items left
  if (length(item_columns) < n_items) {
    cat("Warning: Only", length(item_columns), "items with variance available, using all of them\n")
    n_items <- length(item_columns)
  }
  
  # Randomly select items
  set.seed(42)  # For reproducibility
  selected_items <- sample(item_columns, n_items)
  
  cat("Randomly selected", n_items, "items\n")
  cat("Selected items:", paste(head(selected_items, 10), collapse=", "), "...\n")
  
  # Save the selected items to a file
  write.csv(data.frame(item_id = selected_items), 
           paste0("random_100_", DATASET_NAME, "_selected_items.csv"), 
           row.names = FALSE)
  
  cat("Selected items saved to", paste0("random_100_", DATASET_NAME, "_selected_items.csv"), "\n")
  
  return(selected_items)
}
  
# 5. Function to map selected items to parameter matrix
map_items_to_parameters <- function(selected_items) {
  # Load item parameters
  item_params <- load_item_parameters()
  
  # Print first few selected items for debugging
  cat("First few selected items:", paste(head(selected_items, 10), collapse=", "), "...\n")
  
  # Create mapping table
  mapping_table <- data.frame(
    item_id = character(),
    param_id = character(),
    a1 = numeric(),
    d = numeric(),
    g = numeric(),
    mapped = logical(),
    stringsAsFactors = FALSE
  )
  
  # Extract just the numeric portion from parameters
  cat("Creating numeric mapping for parameters...\n")
  param_numeric <- as.numeric(gsub("[^0-9]", "", item_params$X))
  
  # For each selected item
  for (item_id in selected_items) {
    # Extract the numeric part of the item_id
    item_numeric <- gsub("[^0-9]", "", item_id)
    
    # Try to find a match based on the numeric part
    if (item_numeric != "") {
      idx <- which(param_numeric == as.numeric(item_numeric))
      match_type <- "numeric"
      
      if (length(idx) > 0) {
        param_id <- item_params$X[idx[1]]
      } else {
        # Try direct match
        idx <- which(item_params$X == item_id)
        match_type <- "direct"
        
        # If no match, try removing X prefix
        if (length(idx) == 0) {
          param_id <- gsub("^X", "", item_id)
          idx <- which(item_params$X == param_id)
          match_type <- "removed X"
        }
        
        # If still no match, try adding X prefix
        if (length(idx) == 0) {
          param_id <- paste0("X", item_id)
          idx <- which(item_params$X == param_id)
          match_type <- "added X"
        }
      }
    } else {
      # Try direct match if no numeric part
      idx <- which(item_params$X == item_id)
      match_type <- "direct"
      
      # If no match, try removing X prefix
      if (length(idx) == 0) {
        param_id <- gsub("^X", "", item_id)
        idx <- which(item_params$X == param_id)
        match_type <- "removed X"
      }
      
      # If still no match, try adding X prefix
      if (length(idx) == 0) {
        param_id <- paste0("X", item_id)
        idx <- which(item_params$X == param_id)
        match_type <- "added X"
      }
    }
    
    # If match found
    if (length(idx) > 0) {
      mapping_table <- rbind(mapping_table, data.frame(
        item_id = item_id,
        param_id = item_params$X[idx[1]],
        a1 = item_params$a1[idx[1]],
        d = item_params$d[idx[1]],
        g = item_params$g[idx[1]],
        mapped = TRUE,
        match_type = match_type,
        stringsAsFactors = FALSE
      ))
    } else {
      # No match found
      cat("⚠️ No match found for item", item_id, "\n")
      mapping_table <- rbind(mapping_table, data.frame(
        item_id = item_id,
        param_id = NA,
        a1 = 1,     # Default discrimination parameter
        d = 0,      # Default difficulty parameter
        g = 0.2,    # Default guessing parameter
        mapped = FALSE,
        match_type = "none",
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Report mapping statistics
  mapped_count <- sum(mapping_table$mapped)
  cat("Found parameters for", mapped_count, "out of", length(selected_items), "items\n")
  
  # Display mapping stats by type
  if (mapped_count > 0) {
    match_types <- table(mapping_table$match_type[mapping_table$mapped])
    cat("Matching types:\n")
    print(match_types)
  }
  
  if (mapped_count < length(selected_items)) {
    cat("Warning: Could not find parameters for", length(selected_items) - mapped_count, "items\n")
    cat("Items without parameters will use default values\n")
    
    # Show some examples of unmapped items
    unmapped <- mapping_table[!mapping_table$mapped, "item_id"]
    cat("Examples of unmapped items:", paste(head(unmapped, 5), collapse=", "), "...\n")
  }
  
  return(mapping_table)
}

# 6. Function to calculate theta based on selected items
calculate_theta <- function(selected_items) {
  # Load response matrix
  response_data <- load_response_matrix()
  
  # Extract just the selected columns from the response matrix
  random_data <- response_data[, c(1, which(colnames(response_data) %in% selected_items))] 
  
  # Save the column names as they represent the item IDs
  item_ids <- colnames(random_data)[-1]  # exclude the first column (model names)
  
  # Convert to data frame with proper format for mirt
  model_df <- data.frame(random_data, stringsAsFactors = FALSE)
  colnames(model_df)[1] <- "Model_Name"  # Name the model column
  
  # Extract item response data (exclude Model_Name column)
  response_df <- model_df[, -1, drop = FALSE]
  
  # Map the selected items to parameters
  mapping_table <- map_items_to_parameters(item_ids)
  
  # Fit dummy model just to get structure
  cat("Building dummy model to get structure...\n")
  mod_dummy <- mirt(response_df, 1, itemtype = "3PL", pars = 'values')
  
  # Replace item values
  pars <- mod_dummy
  
  # Set parameter values for each item
  cat("Setting item parameters...\n")
  parameter_count <- 0
  
  # Debug: Check structure of pars
  cat("MIRT pars structure - first 10 rows:\n")
  print(head(pars, 10))
  
  # Create a simpler approach to setting parameters
  # First, create a lookup table for the mapping table keyed by item_id
  item_param_lookup <- list()
  for (i in 1:nrow(mapping_table)) {
    # Add entries for both with and without X prefix to handle mirt model naming
    item_id <- mapping_table$item_id[i]
    item_id_with_x <- paste0("X", gsub("^X", "", item_id))
    
    if (mapping_table$mapped[i] || !is.na(mapping_table$a1[i])) {
      # Store params under both the original ID and with X prefix
      item_param_lookup[[item_id]] <- list(
        a1 = mapping_table$a1[i],
        d = mapping_table$d[i],
        g = mapping_table$g[i]
      )
      
      item_param_lookup[[item_id_with_x]] <- list(
        a1 = mapping_table$a1[i],
        d = mapping_table$d[i],
        g = mapping_table$g[i]
      )
    }
  }
  
  # Print some lookup entries for debugging
  cat("Parameter lookup table has", length(item_param_lookup), "entries\n")
  cat("First few lookup keys:", paste(head(names(item_param_lookup), 10), collapse=", "), "...\n")
  
  # Loop through parameters and set values
  for (i in 1:nrow(pars)) {
    item_name <- pars$item[i]
    param_name <- pars$name[i]
    
    # Only process a1, d, g parameters
    if (param_name %in% c("a1", "d", "g")) {
      # Check if we have parameters for this item
      if (item_name %in% names(item_param_lookup)) {
        # Set the parameter value
        param_value <- item_param_lookup[[item_name]][[param_name]]
        
        # Apply the parameter if it's not NA
        if (!is.na(param_value)) {
          pars$value[i] <- param_value
          parameter_count <- parameter_count + 1
          
          if (i <= 5) {
            cat("Set parameter", param_name, "for item", item_name, "to", param_value, "\n")
          }
        }
      } else if (i <= 10) {
        cat("No parameters found for item:", item_name, "\n")
      }
    }
  }
  
  # Count parameters set by type
  a1_count <- sum(pars$name == "a1" & pars$est == FALSE)
  d_count <- sum(pars$name == "d" & pars$est == FALSE)
  g_count <- sum(pars$name == "g" & pars$est == FALSE)
  
  cat("Parameters set by type: a1=", a1_count, "d=", d_count, "g=", g_count, "\n")
  
  cat("Total parameters set:", parameter_count, "\n")
  
  # Fix all item parameters
  pars$est[pars$name %in% c("a1", "d", "g")] <- FALSE
  
  # Build fixed model with information matrix
  cat("Building the model with fixed parameters...\n")
  mod_fixed <- mirt(response_df, 1, itemtype = "3PL", pars = pars, method = "EM", 
                   technical = list(NCYCLES = 5000, BURNIN = 1000),
                   verbose = FALSE)
  
  # Generate person scores (thetas)
  cat("Generating person scores using WLE...\n")
  thetas <- fscores(mod_fixed, method = "EAP", 
                    full.scores.SE = TRUE,
                    response.pattern = NULL, 
                    quadpts = 61)
  
  # Check for convergence issues
  na_count <- sum(is.na(thetas[,1]))
  nan_count <- sum(is.nan(thetas[,1]))
  cat("Number of NA values:", na_count, "\n")
  cat("Number of NaN values:", nan_count, "\n")
  
  # Handle problematic cases if any
  if (na_count > 0 || nan_count > 0) {
    cat("Handling problematic theta estimates...\n")
    problem_rows <- which(is.na(thetas[,1]) | is.nan(thetas[,1]))
    
    # Use median imputation for problematic cases
    valid_thetas <- thetas[!is.na(thetas[,1]) & !is.nan(thetas[,1]), 1]
    if (length(valid_thetas) > 0) {
      replacement_value <- median(valid_thetas)
      cat("Replacing non-converged values with median theta:", replacement_value, "\n")
      thetas[problem_rows, 1] <- replacement_value
    }
  } else {
    cat("All theta estimates converged successfully.\n")
  }
  
  # Create data frame with model names and theta values
  theta_df <- data.frame(Model_Name = model_df$Model_Name, 
                        Theta_WLE = thetas[, 1], 
                        SE = thetas[, 2])
  
  cat("\n===========================================\n")
  cat("SUMMARY\n")
  cat("===========================================\n")
  cat("Number of models processed:", nrow(theta_df), "\n")
  
  # Save results
  write.csv(theta_df, paste0("random_100_", DATASET_NAME, "_theta_results_EAP.csv"), row.names = FALSE)
  
  # Print first few rows to verify
  cat("First few model names and theta values:\n")
  print(head(theta_df, 5))
  
  return(theta_df)
}
# DATASET_NAME <- "gsm8k"

# 7. Main function to run the analysis
main <- function(n_items = 100) {
  cat("Starting random item analysis with", n_items, "items...\n")
  
  # Step 1: Randomly select items
  selected_items <- randomly_select_items(n_items)
# Load selected items from CSV and extract just the item_id column as a vector
selected_items_df <- read.csv(paste0("experiments/baseline_compare/random_100_", DATASET_NAME, "_selected_items.csv"), stringsAsFactors = FALSE)
selected_items <- as.character(selected_items_df$item_id)
cat("First few selected items:", paste(head(selected_items, 10), collapse=", "), "...\n")
  
  # Step 2: Calculate theta based on selected items
  results <- calculate_theta(selected_items)
  
  cat("Analysis complete. Results saved in the", DIRECTORY, "directory.\n")
  return(results)
}

# 8. Execute the main function
results <- main(100)
