# TinyARC Theta Calculation
# Script to calculate theta values based on pre-selected items from tinyARC_item_ids.csv

# 1. Install and Load Required Packages
if (!require(mirt)) install.packages("mirt")
library(mirt)
library(stats4)
DATASET_NAME <- "arc"
DATASET <- DATASET_NAME
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

# 4. Function to load pre-selected items from tinyARC_numeric_indices.csv
load_preselected_items <- function() {
  # Load the preselected items from CSV
  preselected_data <- read.csv(paste0("tiny", DATASET_NAME, "_numeric_indices.csv"), stringsAsFactors = FALSE)
  
  # Extract the item_index column
  selected_items <- as.character(preselected_data$item_index)
  
  # Remove duplicates if any
  selected_items <- unique(selected_items)
  
  cat("Loaded", length(selected_items), "pre-selected items from tiny", DATASET_NAME, "_numeric_indices.csv\n")
  cat("First few items:", paste(head(selected_items, 10), collapse=", "), "...\n")
  
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
  # First format item_ids to match the column names in response_data
  formatted_item_ids <- paste0("X", selected_items)
  
  # Check which formatted item IDs are present in the response matrix
  available_items <- formatted_item_ids[formatted_item_ids %in% colnames(response_data)]
  if (length(available_items) < length(selected_items)) {
    missing_count <- length(selected_items) - length(available_items)
    cat("Warning:", missing_count, "items not found in the response matrix\n")
    
    # Also try with unformatted IDs
    direct_matches <- selected_items[selected_items %in% colnames(response_data)]
    if (length(direct_matches) > 0) {
      cat("Found", length(direct_matches), "direct matches without X prefix\n")
      available_items <- c(available_items, direct_matches)
    }
  }
  
  # If no items are found, stop execution
  if (length(available_items) == 0) {
    stop("No items from tiny", DATASET, "_numeric_indices.csv found in the response matrix. Check item ID format.")
  }
  
  # Extract the columns we need from the response matrix
  columns_to_extract <- c(1, which(colnames(response_data) %in% c(available_items, selected_items)))
  random_data <- response_data[, columns_to_extract, drop = FALSE]
  
  # Save the column names as they represent the item IDs
  item_ids <- colnames(random_data)[-1]  # exclude the first column (model names)
  
  # Check for items with no variance (all 0s or all 1s) and remove them
  constant_columns <- character(0)
  for (col in colnames(random_data)[-1]) {  # Skip the first column (Model_Name)
    if (length(unique(random_data[[col]])) == 1) {
      cat("Item", col, "has no variance (all responses are", random_data[1, col], ") - removing from analysis\n")
      constant_columns <- c(constant_columns, col)
    }
  }
  
  # Remove constant items
  if (length(constant_columns) > 0) {
    # Remove the constant columns from the data frame
    # First make sure all column names exist in the data frame
    valid_constant_columns <- constant_columns[constant_columns %in% colnames(random_data)]
    
    if (length(valid_constant_columns) > 0) {
      random_data <- random_data[, !(colnames(random_data) %in% valid_constant_columns), drop = FALSE]
      
      # Update item_ids to exclude constant items
      item_ids <- colnames(random_data)[-1]  # exclude the first column (Model_Name)
      
      cat("Removed", length(valid_constant_columns), "constant items. Continuing with", length(item_ids), "items.\n")
      
      # If no items left, exit
      if (length(item_ids) == 0) {
        stop("No items with variance remaining. Cannot calculate theta values.")
      }
    }
  }
  
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
  
  # Counter for parameters by type
  a1_count <- 0
  d_count <- 0
  g_count <- 0
  
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
          pars$est[i] <- FALSE  # Fix the parameter - don't estimate it
          parameter_count <- parameter_count + 1
          
          # Count by parameter type
          if (param_name == "a1") a1_count <- a1_count + 1
          if (param_name == "d") d_count <- d_count + 1
          if (param_name == "g") g_count <- g_count + 1
          
          if (i <= 5) {
            cat("Set parameter", param_name, "for item", item_name, "to", param_value, "\n")
          }
        }
      } else if (i <= 10) {
        cat("No parameters found for item:", item_name, "\n")
      }
    }
  }
  
  # Report parameters set by type
  cat("Parameters set by type: a1=", a1_count, "d=", d_count, "g=", g_count, "\n")
  cat("Total parameters set:", parameter_count, "\n")
  cat("Expected parameters (3 per item):", length(item_ids) * 3, "\n")
  
  # Double check that all parameters are fixed
  pars$est[pars$name %in% c("a1", "d", "g")] <- FALSE
  
  # Build fixed model with information matrix
  cat("Building the model with fixed parameters...\n")
  tryCatch({
    mod_fixed <- mirt(response_df, 1, itemtype = "3PL", pars = pars, method = "EM", 
                     technical = list(NCYCLES = 5000, BURNIN = 1000),
                     verbose = FALSE)
  }, error = function(e) {
    cat("Error in building model:\n", e$message, "\n")
    
    # Check if it's a constant response error and provide better guidance
    if(grepl("only one response category", e$message)) {
      stop("Some items have constant responses (all 0s or all 1s). Please check the data preprocessing.")
    } else {
      stop("Failed to build IRT model. See error message above.")
    }
  })
  
  # Generate person scores (thetas)
  cat("Generating person scores using EAP...\n")
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
  write.csv(theta_df, paste0("tiny_", DATASET_NAME, "_theta_results_EAP.csv"), row.names = FALSE)
  
  # Print first few rows to verify
  cat("First few model names and theta values:\n")
  print(head(theta_df, 5))
  
  return(theta_df)
}

# 7. Main function to run the analysis
main <- function() {
  cat("Starting TinyARC theta analysis...\n")
  
  # Step 1: Load pre-selected items
  selected_items <- load_preselected_items()
  
  # Step 2: Calculate theta based on selected items
  results <- calculate_theta(selected_items)
  
  cat("Analysis complete. Results saved in the", DIRECTORY, "directory.\n")
  return(results)
}

# 8. Execute the main function
results <- main()
