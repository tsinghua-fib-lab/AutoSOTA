library(mirt)
################
# CLEAN DATA
###############
data<-read.csv("../data/gaussian_sampled_hellaswag_response_matrix_test.csv")


# Get dimensions of the original data
cat("Dimensions of origial data:", dim(data), "\n")

data_clean <- na.omit(data)              # remove NA rows
data_clean <- data_clean[, colSums(is.na(data_clean)) == 0]  # remove NA columns

# Drop constant columns (no variance)
constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

# Optionally drop constant rows
constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")

# Get dimensions of the cleaned data
cat("Dimensions of cleaned data:", dim(clean_data), "\n")
data_for_id <- clean_data


# Dimensions of origial data: 2998 845 
# Dropped 3 constant columns.
# Dropped 0 constant rows.
# Dimensions of cleaned data: 2998 842 
######################
# IMPLEMENT MEAN/SIGMA LINKING
######################

# Define chunk sizes based on irt_winogrande.r
chunk_indices <- c(seq(113, 5501, 112), 5608)#⚠️
# Load person scores for each subset
scores_list <- lapply(chunk_indices, function(i) {
  # Read scores and ensure they're numeric
  scores_df <- read.csv(paste0("irt_person_scores_", i, ".csv"))
  # The file contains a single column named 'F1'
  return(scores_df)
})

# Define reference subset (using the first subset as reference)
reference_scores <- scores_list[[1]]

# Print info about scores to debug
cat("Reference scores column names: ", colnames(reference_scores), "\n")
cat("First few reference scores: ", head(reference_scores$F1), "\n")

# Perform Mean/Sigma linking for each subset to the reference scale
linked_params_list <- list()

# First add the reference parameters unchanged
linked_params_list[[1]] <- read.csv(paste0("irt_item_parameters_", 113, ".csv")) #⚠️

# Process each subsequent subset (starting from the second subset)
for (j in 2:length(scores_list)) {
  # Get subset index from chunk_indices
  i <- chunk_indices[j]
  cat("Processing subset ", i, "...\n")
  
  # Load parameters for this subset
  params_j <- read.csv(paste0("irt_item_parameters_", i, ".csv"))
  cat("Loaded parameters for subset ", i, "\n")
  
  # Get scores for common persons - using the 'F1' column directly
  sc_ref <- as.numeric(reference_scores$F1) # Reference scores
  sc_j <- as.numeric(scores_list[[j]]$F1)   # Current subset scores
  
  # Apply Mean/Sigma linking (form j -> reference form)
  A <- sd(sc_ref, na.rm=TRUE)/sd(sc_j, na.rm=TRUE)
  B <- mean(sc_ref, na.rm=TRUE) - A*mean(sc_j, na.rm=TRUE)
  
  # Print linking constants
  cat("Linking constants for subset ", i, ": A=", A, ", B=", B, "\n")
  
  # Transform item parameters
  params_j_star <- params_j
  params_j_star$a1 <- params_j$a1/A
  params_j_star$d <- A*params_j$d + B*params_j$a1
  
  # If using 3PL, keep guessing parameter unchanged
  if ('g' %in% colnames(params_j)) {
    params_j_star$g <- params_j$g
  }
  
  # Add to linked parameters list
  linked_params_list[[j]] <- params_j_star
  
  # Diagnostic: print correlation between transformed scores and reference
  transformed_scores <- A*sc_j + B
  cor_value <- cor(sc_ref, transformed_scores, use='pairwise.complete.obs')
  cat("Correlation after linking: ", cor_value, "\n")
}

# Combine all linked parameters
item_params_combined <- do.call(rbind, linked_params_list)

# Load original parameters for comparison (if needed)
params_list <- lapply(chunk_indices, function(i) read.csv(paste0("irt_item_parameters_", i, ".csv")))
# Using chunk indices: 105, 209, 313, 417, 521, 626, 731, 836, 941, 1046

# Save combined parameters
write.csv(item_params_combined, "irt_item_parameters_combined.csv", row.names = FALSE)

print("Parameter dataframe head:")
print(head(item_params_combined))

# Print column names from data to understand format
print("First few data column names:")
print(head(colnames(clean_data)))

# Look at the first few column names from both datasets to diagnose the issue
print("Data column names:")
print(head(colnames(clean_data)))
print("Parameter rows (X column):")
print(head(item_params_combined$X))

# Extract parameter item names with proper transformation
param_items <- as.character(item_params_combined$X)

#########################################################
# Create mapping between parameters and data columns
#########################################################
# Approach 1: Try matching with 'X' prefix
param_to_data_map <- list()
for (param_item in param_items) {
  # Try direct mapping with X prefix
  if (param_item %in% colnames(clean_data)) {
    param_to_data_map[[param_item]] <- param_item
  } 
  # Try removing X prefix
  else if (gsub("^X", "", param_item) %in% colnames(clean_data)) {
    param_to_data_map[[param_item]] <- gsub("^X", "", param_item)
  }
  # Try adding X prefix
  else if (paste0("X", param_item) %in% colnames(clean_data)) {
    param_to_data_map[[param_item]] <- paste0("X", param_item)
  }
}

# Print mapping statistics
print(paste("Number of mapped parameters:", length(param_to_data_map)))

# Create a clean parameter dataframe with the correct row names
clean_params <- data.frame(
  a1 = item_params_combined$a1,
  d = item_params_combined$d,
  g = item_params_combined$g,
  row.names = param_items
)

# Use a different strategy if mapping fails
if (length(param_to_data_map) == 0) {
  # Try a numeric approach - the item numbers might be the same but format differs
  # Extract numeric portion from parameter names
  param_nums <- as.numeric(gsub("[^0-9]", "", param_items))
  
  # Extract numeric portion from data column names
  data_cols <- colnames(clean_data)
  data_nums <- as.numeric(gsub("[^0-9]", "", data_cols))
  
  # Find common numbers
  common_nums <- intersect(param_nums, data_nums)
  print(paste("Common numeric IDs:", length(common_nums)))
  
  # Create mapping based on numeric IDs
  param_to_data_map <- list()
  for (num in common_nums) {
    param_idx <- which(param_nums == num)[1]
    data_idx <- which(data_nums == num)[1]
    if (!is.na(param_idx) && !is.na(data_idx)) {
      param_item <- param_items[param_idx]
      data_col <- data_cols[data_idx]
      param_to_data_map[[param_item]] <- data_col
    }
  }
  
  print(paste("Number of mapped parameters after numeric matching:", length(param_to_data_map)))
}

#########################################################
# Print parameter summary
cat("Number of items with parameters:", nrow(clean_params), "\n")

# Get list of data columns that have parameters
mapped_data_cols <- unlist(param_to_data_map)
cat("Number of data columns with parameters:", length(mapped_data_cols), "\n")

# Filter data to keep only columns that have parameters
clean_data <- clean_data[, mapped_data_cols, drop=FALSE]

# Fit dummy model just to get structure
mod_dummy <- mirt(clean_data, 1, itemtype = "2PL", pars = 'values')

# Replace item values
pars <- mod_dummy

# At this point, data columns should be renamed to match parameter names
# Print a few column names to verify
print("Final data column names:")
print(head(colnames(clean_data)))

# Set parameter values for each item
print("Setting parameter values...")
parameter_count <- 0

for (i in 1:nrow(pars)) {
  item_name <- pars$item[i]
  param_name <- pars$name[i]
  
  # Only try to set parameters that exist in our clean_params
  if (param_name %in% c("a1", "d", "g") && item_name %in% rownames(clean_params)) {
    if (param_name == "a1") {
      pars$value[i] <- clean_params[item_name, "a1"]
    } else if (param_name == "d") {
      pars$value[i] <- clean_params[item_name, "d"]
    } else if (param_name == "g") {
      pars$value[i] <- clean_params[item_name, "g"]
    }
    parameter_count <- parameter_count + 1
  }
}

print(paste("Total parameters set:", parameter_count))

# Print parameter summary to verify correct mapping
cat("Number of parameters set:", sum(pars$name %in% c("a1", "d", "g")), "\n")

# Fix all item parameters
pars$est[pars$name %in% c("a1", "d", "g")] <- FALSE

# Only proceed if we have data and parameters to work with
if (ncol(clean_data) > 0 && parameter_count > 0) {
  # Build fixed model with information matrix
  print("Building the model...")
  mod_fixed <- mirt(clean_data, 1, itemtype = "2PL", pars = pars, method = "EM", 
                   technical = list(NCYCLES = 5000, BURNIN = 1000),
                   verbose = TRUE)
  
  # Generate person scores (thetas)
  print("Generating person scores...")
  # Use robust optimization settings with WLE (weighted likelihood) estimation
  # WLE doesn't require information matrices for imputation
  thetas <- fscores(mod_fixed, method = "WLE", 
                    full.scores.SE = TRUE,
                    response.pattern = NULL, # Use all response patterns
                    quadpts = 61)           # More quadrature points for better precision
  
  # Check for convergence issues
  na_count <- sum(is.na(thetas))
  nan_count <- sum(is.nan(thetas))
  print(paste("Number of NA values:", na_count))
  print(paste("Number of NaN values:", nan_count))
  
  # Check for explicit warnings from mirt about convergence
  warning_messages <- warnings()
  problem_rows <- c()
  
  for (warning_msg in warning_messages) {
    if (grepl("factor score estimates failed to converge", warning_msg)) {
      # Extract row numbers from the warning message
      row_nums_str <- gsub(".*factor score estimates failed to converge successfully:[[:space:]]*|[[:space:]]*$", "", warning_msg)
      # Split by comma and convert to numeric
      failed_rows <- as.numeric(unlist(strsplit(row_nums_str, "[^0-9]+")))
      failed_rows <- failed_rows[!is.na(failed_rows)]
      
      if (length(failed_rows) > 0) {
        problem_rows <- c(problem_rows, failed_rows)
        print(paste("Found", length(failed_rows), "rows with convergence issues from warnings:"))
        print(paste("  ", paste(failed_rows, collapse=", ")))
      }
    }
  }
  
  # Handle both NA/NaN values and warned problem rows
  explicit_problems <- which(is.na(thetas) | is.nan(thetas))
  problem_rows <- unique(c(problem_rows, explicit_problems))
  
  if (length(problem_rows) > 0) {
    print(paste("Total problem rows:", length(problem_rows)))
    print(paste("Problem row indices:", paste(sort(problem_rows), collapse=", ")))
    
    # Try to compute individual estimates for problematic cases with different settings
    print("Attempting to re-estimate problematic rows with different optimization settings...")
    
    fixed_count <- 0
    for (row_idx in problem_rows) {
      # Try a different method (ML) with adjusted optimization parameters for each problem row
      try({
        row_data <- clean_data[row_idx, , drop=FALSE]
        # Use WLE method to avoid information matrix requirements
        row_theta <- fscores(mod_fixed, method="WLE", response.pattern=row_data,
                            quadpts=81)
        
        # If successful, replace the original estimate
        if (!is.na(row_theta) && !is.nan(row_theta)) {
          thetas[row_idx] <- row_theta
          fixed_count <- fixed_count + 1
        }
      }, silent=TRUE)
    }
    
    print(paste("Fixed", fixed_count, "of", length(problem_rows), "problematic theta estimates"))
    
    # For any remaining problematic rows, use imputation
    remaining_problems <- which(is.na(thetas) | is.nan(thetas) | row.names(thetas) %in% problem_rows)
    
    if (length(remaining_problems) > 0) {
      # Replace with median of valid thetas
      valid_thetas <- thetas[!is.na(thetas) & !is.nan(thetas) & !(row.names(thetas) %in% problem_rows)]
      if (length(valid_thetas) > 0) {
        replacement_value <- median(valid_thetas)
        print(paste("Replacing remaining non-converged values with median theta:", replacement_value))
        thetas[remaining_problems] <- replacement_value
      } else {
        print("No valid thetas to use for imputation.")
      }
    }
  } else {
    print("All theta estimates converged successfully.")
  }
  
  # Print summary statistics
  print("Theta estimate summary:")
  print(summary(thetas))
  
  # Read response matrix to get model names

  # Get model names from the first column
  model_names <- data_for_id[[1]]
  head(model_names)

  # Create data frame with model names and theta values
  theta_df <- data.frame(Model_Name = model_names, Theta_WLE = thetas[, 1], SE = thetas[, 2])
  
  # Save results
  # output_file <- "irt_person_scores_WLE_SE.csv"

  output_file <- "irt_person_scores_WLE_SE_test.csv"
  write.csv(theta_df, output_file, row.names = FALSE)
  print(paste("Saved person scores to", output_file))
  
  # Print first few rows to verify
  print("First few model names and theta values:")
  print(head(theta_df, 10))
} else {
  print("ERROR: Insufficient data or parameters to build the model")
  print(paste("Data columns:", ncol(data_for_id)))
  print(paste("Parameters set:", parameter_count))
}
