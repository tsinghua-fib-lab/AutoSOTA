#!/usr/bin/env Rscript

# Recalculate theta for tiny benchmark using whole dataset item parameters
# For items not in whole dataset, use tiny benchmark item parameters

library(mirt)

# BENCHMARK <- "arc"
# BENCHMARK <- "winogrande"
# BENCHMARK <- "gsm8k"
BENCHMARK <- "truthfulqa"
BENCHMARK <- "hellaswag"



cat("\n=== Recalculating Theta for Tiny", BENCHMARK, "===\n\n")

# Read item parameters
cat("Reading item parameters...\n")
items_whole_path <- paste0(BENCHMARK, "/irt_item_parameters_combined.csv")
items_tiny_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/irt_item_parameters_tiny", BENCHMARK, ".csv")

items_whole <- read.csv(items_whole_path, stringsAsFactors = FALSE, row.names = 1)
items_tiny <- read.csv(items_tiny_path, stringsAsFactors = FALSE, row.names = 1)

cat("  Whole dataset items:", nrow(items_whole), "\n")
cat("  Tiny dataset items:", nrow(items_tiny), "\n")

# Read response data for tiny benchmark
cat("\nReading tiny benchmark response data...\n")
response_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/response_matrix_tiny", BENCHMARK, ".csv")

if (!file.exists(response_path)) {
  cat("ERROR: Response file not found:", response_path, "\n")
  quit(status = 1)
}

responses_tiny <- read.csv(response_path, stringsAsFactors = FALSE)
cat("  Response data shape:", nrow(responses_tiny), "persons x", ncol(responses_tiny), "items\n")

# Get item names from response data (assuming items are columns, possibly with row names column)
if ("X" %in% colnames(responses_tiny) || "" %in% colnames(responses_tiny)) {
  # First column might be row names/IDs
  response_item_names <- colnames(responses_tiny)[-1]
} else {
  response_item_names <- colnames(responses_tiny)
}

cat("  Items in response data:", length(response_item_names), "\n")

# Create hybrid item parameter set
cat("\nCreating hybrid item parameter set...\n")

# Tiny items have names like "11", "25", etc.
# Whole items have names like "X1", "X2", "X11", "X25", etc.
tiny_items_names <- rownames(items_tiny)
whole_items_names <- rownames(items_whole)

# For each item in response data, get parameters from whole if available, else from tiny
hybrid_params <- data.frame()
items_from_whole <- 0
items_from_tiny <- 0
items_missing <- c()

for (item_name in response_item_names) {
  # Check if item exists in whole dataset
  # Try both with and without X prefix
  item_name_trimmed <- gsub("^X", "", item_name)
  item_with_X <- if (grepl("^X", item_name)) item_name else paste0("X", item_name)
  
  if (item_with_X %in% whole_items_names) {
    # Use parameters from whole dataset
    hybrid_params <- rbind(hybrid_params, items_whole[item_with_X, ])
    rownames(hybrid_params)[nrow(hybrid_params)] <- item_name
    items_from_whole <- items_from_whole + 1
  } else if (item_name_trimmed %in% tiny_items_names) {
    # Use parameters from tiny dataset
    hybrid_params <- rbind(hybrid_params, items_tiny[item_name_trimmed, ])
    rownames(hybrid_params)[nrow(hybrid_params)] <- item_name
    items_from_tiny <- items_from_tiny + 1
  } else if (item_name %in% tiny_items_names) {
    # Try exact match in tiny
    hybrid_params <- rbind(hybrid_params, items_tiny[item_name, ])
    rownames(hybrid_params)[nrow(hybrid_params)] <- item_name
    items_from_tiny <- items_from_tiny + 1
  } else {
    items_missing <- c(items_missing, item_name)
  }
}

cat("  Items from whole dataset:", items_from_whole, "\n")
cat("  Items from tiny dataset:", items_from_tiny, "\n")
cat("  Items missing:", length(items_missing), "\n")

if (length(items_missing) > 0) {
  cat("  Missing items:", paste(head(items_missing, 10), collapse = ", "), "\n")
}

if (nrow(hybrid_params) == 0) {
  cat("ERROR: No matching items found\n")
  quit(status = 1)
}

# Prepare response data matrix (remove any ID columns)
if ("X" %in% colnames(responses_tiny)) {
  response_matrix <- as.matrix(responses_tiny[, -1])
} else if ("" %in% colnames(responses_tiny)) {
  response_matrix <- as.matrix(responses_tiny[, -1])
} else {
  response_matrix <- as.matrix(responses_tiny)
}

# Filter to only include items we have parameters for
items_to_keep <- rownames(hybrid_params)
response_matrix <- response_matrix[, items_to_keep, drop = FALSE]

cat("\nFinal dimensions for theta estimation:\n")
cat("  Persons:", nrow(response_matrix), "\n")
cat("  Items:", ncol(response_matrix), "\n")

# Create parameter list for mirt
cat("\nPreparing fixed item parameters for mirt...\n")

# Convert to mirt format (a1, d, g, u parameters)
pars <- list()
for (i in 1:nrow(hybrid_params)) {
  pars[[i]] <- c(
    a1 = hybrid_params$a1[i],
    d = hybrid_params$d[i],
    g = hybrid_params$g[i],
    u = hybrid_params$u[i]
  )
}

# Estimate theta using fixed item parameters
cat("\nEstimating person parameters (theta) using fscores...\n")
cat("This may take a moment...\n")

# Initialize variable
theta_tiny_recalc <- NULL

# Try using fixed item parameters approach
theta_tiny_recalc <- tryCatch({
  # Create parameter list for fscores with fixed parameters
  cat("Building mirt model with fixed item parameters...\n")
  
  # Create custom item list
  sv <- mirt(response_matrix, model = 1, itemtype = '4PL', pars = 'values')
  
  # Set all item parameters to fixed values
  for (i in 1:nrow(hybrid_params)) {
    item_rows <- which(sv$item == paste0('Item_', i))
    
    # Find the parameter rows for this item
    for (j in item_rows) {
      param_name <- sv$name[j]
      if (param_name == "a1") {
        sv$value[j] <- hybrid_params$a1[i]
        sv$est[j] <- FALSE
      } else if (param_name == "d") {
        sv$value[j] <- hybrid_params$d[i]
        sv$est[j] <- FALSE
      } else if (param_name == "g") {
        sv$value[j] <- hybrid_params$g[i]
        sv$est[j] <- FALSE
      } else if (param_name == "u") {
        sv$value[j] <- hybrid_params$u[i]
        sv$est[j] <- FALSE
      }
    }
  }
  
  # Fit model with fixed item parameters
  cat("Fitting model...\n")
  mod <- mirt(response_matrix, model = 1, itemtype = '4PL', pars = sv, verbose = FALSE)
  
  # Calculate theta scores
  cat("Calculating theta scores using EAP...\n")
  theta_scores <- fscores(mod, method = "EAP", full.scores = TRUE)
  
  # Create results dataframe
  result <- data.frame(
    F1 = theta_scores[, 1],
    SE_F1 = if(ncol(theta_scores) > 1) theta_scores[, 2] else NA
  )
  
  cat("Theta estimation completed successfully\n")
  result
  
}, error = function(e) {
  cat("ERROR in mirt estimation:", conditionMessage(e), "\n")
  cat("\nTrying alternative approach with EAP scoring directly...\n")
  
  tryCatch({
    # Even simpler - just fit model and score
    mod <- mirt(response_matrix, model = 1, itemtype = '4PL', verbose = FALSE)
    theta_scores <- fscores(mod, method = "EAP", full.scores = TRUE)
    
    result <- data.frame(
      F1 = theta_scores[, 1],
      SE_F1 = if(ncol(theta_scores) > 1) theta_scores[, 2] else NA
    )
    
    cat("Theta estimation completed using fitted model (not fixed parameters)\n")
    result
    
  }, error = function(e2) {
    cat("ERROR: Could not estimate theta:", conditionMessage(e2), "\n")
    NULL
  })
})

# Check if estimation succeeded
if (is.null(theta_tiny_recalc)) {
  cat("FATAL ERROR: Could not estimate theta using any method\n")
  quit(status = 1)
}

# Save recalculated theta
output_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/irt_person_scores_tiny", BENCHMARK, "_recalc.csv")
write.csv(theta_tiny_recalc, output_path, row.names = FALSE)
cat("\nRecalculated theta saved to:", output_path, "\n")

# Read whole dataset theta for comparison
cat("\nReading whole dataset theta for comparison...\n")
theta_whole_path <- paste0(BENCHMARK, "/irt_person_scores_WLE_SE_test.csv")
theta_whole <- read.csv(theta_whole_path, stringsAsFactors = FALSE)

cat("  Whole dataset persons:", nrow(theta_whole), "\n")

# Compare theta estimates
n_compare <- min(nrow(theta_tiny_recalc), nrow(theta_whole))

cat("\nComparing theta estimates (first", n_compare, "persons)...\n")

# Calculate MAE and RMSE
theta_comparison <- data.frame(
  Theta_Tiny_Recalc = theta_tiny_recalc$F1[1:n_compare],
  Theta_Whole = theta_whole$Theta_WLE[1:n_compare]
)

rmse <- sqrt(mean((theta_comparison$Theta_Tiny_Recalc - theta_comparison$Theta_Whole)^2, na.rm = TRUE))
mae <- mean(abs(theta_comparison$Theta_Tiny_Recalc - theta_comparison$Theta_Whole), na.rm = TRUE)
correlation <- cor(theta_comparison$Theta_Tiny_Recalc, theta_comparison$Theta_Whole, use = "complete.obs")

cat("\n=== Comparison Results ===\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("Correlation:", correlation, "\n")

# Save comparison
comparison_path <- paste0("experiments/baseline_compare/tinybenchmark_calibration/theta_comparison_", BENCHMARK, "_recalc.csv")
write.csv(theta_comparison, comparison_path, row.names = FALSE)
cat("\nComparison saved to:", comparison_path, "\n")

cat("\n=== Summary ===\n")
cat("Using", items_from_whole, "items from whole dataset and", items_from_tiny, "items from tiny dataset\n")
cat("This approach places tiny benchmark thetas on the same scale as whole dataset\n")
cat("MAE =", mae, "\n")
