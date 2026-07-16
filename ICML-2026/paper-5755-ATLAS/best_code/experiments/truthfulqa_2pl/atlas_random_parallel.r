# usage: Rscript atlas_random_parallel.r --se_theta_stop=0.1 --test_length=100 --n_cores=8
# (if test_length is not specified, it will run on all models)
# (if n_cores is not specified, it will use all available cores - 1)
# 
# PARALLEL OPTIMIZED VERSION - Uses multiple CPU cores and eliminates repeated file reads
# Install and Load Required Packages

# Set CRAN mirror for package installation (required for batch/cluster environments)
options(repos = c(CRAN = "https://cloud.r-project.org/"))

if (!require(catR)) install.packages("catR")
library(catR)
if (!require(parallel)) install.packages("parallel")
library(parallel)
if (!require(doParallel)) install.packages("doParallel")
library(doParallel)
if (!require(foreach)) install.packages("foreach")
library(foreach)

# Parse CLI arguments for se_theta_stop, test_length, and n_cores
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (a in args) {
    if (grepl("^--se_theta_stop=", a)) {
      out$se_theta_stop <- as.numeric(sub("^--se_theta_stop=", "", a))
    } else if (grepl("^--test_length=", a)) {
      out$test_length <- as.integer(sub("^--test_length=", "", a))
    } else if (grepl("^--n_cores=", a)) {
      out$n_cores <- as.integer(sub("^--n_cores=", "", a))
    }
  }
  return(out)
}

.cli_args <- parse_args()

if (!is.null(.cli_args$se_theta_stop) && !is.na(.cli_args$se_theta_stop)) {
  se_theta_stop <- .cli_args$se_theta_stop
} else {
  se_theta_stop <- 0.1
}

# Determine number of cores to use
if (!is.null(.cli_args$n_cores) && !is.na(.cli_args$n_cores)) {
  n_cores <- .cli_args$n_cores
} else {
  # Use all cores minus 1 by default
  n_cores <- max(1, detectCores() - 1)
}
cat("Using", n_cores, "CPU cores for parallel processing\n")

# Prepare item bank
# Read and prepare item bank
DIRECTORY <- "atlas_arc_random/"
FILE_PATH <- "irt_item_parameters_combined.csv"
prepare_item_bank <- function(file_path) {
  # Read your CSV
  items <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Ensure required columns exist
  required_cols <- c("X","a1","d","g","u")
  if (!all(required_cols %in% colnames(items))) {
    missing_cols <- setdiff(required_cols, colnames(items))
    stop(paste("Missing required columns in CSV:", paste(missing_cols, collapse = ", ")))
  }
  
  # Create empty matrix for parameters (a,b,c,d format)
  n_items <- nrow(items)
  item_bank <- matrix(nrow = n_items, ncol = 4)
  
  # Convert from (a1,d,g,u) to (a,b,c,d) format
  # a = absolute value of a1 (discrimination parameter)
  item_bank[,1] <- items$a1
  
  # b = -d/a1 (difficulty parameter)
  item_bank[,2] <- -items$d / items$a1
  
  # c = g (guessing parameter)
  item_bank[,3] <- items$g
  
  # d = u (upper asymptote)
  item_bank[,4] <- items$u
  
  # Set column names to standard IRT notation
  colnames(item_bank) <- c("a", "b", "c", "d")
  
  # Set row names to item IDs
  rownames(item_bank) <- items$X
  
  # Ensure all parameters are numeric
  if (!all(is.numeric(item_bank))) {
    stop("Item parameters must be numeric")
  }
  
  # Print summary statistics of the converted parameters
  cat("Parameter ranges after conversion:\n")
  cat("a (discrimination): ", min(item_bank[,1]), "to", max(item_bank[,1]), "\n")
  cat("b (difficulty): ", min(item_bank[,2]), "to", max(item_bank[,2]), "\n")
  cat("c (guessing): ", min(item_bank[,3]), "to", max(item_bank[,3]), "\n")
  cat("d (upper asymptote): ", min(item_bank[,4]), "to", max(item_bank[,4]), "\n")
  
  # Store the item id separately
  item_info <- items[, c("X")]
  
  return(list(item_bank = item_bank, item_info = item_info))
}




# Function to get score from different models
# OPTIMIZED: Now accepts pre-loaded response_data instead of reading from disk every time
score_response <- function(model_name, item_id, response_data, response_model_names) {
  # Find the row index for the model
  row_idx <- which(response_model_names == model_name)
  
  # FIX: Strip "X" prefix from item_id if present (item bank has "X1", response matrix has "1")
  item_number <- item_id
  if (grepl("^X", item_number)) {
    item_number <- sub("^X", "", item_number)
  }
  
  # Check if we found the model and if the item_id exists in columns
  if (length(row_idx) > 0 && item_number %in% colnames(response_data)) {
    return(response_data[row_idx, item_number])
  } else {
    warning(paste("Could not find data for model", model_name, "and item", item_id, "/", item_number))
    return(NA)
  }
}


# Function to run ATLAS 
# OPTIMIZED: Now accepts pre-loaded response data to avoid repeated file reads
run_atlas <- function(item_bank, item_info, model_name,
                               response_data, response_model_names,
                               start_theta = 0, 
                               min_items = 3, 
                               max_items = 8,
                               se_theta_stop = 0.1,
                               verbose = FALSE) {
  
  # Initialize test
  test <- list(
    administered = integer(0),  # Indices of administered items
    responses = numeric(0),     # Response vector (0/1)
    thetas = start_theta,       # Ability estimates
    SEs = NA,                   # Standard errors
    items_used = character(0)  # Item IDs
  )
  
  # ATLAS loop
  for (i in 1:max_items) {
    if (verbose) cat(sprintf("\n--- Item %d ---\n", i))
    
    tryCatch({
    if (i == 1) {
  # First item: select randomly from items with difficulty near start_theta
  # Get items with difficulty within a certain range of start_theta
  difficulty_range <- 0.5  # Adjust this value to control diversity
  candidate_items <- which(abs(item_bank[,"b"] - start_theta) < difficulty_range)
  
  # If no items within range, fall back to closest item
  if(length(candidate_items) == 0) {
    next_item_pos <- which.min(abs(item_bank[,"b"] - start_theta))
  } else {
    # Randomly select from candidates
    next_item_pos <- sample(candidate_items, 1)
  }
      } else {
        # Subsequent items: selection with randomesque
        next_item <- nextItem(item_bank, 
                              theta = tail(test$thetas, 1),
                              out = test$administered,
                              criterion = "MFI", 
                              method = "EAP", 
                              randomesque = 5)  # Randomly select from 5 most informative items
        next_item_pos <- next_item$item
      }
      
      # Get item details
      next_item_id <- rownames(item_bank)[next_item_pos]

      
      if (verbose) cat("Selected item:", next_item_id, "\n")
      
      # Score the response (using pre-loaded data)
      score <- score_response(model_name, next_item_id, response_data, response_model_names)

      if (verbose) cat("Score:", score, "Item id:", next_item_id, "\n")
      
      # Update test information
      test$administered <- c(test$administered, next_item_pos)
      test$responses <- c(test$responses, score)
      test$items_used <- c(test$items_used, next_item_id)

      
      # Estimate new theta
      est <- thetaEst(item_bank[test$administered, , drop = FALSE], 
                      test$responses, 
                      method = "EAP")
      
      # Calculate standard error
      se <- semTheta(est, item_bank[test$administered, , drop = FALSE], test$responses)
      
      test$thetas <- c(test$thetas, est)
      test$SEs <- c(test$SEs, se)
      
      if (verbose) {
        cat(sprintf("Current theta: %.2f, SE: %.3f, Running score: %d/%d\n", 
                  est, se, sum(test$responses), length(test$responses)))
      }
      
      # Check stopping rules
      if (i >= min_items && !is.na(se) && se <= se_theta_stop) {
        if (verbose) cat(sprintf("\nStopping: SE (%.3f) <= %.3f\n", se, se_theta_stop))
        break
      }
      
    }, error = function(e) {
      if (verbose) cat("Error in ATLAS loop:", e$message, "\n")
      # If we have at least one response, continue with the next item
      if (length(test$responses) > 0) {
        if (verbose) cat("Continuing with next item...\n")
      } else {
        stop("Fatal error in ATLAS loop")
      }
    })
  }
  
  # Prepare final results
  final_theta <- tail(test$thetas, 1)
  final_se <- tail(test$SEs, 1)
  
  results <- data.frame(
    item_id = test$items_used,
    score = test$responses,
    theta = test$thetas[-1],
    se = test$SEs[-1],
    stringsAsFactors = FALSE
  )
  
  # Save selected items to a CSV file
  items_df <- data.frame(
    item_id = test$items_used,
    order = 1:length(test$items_used),
    score = test$responses,
    theta = test$thetas[-1],
    se = test$SEs[-1],
    stringsAsFactors = FALSE
  )
  
  # Create directory if it doesn't exist
  dir.create(paste0(DIRECTORY, "selected_items_", se_theta_stop), recursive = TRUE, showWarnings = FALSE)
  
  # Generate filename with model name (cleaned to be safe for filenames)
  safe_model_name <- gsub("/|\\\\|:|\\*|\\?|\"|<|>|\\||\\s", "_", model_name)
  items_file <- paste0(DIRECTORY, "selected_items_", se_theta_stop, "/", safe_model_name, "_items.csv")
  
  # Save to CSV
  write.csv(items_df, items_file, row.names = FALSE)
  if (verbose) cat("Selected items saved to", items_file, "\n")
  
  return(list(
    results = results,
    final_theta = final_theta,
    final_se = final_se,
    num_items = length(test$responses),
    thetas = test$thetas,
    SEs = test$SEs,
    model_name = model_name
  ))
}


#################################################
# MAIN
#################################################

# Prepare item bank
item_data <- prepare_item_bank(FILE_PATH)

# Check item bank structure
str(item_data$item_bank)
head(item_data$item_bank)
dim(item_data$item_bank)

###########################################
# CLEAN RESPONSE MATRIX
# clean_data <- read.csv("clean_response_matrix.csv")#⚠️
data<-read.csv("../data/gaussian_sampled_truthfulqa_response_matrix_test.csv")


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
###########################################
# LOAD RESPONSE MATRIX ONCE (MAJOR OPTIMIZATION)
cat("Loading response matrix (this is done ONCE)...\n")
response_data <- read.csv("../data/gaussian_sampled_truthfulqa_response_matrix_test.csv",#⚠️ 
                          stringsAsFactors = FALSE, check.names = FALSE)
response_model_names <- response_data[[1]]
cat("Response matrix loaded with", nrow(response_data), "models\n")

###########################################
# LOAD THETA FULL DATABASE
theta_full_db <- read.csv("irt_person_scores_WLE_SE_test.csv")

###########################################
# DETERMINE TEST LENGTH
# Determine number of models to run (can be overridden via --test_length)
test_length <- nrow(clean_data)
if (!is.null(.cli_args$test_length) && !is.na(.cli_args$test_length)) {
  # Clamp to valid range
  test_length <- max(1, min(.cli_args$test_length, nrow(clean_data)))
}
cat("Running ATLAS on", test_length, "models\n")

###########################################
# PARALLEL PROCESSING SETUP
cat("Setting up parallel processing with", n_cores, "cores...\n")
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Export necessary objects to workers
cat("Exporting data to worker nodes...\n")

###########################################
# PARALLEL EXECUTION
cat("\nStarting parallel ATLAS execution...\n")
overall_start_time <- Sys.time()

# Run ATLAS in parallel using foreach
results_list <- foreach(i = 1:test_length, 
                        .packages = c("catR"),
                        .export = c("run_atlas", "score_response", "DIRECTORY", "se_theta_stop"),
                        .verbose = FALSE) %dopar% {
  
  # Get model information
  test_model <- clean_data[[1]][i]
  theta_full <- theta_full_db[[2]][i]
  
  # Run ATLAS for this model
  start_time <- Sys.time()
  set.seed(123 + i)  # Different seed for each model for reproducibility
  
  atlas_results <- run_atlas(
    item_bank = item_data$item_bank,
    item_info = item_data$item_info,
    model_name = test_model,
    response_data = response_data,
    response_model_names = response_model_names,
    start_theta = 0,
    min_items = 30,
    max_items = 500,
    se_theta_stop = se_theta_stop,
    verbose = FALSE  # Suppress verbose output in parallel mode
  )
  
  end_time <- Sys.time()
  duration_sec <- as.numeric(end_time - start_time, units = "secs")
  
  # Return results
  list(
    model_name = test_model,
    theta_full = theta_full,
    theta_atlas = atlas_results$final_theta,
    se = atlas_results$final_se,
    num_items = atlas_results$num_items,
    time_taken = duration_sec,
    model_index = i
  )
}

# Stop the cluster
stopCluster(cl)

overall_end_time <- Sys.time()
overall_duration <- as.numeric(overall_end_time - overall_start_time, units = "secs")

cat("\n==============================================\n")
cat("PARALLEL EXECUTION COMPLETED\n")
cat("Total time:", round(overall_duration, 2), "seconds\n")
cat("Average time per model:", round(overall_duration / test_length, 2), "seconds\n")
cat("==============================================\n")

###########################################
# COLLECT AND SAVE RESULTS
cat("\nCollecting results...\n")

theta_full_list <- sapply(results_list, function(x) x$theta_full)
theta_atlas_list <- sapply(results_list, function(x) x$theta_atlas)
se_list <- sapply(results_list, function(x) x$se)
num_items_administered_list <- sapply(results_list, function(x) x$num_items)
time_list <- sapply(results_list, function(x) x$time_taken)

# Get model names from the first column
model_names <- theta_full_db[[1]][1:test_length]

# Create data frame with model names and theta values
theta_df <- data.frame(
  Model_Name = model_names, 
  Theta_ATLAS = theta_atlas_list, 
  Theta_WLE = theta_full_list, 
  SE = se_list, 
  Num_Items = num_items_administered_list, 
  Time_Taken_Sec = time_list
)

# Save results
output_file <- paste0(DIRECTORY, "irt_person_scores_ATLAS_", se_theta_stop, ".csv")
write.csv(theta_df, output_file, row.names = FALSE)
cat("Saved person scores to", output_file, "\n")

# Print summary statistics
cat("\n==============================================\n")
cat("SUMMARY STATISTICS\n")
cat("==============================================\n")
cat("Mean theta (ATLAS):", round(mean(theta_atlas_list), 3), "\n")
cat("Mean theta (WLE):", round(mean(theta_full_list), 3), "\n")
cat("Mean SE:", round(mean(se_list), 3), "\n")
cat("Mean number of items:", round(mean(num_items_administered_list), 1), "\n")
cat("Mean time per model:", round(mean(time_list), 2), "seconds\n")
cat("Total processing time:", round(overall_duration, 2), "seconds\n")
cat("Speedup vs sequential (estimated):", round(sum(time_list) / overall_duration, 1), "x\n")
cat("==============================================\n")

###########################################
# Analyze item selection frequency
###########################################

analyze_item_frequency <- function() {
  # Get all item files
  item_files <- list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$", full.names=TRUE)
  
  if(length(item_files) < 1) {
    cat("No item sets found\n")
    return(NULL)
  }
  
  # Read all item sets
  item_sets <- lapply(item_files, function(f) {
    df <- read.csv(f)
    return(df$item_id)
  })
  
  # Get model names from filenames
  model_names <- basename(item_files)
  model_names <- sub("_items\\.csv$", "", model_names)
  
  # Combine all items
  all_items <- unlist(item_sets)
  
  # Count frequency of each item
  item_freq <- table(all_items)
  
  # Convert to data frame
  item_freq_df <- data.frame(
    item_id = names(item_freq),
    frequency = as.numeric(item_freq),
    percentage = 100 * as.numeric(item_freq) / length(model_names)
  )
  
  # Sort by frequency
  item_freq_df <- item_freq_df[order(-item_freq_df$frequency),]
  
  # Save to CSV
  write.csv(item_freq_df, paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv"), row.names = FALSE)
  
  # Print top items
  cat("\nTop 10 most frequently selected items:\n")
  for(i in 1:min(10, nrow(item_freq_df))) {
    cat(sprintf("%s: %d models (%.1f%%)\n", 
                item_freq_df$item_id[i], 
                item_freq_df$frequency[i],
                item_freq_df$percentage[i]))
  }
  
  cat("\nItem selection frequency saved to ", paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv\n"))
  
  
  return(item_freq_df)
}

# Analyze item frequency if selected items exist
cat("\n---------- Item Selection Frequency Analysis ----------\n")
if(dir.exists(paste0(DIRECTORY, "selected_items_", se_theta_stop)) && 
   length(list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$")) > 0) {
  item_freq_results <- analyze_item_frequency()
} else {
  cat("No selected item files found in, ",paste0(DIRECTORY, "selected_items_", se_theta_stop), ". Run ATLAS first to generate item selection data.\n")
  item_freq_results <- NULL
}

###########################################
# Analyze item selection positions
###########################################
# DISABLED: Item position analysis removed for faster execution

cat("\n==============================================\n")
cat("ATLAS PARALLEL PROCESSING COMPLETE!\n")
cat("==============================================\n")
