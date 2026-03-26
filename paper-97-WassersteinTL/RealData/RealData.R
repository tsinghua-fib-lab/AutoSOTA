################################################################################
# Wasserstein Transfer Learning (WaTL) - Real Data Application
# 
# Analyze the distribution of physical activity intensity.
#
# Data:
#   - Target: Black participants (200 samples)
#   - Sources: White, Mexican Americans, Other Hispanic
#   - Predictors: BMI and Age
#   - Response: Distribution of physical activity intensity
#
# Arguments:
#   seed: Random seed
#   r: Race index (1=Black, 2=White)
#   M: Grid size for quantile functions
#   rate: Source data sampling rate (1.0 = use all source data)
#   Gender: Gender (0=Female, 1=Male)
#
# Output:
#   RMSPR (Root Mean Squared Prediction Risk) using 5-fold cross-validation
################################################################################

################################################################################
# Parse command line arguments
################################################################################
args <- commandArgs(TRUE)
print(args)

seed <- as.numeric(args[[1]])      # Random seed for reproducibility
r <- as.numeric(args[[2]])         # Race index (1=Black target, 2=White target)
M <- as.numeric(args[[3]])         # Grid size for quantile functions
rate <- as.numeric(args[[4]])      # Source sampling rate (typically 1.0 = use all)
Gender <- as.numeric(args[[5]])    # Gender (0=Female, 1=Male)

################################################################################
# Load required libraries
################################################################################
library(rnhanesdata)   # NHANES 2005-2006 data
library(tidyverse)     # Data manipulation
library(truncnorm)     # Truncated normal distributions
library(pracma)        # Practical numerical analysis
library(osqp)          # Quadratic programming for projection
library(Matrix)        # Sparse matrices
library(parallel)      # Parallel computing
library(doParallel)    # Parallel backend
library(foreach)       # Parallel loops
library(doSNOW)        # SNOW parallel backend

# Load helper functions (WaTL algorithm, data processing, etc.)
source("RealDataFunc.R") 

set.seed(seed)  # Set random seed for reproducibility

################################################################################
# Data Loading and Preprocessing
################################################################################
# Target races: Black and White (can be extended to include other races)
Races = c("Black","White")
race = Races[r]  # Select target race

# Load and preprocess NHANES physical activity data
# This computes quantile functions for each participant's activity distribution
data <- get_realdata(M)

# Filter data: remove missing values for BMI, Gender, and Age
realdata = data[!is.na(data$BMI) & !is.na(data$Gender) & !is.na(data$Age), ]

# Filter by race and gender
realdata = realdata[realdata$Race %in% Races,]
realdata = realdata[realdata$Gender == Gender,]

# Remove gender column (already filtered)
realdata$Gender = NULL

# Process data into target and source datasets
# Target: 'race' (e.g., Black), Sources: all other races (e.g., White, Mexican, etc.)
realdata = realdata_process(realdata, M, race)

################################################################################
# Setup for Analysis
################################################################################
K = length(realdata$source_data)  # Number of source datasets (typically 3)
num_fold = 5  # 5-fold cross-validation

# Grids for quantile function evaluation
z_grid <- seq(1 / M, 1 - 1 / M, length.out = M)  # Evaluation points
p_grid <- seq(1 / M, 1 - 1 / M, length.out = M)  # Probability levels

# Extract target and source data
X_t <- realdata$X_t              # Target predictors: BMI and Age (n_t × 2)
Y_t <- realdata$Y_t              # Target quantile functions (n_t × M)
source_d <- realdata$source_data # List of K source datasets
n_t <- dim(X_t)[1]               # Total available target samples

################################################################################
# Subsample target to 200 participants
################################################################################
Index_random = sample(n_t, 200)
X_t = X_t[Index_random,]
Y_t = Y_t[Index_random,]

n_t <- dim(X_t)[1]  # Update sample size (should be 200)

cat("Target Sample Size: ", n_t, "\n")

################################################################################
# Setup 5-fold Cross-Validation
################################################################################
# Create fold indices for outer cross-validation
outer_folds <- create_folds(n_t, num_fold)

# Storage for results from all folds
all_results <- list()
fold_sizes <- numeric(num_fold)

################################################################################
# Main 5-Fold Cross-Validation Loop
################################################################################
for (outer_k in 1:num_fold) 
{
  cat("===== Outer Fold:", outer_k, "=====\n")
  
  #============================================================================
  # Split data into training and test sets for this fold
  #============================================================================
  outer_test_idx <- which(outer_folds == outer_k)
  X_t_train <- X_t[-outer_test_idx,]   # Training predictors (160 × 2)
  Y_t_train <- Y_t[-outer_test_idx,]   # Training quantile functions (160 × M)
  X_t_test <- X_t[outer_test_idx,]     # Test predictors (40 × 2)
  Y_t_test <- Y_t[outer_test_idx,]     # Test quantile functions (40 × M)
  
  n_t <- dim(X_t_train)[1]

  #============================================================================
  # Setup for LOCAL Fréchet regression (different from simulation!)
  #============================================================================
  # Bandwidth selection: use 15% of range for each predictor (BMI and Age)
  # Gaussian kernel for local linear weights
  optns = list(kernel = "gauss", 
               bw = c(diff(range(X_t_train$BMI))*0.15,
                      diff(range(X_t_train$Age))*0.15))
  
  # Prepare training data in format for local regression
  qin.target <- split(Y_t_train, row(Y_t_train))
  qin.target <- lapply(qin.target, as.numeric)
  xin.t <- to_matrix(X_t_train)
  
  cat("Dimension of X Train:",dim(X_t_train),"\n")
  cat("Dimension of Y Train:",dim(Y_t_train),"\n")
  
  # Storage for results on test set
  n_test <- nrow(X_t_test)
  fold_sizes[outer_k] <- n_test
  fold_results <- matrix(NA, nrow = n_test, ncol = 3)
  colnames(fold_results) <- c("d_1", "d_2_target", "d_2_source")
  
  #============================================================================
  # Evaluate on each test point in this fold
  #============================================================================
  for (i in 1:n_test) 
  {
    # Current test point: BMI and Age values
    X_value <- as.vector(as.numeric(X_t_test[i,]))
    
    # True quantile function for this test participant
    true_Et <- Y_t_test[i,]
    
    #==========================================================================
    # Compute local linear weights for this query point
    #==========================================================================
    # Weight function: s_L(x,h) = K_h(X-x){u_2 - u_1(X-x)}/σ_0^2
    # These weights are different from global Fréchet weights used in simulation!
    s_vec_t_train <- LocalLinWeights(qin.target, xin.t, X_value, optns)
    
    cat("Finish Calculate Weights \n")
    
    #==========================================================================
    # ALGORITHM 1, STEP 1: Weighted auxiliary estimator
    #==========================================================================
    # Combines target (k=0) and all sources (k=1,...,K)
    # Uses LOCAL Fréchet regression (lrem) instead of global (grem)
    f1_hat <- compute_f1_hat(source_data_list = source_d,
                             X_value = X_value,
                             M = M,
                             rate = rate,
                             X_t = X_t_train,
                             Y_t = Y_t_train)
    
    cat("Finish Calculate f1_hat \n")
    
    # Compute initial distance (before bias correction)
    d <- compute_L2_distance(true_Et, f1_hat)
    
    #==========================================================================
    # Select regularization parameter λ via inner cross-validation
    #==========================================================================
    # Lambda grid: covers range from 0.0001 to 0.5
    lambda_candidates <- sort(unique(c(
      seq(0.0001, 0.001, length.out = 5),
      seq(0.001, 0.01, length.out = 5),
      seq(0.01, 0.1, length.out = 5),
      seq(0.1, 0.5, length.out = 5)
    )))
    
    # 3-fold CV on training data to select best λ
    cv_out <- cv_search_lambda(
      Y_t = Y_t_train,
      s_vec = s_vec_t_train,
      X_t = X_t_train,
      source_d = source_d,
      M = M,
      rate = rate,
      lambda_grid = lambda_candidates,
      n_folds = 3,              # Inner CV folds
      max_iter = 500,           # Max iterations for gradient descent
      step_size = 0.1,          # Step size for gradient descent
      tol = 1e-8,               # Convergence tolerance
      n_cores = 8,              # Parallel cores
      val_sample_size = 50      # Validation sample size
    )

    best_lambda <- cv_out$best_lambda
    cat("Best lambda =", best_lambda, "\n")
    
    #==========================================================================
    # ALGORITHM 1, STEP 2: Bias correction using target data
    #==========================================================================
    # NOTE: Using fixed λ=0.25 (can also use best_lambda from CV)
    f_hat <- compute_f_L2(Y_t = Y_t_train,
                          s_vec = s_vec_t_train,
                          f1_hat = f1_hat,
                          lambda = 0.25,        # Fixed lambda (alternatively: best_lambda)
                          M = M,
                          max_iter = 200,       # Fewer iterations for real data
                          step_size = 0.1,      # Smaller step size
                          tol = 1e-3)           # Looser tolerance
    
    #==========================================================================
    # ALGORITHM 1, STEP 3: Projection to Wasserstein space
    #==========================================================================
    # Ensure the result is a valid quantile function (monotone increasing)
    # This step is EXPLICIT for real data (implicit in simulation)
    f_hat = Project(f_hat)
    
    #==========================================================================
    # Evaluate all methods
    #==========================================================================
    # d_1: WaTL (proposed method - full Algorithm 1)
    d_1 <- compute_L2_distance(true_Et, f_hat)
    
    # d_2: Only Target (local Fréchet regression on target data only)
    # This is the baseline without transfer learning
    f_target <- lrem(qin.target, xin.t, X_value, optns)$qp[1,]
    d_2 <- compute_L2_distance(true_Et, f_target)
    
    # d: Distance from auxiliary estimator f1_hat (diagnostic)
    # Shows performance before bias correction (Step 2)
    
    # Store results for this test point
    fold_results[i, "d_1"] <- d_1
    fold_results[i, "d_2_target"] <- d_2
    fold_results[i, "d_2_source"] <- d
    
    cat("d:",d,"d1:",d_1,"d2:",d_2,"\n")
    
    if (i %% 10 == 0 || i == n_test) {
      cat(sprintf("Processed %d/%d test points in fold %d\n", i, n_test, outer_k))
    }
  }
  
  # Store results for this fold
  all_results[[outer_k]] <- fold_results
  
  # Print average performance for this fold
  cat("Average results for fold", outer_k, ":\n")
  print(colMeans(fold_results, na.rm=TRUE))
}

################################################################################
# Aggregate and Save Results
################################################################################
# Compute overall statistics across all folds
total_points <- sum(fold_sizes)
weighted_means <- colMeans(do.call(rbind, all_results), na.rm=TRUE)
cat("\nOverall weighted averages (RMSPR):\n")
print(weighted_means)

# Compute standard errors
all_results_matrix <- do.call(rbind, all_results)
standard_errors <- apply(all_results_matrix, 2, function(x) sd(x, na.rm=TRUE) / sqrt(nrow(all_results_matrix)))
cat("\nStandard Errors:\n")
print(standard_errors)

# Save all results to file
# Create data directory if it doesn't exist
dir.create("data", recursive = TRUE, showWarnings = FALSE)

save(all_results, fold_sizes, weighted_means, standard_errors,
     file = paste0('data/', seed, '_', r, '_', M,'_',rate,'_', Gender, '_.RData'))

