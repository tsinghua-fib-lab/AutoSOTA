################################################################################
# Wasserstein Transfer Learning (WaTL) - Simulation Study
# 
# It compares three methods:
#   - WaTL: Wasserstein Transfer Learning (Algorithm 1)
#   - Only Target: Global Fréchet regression using target data only
#   - Only Source: Global Fréchet regression using source data only
#
# Arguments:
#   M: Grid size for quantile functions
#   n_t: Target sample size (varies from 200 to 800)
#   seed: Random seed
#   setting: Data generation setting (1 or 2)
#   ns: Source sample multiplier (either 100 or 200)
#
# Output:
#   RMSPR (Root Mean Squared Prediction Risk) for each method
################################################################################

################################################################################
# Parse command line arguments
################################################################################
args <- commandArgs(TRUE)
print(args)

M <- as.numeric(args[[1]])        # Grid size for discretized quantile functions
n_t <- as.numeric(args[[2]])      # Target sample size (e.g., 200, 400, 600, 800)
seed <- as.numeric(args[[3]])     # Random seed for reproducibility
setting <- as.numeric(args[[4]])  # Data generation setting (1 or 2)
ns <- as.numeric(args[[5]])       # source sample multiplier

################################################################################
# Load required libraries
################################################################################
library(truncnorm)    # For truncated normal distributions
library(pracma)       # For practical numerical analysis
library(Matrix)       # For sparse matrix operations
library(osqp)         # For quadratic programming (projection step)
library(parallel)     # For parallel computing
library(doParallel)   # For parallel backend
library(foreach)      # For parallel loops

# Load helper functions (data generation, WaTL algorithm, etc.)
source("SimulationFunc.R") 

set.seed(seed)  # Set random seed

# Number of source datasets
K <- 5

# Source sample sizes: n_k = k*tau where k=1,...,5 and tau ∈ {100, 200}
# Example: if ns=100, then n_vec = [100, 200, 300, 400, 500]
n_vec <- c(1, 2, 3, 4, 5) * ns

# Number of Monte Carlo repetitions
nRepeat <- 50

# Grids for quantile function evaluation
# z_grid: evaluation points in (0,1) for output
# p_grid: probability levels for input quantile functions
z_grid <- seq(1 / M, 1 - 1 / M, length.out = M)
p_grid <- seq(1 / M, 1 - 1 / M, length.out = M)

# Storage matrix for results
lossMat <- matrix(NA, nrow = nRepeat, ncol = 4)
colnames(lossMat) <- c("d_1", "d_2_target", "d_2_source", "d")

################################################################################
# Define test points (100 predictor values)
################################################################################
if(setting == 1){
  # Setting 1: X ∈ [0, 1]
  X_values <- seq(0, 1, length.out = 100)
} else if(setting == 2){
  # Setting 2: X ∈ [-1, 1]
  X_values <- seq(-1, 1, length.out = 100)
}

################################################################################
# Setup parallel computing for efficiency
################################################################################
cl <- makeCluster(10)  # Use 10 cores
registerDoParallel(cl)

# Array to store results: [X_values × nRepeat × 4 methods]
all_results <- array(NA, dim = c(length(X_values), nRepeat, 4))
dimnames(all_results)[[3]] <- c("d_1", "d_2_target", "d_2_source", "d")

# Export necessary variables to parallel workers
clusterExport(cl, c("n_t", "n_vec", "M", "K", "setting", "z_grid", "p_grid"))

################################################################################
# Main Monte Carlo loop over test points
################################################################################
for (x_idx in 1:length(X_values)) {
  X_value <- X_values[x_idx]
  cat(sprintf("\nProcessing X_value = %.3f (%d/%d)\n", 
              X_value, x_idx, length(X_values)))
  
  # Parallel loop over Monte Carlo repetitions
  results <- foreach(iter = 1:nRepeat, 
                     .combine = 'rbind',
                     .packages = c('truncnorm', 'pracma', 'Matrix', 'osqp')) %dopar% {
                       
                       #================================================================
                       # STEP 0: Generate simulated data
                       #================================================================
                       # Generate target and source data
                       if(setting == 1){
                         simu_data <- simulate_data_setting1(n_t, n_vec, M, K)
                       } else if(setting == 2) {
                         simu_data <- simulate_data_setting2(n_t, n_vec, M, K)
                       }
                       
                       # Extract target data: predictors and response quantile functions
                       X_t <- simu_data$X_t  # Target predictors (n_t × 1)
                       Y_t <- simu_data$Y_t  # Target quantile functions (n_t × M)
                       source_d <- simu_data$source_data  # List of K source datasets
                       
                       #================================================================
                       # ALGORITHM 1 - WASSERSTEIN TRANSFER LEARNING (WaTL)
                       #================================================================
                       
                       # STEP 1: Compute weighted auxiliary estimator f̂(x)
                       # Formula: f̂(x) = (1/(n_0+n_A)) * Σ_{k=0}^K n_k * f̂^(k)(x)
                       # This aggregates information from target (k=0) and all sources (k=1,...,K)
                       f1_hat <- compute_f1_hat(source_data_list = source_d,
                                                X_value = X_value,
                                                M = M,
                                                X_t = X_t,
                                                Y_t = Y_t)
                       
                       #================================================================
                       # Compute global Fréchet weights for target data
                       #================================================================
                       # Weight function: s_G^(0)(x) = 1 + (X_i - X̄)^T Σ^(-1) (x - X̄)
                       X_t_mat <- to_matrix(X_t)
                       xbar <- colMeans(X_t_mat)  # Sample mean of target predictors
                       Sigma <- cov(X_t_mat) * (nrow(X_t_mat)-1)/nrow(X_t_mat)  # Sample covariance
                       invSigma <- solve(Sigma)  # Inverse covariance
                       
                       # Compute weight for each target observation at query point X_value
                       s_vec_t <- sapply(1:nrow(X_t_mat), function(i){
                         as.numeric(1 + (X_t_mat[i,] - xbar) %*% invSigma %*% (X_value - xbar))
                       })
                       
                       # Interpolate quantile functions to common grid
                       F_inv_Yt <- t(apply(Y_t, 1, function(row) {
                         F_inv_t_i_fun <- approxfun(p_grid, row, method = "linear", rule = 2)
                         F_inv_t_i_fun(z_grid)
                       }))
                       
                       #================================================================
                       # Compute discrepancy between target and each source
                       #================================================================
                       # For monitoring/diagnostic purposes (not used in WaTL algorithm)
                       # ψ_k = ||f^(0)(x) - f^(k)(x)||_2
                       d_k_vec <- numeric(K)
                       for(k in 1:K) {
                         # Extract source k data
                         X_s <- source_d[[k]]$X_s
                         X_s_mat <- to_matrix(X_s)
                         xbar_s <- colMeans(X_s_mat)
                         Sigma_s <- cov(X_s_mat) * (nrow(X_s_mat)-1)/nrow(X_s_mat)
                         invSigma_s <- solve(Sigma_s)
                         
                         # Compute global Fréchet weights for source k
                         s_vec_s <- sapply(1:nrow(X_s_mat), function(i){
                           as.numeric(1 + (X_s_mat[i,] - xbar_s) %*% invSigma_s %*% (X_value - xbar_s))
                         })
                         
                         # Interpolate source quantile functions
                         Y_s <- source_d[[k]]$Y_s
                         F_inv_Ys <- t(apply(Y_s, 1, function(row) {
                           F_inv_s_i_fun <- approxfun(p_grid, row, method = "linear", rule = 2)
                           F_inv_s_i_fun(z_grid)
                         }))
                         
                         # Compute L2 distance between weighted target and source estimates
                         diff_k <- colMeans(s_vec_s * F_inv_Ys) - colMeans(s_vec_t * F_inv_Yt)
                         d_k_vec[k] <- sqrt(sum(diff_k^2))
                       }
                       
                       # Maximum discrepancy: ψ = max_{k} ψ_k
                       d <- max(d_k_vec)
                       
                       #================================================================
                       # Select regularization parameter λ via cross-validation
                       #================================================================
                       # Define grid of candidate λ values (depends on setting)
                       if(setting == 1){
                         lambda_candidates <- seq(0, 3, by = 0.1)  # Wider range for Setting 1
                       } else if(setting == 2){
                         lambda_candidates <- seq(0, 0.05, by = 0.01)  # Narrower range for Setting 2
                       }
                       
                       # Perform cross-validation to select best λ
                       cv_out <- cv_search_lambda(
                        Y_t = Y_t,
                        s_vec = s_vec_t,
                        X_t = X_t,
                        source_d = source_d,
                        M = M,
                        lambda_grid = lambda_candidates,
                        n_folds = 3,           # 3-fold cross-validation
                        max_iter = 500,        # Max iterations for gradient descent
                        step_size = 0.1,       # Step size for gradient descent
                        tol = 1e-8,            # Convergence tolerance
                        n_cores = 8,           # Number of cores for parallel CV
                        val_sample_size = 50   # Validation sample size
                      )
                       
                       best_lambda <- cv_out$best_lambda
                       
                       #================================================================
                       # STEP 2: Bias correction using target data
                       #================================================================
                       # Minimize: (1/n_0) Σ_i s_i ||F_i^(-1) - g||_2^2 + λ||g - f̂(x)||_2
                       # This corrects the bias in f̂(x) using target-specific information
                       f_hat <- compute_f_L2(Y_t = Y_t,
                                             s_vec = s_vec_t,
                                             f1_hat = f1_hat,
                                             lambda = best_lambda,
                                             M = M,
                                             max_iter = 1000,    # Max iterations
                                             step_size = 0.25,   # Step size
                                             tol = 1e-8)         # Convergence tolerance
                       
                       # STEP 3: Projection to Wasserstein space
                       # For simulation, this is implicit (quantile functions are already monotone)
                       # No explicit projection needed here
                       
                       #================================================================
                       # Compute true regression function for evaluation
                       #================================================================
                       # True m_G^(0)(x) based on data generation mechanism
                       if(setting == 1){
                         # Setting 1: m_G(x) = (1-x)u + x*F^(-1)_{Z}(u)
                         true_Et <- (1 - X_value)*z_grid + X_value*qtruncnorm(z_grid, a=0, b=1, mean=0.5, sd=1)
                       } else if(setting == 2) {
                         # Setting 2: Different true function
                         true_Et <- 3 * X_value + (3 + 3 * X_value) * qtruncnorm(z_grid, a=0, b=1, mean=0.5, sd=1)
                       }
                       
                       #================================================================
                       # BASELINE 1: WaTL (proposed method - Algorithm 1)
                       #================================================================
                       # Compute L2 distance from truth
                       d_1 <- compute_L2_distance(true_Et, f_hat)
                       
                       #================================================================
                       # BASELINE 2: Only Target 
                       #================================================================
                       # Global Fréchet regression using ONLY target data
                       # This is the standard approach without transfer learning
                       qin.target <- split(Y_t, row(Y_t))
                       qin.target <- lapply(qin.target, as.numeric)
                       xin.t <- to_matrix(X_t)
                       f_target <- grem(qin.target, xin.t, X_value)$qp[1,]  # Get predicted quantile
                       d_2 <- compute_L2_distance(true_Et, f_target)
                       
                       #================================================================
                       # BASELINE 3: Only Source
                       #================================================================
                       # Global Fréchet regression using ONLY source data
                       # This ignores target data completely
                       # Combine all source datasets
                       X_s_all <- NULL
                       Y_s_all <- NULL
                       for(k in 1:K) {
                         X_s_all <- c(source_d[[k]]$X_s, X_s_all)
                         Y_s_all <- rbind(source_d[[k]]$Y_s, Y_s_all)
                       }
                       
                       # Fit global Fréchet regression on pooled source data
                       qin.source <- split(Y_s_all, row(Y_s_all))
                       qin.source <- lapply(qin.source, as.numeric)
                       xin.s <- to_matrix(X_s_all)
                       f_source <- grem(qin.source, xin.s, X_value)$qp[1,]
                       d_3 <- compute_L2_distance(true_Et, f_source)
                       
                       # Return: [WaTL, Only Target, Only Source, Max Discrepancy]
                       c(d_1, d_2, d_3, d)
                     }
  
  # Store results for this X_value across all repetitions
  all_results[x_idx, , ] <- results
  
  # Print average performance for this X_value
  cat(sprintf("Average for X_value = %.3f:\n", X_value))
  print(colMeans(results))
}

################################################################################
# Cleanup and save results
################################################################################
stopCluster(cl)  # Stop parallel cluster

# Compute overall average performance across all X_values and repetitions
cat("\nOverall averages:\n")
overall_means <- apply(all_results, 3, mean, na.rm = TRUE)
print(overall_means)

# Create output directory if it doesn't exist
dir.create(paste0("Setting", setting), recursive = TRUE, showWarnings = FALSE)

# Save results to file
# Filename format: Setting{setting}/{M}_{n_t}_{seed}_{ns}.RData
save(all_results, X_values, 
     file = paste0('Setting', setting, "/", 
                  M, '_', n_t, '_', seed, '_', ns, '.RData'))
