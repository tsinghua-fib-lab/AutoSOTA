to_matrix <- function(x) {
  if (is.vector(x)) {
    matrix(x, ncol=1)
  } else if (is.data.frame(x)) {
    as.matrix(x)
  } else {
    x
  }
}

cv_search_lambda <- function(Y_t,
                             s_vec,
                             X_t,
                             source_d,
                             M,
                             lambda_grid = NULL,
                             n_folds = 5,
                             max_iter = 200,
                             step_size = 0.5,
                             tol = 1e-6,
                             n_cores = NULL,
                             val_sample_size = 30
)
{
  start_time <- Sys.time()
  
  if (is.null(n_cores)) {
    n_cores <- max(1, detectCores() - 1)
  }
  
  n <- nrow(Y_t)
  total_lambdas <- length(lambda_grid)
  
  cat(sprintf("\nStarting cross-validation at %s\n", format(start_time)))
  cat(sprintf("Total lambdas to test: %d\n", total_lambdas))
  cat(sprintf("Number of cores: %d\n", n_cores))
  cat(sprintf("Validation sample size: %d\n", val_sample_size))
  cat("----------------------------------------\n\n")
  
  shuffle_idx <- sample.int(n)
  fold_id <- cut(seq_len(n), breaks = n_folds, labels = FALSE)
  
  cl <- makeCluster(n_cores)
  registerDoSNOW(cl)
  
  pb <- txtProgressBar(max=total_lambdas, style=3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress=progress)
  
  clusterExport(
    cl, 
    c("compute_f1_hat", 
      "compute_f_L2",
      "to_matrix",
      "kerFctn",
      "SetBwRange",
      "plcm",
      "lcm",
      "gcd",
      "bwCV",
      "lrem",
      "compute_L2_distance",
      "Y_t",
      "s_vec",
      "X_t",
      "source_d",
      "M",
      "rate",
      "lambda_grid",
      "n_folds",
      "max_iter",
      "step_size",
      "tol",
      "val_sample_size",
      "shuffle_idx",
      "fold_id"
    ),
    envir = environment()
  )
  
  progress_counter <- 0
  
  results <- foreach(l_idx = seq_along(lambda_grid),
                     .combine = 'rbind',
                     .options.snow = opts) %dopar% {
                       lambda_val <- lambda_grid[l_idx]
                       fold_errors <- numeric(n_folds)
                       
                       for (k in seq_len(n_folds)) {
                         val_idx_full <- shuffle_idx[fold_id == k]
                         
                         if (length(val_idx_full) > val_sample_size) {
                           val_idx <- sample(val_idx_full, val_sample_size)
                         } else {
                           val_idx <- val_idx_full
                         }
                         
                         train_idx <- shuffle_idx[fold_id != k]
                         
                         Y_train <- Y_t[train_idx, , drop = FALSE]
                         s_vec_train <- s_vec[train_idx]
                         X_train <- X_t[train_idx, ]
                         
                         Y_val <- Y_t[val_idx, , drop = FALSE]
                         s_vec_val <- s_vec[val_idx]
                         X_val <- X_t[val_idx, ]
                         
                         val_errors_fold <- numeric(length(val_idx))
                         
                         for (i in seq_along(val_idx)) {
                           X_value <- as.vector(as.numeric(X_val[i,]))
                           
                           f1_hat_i <- compute_f1_hat(
                             source_data_list = source_d,
                             X_value = X_value,
                             M = M,
                             X_t = X_train,
                             Y_t = Y_train
                           )
                           
                           f_train <- tryCatch({
                             compute_f_L2(
                               Y_t = Y_train,
                               s_vec = s_vec_train,
                               f1_hat = f1_hat_i,
                               lambda = lambda_val,
                               M = M,
                               max_iter = max_iter,
                               step_size = step_size,
                               tol = tol
                             )
                           }, error = function(e) NULL)
                           
                           if (!is.null(f_train)) {
                             val_errors_fold[i] <- mean((f_train - Y_val[i,])^2)
                           } else {
                             val_errors_fold[i] <- NA
                           }
                         }
                         
                         valid_errors <- !is.na(val_errors_fold)
                         if (sum(valid_errors) > 0) {
                           fold_errors[k] <- mean(val_errors_fold[valid_errors])
                         } else {
                           fold_errors[k] <- NA
                         }
                       }
                       
                       c(lambda_val, mean(fold_errors, na.rm = TRUE))
                     }
  
  stopCluster(cl)
  close(pb)
  
  end_time <- Sys.time()
  total_time <- difftime(end_time, start_time, units = "mins")
  
  cat("\n\n----------------------------------------\n")
  cat(sprintf("Cross-validation completed at %s\n", format(end_time)))
  cat(sprintf("Total time elapsed: %.2f minutes\n", as.numeric(total_time)))
  cat("----------------------------------------\n")
  
  results <- as.data.frame(results)
  colnames(results) <- c("lambda", "cv_error")
  
  best_lambda <- results$lambda[which.min(results$cv_error)]
  best_error <- min(results$cv_error)
  
  return(list(
    results = results,
    best_lambda = best_lambda,
    total_time = total_time
  ))
}

################################################################################
## Data Generation: Setting 1
################################################################################
# Generates target and source data according to Setting 1 specifications
#
# Target Population (k=0):
#   - X^(0) ~ U(0,1)
#   - F^{-1}_{ν^(0)}(u) = w^(0)(1-u)u + (1-X^(0))u + X^(0) F^{-1}_{Z^(0)}(u)
#   - Z^(0) ~ N(0.5, 1)|_{(0,1)} (truncated normal)
#   - w^(0) ~ N(0, 1)|_{(-0.5, 0.5)} (truncated normal)
#
# Source Population (k=1,...,K):
#   - ψ_k = 0.1k (similarity parameter)
#   - X^(k) ~ U(0,1)
#   - F^{-1}_{ν^(k)}(u) = w^(k)(1-u)u + (1-X^(k))u + X^(k) F^{-1}_{Z^(k)}(u)
#   - Z^(k) ~ N(0.5, 1-ψ_k)|_{(0,1)}
#   - w^(k) ~ N(0, 1)|_{(-0.5, 0.5)}
#
# Arguments:
#   n_t: Target sample size
#   n_vec: Vector of source sample sizes (length K)
#   M: Grid size for quantile functions
#   K: Number of source datasets (default 5)
#
# Returns:
#   List with X_t, Y_t (target data) and source_data (list of K source datasets)
simulate_data_setting1 <- function(n_t, n_vec, M, K=5)
{
  
  # Define quantile function for target distribution Z^(0)
  # Z^(0) ~ N(0.5, 1)|_{(0,1)}
  F_inv_Phi <- function(z) {
    qtruncnorm(z, a=0, b=1, mean=0.5, sd=1)
  }
  
  # Define quantile function for source distribution Z^(k)
  # Z^(k) ~ N(0.5, 1-ψ_k)|_{(0,1)}
  # Note: variance decreases as ψ_k increases (sources become less similar to target)
  F_inv_phi <- function(z, d_k){
    qtruncnorm(z, a=0, b=1, mean=0.5, sd=sqrt(1 - d_k))
  }
  
  #============================================================================
  # Generate TARGET data (k=0)
  #============================================================================
  X_t <- runif(n_t, 0, 1)  # Predictors: X^(0) ~ U(0,1)
  w_t <- rtruncnorm(n_t, a=-0.5, b=0.5, mean=0, sd=1)  # Random effects: w^(0) ~ N(0,1)|_{(-0.5,0.5)}
  
  # Grid for evaluating quantile functions
  z_grid <- seq(1 / M, 1 - 1/ M, length.out=M)
  
  # Generate quantile functions for target responses
  # Each row i: F^{-1}_{ν_i^(0)}(u) = w_i(1-u)u + (1-X_i)u + X_i F^{-1}_{Z^(0)}(u)
  Y_t <- t(sapply(seq_len(n_t), function(i) {
    w_t[i]*(1-z_grid)*z_grid + (1 - X_t[i])*z_grid + X_t[i]*F_inv_Phi(z_grid)
  }))
  
  #============================================================================
  # Generate SOURCE data (k=1,...,K)
  #============================================================================
  source_data <- list()  
  for(k_idx in seq_len(K)){
    # Similarity parameter: ψ_k = 0.1k
    # Larger ψ_k means source k is less similar to target
    d_k <- k_idx * 0.1
    n_k <- n_vec[k_idx]  # Sample size for source k
    
    # Generate predictors and random effects for source k
    X_s_k <- runif(n_k, 0, 1)  # X^(k) ~ U(0,1)
    w_s_k <- rtruncnorm(n_k, a=-0.5, b=0.5, mean=0, sd=1)  # w^(k) ~ N(0,1)|_{(-0.5,0.5)}
    
    # Generate quantile functions for source k responses
    # F^{-1}_{ν_i^(k)}(u) = w_i(1-u)u + (1-X_i)u + X_i F^{-1}_{Z^(k)}(u)
    Y_s_k <- t(sapply(seq_len(n_k), function(i) {
      w_s_k[i]*(1-z_grid)*z_grid + (1 - X_s_k[i])*z_grid + X_s_k[i]*F_inv_phi(z_grid, d_k)
    }))
    
    # Store source k data
    source_data[[k_idx]] <- list(
      X_s = X_s_k,    # Predictors (n_k × 1)
      Y_s = Y_s_k,    # Quantile functions (n_k × M)
      d_k = d_k       # Similarity parameter ψ_k
    )
  }
  
  return(list(
    X_t = X_t,              # Target predictors (n_t × 1)
    Y_t = Y_t,              # Target quantile functions (n_t × M)
    source_data = source_data  # List of K source datasets
  ))
}


## Data Generation Setting 2
simulate_data_setting2 <- function(n_t, n_vec, M, K=5){

  X_t <- runif(n_t, -1, 1)
  mu_t <- rnorm(n_t, mean=3*X_t, sd=0.5) 

  shape_t <- (3 + 3 * X_t)^2
  rate_t <- 3 + 3 * X_t
  sigma_t <- rgamma(n_t, shape=shape_t, rate=rate_t)
  

  F_inv_Phi <- function(z){
    qtruncnorm(z, a=0, b=1, mean=0.5, sd=1)
  }
  
  z_grid <- seq(1 / M, 1 - 1/ M, length.out=M)
  
  Y_t<- t(sapply(seq_len(n_t), function(i){
    mu_t[i] + sigma_t[i] * F_inv_Phi(z_grid)
  }))
  
  source_data <- list()
  for(k_idx in seq_len(K))
  {
    d_k <- k_idx * 0.05
    n_k <- n_vec[k_idx]
    
    X_s_k  <- runif(n_k, -1, 1)
    mu_s_k <- rnorm(n_k, mean=3*X_s_k, sd=0.5)
    shape_s_k <- (3 + d_k + (3 + d_k)*X_s_k)^2
    rate_s_k  <- 3 + d_k + (3 + d_k)*X_s_k
    sigma_s_k <- rgamma(n_k, shape=shape_s_k, rate=rate_s_k)
    
    Y_s_k <- t(sapply(seq_len(n_k), function(i){
      mu_s_k[i] + sigma_s_k[i] * F_inv_Phi(z_grid)
    }))
    
    source_data[[k_idx]] <- list(
      X_s = X_s_k,
      Y_s = Y_s_k,
      d_k = d_k
    )
  }
  
  return(list(
    X_t = X_t,
    Y_t = Y_t,
    source_data = source_data
  ))
}


################################################################################
## ALGORITHM 1, STEP 1: Compute Weighted Auxiliary Estimator
################################################################################
# Implements the weighted auxiliary estimator f̂(x) which aggregates information
# from both target (k=0) and source (k=1,...,K) datasets.
#
# Formula:
#   f̂(x) = (1/(n_0 + n_A)) * Σ_{k=0}^K n_k * f̂^(k)(x)
#
# where:
#   - f̂^(k)(x) = (1/n_k) Σ_{i=1}^{n_k} s_G^(k)(x_i) * F^{-1}_{ν_i^(k)}
#   - s_G^(k)(x) = 1 + (X^(k) - θ_k)^T Σ_k^{-1} (x - θ_k)  [global Fréchet weight]
#   - n_A = Σ_{k=1}^K n_k (total source sample size)
#
# Arguments:
#   source_data_list: List of K source datasets
#   X_value: Query point where to evaluate f̂(x)
#   M: Grid size for quantile functions
#   X_t: Target predictors (optional, but should be provided)
#   Y_t: Target quantile functions (optional, but should be provided)
#
# Returns:
#   f̂(x): Weighted auxiliary estimator (vector of length M)
compute_f1_hat <- function(source_data_list, 
                       X_value, 
                       M,
                       X_t = NULL,
                       Y_t = NULL)
{

  K <- length(source_data_list)  # Number of source datasets

  z_grid <- seq(1 / M, 1 - 1/ M, length.out=M)  # Grid for quantile evaluation
  bigF <- rep(0, length(z_grid))  # Accumulator for weighted sum
  n_total <- 0  # Total sample size counter (n_0 + n_A)
  
  #============================================================================
  # Include TARGET data (k=0) in the weighted estimator
  #============================================================================
  # This is CRITICAL: Step 1 must include target data
  if (!is.null(X_t) && !is.null(Y_t)) {
    n_t <- length(X_t)
    X_t_mat <- to_matrix(X_t)
    
    # Compute empirical mean and covariance for target
    xbar_t <- colMeans(X_t_mat)
    Sigma_t <- cov(X_t_mat) * (nrow(X_t_mat)-1)/nrow(X_t_mat)
    invSigma_t <- solve(Sigma_t)
    
    # Compute global Fréchet weights for target: s_G^(0)(X_i^(0))
    # Weight: 1 + (X_i - X̄)^T Σ^{-1} (x - X̄)
    s_vec_t <- sapply(1:nrow(X_t_mat), function(i){
      as.numeric(1 + (X_t_mat[i,] - xbar_t) %*% invSigma_t %*% (X_value - xbar_t))
    })
    
    # Accumulate weighted target contributions: n_0 * f̂^(0)(x)
    for(i in seq_len(n_t)){
      bigF <- bigF + s_vec_t[i] * Y_t[i,]
    }
    n_total <- n_total + n_t
  }
  
  #============================================================================
  # Add SOURCE data (k=1,...,K) contributions
  #============================================================================
  for(k_idx in seq_len(K))
  {
    # Extract source k data
    X_s   <- source_data_list[[k_idx]]$X_s
    Y_s <- source_data_list[[k_idx]]$Y_s
    n_k   <- length(X_s)

    X_s_mat  <- to_matrix(X_s)
    
    # Compute empirical mean and covariance for source k
    xbar <- colMeans(X_s_mat)
    Sigma <- cov(X_s_mat) * (nrow(X_s_mat)-1)/nrow(X_s_mat)
    invSigma <- solve(Sigma)
    
    # Compute global Fréchet weights for source k: s_G^(k)(X_i^(k))
    s_vec <- sapply(1:nrow(X_s_mat), function(i){
      as.numeric(1 + (X_s_mat[i,] - xbar) %*% invSigma %*% (X_value - xbar))
    })
    
    # Accumulate weighted source k contributions: n_k * f̂^(k)(x)
    for(i in seq_len(n_k)){
      bigF <- bigF + s_vec[i] * Y_s[i,]
    }
    n_total <- n_total + n_k
  }
  
  # Return weighted average: f̂(x) = (Σ_{k=0}^K n_k f̂^(k)(x)) / (n_0 + n_A)
  f1 <- bigF / n_total
  
  return(f1)
}


################################################################################
## ALGORITHM 1, STEP 2: Bias Correction Using Target Data
################################################################################
# Implements the bias-corrected estimator f̂_0(x) using gradient descent.
#
# Formula:
#   f̂_0(x) = argmin_{g∈L^2(0,1)} (1/n_0) Σ_{i=1}^{n_0} s_i ||F^{-1}_{ν_i^(0)} - g||_2^2 
#                                  + λ ||g - f̂(x)||_2
#
# where:
#   - First term: Target data fidelity (uses global Fréchet weights s_i)
#   - Second term: Regularization toward auxiliary estimator f̂(x) from Step 1
#   - λ: Regularization parameter (selected via cross-validation)
#
# This step corrects the bias in f̂(x) by incorporating target-specific information
# while maintaining stability through regularization.
#
# Arguments:
#   Y_t: Target quantile functions (n_0 × M matrix)
#   s_vec: Global Fréchet weights for target (length n_0)
#   f1_hat: Weighted auxiliary estimator from Step 1 (length M)
#   lambda: Regularization parameter
#   M: Grid size
#   max_iter: Maximum iterations for gradient descent
#   step_size: Step size for gradient descent
#   tol: Convergence tolerance
#
# Returns:
#   f̂_0(x): Bias-corrected estimator (vector of length M)
compute_f_L2 <- function(Y_t,
                         s_vec,
                         f1_hat,
                         lambda,
                         M,
                         max_iter = 1000,
                         step_size = 0.5,
                         tol = 1e-8) {
  # Input validation
  n0 <- nrow(Y_t)
  if (length(s_vec) != n0) {
    stop("s_vec length must match row number of Y_t.")
  }
  if (length(f1_hat) != M) {
    stop("f1_hat length must be M.")
  }
  
  # Initialize at auxiliary estimator f̂(x) from Step 1
  f <- as.numeric(f1_hat)
  
  #============================================================================
  # Gradient Descent Optimization
  #============================================================================
  # Minimize: L(g) = (1/n_0) Σ_i s_i ||Y_i - g||_2^2 + λ ||g - f̂||_2
  #
  # Gradient: ∇L(g) = -(2/n_0) Σ_i s_i (Y_i - g) + 2λ(g - f̂)
  #
  for (iter in 1:max_iter) 
  {
    # Compute residuals: Y_i - f for each target observation
    diff_mat <- sweep(Y_t, 2, f, FUN = "-") 
    
    # Negate for gradient computation: -(Y_i - f)
    diff_mat <- -diff_mat                 
    
    # Weight by Fréchet weights: s_i * (-(Y_i - f))
    weighted_diff_mat <- sweep(diff_mat, 1, s_vec, FUN = "*")  
    
    # Sum over observations: Σ_i s_i (-(Y_i - f))
    gradient_target <- colSums(weighted_diff_mat)
    
    # Regularization gradient: λ(f - f̂)
    gradient_reg <- lambda * (f - f1_hat) 
    
    # Total gradient
    grad <- gradient_target + gradient_reg
    
    # Gradient descent update: f ← f - α * ∇L(f)
    f_new <- f - step_size * grad 
    
    # Check convergence: ||f_new - f|| < tolerance
    diff_norm <- sqrt(sum((f_new - f)^2))
    if (diff_norm < tol) {
      f <- f_new
      break
    }
    
    f <- f_new
  }
  
  return(as.numeric(f))
}

grem <- function(y = NULL,
                 x = NULL,
                 xOut = NULL,
                 optns = list()) {
  if (is.null(y) | is.null(x)) {
    stop("requires the input of both y and x")
  }
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector")
    }
  }
  n <- nrow(x) # number of observations
  p <- ncol(x) # number of covariates
  if (!is.list(y)) {
    stop("y must be a list")
  }
  if (length(y) != n) {
    stop("the number of rows in x must be the same as the number of empirical measures in y")
  }
  if (!is.null(xOut)) {
    if (!is.matrix(xOut)) {
      if (is.data.frame(xOut)) {
        xOut <- as.matrix(xOut)
      } else if (is.vector(xOut)) {
        if (p == 1) {
          xOut <- as.matrix(xOut)
        } else {
          xOut <- t(xOut)
        }
      } else {
        stop("xOut must be a matrix or a data frame or a vector")
      }
    }
    if (ncol(xOut) != p) {
      stop("x and xOut must have the same number of columns")
    }
    nOut <- nrow(xOut) # number of predictions
  } else {
    nOut <- 0
  }
  
  N <- sapply(y, length)
  y <- lapply(1:n, function(i) {
    sort(y[[i]])
  }) # sort observed values
  
  M <- min(plcm(N), n * max(N), 5000) # least common multiple of N_i
  yM <- t(sapply(1:n, function(i) {
    residual <- M %% N[i]
    if(residual) {
      sort(c(rep(y[[i]], each = M %/% N[i]), sample(y[[i]], residual)))
    } else {
      rep(y[[i]], each = M %/% N[i])
    }
  })) # n by M
  
  # initialization of OSQP solver
  A <- cbind(diag(M), rep(0, M)) + cbind(rep(0, M), -diag(M))
  if (!is.null(optns$upper) &
      !is.null(optns$lower)) {
    # if lower & upper are neither NULL
    l <- c(optns$lower, rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$upper)) {
    # if lower is NULL
    A <- A[, -1]
    l <- c(rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$lower)) {
    # if upper is NULL
    A <- A[, -ncol(A)]
    l <- c(optns$lower, rep(0, M - 1))
  } else {
    # if both lower and upper are NULL
    A <- A[, -c(1, ncol(A))]
    l <- rep(0, M - 1)
  }
  # P <- as(diag(M), "sparseMatrix")
  # A <- as(t(A), "sparseMatrix")
  P <- diag(M)
  A <- t(A)
  q <- rep(0, M)
  u <- rep(Inf, length(l))
  model <-
    osqp::osqp(
      P = P,
      q = q,
      A = A,
      l = l,
      u = u,
      osqp::osqpSettings(max_iter = 1e05, eps_abs = 1e-05, eps_rel = 1e-05, verbose = FALSE)
    )
  
  xMean <- colMeans(x)
  invVa <- solve(var(x) * (n - 1) / n)
  wc <-
    t(apply(x, 1, function(xi) {
      t(xi - xMean) %*% invVa
    })) # n by p
  if (nrow(wc) != n) {
    wc <- t(wc)
  } # for p=1
  
  qf <- matrix(nrow = n, ncol = M)
  residuals <- rep.int(0, n)
  totVa <- sum((scale(yM, scale = FALSE))^2) / M
  for (i in 1:n) {
    w <- apply(wc, 1, function(wci) {
      1 + t(wci) %*% (x[i, ] - xMean)
    })
    qNew <- apply(yM, 2, weighted.mean, w) # M
    if (any(w < 0)) {
      # if negative weights exist, project
      model$Update(q = -qNew)
      qNew <- sort(model$Solve()$x)
    }
    if (!is.null(optns$upper)) {
      qNew <- pmin(qNew, optns$upper)
    }
    if (!is.null(optns$lower)) {
      qNew <- pmax(qNew, optns$lower)
    }
    qf[i, ] <- qNew
    residuals[i] <- sqrt(sum((yM[i, ] - qf[i, ])^2) / M)
  }
  qfSupp <- 1:M / M
  resVa <- sum(residuals^2)
  RSquare <- 1 - resVa / totVa
  adjRSquare <- RSquare - (1 - RSquare) * p / (n - p - 1)
  
  if (nOut > 0) {
    qp <- matrix(nrow = nOut, ncol = M)
    for (i in 1:nOut) {
      w <- apply(wc, 1, function(wci) {
        1 + t(wci) %*% (xOut[i, ] - xMean)
      })
      qNew <- apply(yM, 2, weighted.mean, w) # M
      if (any(w < 0)) {
        # if negative weights exist
        model$Update(q = -qNew)
        qNew <- sort(model$Solve()$x)
      }
      if (!is.null(optns$upper)) {
        qNew <- pmin(qNew, optns$upper)
      }
      if (!is.null(optns$lower)) {
        qNew <- pmax(qNew, optns$lower)
      }
      qp[i, ] <- qNew
    }
    qpSupp <- 1:M / M
    
    res <-
      list(
        qf = qf,
        qfSupp = qfSupp,
        qp = qp,
        qpSupp = qpSupp,
        RSquare = RSquare,
        adjRSquare = adjRSquare,
        residuals = residuals,
        y = y,
        x = x,
        xOut = xOut,
        optns = optns
      )
  } else {
    res <- list(
      qf = qf,
      qfSupp = qfSupp,
      RSquare = RSquare,
      adjRSquare = adjRSquare,
      residuals = residuals,
      y = y,
      x = x,
      optns = optns
    )
  }
  
  class(res) <- "rem"
  res
}


plcm <- function(x) {
  stopifnot(is.numeric(x))
  # if (any(floor(x) != ceiling(x)) || length(x) < 2)
  #   stop("Argument 'x' must be an integer vector of length >= 2.")
  
  x <- x[x != 0]
  n <- length(x)
  if (n == 0) {
    l <- 0
  } else if (n == 1) {
    l <- x
  } else if (n == 2) {
    l <- lcm(x[1], x[2])
  } else {
    l <- lcm(x[1], x[2])
    for (i in 3:n) {
      l <- lcm(l, x[i])
    }
  }
  return(l)
}

lcm <- function(n, m) {
  stopifnot(is.numeric(n), is.numeric(m))
  if (length(n) != 1 || floor(n) != ceiling(n) ||
      length(m) != 1 || floor(m) != ceiling(m)) {
    stop("Arguments 'n', 'm' must be integer scalars.")
  }
  if (n == 0 && m == 0) {
    return(0)
  }
  
  return(n / gcd(n, m) * m)
}

gcd <- function(n, m) {
  stopifnot(is.numeric(n), is.numeric(m))
  if (length(n) != 1 || floor(n) != ceiling(n) ||
      length(m) != 1 || floor(m) != ceiling(m)) {
    stop("Arguments 'n', 'm' must be integer scalars.")
  }
  if (n == 0 && m == 0) {
    return(0)
  }
  
  n <- abs(n)
  m <- abs(m)
  if (m > n) {
    t <- n
    n <- m
    m <- t
  }
  while (m > 0) {
    t <- n
    n <- m
    m <- t %% m
  }
  return(n)
}

compute_L2_distance <- function(vec1, vec2){
  sqrt(mean( (vec1 - vec2)^2 ))
}

