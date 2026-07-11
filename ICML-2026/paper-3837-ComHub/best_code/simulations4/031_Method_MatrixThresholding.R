#################################################
#################################################
## Description:
##
##  In this document, we define functions for
##  thresholded covariance/correlation matrix
##  estimation. This is given by the functions:
##
##      sta_thresholding_BL2008:
##          Tuning via resampling Frobenius norm. 
##      sta_thresholding_opnorm:
##          Tuning via resampling Operator norm.
##      sta_thresholding_oracle:
##          Tuning via oracle Operator norm.
##      sta_thresholding_perc: 
##          dropping a given percentage of entries.                 
##  
##  To standardize the format of all our functions,
##  all functions have the same inputs and outputs.
##
##  The "sta_***" functions require auxiliary functions
##  provided below.
##


#################################################
#################################################
## Thresholding Auxiliary Functions:
#################################################
#################################################


#################################################
#################################################
## lambda_range_selection:
##   For a given matrix, selects a range of 
##      thresholding values. to consider in later
##      stages.
##
##  INPUTS:
##      mat     : matrix to threshold.
##      length  : number of thresholding values.
##
##  OUTPUTS: 
##      .output  : numerical vector of thresholding
##                  parameter ranges.
##
lambda_range_selection <- function (mat, length = 100, ...) {
    
    ## Selecting non-zero entries.
    .mat_entries <- abs(mat[upper.tri(mat, diag = FALSE)])
    .mat_entries_nonzero <-  .mat_entries[.mat_entries > 0]
    
    ## Min-max range.
    .range_min <- min(.mat_entries_nonzero) * 0.9 
    .range_max <- max(.mat_entries_nonzero) * 1.1
    
    ## Constructing range.
    .output <- 10^(seq(
        log(.range_min, base = 10),
        log(.range_max, base = 10),
        length.out = length))

    return(.output)
}

#################################################
#################################################
## thresholding:
##   For a given matrix and threshold, calculates 
##   the off-diagonal thresholded matrix.
##
##  INPUTS:
##      mat     : matrix to threshold.
##      lambda  : thresholding parameter.
##
##  OUTPUTS: 
##      .output  : numerical matrix. Thresholded matrix.
##
thresholding <- function (mat, lambda, ...) {
    .nodiag <- mat - diag(diag(mat))
    .matHat_thresh <- .nodiag * (abs(.nodiag) > lambda) + diag(diag(mat))
    
    .output <- (.matHat_thresh + t(.matHat_thresh)) / 2
    return(.output)
}


#################################################
#################################################
## threshold_tuning_BL2008:
##   Given a range of thresholding parameters 
##   lambda_range, calculates the lambda threshold
##   which minimizes the resampling Frobenius
##   norm distance, similar to Bickel & Levina (2008).
##
##  INPUTS:
##      X            : data matrix.
##      mat_type     : if "cor", applies to correlations
##                      If "cov", applies to covariance.
##      lambda_range : Numeric vector. Range of 
##                      candidate thresholds. 
##      M            : Number of resamples 
##      true_mat     : ground truth matrix.
##
##  OUTPUTS: 
##      output  : numerical. Threshold value that 
##                  drops perc of the variables.
##
threshold_tuning_BL2008 <- function(X, mat_type = "cov", lambda_range, M = 100, true_mat = NULL, ...) {

    n           <- nrow(X)
    p           <- ncol(X)
    n_lambda    <- length(lambda_range)
    m_ver       <- as.integer(n / log(n))
    Fnorm       <- matrix(0, nrow = M, ncol = n_lambda)

    ## For each choice of lambda and split:
    for (split_ind in 1:M) {
        
        split <- sample(1:n, m_ver, FALSE)
            
        ## Split data by n - n/ log n  and n / log n.
        X1 <- X[-split, ] 
        X2 <- X[split, ]
        
        ## Thresholded matrix for logn/n sample.
        S1 <- get(mat_type)(X1)
        
        ## Direct matrix for logn/n sample.
        S2 <- get(mat_type)(X2)
            
        for (lambda_ind in 1:n_lambda) {
        
            T1 <- thresholding(S1, lambda_range[lambda_ind])
    
            ## Calculate Frobenius Norm:
            Fnorm[split_ind, lambda_ind] <- sum((T1 - S2)^2) 
        }
    }
    ## Average Frobenius Error for each lambda.
    Fnorm_av <- apply(Fnorm, MARGIN = 2, mean)

    opt_ind <- which.min(Fnorm_av)

    output <- list(
        opt_lambda = lambda_range[opt_ind],
        opt_ind = opt_ind)
    return(output)
}



#################################################
#################################################
## threshold_tuning_opnorm:
##   Given a range of thresholding parameters 
##   lambda_range, calculates the lambda threshold
##   which minimizes the resampling operator
##   norm distance, similar to Bickel & Levina (2008).
##
##  INPUTS:
##      X            : data matrix.
##      mat_type     : if "cor", applies to correlations
##                      If "cov", applies to covariance.
##      lambda_range : Numeric vector. Range of 
##                      candidate thresholds. 
##      M            : Number of resamples 
##      true_mat     : ground truth matrix.
##
##  OUTPUTS: 
##      output  : numerical. Threshold value that 
##                  drops perc of the variables.
##
threshold_tuning_opnorm <- function(X, mat_type = "cov", lambda_range, M = 100, true_mat = NULL, ...) {

    n           <- nrow(X)
    p           <- ncol(X)
    n_lambda    <- length(lambda_range)
    m_ver       <- as.integer(n / log(n))
    op_norm       <- matrix(0, nrow = M, ncol = n_lambda)

    ## For each choice of lambda and split:
    for (split_ind in 1:M) {
        
        split <- sample(1:n, m_ver, FALSE)
            
        ## Split data by n - n/ log n  and n / log n.
        X1 <- X[-split, ] 
        X2 <- X[split, ]
        
        ## Thresholded matrix for logn/n sample.
        S1 <- get(mat_type)(X1)
        
        ## Direct matrix for logn/n sample.
        S2 <- get(mat_type)(X2)
            
        for (lambda_ind in 1:n_lambda) {
        
            T1        <- thresholding(S1, lambda_range[lambda_ind])

            ## Sometimes, eigen gives error message. Use try to prevent code from stopping.
            eigvals   <- try(eigen(T1 - S2)$values, silent = TRUE, ...) 
        
            if (typeof(eigvals) %in% c("double", "numeric")) {
                ## If eigen runs, save operator norm.
                op_norm[split_ind, lambda_ind]  <- max(abs(eigvals)) 
            } else if (typeof(eigvals) == "character"){
                ## If eigen fails, save inf and index of error.
                op_norm[split_ind, lambda_ind]  <- Inf
            }
        }
    }
    ## Average Frobenius Error for each lambda.
    op_norm_av <- apply(op_norm, MARGIN = 2, mean)

    opt_ind <- which.min(op_norm_av)

    output <- list(
        opt_lambda = lambda_range[opt_ind],
        opt_ind = opt_ind)
    return(output)
}


#################################################
#################################################
## matrix_threshold_oracle:
##   Given a range of thresholding parameters 
##   lambda_range, calculates the lambda threshold
##   which minimizes the oracle operator norm 
##   distance to the true covariance matrix
##   true_mat.
##
##  INPUTS:
##      X            : data matrix.
##      mat_type     : if "cor", applies to correlations
##                      If "cov", applies to covariance.
##      lambda_range : Numeric vector. Range of 
##                      candidate thresholds. 
##      M            : Number of resamplings. Unused for
##                      this function, but included for consistency.
##      true_mat     : ground truth matrix.
##
##  OUTPUTS: 
##      output  : numerical. Threshold value that 
##                  drops perc of the variables.
##
matrix_threshold_oracle <- function(X, mat_type = "cov", lambda_range, M = NULL, true_mat, ...) {
    
    n           <- nrow(X)
    p           <- ncol(X)
    n_lambda    <- length(lambda_range)
    OpNorm      <- rep(0, n_lambda)
    Sn          <- get(mat_type)(X)

    bug_ind <- rep(FALSE, n_lambda)
    for (lambda_ind in 1:n_lambda) {
        lambda      <- lambda_range[lambda_ind]
        Tn          <- thresholding(Sn, lambda)

        ## Sometimes, eigen gives error message. Use try to prevent code from stopping.
        eigvals     <- try(eigen(Tn - true_mat)$values, silent = TRUE, ...) 
        
        if (typeof(eigvals) %in% c("double", "numeric")) {
            ## If eigen runs, save operator norm.
            OpNorm[lambda_ind]  <- max(abs(eigvals)) 

        } else if (typeof(eigvals) == "character"){
            
            ## If eigen fails, save inf and index of error.
            OpNorm[lambda_ind]  <- Inf
            bug_ind[lambda_ind] <- TRUE
        }
    }   

    opt_ind <- which.min(OpNorm[1:n_lambda])
    output <- list(
        opt_lambda = lambda_range[opt_ind],
        opt_ind = opt_ind)
    return(output)
    
}


#################################################
#################################################
## matrix_threshold_perc:
##   Function that calculates the thresholding 
##   parameter for thresholded covariance
##   matrix with a selected percentage of entries to
##   drop.
##
##  INPUTS:
##      mat     : matrix to threshold.
##      perc    : proportion of entries to drop.
##                 Value in [0,1].
##
##  OUTPUTS: 
##      output  : numerical. Threshold value that 
##                  drops perc of the variables.
##
matrix_threshold_perc <- function(mat, perc = 0.05, ...) { ## Perc = percentage of entries to drop
    
    ## Selecting non-zero entries.
    mat_entries <- abs(mat[upper.tri(mat, diag = FALSE)])
    output      <- quantile(mat_entries, c(perc))
    
    return(output)
}


#################################################
#################################################
#################################################
## Standardized functions for %>% notation
## Start here:
#################################################
#################################################
#################################################


#################################################
#################################################
## sta_thresholding_BL2008:
##   Function that calculates thresholded covariance
##   matrix following thresholding method proposed
##   by Bickel & Levina (2008) paper, which selects
##   tuning parameter minimizing the resampling
##   Frobenius Norm.
##
##  INPUTS:
##      X        : data matrix.
##      mat_type : if "cor", applies to correlations
##                  If "cov", applies to covariance.
##      mat      : covariance/correlation matrix estimate.
##      var_inds : Subset of variables to apply 
##                  thresholding to.
##      true_mat : ground truth matrix.
##
##  OUTPUTS: of the following objects
##      X        : numeric matrix. Same as input.
##      mat_type : character. Same as input.
##      mat      : thresholded covariance/correlation matrix.
##                  Only variables in var_inds are thresholded.
##      var_inds : numeric vector. Same as input.
##

sta_thresholding_BL2008 <- function(X, mat_type, mat, var_inds, true_mat = NULL, ...) {

    p               <- ncol(X)
    n               <- nrow(X)
    mat_red         <- mat[var_inds, var_inds]
    X_red           <- X[, var_inds]
    lambda_range    <- lambda_range_selection(mat_red)
    
    ## Threshold reduced matrix:
    lambda_opt      <- threshold_tuning_BL2008(X_red, mat_type, lambda_range)$opt_lambda
    mat_red_thr     <- thresholding(mat_red, lambda_opt)

    ## Return to full dimension:
    mat_thr_full                        <- matrix(0, p, p)
    mat_thr_full[var_inds, var_inds]    <- mat_red_thr

    output <- list(
        X           = X,
        mat_type    = mat_type,
        mat         = mat_thr_full,
        var_inds    = var_inds
    )
    return(output)
}


#################################################
#################################################
## sta_thresholding_opnorm:
##   Function that calculates thresholded covariance
##   matrix following thresholding method proposed
##   by Bickel & Levina (2008) paper, selecting the
##   tuning parameter minimizing the resampling
##   Operator Norm.
##
##  INPUTS:
##      X        : data matrix.
##      mat_type : if "cor", applies to correlations
##                  If "cov", applies to covariance.
##      mat      : covariance/correlation matrix estimate.
##      var_inds : Subset of variables to apply 
##                  thresholding to.
##      true_mat : ground truth matrix.
##
##  OUTPUTS: of the following objects
##      X        : numeric matrix. Same as input.
##      mat_type : character. Same as input.
##      mat      : thresholded covariance/correlation matrix.
##                  Only variables in var_inds are thresholded.
##      var_inds : numeric vector. Same as input.
##
sta_thresholding_opnorm <- function(X, mat_type, mat, var_inds, true_mat = NULL, ...) {

    p               <- ncol(X)
    n               <- nrow(X)
    mat_red         <- mat[var_inds, var_inds]
    X_red           <- X[, var_inds]
    lambda_range    <- lambda_range_selection(mat_red)
    
    ## Threshold reduced matrix:
    lambda_opt      <- threshold_tuning_opnorm(X_red, mat_type, lambda_range)$opt_lambda
    mat_red_thr     <- thresholding(mat_red, lambda_opt)

    ## Return to full dimension:
    mat_thr_full                        <- matrix(0, p, p)
    mat_thr_full[var_inds, var_inds]    <- mat_red_thr

    output <- list(
        X           = X,
        mat_type    = mat_type,
        mat         = mat_thr_full,
        var_inds    = var_inds
    )
    return(output)
}


#################################################
#################################################
## sta_thresholding_oracle:
##   Function that calculates thresholded covariance
##   matrix with the oracle optimal threshold 
##   parameter.
##
##  INPUTS:
##      X        : data matrix.
##      mat_type : if "cor", applies to correlations
##                  If "cov", applies to covariance.
##      mat      : covariance/correlation matrix estimate.
##      var_inds : Subset of variables to apply 
##                  thresholding to.
##      true_mat : ground truth matrix.
##
##  OUTPUTS: of the following objects
##      X        : numeric matrix. Same as input.
##      mat_type : character. Same as input.
##      mat      : thresholded covariance/correlation matrix.
##                  Only variables in var_inds are thresholded.
##      var_inds : numeric vector. Same as input.
##
sta_thresholding_oracle <- function(X, mat_type, mat, var_inds, true_mat, ...) {

    p               <- ncol(X)
    n               <- nrow(X)
    mat_red         <- mat[var_inds, var_inds]
    X_red           <- X[, var_inds]
    lambda_range    <- lambda_range_selection(mat_red)
    
    ## Threshold reduced matrix:
    lambda_opt      <- matrix_threshold_oracle(X, mat_type, lambda_range, M = NULL, true_mat)$opt_lambda
    mat_red_thr     <- thresholding(mat_red, lambda_opt)

    ## Return to full dimension:
    mat_thr_full                        <- matrix(0, p, p)
    mat_thr_full[var_inds, var_inds]    <- mat_red_thr

    output <- list(
        X           = X,
        mat_type    = mat_type,
        mat         = mat_thr_full,
        var_inds    = var_inds
    )
    return(output)
}


#################################################
#################################################
## sta_thresholding_perc:
##   Calculates a covariance matrix estimator
##   that drops all the lowest entries up to a 
##   given percentage.
##
##  INPUTS:
##      X        : data matrix.
##      mat_type : if "cor", applies to correlations
##                  If "cov", applies to covariance.
##      mat      : covariance/correlation matrix estimate.
##      var_inds : Subset of variables to apply 
##                  thresholding to.
##      true_mat : ground truth matrix.
##      perc     : percentage of entries to drop.
##
##  OUTPUTS: of the following objects
##      X        : numeric matrix. Same as input.
##      mat_type : character. Same as input.
##      mat      : thresholded covariance/correlation matrix.
##                  Only variables in var_inds are thresholded.
##      var_inds : numeric vector. Same as input.
##
sta_thresholding_perc <- function(X, mat_type, mat, var_inds, true_mat, perc = 0.05,...) {

    p               <- ncol(X)
    n               <- nrow(X)
    mat_red         <- mat[var_inds, var_inds]
    X_red           <- X[, var_inds]
    
    ## Threshold reduced matrix:
    lambda_opt      <- matrix_threshold_perc(mat_red, perc)
    mat_red_thr     <- thresholding(mat_red, lambda_opt)

    ## Return to full dimension:
    mat_thr_full                        <- matrix(0, p, p)
    mat_thr_full[var_inds, var_inds]    <- mat_red_thr

    output <- list(
        X           = X,
        mat_type    = mat_type,
        mat         = mat_thr_full,
        var_inds    = var_inds
    )
    return(output)
}
