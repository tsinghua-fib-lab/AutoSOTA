

#################################################
#################################################
## Variable Screening in High-Dimensional
## Covariance/Correlation Estimation
## Sanchez et al. (2024)
#################################################
#################################################

#################################################
#################################################
## screening_vars_SMZL2024:
##   Function that, given a covariance/correlation 
##    matrix, screens out variables with low 
##    columnwise max/l2/l1 norms.
##
##  INPUTS:
##      X        : data matrix.
##      mat_type : if "cor", applies to correlations
##                  If "cov", applies to covariance.
##      mat      : covariance/correlation matrix estimate.
##      var_inds : Set of variables to apply screening
##                  to. Usually, var_inds = 1:p.
##      method   : measure used for screening.
##
##  OUTPUTS: 
##      output   : numeric vector. Indexes of variables 
##                  to keep after screening. 
##
screening_vars_SMZL2024 <- function(X, mat_type, mat, var_inds, method = c("max", "l2", "l1"), ...) {
  
  p             <- ncol(X)
  n             <- nrow(X)
  T0            <- as.integer(min(p, 3 * n / 4))
  var_strength  <- NULL

  if (method == "max") {
    var_strength <- apply(
        mat - diag(diag(mat)),
        MARGIN = 1,
        function(x) max(abs(x)))

  } else if (method == "l2") {
    var_strength <- apply(
        mat - diag(diag(mat)),
        MARGIN = 1,
        function(x) sum(x^2))

  } else if (method == "l1") {
    var_strength <- apply(
        mat - diag(diag(mat)),
        MARGIN = 1,
        function(x) sum(abs(x)))

  }

  ## Choose the variables with top var_strength
  order_vars <- order(var_strength, decreasing = TRUE)

  output <- sort(order_vars[1:T0], decreasing = FALSE)
  
  return(output)
}




#################################################
#################################################
## Post selection screening function:
#################################################
#################################################

#################################################
#################################################
## sta_screened_mat:
##   Given a covariance/correlation matrix estimator,
##    calcualtes screened matrix according to 
##    desired screening method.
##
##  INPUTS:
##      X        : data matrix.
##      mat_type : if "cor", applies to correlations
##                  If "cov", applies to covariance.
##      mat      : covariance/correlation matrix estimate.
##      var_inds : Set of variables to apply screening
##                  to. Usually, var_inds = 1:p.
##      true_mat : ground truth matrix.
##      method   : measure used for screening.
##
##  OUTPUTS: of the following objects
##      X        : numeric matrix. Same as input.
##      mat_type : character. Same as input.
##      mat      : matrix of the original pxp size,
##                  with zeros in all but the row/columns
##                  in var_inds.
##      var_inds : Set of variables selected by the 
##                  thresholding method.
##
sta_screened_mat <- function (X, mat_type, mat, var_inds, true_mat = NULL, method = c("max", "l1", "l2"), ...) {
  
  p                 <- ncol(X)
  n                 <- nrow(X)
  mat_red           <- mat[var_inds, var_inds]
  X_red             <- X[, var_inds]
  p_red             <- length(var_inds)

  screened_vars     <- screening_vars_SMZL2024(X_red, mat_type, mat_red, 1:p_red, method)
  reind_scr_vars    <- var_inds[screened_vars]

  mat_red_screened  <- matrix(0, p, p)
  mat_red_screened[
    reind_scr_vars,
    reind_scr_vars] <- mat[reind_scr_vars, reind_scr_vars]
  
  output <- list(
    X           = X,
    mat_type    = mat_type,
    mat         = mat_red_screened,
    var_inds    = reind_scr_vars
  )
  
  return(output)
}
