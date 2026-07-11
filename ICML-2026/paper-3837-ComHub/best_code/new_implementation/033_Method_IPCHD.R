#################################################
#################################################
## Auxiliary Functions:
#################################################
#################################################

#################################################
#################################################
## eigen_ratios:
##   Function that, given {L_k}_k the eigenvalues of
##    a matrix L_1> L_2 >...L_p, returns {L_k + 2x}_k,
##    where x = max(0, -2 L_p). To ensure positive
##    definiteness.
##
##  INPUTS:
##      eigvals   : numeric vector. Eigenvalues
##                   of the matrix.
##
##  OUTPUTS: 
##      ridge_eig : numeric vector. Shifted eigenvalues
##                   of the matrix.
##
ridge_tuning <- function(eigvals) {
  
  d           <- length(eigvals)
  eig_min     <- min(eigvals)
  tuning_par  <- max(0, -2 * eig_min)
  ridge_eig   <- tuning_par + eigvals
  
  return(ridge_eig)
}


#####################################
## Calculating eigen-ratios
#################################################
#################################################
## eigen_ratios:
##   Function that, given the eigenvalues/vectors of
##    a matrix, returns influence measures on shat
##    eigenvectors for variables in the matrix.
##
##  INPUTS:
##      eig_mat  : output of an "eigen()" function call.
##                  Eigenvalues/vectors of cov/corr matrix.
##      shat     : Estimated number of eigenvectors
##                  used to calculate inf. measures.
##      method   : measure used for screening.
##
##  OUTPUTS: 
##      output   : numeric vector. Influence measures
##                  for variables in var_inds. Zero for
##                  the rest of the variables.
##
eigen_ratios <- function(eigvals, cutoff = NULL) {
  d         <- length(eigvals)
  if (is.null(cutoff)) cutoff <- floor(d / 2) - 1
  
  vals      <- 1 / eigvals[d:1]
  eigen_rat <- vals[1:cutoff] / vals[2:(cutoff + 1)]
  return(eigen_rat)
}


#################################################
#################################################
## eigen_ratios:
##   Function that, given {L_k / L_{k+1}}_k the 
##    eigenvalue ratios of a matrix, returns the
##    ratio index with the largest gap, or an 
##    overestimation.
##
##  INPUTS:
##      eigen_rat : numeric vector. Consecutive
##        eigenvalue ratios.
##      overst_type  : character. Type of overestimation.
##        If "true", always overestimates with d/5.
##        If "frac", overestimates with d/5.
##        If "sqrt", overestimates with sqrt(d).
##      d         : numeric. Dimension.
##
##  OUTPUTS: 
##      shat    : index of largest eigen-ratio, or
##                  overestimated value of largest ratio.
##      overest : if TRUE, overestimation was used.
optimal_gap <- function(eigen_rat, overest_type, d) {
  
  sort      <- sort(eigen_rat, decreasing = TRUE)
  max1      <- sort[1]
  max2      <- sort[2]
  shat      <- NULL
  overest   <- NULL
  
  if (is.numeric(overest_type)) {
    shat      <- overest_type
    overest   <- TRUE
    
  } else if (overest_type == "true") {
    shat      <- floor(sqrt(d))
    overest   <- TRUE
    
  } else if (max1 > 1.5 * max2) {
    shat      <- which.max(eigen_rat)
    overest   <- FALSE
    
  } else if (overest_type == "frac") {
    shat      <- floor(d / 5)
    overest   <- TRUE
    
  } else if (overest_type == "sqrt") {
    shat      <- floor(sqrt(d))
    overest   <- TRUE
    
  }
  
  output <- list(shat = shat, overest = overest)
  return(output)
}






#################################################
#################################################
## influence_measures:
##   Function that, given the eigenvalues/vectors of
##    a matrix, returns influence measures on shat
##    eigenvectors for variables in the matrix.
##
##  INPUTS:
##      eig_mat  : output of an "eigen()" function call.
##                  Eigenvalues/vectors of cov/corr matrix.
##      shat     : Estimated number of eigenvectors
##                  used to calculate inf. measures.
##      method   : measure used for screening.
##
##  OUTPUTS: 
##      output   : numeric vector. Influence measures
##                  for variables in var_inds. Zero for
##                  the rest of the variables.
##
influence_measures <- function(eig_mat, shat, var_inds) {
  
  d         <- length(eig_mat$values)
  inf_meas  <- NULL
  
  if (shat == 1) {
    inf_meas  <- eig_mat$vectors[, d]^2
    
  } else if (shat > 1) {
    vectors   <- eig_mat$vectors[, d:(d - shat + 1)]
    inf_meas  <- apply(vectors^2, MARGIN = 1, sum)
    
  }

  return(inf_meas)

}



#################################################
#################################################
## sta_ipchd:
##   Function that, given a (screened) covariance or
##    correlation matrix, returns influence measures
##    of variables in the (screened) matrix.
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
##      output   : numeric vector. Influence measures
##                  for variables in var_inds. Zero for
##                  the rest of the variables.
##
sta_ipchd <- function (
    X, mat_type, mat, var_inds, overest_type = c("true", "frac", "sqrt")) {
  
  ## Setting parameters:
  p             <- ncol(X)
  n             <- nrow(X)
  mat_red       <- mat[var_inds, var_inds]
  X_red         <- X[, var_inds]
  p_red         <- length(var_inds)
  
  
  ## Calculating optimal gap
  eig_mat_red   <- eigen(mat_red)
  ridge_eig     <- ridge_tuning(eig_mat_red$values)
  eigen_rat     <- eigen_ratios(ridge_eig)
  shat          <- optimal_gap(eigen_rat, overest_type, p_red)$shat
  
  ## Calculating influence measures
  output_red    <- influence_measures(eig_mat_red, shat, var_inds)
  output        <- rep(0, p)
  output[
    var_inds]   <- output_red
  
  return(output)
}











