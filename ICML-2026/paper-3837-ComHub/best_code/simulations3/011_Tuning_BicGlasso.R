
##############################################
## BIC function for HWGLASSO and GLASSO:
original <- TRUE


#################################################
#################################################
## .BICemp:
##   Bayesian Information criteria for GGM estimation
##    as in Gao et al. (2012)
##
##  INPUTS:
##      mat       : sample covariance/correlation matrix.
##      inv_est   : estimated precision matrix/inverse correlation.
##      p         : total dimension.
##      n         : sample size.
##      threshold : degree count in BIC only considers 
##                    absolute entries over threshold.
##
##  OUTPUTS: 
##      ridge_eig : numeric vector. Shifted eigenvalues
##                   of the matrix.
##
.BICemp <- function(mat, inv_est, p, n, threshold = 0){
  ## After symmetrization, there is a legitimate minimizer. Consider this...
  inv_est_symm  <- (inv_est + t(inv_est)) / 2 
  
  .eig          <- eigen(inv_est_symm)
  .eig_val      <- Re(.eig$values)
  # print(paste0("No. of negative eigenvals: ",sum(.eig_val <= 0)))
  # print(paste0("No. of complex eigenvals: ",sum(Im(.eig_val) != 0)))
  .eig_val[.eig_val <= 0] <- 1e-5
  
  .bic <- 0
  .bic <- .bic - n * sum( log(.eig_val) )
  .bic <- .bic + n * Trace(inv_est %*% mat)
  
  .edgenum <- sum(abs(inv_est[upper.tri(inv_est, FALSE)]) > threshold)
  
  .bic <- .bic + log(n) * .edgenum
  return(.bic)
}
