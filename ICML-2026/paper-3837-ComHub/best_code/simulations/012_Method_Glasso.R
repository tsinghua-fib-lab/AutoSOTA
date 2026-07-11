
#################################################
#################################################
## BICglasso:
##  This function computes the GLASSO estimator
##  for the given range of tuning parameters, 
##  and then compute the BIC of each estimator.
##  Outputs the fit with optimal BIC Tun. parameter.
##
##  INPUTS:
##      mat       : sample covariance/correlation matrix.
##      rho       : tuning parameter range vector.
##      p         : total dimension.
##      n         : sample size.
##      threshold : degree count in BIC only considers 
##                    absolute entries over threshold.
##      maxit     : Maximum number of iterations of GLASSO.
##      penalize.diagonal :
##                    if FALSE, off-diagonal L1 penalty.
##
##  OUTPUTS: 
##    OUTPUT        : list of GLASSO outputs for tun. par. rho.
##    BIC           : vector of BIC values for tun. par. rho.
##    optimal.index : index corresponding to optimal rho.
##    optimal.rho   : optimal tuning parameter value. 
##    optimal.model : object glasso trained with optimal.rho
##    total.time    : total time for training.
##
BICglasso <- function(
  mat, rho, p, n, threshold = 0, maxit = 200, 
  penalize.diagonal = FALSE) {

  .rholength <- length(rho)
  .BIC.GL <- rep(NA, .rholength)
  .OUTPUT.GL <- list()
  
  .start_time <- Sys.time()

  for(.i in 1:.rholength){
    
    if(.i == 1){
      .OUTPUT.GL[[.i]] <- glasso(
        s = mat, rho = rho[.i],
        nobs = n, zero = NULL,
        thr = 1.0e-4, maxit = maxit,  approx = FALSE, 
        penalize.diagonal = penalize.diagonal,
        start = "cold", w.init = NULL, wi.init = NULL,
        trace = FALSE)
      
    } else {
      .OUTPUT.GL[[.i]] <- glasso(
        s = mat, rho = rho[.i],
        nobs = n, zero = NULL,
        thr = 1.0e-4, maxit = maxit,  approx = FALSE,
        penalize.diagonal = penalize.diagonal,
        start = "warm",
        w.init  = .OUTPUT.GL[[.i-1]]$w,
        wi.init = .OUTPUT.GL[[.i-1]]$wi,
        trace = FALSE)
      
    }
    
    .BIC.GL[.i] <- .BICemp(
      mat = mat,
      inv_est = .OUTPUT.GL[[.i]]$wi,
      n = n, p = p, threshold = threshold)
  }
  .end_time <- Sys.time()
  .total_time_GL <- .end_time - .start_time
  
  .output <- list(
    OUTPUT = .OUTPUT.GL,
    BIC = .BIC.GL,
    optimal.index = which.min(.BIC.GL),
    optimal.rho = rho[which.min(.BIC.GL)],
    optimal.model = .OUTPUT.GL[[which.min(.BIC.GL)]],
    total.time = .total_time_GL)

  return(.output)
}

