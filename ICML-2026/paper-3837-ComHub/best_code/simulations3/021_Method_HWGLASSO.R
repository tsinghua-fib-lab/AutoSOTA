
#################################################
#################################################
## BIChwglasso:
##  This function computes the HW-GLASSO estimator
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
##    OUTPUT        : list of GLASSO outputs for tun. pars. in rho.
##    BIC           : vector of BIC values for tun. par. in rho.
##    optimal.index : index corresponding to optimal rho.
##    optimal.rho   : optimal tuning parameter value. 
##    optimal.model : object glasso trained with optimal.rho
##    total.time    : total time for training.
##
BIChwglasso <- function(mat, rho, p, n,
                       threshold = 0, maxit = 200,
                       penalize.diagonal = TRUE){
  
  ## Calculate weighting matrix:
  .inv_mat <- solve(mat + 0.1 * diag(p))
  .W  <- matrix(rep(0, p * p), ncol = p)

  for (.i in (2:p)) {
    for(.j in (1:(.i-1))) {
      .a  <- abs(.inv_mat[.i, .j])
      .ai <- sum(abs(.inv_mat[.i, -.i])) 
      .aj <- sum(abs(.inv_mat[.j, -.j])) 
      
      .W[.i, .j] <- (.i != .j) / (.a * .ai * .aj)
    }
  }
  .W <- .W + t(.W)
  
  ## Fit weigthed GLASSO models:
  .rholength = length(rho)
  .BIC.HWGL = rep(NA, .rholength)
  .OUTPUT.HWGL = list()
  
  .start_time = Sys.time()
  for(.i in 1:.rholength){
    if(.i == 1){
      .OUTPUT.HWGL[[.i]] <- glasso(
        s = mat, rho = rho[.i]*.W,
        nobs = n, zero = NULL,
        thr = 1.0e-4, maxit = maxit,  approx = FALSE,
        penalize.diagonal = penalize.diagonal,
        start = "cold", w.init = NULL, wi.init = NULL,
        trace = FALSE)

    } else {
      .OUTPUT.HWGL[[.i]] <- glasso(
        s = mat, rho = rho[.i]*.W,
        nobs = n, zero = NULL, 
        thr = 1.0e-4, maxit = maxit,  approx = FALSE,
        penalize.diagonal = penalize.diagonal, 
        start = "warm",
        w.init  = .OUTPUT.HWGL[[.i-1]]$w, 
        wi.init = .OUTPUT.HWGL[[.i-1]]$wi,
        trace = FALSE)

    }
    .BIC.HWGL[.i] <- .BICemp(
      mat = mat, 
      inv_est = .OUTPUT.HWGL[[.i]]$wi, 
      n = n, p = p, threshold = threshold)

  }
  ## Select optimal model:
  .end_time = Sys.time()
  .total_time_HWGL <- .end_time - .start_time
  
  ## Return outputs:
  .OUTPUT = list(OUTPUT = .OUTPUT.HWGL, 
                 BIC = .BIC.HWGL, 
                 optimal.index = which.min(.BIC.HWGL),
                 optimal.rho = rho[which.min(.BIC.HWGL)],
                 optimal.model = .OUTPUT.HWGL[[which.min(.BIC.HWGL)]],
                 total.time = .total_time_HWGL)
  return(.OUTPUT)
}

