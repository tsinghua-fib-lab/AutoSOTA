

##############################################
##############################################
#################################### HW.GLASSO
##############################################
##############################################
##
## The tuning parameter range can be estimated
## by using the fact that GLASSO = Diagmat
## if the penalty > max(|Sigma|).
##
##



#################################################
#################################################
## tp.hwgl:
##  Finds a tuning parameter for which the output of the
##  HWGL method is diagonal. This bounds the range
##  of values for training the HWGL method.  
##
##  INPUTS
##    X     : nxp data matrix.
##    pm    : true underlying precision matrix. Used to
##              generate sample for pretraining. Used 
##              only if X = NULL.
##    p     : number of variables.
##    n     : sample size.
##    cov   : if FALSE, pretraining on correlation matrix.
##    
##  OUTPUT:
##    .rho  : tuning parameter value for which HWGL method
##              is diagonal.
## 
tp.hwgl <- function(X = NULL, pm = NULL, p, n, cov = TRUE) {
  ##############################################
  ##############################################
  ## Step 1: Generate data/ use the given data:
  if (is.null(X)) {
    .sigma <- solve(pm)
    .X <- rmvnorm(n = n, sigma = .sigma, method = "svd")
  } else {
    .X <- X
  }
  
  ## Find covariance.
  .cov <- NULL
  .rho <- NULL
  if (cov) {
    .cov <- cov(.X)
  } else {
    .cov <- cor(.X)
  } 
  .rho <- 10^30
  
  ##############################################
  ##############################################
  ## Find weight matrix: 
  .inv_cov <- solve(.cov + 0.1 * diag(p))
  .W  <- matrix(rep(0, p * p), ncol = p)
  for (.i in (2:p)) {
    for (.j in (1:(.i-1))) {

      .a  <- abs(.inv_cov[.i, .j])
      .ai <- sum(abs(.inv_cov[.i, -.i])) 
      .aj <- sum(abs(.inv_cov[.j, -.j])) 
      
      .W[.i, .j] <- (.i != .j) / (.a * .ai * .aj)
    }
  }
  .W <- .W + t(.W)
  
  ##############################################
  ##############################################
  ## Estimate the value for which the model is 
  ## first diagonal.
  .output.hwgl <- glasso(
    s = .cov, rho = .rho*.W, nobs = n,
    maxit = 200, penalize.diagonal = FALSE)
  .is.diagonal <- is.diagonal.matrix(.output.hwgl$wi)
  
  .count <- 1
  while (!.is.diagonal & .count < 5) {
    .count <- .count + 1
    .rho <- 10^5 * .rho
    .output.hwgl <- glasso(
      s = .cov, rho = .rho*.W, nobs = n,
      maxit = 200, penalize.diagonal = FALSE)
    .is.diagonal <- is.diagonal.matrix(.output.hwgl$wi)
  }
  if (!.is.diagonal) {
    print("Warning: No diagonal/non-diagonal threshold was found: lower bound only.")
    return(.rho)
  }

  .count <- 1
  while (.is.diagonal & .count < 201) {
    ## Reduce the value of the tuning parameter
    .count <- .count + 1
    .rho <- .rho / 2
    
    ## find the new model:
    .output.hwgl <- glasso(
      s = .cov, rho =  .rho * .W, nobs = n,
      maxit = 200, penalize.diagonal = FALSE)
    .is.diagonal <- is.diagonal.matrix(.output.hwgl$wi)
    
  }   
  if (.is.diagonal) {
    print("Warning: No diagonal/non-diagonal threshold was found: upper bound only.")
    return(.rho)
  }  
  
  print(paste("The diagonal/non-diagonal treshold is:", .rho))
  return(.rho)
}

