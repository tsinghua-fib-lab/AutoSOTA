#################################################
#################################################
#################################################
##
## In the following document, we introduce the
## functions that take a covariance matrix
## estimate and turn them into a hub estimation.
## We measure:
##  1) TP rate.
##  2) FP rate.
##  3) AUC
##
#################################################
#################################################



#################################################
#################################################
## .dalphavals:
##    Given a precision matrix estimation, calculates
##    column-wise L2-norms to measure continuous
##    connectivity as in (1) of our main paper.
##
##  INPUTS
##    theta     : precision matrix estimator.
##    
##  OUTPUT
##    .alpha    : column-wise L2 norms of theta.
##
.dalphavals <- function(Theta) {
  .alpha <- apply(Theta^2, MARGIN = 2, sum)
  return(.alpha)
}

#################################################
#################################################
## .nalphavals:
##    Given a precision matrix estimation, calculates
##    column-wise L2-norms to measure continuous
##    connectivity as in (1) of our main paper.
##
##  INPUTS
##    theta     : precision matrix estimator.
##    
##  OUTPUT
##    .alpha    : column-wise L2 norms of theta.
##
.nalphavals <- function(Theta) {
  .alpha <- apply((Theta - diag(diag(Theta)))^2, MARGIN = 2, sum)
  return(.alpha)
}


#################################################
#################################################
## .degrees:
##    Given a precision matrix estimation, calculates
##    column-wise L0-norms to measure discrete
##    connectivity as in a graphical model.
##
##  INPUTS
##    theta     : precision matrix estimator.
##    
##  OUTPUT
##    .degs    : column-wise L0 norms of theta.
##
.degrees <- function(Theta) {
  .degs <- apply(
    Theta, MARGIN = 2,
    function(x) sum(abs(x) > 0))
  return(.degs)
}


