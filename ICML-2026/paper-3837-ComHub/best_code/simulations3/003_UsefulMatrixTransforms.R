##########################################################
##########################################################
##########################################################
##
## This document contains two types of functions:
##
##  1) Functions to move around PC, COV, CORR and IC.
##
##  2) Trace function.
##
##  3) Function that generates random orthonormal frames.
##
##########################################################
##########################################################
##########################################################

examples = FALSE

#################################################
#################################################
## .PMtoIC: 
##    Auxiliary function. Given a precision
##    matrix Theta, calculates its corresponding
##    correlation matrix.
##
##  INPUTS
##    Theta     : pxp precision matrix.
##
##  OUTPUT
##    .Rho  : pxp correlation matrix 
##                  associated with Theta.
##
.PMtoCOR = function(theta){
  .sigma = solve(theta)
  .diag = diag(.sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% .sigma %*% diag(.diaginvsq)
  
  return(.Rho)
}

#################################################
#################################################
## .PMtoIC: 
##    Auxiliary function. Given a precision
##    matrix Theta, calculates its corresponding
##    inverse correlation matrix.
##
##  INPUTS
##    Theta     : pxp precision matrix.
##
##  OUTPUT
##    .inv_cor  : pxp inverse correlation matrix 
##                  associated with Theta.
##
.PMtoIC = function(theta){
  .sigma = solve(theta)
  .diag = diag(.sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% .sigma %*% diag(.diaginvsq)
  .inv_cor = solve(.Rho)
  
  return(.inv_cor)
}

#################################################
#################################################
## .COVtoIC:
##    Auxiliary function. Given a covariance
##    matrix Sigma, calculates its corresponding
##    inverse correlation matrix.
##
##  INPUTS
##    Sigma     : pxp covariance matrix.
##
##  OUTPUT
##    .inv_cor  : pxp inverse correlation matrix 
##                  associated with Sigma.
##
.COVtoIC = function(sigma){
  .diag = diag(sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% sigma %*% diag(.diaginvsq)
  .inv_cor = solve(.Rho)
  
  return(.inv_cor)
}

#################################################
#################################################
## .COVtoCOR:
##    Auxiliary function. Given a covariance
##    matrix Sigma, calculates its corresponding
##    inverse correlation matrix.
##
##  INPUTS
##    Sigma : pxp covariance matrix.
##
##  OUTPUT
##    .rho  : pxp inverse correlation matrix 
##                  associated with Sigma.
##
.COVtoCOR = function(sigma){
  .diag = diag(sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% sigma %*% diag(.diaginvsq)
  return(.Rho)
}




#################################################
#################################################
## .rand.orthonormal
##  Function that generates a random set of ndir orthonormal
##  vectors in dimension p.
##
##  INPUTS:
##  p     : total dimension.
##  ndir  : number of directiors, ndir <= p.
##
##  OUTPUTS:
##  u     : p x ndir matrix with orthonormal columns.
##
.rand.orthonormal = function(p, ndir){
  .X = matrix(rnorm(p * ndir), nrow = p, ncol = ndir)

  u = svd(.X)$u

  return(u)
}

#################################################
#################################################
## Trace: 
##    Trace of a square matrix.
##
##  INPUTS
##    M : square matrix.
##
##  OUTPUT
##    .tr  : trace of M.
##
Trace = function(M){
  .dim = dim(M)
  if(.dim[1]!= .dim[2]){
    print("Error on trace dimensions: must provide a square matrix.")
    return(-1)
  }
  
  .tr = sum(diag(M))
  return(.tr)
  
}

