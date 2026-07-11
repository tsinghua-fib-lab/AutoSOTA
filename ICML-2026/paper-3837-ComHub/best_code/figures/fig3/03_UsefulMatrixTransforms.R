
##########################################################
##########################################################
## Auxiliary functions useful for jumping from 
## Cov/corr/PM/IC.

.PMtoCOR = function(theta){
  .sigma = solve(theta)
  .diag = diag(.sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% .sigma %*% diag(.diaginvsq)
  
  return(.Rho)
}


.PMtoIC = function(theta){
  .sigma = solve(theta)
  .diag = diag(.sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% .sigma %*% diag(.diaginvsq)
  .inv_cor = solve(.Rho)
  
  return(.inv_cor)
}

.COVtoIC = function(sigma){
  .diag = diag(sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% sigma %*% diag(.diaginvsq)
  .inv_cor = solve(.Rho)
  
  return(.inv_cor)
}

.COVtoCOR = function(sigma){
  .diag = diag(sigma)
  .diaginvsq = 1/sqrt(.diag)
  
  .Rho = diag(.diaginvsq) %*% sigma %*% diag(.diaginvsq)
  return(.Rho)
}


.rand.orthonormal = function(p, ndir){
  .X = matrix(rnorm(p*ndir), nrow = p, ncol = ndir)
  
  .svd = svd(.X)
  
  return(.svd$u)
}


trace = function(M){
  .dim = dim(M)
  if(.dim[1]!= .dim[2]){
    print("Error on trace dimensions: must provide a square matrix.")
    return(-1)
  }
  
  .tr = sum(diag(M))
  return(.tr)
  
}
