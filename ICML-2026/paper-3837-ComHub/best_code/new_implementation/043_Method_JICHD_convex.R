

K <- 3
omega <- rep(1/K, K)


x = seq(0.0001, 2, length.out = 100)
plot(x, x * log(x))


Trace <- function(M) {
  return(sum(diag(M)))
}


####################################################
####################################################
## log_mat
##  
##  INPUT
##    M     : symmetric positive definite matrix.
##
##  OUTPUT:
##    logM  : logarithm matrix = U %*% log(D) %*% t(U).
##
log_mat <- function(M) {
  
  eigM <- eigen(M)
  logd <- diag(log(eigM$values))
  logM <- (eigM$vectors) %*% logd %*% t(eigM$vectors)

  return(logM)  
}


####################################################
####################################################
## Mdist :
##  Bergman distance for positive-definite matrices.
##  
##  INPUT
##    M1     : symmetric positive definite matrix.
##    M2     : symmetric positive definite matrix.
##
##  OUTPUT:
##    dM  : Bergman distance Tr(M1 * (log(M1) - log(M2)))
##
Mdist <- function(M1, M2) {
  return(Trace(M1 %*% (log_mat(M1) - log_mat(M2))))
}


####################################################
####################################################
## Wdist :
##  Bergman distance for positive vectors.
##  
##  INPUT
##    w1     : positive vector.
##    w2     : positive vector.
##
##  OUTPUT:
##    dw  : Bergman distance sum(w1 * (log(w1) - log(w2))).
##
Wdist <- function(w1, w2) {
  return(sum(w1 * (log(w1) - log(w2))))
}

Jdist <- function(v1, v2, a, b) {
  Md <- Mdist(v1$M, v2$M)
  Wd <- Wdist(v1$w, v2$w)
  dM <- a * Md + b * Wd
  return(dM)
}

####################################################
####################################################
## update_w :
##  Bergman distance for positive vectors.
##  
##  INPUT
##
##  OUTPUT:
##
update_w <- function(sigma_list, vt, vt_bar, a, b, eta) {
  
  K <- length(sigma_list)
  Mbar_list <- lapply(1:K, function(k, M) return(M),
                      M = vt_bar$M)
  wbar_list <- lapply(1:K, function(k, w) return(w[k]),
                      w = vt_bar$w)
  w_list <- mapply(
    function(sigma, w, Mbar, a, b, eta) {
      w_bn <- w * exp( Trace(sigma %*% Mbar)  * eta/b ) ## Correct if necessary
    },
    sigma = sigma_list,
    w = wbar_list,
    Mbar = Mbar_list,
    b = rep(b, K), 
    eta = rep(eta, K)) %>% unlist()
  
  w_update <- w_list / sum(w_list)
  return(w_update)
}

#########################
#########################
#########################
example <- FALSE

if (example) {
  
  library(magrittr)
  
  K <- 3
  p <- 10
  w <- rep(1/K, K)
  M <- diag(p) * K / p
  v0 <- list(w = w, M = M)
  
  sigma_list <- list(diag(p), diag(p), diag(p))
  
    
  update_w(sigma_list, v0, v0, 1, 1, 1)
  
  
}



####################################################
####################################################
## update_M :
##  
##  INPUT
##
##  OUTPUT:
##
update_M <- function(sigma_list, vt, vt_bar, a, b, eta) {
  
  K <- length(sigma_list)
  p <- ncol(sigma_list[[1]])
  log_M  <- log_mat(vt$M)
  print(log_M)
  
  sigma_wmean <- matrix(0, p, p)
  for (k in 1:K) 
    print(sigma_list[[k]])
    sigma_wmean <- sigma_wmean + vt_bar$w[k] * sigma_list[[k]] / K
  print(sigma_wmean)
  
  aggr_M <- (eta / a) * sigma_wmean + log_M
  U_bar <- eigen(aggr_M)$vectors
  L_bar <- eigen(aggr_M)$values
  
  
  L_update <- ev_hat(L_bar, 
                     length(sigma_list),
                     nrow(sigma_list[[1]]))
  
  M_update <- U_bar %*% diag(L_update) %*% t(U_bar)
  
  return(M_update)
  
}

####################################################
####################################################
## ev_hat :
##  Generates the sequence of update-eigenvalues of 
##  the matrix M(t+1)_bar and M(t+1).
##  
##  INPUT
##    L_bar : Eigenvalues and eigenvectors of the matrix
##            aggr_M.
##    K     : Number of populations of interest. 
##    d     : dimension of the data.
##
##  OUTPUT
##    
##
ev_hat <- function(L_bar, K, d) {
  
  lower <- log(K / d) - 1
  upper <- - min(L_bar)
  
  
  nu_range <- seq(lower, upper, length.out = 100)
  vals <- lapply(
    nu_range,
    function(nu, lambda) {
      exp(lambda + nu) %>%
      {cbind(.,1)} %>%
        apply(1, min) %>%
        sum() %>%
        return()
    },
    lambda = L_bar
  ) %>% unlist()

  min_ind <- which.min(abs(K - vals))
  nu_opt <- nu_range[min_ind]
  plot(nu_range, vals)
  print(nu_opt)
  L_update <- exp(L_bar + nu_opt) %>%
    {cbind(.,1)} %>%
    apply(1, min) %>%
    unlist()
  
  print(L_update)
  
  return(L_update)
  
}

#########################
#########################
#########################
example <- FALSE

if (example) {
  
  library(magrittr)
  
  K <- 3
  p <- 10
  w <- rep(1/K, K)
  M <- diag(p) * K / p
  v0 <- list(w = w, M = M)
  
  Sigma1 <- diag(c(10,5,5,rep(1, p - K)))
  Sigma2 <- diag(c(5,10,5,rep(1, p - K)))
  Sigma3 <- diag(c(5,5,10,rep(1, p - K)))
  sigma_list <- list(Sigma1, Sigma2, Sigma3)
  
  
  w <- update_w(sigma_list, v0, v0, 1, 1, 1)
  M <- update_M(sigma_list, v0, v0, 1, 1, 1)
  v0 <- list(w = w, M = M)
  w <- update_w(sigma_list, v0, v0, 1, 1, 1)
  M <- update_M(sigma_list, v0, v0, 1, 1, 1)
  v0 <- list(w = w, M = M)
  w <- update_w(sigma_list, v0, v0, 1, 1, 1)
  M <- update_M(sigma_list, v0, v0, 1, 1, 1)
  v0 <- list(w = w, M = M)
  w <- update_w(sigma_list, v0, v0, 1, 1, 1)
  M <- update_M(sigma_list, v0, v0, 1, 1, 1)
  v0 <- list(w = w, M = M)
  
  library(plot.matrix)
  plot(M)
  plot(density(M))
}


  
  
  