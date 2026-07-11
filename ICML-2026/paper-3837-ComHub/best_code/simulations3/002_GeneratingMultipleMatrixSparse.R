#################################################
#################################################
#################################################
##
## In the following document, we will
## define a method for generating positive
## definite matrices with hubs, which is 
## based on first generating the eigen-
## structure.
##
#################################################
#################################################

examples = FALSE

#################################################
#################################################
## .adjmat: 
##    Auxiliary function. Generates a
##    random adjacency matrix of a network with 
##    hubs. Uses non-uniform Erdos Renyi model,
##    where hubs have higher connection
##    probability.
##
##  INPUTS
##    p     : Total dimension of pxp matrix.
##    T0     : Size of T0xT0 highly connected submatrix.
##    H1    : first set of hubs
##    H2    : second set of hubs
##    ph1   : Probability of hub connectivity in 1st set
##    ph2   : Probability of hub connectivity in 2nd set
##    pnh   : non-hub connectivity for entries in txt submatrix.
##    pneff : Probability of connection for all
##              variables outside of the t x t 
##              sumatrix.
##
##  OUTPUT
##    .A    : pxp matrix with {0,1} entries. 
##
.adjmat <- function(p, T0, H1, H2, ph1, ph2, pnh, pneff){
  .A = matrix(rep(0, p*p), ncol = p)
  
  l1 <- length(H1)
  l2 <- length(H2)

  .A[]         <- rbinom(p * p, 1, pneff)
  .A[1:T0, 1:T0] <- rbinom(T0 * T0, 1, pnh)
  if (!is.null(H2)) {
    .A[1:T0, H2]  <- rbinom(T0 * l2, 1, ph2)
    .A[H2, (1:T0)[-H2]]  <- rbinom((T0 - l2) * l2, 1, ph2)
  }
  if (!is.null(H1)) {
    .A[1:T0, H1]  <- rbinom(T0 * l1, 1, ph1)
    .A[H1, (1:T0)[-H1]]  <- rbinom((T0 - l1) * l1, 1, ph1)
  }
  .A[upper.tri(.A, TRUE)] <- 0
  
  .A = .A + t(.A)
  return(.A)
}

.adjmat_list <- function(p, K, T0, Hjoint, Hind_list, ph1, ph2, pnh, pneff) {
  
  .Hind   <- unlist(Hind_list)
  
  .Ajoint <- .adjmat(p, T0, H1 = Hjoint, H2 = .Hind, ph1 = ph1, ph2 = 0, pnh =pnh, pneff = pneff)
  .Aind_list <- list()
  for (k in 1:K) {
    .Aind_list[[k]] <- .adjmat(p, T0, H1 = Hjoint, H2 = Hind_list[[k]], ph1 = 0, ph2 = ph2, pnh = 0, pneff = 0)
  }

  output <- list(Ajoint = .Ajoint, Aind_list = .Aind_list)
  return(output)
}

.rsign <- function(n = 1) {
  return(2*rbinom(n,1,0.5)-1)
}

#################################################
#################################################
## .rsignunif:
##    Auxiliary function. Generates a 
##    random variable with distribution:
##    Unif( [-max,-min]U[min,max] )
##
##  INPUTS
##    n       : size of output vector.
##    min     : minimum absolute value of variables. 
##    max     : maximum absolute value of variables.
##
##  OUTPUT
##    .output : length n vector with entries of 
##                distribution Unif([-max,-min]U[min,max])
## 
.rsignunif <- function(n = 1 , min = 4, max = 5) {
  .cont = runif(n, min, max)
  .sign = .rsign(n)
  return(.cont*.sign )
}

#################################################
#################################################
## .rsymmmatrix: 
##    Auxiliary function. Generates a
##    random matrix with signed-uniform or normal
##    entries. The distribution of the entries 
##    depends on whether variables are hubs, 
##    highly connected or of low connection.
##
##  INPUTS
##    p       : Total dimension of pxp matrix.
##    T0      : Size of T0xT0 highly connected submatrix.
##    H1      : first set of hubs
##    H2      : second set of hubs
##    type    : whether the distribution signed-uniform
##                or zero-mean normal.
##    hmin1   : Used when type = "unif". Minimum absolute 
##                value of hub entries, in 1st set.
##    hmax1   : Used when type = "unif". Maximum absolute 
##                value of hub entries in 1st set.
##    hmin2   : Used when type = "unif". Minimum absolute 
##                value of hub entries in 2nd set.
##    hmax2   : Used when type = "unif". Maximum absolute 
##                value of hub entries in 2nd set. 
##    nhmin   : Used when type = "unif". Minimum absolute 
##                value of entries in T0xT0 submatrix 
##                not related to hub variables.
##    nhmax   : Used when type = "unif". Maximum absolute 
##                value of entries in T0xT0 submatrix 
##                not related to hub variables.
##    neffmin : Used when type = "unif". Minimum absolute 
##                value of entries outside of the 
##                T0xT0 submatrix 
##    neffmin : Used when type = "unif". Minimum absolute 
##                value of entries outside of the 
##                T0xT0 submatrix 
##    hsd1     : Used when type = "gaussian". Standard
##                deviation of normal hub entries in 1st set.
##    hsd2     : Used when type = "gaussian". Standard
##                deviation of normal hub entries in 2nd set.
##    nhsd     : Used when type = "gaussian". Standard
##                deviation of normal entries in T0xT0 
##                submatrix not related to hub variables.
##    neffsd   : Used when type = "gaussian". Standard
##                deviation of normal entries outside of 
##                the T0xT0 submatrix 
##
##  OUTPUT
##    .theta   : random pxp matrix with entrywise normal or
##                signed uniform distribution
##                depending on given parameters.     
##
.rsymmmatrix <- function(
  p, T0, H1, H2, 
  type = c("unif", "gaussian"),
  hmin1 = 0.5, hmax1 = 0.8,
  hmin2 = 0.5, hmax2 = 0.8,
  nhmin = 0.5, nhmax = 0.8,
  neffmin = 0.5, neffmax = 0.8,
  hsd1 = 1, hsd2 = 1, nhsd = 1, neffsd = 0.5){
  
  .theta = matrix(rep(0, p*p), ncol = p)
  l1 <- length(H1)
  l2 <- length(H2)

  if (type == "unif") {
    .theta[]         <- .rsignunif(p * p, min = neffmin, max = neffmax)
    .theta[1:T0, 1:T0] <- .rsignunif(T0 * T0, min = nhmin, max = nhmax)
    if (!is.null(H2)) {
      .theta[1:T0, H2]  <- .rsignunif(T0 * l2, min = hmin2, max = hmax2)
      .theta[H2, (1:T0)[-H2]]  <- .rsignunif((T0 - l2) * l2, min = hmin2, max = hmax2)
    }
    if (!is.null(H1)) {
      .theta[1:T0, H1]  <- .rsignunif(T0 * l1, min = hmin1, max = hmax1)
      .theta[H1, (1:T0)[-H1]]  <- .rsignunif((T0 - l1) * l1, min = hmin1, max = hmax1)
    }
    .theta[upper.tri(.theta, TRUE)] <- 0
  }

  if(type == "gaussian"){
    .theta[]         <- rnorm(p * p, sd = neffsd)
    .theta[1:T0, 1:T0] <- rnorm(T0 * T0, sd = nhsd)
    if (!is.null(H2)) {
      .theta[1:T0, H2]  <- rnorm(T0 * l2, sd = hsd2)
      .theta[H2, (1:T0)[-H2]]  <- rnorm((T0 - l2) * l2, sd = hsd2)
    }
    if (!is.null(H1)) {
      .theta[1:T0, H1]  <- rnorm(T0 * l1, sd = hsd1)
      .theta[H1, (1:T0)[-H1]]  <- rnorm((T0 - l1) * l1, sd = hsd1)
    }
    .theta[upper.tri(.theta, TRUE)] <- 0
  }
  .theta = .theta + t(.theta)
  return(.theta)
}

#################################################
#################################################
## .shufflemat: 
##    Auxiliary function. Shuffles the 
##    position of the hubs from the
##    first couple entries to any 
##    random entry.
##
##  INPUTS
##    p   : Total dimension of pxp matrix.
##    A   : pxp matrix.
##
##  OUTPUT
##    .M  : matrix with randomly shuffled columns/
##            rows of A.
##
.shufflemat <- function(A,p){
  .neworder = sample(1:p, p, replace = FALSE)
  .M = A[.neworder, .neworder]
  return(.M)
}

#################################################
#################################################
## .rhubmat: 
##    Auxiliary function. Generates a 
##    random sparse symmetric matrix with hubs, where 
##    the non-zero entries are either signed-uniform 
##    or Gaussian. 
##    The matrix which contains a  T0xT0 matrix 
##    containing hub variables and high connectivity. 
##    The entries outside the T0xT0 submatrix are 
##    very sparse.
##
##  INPUTS
##    p       : Total dimension of pxp matrix.
##    T0      : Size of T0xT0 highly connected submatrix.
##    H1      : first set of hubs
##    H2      : second set of hubs
##    ph1     : Probability of hub connectivity for 1st set
##    ph1     : Probability of hub connectivity for 2nd set
##    pnh     : Probability of connection for variables 
##                in the high-connection T0xT0 submatrix.
##    pneff   : Probability of connection for all
##                variables outside of the T0xT0 
##                sumatrix.
##    shuffle : If true, shuffles rows/columns to a random
##                position.
##    type    : whether the distribution signed-uniform
##                or zero-mean normal.
##    hmin1   : Used when type = "unif". Minimum absolute 
##                value of hub entries in 1st set.
##    hmax1   : Used when type = "unif". Maximum absolute 
##                value of hub entries in 1st set.
##    hmin2   : Used when type = "unif". Minimum absolute 
##                value of hub entries in 2nd set.
##    hmax2   : Used when type = "unif". Maximum absolute 
##                value of hub entries in 2nd set.
##    nhmin   : Used when type = "unif". Minimum absolute 
##                value of entries in T0xT0 submatrix 
##                not related to hub variables.
##    nhmax   : Used when type = "unif". Maximum absolute 
##                value of entries in T0xT0 submatrix 
##                not related to hub variables.
##    neffmin : Used when type = "unif". Minimum absolute 
##                value of entries outside of the 
##                T0xT0 submatrix 
##    neffmin : Used when type = "unif". Minimum absolute 
##                value of entries outside of the 
##                T0xT0 submatrix 
##    hsd1    : Used when type = "gaussian". Standard
##                deviation of normal hub entries, 1st set.
##    hsd2    : Used when type = "gaussian". Standard
##                deviation of normal hub entries, 2nd set.
##    nhsd    : Used when type = "gaussian". Standard
##                deviation of normal entries in T0xT0 
##                submatrix not related to hub variables. 
##    neffsd  : Used when type = "gaussian". Standard
##                deviation of normal entries outside of 
##                the T0xT0 submatrix 
##
##  OUTPUT
##    .theta  : random sparse pxp matrix with hubs, and a T0xT0
##                highly connected matrix. Non-zero entries 
##                are either normally distributed or
##                signed uniform distribution depending on 
##                given parameters.     
##
.rhubmat <- function(
  p, T0, H1, H2, 
  ph1, ph2, pnh, pneff, 
  shuffle = FALSE, type = c("unif", "gaussian"),
  hmin1 = 0.5, hmax1 = 0.8,
  hmin2 = 0.5, hmax2 = 0.8,
  nhmin = 0.5, nhmax = 0.8,
  neffmin = 0.5, neffmax = 0.8,
  hsd1 = 5, hsd2 = 5, nhsd = 1, neffsd = 0.5){
  
  .A = .adjmat(p, T0, H1, H2, ph1, ph2, pnh, pneff)
  .theta = .A * .rsymmmatrix(
    p, T0, H1, H2,
    type = type,
    hmin1 = hmin1, hmax1 = hmax1,
    hmin2 = hmin2, hmax2 = hmax2,
    nhmin = nhmin, nhmax = nhmax,
    neffmin = neffmin, neffmax = neffmax,
    hsd1 = hsd1, hsd2 = hsd2, nhsd = nhsd, neffsd = nseffsd)

  if( shuffle ){
    .theta = .shufflemat(.theta, p)
  }
  return( .theta )
}




.rhubmat_list <- function(
    p, T0, K, Hjoint, Hind_list, 
    ph1, ph2, pnh, pneff, 
    shuffle = FALSE, type = c("unif", "gaussian"),
    hmin1 = 0.5, hmax1 = 0.8,
    hmin2 = 0.5, hmax2 = 0.8,
    nhmin = 0.5, nhmax = 0.8,
    neffmin = 0.5, neffmax = 0.8,
    hsd1 = 5, hsd2 = 5, nhsd = 1, neffsd = 0.5){
  
  A <- .adjmat_list(p, K, T0, Hjoint, Hind_list, ph1, ph2, pnh, pneff)
  Hind   <- unlist(Hind_list)
  theta = .rsymmmatrix(
    p, T0, Hjoint, Hind,
    type = type,
    hmin1 = hmin1, hmax1 = hmax1,
    hmin2 = hmin2, hmax2 = hmax2,
    nhmin = nhmin, nhmax = nhmax,
    neffmin = neffmin, neffmax = neffmax,
    hsd1 = hsd1, hsd2 = hsd2, nhsd = nhsd, neffsd = nseffsd)
  
  theta_list <- list()
  for (k in 1:K) {
    theta_list[[k]] = theta * (A$Ajoint + A$Aind_list[[k]]) 
  }
  
  return(theta_list)
}

## With this functions, we are able to select the 
## network structure of the precision matrix. Now,
## we have to design methods that select the value
## of the diagonal.

######################
######################
### Example:
if(examples){

  adjmat = .adjmat(p = 20, t = 10, 
                   H1 = c(10), H2 = c(4,5), 
                   ph1 = 1, ph2 = 0.5, pnh = 0)
  par(oma = c(0,0,0,2))
  plot(adjmat) 
  
  ## We generate two conventional matrices with
  ## two types of hubs. 
  rhubmat = .rhubmat(p = 20, t = 10, 
                     H1 = c(5), H2 = c(1,2), 
                     ph1 = 1, ph2 = 0.5, pnh = 1, 
                     shuffle = FALSE,
                     type = "unif", 
                     hmin1 = 100, hmax1 = 101, 
                     hmin2 = 50, hmax2 = 51, 
                     nhmin = 1, nhmax = 2)
  plot(rhubmat)
  rhubmat = .rhubmat(p = 10, t = 8, 
                     H1 = c(8), H2 = c(1,2), 
                     ph1 = 1, ph2 = 0.5, pnh = 1, 
                     shuffle = FALSE,
                     type = "gaussian", 
                     hsd1 = 100, hsd2 = 100, nhsd = 1) 
  plot(abs(rhubmat))
  
  
  
  ## Lets see what happens if r1 = 0
  rhubmat = .rhubmat(p = 20, t = 10, 
                     H1 = NULL, H2 = c(1,2), 
                     ph1 = 1, ph2 = 0.5, pnh = 1, 
                     shuffle = FALSE,
                     type = "unif", 
                     hmin1 = 100, hmax1 = 101, 
                     hmin2 = 50, hmax2 = 51, 
                     nhmin = 1, nhmax = 2)
  par(mar = c(4,4,4,4))
  plot(rhubmat)

  ## Lets see what happens if r2 = 0
  rhubmat = .rhubmat(p = 20, t = 10, 
                     H1 = c(4,5), H2 = NULL, 
                     ph1 = 1, ph2 = 0.5, pnh = 1, 
                     shuffle = FALSE,
                     type = "unif", 
                     hmin1 = 100, hmax1 = 101, 
                     hmin2 = 50, hmax2 = 51, 
                     nhmin = 1, nhmax = 2)
  par(mar = c(4,4,4,4))
  plot(rhubmat)
  
  rm(adjmat, rhubmat, examples)
  
}



#################################################
#################################################
## r.sparse.pdhubmat: 
##    Generates a random sparse positive definite
##    matrix with hubs, where the non-zero entries
##    are either signed-uniform or Gaussian. 
##    The matrix which contains a  T0xT0 matrix 
##    containing hub variables and high connectivity. 
##    The entries outside the T0xT0 submatrix are 
##    very sparse.
##
##  INPUTS
##    p       : Total dimension of pxp matrix.
##    T0      : Size of T0xT0 highly connected submatrix.
##    r       : number of hubs.
##    ph      : Probability of hub connectivity 
##    pnh     : Probability of connection for variables 
##                in the high-connection T0xT0 submatrix.
##    pneff   : Probability of connection for all
##                variables outside of the T0xT0 
##                sumatrix.
##    diagonal_shift : 
##                Size of minimum eigenvalue of the 
##                output matrix.
##    shuffle : If true, shuffles rows/columns to a random
##                position.
##    type    : whether the distribution signed-uniform
##                or zero-mean normal.
##    hmin1   : Used when type = "unif". Minimum absolute 
##                value of hub entries in 1st set.
##    hmax1   : Used when type = "unif". Maximum absolute 
##                value of hub entries in 1st set.
##    hmin2   : Used when type = "unif". Minimum absolute 
##                value of hub entries in 2nd set.
##    hmax2   : Used when type = "unif". Maximum absolute 
##                value of hub entries in 2nd set.
##    nhmin   : Used when type = "unif". Minimum absolute 
##                value of entries in T0xT0 submatrix 
##                not related to hub variables.
##    nhmax   : Used when type = "unif". Maximum absolute 
##                value of entries in T0xT0 submatrix 
##                not related to hub variables.
##    neffmin : Used when type = "unif". Minimum absolute 
##                value of entries outside of the 
##                T0xT0 submatrix 
##    neffmin : Used when type = "unif". Minimum absolute 
##                value of entries outside of the 
##                T0xT0 submatrix 
##    hsd1    : Used when type = "gaussian". Standard
##                deviation of normal hub entries, 1st set.
##    hsd2    : Used when type = "gaussian". Standard
##                deviation of normal hub entries, 2nd set.
##    nhsd    : Used when type = "gaussian". Standard
##                deviation of normal entries in T0xT0 
##                submatrix not related to hub variables. 
##    neffsd  : Used when type = "gaussian". Standard
##                deviation of normal entries outside of 
##                the T0xT0 submatrix 
##
##  OUTPUT
##    .theta  : random sparse and positive definite pxp matrix 
##                with hubs, and a T0xT0
##                highly connected matrix. Non-zero entries 
##                are either normally distributed or
##                signed uniform distribution depending on 
##                given parameters.     
##
r.sparse.pdhubmat <- function(
  p, T0, H1, H2, ph1, ph2, pnh, pneff,
  diagonal_shift = 1, shuffle = FALSE, type = c("unif", "gaussian"),
  hmin1 = 0.5, hmax1 = 0.8,
  hmin2 = 0.5, hmax2 = 0.8,
  nhmin = 0.5, nhmax = 0.8,
  neffmin = 0.5, neffmax = 0.8,
  hsd1 = 1, hsd2 = 1, nhsd = 1, neffsd = 0.5,
  verbose = FALSE){
  .pm = .rhubmat(
    p, T0, H1, H2, ph1, ph2, pnh, pneff, 
    shuffle = shuffle, type = type,
    hmin1 = hmin1, hmax1 = hmax1,
    hmin2 = hmin2, hmax2 = hmax2,
    nhmin = nhmin, nhmax = nhmax,
    neffmin = neffmin, neffmax = neffmax,
    hsd1 = hsd1, hsd2 = hsd2, nhsd = nhsd, neffsd = neffsd)

  .lambda = eigen(.pm)$value[p]
  .I =  (-.lambda + diagonal_shift) * diag(p)
  
  if(verbose){
    print(paste("The dimension of PM is:",dim(.pm)[1],"x", dim(.pm)[2]))
    print(paste("The dimension of .I is:",dim(.I)[1],"x", dim(.I)[2]))
  }
  
  .pm = .pm + .I
  return(.pm)
}






r.sparse.pdhubmat_list <- function(
    p, T0, K, Hjoint, Hind_list, ph1, ph2, pnh, pneff,
    diagonal_shift = 1, shuffle = FALSE, type = c("unif", "gaussian"),
    hmin1 = 0.5, hmax1 = 0.8,
    hmin2 = 0.5, hmax2 = 0.8,
    nhmin = 0.5, nhmax = 0.8,
    neffmin = 0.5, neffmax = 0.8,
    hsd1 = 1, hsd2 = 1, nhsd = 1, neffsd = 0.5,
    verbose = FALSE){
  
  theta_list <- .rhubmat_list(
    p, T0, K, Hjoint, Hind_list, 
    ph1, ph2, pnh, pneff, 
    shuffle, type,
    hmin1, hmax1,
    hmin2, hmax2,
    nhmin, nhmax,
    neffmin, neffmax,
    hsd1, hsd2, nhsd, neffsd)
    
    lambda <- lapply(
      theta_list, 
      function(theta) return(min(eigen(theta)$values)) )  %>% 
      unlist() %>% 
      min()
  
  .I <- (-lambda + diagonal_shift) * diag(p)
  pm_list <- list()
  for (k in 1:K) {
    pm_list[[k]] <- theta_list[[k]] + .I
  }
  
  return(pm_list)
}





######################
######################
### Example:
if(examples){
  Theta = r.sparse.pdhubmat(
    p = 20, T0 = 10, H1 = c(1), H2 = c(5), 
    ph1 = 1, ph2 = 0.5, pnh = 1, pneff = 0.1,
    diagonal_shift = 1,  shuffle = FALSE, type = "unif", 
    hmin1 = 150, hmax1 = 100, 
    hmin2 = 50, hmax2 = 100, 
    nhmin = 25, nhmax = 50,
    neffmin = 0, neffmax = 25)

  plot(abs(Theta - diag(diag(Theta))))
  
  rm(Theta)
  
}

rm(examples)
