#################################################
#################################################
##
## The Stiefel method is to find the bottom r
##  "joint eigenvectors" of Sigma = [Sigma1,...,SigmaK]
## by minimizing the objective:
##
##  V in R^(pxr) |---> max_k Trace( V^T Sigmak V )
##  
##  This method only provides eigenvectors. No
##  eigenvalue can be obtained by this method.  
##
#################################################
#################################################

source("02_GeneratingMultipleMatrixSparse.R")
source("03_UsefulMatrixTransforms.R")
source("04_SpectralCurvesPlots.R")


cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")

if(!require(Matrix)){
  install.packages("Matrix")
  library(Matrix)
}

examples = FALSE

######################
######################
## Step 1: Objective function
.objective.stiefel = function(V0, sigmalist, K, find.eigen = FALSE){
  .dir = dim(V0)[2]
  .f = rep(0, K)
  .eigenvals = matrix(0, ncol = .dir, nrow = K)
  for(.k in 1:K){
    .eigenvalsK =  diag( t(V0) %*% sigmalist[[.k]]%*% V0 )
    .f[.k] = sum( .eigenvalsK )
    .eigenvals[.k, ] = .eigenvalsK
  }
  
  ## Output 1 & 2: value of f and 
  ##                which K maximizes.
  .objective = max(.f)
  .kmax = which(.f == .objective)
  .OUTPUT = list(objective = .objective, 
                 kmax = .kmax)
  
  if(find.eigen){
    ## Output 3: the eigenvalues associated
    ##            with each column of V0
    .eigenvals = apply(.eigenvals, MARGIN = 2, max)
    
    ## Output 4: the ordered eigenvalues
    ##            and the ordered eigenvectors.
    .order = order(.max.eigenvals, decreasing = FALSE)
    .sorted.eigenvals = .max.eigenvals[.max.eigenvals]
    .sorted.V0 = V0[, .max.eigenvals]
    
    .OUTPUT = c(.OUTPUT, 
                list(eigenvals = .eigenvals,
                     sorted.eigenvals = .sorted.eigenvals,
                     sorted.V0 = .sorted.V0))
  }
  
  return(.OUTPUT)
}


######################
######################
## Step 2: get the gradient of the
##          objective function over R^(pxr)
.objective.der = function(V0, sigmalist, K){
  
  .maxvals = .objective.stiefel(V0 = V0, 
                                sigmalist = sigmalist, 
                                K = K)
  
  .maxind = .maxvals$kmax[1]
  
  .der = 2 * sigmalist[[.maxind]] %*% V0
  
  return(.der)
}


######################
######################
## Step 3.0 Function that verifies 
##          if a "vector" U is on the
##          tangent space of V0
.is.tangent = function(V0, U, threshold = 10^(-8), verbose = FALSE){
  .M = t(V0) %*% U + t(U) %*% V0
  
  if(max(abs(.M)) > threshold){
    if(verbose) print("U is not on the Stiefel tangent space on V0.")
    return(FALSE)
  }
  
  return(TRUE)
}


######################
######################
## Step 3.1: Function that projects a
##          tangent vector U in T v0 R^(pxr)
##          into the T v0 St(p,r)
.stiefel.projection.tangent = function(V0, U){
  
  .M = t(V0) %*% U
  .skewM = (.M + t(.M))/2
  
  .proj = U - V0 %*% .skewM
  
  return(.proj)
}


######################
######################
## Step 4: Tangent Gradient of objective
##          on Stiefel manifold.
.objective.stiefel.der = function(V0, sigmalist, K){
  
  .U1 = .objective.der(V0 = V0, 
                       sigmalist = sigmalist, 
                       K = K)
  .U2 = .stiefel.projection.tangent(V0, .U1)
  
  return(.U2)
}


######################
######################
## Step 5.0: Function that determines
##          if a matrix V0 is on the
##          stiefel manifold.
.is.stiefel = function(V0, threshold = 10^(-8), verbose = FALSE){
  .ndir = dim(V0)[2]
  .M = NULL
  
  if(is.null(V0)){
    if(verbose) print("Error: no value of V0.")
    return(-1)
  } else if(.ndir == 1){
    .norm = as.vector(V0)
    .M = sum(.norm^2) - 1
  } else{
    .M = t(V0) %*% V0 - diag(.ndir)
  }
  
  if(max(abs(.M)) > threshold){
    if(verbose) print("U is not on the Stiefel manifold.")
    return(FALSE)
  }
  
  return(TRUE)
  
}


######################
######################
## Step 5.1: Project from tangent space 
##            with metric projection:
.stiefel.projection.mspace = function(M){
  .svd = svd(M)
  .proj = .svd$u %*% t(.svd$v)
  
  return(.proj)
}


######################
######################
## Step 5.2: Project from tangent space
##            to manifold by exponential map.
.stiefel.projection.expspace = function(V0,U){
  
  .U = .stiefel.projection.tangent(V0 = V0, U = U)
  
  .A = t(V0) %*% .U
  
  .expA = expm(-.A)
  .expBr = expm( U %*% t(V0) - V0 %*% t(U) )
  
  .expRiemman = .expBr %*% V0 %*% .expA
  
  return(.expRiemman)
}


######################
######################
## Step 5.3: Project from tangent space to
##            manifold, choosing what to do:
.stiefel.projection.space = function(V0, U, type = c("M","exp")){
  
  .proj = NULL
  if(type == "M"){
    .proj = .stiefel.projection.mspace(M = U + V0)
  } else if(type == "exp"){
    .proj = .stiefel.projection.expspace(V0 = V0, U = U)
  }
  
  return(.proj)
}


######################
######################
## Step 6.1: Manifold Gradient Descent
##          by metric projection.
.sgd.stiefel = function(sigmalist, p, ndir, K, type = c("M", "exp"), 
                        nstarts = 10, alpha = 0.01, max.iter = 10){
  
  ## Define objects:
  .fmatrix = matrix(0, ncol = max.iter, nrow = nstarts)
  .init.points = list()
  .conv.points = list()
  
  for(.start in 1:nstarts){
    
    ## Define starting points / starting values.
    .init.points[[.start]] = .rand.orthonormal(p, ndir)
    .Vit = .init.points[[.start]]
    
    if(!.is.stiefel(V0 = .Vit)){
      print(paste("Error (",.start,",", .it,"): Update point outside the Stiefel manifold."))
    }
    .fmatrix[.start , 1] = .objective.stiefel(.Vit, sigmalist = sigmalist, K)$objective
    
    ## Main iterative step:  
    for(.it in 2:max.iter){
      .Vaux = .Vit
      .tangent = .objective.stiefel.der(V0 = .Vaux, sigmalist = sigmalist, K = K)
      
      if(!.is.tangent(V0 = .Vaux, U = .tangent)){
        print(paste("Error (",.start,",", .it,"): derivative outside tangent space."))
      }
      .tangent.update = - alpha * .tangent / log(.it + 2)
      
      .Vit = .stiefel.projection.space(V0 = .Vaux, 
                                       U = .tangent.update, 
                                       type = type)
      if(!.is.stiefel(V0 = .Vit)){
        print(paste("Error (",.start,",", .it,"): Update point outside the Stiefel manifold."))
      }
      
      .fmatrix[.start, .it] = .objective.stiefel(V0 = .Vit, 
                                             sigmalist = sigmalist, 
                                             K = K)$objective
    }
    
    .conv.points[[.start]] = .Vit
  }
  
  .OUTPUT = list(fmatrix = .fmatrix, 
                 init.points = .init.points,
                 conv.points = .conv.points)
  return(.OUTPUT)
  
}


######################
######################
## Step 7: Final function for the optimization of
##          the objective on the Stiefel manifold
sgd.stiefel = function(sigmalist, p, ndir, K, type = c("M", "exp"), 
                       nstarts = 10, alpha = 0.01, max.iter = 10){
  
  ## Find the convergence points.
  .stiefel = .sgd.stiefel(sigmalist = sigmalist, p = p, ndir = ndir, 
                          K = K, type = type, 
                          nstarts = nstarts, 
                          alpha = alpha, max.iter = max.iter)
  
  ## Find the eigenvalues.
  .chosen.start = which.min(.stiefel$fmatrix[, max.iter])
  .vectors = .stiefel$conv.points[[.chosen.start]]
  
  .values = rep(0, ndir)
  for(.dir in 1:ndir){
    .values[.dir] = .objective.sphere(v0 = .vectors[, .dir], 
                                sigmalist = sigmalist, 
                                K = K)
  }
  
  ## Order vectors/values in increasing order.
  .order = order(.values, decreasing = FALSE)
  .values = .values[.order]  
  .vectors = .vectors[, .order]
  
  ## Organize output:
  .OUTPUT = list(values = .values, 
                 vectors = .vectors)
  return(.OUTPUT)
  
}


#################################################
#################################################
#################################################
#################################################
if(examples){
  
  #################################
  ## Generate example data:
  {
    .H1 = 1:3
    .H2.1 = 10 + 1:3
    .H2.2 = 20 + 1:3
    .H2.3 = 30 + 1:3
    .p = 100
    .t = 100
    
    .ph1 = 0.7
    .ph2 = 0.7
    .pnh = 0.05
    
    .diagonal_shift = 3
    
    ## Generate the three precision matrices with hubs.
    .Theta1 = r.sparse.pdhubmat(p = .p, t = .t, 
                                H1 = .H1, H2 = .H2.1, 
                                ph1 = .ph1, ph2 = .ph2, pnh = .pnh, 
                                diagonal_shift = .diagonal_shift, shuffle = FALSE, type = "unif",
                                hmin1 = 4, hmax1 = 5, 
                                hmin2 = 4, hmax2 = 5, 
                                nhmin = 4, nhmax = 5)
    .Theta2 = r.sparse.pdhubmat(p = .p, t = .t, 
                                H1 = .H1, H2 = .H2.2, 
                                ph1 = .ph1, ph2 = .ph2, pnh = .pnh, 
                                diagonal_shift = .diagonal_shift, shuffle = FALSE, type = "unif",
                                hmin1 = 4, hmax1 = 5, 
                                hmin2 = 4, hmax2 = 5, 
                                nhmin = 4, nhmax = 5)
    .Theta3 = r.sparse.pdhubmat(p = .p, t = .t, 
                                H1 = .H1, H2 = .H2.3, 
                                ph1 = .ph1, ph2 = .ph2, pnh = .pnh, 
                                diagonal_shift = .diagonal_shift, shuffle = FALSE, type = "unif",
                                hmin1 = 4, hmax1 = 5, 
                                hmin2 = 4, hmax2 = 5, 
                                nhmin = 4, nhmax = 5)
    
    ## Obtain the correlation associated with each matrix:
    .cor1 = .PMtoCOR(.Theta1)
    .cor2 = .PMtoCOR(.Theta2)
    .cor3 = .PMtoCOR(.Theta3)
    
    .cor = list(cor1 = .cor1, 
                cor2 = .cor2, 
                cor3 = .cor3)
  }
  
  #################################
  ## Verify the objective function:
  {
    .V = .rand.orthonormal(.p, 10)
    round(t(.V)%*% .V, digits = 5) 
    ## Yes! it generates good basis.
  }
 
  #################################
  ## Verify the projection tangent:
  {
    .V = .rand.orthonormal(.p, 10)
    .X = matrix(rnorm(.p * 10), ncol = 10, nrow = .p)
    .proj = .stiefel.projection.tangent(.V, .X)
    
    round( t(.V)%*% .proj + t(.proj) %*% .V, digits = 5)
    ## Yes! The projection gives an element of the tangent.
    
  }
  
  #################################
  ## Verify the metric projection to the manifold.
  {
    .X = matrix(rnorm(.p * 10), ncol = 10, nrow = .p)
    .proj.st = .stiefel.projection.mspace(M = .X)
    
    round( t(.proj.st) %*% .proj.st, digits = 5 )
    ## Yes! the metric projection works.
  }

  #################################
  ## Verify the exponential map to the Stiefel manifold:
  {
    .V = .rand.orthonormal(.p, 10)
    .X = matrix(rnorm(.p * 10), ncol = 10, nrow = .p)
    .proj.st = .stiefel.projection.expspace(V0 = .V, U = .X)
    
    round( t(.proj.st) %*% .proj.st, digits = 5 )
    ## YES! the exponential map works.
  }
  
  #################################
  ## Verify the projected derivative
  {
    .V = .rand.orthonormal(.p, 10)
    round(t(.V)%*% .V, digits = 5)
    .Vder = .objective.der(V0 = .V, sigmalist = .cor, K = 3)
    .is.tangent(V0 = .V, U = .Vder)
    
    .Vder.st = .objective.stiefel.der(V0 = .V, sigmalist = .cor, K = 3)
    .is.tangent(V0 = .V, U = .Vder.st)
  }
    
  
  
  
  #################################  
  #################################  
  #################################  
  ## Let's see if our optimization framework works:
  {
    .max.iter = 1000
    .nstarts = 10
    .output = .sgd.stiefel(sigmalist = .cor, p = .p, ndir = 5, K = 3, 
                           type = "M", nstarts = .nstarts, alpha = 0.05, max.iter = .max.iter)
    .minval = min(.output$fmatrix[, .max.iter])
    
    par(mfrow = c(1,1))
    plot(0, col = "white",
         xlim = c(0, .max.iter),
         ylim = c(0, max(.output$fmatrix)),
         main = "Manifold Gradient Descent: Evolution",
         xlab = "Iteration step",
         ylab = "Function value")
    for(.start in 1:.nstarts){
      lines(x = 1:.max.iter, y = .output$fmatrix[.start, ], col = cbPalette[1])
    }  
  }

  
  #################################  
  #################################  
  #################################  
  ## Additional verification:
  
  round(t(.output$conv.points[[1]]) %*% (.output$conv.points[[1]]), digits = 5)
  round(t(.output$conv.points[[2]]) %*% (.output$conv.points[[2]]), digits = 5)
  round(t(.output$conv.points[[3]]) %*% (.output$conv.points[[3]]), digits = 5)
  round(t(.output$conv.points[[4]]) %*% (.output$conv.points[[4]]), digits = 5)
  round(t(.output$conv.points[[5]]) %*% (.output$conv.points[[5]]), digits = 5)
  
  
  par(mfrow = c(3,2))
  .start = 1
  plot(apply(.output$conv.points[[.start]]^2, MARGIN = 1, sum), ylim = c(0,1), main = "End 1")
  plot(apply(.output$init.points[[.start]]^2, MARGIN = 1, sum), ylim = c(0,1), main = "Start 1")

  .start = 2
  plot(apply(.output$conv.points[[.start]]^2, MARGIN = 1, sum), ylim = c(0,1), main = "End 2")
  plot(apply(.output$init.points[[.start]]^2, MARGIN = 1, sum), ylim = c(0,1), main = "Start 2")

  .start = which.min(.output$fmatrix[, .max.iter])
  plot(apply(.output$conv.points[[.start]]^2, MARGIN = 1, sum), ylim = c(0,1), main = "End 3")
  plot(apply(.output$init.points[[.start]]^2, MARGIN = 1, sum), ylim = c(0,1), main = "Start 3")
  
  ## It worked exactly how it was supposed to!
    
  ## Step 7: example.
  .max.iter = 1000
  .nstarts = 10
  .bottom.eig = sgd.stiefel(sigmalist = .cor, p = .p, ndir = 20, K = 3, 
                            type = "M", nstarts = .nstarts, alpha = 0.05, 
                            max.iter = .max.iter)
  par(mfrow = c(2,1))
  plot(.bottom.eig$values)
  plot.spectralcurves(V = .bottom.eig$vectors, 
                      p = .p, H1 = 1:3, H2 = c(.H2.1, .H2.2, .H2.3))
  ## Looks good!
  
  
}

rm(examples)

