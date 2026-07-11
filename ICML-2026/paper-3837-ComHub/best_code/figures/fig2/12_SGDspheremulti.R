#################################################
#################################################
#################################################
#################################################
##
## In this code, we generate an algorithm for 
## detecting multiple eigenvalues/eigenvectors
## sequentially via a sequential optimization on
## restricted spheres.
##
#################################################
#################################################
#################################################
#################################################

source("02_GeneratingMultipleMatrixSparse.R")
source("03_UsefulMatrixTransforms.R")
source("04_SpectralCurvesPlots.R")


if(!require(Matrix)){
  install.packages("Matrix")
  library(Matrix)
}

cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")

examples = FALSE

######################
######################
## Step 1.1= code the objective function:
## f(v0) = max_k v0^T Sigma(k) v0
## for v0 on the unit sphere.
.objective.sphere = function(v0, sigmalist, K){
  .vals = rep(0,K)
  for(.k in 1:K){
    .vals[.k] = crossprod( sigmalist[[.k]] %*% v0, v0 )
  }
  return(max(.vals))
}


######################
######################
## Step 2.1: Function that verifies if 
##            v0 is a unit vector.
.is.normalv = function(v0, threshold = 10^(-8),
                       verbose = FALSE){
  .m = sqrt(sum(v0*v0))
  if(abs(.m - 1) > threshold){
    if(verbose) print("Vector v0 is not of norm 1.")
    return(FALSE)
  }
  return(TRUE)
}


######################
######################
## Step 2.2: Function that verifies if a 
##            vector (or the columns of a matrix) 
##            satisfies (satisfy) the restriction
##            Vorth^T v0 = 0.
.is.orthogonal = function(v0, Vorth = NULL, threshold = 10^(-8),
                          verbose = FALSE){
  
  .m = NULL
  if(is.null(Vorth)){
    return(TRUE)
  } else if(dim(Vorth)[2] == 1){
    .norm = as.vector(Vorth)
    .m = sum(.norm * v0)
  } else if(dim(Vorth)[2] > 1){
    .m = t(Vorth) %*% v0   
  } 
  
  if(max(abs(.m)) > threshold){
    if(verbose) print("The vector v0 is not orthogonal to Vorth")
    return(FALSE)
  }
  return(TRUE)
} 


######################
######################
## Step 2.3: Function that determines
##          if a matrix Vorth is on the
##          stiefel manifold.
.is.stiefel = function(V0 = NULL, threshold = 10^(-8), verbose = FALSE){
  
  .m = NULL
  if(is.null(V0)){
    return(TRUE)
  } else if(dim(V0)[2] == 1){
    .norm = as.vector(V0)
    .m = sum(.norm* .norm) - 1
  } else{
    .r = dim(V0)[2]
    .m = t(V0) %*% V0 - diag(.r)
  }
  
  
  if(max(abs(.m)) > threshold){
    if(verbose) print("Vorth is not on the Stiefel manifold.")
    return(FALSE)
  }
  
  return(TRUE)
  
}


######################
######################
## Step 2.4: Function that creates
##            the projection matrix 
##            to the column space of an orth. mat.
.proj.matrix = function(Vorth, p){
  
  .projmat = NULL
  if(is.null(Vorth)){
    .projmat = matrix(0, nrow = p, ncol = p)
  } else if(ncol(Vorth) == 1){
    .vect = as.vector(Vorth) 
    .projmat = outer(.vect, .vect)  
  } else{
    .projmat = Vorth %*% t(Vorth)
  }
  
  return(.projmat)
}


######################
######################
## Step 3: This function finds a subgradient
## for the function 
## f(v0) = max_k v0^T Sigma(k) v0,
## when considering f as a function
## on the sphere. 
## If Vorth = NULL, the derivative is
## taken on the general tangent space.
## If Vorth != 0, we calculate it on
## S(Vorth) = {v | norm(v)=1, Vorth^T v = 0}
.objective.sphere.subgradient = function(v0, sigmalist, K, 
                                         Vorth = NULL, ProjVorth = NULL,
                                         threshold = 10^(-8)){
  
  if(!.is.normalv(v0)){
    print("Error: The provided vector is not of unit norm.")
  }
  if(!.is.stiefel(V0 = Vorth)){
    print("Error: The provided restrictions are not orthonormal.")
  }
  if(!.is.orthogonal(v0 = v0, Vorth = Vorth, threshold = threshold)){
    print("Error: The vector does not satisfy the required restrictions.")
  }
  
  ## Create the projection matrix
  ## to the orthogonal subspace.
  ## to the restricted tangent.
  .vect = as.vector(v0)
  .p = length(.vect)
  
  
  .projmat = matrix(0, .p, .p)
  if(is.null(ProjVorth)){
    .p = length(.vect)
    .projmat = .proj.matrix(Vorth = Vorth, p = .p)
  } else{
    .projmat = ProjVorth
  }
  .vect = .vect - .projmat %*% .vect
  .vect = .vect / sqrt(sum(.vect^2)) 
  
  
  ## Generate the subgradient:
  .vals = rep(0, K)
  for(.k in 1:K){
    .vals[.k] = crossprod( sigmalist[[.k]] %*% .vect, .vect )
  }
  .maxvals = which(.vals == max(.vals))
  .maxind = .maxvals[1]
  
  .subgradient = 2 * sigmalist[[.maxind]] %*% .vect
  .projection = .subgradient - .projmat %*% .subgradient
  
  return(.projection)
}


######################
######################
## Step 4.1: Create a set of random unit vectors
##            orthogonal to Vorth.
.starting.point.sphere = function(p, Vorth = NULL, nstarts){
  .v0s = matrix(rnorm(p*nstarts), ncol = nstarts, nrow = p)
  
  if(!is.null(Vorth)){
    ## Create the projection matrix to
    ##  span(Vorth)
    .projmat = NULL
    if(dim(Vorth)[2] == 1){
      .norm = as.vector(Vorth)
      .projmat = outer(.norm, .norm)
    } else{
      .projmat = Vorth %*% t(Vorth)
    }
    
    ## Project to the orthogonal space to Vorth
    .v0s = .v0s - .projmat %*% .v0s
    
  }
  
  ## Normalize and return.
  if(nstarts == 1){
    .norm = sqrt(sum(.v0s^2))
    .v0s = as.matrix(.v0s, ncol = 1)/.norm
    return(.v0s)
  }
  
  .norms = apply(.v0s, MARGIN = 2, function(x) sqrt(sum(x^2)))
  .v0s = .v0s %*% diag(1/.norms)
  return(.v0s)
  
}


######################
######################
## Step 4.2: Function that projects point back to 
##            a constrained point.
.constraint.proj = function(v0, Vorth, ProjVorth = NULL){
  
  .vect = as.vector(v0)
  .p = length(.vect)
  
  .ProjVorth = ProjVorth
  if(is.null(.ProjVorth)){
    .ProjVorth = .proj.matrix(Vorth = Vorth, p = .p)
  }
  
  .vect = .vect - .ProjVorth %*% .vect
  .vect = .vect/sqrt(sum(.vect^2))
  
  return(.vect)
}


######################
######################
## Step 5: Given a set of constraints of the form
##          {v0 : Vorth^T v0 = 0}
##          we optimize the objective on that
##          constrained space. 
.sgd.sphere.step = function(sigmalist, p, K, Vorth = NULL, 
                            nstarts = 10, alpha = 0.01, max.iter = 10,
                            threshold = 10^(-8),
                            verbose = FALSE){
  
  ## Step 1: create starting points.
  .init.points = .starting.point.sphere(p = p, 
                                        Vorth = Vorth, 
                                        nstarts = nstarts)
  colnames(.init.points) = paste0("start", 1:nstarts)
  
  if(!.is.orthogonal(v0 = .init.points, Vorth = Vorth, threshold = threshold)){
    if(verbose) print("Error: The initial points do not satisfy the restriction Vorth^T v0 = 0.")
    return(-1)
  }
  
  ## Step 2: create matrixes for the endpoints and f values:
  .fmatrix = matrix(0, ncol = max.iter, nrow = nstarts)
  rownames(.fmatrix) = paste0("start", 1:nstarts)
  colnames(.fmatrix) = paste0("iter", 1:max.iter)
  
  .conv.points = matrix(0, ncol = nstarts, nrow = p)
  colnames(.conv.points) = paste0("start", 1:nstarts)
  
  .nsteps = rep(max.iter + 1, nstarts)
  names(.nsteps) = paste0("start", 1:nstarts)
  
  ## Step 4: create the projection to the orthogonal of Vorth:
  .projmat = .proj.matrix(Vorth = Vorth, p = p)
  
  ## Step 3: use a cycle to find optimal point 
  ##          on each starting point:
  for(.start in 1:nstarts){
    
    ## Find the f on starting point.
    .vnew = as.vector(.init.points[, .start])
    .vold = as.vector(.init.points[, .start])
    .fmatrix[.start , 1] = .objective.sphere(v0 = .vnew, 
                                             sigmalist = sigmalist, 
                                             K = K)
    
    ## Optimize vector v until convergence
    .it = 2
    while(.it < max.iter){
      
      ## Move againts the subgradient direction.
      .vold = .vnew
      .subgradient = .objective.sphere.subgradient(v0 = .vold, 
                                                   sigmalist = sigmalist, K = K, 
                                                   Vorth = Vorth, 
                                                   ProjVorth = .projmat,
                                                   threshold = threshold)
      .vnew = .vold - alpha * .subgradient / log(.it + 2)
      .vnew = .constraint.proj(v0 = .vnew, Vorth = Vorth, ProjVorth = .projmat)
      .isrest = .is.orthogonal(v0 = .vnew, Vorth = Vorth, threshold = threshold)
            
      if(!.isrest){
        if(verbose){
          print(paste("Error [", .start, ",", .it,
                      "]: Subgradient is not on the tangent space, Vorth^T v != 0"))
        }
      }
      
      .fmatrix[.start, .it] = .objective.sphere(v0 = .vnew, 
                                                sigmalist = sigmalist, 
                                                K = K) 
      .fstep = .fmatrix[.start, .it] - .fmatrix[.start, .it - 1]
      if( .fstep > 0 & .fstep < threshold ){ 
        .nsteps[.start] = .it
        .fmatrix[.start, (.it+1):max.iter] = rep(.fmatrix[.start, .it], max.iter - .it)
        .it = max.iter + 1
      }
      .it = .it + 1
    }
    
    .nsteps[.start] = min(max.iter, .nsteps[.start])
    .conv.points[, .start] = .vnew
  }
  
  .OUTPUT = list(fmatrix = .fmatrix,
                 init.points = .init.points,
                 conv.points = .conv.points,
                 nsteps = .nsteps)
  return(.OUTPUT)
}


######################
######################
## Step 5: Generate function that finds
##          eigenvectors sequentially considering
##          constraints.
sgd.sphere = function(ndir, sigmalist, p, K,
                      nstarts = 10, alpha = 0.01, max.iter = 10,
                      verbose = FALSE){
  
  .eigvals = rep(0, ndir)
  .Vorth = NULL
  
  for(.dir in 1:ndir){
    if(verbose) print(paste0("Calculating ", .dir, "-th", " eigenvector."))
      
    .step = .sgd.sphere.step(sigmalist = sigmalist, p = p, K = K, Vorth = .Vorth, 
                             nstarts = nstarts, alpha = alpha, max.iter = max.iter)
    
    .opt.start = which.min(.step$fmatrix[, max.iter])
    .vopt = as.vector(.step$conv.points[, .opt.start])
    
    ## Saving vector and value.
    .Vorth = cbind(.Vorth, .vopt)
    .eigvals[.dir] = .objective.sphere(v0 = .vopt, 
                                       sigmalist = sigmalist, K = K)
  }
  
  .OUTPUT = list(vectors = .Vorth,
                 values = .eigvals)
  return(.OUTPUT)
}


######################
######################
if(examples){
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
  
  
  ## Step 1.1: example.
  .x = matrix(rnorm(.p), ncol = 1)
  .x = .x/sqrt(sum(.x^2))
  sqrt(sum(.x^2))
  
  .objective.sphere(v0 = .x, sigmalist = .cor, K = 3)
  ## Good! Step 1.1 works.
  
  ## Step 2.1: example.
  .is.normalv(v0 = .x)
  .is.normalv(v0 = 2*.x)
  ## Good! Step 2.1 works.
  
  ## Step 2.2: example.
  .y = matrix(rnorm(.p), ncol = 1)
  .y = .y/sqrt(sum(.y^2))
  .z = matrix(rnorm(.p), ncol = 1)
  .z = .y/sqrt(sum(.z^2))
  
  .is.orthogonal(v0 = .x, Vorth = .y)
  .is.orthogonal(v0 = .x, Vorth = cbind(.y, .z))
  .Vorth = .starting.point.sphere(p = .p, Vorth = matrix(.x, ncol = 1), nstarts = 3)
  .is.orthogonal(v0 = .x, Vorth = .Vorth)
  
  ## Good! Step 2.2 works!
  
  ## Step 2.3: example.
  .y = .starting.point.sphere(p = .p, Vorth = matrix(.x, ncol = 1), nstarts = 1)
  sqrt(sum(.y^2))
  .z = .starting.point.sphere(p = .p, Vorth = cbind(.x, .y), nstarts = 1)
  sqrt(sum(.z^2))
  
  .Vorth = cbind(.x, .y, .z)
  .is.stiefel(V0 = .Vorth)
  round(t(.Vorth) %*% .Vorth, digits = 5)
  ## Good! Step 2.3 works!
  
  
  ## Step 3: verify how the subgradient works.
  .x = .starting.point.sphere(p = .p, Vorth = .Vorth, nstarts = 1)
  .y = .starting.point.sphere(p = .p, Vorth = NULL, nstarts = 1)
  
  round(t(.Vorth) %*% .x, digits = 5)
  round(t(.Vorth) %*% .y, digits = 5)
  
  .subgrad = .objective.sphere.subgradient(v0 = .x, sigmalist = .cor, K = 3, 
                                           Vorth = .Vorth, threshold = 10^(-8))
  round(t(.Vorth)%*% .subgrad, digits = 5)
  
  .subgrad = .objective.sphere.subgradient(v0 = .y, sigmalist = .cor, K = 3,
                                           Vorth = .Vorth, threshold = 10^(-8))
  ## Step 3 seems to work well.
  
  ## Step 4: example:
  .v1 = .starting.point.sphere(p = 100, Vorth = NULL, nstarts = 5)
  round(t(.v1) %*% .v1, digits = 5)
  
  .v2 = .starting.point.sphere(p = .p, Vorth = .Vorth, nstarts = 5)
  round(t(.v2) %*% .v2, digits = 5)
  round(t(.Vorth) %*% .v2, digits = 5)
  ## Step 4 seems to work well.
  
  
  
  
  ## Step 5: example.
  .step1 = .sgd.sphere.step(sigmalist = .cor, p = .p, K = 3, Vorth = NULL, 
                            nstarts = 10, alpha = 0.05, 
                            max.iter = 1000, threshold = 10^(-8), verbose = TRUE)
  .step1$nsteps
  plot(0, col = "white",
       xlim = c(0, .max.iter),
       ylim = c(0,max(.step1$fmatrix)),
       main = "SGD step results: step 1",
       xlab = "Iteration step",
       ylab = "Function value")
  for(.start in 1:10){
    lines(x = 1:1000, y = .step1$fmatrix[.start,], col = cbPalette[1])
  }
  .Vorth = cbind(.step1$conv.points[, which.min(.step1$fmatrix[,1000])] )
  
  
  
  
  
  
  .step2 = .sgd.sphere.step(sigmalist = .cor, p = .p, K = 3, Vorth = .Vorth, 
                            nstarts = 10, alpha = 0.05, 
                            max.iter = 1000, threshold = 10^(-8), verbose = TRUE)
  .step2$nsteps
  plot(0, col = "white",
       xlim = c(0, .max.iter),
       ylim = c(0,max(.step2$fmatrix)),
       main = "SGD step results: step 2",
       xlab = "Iteration step",
       ylab = "Function value")
  for(.start in 1:10){
    lines(x = 1:1000, y = .step2$fmatrix[.start,], col = cbPalette[1])
  }
  t(.Vorth) %*% .step2$conv.points
  .Vorth = cbind(.Vorth, 
                 .step2$conv.points[, which.min(.step2$fmatrix[,1000])] )
  t(.Vorth)%*% .Vorth
  
  
  
  
  
  .step3 = .sgd.sphere.step(sigmalist = .cor, p = .p, K = 3, Vorth = .Vorth, 
                            nstarts = 10, alpha = 0.05, 
                            max.iter = 1000, threshold = 10^(-8), verbose = TRUE)
  .step3$nsteps
  plot(0, col = "white",
       xlim = c(0, 1000),
       ylim = c(0,max(.step3$fmatrix)),
       main = "SGD step results",
       xlab = "Iteration step",
       ylab = "Function value")
  for(.start in 1:10){
    lines(x = 1:1000, y = .step3$fmatrix[.start,], col = cbPalette[1])
  }
  t(.Vorth) %*% .step3$conv.points[, which.min(.step3$fmatrix[,1000])]
  t(.Vorth) %*% .Vorth
  .Vorth = cbind(.Vorth, 
                 .step3$conv.points[, which.min(.step2$fmatrix[,1000])] )
  
  
  
  .step4 = .sgd.sphere.step(sigmalist = .cor, p = .p, K = 3, Vorth = .Vorth, 
                            nstarts = 10, alpha = 0.05, 
                            max.iter = 1000, threshold = 10^(-8), verbose = TRUE)
  plot(0, col = "white",
       xlim = c(0, 1000),
       ylim = c(0,max(.step4$fmatrix)),
       main = "SGD step results",
       xlab = "Iteration step",
       ylab = "Function value")
  for(.start in 1:10){
    lines(x = 1:1000, y = .step4$fmatrix[.start,], col = cbPalette[1])
  }
  t(.Vorth) %*% .Vorth
  .Vorth = cbind(.Vorth, .step4$conv.points[, which.min(.step4$fmatrix[, 1000])])
  t(.Vorth) %*% .Vorth
  
  
  .step5 = .sgd.sphere.step(sigmalist = .cor, p = .p, K = 3, Vorth = .Vorth, 
                            nstarts = 10, alpha = 0.05, 
                            max.iter = 1000, threshold = 10^(-8), verbose = TRUE)
  plot(0, col = "white",
       xlim = c(0, 1000),
       ylim = c(0,max(.step5$fmatrix)),
       main = "SGD step results: Step 5",
       xlab = "Iteration step",
       ylab = "Function value")
  for(.start in 1:10){
    lines(x = 1:1000, y = .step4$fmatrix[.start,], col = cbPalette[1])
  }
  t(.Vorth) %*% .Vorth
  .Vorth = cbind(.Vorth, .step5$conv.points[, which.min(.step5$fmatrix[, 1000])])
  t(.Vorth) %*% .Vorth
  
  
  .step6 = .sgd.sphere.step(sigmalist = .cor, p = .p, K = 3, Vorth = .Vorth, 
                            nstarts = 10, alpha = 0.05, 
                            max.iter = 1000, threshold = 10^(-8), verbose = TRUE)
  plot(0, col = "white",
       xlim = c(0, 1000),
       ylim = c(0,max(.step6$fmatrix)),
       main = "SGD step results: Step 6",
       xlab = "Iteration step",
       ylab = "Function value")
  for(.start in 1:10){
    lines(x = 1:1000, y = .step6$fmatrix[.start,], col = cbPalette[1])
  }
  t(.Vorth) %*% .Vorth
  .Vorth = cbind(.Vorth, .step6$conv.points[, which.min(.step6$fmatrix[, 1000])])
  t(.Vorth) %*% .Vorth
  
  plot.spectralcurves(V = .Vorth, p = .p, H1 = .H1, H2 = c(.H2.1, .H2.2, .H2.3))
  
  ## Testing step 6
  
  ######################
  ######################
  ## Lets see how the SGD works!
  ##
  .max.iter = 1000
  .nstarts = 10
  .output = sgd.sphere(ndir = 20, sigmalist = .cor, p = .p, K = 3, 
                       nstarts = .nstarts, alpha = 0.01, max.iter = .max.iter, 
                       verbose = TRUE)
  

  plot.spectralcurves(V = .output$vectors, p = .p, 
                      H1 = .H1, H2 = c(.H2.1, .H2.2, .H2.3))
  
  ## It seems like it converges. Why do the other starts converge?
  
  ######################
  ######################
  ## Lets compare the maximum obtained this way with the
  ## maximum obtained with good sphere sampling:
  # .maxsample = 0
  # for(.i in 1:100000){
  #   .x = rnorm(.p)
  #   .x = .x/sqrt(sum(.x^2))
  #   
  #   .val = .objective.sphere(v0 = .x, 
  #                         sigmalist = .cor, 
  #                         K = 3)
  #   .maxsample = max(.maxsample, .val)
  #   
  # }
  # 
  # .maxmax = max(.output$val.matrix)
  # print(paste("The maximum found by the Sphere SGD is:", .maxmax))
  # print(paste("The maximum found by sampling is:", .maxsample))
  
  # .repeat.values = rep(0, .nstarts)
  # for(.ind in 1:.nstarts){
  #   .repeat.values[.ind] = .objective.sphere(.output$conv.p oints[, .ind], 
  #                                         sigmalist = .cor,
  #                                         K = 3)
  # }
  # .repeat.values
  .max.start = which.max(.output$val.matrix[, .max.iter])
  
  
  plot(.output$conv.points[, .max.start]^2)
  
}

rm(examples)


