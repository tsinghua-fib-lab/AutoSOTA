#################################################
#################################################
#################################################
#################################################
##
## In this code, we create an algorithm for 
## optimizing an objective function on the sphere.
## This code only deals with finding a single 
## direction. 
##
## In later code we will explore how to optimize
## sequentially multiple directions. This will be
## done via a greedy algorithm.
##
#################################################
#################################################
#################################################
#################################################

source("02_GeneratingMultipleMatrixSparse.R")
source("03_UsefulMatrixTransforms.R")


cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")
examples = FALSE

#################################################

######################
######################
.objective.sphere = function(v0, sigmalist, K){
  .vals = rep(0,K)
  for(.k in 1:K){
    .vals[.k] = crossprod( sigmalist[[.k]] %*% v0, v0 )
  }
  return(max(.vals))
}

######################
######################
.tangent.vect = function(v0, sigmalist, K){
  
  .vals = rep(0,K)
  for(.k in 1:K){
    .vals[.k] = crossprod( sigmalist[[.k]] %*% v0, v0 )
  }
  
  .maxvals = which(.vals == max(.vals))
  
  .tangent = NULL
  .maxind = .maxvals[1]
    
  .tangent = 2 * sigmalist[[.maxind]] %*% v0 - 
                2 * sum( (sigmalist[[.maxind]] %*% v0) * v0) * v0
  
  return(.tangent)
}

######################
######################
.sgd.sphere = function(sigmalist, p, K, nstarts = 10, alpha = 0.01, max.iter = 10){
  
  ## Step 1: create starting points.
  .v0s = matrix(rnorm(p*nstarts), nrow = p, ncol = nstarts)
  .norms = apply(.v0s, MARGIN = 2, function(x) sqrt(sum(x^2)))
  .v0s = .v0s %*% diag(1/.norms)
  
  ## Step 2: do a couple of iterations:
  .max = matrix(0, ncol = max.iter, nrow = nstarts)
  .vkmat = matrix(0, ncol = .nstarts, nrow = p)
  for(.ind in 1:nstarts){
    
    ## Find the value on starting point.
    .vk = as.vector(.v0s[, .ind])
    .vaux = as.vector(.v0s[, .ind])
    .max[.ind , 1] = .objective.sphere(v0 = .vk, 
                                       sigmalist = sigmalist, 
                                       K = K)
    
    for(.it in 2:max.iter){
      .vaux = .vk
      .tangent.v = .tangent.vect(v0 = .vaux, 
                                 sigmalist = sigmalist, 
                                 K = K)
      .vk = .vaux + alpha * .tangent.v 
      .vk = .vk / sqrt(sum(.vk^2))
      
      .max[.ind, .it] = .objective.sphere(v0 = .vk, 
                                          sigmalist = sigmalist, 
                                          K = K)
    }
    
    .vkmat[, .ind] = .vk
  }
  
  .OUTPUT = list(val.matrix = .max, 
                 conv.points = .vkmat)
  return(.OUTPUT)
}


######################
######################
if(examples){
  .H1 = NULL
  .H2.1 = 1:3
  .H2.2 = 1:3
  .H2.3 = 1:3
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
  
  ## Find a random vector to find the normal vector:
  .x = rnorm(.p)
  .x = .x/sqrt(sum(.x^2))
  
  .tangent = .tangent.vect(.x, sigmalist = .cor, K = 3)
  
  sum(.x^2)
  sum(.tangent^2)  
  sum(.x * .tangent)
  ## It works!
  
  
  ######################
  ######################
  ## Lets see how the SGD works!
  ##
  .max.iter = 100
  .nstarts = 30
  .output = .sgd.sphere(sigmalist = .cor, p = .p, K = 3, 
                     nstarts = .nstarts, alpha = 0.01, max.iter = .max.iter)
  
  plot(0, col = "white",
       xlim = c(0, .max.iter),
       ylim = c(0,15),
       main = "SGD step results",
       xlab = "Iteration step",
       ylab = "Function value")
  for(.ind in 1:.nstarts){
    lines(x = 1:.max.iter, y = .output$val.matrix[.ind,], col = cbPalette[1])
  }
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

