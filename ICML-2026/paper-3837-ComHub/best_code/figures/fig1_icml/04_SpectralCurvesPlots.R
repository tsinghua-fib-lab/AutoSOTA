#################################################
#################################################
#################################################
##
## This document provides the function for 
## visualizing the evolution of the influence
## measures when we increase the number of 
## included bottom eigenvalues. 
##
#################################################
#################################################
#################################################

cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")


#################################################
#################################################
## Ploting spectral curves directly:
plot.spectralcurves = function(Sigma = NULL, Theta = NULL, V = NULL, r = NULL, p, H1, H2,
                               main = ""){
  .cond = .plot.spectralcurves.check(Sigma = Sigma, 
                                     Theta = Theta, 
                                     V = V)
  
  ## Define the variables used:
  .r = NULL; .vecmat = NULL;
  
  if(is.null(r)){
    .r = floor(p/2)
  } else{
    .r = r
  }
  
  if(.cond < 1){
    print("Error: no output generated.")
    return(-1)
  } else if(.cond == 1){
    .r = dim(V)[2] 
    .vecmat = V
    
  } else if(.cond == 2){
    .eigpm = eigen(Theta)
    .vecmat = .eigpm$vectors[, 1:.r]
  } else if(.cond == 3){
    .eigcov = eigen(Sigma)
    .vecmat = .eigcov$vectors[, p + 1 - 1:.r]
  }
  
  ## Find the spectral curve trajectories:
  .ow.traj = matrix(rep(0, p * .r), ncol = .r)
  .ow.traj[,1] = .vecmat[,1]^2
  for(.i in 2:.r){
    .ow.traj[, .i] = .ow.traj[, .i-1] + .vecmat[, .i]^2
  }
  
  ## Plot the trajectories.
  plot(c(0,0), col = "white",
       xlim = c(0, .r),
       ylim = c(0, 1),
       main = main,
       xlab = "Influence Measure",
       ylab = "# of eigenvectors")
  
  .indexlist = c(setdiff(x = 1:p, y = c(H1,H2)), H2, H1)
  for(.j in .indexlist){
    if(!(.j %in% c(H1,H2))){ ## Choose colors.
      lines(x = 0:.r, y = c(0,.ow.traj[.j, ]), col = cbPalette[1])
    } else if(.j %in% H2){
      lines(x = 0:.r, y = c(0,.ow.traj[.j, ]), col = cbPalette[3])  
    } else{
      lines(x = 0:.r, y = c(0,.ow.traj[.j, ]), col = cbPalette[2])
    }
  }
  
  return(1)
  
}

#################################################
#################################################
## Ploting spectral curves directly:
plot.spectralcurves2 = function(Sigma = NULL, Theta = NULL, V = NULL, r = NULL, p, K, Hjoint, Hind,
                               main = "", coljoint = NULL, colind = NULL, colnon = NULL){
  .cond = .plot.spectralcurves.check(Sigma = Sigma, 
                                     Theta = Theta, 
                                     V = V)
  
  ## Define the variables used:
  .r = NULL; .vecmat = NULL;
  
  if(is.null(r)){
    .r = floor(p/2)
  } else{
    .r = r
  }
  
  if(.cond < 1){
    print("Error: no output generated.")
    return(-1)
  } else if(.cond == 1){
    .r = dim(V)[2] 
    .vecmat = V
    
  } else if(.cond == 2){
    .eigpm = eigen(Theta)
    .vecmat = .eigpm$vectors[, 1:.r]
  } else if(.cond == 3){
    .eigcov = eigen(Sigma)
    .vecmat = .eigcov$vectors[, p + 1 - 1:.r]
    
  }

  ## Find the spectral curve trajectories:
  .ow.traj = matrix(rep(0, p * (.r+1)), ncol = .r+1)
  .ow.traj[,2] = .vecmat[,1]^2
  for(.i in 2:.r){
    .ow.traj[, .i+1] = .ow.traj[, .i] + .vecmat[, .i]^2
  }
  
  ## Plot the trajectories.
  plot(c(0,0), col = "white",
       xlim = c(0, .r),
       ylim = c(0, 1),
       main = main,
       ylab = "Influence Measure",
       xlab = "# of eigenvectors")
  .Hinds = NULL
  for(.k in 1:K){
    .Hinds = c(.Hinds, Hind[[.k]])
  }
  
  .indexes = c(setdiff(1:p, c(Hjoint,.Hinds)), .Hinds, Hjoint)
  for(.j in .indexes){
    if(.j %in% Hjoint){
      lines(x = 0:.r, y = .ow.traj[.j, ], col = coljoint)
    } else if(.j %in% .Hinds){
      for(.k in 1:K){
        if(.j %in% Hind[[.k]]){
          lines(x = 0:.r, y = .ow.traj[.j, ], col = colind[.k])
        }
      }
    } else{
      lines(x = 0:.r, y = .ow.traj[.j, ], col = colnon)
    } 
  }
  
  return(1)
  
}


#################################################
#################################################
## Verifying conditions are satisfied 
## for safe use of function.
.plot.spectralcurves.check = function(Sigma, Theta, V){
  x = NULL
  
  if(is.null(V) & is.null(Sigma) & is.null(Theta)){
    print("Error: no proper inputs provided.")
    x = -1
  } else if(!is.null(V)){
    x = 1
  } else if(!is.null(Theta)){
    x = 2
  } else if(!is.null(Sigma)){
    x = 3
  } 
  
  return(x)
}



#################################################
#################################################
## 
##

