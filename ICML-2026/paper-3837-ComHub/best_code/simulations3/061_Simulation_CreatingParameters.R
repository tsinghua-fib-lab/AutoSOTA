#################################################
#################################################
#################################################
##
## In the following document, we introduce the
## functions that run the simulations and
## estimate and turn them into manageable outputs
##
#################################################
#################################################

#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

CreateParameters <- function(id_task, runtype = c(1, 2, 3)) {
  
  if(id_task == 0){
    .args = list()

    .args$p       <- 20
    .args$n_prop  <- 0.5
    .args$T0_prop <- 0.5
    .args$n       <- 100
    .args$T0      <- .args$p * .args$T0_prop
    .args$K       <- 3
    .args$r2      <- 1
    .args$r1      <- 1
    .args$ph1     <- 1
    .args$ph2     <- 1
    .args$pnh     <- 0
    .args$pneff   <- 0
    
    ## Other parameters
    .args$shuffle <- FALSE; .args$type <- "unif"
    .args$diagonal_shift <- 2
    .args$hmin1   <- 9; .args$hmax1   <- 10; 
    .args$hmin2   <- 9; .args$hmax2   <- 10; 
    .args$nhmin   <- 9; .args$nhmax   <- 10; 
    .args$neffmin <- 9; .args$neffmax <- 10;
    
    .args$nsim    <- 3
    .args$id_task <- id_task
    

    .args$Hjoint  <- 1:.args$r1
    .args$Hind    <- lapply(1:.args$K, function(k) {k * 3 + 1:.args$r2})

    .args$running_days <- 4

    return(.args)
    
  }

  sim_par_table <- expand.grid(
    K              = 3,
    hmin1          = 4, hmax1         = 5,
    hmin2          = 4, hmax2         = 5,
    nhmin          = 4, nhmax         = 5,
    neffmin        = 4, neffmax       = 5,
    shuffle       = FALSE,
    type          = "unif",
    running_days  = ifelse(runtype <= 2, 1, 5),
    threshold     = 2,
    
    r2              = c(3),
    r1              = c(5),
    pneff           = c(0.01),
    pnh             = c(0.05),
    ph2             = c(0.3, 0.5),
    ph1             = c(0.3, 0.4, 0.5),
    
    nsim            = ifelse(runtype <= 2, 2, 10),
    diagonal_shift  = c(2,5),
    n_prop          = c(0.5, 0.75, 1, 1.25),
    T0_prop         = c(1),
    p               = c(100, 200, 400))

  args <- sim_par_table[id_task, ]
  args_list <- list()
  for (i in 1:ncol(args)) {
    if (class(args[, i]) == "factor") {
      args_list[[i]] <- as.character(args[1,i])
    } else args_list[[i]] <- args[1,i]
  }
  names(args_list)  <- colnames(args)
  
  args_list$id_task <- id_task
  args_list$n       <- as.integer(args$p * args$n_prop)
  args_list$T0      <- as.integer(args$p * args$T0_prop)
  
  args_list$Hjoint  <- 1:args$r1
  args_list$Hind    <- lapply(1:args$K, function(k) {k * 10 + 1:args$r2})
  
  args_list$running_days <- 4

  return(args_list)

}