
#####################################################
#####################################################
#####################################################

FullSimulation <- function(args, index) {

  #####################################################
  ## Runtype 2, 3, 4:
  ## Define output saving object.
  #####################################################
  
  method_names <- c(
    "ST.OVER.CORR.IM", "ST.ORAC.CORR.IM", 
    "ST.OVER.THR.IM", "ST.ORAC.THR.IM")
  n_methods <- length(method_names)

  ## Output of simulation:
  output_sim <- NULL
  
  ## BIC output:
  output_bic <- NULL

  #################################################
  #################################################
  ## BIC output setting:
  
  
  #################################################
  #################################################
  ## Cycle:
  loop_start_time <- Sys.time()
  sim_ind <- 1
  count <- 1

  while (sim_ind < args$nsim + 1) {
    
    ############################
    ######## Generate data:
    ########
    {
      print(paste0("Step ", sim_ind,".1: Generate Precsion Matrices."))
      
      .pmlist   <- lapply(
        1:args$K, function(k) {
          return(r.sparse.pdhubmat(
            p = args$p, T0 = args$T0,
            H1 = args$Hjoint, H2 = args$Hind[[k]],
            ph1 = args$ph1, ph2 = args$ph2,
            pnh = args$pnh, pneff = args$pneff,
            diagonal_shift = args$diagonal_shift,
            shuffle = args$shuffle, type = args$type,
            hmin1 = args$hmin1, hmax1 = args$hmax1,
            hmin2 = args$hmin2, hmax2 = args$hmax2,
            nhmin = args$nhmin, nhmax = args$nhmax,
            neffmin = args$neffmin, neffmax = args$neffmax,
            verbose = FALSE))} )
      .covlist  <- lapply(.pmlist, solve)
      .iclist   <- lapply(.covlist, .COVtoCOR)

    }
    #####################################################
    #####################################################
    ## STEP 2: Generate data.
    {
      print(paste0("Step ", sim_ind,".2: Generate Gaussian Data."))
      
      .Xlist    <- lapply(
        .covlist, function(sigma) {
          rmvnorm(n = args$n, sigma = sigma, method = "svd")})
      .scorlist <- lapply(.Xlist, cor)
      .scovlist <- lapply(.Xlist, cov)

      .thrcor_list <- list()
      for (.k in seq_along(.scorlist)) {
          .thr                <- matrix_threshold_perc(.scorlist[[.k]], perc = 0.7)
          .thrcor_list[[.k]]  <- thresholding(.scorlist[[.k]], .thr) 
      }

      .ns       <- rep(args$n, args$K)
      .ndirOver = floor(sqrt(args$p))
      .ndirOrac = length(args$Hjoint)

    }


    #####################################################
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    ## Step 3: STIEFEL OVERESTIMATION
    {
      print(paste0("Step ", sim_ind,".3: STIEFEL SAMPLE-COR OVEREST."))
      
      ##################
      ## Joint Estimation
      ##################
      .start.time = Sys.time()

      .cor_StOver_obj = sgd.stiefel(
        sigmalist = .scorlist,   ## Find common eigenvectors.
        p = args$p, ndir = .ndirOver, K = args$K,
        type = "M", nstarts = 10,
        alpha = 0.1, max.iter = 500)
      
          
      .cor_StOver_im <- apply(t(t(.cor_StOver_obj$vectors^2)), MARGIN = 1, sum)
      .cor_StOver_evals <- .cor_StOver_obj$values
      .end.time = Sys.time()

      .cor_StOver_time <- difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]]

    }
    #####################################################
    #####################################################
    ## Step 3: STIEFEL OVERESTIMATION
    {
      print(paste0("Step ", sim_ind,".4: STIEFEL SAMPLE-COR ORACLE."))
        
      ##################
      ## Joint Estimation
      ##################
      .start.time = Sys.time()

      .cor_StOrac_obj = sgd.stiefel(
        sigmalist = .scorlist,   ## Find common eigenvectors.
        p = args$p, ndir = .ndirOrac, K = args$K,
        type = "M", nstarts = 10,
        alpha = 0.1, max.iter = 500)
      
          
      .cor_StOrac_im <- apply(t(t(.cor_StOrac_obj$vectors^2)), MARGIN = 1, sum)
      .cor_StOrac_evals <- .cor_StOrac_obj$values
      .end.time = Sys.time()

      .cor_StOrac_time <- difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]]

    }
    #####################################################
    #####################################################
    ## Step 4: STIEFEL ORACLE
    {
      print(paste0("Step ", sim_ind,".4: STIEFEL THRESHOLD-COR OVEREST"))

      ##################
      ## Joint Estimation
      ##################
      .start.time = Sys.time()

      .thr_StOver_obj = sgd.stiefel(
        sigmalist = .scorlist,   ## Find common eigenvectors.
        p = args$p, ndir = .ndirOver, K = args$K,
        type = "M", nstarts = 10,
        alpha = 0.1, max.iter = 500)
      
          
      .thr_StOver_im <- apply(t(t(.thr_StOver_obj$vectors^2)), MARGIN = 1, sum)

      .end.time = Sys.time()

      .thr_StOver_time <- difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]]

    }
    #####################################################
    #####################################################
    ## Step 4: STIEFEL ORACLE
    {
      print(paste0("Step ", sim_ind,".4: STIEFEL THRESHOLD-COR ORACLE"))

      ##################
      ## Joint Estimation
      ##################
      .start.time = Sys.time()

      .thr_StOrac_obj = sgd.stiefel(
        sigmalist = .scorlist,   ## Find common eigenvectors.
        p = args$p, ndir = .ndirOrac, K = args$K,
        type = "M", nstarts = 10,
        alpha = 0.1, max.iter = 500)
      
          
      .thr_StOrac_im <- apply(t(t(.thr_StOrac_obj$vectors^2)), MARGIN = 1, sum)

      .end.time = Sys.time()

      .thr_StOrac_time <- difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]]

    }
    #####################################################
    #####################################################
    ## Step 7: Saving Data:
    {
      ## ID data:
      .data_id_temp <- data.frame(
        TASK_ID   = rep(args$id_task, n_methods),
        SIM_NUM   = rep(sim_ind, n_methods),
        K_MAT_NUM = rep(0, n_methods),
        METHOD    = method_names,
        TIME      = c(.cor_StOver_time, .cor_StOrac_time, 
                      .thr_StOver_time, .thr_StOrac_time))
      
      ## Connectivity data:
      .data_deg_temp <- data.frame(rbind(
        .cor_StOver_im, .cor_StOrac_im,
        .thr_StOver_im, .thr_StOrac_im))
      colnames(.data_deg_temp) <- paste0("var", 1:args$p)
      
      ## Saving data:
      .data_temp <- cbind(.data_id_temp, .data_deg_temp)
      output_sim <- rbind(output_sim, .data_temp)

      rm(.data_deg_temp, .data_id_temp, .data_temp)

    }
    #####################################################
    #####################################################
    ## Step 12: Add stopping condition. 
    {
      print(paste0("Step ", sim_ind,": Time Analysis."))
      
      ## If it will take more than X days to run,
      ## save results and leave.
      time_stamp <- Sys.time()
      current.rt.hour   <- 
        difftime(time_stamp, loop_start_time, units = "hours") %>%
        as.numeric()
      current.rt.days   <- 
        difftime(time_stamp, loop_start_time, units = "days") %>%
        as.numeric()
      mean.rt.days      <- current.rt.days / sim_ind
      expected.rt.days  <- current.rt.days + 1.5 * mean.rt.days
      ncompleted        <- sim_ind
      
      if (expected.rt.days >= args$running_days) { ## days.
        print(paste("---> Expected running time (+1):", 
                    round(expected.rt.days, digits = 4),
                    "days."))
        print("---> Stopping process...")
        sim_ind = args$nsim + 1
      }
      sim_ind <- sim_ind + 1
    }

  }

  output <- list(
    output_sim  = output_sim,
    output_bic  = output_bic)
    
  return(output)

}
