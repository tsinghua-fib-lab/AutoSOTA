
#####################################################
#####################################################
#####################################################

FullSimulation <- function(args, index) {

  #####################################################
  ## Runtype 2, 3, 4:
  ## Define output saving object.
  #####################################################
  
  method_names <- c(
    "COR_Scr_IPCHD",              ## IPCHD with screening
    "COR_Thr_IPCHD")             ## IPCHD with thresholding

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

      .pmlist <- r.sparse.pdhubmat_list(
        p = args$p, T0 = args$T0, K = args$K,
        Hjoint = args$Hjoint, Hind_list = args$Hind,
        ph1 = args$ph1, ph2 = args$ph2,
        pnh = args$pnh, pneff = args$pneff,
        diagonal_shift = args$diagonal_shift,
        shuffle = args$shuffle, type = args$type,
        hmin1 = args$hmin1, hmax1 = args$hmax1,
        hmin2 = args$hmin2, hmax2 = args$hmax2,
        nhmin = args$nhmin, nhmax = args$nhmax,
        neffmin = args$neffmin, neffmax = args$neffmax,
        verbose = FALSE)      
      #.pmlist   <- lapply(
      #  1:args$K, function(k) {
      #    return(r.sparse.pdhubmat(
      #      p = args$p, T0 = args$T0,
      #      H1 = args$Hjoint, H2 = args$Hind[[k]],
      #      ph1 = args$ph1, ph2 = args$ph2,
      #      pnh = args$pnh, pneff = args$pneff,
      #      diagonal_shift = args$diagonal_shift,
      #      shuffle = args$shuffle, type = args$type,
      #      hmin1 = args$hmin1, hmax1 = args$hmax1,
      #      hmin2 = args$hmin2, hmax2 = args$hmax2,
      #      nhmin = args$nhmin, nhmax = args$nhmax,
      #      neffmin = args$neffmin, neffmax = args$neffmax,
      #      verbose = FALSE))} )
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
    ## Step 3: SCREENED IPCHD
    {
      print(paste0("Step ", sim_ind,".3: SCREENED IPCHD"))
      
      ##################
      ## Joint Estimation
      ##################
      .start.time = Sys.time()

      .input_cor_list   <- lapply(.Xlist, function(X) {
          output <- list(
            X = X, mat_type = "cor",
            mat = cor(X), var_inds = 1:(args$p))
          return(output)})
      .result_scr_mat      <- sapply(.input_cor_list, function(input_cor) {
        result_cor <- input_cor %>%
          c(list(true_mat = NULL, method = "max")) %>%
          do.call(sta_screened_mat, .) %>%
          c(list(overest_type = "frac")) %>%
          do.call(sta_ipchd, .)

        return(result_cor)
      }) %>% t()
      print(.result_scr_mat)

      .end.time = Sys.time()

      .cor_scr_time <- difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]]

    }
    #####################################################
    #####################################################
    ## Step 3: THRESHOLDED IPCHD
    {
      print(paste0("Step ", sim_ind,".4: THRESHOLDED IPCHD."))
        
      ##################
      ## Joint Estimation
      ##################
      .start.time = Sys.time()

      .input_cor_list   <- lapply(.Xlist, function(X) {
          output <- list(
            X = X, mat_type = "cor",
            mat = cor(X), var_inds = 1:(args$p))
          return(output)})
      .result_thr_mat      <- sapply(.input_cor_list, function(input_cor) {
        result_cor <- input_cor %>%
          c(list(true_mat = NULL, perc = 0.7)) %>%
          do.call(sta_thresholding_perc, .) %>%
          c(list(overest_type = "frac")) %>%
          do.call(sta_ipchd, .) 

        return(result_cor)
      }) %>% t()
      print(.result_scr_mat)
      
      .end.time = Sys.time()

      .cor_thr_time <- difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]]

    }
    #####################################################
    #####################################################
    ## Step 7: Saving Data:
    {
      ## ID data:
      .data_id_temp <- data.frame(
        TASK_ID   = rep(args$id_task, args$K * n_methods),
        SIM_NUM   = rep(sim_ind, args$K * n_methods),
        K_MAT_NUM = rep(1:args$K, n_methods),
        METHOD    = rep(method_names, rep(args$K, n_methods)),
        TIME      = rep(c(.cor_scr_time / args$K, .cor_thr_time / args$K), c(args$K, args$K)))
      print(.data_id_temp)

      ## Connectivity data:
      .data_deg_temp <- data.frame(rbind(
        .result_scr_mat, .result_thr_mat))
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
