
#####################################################
#####################################################
#####################################################

load_pretraining <- function(args, index) {
  pretraining_subfolder <- paste0(main_folder, "pretrainings", index, "/data/")
  print(paste("ID-Task:", args$id_task))

  pretraining_file <- paste0(
    pretraining_subfolder,
    "output", args$id_task, ".RData")
    
  print(paste0("pretraining file: ", pretraining_file))
  load(pretraining_file)

  output <- list()

  hwgl_corr <- get(paste0("output", args$id_task))
  output$tp_hwgl_corr <- hwgl_corr$pretuning_par_HWGL_corr
  output$pretuning_time_HWGL_corr <- hwgl_corr$pretuning_time_HWGL_corr

  return(output)
}

#####################################################
#####################################################
#####################################################

FullSimulation <- function(args, index) {

  #####################################################
  ## Runtype 2, 3:
  ## Load pre-training to environment.
  #####################################################
  
  pretraining_vals <- load_pretraining(args, 1)
  list2env(pretraining_vals, globalenv())

  #####################################################
  ## Runtype 2, 3, 4:
  ## Define output saving object.
  #####################################################
  
  method_names <- c(
    "HWGL.CORR.d", "HWGL.CORR.ad", "HWGL.CORR.an")  ## HWGL-methdos.
  n_methods <- length(method_names)

  ## Output of simulation:
  output_sim <- NULL
  
  ## BIC output:
  output_bic <- NULL

  #################################################
  #################################################
  ## BIC output setting:
  
  n_rho <- 10
  
  #ind_tuning <- list( ## First try! HWGL
  #  10^(seq(from = -2, to = 0, length.out = n_bic)), 
  #  10^(seq(from = -2.33, to = -0.33, length.out = n_bic)),
  #  10^(seq(from = -2.66, to = -0.66, length.out = n_bic)),
  #  10^(seq(from = -3,    to = -1, length.out = n_bic)))
  ind_tuning <- 10^(seq(from = -4, to = 2, length.out = n_rho))

  # tuning_hwgl   <- tp_hwgl_corr * ind_tuning[[4 * args$n_prop]]
  tuning_hwgl   <- tp_hwgl_corr * ind_tuning
  rho_min_vec   <- rep(0, n_rho)

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

      .ns       <- rep(args$n, args$K)

    }
    #####################################################
    #####################################################
    ## Step 3: Apply individual GLASSO.
    {
      print(paste0("Step ", sim_ind,".6: Individual GLASSO."))

      for (.k in 1:args$K) {
        
        ##################
        ## Individual
        ## Calculations:
        ##################
        .start.time = Sys.time()

        #.tuning_gl  <- ind_tuning[[4 * args$n_prop]]
        print(tuning_hwgl)
        print(dim(.scorlist[[.k]]))
        .hwgl_obj   <- BIChwglasso(
          mat = .scorlist[[.k]], rho = tuning_hwgl,
          p = args$p, n = args$n,
          penalize.diagonal = FALSE)
      
          
        .hwgl_mat <- .hwgl_obj$wi

        .hwgl_degs    <- .degrees((.hwgl_obj$optimal.model)$wi)
        .hwgl_dalpha  <- .dalphavals((.hwgl_obj$optimal.model)$wi)
        .hwgl_nalph   <- .nalphavals((.hwgl_obj$optimal.model)$wi)

        .end.time = Sys.time()

        ##################
        ## Saving individual 
        ## data separately:
        ##################
        

        ## ID data:
        .data_id_temp <- data.frame(
          TASK_ID   = rep(args$id_task, 3),
          SIM_NUM   = rep(sim_ind, 3),
          K_MAT_NUM = rep(.k, 3),
          METHOD    = method_names,
          TIME      = rep(difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]], 3))

        ## Connectivity data:
        .data_deg_temp <- data.frame(rbind(
          .hwgl_degs,
          .hwgl_dalpha,
          .hwgl_nalph))
        colnames(.data_deg_temp) <- paste0("var", 1:args$p)

        ## Saving data:
        .data_temp <- cbind(.data_id_temp, .data_deg_temp)
        output_sim <- rbind(output_sim, .data_temp)

        rm(.data_deg_temp, .data_id_temp, .data_temp)

        ##################
        ## Saving individual
        ## BIC data separately:
        ##################
        
        ## ID data:
        .data_id_temp <- data.frame(
          TASK_ID   = args$id_task,
          SIM_NUM   = sim_ind,
          K_MAT_NUM = .k)

        ## BIC data:    
        .data_bic_temp <- c(.hwgl_obj$BIC, tuning_hwgl)
        names(.data_bic_temp) <- paste0(rep(c("BIC", "rho"), c(n_rho, n_rho)), rep(1:n_rho, 2))
        
        .data_temp    <- cbind(.data_id_temp, t(.data_bic_temp))
        output_bic   <- rbind(output_bic, .data_temp)

        rm(.data_bic_temp, .data_id_temp, .data_temp)
        
      }
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
