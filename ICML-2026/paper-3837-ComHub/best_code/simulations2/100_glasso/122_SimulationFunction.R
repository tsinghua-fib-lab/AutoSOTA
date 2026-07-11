
#####################################################
#####################################################
#####################################################

FullSimulation <- function(args, index) {

  
  #####################################################
  ## Runtype 2, 3:
  ## Define output saving object.
  #####################################################
  
  method_names <- c(
    "GL.CORR.d", "GL.CORR.ad", "GL.CORR.an")  ## HWGL-methdos.
  n_methods <- length(method_names)

  ## Output of simulation:
  output_sim <- NULL
  
  ## BIC output:
  output_bic <- NULL

  #################################################
  #################################################
  ## BIC output setting:
  
  n_rho <- 10
  ind_tuning <- 10^(seq(from = -4, to = 2, length.out = n_rho))

  #################################################
  #################################################
  ## Cycle:
  loop_start_time <- Sys.time()
  sim_ind   <- 1
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

      .ns       <- rep(args$n, args$K)

    }

    #####################################################
    #####################################################
    ## Step 3: Apply individual GLASSO.
    {
      print(paste0("Step ", sim_ind,".6: Individual GLASSO."))

      ######################################################
      ## Calculating all individual matrices GLASSO outputs:
      GL_list <- lapply(
        1:args$K,
        function(.k) {
          ##################
          ## Individual
          ## Calculations:
          ##################
          .start.time = Sys.time()

          .max_entry <- max(abs(.scorlist[[.k]][upper.tri(.scorlist[[.k]], FALSE)] ))
          .gl_rho <- .max_entry * ind_tuning 
          print(.gl_rho)
          print(dim(.scorlist[[.k]]))
          .gl_obj <- BICglasso(
            mat = .scorlist[[.k]], rho = .gl_rho,
            p = args$p, n = args$n,
            penalize.diagonal = FALSE)
          
          .gl_mat <- (.gl_obj$optimal.model)$wi
          .gl_bic <- .gl_obj$BIC
          
          .end.time = Sys.time()
          print("done")
          output <- list(
            gl_mat = .gl_mat,
            gl_adj = (abs(.gl_mat) > 1e-10),
            gl_bic = .gl_bic,
            gl_rho = .gl_rho,
            time = difftime(time1 = .end.time, time2 = .start.time, units = "secs")[[1]])
          return(output)
        }
      )

      ######################################################
      ## Saving "intersected" joint outcomes:
      GL_joint  <- 
        ((GL_list[[1]]$gl_adj != 0) & 
        (GL_list[[2]]$gl_adj != 0) & 
        (GL_list[[3]]$gl_adj != 0)) * 1
      .gl_degs  <- .degrees(GL_joint)
      .gl_time  <- GL_list[[1]]$time + GL_list[[2]]$time + GL_list[[3]]$time
      
      ## ID data:
      .data_id_temp <- data.frame(
        TASK_ID   = args$id_task,
        SIM_NUM   = sim_ind,
        K_MAT_NUM = 0,
        METHOD    = "GL.CORR.d",
        TIME      = .gl_time)

      ## Connectivity data:
      .data_deg_temp <- data.frame(rbind(
        .gl_degs))
      colnames(.data_deg_temp) <- paste0("var", 1:args$p)

      ## Saving data:
      .data_temp <- cbind(.data_id_temp, .data_deg_temp)
      output_sim <- rbind(output_sim, .data_temp)

      ######################################################
      ## Saving individual degree/weighted degree outcomes!
      for (.k in 1:args$K) {
        ##################
        ## Saving individual 
        ## data separately:
        ##################

        .gl_degs    <- .degrees(GL_list[[.k]]$gl_mat)
        .gl_dalpha  <- .dalphavals(GL_list[[.k]]$gl_mat)
        .gl_nalph   <- .nalphavals(GL_list[[.k]]$gl_mat)

        ## ID data:
        .data_id_temp <- data.frame(
          TASK_ID   = rep(args$id_task, 3),
          SIM_NUM   = rep(sim_ind, 3),
          K_MAT_NUM = rep(.k, 3),
          METHOD    = method_names,
          TIME      = rep((GL_list[[.k]]$time)[[1]], 3))

        ## Connectivity data:
        .data_deg_temp <- data.frame(rbind(
          .gl_degs,
          .gl_dalpha,
          .gl_nalph))
        colnames(.data_deg_temp) <- paste0("var", 1:args$p)

        ## Saving data:
        .data_temp <- cbind(.data_id_temp, .data_deg_temp)
        output_sim <- rbind(output_sim, .data_temp)

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
        .data_bic_temp <- c(GL_list[[.k]]$gl_bic, GL_list[[.k]]$gl_rho)
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
    output_sim = output_sim,
    output_bic = output_bic)
  print(output_sim)
  print(output_bic)

  return(output)

}
