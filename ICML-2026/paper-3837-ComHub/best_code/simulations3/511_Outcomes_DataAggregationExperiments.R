
############################################
############################################
## SETTING PARAMETERS:
wd <- getwd()
.libPaths(paste0(wd,"/req_lib"))

library(readr)
library(magrittr)
library(ggplot2)
library(tidyr)
library(tibble)
library(dplyr)
library(stringr)
library(lazyeval)


###################### Parameter table:
runtype       <- 2 # FOR EXPERIMENT RUNS
#runtype       <- 3 # FOR FULL RUNS
index_old     <- 1 # run index to use
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
    
  nsim            = ifelse(runtype <= 2, 2, 5),
  diagonal_shift  = c(2,5),
  n_prop          = c(0.5, 0.75, 1, 1.25),
  T0_prop         = c(0.5, 0.75, 1),
  p               = c(100, 200, 400))
attach(sim_par_table)


###################### Creating folders:
subfolder_new        <- paste0("500_AggregatedDataExperiments/")
subfolder_data_new   <- paste0(subfolder_new, "data_all/")
subfolder_plots_new  <- paste0(subfolder_new, "plots_all/")

if (!dir.exists(subfolder_new)) {
       dir.create(subfolder_new)
}
if (!dir.exists(subfolder_data_new)) {
       dir.create(subfolder_data_new)
}
if (!dir.exists(subfolder_plots_new)) {
       dir.create(subfolder_plots_new)
}

##################################################################
##################################################################
## LOAD + MERGE RESULTS:
##################################################################
##################################################################
# Which simulations are we using?
##  RUNTYPE = 2: experiment partial runs.
##  RUNTYPE = 3: full simulation runs. 
run_info <- list(
  list(
    main_dir       = "100_glasso/",
    run_index      = 1,
    runtype        = runtype,
    abrev_name     = "glasso"),

  list(
    main_dir       = "200_hwgl/",
    run_index      = 1,
    runtype        = runtype,
    abrev_name     = "hwgl"),
    
  list(
    main_dir       = "300_ipchd/",
    run_index      = 1,
    runtype        = runtype,
    abrev_name     = "ipchd"),

  list(
    main_dir       = "400_jichd/",
    run_index      = 1,
    runtype        = runtype,
    abrev_name     = "stisph")
  )

###########################
## LOOP OVER ALL 288 SIMULATION PARAMETER COMBINATIONS
for (id_task in 1:144) {
  print(paste("XXXXXXXXXXXXXXXX ID-TASK", id_task))

  output <- NULL

  for(id_microrun in 0:9) {

    print(paste("XXXXXXXX MICRO-RUN", id_microrun))

    ###########################
    ## Merge all data, from all methods into output.
    for (method_ind in 1:length(run_info)) {
      main_dir     <- run_info[[method_ind]]$main_dir
      run_index    <- run_info[[method_ind]]$run_index
      runtype      <- run_info[[method_ind]]$runtype
      runtype_name <- c("pretrainings", "experiments", "outputs")[runtype]
      abrev_name   <- run_info[[method_ind]]$abrev_name

      load(paste0(
        main_dir, runtype_name, run_index,
        "/data/output", id_task, "_", id_microrun, ".RData"))

      ## Add id-task info:
      output_temp <- get(paste0("output", id_task, "_", id_microrun))
      args_temp   <- get(paste0("args", id_task))
      output_temp <- output_temp %>%
        mutate(
          p = args_temp$p, 
          T0 = args_temp$T0, n = args_temp$n,
          ph1 = args_temp$ph1, ph2 = args_temp$ph2, 
          .before = METHOD)
      
      ## Clean names:
      colnames(output_temp) <- gsub(" ", "", colnames(output_temp))
      
      ## Visualize current data:
      print(paste(
        abrev_name, nrow(output_temp), ncol(output_temp)))

      ## Add data to output:
      output <- rbind(
        output,
        output_temp)
      rm(output_temp, args_temp)
      rm(list = paste0("output", id_task, "_", id_microrun))
      rm(list = paste0("rho_min", id_task, "_", id_microrun))
    }
  }
  


  ###########################
  ## Saving data:
  print(paste("Saving data: Task-ID", id_task))
  print(names(output))
  print(ls())
  print(paste0(subfolder_data_new, "output", id_task, ".RData"))
  
  args <- get(paste0("args", id_task))
  print(paste(
      "TOTAL",
      nrow(output),
      ncol(output)))

  tmp.env <- new.env()
  assign(
    paste0("output", id_task),
    output,
    pos = tmp.env)
  assign(
    paste0("args", id_task),
    args,
    pos = tmp.env)
  save(
    list = ls(all.names = TRUE, pos = tmp.env), envir = tmp.env, 
    file = paste0(subfolder_data_new, "output", id_task, ".RData"))
  
  rm(output)
  rm(args)
  rm(list = paste0("args", id_task))
}

ls()
rm(list = ls())
