############################################
############################################
## SETTING PARAMETERS:
wd <- getwd()
.libPaths(paste0(wd,"/req_lib"))

library(readr)
library(ggplot2)
library(tidyr)
library(tibble)
library(dplyr)
library(stringr)
library(lazyeval)
library(pROC)

input <- commandArgs(trailingOnly = TRUE)
diag_shift_val <- as.numeric(input[1])
T0_prop_val    <- as.numeric(input[2])


###################### Parameter table:
runtype       <- 3 # FOR FULL SIMULATIONS
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
  ph2             = c(0.05, 0.25, 0.5),
  ph1             = c(0.25, 0.5),
    
  nsim            = ifelse(runtype <= 2, 2, 5),
  diagonal_shift  = c(2,5),
  #n_prop          = c(0.25, 0.5, 0.75, 1),
  n_prop          = c(0.5, 0.75, 1, 1.25),
  T0_prop         = c(0.5, 0.75, 1),
  p               = c(100, 200, 300))
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
###################### Processing Data
##################################################################
##################################################################
method_names <- c(
    "GL.CORR.d",          ## GLASSO-methods
    "HWGL.CORR.d",        ## HWGL-methdos.
    "COR_Scr_IPCHD", 
    "COR_Thr_IPCHD",
    "ST.OVER.CORR.IM",
    "ST.OVER.THR.IM")
method_names_clean <- c(
    "GLASSO",          ## GLASSO-methods
    "HWGL",        ## HWGL-methdos.
    "IPC-HD: Screening",
    "IPC-HD: Threshold",
    "JIC-HD: Sample Cov",
    "JIC-HD: Thresholded Cov")


outputs_merged_list <- list()

for (p_val in c(100,200,300)) {
  
  ##############################
  ##############################
  ## LOADING ALL DATA WITH T0 = P.
  sim_ind_load    <- which(T0_prop == T0_prop_val & p == p_val & diagonal_shift == diag_shift_val)
  type            <- "all"
  results_dir     <- paste0(subfolder_new, "plots_", type, "/")

  for (sim_ind in sim_ind_load) {
    load(paste0(
      subfolder_new, "data_all/",
      "output", sim_ind, ".RData"))
  }

  ##############################
  ##############################
  ## TRUE POSITIVE RATE OF JIC-HD METHODS:

  ## Merge dataset of JIC-HD-derived data.
  output_merged_jic  <- distinct(bind_rows(mget(ls(pattern = '^output\\d+')))) %>%
    filter(METHOD %in% method_names[-c(1:4)]) %>%
    mutate(METHOD = str_replace_all(METHOD, setNames(method_names_clean, method_names))) %>%
    arrange(TASK_ID, SIM_NUM, K_MAT_NUM, METHOD) %>%
    select(-K_MAT_NUM, -TIME)
  output_merged_jic$microrun <- rep(1:10, nrow(output_merged_jic) / 10) 
  output_merged_jic <- output_merged_jic %>% 
    relocate(microrun, .after = 1)
  dim(output_merged_jic)
  head(output_merged_jic[,1:15], 15)
  colnames(output_merged_jic)

  ## For each row, we calculate the TPR/FPR:
  mat_jic <- t(apply(
    output_merged_jic, MARGIN = 1, 
    function(x) {
      nhubs     <- 5
      p_val     <- length(x) - 9
      id_task   <- x[1]
      args_temp <- get(gsub(" ", "", paste0("args", id_task, sep = "")))
      trueHubs  <- (1:p_val) %in% c(args_temp$Hjoint)
      nhubs     <- length(args_temp$Hjoint)

      vals      <- as.numeric(x[-(1:9)])
      vals_pos  <- vals

      tr_mean   <- mean(vals_pos)
      tr_sd     <- sd(vals_pos)

      hubshat   <- vals > tr_mean + 2 * tr_sd

      tp <- sum(hubshat & trueHubs) / (nhubs)
      fp <- sum(hubshat & !trueHubs) / (p_val - nhubs)

      return(c(tp, fp))
    }
  ))
  output_merged_jic$tp <- mat_jic[,1]
  output_merged_jic$fp <- mat_jic[,2]
  output_merged_jic <- output_merged_jic %>%
    dplyr::select(!starts_with("var"))

  dim(output_merged_jic)
  colnames(output_merged_jic)
  head(output_merged_jic)

  
  ##############################
  ##############################
  # TPR of GLASSO-methods.

  output_merged_gl   <- distinct(bind_rows(mget(ls(pattern = '^output\\d+')))) %>%
    filter(METHOD %in% method_names[c(1:4)]) %>%
    filter(K_MAT_NUM != 0) %>%
    mutate(METHOD = str_replace_all(METHOD, setNames(method_names_clean, method_names))) %>%
    arrange(TASK_ID, SIM_NUM, K_MAT_NUM, METHOD) %>%
    select(-TIME)
  output_merged_gl$microrun <- rep(1:10, nrow(output_merged_gl) / 10) ## adding micro-run identifier...
  output_merged_gl <- output_merged_gl %>% 
    relocate(microrun, .after = 1)
  dim(output_merged_gl)
  head(output_merged_gl[, 1:15], 10)

  ## Calculate hubs for each of the simulations ran
  hubsdata_gl <- t(apply(
    output_merged_gl, MARGIN = 1, 
    function(x) {
      nhubs     <- 5
      p_val     <- length(x) - 10
      id_task   <- x[1]
      args_temp <- get(gsub(" ", "", paste0("args", id_task, sep = "")))
      trueHubs  <- (1:p_val) %in% c(args_temp$Hjoint)
      nhubs     <- length(args_temp$Hjoint)
      
      vals      <- as.numeric(x[-c(1:10)])
      vals_pos  <- vals        
      
      tr_mean   <- mean(vals_pos)
      tr_sd     <- sd(vals_pos)

      hubshat   <- (vals > tr_mean + 2 * tr_sd) # 2.32 * tr_sd
      return(hubshat)

    }
  ))
    
  ## Aggregate hubs of K_MAT_NUM = 1,2,3 to obtain common hub estimation rate.
  colnames(hubsdata_gl) <- paste0("ishub", 1:p_val)
  output_merged_gl <- cbind(output_merged_gl, hubsdata_gl) %>%
    dplyr::select(!starts_with("var")) %>%
    group_by(TASK_ID, microrun, SIM_NUM, p, T0, n, ph1, ph2, METHOD) %>%
    summarise_at(vars(starts_with("ishub")),
      function(x) {1 * (sum(x) == 3)}) %>%
    ungroup()
  dim(output_merged_gl)
  head(output_merged_gl[, 1:15], 15)

  mat_gl <- t(apply(
    output_merged_gl, MARGIN = 1, 
    function(x) {
      nhubs     <- 5
      p_val     <- length(x) - 9
      id_task   <- x[1]
      args_temp <- get(gsub(" ", "", paste0("args", id_task, sep = "")))
      trueHubs  <- (1:p_val) %in% c(args_temp$Hjoint)
      nhubs     <- length(args_temp$Hjoint)

      hubshat   <- as.numeric(x[-(1:9)])

      tp <- sum(hubshat & trueHubs) / (nhubs)
      fp <- sum(hubshat & !trueHubs) / (p_val - nhubs)
      
      return(c(tp, fp))
    }
  ))
  output_merged_gl$tp <- mat_gl[,1]
  output_merged_gl$fp <- mat_gl[,2]
  output_merged_gl <- output_merged_gl %>%
    dplyr::select(!starts_with("ishub"))
  
  dim(output_merged_gl)
  head(output_merged_gl, 15)



  ##############################
  ##############################
  ## MERGING OUTPUTS:

  output_merged_p <- rbind(output_merged_gl, output_merged_jic)   
  outputs_merged_list[[length(outputs_merged_list) + 1]] <- output_merged_p


  rm(list = grep('^output\\d+', ls(), value = TRUE))
  rm(list = grep("args", ls(), value = TRUE))
  rm(
    output_merged_gl, output_merged_jic, output_merged_p,
    sim_ind_load, mat_gl, mat_jic, hubsdata_gl)
}

output_merged <- do.call("rbind", outputs_merged_list)
dim(output_merged)



##################################################################
##################################################################
###################### Generating Plot:
##################################################################
##################################################################



output_summarised <- output_merged %>%
  dplyr::select(!starts_with("var")) %>%
  pivot_longer(
    cols          = c("tp", "fp"),
    names_to      = "eval_par",
    values_to     = "eval") %>%
      
  group_by(METHOD, p, ph1 , ph2, T0, n, eval_par) %>%
  summarise(mean = mean(eval), sd = sd(eval))
    

## Plot 1: ph = 0.4
file_name <- paste0(
  subfolder_plots_new, 
  "521_MainTPR",
  "_d", diag_shift_val, 
  "_T0prop", T0_prop_val,
  ".pdf")
pdf(file_name, width = 8, height = 5)
## Plot 1: TPR 
p1 <-  output_summarised %>% filter(eval_par == "tp") %>%
  mutate(
    ph1_name = ifelse(ph1 == 0.5, "p[C] == 0.5", "p[C] == 0.25"),
    ph2_name = ifelse(ph2 == 0.5, "p[I] == 0.5", ifelse(ph2 == 0.25, "p[I] == 0.25", "p[I] == 0.05")),
    p_name = ifelse(p == 100, "p == 100", ifelse(p == 200, "p == 200", "p == 300")),
    METHOD   = factor(METHOD),
    TPR = mean) %>%
  ggplot(aes(x = n, y = TPR)) + 
  geom_line(aes(col = METHOD, linetype = METHOD), linewidth = 1) + 
  #geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd, fill = METHOD), alpha = 0.1) +
  geom_hline(yintercept = c(0,1), linetype = 2) +
  #facet_grid(rows = vars(ph2), cols = vars())
  facet_grid(ph2_name ~ p_name + ph1_name, scales = "free_x", labeller = label_parsed) +
  theme(legend.position="bottom")
print(p1)
dev.off()


## Plot 1: ph = 0.4
file_name <- paste0(
  subfolder_plots_new, 
  "521_MainFPR",
  "_d", diag_shift_val, 
  "_T0prop", T0_prop_val,
  ".pdf")
pdf(file_name, width = 8, height = 5)
## Plot 1: TPR 
p1 <-  output_summarised %>% filter(eval_par == "fp") %>%
  mutate(
    ph1_name = ifelse(ph1 == 0.5, "p[C] == 0.5", "p[C] == 0.25"),
    ph2_name = ifelse(ph2 == 0.5, "p[I] == 0.5", ifelse(ph2 == 0.25, "p[I] == 0.25", "p[I] == 0.05")),
    p_name = ifelse(p == 100, "p == 100", ifelse(p == 200, "p == 200", "p == 300")),
    METHOD   = factor(METHOD),
    FPR = mean) %>%
  ggplot(aes(x = n, y = FPR)) + 
  geom_line(aes(col = METHOD, linetype = METHOD), linewidth = 1) + 
  #geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd, fill = METHOD), alpha = 0.1) +
  geom_hline(yintercept = c(0,1), linetype = 2) +
  #facet_grid(rows = vars(ph2), cols = vars())
  facet_grid(ph2_name ~ p_name + ph1_name, scales = "free_x", labeller = label_parsed) +
  theme(legend.position="bottom")
print(p1)
dev.off()

