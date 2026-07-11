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
  n_prop          = c(0.5, 0.75, 1, 1.25),
  T0_prop         = c(0.5, 0.75, 1),
  p               = c(100, 200, 300))
attach(sim_par_table)


###################### Creating folders:
subfolder_new        <- paste0("600_AggregatedDataFull/")
subfolder_data_new   <- paste0(subfolder_new, "data_all/")
subfolder_plots_new  <- paste0(subfolder_new, "plots_all/")
subfolder_time_new   <- paste0(subfolder_new, "time_all/")

if (!dir.exists(subfolder_new)) {
       dir.create(subfolder_new)
}
if (!dir.exists(subfolder_data_new)) {
       dir.create(subfolder_data_new)
}
if (!dir.exists(subfolder_time_new)) {
       dir.create(subfolder_time_new)
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


output_merged <- NULL

for(p_val in c(100, 200, 300)) {
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
  ## Total time for individual methods
  output_merged_gl  <- distinct(bind_rows(mget(ls(pattern = '^output\\d+')))) %>%
    filter(METHOD %in% method_names[1:4], K_MAT_NUM != 0) %>%
    mutate(METHOD = str_replace_all(METHOD, setNames(method_names_clean, method_names))) %>%
    arrange(TASK_ID, SIM_NUM, K_MAT_NUM, METHOD) %>%
    select(-starts_with("var"))
  output_merged_gl$microrun <- rep(1:10, nrow(output_merged_gl) / 10) 
  
  output_merged_gl <- output_merged_gl %>%
    group_by(TASK_ID, SIM_NUM, microrun, METHOD, p, T0, n, ph1, ph2) %>%
    summarise(total_time = sum(TIME)) %>%
    group_by(TASK_ID, METHOD, p, T0, n, ph1, ph2) %>%
    summarize(MeanTime = mean(total_time), SdTime = sd(total_time))

  ##############################
  ##############################
  ## Total time for JIC-HD
  output_merged_jic  <- distinct(bind_rows(mget(ls(pattern = '^output\\d+')))) %>%
    filter(METHOD %in% method_names[-c(1:4)]) %>%
    mutate(METHOD = str_replace_all(METHOD, setNames(method_names_clean, method_names))) %>%
    arrange(TASK_ID, SIM_NUM, K_MAT_NUM, METHOD) %>%
    select(-starts_with("var"))
  output_merged_jic$microrun <- rep(1:10, nrow(output_merged_jic) / 10) 
    
  output_merged_jic <- output_merged_jic %>%
    group_by(TASK_ID, SIM_NUM, microrun, METHOD, p, T0, n, ph1, ph2) %>%
    summarise(total_time = sum(TIME)) %>%
    group_by(TASK_ID, METHOD, p, T0, n, ph1, ph2) %>%
    summarize(MeanTime = mean(total_time), SdTime = sd(total_time))

  output_merged <- rbind(output_merged, output_merged_gl, output_merged_jic)

}



## Plot 1: ph = 0.4
file_name <- paste0(
  subfolder_time_new, 
  "631_logtime",
  "_d", diag_shift_val, 
  "_T0prop", T0_prop_val,
  ".pdf")
pdf(file_name, width = 7, height = 7)
p1 <- output_merged %>%

  mutate(
    ph1_name = ifelse(ph1 == 0.5, "p[C] == 0.5", "p[C] == 0.25"),
    ph2_name = ifelse(ph2 == 0.5, "p[I] == 0.5", ifelse(ph2 == 0.25, "p[I] == 0.25", "p[I] == 0.05")),
    p_name   = ifelse(p == 100, "p == 100", ifelse(p == 200, "p == 200", "p == 300"))) %>%

  ggplot(aes(x = n, y = log(MeanTime, base = 60))) + 
    geom_line(aes(col = METHOD, linetype = METHOD), linewidth = 1) + 
    geom_point(aes(col = METHOD, shape = METHOD)) + 
    geom_hline(yintercept = c(0, 0.56, 1, 1.56, 2), linetype = 2, linewidth = 0.25) +
    ylab(expression(log[60]("seconds"))) +
    xlab("Sample Size n") +
    #annotate("text", x = p * 0.93, y = 0.15, label = "1 s", size = 2.5) + 
    #annotate("text", x = p * 0.93, y = 0.71, label = "10 s", size = 2.5) +
    #annotate("text", x = p * 0.93, y = 1.15, label = "1 m", size = 2.5) + 
    #annotate("text", x = p * 0.93, y = 1.71, label = "10 m", size = 2.5) + 
    #annotate("text", x = p * 0.93, y = 2.15, label = "1 h", size = 2.5) + 
    theme(legend.title = element_blank()) +
    facet_grid(ph2_name ~ p_name + ph1_name, scales = "free_x", labeller = label_parsed) +
    theme(legend.position="bottom")
print(p1)
dev.off()
  

