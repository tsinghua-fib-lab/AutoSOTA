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
#runtype       <- 2 # FOR EXPERIMENT RUNS
runtype       <- 3 # FOR FULL RUNS
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
    
  r2              = c(5),
  r1              = c(5, 10, 15),
  pneff           = c(0.01),
  pnh             = c(0.05),
  ph2min          = c(0.3, 0.5),
  ph1min          = c(0.3, 0.4, 0.5),
    
  nsim            = ifelse(runtype <= 2, 2, 10),
  diagonal_shift  = c(2),
  n_prop          = c(0.5, 0.75, 1, 1.25),
  T0_prop         = c(1),
  p               = c(100, 200, 400))
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
    "ST.2OVER.CORR.IM",
    "ST.2OVER.THR.IM")
method_names_clean <- c(
    "GLASSO",          ## GLASSO-methods
    "HWGL",        ## HWGL-methdos.
    "IPC-HD: Screening",
    "IPC-HD: Thresholding",
    "JIC-HD: Sample Cov",
    "JIC-HD: Thresholding")


output_merged <- NULL
diag_shift_val <- 2
r1_val <- 5

for(diag_shift_val in c(2)) {
for (p_val in c(100, 200, 400)) {
  ##############################
  ##############################
  ## LOADING ALL DATA WITH T0 = P.
  sim_ind_load    <- which(T0_prop == 1 & p == p_val & diagonal_shift == diag_shift_val, r1 == r1_val)
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
    filter(METHOD %in% method_names[1:4], K_MAT_NUM != 0) 
      
  output_merged_gl  <- output_merged_gl %>%
    select(-starts_with("var")) %>%
    mutate(METHOD = str_replace_all(METHOD, setNames(method_names_clean, method_names))) %>%
    arrange(TASK_ID, SIM_NUM, K_MAT_NUM, METHOD) 
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

cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")
gv <- guide_legend(nrow = 1, byrow = TRUE, title = "")

## Plot 1: ph = 0.4
file_name <- paste0(
  subfolder_time_new, 
  "631_logtime",
  "_d", diag_shift_val, 
  ".pdf")
pdf(file_name, width = 8, height = 5)


data_plot <- output_merged %>%
  filter(
    METHOD != "IPC-HD: Screening",
    METHOD != "JIC-HD: Thresholding"
  ) %>%
  mutate(
    METHOD = ifelse(METHOD == "IPC-HD: Threshold", "IPC-HD", METHOD),
    METHOD = ifelse(METHOD == "JIC-HD: Sample Cov", "JIC-HD", METHOD)) %>%

  mutate(
    ph1_name = ifelse(ph1 == 0.3, "p[C] %in% \"[0.3, 0.6]\"", ifelse(ph1 == 0.4, "p[C] %in% \"[0.4, 0.7]\"", "p[C] %in% \"[0.5, 0.8]\"")),
    ph2_name = ifelse(ph2 == 0.3, "p[I] == 0.3", ifelse(ph2 == 0.4, "p[I] == 0.4", "p[I] == 0.5")),
    p_name   = ifelse(p == 100, "p == 100", ifelse(p == 200, "p == 200", "p == 400")),
    METHOD = factor(METHOD))

print(unique(data_plot$METHOD))
print(colnames(data_plot))
print(unique(data_plot$n))

p1 <- data_plot %>%

  ggplot(aes(x = n, y = log(MeanTime, base = 60))) + 
    #geom_ribbon(aes(ymin = MeanTime - SdTime, ymax = MeanTime + SdTime, fill = METHOD), alpha = 0.3) +
    geom_line(aes(col = METHOD, linetype = METHOD), linewidth = 1) + 
    geom_point(aes(col = METHOD, shape = METHOD)) + 
    geom_hline(yintercept = c(0, 0.56, 1, 1.56, 2), linetype = 2, linewidth = 0.25) +
    ylab(expression(log[60]("seconds"))) +
    xlab("Sample Size n") +
    annotate("text", x = 25, y = 0.15, label = "     1 s", size = 2.5) + 
    annotate("text", x = 25, y = 0.71, label = "     10 s", size = 2.5) +
    annotate("text", x = 25, y = 1.15, label = "     1 m", size = 2.5) + 
    annotate("text", x = 25, y = 1.71, label = "     10 m", size = 2.5) + 
    annotate("text", x = 25, y = 2.15, label = "     1 h", size = 2.5) + 
    scale_linetype_manual(values = c(2, 3, 4, 1)) +
    scale_discrete_manual("linewidth", values = c(0.75, 0.75, 0.75, 1)) +
    scale_shape_manual(values = c(2, 5, 13, 19)) +
    scale_color_manual(values=c(cbPalette[c(2,4,8)], "#000000")) +
    scale_fill_manual(values=c(cbPalette[c(2,4,8)], "#000000")) +
      
    theme(legend.title = element_blank()) +
    facet_grid(ph1_name ~ p_name + ph2_name, scales = "free_x", labeller = label_parsed) +
    theme(legend.position="bottom") + 
    guides(colour = gv, shape = gv, size = gv, linetype = gv, fill = gv)
print(p1)
dev.off()

output_merged <- NULL
output_merged_gl <- NULL
output_merged_jic <- NULL

}

rm(list = ls())
