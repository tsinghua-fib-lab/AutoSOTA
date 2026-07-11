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
runtype       <- 2 # FOR EXPERIMENTS
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
  ph1min          = c(0.3, 0.5),
    
  nsim            = ifelse(runtype <= 2, 1, 5),
  diagonal_shift  = c(2),
  n_prop          = c(0.5, 0.75, 1),
  T0_prop         = c(1),
  p               = c(200, 400))
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
    "ST.1OVER.CORR.IM", "ST.2OVER.CORR.IM", "ST.3OVER.CORR.IM")
    #"ST.ORAC.THR.IM", 
    #"ST.1OVER.THR.IM", "ST.2OVER.THR.IM", "ST.3OVER.THR.IM")
method_names_clean <- c(
    "JIC-HD 1", "JIC-HD 2", "JIC-HD 3")
    #"Threshold: Oracle",
    #"Threshold: Overest 1", "Threshold: Overest 2", "Threshold: Overest 3")


outputs_merged_list <- list()
sd_const <- 2

for (diag_shift_val in c(2)) {
  for (p_val in c(200,400)) {
  
    ##############################
    ##############################
    ## LOADING ALL DATA WITH T0 = P.
    sim_ind_load    <- which(T0_prop == 1 & p == p_val & diagonal_shift == diag_shift_val)
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
      filter(METHOD %in% method_names) %>% 
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
        p_val     <- length(x) - 10
        id_task   <- x[1]
        args_temp <- get(gsub(" ", "", paste0("args", id_task, sep = "")))
        trueHubs  <- (1:p_val) %in% c(args_temp$Hjoint)
        nhubs     <- length(args_temp$Hjoint)

        vals      <- as.numeric(x[-(1:10)])
        vals_pos  <- vals

        tr_mean   <- mean(vals_pos)
        tr_sd     <- sd(vals_pos)

        hubshat   <- vals > tr_mean + sd_const * tr_sd

        tp <- sum(hubshat & trueHubs) / (nhubs)
        fp <- sum(hubshat & !trueHubs) / (p_val - nhubs)
        fn <- sum(!hubshat & trueHubs) / (nhubs)
        prec <- ifelse(
          sum(hubshat) == 0,
          0,
          sum(hubshat & trueHubs) / (sum(hubshat)))
        rcll <- sum(hubshat & trueHubs) / (nhubs)
        fscr <- ifelse(
          (prec == 0) || (rcll == 0) ,
          0, 
          2 * prec * rcll / (prec + rcll) )

        return(c(tp, fp, fn, prec, rcll, fscr))
      }
    ))
    output_merged_jic$tp <- mat_jic[,1]
    output_merged_jic$fp <- mat_jic[,2]
    output_merged_jic$fn <- mat_jic[,3]
    output_merged_jic$prec <- mat_jic[,4]
    output_merged_jic$rcll <- mat_jic[,5]
    output_merged_jic$fscr <- mat_jic[,6]
    output_merged_jic <- output_merged_jic %>%
     dplyr::select(!starts_with("var"))

    dim(output_merged_jic)
    colnames(output_merged_jic)
    head(output_merged_jic)

  
    
    ##############################
    ##############################
    ## MERGING OUTPUTS:

    output_merged_p <- output_merged_jic
    outputs_merged_list[[length(outputs_merged_list) + 1]] <- output_merged_p


    rm(list = grep('^output\\d+', ls(), value = TRUE))
    rm(list = grep("args", ls(), value = TRUE))
    rm(
      output_merged_jic, output_merged_p,
      sim_ind_load, mat_jic)
  }
}

output_merged <- do.call("rbind", outputs_merged_list)
dim(output_merged)
output_merged$METHOD %>% unique()


##################################################################
##################################################################
###################### Generating Plot:
##################################################################
##################################################################

cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")

output_summarised <- output_merged %>%
  dplyr::select(!starts_with("var")) %>%
  pivot_longer(
    cols          = c("tp", "fp", "fn", "prec", "rcll", "fscr"),
    names_to      = "eval_par",
    values_to     = "eval") %>%
      
  group_by(METHOD, p, ph1 , ph2, T0, n, nhubs, eval_par) %>%
  summarise(mean = mean(eval), sd = sd(eval))
    
T0_prop_val <- 1

gv <- guide_legend(nrow = 2, byrow = TRUE, title = "")
## Plot 1: ph = 0.4
file_name <- paste0(
  subfolder_plots_new, 
  "525_Sens.pdf")
pdf(file_name, width = 5, height = 4.5)
## Plot 1: TPR 
p1 <-  output_summarised %>%
  filter(
    ph1 == 0.3,
    ph2 == 0.3, 
    p  == 400,
    eval_par %in% c("tp", "fp")) %>%
  mutate(
    nhubs_name = ifelse(nhubs == 5, "H[c] == 5", ifelse(nhubs == 10, "H[c] == 10", "H[c] == 15")),
    nhubs_name = factor(nhubs_name, levels = c("H[c] == 5", "H[c] == 10", "H[c] == 15"), ordered = TRUE),
    METHOD     = factor(METHOD, levels = c("JIC-HD 1", "JIC-HD 2", "JIC-HD 3"), ordered = TRUE),
    eval_par   = ifelse(eval_par == "tp", "TPR", ifelse(eval_par == "fp", "FPR", "F-score")),
    eval_par  = factor(eval_par, levels = c("FPR", "TPR", "F-score"), ordered = TRUE)  
  ) %>%
  ggplot(aes(x = n, y = mean)) + 
    geom_line(aes(col = METHOD, linetype = METHOD), linewidth = 1) + 
    geom_point(aes(col = METHOD, shape = METHOD), size = 2.2, alpha = 1) +
    geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd, fill = METHOD), alpha = 0.3) +
    geom_hline(yintercept = c(0,1), linetype = 2) +
    scale_color_manual(
      values = c(
        "JIC-HD 1" = cbPalette[2],
        "JIC-HD 2" = cbPalette[4],
        "JIC-HD 3" = cbPalette[8]),
      labels = c(
        expression("JIC-HD: " * hat(s) == sqrt(p) / 2),
        expression("JIC-HD: " * hat(s) == sqrt(p)),
        expression("JIC-HD: " * hat(s) == 3 * sqrt(p) / 2) ) ) +
    scale_linetype_manual(
      values = c(
        "JIC-HD 1" = 2,
        "JIC-HD 2" = 3,
        "JIC-HD 3" = 4),
      labels = c(
        expression("JIC-HD: " * hat(s) == sqrt(p) / 2),
        expression("JIC-HD: " * hat(s) == sqrt(p)),
        expression("JIC-HD: " * hat(s) == 3 * sqrt(p) / 2) ) ) +
    scale_shape_manual(
      values = c(
        "JIC-HD 1" = 2,
        "JIC-HD 2" = 5,
        "JIC-HD 3" = 13),
      labels = c(
        expression("JIC-HD: " * hat(s) == sqrt(p) / 2),
        expression("JIC-HD: " * hat(s) == sqrt(p)),
        expression("JIC-HD: " * hat(s) == 3 * sqrt(p) / 2) ) ) +
    scale_fill_manual(
      values = c(
        "JIC-HD 1" = cbPalette[2],
        "JIC-HD 2" = cbPalette[4],
        "JIC-HD 3" = cbPalette[8]),
      labels = c(
        expression("JIC-HD: " * hat(s) == sqrt(p) / 2),
        expression("JIC-HD: " * hat(s) == sqrt(p)),
        expression("JIC-HD: " * hat(s) == 3 * sqrt(p) / 2) ) ) +
    #facet_grid(rows = vars(ph2), cols = vars())
    facet_grid(eval_par ~ nhubs_name, scales = "free_y", labeller = label_parsed) +
    theme(legend.position="bottom")  + 
    guides(
      color = guide_legend(nrow = 2, byrow = TRUE, title = ""),
      linetype = guide_legend(nrow = 2, byrow = TRUE, title = ""),
      shape = guide_legend(nrow = 2, byrow = TRUE, title = ""),
      fill = guide_legend(nrow = 2, byrow = TRUE, title = ""))
    #guides(colour = gv, shape = gv, linetype = gv, fill = gv)
  print(p1)
  dev.off()




