############################################
############################################
## SETTING PARAMETERS:
.libPaths("../req_lib")
library(readr)
library(plot.matrix)
library(ggplot2)
library(tidyr)
library(tibble)
library(dplyr)
library(stringr)
library(lazyeval)
library(magrittr)

plot_version <- 2

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}



###################### Parameter table:
runtype <- 2 # for threshold simulations
index   <- 1 # run index to use
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
    ph2             = c(0, 0.25, 0.5),
    ph1             = c(0.25, 0.5),
    
    nsim            = ifelse(runtype <= 2, 2, 10),
    diagonal_shift  = c(1,2,5),
    #n_prop          = c(0.25, 0.5, 0.75, 1),
    n_prop          = c(1.25, 1.5, 2, 3, 5),
    T0_prop         = c(0.5, 0.75, 1),
    p               = c(100, 200, 300, 1000))
sim_par_table$T0 <- sim_par_table$p * sim_par_table$T0_prop
attach(sim_par_table)

############################################
############################################

for(d_shift in c(1, 2, 5)) {
  for(p_val in c(100, 200, 300)) {
    
    ## Set simulation parameters:
    sim_ind_load    <- which(diagonal_shift == d_shift & p == p_val)
    runtype_name    <- c("pretraining", "experiments", "outputs")[runtype]
    run_index       <- index
  
    ## Create directory where results will be saved:
    results_dir     <- paste0(runtype_name, index, "/bic_plots/")
    if (!dir.exists(results_dir)){
      dir.create(results_dir)
    }

    ## Load data corresponding to specified parameters:
    for (.task_id in sim_ind_load) {
      for (.sim_ind in 0:9) {
        load(paste0(
          runtype_name, run_index, "/",
          "bic_data/",
          "output_bic", .task_id, "_", .sim_ind,".RData"))
      
      

        .args         <- get(paste0("args", .task_id))
        .output_bic   <- get(paste0("output_bic", .task_id, "_", .sim_ind)) %>%
          pivot_longer(
            cols = - c(SIM_NUM, K_MAT_NUM, TASK_ID),
            names_to = "val_ind",
            values_to = "val") %>%
    
          mutate(
            val_type    = str_sub(val_ind, 1, 3),
            val_ind     = as.numeric(str_remove(val_ind, "(BIC|rho)"))) %>%
    
          pivot_wider(
            names_from = "val_type",
            names_prefix = "",
            values_from = "val") %>%

          mutate(
            .before = SIM_NUM,
            p = .args$p, ph1 = .args$ph1, ph2 = .args$ph2,
            pnh = .args$pnh, pneff = .args$pneff,
            T0 = .args$T0, n = .args$n)
      
        assign(paste0("output_bic", .task_id, "_", .sim_ind), .output_bic)  
      }
      assign(
        paste0("output_bic", .task_id), 
        distinct(bind_rows(mget(ls(pattern = paste0("^output_bic",.task_id,"_\\d+")))) %>%
        filter(p == p_val)))
        
        rm(list = paste0("output_bic", .task_id, "_", 0:9))
        rm(list = paste0("rho_vals", .task_id, "_", 0:9))
        rm(list = paste0("args", .task_id))

    }

    output_merged   <- distinct(bind_rows(mget(ls(pattern = '^output_bic\\d+')))) %>%
        filter(p == p_val)

    results_dir <- paste0(
        runtype_name, index, "/")
    file_name <- paste0(
        results_dir, "bic_plots/", 
        "v", plot_version, "_bic_p", p_val, "_delta_", d_shift,"_test_plot", ".pdf")
    pdf(file_name, width = 7.5, height = 15)
    
    
    T0_val = unique(output_merged$T0)[1]
    p1 <- output_merged %>% 
        filter(T0 == T0_val, SIM_NUM < 11) %>% 
        ggplot(aes(x = log(rho, 10), y = BIC)) +
                geom_line(aes(col = factor(K_MAT_NUM), linetype = factor(ph2))) + 
                facet_grid(rows = vars(ph1), cols = vars(n)) +
                ggtitle(paste0("T0 = ", T0_val))
        
    T0_val = unique(output_merged$T0)[2]
    p2 <- output_merged %>% 
        filter(T0 == T0_val, SIM_NUM < 11) %>%
        ggplot(aes(x = log(rho, 10), y = BIC)) +
                geom_line(aes(col = factor(K_MAT_NUM), linetype = factor(ph2))) + 
                facet_grid(rows = vars(ph1), cols = vars(n)) +
                ggtitle(paste0("T0 = ", T0_val))
        
    T0_val = unique(output_merged$T0)[3]
    p3 <- output_merged %>% 
        filter(T0 == T0_val, SIM_NUM < 11) %>%
        ggplot(aes(x = log(rho, 10), y = BIC)) +
                geom_line(aes(col = factor(K_MAT_NUM), linetype = factor(ph2))) + 
                facet_grid(rows = vars(ph1), cols = vars(n)) +
                ggtitle(paste0("T0 = ", T0_val))
    multiplot(p1, p2, p3, rows = 3)
    
    dev.off()

    rm(list = grep("output_bic", ls(), value = TRUE))
    rm(list = grep("args", ls(), value = TRUE))
    rm("p1", "p2", "p3")


  }
}

rm(list = ls())
