source("02_GeneratingMultipleMatrixSparse.R")
source("03_UsefulMatrixTransforms.R")
source("04_SpectralCurvesPlots.R")
source("05_MatrixPlot.R")


library(magrittr)




###############################################################
###############################################################
###############################################################
###############################################################
## Figure 1 plot:
library(plot.matrix)
set.seed(100)
p <- 40
cex <- 0.75
Hjoint <- 10
Hind <- list(20 + c(1), 30 + c(1))
mat_list <- .rhubmat_list(p, p, 2, 
                          Hjoint = Hjoint, Hind_list = Hind,
                          ph1 = 0.5, ph2 = 0.4, pnh = 0.05, pneff = 0.05, 
                          shuffle = FALSE, type = "unif", 
                          hmin1 = 5, hmax1 = 7,
                          hmin2 = 5, hmax2 = 7,
                          nhmin = 4, nhmax = 5,
                          neffmin = 4, neffmax = 5)
min_eigen <- lapply(mat_list[[1]], function(x) min(eigen(x)$values)) %>%
  unlist() %>% min()
pm_list <- lapply(mat_list[[1]], 
                  function(x) {x + (5 - min_eigen) * diag(p)})



ic_list <-  lapply(pm_list, .PMtoIC)

rowinds <- function(p) {
  return(matrix(rep(1:p, p), ncol = p))
}
colinds <- function(p) {
  return(matrix(rep(1:p, p), ncol = p, byrow = TRUE))
}

range <- lapply(ic_list, function(x) {abs(x - diag(diag(x)))} ) %>%
  lapply(max) %>% unlist() %>% max()
## Fig 1:
par(mfrow = c(2,4), mar = c(5.1, 4.1, 4.1, 3.1))  
for (k in 1:2) {
  plot.pdm(ic_list[[k]],
           main = paste0("Precision Matrix: Pop ", k), cex = cex,
           entryrange = range)    
  
  plot.pdm(
    ic_list[[k]] *
      (rowinds(p) %in% c(Hjoint) | colinds(p) %in% c(Hjoint)), 
    main = "Hubs: Low rank", cex = cex,
    entryrange = range)
  
  plot.pdm(
    ic_list[[k]] *
      (rowinds(p) %in% c(Hind[[k]]) | colinds(p) %in% c(Hind[[k]])), 
    main = "Hubs: Low rank", cex = cex,
    entryrange = range)
  
  plot.pdm(
    ic_list[[k]] * 
      !(rowinds(p) %in% c(Hjoint, Hind[[k]]) | colinds(p) %in% c(Hjoint, Hind[[k]])),
    main = "Remaining Connections", cex = cex,
    entryrange = range)
  
}

par(mfrow = c(1,1))
theta_j <- ic_list[[k]] *
  (rowinds(p) %in% c(Hjoint) | colinds(p) %in% c(Hjoint))
v <- eigen(theta_j)$vectors[,1]

plot(v, pch = 19, xlab = "Variable Index", ylab = "Eigenvector Entry",
     ylim = c(1.2* min(v),1.2* max(v)))
abline(v = Hjoint, col = "red", lty = 2)
abline(h = 0)

