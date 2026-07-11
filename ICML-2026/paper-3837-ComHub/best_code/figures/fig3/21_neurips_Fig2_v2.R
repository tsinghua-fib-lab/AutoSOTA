source("02_GeneratingMultipleMatrixSparse.R")
source("03_UsefulMatrixTransforms.R")
source("04_SpectralCurvesPlots.R")
source("05_MatrixPlot.R")

source("12_SGDspheremulti.R")
source("13_SGDstiefel.R")

library(magrittr)


source("05_MatrixPlot.R")



###############################################################
###############################################################
###############################################################
###############################################################

library(plot.matrix)
set.seed(100)
p <- 40
cex <- 0.75
Hjoint <- 10
#Hind <- list(20 + c(1), 30 + c(1))
Hind <- list(NULL, NULL, NULL)
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


#ic_list <-  pm_list 
ic_list <-  lapply(pm_list, .PMtoIC)

range <- lapply(ic_list, function(x) {abs(x - diag(diag(x)))} ) %>%
  lapply(max) %>% unlist() %>% max()


## Fig 0:
par(mfrow = c(1,3), mar = c(5.1, 4.1, 4.1, 3.1))  
for (k in 1:2) {
  plot.pdm(ic_list[[k]],
           main = paste0("Precision Matrix: Pop ", k), cex = cex,
           entryrange = range)    
  
  #plot.pdm(
  #  ic_list[[k]] *
  #    (rowinds(p) %in% c(Hjoint, Hind[[k]]) | colinds(p) %in% c(Hjoint, Hind[[k]])), 
  #  main = "Hubs: Low rank", cex = cex,
  #  entryrange = range)
  
  #plot.pdm(
  #  ic_list[[k]] * 
  #    !(rowinds(p) %in% c(Hjoint, Hind[[k]]) | colinds(p) %in% c(Hjoint, Hind[[k]])),
  #  main = "Remaining Connections", cex = cex,
  #  entryrange = range)
  
  plot(eigen(ic_list[[k]])$values, #[1:floor(p/2)],
       pch = 19,
       ylab = "Eigenvalues",
       main = "P. M. Eigenvalues")
  
  ic_list[[k]] %>% eigen() %>%
    {sign(.$vectors[Hjoint,1]) * .$vectors[,1]} %>%
    #{.[,1:2]^2} %>%
    #apply(1, sum) %>%
    plot(main = "Leading Eigenvector",
         ylab = "Vector Entries")
  abline(h = 0)
  abline(v = Hjoint, col = "red", lty = 2)
}



###############################################################
###############################################################
###############################################################
###############################################################

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


#ic_list <-  pm_list 
ic_list <-  lapply(pm_list, .PMtoIC)

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









###############################################################
###############################################################
###############################################################
###############################################################


library(plot.matrix)
set.seed(100)
p <- 40
cex <- 0.75
Hjoint <- 10 
Hind <- list(20 + c(1), 30 + c(1))
#Hind <- list(NULL, NULL, NULL)
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


#ic_list <-  pm_list 
ic_list <-  lapply(pm_list, .PMtoIC)

range <- lapply(ic_list, function(x) {abs(x - diag(diag(x)))} ) %>%
  lapply(max) %>% unlist() %>% max()

jichd <- sgd.stiefel(
  ic_list, p = p, ndir = 1, K = 2, 
  type = "M", nstarts = 5, 
  alpha = 0.1, max.iter = 3000)


par(mfrow = c(1,3), mar = c(5.1, 4.1, 4.1, 3.1))  
for (k in 1:2) {
  plot.pdm(ic_list[[k]],
           main = paste0("Precision Matrix: Pop ", k), cex = cex,
           entryrange = range)    
  
  #plot.pdm(
  #  ic_list[[k]] *
  #    (rowinds(p) %in% c(Hjoint, Hind[[k]]) | colinds(p) %in% c(Hjoint, Hind[[k]])), 
  #  main = "Hubs: Low rank", cex = cex,
  #  entryrange = range)
  
  #plot.pdm(
  #  ic_list[[k]] * 
  #    !(rowinds(p) %in% c(Hjoint, Hind[[k]]) | colinds(p) %in% c(Hjoint, Hind[[k]])),
  #  main = "Remaining Connections", cex = cex,
  #  entryrange = range)
  
  #plot(eigen(ic_list[[k]])$values[1:floor(p/2)])
  
}



barplot(jichd$vectors^2,
     ylab = "Joint Minimax Influence Measures",
     xlab = "Variable Index",
     main = "Joint Minimax Eigenvector",
     type = "",
     col = ifelse(1:p %in% c(Hjoint), "red", "grey80"))
abline(v = Hjoint+ 1.5, col = "red", lty = 2)


