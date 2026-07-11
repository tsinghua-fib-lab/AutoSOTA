source("02_GeneratingMultipleMatrixSparse.R")
source("03_UsefulMatrixTransforms.R")
source("04_SpectralCurvesPlots.R")
source("05_MatrixPlot.R")

source("12_SGDspheremulti.R")
source("13_SGDstiefel.R")


#######################################################
#######################################################
## Checking inverse correlation: 

args = list(p = 100, t = 100, K = 3, r = 5, 
            Hjoint = 1:5, Hind = list(21:23, 41:43, 61:63, 81:83), 
            ph1 = 0.4, ph2 = 0.4, pnh = 0.05, 
            diagonal_shift = 2, 
            shuffle = FALSE, type = "unif", 
            hmin1 = 4, hmax1 = 5,
            hmin2 = 4, hmax2 = 5, 
            nhmin = 4, nhmax = 5, 
            verbose = FALSE)

## Step 2: We create the covariance/precision matrices of the model.
Sigmalist = list()
Thetalist = list()
Rholist = list()
IClist = list()
maxentry    = 0
maxentry.IC = 0
maxentry.PM = 0
par(mfrow = c(3,2))
for(.k in 1:args$K){
  Thetalist[[.k]] = r.sparse.pdhubmat(p = args$p, t = args$t, 
                                      H1 = args$Hjoint, H2 = args$Hind[[.k]], 
                                      ph1 = args$ph1, ph2 = args$ph2, pnh = args$pnh, 
                                      diagonal_shift = args$diagonal_shift, 
                                      shuffle = args$shuffle, type = args$type, 
                                      hmin1 = args$hmin1, hmax1 = args$hmax1,
                                      hmin2 = args$hmin2, hmax2 = args$hmax2, 
                                      nhmin = args$nhmin, nhmax = args$nhmax, 
                                      verbose = FALSE)
  Sigmalist[[.k]] = solve(Thetalist[[.k]])
  Rholist[[.k]] = .COVtoCOR(Sigmalist[[.k]])
  IClist[[.k]] = .COVtoIC(Sigmalist[[.k]])
  
  maxentry.IC = max(maxentry, max(abs(IClist[[.k]]- diag(diag(IClist[[.k]])))))
  maxentry.PM = max(maxentry, max(abs(Thetalist[[.k]]- diag(diag(Thetalist[[.k]])))))
  
}


## Extra step: generate plots.

TH = c("st","nd","rd")


output.stiefel1 = sgd.stiefel(sigmalist = Rholist, p = args$p, ndir = 5, 
                              K = args$K, type = "M", 
                              nstarts = 5, alpha = 0.2, max.iter = 2000)

##########################################################
##########################################################
##########################################################
## Modified version of figure.
#plot(1:8, col = cbPalette)

colsH = cbPalette[c(3, 4, 8, 2)]
pchH = c(8,9,12)
colinds = rep(cbPalette[1], args$p)
colinds[args$Hjoint] = cbPalette[7]
pchinds = rep(1, args$p)
pchinds[args$Hjoint] = 19

colinds[args$Hind[[1]]] = colsH[1]
colinds[args$Hind[[2]]] = colsH[2]
colinds[args$Hind[[3]]] = colsH[3]
pchinds[args$Hind[[1]]] = pchH[1]
pchinds[args$Hind[[2]]] = pchH[2]
pchinds[args$Hind[[3]]] = pchH[3]

layout(mat = matrix(c(1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,5,5,5), ncol = 9, byrow = TRUE))
par(mar= c(3.1, 4.1, 2.6, 1.6))
for(.k in 1:args$K){
  plot(1,1,
       col = "white",
       xlim = c(1, args$p),
       ylim = c(1, args$p),
       xlab = "Row Index",
       ylab = "Column Index",
       xaxt = "none",
       yaxt = "none",
       main = paste0("Population ", .k))
  axis(2, at = c(1, seq(from = 0, to = args$p, length.out = 6)[-1]), 
       labels = c(seq(from = args$p, to = 0, length.out = 6)[-6], 1))
  axis(1, at = c(1, seq(from = 0, to = args$p, length.out = 6)[-1]), 
       labels = c(1, seq(from = 0, to = args$p, length.out = 6)[-1]))

  
  rect(xleft = 0.5, 
       ybottom = 0.5, 
       xright = args$p+0.5, 
       ytop = args$p+0.5,
       col = "gray86",
       density = 75, angle = 45)
  
  rect(xleft = c(min(args$Hind[[.k]])-0.5, 0.5), 
       ybottom = c(0.5, args$p-max(args$Hind[[.k]])-0.5), 
       xright = c(max(args$Hind[[.k]])+0.5, args$p+0.5), 
       ytop = c(args$p+0.5, args$p-min(args$Hind[[.k]])+0.5),
       col = c(colsH[.k], colsH[.k]), border = NA, 
       density = 75, angle = 135)
  
  rect(xleft = c(0.5, 0.5), 
       ybottom = c(0.5, args$p - args$r+0.5), 
       xright = c(args$r+0.5, args$p+0.5), 
       ytop = c(args$p+0.5,args$p+0.5),
       col = c(cbPalette[7], cbPalette[7]), border = NA, 
       density = 75, angle = 45)
}

# plot.spectralcurves2(Theta = Rho.av, r = ndir, p = args$p, 
#                     K = args$K, Hjoint = args$Hjoint, Hind = args$Hind, 
#                     coljoint = cbPalette[7], colind = colsH, colnon = cbPalette[1],
#                     main = "Average Matrix Eigenvectors")
# 
# plot.spectralcurves2(V = output$vectors, r = ndir, p = args$p, 
#                     K = args$K, Hjoint = args$Hjoint, Hind = args$Hind, 
#                     coljoint = cbPalette[7], colind = colsH, colnon = cbPalette[1],
#                     main = "Average Matrix Eigenvectors")
#plot(1/output$values, 
#     xlab = "Index",
#     ylab = "Joint Eigenvalue",
#     pch = 19,
#     main = "Minimax Joint Eigenvalues")
plot(apply(output.stiefel1$vectors^2, MARGIN = 1, sum), ylim = c(0,1),
     col = colinds, pch = pchinds,
     xlab = "Variable Index",
     ylab = "Influence Measure value",
     main = "Joint Influence Measures")
par(mar= c(1, 1, 1, 1))
plot(x = 0, y= 0, col = "white",
     xlim = c(-1,1), ylim = c(-1,1), 
     xlab ="", ylab = "", axes=F)
legend(x = "center", 
       legend = c("Common Hubs",
                  "Ind. Hubs 1",
                  "Ind. Hubs 2",
                  "Ind. Hubs 3",
                  "Non-hubs"),
       col = c(cbPalette[7], colsH[-4], cbPalette[1]), 
       pch = c(19, pchH, 1), cex = 1,
       box.col = "white")











