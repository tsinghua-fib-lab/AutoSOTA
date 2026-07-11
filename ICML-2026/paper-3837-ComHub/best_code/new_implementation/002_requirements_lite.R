#################################################
#################################################
#################################################
##
## In the following file, we import all packages
## needed to run the simulations.
##
#################################################
#################################################
#################################################


###################### Creating folders:
wd <- getwd()
req_lib_dir <- paste0(wd,"/req_lib")
subfolder_new        <- paste0("req_lib/")
if (!dir.exists(subfolder_new)) {
       dir.create(subfolder_new)
}
.libPaths(req_lib_dir)
print(.libPaths())

library(magrittr)
library(tidyverse)
library(mvtnorm)
library(glasso)
library(e1071)
library(mvtnorm)
library(tidyr)
library(dplyr)
library(pracma)
library(MASS)
library(Matrix)
library(RSpectra)
library(spam)
library(matrixcalc)
library(JGL)
library(ggh4x)

## Colorblind friendly palette.
cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")

