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
print("Installing packages in directories:")
print(.libPaths())

## For 01_GeneratingMatrixSparse.R
#if(!require(plot.matrix)){
#  .libPaths(req_lib_dir)
#  install.packages("plot.matrix", repos = "https://archive.linux.duke.edu/cran/")
#  library(plot.matrix)
#}

## For 02_BIC.R
if(!require(magrittr)){
  .libPaths(req_lib_dir)
  print(.libPaths)
  install.packages("magrittr", repos = "https://archive.linux.duke.edu/cran/")
  library(magrittr)
}
if(!require(tidyverse)){
  .libPaths(req_lib_dir)
  print(.libPaths)
  install.packages("tidyverse", repos = "https://archive.linux.duke.edu/cran/")
  library(tidyverse)
}

## For 02_BIC.R
if(!require(mvtnorm)){
  .libPaths(req_lib_dir)
  print(.libPaths)
  install.packages("mvtnorm", repos = "https://archive.linux.duke.edu/cran/")
  library(mvtnorm)
}



## For 0_MatrixEstimators.R
if(!require(glasso)){
  .libPaths(req_lib_dir)
  print(.libPaths)
  install.packages("glasso", repos = "https://archive.linux.duke.edu/cran/")
  library(glasso)
}
#if(!require(hglasso)){
#  .libPaths(req_lib_dir)
#  install.packages("hglasso", repos = "https://archive.linux.duke.edu/cran/")
#  library(hglasso)
#}
if(!require(e1071)){
  .libPaths(req_lib_dir)
  print(.libPaths)
  install.packages("e1071", repos = "https://archive.linux.duke.edu/cran/")
  library(e1071)
}

## For 0_glassoplsp.R
if(!require(mvtnorm)){
  .libPaths(req_lib_dir)
  install.packages("mvtnorm", repos = "https://archive.linux.duke.edu/cran/")
  library(mvtnorm)
}

## For 0_EstimationMeasures.R
if(!require(tidyr)){
  .libPaths(req_lib_dir)
  install.packages("tidyr", repos = "https://archive.linux.duke.edu/cran/")
  library(tidyr)
}
if(!require(dplyr)){
  .libPaths(req_lib_dir)
  install.packages("dplyr", repos = "https://archive.linux.duke.edu/cran/")
  library(dplyr)
}
if(!require(pracma)){
  .libPaths(req_lib_dir)
  install.packages("pracma", repos = "https://archive.linux.duke.edu/cran/")
  library(pracma)
}
#if(!require(pROC)){
#  .libPaths(req_lib_dir)
#  install.packages("pROC_1.17.0.1.tar.gz", repos = NULL, type="source")
#  library(pROC)
#}
#if(!require(Ckmeans.1d.dp)){
#  .libPaths(req_lib_dir)
#  install.packages("Ckmeans.1d.dp_4.3.3.tar.gz", repos = NULL, type="source")
#  library(Ckmeans.1d.dp)
#}
#if(!require(pryr)){
#  .libPaths(req_lib_dir)
#  install.packages("pryr_0.1.4.tar.gz", repos = NULL, type="source")
#  library(pryr)
#}
if(!require(e1071)){
  .libPaths(req_lib_dir)
  install.packages("e1071", repos = "https://archive.linux.duke.edu/cran/")
  library(e1071)
}
if(!require(MASS)){
  .libPaths(req_lib_dir)
  install.packages("MASS", repos = "https://archive.linux.duke.edu/cran/")
  library(MASS)
}
if(!require(Matrix)){
  .libPaths(req_lib_dir)
  install.packages("Matrix", repos = "https://archive.linux.duke.edu/cran/")
  library(Matrix)
}
if(!require(RSpectra)){
  .libPaths(req_lib_dir)
  install.packages("RSpectra", repos = "https://archive.linux.duke.edu/cran/")
  library(RSpectra)
}
if(!require(spam)){
  .libPaths(req_lib_dir)
  install.packages("spam", repos = "https://archive.linux.duke.edu/cran/")
  library(spam)
}

## For 0_ChoosingTuningParameters.R
if(!require(matrixcalc)){
  .libPaths(req_lib_dir)
  install.packages("matrixcalc", repos = "https://archive.linux.duke.edu/cran/")
  library(matrixcalc)
}

## For 03_ggl.R and 03_fgl.R
if(!require(JGL)){
  .libPaths(req_lib_dir)
  install.packages("JGL_2.3.1.tar.gz", repos = NULL, type="source")
  library(JGL)
}


if(!require(ggh4x)) {
  .libPaths(req_lib_dir)
  install.packages("ggh4x_0.3.0.tar.gz", repos = NULL, type="source")
  library(ggh4x)
}


## Colorblind friendly palette.
cbPalette <- c("#999999", "#E69F00", "#56B4E9", 
               "#009E73", "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")
