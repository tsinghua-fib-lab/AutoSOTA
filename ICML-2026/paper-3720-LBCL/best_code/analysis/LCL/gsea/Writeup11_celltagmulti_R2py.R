# https://github.com/linnykos/Was2CoDE_analysis/blob/kevin/code/kevin/Writeup27_reprocessing-data/Writeup27_sea-ad_microglia_R2py.R

rm(list=ls())
library(Seurat)
library(SeuratObject)
library(SeuratDisk)

load("~/kzlinlab/data/celltagging-multi_fibroblast/celltagging-multi_fibroblast.RData")

for(variable in colnames(seurat_obj@meta.data)){
  if(is.factor(seurat_obj@meta.data[,variable]) | is.logical(seurat_obj@meta.data[,variable]) ){
    seurat_obj@meta.data[,variable] <- as.character(seurat_obj@meta.data[,variable])
  }
  
  stopifnot(is.numeric(seurat_obj@meta.data[,variable]) | is.character(seurat_obj@meta.data[,variable]))
}
summary(seurat_obj@meta.data)

seurat_obj[["RNA"]] <- as(object = seurat_obj[["RNA"]], Class = "Assay")

####################

# only keep the cells with lineage and more than 5 cells
seurat_obj <- subset(seurat_obj, has_lineage == TRUE)
tab_vec <- table(seurat_obj$assigned_lineage)
lineage_names <- names(tab_vec)[tab_vec >= 5]
keep_vec <- rep(FALSE, length(Seurat::Cells(seurat_obj)))
keep_vec[seurat_obj$assigned_lineage %in% lineage_names] <- TRUE
seurat_obj$keep <- keep_vec
seurat_obj <- subset(seurat_obj, keep == TRUE)

#################################

# create a timepoint (integer) column
time_vec <- seurat_obj$sample
time_vec <- as.numeric(gsub("B4D", "", time_vec))
seurat_obj$time_info <- time_vec

# create a X_clone matrix
lin <- seurat_obj$assigned_lineage
cells <- colnames(seurat_obj)

lin <- lin[cells]  # align

lineages <- unique(lin)
j <- match(lin, lineages)

library(Matrix)
X_clone <- sparseMatrix(
  i = seq_along(cells),
  j = j,
  x = 1L,
  dims = c(length(cells), length(lineages)),
  dimnames = list(cells, lineages)
)
colnames(X_clone) <- paste0("NewLineage_", 1:ncol(X_clone))
X_clone <- as.matrix(X_clone)

seurat_obj[["clone"]] <- Seurat::CreateDimReducObject(X_clone)
seurat_obj$assigned_lineage <- NULL

###############################

seurat_obj
# An object of class Seurat 
# 31761 features across 22238 samples within 1 assay 
# Active assay: RNA (31761 features, 2000 variable features)
# 3 layers present: counts, data, scale.data
# 3 dimensional reductions calculated: pca, umap, X_clone

# assigned_lineage: lineage information
# sample: timepoint 
# predicted.id_cca_co: cell-type
# md_fate_coarse_rev1: the fates

table(seurat_obj$time_point)
# 3    12    21 
# 10890  3258  8090 

table(seurat_obj$predicted.id_cca_co)
# Dead-end_0 Dead-end_1 Dead-end_2    Early_0    Early_1    Early_2      Fib_0 
# 339        254        512       1528       3613       2189        555 
# Fib_1      Fib_2      iEP_0      iEP_1      iEP_2     Tran_0     Tran_1 
# 1666        225        590        654        517       1907       6890 
# Tran_2 
# 799

table(seurat_obj$md_fate_coarse_rev1) 
# dead-end            na reprogramming 
# 4828         13614          3796

table(seurat_obj$md_fate_rev1)
# Day12 dead-end Day12 reprogramming      Day21 dead-end Day21 reprogramming 
# 728                 825                3700                2157 
# Day3 dead-end  Day3 reprogramming                  na 
# 400                 814               13614 

SeuratDisk::SaveH5Seurat(seurat_obj,
                         filename = "~/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/celltagging-multi_fibroblast.h5Seurat",
                         overwrite = TRUE)
SeuratDisk::Convert("~/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/celltagging-multi_fibroblast.h5Seurat",
                    dest = "h5ad",
                    misc = FALSE,
                    overwrite = TRUE)