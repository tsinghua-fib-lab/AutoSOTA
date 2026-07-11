# https://github.com/SZ-yang/SCSeq_LineageBarcoding/blob/kevin/kevin/Writeup5_GEMLI/Writeup5_LARRY_gsea-plotting.R

rm(list=ls())

library(Seurat)
library(SeuratDisk)
library(SeuratData)
library(org.Mm.eg.db)
library(clusterProfiler)
library(ggplot2)
library(variancePartition)

seurat_obj <- SeuratDisk::LoadH5Seurat("~/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/celltagging-multi_fibroblast.h5Seurat")
df <- read.csv("~/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/Writeup11_celltagmulti_cospar_postprocess_adata-obs.csv",
               row.names = 1)
df <- df[Seurat::Cells(seurat_obj),]
seurat_obj$fate_bias <- df[,"fate_bias_transition_map_reprogramming.dead.end"]

fate_vec <- seurat_obj$fate_bias
names(fate_vec) <- Seurat::Cells(seurat_obj)

mat <- SeuratObject::LayerData(
  seurat_obj,
  layer = "data",
  assay = "RNA",
  features = Seurat::VariableFeatures(seurat_obj)
)
mat <- mat[,names(fate_vec)]
teststat_vec <- apply(mat, 1, function(x){
  stats::cor(fate_vec, x)
})
teststat_vec <- teststat_vec[!is.na(teststat_vec)]
teststat_vec <- sort(teststat_vec, decreasing = TRUE)


set.seed(10)
gse <- clusterProfiler::gseGO(
  teststat_vec,
  ont = "BP", # what kind of pathways are you interested in
  keyType = "SYMBOL",
  OrgDb = "org.Mm.eg.db",
  pvalueCutoff = 1,
  minGSSize = 10,            # minimum gene set size
  maxGSSize = 500            # maximum gene set size
)

gse_df_cospar <- as.data.frame(gse)
gse_df_cospar <- gse_df_cospar[order(gse_df_cospar$p.adjust, decreasing = TRUE),]
length(which(gse_df_cospar$p.adjust <= 0.05))

########################


df <- read.csv("~/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup10b/IG_analysis_res.csv")

teststat_vec <- df[,"score_001_15"]
names(teststat_vec) <- df[,"gene"]
teststat_vec <- sort(teststat_vec, decreasing = TRUE)

set.seed(10)
gse <- clusterProfiler::gseGO(
  teststat_vec,
  ont = "BP", # what kind of pathways are you interested in
  keyType = "SYMBOL",
  OrgDb = "org.Mm.eg.db",
  pvalueCutoff = 1,
  minGSSize = 10,            # minimum gene set size
  maxGSSize = 500            # maximum gene set size
)

gse_df_lcl <- as.data.frame(gse)
gse_df_lcl <- gse_df_lcl[order(gse_df_lcl$p.adjust, decreasing = TRUE),]
length(which(gse_df_lcl$p.adjust <= 0.05))

gse_df_lcl[which(gse_df_lcl$p.adjust <= 0.05)[1:50],c("ID", "Description","core_enrichment")]

#########################

load("~/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup10c/Writeup10c_celltagmulti_variancePartition.RData")

teststat_vec <- varPart[,"assigned_lineage"]
names(teststat_vec) <- rownames(varPart)
teststat_vec <- sort(teststat_vec, decreasing = TRUE)

set.seed(10)
gse <- clusterProfiler::gseGO(
  teststat_vec,
  ont = "BP", # what kind of pathways are you interested in
  keyType = "SYMBOL",
  OrgDb = "org.Mm.eg.db",
  pvalueCutoff = 1,
  minGSSize = 10,            # minimum gene set size
  maxGSSize = 500,            # maximum gene set size
  scoreType = "pos"
)

gse_df_vp <- as.data.frame(gse)
gse_df_vp <- gse_df_vp[order(gse_df_vp$p.adjust, decreasing = TRUE),]
length(which(gse_df_vp$p.adjust <= 0.05))

gse_df_vp[which(gse_df_vp$p.adjust <= 0.05),c("ID", "Description")]

#########################

# pathways
pathways_of_interest <- c("GO:0045596", # contains Wnt4 and Sfpr1: negative regulation of cell differentiation
                          "GO:0010594", # contains Bmp4:  regulation of endothelial cell migration
                          "GO:0017015", # TGF-beta signaling: regulation of transforming growth factor beta receptor signaling pathway
                          "GO:0085029") # contains Col1a2: extracellular matrix assembly


gse_df_lcl[pathways_of_interest, c("ID", "Description", "p.adjust")]
gse_df_cospar[pathways_of_interest, c("ID", "Description", "p.adjust")]
gse_df_vp[pathways_of_interest, c("ID", "Description", "p.adjust")]

