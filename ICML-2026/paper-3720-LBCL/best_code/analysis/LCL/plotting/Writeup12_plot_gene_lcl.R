rm(list=ls())

library(Seurat)
library(scCustomize)

fig_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/git/SCSeq_LineageBarcoding/fig/kevin/Writeup12/"

load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/cell_tag_integration.RData")

seurat_obj <- subset(integrated, batch == "tag")

lcl_csv <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup10b_joshua-lcl/test_proj_embe_w_clone_id.csv")
rownames(lcl_csv) <- lcl_csv[,"X"]
lcl_csv <- lcl_csv[,grep("embed_", colnames(lcl_csv))]
lcl_csv <- lcl_csv[Seurat::Cells(seurat_obj),]
colnames(lcl_csv) <- paste0("LCL_", 1:ncol(lcl_csv))
lcl_csv <- as.matrix(lcl_csv)

umap_csv <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup10b_joshua-lcl/cellTag-cellTagMulti_UMAP_test.csv")
rownames(umap_csv) <- umap_csv$X
umap_csv <- umap_csv[,c("UMAP0", "UMAP1")]
colnames(umap_csv) <- paste0("lclumap_", 1:2)
umap_csv <- umap_csv[Seurat::Cells(seurat_obj),]
umap_csv <- as.matrix(umap_csv)

seurat_obj[["LCL"]] <- Seurat::CreateDimReducObject(lcl_csv,
                                                    assay = "integrated")
seurat_obj[["LCLUMAP"]] <- Seurat::CreateDimReducObject(umap_csv,
                                                        assay = "integrated")

##########

set.seed(10)
seurat_obj <- Seurat::FindNeighbors(seurat_obj, 
                                    dims = 1:31, 
                                    reduction = "LCL",
                                    k.param = 5)
seurat_obj <- Seurat::FindClusters(seurat_obj, 
                                   resolution = 0.1,
                                   group.singletons = FALSE)

Seurat::FeaturePlot(seurat_obj,
                    features = "Apoa1",
                    reduction = "LCLUMAP")

plot1 <- DimPlot(seurat_obj,
                 reduction = "LCLUMAP",
                 group.by = "seurat_clusters",
                 label = TRUE)
plot1 <- plot1 + ggplot2::labs(x = "UMAP1", y = "UMAP2", title = "Leiden clustering on CellTag")

ggplot2::ggsave(plot1,
                filename = paste0(fig_folder, "Writeup12_lcl-clustering.png"),
                height = 4, 
                width = 5.5)

DimPlot(seurat_obj,
        reduction = "LCLUMAP",
        group.by = "cell_type",
        label = TRUE)

de_res <- Seurat::FindAllMarkers(seurat_obj, group.by = "seurat_clusters")
save(de_res, 
     file = "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/Writeup12_FindAllMarkers.csv")

de_res2 <- de_res
de_res2 <- de_res2[which(de_res2$avg_log2FC >= 3),]
de_res2 <- de_res2[which(de_res2$pct.1 >= 0.3),]
de_res2 <- de_res2[which(de_res2$p_val_adj <= 0.05),]
de_res2 <- de_res2[order(abs(de_res2$avg_log2FC), decreasing = TRUE),]
rownames(de_res2) <- de_res2$gene

gene_vec <- c("Ptn", "Sfrp1", "Shh")
tmp <- de_res2[gene_vec, c("p_val_adj", "avg_log2FC", "cluster")]

for(gene in gene_vec){
  plot1 <- scCustomize::FeaturePlot_scCustom(seurat_obj,
                                             features = gene,
                                             reduction = "LCLUMAP",
                                             max.cutoff = "q99")
  plot1 <- plot1 + ggplot2::labs(x = "UMAP1", y = "UMAP2")
  
  ggplot2::ggsave(plot1,
                  filename = paste0(fig_folder, "Writeup12_featureplot_", gene, "-lcl.png"),
                  height = 4, 
                  width = 5)
}
