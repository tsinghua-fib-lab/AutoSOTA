rm(list=ls())

library(Seurat)

load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/cell_tag_integration.RData")

fig_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/git/SCSeq_LineageBarcoding/fig/kevin/Writeup12/"

integrated$batch <- ifelse(integrated$batch == "multi", "CellTag-multi", "CellTag")

plot1 <- DimPlot(integrated,
                 group.by = "batch")
plot1 <- plot1 + ggplot2::labs(x = "UMAP1", y = "UMAP2", title = "After batch correction")

ggplot2::ggsave(plot1,
                filename = paste0(fig_folder, "Writeup12_batch_after.png"),
                height = 4, 
                width = 5.5)

##################


load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/cell_tag-cell_tag_multi.RData")

cell_tag_multi$batch <- ifelse(cell_tag_multi$batch == "multi", "CellTag-multi", "CellTag")

Seurat::VariableFeatures(cell_tag_multi) <- SeuratObject::Features(cell_tag_multi)
cell_tag_multi <- Seurat::ScaleData(cell_tag_multi)
cell_tag_multi <- Seurat::RunPCA(cell_tag_multi,
                                 features = Seurat::VariableFeatures(cell_tag_multi),
                                 verbose = FALSE)
cell_tag_multi <- Seurat::RunUMAP(cell_tag_multi,
                                  dims = c(1:30))

plot1 <- DimPlot(cell_tag_multi,
                 group.by = "batch") 
plot1 <- plot1 + ggplot2::labs(x = "UMAP1", y = "UMAP2", title = "Before batch correction")

ggplot2::ggsave(plot1,
                filename = paste0(fig_folder, "Writeup12_batch_before.png"),
                height = 4, 
                width = 5.5)