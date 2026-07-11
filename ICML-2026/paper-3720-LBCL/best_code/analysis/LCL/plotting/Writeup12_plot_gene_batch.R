rm(list=ls())

library(Seurat)
library(scCustomize)

fig_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/git/SCSeq_LineageBarcoding/fig/kevin/Writeup12/"

load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/cell_tag_integration.RData")

seurat_obj <- subset(integrated, batch == "tag")

gene_vec <- c("Ptn", "Sfrp1", "Shh")
for(gene in gene_vec){
  plot1 <- scCustomize::FeaturePlot_scCustom(seurat_obj,
                                             features = gene,
                                             reduction = "umap",
                                             max.cutoff = "q99")
  plot1 <- plot1 + ggplot2::labs(x = "UMAP1", y = "UMAP2")
  
  ggplot2::ggsave(plot1,
                  filename = paste0(fig_folder, "Writeup12_featureplot_", gene, "-batch.png"),
                  height = 4, 
                  width = 5)
}

