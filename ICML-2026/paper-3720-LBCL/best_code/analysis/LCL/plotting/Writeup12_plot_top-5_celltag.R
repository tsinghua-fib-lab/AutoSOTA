rm(list=ls())

library(Seurat)

load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/cell_tag_integration.RData")

table(integrated$batch)

# UMAP:
table(integrated$batch)
DimPlot(
  integrated,
  reduction = "umap",
  group.by  = "batch",  # or your actual meta column
  pt.size   = 0.4
)


# metadata
metadata <- integrated@meta.data

seurat_obj <- subset(integrated, batch == "tag")
table(seurat_obj$assigned_lineage)
table(seurat_obj$clone_id)

cols <- c(
  rgb(209, 57, 44, maxColorValue = 255),
  rgb(74, 124, 179, maxColorValue = 255),
  rgb(103, 173, 87, maxColorValue = 255),
  rgb(142, 82, 159, maxColorValue = 255),
  rgb(238, 134, 50, maxColorValue = 255)
)

umap_mat <- seurat_obj[["umap"]]@cell.embeddings
col_vec <- rep("gray", nrow(umap_mat))
clone_vec <- paste0("Lineage:", seurat_obj$clone_id)
tab_vec <- sort(table(clone_vec), decreasing = TRUE)
for(i in 1:5){
  idx <- which(clone_vec == names(tab_vec)[i])
  col_vec[idx] <- cols[i]
}

# put the grays behind
idx <- which(col_vec == "gray")
col_vec <- c(col_vec[idx], col_vec[-idx])
umap_mat <- rbind(
  umap_mat[idx,],
  umap_mat[-idx,]
)

fig_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/git/SCSeq_LineageBarcoding/fig/kevin/Writeup12/"

png(filename = paste0(fig_folder, "Writeup12_top5_celltag_umap.png"),
    height = 2000, 
    width = 2000,
    units = "px",
    res = 300)
plot(umap_mat[,1],
     umap_mat[,2],
     col = col_vec,
     pch = 16,
     xlab = "UMAP1",
     ylab = "UMAP2")

graphics.off()
