rm(list=ls())

library(Seurat)

load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/cell_tag_integration.RData")
fig_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/git/SCSeq_LineageBarcoding/fig/kevin/Writeup12/"

umap_csv_ctm <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup10b_joshua-lcl/cellTag-cellTagMulti_UMAP_train.csv")
rownames(umap_csv_ctm) <- umap_csv_ctm$X
umap_csv_ctm <- umap_csv_ctm[,c("UMAP0", "UMAP1")]
colnames(umap_csv_ctm) <- paste0("lclumap_", 1:2)
umap_csv_ctm <- as.matrix(umap_csv_ctm)

umap_csv_ct <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup10b_joshua-lcl/cellTag-cellTagMulti_UMAP_test.csv")
rownames(umap_csv_ct) <- umap_csv_ct$X
umap_csv_ct <- umap_csv_ct[,c("UMAP0", "UMAP1")]
colnames(umap_csv_ct) <- paste0("lclumap_", 1:2)
umap_csv_ct <- as.matrix(umap_csv_ct)

umap_csv <- rbind(umap_csv_ctm,
                  umap_csv_ct)
umap_csv <- umap_csv[Seurat::Cells(integrated),]


integrated[["LCLUMAP"]] <- Seurat::CreateDimReducObject(umap_csv,
                                                        assay = "integrated")
integrated$batch <- ifelse(integrated$batch == "multi", "CellTag-multi", "CellTag")

cols <- c("CellTag" = "#F8766D", "CellTag-multi" = "#00BFC4")
cols2 <- adjustcolor(cols, alpha.f = 0.5) 
names(cols2) <- names(cols)

set.seed(10)
umap_mat <- integrated[["LCLUMAP"]]@cell.embeddings
col_vec <- cols2[integrated$batch]
shuff_idx <- sample(1:nrow(umap_mat))

png(paste0(fig_folder, "Writeup12_lcl-integration.png"),
    height = 1800, 
    width = 1800,
    units = "px",
    res = 300)
par(mar = rep(0.5,4))
plot(umap_mat[shuff_idx,1],
     umap_mat[shuff_idx,2],
     pch = 16,
     cex = 0.75,
     col = col_vec[shuff_idx],
     xlab = "",
     ylab = "",
     main = "",
     xaxt = "n",
     yaxt = "n",
     bty = "n")
graphics.off()