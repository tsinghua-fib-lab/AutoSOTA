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


tmp1 <- integrated$predicted.id_cca_co
tmp2 <- integrated$cell_type
tmp_df <- cbind(tmp1, tmp2)
celltype_vec <- sapply(1:nrow(tmp_df), function(i){
  ifelse(nchar(tmp_df[i,1]) != 0, tmp_df[i,1], tmp_df[i,2])
})
names(celltype_vec) <- Seurat::Cells(integrated)
integrated$final_celltype <- celltype_vec

celltype_cols <- c(
  "Ambiguous"   = "gray30",
  Fib_0       = "#2A7ABF",
  Fib_1       = "#5AA6E6",
  Fib_2       = "#0B4F8A",
  Fibroblast  = "#1F77B4",
  Early_0     = "#2FBF9B",
  Early_1     = "#66D2B8",
  Early_2     = "#0F8F73",
  Tran_0      = "#FF8C2A",
  Tran_1      = "#FFB066",
  Tran_2      = "#D96B00",
  iEP_0       = "#8E63C7",
  iEP_1       = "#B08AE0",
  iEP_2       = "#6A3D9A",
  iEP         = "#9467BD",
  "Dead-end_0" = "#D62728",
  "Dead-end_1" = "#FF5A5F",
  "Dead-end_2" = "#A50F15"
)

celltype_cols <- c(
  "Ambiguous"  = "#D7644A",  # salmon
  
  # Fibroblasts = purples (includes #471F4E, plus lighter)
  "Fib_0"      = "#471F4E",
  "Fib_1"      = "#6A3D7C",
  "Fib_2"      = "#9B77AE",
  "Fibroblast" = "#7A4E86",
  
  # Early = muted yellows (not too bright)
  "Early_0"    = "#FF8C2A",  # ochre
  "Early_1"    = "#FFB066",  # warm mustard
  "Early_2"    = "#D96B00",  # pale muted yellow
  
  # Transitions = cyan/blue shades
  "Tran_0"     = "#0F5E9C",  # deep blue
  "Tran_1"     = "#2A9CC9",  # cyan-blue
  "Tran_2"     = "#7BC8E6",  # light cyan
  
  # iEPs = greens (includes #367A76, plus lighter)
  "iEP_0"      = "#367A76",
  "iEP_1"      = "#55A19B",
  "iEP_2"      = "#86C6BF",
  "iEP"        = "#4B908B",
  
  # Dead-ends = dark grays
  "Dead-end_0" = "#1F1F1F",
  "Dead-end_1" = "#3A3A3A",
  "Dead-end_2" = "#5A5A5A"
)



umap_mat <- integrated[["LCLUMAP"]]@cell.embeddings
col_vec <- celltype_cols[integrated$final_celltype]
shuff_idx <- c(which(integrated$batch == "multi"),
              which(integrated$batch == "tag"))

png(paste0(fig_folder, "Writeup12_lcl-celltype.png"),
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

