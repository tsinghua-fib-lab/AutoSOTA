rm(list=ls())

library(Seurat)
# library(GEMLI)  # not needed if you're using the custom scripts

out_folder <- "/Users/apple/Desktop/KB/SCSeq_LineageBarcoding/SCLineage_ConstrativeLearning/GEMLI_AUPRC"
seurat_obj <- readRDS(file.path(out_folder, "cellTagMulti_normX_seurat.rds"))

date_of_run <- Sys.time()
session_info <- sessionInfo()  # devtools optional; avoid dependency

code_folder <- "/Users/apple/Desktop/KB/SCSeq_LineageBarcoding/SCLineage_ConstrativeLearning/GEMLI_AUPRC/"

source(paste0(code_folder, "predict_lineages_custom.R"))
source(paste0(code_folder, "quantify_clusters_iterative_custom.R"))
source(paste0(code_folder, "test_lineages_custom.R"))

# ----------------------------
# 1) filter to the top 50 lineages
# ----------------------------
lineage_tab <- table(seurat_obj$clone_id)
largest_lineages <- names(lineage_tab)[order(lineage_tab, decreasing = TRUE)[1:50]]

keep_vec <- seurat_obj$clone_id %in% largest_lineages
seurat_obj$keep <- keep_vec
seurat_obj <- subset(seurat_obj, keep == TRUE)

# ----------------------------
# 2) GEMLI input = normalized expression (RNA "data" layer)
#    IMPORTANT: keep sparse, don't as.matrix()
# ----------------------------
data_matrix <- SeuratObject::LayerData(seurat_obj, assay = "RNA", layer = "data")

# drop empty cells (all-zero columns) just in case
colsum <- Matrix::colSums(data_matrix)
if (any(colsum == 0)) {
  data_matrix <- data_matrix[, colsum != 0, drop = FALSE]
  seurat_obj <- subset(seurat_obj, cells = colnames(data_matrix))
}

# drop empty genes (all-zero rows)
rowsum <- Matrix::rowSums(data_matrix)
data_matrix <- data_matrix[rowsum != 0, , drop = FALSE]

# align clone_id names to columns
seurat_obj$clone_id <- seurat_obj$clone_id[colnames(data_matrix)]
stopifnot(all(colnames(data_matrix) == names(seurat_obj$clone_id)))

# ----------------------------
# 3) run GEMLI
# ----------------------------
GEMLI_items <- list()
GEMLI_items[["gene_expression"]] <- data_matrix   # genes x cells
GEMLI_items[["barcodes"]] <- seurat_obj$clone_id  # named vector, length = cells

set.seed(10)
GEMLI_items <- predict_lineages_custom(
  GEMLI_items,
  bool_find_markers = FALSE,          # critical for normX (can have negatives)
  desired_cluster_size = c(50,200),
  verbose = 1
)

save(date_of_run, session_info, GEMLI_items,
     file = file.path(out_folder, "cellTagMulti_normX_GEMLI.RData"))

zz <- GEMLI_items[["prediction"]]
quantile(zz[zz != 0])

# ----------------------------
# 4) evaluate (precision/recall table)
# ----------------------------
GEMLI_items <- test_lineages_custom(
  GEMLI_items,
  valid_fam_sizes = c(50,200)
)
res <- GEMLI_items$testing_results
print(res)

save(date_of_run, session_info, GEMLI_items,
     file = file.path(out_folder, "cellTagMulti_normX_GEMLI.RData"))

print("Done! :)")

# ----------------------------
# 5) PR curve + AUPRC
# ----------------------------
plot_folder <- out_folder

recall <- res[, "sensitivity"]
precision <- res[, "precision"]

ok <- is.finite(recall) & is.finite(precision)
recall <- recall[ok]
precision <- precision[ok]

ord <- order(recall)
recall <- recall[ord]
precision <- precision[ord]

# enforce endpoints (optional)
if (length(recall) > 0) {
  if (min(recall) > 0) {
    recall <- c(0, recall)
    precision <- c(precision[1], precision)
  }
  if (max(recall) < 1) {
    recall <- c(recall, 1)
    precision <- c(precision[length(precision)], precision)
  }
}

auprc <- sum(diff(recall) * (precision[-1] + precision[-length(precision)]) / 2, na.rm = TRUE)

png(file.path(plot_folder, "cellTagMulti_normX_GEMLI_PR.png"),
    height = 1200, width = 1200, units = "px", res = 300)

plot(x = recall, y = precision,
     xlim = c(0,1), ylim = c(0,1),
     xlab = "Recall", ylab = "Precision",
     pch = 16, main = paste0("PR curve (AUPRC=", signif(auprc, 4), ")"))
lines(recall, precision)

baseline <- res["0","TP"] / (res["0","TP"] + res["0","FP"])
abline(h = baseline, col = 2, lwd = 2, lty = 2)

graphics.off()

cat("AUPRC:", signif(auprc, 6), "\n")
cat("Baseline precision:", signif(baseline, 6), "\n")
