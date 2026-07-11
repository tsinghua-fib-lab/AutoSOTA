rm(list=ls())

library(Seurat)
library(GEMLI)

out_folder <- "/Users/apple/Desktop/KB/SCSeq_LineageBarcoding/SCLineage_ConstrativeLearning/GEMLI_AUROC/"
seurat_obj <- readRDS(file.path(out_folder, "cellTagMulti_normX_seurat.rds"))
# load(paste0(out_folder, "Larry_41093_2000_norm_log_cleaned.RData"))

date_of_run <- Sys.time()
session_info <- devtools::session_info()

code_folder <- "/Users/apple/Desktop/KB/SCSeq_LineageBarcoding/SCLineage_ConstrativeLearning/GEMLI_AUROC/"

source(paste0(code_folder, "predict_lineages_custom.R"))
source(paste0(code_folder, "quantify_clusters_iterative_custom.R"))
source(paste0(code_folder, "test_lineages_custom.R"))

# first filter to the top 50 lineages
lineage_tab <- table(seurat_obj$clone_id)
largest_lineages <- names(lineage_tab)[order(lineage_tab, decreasing = TRUE)[1:50]]
keep_vec <- rep(FALSE, length(Seurat::Cells(seurat_obj)))
keep_vec[seurat_obj$clone_id %in% largest_lineages] <- TRUE
seurat_obj$keep <- keep_vec
seurat_obj <- subset(seurat_obj, keep == TRUE)

# remove any cells that are empty
data_matrix <- SeuratObject::LayerData(seurat_obj,
                                       layer = "data",
                                       assay = "RNA")
# remove empty cells
sum_vec <- Matrix::colSums(data_matrix)
if (any(sum_vec == 0)) {
  data_matrix <- data_matrix[, sum_vec != 0, drop = FALSE]
  seurat_obj <- subset(seurat_obj, cells = colnames(data_matrix))
}

# remove all-zero genes
rowsum_vec <- Matrix::rowSums(data_matrix)
data_matrix <- data_matrix[rowsum_vec != 0, , drop = FALSE]

# align clone_id names to columns
seurat_obj$clone_id <- seurat_obj$clone_id[colnames(data_matrix)]
stopifnot(all(colnames(data_matrix) == names(seurat_obj$clone_id)))

GEMLI_items <- list(
  gene_expression = data_matrix,
  barcodes = seurat_obj$clone_id
)

set.seed(10)
GEMLI_items <- predict_lineages_custom(
  GEMLI_items,
  bool_find_markers = FALSE,          # <-- key fix
  desired_cluster_size = c(50, 200),
  verbose = 1
)

save(date_of_run, session_info,
     GEMLI_items,
     file = paste0(out_folder, "cellTagMulti_log_GEMLI.RData"))

zz <- GEMLI_items[['prediction']] # we need to look at how they did it on LARRY...
quantile(zz[zz!=0])

GEMLI_items <- test_lineages_custom(GEMLI_items,
                                    valid_fam_sizes = c(50,200))
GEMLI_items$testing_results

save(date_of_run, session_info,
     GEMLI_items,
     file = paste0(out_folder, "cellTagMulti_log_GEMLI.RData"))

print("Done! :)")

###############

plot_folder <- "/Users/apple/Desktop/KB/SCSeq_LineageBarcoding/SCLineage_ConstrativeLearning/GEMLI_AUROC/"


total_true <- GEMLI_items$testing_results["0","TP"]
total_false <- GEMLI_items$testing_results["0","FP"]

tpr <- GEMLI_items$testing_results[,"TP"]/total_true
fpr <- GEMLI_items$testing_results[,"FP"]/total_false


# Remove NA / NaN
ok <- is.finite(tpr) & is.finite(fpr)
tpr <- tpr[ok]
fpr <- fpr[ok]

# Sort by FPR (important!)
ord <- order(fpr)
fpr <- fpr[ord]
tpr <- tpr[ord]

# Enforce ROC endpoints
if (min(fpr) > 0) {
  fpr <- c(0, fpr)
  tpr <- c(0, tpr)
}
if (max(fpr) < 1) {
  fpr <- c(fpr, 1)
  tpr <- c(tpr, 1)
}

# AUROC via trapezoidal integration
auroc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2, na.rm = TRUE)

cat("AUROC =", signif(auroc, 6), "\n")
# AUROC = 0.637563

png(paste0(plot_folder, "Writeup5_LARRY_log_GEMLI_ROC.png"),
    height = 1200, width = 1200, units = "px", res = 300)
plot(x = fpr,
     y = tpr,
     xlim = c(0,1),
     ylim = c(0,1),
     xlab = "FPR",
     ylab = "TPR",
     pch = 16)
lines(fpr, tpr)
lines(c(0,1), c(0,1), col = 2, lwd = 2, lty = 2)
graphics.off()
