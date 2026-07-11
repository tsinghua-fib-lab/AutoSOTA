############################################
## 0. Setup Python (reticulate + conda env)
############################################
library(reticulate)
use_condaenv("r-h5ad", required = TRUE)

############################################
## 1. Read h5ad via anndata (NO basilisk)
############################################
ad <- import("anndata", delay_load = FALSE)
adata <- ad$read_h5ad("cellTag_test_multi_Seurat_clone_id_LCL.h5ad")

############################################
## 2. Extract LCL embedding + clone IDs
############################################

# LCL embedding: cells x dims
emb <- py_to_r(adata$obsm[["scVI"]])
emb <- as.matrix(emb)

# clone IDs
obs <- py_to_r(adata$obs)
clone_id <- obs$clone_id

# cell names
cell_names <- as.character(py_to_r(adata$obs_names$to_list()))

# attach names
rownames(emb) <- cell_names
names(clone_id) <- cell_names

############################################
## 3. Create minimal Seurat object
############################################
library(Matrix)
library(Seurat)

dummy_counts <- Matrix(0, nrow = 2, ncol = length(cell_names), sparse = TRUE)
rownames(dummy_counts) <- c("dummy1", "dummy2")
colnames(dummy_counts) <- cell_names

seurat_obj <- CreateSeuratObject(counts = dummy_counts)


# add clone_id metadata
seurat_obj$clone_id <- clone_id[colnames(seurat_obj)]

############################################
## 4. Add LCL embedding as DimReduc
############################################
seurat_obj[["scVI"]] <- CreateDimReducObject(
  embeddings = emb[colnames(seurat_obj), , drop = FALSE],
  key = "scVI_",
  assay = DefaultAssay(seurat_obj)
)

############################################
## 5. Sanity checks (important)
############################################
stopifnot(
  all(Cells(seurat_obj) == rownames(seurat_obj[["scVI"]]@cell.embeddings)),
  length(seurat_obj$clone_id) == ncol(seurat_obj)
)

############################################
## 6. GEMLI-ready matrix (UNCHANGED pipeline)
############################################
data_matrix <- t(seurat_obj[["scVI"]]@cell.embeddings)

# data_matrix: dims x cells
dim(data_matrix)
head(seurat_obj$clone_id)

############################################
## 7. (Optional) Save for reuse
############################################
saveRDS(seurat_obj, "cellTagMulti_scVI_seurat.rds")
