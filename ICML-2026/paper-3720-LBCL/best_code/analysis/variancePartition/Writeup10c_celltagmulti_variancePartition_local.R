rm(list=ls())

library(Seurat)
library(Matrix)
library(variancePartition)
library(lme4)
library(reformulas)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(scales)

load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup12_joshua-celltagmulti/cell_tag_integration.RData")
seurat_obj <- subset(integrated, batch == "multi")

keep_vec <- !is.na(seurat_obj$assigned_lineage)
seurat_obj$keep <- keep_vec
seurat_obj <- subset(seurat_obj, keep == TRUE)

tab_vec <- table(seurat_obj$assigned_lineage)
lineage_names <- names(tab_vec)[which(tab_vec >= 50)]
keep_vec <- seurat_obj$assigned_lineage %in% lineage_names
seurat_obj$keep <- keep_vec
seurat_obj <- subset(seurat_obj, keep == TRUE)

#############################

tmp <- as.character(seurat_obj$predicted.id_cca_co)
tmp2 <- sapply(tmp, function(x){
  strsplit(x, split = "_")[[1]][1]
})
names(tmp2) <- NULL
seurat_obj$celltype <- as.factor(tmp2)
seurat_obj$predicted.id_cca_co <- as.factor(seurat_obj$predicted.id_cca_co)

#############################

# fit
var_data <- SeuratObject::LayerData(seurat_obj,
                                    layer = "scale.data",
                                    assay = "integrated")
var_info <- seurat_obj@meta.data
form <- ~ (1 | celltype) + (1 | assigned_lineage)  + (1 | sample)
old_warn <- getOption("warn")
options(warn = 0)
on.exit(options(warn = old_warn), add = TRUE)

varPart <- suppressWarnings(
  variancePartition::fitExtractVarPartModel(var_data, form, var_info, showWarnings = FALSE)
)

save(varPart,
     file = "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/out/kevin/Writeup10c/Writeup10c_celltag_variancePartition_local.RData")

variance_explained <- Matrix::rowSums(varPart[,c("assigned_lineage", "celltype", "sample")])
varPart <- varPart[order(variance_explained, decreasing = TRUE),]
head(varPart[1:50,])

round(varPart[1:50,1:3],2)

print("Done! :)")

############################

# 1) Pick top 50 genes by total explained variation
top_n <- 40
varPart <- varPart[,c("assigned_lineage", "celltype", "sample")]
colnames(varPart) <- c("Lineage_label", "Cell_type", "Time_point")

df_top <- as.data.frame(varPart) %>%
  rownames_to_column("gene") %>%
  mutate(total_explained = rowSums(across(-gene), na.rm = TRUE))

df_top <- df_top[order(df_top$total_explained, decreasing = TRUE)[1:top_n],]

# 2) Order covariates: assigned_lineage first; others by total contribution among top 50
covars <- setdiff(colnames(varPart), "Lineage_label")
other_order <- df_top %>%
  summarise(across(all_of(covars), ~ sum(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "covariate", values_to = "tot") %>%
  arrange(desc(tot)) %>%
  pull(covariate)

covariate_levels <- rev(c("Lineage_label", "Time_point", "Cell_type"))

# 3) Long format for ggplot
plot_df <- df_top %>%
  select(-total_explained) %>%
  pivot_longer(-gene, names_to = "covariate", values_to = "var_explained") %>%
  mutate(
    gene = factor(gene, levels = df_top$gene),                  # left-to-right by total explained
    covariate = factor(covariate, levels = covariate_levels)    # stacking order
  )

# 4) Colors (Study_Designation fixed to purple; others distinct & colorblind-friendly)
fill_cols <- c(
  "Lineage_label"                  = "#6a0dad",  # purple (focus)
  "Time_point"                   = "#009E73",
  "Cell_type"            = "#E69F00"
)

# If your column names differ in capitalization/spaces, you can inspect:
# setdiff(levels(plot_df$covariate), names(fill_cols))

y_max <- max(df_top$total_explained)

plot1 <- ggplot(plot_df, aes(x = gene, y = var_explained, fill = covariate)) +
  geom_col(width = 0.9, color = NA) +
  scale_fill_manual(values = fill_cols, drop = FALSE) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  coord_cartesian(ylim = c(0, y_max)) +
  labs(x = NULL, y = "Variance explained") +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8),
    axis.ticks.x = element_blank()
  )


fig_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/Joshua/git/SCSeq_LineageBarcoding/fig/kevin/Writeup10c/"

ggplot2::ggsave(plot1,
                filename = paste0(fig_folder, "Writeup10c_celltagmulti_variancePartition_local.png"),
                width = 6, height = 6)




