# 1. Load libraries (Install these via BiocManager if you don't have them)
library(clusterProfiler)
library(org.Mm.eg.db)
library(enrichplot)

# 2. Load your data (Change to ce_ig_scores.csv for the baseline)
df <- read.csv("/Users/apple/Project/Lineage_aware_ContraLearn/analysis/CrossEntropy_sup/train_test/LARRY_top200/lcl_ig_scores.csv")

# 3. Create a named vector and sort it descending
# This is the exact format clusterProfiler requires
teststat_vec <- df$Score
names(teststat_vec) <- df$Gene
teststat_vec <- sort(teststat_vec, decreasing = TRUE)

# 4. Run GSEA
set.seed(10)
gse <- clusterProfiler::gseGO(
  geneList     = teststat_vec,
  ont          = "BP",           # "BP" = Biological Process pathways
  keyType      = "SYMBOL",       # Assuming LARRY genes are symbols (e.g., "Sca1")
  OrgDb        = "org.Mm.eg.db", # Mouse database
  pvalueCutoff = 0.05,           # Only keep statistically significant pathways
  minGSSize    = 10,
  maxGSSize    = 500,
  scoreType    = "pos"           # Because your IG scores are absolute/positive
)

# 5. Extract results to a data frame and view the top hits
gse_df <- as.data.frame(gse)
gse_df <- gse_df[order(gse_df$p.adjust, decreasing = FALSE), ]

# Print the significant pathways
print(head(gse_df[, c("ID", "Description", "p.adjust")]))
write.csv(gse_df,
          "/Users/apple/Project/Lineage_aware_ContraLearn/analysis/CrossEntropy_sup/train_test/LARRY_top200/lcl_gsea_results.csv",
          row.names = FALSE)

# 6. (Optional) Plot the top pathway
enrichplot::gseaplot2(gse, geneSetID = 1, title = gse$Description[1])
