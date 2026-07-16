#!/usr/bin/env Rscript
# Mean/Sigma linking across IRT chunks, then WLE person-score estimation.
#
# Usage (run from repo root):
#   Rscript scripts/02_wle_scoring.r --benchmark=truthfulqa

library(mirt)

# ── CLI args ──────────────────────────────────────────────────────────────────
parse_args <- function() {
  out <- list()
  for (a in commandArgs(trailingOnly = TRUE))
    if (grepl("^--benchmark=", a)) out$benchmark <- sub("^--benchmark=", "", a)
  out
}
args <- parse_args()
stopifnot(!is.null(args$benchmark))
benchmark <- args$benchmark

# ── Benchmark config (must match 01_fit_irt.r) ───────────────────────────────
CONFIGS <- list(
  arc        = list(chunk_ends = c(seq(106L, 737L, by = 105L), 840L)),
  gsm8k      = list(chunk_ends = c(seq(110L, 1200L, by = 109L), 1307L)),
  hellaswag  = list(chunk_ends = c(seq(113L, 5501L, by = 112L), 5608L)),
  truthfulqa = list(chunk_ends = c(105L, 209L, 313L, 418L, 523L, 628L)),
  winogrande = list(chunk_ends = c(105L, 209L, 313L, 417L, 521L, 626L, 731L, 836L, 941L, 1046L))
)

cfg          <- CONFIGS[[benchmark]]
if (is.null(cfg)) stop("Unknown benchmark: ", benchmark)
chunk_indices <- cfg$chunk_ends
outdir        <- benchmark

# ── Load & clean test response matrix ────────────────────────────────────────
data_file <- paste0("data/gaussian_sampled_", benchmark, "_response_matrix_test.csv")
cat("Loading:", data_file, "\n")
data <- read.csv(data_file)
cat("Dimensions of original data:", dim(data), "\n")

data_clean <- na.omit(data)
data_clean <- data_clean[, colSums(is.na(data_clean)) == 0]

constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data    <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data    <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")
cat("Dimensions of cleaned data:", dim(clean_data), "\n")
data_for_id <- clean_data

# ── Mean/Sigma linking across chunks ─────────────────────────────────────────
scores_list <- lapply(chunk_indices, function(i) {
  read.csv(file.path(outdir, paste0("irt_person_scores_", i, ".csv")))
})
reference_scores <- scores_list[[1]]
cat("Reference scores columns:", colnames(reference_scores), "\n")

linked_params_list <- list()
linked_params_list[[1]] <- read.csv(
  file.path(outdir, paste0("irt_item_parameters_", chunk_indices[1], ".csv")))

for (j in 2:length(scores_list)) {
  i        <- chunk_indices[j]
  params_j <- read.csv(file.path(outdir, paste0("irt_item_parameters_", i, ".csv")))

  sc_ref <- as.numeric(reference_scores$F1)
  sc_j   <- as.numeric(scores_list[[j]]$F1)

  A <- sd(sc_ref, na.rm = TRUE) / sd(sc_j, na.rm = TRUE)
  B <- mean(sc_ref, na.rm = TRUE) - A * mean(sc_j, na.rm = TRUE)
  cat("Chunk", i, "linking: A=", A, "B=", B, "\n")

  params_j_star    <- params_j
  params_j_star$a1 <- params_j$a1 / A
  params_j_star$d  <- A * params_j$d + B * params_j$a1
  if ("g" %in% colnames(params_j)) params_j_star$g <- params_j$g

  linked_params_list[[j]] <- params_j_star
  cat("Correlation after linking:",
      cor(sc_ref, A * sc_j + B, use = "pairwise.complete.obs"), "\n")
}

item_params_combined <- do.call(rbind, linked_params_list)
write.csv(item_params_combined, file.path(outdir, "irt_item_parameters_combined.csv"),
          row.names = FALSE)

# ── Map parameters to data columns ───────────────────────────────────────────
param_items     <- as.character(item_params_combined$X)
param_to_data_map <- list()
for (param_item in param_items) {
  if      (param_item %in% colnames(clean_data))                   param_to_data_map[[param_item]] <- param_item
  else if (gsub("^X", "", param_item) %in% colnames(clean_data))   param_to_data_map[[param_item]] <- gsub("^X", "", param_item)
  else if (paste0("X", param_item) %in% colnames(clean_data))      param_to_data_map[[param_item]] <- paste0("X", param_item)
}

if (length(param_to_data_map) == 0) {
  param_nums <- as.numeric(gsub("[^0-9]", "", param_items))
  data_cols  <- colnames(clean_data)
  data_nums  <- as.numeric(gsub("[^0-9]", "", data_cols))
  for (num in intersect(param_nums, data_nums)) {
    p <- which(param_nums == num)[1]; d <- which(data_nums == num)[1]
    if (!is.na(p) && !is.na(d)) param_to_data_map[[param_items[p]]] <- data_cols[d]
  }
}
cat("Mapped parameters:", length(param_to_data_map), "\n")

clean_params <- data.frame(
  a1 = item_params_combined$a1,
  d  = item_params_combined$d,
  g  = item_params_combined$g,
  row.names = param_items
)
clean_data <- clean_data[, unlist(param_to_data_map), drop = FALSE]

# ── Fit fixed model and estimate WLE scores ───────────────────────────────────
mod_dummy <- mirt(clean_data, 1, itemtype = "3PL", pars = "values")
pars      <- mod_dummy

for (i in 1:nrow(pars)) {
  item_name  <- pars$item[i]
  param_name <- pars$name[i]
  if (param_name %in% c("a1", "d", "g") && item_name %in% rownames(clean_params))
    pars$value[i] <- clean_params[item_name, param_name]
}
pars$est[pars$name %in% c("a1", "d", "g")] <- FALSE

mod_fixed <- mirt(clean_data, 1, itemtype = "3PL", pars = pars, method = "EM",
                  technical = list(NCYCLES = 5000, BURNIN = 1000), verbose = TRUE)

thetas <- fscores(mod_fixed, method = "WLE", full.scores.SE = TRUE,
                  response.pattern = NULL, quadpts = 61)

# ── Handle non-convergence ────────────────────────────────────────────────────
problem_rows <- which(is.na(thetas[, 1]) | is.nan(thetas[, 1]))
if (length(problem_rows) > 0) {
  cat("Re-estimating", length(problem_rows), "non-converged rows...\n")
  fixed <- 0
  for (idx in problem_rows) {
    try({
      r <- fscores(mod_fixed, method = "WLE", response.pattern = clean_data[idx, , drop = FALSE], quadpts = 81)
      if (!is.na(r[1]) && !is.nan(r[1])) { thetas[idx, ] <- r; fixed <- fixed + 1 }
    }, silent = TRUE)
  }
  cat("Fixed", fixed, "of", length(problem_rows), "\n")
  remaining <- which(is.na(thetas[, 1]) | is.nan(thetas[, 1]))
  if (length(remaining) > 0)
    thetas[remaining, 1] <- median(thetas[, 1], na.rm = TRUE)
}

# ── Save ──────────────────────────────────────────────────────────────────────
model_names <- data_for_id[[1]]
theta_df    <- data.frame(Model_Name = model_names, Theta_WLE = thetas[, 1], SE = thetas[, 2])
output_file <- file.path(outdir, "irt_person_scores_WLE_SE_test.csv")
write.csv(theta_df, output_file, row.names = FALSE)
cat("Saved WLE scores to", output_file, "\n")
print(head(theta_df, 10))
