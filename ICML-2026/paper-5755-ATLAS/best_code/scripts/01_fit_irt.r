#!/usr/bin/env Rscript
# Fit IRT model for one column chunk of a benchmark response matrix.
#
# Usage (run from repo root):
#   Rscript scripts/01_fit_irt.r --benchmark=arc --chunk_end=106
#   Rscript scripts/01_fit_irt.r --benchmark=arc --chunk_end=106 --itemtype=2PL --outdir=experiments/arc_2pl
#
# chunk_end must be one of the pre-defined values in CONFIGS below.
# For 2PL runs, pass --itemtype=2PL and --outdir=experiments/<benchmark>_2pl.

library(mirt)

# ── CLI args ──────────────────────────────────────────────────────────────────
parse_args <- function() {
  out <- list(itemtype = "3PL", outdir = NULL)
  for (a in commandArgs(trailingOnly = TRUE)) {
    if      (grepl("^--benchmark=", a)) out$benchmark <- sub("^--benchmark=", "", a)
    else if (grepl("^--chunk_end=", a)) out$chunk_end <- as.integer(sub("^--chunk_end=", "", a))
    else if (grepl("^--itemtype=",  a)) out$itemtype  <- sub("^--itemtype=",  "", a)
    else if (grepl("^--outdir=",    a)) out$outdir    <- sub("^--outdir=",    "", a)
  }
  out
}
args <- parse_args()
stopifnot(!is.null(args$benchmark), !is.null(args$chunk_end))

# ── Benchmark config ──────────────────────────────────────────────────────────
# chunk_ends: the column index at which each chunk ends (1-indexed, inclusive).
# The first chunk always starts at column 2 (column 1 is the model-name label).
# Each subsequent chunk starts at prev_chunk_end + 1.
CONFIGS <- list(
  arc = list(
    data_file  = "data/gaussian_sampled_arc_response_matrix_train_with_scores.csv",
    chunk_ends = c(seq(106L, 737L, by = 105L), 840L),
    outdir     = "arc"
  ),
  gsm8k = list(
    data_file  = "data/gaussian_sampled_gsm8k_response_matrix_train_with_scores.csv",
    chunk_ends = c(seq(110L, 1200L, by = 109L), 1307L),
    outdir     = "gsm8k"
  ),
  hellaswag = list(
    data_file  = "data/gaussian_sampled_hellaswag_response_matrix_train_with_scores.csv",
    chunk_ends = c(seq(113L, 5501L, by = 112L), 5608L),
    outdir     = "hellaswag"
  ),
  truthfulqa = list(
    data_file  = "data/gaussian_sampled_truthfulqa_response_matrix_train_with_scores.csv",
    chunk_ends = c(105L, 209L, 313L, 418L, 523L, 628L),
    outdir     = "truthfulqa"
  ),
  winogrande = list(
    data_file  = "data/gaussian_sampled_winogrande_response_matrix_train_with_scores.csv",
    chunk_ends = c(105L, 209L, 313L, 417L, 521L, 626L, 731L, 836L, 941L, 1046L),
    outdir     = "winogrande"
  )
)

cfg <- CONFIGS[[args$benchmark]]
if (is.null(cfg)) stop("Unknown benchmark: ", args$benchmark)

chunk_end <- args$chunk_end
itemtype  <- args$itemtype
outdir    <- if (!is.null(args$outdir)) args$outdir else cfg$outdir

if (!chunk_end %in% cfg$chunk_ends)
  stop("chunk_end=", chunk_end, " not valid for benchmark=", args$benchmark,
       ". Valid values: ", paste(cfg$chunk_ends, collapse = ", "))

# ── Load & clean data ─────────────────────────────────────────────────────────
cat("Loading:", cfg$data_file, "\n")
data <- read.csv(cfg$data_file)
cat("Dimensions of original data:", dim(data), "\n")

data_clean <- na.omit(data)
data       <- data_clean[, colSums(is.na(data_clean)) == 0]

constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data    <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data    <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")
cat("Dimensions of cleaned data:", dim(clean_data), "\n")

# ── Select chunk columns ──────────────────────────────────────────────────────
chunk_idx <- which(cfg$chunk_ends == chunk_end)
start_col <- if (chunk_idx == 1L) 2L else cfg$chunk_ends[chunk_idx - 1L] + 1L
dat       <- clean_data[, start_col:chunk_end]
cat("Chunk", chunk_idx, "of", length(cfg$chunk_ends),
    ": columns", start_col, "to", chunk_end, "(", ncol(dat), "items)\n")

# ── Fit model ─────────────────────────────────────────────────────────────────
cat("Fitting", itemtype, "model...\n")
model <- mirt(dat, 1, itemtype = itemtype, method = "EM",
              technical = list(NCYCLES = 100000))
print(model)

m2           <- M2(model)
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE,
                        full.scores.SE = TRUE, quadpts = 61)
item_params  <- coef(model, simplify = TRUE)$items
print(item_params)

# ── Save outputs ──────────────────────────────────────────────────────────────
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
tag <- as.character(chunk_end)

write.csv(theta_scores, file.path(outdir, paste0("irt_person_scores_",    tag, ".csv")), row.names = FALSE)
write.csv(item_params,  file.path(outdir, paste0("irt_item_parameters_",  tag, ".csv")), row.names = TRUE)
write.csv(m2,           file.path(outdir, paste0("m2_",                   tag, ".csv")), row.names = TRUE)
cat("Saved outputs to", outdir, "\n")

# ── 2PL-only diagnostics (item fit + Q3 local dependence) ────────────────────
if (itemtype == "2PL") {
  item_fit <- itemfit(model, fit_stats = "S_X2")
  write.csv(item_fit[order(-item_fit$RMSEA.S_X2), ],
            file.path(outdir, paste0("irt_item_fit_", tag, ".csv")), row.names = TRUE)

  q3mat    <- residuals(model, type = "Q3")
  q3_upper <- q3mat
  q3_upper[lower.tri(q3_upper, diag = TRUE)] <- NA
  bad_pairs <- which(q3_upper > 0.20, arr.ind = TRUE)
  q3 <- data.frame(
    Item1 = rownames(q3_upper)[bad_pairs[, 1]],
    Item2 = colnames(q3_upper)[bad_pairs[, 2]],
    Q3    = q3_upper[bad_pairs]
  )
  write.csv(q3[order(-q3$Q3), ],
            file.path(outdir, paste0("q3_", tag, ".csv")), row.names = TRUE)
  cat("Saved 2PL diagnostics to", outdir, "\n")
}
