#!/usr/bin/env Rscript
# COLSA TCGA Experiment - Reproducible Evaluation
# Usage: Rscript evaluate.R [--real-tcga PATH]
# Default: simulation with paper-derived parameters (reproducible)

library(COLSA)
library(survival)

# ---- Fixed configuration ----
SEED <- 42
N_TOTAL <- 7315
K_BATCHES <- 18
N_GENES <- 23
SIG_THRESH <- 1.96
PRE_ESTIMATION_FACTOR <- 2
NU <- 0.2

# 23 target genes from Table 3
TARGET_GENES <- c(
  "FLNC", "GATA4", "PIK3R3", "ISL2", "TREML3P", "SERPINE1",
  "PTX3", "IL1RAP", "PPP1R1C", "CDK6.AS1", "CYP19A1", "FKBP1C",
  "SEMA3G", "PGR", "PRLR", "PDCD4", "HOXA2", "METTL7B",
  "BCL2", "FLNC.AS1", "SLC11A1", "SEC61G", "MYRF"
)

# Oracle log-HR from Table 3 (paper baseline)
ORACLE_LOGHR <- log(c(1.06, 1.03, 0.97, 1.03, 1.05, 1.08, 1.05, 1.00,
                      1.04, 1.03, 1.06, 1.01, 0.90, 0.99, 0.98, 0.95,
                      1.05, 1.03, 0.97, 1.02, 0.98, 1.03, 1.04))

# ---- Parse args ----
args <- commandArgs(trailingOnly = TRUE)
use_real <- FALSE
real_path <- NULL
if (length(args) > 0 && args[1] == "--real-tcga") {
  use_real <- TRUE
  real_path <- args[2]
}

cat(sprintf("=== COLSA TCGA Experiment (seed=%d) ===\n", SEED))

if (use_real) {
  stop("Real TCGA data path not configured. Set up TCGA data first.")
}

# ---- Generate simulation data ----
set.seed(SEED)

# Batch sizes matching TCGA distribution
batch_sizes <- round(c(1073, 866, 707, 578, 472, 385, 315, 257, 210,
                       171, 140, 114, 93, 76, 62, 51, 41, 34))
batch_sizes <- round(batch_sizes * N_TOTAL / sum(batch_sizes))
batch_sizes[1] <- batch_sizes[1] + (N_TOTAL - sum(batch_sizes))

# Correlated gene expression
Sigma <- matrix(0.3, N_GENES, N_GENES)
diag(Sigma) <- 1.0
X <- MASS::mvrnorm(N_TOTAL, mu = rep(0, N_GENES), Sigma = Sigma)
colnames(X) <- TARGET_GENES

# Generate survival times from Cox model
lp <- X %*% ORACLE_LOGHR
U <- runif(N_TOTAL)
event_time <- (-log(U) / (0.1 * exp(lp)))^(1 / 1.5)
cens_time <- rexp(N_TOTAL, 0.05)
obs_time <- pmin(event_time, cens_time)
status <- as.numeric(event_time <= cens_time)

# Create data frame
df <- as.data.frame(X)
df$time <- obs_time
df$status <- status

# Randomize and assign batches
df <- df[sample(N_TOTAL), ]
df$batch <- rep(1:K_BATCHES, batch_sizes)

cat(sprintf("Data: %d samples, %d events (%.1f%% censored)\n",
            N_TOTAL, sum(status), 100 * mean(status == 0)))

# ---- Oracle (full-data Cox) ----
fmla <- as.formula(paste("Surv(time, status) ~", paste(TARGET_GENES, collapse = " + ")))
boundary <- c(0, max(df$time[df$status == 1]))
cat(sprintf("Boundary: [0, %.2f] (max event time, was max=%.2f)
", boundary[2], max(df$time)))

oracle_fit <- coxph(fmla, data = df)
oracle_z <- summary(oracle_fit)$coefficients[, "z"]
oracle_lhr <- log(exp(coef(oracle_fit)))

# ---- COLSA (online with 18 batches) ----
batch1 <- df[df$batch == 1, ]
bics <- sapply(1:15, function(nb) {
  tryCatch(BIC(colsa(fmla, batch1, nb, boundary, scale = 1)),
           error = function(e) Inf)
})
p0 <- which.min(bics)
alpha_best <- p0 / nrow(batch1)^NU
cat(sprintf("BIC-selected p0: %d (alpha=%.3f)\n", p0, alpha_best))

colsa_fit <- colsa(fmla, batch1, p0, boundary, scale = PRE_ESTIMATION_FACTOR)
for (b in 2:K_BATCHES) {
  bdat <- df[df$batch == b, ]
  colsa_fit <- update(colsa_fit, bdat, alpha = alpha_best, nu = NU)
}

colsa_summ <- summary(colsa_fit)
colsa_z <- colsa_summ$coefficients[, "z"]
colsa_lhr <- log(colsa_summ$coefficients[, "exp(coef)"])

# ---- Metrics ----
oracle_sig <- abs(oracle_z) > SIG_THRESH
colsa_sig <- abs(colsa_z) > SIG_THRESH
sign_match <- sign(oracle_z) == sign(colsa_z)
n_correct <- sum((oracle_sig == colsa_sig) & sign_match)

pearson_r <- cor(oracle_z, colsa_z)
mad_lhr <- mean(abs(oracle_lhr - colsa_lhr))

# Output
cat("\n=== RESULTS ===\n")
cat(sprintf("Correctly Recovered Inference: %d/%d\n", n_correct, N_GENES))
cat(sprintf("Pearson r of Z-statistics: %.4f\n", pearson_r))
cat(sprintf("Mean Absolute Log-HR Difference: %.5f\n", mad_lhr))

# Save results
res <- list(
  seed = SEED,
  n_correct = n_correct,
  n_total = N_GENES,
  pearson_r = pearson_r,
  mad_loghr = mad_lhr,
  oracle_z = oracle_z,
  colsa_z = colsa_z,
  oracle_lhr = oracle_lhr,
  colsa_lhr = colsa_lhr,
  timestamp = Sys.time()
)
saveRDS(res, "/repo/eval_results.rds")
cat("\nResults saved to /repo/eval_results.rds\n")
