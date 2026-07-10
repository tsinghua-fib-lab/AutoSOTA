#!/usr/bin/env Rscript
# Multi-seed robustness evaluation for CODE-02
library(COLSA)
library(survival)

SEEDS <- c(42, 123, 456, 789, 1024)
N_TOTAL <- 7315
K_BATCHES <- 18
N_GENES <- 23
SIG_THRESH <- 1.96
PRE_ESTIMATION_FACTOR <- 2
NU <- 0.2

TARGET_GENES <- c(
  "FLNC", "GATA4", "PIK3R3", "ISL2", "TREML3P", "SERPINE1",
  "PTX3", "IL1RAP", "PPP1R1C", "CDK6.AS1", "CYP19A1", "FKBP1C",
  "SEMA3G", "PGR", "PRLR", "PDCD4", "HOXA2", "METTL7B",
  "BCL2", "FLNC.AS1", "SLC11A1", "SEC61G", "MYRF"
)

ORACLE_LOGHR <- log(c(1.06, 1.03, 0.97, 1.03, 1.05, 1.08, 1.05, 1.00,
                      1.04, 1.03, 1.06, 1.01, 0.90, 0.99, 0.98, 0.95,
                      1.05, 1.03, 0.97, 1.02, 0.98, 1.03, 1.04))

batch_sizes <- round(c(1073, 866, 707, 578, 472, 385, 315, 257, 210,
                       171, 140, 114, 93, 76, 62, 51, 41, 34))
batch_sizes <- round(batch_sizes * N_TOTAL / sum(batch_sizes))
batch_sizes[1] <- batch_sizes[1] + (N_TOTAL - sum(batch_sizes))

Sigma <- matrix(0.3, N_GENES, N_GENES)
diag(Sigma) <- 1.0

results <- data.frame(
  seed = integer(), n_correct = integer(), pearson_r = numeric(),
  mad_loghr = numeric(), stringsAsFactors = FALSE
)

for (s in SEEDS) {
  cat(sprintf("\n=== Seed %d ===\n", s))
  set.seed(s)
  
  X <- MASS::mvrnorm(N_TOTAL, mu = rep(0, N_GENES), Sigma = Sigma)
  colnames(X) <- TARGET_GENES
  
  lp <- X %*% ORACLE_LOGHR
  U <- runif(N_TOTAL)
  event_time <- (-log(U) / (0.1 * exp(lp)))^(1 / 1.5)
  cens_time <- rexp(N_TOTAL, 0.05)
  obs_time <- pmin(event_time, cens_time)
  status <- as.numeric(event_time <= cens_time)
  
  df <- as.data.frame(X)
  df$time <- obs_time
  df$status <- status
  df <- df[sample(N_TOTAL), ]
  df$batch <- rep(1:K_BATCHES, batch_sizes)
  
  cat(sprintf("Events: %d/%d (%.1f%% censored)\n", 
              sum(status), N_TOTAL, 100*mean(status==0)))
  
  fmla <- as.formula(paste("Surv(time, status) ~", paste(TARGET_GENES, collapse = " + ")))
  boundary <- c(0, max(df$time))
  
  oracle_fit <- coxph(fmla, data = df)
  oracle_z <- summary(oracle_fit)$coefficients[, "z"]
  oracle_lhr <- log(exp(coef(oracle_fit)))
  
  batch1 <- df[df$batch == 1, ]
  aics <- sapply(1:5, function(nb) {
    tryCatch(AIC(colsa(fmla, batch1, nb, boundary, scale = 1)),
             error = function(e) Inf)
  })
  p0 <- which.min(aics)
  alpha_best <- p0 / nrow(batch1)^NU
  
  colsa_fit <- colsa(fmla, batch1, p0, boundary, scale = PRE_ESTIMATION_FACTOR)
  for (b in 2:K_BATCHES) {
    bdat <- df[df$batch == b, ]
    colsa_fit <- update(colsa_fit, bdat, alpha = alpha_best, nu = NU)
  }
  
  colsa_summ <- summary(colsa_fit)
  colsa_z <- colsa_summ$coefficients[, "z"]
  colsa_lhr <- log(colsa_summ$coefficients[, "exp(coef)"])
  
  oracle_sig <- abs(oracle_z) > SIG_THRESH
  colsa_sig <- abs(colsa_z) > SIG_THRESH
  sign_match <- sign(oracle_z) == sign(colsa_z)
  n_correct <- sum((oracle_sig == colsa_sig) & sign_match)
  
  pearson_r <- cor(oracle_z, colsa_z)
  mad_lhr <- mean(abs(oracle_lhr - colsa_lhr))
  
  results <- rbind(results, data.frame(
    seed = s, n_correct = n_correct, pearson_r = pearson_r, mad_loghr = mad_lhr
  ))
  
  cat(sprintf("Correct: %d/23, r=%.4f, MAD=%.5f\n", n_correct, pearson_r, mad_lhr))
}

cat("\n=== AGGREGATE RESULTS ===\n")
cat(sprintf("Correctly Recovered: min=%d, median=%.1f, max=%d\n",
            min(results$n_correct), median(results$n_correct), max(results$n_correct)))
cat(sprintf("Pearson r: min=%.4f, median=%.4f, max=%.4f\n",
            min(results$pearson_r), median(results$pearson_r), max(results$pearson_r)))
cat(sprintf("MAD logHR: min=%.5f, median=%.5f, max=%.5f\n",
            min(results$mad_loghr), median(results$mad_loghr), max(results$mad_loghr)))

# Report worst-case (minimum) as conservative primary metric
worst_n_correct <- min(results$n_correct)
worst_r <- min(results$pearson_r)
worst_mad <- max(results$mad_loghr)

cat(sprintf("\n=== CONSERVATIVE (min primary, min r, max MAD) ===\n"))
cat(sprintf("Correctly Recovered Inference: %d/23\n", worst_n_correct))
cat(sprintf("Pearson r of Z-statistics: %.4f\n", worst_r))
cat(sprintf("Mean Absolute Log-HR Difference: %.5f\n", worst_mad))

saveRDS(results, "/repo/eval_multi_seed_results.rds")
