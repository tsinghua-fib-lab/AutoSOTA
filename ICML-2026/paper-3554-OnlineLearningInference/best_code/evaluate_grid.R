#!/usr/bin/env Rscript
# COLSA Alpha-Nu Grid Search Evaluation
library(COLSA)
library(survival)

SEED <- 42
N_TOTAL <- 7315; K_BATCHES <- 18; N_GENES <- 23; SIG_THRESH <- 1.96
PRE_ESTIMATION_FACTOR <- 2

TARGET_GENES <- c(
  "FLNC","GATA4","PIK3R3","ISL2","TREML3P","SERPINE1","PTX3","IL1RAP",
  "PPP1R1C","CDK6.AS1","CYP19A1","FKBP1C","SEMA3G","PGR","PRLR","PDCD4",
  "HOXA2","METTL7B","BCL2","FLNC.AS1","SLC11A1","SEC61G","MYRF"
)
ORACLE_LOGHR <- log(c(1.06,1.03,0.97,1.03,1.05,1.08,1.05,1.00,
  1.04,1.03,1.06,1.01,0.90,0.99,0.98,0.95,1.05,1.03,0.97,1.02,0.98,1.03,1.04))

batch_sizes <- round(c(1073,866,707,578,472,385,315,257,210,171,140,114,93,76,62,51,41,34))
batch_sizes <- round(batch_sizes * N_TOTAL / sum(batch_sizes))
batch_sizes[1] <- batch_sizes[1] + (N_TOTAL - sum(batch_sizes))

Sigma <- matrix(0.3, N_GENES, N_GENES); diag(Sigma) <- 1.0

set.seed(SEED)
X <- MASS::mvrnorm(N_TOTAL, mu=rep(0,N_GENES), Sigma=Sigma)
colnames(X) <- TARGET_GENES
lp <- X %*% ORACLE_LOGHR
event_time <- (-log(runif(N_TOTAL))/(0.1*exp(lp)))^(1/1.5)
cens_time <- rexp(N_TOTAL, 0.05)
obs_time <- pmin(event_time, cens_time)
status <- as.numeric(event_time <= cens_time)
df <- as.data.frame(X); df$time <- obs_time; df$status <- status
df <- df[sample(N_TOTAL),]; df$batch <- rep(1:K_BATCHES, batch_sizes)

fmla <- as.formula(paste("Surv(time, status) ~", paste(TARGET_GENES, collapse=" + ")))
boundary <- c(0, max(df$time))

oracle_fit <- coxph(fmla, data=df)
oracle_z <- summary(oracle_fit)$coefficients[,"z"]
oracle_lhr <- log(exp(coef(oracle_fit)))
oracle_sig <- abs(oracle_z) > SIG_THRESH

batch1 <- df[df$batch==1,]

# Grid: alpha in [0.5, 0.8, 0.94 (default), 1.2, 1.5, 2.0], nu in [0.1, 0.2, 0.3, 0.4, 0.5]
alphas <- c(0.5, 0.7, 0.8, 0.94, 1.0, 1.2, 1.5, 2.0)
nus <- c(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)

results <- data.frame(alpha=numeric(), nu=numeric(), p0=integer(),
  n_correct=integer(), pearson_r=numeric(), mad_loghr=numeric())

cat(sprintf("Grid search: %d alpha x %d nu = %d combinations\n", 
  length(alphas), length(nus), length(alphas)*length(nus)))

for (a in alphas) {
  for (nu_val in nus) {
    cat(sprintf("\n--- alpha=%.2f, nu=%.2f ---\n", a, nu_val))
    tryCatch({
      # Determine p0 from given alpha (reverse of alpha = p0 / n^nu)
      p0 <- max(1, round(a * nrow(batch1)^nu_val))
      cat(sprintf("  Derived p0: %d\n", p0))
      
      colsa_fit <- colsa(fmla, batch1, p0, boundary, scale=PRE_ESTIMATION_FACTOR)
      for (b in 2:K_BATCHES) {
        bdat <- df[df$batch==b,]
        colsa_fit <- update(colsa_fit, bdat, alpha=a, nu=nu_val)
      }
      
      colsa_summ <- summary(colsa_fit)
      colsa_z <- colsa_summ$coefficients[,"z"]
      colsa_lhr <- log(colsa_summ$coefficients[,"exp(coef)"])
      
      colsa_sig <- abs(colsa_z) > SIG_THRESH
      sign_match <- sign(oracle_z) == sign(colsa_z)
      n_correct <- sum((oracle_sig == colsa_sig) & sign_match)
      pearson_r <- cor(oracle_z, colsa_z)
      mad_lhr <- mean(abs(oracle_lhr - colsa_lhr))
      
      results <- rbind(results, data.frame(
        alpha=a, nu=nu_val, p0=p0, n_correct=n_correct,
        pearson_r=pearson_r, mad_loghr=mad_lhr))
      
      cat(sprintf("  Result: %d/23, r=%.4f, MAD=%.5f\n", n_correct, pearson_r, mad_lhr))
    }, error=function(e) {
      cat(sprintf("  ERROR: %s\n", e$message))
      results <<- rbind(results, data.frame(
        alpha=a, nu=nu_val, p0=NA, n_correct=NA,
        pearson_r=NA, mad_loghr=NA))
    })
  }
}

cat("\n=== GRID RESULTS ===\n")
results_valid <- results[!is.na(results$n_correct),]
results_valid <- results_valid[order(-results_valid$n_correct, -results_valid$pearson_r, results_valid$mad_loghr),]
print(results_valid, row.names=FALSE)

best <- results_valid[1,]
cat(sprintf("\n=== BEST: alpha=%.2f, nu=%.2f, p0=%d ===\n", best$alpha, best$nu, best$p0))
cat(sprintf("Correctly Recovered Inference: %d/23\n", best$n_correct))
cat(sprintf("Pearson r of Z-statistics: %.4f\n", best$pearson_r))
cat(sprintf("Mean Absolute Log-HR Difference: %.5f\n", best$mad_loghr))

saveRDS(results, "/repo/grid_results.rds")
