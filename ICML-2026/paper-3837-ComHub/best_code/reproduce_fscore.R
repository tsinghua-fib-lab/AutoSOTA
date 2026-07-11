#####################################################
## Reproduction script for paper 3837:
## "Identifying Common Hubs in Multiple Gaussian Graphical Models"
##
## Target rubric metric:
##   F-score for JIC-HD method
##   K=3, p=100, r=5, pC=0.4, pI=0.5, pN=0.05, n=50
##   n_replicates=100, hub_threshold=2sd_above_mean
##   s_hat=floor(sqrt(p))
##
## Reproduction by: AutoSOTA Reproduction Agent
#####################################################

cat("\n========================================\n")
cat("Paper 3837 - JIC-HD F-score Reproduction\n")
cat("========================================\n\n")

# Redirect all output to both console and log file
log_file <- "/repo/reproduction_output.log"
sink(log_file, split = TRUE)

start_time <- Sys.time()
cat("Start time:", format(start_time), "\n\n")

#####################################################
## STEP 1: Load required libraries
#####################################################
cat("Step 1: Loading libraries...\n")
library(magrittr)
library(mvtnorm)
library(Matrix)
library(pracma)

#####################################################
## STEP 2: Source the necessary implementation files
#####################################################
cat("Step 2: Sourcing implementation files...\n")

# Source from simulations4 (most complete parameterization)
source("/repo/simulations4/002_GeneratingMultipleMatrixSparse.R")
source("/repo/simulations4/003_UsefulMatrixTransforms.R")
source("/repo/simulations4/041_Method_JICHD.R")
source("/repo/simulations4/051_Estimation_HubSelection.R")

cat("  All source files loaded.\n")

#####################################################
## STEP 3: Define simulation parameters from rubric
#####################################################
cat("\nStep 3: Setting simulation parameters...\n")

params <- list(
  K              = 3,     # Number of groups
  p              = 100,   # Dimension
  T0             = 100,   # T0 = p (fully connected high-conn submatrix)
  r1             = 5,     # Number of common hubs
  r2             = 5,     # Number of individual hubs per group

  # Hub connection probabilities
  ph1min         = 0.4,   # pC = 0.4 (common hubs, lower bound)
  ph1max         = 0.7,   # pC range upper bound (ph1min + 0.3)
  ph2min         = 0.5,   # pI = 0.5 (individual hubs, lower bound)
  ph2max         = 0.8,   # pI range upper bound (ph2min + 0.3)

  # Noise probabilities
  pnh            = 0.05,  # pN = 0.05 (non-hub edge prob in T0xT0)
  pneff          = 0.01,  # Edge prob outside T0xT0

  # Sample size
  n              = 50,    # n = 50 samples per group

  # Matrix generation
  diagonal_shift = 2,     # Minimum eigenvalue shift
  shuffle        = FALSE,
  type           = "unif",

  # Edge weight magnitudes
  hmin1          = 4, hmax1 = 5,  # Common hub edge weights
  hmin2          = 4, hmax2 = 5,  # Individual hub edge weights
  nhmin          = 4, nhmax = 5,  # Non-hub edge weights
  neffmin        = 4, neffmax = 5, # Out-of-T0 edge weights

  # Hub sets
  Hjoint         = 1:5,   # Common hubs: nodes 1-5
  Hind           = lapply(1:3, function(k) { k * 25 + 1:5 }),
                          # Individual hubs: {26-30}, {51-55}, {76-80}

  # JIC-HD method parameters
  ndir           = 10,    # floor(sqrt(p)) = floor(sqrt(100)) = 10
  nstarts        = 10,    # Number of random starts for SGD
  alpha          = 0.1,   # Learning rate
  max_iter       = 500,   # Max SGD iterations
  tau           = 5.0,   # Softmax temperature for gradient weighting

  # Detection threshold
  hub_threshold_sd = 2,   # 2 standard deviations above mean

  # Number of replicates
  n_replicates   = 100
)

cat("  Parameters:\n")
cat(sprintf("    K=%d, p=%d, r1=%d, r2=%d\n", params$K, params$p, params$r1, params$r2))
cat(sprintf("    ph1=[%.1f,%.1f], ph2=[%.1f,%.1f]\n",
            params$ph1min, params$ph1max, params$ph2min, params$ph2max))
cat(sprintf("    pnh=%.2f, pneff=%.2f\n", params$pnh, params$pneff))
cat(sprintf("    n=%d, ndir=%d, n_replicates=%d\n",
            params$n, params$ndir, params$n_replicates))

true_hubs <- params$Hjoint
cat(sprintf("    True common hubs: %s\n", paste(true_hubs, collapse = ", ")))

#####################################################
## STEP 4: Run simulation replicates
#####################################################
cat("\nStep 4: Running simulation replicates...\n")
cat(sprintf("  Target: %d replicates\n", params$n_replicates))

# Storage for results
results <- data.frame(
  replicate   = integer(params$n_replicates),
  tp           = numeric(params$n_replicates),
  fp           = numeric(params$n_replicates),
  tn           = numeric(params$n_replicates),
  fn           = numeric(params$n_replicates),
  precision    = numeric(params$n_replicates),
  recall       = numeric(params$n_replicates),
  fscore       = numeric(params$n_replicates),
  time_sec     = numeric(params$n_replicates),
  stringsAsFactors = FALSE
)

# Progress tracking
loop_start <- Sys.time()
completed <- 0

for (rep_idx in 1:params$n_replicates) {
  rep_start <- Sys.time()

  #############################################
  ## 4.1 Generate precision matrices
  cat(sprintf("\n  Replicate %d/%d: ", rep_idx, params$n_replicates))

  pmlist <- r.sparse.pdhubmat_list(
    p = params$p, T0 = params$T0, K = params$K,
    Hjoint = params$Hjoint, Hind_list = params$Hind,
    ph1min = params$ph1min, ph1max = params$ph1max,
    ph2min = params$ph2min, ph2max = params$ph2max,
    pnh = params$pnh, pneff = params$pneff,
    diagonal_shift = params$diagonal_shift,
    shuffle = params$shuffle, type = params$type,
    hmin1 = params$hmin1, hmax1 = params$hmax1,
    hmin2 = params$hmin2, hmax2 = params$hmax2,
    nhmin = params$nhmin, nhmax = params$nhmax,
    neffmin = params$neffmin, neffmax = params$neffmax,
    verbose = FALSE)

  covlist <- lapply(pmlist, solve)

  #############################################
  ## 4.2 Generate Gaussian data
  Xlist <- lapply(covlist, function(sigma) {
    rmvnorm(n = params$n, sigma = sigma, method = "svd")
  })
  scorlist <- lapply(Xlist, cor)

  #############################################
  ## 4.3 Run JIC-HD method
  stiefel_result <- sgd.stiefel(
    sigmalist = scorlist,
    p = params$p,
    ndir = params$ndir,
    K = params$K,
    type = "M",
    nstarts = params$nstarts,
    alpha = params$alpha,
    max.iter = params$max_iter,
    tau = params$tau)

  #############################################
  ## 4.4 Compute importance measures
  # Importance = row-wise sum of squared eigenvector loadings
  importance <- apply(t(t(stiefel_result$vectors^2)), MARGIN = 1, sum)

  #############################################
  ## 4.5 Hub selection: 2-sigma threshold
  imp_mean <- mean(importance)
  imp_sd   <- sd(importance)
  threshold <- imp_mean + params$hub_threshold_sd * imp_sd

  predicted_hubs <- which(importance > threshold)

  #############################################
  ## 4.6 Compute metrics
  tp_count <- sum(predicted_hubs %in% true_hubs)
  fp_count <- sum(!(predicted_hubs %in% true_hubs))
  fn_count <- sum(!(true_hubs %in% predicted_hubs))
  tn_count <- params$p - tp_count - fp_count - fn_count

  # Precision = TP / (TP + FP)
  precision_val <- if (tp_count + fp_count > 0) tp_count / (tp_count + fp_count) else 0

  # Recall = TP / (TP + FN) = TPR
  recall_val <- tp_count / length(true_hubs)

  # F-score = 2 * P * R / (P + R)
  fscore_val <- if (precision_val + recall_val > 0) {
    2 * precision_val * recall_val / (precision_val + recall_val)
  } else 0

  rep_time <- as.numeric(difftime(Sys.time(), rep_start, units = "secs"))

  results[rep_idx, ] <- c(
    rep_idx, tp_count, fp_count, tn_count, fn_count,
    round(precision_val, 6), round(recall_val, 6),
    round(fscore_val, 6), round(rep_time, 2))

  cat(sprintf("F=%.4f (P=%.4f, R=%.4f) in %.1fs",
              fscore_val, precision_val, recall_val, rep_time))

  completed <- completed + 1

  # Progress estimate
  elapsed <- as.numeric(difftime(Sys.time(), loop_start, units = "mins"))
  if (rep_idx %% 10 == 0) {
    eta <- elapsed / rep_idx * (params$n_replicates - rep_idx)
    cat(sprintf("\n  Progress: %d/%d (%.1f%%) | Elapsed: %.1f min | ETA: %.1f min",
                rep_idx, params$n_replicates,
                100 * rep_idx / params$n_replicates,
                elapsed, eta))
  }
}

#####################################################
## STEP 5: Summarize results
#####################################################
cat("\n\n========================================\n")
cat("Step 5: Results Summary\n")
cat("========================================\n\n")

# Compute summary statistics
mean_fscore   <- mean(results$fscore)
sd_fscore     <- sd(results$fscore)
mean_prec     <- mean(results$precision)
mean_recall   <- mean(results$recall)
se_fscore     <- sd_fscore / sqrt(params$n_replicates)

cat(sprintf("Number of replicates:      %d\n", params$n_replicates))
cat(sprintf("Mean F-score:              %.4f\n", mean_fscore))
cat(sprintf("SD of F-score:             %.4f\n", sd_fscore))
cat(sprintf("SE of F-score:             %.4f\n", se_fscore))
cat(sprintf("Mean Precision:            %.4f\n", mean_prec))
cat(sprintf("Mean Recall (TPR):         %.4f\n", mean_recall))
cat(sprintf("\nRubric target F-score:     %.2f\n", 0.73))
cat(sprintf("Rubric CI bounds:          [%.4f, %.4f]\n", 0.0800, 0.7950))
cat(sprintf("\nWithin CI bounds:          %s\n",
            ifelse(mean_fscore >= 0.0800 && mean_fscore <= 0.7950, "YES", "NO")))

cat(sprintf("\nTotal elapsed time:        %.1f minutes\n",
            as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

cat("\nDetailed per-replicate results:\n")
print(results, row.names = FALSE)

# Save results
saveRDS(results, file = "/repo/reproduction_results.rds")
cat("\nResults saved to /repo/reproduction_results.rds\n")

# Print the key metric for manifest
cat(sprintf("\n\n>>> REPRODUCTION_METRIC: F-score = %.4f\n", mean_fscore))

sink()
cat("\nReproduction complete. Output saved to /repo/reproduction_output.log\n")
