################################################################################
# Wasserstein Transfer Learning - Human Mortality Data Experiment
# Reproduces Table 1 (Appendix A) from the paper
#
# Data: UN World Population Prospects 2015
# Target: Developed countries (45 countries)
# Source: Developing countries (156 countries)
# Response: Age-at-death quantile functions
################################################################################

library(wpp2015)
library(osqp)
library(Matrix)
library(pracma)
library(parallel)

# Set seed for reproducibility
set.seed(42)

M <- 100  # Grid size for quantile functions

cat("Loading WPP2015 mortality data...\n")
data(UNlocations)
data(mxM)  # Male mortality rates
data(mxF)  # Female mortality rates

################################################################################
# Helper functions
################################################################################

to_matrix <- function(x) {
  if (is.vector(x)) matrix(x, ncol=1)
  else if (is.data.frame(x)) as.matrix(x)
  else x
}

compute_L2_distance <- function(vec1, vec2) sqrt(mean((vec1 - vec2)^2))

# Step 1: Weighted auxiliary estimator (global Frechet regression)
compute_f1_hat <- function(source_data_list, X_value, M, X_t = NULL, Y_t = NULL) {
  K <- length(source_data_list)
  bigF <- rep(0, M)
  n_total <- 0

  if (!is.null(X_t) && !is.null(Y_t)) {
    X_t_mat <- to_matrix(X_t)
    n_t <- nrow(X_t_mat)
    xbar_t <- colMeans(X_t_mat)
    Sigma_t <- cov(X_t_mat) * (n_t-1)/n_t
    # Handle degenerate case
    if (any(diag(Sigma_t) < 1e-10)) {
      Sigma_t <- Sigma_t + diag(1e-6, ncol(Sigma_t))
    }
    invSigma_t <- solve(Sigma_t)
    s_vec_t <- sapply(1:n_t, function(i) {
      as.numeric(1 + (X_t_mat[i,] - xbar_t) %*% invSigma_t %*% (X_value - xbar_t))
    })
    for (i in seq_len(n_t)) bigF <- bigF + s_vec_t[i] * Y_t[i,]
    n_total <- n_total + n_t
  }

  for (k_idx in seq_len(K)) {
    X_s <- source_data_list[[k_idx]]$X_s
    Y_s <- source_data_list[[k_idx]]$Y_s
    X_s_mat <- to_matrix(X_s)
    n_k <- nrow(X_s_mat)
    xbar <- colMeans(X_s_mat)
    Sigma <- cov(X_s_mat) * (n_k-1)/n_k
    if (any(diag(Sigma) < 1e-10)) Sigma <- Sigma + diag(1e-6, ncol(Sigma))
    invSigma <- solve(Sigma)
    s_vec <- sapply(1:n_k, function(i) {
      as.numeric(1 + (X_s_mat[i,] - xbar) %*% invSigma %*% (X_value - xbar))
    })
    for (i in seq_len(n_k)) bigF <- bigF + s_vec[i] * Y_s[i,]
    n_total <- n_total + n_k
  }

  bigF / n_total
}

# Step 2: Bias correction via gradient descent
compute_f_L2 <- function(Y_t, s_vec, f1_hat, lambda, M, max_iter = 1000, step_size = 0.5, tol = 1e-8) {
  f <- as.numeric(f1_hat)
  for (iter in 1:max_iter) {
    diff_mat <- -sweep(Y_t, 2, f, FUN = "-")
    weighted_diff_mat <- sweep(diff_mat, 1, s_vec, FUN = "*")
    grad <- colSums(weighted_diff_mat) + lambda * (f - f1_hat)
    f_new <- f - step_size * grad
    if (sqrt(sum((f_new - f)^2)) < tol) { f <- f_new; break }
    f <- f_new
  }
  as.numeric(f)
}

################################################################################
# Process mortality data
################################################################################

# Age group definitions for WPP 2015
# Ages: 0, 1, 5, 10, ..., 95, 100 (22 groups)
age_starts <- c(0, 1, seq(5, 95, by=5), 100)
age_ends   <- c(1, 5, seq(10, 95, by=5), 100, 110)  # 110 for 100+ open interval
age_widths <- age_ends - age_starts  # widths: 1, 4, 5, 5, ..., 5, 10

# Convert age-specific mortality rates to age-at-death quantile function
mx_to_quantile <- function(mx_vec, age_starts, age_widths, M_grid = 100) {
  n_ages <- length(mx_vec)
  widths <- age_widths[1:n_ages]

  # Life table: qx = probability of death in interval
  qx <- 1 - exp(-mx_vec * widths)
  qx[is.na(qx) | is.infinite(qx) | qx < 0] <- 0
  qx <- pmin(qx, 1)

  # Survivors: lx
  lx <- cumprod(c(1, 1 - qx))
  # Deaths in each interval
  dx <- -diff(lx)
  dx[dx < 0] <- 0

  total_deaths <- sum(dx)
  if (total_deaths < 1e-10) {
    # Fallback: uniform distribution
    return(seq(0, 100, length.out = M_grid))
  }
  prop <- dx / total_deaths

  # Create CDF over fine age grid
  max_age <- age_ends[n_ages]
  age_fine <- seq(0, max_age, length.out = 2000)
  cdf_fine <- numeric(length(age_fine))

  for (j in seq_along(age_fine)) {
    a <- age_fine[j]
    cum <- 0
    for (i in 1:n_ages) {
      if (a >= age_ends[i]) {
        cum <- cum + prop[i]
      } else if (a > age_starts[i]) {
        cum <- cum + prop[i] * (a - age_starts[i]) / widths[i]
        break
      } else {
        break
      }
    }
    cdf_fine[j] <- min(cum, 1)
  }

  # Quantile function: inverse CDF
  p_grid <- seq(1/M_grid, 1 - 1/M_grid, length.out = M_grid)
  qf <- sapply(p_grid, function(p) {
    if (p <= cdf_fine[1]) return(age_fine[1])
    if (p >= cdf_fine[length(cdf_fine)]) return(age_fine[length(age_fine)])
    approx(cdf_fine, age_fine, xout = p, rule = 2)$y
  })

  return(qf)
}

# Compute life expectancy at birth from mx rates (used as predictor X)
compute_e0 <- function(mx_vec, age_starts, age_widths) {
  n_ages <- length(mx_vec)
  widths <- age_widths[1:n_ages]
  qx <- pmin(1 - exp(-mx_vec * widths), 1)
  qx[is.na(qx)] <- 0
  lx <- cumprod(c(1, 1 - qx))
  # Person-years in each interval: Lx = (lx[i] + lx[i+1]) / 2 * width
  Lx <- (lx[1:n_ages] + lx[2:(n_ages+1)]) / 2 * widths
  sum(Lx)  # e0 = sum of all Lx (since l0 = 1)
}

# Get classification of countries
indiv <- UNlocations[UNlocations$location_type == 4,
                     c("name","country_code","agcode_901","agcode_902")]
mx_ccs <- unique(mxM$country_code)
in_both <- intersect(indiv$country_code, mx_ccs)

dev_codes   <- indiv[indiv$country_code %in% in_both & indiv$agcode_901 > 0, "country_code"]
less_codes  <- indiv[indiv$country_code %in% in_both & indiv$agcode_902 > 0, "country_code"]

cat("Developed countries (UN classification):", length(dev_codes), "\n")
cat("Developing countries (UN classification):", length(less_codes), "\n")

year_col <- "2010-2015"

# Process all countries
process_country <- function(cc, mxM_data, mxF_data, year_col, age_starts, age_widths, M_grid) {
  mx_m <- mxM_data[mxM_data$country_code == cc, ]
  mx_f <- mxF_data[mxF_data$country_code == cc, ]

  if (nrow(mx_m) == 0 || !(year_col %in% names(mx_m))) return(NULL)

  # Sort by age (convert 100+ to 100 for sorting)
  age_str <- mx_m$age
  age_num <- suppressWarnings(as.numeric(age_str))
  age_num[is.na(age_num)] <- 100  # "100+" -> 100

  ord <- order(age_num)
  mx_m <- mx_m[ord, ]
  mx_f <- mx_f[ord, ]

  rates_m <- as.numeric(mx_m[[year_col]])
  rates_f <- as.numeric(mx_f[[year_col]])

  if (any(is.na(rates_m)) || any(is.na(rates_f))) return(NULL)
  if (length(rates_m) != length(age_starts)) {
    # Truncate or pad
    n_use <- min(length(rates_m), length(age_starts))
    rates_m <- rates_m[1:n_use]
    rates_f <- rates_f[1:n_use]
    as_use <- age_starts[1:n_use]
    aw_use <- age_widths[1:n_use]
  } else {
    as_use <- age_starts
    aw_use <- age_widths
  }

  rates <- (rates_m + rates_f) / 2

  qf <- tryCatch(
    mx_to_quantile(rates, as_use, aw_use, M_grid),
    error = function(e) NULL
  )
  if (is.null(qf) || any(is.na(qf))) return(NULL)

  e0 <- tryCatch(
    compute_e0(rates, as_use, aw_use),
    error = function(e) NA
  )

  list(qf = qf, e0 = e0, cc = cc)
}

cat("Processing developed countries...\n")
dev_results <- lapply(dev_codes, process_country, mxM, mxF, year_col, age_starts, age_widths, M)
dev_results <- Filter(Negate(is.null), dev_results)
dev_results <- Filter(function(x) !is.na(x$e0), dev_results)

cat("Processing developing countries...\n")
less_results <- lapply(less_codes, process_country, mxM, mxF, year_col, age_starts, age_widths, M)
less_results <- Filter(Negate(is.null), less_results)
less_results <- Filter(function(x) !is.na(x$e0), less_results)

n_dev <- length(dev_results)
n_less <- length(less_results)
cat("Valid developed countries:", n_dev, "\n")
cat("Valid developing countries:", n_less, "\n")

# Build matrices
X_t_all <- matrix(sapply(dev_results, function(x) x$e0), ncol=1)
Y_t_all <- do.call(rbind, lapply(dev_results, function(x) x$qf))

X_s <- matrix(sapply(less_results, function(x) x$e0), ncol=1)
Y_s <- do.call(rbind, lapply(less_results, function(x) x$qf))

# Single source dataset (all developing countries pooled)
source_data <- list(list(X_s = X_s, Y_s = Y_s))

cat("\nData summary:\n")
cat("Target (developed):", n_dev, "countries, E0 range:", round(range(X_t_all), 1), "\n")
cat("Source (developing):", n_less, "countries, E0 range:", round(range(X_s), 1), "\n")
cat("Quantile function range (target):", round(range(Y_t_all), 1), "\n")

################################################################################
# Run WaTL experiment: n_target = 14
# Evaluate on remaining developed countries
################################################################################

n_target_train <- 14
n_replications <- 100
lambda_fixed <- 0.25  # Fixed lambda as in RealData.R

cat(sprintf("\n=== WaTL Experiment: n_target=%d, lambda=%.2f ===\n", n_target_train, lambda_fixed))

set.seed(42)
rmspr_results <- numeric(n_replications)
time_results <- numeric(n_replications)

for (rep in 1:n_replications) {
  # Random train/test split of developed countries
  train_idx <- sample(n_dev, n_target_train)
  test_idx <- setdiff(1:n_dev, train_idx)

  X_train <- X_t_all[train_idx, , drop=FALSE]
  Y_train <- Y_t_all[train_idx, , drop=FALSE]
  X_test  <- X_t_all[test_idx, , drop=FALSE]
  Y_test  <- Y_t_all[test_idx, , drop=FALSE]

  n_test <- nrow(X_test)
  d_vals <- numeric(n_test)

  t_start <- proc.time()["elapsed"]

  for (i in 1:n_test) {
    X_val <- as.numeric(X_test[i,])
    true_qf <- Y_test[i,]

    # Step 1: auxiliary estimator
    f1 <- compute_f1_hat(source_data, X_val, M, X_train, Y_train)

    # Weights for training data
    X_tr_mat <- to_matrix(X_train)
    xb <- mean(X_tr_mat)
    Sg <- var(as.numeric(X_tr_mat)) * (nrow(X_tr_mat)-1)/nrow(X_tr_mat)
    if (Sg < 1e-10) Sg <- 1e-6
    sv <- as.numeric(1 + (X_tr_mat - xb) / Sg * (X_val - xb))

    # Step 2: bias correction
    f2 <- compute_f_L2(Y_t=Y_train, s_vec=sv, f1_hat=f1,
                       lambda=lambda_fixed, M=M,
                       max_iter=200, step_size=0.1, tol=1e-3)

    d_vals[i] <- compute_L2_distance(true_qf, f2)
  }

  t_end <- proc.time()["elapsed"]
  elapsed_ms <- (t_end - t_start) * 1000

  rmspr_results[rep] <- mean(d_vals)
  time_results[rep] <- elapsed_ms / n_test  # Per-test-point time

  if (rep %% 20 == 0) {
    cat(sprintf("  Rep %3d/%d: RMSPR=%.4f, Time/pt=%.3fms\n",
                rep, n_replications, rmspr_results[rep], time_results[rep]))
  }
}

rmspr_mean <- mean(rmspr_results)
rmspr_se   <- sd(rmspr_results) / sqrt(n_replications)
time_mean  <- mean(time_results)
time_se    <- sd(time_results) / sqrt(n_replications)

cat(sprintf("\n=== RESULTS (n_target=%d) ===\n", n_target_train))
cat(sprintf("RMSPR: %.4f ± %.4f\n", rmspr_mean, rmspr_se))
cat(sprintf("Training Time: %.3f ± %.3f ms\n", time_mean, time_se))

cat("\n=== REPRODUCTION RESULTS ===\n")
cat(sprintf("Metric: RMSPR, Dataset: Human Mortality Data (Age-at-Death / %d Countries 2015 / %d Target Samples)\n",
            n_dev + n_less, n_target_train))
cat(sprintf("Paper reported value: 0.028, CI: [0.02744, 0.02856]\n"))
cat(sprintf("Reproduced value: %.4f\n", rmspr_mean))
within_rmspr <- rmspr_mean >= 0.02744 && rmspr_mean <= 0.02856
cat(sprintf("Within CI: %s\n", ifelse(within_rmspr, "Yes", "No")))
cat("---\n")
cat(sprintf("Metric: Training Time (ms), Dataset: Human Mortality Data\n"))
cat(sprintf("Paper reported value: 0.598, CI: [0.58604, 0.60996]\n"))
cat(sprintf("Reproduced value: %.3f\n", time_mean))
within_time <- time_mean >= 0.58604 && time_mean <= 0.60996
cat(sprintf("Within CI: %s\n", ifelse(within_time, "Yes", "No")))
