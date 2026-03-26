################################################################################
# Wasserstein Transfer Learning - Human Mortality Data Experiment
# Reproduces Table 1 (Appendix A) from the paper
#
# Strategy: Use top 24 developed + 138 developing = 162 total countries
# This matches the paper's country count exactly.
################################################################################

library(wpp2015)
library(osqp)
library(Matrix)
library(pracma)

set.seed(42)
M <- 100

cat("Loading WPP2015 mortality data...\n")
data(UNlocations)
data(mxM)
data(mxF)

################################################################################
# Helper functions
################################################################################

to_matrix <- function(x) {
  if (is.vector(x)) matrix(x, ncol=1)
  else if (is.data.frame(x)) as.matrix(x)
  else x
}

compute_L2_distance <- function(vec1, vec2) sqrt(mean((vec1 - vec2)^2))

compute_f1_hat <- function(source_data_list, X_value, M, X_t = NULL, Y_t = NULL) {
  K <- length(source_data_list)
  bigF <- rep(0, M)
  n_total <- 0

  if (!is.null(X_t) && !is.null(Y_t)) {
    X_t_mat <- to_matrix(X_t)
    n_t <- nrow(X_t_mat)
    xbar_t <- colMeans(X_t_mat)
    Sigma_t <- var(as.numeric(X_t_mat)) * (n_t-1)/n_t
    if (Sigma_t < 1e-12) Sigma_t <- 1e-6
    s_vec_t <- as.numeric(1 + (X_t_mat - xbar_t) / Sigma_t * (X_value - xbar_t))
    for (i in seq_len(n_t)) bigF <- bigF + s_vec_t[i] * Y_t[i,]
    n_total <- n_total + n_t
  }

  for (k_idx in seq_len(K)) {
    X_s <- source_data_list[[k_idx]]$X_s
    Y_s <- source_data_list[[k_idx]]$Y_s
    X_s_mat <- to_matrix(X_s)
    n_k <- nrow(X_s_mat)
    xbar <- colMeans(X_s_mat)
    Sigma <- var(as.numeric(X_s_mat)) * (n_k-1)/n_k
    if (Sigma < 1e-12) Sigma <- 1e-6
    s_vec <- as.numeric(1 + (X_s_mat - xbar) / Sigma * (X_value - xbar))
    for (i in seq_len(n_k)) bigF <- bigF + s_vec[i] * Y_s[i,]
    n_total <- n_total + n_k
  }

  bigF / n_total
}

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

age_starts <- c(0, 1, seq(5, 95, by=5), 100)
age_ends   <- c(1, 5, seq(10, 95, by=5), 100, 110)
age_widths <- age_ends - age_starts

mx_to_quantile_norm <- function(mx_vec, M_grid = 100, max_age = 110) {
  n_ages <- length(mx_vec)
  widths <- age_widths[1:n_ages]
  qx <- pmin(1 - exp(-mx_vec * widths), 1)
  qx[is.na(qx) | qx < 0] <- 0
  lx <- cumprod(c(1, 1 - qx))
  dx <- pmax(-diff(lx), 0)
  total <- sum(dx)
  if (total < 1e-10) return(rep(0.5, M_grid))
  prop <- dx / total

  n_fine <- 5000
  age_fine <- seq(0, max_age, length.out = n_fine)
  ae <- age_ends[1:n_ages]
  as_ <- age_starts[1:n_ages]
  wd <- widths

  cdf_fine <- numeric(n_fine)
  for (j in 1:n_fine) {
    a <- age_fine[j]
    cum <- 0
    for (i in 1:n_ages) {
      if (a >= ae[i]) cum <- cum + prop[i]
      else if (a > as_[i]) { cum <- cum + prop[i] * (a - as_[i]) / wd[i]; break }
      else break
    }
    cdf_fine[j] <- min(cum, 1)
  }

  p_grid <- seq(1/M_grid, 1-1/M_grid, length.out = M_grid)
  qf_raw <- sapply(p_grid, function(p) {
    if (p <= cdf_fine[1]) return(age_fine[1])
    if (p >= tail(cdf_fine, 1)) return(age_fine[n_fine])
    approx(cdf_fine, age_fine, xout = p, rule = 2)$y
  })
  qf_raw / max_age  # Normalize to [0,1]
}

compute_e0 <- function(mx_vec) {
  n_ages <- length(mx_vec)
  widths <- age_widths[1:n_ages]
  qx <- pmin(1 - exp(-mx_vec * widths), 1)
  qx[is.na(qx)] <- 0
  lx <- cumprod(c(1, 1 - qx))
  Lx <- (lx[1:n_ages] + lx[2:(n_ages+1)]) / 2 * widths
  sum(Lx)
}

process_country <- function(cc, year_col = "2010-2015") {
  mx_m <- mxM[mxM$country_code == cc, ]
  mx_f <- mxF[mxF$country_code == cc, ]
  if (nrow(mx_m) == 0 || !(year_col %in% names(mx_m))) return(NULL)

  age_num <- suppressWarnings(as.numeric(as.character(mx_m$age)))
  age_num[is.na(age_num)] <- 100
  ord <- order(age_num)
  mx_m <- mx_m[ord, ]; mx_f <- mx_f[ord, ]

  rates_m <- as.numeric(mx_m[[year_col]])
  rates_f <- as.numeric(mx_f[[year_col]])
  if (any(is.na(rates_m)) || any(is.na(rates_f)) || length(rates_m) < 20) return(NULL)

  n_use <- min(length(rates_m), length(age_starts))
  rates <- (rates_m[1:n_use] + rates_f[1:n_use]) / 2

  qf <- tryCatch(mx_to_quantile_norm(rates, M), error = function(e) NULL)
  if (is.null(qf) || any(is.na(qf)) || any(!is.finite(qf))) return(NULL)

  e0 <- tryCatch(compute_e0(rates), error = function(e) NA)
  if (is.na(e0)) return(NULL)

  list(cc = cc, qf = qf, e0 = e0)
}

# Get country classifications
indiv <- UNlocations[UNlocations$location_type == 4,
                     c("name","country_code","agcode_901","agcode_902")]
mx_ccs <- unique(mxM$country_code)
in_both <- intersect(indiv$country_code, mx_ccs)

dev_codes  <- indiv[indiv$country_code %in% in_both & indiv$agcode_901 > 0, "country_code"]
less_codes <- indiv[indiv$country_code %in% in_both & indiv$agcode_902 > 0, "country_code"]

cat("Processing all countries...\n")
dev_res  <- Filter(Negate(is.null), lapply(dev_codes, process_country))
less_res <- Filter(Negate(is.null), lapply(less_codes, process_country))

cat("Valid developed:", length(dev_res), ", Valid developing:", length(less_res), "\n")

# Sort developed countries by life expectancy to pick top 24
dev_e0 <- sapply(dev_res, function(x) x$e0)
ord_dev <- order(-dev_e0)

# Try different n_dev configurations to match paper (24 developed, 138 developing)
# Paper: 24 + 138 = 162

# Try selecting top 24 developed countries and 138 developing countries
dev_24  <- dev_res[ord_dev[1:24]]
less_138 <- less_res[sample(length(less_res), 138)]  # Random sample of 138 from 156

cat("Using 24 developed + 138 developing = 162 countries\n")

X_t_all <- matrix(sapply(dev_24, function(x) x$e0), ncol=1)
Y_t_all <- do.call(rbind, lapply(dev_24, function(x) x$qf))
X_s <- matrix(sapply(less_138, function(x) x$e0), ncol=1)
Y_s <- do.call(rbind, lapply(less_138, function(x) x$qf))

# Normalize life expectancy
all_e0 <- c(X_t_all, X_s)
e0_min <- min(all_e0); e0_max <- max(all_e0)
X_t_norm <- (X_t_all - e0_min) / (e0_max - e0_min)
X_s_norm  <- (X_s - e0_min) / (e0_max - e0_min)

cat("LE range (target norm):", round(range(X_t_norm), 3), "\n")
cat("LE range (source norm):", round(range(X_s_norm), 3), "\n")
cat("QF range:", round(range(Y_t_all), 4), "\n")

source_data <- list(list(X_s = X_s_norm, Y_s = Y_s))

################################################################################
# Run WaTL: n_target = 14, leave-out 10 for testing
################################################################################

n_target_train <- 14
n_dev <- nrow(X_t_all)
n_reps <- 500
lambda_fixed <- 0.25

cat(sprintf("\n=== WaTL: n_target=%d, n_reps=%d ===\n", n_target_train, n_reps))

set.seed(42)
rmspr_vec <- numeric(n_reps)
time_vec  <- numeric(n_reps)

for (rep in 1:n_reps) {
  train_idx <- sample(n_dev, n_target_train)
  test_idx  <- setdiff(1:n_dev, train_idx)

  X_tr <- X_t_norm[train_idx, , drop=FALSE]
  Y_tr <- Y_t_all[train_idx, , drop=FALSE]
  X_te <- X_t_norm[test_idx, , drop=FALSE]
  Y_te <- Y_t_all[test_idx, , drop=FALSE]

  n_te <- nrow(X_te)
  dvals <- numeric(n_te)

  t0 <- proc.time()["elapsed"]

  for (i in 1:n_te) {
    XV  <- as.numeric(X_te[i,])
    tqf <- Y_te[i,]

    f1 <- compute_f1_hat(source_data, XV, M, X_tr, Y_tr)

    xb <- mean(X_tr); Sg <- var(as.numeric(X_tr)) * (nrow(X_tr)-1)/nrow(X_tr)
    if (Sg < 1e-12) Sg <- 1e-6
    sv <- as.numeric(1 + (X_tr - xb) / Sg * (XV - xb))

    f2 <- compute_f_L2(Y_t=Y_tr, s_vec=sv, f1_hat=f1,
                       lambda=lambda_fixed, M=M,
                       max_iter=200, step_size=0.1, tol=1e-3)

    dvals[i] <- compute_L2_distance(tqf, f2)
  }

  t1 <- proc.time()["elapsed"]

  rmspr_vec[rep] <- mean(dvals)
  time_vec[rep]  <- (t1 - t0) * 1000 / n_te

  if (rep %% 100 == 0) {
    cat(sprintf("  Rep %4d: RMSPR=%.5f, Time=%.3fms\n",
                rep, rmspr_vec[rep], time_vec[rep]))
  }
}

cat(sprintf("\nFinal (24 dev, 138 source, n_train=14):\n"))
cat(sprintf("  RMSPR = %.5f ± %.5f\n", mean(rmspr_vec), sd(rmspr_vec)/sqrt(n_reps)))
cat(sprintf("  Time  = %.3f ± %.3f ms\n", mean(time_vec), sd(time_vec)/sqrt(n_reps)))

cat("\n=== REPRODUCTION RESULTS ===\n")
cat(sprintf("Metric: RMSPR, Dataset: Human Mortality Data (Age-at-Death Distributions / 162 Countries 2015 / 14 Target Samples)\n"))
cat(sprintf("Paper reported value: 0.028, CI: [0.02744, 0.02856]\n"))
cat(sprintf("Reproduced value: %.5f\n", mean(rmspr_vec)))
cat(sprintf("Within CI: %s\n", ifelse(mean(rmspr_vec) >= 0.02744 & mean(rmspr_vec) <= 0.02856, "Yes", "No")))
cat("---\n")
cat(sprintf("Metric: Training Time (ms), Dataset: Human Mortality Data\n"))
cat(sprintf("Paper reported value: 0.598, CI: [0.58604, 0.60996]\n"))
cat(sprintf("Reproduced value: %.3f\n", mean(time_vec)))
cat(sprintf("Within CI: %s\n", ifelse(mean(time_vec) >= 0.58604 & mean(time_vec) <= 0.60996, "Yes", "No")))
