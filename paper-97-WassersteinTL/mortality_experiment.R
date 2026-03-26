################################################################################
# Wasserstein Transfer Learning - Human Mortality Data Experiment
# Reproduces Table 1 (Appendix A) from the paper
#
# Data: UN World Population Prospects 2015 age-specific mortality rates
# Target: Developed countries (24 countries)
# Source: Developing countries (138 countries)
# Response: Age-at-death quantile functions
#
# This implements the WaTL algorithm using global Frechet regression
# following the same structure as Simulation.R but with real mortality data.
################################################################################

library(wpp2015)
library(osqp)
library(Matrix)
library(pracma)
library(parallel)
library(doParallel)
library(foreach)

# Set seed for reproducibility
set.seed(42)

# Grid size for quantile functions
M <- 100

cat("Loading WPP2015 mortality data...\n")
data(UNlocations)
data(mxM)  # Male mortality rates
data(mxF)  # Female mortality rates

################################################################################
# Helper functions (from SimulationFunc.R)
################################################################################

to_matrix <- function(x) {
  if (is.vector(x)) {
    matrix(x, ncol=1)
  } else if (is.data.frame(x)) {
    as.matrix(x)
  } else {
    x
  }
}

compute_L2_distance <- function(vec1, vec2){
  sqrt(mean((vec1 - vec2)^2))
}

gcd <- function(n, m) {
  n <- abs(n); m <- abs(m)
  if (n == 0 && m == 0) return(0)
  if (m > n) { t <- n; n <- m; m <- t }
  while (m > 0) { t <- n; n <- m; m <- t %% m }
  n
}

lcm <- function(n, m) {
  if (n == 0 && m == 0) return(0)
  n / gcd(n, m) * m
}

plcm <- function(x) {
  x <- x[x != 0]
  n <- length(x)
  if (n == 0) return(0)
  if (n == 1) return(x)
  l <- lcm(x[1], x[2])
  if (n > 2) for (i in 3:n) l <- lcm(l, x[i])
  l
}

# Global Frechet regression (from SimulationFunc.R)
grem <- function(y = NULL, x = NULL, xOut = NULL, optns = list()) {
  if (is.null(y) | is.null(x)) stop("requires the input of both y and x")
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) x <- as.matrix(x)
    else stop("x must be a matrix or a data frame or a vector")
  }
  n <- nrow(x); p <- ncol(x)
  if (!is.list(y)) stop("y must be a list")
  if (length(y) != n) stop("length mismatch")
  if (!is.null(xOut)) {
    if (!is.matrix(xOut)) {
      if (is.data.frame(xOut)) xOut <- as.matrix(xOut)
      else if (is.vector(xOut)) {
        if (p == 1) xOut <- as.matrix(xOut)
        else xOut <- t(xOut)
      }
    }
    nOut <- nrow(xOut)
  } else {
    nOut <- 0
  }

  N <- sapply(y, length)
  y <- lapply(1:n, function(i) sort(y[[i]]))
  M_grid <- min(plcm(N), n * max(N), 5000)
  yM <- t(sapply(1:n, function(i) {
    residual <- M_grid %% N[i]
    if (residual) sort(c(rep(y[[i]], each = M_grid %/% N[i]), sample(y[[i]], residual)))
    else rep(y[[i]], each = M_grid %/% N[i])
  }))

  A <- cbind(diag(M_grid), rep(0, M_grid)) + cbind(rep(0, M_grid), -diag(M_grid))
  A <- A[, -c(1, ncol(A))]
  l <- rep(0, M_grid - 1)
  P <- diag(M_grid)
  A <- t(A)
  q <- rep(0, M_grid)
  u <- rep(Inf, length(l))
  model <- osqp::osqp(P = P, q = q, A = A, l = l, u = u,
                      osqp::osqpSettings(max_iter = 1e05, eps_abs = 1e-05, eps_rel = 1e-05, verbose = FALSE))

  xMean <- colMeans(x)
  invVa <- solve(var(x) * (n - 1) / n)
  wc <- t(apply(x, 1, function(xi) t(xi - xMean) %*% invVa))
  if (nrow(wc) != n) wc <- t(wc)

  qf <- matrix(nrow = n, ncol = M_grid)
  residuals <- rep.int(0, n)
  totVa <- sum((scale(yM, scale = FALSE))^2) / M_grid
  for (i in 1:n) {
    w <- apply(wc, 1, function(wci) 1 + t(wci) %*% (x[i, ] - xMean))
    qNew <- apply(yM, 2, weighted.mean, w)
    if (any(w < 0)) { model$Update(q = -qNew); qNew <- sort(model$Solve()$x) }
    qf[i, ] <- qNew
    residuals[i] <- sqrt(sum((yM[i, ] - qf[i, ])^2) / M_grid)
  }

  if (nOut > 0) {
    qp <- matrix(nrow = nOut, ncol = M_grid)
    for (i in 1:nOut) {
      w <- apply(wc, 1, function(wci) 1 + t(wci) %*% (xOut[i, ] - xMean))
      qNew <- apply(yM, 2, weighted.mean, w)
      if (any(w < 0)) { model$Update(q = -qNew); qNew <- sort(model$Solve()$x) }
      qp[i, ] <- qNew
    }
    res <- list(qf = qf, qp = qp)
  } else {
    res <- list(qf = qf)
  }
  class(res) <- "rem"
  res
}

# Step 1: Weighted auxiliary estimator
compute_f1_hat <- function(source_data_list, X_value, M, X_t = NULL, Y_t = NULL) {
  K <- length(source_data_list)
  z_grid <- seq(1/M, 1-1/M, length.out = M)
  bigF <- rep(0, M)
  n_total <- 0

  if (!is.null(X_t) && !is.null(Y_t)) {
    n_t <- nrow(Y_t)
    X_t_mat <- to_matrix(X_t)
    xbar_t <- colMeans(X_t_mat)
    Sigma_t <- cov(X_t_mat) * (nrow(X_t_mat)-1)/nrow(X_t_mat)
    invSigma_t <- solve(Sigma_t)
    s_vec_t <- sapply(1:nrow(X_t_mat), function(i) {
      as.numeric(1 + (X_t_mat[i,] - xbar_t) %*% invSigma_t %*% (X_value - xbar_t))
    })
    for (i in seq_len(n_t)) bigF <- bigF + s_vec_t[i] * Y_t[i,]
    n_total <- n_total + n_t
  }

  for (k_idx in seq_len(K)) {
    X_s <- source_data_list[[k_idx]]$X_s
    Y_s <- source_data_list[[k_idx]]$Y_s
    n_k <- nrow(Y_s)
    X_s_mat <- to_matrix(X_s)
    xbar <- colMeans(X_s_mat)
    Sigma <- cov(X_s_mat) * (nrow(X_s_mat)-1)/nrow(X_s_mat)
    invSigma <- solve(Sigma)
    s_vec <- sapply(1:nrow(X_s_mat), function(i) {
      as.numeric(1 + (X_s_mat[i,] - xbar) %*% invSigma %*% (X_value - xbar))
    })
    for (i in seq_len(n_k)) bigF <- bigF + s_vec[i] * Y_s[i,]
    n_total <- n_total + n_k
  }

  bigF / n_total
}

# Step 2: Bias correction via gradient descent
compute_f_L2 <- function(Y_t, s_vec, f1_hat, lambda, M, max_iter = 1000, step_size = 0.5, tol = 1e-8) {
  n0 <- nrow(Y_t)
  f <- as.numeric(f1_hat)

  for (iter in 1:max_iter) {
    diff_mat <- sweep(Y_t, 2, f, FUN = "-")
    diff_mat <- -diff_mat
    weighted_diff_mat <- sweep(diff_mat, 1, s_vec, FUN = "*")
    gradient_target <- colSums(weighted_diff_mat)
    gradient_reg <- lambda * (f - f1_hat)
    grad <- gradient_target + gradient_reg
    f_new <- f - step_size * grad
    diff_norm <- sqrt(sum((f_new - f)^2))
    if (diff_norm < tol) { f <- f_new; break }
    f <- f_new
  }
  as.numeric(f)
}

################################################################################
# Process mortality data
################################################################################
# Convert age-specific mortality rates to age-at-death quantile functions
# Using both male and female data (combined)
# Year: 2010-2015 period (best available for 2015 data in wpp2015 package)

year_col <- "2010-2015"

# Standard age groups (0, 1, 5, 10, ..., 95, 100)
# Age midpoints for each group
age_starts <- c(0, 1, 5, seq(10, 95, by=5), 100)
age_ends <- c(1, 5, seq(10, 95, by=5), 100, 120)
age_widths <- age_ends - age_starts
age_mids <- (age_starts + age_ends) / 2

# Function to convert mortality rates (mx) to age-at-death quantile function
# Method: Use life table to get density, then compute quantile function
mx_to_quantile <- function(mx_vec, M_grid = 100) {
  n_ages <- length(mx_vec)
  widths <- age_widths[1:n_ages]

  # Convert mx to probability of death in each interval (qx)
  # Using: qx = 1 - exp(-mx * width) for each age group
  qx <- 1 - exp(-mx_vec * widths)
  qx[is.na(qx) | is.infinite(qx)] <- 1
  qx <- pmin(qx, 1)

  # Build life table: lx (survivors)
  lx <- numeric(n_ages + 1)
  lx[1] <- 1
  for (i in 1:n_ages) {
    lx[i+1] <- lx[i] * (1 - qx[i])
  }

  # Deaths in each age group: dx = lx - l(x+1)
  dx <- diff(lx)
  dx[dx < 0] <- 0

  # Expected deaths distribution (density)
  # Each age group contributes deaths uniformly within the interval
  total_deaths <- sum(dx)
  if (total_deaths < 1e-10) return(seq(age_starts[1], age_ends[n_ages], length.out = M_grid))

  prop <- dx / total_deaths

  # Create fine-grained CDF
  # Sample ages from the distribution
  age_grid <- seq(0, 110, length.out = 1000)
  cdf <- numeric(length(age_grid))

  for (j in seq_along(age_grid)) {
    a <- age_grid[j]
    cum <- 0
    for (i in 1:n_ages) {
      if (a >= age_ends[i]) {
        cum <- cum + prop[i]
      } else if (a > age_starts[i]) {
        # Partial age group: assume uniform distribution within group
        cum <- cum + prop[i] * (a - age_starts[i]) / widths[i]
        break
      } else {
        break
      }
    }
    cdf[j] <- cum
  }

  # Quantile function: inverse of CDF
  p_grid <- seq(1/M_grid, 1-1/M_grid, length.out = M_grid)

  # Interpolate to get quantile function values
  # For each probability level p, find corresponding age
  qf <- sapply(p_grid, function(p) {
    if (p <= min(cdf)) return(age_grid[1])
    if (p >= max(cdf)) return(age_grid[length(age_grid)])
    approx(cdf, age_grid, xout = p, rule = 2)$y
  })

  return(qf)
}

# Get countries with mortality data
indiv <- UNlocations[UNlocations$location_type == 4, c("name","country_code","agcode_901","agcode_902")]
mx_countries <- unique(mxM$country_code)
in_both <- intersect(indiv$country_code, mx_countries)

# Classify countries
dev_codes <- indiv[indiv$country_code %in% in_both & indiv$agcode_901 > 0, "country_code"]
less_codes <- indiv[indiv$country_code %in% in_both & indiv$agcode_902 > 0, "country_code"]

cat("Developed countries:", length(dev_codes), "\n")
cat("Developing countries:", length(less_codes), "\n")

# Process mortality data for all countries
# Use both male and female combined
process_country_mortality <- function(country_code, year_col = "2010-2015", M_grid = 100) {
  mx_m <- mxM[mxM$country_code == country_code, ]
  mx_f <- mxF[mxF$country_code == country_code, ]

  # Check if data exists for this year
  if (!(year_col %in% names(mx_m)) || nrow(mx_m) == 0) return(NULL)

  # Get mortality rates (average of male and female)
  rates_m <- as.numeric(mx_m[[year_col]])
  rates_f <- as.numeric(mx_f[[year_col]])

  # Handle missing/NA values
  if (any(is.na(rates_m)) || any(is.na(rates_f))) return(NULL)

  rates_combined <- (rates_m + rates_f) / 2

  # Convert to quantile function
  qf <- mx_to_quantile(rates_combined, M_grid)
  return(qf)
}

cat("Processing developed countries...\n")
dev_qf_list <- list()
dev_codes_valid <- c()

for (code in dev_codes) {
  qf <- process_country_mortality(code, year_col, M)
  if (!is.null(qf) && !any(is.na(qf))) {
    dev_qf_list[[as.character(code)]] <- qf
    dev_codes_valid <- c(dev_codes_valid, code)
  }
}

cat("Processing developing countries...\n")
less_qf_list <- list()
less_codes_valid <- c()

for (code in less_codes) {
  qf <- process_country_mortality(code, year_col, M)
  if (!is.null(qf) && !any(is.na(qf))) {
    less_qf_list[[as.character(code)]] <- qf
    less_codes_valid <- c(less_codes_valid, code)
  }
}

cat("Valid developed countries:", length(dev_codes_valid), "\n")
cat("Valid developing countries:", length(less_codes_valid), "\n")

################################################################################
# Setup predictor variable
# The paper doesn't specify which predictor X is used.
# Common approach: Use GDP per capita, HDI, or geographic coordinates.
# Since not specified, we use a rank-based predictor (development level proxy).
#
# Alternative: Use GDP data from the World Bank or use a simple rank predictor.
# We use the UN location's development classification as a continuous predictor.
#
# Actually, for Global Frechet regression with mortality data, the predictor X
# is likely a country-level covariate like GDP, life expectancy at birth, etc.
#
# The most natural choice for mortality data regression is to use
# the HDI (Human Development Index) or GDP as predictor.
# Since we don't have that, we use a simulated predictor based on mortality level.
#
# Better approach: use the life expectancy at birth as the predictor
# (this is computable from the mortality data itself and is a standard
# measure of development level)
################################################################################

# Compute life expectancy at birth from mortality rates (as predictor X)
compute_life_expectancy <- function(country_code, year_col = "2010-2015") {
  mx_m <- mxM[mxM$country_code == country_code, ]
  mx_f <- mxF[mxF$country_code == country_code, ]

  if (!(year_col %in% names(mx_m)) || nrow(mx_m) == 0) return(NA)

  rates_m <- as.numeric(mx_m[[year_col]])
  rates_f <- as.numeric(mx_f[[year_col]])

  if (any(is.na(rates_m)) || any(is.na(rates_f))) return(NA)

  rates <- (rates_m + rates_f) / 2

  n_ages <- length(rates)
  widths <- age_widths[1:n_ages]

  # Life table
  qx <- 1 - exp(-rates * widths)
  qx <- pmin(qx, 1)

  lx <- numeric(n_ages + 1)
  lx[1] <- 1
  for (i in 1:n_ages) lx[i+1] <- lx[i] * (1 - qx[i])

  dx <- diff(lx)
  dx[dx < 0] <- 0

  # Life expectancy at each age: sum of lx / l0 * width
  # At birth: e0 = sum(Lx) where Lx = (lx + l(x+1))/2 * width
  Lx <- ((lx[1:n_ages] + lx[2:(n_ages+1)]) / 2) * widths
  e0 <- sum(Lx) + lx[n_ages+1] * 10  # Add remaining years for 100+
  return(e0)
}

# Compute life expectancy for all valid countries
cat("Computing life expectancy (predictor X)...\n")
dev_le <- sapply(dev_codes_valid, compute_life_expectancy, year_col = year_col)
less_le <- sapply(less_codes_valid, compute_life_expectancy, year_col = year_col)

# Remove any with NA LE
valid_dev <- !is.na(dev_le)
valid_less <- !is.na(less_le)
dev_codes_valid <- dev_codes_valid[valid_dev]
dev_qf_list <- dev_qf_list[valid_dev]
dev_le <- dev_le[valid_dev]

less_codes_valid <- less_codes_valid[valid_less]
less_qf_list <- less_qf_list[valid_less]
less_le <- less_le[valid_less]

cat("Final developed countries:", length(dev_codes_valid), "\n")
cat("Final developing countries:", length(less_codes_valid), "\n")
cat("Life expectancy range (target):", range(dev_le), "\n")
cat("Life expectancy range (source):", range(less_le), "\n")

################################################################################
# Prepare data matrices
################################################################################
# Target: developed countries
n_dev_total <- length(dev_codes_valid)
X_t_all <- matrix(dev_le, ncol = 1)
Y_t_all <- do.call(rbind, dev_qf_list)
rownames(Y_t_all) <- NULL

# Source: developing countries (single source dataset)
n_less <- length(less_codes_valid)
X_s <- matrix(less_le, ncol = 1)
Y_s <- do.call(rbind, less_qf_list)
rownames(Y_s) <- NULL

source_data <- list(list(X_s = X_s, Y_s = Y_s))

cat("\nData prepared:\n")
cat("Target (developed):", n_dev_total, "countries\n")
cat("Source (developing):", n_less, "countries\n")

p_grid <- seq(1/M, 1-1/M, length.out = M)
z_grid <- seq(1/M, 1-1/M, length.out = M)

################################################################################
# Run WaTL experiment with varying target sample sizes (Table 1)
# n_target_sizes = c(14, 19, 24)
################################################################################

run_watl_experiment <- function(n_target, X_t_all, Y_t_all, source_data, M,
                                n_replications = 20, seed = 42) {
  set.seed(seed)
  n_total <- nrow(Y_t_all)

  rmspr_watl <- numeric(n_replications)
  train_times <- numeric(n_replications)

  for (rep in 1:n_replications) {
    # Sample n_target countries as training set, rest as test
    # When n_target < n_total, use a random sample for training and test on all
    train_idx <- sample(n_total, n_target)
    test_idx <- setdiff(1:n_total, train_idx)

    # When n_target == n_total, use leave-one-out or 5-fold CV
    if (length(test_idx) == 0) {
      test_idx <- train_idx  # Use training data for evaluation (over-fitted estimate)
    }

    X_train <- X_t_all[train_idx, , drop = FALSE]
    Y_train <- Y_t_all[train_idx, , drop = FALSE]
    X_test <- X_t_all[test_idx, , drop = FALSE]
    Y_test <- Y_t_all[test_idx, , drop = FALSE]

    n_test <- nrow(X_test)
    d_watl <- numeric(n_test)

    start_time <- proc.time()["elapsed"]

    for (i in 1:n_test) {
      X_value <- as.numeric(X_test[i, ])
      true_qf <- Y_test[i, ]

      # Step 1: Weighted auxiliary estimator
      f1_hat <- compute_f1_hat(source_data, X_value, M, X_train, Y_train)

      # Compute global Frechet weights for target training data
      X_train_mat <- to_matrix(X_train)
      xbar <- colMeans(X_train_mat)
      Sigma <- cov(X_train_mat) * (nrow(X_train_mat)-1)/nrow(X_train_mat)
      invSigma <- solve(Sigma)
      s_vec_t <- sapply(1:nrow(X_train_mat), function(j) {
        as.numeric(1 + (X_train_mat[j,] - xbar) %*% invSigma %*% (X_value - xbar))
      })

      # Step 2: Bias correction
      f_hat <- compute_f_L2(Y_t = Y_train,
                            s_vec = s_vec_t,
                            f1_hat = f1_hat,
                            lambda = 0.1,
                            M = M,
                            max_iter = 200,
                            step_size = 0.1,
                            tol = 1e-6)

      d_watl[i] <- compute_L2_distance(true_qf, f_hat)
    }

    end_time <- proc.time()["elapsed"]
    elapsed_ms <- (end_time - start_time) * 1000

    rmspr_watl[rep] <- mean(d_watl)
    train_times[rep] <- elapsed_ms / n_test  # Per-prediction time

    cat(sprintf("  Rep %d/%d: RMSPR=%.4f, Time/pred=%.3fms\n",
                rep, n_replications, rmspr_watl[rep], train_times[rep]))
  }

  list(
    rmspr = mean(rmspr_watl),
    rmspr_se = sd(rmspr_watl) / sqrt(n_replications),
    train_time_ms = mean(train_times),
    train_time_se = sd(train_times) / sqrt(n_replications)
  )
}

################################################################################
# Run for n_target = 14 (main rubric target)
################################################################################
cat("\n=== Running WaTL with n_target = 14 ===\n")

# For n_target = 14, we test on remaining developed countries
# Use leave-one-out style: split 24 target countries into 14 training / 10 test
# Run multiple times and average

n_dev <- nrow(Y_t_all)
cat("Total developed countries:", n_dev, "\n")

# Since the paper mentions "14 target samples" and evaluates on all 24,
# we do: randomly split 24 countries into 14 train + 10 test, repeat many times
# and average RMSPR over test predictions

n_target <- 14
n_replications <- 50  # Multiple random splits for stable estimate

set.seed(42)

rmspr_list <- c()
time_list <- c()

for (rep in 1:n_replications) {
  # Random split
  train_idx <- sample(n_dev, n_target)
  test_idx <- setdiff(1:n_dev, train_idx)

  X_train <- X_t_all[train_idx, , drop = FALSE]
  Y_train <- Y_t_all[train_idx, , drop = FALSE]
  X_test <- X_t_all[test_idx, , drop = FALSE]
  Y_test <- Y_t_all[test_idx, , drop = FALSE]

  n_test <- nrow(X_test)
  d_watl <- numeric(n_test)

  start_time <- proc.time()["elapsed"]

  for (i in 1:n_test) {
    X_value <- as.numeric(X_test[i, ])
    true_qf <- Y_test[i, ]

    # Step 1: Weighted auxiliary estimator (target + source combined)
    f1_hat <- compute_f1_hat(source_data, X_value, M, X_train, Y_train)

    # Compute global Frechet weights for training data at test point
    X_train_mat <- to_matrix(X_train)
    xbar <- colMeans(X_train_mat)
    if (ncol(X_train_mat) == 1) {
      Sigma <- matrix(var(X_train_mat) * (nrow(X_train_mat)-1)/nrow(X_train_mat), 1, 1)
    } else {
      Sigma <- cov(X_train_mat) * (nrow(X_train_mat)-1)/nrow(X_train_mat)
    }
    invSigma <- solve(Sigma)

    s_vec_t <- sapply(1:nrow(X_train_mat), function(j) {
      as.numeric(1 + (X_train_mat[j,] - xbar) %*% invSigma %*% (X_value - xbar))
    })

    # Step 2: Bias correction with lambda = 0.25 (as in RealData.R)
    f_hat <- compute_f_L2(Y_t = Y_train,
                          s_vec = s_vec_t,
                          f1_hat = f1_hat,
                          lambda = 0.25,
                          M = M,
                          max_iter = 200,
                          step_size = 0.1,
                          tol = 1e-3)

    d_watl[i] <- compute_L2_distance(true_qf, f_hat)
  }

  end_time <- proc.time()["elapsed"]
  elapsed_ms <- (end_time - start_time) * 1000

  rmspr_rep <- mean(d_watl)
  time_rep <- elapsed_ms / n_test

  rmspr_list <- c(rmspr_list, rmspr_rep)
  time_list <- c(time_list, time_rep)

  if (rep %% 10 == 0) {
    cat(sprintf("Rep %d/%d: RMSPR=%.4f, Time/pred=%.3fms\n",
                rep, n_replications, rmspr_rep, time_rep))
  }
}

cat("\n=== FINAL RESULTS (n_target=14) ===\n")
cat(sprintf("RMSPR: %.4f (SE=%.4f)\n", mean(rmspr_list), sd(rmspr_list)/sqrt(n_replications)))
cat(sprintf("Training Time: %.3f ms (SE=%.3f)\n", mean(time_list), sd(time_list)/sqrt(n_replications)))

cat("\n=== REPRODUCTION RESULTS ===\n")
cat(sprintf("Metric: RMSPR, Dataset: Human Mortality Data (Age-at-Death / 162 Countries 2015 / 14 Target Samples)\n"))
cat(sprintf("Paper reported value: 0.028, CI: [0.02744, 0.02856]\n"))
cat(sprintf("Reproduced value: %.4f\n", mean(rmspr_list)))
within_ci_rmspr <- mean(rmspr_list) >= 0.02744 && mean(rmspr_list) <= 0.02856
cat(sprintf("Within CI: %s\n", ifelse(within_ci_rmspr, "Yes", "No")))
cat("---\n")
cat(sprintf("Metric: Training Time (ms), Dataset: Human Mortality Data\n"))
cat(sprintf("Paper reported value: 0.598, CI: [0.58604, 0.60996]\n"))
cat(sprintf("Reproduced value: %.3f\n", mean(time_list)))
within_ci_time <- mean(time_list) >= 0.58604 && mean(time_list) <= 0.60996
cat(sprintf("Within CI: %s\n", ifelse(within_ci_time, "Yes", "No")))
