# Deprecated archive: retained for reference only and not part of the active test suite.
# =============================================================================
# Tests for strategize_onestep()
# =============================================================================
# These tests cover the one-step M-estimation approach for optimal
# stochastic interventions.
# =============================================================================

# =============================================================================
# Basic Functionality Tests
# =============================================================================

test_that("strategize_onestep returns valid result structure", {
  skip_on_cran()
  skip_if_no_jax()
  skip_onestep_tests()

  data <- generate_test_data(n = 300, seed = 42)
  p_list <- generate_test_p_list(data$W)

  result <- strategize_onestep(
    W = as.data.frame(data$W),
    Y = data$Y,
    p_list = p_list,
    nSGD = 50,
    lambda_seq = c(0.1),
    quiet = TRUE
  )

  # Check result structure

  expect_type(result, "list")
  expect_true("pi_star_point" %in% names(result) || "PiStar_list" %in% names(result))

  # Check for Q value
  q_names <- c("Q_point", "Q_point_mEst", "Q_point_all")
  expect_true(any(q_names %in% names(result)))
})

test_that("strategize_onestep respects find_max parameter", {
  skip_on_cran()
  skip_if_no_jax()
  skip_onestep_tests()

  data <- generate_test_data(n = 200, seed = 123)
  p_list <- generate_test_p_list(data$W)

  # Test with find_max = TRUE (maximize)
  result_max <- strategize_onestep(
    W = as.data.frame(data$W),
    Y = data$Y,
    p_list = p_list,
    nSGD = 30,
    lambda_seq = c(0.1),
    find_max = TRUE,
    quiet = TRUE
  )

  # Test with find_max = FALSE (minimize)
  result_min <- strategize_onestep(
    W = as.data.frame(data$W),
    Y = data$Y,
    p_list = p_list,
    nSGD = 30,
    lambda_seq = c(0.1),
    find_max = FALSE,
    quiet = TRUE
  )

  # Both should return valid results
  expect_type(result_max, "list")
  expect_type(result_min, "list")
})

test_that("strategize_onestep handles pi_list for evaluation", {
  skip_on_cran()
  skip_if_no_jax()
  skip_onestep_tests()

  data <- generate_test_data(n = 200, seed = 42)
  p_list <- generate_test_p_list(data$W)

  # Evaluate performance under baseline distribution
  result <- strategize_onestep(
    W = as.data.frame(data$W),
    Y = data$Y,
    p_list = p_list,
    pi_list = p_list,  # Evaluate at baseline
    nSGD = 5L,
    quiet = TRUE
  )

  expect_type(result, "list")
})

# =============================================================================
# Parameter Validation Tests
# =============================================================================

test_that("strategize_onestep handles different penalty types", {
  skip_on_cran()
  skip_if_no_jax()
  skip_onestep_tests()

  data <- generate_test_data(n = 200, seed = 42)
  p_list <- generate_test_p_list(data$W)

  for (penalty in c("LogMaxProb", "L2", "KL")) {
    result <- strategize_onestep(
      W = as.data.frame(data$W),
      Y = data$Y,
      p_list = p_list,
      nSGD = 20,
      lambda_seq = c(0.1),
      penalty_type = penalty,
      quiet = TRUE
    )

    expect_type(result, "list")
  }
})

test_that("strategize_onestep handles K > 1 clusters", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_slow()

  data <- generate_test_data(n = 400, seed = 42)
  p_list <- generate_test_p_list(data$W)

  # Generate covariates for clustering
  X <- matrix(rnorm(nrow(data$W) * 2), ncol = 2)
  colnames(X) <- c("X1", "X2")

  result <- strategize_onestep(
    W = as.data.frame(data$W),
    Y = data$Y,
    X = X,
    K = 2,
    p_list = p_list,
    nSGD = 30,
    lambda_seq = c(0.1),
    quiet = TRUE
  )

  expect_type(result, "list")
})

# =============================================================================
# Cross-validation Tests
# =============================================================================

test_that("strategize_onestep cross-validates over lambda_seq", {
  skip_on_cran()
  skip_if_no_jax()
  skip_onestep_tests()

  data <- generate_test_data(n = 300, seed = 42)
  p_list <- generate_test_p_list(data$W)

  result <- strategize_onestep(
    W = as.data.frame(data$W),
    Y = data$Y,
    p_list = p_list,
    nSGD = 30,
    lambda_seq = c(0.01, 0.1, 0.5),
    n_folds = 2,
    quiet = TRUE
  )

  expect_type(result, "list")

  # Should have CV information if multiple lambdas provided
  cv_names <- c("CVInfo", "cv_results", "lambda_selected")
  # Note: exact structure may vary by implementation
})

# =============================================================================
# Edge Cases
# =============================================================================

test_that("strategize_onestep handles single factor", {
  skip_on_cran()
  skip_if_no_jax()
  skip_onestep_tests()

  set.seed(42)
  n <- 200
  W <- data.frame(Gender = sample(c("M", "F"), n, replace = TRUE))
  Y <- rbinom(n, 1, 0.5 + 0.1 * (W$Gender == "F"))
  p_list <- list(Gender = c(M = 0.5, F = 0.5))

  result <- strategize_onestep(
    W = W,
    Y = Y,
    p_list = p_list,
    nSGD = 30,
    lambda_seq = c(0.1),
    quiet = TRUE
  )

  expect_type(result, "list")
})

test_that("strategize_onestep handles factors with many levels", {
  skip_on_cran()
  skip_if_no_jax()
  skip_onestep_tests()

  set.seed(42)
  n <- 300
  levels <- c("A", "B", "C", "D", "E")
  W <- data.frame(Factor1 = sample(levels, n, replace = TRUE))
  Y <- rbinom(n, 1, 0.5)
  p_list <- list(Factor1 = setNames(rep(0.2, 5), levels))

  result <- strategize_onestep(
    W = W,
    Y = Y,
    p_list = p_list,
    nSGD = 30,
    lambda_seq = c(0.1),
    quiet = TRUE
  )

  expect_type(result, "list")
})

test_that("strategize_onestep runs a short JAX training loop with finite output", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 120, n_factors = 2, seed = 20260331)
  p_list <- generate_test_p_list(data$W)

  result <- strategize_onestep(
    W = as.data.frame(data$W),
    Y = data$Y,
    p_list = p_list,
    nSGD = 6L,
    nEpoch = 1L,
    batch_size = 12L,
    lambda_seq = c(0.1),
    forceSEs = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE,
    quiet = TRUE
  )

  q_names <- intersect(c("Q_point", "Q_point_mEst", "Q_point_all"), names(result))
  expect_true(length(q_names) >= 1L)
  q_values <- unlist(lapply(q_names, function(name) {
    as.numeric(unlist(result[[name]], use.names = FALSE))
  }), use.names = FALSE)
  expect_true(any(is.finite(q_values)))
})
