# =============================================================================
# Edge Case Tests
# =============================================================================
# Tests for boundary conditions and special cases.
# =============================================================================

test_that("strategize handles single factor", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_seed(42)

  n <- 200
  W <- matrix(sample(c("A", "B"), n, replace = TRUE), ncol = 1)
  colnames(W) <- "V1"

  pair_id <- c(seq_len(n/2), seq_len(n/2))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n/2), rep(2L, n/2))

  Y <- as.numeric(ave(
    as.numeric(W == "B") * 0.5,
    respondent_task_id,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  params <- default_strategize_params(fast = TRUE)

  res <- strategize(
    Y = Y,
    W = W,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    lambda = params$lambda,
    K = params$K,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    diff = params$diff,
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  expect_valid_strategize_output(res, n_factors = 1)
})

test_that("strategize handles multiple factor levels (>2)", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_seed(42)

  n <- 300
  W <- cbind(
    matrix(sample(c("A", "B", "C", "D"), n, replace = TRUE), ncol = 1),
    matrix(sample(c("X", "Y", "Z"), n, replace = TRUE), ncol = 1)
  )
  colnames(W) <- c("Factor1", "Factor2")

  pair_id <- c(seq_len(n/2), seq_len(n/2))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n/2), rep(2L, n/2))

  Y <- as.numeric(ave(
    as.numeric(W[, 1] == "D") * 0.4 + as.numeric(W[, 2] == "Z") * 0.3,
    respondent_task_id,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  params <- default_strategize_params(fast = TRUE)

  res <- strategize(
    Y = Y,
    W = W,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    lambda = params$lambda,
    K = params$K,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    diff = params$diff,
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  expect_valid_strategize_output(res, n_factors = 2)
})

test_that("strategize handles moderate sample size", {
  skip_on_cran()
  skip_if_no_jax()

  # Smaller dataset to test boundary conditions
  data <- generate_test_data(n = 200, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})
