# =============================================================================
# Core strategize() Function Tests
# =============================================================================
# Tests for the main strategize() function.
# These tests require the conda environment with JAX.
# =============================================================================

test_that("GLM interaction variation filter preserves zero-column shape", {
  interacted_dat <- matrix(0, nrow = 6L, ncol = 3L)
  interaction_info <- data.frame(
    d = c(1L, 1L, 2L),
    l = c(1L, 1L, 1L),
    dl_index = c(1L, 1L, 2L),
    dp = c(2L, 3L, 3L),
    lp = c(1L, 1L, 1L),
    dplp_index = c(2L, 3L, 3L),
    inter_index = seq_len(3L)
  )

  filtered <- strategize:::cs2step_filter_varying_interactions(
    interacted_dat = interacted_dat,
    interaction_info = interaction_info
  )

  expect_true(is.matrix(filtered$interacted_dat))
  expect_equal(nrow(filtered$interacted_dat), nrow(interacted_dat))
  expect_equal(ncol(filtered$interacted_dat), 0L)
  expect_s3_class(filtered$interaction_info, "data.frame")
  expect_equal(nrow(filtered$interaction_info), 0L)
  expect_length(filtered$interaction_info$inter_index, 0L)
})

test_that("strategize returns valid result with GLM outcome model", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res, n_factors = ncol(data$W))
  expect_true(is.finite(res$Q_reference_in_sample))
  expect_true(is.finite(res$Q_gain_in_sample))
})

test_that("strategize GLM handles pairwise designs with zero-variance interactions", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_seed(20260614)
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")

  n_pairs <- 60L
  n_factors <- 3L
  W_left <- matrix("A", nrow = n_pairs, ncol = n_factors)
  W_right <- matrix("A", nrow = n_pairs, ncol = n_factors)
  for (i in seq_len(n_pairs)) {
    active <- ((i - 1L) %% n_factors) + 1L
    alt <- rep("A", n_factors)
    alt[active] <- "B"
    if (i %% 2L == 1L) {
      W_left[i, ] <- alt
    } else {
      W_right[i, ] <- alt
    }
  }
  W <- rbind(W_left, W_right)
  colnames(W) <- paste0("V", seq_len(n_factors))

  p_list <- lapply(seq_len(n_factors), function(j) {
    prop_b <- mean(W[, j] == "B")
    c(A = 1 - prop_b, B = prop_b)
  })
  names(p_list) <- colnames(W)

  pair_id <- rep(seq_len(n_pairs), times = 2L)
  profile_order <- rep(c(1L, 2L), each = n_pairs)
  active <- ((seq_len(n_pairs) - 1L) %% n_factors) + 1L
  effects <- c(0.8, -0.3, 0.4)
  left_minus_right <- ifelse(seq_len(n_pairs) %% 2L == 1L,
                             effects[active],
                             -effects[active])
  Y_left <- rbinom(n_pairs, size = 1L, prob = plogis(left_minus_right))
  Y <- c(Y_left, 1L - Y_left)

  res <- NULL
  expect_no_error({
    res <- strategize(
      Y = Y,
      W = W,
      pair_id = pair_id,
      respondent_id = pair_id,
      respondent_task_id = pair_id,
      profile_order = profile_order,
      p_list = p_list,
      lambda = 0.1,
      K = 1,
      nSGD = 2L,
      outcome_model_type = "glm",
      diff = TRUE,
      use_regularization = TRUE,
      crossfit_q = FALSE,
      compute_hessian = FALSE,
      force_gaussian = FALSE,
      nMonte_Qglm = 2L,
      nMonte_adversarial = 2L,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    )
  })
  expect_valid_strategize_output(res, n_factors = n_factors)
})

test_that("strategize GLM refits intercept-only when all pairwise columns are aliased", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_seed(20260615)
  withr::local_envvar(STRATEGIZE_GLM_SKIP_EVAL = "1")

  n_pairs <- 36L
  n_factors <- 3L
  W_pair <- cbind(
    rep(c("A", "B"), length.out = n_pairs),
    rep(c("A", "A", "B", "B"), length.out = n_pairs),
    rep(c("A", "B", "B", "A"), length.out = n_pairs)
  )
  W <- rbind(W_pair, W_pair)
  colnames(W) <- paste0("V", seq_len(n_factors))

  p_list <- lapply(seq_len(n_factors), function(j) {
    prop_b <- mean(W[, j] == "B")
    c(A = 1 - prop_b, B = prop_b)
  })
  names(p_list) <- colnames(W)

  pair_id <- rep(seq_len(n_pairs), times = 2L)
  profile_order <- rep(c(1L, 2L), each = n_pairs)
  Y_left <- rbinom(n_pairs, size = 1L, prob = 0.5)
  Y <- c(Y_left, 1L - Y_left)

  res <- NULL
  expect_warning({
    res <- strategize(
      Y = Y,
      W = W,
      pair_id = pair_id,
      respondent_id = pair_id,
      respondent_task_id = pair_id,
      profile_order = profile_order,
      p_list = p_list,
      lambda = 0.1,
      K = 1,
      nSGD = 1L,
      outcome_model_type = "glm",
      diff = TRUE,
      use_regularization = TRUE,
      crossfit_q = FALSE,
      compute_hessian = FALSE,
      force_gaussian = FALSE,
      nMonte_Qglm = 2L,
      nMonte_adversarial = 2L,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    )
  }, "GLM coefficients contained NA")
  expect_valid_strategize_output(res, n_factors = n_factors)
})

test_that("strategize handles K > 1 (multi-cluster)", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_no_factorhet_stack()

  data <- generate_test_data(n = 500, seed = 42)
  data <- add_respondent_covariates(data)
  params <- default_strategize_params(fast = TRUE)
  params$K <- 2

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, X = data$X),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
  expect_equal(length(res$pi_star_point), 2)
  expect_true(all(c("k1", "k2") %in% names(res$pi_star_point)))
})

test_that("strategize handles diff = FALSE (non-difference mode)", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})

test_that("strategize handles use_regularization = FALSE", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W, use_regularization = FALSE),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})

test_that("strategize computes standard errors when requested", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$compute_se <- TRUE

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
  expect_true("pi_star_se" %in% names(res))
  expect_true("Q_se" %in% names(res))
})

test_that("strategize validates Y and W dimensions", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  # Use mismatched W dimension
  W_wrong <- data$W[1:100, ]

  expect_error(
    do.call(strategize, c(
      list(Y = data$Y, W = W_wrong),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))
  )
})

test_that("strategize returns valid probability distributions", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  # Check pi_star_point distributions
 for (k in seq_along(res$pi_star_point)) {
    pi_k <- res$pi_star_point[[k]]
    for (d in seq_along(pi_k)) {
      expect_valid_probability(pi_k[[d]])
    }
  }

  # Check p_list distributions
  for (d in seq_along(res$p_list)) {
    expect_valid_probability(res$p_list[[d]])
  }
})

test_that("strategize returns all expected output fields", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expected_fields <- c("pi_star_point", "Q_point", "p_list")
  for (field in expected_fields) {
    expect_true(field %in% names(res), info = paste("Missing field:", field))
  }
  expect_identical(anyDuplicated(names(res)), 0L)

  expect_type(res$pi_star_point, "list")
  expect_type(res$p_list, "list")
  expect_equal(length(res$p_list), ncol(data$W))
})
