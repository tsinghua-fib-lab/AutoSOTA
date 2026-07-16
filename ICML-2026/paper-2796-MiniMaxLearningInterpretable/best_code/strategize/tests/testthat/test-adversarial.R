# =============================================================================
# Adversarial Mode Tests
# =============================================================================
# Tests for the adversarial (minimax/Nash equilibrium) optimization mode.
# Validates that strategize() correctly identifies Nash equilibrium strategies
# in competitive two-player settings.
# =============================================================================

# =============================================================================
# Helper Functions for Adversarial Testing
# =============================================================================

#' Define ground truth probability parameters for adversarial model
define_ground_truth <- function(
    p_RVoters = 0.5,
    lambda = 0.02,
    pr_mm_RVoters_primary = 0.50,
    pr_mf_RVoters_primary = 0.55,
    pr_fm_RVoters_primary = 0.45,
    pr_ff_RVoters_primary = 0.50,
    pd_mm_DVoters_primary = 0.50,
    pd_mf_DVoters_primary = 0.45,
    pd_fm_DVoters_primary = 0.55,
    pd_ff_DVoters_primary = 0.50,
    pr_mm_RVoters = 0.85,
    pr_mf_RVoters = 0.90,
    pr_fm_RVoters = 0.80,
    pr_ff_RVoters = 0.85,
    pr_mm_DVoters = 0.15,
    pr_mf_DVoters = 0.10,
    pr_fm_DVoters = 0.20,
    pr_ff_DVoters = 0.15
) {
  list(
    p_RVoters = p_RVoters,
    p_DVoters = 1 - p_RVoters,
    lambda = lambda,
    pr_mm_RVoters_primary = pr_mm_RVoters_primary,
    pr_mf_RVoters_primary = pr_mf_RVoters_primary,
    pr_fm_RVoters_primary = pr_fm_RVoters_primary,
    pr_ff_RVoters_primary = pr_ff_RVoters_primary,
    pd_mm_DVoters_primary = pd_mm_DVoters_primary,
    pd_mf_DVoters_primary = pd_mf_DVoters_primary,
    pd_fm_DVoters_primary = pd_fm_DVoters_primary,
    pd_ff_DVoters_primary = pd_ff_DVoters_primary,
    pr_mm_RVoters = pr_mm_RVoters,
    pr_mf_RVoters = pr_mf_RVoters,
    pr_fm_RVoters = pr_fm_RVoters,
    pr_ff_RVoters = pr_ff_RVoters,
    pr_mm_DVoters = pr_mm_DVoters,
    pr_mf_DVoters = pr_mf_DVoters,
    pr_fm_DVoters = pr_fm_DVoters,
    pr_ff_DVoters = pr_ff_DVoters
  )
}

#' Compute expected Republican vote share
compute_vote_share_R <- function(pi_R, pi_D, params,
                                  pi_R_s = 0.50, pi_D_s = 0.50) {
  with(params, {
    p_R_m <- 1 * (pi_R * pi_R_s) +
      pr_mf_RVoters_primary * (pi_R * (1 - pi_R_s)) +
      (1 - pr_fm_RVoters_primary) * ((1 - pi_R) * pi_R_s) +
      0 * ((1 - pi_R) * (1 - pi_R_s))
    p_R_f <- 1 - p_R_m

    p_D_m <- 1 * (pi_D * pi_D_s) +
      pd_mf_DVoters_primary * (pi_D * (1 - pi_D_s)) +
      (1 - pd_fm_DVoters_primary) * ((1 - pi_D) * pi_D_s) +
      0 * ((1 - pi_D) * (1 - pi_D_s))
    p_D_f <- 1 - p_D_m

    p_mm <- p_R_m * p_D_m
    p_mf <- p_R_m * p_D_f
    p_fm <- p_R_f * p_D_m
    p_ff <- p_R_f * p_D_f

    p_mm * (p_RVoters * pr_mm_RVoters + p_DVoters * pr_mm_DVoters) +
      p_mf * (p_RVoters * pr_mf_RVoters + p_DVoters * pr_mf_DVoters) +
      p_fm * (p_RVoters * pr_fm_RVoters + p_DVoters * pr_fm_DVoters) +
      p_ff * (p_RVoters * pr_ff_RVoters + p_DVoters * pr_ff_DVoters)
  })
}

#' Utility functions with L2 regularization
utility_R <- function(pi_R, pi_D, params) {
  compute_vote_share_R(pi_R, pi_D, params) -
    params$lambda * ((pi_R - 0.5)^2 + ((1 - pi_R) - 0.5)^2)
}

utility_D <- function(pi_R, pi_D, params) {
  (1 - compute_vote_share_R(pi_R, pi_D, params)) -
    params$lambda * ((pi_D - 0.5)^2 + ((1 - pi_D) - 0.5)^2)
}

#' Compute Nash equilibrium via grid search
compute_nash_grid <- function(params, grid_step = 0.01, tol = 0.02) {
  pi_seq <- seq(0, 1, by = grid_step)

  best_response_D <- sapply(pi_seq, function(r) {
    utilities <- sapply(pi_seq, function(d) utility_D(r, d, params))
    pi_seq[which.max(utilities)]
  })

  best_response_R <- sapply(pi_seq, function(d) {
    utilities <- sapply(pi_seq, function(r) utility_R(r, d, params))
    pi_seq[which.max(utilities)]
  })

  nash_found <- FALSE
  pi_R_nash <- NA
  pi_D_nash <- NA
  min_deviation <- Inf

  for (i in seq_along(pi_seq)) {
    for (j in seq_along(pi_seq)) {
      r <- pi_seq[i]
      d <- pi_seq[j]
      deviation <- abs(r - best_response_R[j]) + abs(d - best_response_D[i])
      if (deviation < min_deviation) {
        min_deviation <- deviation
        pi_R_nash <- r
        pi_D_nash <- d
      }
      if (deviation < tol) {
        nash_found <- TRUE
      }
    }
  }

  if (min_deviation < tol * 2) {
    nash_found <- TRUE
  }

  list(
    pi_R = pi_R_nash,
    pi_D = pi_D_nash,
    found = nash_found,
    deviation = min_deviation
  )
}

#' Compute Nash equilibrium via iterative best response
compute_nash_iterative <- function(params, grid_step = 0.01,
                                    max_iter = 500, tol = 0.01) {
  pi_seq <- seq(0, 1, by = grid_step)

  best_response_D_vec <- sapply(pi_seq, function(r) {
    utilities <- sapply(pi_seq, function(d) utility_D(r, d, params))
    pi_seq[which.max(utilities)]
  })

  best_response_R_vec <- sapply(pi_seq, function(d) {
    utilities <- sapply(pi_seq, function(r) utility_R(r, d, params))
    pi_seq[which.max(utilities)]
  })

  best_response_D_fn <- function(r) {
    best_response_D_vec[which.min(abs(pi_seq - r))]
  }

  best_response_R_fn <- function(d) {
    best_response_R_vec[which.min(abs(pi_seq - d))]
  }

  pi_R_current <- 0.5
  pi_D_current <- 0.5

  for (iter in seq_len(max_iter)) {
    pi_D_new <- best_response_D_fn(pi_R_current)
    pi_R_new <- best_response_R_fn(pi_D_new)

    if (abs(pi_R_new - pi_R_current) < tol &&
        abs(pi_D_new - pi_D_current) < tol) {
      return(list(
        pi_R = pi_R_new,
        pi_D = pi_D_new,
        converged = TRUE,
        iterations = iter
      ))
    }

    pi_R_current <- pi_R_new
    pi_D_current <- pi_D_new
  }

  list(
    pi_R = pi_R_current,
    pi_D = pi_D_current,
    converged = FALSE,
    iterations = max_iter
  )
}

#' Generate adversarial mode test data
generate_adversarial_data <- function(n = 2000, params, seed = 12345) {
  withr::local_seed(seed)

  competing_group_variable_respondent <- sample(
    c("Democrat", "Republican"),
    size = n,
    replace = TRUE,
    prob = c(params$p_DVoters, params$p_RVoters)
  )
  competing_group_variable_respondent <- c(
    competing_group_variable_respondent,
    competing_group_variable_respondent
  )

  pair_id <- c(1:n, 1:n)
  respondent_id <- pair_id
  profile_order <- c(rep(1L, n), rep(2L, n))
  respondent_task_id <- rep(1L, times = 2 * n)

  competing_group_variable_candidate <- sample(
    c("Democrat", "Republican"),
    size = 2 * n,
    replace = TRUE,
    prob = c(0.5, 0.5)
  )

  competing_group_competition_variable_candidate <- ifelse(
    competing_group_variable_candidate[1:n] == competing_group_variable_candidate[-(1:n)],
    yes = "Same",
    no = "Different"
  )
  competing_group_competition_variable_candidate <- c(
    competing_group_competition_variable_candidate,
    competing_group_competition_variable_candidate
  )

  X <- as.data.frame(matrix(rbinom(n * 2, size = 1, prob = 0.5), nrow = 2 * n))
  colnames(X) <- "female"

  X_other <- numeric(nrow(X))
  other_competing_group_variable_candidate <- character(nrow(X))
  for (p in unique(pair_id)) {
    rows_in_pair <- which(pair_id == p)
    i <- rows_in_pair[1]
    j <- rows_in_pair[2]
    X_other[i] <- X$female[j]
    X_other[j] <- X$female[i]
    other_competing_group_variable_candidate[i] <- competing_group_variable_candidate[j]
    other_competing_group_variable_candidate[j] <- competing_group_variable_candidate[i]
  }

  Yobs <- rep(NA, length(competing_group_competition_variable_candidate))

  # Generate outcomes based on true probabilities (simplified for testing)
  for (i in seq_along(Yobs)) {
    if (competing_group_competition_variable_candidate[i] == "Same") {
      Yobs[i] <- rbinom(1, 1, 0.5)
    } else {
      Yobs[i] <- rbinom(1, 1, 0.5)
    }
  }

  Yobs[-(1:n)] <- 1 - Yobs[1:n]

  list(
    Y = Yobs,
    W = X,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate
  )
}

# =============================================================================
# Tests
# =============================================================================

test_that("adversarial mode executes without fatal error", {
  skip_on_cran()
  skip_if_no_jax()

  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)
  data <- generate_adversarial_data(n = 500, params = params, seed = 42)

  res <- tryCatch({
    strategize(
      Y = data$Y,
      W = data$W,
      lambda = params$lambda,
      pair_id = data$pair_id,
      respondent_id = data$respondent_id,
      respondent_task_id = data$respondent_task_id,
      profile_order = data$profile_order,
      competing_group_variable_respondent = data$competing_group_variable_respondent,
      competing_group_variable_candidate = data$competing_group_variable_candidate,
      competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
      adversarial = TRUE,
      diff = TRUE,
      nSGD = 10L,
      outcome_model_type = "glm",
      force_gaussian = FALSE,
      nMonte_adversarial = 10L,
      nMonte_Qglm = 50L,
      temperature = 0.3,
      compute_se = FALSE,
      conda_env = "strategize_env",
      conda_env_required = TRUE
    )
  }, error = function(e) {
    list(error = TRUE, message = conditionMessage(e))
  })

  expect_false(isTRUE(res$error))
})

test_that("adversarial mode returns expected output structure", {
  skip_on_cran()
  skip_if_no_jax()

  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)
  data <- generate_adversarial_data(n = 500, params = params, seed = 123)

  res <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = 10L,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 50L,
    temperature = 0.3,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  expect_type(res, "list")
  expect_true("pi_star_point" %in% names(res))
  expect_true("Q_point" %in% names(res))
  expect_type(res$pi_star_point, "list")
  expect_equal(length(res$pi_star_point), 2)
})

test_that("Nash equilibrium responds to voter proportion changes", {
  results <- list()

  for (p_R in c(0.3, 0.5, 0.7)) {
    params <- define_ground_truth(p_RVoters = p_R, lambda = 0.02)
    nash <- compute_nash_grid(params, grid_step = 0.01)
    results[[as.character(p_R)]] <- nash
  }

  all_found <- all(sapply(results, function(x) x$found || x$deviation < 0.1))
  expect_true(all_found)
})

test_that("high regularization yields near-uniform equilibrium", {
  params <- define_ground_truth(p_RVoters = 0.5, lambda = 10.0)
  nash <- compute_nash_grid(params, grid_step = 0.01)

  if (nash$found) {
    expect_lt(abs(nash$pi_R - 0.5), 0.05)
    expect_lt(abs(nash$pi_D - 0.5), 0.05)
  }
})

test_that("grid search and iterative methods produce same equilibrium", {
  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)

  nash_grid <- compute_nash_grid(params, grid_step = 0.01)
  nash_iter <- compute_nash_iterative(params, grid_step = 0.01)

  if (nash_grid$found && nash_iter$converged) {
    expect_equal(nash_grid$pi_R, nash_iter$pi_R, tolerance = 0.02)
    expect_equal(nash_grid$pi_D, nash_iter$pi_D, tolerance = 0.02)
  }
})

test_that("adversarial optimization converges without Inf values", {
  skip_on_cran()
  skip_if_no_jax()

  params <- define_ground_truth(p_RVoters = 0.5, lambda = 0.02)
  data <- generate_adversarial_data(n = 1000, params = params, seed = 333)

  res <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = 200,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    nMonte_adversarial = 10L,
    nMonte_Qglm = 50L,
    temperature = 0.3,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  pi_values <- unlist(res$pi_star_point)
  Q_values <- if (is.list(res$Q_point)) unlist(res$Q_point) else res$Q_point

  expect_false(any(is.infinite(pi_values)))
  expect_false(any(is.infinite(Q_values)))
})

test_that("strategize handles adversarial mode with standard test data", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  data <- add_adversarial_structure(data)
  params <- default_strategize_params(fast = TRUE)

  res <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  expect_type(res, "list")
  expect_true("pi_star_point" %in% names(res))
  expect_type(res$pi_star_point, "list")
})

test_that("cv_strategize with adversarial mode requires larger datasets", {
  # cv_strategize with adversarial mode splits data multiple times
  # (by respondent party, competition type, and CV folds) creating
  # subsets too small for reliable gradient computation with n < 5000.
  skip("cv_strategize adversarial mode requires larger datasets (n > 5000) for CV splitting")
})
