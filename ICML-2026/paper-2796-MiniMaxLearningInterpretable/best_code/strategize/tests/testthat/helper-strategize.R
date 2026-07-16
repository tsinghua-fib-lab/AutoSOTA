# =============================================================================
# Test Helpers for the strategize Package
# =============================================================================
# This file is automatically loaded by testthat before running tests.
# It contains common utilities, skip conditions, and data generators.
# =============================================================================

# =============================================================================
# Skip Conditions
# =============================================================================

#' Skip tests if conda environment is not available
skip_if_no_conda <- function(conda_env = "strategize_env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    skip("reticulate package not available")
  }

  conda_list <- tryCatch(
    reticulate::conda_list(),
    error = function(e) NULL
  )

  if (is.null(conda_list) || !conda_env %in% conda_list$name) {
    skip(paste0("Conda environment '", conda_env, "' not available"))
  }
}

#' Skip tests if JAX is not available
skip_if_no_jax <- function(conda_env = "strategize_env") {
  skip_if_no_conda(conda_env)

  jax_available <- tryCatch({
    reticulate::use_condaenv(conda_env, required = TRUE)
    reticulate::py_module_available("jax")
  }, error = function(e) FALSE)

  if (!isTRUE(jax_available)) {
    skip("JAX not available in conda environment")
  }
}

#' Skip slow tests unless explicitly enabled
skip_if_slow <- function() {
  if (!identical(Sys.getenv("STRATEGIZE_RUN_SLOW_TESTS"), "true")) {
    skip("Slow test (set STRATEGIZE_RUN_SLOW_TESTS=true to run)")
  }
}

#' Skip tests if tgp package is not available (required for K > 1 multi-cluster)
skip_if_no_tgp <- function() {
  if (!requireNamespace("tgp", quietly = TRUE)) {
    skip("tgp package not available (required for K > 1 multi-cluster tests)")
  }
}

#' Skip tests if the full FactorHet/mlrMBO/lhs stack is unavailable
skip_if_no_factorhet_stack <- function() {
  required_pkgs <- c("tgp", "FactorHet", "mlrMBO", "ParamHelpers", "lhs")
  missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0L) {
    skip(sprintf(
      "K > 1 backend stack unavailable: missing %s",
      paste(missing_pkgs, collapse = ", ")
    ))
  }

  lhs_ok <- tryCatch({
    lhs::randomLHS(4L, 1L)
    TRUE
  }, error = function(e) FALSE)
  if (!isTRUE(lhs_ok)) {
    skip("K > 1 backend stack unavailable: lhs::randomLHS failed")
  }

  design_ok <- tryCatch({
    par.set <- ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericParam("l", lower = -5, upper = 0)
    )
    ParamHelpers::generateDesign(4L, par.set, lhs::randomLHS)
    TRUE
  }, error = function(e) FALSE)
  if (!isTRUE(design_ok)) {
    skip("K > 1 backend stack unavailable: ParamHelpers/lhs design generation failed")
  }
}

# =============================================================================
# Test Data Generators
# =============================================================================

#' Generate standard conjoint test data
#'
#' @param n Number of observations (profiles)
#' @param n_factors Number of treatment factors
#' @param n_levels Number of levels per factor
#' @param seed Random seed
#' @return List with Y, W, pair_id, respondent_id, respondent_task_id, profile_order
generate_test_data <- function(n = 1000, n_factors = 3, n_levels = 2, seed = 1234321) {
  withr::local_seed(seed)

  # Generate factor matrix
  levels <- LETTERS[seq_len(n_levels)]
  W <- matrix(
    sample(levels, n * n_factors, replace = TRUE),
    nrow = n,
    ncol = n_factors
  )
  colnames(W) <- paste0("V", seq_len(n_factors))

  # Generate IDs for forced-choice pairs
  n_pairs <- n / 2
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))

  # Generate outcome based on factor effects
  effects <- seq(0.4, 0.2, length.out = n_factors)
  latent_utility <- drop((W == "B") %*% effects)

  Y <- as.numeric(ave(
    latent_utility,
    respondent_task_id,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  list(
    Y = Y,
    W = W,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order
  )
}

#' Generate large, non-degenerate pairwise test data for performance checks
#'
#' @param n_pairs Number of pairwise evaluation examples
#' @param n_factors Number of treatment factors
#' @param n_levels Number of levels per factor
#' @param seed Random seed
#' @return List with pairwise data and pair-level diagnostics
generate_pairwise_performance_test_data <- function(n_pairs = 1000L,
                                                    n_factors = 3,
                                                    n_levels = 2,
                                                    seed = 1234321) {
  withr::local_seed(seed)

  levels <- LETTERS[seq_len(n_levels)]
  sample_profiles <- function(n_rows) {
    mat <- matrix(
      sample(levels, n_rows * n_factors, replace = TRUE),
      nrow = n_rows,
      ncol = n_factors
    )
    colnames(mat) <- paste0("V", seq_len(n_factors))
    mat
  }

  effect_sizes <- seq(0.4, 0.2, length.out = n_factors)
  latent_utility <- function(W) {
    drop((W == "B") %*% effect_sizes)
  }

  W_left <- sample_profiles(n_pairs)
  W_right <- sample_profiles(n_pairs)
  bad_pairs <- rep(TRUE, n_pairs)
  max_attempts <- 100L
  for (attempt in seq_len(max_attempts)) {
    if (!any(bad_pairs)) {
      break
    }
    left_bad <- W_left[bad_pairs, , drop = FALSE]
    right_bad <- W_right[bad_pairs, , drop = FALSE]
    identical_bad <- rowSums(left_bad != right_bad) == 0L
    margin_bad <- abs(latent_utility(left_bad) - latent_utility(right_bad))
    bad_now <- identical_bad | (margin_bad <= 0)
    bad_idx <- which(bad_pairs)[bad_now]
    if (!length(bad_idx)) {
      bad_pairs[] <- FALSE
      break
    }
    W_right[bad_idx, ] <- sample_profiles(length(bad_idx))
    bad_pairs[] <- FALSE
    bad_pairs[bad_idx] <- TRUE
  }

  utility_left <- latent_utility(W_left)
  utility_right <- latent_utility(W_right)
  identical_pairs <- rowSums(W_left != W_right) == 0L
  pair_margin <- abs(utility_left - utility_right)
  if (any(identical_pairs) || any(pair_margin <= 0)) {
    stop("Failed to generate non-degenerate pairwise performance data.", call. = FALSE)
  }

  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))
  Y_left <- as.numeric(utility_left > utility_right)
  Y <- c(Y_left, 1 - Y_left)
  W <- rbind(W_left, W_right)

  list(
    Y = Y,
    W = W,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    pair_margin = pair_margin,
    identical_pair = identical_pairs
  )
}

#' Generate test data with respondent-level covariates for K > 1
#'
#' @param base_data Output from generate_test_data
#' @param n_covariates Number of respondent-level covariates
#' @param seed Random seed
#' @return Base data with X matrix added
add_respondent_covariates <- function(base_data, n_covariates = 3, seed = 42) {
  withr::local_seed(seed)

  n_respondents <- length(unique(base_data$respondent_id))
  X_base <- matrix(rnorm(n_respondents * n_covariates), ncol = n_covariates)
  colnames(X_base) <- paste0("X", seq_len(n_covariates))

  # Duplicate for both profiles per respondent
  base_data$X <- X_base[base_data$respondent_id, ]
  base_data
}

#' Generate adversarial mode test data
#'
#' @param base_data Output from generate_test_data
#' @param seed Random seed
#' @return Base data with adversarial variables added
add_adversarial_structure <- function(base_data, seed = 42) {
  withr::local_seed(seed)

  n <- length(base_data$Y)
  n_pairs <- n / 2
  n_unique_respondents <- n_pairs

  # Assign each unique respondent to a party (as close to 50/50 as possible)
  n_party_a <- ceiling(n_unique_respondents / 2)
  n_party_b <- floor(n_unique_respondents / 2)
  respondent_party_base <- c(rep("PartyA", n_party_a), rep("PartyB", n_party_b))
  competing_group_variable_respondent <- respondent_party_base[base_data$respondent_id]

  # Determine competition type for each pair
  competition_type_base <- sample(c(rep("Same", n_party_a), rep("Different", n_party_b)))
  pair_competition_type <- competition_type_base[base_data$pair_id]

  # Assign candidate party based on competition type
  competing_group_variable_candidate <- ifelse(
    pair_competition_type == "Same",
    competing_group_variable_respondent,
    ifelse(base_data$profile_order == 1, "PartyA", "PartyB")
  )

  base_data$competing_group_variable_respondent <- competing_group_variable_respondent
  base_data$competing_group_variable_candidate <- competing_group_variable_candidate
  base_data$competing_group_competition_variable_candidate <- pair_competition_type
  base_data
}

#' Attach pairwise competition-group metadata to a foundation-style experiment
#'
#' @param experiment Experiment list returned by foundation_test_experiment()
#' @param seed Random seed for reproducible party assignments
#' @param single_stage Logical; if TRUE, force all pairs into the same-stage regime
#' @return Experiment list with candidate/respondent competition-group fields added
add_foundation_pairwise_context <- function(experiment,
                                            seed = 42,
                                            single_stage = FALSE) {
  base_data <- list(
    Y = experiment$Y,
    W = as.matrix(experiment$W),
    pair_id = experiment$pair_id,
    respondent_id = experiment$respondent_id,
    respondent_task_id = experiment$respondent_task_id,
    profile_order = experiment$profile_order
  )
  structured <- add_adversarial_structure(base_data, seed = seed)
  experiment$competing_group_variable_candidate <- structured$competing_group_variable_candidate
  experiment$competing_group_variable_respondent <- structured$competing_group_variable_respondent
  if (isTRUE(single_stage)) {
    experiment$competing_group_variable_candidate <-
      experiment$competing_group_variable_respondent
  }
  experiment
}

#' Generate probability list from factor matrix
#'
#' Creates a list of probability distributions for each factor,
#' where each distribution is uniform over the factor's levels.
#' Returns a named vector format compatible with strategize().
#'
#' @param W Factor matrix from generate_test_data
#' @return List of named probability vectors for each factor
generate_test_p_list <- function(W) {
  suppressMessages(create_p_list(W, uniform = TRUE))
}

#' Build pairwise interaction columns for a binary factor matrix
#'
#' @param X Numeric matrix
#' @param KChoose2_combs Two-row matrix of pair indices from combn()
#' @return Numeric interaction matrix
build_pairwise_interaction_matrix <- function(X, KChoose2_combs) {
  X <- as.matrix(X)
  X_inter <- apply(KChoose2_combs, 2, function(idx) {
    X[, idx[1]] * X[, idx[2]]
  })
  if (is.null(dim(X_inter))) {
    X_inter <- matrix(X_inter, ncol = 1L)
  }
  X_inter
}

#' Generate the deterministic linear average-case fixture used in OptimizingSI
#'
#' This reproduces the misspecification-0 average-case branch of the linear
#' simulation design inside package tests without sourcing external code.
#'
#' @param n_obs Number of observations
#' @param k_factors Number of binary factors
#' @param monte_i Monte Carlo index used in the simulation seed path
#' @return List with deterministic fixture data and oracle quantities
generate_linear_average_case_fixture <- function(n_obs = 2000L,
                                                 k_factors = 10L,
                                                 monte_i = 1L) {
  seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (seed_exists) {
    seed_value <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (seed_exists) {
      assign(".Random.seed", seed_value, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  SEED_SCALER <- 10L
  treatProb <- 0.5
  sigma2 <- 0.1^2
  LinearR2_TARGET <- 0.75
  TARGETQ <- 1
  penaltyType <- "L2"
  KChoose2_combs <- combn(seq_len(k_factors), 2)

  getInteractionWts <- function(pi) {
    c(pi, pi[KChoose2_combs[1, ]] * pi[KChoose2_combs[2, ]])
  }

  penalty <- function(pi, lambda) {
    lambda * sum(c((pi - 0.5)^2, ((1 - pi) - 0.5)^2))
  }

  outer_seed <- SEED_SCALER * k_factors^2
  set.seed(outer_seed)
  beta_master <- rnorm(20L + choose(20L, 2), sd = 1)
  my_beta <- my_beta_orig <- beta_master[seq_len(k_factors + choose(k_factors, 2))]

  X_calibration <- matrix(
    rbinom(k_factors * 10000L, size = 1, prob = treatProb),
    nrow = 10000L
  )
  X_inter_calibration <- build_pairwise_interaction_matrix(
    X_calibration,
    KChoose2_combs
  )
  NormalizedInteractionWts <- my_beta_orig[-seq_len(k_factors)] /
    sqrt(sum(my_beta_orig[-seq_len(k_factors)]^2))
  interactionWts_vec <- 10^seq(-5, 5, length.out = 100)
  proposalR2_vec <- vapply(interactionWts_vec, function(scale_value) {
    tmp_beta <- my_beta_orig
    tmp_beta[-seq_len(k_factors)] <- NormalizedInteractionWts * scale_value
    Y_signal <- cbind(X_calibration, X_inter_calibration) %*% tmp_beta
    summary(lm(Y_signal ~ X_calibration))$adj.r.squared
  }, numeric(1))
  my_beta_orig[-seq_len(k_factors)] <- NormalizedInteractionWts *
    interactionWts_vec[which.min(abs(proposalR2_vec - LinearR2_TARGET))[1]]
  my_beta <- my_beta_orig

  getQ <- function(pi) {
    sum(my_beta * getInteractionWts(pi))
  }

  LambdaProposal_seq <- 10^seq(-3, 2, length.out = 500)
  SolAtLambdaProposal <- matrix(
    NA_real_,
    nrow = length(LambdaProposal_seq),
    ncol = k_factors
  )

  for (lambda_index in seq_along(LambdaProposal_seq)) {
    lambda_value <- LambdaProposal_seq[lambda_index]
    my_beta_main <- my_beta[seq_len(k_factors)]
    my_beta_inter <- my_beta[-seq_len(k_factors)]
    COEF_MAT <- sapply(seq_len(k_factors), function(k_index) {
      zero_vec <- rep(0, k_factors)
      zero_vec[k_index] <- -4 * lambda_value
      interindices_ref <- which(
        KChoose2_combs[1, ] == k_index | KChoose2_combs[2, ] == k_index
      )
      interindices <- vapply(interindices_ref, function(interaction_index) {
        KChoose2_combs[, interaction_index][KChoose2_combs[, interaction_index] != k_index]
      }, integer(1))
      zero_vec[interindices] <- my_beta_inter[interindices_ref]
      zero_vec
    })
    B_VEC <- -4 * lambda_value * rep(0.5, k_factors) - my_beta_main
    SolAtLambdaProposal[lambda_index, ] <- as.numeric(
      solve(COEF_MAT, as.matrix(B_VEC))
    )
  }

  whichEligible <- which(apply(SolAtLambdaProposal, 1, function(pi_star) {
    all(pi_star > 0.15) && all(pi_star < 0.85) && sd(pi_star) > 0.1
  }))
  impliedQ_vec <- apply(SolAtLambdaProposal[whichEligible, , drop = FALSE], 1, getQ)
  whichSelected <- whichEligible[which.min(abs(impliedQ_vec - TARGETQ))]
  lambda <- LambdaProposal_seq[whichSelected]
  pi_star_true <- as.numeric(SolAtLambdaProposal[whichSelected, ])
  trueQ <- as.numeric(getQ(pi_star_true))

  inner_seed <- SEED_SCALER * (k_factors * n_obs * monte_i)
  set.seed(inner_seed)
  X_obs <- matrix(
    rbinom(k_factors * n_obs, size = 1, prob = treatProb),
    nrow = n_obs
  )
  X_inter_obs <- build_pairwise_interaction_matrix(X_obs, KChoose2_combs)
  mu_true <- as.numeric(cbind(X_obs, X_inter_obs) %*% my_beta)
  observation_seed <- inner_seed + 303L
  set.seed(observation_seed)
  Yobs <- mu_true + rnorm(n_obs, sd = sqrt(sigma2))

  W <- apply(X_obs, 2, as.character)
  colnames(W) <- as.character(seq_len(k_factors))

  list(
    Y = as.numeric(Yobs),
    W = W,
    lambda = as.numeric(lambda),
    pi_star_true = pi_star_true,
    trueQ = trueQ,
    mu_true = as.numeric(mu_true),
    my_beta = as.numeric(my_beta),
    KChoose2_combs = KChoose2_combs,
    outer_seed = as.integer(outer_seed),
    inner_seed = as.integer(inner_seed),
    getInteractionWts = getInteractionWts,
    penalty = penalty,
    penaltyType = penaltyType
  )
}

# =============================================================================
# Common Strategize Call Parameters
# =============================================================================

#' Get default strategize parameters for testing
#'
#' @param fast Logical; if TRUE, use minimal iterations for faster tests
#' @return Named list of default parameters
default_strategize_params <- function(fast = TRUE) {
  params <- list(
    lambda = 0.1,
    K = 1,
    nSGD = if (fast) 5L else 100L,
    outcome_model_type = "glm",
    force_gaussian = TRUE,
    nMonte_adversarial = if (fast) 5L else 24L,
    nMonte_Qglm = if (fast) 5L else 100L,
    diff = TRUE,
    compute_se = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE
  )

  if (fast) {
    params$neural_mcmc_control <- list(
      n_samples_warmup = 10L,
      n_samples_mcmc = 10L
    )
  }

  params
}

# =============================================================================
# Assertion Helpers
# =============================================================================

#' Check that a probability distribution is valid
#'
#' @param probs Numeric vector of probabilities
#' @param tolerance Tolerance for sum-to-one check
expect_valid_probability <- function(probs, tolerance = 1e-6) {
  testthat::expect_true(all(probs >= 0), info = "All probabilities must be non-negative")
  testthat::expect_equal(sum(probs), 1, tolerance = tolerance,
                         info = "Probabilities must sum to 1")
}

#' Check that strategize output has expected structure
#'
#' @param res Result from strategize()
#' @param n_factors Expected number of factors
expect_valid_strategize_output <- function(res, n_factors = NULL) {
  testthat::expect_type(res, "list")
  testthat::expect_true("pi_star_point" %in% names(res))
  testthat::expect_true("Q_point" %in% names(res))
  testthat::expect_true("p_list" %in% names(res))
  testthat::expect_type(res$pi_star_point, "list")
  testthat::expect_type(res$p_list, "list")

  testthat::expect_true("Q_point_in_sample" %in% names(res))
  testthat::expect_true("Q_reference_in_sample" %in% names(res))
  testthat::expect_true("Q_gain_in_sample" %in% names(res))
  if (all(is.finite(res$Q_reference_in_sample))) {
    testthat::expect_equal(
      as.numeric(res$Q_gain_in_sample),
      as.numeric(res$Q_point_in_sample) - as.numeric(res$Q_reference_in_sample),
      tolerance = 1e-8
    )
  }

  if (!is.null(n_factors)) {
    testthat::expect_equal(length(res$p_list), n_factors)
  }
}

format_test_metric_value <- function(x, digits = 4) {
  if (is.null(x) || length(x) != 1L || is.na(x) || !is.finite(x)) {
    return("NA")
  }
  sprintf(paste0("%.", digits, "f"), x)
}

format_test_bool_value <- function(x) {
  if (is.null(x) || length(x) != 1L || is.na(x)) {
    return("NA")
  }
  if (isTRUE(x)) "TRUE" else "FALSE"
}

format_test_named_numeric <- function(x, digits = 3) {
  if (is.null(x) || !length(x)) {
    return("NA")
  }
  vals <- vapply(x, format_test_metric_value, character(1), digits = digits)
  paste(sprintf("%s=%s", names(vals), vals), collapse = ", ")
}

format_test_named_integer <- function(x) {
  if (is.null(x) || !length(x)) {
    return("NA")
  }
  vals <- ifelse(is.na(x), "NA", as.character(as.integer(x)))
  paste(sprintf("%s=%s", names(x), vals), collapse = ", ")
}

format_metric_line <- function(label, metrics, indent = "") {
  if (is.null(metrics) || !is.list(metrics)) {
    return(sprintf("%s%s: <missing>", indent, label))
  }

  parts <- c()
  if (!is.null(metrics$n_eval)) parts <- c(parts, sprintf("n=%s", metrics$n_eval))
  if (!is.null(metrics$auc)) parts <- c(parts, sprintf("auc=%s", format_test_metric_value(metrics$auc, 4)))
  if (!is.null(metrics$log_loss)) parts <- c(parts, sprintf("log_loss=%s", format_test_metric_value(metrics$log_loss, 4)))
  if (!is.null(metrics$accuracy)) parts <- c(parts, sprintf("accuracy=%s", format_test_metric_value(metrics$accuracy, 3)))
  if (!is.null(metrics$brier)) parts <- c(parts, sprintf("brier=%s", format_test_metric_value(metrics$brier, 4)))
  if (!is.null(metrics$rmse)) parts <- c(parts, sprintf("rmse=%s", format_test_metric_value(metrics$rmse, 4)))
  if (!is.null(metrics$mae)) parts <- c(parts, sprintf("mae=%s", format_test_metric_value(metrics$mae, 4)))
  if (!is.null(metrics$nll)) parts <- c(parts, sprintf("nll=%s", format_test_metric_value(metrics$nll, 4)))
  if (!is.null(metrics$truth_pred_correlation)) {
    parts <- c(parts, sprintf("cor=%s", format_test_metric_value(metrics$truth_pred_correlation, 4)))
  }
  if (!is.null(metrics$y_mean)) parts <- c(parts, sprintf("y_mean=%s", format_test_metric_value(metrics$y_mean, 3)))
  if (!is.null(metrics$pred_mean)) parts <- c(parts, sprintf("pred_mean=%s", format_test_metric_value(metrics$pred_mean, 3)))
  if (!is.null(metrics$pred_sd)) parts <- c(parts, sprintf("pred_sd=%s", format_test_metric_value(metrics$pred_sd, 3)))
  if (!is.null(metrics$protocol)) parts <- c(parts, sprintf("protocol=%s", metrics$protocol))
  if (!is.null(metrics$n_train)) parts <- c(parts, sprintf("n_train=%s", metrics$n_train))
  if (!is.null(metrics$n_test)) parts <- c(parts, sprintf("n_test=%s", metrics$n_test))
  if (!is.null(metrics$n_main)) parts <- c(parts, sprintf("n_main=%s", metrics$n_main))
  if (!is.null(metrics$n_inter)) parts <- c(parts, sprintf("n_inter=%s", metrics$n_inter))
  if (!is.null(metrics$n_stage)) parts <- c(parts, sprintf("n_stage=%s", metrics$n_stage))
  if (!is.null(metrics$screening_used)) {
    parts <- c(parts, sprintf("screening_used=%s", format_test_bool_value(metrics$screening_used)))
  }

  lines <- sprintf("%s%s: %s", indent, label, paste(parts, collapse = ", "))
  if (!is.null(metrics$pred_quantiles)) {
    lines <- c(lines, sprintf("%s  pred_q: %s", indent, format_test_named_numeric(metrics$pred_quantiles, 3)))
  }
  if (!is.null(metrics$confusion_0_5)) {
    lines <- c(lines, sprintf("%s  confusion@0.5: %s", indent, format_test_named_integer(metrics$confusion_0_5)))
  }
  lines
}

format_oos_failure_details <- function(metrics,
                                       null_metrics = NULL,
                                       stage_diagnostics = NULL,
                                       label = "OOS diagnostics") {
  lines <- c(label)
  lines <- c(lines, format_metric_line("oos", metrics))

  if (!is.null(null_metrics)) {
    lines <- c(lines, format_metric_line("null", null_metrics))
    deltas <- c(
      sprintf(
        "delta log_loss (oos-null)=%s",
        format_test_metric_value(metrics$log_loss - null_metrics$log_loss, 4)
      ),
      sprintf(
        "delta brier (oos-null)=%s",
        format_test_metric_value(metrics$brier - null_metrics$brier, 4)
      ),
      sprintf(
        "delta accuracy (oos-null)=%s",
        format_test_metric_value(metrics$accuracy - null_metrics$accuracy, 3)
      )
    )
    lines <- c(lines, paste(deltas, collapse = ", "))
  }

  if (!is.null(metrics$in_sample_metrics)) {
    lines <- c(lines, format_metric_line("in_sample_full_fit", metrics$in_sample_metrics))
    if (!is.null(metrics$in_sample_metrics$by_stage)) {
      for (nm in names(metrics$in_sample_metrics$by_stage)) {
        lines <- c(lines, format_metric_line(
          sprintf("in_sample stage %s", nm),
          metrics$in_sample_metrics$by_stage[[nm]],
          indent = "  "
        ))
      }
    }
  }

  if (!is.null(metrics$by_fold) && length(metrics$by_fold) > 0L) {
    for (i in seq_along(metrics$by_fold)) {
      lines <- c(lines, format_metric_line(
        sprintf("fold %d", i),
        metrics$by_fold[[i]],
        indent = "  "
      ))
    }
  }

  if (!is.null(metrics$by_stage) && length(metrics$by_stage) > 0L) {
    for (nm in names(metrics$by_stage)) {
      lines <- c(lines, format_metric_line(
        sprintf("oos stage %s", nm),
        metrics$by_stage[[nm]],
        indent = "  "
      ))
    }
  }

  if (!is.null(stage_diagnostics)) {
    diag_parts <- c()
    if (!is.null(stage_diagnostics$pct_primary)) {
      diag_parts <- c(diag_parts, sprintf("pct_primary=%s", format_test_metric_value(stage_diagnostics$pct_primary, 3)))
    }
    if (!is.null(stage_diagnostics$min_cell_n)) {
      diag_parts <- c(diag_parts, sprintf("min_cell_n=%s", stage_diagnostics$min_cell_n))
    }
    if (!is.null(stage_diagnostics$single_stage_only)) {
      diag_parts <- c(diag_parts, sprintf("single_stage_only=%s", format_test_bool_value(stage_diagnostics$single_stage_only)))
    }
    if (!is.null(stage_diagnostics$sparse_cells)) {
      diag_parts <- c(diag_parts, sprintf("sparse_cells=%s", format_test_bool_value(stage_diagnostics$sparse_cells)))
    }
    if (length(diag_parts) > 0L) {
      lines <- c(lines, sprintf("stage_diagnostics: %s", paste(diag_parts, collapse = ", ")))
    }
  }

  paste(lines, collapse = "\n")
}
