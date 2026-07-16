# =============================================================================
# Implicit standard error approximation tests
# =============================================================================

local_force_no_interactions <- function(value = TRUE) {
  old_exists <- exists(
    "force_no_interactions",
    envir = .GlobalEnv,
    inherits = FALSE
  )
  if (old_exists) {
    old_value <- get(
      "force_no_interactions",
      envir = .GlobalEnv,
      inherits = FALSE
    )
  }

  assign("force_no_interactions", value, envir = .GlobalEnv)

  withr::defer({
    if (old_exists) {
      assign("force_no_interactions", old_value, envir = .GlobalEnv)
    } else if (exists("force_no_interactions", envir = .GlobalEnv, inherits = FALSE)) {
      rm("force_no_interactions", envir = .GlobalEnv)
    }
  })
}

run_se_method_comparison <- function(adversarial = TRUE) {
  skip_on_cran()
  skip_if_no_jax()

  local_force_no_interactions(TRUE)

  data <- generate_test_data(n = 120, n_factors = 2, seed = 123)
  data <- add_adversarial_structure(data, seed = 456)

  params <- default_strategize_params(fast = TRUE)
  params$lambda <- 0.05
  params$nSGD <- 15L
  params$nMonte_adversarial <- 3L
  params$nMonte_Qglm <- 5L
  params$compute_se <- TRUE
  params$primary_pushforward <- "linearized"
  params$a_init_sd <- 0.0
  params$compute_hessian <- FALSE
  params$optimism <- "none"

  args <- c(
    list(
      Y = data$Y,
      W = data$W,
      pair_id = data$pair_id,
      respondent_id = data$respondent_id,
      respondent_task_id = data$respondent_task_id,
      profile_order = data$profile_order,
      adversarial = adversarial,
      competing_group_variable_respondent = data$competing_group_variable_respondent,
      competing_group_variable_candidate = data$competing_group_variable_candidate,
      competing_group_competition_variable_candidate =
        data$competing_group_competition_variable_candidate
    ),
    params
  )

  set.seed(999)
  res_full <- do.call(strategize, c(args, list(se_method = "full")))
  set.seed(999)
  res_impl <- do.call(strategize, c(args, list(se_method = "implicit")))

  list(full = res_full, implicit = res_impl)
}

expect_se_methods_close <- function(results,
                                    mean_tolerance,
                                    median_tolerance,
                                    max_tolerance,
                                    q_tolerance) {
  full_se <- results$full$pi_star_se_vec
  impl_se <- results$implicit$pi_star_se_vec
  keep <- is.finite(full_se) & is.finite(impl_se)
  diffs <- abs(full_se[keep] - impl_se[keep])

  expect_true(any(keep))
  expect_lt(mean(diffs), mean_tolerance)
  expect_lt(stats::median(diffs), median_tolerance)
  expect_lt(max(diffs), max_tolerance)
  expect_true(is.finite(results$full$Q_se))
  expect_true(is.finite(results$implicit$Q_se))
  expect_lt(abs(results$full$Q_se - results$implicit$Q_se), q_tolerance)
}

test_that("REINFORCE diagnostics are tracer-safe under jacrev", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  x0 <- strategize:::strenv$jnp$array(c(1, 2), dtype = strategize:::strenv$dtj)
  jacobian <- tryCatch(
    strategize:::strenv$jax$jacrev(function(x) {
      diag <- strategize:::build_reinforce_diag(
        reward_mean = x[[1L]],
        reward_var = x[[1L]] * x[[1L]],
        baseline_prev = x[[1L]],
        grad_total = x
      )
      strategize:::strenv$jnp$add(diag$reward_mean, diag$reward_var)
    })(x0),
    error = identity
  )

  expect_false(
    inherits(jacobian, "error"),
    info = if (inherits(jacobian, "error")) conditionMessage(jacobian) else NULL
  )
  if (!inherits(jacobian, "error")) {
    expect_true(all(is.finite(as.numeric(strategize:::strenv$np$array(jacobian)))))
  }
})

test_that("implicit SEs match full-trace SEs in adversarial mode", {
  results <- run_se_method_comparison(adversarial = TRUE)
  expect_se_methods_close(
    results,
    mean_tolerance = 0.1,
    median_tolerance = 0.05,
    max_tolerance = 0.25,
    q_tolerance = 0.05
  )
})

test_that("implicit SEs match full-trace SEs in non-adversarial mode", {
  results <- run_se_method_comparison(adversarial = FALSE)
  expect_se_methods_close(
    results,
    mean_tolerance = 0.01,
    median_tolerance = 0.005,
    max_tolerance = 0.01,
    q_tolerance = 0.01
  )
})

test_that("implicit SEs complete for large-support neural average-case fits", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  fixture <- generate_linear_average_case_fixture(
    n_obs = 120L,
    k_factors = 12L,
    monte_i = 1L
  )
  params <- default_strategize_params(fast = TRUE)
  params$lambda <- fixture$lambda
  params$nSGD <- 15L
  params$nMonte_Qglm <- 5L
  params$outcome_model_type <- "neural"
  params$diff <- FALSE
  params$adversarial <- FALSE
  params$compute_se <- TRUE
  params$se_method <- "implicit"
  params$penalty_type <- "L2"
  params$use_regularization <- FALSE
  params$use_optax <- FALSE
  params$force_gaussian <- FALSE
  params$a_init_sd <- 0.0
  params$primary_pushforward <- "linearized"
  params$compute_hessian <- FALSE
  params$optimism <- "none"
  params$neural_mcmc_control <- modifyList(
    params$neural_mcmc_control,
    list(
      subsample_method = "batch_vi",
      batch_size = 16L,
      ModelDims = 8L,
      ModelDepth = 1L,
      vi_guide = "auto_diagonal",
      uncertainty_scope = "output",
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  withr::local_seed(20260326L)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = fixture$Y, W = fixture$W),
    params
  )))

  info_msg <- sprintf(
    paste0(
      "objective_gradient_mode=%s; Q_se=%s; finite_pi_se=%d; ",
      "reinforce_nonfinite_ast_steps=%d"
    ),
    res$convergence_history$objective_gradient_mode,
    format(res$Q_se, digits = 6),
    sum(is.finite(res$pi_star_se_vec)),
    as.integer(res$convergence_history$reinforce_nonfinite_ast_steps)
  )

  expect_identical(res$convergence_history$objective_gradient_mode, "reinforce", info = info_msg)
  expect_true(is.finite(res$Q_se), info = info_msg)
  expect_true(any(is.finite(res$pi_star_se_vec)), info = info_msg)
  expect_identical(as.integer(res$convergence_history$reinforce_nonfinite_ast_steps), 0L, info = info_msg)
  expect_null(strategize:::strenv$reinforce_baseline_ast, info = info_msg)
  expect_null(strategize:::strenv$reinforce_baseline_dag, info = info_msg)

  small_data <- local({
    withr::local_seed(20260329L)
    levels <- LETTERS[1:2]
    W <- matrix(sample(levels, 24L * 3L, replace = TRUE), nrow = 24L, ncol = 3L)
    colnames(W) <- paste0("V", seq_len(3L))
    effect_sizes <- seq(0.5, 0.2, length.out = 3L)
    signal <- rowSums((W == "B") * rep(effect_sizes, each = 24L))
    list(
      Y = as.numeric(signal + rnorm(24L, sd = 0.1)),
      W = W
    )
  })

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))
  small_params <- default_strategize_params(fast = TRUE)
  small_params$diff <- FALSE
  small_params$force_gaussian <- TRUE
  small_params$force_reinforce <- TRUE
  small_params$compute_se <- FALSE
  small_params$outcome_model_type <- "neural"
  small_params$nMonte_Qglm <- 8L
  small_params$neural_mcmc_control <- modifyList(
    small_params$neural_mcmc_control,
    list(
      subsample_method = "batch_vi",
      uncertainty_scope = "output",
      vi_guide = "auto_diagonal",
      ModelDims = 8L,
      ModelDepth = 1L,
      batch_size = 16L,
      svi_steps = 20L,
      svi_num_draws = 5L,
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  followup <- suppressWarnings(do.call(strategize, c(
    list(
      Y = small_data$Y,
      W = small_data$W,
      p_list = generate_test_p_list(small_data$W)
    ),
    small_params
  )))

  followup_info <- sprintf(
    paste0(
      "followup objective_gradient_mode=%s; ",
      "reinforce_nonfinite_ast_steps=%d; Q_point=%s"
    ),
    followup$convergence_history$objective_gradient_mode,
    as.integer(followup$convergence_history$reinforce_nonfinite_ast_steps),
    format(as.numeric(followup$Q_point), digits = 6)
  )

  expect_identical(followup$convergence_history$objective_gradient_mode, "reinforce", info = followup_info)
  expect_identical(as.integer(followup$convergence_history$reinforce_nonfinite_ast_steps), 0L, info = followup_info)
  expect_true(all(is.finite(as.numeric(followup$Q_point))), info = followup_info)
  expect_null(strategize:::strenv$reinforce_baseline_ast, info = followup_info)
  expect_null(strategize:::strenv$reinforce_baseline_dag, info = followup_info)
})

test_that("full-trace SEs complete for large-support neural REINFORCE fits", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))

  fixture <- generate_linear_average_case_fixture(
    n_obs = 120L,
    k_factors = 12L,
    monte_i = 1L
  )
  params <- default_strategize_params(fast = TRUE)
  params$lambda <- fixture$lambda
  params$nSGD <- 5L
  params$nMonte_Qglm <- 4L
  params$outcome_model_type <- "neural"
  params$diff <- FALSE
  params$adversarial <- FALSE
  params$compute_se <- TRUE
  params$se_method <- "full"
  params$penalty_type <- "L2"
  params$use_regularization <- FALSE
  params$use_optax <- FALSE
  params$force_gaussian <- FALSE
  params$force_reinforce <- TRUE
  params$a_init_sd <- 0.0
  params$primary_pushforward <- "linearized"
  params$compute_hessian <- FALSE
  params$optimism <- "none"
  params$neural_mcmc_control <- modifyList(
    params$neural_mcmc_control,
    list(
      subsample_method = "batch_vi",
      optimizer = "adamw",
      batch_size = 16L,
      ModelDims = 8L,
      ModelDepth = 1L,
      vi_guide = "auto_diagonal",
      uncertainty_scope = "output",
      svi_steps = 20L,
      svi_num_draws = 5L,
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  withr::local_seed(20260408L)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = fixture$Y, W = fixture$W),
    params
  )))

  info_msg <- sprintf(
    paste0(
      "objective_gradient_mode=%s; force_reinforce=%s; Q_se=%s; ",
      "finite_pi_se=%d; reinforce_nonfinite_ast_steps=%d"
    ),
    res$convergence_history$objective_gradient_mode,
    as.character(res$force_reinforce),
    format(res$Q_se, digits = 6),
    sum(is.finite(res$pi_star_se_vec)),
    as.integer(res$convergence_history$reinforce_nonfinite_ast_steps)
  )

  expect_true(isTRUE(res$force_reinforce), info = info_msg)
  expect_identical(res$convergence_history$objective_gradient_mode, "reinforce", info = info_msg)
  expect_true(is.finite(res$Q_se), info = info_msg)
  expect_true(any(is.finite(res$pi_star_se_vec)), info = info_msg)
  expect_identical(as.integer(res$convergence_history$reinforce_nonfinite_ast_steps), 0L, info = info_msg)
  expect_null(strategize:::strenv$reinforce_baseline_ast, info = info_msg)
  expect_null(strategize:::strenv$reinforce_baseline_dag, info = info_msg)
})
