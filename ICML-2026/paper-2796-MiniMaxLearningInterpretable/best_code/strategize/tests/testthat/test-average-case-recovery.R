extract_average_case_pi_hat <- function(res) {
  pi_obj <- res$pi_star_point
  if (is.list(pi_obj) && length(pi_obj) == 1L && is.list(pi_obj[[1L]])) {
    pi_obj <- pi_obj[[1L]]
  }
  vapply(pi_obj, function(prob_vec) {
    if (!is.null(names(prob_vec)) && "1" %in% names(prob_vec)) {
      return(as.numeric(prob_vec[["1"]]))
    }
    as.numeric(prob_vec[[2L]])
  }, numeric(1))
}

extract_average_case_neural_mu_hat <- function(res, W) {
  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  if (!is.function(model)) {
    stop("Neural average-case fit did not expose a prediction function.", call. = FALSE)
  }

  W_df <- as.data.frame(W, stringsAsFactors = FALSE)
  if (!is.null(names(res$p_list)) && !is.null(colnames(W_df))) {
    W_df <- W_df[, names(res$p_list), drop = FALSE]
  }

  W_num <- as.matrix(vapply(seq_along(res$p_list), function(d_) {
    level_names <- names(res$p_list[[d_]])
    match(as.character(W_df[[d_]]), level_names)
  }, numeric(nrow(W_df))))
  colnames(W_num) <- names(res$p_list)

  pred <- model(X_new = W_num)
  if (is.list(pred) && !is.null(pred$mu)) {
    return(as.numeric(pred$mu))
  }
  as.numeric(pred)
}

get_linear_average_case_fixture_cached <- local({
  fixture <- NULL
  function() {
    if (is.null(fixture)) {
      fixture <<- generate_linear_average_case_fixture()
    }
    fixture
  }
})

get_large_support_average_case_fixture_cached <- local({
  fixture_cache <- list()
  function(k_factors) {
    key <- as.character(as.integer(k_factors))
    if (is.null(fixture_cache[[key]])) {
      fixture_cache[[key]] <<- generate_linear_average_case_fixture(
        k_factors = as.integer(k_factors)
      )
    }
    fixture_cache[[key]]
  }
})

test_that("strategize recovers linear average-case pi* and Q with glm", {
  fixture <- get_linear_average_case_fixture_cached()

  for (seed in 20260326L + 0:3) {
    withr::local_seed(seed)
    res <- strategize(
      Y = fixture$Y,
      W = fixture$W,
      lambda = fixture$lambda,
      outcome_model_type = "glm",
      diff = FALSE,
      adversarial = FALSE,
      compute_se = FALSE,
      penalty_type = "L2",
      use_regularization = FALSE,
      use_optax = FALSE,
      force_gaussian = FALSE,
      nSGD = 1000L,
      nMonte_Qglm = 1000L,
      a_init_sd = 0.001
    )

    pi_hat <- extract_average_case_pi_hat(res)
    Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
      as.numeric(res$Q_point_mEst)
    } else {
      as.numeric(res$Q_point)
    }
    pi_rel_err <- mean(
      abs(pi_hat - fixture$pi_star_true) / pmax(abs(fixture$pi_star_true), 1e-8)
    )
    Q_rel_err <- abs(Q_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8)
    info_msg <- sprintf(
      "seed=%d; pi_rel_err=%.6f; Q_rel_err=%.6f; Q_hat=%.6f; trueQ=%.6f",
      seed,
      pi_rel_err,
      Q_rel_err,
      Q_hat,
      fixture$trueQ
    )

    expect_true(pi_rel_err <= 0.25, info = info_msg)
    expect_true(Q_rel_err <= 0.25, info = info_msg)
  }
})

test_that("strategize recovers linear average-case pi* and Q with neural", {
  skip_on_cran()
  skip_if_no_jax()

  fixture <- get_linear_average_case_fixture_cached()
  expect_gte(length(fixture$Y), 1000L)

  for (seed in 20260326L + 0:3) {
    withr::local_seed(seed)
    res <- strategize(
      Y = fixture$Y,
      W = fixture$W,
      lambda = fixture$lambda,
      outcome_model_type = "neural",
      diff = FALSE,
      adversarial = FALSE,
      compute_se = FALSE,
      penalty_type = "L2",
      use_regularization = FALSE,
      use_optax = FALSE,
      force_gaussian = FALSE,
      nSGD = 1000L,
      nMonte_Qglm = 1000L,
      a_init_sd = 0.001,
      optim_type = "gd",
      neural_mcmc_control = list(
        subsample_method = "batch_vi",
        ModelDims = 64L,
        ModelDepth = 2L,
        qk_norm = FALSE,
        batch_size = 32L,
        optimizer = "adam",
        svi_lr = 0.005,
        early_stopping_n_checks = 20L,
        vi_guide = "auto_diagonal",
        uncertainty_scope = "output",
        eval_enabled = FALSE
      )
    )

    pi_hat <- extract_average_case_pi_hat(res)
    Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
      as.numeric(res$Q_point_mEst)
    } else {
      as.numeric(res$Q_point)
    }
    pi_rel_err <- mean(
      abs(pi_hat - fixture$pi_star_true) / pmax(abs(fixture$pi_star_true), 1e-8)
    )
    Q_rel_err <- abs(Q_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8)
    mu_hat <- extract_average_case_neural_mu_hat(res, fixture$W)
    rmse_y <- sqrt(mean((mu_hat - fixture$Y) ^ 2))
    rmse_null <- sqrt(mean((mean(fixture$Y) - fixture$Y) ^ 2))
    rmse_mu_true <- sqrt(mean((mu_hat - fixture$mu_true) ^ 2))
    cor_mu <- stats::cor(mu_hat, fixture$mu_true)
    info_msg <- sprintf(
      paste0(
        "seed=%d; pi_rel_err=%.6f; Q_rel_err=%.6f; Q_hat=%.6f; trueQ=%.6f; ",
        "rmse_y=%.6f; rmse_null=%.6f; rmse_mu_true=%.6f; cor_mu=%.6f"
      ),
      seed,
      pi_rel_err,
      Q_rel_err,
      Q_hat,
      fixture$trueQ,
      rmse_y,
      rmse_null,
      rmse_mu_true,
      cor_mu
    )

    expect_true(pi_rel_err <= 0.25, info = info_msg)
    expect_true(Q_rel_err <= 0.25, info = info_msg)
    expect_true(rmse_y < 0.5 * rmse_null, info = info_msg)
    expect_true(rmse_mu_true < 0.25 * rmse_null, info = info_msg)
    expect_true(cor_mu > 0.90, info = info_msg)
  }
})

test_that("strategize uses REINFORCE fallback for large-support neural average-case recovery", {
  skip_on_cran()
  skip_if_no_jax()

  thresholds <- c(`12` = 0.20, `20` = 0.10)

  for (k_factors in c(12L, 20L)) {
    fixture <- get_large_support_average_case_fixture_cached(k_factors)
    withr::local_seed(20260326L)
    res <- strategize(
      Y = fixture$Y,
      W = fixture$W,
      lambda = fixture$lambda,
      outcome_model_type = "neural",
      diff = FALSE,
      adversarial = FALSE,
      compute_se = FALSE,
      penalty_type = "L2",
      use_regularization = FALSE,
      use_optax = FALSE,
      force_gaussian = FALSE,
      nSGD = 1000L,
      nMonte_Qglm = 1000L,
      a_init_sd = 0.001,
      optim_type = "gd",
      neural_mcmc_control = list(
        subsample_method = "batch_vi",
        ModelDims = 64L,
        ModelDepth = 2L,
        qk_norm = FALSE,
        batch_size = 32L,
        optimizer = "adam",
        svi_lr = 0.005,
        early_stopping_n_checks = 20L,
        vi_guide = "auto_diagonal",
        uncertainty_scope = "output",
        eval_enabled = FALSE
      )
    )

    pi_hat <- extract_average_case_pi_hat(res)
    Q_hat <- if (!is.null(res$Q_point_mEst) && all(is.finite(res$Q_point_mEst))) {
      as.numeric(res$Q_point_mEst)
    } else {
      as.numeric(res$Q_point)
    }
    pi_rel_err <- mean(
      abs(pi_hat - fixture$pi_star_true) / pmax(abs(fixture$pi_star_true), 1e-8)
    )
    Q_rel_err <- abs(Q_hat - fixture$trueQ) / pmax(abs(fixture$trueQ), 1e-8)
    info_msg <- sprintf(
      paste0(
        "k=%d; pi_rel_err=%.6f; Q_rel_err=%.6f; Q_hat=%.6f; trueQ=%.6f; ",
        "objective_gradient_mode=%s; reinforce_nonfinite_ast_steps=%d"
      ),
      k_factors,
      pi_rel_err,
      Q_rel_err,
      Q_hat,
      fixture$trueQ,
      res$convergence_history$objective_gradient_mode,
      res$convergence_history$reinforce_nonfinite_ast_steps
    )

    expect_identical(res$convergence_history$objective_gradient_mode, "reinforce", info = info_msg)
    expect_true(all(is.finite(pi_hat)), info = info_msg)
    expect_true(is.finite(Q_hat), info = info_msg)
    expect_true(pi_rel_err <= 0.25, info = info_msg)
    expect_true(Q_rel_err <= unname(thresholds[as.character(k_factors)]), info = info_msg)
    expect_identical(as.integer(res$convergence_history$reinforce_nonfinite_ast_steps), 0L, info = info_msg)
  }
})
