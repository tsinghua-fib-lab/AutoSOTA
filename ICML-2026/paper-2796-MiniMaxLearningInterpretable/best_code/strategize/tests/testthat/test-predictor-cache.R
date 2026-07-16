test_that("strategic predictors can be saved and loaded (GLM)", {
  dat <- generate_test_data(n = 300, n_factors = 3, n_levels = 2, seed = 20260201)

  fit <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )

  tmp <- tempfile(fileext = ".rds")
  save_strategic_predictor(fit, tmp, overwrite = TRUE)
  fit_loaded <- load_strategic_predictor(tmp)

  preds1 <- predict(fit, newdata = dat$W)
  preds2 <- predict(fit_loaded, newdata = dat$W)

  expect_equal(preds1, preds2, tolerance = 1e-8)
  expect_true(is.list(fit_loaded$fit$fit_metrics))
  expect_true(is.list(fit_loaded$fit$fit_metrics$in_sample_metrics))
})

test_that("strategic_prediction() can reuse cached predictors", {
  dat <- generate_test_data(n = 200, n_factors = 2, n_levels = 2, seed = 20260202)

  tmp <- tempfile(fileext = ".rds")
  fit_cached <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE,
    cache_path = tmp,
    cache_overwrite = TRUE
  )

  fit_reused <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE,
    cache_path = tmp,
    cache_overwrite = FALSE
  )

  preds_cached <- predict(fit_cached, newdata = dat$W)
  preds_reused <- predict(fit_reused, newdata = dat$W)

  expect_equal(preds_cached, preds_reused, tolerance = 1e-8)
})

test_that("strategic_prediction() rejects stale or legacy caches", {
  dat <- generate_test_data(n = 200, n_factors = 2, n_levels = 2, seed = 20260203)
  tmp <- tempfile(fileext = ".rds")

  fit_cached <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE,
    cache_path = tmp,
    cache_overwrite = TRUE
  )
  bundle <- readRDS(tmp)
  bundle$metadata$fingerprint <- NULL
  saveRDS(bundle, tmp)

  expect_error(
    strategic_prediction(
      Y = dat$Y,
      W = dat$W,
      model = "glm",
      mode = "single",
      use_regularization = FALSE,
      cache_path = tmp,
      cache_overwrite = FALSE
    ),
    "missing an artifact fingerprint"
  )

  fit_cached$metadata$fingerprint <- strategize:::cs2step_predictor_request_fingerprint(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )
  save_strategic_predictor(fit_cached, tmp, overwrite = TRUE)
  expect_error(
    strategic_prediction(
      Y = rev(dat$Y),
      W = dat$W,
      model = "glm",
      mode = "single",
      use_regularization = FALSE,
      cache_path = tmp,
      cache_overwrite = FALSE
    ),
    "fingerprint mismatch"
  )
})

test_that("strategic_prediction() derives neural checkpoint paths from cache_path", {
  captured_control <- NULL
  saved_path <- NULL

  testthat::local_mocked_bindings(
    cs2step_eval_outcome_model_neural = function(Y,
                                                 W_idx,
                                                 names_list = NULL,
                                                 factor_levels,
                                                 diff,
                                                 pair_id = NULL,
                                                 profile_order = NULL,
                                                 X = NULL,
                                                 conda_env = NULL,
                                                 conda_env_required = NULL,
                                                 neural_mcmc_control = NULL,
                                                 varcov_cluster_variable = NULL,
                                                 nFolds_glm = NULL,
                                                 ...) {
      captured_control <<- neural_mcmc_control
      list(
        neural_model_info = list(fit_metrics = list(in_sample_metrics = list())),
        fit_metrics = NULL
      )
    },
    save_strategic_predictor = function(fit,
                                        file,
                                        overwrite = FALSE,
                                        compress = TRUE,
                                        include_metrics = TRUE) {
      saved_path <<- file
      invisible(file)
    },
    .package = "strategize"
  )

  cache_path <- tempfile(fileext = ".rds")
  fit <- strategic_prediction(
    Y = c(0, 1, 0, 1),
    W = data.frame(feature = c("a", "b", "a", "b")),
    model = "neural",
    mode = "single",
    cache_path = cache_path,
    conda_env_required = FALSE
  )

  expect_s3_class(fit, "strategic_predictor")
  expect_equal(captured_control$checkpoint_path, paste0(cache_path, ".inprogress"))
  expect_equal(saved_path, cache_path)
})

test_that("strategic_prediction() cache_overwrite preserves explicit neural checkpoints", {
  captured_control <- NULL

  testthat::local_mocked_bindings(
    cs2step_eval_outcome_model_neural = function(Y,
                                                 W_idx,
                                                 names_list = NULL,
                                                 factor_levels,
                                                 diff,
                                                 pair_id = NULL,
                                                 profile_order = NULL,
                                                 X = NULL,
                                                 conda_env = NULL,
                                                 conda_env_required = NULL,
                                                 neural_mcmc_control = NULL,
                                                 varcov_cluster_variable = NULL,
                                                 nFolds_glm = NULL,
                                                 ...) {
      captured_control <<- neural_mcmc_control
      list(
        neural_model_info = list(fit_metrics = list(in_sample_metrics = list())),
        fit_metrics = NULL
      )
    },
    save_strategic_predictor = function(fit,
                                        file,
                                        overwrite = FALSE,
                                        compress = TRUE,
                                        include_metrics = TRUE) {
      invisible(file)
    },
    .package = "strategize"
  )

  cache_path <- tempfile(fileext = ".rds")
  explicit_checkpoint <- tempfile("explicit-checkpoint-")
  dir.create(explicit_checkpoint, recursive = TRUE)

  strategic_prediction(
    Y = c(0, 1, 0, 1),
    W = data.frame(feature = c("a", "b", "a", "b")),
    model = "neural",
    mode = "single",
    cache_path = cache_path,
    cache_overwrite = TRUE,
    conda_env_required = FALSE,
    neural_mcmc_control = list(checkpoint_path = explicit_checkpoint)
  )

  expect_equal(captured_control$checkpoint_path, explicit_checkpoint)
  expect_true(dir.exists(explicit_checkpoint))
})
