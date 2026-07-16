test_that("GLM outcome model exports overall fit metrics (pairwise)", {
  dat <- generate_pairwise_performance_test_data(
    n_pairs = 1000L,
    n_factors = 3,
    n_levels = 2,
    seed = 20260127
  )

  predictor <- strategic_prediction(
    Y = dat$Y,
    W = dat$W,
    model = "glm",
    mode = "pairwise",
    pair_id = dat$pair_id,
    profile_order = dat$profile_order
  )

  metrics <- predictor$fit$fit_metrics
  diag_info <- format_oos_failure_details(metrics)
  non_null_folds <- Filter(is.list, metrics$by_fold)

  expect_type(metrics, "list")
  expect_identical(metrics$likelihood, "binomial")
  expect_equal(metrics$n_eval, length(dat$Y) / 2)
  expect_gte(metrics$n_eval, 1000L)
  expect_true(is.list(metrics$in_sample_metrics), info = diag_info)
  expect_true(is.numeric(metrics$pred_quantiles), info = diag_info)
  expect_setequal(names(metrics$pred_quantiles), c("p05", "p25", "p50", "p75", "p95"))
  expect_setequal(names(metrics$confusion_0_5), c("tn", "fp", "fn", "tp"))
  expect_true(length(non_null_folds) >= 1L, info = diag_info)
  first_fold <- non_null_folds[[1]]
  expect_true(is.numeric(first_fold$pred_quantiles), info = diag_info)

  expect_true(is.finite(metrics$auc), info = diag_info)
  expect_true(metrics$auc >= 0, info = diag_info)
  expect_true(metrics$auc <= 1, info = diag_info)

  expect_true(is.finite(metrics$accuracy), info = diag_info)
  expect_true(metrics$accuracy >= 0, info = diag_info)
  expect_true(metrics$accuracy <= 1, info = diag_info)

  expect_true(is.finite(metrics$log_loss), info = diag_info)
  expect_true(metrics$log_loss >= 0, info = diag_info)

  expect_true(is.finite(metrics$brier), info = diag_info)
  expect_true(metrics$brier >= 0, info = diag_info)
  expect_true(metrics$brier <= 1, info = diag_info)
})
