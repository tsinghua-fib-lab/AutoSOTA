# =============================================================================
# Tests for SVI step scaling
# =============================================================================

test_that("neural_optimal_svi_steps returns a positive integer", {
  steps <- strategize:::neural_optimal_svi_steps(
    n_obs = 1000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "full",
    batch_size = 512
  )

  expect_true(is.integer(steps))
  expect_gte(steps, 1L)
})

test_that("neural_optimal_svi_steps scales with model size and data size", {
  steps_base <- strategize:::neural_optimal_svi_steps(
    n_obs = 1000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "full",
    batch_size = 512
  )

  steps_bigger_model <- strategize:::neural_optimal_svi_steps(
    n_obs = 1000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 128,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "full",
    batch_size = 512
  )
  expect_gt(steps_bigger_model, steps_base)

  steps_more_data <- strategize:::neural_optimal_svi_steps(
    n_obs = 10000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "full",
    batch_size = 512
  )
  expect_lt(steps_more_data, steps_base)

  steps_batch_vi <- strategize:::neural_optimal_svi_steps(
    n_obs = 10000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "batch_vi",
    batch_size = 512
  )
  steps_smaller_batch <- strategize:::neural_optimal_svi_steps(
    n_obs = 10000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "batch_vi",
    batch_size = 64
  )
  expect_gt(steps_smaller_batch, steps_batch_vi)
})

test_that("neural_optimal_svi_steps handles pairwise cross-encoder configs", {
  steps <- strategize:::neural_optimal_svi_steps(
    n_obs = 500,
    n_factors = 6,
    factor_levels = rep(4L, 6),
    model_dims = 96,
    model_depth = 3,
    n_party_levels = 3,
    n_resp_party_levels = 3,
    n_resp_covariates = 2,
    n_outcomes = 2,
    pairwise_mode = TRUE,
    use_matchup_token = TRUE,
    use_cross_encoder = TRUE,
    use_cross_term = TRUE,
    subsample_method = "batch_vi",
    batch_size = 128
  )

  expect_true(is.integer(steps))
  expect_gte(steps, 1L)
})

test_that("neural_optimal_svi_steps pins exact pairwise batch_vi steps by mode", {
  common_args <- list(
    n_obs = 500,
    n_factors = 6,
    factor_levels = rep(4L, 6),
    model_dims = 96,
    model_depth = 3,
    n_party_levels = 3,
    n_resp_party_levels = 3,
    n_resp_covariates = 2,
    n_outcomes = 2,
    pairwise_mode = TRUE,
    use_matchup_token = TRUE,
    use_qk_norm = TRUE,
    subsample_method = "batch_vi",
    batch_size = 128
  )
  cases <- list(
    term = list(
      args = list(use_cross_encoder = FALSE, use_cross_term = TRUE, use_cross_attn = FALSE),
      expected = 2010L
    ),
    attn = list(
      args = list(use_cross_encoder = FALSE, use_cross_term = FALSE, use_cross_attn = TRUE),
      expected = 2177L
    ),
    full = list(
      args = list(use_cross_encoder = TRUE, use_cross_term = FALSE, use_cross_attn = FALSE),
      expected = 2212L
    )
  )

  for (mode_name in names(cases)) {
    case <- cases[[mode_name]]
    steps <- do.call(
      strategize:::neural_optimal_svi_steps,
      c(common_args, case$args)
    )
    expect_identical(
      steps,
      case$expected,
      info = sprintf("Expected exact optimal SVI steps for %s mode", mode_name)
    )
  }
})

test_that("neural_optimal_svi_steps accounts for QK-norm parameters", {
  base_args <- list(
    n_obs = 500,
    n_factors = 6,
    factor_levels = rep(4L, 6),
    model_dims = 96,
    model_depth = 3,
    n_party_levels = 3,
    n_resp_party_levels = 3,
    n_resp_covariates = 2,
    n_outcomes = 2,
    pairwise_mode = TRUE,
    use_matchup_token = TRUE,
    use_cross_encoder = TRUE,
    use_cross_term = FALSE,
    use_cross_attn = FALSE,
    subsample_method = "batch_vi",
    batch_size = 128
  )

  steps_without_qk <- do.call(
    strategize:::neural_optimal_svi_steps,
    c(base_args, list(use_qk_norm = FALSE))
  )
  steps_with_qk <- do.call(
    strategize:::neural_optimal_svi_steps,
    c(base_args, list(use_qk_norm = TRUE))
  )

  expect_gte(steps_with_qk, steps_without_qk)
  expect_gt(steps_with_qk, steps_without_qk)
})

test_that("neural_optimal_svi_steps treats subsample_method='batch' like 'batch_vi'", {
  steps_small_batch <- strategize:::neural_optimal_svi_steps(
    n_obs = 10000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "batch",
    batch_size = 64
  )
  steps_small_batch_vi <- strategize:::neural_optimal_svi_steps(
    n_obs = 10000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "batch_vi",
    batch_size = 64
  )
  expect_identical(steps_small_batch, steps_small_batch_vi)

  steps_large_batch <- strategize:::neural_optimal_svi_steps(
    n_obs = 10000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "batch",
    batch_size = 512
  )
  steps_large_batch_vi <- strategize:::neural_optimal_svi_steps(
    n_obs = 10000,
    n_factors = 10,
    factor_levels = rep(3L, 10),
    model_dims = 64,
    model_depth = 2,
    n_party_levels = 2,
    n_resp_party_levels = 2,
    n_resp_covariates = 0,
    n_outcomes = 1,
    pairwise_mode = FALSE,
    subsample_method = "batch_vi",
    batch_size = 512
  )
  expect_identical(steps_large_batch, steps_large_batch_vi)
})
