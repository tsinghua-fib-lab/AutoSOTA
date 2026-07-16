# =============================================================================
# Neural Outcome Model Tests
# =============================================================================

test_that("schema dropout resolver handles defaults and overrides", {
  zero <- strategize:::neural_resolve_schema_dropout(NULL)
  expect_equal(unlist(zero, use.names = FALSE), rep(0, 6))

  defaults <- strategize:::neural_resolve_schema_dropout(TRUE)
  expect_equal(defaults$experiment_token, 0.25)
  expect_equal(defaults$covariate_token, 0.05)

  custom <- strategize:::neural_resolve_schema_dropout(list(
    schema_text = 0.2,
    covariate_token = 0.08
  ))
  expect_equal(custom$schema_text, 0.2)
  expect_equal(custom$covariate_token, 0.08)
  expect_equal(custom$experiment_token, 0)

  custom_defaults <- strategize:::neural_resolve_schema_dropout(list(
    defaults = TRUE,
    factor_token = 0.01
  ))
  expect_equal(custom_defaults$schema_text, 0.10)
  expect_equal(custom_defaults$factor_token, 0.01)

  expect_error(
    strategize:::neural_resolve_schema_dropout(list(context_token = 1)),
    "\\[0, 1\\)"
  )
  expect_error(
    strategize:::neural_resolve_schema_dropout(list(context = 0.1)),
    "Unknown schema_dropout"
  )
})

test_that("neural init policy resolves RMS and residual depth defaults", {
  expect_equal(strategize:::neural_resolve_rms_scale(NULL), 0.25)
  expect_equal(strategize:::neural_resolve_rms_scale(0.5), 0.5)
  expect_identical(
    strategize:::neural_resolve_residual_weight_depth_scale("floor-one"),
    "floor_one"
  )

  shallow <- strategize:::neural_resolve_init_policy(
    model_depth = 1L,
    model_dims = 16L
  )
  expect_equal(shallow$RMS_scale, 0.25)
  expect_identical(shallow$residual_weight_depth_scale, "floor_one")
  expect_equal(shallow$depth_prior_scale, sqrt(2))
  expect_equal(shallow$residual_weight_multiplier, sqrt(2))
  expect_equal(shallow$gate_sd_scale, 0.1 * sqrt(2))

  depth_two <- strategize:::neural_resolve_init_policy(
    model_depth = 2L,
    model_dims = 16L
  )
  expect_equal(depth_two$residual_weight_multiplier, 1)

  deep_default <- strategize:::neural_resolve_init_policy(
    model_depth = 4L,
    model_dims = 16L
  )
  expect_equal(deep_default$residual_weight_multiplier, 1)
  expect_equal(deep_default$weight_sd_scale, sqrt(2) / 4)
  expect_equal(deep_default$residual_weight_sd_scale, sqrt(2) / 4)
  expect_equal(deep_default$gate_depth_multiplier, sqrt(2) / 2)

  legacy <- strategize:::neural_resolve_init_policy(
    model_depth = 4L,
    model_dims = 16L,
    RMS_scale = 0.5,
    residual_weight_depth_scale = "legacy"
  )
  expect_equal(legacy$RMS_scale, 0.5)
  expect_equal(legacy$residual_weight_multiplier, sqrt(2) / 2)
  expect_equal(legacy$residual_weight_sd_scale, (sqrt(2) / 4) * (sqrt(2) / 2))

  no_depth <- strategize:::neural_resolve_init_policy(
    model_depth = 4L,
    model_dims = 16L,
    residual_weight_depth_scale = "none"
  )
  expect_equal(no_depth$residual_weight_multiplier, 1)

  expect_error(
    strategize:::neural_resolve_rms_scale(0),
    "RMS_scale"
  )
  expect_error(
    strategize:::neural_resolve_residual_weight_depth_scale("wide"),
    "residual_weight_depth_scale"
  )
})

test_that("schema dropout helper scales unit masks and keeps token masks binary", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  key <- strategize:::strenv$jax$random$PRNGKey(23L)
  keys <- strategize:::strenv$jax$random$split(key, 2L)
  binary <- strategize:::neural_schema_dropout_random_keep(
    strategize:::strenv$jnp$take(keys, 0L, axis = 0L),
    0.5,
    8L,
    4L
  )
  inverted <- strategize:::neural_schema_dropout_random_keep(
    strategize:::strenv$jnp$take(keys, 1L, axis = 0L),
    0.5,
    8L,
    4L,
    inverted = TRUE
  )
  binary_vals <- unique(as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(binary))))
  inverted_vals <- unique(as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(inverted))))

  expect_true(all(binary_vals %in% c(0, 1)))
  expect_true(all(inverted_vals %in% c(0, 2)))
})

test_that("attention and sigma numeric guards use dtype-aware floors", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  scores <- strategize:::strenv$jnp$ones(
    reticulate::tuple(1L, 1L, 1L, 1L),
    dtype = strategize:::strenv$jnp$float16
  )
  mask_value <- as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(
    strategize:::neural_attention_mask_value(scores)
  )))
  expect_true(is.finite(mask_value))
  expect_lte(mask_value, -60000)

  sigma <- strategize:::strenv$jnp$array(1e-9, dtype = strategize:::strenv$dtj)
  sigma_floor <- as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(
    strategize:::neural_floor_sigma(sigma, dtype = strategize:::strenv$dtj)
  )))
  expect_lt(abs(sigma_floor - 1e-3), 1e-7)
})

test_that("pairwise Bernoulli logit scale controls final R logits", {
  expect_false(strategize:::neural_resolve_learned_pairwise_bernoulli_logit_scale(NULL))
  expect_true(strategize:::neural_resolve_learned_pairwise_bernoulli_logit_scale("learned"))
  expect_error(
    strategize:::neural_resolve_learned_pairwise_bernoulli_logit_scale("maybe"),
    "learned_pairwise_bernoulli_logit_scale"
  )
  expect_equal(
    strategize:::neural_resolve_pairwise_bernoulli_logit_scale_prior_sd(0.25, enabled = TRUE),
    0.25
  )
  expect_error(
    strategize:::neural_resolve_pairwise_bernoulli_logit_scale_prior_sd(0, enabled = TRUE),
    "pairwise_bernoulli_logit_scale_prior_sd"
  )

  logits <- c(-0.5, 0, 0.5)
  model_info <- list(
    likelihood = "bernoulli",
    learned_pairwise_bernoulli_logit_scale = TRUE,
    pairwise_bernoulli_logit_scale = 4,
    low_rank_interaction_rank = 0L,
    low_rank_logit_transform = "none"
  )
  expect_equal(
    strategize:::neural_apply_pairwise_bernoulli_logit_adjustment_r(logits, model_info),
    logits * 4
  )
  model_info$learned_pairwise_bernoulli_logit_scale <- FALSE
  expect_equal(
    strategize:::neural_apply_pairwise_bernoulli_logit_adjustment_r(logits, model_info),
    logits
  )
})

test_that("neural architecture control resolvers preserve legacy upgrade defaults", {
  expect_identical(strategize:::neural_resolve_pairwise_antisymmetry(NULL), "strict")
  expect_identical(strategize:::neural_resolve_pairwise_antisymmetry(FALSE), "legacy")
  expect_true(strategize:::neural_pairwise_antisymmetry_strict(list(
    pairwise_antisymmetry = "strict"
  )))
  expect_false(strategize:::neural_pairwise_antisymmetry_strict(list()))

  expect_identical(
    strategize:::neural_resolve_additive_utility_mode("auto", factor_tokenization = "fused"),
    "on"
  )
  expect_identical(
    strategize:::neural_resolve_additive_utility_mode("auto", factor_tokenization = "legacy"),
    "off"
  )
  expect_identical(
    strategize:::neural_resolve_additive_utility_normalization(
      additive_utility_mode = "on"
    ),
    "rms"
  )
  expect_identical(
    strategize:::neural_resolve_additive_utility_normalization(
      value = "none",
      additive_utility_mode = "on",
      supplied = TRUE
    ),
    "none"
  )
  expect_identical(
    strategize:::neural_resolve_additive_utility_normalization(
      additive_utility_mode = "off"
    ),
    "none"
  )
  expect_equal(
    strategize:::neural_resolve_additive_head_weight_target_rms(
      model_dims = 16L,
      normalization = "rms"
    ),
    0.25
  )
  expect_true(strategize:::neural_additive_utility_normalization_enabled(list(
    additive_utility_mode = "on",
    additive_utility_normalization = "rms",
    additive_head_weight_target_rms = 0.25
  )))
  expect_error(
    strategize:::neural_resolve_additive_utility_normalization(
      value = "batchnorm",
      additive_utility_mode = "on",
      supplied = TRUE
    ),
    "additive_utility_normalization"
  )

  calibration <- strategize:::neural_resolve_calibration_control(
    list(enabled = "auto", prior_sd = 0.25),
    likelihood = "categorical"
  )
  expect_true(calibration$enabled)
  expect_identical(calibration$method, "logit_scale")
  expect_equal(calibration$prior_sd, 0.25)

  no_calibration <- strategize:::neural_resolve_calibration_control(
    list(enabled = "auto"),
    likelihood = "normal"
  )
  expect_false(no_calibration$enabled)
  expect_identical(no_calibration$method, "none")

  expect_error(
    strategize:::neural_validate_calibration_temperature_compatibility(
      TRUE,
      list(enabled = TRUE, method = "logit_scale", prior_sd = 0.5)
    ),
    "cannot learn both"
  )
  expect_no_error(
    strategize:::neural_validate_calibration_temperature_compatibility(
      TRUE,
      list(enabled = FALSE, method = "none", prior_sd = NULL)
    )
  )
})

test_that("context_present_masking false disables runtime context masks", {
  info <- strategize:::neural_set_pairwise_context_model_info(
    list(),
    context_present_masking = FALSE
  )
  expect_false(info$context_present_masking)
  expect_null(strategize:::neural_context_present_mask(
    context_present = c(1L, 0L),
    model_info = info
  ))
})

test_that("balanced compact sampler draws studies and respondents hierarchically", {
  config <- strategize:::neural_resolve_balanced_sampling(list(
    scheme = "study_equal_respondent",
    within_respondent = "uniform_observation",
    replacement = TRUE
  ))
  expect_true(isTRUE(config$enabled))

  obs_idx <- seq_len(9L)
  study_index <- c(rep(0L, 6L), rep(1L, 3L))
  respondent_id <- c("a", "a", "b", "b", "b", "c", "d", "e", "e")
  state <- strategize:::neural_build_balanced_sampling_state(
    obs_idx = obs_idx,
    study_index = study_index,
    respondent_id = respondent_id,
    config = config
  )

  expect_identical(as.integer(state$n_studies), 2L)
  expect_equal(unname(state$respondent_counts_by_study), c(3L, 2L))
  expect_equal(unname(state$observation_counts_by_study), c(6L, 3L))

  set.seed(42)
  draws <- strategize:::neural_sample_balanced_obs_idx(state, batch_size = 20000L)
  study_tab <- prop.table(table(study_index[draws]))
  expect_lt(abs(unname(study_tab[["0"]]) - 0.5), 0.03)
  expect_lt(abs(unname(study_tab[["1"]]) - 0.5), 0.03)

  respondent_tab_0 <- prop.table(table(respondent_id[draws[study_index[draws] == 0L]]))
  expect_lt(max(abs(unname(respondent_tab_0[c("a", "b", "c")]) - 1 / 3)), 0.04)
  respondent_tab_1 <- prop.table(table(respondent_id[draws[study_index[draws] == 1L]]))
  expect_lt(max(abs(unname(respondent_tab_1[c("d", "e")]) - 1 / 2)), 0.04)

  expect_error(
    strategize:::neural_build_balanced_sampling_state(
      obs_idx = obs_idx,
      study_index = study_index,
      respondent_id = c(respondent_id[-1L], NA_character_),
      config = config
    ),
    "non-missing respondent_id"
  )
})

test_that("universal mixed loss weighting supports empirical and balanced-cell policies", {
  counts <- c(
    pair_bernoulli = 2291L,
    single_bernoulli = 43L,
    single_categorical = 36L,
    single_normal = 68L,
    single_ordinal = 39L
  )
  task_mode <- rep(
    c("pairwise", "single", "single", "single", "single"),
    counts
  )
  likelihood_code <- rep(c(0L, 0L, 1L, 2L, 3L), counts)

  empirical <- strategize:::neural_universal_loss_weighting(
    task_mode = task_mode,
    likelihood_code = likelihood_code,
    policy = "empirical"
  )
  expect_null(empirical$weights)
  expect_equal(empirical$observation_weights, rep(1, sum(counts)))
  empirical_cells <- empirical$diagnostics$cells
  expect_equal(
    empirical_cells$effective_weighted_share,
    empirical_cells$raw_observation_share,
    tolerance = 1e-12
  )

  balanced <- strategize:::neural_universal_loss_weighting(
    task_mode = task_mode,
    likelihood_code = likelihood_code,
    policy = "balanced_cell",
    clip = c(0.25, 4.0)
  )
  balanced_cells <- balanced$diagnostics$cells
  names(counts) <- c("pairwise::0", "single::0", "single::1", "single::2", "single::3")
  counts <- counts[balanced_cells$cell]
  raw_weight <- sum(counts) / (length(counts) * counts)
  clipped_weight <- pmin(pmax(raw_weight, 0.25), 4.0)
  normalized_weight <- clipped_weight / sum(clipped_weight * counts) * sum(counts)
  expected_share <- counts * normalized_weight / sum(counts * normalized_weight)

  expect_equal(
    balanced_cells$effective_observation_weight,
    as.numeric(normalized_weight),
    tolerance = 1e-12
  )
  expect_equal(
    balanced_cells$effective_weighted_share,
    as.numeric(expected_share),
    tolerance = 1e-12
  )
  expect_lt(
    balanced_cells$effective_weighted_share[balanced_cells$cell == "pairwise::0"],
    balanced_cells$raw_observation_share[balanced_cells$cell == "pairwise::0"]
  )
  expect_gt(
    balanced_cells$effective_weighted_share[balanced_cells$cell == "single::2"],
    balanced_cells$raw_observation_share[balanced_cells$cell == "single::2"]
  )

  expect_error(
    strategize:::neural_resolve_universal_loss_weighting("equal_cells"),
    "universal_loss_weighting"
  )
})

test_that("compact mixed branch loss scales preserve empirical branch mass", {
  scales <- strategize:::neural_compact_mixed_branch_loss_scales(
    n_pair = 92L,
    n_single = 8L,
    pair_batch_n = 9L,
    single_batch_n = 1L
  )
  expect_equal(scales$pair, 92 / 9)
  expect_equal(scales$single, 8)
  expect_equal(scales$pair * 9, 92)
  expect_equal(scales$single * 1, 8)

  padded <- strategize:::neural_compact_mixed_branch_loss_scales(
    n_pair = 92L,
    n_single = 8L,
    pair_batch_n = 9L,
    single_batch_n = 0L
  )
  expect_equal(padded$single, 0)

  balanced_sampler <- strategize:::neural_compact_mixed_branch_loss_scales(
    n_pair = 92L,
    n_single = 8L,
    pair_batch_n = 9L,
    single_batch_n = 1L,
    total_n = 100L,
    batch_n = 10L,
    balanced_sampling = TRUE
  )
  expect_equal(balanced_sampler$pair, 10)
  expect_equal(balanced_sampler$single, 10)
})

test_that("schema dropout token masks preserve one factor token", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  token_mask <- strategize:::strenv$jnp$array(
    matrix(c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE), nrow = 2, byrow = TRUE)
  )
  keep <- strategize:::strenv$jnp$array(
    matrix(c(0, 0, 0, 0, 1, 0), nrow = 2, byrow = TRUE)
  )
  factor_out <- strategize:::neural_schema_dropout_apply_token_mask(
    token_mask,
    list(factor_token = keep),
    "factor_token",
    preserve_one = TRUE
  )
  expect_equal(
    matrix(as.logical(strategize:::cs2step_neural_to_r_array(factor_out)), nrow = 2),
    matrix(c(TRUE, FALSE, FALSE, FALSE, TRUE, FALSE), nrow = 2, byrow = TRUE)
  )

  cov_out <- strategize:::neural_schema_dropout_apply_token_mask(
    token_mask,
    list(covariate_token = keep),
    "covariate_token",
    preserve_one = FALSE
  )
  expect_equal(
    matrix(as.logical(strategize:::cs2step_neural_to_r_array(cov_out)), nrow = 2),
    matrix(c(FALSE, FALSE, FALSE, FALSE, TRUE, FALSE), nrow = 2, byrow = TRUE)
  )
})

get_neural_fit <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 40, seed = 123)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 32L,
        svi_lr = 0.005,
        early_stopping_n_checks = 20L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_perf <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))
    withr::local_seed(123)

    data <- generate_pairwise_performance_test_data(
      n_pairs = 1000L,
      n_factors = 3,
      n_levels = 2,
      seed = 20260327
    )
    data <- add_adversarial_structure(data, seed = 20260328)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 32L,
        ModelDims = 16L,
        ModelDepth = 1L,
        low_rank_interaction_rank = 4L,
        svi_lr = 0.005,
        early_stopping_n_checks = 20L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c(
        "pair_id",
        "respondent_id",
        "respondent_task_id",
        "profile_order",
        "competing_group_variable_respondent",
        "competing_group_variable_candidate",
        "competing_group_competition_variable_candidate"
      )],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_attn <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 30, seed = 321)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 16L,
        cross_candidate_encoder = "attn",
        ModelDims = 16L,
        ModelDepth = 1L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_none <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 24, seed = 20260329)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 16L,
        cross_candidate_encoder = FALSE,
        ModelDims = 16L,
        ModelDepth = 1L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_true <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 24, seed = 20260330)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 16L,
        cross_candidate_encoder = TRUE,
        ModelDims = 16L,
        ModelDepth = 1L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_full_attn_res <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true",
      STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
      STRATEGIZE_NEURAL_EVAL_SEED = "123"
    ))

    data <- generate_test_data(n = 24, seed = 20260331)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    base_neural_control <- params$neural_mcmc_control
    if (is.null(base_neural_control)) {
      base_neural_control <- list()
    }
    params$neural_mcmc_control <- modifyList(
      base_neural_control,
      list(
        subsample_method = "batch_vi",
        batch_size = 16L,
        cross_candidate_encoder = FALSE,
        residual_mode = "full_attn",
        ModelDims = 16L,
        ModelDepth = 2L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    ))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_attn_output_vi <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true"
    ))

    data <- generate_test_data(n = 24, seed = 20260326)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    params$neural_mcmc_control <- modifyList(
      params$neural_mcmc_control,
      list(
        cross_candidate_encoder = "attn",
        ModelDims = 16L,
        ModelDepth = 1L,
        subsample_method = "batch_vi",
        uncertainty_scope = "output",
        svi_steps = "optimal",
        batch_size = 16L,
        early_stopping = FALSE,
        eval_enabled = FALSE,
        warn_stage_imbalance_pct = 0,
        warn_min_cell_n = 0L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- suppressWarnings(do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    )))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

get_neural_fit_attn_output_vi_default_es <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()

    withr::local_envvar(c(
      STRATEGIZE_NEURAL_FAST_MCMC = "true"
    ))

    data <- generate_test_data(n = 24, seed = 20260330)
    params <- default_strategize_params(fast = TRUE)
    params$outcome_model_type <- "neural"
    params$neural_mcmc_control <- modifyList(
      params$neural_mcmc_control,
      list(
        cross_candidate_encoder = "attn",
        ModelDims = 16L,
        ModelDepth = 1L,
        subsample_method = "batch_vi",
        uncertainty_scope = "output",
        svi_steps = 25L,
        batch_size = 16L,
        eval_enabled = FALSE,
        warn_stage_imbalance_pct = 0,
        warn_min_cell_n = 0L
      )
    )

    p_list <- generate_test_p_list(data$W)

    res <- suppressWarnings(do.call(strategize, c(
      list(Y = data$Y, W = data$W, p_list = p_list),
      data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
      params
    )))

    cache <<- list(res = res, data = data, p_list = p_list)
    cache
  }
})

count_svi_run_calls <- function(expr) {
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  reticulate::py_run_string(
    paste(
      "import numpyro.infer.svi as _strategize_svi_mod",
      "_strategize_original_svi_run = _strategize_svi_mod.SVI.run",
      "_strategize_svi_run_call_count = 0",
      "def _strategize_counted_svi_run(self, *args, **kwargs):",
      "    global _strategize_svi_run_call_count",
      "    _strategize_svi_run_call_count += 1",
      "    return _strategize_original_svi_run(self, *args, **kwargs)",
      "_strategize_svi_mod.SVI.run = _strategize_counted_svi_run",
      sep = "\n"
    )
  )
  on.exit(
    reticulate::py_run_string(
      paste(
        "import numpyro.infer.svi as _strategize_svi_mod",
        "_strategize_svi_mod.SVI.run = _strategize_original_svi_run",
        "del _strategize_original_svi_run",
        sep = "\n"
      )
    ),
    add = TRUE
  )

  value <- force(expr)
  count <- as.integer(reticulate::py_to_r(reticulate::py_eval("_strategize_svi_run_call_count")))
  list(value = value, count = count)
}

local_strategize_binding <- function(name, value, env = parent.frame()) {
  ns <- asNamespace("strategize")
  old_value <- get(name, envir = ns, inherits = FALSE)
  was_locked <- bindingIsLocked(name, ns)
  if (was_locked) {
    unlockBinding(name, ns)
  }
  assign(name, value, envir = ns)
  if (was_locked) {
    lockBinding(name, ns)
  }

  withr::defer({
    if (bindingIsLocked(name, ns)) {
      unlockBinding(name, ns)
    }
    assign(name, old_value, envir = ns)
    if (was_locked) {
      lockBinding(name, ns)
    }
  }, envir = env)

  invisible(old_value)
}

capture_messages <- function(expr) {
  messages <- character()
  value <- withCallingHandlers(
    expr,
    message = function(cnd) {
      message_text <- sub("[\r\n]+$", "", conditionMessage(cnd), perl = TRUE)
      messages <<- c(messages, message_text)
      invokeRestart("muffleMessage")
    }
  )
  list(value = value, messages = messages)
}

extract_log_numeric_field <- function(line, field) {
  pattern <- paste0("(?:^|; )", field, "=([^;]+)")
  hit <- regmatches(line, regexpr(pattern, line, perl = TRUE))
  if (!length(hit) || identical(hit, "")) {
    stop(sprintf("Field '%s' not found in log line: %s", field, line), call. = FALSE)
  }
  value <- sub(paste0("^.*", field, "="), "", hit)
  value <- sub("\\.$", "", value)
  value <- sub("s$", "", value)
  if (identical(value, "NA")) {
    return(NA_real_)
  }
  as.numeric(value)
}

expect_log_rate <- function(line, rate_field, count_field, elapsed_field, tolerance = 0.05) {
  rate <- extract_log_numeric_field(line, rate_field)
  count <- extract_log_numeric_field(line, count_field)
  elapsed <- extract_log_numeric_field(line, elapsed_field)
  expect_true(is.finite(rate))
  expect_true(is.finite(count))
  expect_true(is.finite(elapsed))
  expect_gt(elapsed, 0)
  expect_equal(rate, count / elapsed, tolerance = tolerance)
}

run_output_only_attn_vi_fit <- function(seed,
                                        early_stopping = TRUE,
                                        svi_steps = 25L,
                                        residual_mode = "standard",
                                        eval_enabled = FALSE,
                                        optimizer = NULL,
                                        neural_mcmc_control_overrides = NULL) {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  data <- generate_test_data(n = 24, seed = seed)
  params <- default_strategize_params(fast = TRUE)
  params$outcome_model_type <- "neural"
  params$neural_mcmc_control <- modifyList(
    params$neural_mcmc_control,
      list(
        cross_candidate_encoder = "attn",
        residual_mode = residual_mode,
        ModelDims = 16L,
        ModelDepth = 1L,
        subsample_method = "batch_vi",
      uncertainty_scope = "output",
      svi_steps = svi_steps,
      batch_size = 16L,
      early_stopping = early_stopping,
      eval_enabled = eval_enabled,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )
  if (!is.null(optimizer)) {
    params$neural_mcmc_control$optimizer <- optimizer
  }
  if (!is.null(neural_mcmc_control_overrides)) {
    params$neural_mcmc_control <- modifyList(
      params$neural_mcmc_control,
      neural_mcmc_control_overrides
    )
    if ("early_stopping_validation_max_n" %in% names(neural_mcmc_control_overrides) &&
        is.null(neural_mcmc_control_overrides$early_stopping_validation_max_n)) {
      params$neural_mcmc_control$early_stopping_validation_max_n <- NULL
    }
  }

  p_list <- generate_test_p_list(data$W)

  suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  )))
}

get_neural_model_info <- function(res) {
  model_info <- res$neural_model_info$ast
  if (is.null(model_info)) model_info <- res$neural_model_info$dag
  if (is.null(model_info)) model_info <- res$neural_model_info$ast0
  if (is.null(model_info)) model_info <- res$neural_model_info$dag0
  model_info
}

compact_w_idx_from_matrix <- function(W_idx, factor_levels) {
  factor_names <- colnames(W_idx)
  holdout_codes <- as.integer(factor_levels) + 1L
  block <- structure(
    list(
      n_rows = as.integer(nrow(W_idx)),
      n_factors = as.integer(ncol(W_idx)),
      factor_names = factor_names,
      present_cols = seq_len(ncol(W_idx)),
      values = W_idx,
      holdout_codes = holdout_codes,
      experiment_index = 0L
    ),
    class = c("cs_w_idx_block", "cs_w_idx_compact")
  )
  structure(
    list(
      blocks = list(block),
      n_rows = as.integer(nrow(W_idx)),
      n_factors = as.integer(ncol(W_idx)),
      factor_names = factor_names,
      holdout_codes = holdout_codes
    ),
    class = c("cs_w_idx_blocks", "cs_w_idx_compact")
  )
}

test_that("compact W holdout codes target the explicit missing level", {
  bad_block <- structure(
    list(
      n_rows = 1L,
      n_factors = 1L,
      factor_names = "feature",
      present_cols = integer(0),
      values = matrix(integer(0), nrow = 1L, ncol = 0L),
      holdout_codes = 2L,
      experiment_index = 0L
    ),
    class = c("cs_w_idx_block", "cs_w_idx_compact")
  )
  bad <- structure(
    list(
      blocks = list(bad_block),
      n_rows = 1L,
      n_factors = 1L,
      factor_names = "feature",
      holdout_codes = 2L
    ),
    class = c("cs_w_idx_blocks", "cs_w_idx_compact")
  )

  expect_error(
    strategize:::cs2step_validate_w_idx_compact(bad, factor_levels = 2L),
    "factor_levels \\+ 1"
  )

  good <- bad
  good$holdout_codes <- 3L
  good$blocks[[1L]]$holdout_codes <- 3L
  expect_silent(strategize:::cs2step_validate_w_idx_compact(good, factor_levels = 2L))
  expect_equal(
    unname(strategize:::cs2step_materialize_w_idx_compact(good)),
    matrix(3L, nrow = 1L, ncol = 1L)
  )
})

test_that("compact X materialization follows present column order and masks missing rows", {
  block <- structure(
    list(
      n_rows = 2L,
      n_covariates = 2L,
      covariate_names = c("age", "income"),
      present_cols = c(2L, 1L),
      values = cbind(income = c(100, 200), age = c(30, 40)),
      missing_rows_by_col = list(`1` = 2L),
      experiment_index = 0L
    ),
    class = c("cs_covariate_values_block", "cs_covariate_values_compact")
  )
  compact <- structure(
    list(
      blocks = list(block),
      n_rows = 2L,
      n_covariates = 2L,
      covariate_names = c("age", "income")
    ),
    class = c("cs_covariate_values_blocks", "cs_covariate_values_compact")
  )

  values <- strategize:::cs2step_materialize_x_compact(compact)
  present <- strategize:::cs2step_materialize_x_present_compact(compact)
  expect_equal(unname(values), matrix(c(30, 40, 100, 200), nrow = 2L))
  expect_equal(unname(present), matrix(c(1, 0, 1, 1), nrow = 2L))

  bad <- compact
  bad$blocks[[1L]]$values <- matrix(c(100, 200), ncol = 1L)
  expect_error(
    strategize:::cs2step_materialize_x_compact(bad),
    "one column per present column"
  )
})

run_compact_svi_fit <- function(compact_update_scan = "required",
                                compact_update_chunk_size = 2L,
                                svi_steps = 4L,
                                early_stopping = FALSE,
                                early_stopping_n_checks = 2L,
                                checkpoint_path = NULL,
                                checkpoint_n_checks = 2L) {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  W <- data.frame(feature = rep(c("A", "B"), 4L), stringsAsFactors = FALSE)
  names_list <- strategize:::cs2step_build_names_list(W)
  W_idx <- strategize:::cs2step_encode_W_indices(
    W,
    names_list = names_list,
    unknown = "error",
    pad_unknown = 0L
  )
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))
  W_idx_compact <- compact_w_idx_from_matrix(W_idx, factor_levels)

  fit <- suppressWarnings(strategize:::cs2step_eval_outcome_model_neural(
    Y = c(1, 0, 1, 0, 1, 0, 1, 0),
    W_idx = NULL,
    W_idx_compact = W_idx_compact,
    names_list = names_list,
    factor_levels = factor_levels,
    diff = TRUE,
    pair_id = rep(seq_len(4L), each = 2L),
    profile_order = rep(1:2, 4L),
    conda_env_required = TRUE,
    neural_mcmc_control = list(
      ModelDims = 8L,
      ModelDepth = 1L,
      subsample_method = "batch_vi",
      optimizer = "adam",
      svi_steps = as.integer(svi_steps),
      svi_num_draws = 1L,
      batch_size = 4L,
      compact_update_chunk_size = as.integer(compact_update_chunk_size),
      compact_update_scan = compact_update_scan,
      early_stopping = isTRUE(early_stopping),
      early_stopping_n_checks = as.integer(early_stopping_n_checks),
      early_stopping_validation_frac = 1,
      early_stopping_validation_max_n = NULL,
      eval_enabled = FALSE,
      checkpoint_path = checkpoint_path,
      checkpoint_n_checks = as.integer(checkpoint_n_checks),
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  ))
  get_neural_model_info(list(neural_model_info = list(ast = fit$neural_model_info)))
}

compute_binary_null_metrics <- function(y) {
  y <- as.numeric(y)
  y <- y[is.finite(y)]
  p_null <- mean(y)
  p_null <- min(max(p_null, 1e-6), 1 - 1e-6)
  list(
    log_loss = -mean(y * log(p_null) + (1 - y) * log(1 - p_null)),
    accuracy = max(mean(y), 1 - mean(y)),
    brier = mean((p_null - y) ^ 2)
  )
}

local_strenv_bindings <- function(bindings, .local_envir = parent.frame()) {
  force(.local_envir)
  binding_names <- as.character(bindings)
  had_binding <- setNames(logical(length(binding_names)), binding_names)
  old_values <- setNames(vector("list", length(binding_names)), binding_names)

  for (binding in binding_names) {
    had_binding[[binding]] <- exists(binding, envir = strategize:::strenv, inherits = FALSE)
    if (had_binding[[binding]]) {
      old_values[[binding]] <- get(binding, envir = strategize:::strenv, inherits = FALSE)
    }
  }

  withr::defer({
    for (binding in binding_names) {
      if (had_binding[[binding]]) {
        assign(binding, old_values[[binding]], envir = strategize:::strenv)
      } else if (exists(binding, envir = strategize:::strenv, inherits = FALSE)) {
        rm(list = binding, envir = strategize:::strenv)
      }
    }
  }, envir = .local_envir)
}

# pkgload::load_all() can error on strategize:::strenv$foo <- value and surface
# loadNamespace("*tmp*"), so use assign() for test-time strenv mutations.
set_strenv_bindings <- function(values) {
  for (binding in names(values)) {
    assign(binding, values[[binding]], envir = strategize:::strenv)
  }
}

test_that("local_strenv_bindings cleans up bindings at caller scope", {
  binding <- paste0(".test_local_strenv_binding_", Sys.getpid())
  if (exists(binding, envir = strategize:::strenv, inherits = FALSE)) {
    rm(list = binding, envir = strategize:::strenv)
  }

  local({
    local_strenv_bindings(binding)
    set_strenv_bindings(setNames(list("temporary"), binding))
    expect_identical(
      get(binding, envir = strategize:::strenv, inherits = FALSE),
      "temporary"
    )
  })

  expect_false(exists(binding, envir = strategize:::strenv, inherits = FALSE))
})

generate_average_case_neural_data <- function(n = 24, n_factors = 3, seed = 20260326) {
  withr::local_seed(seed)

  levels <- LETTERS[1:2]
  W <- matrix(
    sample(levels, n * n_factors, replace = TRUE),
    nrow = n,
    ncol = n_factors
  )
  colnames(W) <- paste0("V", seq_len(n_factors))

  effect_sizes <- seq(0.5, 0.2, length.out = n_factors)
  signal <- rowSums((W == "B") * rep(effect_sizes, each = n))
  Y <- as.numeric(signal + rnorm(n, sd = 0.1))

  list(Y = Y, W = W)
}

run_average_case_neural_fit <- function(vi_guide = "auto_diagonal",
                                        compute_se = FALSE,
                                        nMonte_Qglm = 4L,
                                        svi_steps = 20L,
                                        force_reinforce = FALSE,
                                        seed = 20260326) {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))

  data <- generate_average_case_neural_data(seed = seed)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE
  params$force_gaussian <- TRUE
  params$force_reinforce <- force_reinforce
  params$compute_se <- compute_se
  params$outcome_model_type <- "neural"
  params$nMonte_Qglm <- as.integer(nMonte_Qglm)
  base_neural_control <- params$neural_mcmc_control
  if (is.null(base_neural_control)) {
    base_neural_control <- list()
  }
  params$neural_mcmc_control <- modifyList(
    base_neural_control,
    list(
      subsample_method = "batch_vi",
      uncertainty_scope = "output",
      vi_guide = vi_guide,
      ModelDims = 8L,
      ModelDepth = 1L,
      batch_size = 16L,
      svi_steps = as.integer(svi_steps),
      svi_num_draws = 5L,
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  p_list <- generate_test_p_list(data$W)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    params
  )))

  list(res = res, data = data, p_list = p_list)
}

test_that("capture_messages strips trailing newlines without trimming leading whitespace", {
  captured <- capture_messages({
    message("plain text")
    message("\n indented text")
  })

  expect_identical(captured$messages[[1]], "plain text")
  expect_identical(captured$messages[[2]], "\n indented text")
})

test_that("neural output site init values target scalar normal regression only", {
  init <- strategize:::neural_build_output_site_init_values(
    Y = c(1, 3, 5, NA_real_),
    likelihood = "normal",
    nOutcomes = 1L
  )
  expect_equal(init$b_out, mean(c(1, 3, 5)))
  expect_equal(init$tau_b, abs(mean(c(1, 3, 5))))
  expect_equal(init$sigma, stats::mad(c(1, 3, 5)))

  constant_init <- strategize:::neural_build_output_site_init_values(
    Y = rep(2, 4),
    likelihood = "normal",
    nOutcomes = 1L
  )
  expect_equal(constant_init$b_out, 2)
  expect_equal(constant_init$tau_b, 2)
  expect_true(is.finite(constant_init$sigma))
  expect_gt(constant_init$sigma, 0)

  latent_init <- strategize:::neural_build_output_site_init_values(
    Y = c(-2, -2, -2),
    likelihood = "normal",
    nOutcomes = 1L,
    b_out_site_name = "b_out_z"
  )
  expect_equal(latent_init$tau_b, 2)
  expect_equal(latent_init$b_out_z * latent_init$tau_b, -2)
  expect_true(is.finite(latent_init$sigma))
  expect_gt(latent_init$sigma, 0)

  expect_equal(
    strategize:::neural_build_output_site_init_values(
      Y = c(0, 1),
      likelihood = "bernoulli",
      nOutcomes = 1L
    ),
    list()
  )
  expect_equal(
    strategize:::neural_build_output_site_init_values(
      Y = c(0, 1, 2),
      likelihood = "categorical",
      nOutcomes = 3L
    ),
    list()
  )
  expect_equal(
    strategize:::neural_build_output_site_init_values(
      Y = c(0, 1, 2),
      likelihood = "normal",
      nOutcomes = 2L
    ),
    list()
  )
})

test_that("special-token helper transforms stay centered and symmetric", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  feature_raw <- strategize:::strenv$jnp$array(
    matrix(c(2, 0,
             0, 4,
             -2, -4),
           nrow = 3L, byrow = TRUE),
    dtype = strategize:::strenv$jnp$float32
  )
  feature_centered <- strategize:::neural_center_token_rows(feature_raw)
  feature_mat <- reticulate::py_to_r(strategize:::strenv$np$array(feature_centered))
  expect_equal(colMeans(feature_mat), c(0, 0), tolerance = 1e-6)

  delta <- strategize:::strenv$jnp$array(c(2, -4), dtype = strategize:::strenv$jnp$float32)
  segment <- strategize:::neural_build_symmetric_segment_embeddings(delta)
  segment_mat <- reticulate::py_to_r(strategize:::strenv$np$array(segment))
  expect_equal(segment_mat[1, ] + segment_mat[2, ], c(0, 0), tolerance = 1e-6)
  expect_equal(segment_mat[2, ] - segment_mat[1, ], c(2, -4), tolerance = 1e-6)
})

test_that("covariate context tokens distinguish absent from present zero via presence masks", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels()
  covariate_family_idx <- match("covariate_fused", token_levels) - 1L
  family_mat <- matrix(0, nrow = length(token_levels), ncol = 4L)
  family_mat[covariate_family_idx + 1L, ] <- c(1000, 2000, 3000, 4000)
  metadata_dim <- length(strategize:::neural_covariate_value_metadata_names())
  model_info <- list(
    model_dims = 4L,
    token_family_levels = token_levels,
    covariate_names = "income",
    default_covariate_order = 0L,
    max_covariate_tokens = 1L,
    covariate_name_text = NULL,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "name_dist_moe"
  )
  params <- list(
    E_covariate_fused_base = strategize:::strenv$jnp$array(
      c(0, 0, 0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_missing = strategize:::strenv$jnp$array(
      c(10, 20, 30, 40),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_fuse_1 = strategize:::strenv$jnp$array(
      {
        w1 <- matrix(0, nrow = 8L + metadata_dim, ncol = 16L)
        w1[5:8, 9:12] <- diag(4L)
        w1
      },
      dtype = strategize:::strenv$jnp$float32
    ),
    b_covariate_fuse_1 = strategize:::strenv$jnp$array(
      c(rep(neural_test_swiglu_unit_gate(), 8L), rep(0, 8L)),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_fuse_2 = strategize:::strenv$jnp$array(
      rbind(diag(4L), matrix(0, nrow = 4L, ncol = 4L)),
      dtype = strategize:::strenv$jnp$float32
    ),
    b_covariate_fuse_2 = strategize:::strenv$jnp$zeros(list(4L), dtype = strategize:::strenv$jnp$float32),
    E_token_family = strategize:::strenv$jnp$array(
      family_mat,
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = NULL
  )

  tok_absent <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(0, nrow = 1L, ncol = 1L),
    resp_cov_present = matrix(0, nrow = 1L, ncol = 1L),
    params = params,
    batch = FALSE
  )
  tok_zero <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(0, nrow = 1L, ncol = 1L),
    resp_cov_present = matrix(1, nrow = 1L, ncol = 1L),
    params = params,
    batch = FALSE
  )

  tok_absent_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_absent))
  tok_zero_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_zero))
  diff <- drop(tok_absent_r - tok_zero_r)

  expect_false(isTRUE(all.equal(tok_absent_r, tok_zero_r)))
  expect_true(all(diff > 9))
})

test_that("runtime token model info carries experiment text semantics", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels()
  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    token_family_levels = token_levels,
    experiment_token_mode = "description",
    experiment_description_text = matrix(
      c(1, 0,
        0, 1),
      nrow = 2L,
      byrow = TRUE
    ),
    experiment_description_present = c(TRUE, TRUE)
  )

  params <- list(
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = NULL,
    W_experiment_text = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    )
  )

  expect_identical(model_info$experiment_token_mode, "description")

  tok_exp0 <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(0, nrow = 1L, ncol = 1L),
    resp_cov_present = matrix(0, nrow = 1L, ncol = 1L),
    experiment_idx = 0L,
    params = params,
    batch = FALSE
  )
  tok_exp1 <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(0, nrow = 1L, ncol = 1L),
    resp_cov_present = matrix(0, nrow = 1L, ncol = 1L),
    experiment_idx = 1L,
    params = params,
    batch = FALSE
  )

  tok_exp0_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_exp0))
  tok_exp1_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_exp1))
  expect_equal(drop(tok_exp0_r[1, 1, ]), c(1, 0), tolerance = 1e-6)
  expect_equal(drop(tok_exp1_r[1, 1, ]), c(0, 1), tolerance = 1e-6)
})

test_that("time context encoding matches foundation year basis", {
  emb_2025 <- strategize:::neural_encode_time_context(2025, present = TRUE)
  emb_missing <- strategize:::neural_encode_time_context(NA_real_, present = FALSE)

  expect_identical(names(emb_2025), strategize:::neural_time_feature_names())
  expect_equal(unname(emb_2025[["linear_year_2000_25"]]), 1, tolerance = 1e-8)
  expect_equal(
    unname(emb_2025[["sin_period_10"]]),
    sin(2 * pi * 25 / 10),
    tolerance = 1e-8
  )
  expect_equal(
    unname(emb_2025[["cos_period_10"]]),
    cos(2 * pi * 25 / 10),
    tolerance = 1e-8
  )
  expect_equal(unname(emb_2025[["missing_year"]]), 0, tolerance = 1e-8)
  expect_equal(unname(emb_missing[["missing_year"]]), 1, tolerance = 1e-8)
  expect_equal(sum(abs(emb_missing[names(emb_missing) != "missing_year"])), 0)
})

test_that("place context encoding matches foundation country basis", {
  canada <- strategize:::cs2step_normalize_country("Canada")
  emb_canada <- strategize:::neural_encode_place_context(
    canada$country_latitude,
    canada$country_longitude,
    present = TRUE
  )
  emb_missing <- strategize:::neural_encode_place_context(
    NA_real_,
    NA_real_,
    present = FALSE
  )

  expect_identical(names(emb_canada), strategize:::neural_place_feature_names())
  expect_equal(unname(emb_canada[["missing_country"]]), 0, tolerance = 1e-8)
  expect_equal(unname(emb_missing[["missing_country"]]), 1, tolerance = 1e-8)
  expect_equal(sum(abs(emb_missing[names(emb_missing) != "missing_country"])), 0)
  expect_true(any(abs(emb_canada[c("sphere_x", "sphere_y", "sphere_z")]) > 0))
})

test_that("place context token is emitted only when place context is enabled", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels(include_place = TRUE)
  model_enabled <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    token_family_levels = token_levels,
    place_context_enabled = TRUE,
    default_place_embedding = matrix(c(1, 2), nrow = 1L)
  )
  model_disabled <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    token_family_levels = token_levels,
    place_context_enabled = FALSE,
    default_place_embedding = matrix(c(1, 2), nrow = 1L)
  )
  params <- list(
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = NULL,
    W_place_context = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    )
  )

  tok_enabled <- strategize:::add_context_tokens(
    model_info = model_enabled,
    resp_party_idx = 0L,
    params = params,
    batch = FALSE
  )
  tok_disabled <- strategize:::add_context_tokens(
    model_info = model_disabled,
    resp_party_idx = 0L,
    params = params,
    batch = FALSE
  )

  expect_equal(dim(reticulate::py_to_r(strategize:::strenv$np$array(tok_enabled)))[2], 1L)
  expect_null(tok_disabled)
})

test_that("place context token changes with supplied country embedding", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels(include_place = TRUE)
  place_family_idx <- match("place", token_levels) - 1L
  family_mat <- matrix(0, nrow = length(token_levels), ncol = 2L)
  family_mat[place_family_idx + 1L, ] <- c(100, 200)
  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    token_family_levels = token_levels,
    place_context_enabled = TRUE,
    default_place_embedding = matrix(c(0, 0), nrow = 1L)
  )
  params <- list(
    E_token_family = strategize:::strenv$jnp$array(
      family_mat,
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = NULL,
    W_place_context = strategize:::strenv$jnp$array(
      matrix(c(1, 0, 0, 1), nrow = 2L),
      dtype = strategize:::strenv$jnp$float32
    )
  )
  emb_usa <- matrix(c(1, 2), nrow = 1L)
  emb_can <- matrix(c(3, 5), nrow = 1L)

  tok_usa <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    place_embedding = emb_usa,
    params = params,
    batch = FALSE
  )
  tok_can <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    place_embedding = emb_can,
    params = params,
    batch = FALSE
  )
  tok_usa_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_usa))
  tok_can_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_can))

  expect_equal(drop(tok_usa_r[1, 1, ]), c(101, 202), tolerance = 1e-6)
  expect_equal(drop(tok_can_r[1, 1, ]), c(103, 205), tolerance = 1e-6)
  expect_false(isTRUE(all.equal(tok_usa_r, tok_can_r)))
})

test_that("pack and upgrade preserve place context model-info fields", {
  model_info <- list(
    place_embedding = matrix(1:4, nrow = 2L),
    place_present = c(TRUE, FALSE),
    place_context_enabled = TRUE,
    place_feature_names = c("a", "b"),
    default_place_embedding = matrix(c(0, 1), nrow = 1L),
    default_place_present = FALSE,
    place_context_dim = 2L
  )
  packed <- strategize:::cs2step_neural_pack_model_info(model_info, drop_params = TRUE)
  upgraded <- strategize:::cs2step_neural_upgrade_model_info(packed)

  expect_equal(packed$place_embedding, model_info$place_embedding)
  expect_identical(packed$place_context_enabled, TRUE)
  expect_identical(upgraded$place_feature_names, c("a", "b"))
  expect_identical(upgraded$place_context_dim, 2L)
  expect_identical(upgraded$default_place_present, FALSE)
})

test_that("time context token changes with supplied year embedding", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels(include_time = TRUE)
  time_family_idx <- match("time", token_levels) - 1L
  family_mat <- matrix(0, nrow = length(token_levels), ncol = 2L)
  family_mat[time_family_idx + 1L, ] <- c(100, 200)
  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    token_family_levels = token_levels,
    time_context_enabled = TRUE,
    default_time_embedding = strategize:::neural_default_time_context_matrix()
  )
  params <- list(
    E_token_family = strategize:::strenv$jnp$array(
      family_mat,
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = NULL,
    W_time_context = strategize:::strenv$jnp$array(
      matrix(c(1, 0, 0, 1), nrow = 2L),
      dtype = strategize:::strenv$jnp$float32
    )
  )
  emb_2020 <- matrix(c(1, 2), nrow = 1L)
  emb_2025 <- matrix(c(3, 5), nrow = 1L)

  tok_2020 <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    time_embedding = emb_2020,
    params = params,
    batch = FALSE
  )
  tok_2025 <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    time_embedding = emb_2025,
    params = params,
    batch = FALSE
  )
  tok_2020_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_2020))
  tok_2025_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_2025))

  expect_equal(drop(tok_2020_r[1, 1, ]), c(101, 202), tolerance = 1e-6)
  expect_equal(drop(tok_2025_r[1, 1, ]), c(103, 205), tolerance = 1e-6)
  expect_false(isTRUE(all.equal(tok_2020_r, tok_2025_r)))
})

test_that("runtime covariate defaults are JAX-normalized for covariate transforms", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    covariate_names = c("alpha", "beta"),
    resp_cov_mean = c(1, 2),
    resp_cov_scale = c(2, 4),
    resp_cov_default_present = c(1, 1)
  )
  resp_cov_mat <- strategize:::strenv$jnp$array(
    matrix(
      c(3, 6,
        1, 2),
      nrow = 2L,
      byrow = TRUE
    ),
    dtype = strategize:::strenv$jnp$float32
  )
  resp_cov_present_mat <- strategize:::strenv$jnp$array(
    matrix(
      c(1, 1,
        1, 0),
      nrow = 2L,
      byrow = TRUE
    ),
    dtype = strategize:::strenv$jnp$float32
  )

  z <- strategize:::neural_covariate_z_scores(resp_cov_mat, model_info)
  basis <- strategize:::neural_covariate_basis(
    resp_cov_mat = resp_cov_mat,
    resp_cov_present_mat = resp_cov_present_mat,
    model_info = model_info
  )
  z_r <- reticulate::py_to_r(strategize:::strenv$np$array(z))
  basis_r <- reticulate::py_to_r(strategize:::strenv$np$array(basis))

  expect_true(strategize:::neural_has_shape(model_info$resp_cov_mean))
  expect_equal(dim(z_r), c(2L, 2L))
  expect_equal(dim(basis_r), c(2L, 2L, 3L))
  expect_true(all(is.finite(as.numeric(z_r))))
  expect_true(all(is.finite(as.numeric(basis_r))))
})

test_that("shared_projection emits ordered fused covariate tokens with padding masks", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels()
  cov_text <- matrix(
    c(1, 0,
      0, 1),
    nrow = 2L,
    byrow = TRUE
  )
  rownames(cov_text) <- c("alpha", "beta")
  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    covariate_names = c("alpha", "beta"),
    covariate_name_text = cov_text,
    resp_cov_mean = c(0, 0),
    resp_cov_scale = c(1, 1),
    default_covariate_order = c(1L, 0L),
    max_covariate_tokens = 3L,
    token_family_levels = token_levels,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "legacy_scalar"
  )

  params <- list(
    E_covariate_start = strategize:::strenv$jnp$array(
      c(1, 2),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_end = strategize:::strenv$jnp$array(
      c(3, 4),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_role = strategize:::strenv$jnp$array(
      matrix(0, nrow = 4L, ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_value_shared = strategize:::strenv$jnp$array(
      matrix(c(10, 20), nrow = 1L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    )
  )
  params <- neural_test_add_fused_covariate_value_params(
    params,
    model_dims = 2L,
    token_family_levels = token_levels
  )

  cov_info <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(c(4, 8), nrow = 1L),
    resp_cov_present = matrix(c(1, 1), nrow = 1L),
    params = params,
    batch = FALSE,
    return_mask = TRUE
  )

  tok_cov_r <- reticulate::py_to_r(strategize:::strenv$np$array(cov_info$tokens))
  mask_r <- reticulate::py_to_r(strategize:::strenv$np$array(cov_info$mask))

  expect_equal(dim(tok_cov_r), c(1L, 3L, 2L))
  expect_equal(as.numeric(mask_r[1, ]), c(1, 1, 0))
  expect_equal(drop(tok_cov_r[1, 1, ]), neural_test_swiglu_value(c(80, 160)), tolerance = 1e-6)
  expect_equal(drop(tok_cov_r[1, 2, ]), neural_test_swiglu_value(c(40, 80)), tolerance = 1e-6)
  expect_equal(drop(tok_cov_r[1, 3, ]), c(0, 0), tolerance = 1e-6)
})

test_that("shared_projection emits active fused tokens for missing-in-row covariates", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels()
  cov_text <- matrix(
    c(1, 0,
      0, 1),
    nrow = 2L,
    byrow = TRUE
  )
  rownames(cov_text) <- c("alpha", "beta")
  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    covariate_names = c("alpha", "beta"),
    covariate_name_text = cov_text,
    resp_cov_mean = c(0, 0),
    resp_cov_scale = c(1, 1),
    default_covariate_order = c(0L, 1L),
    max_covariate_tokens = 3L,
    token_family_levels = token_levels,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "legacy_scalar"
  )

  params <- list(
    E_covariate_start = strategize:::strenv$jnp$array(
      c(1, 2),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_end = strategize:::strenv$jnp$array(
      c(3, 4),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_role = strategize:::strenv$jnp$array(
      matrix(0, nrow = 4L, ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_missing = strategize:::strenv$jnp$array(
      c(7, 9),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_value_shared = strategize:::strenv$jnp$array(
      matrix(c(10, 20), nrow = 1L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    )
  )
  params <- neural_test_add_fused_covariate_value_params(
    params,
    model_dims = 2L,
    token_family_levels = token_levels
  )

  cov_info <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(c(4, 8), nrow = 1L),
    resp_cov_present = matrix(c(1, 0), nrow = 1L),
    params = params,
    batch = FALSE,
    return_mask = TRUE
  )
  tok_cov_r <- reticulate::py_to_r(strategize:::strenv$np$array(cov_info$tokens))
  mask_r <- reticulate::py_to_r(strategize:::strenv$np$array(cov_info$mask))

  expect_equal(as.numeric(mask_r[1, ]), c(1, 1, 0))
  expect_equal(drop(tok_cov_r[1, 1, ]), neural_test_swiglu_value(c(40, 80)), tolerance = 1e-6)
  expect_equal(drop(tok_cov_r[1, 2, ]), neural_test_swiglu_value(c(7, 9)), tolerance = 1e-6)
  expect_equal(drop(tok_cov_r[1, 3, ]), c(0, 0), tolerance = 1e-6)

  cov_absent <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(c(4, 8), nrow = 1L),
    resp_cov_present = matrix(c(1, 0), nrow = 1L),
    resp_cov_order = matrix(c(0L, -1L, -1L), nrow = 1L),
    params = params,
    batch = FALSE,
    return_mask = TRUE
  )
  absent_r <- reticulate::py_to_r(strategize:::strenv$np$array(cov_absent$tokens))
  absent_mask <- reticulate::py_to_r(strategize:::strenv$np$array(cov_absent$mask))
  expect_equal(as.numeric(absent_mask[1, ]), c(1, 0, 0))
  expect_equal(absent_r[1, 2:3, ], matrix(0, nrow = 2L, ncol = 2L), tolerance = 1e-6)
})

test_that("shared_projection uses categorical value text by value code", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels()
  value_text <- array(0, dim = c(3L, 3L, 2L))
  value_text[1L, 3L, ] <- c(5, 7)
  value_text[2L, 2L, ] <- c(1, 2)
  value_text[3L, 3L, ] <- c(99, 99)
  value_present <- matrix(0, nrow = 3L, ncol = 3L)
  value_present[1L, 3L] <- 1
  value_present[2L, 2L] <- 1
  value_present[3L, 3L] <- 1
  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    covariate_names = c("party", "ordered", "numeric"),
    covariate_name_text = matrix(0, nrow = 3L, ncol = 2L),
    resp_cov_mean = c(0, 0, 0),
    resp_cov_scale = c(1, 1, 1),
    default_covariate_order = c(0L, 1L, 2L),
    max_covariate_tokens = 3L,
    token_family_levels = token_levels,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "legacy_scalar",
    covariate_value_text = value_text,
    covariate_value_text_present = value_present,
    covariate_value_type = c(1L, 2L, 0L)
  )

  params <- list(
    E_covariate_start = strategize:::strenv$jnp$array(
      c(0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_end = strategize:::strenv$jnp$array(
      c(0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_role = strategize:::strenv$jnp$array(
      matrix(0, nrow = 4L, ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_missing = strategize:::strenv$jnp$array(
      c(0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_value_shared = strategize:::strenv$jnp$array(
      matrix(c(10, 20), nrow = 1L),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_value_text = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = strategize:::strenv$jnp$array(
      matrix(0, nrow = 2L, ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    )
  )
  params <- neural_test_add_fused_covariate_value_params(
    params,
    model_dims = 2L,
    token_family_levels = token_levels
  )

  cov_info <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(c(2, 1, 2), nrow = 1L),
    resp_cov_present = matrix(c(1, 1, 1), nrow = 1L),
    params = params,
    batch = FALSE,
    return_mask = TRUE
  )
  tok_cov_r <- reticulate::py_to_r(strategize:::strenv$np$array(cov_info$tokens))

  expect_equal(drop(tok_cov_r[1, 1, ]), neural_test_swiglu_value(c(5, 7)), tolerance = 1e-6)
  expect_equal(drop(tok_cov_r[1, 2, ]), neural_test_swiglu_value(c(11, 22)), tolerance = 1e-6)
  expect_equal(drop(tok_cov_r[1, 3, ]), neural_test_swiglu_value(c(20, 40)), tolerance = 1e-6)

  cov_fallback <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = matrix(c(1, 1, 2), nrow = 1L),
    resp_cov_present = matrix(c(1, 1, 1), nrow = 1L),
    params = params,
    batch = FALSE,
    return_mask = TRUE
  )
  fallback_r <- reticulate::py_to_r(strategize:::strenv$np$array(cov_fallback$tokens))
  expect_equal(drop(fallback_r[1, 1, ]), neural_test_swiglu_value(c(10, 20)), tolerance = 1e-6)
})

test_that("covariate distribution profiles summarize local metadata", {
  profiles <- strategize:::neural_build_covariate_distribution_profiles(
    X_mat = cbind(
      cont = c(-2, -1, 0, 1, 2, 3),
      binary = c(0, 1, 0, 1, 0, 1),
      constant = rep(5, 6)
    ),
    X_present_mat = cbind(
      cont = c(1, 1, 1, 1, 1, 1),
      binary = c(1, 1, 1, 1, 0, 1),
      constant = c(1, 1, 1, 1, 1, 0)
    ),
    experiment_index = c(0L, 0L, 0L, 1L, 1L, 1L),
    covariate_names = c("cont", "binary", "constant"),
    default_experiment_index = 1L
  )

  expect_named(
    profiles,
    c("by_experiment", "metadata_by_experiment", "default_stats", "default_metadata")
  )
  expect_equal(length(profiles$by_experiment), 2L)
  expect_equal(dim(profiles$by_experiment[[1]]), c(3L, 9L))
  expect_equal(dim(profiles$metadata_by_experiment[[1]]), c(3L, 15L))
  expect_true(profiles$metadata_by_experiment[[1]]["binary", "binary_like"] > 0.5)
  expect_true(profiles$metadata_by_experiment[[1]]["binary", "integer_like"] > 0.5)
  expect_true(profiles$metadata_by_experiment[[2]]["constant", "sd_log"] < 1e-6)
  expect_equal(
    profiles$default_stats[, "mean"],
    profiles$by_experiment[[2]][, "mean"],
    tolerance = 1e-8
  )
})

test_that("shared_projection name_dist_moe conditions value token on metadata and name", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels()
  cov_text <- matrix(
    c(1, 0,
      0, 1),
    nrow = 2L,
    byrow = TRUE
  )
  rownames(cov_text) <- c("alpha", "beta")
  stats_mat <- matrix(
    c(
      0, 1, -2, -1, -0.5, 0, 0.5, 1, 2,
      1, 2, -1, 0, 0.5, 1, 1.5, 2, 3
    ),
    nrow = 2L,
    byrow = TRUE
  )
  colnames(stats_mat) <- strategize:::neural_covariate_value_stat_names()
  rownames(stats_mat) <- c("alpha", "beta")
  meta_a <- matrix(0, nrow = 2L, ncol = length(strategize:::neural_covariate_value_metadata_names()))
  meta_b <- meta_a
  colnames(meta_a) <- strategize:::neural_covariate_value_metadata_names()
  colnames(meta_b) <- strategize:::neural_covariate_value_metadata_names()
  rownames(meta_a) <- c("alpha", "beta")
  rownames(meta_b) <- c("alpha", "beta")
  meta_a[, "missing_rate"] <- c(0.10, 0.20)
  meta_a[, "unique_ratio"] <- c(0.80, 0.60)
  meta_a[, "mean_signlog"] <- c(0.00, 0.70)
  meta_b <- meta_a
  meta_b[, "missing_rate"] <- c(0.55, 0.75)
  meta_b[, "unique_ratio"] <- c(0.35, 0.25)

  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    covariate_names = c("alpha", "beta"),
    covariate_name_text = cov_text,
    resp_cov_mean = c(0, 1),
    resp_cov_scale = c(1, 2),
    default_covariate_order = c(0L, 1L),
    max_covariate_tokens = 2L,
    token_family_levels = token_levels,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "name_dist_moe",
    default_covariate_value_stats = stats_mat,
    default_covariate_value_metadata = meta_a,
    covariate_value_metadata_by_experiment = list(meta_a, meta_b),
    covariate_value_stats_by_experiment = list(stats_mat, stats_mat)
  )

  params <- list(
    E_covariate_start = strategize:::strenv$jnp$array(
      c(0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_end = strategize:::strenv$jnp$array(
      c(0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_covariate_role = strategize:::strenv$jnp$array(
      matrix(0, nrow = 4L, ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_value_basis = strategize:::strenv$jnp$array(
      array(
        c(
          1, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0.5, 0,
          0, 1,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0.25,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0.1, 0.1,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0, 0,
          0.2, 0.2
        ),
        dim = c(4L, 7L, 2L)
      ),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_value_conditioner_1 = strategize:::strenv$jnp$array(
      matrix(
        c(
          1, 0,
          0, 1,
          0.4, 0.0,
          0.0, 0.3,
          0.2, 0.0,
          0.0, 0.2,
          rep(0, 22)
        ),
        nrow = 17L,
        byrow = TRUE
      ),
      dtype = strategize:::strenv$jnp$float32
    ),
    b_covariate_value_conditioner_1 = strategize:::strenv$jnp$array(
      c(0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_covariate_value_conditioner_2 = strategize:::strenv$jnp$array(
      matrix(
        c(
          1.0, 0.1, -0.2, 0.0,
          0.0, 0.8, 0.2, -0.1
        ),
        nrow = 2L,
        byrow = TRUE
      ),
      dtype = strategize:::strenv$jnp$float32
    ),
    b_covariate_value_conditioner_2 = strategize:::strenv$jnp$array(
      c(0, 0, 0, 0),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_experiment = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_covariate_name_text = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    )
  )
  params <- neural_test_add_fused_covariate_value_params(
    params,
    model_dims = 2L,
    token_family_levels = token_levels
  )

  build_tokens <- function(info, experiment_idx = NULL) {
    strategize:::add_context_tokens(
      model_info = info,
      resp_party_idx = 0L,
      resp_cov = matrix(c(1, 2), nrow = 1L),
      resp_cov_present = matrix(c(1, 1), nrow = 1L),
      experiment_idx = experiment_idx,
      params = params,
      batch = FALSE,
      return_mask = TRUE
    )
  }

  tok_default <- reticulate::py_to_r(strategize:::strenv$np$array(build_tokens(model_info)$tokens))
  tok_exp1 <- reticulate::py_to_r(strategize:::strenv$np$array(build_tokens(model_info, experiment_idx = 1L)$tokens))
  tok_fallback <- strategize:::add_context_tokens(
    model_info = model_info,
    resp_party_idx = 0L,
    resp_cov = NULL,
    resp_cov_present = NULL,
    experiment_idx = 1L,
    params = params,
    batch = FALSE,
    return_mask = TRUE
  )
  tok_fallback_r <- reticulate::py_to_r(strategize:::strenv$np$array(tok_fallback$tokens))

  cov_text_beta <- cov_text
  cov_text_beta[2, ] <- c(2, 0)
  model_info_name <- model_info
  model_info_name$covariate_name_text <- cov_text_beta
  tok_name <- reticulate::py_to_r(strategize:::strenv$np$array(build_tokens(model_info_name)$tokens))

  expect_equal(dim(tok_default), c(1L, 2L, 2L))
  expect_equal(dim(tok_fallback_r), c(1L, 2L, 2L))
  expect_true(all(is.finite(as.numeric(tok_fallback_r))))
  expect_false(isTRUE(all.equal(
    drop(tok_default[1, 1, ]),
    drop(tok_exp1[1, 1, ]),
    tolerance = 1e-6
  )))
  expect_false(isTRUE(all.equal(
    drop(tok_default[1, 2, ]),
    drop(tok_name[1, 2, ]),
    tolerance = 1e-6
  )))
  expect_equal(
    reticulate::py_to_r(strategize:::strenv$np$array(build_tokens(model_info)$mask)),
    reticulate::py_to_r(strategize:::strenv$np$array(build_tokens(model_info_name)$mask))
  )
})

test_that("fused factor tokenization emits ordered factor/value tokens with padding masks", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels()
  factor_text <- matrix(
    c(1, 0,
      0, 1,
      1, 1),
    nrow = 3L,
    byrow = TRUE
  )
  rownames(factor_text) <- c("alpha", "beta", "gamma")
  level_text <- list(
    alpha = matrix(c(10, 0,
                     20, 0,
                     0, 0), nrow = 3L, byrow = TRUE),
    beta = matrix(c(0, 10,
                    0, 20,
                    0, 0), nrow = 3L, byrow = TRUE),
    gamma = matrix(c(5, 5,
                     6, 6,
                     0, 0), nrow = 3L, byrow = TRUE)
  )
  factor_struct <- matrix(
    c(0.1, 0.0,
      0.0, 0.2,
      0.3, 0.4),
    nrow = 3L,
    byrow = TRUE,
    dimnames = list(c("alpha", "beta", "gamma"), c("s1", "s2"))
  )
  level_struct <- list(
    alpha = matrix(c(0.0, 0.0,
                     0.7, 0.8,
                     0.0, 0.0), nrow = 3L, byrow = TRUE),
    beta = matrix(c(0.2, 0.0,
                    0.0, 0.2,
                    0.0, 0.0), nrow = 3L, byrow = TRUE),
    gamma = matrix(c(0.5, 0.6,
                     0.0, 0.0,
                     0.0, 0.0), nrow = 3L, byrow = TRUE)
  )

  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    factor_name_text = factor_text,
    level_name_text = level_text,
    factor_struct_matrix = factor_struct,
    level_struct_matrices = level_struct,
    factor_struct_feature_names = colnames(factor_struct),
    level_struct_feature_names = c("s1", "s2"),
    default_factor_order = c(2L, 0L),
    factor_tokenization = "fused",
    max_factor_tokens = 3L,
    token_family_levels = token_levels,
    likelihood = "bernoulli"
  )

  params <- list(
    E_factor_fused_base = strategize:::strenv$jnp$array(
      c(1, 2),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_factor_fuse_1 = strategize:::strenv$jnp$array(
      cbind(
        matrix(0, nrow = 8L, ncol = 4L),
        rbind(
          diag(2L),
          matrix(0, nrow = 2L, ncol = 2L),
          diag(2L),
          matrix(0, nrow = 2L, ncol = 2L)
        ),
        matrix(0, nrow = 8L, ncol = 2L)
      ),
      dtype = strategize:::strenv$jnp$float32
    ),
    b_factor_fuse_1 = strategize:::strenv$jnp$array(
      c(rep(neural_test_swiglu_unit_gate(), 4L), rep(0, 4L)),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_factor_fuse_2 = strategize:::strenv$jnp$array(
      rbind(diag(2L), matrix(0, nrow = 2L, ncol = 2L)),
      dtype = strategize:::strenv$jnp$float32
    ),
    b_factor_fuse_2 = strategize:::strenv$jnp$zeros(list(2L), dtype = strategize:::strenv$jnp$float32),
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_party = NULL,
    E_rel = NULL,
    E_resp_party = NULL,
    E_experiment = NULL,
    E_stage = NULL,
    E_matchup = NULL,
    W_factor_name_text = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_level_name_text = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_factor_struct = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    ),
    W_level_struct = strategize:::strenv$jnp$array(
      diag(2),
      dtype = strategize:::strenv$jnp$float32
    )
  )

  cand_info <- strategize:::neural_build_candidate_tokens_hard(
    X_idx = strategize:::strenv$jnp$array(matrix(c(1L, 0L, 0L), nrow = 1L))$astype(strategize:::strenv$jnp$int32),
    party_idx = strategize:::strenv$jnp$array(0L)$astype(strategize:::strenv$jnp$int32),
    model_info = model_info,
    params = params,
    return_mask = TRUE
  )

  tok_r <- reticulate::py_to_r(strategize:::strenv$np$array(cand_info$tokens))
  mask_r <- reticulate::py_to_r(strategize:::strenv$np$array(cand_info$mask))

  expect_equal(dim(tok_r), c(1L, 3L, 2L))
  expect_equal(as.numeric(mask_r[1, ]), c(1, 1, 0))
  expect_equal(drop(tok_r[1, 1, ]), c(7.4, 8.5), tolerance = 1e-4)
  expect_equal(drop(tok_r[1, 2, ]), c(22.4, 2.4), tolerance = 1e-4)
  expect_equal(drop(tok_r[1, 3, ]), c(0, 0), tolerance = 1e-6)
  expect_true(isTRUE(model_info$factor_schema_supplied))

  model_info_normal <- model_info
  model_info_normal$likelihood <- "normal"
  cand_info_normal <- strategize:::neural_build_candidate_tokens_hard(
    X_idx = strategize:::strenv$jnp$array(matrix(c(1L, 0L, 0L), nrow = 1L))$astype(strategize:::strenv$jnp$int32),
    party_idx = strategize:::strenv$jnp$array(0L)$astype(strategize:::strenv$jnp$int32),
    model_info = model_info_normal,
    params = params,
    return_mask = TRUE
  )
  tok_normal <- reticulate::py_to_r(strategize:::strenv$np$array(cand_info_normal$tokens))
  expect_equal(drop(tok_normal[1, 1, ]), c(7, 8), tolerance = 1e-4)
  expect_equal(drop(tok_normal[1, 2, ]), c(22, 2), tolerance = 1e-4)

  prepared_info <- strategize:::neural_make_prepared_prediction_model_info(
    model_depth = 1L,
    model_dims = 2L,
    n_heads = 1L,
    head_dim = 2L,
    n_candidate_tokens = 3L,
    factor_tokenization = "fused",
    max_factor_tokens = 3L,
    factor_name_text = factor_text,
    level_name_text = level_text,
    factor_order_by_experiment = list(c(2L, 0L)),
    default_factor_order = c(2L, 0L),
    factor_struct_matrix = factor_struct,
    level_struct_matrices = level_struct,
    factor_struct_feature_names = colnames(factor_struct),
    level_struct_feature_names = c("s1", "s2")
  )
  prepared_info$token_family_levels <- token_levels
  expect_true(isTRUE(prepared_info$factor_schema_supplied))

  prepared_missing_flag <- prepared_info
  prepared_missing_flag$factor_schema_supplied <- NULL
  expect_true(strategize:::neural_model_info_factor_schema_supplied(prepared_missing_flag))
  upgraded_info <- strategize:::cs2step_neural_upgrade_model_info(prepared_missing_flag)
  expect_true(isTRUE(upgraded_info$factor_schema_supplied))

  cand_info_missing_flag <- strategize:::neural_build_candidate_tokens_hard(
    X_idx = strategize:::strenv$jnp$array(matrix(c(1L, 0L, 0L), nrow = 1L))$astype(strategize:::strenv$jnp$int32),
    party_idx = strategize:::strenv$jnp$array(0L)$astype(strategize:::strenv$jnp$int32),
    model_info = prepared_missing_flag,
    params = params,
    return_mask = TRUE
  )
  mask_missing_flag <- reticulate::py_to_r(strategize:::strenv$np$array(cand_info_missing_flag$mask))
  expect_equal(as.numeric(mask_missing_flag[1, ]), c(1, 1, 0))
})

test_that("default fused structural metadata distinguishes same-cardinality factors", {
  struct <- strategize:::neural_make_default_fused_structural_info(
    factor_names = c("price", "message", "brand"),
    factor_levels = c(2L, 2L, 2L)
  )

  factor_mat <- struct$factor_struct_matrix
  identity_cols <- grep("^factor_identity_", colnames(factor_mat), value = TRUE)

  expect_length(identity_cols, 3L)
  expect_equal(unname(factor_mat[, identity_cols, drop = FALSE]), diag(3L))
  expect_false(isTRUE(all.equal(
    unname(factor_mat["price", ]),
    unname(factor_mat["message", ])
  )))
})

test_that("pairwise context tokens are masked for context-absent rows", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  token_levels <- strategize:::neural_token_family_levels(
    include_candidate_group = TRUE,
    include_relation = TRUE,
    include_stage = TRUE,
    include_respondent_group = TRUE,
    include_matchup = TRUE
  )
  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    cand_party_to_resp_idx = c(0L, 1L, -1L),
    n_party_levels = 3L,
    token_family_levels = token_levels
  )
  model_info <- strategize:::neural_set_pairwise_context_model_info(
    info = model_info,
    pairwise_context_mode = "stage_aware",
    has_candidate_group_context = TRUE,
    has_respondent_group_context = TRUE,
    has_relation_token_context = TRUE,
    has_stage_context = TRUE,
    has_matchup_context = TRUE,
    n_resp_party_levels = 3L,
    party_missing_index = 2L,
    resp_party_missing_index = 2L
  )
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 2L
  )
  params <- list(
    E_factor_1 = strategize:::strenv$jnp$array(
      matrix(c(1, 2, 3, 4), nrow = 2L, byrow = TRUE),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_party = strategize:::strenv$jnp$array(
      matrix(c(10, 11, 12, 13, 14, 15), nrow = 3L, byrow = TRUE),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_rel = strategize:::strenv$jnp$array(
      matrix(c(20, 21, 22, 23, 24, 25), nrow = 3L, byrow = TRUE),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_resp_party = strategize:::strenv$jnp$array(
      matrix(c(30, 31, 32, 33, 34, 35), nrow = 3L, byrow = TRUE),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_stage = strategize:::strenv$jnp$array(
      array(seq_len(12), dim = c(3L, 2L, 2L)),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_matchup = strategize:::strenv$jnp$array(
      matrix(seq_len(12), nrow = 6L),
      dtype = strategize:::strenv$jnp$float32
    ),
    E_token_family = strategize:::strenv$jnp$array(
      matrix(0, nrow = length(token_levels), ncol = 2L),
      dtype = strategize:::strenv$jnp$float32
    )
  )
  params <- neural_test_add_fused_factor_params(
    params,
    model_dims = 2L,
    model_info = model_info,
    token_family_levels = token_levels
  )

  cand_info <- strategize:::neural_build_candidate_tokens_hard(
    X_idx = strategize:::strenv$jnp$array(matrix(c(0L, 1L), ncol = 1L))$astype(strategize:::strenv$jnp$int32),
    party_idx = strategize:::strenv$jnp$array(c(0L, 2L))$astype(strategize:::strenv$jnp$int32),
    resp_party_idx = strategize:::strenv$jnp$array(c(0L, 2L))$astype(strategize:::strenv$jnp$int32),
    context_present = strategize:::strenv$jnp$array(c(1L, 0L)),
    model_info = model_info,
    params = params,
    return_mask = TRUE
  )
  cand_tokens <- reticulate::py_to_r(strategize:::strenv$np$array(cand_info$tokens))
  cand_mask <- reticulate::py_to_r(strategize:::strenv$np$array(cand_info$mask))
  expect_equal(cand_mask, matrix(c(1, 1, 1, 1, 0, 0), nrow = 2L, byrow = TRUE))
  expect_equal(cand_tokens[2, 2:3, ], matrix(0, nrow = 2L, ncol = 2L), tolerance = 1e-6)

  ctx_info <- strategize:::neural_build_context_tokens_batch(
    model_info = model_info,
    resp_party_idx = strategize:::strenv$jnp$array(c(0L, 2L))$astype(strategize:::strenv$jnp$int32),
    stage_idx = strategize:::strenv$jnp$array(c(1L, 0L))$astype(strategize:::strenv$jnp$int32),
    matchup_idx = strategize:::strenv$jnp$array(c(0L, 5L))$astype(strategize:::strenv$jnp$int32),
    context_present = strategize:::strenv$jnp$array(c(1L, 0L)),
    params = params,
    return_mask = TRUE
  )
  ctx_tokens <- reticulate::py_to_r(strategize:::strenv$np$array(ctx_info$tokens))
  ctx_mask <- reticulate::py_to_r(strategize:::strenv$np$array(ctx_info$mask))
  expect_equal(ctx_mask, matrix(c(1, 1, 1, 0, 0, 0), nrow = 2L, byrow = TRUE))
  expect_equal(ctx_tokens[2, , ], matrix(0, nrow = 3L, ncol = 2L), tolerance = 1e-6)
})

test_that("stage diagnostics ignore context-absent pairwise rows", {
  skip_on_cran()
  skip_if_no_jax()
  withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

  W <- data.frame(feature = rep(c("A", "B"), 4L), stringsAsFactors = FALSE)
  names_list <- strategize:::cs2step_build_names_list(W)
  W_idx <- strategize:::cs2step_encode_W_indices(
    W,
    names_list = names_list,
    unknown = "error",
    pad_unknown = 0L
  )
  fit <- suppressWarnings(strategize:::cs2step_eval_outcome_model_neural(
    Y = c(1, 0, 1, 0, 1, 0, 1, 0),
    W_idx = W_idx,
    names_list = names_list,
    factor_levels = vapply(names_list, function(x) length(x[[1]]), integer(1)),
    diff = TRUE,
    pair_id = rep(seq_len(4L), each = 2L),
    profile_order = rep(1:2, 4L),
    competing_group_variable_candidate = c("A", "A", "A", "B", NA, NA, "A", "B"),
    competing_group_variable_respondent = c("A", "A", "A", "A", NA, NA, NA, NA),
    conda_env_required = TRUE,
    neural_mcmc_control = list(
      ModelDims = 8L,
      ModelDepth = 1L,
      subsample_method = "batch_vi",
      optimizer = "adam",
      svi_steps = 4L,
      svi_num_draws = 1L,
      batch_size = 4L,
      early_stopping = FALSE,
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  ))
  stage <- fit$neural_model_info$stage_diagnostics

  expect_identical(stage$stage_context_policy, "masked_optional")
  expect_identical(stage$n_context_present_pairs, 2L)
  expect_identical(stage$n_context_absent_pairs, 2L)
  expect_identical(stage$n_primary, 1L)
  expect_identical(stage$n_general, 1L)
  expect_equal(stage$pct_primary, 0.5)
  expect_true(isTRUE(stage$stage_context_enabled))
})

test_that("fused factor tokenization requires structural factor and level token info", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  expect_error(
    strategize:::neural_make_runtime_token_model_info(
      model_dims = 2L,
      factor_name_text = matrix(c(1, 0, 0, 1), nrow = 2L, byrow = TRUE),
      level_name_text = list(
        matrix(c(1, 0, 0, 1, 0, 0), nrow = 3L, byrow = TRUE),
        matrix(c(0, 1, 1, 0, 0, 0), nrow = 3L, byrow = TRUE)
      ),
      factor_tokenization = "fused",
      max_factor_tokens = 2L
    ),
    "requires structural token_info fields"
  )
  expect_error(
    strategize:::neural_make_prepared_prediction_model_info(
      model_depth = 1L,
      model_dims = 2L,
      n_heads = 1L,
      head_dim = 2L,
      factor_name_text = matrix(c(1, 0, 0, 1), nrow = 2L, byrow = TRUE),
      level_name_text = list(
        matrix(c(1, 0, 0, 1, 0, 0), nrow = 3L, byrow = TRUE),
        matrix(c(0, 1, 1, 0, 0, 0), nrow = 3L, byrow = TRUE)
      ),
      factor_tokenization = "fused",
      max_factor_tokens = 2L
    ),
    "requires structural token_info fields"
  )
})

test_that("default term-mode neural predictor returns valid pairwise probabilities", {
  fit <- get_neural_fit()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  expect_valid_strategize_output(res, n_factors = ncol(data$W))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  X_left <- W_numeric[idx_left, , drop = FALSE]
  X_right <- W_numeric[idx_right, , drop = FALSE]
  p_lr <- as.numeric(model(X_left_new = X_left, X_right_new = X_right))
  p_rl <- as.numeric(model(X_left_new = X_right, X_right_new = X_left))

  # The default pairwise neural mode uses the opponent-dependent "term"
  # interaction, so swap-complementarity is not a valid invariant here.
  expect_length(p_lr, nrow(X_left))
  expect_length(p_rl, nrow(X_left))
  expect_true(all(is.finite(p_lr)))
  expect_true(all(is.finite(p_rl)))
  expect_true(all(p_lr >= 0 & p_lr <= 1))
  expect_true(all(p_rl >= 0 & p_rl <= 1))
})

test_that("neural attn predictor remains antisymmetric", {
  fit <- get_neural_fit_attn()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  expect_valid_strategize_output(res, n_factors = ncol(data$W))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  X_left <- W_numeric[idx_left, , drop = FALSE]
  X_right <- W_numeric[idx_right, , drop = FALSE]
  p_lr <- as.numeric(model(X_left_new = X_left, X_right_new = X_right))
  p_rl <- as.numeric(model(X_left_new = X_right, X_right_new = X_left))
  expect_equal(p_lr + p_rl, rep(1, length(p_lr)), tolerance = 1e-4)
})

test_that("default pairwise neural metadata uses term interactions", {
  fit <- get_neural_fit()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "term")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_true(isTRUE(model_info$has_cross_term))
  expect_false(isTRUE(model_info$has_cross_attn))
})

test_that("explicit none override disables the pairwise interaction default", {
  fit <- get_neural_fit_none()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "none")
  expect_false(isTRUE(model_info$cross_candidate_encoder))
  expect_false(isTRUE(model_info$has_cross_term))
  expect_false(isTRUE(model_info$has_cross_attn))
})

test_that("logical TRUE override normalizes to the term interaction mode", {
  fit <- get_neural_fit_true()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "term")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_true(isTRUE(model_info$has_cross_term))
  expect_false(isTRUE(model_info$has_cross_attn))
})

run_low_rank_pairwise_fit <- function(cross_candidate_encoder = NULL) {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "1"
  ))

  data <- generate_test_data(n = 24, seed = 20260411)
  params <- default_strategize_params(fast = TRUE)
  params$outcome_model_type <- "neural"
  control <- modifyList(
    params$neural_mcmc_control,
    list(
      subsample_method = "batch_vi",
      batch_size = 16L,
      ModelDims = 8L,
      ModelDepth = 1L,
      low_rank_interaction_rank = 2L,
      optimizer = "adam",
      svi_steps = 4L,
      svi_num_draws = 1L,
      eval_enabled = FALSE,
      early_stopping = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )
  if (!is.null(cross_candidate_encoder)) {
    control$cross_candidate_encoder <- cross_candidate_encoder
  }
  params$neural_mcmc_control <- control
  p_list <- generate_test_p_list(data$W)

  suppressMessages(suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))))
}

test_that("low-rank pairwise default disables the implicit cross term", {
  res <- run_low_rank_pairwise_fit()
  model_info <- get_neural_model_info(res)

  expect_false(is.null(model_info))
  expect_identical(model_info$low_rank_interaction_rank, 2L)
  expect_identical(model_info$cross_candidate_encoder_mode, "none")
  expect_false(isTRUE(model_info$cross_candidate_encoder))
  expect_false(isTRUE(model_info$has_cross_term))
  expect_identical(model_info$low_rank_logit_transform, "none")
  expect_null(model_info$low_rank_logit_bound)
  expect_null(model_info$low_rank_logit_softness)
  expect_identical(model_info$low_rank_logit_normalization, "rms")
  expect_equal(model_info$low_rank_head_weight_target_rms, 1 / (sqrt(2) * 8))
  expect_equal(model_info$low_rank_rc_out_target_rms, 1 / (sqrt(2) * 2))
  expect_identical(model_info$additive_utility_mode, "on")
  expect_identical(model_info$additive_utility_normalization, "rms")
  expect_equal(model_info$additive_head_weight_target_rms, 1 / sqrt(8))
  expect_true(isTRUE(model_info$has_additive_utility))
})

test_that("explicit term override remains honored with low-rank pairwise interaction", {
  res <- run_low_rank_pairwise_fit(cross_candidate_encoder = "term")
  model_info <- get_neural_model_info(res)

  expect_false(is.null(model_info))
  expect_identical(model_info$low_rank_interaction_rank, 2L)
  expect_identical(model_info$cross_candidate_encoder_mode, "term")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_true(isTRUE(model_info$has_cross_term))
  expect_match(model_info$cross_candidate_encoder_note, "low_rank_interaction_rank")
})

test_that("learned pairwise Bernoulli scale yields non-degenerate prior logits", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "1"
  ))
  withr::local_seed(20260605)

  data <- generate_pairwise_performance_test_data(
    n_pairs = 24L,
    n_factors = 3L,
    n_levels = 2L,
    seed = 20260605
  )
  W <- as.data.frame(data$W, stringsAsFactors = FALSE)
  names_list <- strategize:::cs2step_build_names_list(W)
  W_idx <- strategize:::cs2step_encode_W_indices(
    W,
    names_list = names_list,
    unknown = "error",
    pad_unknown = 0L
  )
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))

  fit <- suppressMessages(suppressWarnings(strategize:::cs2step_eval_outcome_model_neural(
    Y = data$Y,
    W_idx = W_idx,
    names_list = names_list,
    factor_levels = factor_levels,
    diff = TRUE,
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    conda_env_required = TRUE,
    neural_mcmc_control = list(
      ModelDims = 8L,
      ModelDepth = 1L,
      subsample_method = "batch_vi",
      uncertainty_scope = "all",
      optimizer = "adam",
      batch_size = 16L,
      svi_steps = 4L,
      svi_num_draws = 1L,
      low_rank_interaction_rank = 0L,
      cross_candidate_encoder = "none",
      learned_pairwise_bernoulli_logit_scale = TRUE,
      pairwise_bernoulli_logit_scale_prior_sd = 0.5,
      eval_enabled = FALSE,
      early_stopping = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )))

  model_info <- fit$neural_model_info
  expect_true(isTRUE(model_info$learned_pairwise_bernoulli_logit_scale))
  expect_equal(model_info$pairwise_bernoulli_logit_scale_prior_sd, 0.5)
  expect_true("log_pairwise_bernoulli_logit_scale" %in% model_info$param_names)
  expect_false(is.null(model_info$params$log_pairwise_bernoulli_logit_scale))

  model_env <- environment(fit$my_model)
  strenv <- get("strenv", envir = model_env)
  model_fn <- get("BayesianPairTransformerModel", envir = model_env)
  to_index_matrix <- get("to_index_matrix", envir = model_env)

  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  n_obs <- length(idx_left)
  X_left_jnp <- strenv$jnp$array(
    to_index_matrix(W_idx[idx_left, , drop = FALSE])
  )$astype(strenv$jnp$int32)
  X_right_jnp <- strenv$jnp$array(
    to_index_matrix(W_idx[idx_right, , drop = FALSE])
  )$astype(strenv$jnp$int32)
  party_left_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
  party_right_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
  resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
  resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)

  get_trace_site <- function(trace, name) {
    tryCatch(trace[[name]], error = function(e) NULL)
  }
  to_numeric <- function(x) {
    if (is.null(x)) {
      return(numeric(0))
    }
    out <- tryCatch(
      reticulate::py_to_r(strenv$np$asarray(x)),
      error = function(e) NULL
    )
    if (is.null(out)) {
      out <- tryCatch(
        reticulate::py_to_r(strenv$jax$device_get(x)),
        error = function(e) NULL
      )
    }
    if (is.null(out)) {
      out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
    }
    if (is.list(out)) {
      out <- unlist(out, use.names = FALSE)
    }
    if (!is.numeric(out)) {
      return(numeric(0))
    }
    as.numeric(out)
  }

  logit_samples <- numeric(0)
  for (i in seq_len(20L)) {
    rng_key <- strenv$jax$random$PRNGKey(as.integer(20260605L + i))
    tracer <- strenv$numpyro$handlers$trace(
      strenv$numpyro$handlers$seed(model_fn, rng_key)
    )
    trace <- tracer$get_trace(
      X_left = X_left_jnp,
      X_right = X_right_jnp,
      party_left = party_left_jnp,
      party_right = party_right_jnp,
      resp_party = resp_party_jnp,
      resp_cov = resp_cov_jnp,
      Y_obs = NULL
    )

    expect_false(is.null(get_trace_site(trace, "log_pairwise_bernoulli_logit_scale")))
    obs_site <- get_trace_site(trace, "obs_pair")
    expect_false(is.null(obs_site))
    logits <- tryCatch(obs_site$fn$logits, error = function(e) NULL)
    expect_false(is.null(logits))
    logit_samples <- c(logit_samples, to_numeric(logits))
  }

  logit_samples <- logit_samples[is.finite(logit_samples)]
  expect_true(length(logit_samples) > 0L)
  sd_logits <- stats::sd(logit_samples)
  expect_true(is.finite(sd_logits))
  expect_true(
    sd_logits >= 0.05,
    info = sprintf("Prior predictive pairwise Bernoulli logit SD %.4f below 0.05", sd_logits)
  )
})

test_that("low-rank RMS logit normalization resolves defaults and column scales", {
  expect_identical(
    strategize:::neural_resolve_low_rank_logit_normalization(
      low_rank_interaction_rank = 2L,
      pairwise_mode = TRUE,
      likelihood = "bernoulli"
    ),
    "rms"
  )
  expect_identical(
    strategize:::neural_resolve_low_rank_logit_normalization(
      low_rank_interaction_rank = 0L,
      pairwise_mode = TRUE,
      likelihood = "bernoulli"
    ),
    "none"
  )
  expect_identical(
    strategize:::neural_resolve_low_rank_logit_normalization(
      low_rank_interaction_rank = 2L,
      pairwise_mode = TRUE,
      likelihood = "mixed"
    ),
    "rms"
  )
  expect_identical(
    strategize:::neural_resolve_low_rank_logit_normalization(
      value = "none",
      supplied = TRUE,
      low_rank_interaction_rank = 2L,
      pairwise_mode = TRUE,
      likelihood = "bernoulli"
    ),
    "none"
  )
  expect_true(strategize:::neural_low_rank_logit_normalization_enabled(
    list(
      likelihood = "mixed",
      low_rank_interaction_rank = 2L,
      low_rank_logit_normalization = "rms",
      low_rank_head_weight_target_rms = 0.1,
      low_rank_rc_out_target_rms = 0.25
    ),
    pairwise_obs = TRUE
  ))
  expect_equal(
    strategize:::neural_resolve_low_rank_head_weight_target_rms(
      model_dims = 8L,
      normalization = "rms"
    ),
    1 / (sqrt(2) * 8)
  )
  expect_equal(
    strategize:::neural_resolve_low_rank_rc_out_target_rms(
      low_rank_interaction_rank = 2L,
      normalization = "rms"
    ),
    1 / (sqrt(2) * 2)
  )

  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  strenv <- strategize:::strenv
  W <- strenv$jnp$array(matrix(c(1, 2, 3, 4, 5, 6), nrow = 3L), dtype = strenv$dtj)
  target <- 0.125
  scaled <- strategize:::neural_column_rms_normalize(W, target)
  scaled_r <- as.matrix(strenv$np$array(scaled))
  expect_equal(sqrt(colMeans(scaled_r ^ 2)), rep(target, 2L), tolerance = 1e-5)
})

test_that("additive utility RMS normalization bounds pooled-token head logits", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  tokens <- strenv$jnp$array(array(100, dim = c(2L, 3L, 4L)), dtype = strenv$dtj)
  token_mask <- strenv$jnp$ones(list(2L, 3L), dtype = strenv$jnp$int32)
  params <- list(
    W_out = strenv$jnp$zeros(list(4L, 1L), dtype = strenv$dtj),
    W_add_out = strenv$jnp$array(matrix(100, nrow = 4L, ncol = 1L), dtype = strenv$dtj)
  )
  base_info <- list(
    model_dims = 4L,
    factor_tokenization = "fused",
    additive_utility_mode = "on"
  )
  raw <- strategize:::neural_additive_logits_from_candidate_tokens(
    tokens = tokens,
    token_mask = token_mask,
    params = params,
    model_info = modifyList(base_info, list(additive_utility_normalization = "none")),
    out_dim = 1L,
    dtype = strenv$dtj
  )
  normalized <- strategize:::neural_additive_logits_from_candidate_tokens(
    tokens = tokens,
    token_mask = token_mask,
    params = params,
    model_info = modifyList(base_info, list(
      additive_utility_normalization = "rms",
      additive_head_weight_target_rms = 0.5
    )),
    out_dim = 1L,
    dtype = strenv$dtj
  )

  raw_values <- as.numeric(strenv$np$array(raw))
  normalized_values <- as.numeric(strenv$np$array(normalized))
  expect_true(all(is.finite(raw_values)))
  expect_true(all(is.finite(normalized_values)))
  expect_gt(max(abs(raw_values)), 1000)
  expect_equal(normalized_values, rep(2, 2L), tolerance = 1e-4)
})

test_that("full attention residual mode exposes depth-attention metadata", {
  fit <- get_neural_fit_full_attn_res()
  model_info <- get_neural_model_info(fit$res)

  expect_valid_strategize_output(fit$res, n_factors = ncol(fit$data$W))
  expect_false(is.null(model_info))
  expect_identical(model_info$residual_mode, "full_attn")
  expect_true("pseudo_query_attn_l1" %in% model_info$param_names)
  expect_true("pseudo_query_ff_l1" %in% model_info$param_names)
  expect_true("pseudo_query_final" %in% model_info$param_names)
  expect_false("alpha_attn_l1" %in% model_info$param_names)
  expect_false("alpha_ff_l1" %in% model_info$param_names)
})

test_that("pure MCMC standard residual fits preserve fixed ReZero gates", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  data <- generate_test_data(n = 12, seed = 20260704)
  names_list <- strategize:::cs2step_build_names_list(data$W)
  W_idx <- strategize:::cs2step_encode_W_indices(
    data$W,
    names_list = names_list,
    unknown = "error",
    pad_unknown = 0L
  )
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))

  fit <- suppressMessages(suppressWarnings(strategize:::cs2step_eval_outcome_model_neural(
    Y = data$Y,
    W_idx = W_idx,
    names_list = names_list,
    factor_levels = factor_levels,
    diff = TRUE,
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    conda_env_required = TRUE,
    neural_mcmc_control = list(
      ModelDims = 4L,
      ModelDepth = 1L,
      residual_mode = "standard",
      subsample_method = "full",
      uncertainty_scope = "all",
      n_samples_warmup = 1L,
      n_samples_mcmc = 1L,
      n_chains = 1L,
      chain_method = "sequential",
      cross_candidate_encoder = "none",
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )))

  model_info <- fit$neural_model_info
  expect_false(is.null(model_info))
  expect_identical(model_info$residual_mode, "standard")
  expect_true("alpha_attn_layers" %in% model_info$param_names)
  expect_true("alpha_ff_layers" %in% model_info$param_names)
  expect_false("alpha_attn_l1" %in% model_info$param_names)
  expect_false("alpha_ff_l1" %in% model_info$param_names)
  # Fixed ReZero gates initialize at the depth-aware policy scale
  # 0.1 * sqrt(2 / model_depth) (equals the historical 0.1 at depth 2);
  # "preserved" means pure MCMC leaves them at that init.
  expected_gate <- 0.1 * sqrt(2 / as.numeric(model_info$model_depth))
  expect_equal(
    as.numeric(strategize:::cs2step_neural_to_r_array(model_info$params$alpha_attn_layers)),
    rep(expected_gate, as.integer(model_info$model_depth)),
    tolerance = 1e-6
  )
  expect_equal(
    as.numeric(strategize:::cs2step_neural_to_r_array(model_info$params$alpha_ff_layers)),
    rep(expected_gate, as.integer(model_info$model_depth)),
    tolerance = 1e-6
  )
})

test_that("full attention residual readout aggregates layer history", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  jnp <- strategize:::strenv$jnp
  dtj <- strategize:::strenv$dtj
  np <- strategize:::strenv$np

  model_info <- strategize:::neural_make_transformer_model_info(
    model_depth = 1L,
    model_dims = 1L,
    n_heads = 1L,
    head_dim = 1L,
    residual_mode = "full_attn"
  )
  params <- list(
    pseudo_query_attn_l1 = jnp$zeros(reticulate::tuple(1L), dtype = dtj),
    pseudo_query_ff_l1 = jnp$zeros(reticulate::tuple(1L), dtype = dtj),
    pseudo_query_final = jnp$zeros(reticulate::tuple(1L), dtype = dtj),
    RMS_attn_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_ff_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_q_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_k_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_final = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    W_q_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_k_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_v_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_o_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_ff1_l1 = jnp$zeros(reticulate::tuple(1L, 2L), dtype = dtj),
    W_ff2_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj)
  )
  tokens <- jnp$array(array(3, dim = c(1L, 1L, 1L)), dtype = dtj)

  transformer_out <- strategize:::neural_run_transformer(
    tokens = tokens,
    model_info = model_info,
    params = params,
    return_details = TRUE
  )
  last_state <- as.numeric(reticulate::py_to_r(np$array(transformer_out$tokens)))
  readout_state <- as.numeric(reticulate::py_to_r(np$array(transformer_out$readout_tokens)))

  expect_equal(last_state, 0, tolerance = 1e-6)
  expect_equal(readout_state, 1, tolerance = 1e-6)
})

test_that("candidate extraction uses readout tokens under full attention residual mode", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  jnp <- strategize:::strenv$jnp
  dtj <- strategize:::strenv$dtj
  np <- strategize:::strenv$np

  transformer_out <- list(
    tokens = jnp$array(array(1:6, dim = c(1L, 6L, 1L)), dtype = dtj),
    readout_tokens = jnp$array(array(11:16, dim = c(1L, 6L, 1L)), dtype = dtj)
  )

  model_info_standard <- strategize:::neural_make_transformer_model_info(
    model_depth = 1L,
    model_dims = 1L,
    n_heads = 1L,
    head_dim = 1L,
    residual_mode = "standard"
  )
  model_info_standard$n_candidate_tokens <- 3L

  model_info_full_attn <- strategize:::neural_make_transformer_model_info(
    model_depth = 1L,
    model_dims = 1L,
    n_heads = 1L,
    head_dim = 1L,
    residual_mode = "full_attn"
  )
  model_info_full_attn$n_candidate_tokens <- 3L

  standard_tokens <- strategize:::neural_extract_candidate_tokens(
    transformer_out,
    model_info_standard
  )
  full_attn_tokens <- strategize:::neural_extract_candidate_tokens(
    transformer_out,
    model_info_full_attn
  )

  expect_equal(
    as.numeric(reticulate::py_to_r(np$array(standard_tokens))),
    c(4, 5, 6),
    tolerance = 1e-6
  )
  expect_equal(
    as.numeric(reticulate::py_to_r(np$array(full_attn_tokens))),
    c(14, 15, 16),
    tolerance = 1e-6
  )
})

test_that("neural attn metadata marks the cross-candidate encoder as enabled", {
  fit <- get_neural_fit_attn()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_true(isTRUE(model_info$has_qk_norm))
  expect_false(is.null(model_info$params$RMS_merge_cross))
})

test_that("neural outcome bundles save and reload cleanly", {
  fit <- get_neural_fit_attn()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model_info <- get_neural_model_info(res)
  expect_true(!is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_false(is.null(model_info$params$RMS_merge_cross))

  theta_mean <- tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      tryCatch(
        as.numeric(res$est_coefficients_jnp),
        error = function(e2) {
          as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
        }
      )
    }
  )
  expect_true(is.numeric(theta_mean))

  vcov_vec <- res$vcov_outcome_model
  theta_var <- if (!is.null(vcov_vec) && length(vcov_vec) > 1L) {
    as.numeric(vcov_vec[-1])
  } else {
    NULL
  }

  tmp <- tempfile(fileext = ".rds")
  save_neural_outcome_bundle(
    file = tmp,
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = model_info,
    p_list = p_list,
    mode = "pairwise",
    overwrite = TRUE
  )

  bundle <- readRDS(tmp)
  has_py_object <- function(x) {
    if (reticulate::is_py_object(x)) {
      return(TRUE)
    }
    if (is.list(x)) {
      return(any(vapply(x, has_py_object, logical(1))))
    }
    FALSE
  }
  expect_false(has_py_object(bundle))
  expect_false(has_py_object(bundle$fit$neural_model_info))
  expect_identical(bundle$fit$neural_model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(bundle$fit$neural_model_info$cross_candidate_encoder))
  expect_true(isTRUE(bundle$fit$neural_model_info$has_qk_norm))
  expect_true("RMS_merge_cross" %in% bundle$fit$neural_model_info$param_names)

  fit_loaded <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  expect_true(inherits(fit_loaded, "strategic_predictor"))
  expect_true(is.null(fit_loaded$fit$params))
  expect_identical(fit_loaded$fit$neural_model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(fit_loaded$fit$neural_model_info$cross_candidate_encoder))
  expect_true(isTRUE(fit_loaded$fit$neural_model_info$has_qk_norm))
  expect_true("RMS_merge_cross" %in% fit_loaded$fit$neural_model_info$param_names)

  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  W_left <- data$W[idx_left, , drop = FALSE]
  W_right <- data$W[idx_right, , drop = FALSE]
  p <- predict_pair(fit_loaded, W_left = W_left, W_right = W_right)
  expect_true(is.numeric(p))
  expect_true(all(is.finite(p)))
  expect_true(all(p >= 0 & p <= 1))
})

test_that("legacy attn bundles fail clearly when required cross-attn params are missing", {
  fit <- get_neural_fit_attn()
  res <- fit$res
  model_info <- get_neural_model_info(res)

  expect_true(!is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")

  theta_mean <- tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      tryCatch(
        as.numeric(res$est_coefficients_jnp),
        error = function(e2) {
          as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
        }
      )
    }
  )
  expect_true(is.numeric(theta_mean))

  vcov_vec <- res$vcov_outcome_model
  theta_var <- if (!is.null(vcov_vec) && length(vcov_vec) > 1L) {
    as.numeric(vcov_vec[-1])
  } else {
    NULL
  }

  tmp <- tempfile(fileext = ".rds")
  save_neural_outcome_bundle(
    file = tmp,
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = model_info,
    p_list = fit$p_list,
    mode = "pairwise",
    overwrite = TRUE
  )

  fit_loaded <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  expect_true(inherits(fit_loaded, "strategic_predictor"))

  bundle <- readRDS(tmp)
  bundle$fit$neural_model_info$param_names <- setdiff(
    bundle$fit$neural_model_info$param_names,
    "W_q_cross"
  )
  bundle$neural_model_info <- bundle$fit$neural_model_info
  saveRDS(bundle, tmp)

  expect_error(
    load_neural_outcome_bundle(tmp, preload_params = FALSE),
    "W_q_cross"
  )
})

test_that("full attention bundles reload, but legacy full attention bundles fail clearly", {
  fit <- get_neural_fit_full_attn_res()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model_info <- get_neural_model_info(res)
  expect_true(!is.null(model_info))
  expect_true("pseudo_query_final" %in% model_info$param_names)

  theta_mean <- tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      tryCatch(
        as.numeric(res$est_coefficients_jnp),
        error = function(e2) {
          as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
        }
      )
    }
  )
  expect_true(is.numeric(theta_mean))

  vcov_vec <- res$vcov_outcome_model
  theta_var <- if (!is.null(vcov_vec) && length(vcov_vec) > 1L) {
    as.numeric(vcov_vec[-1])
  } else {
    NULL
  }

  tmp <- tempfile(fileext = ".rds")
  save_neural_outcome_bundle(
    file = tmp,
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = model_info,
    p_list = p_list,
    mode = "pairwise",
    overwrite = TRUE
  )

  fit_loaded <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  expect_true(inherits(fit_loaded, "strategic_predictor"))
  expect_true("pseudo_query_final" %in% fit_loaded$fit$neural_model_info$param_names)

  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  p <- predict_pair(
    fit_loaded,
    W_left = data$W[idx_left, , drop = FALSE],
    W_right = data$W[idx_right, , drop = FALSE]
  )
  expect_true(is.numeric(p))
  expect_true(all(is.finite(p)))

  bundle <- readRDS(tmp)
  bundle$fit$neural_model_info$param_names <- setdiff(
    bundle$fit$neural_model_info$param_names,
    "pseudo_query_final"
  )
  bundle$neural_model_info <- bundle$fit$neural_model_info
  saveRDS(bundle, tmp)

  expect_error(
    load_neural_outcome_bundle(tmp, preload_params = FALSE),
    "pseudo_query_final"
  )
})

test_that("full attention bundle intervals reuse the vmapped draw path", {
  fit <- get_neural_fit_full_attn_res()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model_info <- get_neural_model_info(res)
  expect_true(!is.null(model_info))
  expect_true("pseudo_query_final" %in% model_info$param_names)

  theta_mean <- tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      tryCatch(
        as.numeric(res$est_coefficients_jnp),
        error = function(e2) {
          as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
        }
      )
    }
  )
  expect_true(is.numeric(theta_mean))

  vcov_vec <- res$vcov_outcome_model
  theta_var <- if (!is.null(vcov_vec) && length(vcov_vec) > 1L) {
    as.numeric(vcov_vec[-1])
  } else {
    NULL
  }
  expect_false(is.null(theta_var))

  tmp <- tempfile(fileext = ".rds")
  save_neural_outcome_bundle(
    file = tmp,
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = model_info,
    p_list = p_list,
    mode = "pairwise",
    overwrite = TRUE
  )

  fit_loaded <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  expect_true(inherits(fit_loaded, "strategic_predictor"))

  idx_left <- which(data$profile_order == 1L)
  idx_right <- which(data$profile_order == 2L)
  W_left <- data$W[idx_left, , drop = FALSE]
  W_right <- data$W[idx_right, , drop = FALSE]

  ci_first <- predict_pair(
    fit_loaded,
    W_left = W_left,
    W_right = W_right,
    interval = "ci",
    n_draws = 8L,
    seed = 123
  )
  ci_second <- predict_pair(
    fit_loaded,
    W_left = W_left,
    W_right = W_right,
    interval = "ci",
    n_draws = 8L,
    seed = 123
  )

  expect_s3_class(ci_first, "data.frame")
  expect_true(all(c("fit", "lo", "hi") %in% names(ci_first)))
  expect_equal(ci_first, ci_second, tolerance = 1e-6)
  expect_true(all(is.finite(as.matrix(ci_first[, c("fit", "lo", "hi")]))))
  expect_true(all(ci_first$lo <= ci_first$hi))
})

test_that("default term bundles preserve pairwise interaction metadata", {
  fit <- get_neural_fit()
  res <- fit$res

  model_info <- get_neural_model_info(res)
  expect_false(is.null(model_info))
  expect_identical(model_info$cross_candidate_encoder_mode, "term")
  expect_true(isTRUE(model_info$has_cross_term))

  theta_mean <- tryCatch(
    as.numeric(reticulate::py_to_r(res$est_coefficients_jnp)),
    error = function(e) {
      tryCatch(
        as.numeric(res$est_coefficients_jnp),
        error = function(e2) {
          as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(res$est_coefficients_jnp)))
        }
      )
    }
  )
  expect_true(is.numeric(theta_mean))

  vcov_vec <- res$vcov_outcome_model
  theta_var <- if (!is.null(vcov_vec) && length(vcov_vec) > 1L) {
    as.numeric(vcov_vec[-1])
  } else {
    NULL
  }

  tmp <- tempfile(fileext = ".rds")
  save_neural_outcome_bundle(
    file = tmp,
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = model_info,
    p_list = fit$p_list,
    mode = "pairwise",
    overwrite = TRUE
  )

  bundle <- readRDS(tmp)
  expect_identical(bundle$fit$neural_model_info$cross_candidate_encoder_mode, "term")
  expect_true(isTRUE(bundle$fit$neural_model_info$has_cross_term))
  expect_true("M_cross" %in% bundle$fit$neural_model_info$param_names)

  fit_loaded <- load_neural_outcome_bundle(tmp, preload_params = FALSE)
  expect_true(inherits(fit_loaded, "strategic_predictor"))
  expect_identical(fit_loaded$fit$neural_model_info$cross_candidate_encoder_mode, "term")
  expect_true(isTRUE(fit_loaded$fit$neural_model_info$has_cross_term))
  expect_true("M_cross" %in% fit_loaded$fit$neural_model_info$param_names)
})

test_that("output-only optimal SVI uses the pairwise batch_vi heuristic path", {
  fit <- get_neural_fit_attn_output_vi()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_true(isTRUE(model_info$pairwise_mode))
  expect_identical(model_info$uncertainty_scope, "output")
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")
  expect_true(isTRUE(model_info$cross_candidate_encoder))
  expect_true(isTRUE(model_info$has_qk_norm))
  expect_false(is.null(model_info$params$RMS_merge_cross))
  expect_true(is.numeric(model_info$svi_loss_curve))
  expect_gt(length(model_info$svi_loss_curve), 0L)

  expected_steps <- strategize:::neural_optimal_svi_steps(
    n_obs = length(unique(fit$data$pair_id)),
    n_factors = as.integer(model_info$n_factors),
    factor_levels = as.integer(model_info$factor_levels),
    model_dims = as.integer(model_info$model_dims),
    model_depth = as.integer(model_info$model_depth),
    n_party_levels = as.integer(model_info$n_party_levels),
    n_resp_party_levels = length(model_info$resp_party_levels),
    n_resp_covariates = as.integer(model_info$n_resp_covariates),
    n_outcomes = 1L,
    pairwise_mode = isTRUE(model_info$pairwise_mode),
    use_matchup_token = isTRUE(model_info$has_matchup_token),
    use_cross_encoder = identical(model_info$cross_candidate_encoder_mode, "full"),
    use_cross_term = identical(model_info$cross_candidate_encoder_mode, "term"),
    use_cross_attn = identical(model_info$cross_candidate_encoder_mode, "attn"),
    use_qk_norm = isTRUE(model_info$has_qk_norm),
    batch_size = 16L,
    subsample_method = "batch_vi"
  )

  expect_length(model_info$svi_loss_curve, expected_steps)
  expect_identical(as.integer(model_info$svi_steps), as.integer(expected_steps))
  expect_identical(as.integer(model_info$svi_steps_completed), as.integer(expected_steps))
  expect_false(isTRUE(model_info$early_stopping$enabled))
  expect_identical(model_info$early_stopping$reason, "disabled")
})

test_that("output-only neural SVI enables early stopping by default", {
  fit <- get_neural_fit_attn_output_vi_default_es()
  model_info <- get_neural_model_info(fit$res)

  expect_false(is.null(model_info))
  expect_true(isTRUE(model_info$early_stopping$enabled))
  expect_true(isTRUE(model_info$early_stopping$active))
  expect_identical(model_info$early_stopping$metric, "log_loss")
  expect_gt(as.integer(model_info$early_stopping$n_train), 0L)
  expect_gt(as.integer(model_info$early_stopping$n_validation), 0L)
  expect_identical(model_info$early_stopping$validation_prediction_mode, "single_jit_call")
  expect_identical(as.integer(model_info$early_stopping$validation_n_batches), 1L)
  expect_true(as.integer(model_info$svi_steps_completed) <= as.integer(model_info$svi_steps))
  expect_length(model_info$svi_loss_curve, as.integer(model_info$svi_steps_completed))
  expect_false(identical(model_info$early_stopping$reason, "disabled"))
  expect_false(is.na(model_info$early_stopping$best_step))
})

test_that("output-only neural SVI without early stopping uses a single SVI.run call", {
  counted <- count_svi_run_calls(
    run_output_only_attn_vi_fit(
      seed = 20260401,
      early_stopping = FALSE,
      svi_steps = 25L
    )
  )
  model_info <- get_neural_model_info(counted$value)

  expect_identical(counted$count, 1L)
  expect_false(isTRUE(model_info$early_stopping$enabled))
  expect_identical(model_info$early_stopping$reason, "disabled")
  expect_identical(
    as.integer(model_info$svi_steps_completed),
    as.integer(model_info$svi_steps)
  )
})

test_that("output-only neural early stopping advances SVI through chunked run calls", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  strategize:::strenv$jax_svi_gradient_jit_cache_clear()

  counted <- count_svi_run_calls(
    run_output_only_attn_vi_fit(
      seed = 20260402,
      early_stopping = TRUE,
      svi_steps = 25L
    )
  )
  model_info <- get_neural_model_info(counted$value)

  expect_gt(counted$count, 1L)
  expect_true(isTRUE(model_info$early_stopping$enabled))
  expect_true(isTRUE(model_info$early_stopping$active))
  expect_identical(model_info$early_stopping$metric, "log_loss")
  expect_true(as.integer(model_info$svi_steps_completed) <= as.integer(model_info$svi_steps))
  expect_length(model_info$svi_loss_curve, as.integer(model_info$svi_steps_completed))
  expect_false(is.na(model_info$early_stopping$best_step))
  expect_identical(
    as.integer(model_info$early_stopping$stop_step),
    as.integer(model_info$svi_steps_completed)
  )
  expect_true(is.list(model_info$gradient_diagnostics))
  expect_identical(model_info$gradient_diagnostics$gradient_status, "ok")
  expect_length(
    model_info$gradient_diagnostics$checkpoint_global_l2_norm,
    length(model_info$early_stopping$validation_loss_history)
  )
  expect_true(is.finite(model_info$gradient_diagnostics$global_l2_norm))
  expect_true(is.finite(model_info$gradient_diagnostics$global_rms))
  expect_true(is.finite(model_info$gradient_diagnostics$global_max_abs))
  expect_gte(as.integer(tail(model_info$gradient_diagnostics$checkpoint_n_elements, 1L)), 1L)
  gradient_cache_info <- as.list(strategize:::strenv$jax_svi_gradient_jit_cache_info())
  # The per-fit jitted gradient closures capture the svi object (and through
  # it the training arrays), so the cache is cleared when the fit returns --
  # size must be 0 (no cross-fit memory retention). compile_count is lifetime
  # telemetry and proves the jitted diagnostics path actually ran.
  expect_identical(as.integer(gradient_cache_info$size), 0L)
  expect_gte(as.integer(gradient_cache_info$compile_count), 1L)
})

test_that("output-only neural checkpoint gradient diagnostics can be disabled", {
  captured <- capture_messages(
    suppressWarnings(run_output_only_attn_vi_fit(
      seed = 20260421,
      early_stopping = TRUE,
      svi_steps = 25L,
      neural_mcmc_control_overrides = list(gradient_diagnostics = FALSE)
    ))
  )
  model_info <- get_neural_model_info(captured$value)
  early_stop_lines <- grep("^SVI early-stop check [0-9]+/[0-9]+: ", captured$messages, value = TRUE)

  expect_identical(model_info$gradient_diagnostics$gradient_status, "disabled")
  expect_length(model_info$gradient_diagnostics$checkpoint_global_l2_norm, 0L)
  expect_gt(length(early_stop_lines), 0L)
  expect_false(any(grepl("grad_l2=|grad_rms=|grad_max=|grad_bad=", early_stop_lines)))
})

test_that("early stopping validation target helper applies fraction and optional cap", {
  expect_identical(
    strategize:::neural_resolve_early_stopping_validation_target_n(
      n_eval = 5000L,
      n_validation_available = 1000L
    ),
    250L
  )
  expect_identical(
    strategize:::neural_resolve_early_stopping_validation_target_n(
      n_eval = 50000L,
      n_validation_available = 10000L,
      validation_frac = 0.05,
      validation_max_n = 2048L
    ),
    2048L
  )
  expect_identical(
    strategize:::neural_resolve_early_stopping_validation_target_n(
      n_eval = 50000L,
      n_validation_available = 10000L,
      validation_frac = 0.05,
      validation_max_n = NULL
    ),
    2500L
  )
})

test_that("early stopping validation batch helper bounds prediction chunk size", {
  expect_identical(
    strategize:::neural_resolve_early_stopping_validation_batch_size(
      validation_target_n = 250L
    ),
    128L
  )
  expect_identical(
    strategize:::neural_resolve_early_stopping_validation_batch_size(
      validation_target_n = 250L,
      validation_batch_size = 64L
    ),
    64L
  )
  expect_identical(
    strategize:::neural_resolve_early_stopping_validation_batch_size(
      validation_target_n = 32L,
      validation_batch_size = 128L
    ),
    32L
  )
  expect_identical(
    strategize:::neural_resolve_early_stopping_validation_batch_size(
      validation_target_n = 32L,
      validation_batch_size = NULL
    ),
    32L
  )
})

test_that("compact SVI scan mode resolver defaults to required for chunked compact updates", {
  expect_identical(
    strategize:::neural_resolve_compact_update_scan(
      compact_training = TRUE,
      compact_update_chunk_size = 8L,
      compact_update_scan = NULL
    ),
    "required"
  )
  expect_identical(
    strategize:::neural_resolve_compact_update_scan(
      compact_training = TRUE,
      compact_update_chunk_size = 1L,
      compact_update_scan = "required"
    ),
    "fallback"
  )
  expect_identical(
    strategize:::neural_resolve_compact_update_scan(
      compact_training = TRUE,
      compact_update_chunk_size = 8L,
      compact_update_scan = "fallback"
    ),
    "fallback"
  )
  expect_error(
    strategize:::neural_resolve_compact_update_scan(
      compact_training = TRUE,
      compact_update_chunk_size = 8L,
      compact_update_scan = "silent"
    ),
    "compact_update_scan"
  )
})

test_that("compact SVI effective scan steps round only scanned tail chunks", {
  expect_identical(
    strategize:::neural_compact_effective_scan_steps(
      svi_steps = 5L,
      completed_steps = 0L,
      chunk_size = 4L,
      scan_available = TRUE
    ),
    8L
  )
  expect_identical(
    strategize:::neural_compact_effective_scan_steps(
      svi_steps = 5L,
      completed_steps = 2L,
      chunk_size = 4L,
      scan_available = TRUE
    ),
    6L
  )
  expect_identical(
    strategize:::neural_compact_effective_scan_steps(
      svi_steps = 5L,
      completed_steps = 0L,
      chunk_size = 4L,
      scan_available = FALSE
    ),
    5L
  )
  expect_identical(
    strategize:::neural_compact_effective_scan_steps(
      svi_steps = 5L,
      completed_steps = 0L,
      chunk_size = 1L,
      scan_available = TRUE
    ),
    5L
  )
  expect_identical(
    strategize:::neural_compact_effective_scan_steps(
      svi_steps = 5L,
      completed_steps = 5L,
      chunk_size = 4L,
      scan_available = TRUE
    ),
    5L
  )
})

test_that("compact SVI validation cadence respects chunk boundaries and final step", {
  expect_identical(
    strategize:::neural_resolve_positive_int(
      NA_integer_,
      ceiling(3L / 10L)
    ),
    1L
  )
  expect_identical(
    strategize:::neural_resolve_positive_int(
      NaN,
      4L
    ),
    4L
  )
  expect_identical(
    strategize:::neural_compact_chunk_boundary_checks(
      svi_steps = 10L,
      n_checks = 5L,
      chunk_size = 2L
    ),
    c(2L, 4L, 6L, 8L, 10L)
  )
  expect_identical(
    strategize:::neural_compact_chunk_boundary_checks(
      svi_steps = 10L,
      n_checks = 5L,
      chunk_size = 4L
    ),
    c(4L, 8L, 10L)
  )
  expect_identical(
    strategize:::neural_compact_chunk_boundary_checks(
      svi_steps = 3L,
      n_checks = 10L,
      chunk_size = 8L
    ),
    3L
  )
})

test_that("required compact SVI scan failures are explicit", {
  expect_error(
    strategize:::neural_stop_compact_scan_required("synthetic scan failure"),
    "synthetic scan failure"
  )
})

test_that("compact SVI jitted update and gradient helpers register cache diagnostics", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  expect_false(is.null(strategize:::strenv$jax_svi_update))
  expect_false(is.null(strategize:::strenv$jax_svi_update_scan))
  expect_false(is.null(strategize:::strenv$jax_svi_update_stable_available))
  expect_false(is.null(strategize:::strenv$jax_svi_update_jit_cache_info))
  expect_false(is.null(strategize:::strenv$jax_svi_update_jit_cache_clear))
  expect_false(is.null(strategize:::strenv$jax_svi_gradient_diagnostics))
  expect_false(is.null(strategize:::strenv$jax_svi_gradient_jit_cache_info))
  expect_false(is.null(strategize:::strenv$jax_svi_gradient_jit_cache_clear))

  strategize:::strenv$jax_svi_update_jit_cache_clear()
  cache_info <- as.list(strategize:::strenv$jax_svi_update_jit_cache_info())
  expect_identical(as.integer(cache_info$size), 0L)
  # compile_count is lifetime telemetry: clearing frees the cached closures
  # (the memory concern) but deliberately does not reset the counter.
  expect_gte(as.integer(cache_info$compile_count), 0L)
  strategize:::strenv$jax_svi_gradient_jit_cache_clear()
  gradient_cache_info <- as.list(strategize:::strenv$jax_svi_gradient_jit_cache_info())
  expect_identical(as.integer(gradient_cache_info$size), 0L)
  expect_gte(as.integer(gradient_cache_info$compile_count), 0L)
})

test_that("compact SVI required scan mode errors when scan helper fails", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  local_strenv_bindings("jax_svi_update_scan")
  set_strenv_bindings(list(
    jax_svi_update_scan = function(...) stop("synthetic scan failure")
  ))

  expect_error(
    run_compact_svi_fit(
      compact_update_scan = "required",
      compact_update_chunk_size = 2L,
      svi_steps = 2L
    ),
    "synthetic scan failure"
  )
})

test_that("compact SVI fallback mode records single-step fallback when scan helper fails", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  local_strenv_bindings("jax_svi_update_scan")
  set_strenv_bindings(list(
    jax_svi_update_scan = function(...) stop("synthetic scan failure")
  ))

  model_info <- run_compact_svi_fit(
    compact_update_scan = "fallback",
    compact_update_chunk_size = 2L,
    svi_steps = 2L
  )
  diagnostics <- model_info$optimizer_diagnostics

  expect_identical(diagnostics$compact_update_scan_mode, "fallback")
  expect_identical(diagnostics$compact_update_scan_status, "fallback_single_step")
  expect_match(diagnostics$compact_update_scan_error, "synthetic scan failure")
  expect_identical(as.integer(diagnostics$compact_update_chunk_size_effective), 1L)
  expect_identical(as.integer(diagnostics$compact_update_steps_requested), 2L)
  expect_identical(as.integer(diagnostics$compact_update_steps_effective), 2L)
  expect_identical(as.integer(diagnostics$compact_update_steps_overrun), 0L)
  expect_identical(diagnostics$compact_update_jit_required, TRUE)
  expect_identical(diagnostics$compact_update_jit_status, "ok")
  expect_identical(diagnostics$compact_update_jit_path, "scan_to_single_fallback")
  expect_gte(as.integer(diagnostics$compact_update_jit_compile_count), 1L)
})

test_that("compact SVI single-step updates use cached jitted update", {
  model_info <- run_compact_svi_fit(
    compact_update_scan = "fallback",
    compact_update_chunk_size = 1L,
    svi_steps = 2L
  )
  diagnostics <- model_info$optimizer_diagnostics

  expect_identical(diagnostics$compact_update_scan_status, "single_step")
  expect_identical(as.integer(diagnostics$compact_update_chunk_size_effective), 1L)
  expect_identical(diagnostics$compact_update_jit_required, TRUE)
  expect_identical(diagnostics$compact_update_jit_status, "ok")
  expect_identical(diagnostics$compact_update_jit_path, "single")
  expect_gte(as.integer(diagnostics$compact_update_jit_cache_size), 1L)
  expect_gte(as.integer(diagnostics$compact_update_jit_compile_count), 1L)
})

test_that("compact SVI skips non-finite single-step updates without stopping", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  strategize:::strategize_register_jax_svi_helpers()
  local_strenv_bindings("jax_svi_update")
  original_update <- strategize:::strenv$jax_svi_update
  update_calls <- 0L
  set_strenv_bindings(list(
    jax_svi_update = function(...) {
      update_calls <<- update_calls + 1L
      result <- original_update(...)
      if (identical(update_calls, 1L)) {
        parts <- as.list(result)
        return(list(parts[[1L]], NaN))
      }
      result
    }
  ))

  model_info <- run_compact_svi_fit(
    compact_update_scan = "fallback",
    compact_update_chunk_size = 1L,
    svi_steps = 3L
  )
  diagnostics <- model_info$optimizer_diagnostics

  expect_identical(as.integer(model_info$svi_steps_completed), 3L)
  expect_false(is.finite(model_info$svi_loss_curve[[1L]]))
  expect_identical(diagnostics$optimizer_status, "ok_with_skipped_updates")
  expect_identical(as.integer(diagnostics$compact_update_nonfinite_count), 1L)
  expect_identical(as.integer(diagnostics$compact_update_skipped_steps), 1L)
  expect_identical(as.integer(diagnostics$compact_update_nonfinite_steps[[1L]]), 1L)
  expect_identical(diagnostics$compact_update_nonfinite_path[[1L]], "single")
  expect_true(isTRUE(model_info$convergence_diagnostics$loss_nonfinite))
  expect_true(any(grepl("skipped", model_info$convergence_diagnostics$notes, fixed = TRUE)))
})

test_that("compact SVI fallback mode keeps requested step budget after scan failure", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  local_strenv_bindings("jax_svi_update_scan")
  set_strenv_bindings(list(
    jax_svi_update_scan = function(...) stop("synthetic scan failure")
  ))

  model_info <- run_compact_svi_fit(
    compact_update_scan = "fallback",
    compact_update_chunk_size = 4L,
    svi_steps = 5L
  )
  diagnostics <- model_info$optimizer_diagnostics

  expect_identical(as.integer(model_info$svi_steps), 5L)
  expect_identical(as.integer(model_info$svi_steps_completed), 5L)
  expect_length(model_info$svi_loss_curve, 5L)
  expect_identical(diagnostics$compact_update_scan_status, "fallback_single_step")
  expect_identical(as.integer(diagnostics$compact_update_steps_requested), 5L)
  expect_identical(as.integer(diagnostics$compact_update_steps_effective), 5L)
  expect_identical(as.integer(diagnostics$compact_update_steps_overrun), 0L)
})

test_that("compact SVI errors when jitted single-step update fails", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  local_strenv_bindings("jax_svi_update")
  set_strenv_bindings(list(
    jax_svi_update = function(...) stop("synthetic jit failure")
  ))

  expect_error(
    run_compact_svi_fit(
      compact_update_scan = "fallback",
      compact_update_chunk_size = 1L,
      svi_steps = 2L
    ),
    "synthetic jit failure"
  )
})

test_that("compact SVI required scan mode records ok for live scanned updates", {
  model_info <- run_compact_svi_fit(
    compact_update_scan = "required",
    compact_update_chunk_size = 2L,
    svi_steps = 4L
  )
  diagnostics <- model_info$optimizer_diagnostics

  expect_identical(diagnostics$compact_update_scan_mode, "required")
  expect_identical(diagnostics$compact_update_scan_status, "ok")
  expect_gte(as.integer(diagnostics$compact_update_chunk_size_effective), 2L)
  expect_identical(diagnostics$compact_update_jit_required, TRUE)
  expect_identical(diagnostics$compact_update_jit_status, "ok")
  expect_identical(diagnostics$compact_update_jit_path, "scan")
  expect_gte(as.integer(diagnostics$compact_update_jit_cache_size), 1L)
  expect_gte(as.integer(diagnostics$compact_update_jit_compile_count), 1L)
})

test_that("compact SVI skips non-finite scanned updates without stopping", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  strategize:::strategize_register_jax_svi_helpers()
  local_strenv_bindings("jax_svi_update_scan")
  original_scan <- strategize:::strenv$jax_svi_update_scan
  scan_calls <- 0L
  set_strenv_bindings(list(
    jax_svi_update_scan = function(...) {
      scan_calls <<- scan_calls + 1L
      result <- original_scan(...)
      if (identical(scan_calls, 1L)) {
        parts <- as.list(result)
        losses <- as.numeric(strategize:::strenv$np$array(parts[[2L]]))
        losses[[min(2L, length(losses))]] <- NaN
        return(list(parts[[1L]], losses))
      }
      result
    }
  ))

  model_info <- run_compact_svi_fit(
    compact_update_scan = "required",
    compact_update_chunk_size = 2L,
    svi_steps = 4L
  )
  diagnostics <- model_info$optimizer_diagnostics

  expect_identical(as.integer(model_info$svi_steps_completed), 4L)
  expect_false(is.finite(model_info$svi_loss_curve[[2L]]))
  expect_identical(diagnostics$optimizer_status, "ok_with_skipped_updates")
  expect_identical(as.integer(diagnostics$compact_update_nonfinite_count), 1L)
  expect_identical(as.integer(diagnostics$compact_update_skipped_steps), 1L)
  expect_identical(as.integer(diagnostics$compact_update_nonfinite_steps[[1L]]), 2L)
  expect_identical(diagnostics$compact_update_nonfinite_path[[1L]], "scan")
  expect_true(isTRUE(diagnostics$compact_update_stable_available))
  expect_true(isTRUE(model_info$convergence_diagnostics$loss_nonfinite))
})

test_that("compact SVI required scan rounds effective steps to full scan shape", {
  model_info <- run_compact_svi_fit(
    compact_update_scan = "required",
    compact_update_chunk_size = 4L,
    svi_steps = 5L
  )
  diagnostics <- model_info$optimizer_diagnostics

  expect_identical(as.integer(model_info$svi_steps), 5L)
  expect_identical(as.integer(model_info$svi_steps_completed), 8L)
  expect_length(model_info$svi_loss_curve, 8L)
  expect_identical(diagnostics$compact_update_scan_status, "ok")
  expect_identical(diagnostics$compact_update_jit_path, "scan")
  expect_identical(as.integer(diagnostics$compact_update_chunk_size_effective), 4L)
  expect_identical(as.integer(diagnostics$compact_update_steps_requested), 5L)
  expect_identical(as.integer(diagnostics$compact_update_steps_effective), 8L)
  expect_identical(as.integer(diagnostics$compact_update_steps_overrun), 3L)
})

test_that("compact SVI validation logs scanned chunk throughput from completed steps", {
  captured <- capture_messages(run_compact_svi_fit(
    compact_update_scan = "required",
    compact_update_chunk_size = 4L,
    svi_steps = 4L,
    early_stopping = TRUE,
    early_stopping_n_checks = 1L
  ))
  validation_lines <- grep("^Compact SVI validation check [0-9]+/[0-9]+: ", captured$messages, value = TRUE)

  expect_gt(length(validation_lines), 0L)
  expect_false(any(grepl("iter_per_s", validation_lines)))
  line <- validation_lines[[1L]]
  expect_identical(as.integer(extract_log_numeric_field(line, "chunk_steps")), 4L)
  expect_log_rate(line, "chunk_step_per_s", "chunk_steps", "chunk_elapsed_s")
})

test_that("compact SVI validation logs fallback chunks as one completed step", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  local_strenv_bindings("jax_svi_update_scan")
  set_strenv_bindings(list(
    jax_svi_update_scan = function(...) stop("synthetic scan failure")
  ))

  captured <- capture_messages(run_compact_svi_fit(
    compact_update_scan = "fallback",
    compact_update_chunk_size = 4L,
    svi_steps = 2L,
    early_stopping = TRUE,
    early_stopping_n_checks = 2L
  ))
  validation_lines <- grep("^Compact SVI validation check [0-9]+/[0-9]+: ", captured$messages, value = TRUE)

  expect_gt(length(validation_lines), 0L)
  expect_false(any(grepl("iter_per_s", validation_lines)))
  expect_false(any(grepl("chunk_steps=4", validation_lines, fixed = TRUE)))
  line <- validation_lines[[1L]]
  expect_identical(as.integer(extract_log_numeric_field(line, "chunk_steps")), 1L)
  expect_log_rate(line, "chunk_step_per_s", "chunk_steps", "chunk_elapsed_s")
})

test_that("compact SVI progress logs window and sampled-observation throughput", {
  captured <- capture_messages(run_compact_svi_fit(
    compact_update_scan = "fallback",
    compact_update_chunk_size = 1L,
    svi_steps = 2L,
    early_stopping = FALSE
  ))
  progress_lines <- grep("^Compact SVI progress: ", captured$messages, value = TRUE)

  expect_gt(length(progress_lines), 0L)
  expect_false(any(grepl("iter_per_s", progress_lines)))
  line <- progress_lines[[1L]]
  window_steps <- extract_log_numeric_field(line, "window_steps")
  progress_elapsed_s <- extract_log_numeric_field(line, "progress_elapsed_s")
  window_update_elapsed_s <- extract_log_numeric_field(line, "window_update_elapsed_s")

  expect_identical(as.integer(window_steps), 1L)
  expect_identical(as.integer(extract_log_numeric_field(line, "window_chunks")), 1L)
  expect_gt(progress_elapsed_s, 0)
  expect_gt(window_update_elapsed_s, 0)
  expect_equal(
    extract_log_numeric_field(line, "step_per_s"),
    window_steps / progress_elapsed_s,
    tolerance = 0.05
  )
  expect_equal(
    extract_log_numeric_field(line, "sampled_train_obs_per_s"),
    window_steps * 4 / progress_elapsed_s,
    tolerance = 0.05
  )
  expect_equal(
    extract_log_numeric_field(line, "update_step_per_s"),
    window_steps / window_update_elapsed_s,
    tolerance = 0.05
  )
  expect_equal(
    extract_log_numeric_field(line, "update_sampled_train_obs_per_s"),
    window_steps * 4 / window_update_elapsed_s,
    tolerance = 0.05
  )
})

test_that("compact SVI validation logs validation observation throughput", {
  captured <- capture_messages(run_compact_svi_fit(
    compact_update_scan = "required",
    compact_update_chunk_size = 4L,
    svi_steps = 4L,
    early_stopping = TRUE,
    early_stopping_n_checks = 1L
  ))
  validation_lines <- grep("^Compact SVI validation check [0-9]+/[0-9]+: ", captured$messages, value = TRUE)

  expect_gt(length(validation_lines), 0L)
  line <- validation_lines[[1L]]
  expect_true(grepl("validation_obs_per_s=", line, fixed = TRUE))
  expect_false(grepl("(^|; )step_per_s=", line, perl = TRUE))
  expect_log_rate(line, "validation_obs_per_s", "validation_obs", "validation_elapsed_s", tolerance = 0.2)
})

test_that("compact SVI progress windows start from resumed process deltas", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  tmp <- tempfile()
  local_strenv_bindings("jax_svi_update")
  original_update <- strategize:::strenv$jax_svi_update
  update_calls <- 0L
  set_strenv_bindings(list(
    jax_svi_update = function(...) {
      update_calls <<- update_calls + 1L
      if (update_calls > 2L) {
        stop("synthetic interruption")
      }
      original_update(...)
    }
  ))

  expect_error(
    run_compact_svi_fit(
      compact_update_scan = "fallback",
      compact_update_chunk_size = 1L,
      svi_steps = 4L,
      early_stopping = FALSE,
      checkpoint_path = tmp,
      checkpoint_n_checks = 2L
    ),
    "synthetic interruption"
  )

  set_strenv_bindings(list(jax_svi_update = original_update))
  captured <- capture_messages(run_compact_svi_fit(
    compact_update_scan = "fallback",
    compact_update_chunk_size = 1L,
    svi_steps = 4L,
    early_stopping = FALSE,
    checkpoint_path = tmp,
    checkpoint_n_checks = 2L
  ))
  progress_lines <- grep("^Compact SVI progress: ", captured$messages, value = TRUE)

  expect_true(any(grepl("^Resuming neural SVI checkpoint from .+ at step 2/4\\.$", captured$messages)))
  expect_gt(length(progress_lines), 0L)
  expect_match(progress_lines[[1L]], "step=4/4")
  expect_identical(as.integer(extract_log_numeric_field(progress_lines[[1L]], "window_steps")), 2L)
  expect_identical(as.integer(extract_log_numeric_field(progress_lines[[1L]], "window_chunks")), 2L)
})

test_that("compact SVI validation checkpoints write latest and best without early stopping", {
  tmp <- tempfile()
  metric_values <- c(0.1, 1)
  metric_i <- 0L
  local_strategize_binding(
    "cs_compute_outcome_metrics",
    function(...) {
      metric_i <<- metric_i + 1L
      value <- metric_values[[min(metric_i, length(metric_values))]]
      list(log_loss = value, accuracy = 1, brier = 0)
    },
    env = environment()
  )
  model_info <- run_compact_svi_fit(
    compact_update_scan = "required",
    compact_update_chunk_size = 2L,
    svi_steps = 4L,
    early_stopping = TRUE,
    early_stopping_n_checks = 2L,
    checkpoint_path = tmp
  )

  expect_true(file.exists(file.path(tmp, "latest.rds")))
  expect_true(file.exists(file.path(tmp, "best.rds")))
  expect_identical(as.integer(model_info$svi_steps_completed), 4L)
  expect_true(isTRUE(model_info$early_stopping$enabled))
  expect_true(isTRUE(model_info$early_stopping$active))
  expect_false(isTRUE(model_info$early_stopping$stopped_early))
  expect_identical(model_info$early_stopping$reason, "completed_budget")
  expect_gt(length(model_info$early_stopping$validation_loss_history), 0L)
  expect_true(is.finite(model_info$early_stopping$best_metric))
  expect_identical(as.integer(model_info$early_stopping$best_step), 2L)
  expect_equal(model_info$early_stopping$best_metric, 0.1)
  expect_equal(model_info$early_stopping$final_metric, 0.1)

  latest <- strategize:::neural_svi_checkpoint_load_snapshot(tmp, "latest")
  best <- strategize:::neural_svi_checkpoint_load_snapshot(tmp, "best")
  expect_identical(as.integer(latest$completed_step), 4L)
  expect_true(is.finite(best$best_metric))
  expect_identical(as.integer(best$best_step), 2L)
  expect_equal(best$best_metric, 0.1)
})

test_that("JAX block helper accepts nested non-JAX values", {
  x <- list(a = 1, b = list(c = "value"))

  expect_no_error(strategize:::strategize_jax_block_until_ready(x))
  expect_identical(x$a, 1)
  expect_identical(x$b$c, "value")
})

test_that("JAX block helper walks nested JAX arrays", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  arr <- strategize:::strenv$jnp$array(c(1, 2, 3)) + 1
  nested <- list(arr = arr, inner = list(arr))

  expect_no_error(strategize:::strategize_jax_block_until_ready(nested))
  expect_equal(as.numeric(strategize:::cs2step_neural_to_r_array(arr)), c(2, 3, 4))
})

test_that("output-only neural early stopping exposes resolved validation size controls", {
  fit <- run_output_only_attn_vi_fit(
    seed = 20260410,
    early_stopping = TRUE,
    svi_steps = 25L,
    neural_mcmc_control_overrides = list(
      early_stopping_validation_frac = 1,
      early_stopping_validation_max_n = 4L,
      early_stopping_validation_batch_size = 1L
    )
  )
  model_info <- get_neural_model_info(fit)

  expect_identical(model_info$early_stopping$validation_frac, 1)
  expect_identical(as.integer(model_info$early_stopping$validation_max_n), 4L)
  expect_identical(as.integer(model_info$early_stopping$validation_batch_size), 1L)
  expect_identical(model_info$early_stopping$validation_prediction_mode, "batched_fallback")
  expect_gte(as.integer(model_info$early_stopping$validation_n_batches), 2L)
  expect_lte(as.integer(model_info$early_stopping$validation_target_n), 4L)
  expect_identical(
    as.integer(model_info$early_stopping$validation_target_n),
    as.integer(model_info$early_stopping$n_validation)
  )
  expect_lte(as.integer(model_info$early_stopping$n_validation), 4L)
})

test_that("output-only neural early stopping validates ordered factor and covariate spans", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_EVAL_FOLDS = "2",
    STRATEGIZE_NEURAL_EVAL_SEED = "123"
  ))

  data <- add_respondent_covariates(
    generate_test_data(n = 24, seed = 20260408),
    n_covariates = 2,
    seed = 20260409
  )
  params <- default_strategize_params(fast = TRUE)
  params$outcome_model_type <- "neural"
  params$neural_mcmc_control <- modifyList(
    params$neural_mcmc_control,
    list(
      cross_candidate_encoder = "attn",
      ModelDims = 16L,
      ModelDepth = 1L,
      subsample_method = "batch_vi",
      uncertainty_scope = "output",
      svi_steps = 25L,
      batch_size = 16L,
      early_stopping = TRUE,
      eval_enabled = FALSE,
      covariate_value_encoding = "shared_projection",
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  p_list <- generate_test_p_list(data$W)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, X = data$X, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  )))
  model_info <- get_neural_model_info(res)

  expect_true(isTRUE(model_info$early_stopping$enabled))
  expect_true(isTRUE(model_info$early_stopping$active))
  expect_identical(model_info$factor_tokenization, "fused")
  expect_identical(model_info$covariate_value_encoding, "shared_projection")
  expect_false(isTRUE(model_info$has_factor_span_tokens))
  expect_true(isTRUE(model_info$has_covariate_fused_tokens))
  expect_false(isTRUE(model_info$has_covariate_span_tokens))
  expect_true(isTRUE(model_info$has_shared_covariate_value_projection))
  expect_null(model_info$early_stopping$error_message)
  expect_false(identical(model_info$early_stopping$reason, "validation_error"))
  expect_false(identical(model_info$early_stopping$reason, "metric_failed"))
  expect_true(length(model_info$early_stopping$validation_loss_history) >= 1L)
})

test_that("output-only neural early stopping still honors patience exhaustion", {
  local_strategize_binding(
    "cs_compute_outcome_metrics",
    function(...) {
      list(log_loss = 1, nll = 1)
    },
    env = environment()
  )

  counted <- count_svi_run_calls(
    run_output_only_attn_vi_fit(
      seed = 20260403,
      early_stopping = TRUE,
      svi_steps = 60L
    )
  )
  model_info <- get_neural_model_info(counted$value)

  expect_gt(counted$count, 1L)
  expect_true(isTRUE(model_info$early_stopping$stopped_early))
  expect_identical(model_info$early_stopping$reason, "patience_exhausted")
  expect_identical(as.integer(model_info$early_stopping$best_step), 6L)
  expect_identical(as.integer(model_info$early_stopping$stop_step), 24L)
  expect_identical(as.integer(model_info$svi_steps_completed), 24L)
  expect_length(model_info$svi_loss_curve, 24L)
})

test_that("output-only neural early stopping surfaces validation errors explicitly", {
  local_strategize_binding(
    "cs_compute_outcome_metrics",
    function(...) {
      stop("synthetic validation failure", call. = FALSE)
    },
    env = environment()
  )

  captured <- capture_messages(
    suppressWarnings(run_output_only_attn_vi_fit(
      seed = 20260404,
      early_stopping = TRUE,
      svi_steps = 25L
    ))
  )
  messages <- captured$messages
  model_info <- get_neural_model_info(captured$value)

  summary_idx <- match(
    TRUE,
    grepl(
      "^SVI fit summary: steps=[0-9]+/[0-9]+; validation error at step [0-9]+ \\(synthetic validation failure\\)\\.$",
      messages
    )
  )

  expect_true(is.finite(summary_idx))
  expect_false(any(grepl("^SVI fit summary: steps=[0-9]+/[0-9]+; final ELBO=", messages)))
  expect_identical(model_info$early_stopping$reason, "validation_error")
  expect_match(model_info$early_stopping$error_message, "synthetic validation failure")
  expect_identical(
    as.integer(model_info$early_stopping$stop_step),
    as.integer(model_info$svi_steps_completed)
  )
  expect_length(model_info$early_stopping$validation_loss_history, 0L)
})

test_that("output-only neural logging surfaces structure and fit summary before optimization", {
  captured <- capture_messages(
    suppressWarnings(run_output_only_attn_vi_fit(
      seed = 20260405,
      early_stopping = FALSE,
      svi_steps = 5L
    ))
  )
  messages <- captured$messages

  enlist_idx <- match(TRUE, grepl("^Enlisting SVI with autoguide for output-only uncertainty\\.\\.\\.$", messages))
  structure_idx <- match(TRUE, grepl("^Bayesian Transformer complete\\. Pairwise=", messages))
  summary_idx <- match(TRUE, grepl("^SVI fit summary: steps=[0-9]+/[0-9]+; final ELBO=", messages))
  optimize_idx <- match(TRUE, grepl("^Done initializing outcome models & starting optimization sequence\\.\\.\\.$", messages))

  expect_true(all(is.finite(c(enlist_idx, structure_idx, summary_idx, optimize_idx))))
  expect_lt(enlist_idx, structure_idx)
  expect_lt(structure_idx, summary_idx)
  expect_lt(summary_idx, optimize_idx)
})

test_that("output-only neural post-fit logging reuses computed OOS metrics", {
  captured <- capture_messages(
    suppressWarnings(run_output_only_attn_vi_fit(
      seed = 20260406,
      early_stopping = FALSE,
      svi_steps = 5L,
      eval_enabled = TRUE
    ))
  )
  messages <- captured$messages

  metric_idx <- match(TRUE, grepl("^Neural fit metrics \\(.+oos_2fold.+\\): ", messages))
  optimize_idx <- match(TRUE, grepl("^Done initializing outcome models & starting optimization sequence\\.\\.\\.$", messages))

  expect_true(is.finite(metric_idx))
  expect_true(grepl("LogLoss=", messages[[metric_idx]]))
  expect_lt(metric_idx, optimize_idx)
})

test_that("average-case normal neural logging reports per-check validation nll summaries", {
  local_strategize_binding(
    "cs_compute_outcome_metrics",
    function(...) {
      list(log_loss = 1, nll = 1, rmse = 0.25, mae = 0.125)
    },
    env = environment()
  )

  captured <- capture_messages(
    suppressWarnings(run_average_case_neural_fit(
      seed = 20260407,
      svi_steps = 60L
    ))
  )
  messages <- captured$messages

  check_idx <- match(TRUE, grepl("^SVI early-stop check [0-9]+/[0-9]+: ", messages))
  early_idx <- match(TRUE, grepl("^SVI early stopping at step [0-9]+/[0-9]+ on validation nll=1\\.000000\\.$", messages))
  summary_idx <- match(TRUE, grepl("^SVI fit summary: steps=[0-9]+/[0-9]+; best nll=1\\.000000 at step [0-9]+\\.$", messages))

  expect_true(all(is.finite(c(check_idx, early_idx, summary_idx))))
  expect_match(
    messages[[check_idx]],
    "^SVI early-stop check [0-9]+/[0-9]+: step=[0-9]+/[0-9]+; validation nll=1\\.000000; train_elbo=(NA|-?[0-9]+\\.[0-9]{2}); best=1\\.000000 at step [0-9]+; delta_prev=(NA|\\+0\\.000000); iter_per_s=(NA|[0-9]+\\.[0-9]{3}); rss_mb=(NA|[0-9]+\\.[0-9]); elapsed=[0-9]+\\.[0-9]{3}s; grad_l2=[^;]+; grad_rms=[^;]+; grad_max=[^;]+; grad_bad=(NA|[0-9]+)\\.$"
  )
  expect_lt(check_idx, early_idx)
  expect_lt(early_idx, summary_idx)
})

test_that("output-only attn VI logging reports compact early-stop summaries when Muon stays active", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  if (!reticulate::py_has_attr(strategize:::strenv$optax, "contrib") ||
      !reticulate::py_has_attr(strategize:::strenv$optax$contrib, "muon")) {
    skip("optax.contrib.muon not available")
  }

  captured <- capture_messages(
    suppressWarnings(run_output_only_attn_vi_fit(
      seed = 20260408,
      svi_steps = 25L,
      optimizer = "muon"
    ))
  )
  messages <- captured$messages
  early_stop_lines <- grep("^SVI early-stop check [0-9]+/[0-9]+: ", messages, value = TRUE)

  expect_gt(length(early_stop_lines), 0L)
  expect_match(
    early_stop_lines[[1]],
    "train_elbo=(NA|-?[0-9]+\\.[0-9]{2}); best=-?[0-9]+\\.[0-9]{6} at step [0-9]+; delta_prev=NA; iter_per_s=(NA|[0-9]+\\.[0-9]{3}); rss_mb=(NA|[0-9]+\\.[0-9]); elapsed=[0-9]+\\.[0-9]{3}s; grad_l2=[^;]+; grad_rms=[^;]+; grad_max=[^;]+; grad_bad=(NA|[0-9]+)\\.$"
  )
  expect_false(any(grepl("param_rms=|param_delta_rms=|muon_mu_l2=|adam_mu_l2=|adam_nu_rms=", early_stop_lines)))
})

test_that("output-only attn VI logging uses the same compact summary for non-Muon optimizers", {
  captured <- capture_messages(
    suppressWarnings(run_output_only_attn_vi_fit(
      seed = 20260409,
      svi_steps = 25L,
      optimizer = "adabelief"
    ))
  )
  messages <- captured$messages
  early_stop_lines <- grep("^SVI early-stop check [0-9]+/[0-9]+: ", messages, value = TRUE)

  expect_gt(length(early_stop_lines), 0L)
  expect_match(
    early_stop_lines[[1]],
    "train_elbo=(NA|-?[0-9]+\\.[0-9]{2}); best=-?[0-9]+\\.[0-9]{6} at step [0-9]+; delta_prev=NA; iter_per_s=(NA|[0-9]+\\.[0-9]{3}); rss_mb=(NA|[0-9]+\\.[0-9]); elapsed=[0-9]+\\.[0-9]{3}s; grad_l2=[^;]+; grad_rms=[^;]+; grad_max=[^;]+; grad_bad=(NA|[0-9]+)\\.$"
  )
  expect_false(any(grepl("param_rms=|param_delta_rms=|muon_mu_l2=|adam_mu_l2=|adam_nu_rms=", early_stop_lines)))
})

test_that("output-only full-attn attn VI routes training through shared candidate extraction", {
  helper_calls <- 0L
  original_extract_candidate_tokens <- strategize:::neural_extract_candidate_tokens

  local_strategize_binding(
    "neural_extract_candidate_tokens",
    function(...) {
      helper_calls <<- helper_calls + 1L
      original_extract_candidate_tokens(...)
    },
    env = environment()
  )

  fit <- run_output_only_attn_vi_fit(
    seed = 20260404,
    early_stopping = FALSE,
    svi_steps = 5L,
    residual_mode = "full_attn"
  )
  model_info <- get_neural_model_info(fit)

  expect_identical(model_info$residual_mode, "full_attn")
  expect_identical(model_info$cross_candidate_encoder_mode, "attn")
  expect_gt(helper_calls, 0L)
})

test_that("SVI ELBO plot title reports the rounded last-20 finite mean", {
  expect_identical(
    strategize:::neural_format_svi_elbo_plot_title(seq_len(25)),
    "SVI ELBO Loss [15.5000]"
  )

  expect_identical(
    strategize:::neural_format_svi_elbo_plot_title(c(Inf, NA_real_, 2, 4, 6)),
    "SVI ELBO Loss [4.0000]"
  )

  expect_identical(
    strategize:::neural_format_svi_elbo_plot_title(c(NA_real_, Inf, -Inf)),
    "SVI ELBO Loss"
  )
})

test_that("output-only single-model normal batch_vi keeps the production step floor without inflating draws", {
  budget <- strategize:::neural_resolve_svi_budget(
    svi_steps_input = "optimal",
    svi_num_draws_input = 100L,
    user_supplied_svi_steps = FALSE,
    user_supplied_svi_num_draws = FALSE,
    n_obs = 2000L,
    n_factors = 10L,
    factor_levels = rep(2L, 10L),
    model_dims = 64L,
    model_depth = 2L,
    n_party_levels = 1L,
    n_resp_party_levels = 1L,
    n_resp_covariates = 0L,
    n_outcomes = 1L,
    pairwise_mode = FALSE,
    use_matchup_token = FALSE,
    use_cross_encoder = FALSE,
    use_cross_term = FALSE,
    use_cross_attn = FALSE,
    use_qk_norm = FALSE,
    batch_size = 512L,
    subsample_method = "batch_vi",
    output_only_mode = TRUE,
    likelihood = "normal"
  )

  expect_true(isTRUE(budget$used_optimal))
  expect_true(isTRUE(budget$used_default_svi_steps))
  expect_true(isTRUE(budget$used_default_svi_num_draws))
  expect_true(isTRUE(budget$output_single_normal_batch_vi))
  expect_true(isTRUE(budget$applied_output_single_normal_batch_vi_floor))
  expect_false(isTRUE(budget$applied_output_single_normal_batch_vi_draw_floor))
  expect_identical(as.integer(budget$svi_steps), 2000L)
  expect_identical(as.integer(budget$svi_num_draws), 100L)
})

test_that("explicit SVI overrides are honored even when below the production floor", {
  budget <- strategize:::neural_resolve_svi_budget(
    svi_steps_input = 200L,
    svi_num_draws_input = 100L,
    user_supplied_svi_steps = TRUE,
    user_supplied_svi_num_draws = TRUE,
    n_obs = 2000L,
    n_factors = 10L,
    factor_levels = rep(2L, 10L),
    model_dims = 64L,
    model_depth = 2L,
    n_party_levels = 1L,
    n_resp_party_levels = 1L,
    n_resp_covariates = 0L,
    n_outcomes = 1L,
    pairwise_mode = FALSE,
    use_matchup_token = FALSE,
    use_cross_encoder = FALSE,
    use_cross_term = FALSE,
    use_cross_attn = FALSE,
    use_qk_norm = FALSE,
    batch_size = 512L,
    subsample_method = "batch_vi",
    output_only_mode = TRUE,
    likelihood = "normal"
  )

  expect_false(isTRUE(budget$applied_output_single_normal_batch_vi_floor))
  expect_false(isTRUE(budget$applied_output_single_normal_batch_vi_draw_floor))
  expect_identical(as.integer(budget$svi_steps), 200L)
  expect_identical(as.integer(budget$svi_num_draws), 100L)
})

test_that("neural prior predictive probabilities are not overly concentrated", {
  fit <- get_neural_fit()
  res <- fit$res
  data <- fit$data
  p_list <- fit$p_list

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  model_env <- environment(model)
  strenv <- get("strenv", envir = model_env)
  likelihood <- get("likelihood", envir = model_env)
  if (!likelihood %in% c("bernoulli", "categorical")) {
    skip("Prior predictive check supports bernoulli/categorical only.")
  }
  pairwise_mode <- isTRUE(get("pairwise_mode", envir = model_env))
  model_fn <- if (pairwise_mode) {
    get("BayesianPairTransformerModel", envir = model_env)
  } else {
    get("BayesianSingleTransformerModel", envir = model_env)
  }

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  to_index_matrix <- if (exists("to_index_matrix", envir = model_env, inherits = TRUE)) {
    get("to_index_matrix", envir = model_env)
  } else {
    function(x_mat) {
      x_mat <- as.matrix(x_mat)
      if (anyNA(x_mat)) {
        x_mat[is.na(x_mat)] <- 1L
      }
      x_int <- matrix(as.integer(x_mat) - 1L, nrow = nrow(x_mat), ncol = ncol(x_mat))
      x_int[x_int < 0L] <- 0L
      x_int
    }
  }

  if (pairwise_mode) {
    idx_left <- which(data$profile_order == 1L)
    idx_right <- which(data$profile_order == 2L)
    X_left <- W_numeric[idx_left, , drop = FALSE]
    X_right <- W_numeric[idx_right, , drop = FALSE]
    n_obs <- nrow(X_left)
    X_left_jnp <- strenv$jnp$array(to_index_matrix(X_left))$astype(strenv$jnp$int32)
    X_right_jnp <- strenv$jnp$array(to_index_matrix(X_right))$astype(strenv$jnp$int32)
    party_left_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    party_right_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)
  } else {
    n_obs <- nrow(W_numeric)
    X_single_jnp <- strenv$jnp$array(to_index_matrix(W_numeric))$astype(strenv$jnp$int32)
    party_single_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
    resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)
  }

  coerce_prob_numeric <- function(x) {
    if (is.null(x)) {
      return(numeric(0))
    }
    out <- tryCatch(
      reticulate::py_to_r(strenv$np$asarray(x)),
      error = function(e) NULL
    )
    if (is.null(out)) {
      out <- tryCatch(
        reticulate::py_to_r(strenv$np$array(x)),
        error = function(e) NULL
      )
    }
    if (is.null(out)) {
      out <- tryCatch(
        reticulate::py_to_r(strenv$jax$device_get(x)),
        error = function(e) NULL
      )
    }
    if (is.null(out)) {
      out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
    }
    if (is.null(out)) {
      return(numeric(0))
    }
    if (is.list(out)) {
      out <- unlist(out, use.names = FALSE)
    }
    if (!is.numeric(out)) {
      return(numeric(0))
    }
    as.numeric(out)
  }

  model_info <- NULL
  if (!is.null(res$neural_model_info)) {
    if (!is.null(res$Y_models$my_model_ast_jnp)) {
      model_info <- res$neural_model_info$ast
    } else if (!is.null(res$Y_models$my_model_dag_jnp)) {
      model_info <- res$neural_model_info$dag
    }
  }
  if (is.null(model_info)) {
    skip("Neural model info unavailable for prior predictive check.")
  }
  model_dims <- as.integer(model_info$model_dims)
  model_depth <- as.integer(model_info$model_depth)
  n_heads <- as.integer(model_info$n_heads)
  head_dim <- as.integer(model_info$head_dim)
  cross_candidate_encoder <- isTRUE(model_info$cross_candidate_encoder)
  n_factors <- ncol(W_numeric)
  n_resp_covariates <- if (!is.null(model_info$n_resp_covariates)) {
    as.integer(model_info$n_resp_covariates)
  } else {
    0L
  }

  get_trace_value <- function(trace, name) {
    site <- tryCatch(trace[[name]], error = function(e) NULL)
    if (is.null(site)) {
      return(NULL)
    }
    val <- tryCatch(site$value, error = function(e) NULL)
    if (is.null(val)) {
      val <- site
    }
    val
  }

  build_params_from_trace <- function(trace) {
    params <- list()
    for (d_ in seq_len(n_factors)) {
      name <- paste0("E_factor_", d_)
      val <- get_trace_value(trace, name)
      if (is.null(val)) {
        raw <- get_trace_value(trace, paste0(name, "_raw"))
        if (!is.null(raw)) {
          n_real <- if (!is.null(model_info$factor_levels)) {
            as.integer(model_info$factor_levels[[d_]])
          } else {
            NA_integer_
          }
          n_raw <- tryCatch(
            as.integer(reticulate::py_to_r(raw$shape[[1]])),
            error = function(e) NA_integer_
          )
          if (!is.na(n_real) && !is.na(n_raw) && n_raw > n_real) {
            real_idx <- strenv$jnp$arange(as.integer(n_real))
            real_rows <- strenv$jnp$take(raw, real_idx, axis = 0L)
            real_mean <- strenv$jnp$mean(real_rows, axis = 0L, keepdims = TRUE)
            real_centered <- real_rows - real_mean
            missing_row <- strenv$jnp$take(raw, as.integer(n_real), axis = 0L)
            missing_row <- strenv$jnp$reshape(missing_row, list(1L, model_dims))
            val <- strenv$jnp$concatenate(list(real_centered, missing_row), axis = 0L)
          } else {
            val <- raw - strenv$jnp$mean(raw, axis = 0L, keepdims = TRUE)
          }
        }
      }
      params[[name]] <- val
    }
    base_names <- c(
      "E_feature_id", "E_party", "E_rel", "E_resp_party", "E_stage", "E_matchup", "E_choice",
      "E_sep", "E_segment",
      "W_resp_x", "pseudo_query_final", "RMS_final", "W_out", "b_out", "M_cross", "W_cross_out",
      "RMS_q_cross", "RMS_k_cross"
    )
    for (nm in base_names) {
      params[[nm]] <- get_trace_value(trace, nm)
    }
    if (is.null(params$E_feature_id)) {
      raw <- get_trace_value(trace, "E_feature_id_raw")
      if (!is.null(raw)) {
        params$E_feature_id <- strategize:::neural_center_token_rows(raw)
      }
    }
    if (is.null(params$E_segment)) {
      delta <- get_trace_value(trace, "E_segment_delta")
      if (!is.null(delta)) {
        params$E_segment <- strategize:::neural_build_symmetric_segment_embeddings(delta)
      }
    }
    for (l_ in seq_len(model_depth)) {
      params[[paste0("RMS_attn_l", l_)]] <- get_trace_value(trace, paste0("RMS_attn_l", l_))
      params[[paste0("RMS_q_l", l_)]] <- get_trace_value(trace, paste0("RMS_q_l", l_))
      params[[paste0("RMS_k_l", l_)]] <- get_trace_value(trace, paste0("RMS_k_l", l_))
      params[[paste0("RMS_ff_l", l_)]] <- get_trace_value(trace, paste0("RMS_ff_l", l_))
      params[[paste0("W_q_l", l_)]] <- get_trace_value(trace, paste0("W_q_l", l_))
      params[[paste0("W_k_l", l_)]] <- get_trace_value(trace, paste0("W_k_l", l_))
      params[[paste0("W_v_l", l_)]] <- get_trace_value(trace, paste0("W_v_l", l_))
      params[[paste0("W_o_l", l_)]] <- get_trace_value(trace, paste0("W_o_l", l_))
      params[[paste0("W_ff1_l", l_)]] <- get_trace_value(trace, paste0("W_ff1_l", l_))
      params[[paste0("W_ff2_l", l_)]] <- get_trace_value(trace, paste0("W_ff2_l", l_))
    }
    params
  }

  rms_norm <- function(x, g, eps = 1e-6) {
    if (is.null(g)) {
      return(x)
    }
    mean_sq <- strenv$jnp$mean(x * x, axis = -1L, keepdims = TRUE)
    inv_rms <- strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps))
    x * inv_rms * g
  }

  run_transformer <- function(tokens, params) {
    for (l_ in seq_len(model_depth)) {
      Wq <- params[[paste0("W_q_l", l_)]]
      Wk <- params[[paste0("W_k_l", l_)]]
      Wv <- params[[paste0("W_v_l", l_)]]
      Wo <- params[[paste0("W_o_l", l_)]]
      Wff1 <- params[[paste0("W_ff1_l", l_)]]
      Wff2 <- params[[paste0("W_ff2_l", l_)]]
      RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
      RMS_q <- params[[paste0("RMS_q_l", l_)]]
      RMS_k <- params[[paste0("RMS_k_l", l_)]]
      RMS_ff <- params[[paste0("RMS_ff_l", l_)]]

      tokens_norm <- rms_norm(tokens, RMS_attn)
      Q <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wq)
      K <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wk)
      V <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wv)

      Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]], n_heads, head_dim))
      Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]], n_heads, head_dim))
      Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]], n_heads, head_dim))
      Qh <- rms_norm(Qh, RMS_q)
      Kh <- rms_norm(Kh, RMS_k)
      scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(head_dim)))
      scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
      attn <- strenv$jax$nn$softmax(scores, axis = -1L)
      context_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
      context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]], context_h$shape[[2]], model_dims))
      attn_out <- strenv$jnp$einsum("ntm,mk->ntk", context, Wo)

      h1 <- tokens + attn_out
      h1_norm <- rms_norm(h1, RMS_ff)
      ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
      hidden_width <- strategize:::ai(ff_pre$shape[[3]]) %/% 2L
      gate_idx <- strenv$jnp$arange(0L, hidden_width)
      value_idx <- strenv$jnp$arange(hidden_width, strategize:::ai(ff_pre$shape[[3]]))
      gate <- strenv$jnp$take(ff_pre, gate_idx, axis = 2L)
      value <- strenv$jnp$take(ff_pre, value_idx, axis = 2L)
      ff_act <- strenv$jax$nn$swish(gate) * value
      ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
      tokens <- h1 + ff_out
    }
    rms_norm(tokens, params$RMS_final)
  }

  embed_candidate <- function(X_idx, party_idx, resp_p, params) {
    N_batch <- as.integer(X_idx$shape[[1]])
    D_local <- as.integer(X_idx$shape[[2]])
    token_list <- vector("list", D_local)
    for (d_ in seq_len(D_local)) {
      E_d <- params[[paste0("E_factor_", d_)]]
      idx_d <- strenv$jnp$take(X_idx, as.integer(d_ - 1L), axis = 1L)
      token_list[[d_]] <- strenv$jnp$take(E_d, idx_d, axis = 0L)
    }
    tokens <- strenv$jnp$stack(token_list, axis = 1L)
    if (!is.null(params$E_feature_id)) {
      feature_tok <- strenv$jnp$reshape(params$E_feature_id, list(1L, D_local, model_dims))
      tokens <- tokens + feature_tok
    }
    if (!is.null(params$E_party)) {
      party_tok <- strenv$jnp$take(params$E_party, party_idx, axis = 0L)
      party_tok <- strenv$jnp$reshape(party_tok, list(N_batch, 1L, model_dims))
      tokens <- strenv$jnp$concatenate(list(tokens, party_tok), axis = 1L)
    }
    if (!is.null(params$E_rel) && !is.null(model_info$cand_party_to_resp_idx)) {
      cand_map <- strenv$jnp$atleast_1d(model_info$cand_party_to_resp_idx)
      cand_resp_idx <- strenv$jnp$take(cand_map, party_idx, axis = 0L)
      is_known <- cand_resp_idx >= 0L
      is_match <- strenv$jnp$equal(cand_resp_idx, resp_p)
      rel_idx <- strenv$jnp$where(is_match, as.integer(0L),
                                  strenv$jnp$where(is_known, as.integer(1L), as.integer(2L)))
      rel_idx <- strenv$jnp$astype(rel_idx, strenv$jnp$int32)
      rel_tok <- strenv$jnp$take(params$E_rel, rel_idx, axis = 0L)
      rel_tok <- strenv$jnp$reshape(rel_tok, list(N_batch, 1L, model_dims))
      tokens <- strenv$jnp$concatenate(list(tokens, rel_tok), axis = 1L)
    }
    tokens
  }

  build_context_tokens <- function(stage_idx, resp_p, resp_c, matchup_idx, params) {
    N_batch <- as.integer(resp_p$shape[[1]])
    token_list <- list()
    if (!is.null(params$E_stage) && !is.null(stage_idx)) {
      stage_tok <- params$E_stage[resp_p, stage_idx]
      stage_tok <- strenv$jnp$reshape(stage_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- stage_tok
    }
    if (!is.null(params$E_resp_party)) {
      resp_tok <- strenv$jnp$take(params$E_resp_party, resp_p, axis = 0L)
      resp_tok <- strenv$jnp$reshape(resp_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- resp_tok
    }
    if (!is.null(params$E_matchup) && !is.null(matchup_idx)) {
      matchup_tok <- strenv$jnp$take(params$E_matchup, matchup_idx, axis = 0L)
      matchup_tok <- strenv$jnp$reshape(matchup_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- matchup_tok
    }
    if (!is.null(params$W_resp_x) && n_resp_covariates > 0L) {
      resp_cov_tok <- strenv$jnp$einsum("nc,cm->nm", resp_c, params$W_resp_x)
      resp_cov_tok <- strenv$jnp$reshape(resp_cov_tok, list(N_batch, 1L, model_dims))
      token_list[[length(token_list) + 1L]] <- resp_cov_tok
    }
    if (length(token_list) == 0L) {
      return(NULL)
    }
    strenv$jnp$concatenate(token_list, axis = 1L)
  }

  encode_candidate <- function(X_idx, party_idx, resp_p, resp_c, stage_idx, matchup_idx, params) {
    N_batch <- as.integer(X_idx$shape[[1]])
    choice_vec <- if (!is.null(params$E_choice)) {
      params$E_choice
    } else {
      strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj)
    }
    choice_tok <- strenv$jnp$reshape(choice_vec, list(1L, 1L, model_dims))
    choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
    ctx_tokens <- build_context_tokens(stage_idx, resp_p, resp_c, matchup_idx, params)
    cand_tokens <- embed_candidate(X_idx, party_idx, resp_p, params)
    if (!is.null(ctx_tokens)) {
      tokens <- strenv$jnp$concatenate(list(choice_tok, ctx_tokens, cand_tokens), axis = 1L)
    } else {
      tokens <- strenv$jnp$concatenate(list(choice_tok, cand_tokens), axis = 1L)
    }
    tokens <- run_transformer(tokens, params)
    choice_out <- strenv$jnp$take(tokens, strenv$jnp$arange(1L), axis = 1L)
    strenv$jnp$squeeze(choice_out, axis = 1L)
  }

  prior_predictive_probs <- function(trace) {
    site_name <- if (pairwise_mode) "obs_pair" else "obs_single"
    obs_site <- tryCatch(trace[[site_name]], error = function(e) NULL)
    if (is.null(obs_site)) {
      return(NULL)
    }
    obs_dist <- tryCatch(obs_site$fn, error = function(e) NULL)
    if (is.null(obs_dist)) {
      return(NULL)
    }
    probs <- tryCatch(obs_dist$probs, error = function(e) NULL)
    if (!is.null(probs)) {
      return(probs)
    }
    logits <- tryCatch(obs_dist$logits, error = function(e) NULL)
    if (is.null(logits)) {
      return(NULL)
    }
    if (likelihood == "bernoulli") {
      return(strenv$jax$nn$sigmoid(logits))
    }
    strenv$jax$nn$softmax(logits, axis = -1L)
  }

  n_draws <- 25L
  prob_samples <- numeric(0)
  for (i in seq_len(n_draws)) {
    rng_key <- strenv$jax$random$PRNGKey(as.integer(1000L + i))
    tracer <- strenv$numpyro$handlers$trace(
      strenv$numpyro$handlers$seed(model_fn, rng_key)
    )
    trace <- if (pairwise_mode) {
      tracer$get_trace(
        X_left = X_left_jnp,
        X_right = X_right_jnp,
        party_left = party_left_jnp,
        party_right = party_right_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        Y_obs = NULL
      )
    } else {
      tracer$get_trace(
        X = X_single_jnp,
        party = party_single_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        Y_obs = NULL
      )
    }
    prob <- prior_predictive_probs(trace)
    prob_samples <- c(
      prob_samples,
      coerce_prob_numeric(prob)
    )
  }

  prob_samples <- prob_samples[is.finite(prob_samples)]
  expect_true(length(prob_samples) > 0L)
  sd_prob <- stats::sd(prob_samples)
  expect_true(is.finite(sd_prob))
  expect_true(
    sd_prob >= 0.09,
    info = sprintf("Prior predictive SD %.3f below 0.09", sd_prob)
  )
})

test_that("non-pairwise neural prior trace handles NULL obs under minibatching", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "true",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))

  data <- generate_test_data(n = 24, seed = 20260329)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE
  params$outcome_model_type <- "neural"
  base_neural_control <- params$neural_mcmc_control
  if (is.null(base_neural_control)) {
    base_neural_control <- list()
  }
  params$neural_mcmc_control <- modifyList(
    base_neural_control,
    list(
      subsample_method = "batch_vi",
      batch_size = 16L,
      ModelDims = 16L,
      ModelDepth = 1L,
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  p_list <- generate_test_p_list(data$W)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  )))

  model <- res$Y_models$my_model_ast_jnp
  if (is.null(model)) {
    model <- res$Y_models$my_model_dag_jnp
  }
  expect_true(is.function(model))

  model_env <- environment(model)
  strenv <- get("strenv", envir = model_env)
  likelihood <- get("likelihood", envir = model_env)
  pairwise_mode <- isTRUE(get("pairwise_mode", envir = model_env))
  expect_identical(likelihood, "bernoulli")
  expect_false(pairwise_mode)

  model_fn <- get("BayesianSingleTransformerModel", envir = model_env)
  expect_true(is.function(model_fn))

  W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
    match(data$W[, d_], names(p_list[[d_]]))
  }))
  to_index_matrix <- if (exists("to_index_matrix", envir = model_env, inherits = TRUE)) {
    get("to_index_matrix", envir = model_env)
  } else {
    function(x_mat) {
      x_mat <- as.matrix(x_mat)
      if (anyNA(x_mat)) {
        x_mat[is.na(x_mat)] <- 1L
      }
      x_int <- matrix(as.integer(x_mat) - 1L, nrow = nrow(x_mat), ncol = ncol(x_mat))
      x_int[x_int < 0L] <- 0L
      x_int
    }
  }

  coerce_prob_numeric <- function(x) {
    if (is.null(x)) {
      return(numeric(0))
    }
    out <- tryCatch(
      reticulate::py_to_r(strenv$np$asarray(x)),
      error = function(e) NULL
    )
    if (is.null(out)) {
      out <- tryCatch(reticulate::py_to_r(x), error = function(e) NULL)
    }
    if (is.null(out)) {
      return(numeric(0))
    }
    if (is.list(out)) {
      out <- unlist(out, use.names = FALSE)
    }
    if (!is.numeric(out)) {
      return(numeric(0))
    }
    as.numeric(out)
  }

  n_obs <- nrow(W_numeric)
  X_single_jnp <- strenv$jnp$array(to_index_matrix(W_numeric))$astype(strenv$jnp$int32)
  party_single_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
  resp_party_jnp <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
  resp_cov_jnp <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)

  rng_key <- strenv$jax$random$PRNGKey(20260329L)
  tracer <- strenv$numpyro$handlers$trace(
    strenv$numpyro$handlers$seed(model_fn, rng_key)
  )
  trace <- tracer$get_trace(
    X = X_single_jnp,
    party = party_single_jnp,
    resp_party = resp_party_jnp,
    resp_cov = resp_cov_jnp,
    Y_obs = NULL
  )

  prob_obj <- tryCatch(trace$obs$fn$probs, error = function(e) NULL)
  if (is.null(prob_obj)) {
    logits_obj <- tryCatch(trace$obs$fn$logits, error = function(e) NULL)
    if (!is.null(logits_obj)) {
      prob_obj <- strenv$jax$nn$sigmoid(logits_obj)
    }
  }
  probs <- coerce_prob_numeric(prob_obj)

  expect_true(length(probs) > 0L)
  expect_true(all(is.finite(probs)))
  expect_true(all(probs >= 0 & probs <= 1))
})

test_that("neural outcome model exports cross-fitted OOS fit metrics", {
  fit <- get_neural_fit_perf()
  res <- fit$res

  info <- NULL
  if (!is.null(res$neural_model_info)) {
    info <- res$neural_model_info$ast
    if (is.null(info)) {
      info <- res$neural_model_info$dag
    }
  }
  if (is.null(info)) {
    skip("Neural model info unavailable for fit-metrics check.")
  }

  metrics <- info$fit_metrics
  diag_info <- format_oos_failure_details(
    metrics,
    stage_diagnostics = info$stage_diagnostics
  )
  expect_type(metrics, "list")
  expect_true(is.character(metrics$eval_note))
  expect_match(metrics$eval_note, "^oos_\\d+fold$")
  expect_equal(metrics$n_folds, 2L)
  expect_equal(metrics$seed, 123L)
  expect_true(!is.null(metrics$n_eval), info = diag_info)
  expect_true(is.numeric(metrics$n_eval) && length(metrics$n_eval) == 1L && !is.na(metrics$n_eval), info = diag_info)
  expect_true(metrics$n_eval >= 1000L, info = diag_info)
  expect_true(is.list(metrics$by_fold))
  expect_true(length(metrics$by_fold) >= 2L)
  expect_true(is.list(metrics$in_sample_metrics), info = diag_info)
  expect_false(isTRUE(info$stage_diagnostics$single_stage_only), info = diag_info)
  expect_false(isTRUE(info$stage_diagnostics$sparse_cells), info = diag_info)

  if (identical(metrics$likelihood, "bernoulli")) {
    expect_true(is.numeric(metrics$pred_quantiles), info = diag_info)
    expect_setequal(names(metrics$pred_quantiles), c("p05", "p25", "p50", "p75", "p95"))
    expect_setequal(names(metrics$confusion_0_5), c("tn", "fp", "fn", "tp"))
    expect_true(is.finite(metrics$log_loss), info = diag_info)
    expect_true(metrics$log_loss >= 0, info = diag_info)
  } else if (identical(metrics$likelihood, "categorical")) {
    expect_true(is.finite(metrics$log_loss), info = diag_info)
    expect_true(metrics$log_loss >= 0, info = diag_info)
  } else if (identical(metrics$likelihood, "normal")) {
    expect_true(is.finite(metrics$rmse), info = diag_info)
    expect_true(metrics$rmse >= 0, info = diag_info)
  }
})

test_that("neural pairwise OOS fit beats an intercept-only observable baseline", {
  fit <- get_neural_fit_perf()
  res <- fit$res
  data <- fit$data

  info <- get_neural_model_info(res)
  if (is.null(info) || is.null(info$fit_metrics)) {
    skip("Neural fit metrics unavailable for observable-fit check.")
  }

  metrics <- info$fit_metrics
  if (!identical(metrics$likelihood, "bernoulli")) {
    skip("Observable-fit baseline check currently supports bernoulli pairwise outcomes only.")
  }

  y_eval <- data$Y[data$profile_order == 1L]
  null_metrics <- compute_binary_null_metrics(y_eval)
  diag_info <- format_oos_failure_details(
    metrics,
    null_metrics = null_metrics,
    stage_diagnostics = info$stage_diagnostics,
    label = "Neural pairwise observable-baseline comparison"
  )

  expect_gte(length(y_eval), 1000L)
  expect_false(any(data$identical_pair))
  expect_true(all(data$pair_margin > 0))
  expect_false(isTRUE(info$stage_diagnostics$single_stage_only), info = diag_info)
  expect_false(isTRUE(info$stage_diagnostics$sparse_cells), info = diag_info)
  expect_gt(as.integer(info$low_rank_interaction_rank), 0L)
  expect_identical(info$cross_candidate_encoder_mode, "none")
  expect_identical(info$low_rank_logit_transform, "none")
  expect_identical(info$low_rank_logit_normalization, "rms")
  expect_equal(metrics$n_eval, length(y_eval))
  expect_true(is.finite(metrics$auc), info = diag_info)
  expect_true(metrics$auc > 0.5, info = diag_info)
  expect_true(metrics$log_loss < null_metrics$log_loss, info = diag_info)
  expect_true(metrics$brier < null_metrics$brier, info = diag_info)
})

test_that("non-pairwise AutoDelta runs on the average-case normal neural path", {
  fit <- run_average_case_neural_fit(
    vi_guide = "auto_delta",
    compute_se = FALSE,
    svi_steps = 5L
  )
  res <- fit$res
  model_info <- get_neural_model_info(res)

  expect_valid_strategize_output(res, n_factors = ncol(fit$data$W))
  expect_false(is.null(model_info))
  expect_false(isTRUE(model_info$pairwise_mode))
  expect_identical(model_info$likelihood, "normal")
  expect_true(all(is.finite(as.numeric(res$Q_point))))
})

test_that("full-data normal neural path runs with direct MCMC warm-start init", {
  skip_on_cran()
  skip_if_no_jax()

  withr::local_envvar(c(
    STRATEGIZE_NEURAL_FAST_MCMC = "false",
    STRATEGIZE_NEURAL_SKIP_EVAL = "true"
  ))

  data <- generate_average_case_neural_data(seed = 20260327)
  params <- default_strategize_params(fast = TRUE)
  params$diff <- FALSE
  params$force_gaussian <- TRUE
  params$compute_se <- FALSE
  params$outcome_model_type <- "neural"
  params$nMonte_Qglm <- 4L
  params$nSGD <- 1L
  params$optim_type <- "gd"
  base_neural_control <- params$neural_mcmc_control
  if (is.null(base_neural_control)) {
    base_neural_control <- list()
  }
  params$neural_mcmc_control <- modifyList(
    base_neural_control,
    list(
      subsample_method = "full",
      uncertainty_scope = "all",
      ModelDims = 8L,
      ModelDepth = 1L,
      n_samples_warmup = 1L,
      n_samples_mcmc = 1L,
      n_chains = 1L,
      chain_method = "sequential",
      eval_enabled = FALSE,
      warn_stage_imbalance_pct = 0,
      warn_min_cell_n = 0L
    )
  )

  p_list <- generate_test_p_list(data$W)
  res <- suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    params
  )))

  model_info <- get_neural_model_info(res)
  expect_valid_strategize_output(res, n_factors = ncol(data$W))
  expect_false(is.null(model_info))
  expect_identical(model_info$likelihood, "normal")
  expect_identical(model_info$gradient_diagnostics$gradient_status, "not_svi")
  expect_length(model_info$gradient_diagnostics$checkpoint_global_l2_norm, 0L)
  expect_true(all(is.finite(as.numeric(res$Q_point))))
})

test_that("AutoDelta is rejected when neural SEs are requested", {
  expect_error(
    run_average_case_neural_fit(
      vi_guide = "auto_delta",
      compute_se = TRUE,
      svi_steps = 5L
    ),
    "compute_se = TRUE is not supported when neural_mcmc_control\\$vi_guide = 'auto_delta'"
  )
})

test_that("average-case neural gaussian Q helper can enumerate exact single-party support", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  pi_ast <- strategize:::strenv$jnp$array(c(0.60, 0.40, 0.30, 0.70), dtype = strategize:::strenv$dtj)
  pi_dag <- strategize:::strenv$jnp$array(c(0.55, 0.45, 0.20, 0.80), dtype = strategize:::strenv$dtj)
  seed <- strategize:::strenv$jax$random$PRNGKey(123L)
  d_locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L))

  neural_draws <- strategize:::draw_average_case_q_profiles(
    pi_star_ast = pi_ast,
    pi_star_dag = pi_dag,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 4L,
    seed_in = seed,
    temperature = 0.5,
    ParameterizationType = "Full",
    d_locator_use = d_locator,
    sampler = strategize:::strenv$getMultinomialSamp,
    use_exact_support = TRUE,
    exact_support_single_party = TRUE
  )

  ast_profiles <- reticulate::py_to_r(strategize:::strenv$np$array(neural_draws$pi_star_ast_f_all))
  dag_profiles <- reticulate::py_to_r(strategize:::strenv$np$array(neural_draws$pi_star_dag_f_all))
  profile_weights <- as.numeric(reticulate::py_to_r(strategize:::strenv$np$array(neural_draws$profile_weights)))

  expect_true(isTRUE(neural_draws$use_mc_q))
  expect_true(isTRUE(neural_draws$exact_support))
  expect_identical(as.integer(neural_draws$n_draws), 4L)
  expect_identical(dim(ast_profiles), c(4L, 4L))
  expect_identical(dim(dag_profiles), c(4L, 4L))
  expect_equal(rowSums(ast_profiles[, 1:2, drop = FALSE]), rep(1, 4))
  expect_equal(rowSums(ast_profiles[, 3:4, drop = FALSE]), rep(1, 4))
  expect_true(all(vapply(seq_len(nrow(dag_profiles)), function(i) {
    isTRUE(all.equal(dag_profiles[i, ], c(0.55, 0.45, 0.20, 0.80), tolerance = 1e-6))
  }, logical(1))))
  expect_equal(sum(profile_weights), 1, tolerance = 1e-6)
})

test_that("multinomial group spec prefers concrete locator over stale globals", {
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors"))
  set_strenv_bindings(list(
    nUniqueFactors = 3L,
    nUniqueLevelsByFactors = c(2L, 2L, 2L)
  ))

  spec <- strategize:::resolve_multinomial_group_spec(
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L)),
    ParameterizationType = "Full"
  )

  expect_identical(spec$n_unique_factors, 1L)
  expect_identical(spec$n_unique_levels_by_factors, 2L)
})

test_that("multinomial group spec falls back to globals when locator unavailable", {
  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors"))
  set_strenv_bindings(list(
    nUniqueFactors = 2L,
    nUniqueLevelsByFactors = c(2L, 3L)
  ))

  spec <- strategize:::resolve_multinomial_group_spec(
    d_locator_use = NULL,
    ParameterizationType = "Full"
  )

  expect_identical(spec$n_unique_factors, 2L)
  expect_identical(spec$n_unique_levels_by_factors, c(2L, 3L))
})

test_that("multinomial group spec preserves implicit holdout levels", {
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors"))
  set_strenv_bindings(list(
    nUniqueFactors = 2L,
    nUniqueLevelsByFactors = c(4L, 4L)
  ))

  spec <- strategize:::resolve_multinomial_group_spec(
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L)),
    ParameterizationType = "Implicit"
  )

  expect_identical(spec$n_unique_factors, 1L)
  expect_identical(spec$n_unique_levels_by_factors, 3L)
})

test_that("Q evaluation resolver aligns non-adversarial neural objective/report to hard semantics", {
  avg_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L)),
    single_party = TRUE
  )
  avg_force_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L)),
    single_party = TRUE,
    force_reinforce = TRUE
  )
  avg_force_report <- strategize:::resolve_q_eval_spec(
    phase = "report",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L)),
    single_party = TRUE,
    force_reinforce = TRUE
  )
  avg_report <- strategize:::resolve_q_eval_spec(
    phase = "report",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L)),
    single_party = TRUE
  )
  adv_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = TRUE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    nMonte_adversarial = 5L
  )
  adv_large_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = TRUE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    nMonte_adversarial = 5L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(rep(seq_len(12L), each = 2L))
  )
  avg_large_obj <- strategize:::resolve_q_eval_spec(
    phase = "objective",
    adversarial = FALSE,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 9L,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(rep(seq_len(12L), each = 2L)),
    single_party = TRUE
  )
  glm_exact <- strategize:::resolve_q_eval_spec(
    phase = "report",
    adversarial = FALSE,
    outcome_model_type = "glm",
    glm_family = "gaussian",
    nMonte_Qglm = 9L
  )

  expect_identical(avg_obj$profile_draw_mode, "hard")
  expect_identical(avg_report$profile_draw_mode, "hard")
  expect_true(isTRUE(avg_obj$use_exact_support))
  expect_true(isTRUE(avg_report$use_exact_support))
  expect_identical(avg_obj$objective_gradient_mode, "exact")
  expect_identical(avg_report$objective_gradient_mode, "exact")
  expect_identical(as.integer(avg_obj$n_draws), 4L)
  expect_identical(as.integer(avg_report$n_draws), 4L)
  expect_identical(avg_force_obj$profile_draw_mode, "hard")
  expect_false(isTRUE(avg_force_obj$use_exact_support))
  expect_identical(avg_force_obj$objective_gradient_mode, "reinforce")
  expect_identical(avg_force_obj$n_draws, 9L)
  expect_true(isTRUE(avg_force_report$use_exact_support))
  expect_identical(avg_force_report$objective_gradient_mode, "exact")
  expect_identical(as.integer(avg_force_report$n_draws), 4L)
  expect_identical(adv_obj$profile_draw_mode, "relaxed")
  expect_identical(adv_obj$objective_gradient_mode, "pathwise")
  expect_identical(adv_obj$n_draws, 5L)
  expect_identical(avg_large_obj$profile_draw_mode, "hard")
  expect_false(isTRUE(avg_large_obj$use_exact_support))
  expect_true(isTRUE(avg_large_obj$is_large_support))
  expect_identical(avg_large_obj$objective_gradient_mode, "reinforce")
  expect_identical(adv_large_obj$profile_draw_mode, "hard")
  expect_true(isTRUE(adv_large_obj$is_large_support))
  expect_identical(adv_large_obj$objective_gradient_mode, "reinforce")
  expect_true(isTRUE(glm_exact$use_exact_q))
  expect_identical(glm_exact$profile_draw_mode, "exact")
})

test_that("strategize can force REINFORCE for small-support neural average-case optimization", {
  fit <- run_average_case_neural_fit(
    nMonte_Qglm = 8L,
    svi_steps = 20L,
    force_reinforce = TRUE,
    seed = 20260329
  )

  res <- fit$res
  info_msg <- sprintf(
    paste0(
      "objective_gradient_mode=%s; force_reinforce=%s; ",
      "reinforce_nonfinite_ast_steps=%d; Q_point=%s"
    ),
    res$convergence_history$objective_gradient_mode,
    as.character(res$force_reinforce),
    as.integer(res$convergence_history$reinforce_nonfinite_ast_steps),
    format(as.numeric(res$Q_point), digits = 6)
  )

  expect_true(isTRUE(res$force_reinforce), info = info_msg)
  expect_identical(res$convergence_history$objective_gradient_mode, "reinforce", info = info_msg)
  expect_identical(as.integer(res$convergence_history$reinforce_nonfinite_ast_steps), 0L, info = info_msg)
  expect_true(all(is.finite(as.numeric(res$Q_point))), info = info_msg)
})

test_that("hard profile draw mode emits one-hot average-case samples", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  pi_ast <- strategize:::strenv$jnp$array(c(0.25, 0.75), dtype = strategize:::strenv$dtj)
  pi_dag <- strategize:::strenv$jnp$array(c(0.60, 0.40), dtype = strategize:::strenv$dtj)
  seed <- strategize:::strenv$jax$random$PRNGKey(123L)
  local_strenv_bindings(c("nUniqueFactors", "nUniqueLevelsByFactors", "getMultinomialSampHard"))
  set_strenv_bindings(list(
    nUniqueFactors = 3L,
    nUniqueLevelsByFactors = c(2L, 2L, 2L),
    getMultinomialSampHard = strategize:::strenv$jax$jit(
      strategize:::getMultinomialSampHard_R,
      static_argnums = 3L,
      static_argnames = c("ParameterizationType")
    )
  ))

  hard_draws <- strategize:::draw_average_case_q_profiles(
    pi_star_ast = pi_ast,
    pi_star_dag = pi_dag,
    outcome_model_type = "neural",
    glm_family = "gaussian",
    nMonte_Qglm = 8L,
    seed_in = seed,
    temperature = 0.5,
    ParameterizationType = "Full",
    d_locator_use = strategize:::strenv$jnp$array(c(1L, 1L)),
    profile_draw_mode = "hard"
  )

  ast_mat <- unclass(strategize:::strenv$np$array(hard_draws$pi_star_ast_f_all))
  dag_mat <- unclass(strategize:::strenv$np$array(hard_draws$pi_star_dag_f_all))

  expect_true(all(ast_mat %in% c(0, 1)))
  expect_true(all(dag_mat %in% c(0, 1)))
  expect_identical(dim(ast_mat), c(8L, 2L))
  expect_identical(dim(dag_mat), c(8L, 2L))
  expect_true(all(rowSums(ast_mat) == 1))
  expect_true(all(rowSums(dag_mat) == 1))
})
