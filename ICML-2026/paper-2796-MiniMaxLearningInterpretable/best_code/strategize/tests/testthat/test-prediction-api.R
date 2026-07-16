prediction_test_foundation_control <- function() {
  modifyList(
    strategize:::cs_foundation_default_control(),
    list(
      neural_mcmc_control = list(
        ModelDims = 16L,
        ModelDepth = 1L,
        subsample_method = "batch_vi",
        uncertainty_scope = "output",
        optimizer = "adam",
        svi_steps = 8L,
        svi_num_draws = 2L,
        batch_size = 16L,
        early_stopping = FALSE
      )
    )
  )
}

prediction_test_experiment <- function(seed, experiment_id) {
  data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = seed)
  W_df <- as.data.frame(data$W, stringsAsFactors = FALSE)
  colnames(W_df) <- c("price", "message")
  list(
    experiment_id = experiment_id,
    experiment_description = paste("Experiment", experiment_id),
    Y = data$Y,
    W = W_df,
    X = data.frame(income = seq_len(nrow(W_df)) / nrow(W_df)),
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id
  )
}

prediction_test_cache_model_info <- function() {
  factor_names <- c("price", "message")
  factor_levels <- c(2L, 2L)
  struct <- strategize:::neural_make_default_fused_structural_info(
    factor_names = factor_names,
    factor_levels = factor_levels
  )
  model_info <- list(
    model_depth = 1L,
    model_dims = 8L,
    n_heads = 1L,
    head_dim = 8L,
    residual_mode = "standard",
    cross_candidate_encoder_mode = "none",
    likelihood = "bernoulli",
    experiment_token_mode = "description",
    factor_tokenization = "fused",
    n_factors = 2L,
    factor_levels = factor_levels,
    factor_struct_matrix = struct$factor_struct_matrix,
    factor_struct_feature_names = struct$factor_struct_feature_names,
    factor_struct_dim = ncol(struct$factor_struct_matrix),
    level_struct_matrices = struct$level_struct_matrices,
    level_struct_feature_names = struct$level_struct_feature_names,
    level_struct_dim = ncol(struct$level_struct_matrices[[1L]]),
    factor_schema_supplied = TRUE,
    default_factor_order = seq.int(0L, length(factor_names) - 1L),
    token_family_levels = strategize:::neural_token_family_levels(),
    max_factor_tokens = 2L,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "none",
    max_covariate_tokens = 0L,
    n_candidate_tokens = 2L,
    n_party_levels = 2L,
    cand_party_to_resp_idx = c(0L, 1L),
    text_semantic_dim = 2L
  )
  model_info
}

prediction_test_resp_cov_model_info <- function(strict = TRUE) {
  model_info <- prediction_test_cache_model_info()
  covariate_names <- c("local::target::income", "local::target::local_bonus")
  model_info$covariate_names <- covariate_names
  model_info$n_resp_covariates <- length(covariate_names)
  model_info$resp_cov_mean <- rep(0, length(covariate_names))
  model_info$resp_cov_default_present <- rep(0, length(covariate_names))
  model_info$default_covariate_order <- seq.int(0L, length(covariate_names) - 1L)
  model_info$max_covariate_tokens <- 4L
  model_info$shared_projection_value_encoder <- "legacy_scalar"
  model_info$covariate_aliases <- c(
    income = "local::target::income",
    local_bonus = "local::target::local_bonus"
  )
  model_info$foundation_prediction_schema <- list(
    strict_covariate_schema = strict,
    covariate_raw_to_internal = model_info$covariate_aliases
  )
  model_info$strict_prediction_covariate_schema <- strict
  model_info
}

prediction_test_text_embedding <- function(text) {
  text <- as.character(text)
  t(vapply(text, function(x) {
    bytes <- utf8ToInt(x)
    c(sum(bytes), length(bytes))
  }, numeric(2)))
}

test_that("neural prediction W preparation defaults unnamed matrices and rejects ambiguous names", {
  factor_names <- c("price", "message")

  unnamed <- matrix(c("low", "high", "short", "long"), nrow = 2L)
  prepped_unnamed <- strategize:::cs2step_neural_prepare_W_for_prediction(
    unnamed,
    factor_names
  )
  testthat::expect_identical(colnames(prepped_unnamed), factor_names)

  reordered <- data.frame(
    message = c("short", "long"),
    price = c("low", "high"),
    check.names = FALSE
  )
  prepped_reordered <- strategize:::cs2step_neural_prepare_W_for_prediction(
    reordered,
    factor_names
  )
  testthat::expect_identical(colnames(prepped_reordered), factor_names)
  testthat::expect_identical(as.character(prepped_reordered$price), c("low", "high"))

  partial <- data.frame(
    price = c("low", "high"),
    other = c("x", "y"),
    check.names = FALSE
  )
  testthat::expect_error(
    strategize:::cs2step_neural_prepare_W_for_prediction(partial, factor_names),
    "either all match"
  )

  unmatched <- data.frame(
    color = c("red", "blue"),
    slogan = c("jobs", "taxes"),
    check.names = FALSE
  )
  testthat::expect_error(
    strategize:::cs2step_neural_prepare_W_for_prediction(unmatched, factor_names),
    "factor names do not match"
  )
})

test_that("prediction-time X raw aliases activate foundation covariate tokens", {
  model_info <- prediction_test_resp_cov_model_info(strict = TRUE)
  x_new <- data.frame(
    income = c(1, 2),
    local_bonus = c(0.5, 0.75),
    check.names = FALSE
  )

  prepped <- strategize:::cs2step_neural_prepare_resp_cov(
    resp_cov_new = x_new,
    model_info = model_info,
    n_rows = 2L
  )
  expected_values <- as.matrix(x_new)
  colnames(expected_values) <- model_info$covariate_names
  expected_present <- matrix(
    1,
    nrow = 2L,
    ncol = 2L,
    dimnames = list(NULL, model_info$covariate_names)
  )

  testthat::expect_equal(
    prepped$values[, model_info$covariate_names, drop = FALSE],
    expected_values
  )
  testthat::expect_equal(
    prepped$present[, model_info$covariate_names],
    expected_present
  )
  testthat::expect_equal(prepped$order[, 1:2, drop = FALSE], cbind(c(0L, 0L), c(1L, 1L)))
})

test_that("prediction-time omitted respondent covariates stay missing under shared projection", {
  model_info <- prediction_test_resp_cov_model_info(strict = TRUE)
  model_info$resp_cov_mean <- c(10, 20)
  model_info$resp_cov_default_present <- c(1, 1)

  prepped_none <- strategize:::cs2step_neural_prepare_resp_cov(
    resp_cov_new = NULL,
    model_info = model_info,
    n_rows = 2L
  )
  expected_zero <- matrix(
    0,
    nrow = 2L,
    ncol = 2L,
    dimnames = list(NULL, model_info$covariate_names)
  )
  testthat::expect_equal(prepped_none$values, expected_zero)
  testthat::expect_equal(prepped_none$present, expected_zero)
  testthat::expect_equal(prepped_none$order[, 1:2, drop = FALSE], cbind(c(0L, 0L), c(1L, 1L)))
  testthat::expect_true(all(prepped_none$order[, 3:4, drop = FALSE] == -1L))

  prepped_partial <- strategize:::cs2step_neural_prepare_resp_cov(
    resp_cov_new = data.frame(income = c(1, NA), check.names = FALSE),
    model_info = model_info,
    n_rows = 2L
  )
  testthat::expect_equal(prepped_partial$values[, "local::target::income"], c(1, 0))
  testthat::expect_equal(prepped_partial$present[, "local::target::income"], c(1, 0))
  testthat::expect_equal(prepped_partial$values[, "local::target::local_bonus"], c(0, 0))
  testthat::expect_equal(prepped_partial$present[, "local::target::local_bonus"], c(0, 0))
  testthat::expect_equal(prepped_partial$order[, 1:2, drop = FALSE], cbind(c(0L, 0L), c(1L, 1L)))

  testthat::expect_error(
    strategize:::cs2step_neural_prepare_resp_cov(
      resp_cov_new = data.frame(income = c(Inf, 1), check.names = FALSE),
      model_info = model_info,
      n_rows = 2L
    ),
    "Inf"
  )
  testthat::expect_error(
    strategize:::cs2step_neural_prepare_resp_cov(
      resp_cov_new = data.frame(income = c("bad", "2"), check.names = FALSE),
      model_info = model_info,
      n_rows = 2L
    ),
    "numeric values or NA"
  )
})

test_that("prediction-time X rejects conflicting raw and internal covariate aliases", {
  model_info <- prediction_test_resp_cov_model_info(strict = TRUE)
  x_new <- data.frame(
    income = c(1, 2),
    check.names = FALSE
  )
  x_new[["local::target::income"]] <- c(1, 3)

  testthat::expect_error(
    strategize:::cs2step_neural_prepare_resp_cov(
      resp_cov_new = x_new,
      model_info = model_info,
      n_rows = 2L
    ),
    "Conflicting prediction-time X columns"
  )
})

test_that("strict foundation predictors reject unknown prediction-time X columns", {
  model_info <- prediction_test_resp_cov_model_info(strict = TRUE)
  x_new <- data.frame(
    income = c(1, 2),
    unknown_covariate = c(9, 9),
    check.names = FALSE
  )

  testthat::expect_error(
    strategize:::cs2step_neural_prepare_resp_cov(
      resp_cov_new = x_new,
      model_info = model_info,
      n_rows = 2L
    ),
    "Unknown prediction-time X columns"
  )
})

test_that("legacy predictors keep permissive no-match prediction-time X behavior", {
  model_info <- prediction_test_resp_cov_model_info(strict = FALSE)
  model_info$foundation_prediction_schema <- NULL
  model_info$covariate_aliases <- NULL
  model_info$strict_prediction_covariate_schema <- NULL
  x_new <- data.frame(income = c(1, 2), check.names = FALSE)

  prepped <- strategize:::cs2step_neural_prepare_resp_cov(
    resp_cov_new = x_new,
    model_info = model_info,
    n_rows = 2L
  )

  testthat::expect_equal(prepped$order[, 1:2, drop = FALSE], cbind(c(0L, 0L), c(1L, 1L)))
  testthat::expect_true(all(prepped$order[, 3:4, drop = FALSE] == -1L))
  testthat::expect_equal(
    prepped$values,
    matrix(0, nrow = 2L, ncol = 2L, dimnames = list(NULL, model_info$covariate_names))
  )
  testthat::expect_equal(
    prepped$present,
    matrix(0, nrow = 2L, ncol = 2L, dimnames = list(NULL, model_info$covariate_names))
  )
})

test_that("explicit neural factor_schema builds prediction-time semantic token metadata", {
  base_model <- prediction_test_cache_model_info()
  params <- list(
    W_factor_name_text = matrix(0, nrow = 2L, ncol = base_model$model_dims),
    W_level_name_text = matrix(0, nrow = 2L, ncol = base_model$model_dims)
  )
  object <- structure(
    list(
      model_type = "neural",
      metadata = list(text_embedding_fn = prediction_test_text_embedding)
    ),
    class = "strategic_predictor"
  )
  W_new <- data.frame(
    color = c("red", "blue", "green"),
    slogan = c("jobs", "taxes", "jobs"),
    check.names = FALSE
  )
  schema <- list(
    names_list = list(
      color = list(c("red", "blue", "green")),
      slogan = list(c("jobs", "taxes"))
    )
  )

  unpacked <- strategize:::cs2step_unpack_newdata(
    W_new,
    factor_names = c("price", "message"),
    mode = "single",
    factor_schema = schema
  )
  testthat::expect_identical(colnames(unpacked$W), c("color", "slogan"))

  prepared <- strategize:::cs2step_neural_prepare_factor_schema_prediction(
    object = object,
    W = W_new,
    model_info = base_model,
    params = params,
    factor_schema = schema
  )

  testthat::expect_identical(colnames(prepared$W), c("color", "slogan"))
  testthat::expect_equal(dim(prepared$W_idx), c(3L, 2L))
  testthat::expect_identical(prepared$model_info$n_factors, 2L)
  testthat::expect_identical(rownames(prepared$model_info$factor_name_text), c("color", "slogan"))
  testthat::expect_identical(
    rownames(prepared$model_info$level_name_text$color),
    c("red", "blue", "green", "__holdout__")
  )
  testthat::expect_identical(
    colnames(prepared$model_info$factor_struct_matrix),
    base_model$factor_struct_feature_names
  )
  identity_cols <- grep("^factor_identity_", colnames(prepared$model_info$factor_struct_matrix))
  if (length(identity_cols) > 0L) {
    testthat::expect_equal(sum(abs(prepared$model_info$factor_struct_matrix[, identity_cols, drop = FALSE])), 0)
  }
  testthat::expect_false(identical(
    strategize:::neural_model_jit_cache_key(base_model),
    strategize:::neural_model_jit_cache_key(prepared$model_info)
  ))

  bad_embedding <- function(text) {
    matrix(1, nrow = length(as.character(text)), ncol = 3L)
  }
  testthat::expect_error(
    strategize:::cs2step_neural_prepare_factor_schema_prediction(
      object = object,
      W = W_new,
      model_info = base_model,
      params = params,
      factor_schema = schema,
      text_embedding_fn = bad_embedding
    ),
    "incompatible width"
  )
})

test_that("strategic_prediction() fits GLM predictor (pairwise) and predicts probabilities", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 101)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  preds <- predict(
    fit,
    newdata = list(W = data$W, pair_id = data$pair_id, profile_order = data$profile_order)
  )

  testthat::expect_type(preds, "double")
  testthat::expect_length(preds, length(unique(data$pair_id)))
  testthat::expect_true(all(is.finite(preds)))
  testthat::expect_true(all(preds >= 0 & preds <= 1))
})

test_that("GLM pairwise predict preserves first-seen pair order across locales", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 606)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  candidate_pairs <- unique(data$pair_id)[seq_len(40L)]
  candidate_pred <- vapply(candidate_pairs, function(pair) {
    idx <- which(data$pair_id == pair)
    predict(
      fit,
      newdata = list(
        W = data$W[idx, , drop = FALSE],
        pair_id = rep("candidate_pair", length(idx)),
        profile_order = data$profile_order[idx]
      )
    )[[1L]]
  }, numeric(1))
  ordered_candidates <- order(candidate_pred)
  selected_pairs <- candidate_pairs[c(
    ordered_candidates[[length(ordered_candidates)]],
    ordered_candidates[[1L]],
    ordered_candidates[[ceiling(length(ordered_candidates) / 2L)]]
  )]

  rows <- unlist(lapply(selected_pairs, function(pair) {
    which(data$pair_id == pair)
  }), use.names = FALSE)
  first_seen_pair_id <- c("é_pair", "a_pair", "Z_pair")
  pair_id <- rep(first_seen_pair_id, each = 2L)
  W_new <- data$W[rows, , drop = FALSE]
  profile_order <- data$profile_order[rows]

  expected <- vapply(seq_along(first_seen_pair_id), function(i) {
    idx <- which(pair_id == first_seen_pair_id[[i]])
    predict(
      fit,
      newdata = list(
        W = W_new[idx, , drop = FALSE],
        pair_id = pair_id[idx],
        profile_order = profile_order[idx]
      )
    )[[1L]]
  }, numeric(1))
  testthat::expect_gt(diff(range(expected)), 1e-6)

  predict_batch <- function() {
    predict(
      fit,
      newdata = list(W = W_new, pair_id = pair_id, profile_order = profile_order)
    )
  }

  batch <- predict_batch()
  testthat::expect_equal(unname(batch), unname(expected), tolerance = 1e-10)

  old_collate <- Sys.getlocale("LC_COLLATE")
  withr::defer(Sys.setlocale("LC_COLLATE", old_collate))

  predict_under_locale <- function(locale) {
    set_locale <- suppressWarnings(Sys.setlocale("LC_COLLATE", locale))
    if (is.na(set_locale) || !nzchar(set_locale)) {
      return(NULL)
    }
    predict_batch()
  }

  c_batch <- predict_under_locale("C")
  testthat::expect_false(is.null(c_batch))
  testthat::expect_equal(unname(c_batch), unname(expected), tolerance = 1e-10)

  utf8_batch <- NULL
  for (locale in c("C.UTF-8", "en_US.UTF-8", "en_US.utf8")) {
    utf8_batch <- predict_under_locale(locale)
    if (!is.null(utf8_batch)) {
      break
    }
  }
  testthat::skip_if(is.null(utf8_batch), "No UTF-8 collation locale available")
  testthat::expect_equal(unname(utf8_batch), unname(expected), tolerance = 1e-10)
})

test_that("predict_pair() is approximately swap-invariant for symmetric data", {
  data <- generate_test_data(n = 2000, n_factors = 3, n_levels = 2, seed = 202)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  W_left <- as.data.frame(data$W[data$profile_order == 1L, , drop = FALSE])
  W_right <- as.data.frame(data$W[data$profile_order == 2L, , drop = FALSE])
  W_left <- W_left[seq_len(25), , drop = FALSE]
  W_right <- W_right[seq_len(25), , drop = FALSE]

  p_lr <- predict_pair(fit, W_left = W_left, W_right = W_right)
  p_rl <- predict_pair(fit, W_left = W_right, W_right = W_left)

  testthat::expect_true(all(is.finite(p_lr)))
  testthat::expect_true(all(is.finite(p_rl)))
  testthat::expect_equal(p_lr + p_rl, rep(1, length(p_lr)), tolerance = 0.05)
})

test_that("unseen levels map to holdout by default (GLM, single)", {
  data <- generate_test_data(n = 400, n_factors = 2, n_levels = 2, seed = 303)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )

  W_new <- as.data.frame(data$W[1:10, , drop = FALSE])
  base_level <- sort(unique(as.character(data$W[, 1])))[2]
  W_holdout <- W_new
  W_holdout[1, 1] <- base_level

  W_new[1, 1] <- "UNSEEN_LEVEL"

  p_unseen <- predict(fit, newdata = W_new)
  p_holdout <- predict(fit, newdata = W_holdout)

  testthat::expect_true(is.finite(p_unseen[1]))
  testthat::expect_equal(p_unseen[1], p_holdout[1], tolerance = 1e-10)
})

test_that("predict() intervals return expected structure (GLM)", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 404)

  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )

  ci <- predict(
    fit,
    newdata = list(W = data$W, pair_id = data$pair_id, profile_order = data$profile_order),
    interval = "ci",
    n_draws = 200,
    seed = 1
  )

  testthat::expect_s3_class(ci, "data.frame")
  testthat::expect_true(all(c("fit", "lo", "hi") %in% colnames(ci)))
  testthat::expect_true(all(is.finite(ci$fit)))
  testthat::expect_true(all(ci$lo <= ci$fit & ci$fit <= ci$hi, na.rm = TRUE))
})

test_that("as_function() returns a scoring closure", {
  data <- generate_test_data(n = 400, n_factors = 3, n_levels = 2, seed = 505)

  fit_single <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )
  f_single <- as_function(fit_single)
  p1 <- predict(fit_single, newdata = as.data.frame(data$W[1:15, , drop = FALSE]))
  p2 <- f_single(as.data.frame(data$W[1:15, , drop = FALSE]))
  testthat::expect_equal(p1, p2)

  fit_pair <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "pairwise",
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    use_regularization = FALSE
  )
  f_pair <- as_function(fit_pair)

  W_left <- as.data.frame(data$W[data$profile_order == 1L, , drop = FALSE])[1:10, , drop = FALSE]
  W_right <- as.data.frame(data$W[data$profile_order == 2L, , drop = FALSE])[1:10, , drop = FALSE]
  p3 <- predict_pair(fit_pair, W_left = W_left, W_right = W_right)
  p4 <- f_pair(W_left, W_right)
  testthat::expect_equal(p3, p4)
})

test_that("cs2step_unpack_newdata keeps pairwise group metadata separate from X", {
  newdata <- data.frame(
    price = c("A", "B"),
    message = c("A", "B"),
    pair_id = c(1L, 1L),
    profile_order = c(1L, 2L),
    competing_group_variable_candidate = c("PartyA", "PartyB"),
    competing_group_variable_respondent = c("PartyA", "PartyA"),
    experiment_country = c("USA", "CAN"),
    experiment_year = c(2024L, 2025L),
    income = c(0.1, 0.2),
    stringsAsFactors = FALSE
  )

  unpacked <- strategize:::cs2step_unpack_newdata(
    newdata = newdata,
    factor_names = c("price", "message"),
    mode = "pairwise"
  )

  testthat::expect_identical(
    unpacked$competing_group_variable_candidate,
    c("PartyA", "PartyB")
  )
  testthat::expect_identical(
    unpacked$competing_group_variable_respondent,
    c("PartyA", "PartyA")
  )
  testthat::expect_identical(unpacked$experiment_country, c("USA", "CAN"))
  testthat::expect_identical(unpacked$experiment_year, c(2024L, 2025L))
  testthat::expect_identical(colnames(unpacked$X), "income")
})

test_that("country normalization accepts aliases and rejects unknown countries", {
  us_iso2 <- strategize:::cs2step_normalize_country("US")
  us_name <- strategize:::cs2step_normalize_country("United States")
  canada <- strategize:::cs2step_normalize_country("Canada")

  testthat::expect_identical(us_iso2$country_key, "USA")
  testthat::expect_identical(us_name$country_key, "USA")
  testthat::expect_identical(canada$country_key, "CAN")
  testthat::expect_false(isTRUE(strategize:::cs2step_normalize_country(NA_character_)$country_present))
  testthat::expect_error(
    strategize:::cs2step_normalize_country("Untied Stats"),
    "Unknown experiment_country.*Did you mean"
  )
})

test_that("prediction-time experiment_country accepts scalar and row-aligned values", {
  model_info <- list(place_feature_names = strategize:::neural_place_feature_names())
  emb_scalar <- strategize:::cs2step_neural_prepare_place_embedding(
    experiment_country = "Canada",
    model_info = model_info,
    n_rows = 3L
  )
  emb_rows <- strategize:::cs2step_neural_prepare_place_embedding(
    experiment_country = c("USA", NA_character_, "CAN"),
    model_info = model_info,
    n_rows = 3L
  )

  testthat::expect_equal(nrow(emb_scalar), 3L)
  testthat::expect_equal(unname(emb_rows[2L, "missing_country"]), 1)
  testthat::expect_false(isTRUE(all.equal(emb_rows[1L, ], emb_rows[3L, ])))
  testthat::expect_error(
    strategize:::cs2step_neural_prepare_place_embedding(
      experiment_country = c("USA", "CAN"),
      model_info = model_info,
      n_rows = 3L
    ),
    "length one or the same number of rows"
  )
})

test_that("prediction-time experiment_year accepts scalar and row-aligned values", {
  emb_scalar <- strategize:::cs2step_neural_prepare_time_embedding(
    experiment_year = 2030,
    model_info = list(time_feature_names = strategize:::neural_time_feature_names()),
    n_rows = 3L
  )
  emb_rows <- strategize:::cs2step_neural_prepare_time_embedding(
    experiment_year = c(2024L, NA_integer_, 2026L),
    model_info = list(time_feature_names = strategize:::neural_time_feature_names()),
    n_rows = 3L
  )

  testthat::expect_equal(nrow(emb_scalar), 3L)
  testthat::expect_equal(
    unname(emb_scalar[, "linear_year_2000_25"]),
    rep((2030 - 2000) / 25, 3L),
    tolerance = 1e-8
  )
  testthat::expect_equal(unname(emb_rows[2L, "missing_year"]), 1)
  testthat::expect_error(
    strategize:::cs2step_neural_prepare_time_embedding(
      experiment_year = c(2024L, 2025L),
      model_info = list(time_feature_names = strategize:::neural_time_feature_names()),
      n_rows = 3L
    ),
    "length one or the same number of rows"
  )
  testthat::expect_error(
    strategize:::cs2step_neural_prepare_time_embedding(
      experiment_year = 2025.5,
      model_info = list(time_feature_names = strategize:::neural_time_feature_names()),
      n_rows = 1L
    ),
    "integer-like"
  )
})

test_that("pairwise row-aligned experiment_country uses the left profile row", {
  skip_if_no_jax()
  strategize:::initialize_jax()

  model_info <- list(
    factor_levels = 1L,
    party_levels = strategize:::neural_missing_group_label("candidate"),
    resp_party_levels = strategize:::neural_missing_group_label("respondent"),
    party_missing_label = strategize:::neural_missing_group_label("candidate"),
    resp_party_missing_label = strategize:::neural_missing_group_label("respondent"),
    covariate_names = character(0),
    place_context_enabled = TRUE,
    place_feature_names = strategize:::neural_place_feature_names(),
    factor_tokenization = "fused",
    max_factor_tokens = 1L
  )
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 1L,
    factor_names = "factor_1",
    max_factor_tokens = 1L
  )
  model_info$place_context_enabled <- TRUE
  model_info$has_place_context <- TRUE
  model_info$place_feature_names <- strategize:::neural_place_feature_names()
  prep <- strategize:::cs2step_neural_prepare_prediction_data(
    W_idx = matrix(1L, nrow = 4L, ncol = 1L),
    model_info = model_info,
    experiment_country = c("USA", "CAN", "USA", "CAN"),
    pair_id = c(1L, 1L, 2L, 2L),
    profile_order = c(1L, 2L, 2L, 1L),
    mode = "pairwise"
  )
  place_r <- reticulate::py_to_r(strategize:::strenv$np$array(prep$place_embedding))
  expected <- strategize:::cs2step_neural_prepare_place_embedding(
    experiment_country = c("USA", "CAN"),
    model_info = model_info,
    n_rows = 2L
  )

  testthat::expect_equal(unname(place_r), unname(expected), tolerance = 1e-6)
})

test_that("old predictors error when experiment_country is supplied", {
  data <- generate_test_data(n = 80, n_factors = 2, n_levels = 2, seed = 609)
  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )
  newdata <- as.data.frame(data$W[1:3, , drop = FALSE])
  newdata$experiment_country <- "Canada"

  testthat::expect_error(
    predict(fit, newdata = newdata),
    "neural predictor trained with place context"
  )
})

test_that("cs2step_build_pair_mat preserves first-seen pair order", {
  W <- data.frame(
    price = c("A", "B", "A", "B", "A", "B"),
    message = c("X", "Y", "Y", "X", "X", "Y"),
    stringsAsFactors = FALSE
  )
  pair_id <- c("b_pair", "b_pair", "a_pair", "a_pair", "c_pair", "c_pair")
  profile_order <- c(2L, 1L, 1L, 2L, 2L, 1L)

  pair_info <- strategize:::cs2step_build_pair_mat(
    pair_id = pair_id,
    W = W,
    profile_order = profile_order
  )

  testthat::expect_identical(names(pair_info$pair_sizes), c("b_pair", "a_pair", "c_pair"))
  testthat::expect_identical(unname(pair_info$pair_sizes), c(2L, 2L, 2L))
  testthat::expect_identical(
    unname(pair_info$pair_mat),
    matrix(c(2L, 1L, 3L, 4L, 6L, 5L), ncol = 2L, byrow = TRUE)
  )
})

test_that("cs2step_build_pair_mat row hash avoids integer overflow warnings", {
  long_value <- paste(rep("long-row-token", 5000L), collapse = "|")
  W <- data.frame(
    price = c(long_value, paste0(long_value, "-right"), long_value, paste0(long_value, "-alt")),
    stringsAsFactors = FALSE
  )
  pair_id <- c("b_pair", "b_pair", "a_pair", "a_pair")
  profile_order <- c(2L, 1L, 1L, 2L)

  pair_info <- testthat::expect_warning(
    strategize:::cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W,
      profile_order = profile_order
    ),
    NA
  )
  pair_info_again <- strategize:::cs2step_build_pair_mat(
    pair_id = pair_id,
    W = W,
    profile_order = profile_order
  )

  testthat::expect_identical(pair_info$pair_mat, pair_info_again$pair_mat)
  testthat::expect_identical(names(pair_info$pair_sizes), c("b_pair", "a_pair"))
  testthat::expect_identical(
    unname(pair_info$pair_mat),
    matrix(c(2L, 1L, 3L, 4L), ncol = 2L, byrow = TRUE)
  )
})

test_that("cs2step_build_pair_mat is stable across collation locales", {
  W <- data.frame(
    price = c("A", "B", "A", "B", "A", "B"),
    message = c("X", "Y", "Y", "X", "X", "Y"),
    stringsAsFactors = FALSE
  )
  pair_id <- c("é_pair", "é_pair", "a_pair", "a_pair", "Z_pair", "Z_pair")
  profile_order <- c(1L, 2L, 1L, 2L, 1L, 2L)
  old_collate <- Sys.getlocale("LC_COLLATE")
  withr::defer(Sys.setlocale("LC_COLLATE", old_collate))

  build_under_locale <- function(locale) {
    set_locale <- suppressWarnings(Sys.setlocale("LC_COLLATE", locale))
    if (is.na(set_locale) || !nzchar(set_locale)) {
      return(NULL)
    }
    strategize:::cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W,
      profile_order = profile_order
    )
  }

  c_pair_info <- build_under_locale("C")
  testthat::expect_false(is.null(c_pair_info))
  utf8_pair_info <- NULL
  for (locale in c("C.UTF-8", "en_US.UTF-8", "en_US.utf8")) {
    utf8_pair_info <- build_under_locale(locale)
    if (!is.null(utf8_pair_info)) {
      break
    }
  }
  testthat::skip_if(is.null(utf8_pair_info), "No UTF-8 collation locale available")

  testthat::expect_identical(utf8_pair_info$pair_mat, c_pair_info$pair_mat)
  testthat::expect_identical(utf8_pair_info$pair_sizes, c_pair_info$pair_sizes)
  testthat::expect_identical(names(c_pair_info$pair_sizes), unique(pair_id))
})

test_that("neural_model_jit_cache_key ignores legacy jit_cache_key noise", {
  model_a <- prediction_test_cache_model_info()
  model_b <- prediction_test_cache_model_info()
  model_a$jit_cache_key <- "legacy_random_a"
  model_b$jit_cache_key <- "legacy_random_b"

  testthat::expect_identical(
    strategize:::neural_model_jit_cache_key(model_a),
    strategize:::neural_model_jit_cache_key(model_b)
  )
})

test_that("experiment description content changes deterministic neural jit keys", {
  base_model <- prediction_test_cache_model_info()
  same_a <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "alpha experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )
  same_b <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "alpha experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )
  other <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "beta experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )

  testthat::expect_identical(
    strategize:::neural_model_jit_cache_key(same_a),
    strategize:::neural_model_jit_cache_key(same_b)
  )
  testthat::expect_false(identical(
    strategize:::neural_model_jit_cache_key(same_a),
    strategize:::neural_model_jit_cache_key(other)
  ))
})

test_that("upgrade path strips legacy jit cache keys", {
  upgraded <- strategize:::cs2step_neural_upgrade_model_info(list(
    jit_cache_key = "legacy_random_value"
  ))

  testthat::expect_null(upgraded$jit_cache_key)
})

test_that("legacy neural model upgrades use portable attention defaults", {
  upgraded <- strategize:::cs2step_neural_upgrade_model_info(list())

  testthat::expect_identical(upgraded$attention_backend, "xla")
  testthat::expect_identical(upgraded$attention_dtype, "float32")
  testthat::expect_identical(upgraded$attention_padding_multiple, 8L)
  testthat::expect_identical(upgraded$attention_resolved_backend, "xla")
  testthat::expect_true(is.na(upgraded$attention_fallback_reason))
  testthat::expect_identical(upgraded$low_rank_interaction_rank, 0L)
  testthat::expect_identical(upgraded$low_rank_logit_transform, "none")
  testthat::expect_null(upgraded$low_rank_logit_bound)
  testthat::expect_null(upgraded$low_rank_logit_softness)
  testthat::expect_identical(upgraded$low_rank_logit_normalization, "none")
  testthat::expect_null(upgraded$low_rank_head_weight_target_rms)
  testthat::expect_null(upgraded$low_rank_rc_out_target_rms)
  testthat::expect_identical(upgraded$additive_utility_mode, "off")
  testthat::expect_identical(upgraded$additive_utility_normalization, "none")
  testthat::expect_null(upgraded$additive_head_weight_target_rms)
  testthat::expect_false(upgraded$learned_pairwise_bernoulli_logit_scale)
  testthat::expect_equal(upgraded$pairwise_bernoulli_logit_scale, 1)
  testthat::expect_null(upgraded$pairwise_bernoulli_logit_scale_prior_sd)
  testthat::expect_false(upgraded$has_respondent_cls)
  testthat::expect_false(upgraded$has_candidate_cls)
  testthat::expect_false(upgraded$has_low_rank_interaction)
  testthat::expect_identical(
    upgraded$readout_embedding_families,
    "choice"
  )
})

test_that("neural model packing preserves readout and low-rank metadata", {
  rank_zero <- strategize:::cs2step_neural_pack_model_info(
    list(
      low_rank_interaction_rank = 0L,
      has_respondent_cls = TRUE,
      has_candidate_cls = TRUE,
      has_low_rank_interaction = TRUE,
      token_family_levels = c("factor_candidate", "respondent_cls", "candidate_cls", "choice"),
      readout_embedding_families = c("choice", "respondent_cls", "candidate_cls")
    ),
    drop_params = TRUE
  )

  testthat::expect_identical(rank_zero$low_rank_interaction_rank, 0L)
  testthat::expect_false(rank_zero$has_respondent_cls)
  testthat::expect_false(rank_zero$has_candidate_cls)
  testthat::expect_false(rank_zero$has_low_rank_interaction)
  testthat::expect_false("respondent_cls" %in% rank_zero$token_family_levels)
  testthat::expect_false("candidate_cls" %in% rank_zero$token_family_levels)
  testthat::expect_identical(rank_zero$readout_embedding_families, "choice")

  model_info <- list(
    low_rank_interaction_rank = 8L,
    low_rank_logit_transform = "softclip",
    low_rank_logit_bound = 1.5,
    low_rank_logit_softness = 0.25,
    low_rank_logit_normalization = "rms",
    low_rank_head_weight_target_rms = 1 / (sqrt(2) * 16),
    low_rank_rc_out_target_rms = 1 / (sqrt(2) * 8),
    additive_utility_mode = "on",
    additive_utility_normalization = "rms",
    additive_head_weight_target_rms = 0.25,
    has_additive_utility = TRUE,
    learned_pairwise_bernoulli_logit_scale = TRUE,
    pairwise_bernoulli_logit_scale_prior_sd = 0.5,
    pairwise_bernoulli_logit_scale = 3.25,
    has_respondent_cls = TRUE,
    has_candidate_cls = TRUE,
    has_low_rank_interaction = TRUE,
    readout_embedding_families = c("choice", "respondent_cls", "candidate_cls")
  )

  packed <- strategize:::cs2step_neural_pack_model_info(model_info, drop_params = TRUE)

  testthat::expect_identical(packed$low_rank_interaction_rank, 8L)
  testthat::expect_identical(packed$low_rank_logit_transform, "softclip")
  testthat::expect_equal(packed$low_rank_logit_bound, 1.5)
  testthat::expect_equal(packed$low_rank_logit_softness, 0.25)
  testthat::expect_identical(packed$low_rank_logit_normalization, "rms")
  testthat::expect_equal(packed$low_rank_head_weight_target_rms, 1 / (sqrt(2) * 16))
  testthat::expect_equal(packed$low_rank_rc_out_target_rms, 1 / (sqrt(2) * 8))
  testthat::expect_identical(packed$additive_utility_mode, "on")
  testthat::expect_identical(packed$additive_utility_normalization, "rms")
  testthat::expect_equal(packed$additive_head_weight_target_rms, 0.25)
  testthat::expect_true(packed$has_additive_utility)
  testthat::expect_true(packed$learned_pairwise_bernoulli_logit_scale)
  testthat::expect_equal(packed$pairwise_bernoulli_logit_scale_prior_sd, 0.5)
  testthat::expect_equal(packed$pairwise_bernoulli_logit_scale, 3.25)
  testthat::expect_true(packed$has_respondent_cls)
  testthat::expect_true(packed$has_candidate_cls)
  testthat::expect_true(packed$has_low_rank_interaction)
  testthat::expect_identical(
    packed$readout_embedding_families,
    c("choice", "respondent_cls", "candidate_cls")
  )
})

test_that("mixed pairwise prediction coercion applies learned Bernoulli scale once", {
  model_info <- list(
    likelihood = "mixed",
    learned_pairwise_bernoulli_logit_scale = TRUE,
    pairwise_bernoulli_logit_scale = 4,
    low_rank_interaction_rank = 0L,
    low_rank_logit_transform = "none"
  )
  pred <- matrix(c(0.25, -0.25), ncol = 1L)

  pairwise_prob <- strategize:::cs2step_neural_coerce_prediction_output(
    pred,
    likelihood = "mixed",
    target_likelihood = "bernoulli",
    model_info = model_info,
    pairwise_prediction = TRUE
  )
  single_prob <- strategize:::cs2step_neural_coerce_prediction_output(
    pred,
    likelihood = "mixed",
    target_likelihood = "bernoulli",
    model_info = model_info,
    pairwise_prediction = FALSE
  )

  testthat::expect_equal(pairwise_prob, stats::plogis(c(1, -1)))
  testthat::expect_equal(single_prob, stats::plogis(c(0.25, -0.25)))
})

test_that("neural jit wrapper cache reuses identical experiment descriptions", {
  skip_on_cran()
  skip_if_no_jax()

  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }

  cache_env <- strategize:::neural_prediction_jit_cache
  base_model <- prediction_test_cache_model_info()
  model_same <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "alpha experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )
  model_other <- strategize:::cs2step_neural_apply_experiment_description(
    model_info = base_model,
    experiment_description = "beta experiment",
    n_rows = 1L,
    text_embedding_fn = prediction_test_text_embedding
  )

  same_entry <- paste(
    "predict",
    strategize:::neural_model_jit_cache_key(model_same),
    "single",
    "response",
    sep = "::"
  )
  other_entry <- paste(
    "predict",
    strategize:::neural_model_jit_cache_key(model_other),
    "single",
    "response",
    sep = "::"
  )
  existed_same <- exists(same_entry, envir = cache_env, inherits = FALSE)
  existed_other <- exists(other_entry, envir = cache_env, inherits = FALSE)
  on.exit({
    if (!existed_same && exists(same_entry, envir = cache_env, inherits = FALSE)) {
      rm(list = same_entry, envir = cache_env)
    }
    if (!existed_other && exists(other_entry, envir = cache_env, inherits = FALSE)) {
      rm(list = other_entry, envir = cache_env)
    }
  }, add = TRUE)

  strategize:::neural_get_predict_jit(
    model_info = model_same,
    pairwise = FALSE,
    return_logits = FALSE
  )
  entries_after_first <- ls(envir = cache_env, all.names = TRUE)

  strategize:::neural_get_predict_jit(
    model_info = model_same,
    pairwise = FALSE,
    return_logits = FALSE
  )
  entries_after_second <- ls(envir = cache_env, all.names = TRUE)

  strategize:::neural_get_predict_jit(
    model_info = model_other,
    pairwise = FALSE,
    return_logits = FALSE
  )
  entries_after_third <- ls(envir = cache_env, all.names = TRUE)

  testthat::expect_true(same_entry %in% entries_after_first)
  testthat::expect_identical(entries_after_second, entries_after_first)
  testthat::expect_true(other_entry %in% entries_after_third)
  testthat::expect_false(identical(same_entry, other_entry))
})

test_that("stage-aware neural predictors require long-format predict()", {
  predictor <- structure(
    list(
      model_type = "neural",
      mode = "pairwise",
      encoder = list(
        factor_names = c("price", "message"),
        names_list = list(price = c("A", "B"), message = c("A", "B")),
        factor_levels = c(price = 2L, message = 2L)
      ),
      fit = list(
        neural_model_info = list(pairwise_context_mode = "stage_aware")
      ),
      metadata = list()
    ),
    class = "strategic_predictor"
  )
  W_left <- data.frame(price = "A", message = "A")
  W_right <- data.frame(price = "B", message = "B")

  testthat::expect_error(
    predict_pair(predictor, W_left = W_left, W_right = W_right),
    "long-format newdata"
  )

  predictor$fit$neural_model_info <- list(
    has_stage_context = TRUE,
    has_matchup_token = TRUE
  )
  testthat::expect_error(
    predict_pair(predictor, W_left = W_left, W_right = W_right),
    "competing_group_variable_candidate"
  )
})

test_that("neural backend errors with build_backend() hint when unavailable", {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    testthat::skip("reticulate not available")
  }
  conda_list <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
  if (!is.null(conda_list) && "strategize_env" %in% conda_list$name) {
    jax_available <- tryCatch({
      reticulate::use_condaenv("strategize_env", required = TRUE)
      reticulate::py_module_available("jax")
    }, error = function(e) FALSE)
    if (isTRUE(jax_available)) {
      testthat::skip("JAX available; neural fit is slow and not exercised here")
    }
  }

  data <- generate_test_data(n = 40, n_factors = 2, n_levels = 2, seed = 606)
  testthat::expect_error(
    strategic_prediction(
      Y = data$Y,
      W = data$W,
      model = "neural",
      mode = "pairwise",
      pair_id = data$pair_id,
      profile_order = data$profile_order,
      conda_env_required = TRUE
    ),
    "build_backend"
  )
})

test_that("prediction API evaluates model generator bodies directly", {
  glm_src <- paste(deparse(strategize:::cs2step_eval_outcome_model_glm), collapse = "\n")
  neural_src <- paste(deparse(strategize:::cs2step_eval_outcome_model_neural), collapse = "\n")
  master_src <- paste(deparse(strategize::strategize), collapse = "\n")

  expect_false(grepl("deparse(generate_ModelOutcome)", glm_src, fixed = TRUE))
  expect_false(grepl("deparse(generate_ModelOutcome_neural)", neural_src, fixed = TRUE))
  expect_false(grepl("deparse(generate_ModelOutcome)", master_src, fixed = TRUE))
  expect_false(grepl("deparse(generate_ModelOutcome_neural)", master_src, fixed = TRUE))
  expect_true(grepl("eval(body(generate_ModelOutcome)", glm_src, fixed = TRUE))
  expect_true(grepl("eval(body(generate_ModelOutcome_neural)", neural_src, fixed = TRUE))
})
