embedding_test_neural_control <- function(cross_candidate_encoder = NULL,
                                          low_rank_interaction_rank = NULL) {
  control <- list(
    ModelDims = 12L,
    ModelDepth = 1L,
    subsample_method = "batch_vi",
    uncertainty_scope = "output",
    optimizer = "adam",
    svi_steps = 6L,
    svi_num_draws = 2L,
    batch_size = 16L,
    early_stopping = FALSE
  )
  if (!is.null(cross_candidate_encoder)) {
    control$cross_candidate_encoder <- cross_candidate_encoder
  }
  if (!is.null(low_rank_interaction_rank)) {
    control$low_rank_interaction_rank <- low_rank_interaction_rank
  }
  control
}

embedding_test_text_embedding_fn <- function(x) {
  x <- as.character(x)
  cbind(
    nchar = nchar(x),
    has_space = 1 * grepl("\\s", x),
    has_upper = 1 * grepl("[A-Z]", x)
  )
}

embedding_test_covariates <- function(W_df, pair_id = NULL) {
  n <- nrow(W_df)
  if (is.null(pair_id)) {
    idx <- seq_len(n)
    scale_n <- max(n, 1L)
  } else {
    # Respondent covariates must be constant within each forced-choice pair
    # (one respondent evaluates both candidates); per-row or candidate-
    # attribute-dependent values trip the backend's pair-constancy
    # validation, so index everything by pair when pair_id is supplied.
    idx <- as.integer(factor(pair_id))
    scale_n <- max(idx, 1L)
  }
  data.frame(
    income = idx / scale_n,
    `household size` = 1 + (idx %% 4L) +
      if (is.null(pair_id)) 0.5 * (W_df[[1]] == "B") else 0,
    GOPScore = if (is.null(pair_id)) {
      as.numeric(W_df[[ncol(W_df)]] == "B") - 0.5
    } else {
      ((idx %% 3L) - 1) / 2
    },
    local_bonus = seq(-1, 1, length.out = scale_n)[idx],
    check.names = FALSE
  )
}

embedding_test_foundation_control <- function(with_embeddings = TRUE) {
  control <- strategize:::cs_foundation_default_control()
  control$neural_mcmc_control <- embedding_test_neural_control()
  if (isTRUE(with_embeddings)) {
    control$text_embedding_fn <- embedding_test_text_embedding_fn
  } else {
    control$add_text_semantics <- FALSE
    control$text_embedding_fn <- NULL
  }
  control
}

embedding_test_pairwise_experiment <- function(seed,
                                               experiment_id,
                                               factor_names,
                                               x_names = NULL,
                                               canonical_factor_id = NULL) {
  data <- generate_test_data(
    n = 40,
    n_factors = length(factor_names),
    n_levels = 2,
    seed = seed
  )
  W_df <- as.data.frame(data$W, stringsAsFactors = FALSE)
  colnames(W_df) <- factor_names
  if (is.null(canonical_factor_id)) {
    canonical_factor_id <- stats::setNames(factor_names, factor_names)
  }
  x_full <- embedding_test_covariates(W_df, pair_id = data$pair_id)
  X <- if (is.null(x_names) || length(x_names) < 1L) {
    NULL
  } else {
    x_full[, x_names, drop = FALSE]
  }
  list(
    experiment_id = experiment_id,
    experiment_description = paste(
      "Experiment", experiment_id, "with factors", paste(factor_names, collapse = ", ")
    ),
    Y = data$Y,
    W = W_df,
    X = X,
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    canonical_factor_id = canonical_factor_id
  )
}

embedding_test_single_experiment <- function(seed,
                                             experiment_id,
                                             likelihood = c("bernoulli", "normal")) {
  likelihood <- match.arg(likelihood)
  withr::local_seed(seed)
  W_df <- data.frame(
    price = sample(c("A", "B"), 24, replace = TRUE),
    message = sample(c("A", "B"), 24, replace = TRUE),
    stringsAsFactors = FALSE
  )
  Y <- if (identical(likelihood, "bernoulli")) {
    as.numeric(W_df$price == "B")
  } else {
    stats::rnorm(24, mean = 0.5 * (W_df$message == "B"), sd = 0.1)
  }
  list(
    experiment_id = experiment_id,
    Y = Y,
    W = W_df,
    mode = "single",
    likelihood = likelihood
  )
}

embedding_test_predictor_fit <- local({
  cache <- new.env(parent = emptyenv())

  function(mode = c("single", "pairwise"),
           cross_candidate_encoder = NULL,
           low_rank_interaction_rank = NULL,
           seed = 1L) {
    mode <- match.arg(mode)
    cross_key <- if (is.null(cross_candidate_encoder)) "default" else as.character(cross_candidate_encoder)
    rank_key <- if (is.null(low_rank_interaction_rank)) "default" else as.integer(low_rank_interaction_rank)
    cache_key <- paste(mode, cross_key, rank_key, as.integer(seed), sep = "::")
    if (exists(cache_key, envir = cache, inherits = FALSE)) {
      return(get(cache_key, envir = cache, inherits = FALSE))
    }

    skip_on_cran()
    skip_if_no_jax()
    withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

    data <- generate_test_data(
      n = if (identical(mode, "pairwise")) 40 else 30,
      n_factors = 3,
      n_levels = 2,
      seed = seed
    )
    fit <- suppressWarnings(strategic_prediction(
      Y = data$Y,
      W = data$W,
      model = "neural",
      mode = mode,
      pair_id = if (identical(mode, "pairwise")) data$pair_id else NULL,
      profile_order = if (identical(mode, "pairwise")) data$profile_order else NULL,
      neural_mcmc_control = embedding_test_neural_control(
        cross_candidate_encoder = cross_candidate_encoder,
        low_rank_interaction_rank = low_rank_interaction_rank
      ),
      conda_env_required = TRUE
    ))

    out <- list(fit = fit, data = data)
    assign(cache_key, out, envir = cache)
    out
  }
})

embedding_test_context_predictor_fit <- local({
  cache <- new.env(parent = emptyenv())

  function(mode = c("single", "pairwise"),
           cross_candidate_encoder = NULL,
           low_rank_interaction_rank = NULL,
           seed = 1L) {
    mode <- match.arg(mode)
    cross_key <- if (is.null(cross_candidate_encoder)) "default" else as.character(cross_candidate_encoder)
    rank_key <- if (is.null(low_rank_interaction_rank)) "default" else as.integer(low_rank_interaction_rank)
    cache_key <- paste(mode, cross_key, rank_key, as.integer(seed), sep = "::")
    if (exists(cache_key, envir = cache, inherits = FALSE)) {
      return(get(cache_key, envir = cache, inherits = FALSE))
    }

    skip_on_cran()
    skip_if_no_jax()
    withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

    data <- generate_test_data(
      n = if (identical(mode, "pairwise")) 64 else 32,
      n_factors = 3,
      n_levels = 2,
      seed = seed
    )
    data <- add_adversarial_structure(data, seed = seed + 101L)
    W_df <- as.data.frame(data$W, stringsAsFactors = FALSE)
    X <- embedding_test_covariates(
      W_df,
      pair_id = if (identical(mode, "pairwise")) data$pair_id else NULL
    )[, c("income", "household size"), drop = FALSE]
    names_list <- strategize:::cs2step_build_names_list(W_df)
    factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))
    W_idx <- strategize:::cs2step_encode_W_indices(
      W_df,
      names_list = names_list,
      unknown = "error",
      pad_unknown = 0L
    )
    fit <- suppressWarnings(strategize:::cs2step_eval_outcome_model_neural(
      Y = as.numeric(data$Y),
      W_idx = W_idx,
      names_list = names_list,
      factor_levels = factor_levels,
      diff = identical(mode, "pairwise"),
      pair_id = if (identical(mode, "pairwise")) data$pair_id else NULL,
      profile_order = if (identical(mode, "pairwise")) data$profile_order else NULL,
      competing_group_variable_candidate = data$competing_group_variable_candidate,
      competing_group_variable_respondent = data$competing_group_variable_respondent,
      X = X,
      conda_env_required = TRUE,
      neural_mcmc_control = embedding_test_neural_control(
        cross_candidate_encoder = cross_candidate_encoder,
        low_rank_interaction_rank = low_rank_interaction_rank
      )
    ))

    predictor <- structure(
      list(
        model_type = "neural",
        mode = mode,
        encoder = list(
          factor_names = names(names_list),
          names_list = names_list,
          factor_levels = factor_levels
        ),
        fit = fit,
        metadata = list(
          conda_env = "strategize_env",
          conda_env_required = TRUE
        )
      ),
      class = "strategic_predictor"
    )
    out <- list(fit = predictor, data = data, W = W_df, X = X)
    assign(cache_key, out, envir = cache)
    out
  }
})

embedding_test_newdata_subset <- function(fit_obj, rows) {
  out <- list(
    W = fit_obj$W[rows, , drop = FALSE],
    X = fit_obj$X[rows, , drop = FALSE],
    competing_group_variable_candidate =
      fit_obj$data$competing_group_variable_candidate[rows],
    competing_group_variable_respondent =
      fit_obj$data$competing_group_variable_respondent[rows]
  )
  if (identical(fit_obj$fit$mode, "pairwise")) {
    out$pair_id <- fit_obj$data$pair_id[rows]
    out$profile_order <- fit_obj$data$profile_order[rows]
  }
  out
}

embedding_test_pairwise_foundation_fit <- local({
  cache <- NULL

  function() {
    skip("Pooled foundation-model training moved to preference.fm.")
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()
    withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

    foundation_fit <- suppressWarnings(fit_conjoint_foundation_model(
      experiments = list(
        embedding_test_pairwise_experiment(
          seed = 9101,
          experiment_id = "study_a",
          factor_names = c("price", "message"),
          x_names = c("income", "household size")
        ),
        embedding_test_pairwise_experiment(
          seed = 9102,
          experiment_id = "study_b",
          factor_names = c("price", "message", "messenger"),
          x_names = c("income", "GOPScore")
        )
      ),
      foundation_control = embedding_test_foundation_control(with_embeddings = TRUE)
    ))

    cache <<- foundation_fit
    foundation_fit
  }
})

embedding_test_multigroup_single_foundation_fit <- local({
  cache <- NULL

  function() {
    skip("Pooled foundation-model training moved to preference.fm.")
    if (!is.null(cache)) {
      return(cache)
    }

    skip_on_cran()
    skip_if_no_jax()
    withr::local_envvar(c(STRATEGIZE_NEURAL_SKIP_EVAL = "1"))

    foundation_fit <- suppressWarnings(fit_conjoint_foundation_model(
      experiments = list(
        embedding_test_single_experiment(
          seed = 9201,
          experiment_id = "single_binary",
          likelihood = "bernoulli"
        ),
        embedding_test_single_experiment(
          seed = 9202,
          experiment_id = "single_normal",
          likelihood = "normal"
        )
      ),
      foundation_control = embedding_test_foundation_control(with_embeddings = TRUE)
    ))

    cache <<- foundation_fit
    foundation_fit
  }
})

embedding_test_universal_group_key <- function() {
  "universal::mixed::v1"
}

test_that("extract_embeddings returns single-mode neural embeddings", {
  fit_obj <- embedding_test_predictor_fit(mode = "single", seed = 9301)
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = as.data.frame(fit_obj$data$W[1:12, , drop = FALSE])
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(emb$mode, "single")
  expect_true(is.matrix(emb$embeddings))
  expect_equal(nrow(emb$embeddings), 12L)
  expect_true(all(is.finite(emb$embeddings)))
  expect_identical(emb$metadata$source_class, "strategic_predictor")
  expect_identical(emb$metadata$cross_candidate_encoder, "none")
})

test_that("extract_embeddings returns single-mode respondent-context embeddings", {
  fit_obj <- embedding_test_context_predictor_fit(mode = "single", seed = 9311)
  newdata <- embedding_test_newdata_subset(fit_obj, 1:12)
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = newdata,
    level = "respondent_context"
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(emb$mode, "single")
  expect_null(emb$embeddings)
  expect_true(is.matrix(emb$respondent_context))
  expect_equal(dim(emb$respondent_context), c(12L, 12L))
  expect_true(all(is.finite(emb$respondent_context)))
  expect_identical(emb$metadata$level, "respondent_context")
  expect_true(isTRUE(emb$metadata$context_components$respondent_group))
  expect_true(isTRUE(emb$metadata$context_components$respondent_covariates))
})

test_that("respondent-context embeddings react to covariates and respondent groups", {
  fit_obj <- embedding_test_context_predictor_fit(mode = "single", seed = 9311)
  newdata <- embedding_test_newdata_subset(fit_obj, 1:12)
  emb_base <- extract_embeddings(
    fit_obj$fit,
    newdata = newdata,
    level = "respondent_context"
  )

  cov_changed <- newdata
  cov_changed$X$income <- cov_changed$X$income + 3
  emb_cov <- extract_embeddings(
    fit_obj$fit,
    newdata = cov_changed,
    level = "respondent_context"
  )

  group_changed <- newdata
  group_changed$competing_group_variable_respondent <- ifelse(
    group_changed$competing_group_variable_respondent == "PartyA",
    "PartyB",
    "PartyA"
  )
  emb_group <- extract_embeddings(
    fit_obj$fit,
    newdata = group_changed,
    level = "respondent_context"
  )

  expect_false(isTRUE(all.equal(
    emb_base$respondent_context,
    emb_cov$respondent_context,
    tolerance = 1e-8
  )))
  expect_false(isTRUE(all.equal(
    emb_base$respondent_context,
    emb_group$respondent_context,
    tolerance = 1e-8
  )))
})

test_that("extract_embeddings returns narrow single-mode respondent readouts", {
  fit_obj <- embedding_test_context_predictor_fit(
    mode = "single",
    low_rank_interaction_rank = 2L,
    seed = 9311
  )
  newdata <- embedding_test_newdata_subset(fit_obj, 1:12)

  for (level in c("respondent_final", "respondent_cls", "respondent_pool", "respondent_mean")) {
    emb <- extract_embeddings(
      fit_obj$fit,
      newdata = newdata,
      level = level
    )

    expect_s3_class(emb, "strategic_embeddings")
    expect_identical(emb$mode, "single")
    expect_null(emb$embeddings)
    expect_null(emb$respondent_context)
    expect_true(is.matrix(emb$readouts[[level]]))
    expect_equal(dim(emb$readouts[[level]]), c(12L, 12L))
    expect_true(all(is.finite(emb$readouts[[level]])))
    expect_identical(emb$metadata$level, level)
  }
})

test_that("extract_embeddings returns left and right matrices for pairwise term mode", {
  fit_obj <- embedding_test_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "term",
    seed = 9302
  )
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = list(
      W = fit_obj$data$W,
      pair_id = fit_obj$data$pair_id,
      profile_order = fit_obj$data$profile_order
    )
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(emb$mode, "pairwise")
  expect_true(is.matrix(emb$left))
  expect_true(is.matrix(emb$right))
  expect_equal(nrow(emb$left), length(unique(fit_obj$data$pair_id)))
  expect_equal(dim(emb$left), dim(emb$right))
  expect_true(all(is.finite(emb$left)))
  expect_true(all(is.finite(emb$right)))
  expect_null(emb$joint)
  expect_identical(emb$metadata$cross_candidate_encoder, "term")
})

test_that("extract_embeddings returns pairwise respondent-context embeddings", {
  fit_obj <- embedding_test_context_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "term",
    seed = 9313
  )
  pair_keep <- unique(fit_obj$data$pair_id)[1:10]
  rows <- which(fit_obj$data$pair_id %in% pair_keep)
  newdata <- embedding_test_newdata_subset(fit_obj, rows)
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = newdata,
    level = "respondent_context"
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(emb$mode, "pairwise")
  expect_null(emb$left)
  expect_null(emb$right)
  expect_null(emb$joint)
  expect_true(is.matrix(emb$respondent_context))
  expect_equal(dim(emb$respondent_context), c(length(pair_keep), 12L))
  expect_true(all(is.finite(emb$respondent_context)))
})

test_that("extract_embeddings returns narrow pairwise respondent readouts", {
  fit_obj <- embedding_test_context_predictor_fit(
    mode = "pairwise",
    low_rank_interaction_rank = 2L,
    seed = 9313
  )
  pair_keep <- unique(fit_obj$data$pair_id)[1:10]
  rows <- which(fit_obj$data$pair_id %in% pair_keep)
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = embedding_test_newdata_subset(fit_obj, rows),
    level = "respondent_final"
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(emb$mode, "pairwise")
  expect_null(emb$left)
  expect_null(emb$right)
  expect_null(emb$joint)
  expect_null(emb$respondent_context)
  expect_true(is.matrix(emb$readouts$respondent_final))
  expect_equal(dim(emb$readouts$respondent_final), c(length(pair_keep), 12L))
  expect_true(all(is.finite(emb$readouts$respondent_final)))
  expect_identical(emb$metadata$level, "respondent_final")
})

test_that("extract_embeddings can return candidate and respondent-context levels together", {
  fit_obj <- embedding_test_context_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "term",
    seed = 9313
  )
  pair_keep <- unique(fit_obj$data$pair_id)[1:8]
  rows <- which(fit_obj$data$pair_id %in% pair_keep)
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = embedding_test_newdata_subset(fit_obj, rows),
    level = "all"
  )

  expect_true(is.matrix(emb$left))
  expect_true(is.matrix(emb$right))
  expect_true(is.matrix(emb$respondent_context))
  expect_equal(nrow(emb$left), length(pair_keep))
  expect_equal(nrow(emb$respondent_context), length(pair_keep))
  expect_identical(emb$metadata$level, "all")
})

test_that("respondent readout levels error when readouts are unavailable", {
  expect_error(
    strategize:::cs2step_neural_extract_prepared(
      params = list(),
      model_info = list(low_rank_interaction_rank = 0L),
      prep = list(),
      level = "respondent_final"
    ),
    "does not expose respondent readout"
  )
})

test_that("pairwise respondent-context extraction passes through group fields", {
  fit_obj <- embedding_test_context_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "term",
    seed = 9313
  )
  expect_true(isTRUE(fit_obj$fit$fit$neural_model_info$has_stage_context))
  pair_keep <- unique(fit_obj$data$pair_id)[1:12]
  rows <- which(fit_obj$data$pair_id %in% pair_keep)
  newdata <- embedding_test_newdata_subset(fit_obj, rows)

  emb_with_groups <- extract_embeddings(
    fit_obj$fit,
    newdata = newdata,
    level = "respondent_context"
  )
  newdata_without_groups <- newdata
  newdata_without_groups$competing_group_variable_candidate <- NULL
  newdata_without_groups$competing_group_variable_respondent <- NULL
  emb_without_groups <- extract_embeddings(
    fit_obj$fit,
    newdata = newdata_without_groups,
    level = "respondent_context"
  )

  expect_true(isTRUE(emb_with_groups$metadata$context_components$stage))
  expect_true(isTRUE(emb_with_groups$metadata$context_components$matchup))
  expect_false(isTRUE(all.equal(
    emb_with_groups$respondent_context,
    emb_without_groups$respondent_context,
    tolerance = 1e-8
  )))
})

test_that("extract_embeddings returns post-attention pairwise embeddings for attn mode", {
  fit_obj <- embedding_test_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "attn",
    seed = 9303
  )
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = list(
      W = fit_obj$data$W,
      pair_id = fit_obj$data$pair_id,
      profile_order = fit_obj$data$profile_order
    )
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_true(is.matrix(emb$left))
  expect_true(is.matrix(emb$right))
  expect_equal(nrow(emb$left), length(unique(fit_obj$data$pair_id)))
  expect_true(all(is.finite(emb$left)))
  expect_true(all(is.finite(emb$right)))
  expect_null(emb$joint)
  expect_identical(emb$metadata$cross_candidate_encoder, "attn")
})

test_that("extract_embeddings returns joint readout for pairwise full mode", {
  fit_obj <- embedding_test_predictor_fit(
    mode = "pairwise",
    cross_candidate_encoder = "full",
    seed = 9304
  )
  emb <- extract_embeddings(
    fit_obj$fit,
    newdata = list(
      W = fit_obj$data$W,
      pair_id = fit_obj$data$pair_id,
      profile_order = fit_obj$data$profile_order
    )
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_true(is.matrix(emb$joint))
  expect_equal(nrow(emb$joint), length(unique(fit_obj$data$pair_id)))
  expect_true(all(is.finite(emb$joint)))
  expect_null(emb$left)
  expect_null(emb$right)
  expect_identical(emb$metadata$cross_candidate_encoder, "full")
})

test_that("extract_embeddings errors on non-neural strategic predictors", {
  data <- generate_test_data(n = 200, n_factors = 2, n_levels = 2, seed = 9305)
  fit <- strategic_prediction(
    Y = data$Y,
    W = data$W,
    model = "glm",
    mode = "single",
    use_regularization = FALSE
  )

  expect_error(
    extract_embeddings(fit, newdata = as.data.frame(data$W[1:10, , drop = FALSE])),
    "only available for neural models"
  )
})

test_that("extract_embeddings works on raw foundation groups with factor metadata supplied separately", {
  foundation_fit <- embedding_test_pairwise_foundation_fit()
  study_a <- embedding_test_pairwise_experiment(
    seed = 9101,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income", "household size")
  )
  newdata <- data.frame(
    study_a$W,
    study_a$X,
    pair_id = study_a$pair_id,
    profile_order = study_a$profile_order,
    experiment_id = study_a$experiment_id,
    check.names = FALSE
  )

  emb <- extract_embeddings(
    foundation_fit,
    newdata = newdata,
    group_key = "pairwise::bernoulli::1",
    p_list = suppressMessages(create_p_list(study_a$W, uniform = TRUE))
  )

  expect_s3_class(emb, "strategic_embeddings")
  expect_true(is.matrix(emb$left))
  expect_true(is.matrix(emb$right))
  expect_equal(nrow(emb$left), length(unique(study_a$pair_id)))
  expect_identical(emb$metadata$source_class, "conjoint_foundation_model")
  expect_identical(emb$metadata$foundation_group_key, "pairwise::bernoulli::1")
  expect_identical(
    emb$metadata$foundation_group_key_canonical,
    embedding_test_universal_group_key()
  )
  expect_identical(emb$metadata$unmatched_factors, character(0))
  expect_identical(emb$metadata$unmatched_levels, character(0))
})

test_that("extract_embeddings auto-selects the universal group without ambiguity", {
  foundation_fit <- embedding_test_pairwise_foundation_fit()
  study_a <- embedding_test_pairwise_experiment(
    seed = 9101,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income", "household size")
  )
  newdata <- data.frame(
    study_a$W,
    study_a$X,
    pair_id = study_a$pair_id,
    profile_order = study_a$profile_order,
    experiment_id = study_a$experiment_id,
    check.names = FALSE
  )

  emb <- extract_embeddings(
    foundation_fit,
    newdata = newdata,
    p_list = suppressMessages(create_p_list(study_a$W, uniform = TRUE))
  )
  expect_s3_class(emb, "strategic_embeddings")
  expect_identical(
    emb$metadata$foundation_group_key_canonical,
    embedding_test_universal_group_key()
  )
})

test_that("saved and loaded foundation bundles preserve extracted embeddings", {
  skip("Foundation bundle save/load is owned by preference.fm.")
  foundation_fit <- embedding_test_pairwise_foundation_fit()
  study_a <- embedding_test_pairwise_experiment(
    seed = 9101,
    experiment_id = "study_a",
    factor_names = c("price", "message"),
    x_names = c("income", "household size")
  )
  newdata <- list(
    W = study_a$W,
    X = study_a$X,
    pair_id = study_a$pair_id,
    profile_order = study_a$profile_order,
    experiment_id = study_a$experiment_id
  )
  emb_before <- extract_embeddings(
    foundation_fit,
    newdata = newdata,
    group_key = "pairwise::bernoulli::1",
    names_list = strategize:::cs_build_names_list(study_a$W)
  )

  tmp <- tempfile(fileext = ".rds")
  save_conjoint_foundation_bundle(tmp, foundation_fit, overwrite = TRUE)
  loaded <- load_conjoint_foundation_bundle(tmp, preload_params = FALSE)
  emb_after <- extract_embeddings(
    loaded,
    newdata = newdata,
    group_key = "pairwise::bernoulli::1",
    names_list = strategize:::cs_build_names_list(study_a$W)
  )

  expect_equal(emb_before$left, emb_after$left, tolerance = 1e-6)
  expect_equal(emb_before$right, emb_after$right, tolerance = 1e-6)
})

test_that("saved and loaded adapted predictors preserve extracted embeddings", {
  foundation_fit <- embedding_test_pairwise_foundation_fit()
  target <- embedding_test_pairwise_experiment(
    seed = 9306,
    experiment_id = "target_study",
    factor_names = c("price", "message"),
    x_names = c("income", "household size", "local_bonus")
  )
  predictor <- suppressWarnings(adapt_conjoint_foundation_model(
    foundation_model = foundation_fit,
    Y = target$Y,
    W = target$W,
    X = target$X,
    mode = "pairwise",
    pair_id = target$pair_id,
    profile_order = target$profile_order,
    experiment_id = target$experiment_id,
    experiment_description = target$experiment_description,
    canonical_factor_id = target$canonical_factor_id,
    neural_mcmc_control = embedding_test_neural_control()
  ))
  emb_before <- extract_embeddings(
    predictor,
    newdata = list(
      W = target$W,
      X = target$X,
      pair_id = target$pair_id,
      profile_order = target$profile_order
    )
  )

  tmp <- tempfile(fileext = ".rds")
  save_strategic_predictor(predictor, tmp, overwrite = TRUE)
  predictor_loaded <- load_strategic_predictor(tmp)
  emb_after <- extract_embeddings(
    predictor_loaded,
    newdata = list(
      W = target$W,
      X = target$X,
      pair_id = target$pair_id,
      profile_order = target$profile_order
    )
  )

  expect_equal(emb_before$left, emb_after$left, tolerance = 1e-6)
  expect_equal(emb_before$right, emb_after$right, tolerance = 1e-6)
})
