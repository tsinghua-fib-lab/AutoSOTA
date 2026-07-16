#' Prediction-Only Outcome Model API
#'
#' @description
#' Fit the package's outcome model (GLM or neural) without running the stochastic
#' intervention optimization in \code{\link{strategize}}. Returns a fitted
#' \code{strategic_predictor} object with a \code{\link[stats]{predict}} method.
#'
#' @name prediction-api
NULL

cs2step_build_names_list <- function(W) {
  cs_build_names_list(W = W)
}

cs2step_align_W <- function(W, factor_names) {
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  W <- as.data.frame(W)
  if (is.null(colnames(W))) {
    stop("'W' must have column names to align with the fitted model.", call. = FALSE)
  }
  missing_cols <- setdiff(factor_names, colnames(W))
  extra_cols <- setdiff(colnames(W), factor_names)
  if (length(missing_cols) > 0) {
    stop(
      "Missing factor columns in newdata: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  # Ignore extra columns (e.g., IDs) by default.
  W[, factor_names, drop = FALSE]
}

cs2step_neural_prepare_W_for_prediction <- function(W, factor_names) {
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  factor_names <- as.character(factor_names %||% character(0))
  W_cols <- colnames(W)
  W_df <- as.data.frame(W, check.names = FALSE)
  if (length(factor_names) < 1L) {
    return(W_df)
  }

  has_column_names <- !is.null(W_cols) &&
    length(W_cols) > 0L &&
    any(!is.na(W_cols) & nzchar(W_cols))
  if (!isTRUE(has_column_names)) {
    if (ncol(W_df) != length(factor_names)) {
      stop(
        "Unnamed prediction-time W must match the fitted factor width exactly.",
        call. = FALSE
      )
    }
    colnames(W_df) <- factor_names
    return(W_df)
  }

  if (!any(W_cols %in% factor_names)) {
    stop(
      paste(
        "Prediction-time factor names do not match the fitted schema.",
        "For genuinely new factor schemas, supply factor_schema with text embeddings or refit."
      ),
      call. = FALSE
    )
  }

  order_idx <- neural_factor_order_from_names(
    W_cols,
    factor_names,
    error_on_partial = TRUE
  )
  if (length(order_idx) < 1L) {
    stop(
      paste(
        "Prediction-time factor names do not match the fitted schema.",
        "For genuinely new factor schemas, supply factor_schema with text embeddings or refit."
      ),
      call. = FALSE
    )
  }

  cs2step_align_W(W_df, factor_names)
}

cs2step_encode_W_indices <- function(W, names_list, unknown = c("holdout", "error"), pad_unknown = 0L) {
  enc <- cs_encode_W_indices(
    W = W,
    names_list = names_list,
    unknown = unknown,
    pad_unknown = pad_unknown,
    align = "by_name"
  )
  enc$W_idx
}

cs2step_neural_factor_schema_known_fields <- function() {
  c(
    "names_list",
    "p_list",
    "factor_name_text",
    "level_name_text",
    "factor_struct_matrix",
    "level_struct_matrices",
    "factor_struct_feature_names",
    "level_struct_feature_names",
    "text_embedding_fn"
  )
}

cs2step_neural_factor_schema_names_hint <- function(factor_schema = NULL) {
  if (is.null(factor_schema)) {
    return(NULL)
  }
  if (!is.list(factor_schema)) {
    return(NULL)
  }
  schema_names <- names(factor_schema)
  known_fields <- cs2step_neural_factor_schema_known_fields()
  if (is.null(schema_names) || !any(schema_names %in% known_fields)) {
    names_list <- factor_schema
  } else {
    names_list <- factor_schema$names_list %||% NULL
    if (is.null(names_list) && !is.null(factor_schema$p_list)) {
      names_list <- lapply(factor_schema$p_list, function(x) {
        list(names(x %||% character(0)))
      })
      if (!is.null(names(factor_schema$p_list))) {
        names(names_list) <- names(factor_schema$p_list)
      }
    }
  }
  out <- names(names_list %||% list())
  if (is.null(out) || !length(out) || any(is.na(out) | !nzchar(out))) {
    return(NULL)
  }
  as.character(out)
}

cs2step_unpack_newdata <- function(newdata, factor_names, mode, factor_schema = NULL) {
  if (is.null(newdata)) {
    stop("'newdata' is required for prediction.", call. = FALSE)
  }

  if (is.list(newdata) && !is.data.frame(newdata)) {
    if (!"W" %in% names(newdata)) {
      stop("When newdata is a list, it must contain element 'W'.", call. = FALSE)
    }
    out <- list(
      W = newdata$W,
      X = newdata$X %||% NULL,
      pair_id = newdata$pair_id %||% NULL,
      profile_order = newdata$profile_order %||% NULL,
      competing_group_variable_candidate = newdata$competing_group_variable_candidate %||% NULL,
      competing_group_variable_respondent = newdata$competing_group_variable_respondent %||% NULL,
      experiment_id = newdata$experiment_id %||% NULL,
      experiment_description = newdata$experiment_description %||% NULL,
      experiment_country = if ("experiment_country" %in% names(newdata)) {
        newdata$experiment_country %||% NA_character_
      } else {
        NULL
      },
      experiment_year = newdata$experiment_year %||% NULL,
      factor_schema = newdata$factor_schema %||% NULL,
      names_list = newdata$names_list %||% NULL,
      p_list = newdata$p_list %||% NULL
    )
    return(out)
  }

  newdata <- as.data.frame(newdata)
  schema_factor_names <- cs2step_neural_factor_schema_names_hint(factor_schema)
  factor_names_use <- schema_factor_names %||% factor_names
  pair_id <- NULL
  profile_order <- NULL
  competing_group_variable_candidate <- NULL
  competing_group_variable_respondent <- NULL
  experiment_id <- NULL
  experiment_description <- NULL
  experiment_country <- NULL
  experiment_year <- NULL
  if (identical(mode, "pairwise")) {
    if ("pair_id" %in% colnames(newdata)) {
      pair_id <- newdata[["pair_id"]]
    }
    if ("profile_order" %in% colnames(newdata)) {
      profile_order <- newdata[["profile_order"]]
    }
  }
  if ("experiment_id" %in% colnames(newdata)) {
    experiment_id <- newdata[["experiment_id"]]
  }
  if ("experiment_description" %in% colnames(newdata)) {
    experiment_description <- newdata[["experiment_description"]]
  }
  if ("experiment_country" %in% colnames(newdata)) {
    experiment_country <- newdata[["experiment_country"]]
  }
  if ("experiment_year" %in% colnames(newdata)) {
    experiment_year <- newdata[["experiment_year"]]
  }
  if ("competing_group_variable_candidate" %in% colnames(newdata)) {
    competing_group_variable_candidate <- newdata[["competing_group_variable_candidate"]]
  }
  if ("competing_group_variable_respondent" %in% colnames(newdata)) {
    competing_group_variable_respondent <- newdata[["competing_group_variable_respondent"]]
  }
  missing_factor_cols <- setdiff(factor_names_use, colnames(newdata))
  if (length(missing_factor_cols) > 0L) {
    stop(
      "Missing factor columns in newdata: ",
      paste(missing_factor_cols, collapse = ", "),
      call. = FALSE
    )
  }
  W <- newdata[, factor_names_use, drop = FALSE]
  extra_cols <- setdiff(
    colnames(newdata),
    c(
      factor_names_use,
      "pair_id",
      "profile_order",
      "competing_group_variable_candidate",
      "competing_group_variable_respondent",
      "experiment_id",
      "experiment_description",
      "experiment_country",
      "experiment_year"
    )
  )
  X <- if (length(extra_cols) > 0L) {
    newdata[, extra_cols, drop = FALSE]
  } else {
    NULL
  }
  list(
    W = W,
    X = X,
    pair_id = pair_id,
    profile_order = profile_order,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_variable_respondent = competing_group_variable_respondent,
    experiment_id = experiment_id,
    experiment_description = experiment_description,
    experiment_country = experiment_country,
    experiment_year = experiment_year,
    factor_schema = NULL,
    names_list = NULL,
    p_list = NULL
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x

cs2step_neural_param_cache <- new.env(parent = emptyenv())

cs2step_neural_cache_get <- function(cache_id) {
  if (is.null(cache_id) || !nzchar(cache_id)) {
    return(NULL)
  }
  if (exists(cache_id, envir = cs2step_neural_param_cache, inherits = FALSE)) {
    return(get(cache_id, envir = cs2step_neural_param_cache, inherits = FALSE))
  }
  NULL
}

cs2step_neural_cache_set <- function(cache_id, params) {
  if (is.null(cache_id) || !nzchar(cache_id)) {
    return(invisible(NULL))
  }
  assign(cache_id, params, envir = cs2step_neural_param_cache)
  invisible(NULL)
}

cs2step_has_reticulate <- function() {
  requireNamespace("reticulate", quietly = TRUE)
}

cs2step_py_last_error_message <- function() {
  if (!cs2step_has_reticulate()) {
    return(character(0))
  }
  err <- tryCatch(reticulate::py_last_error(), error = function(e) NULL)
  if (is.null(err)) {
    return(character(0))
  }
  pieces <- c(
    type = err$type %||% NULL,
    value = err$value %||% NULL,
    traceback = if (is.null(err$traceback)) NULL else paste(err$traceback, collapse = "\n")
  )
  pieces <- as.character(pieces)
  pieces[nzchar(pieces)]
}

cs2step_py_object_summary <- function(x) {
  if (!cs2step_has_reticulate() || !reticulate::is_py_object(x)) {
    return("")
  }
  object_type <- tryCatch(
    as.character(reticulate::py_to_r(x$`__class__`$`__name__`)),
    error = function(e) NULL
  )
  object_shape <- tryCatch(
    paste(as.character(reticulate::py_to_r(x$shape)), collapse = "x"),
    error = function(e) NULL
  )
  object_dtype <- tryCatch(
    as.character(reticulate::py_to_r(x$dtype)),
    error = function(e) NULL
  )
  details <- c(
    if (length(object_type) && nzchar(object_type)) sprintf("type=%s", object_type),
    if (length(object_shape) && nzchar(object_shape)) sprintf("shape=%s", object_shape),
    if (length(object_dtype) && nzchar(object_dtype)) sprintf("dtype=%s", object_dtype)
  )
  if (!length(details)) {
    return("")
  }
  sprintf(" (%s)", paste(details, collapse = ", "))
}

cs2step_stop_py_conversion_failure <- function(context, x, attempts) {
  last_error <- cs2step_py_last_error_message()
  if (length(last_error)) {
    attempts <- c(attempts, sprintf("reticulate::py_last_error(): %s", paste(last_error, collapse = "\n")))
  }
  attempts <- unique(attempts[nzchar(attempts)])
  attempt_text <- if (length(attempts)) {
    paste0("\n- ", paste(attempts, collapse = "\n- "))
  } else {
    ""
  }
  if (nchar(attempt_text) > 6000L) {
    attempt_text <- paste0(substr(attempt_text, 1L, 6000L), "\n- <conversion diagnostics truncated>")
  }
  stop(
    sprintf(
      "Failed to convert %s from Python to R%s.%s",
      context,
      cs2step_py_object_summary(x),
      attempt_text
    ),
    call. = FALSE
  )
}

cs2step_py_to_r <- function(x, context = "Python object") {
  if (is.null(x)) {
    return(NULL)
  }
  if (exists("strategize_jax_block_until_ready", mode = "function")) {
    tryCatch(
      strategize_jax_block_until_ready(x),
      error = function(e) {
        stop(
          sprintf(
            "Failed while waiting for %s to become ready%s: %s",
            context,
            cs2step_py_object_summary(x),
            conditionMessage(e)
          ),
          call. = FALSE
        )
      }
    )
  }
  if (!cs2step_has_reticulate()) {
    return(x)
  }
  if (reticulate::is_py_object(x)) {
    attempts <- character(0)
    try_convert <- function(label, expr) {
      out <- tryCatch(
        force(expr),
        error = function(e) {
          attempts <<- c(attempts, sprintf("%s: %s", label, conditionMessage(e)))
          NULL
        }
      )
      if (is.null(out)) {
        return(NULL)
      }
      if (reticulate::is_py_object(out)) {
        attempts <<- c(
          attempts,
          sprintf("%s: returned an unconverted Python object%s", label, cs2step_py_object_summary(out))
        )
        return(NULL)
      }
      out
    }

    has_jax <- exists("strenv") &&
      exists("jax", envir = strenv, inherits = FALSE)
    has_np <- exists("strenv") && exists("np", envir = strenv, inherits = FALSE)
    if (isTRUE(has_jax) && isTRUE(has_np)) {
      out <- try_convert(
        "py_to_r(np.asarray(jax.device_get(x)))",
        reticulate::py_to_r(strenv$np$asarray(strenv$jax$device_get(x)))
      )
      if (!is.null(out)) {
        return(out)
      }
    }
    if (isTRUE(has_np)) {
      out <- try_convert("py_to_r(np.asarray(x))", reticulate::py_to_r(strenv$np$asarray(x)))
      if (!is.null(out)) {
        return(out)
      }
      out <- try_convert("py_to_r(np.array(x))", reticulate::py_to_r(strenv$np$array(x)))
      if (!is.null(out)) {
        return(out)
      }
    }
    out <- try_convert("py_to_r(x)", reticulate::py_to_r(x))
    if (!is.null(out)) {
      return(out)
    }
    cs2step_stop_py_conversion_failure(context, x, attempts)
  }
  x
}

cs2step_eval_outcome_model_glm <- function(Y,
                                          W_idx,
                                          factor_levels,
                                          diff,
                                          pair_id = NULL,
                                          profile_order = NULL,
                                          varcov_cluster_variable = NULL,
                                          use_regularization = TRUE,
                                          nFolds_glm = 3L) {
  eval_env <- new.env(parent = environment())

  # Minimal strenv stub: generate_ModelOutcome uses only jnp$array + dtj in GLM path.
  strenv_stub <- list(
    dtj = NULL,
    jnp = list(array = function(x, dtype = NULL) x),
    np = list(array = function(x) x)
  )

  eval_env$strenv <- strenv_stub

  eval_env$adversarial <- FALSE
  eval_env$adversarial_model_strategy <- "four"
  eval_env$GroupsPool <- 1
  eval_env$GroupCounter <- 1
  eval_env$Round_ <- 1
  eval_env$outcome_model_key <- NULL
  eval_env$save_outcome_model <- FALSE
  eval_env$presaved_outcome_model <- FALSE
  eval_env$use_regularization <- isTRUE(use_regularization)
  eval_env$nFolds_glm <- as.integer(nFolds_glm)
  eval_env$folds <- NULL
  eval_env$K <- 1L
  eval_env$holdout_indicator <- 1L
  eval_env$diff <- isTRUE(diff)
  eval_env$glm_family <- "binomial"

  eval_env$w_orig <- W_idx
  eval_env$W <- W_idx
  eval_env$W_ <- W_idx
  eval_env$Y <- Y
  eval_env$Y_ <- Y

  eval_env$factor_levels <- factor_levels

  eval_env$varcov_cluster_variable <- varcov_cluster_variable
  eval_env$varcov_cluster_variable_ <- varcov_cluster_variable
  eval_env$pair_id <- pair_id
  eval_env$pair_id_ <- pair_id
  eval_env$profile_order <- profile_order
  eval_env$profile_order_ <- profile_order
  eval_env$competing_group_variable_candidate_ <- NULL
  eval_env$competing_group_competition_variable_candidate_ <- NULL
  eval_env$competing_group_variable_respondent_ <- NULL
  eval_env$X_ <- NULL
  eval_env$respondent_id <- NULL
  eval_env$respondent_task_id <- NULL

  eval(body(generate_ModelOutcome), envir = eval_env)

  list(
    intercept = as.numeric(eval_env$EST_INTERCEPT_tf),
    coefficients = as.numeric(eval_env$EST_COEFFICIENTS_tf),
    vcov = eval_env$vcov_OutcomeModel,
    main_info = eval_env$main_info,
    interaction_info = eval_env$interaction_info,
    family = eval_env$glm_family,
    fit_metrics = eval_env$fit_metrics
  )
}

cs2step_eval_outcome_model_neural <- function(Y,
                                             W_idx,
                                             W_idx_compact = NULL,
                                             names_list = NULL,
                                             factor_levels,
                                             diff,
                                             pair_id = NULL,
                                             profile_order = NULL,
                                             competing_group_variable_candidate = NULL,
                                             competing_group_variable_respondent = NULL,
                                             X = NULL,
                                             X_compact = NULL,
                                             X_present = NULL,
                                             X_present_compact = NULL,
                                             respondent_id = NULL,
                                             respondent_task_id = NULL,
                                             neural_token_info = NULL,
                                             likelihood_override = NULL,
                                             n_outcomes_override = NULL,
                                             conda_env = "strategize_env",
                                             conda_env_required = TRUE,
                                             neural_mcmc_control = NULL,
                                             varcov_cluster_variable = NULL,
                                             nFolds_glm = 3L) {
  if (!"jnp" %in% ls(envir = strenv) || !"np" %in% ls(envir = strenv)) {
    ok <- tryCatch({
      initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
      TRUE
    }, error = function(e) FALSE)
    if (!isTRUE(ok)) {
      stop(
        "Neural backend not available.\n",
        "  Run strategize::build_backend() to create the JAX environment, then retry.\n",
        "  (You can also set conda_env=... to choose a different environment.)",
        call. = FALSE
      )
    }
  }

  eval_env <- new.env(parent = environment())
  eval_env$adversarial <- FALSE
  eval_env$adversarial_model_strategy <- "neural"
  eval_env$GroupsPool <- 1
  eval_env$GroupCounter <- 1
  eval_env$Round_ <- 1
  eval_env$outcome_model_key <- NULL
  eval_env$save_outcome_model <- FALSE
  eval_env$presaved_outcome_model <- FALSE
  eval_env$use_regularization <- FALSE
  eval_env$K <- 1L
  eval_env$holdout_indicator <- 1L
  eval_env$diff <- isTRUE(diff)
  eval_env$glm_family <- "binomial"

  n_training_rows <- if (!is.null(W_idx)) {
    nrow(W_idx)
  } else {
    cs2step_compact_n_rows(W_idx_compact)
  }
  eval_env$w_orig <- W_idx
  eval_env$W <- W_idx
  eval_env$W_ <- W_idx
  eval_env$W_idx_compact <- W_idx_compact
  eval_env$Y <- Y
  eval_env$Y_ <- Y
  eval_env$names_list <- names_list
  eval_env$factor_levels <- factor_levels
  eval_env$indi_ <- seq_len(n_training_rows)

  eval_env$pair_id <- pair_id
  eval_env$pair_id_ <- pair_id
  eval_env$profile_order <- profile_order
  eval_env$profile_order_ <- profile_order
  eval_env$varcov_cluster_variable <- varcov_cluster_variable
  eval_env$varcov_cluster_variable_ <- varcov_cluster_variable
  eval_env$competing_group_variable_candidate_ <- competing_group_variable_candidate
  eval_env$competing_group_competition_variable_candidate_ <- NULL
  eval_env$competing_group_variable_respondent_ <- competing_group_variable_respondent
  eval_env$neural_mcmc_control <- neural_mcmc_control
  eval_env$nFolds_glm <- if (is.null(nFolds_glm)) NULL else as.integer(nFolds_glm)
  eval_env$X <- if (is.null(X)) NULL else as.matrix(X)
  eval_env$X_ <- if (is.null(X)) NULL else as.matrix(X)
  eval_env$X_compact <- X_compact
  eval_env$X_present <- if (is.null(X_present)) NULL else as.matrix(X_present)
  eval_env$X_present_ <- if (is.null(X_present)) NULL else as.matrix(X_present)
  eval_env$X_present_compact <- X_present_compact
  eval_env$respondent_id <- respondent_id
  eval_env$respondent_task_id <- respondent_task_id
  eval_env$neural_token_info <- neural_token_info
  eval_env$neural_likelihood_override <- if (!is.null(likelihood_override)) {
    tolower(as.character(likelihood_override))
  } else {
    NULL
  }
  eval_env$neural_nOutcomes_override <- if (!is.null(n_outcomes_override)) {
    as.integer(n_outcomes_override)
  } else {
    NULL
  }

  eval(body(generate_ModelOutcome_neural), envir = eval_env)

  theta_mean <- tryCatch(as.numeric(reticulate::py_to_r(strenv$np$array(eval_env$EST_COEFFICIENTS_tf))),
                         error = function(e) as.numeric(eval_env$EST_COEFFICIENTS_tf))
  vcov_vec <- eval_env$vcov_OutcomeModel
  if (!is.null(vcov_vec) && length(vcov_vec) >= 2L) {
    theta_var <- as.numeric(vcov_vec[-1])
  } else {
    theta_var <- numeric(0)
  }

  predict_pair_fxn <- if (exists("TransformerPredict_pair", envir = eval_env, inherits = FALSE)) {
    eval_env$TransformerPredict_pair
  } else {
    NULL
  }
  predict_single_fxn <- if (exists("TransformerPredict_single", envir = eval_env, inherits = FALSE)) {
    eval_env$TransformerPredict_single
  } else {
    NULL
  }

  # Drop large training artifacts that aren't needed for prediction.
  drop_names <- c(
    "PosteriorDraws", "posterior_samples",
    "X_left_jnp", "X_right_jnp", "X_single_jnp", "Y_jnp",
    "sampler", "kernel", "svi", "svi_result", "svi_state", "SVIParams"
  )
  rm(list = intersect(drop_names, ls(envir = eval_env, all.names = TRUE)), envir = eval_env)

  list(
    my_model = eval_env$my_model,
    predict_pair = predict_pair_fxn,
    predict_single = predict_single_fxn,
    neural_model_info = eval_env$neural_model_info,
    theta_mean = theta_mean,
    theta_var = theta_var,
    fit_metrics = eval_env$fit_metrics %||% eval_env$neural_model_info$fit_metrics
  )
}

cs2step_validate_binary_outcome <- function(Y) {
  if (is.null(Y) || missing(Y)) {
    stop("'Y' is required.", call. = FALSE)
  }
  if (!is.numeric(Y) && !is.integer(Y)) {
    stop("'Y' must be numeric (0/1).", call. = FALSE)
  }
  vals <- unique(stats::na.omit(as.numeric(Y)))
  if (!all(vals %in% c(0, 1))) {
    stop("This prediction API currently supports binary outcomes only (Y in {0,1}).",
         call. = FALSE)
  }
  TRUE
}

cs2step_validate_pairwise_ids <- function(pair_id, n) {
  if (is.null(pair_id)) {
    stop("Pairwise mode requires 'pair_id'.", call. = FALSE)
  }
  if (length(pair_id) != n) {
    stop(sprintf("'pair_id' has %d elements but W has %d rows.", length(pair_id), n),
         call. = FALSE)
  }
  sizes <- table(pair_id)
  if (any(sizes != 2L)) {
    bad <- names(sizes)[sizes != 2L]
    stop(
      "Pairwise mode requires exactly 2 rows per pair_id.\n",
      "  Bad pair_id values: ", paste(head(bad, 10), collapse = ", "),
      if (length(bad) > 10) " ..." else "",
      call. = FALSE
    )
  }
  TRUE
}

#' Fit a prediction-only outcome model
#'
#' @param Y Binary outcome in \code{0/1}.
#' @param W Factor matrix/data.frame (one column per conjoint factor).
#' @param X Optional covariate matrix/data.frame (neural backend).
#' @param ... Reserved for future extensions.
#' @param model \code{"glm"} or \code{"neural"}.
#' @param mode \code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#' @param pair_id Optional pair identifier (required for pairwise).
#' @param profile_order Optional within-pair ordering (1/2).
#' @param varcov_cluster_variable Optional cluster IDs for robust GLM vcov.
#' @param conda_env Conda env name for neural backend.
#' @param conda_env_required Require conda env to exist (neural backend).
#' @param neural_mcmc_control Optional list passed to neural backend.
#' @param use_regularization Logical; run glinternet screening for GLM.
#' @param nFolds_glm Number of folds for glinternet CV (GLM).
#' @param cache_path Optional path to a cached predictor (.rds). If it exists and
#'   \code{cache_overwrite} is \code{FALSE}, the cached model is loaded instead of refitting.
#'   When a new model is fit, it is saved to this path. For neural fits,
#'   \code{cache_path} also enables restartable SVI checkpoints under
#'   \code{paste0(cache_path, ".inprogress")}; successful final saves remove
#'   that in-progress directory.
#' @param cache_overwrite Logical; refit and overwrite any existing cache at \code{cache_path}.
#' @param cache_compress Compression setting passed to \code{saveRDS()}.
#' @return An object of class \code{strategic_predictor}.
#' @export
strategic_prediction <- function(Y,
                                W,
                                X = NULL,
                                ...,
                                model = c("glm", "neural"),
                                mode = c("auto", "pairwise", "single"),
                                pair_id = NULL,
                                profile_order = NULL,
                                varcov_cluster_variable = NULL,
                                conda_env = "strategize_env",
                                conda_env_required = TRUE,
                                neural_mcmc_control = NULL,
                                use_regularization = TRUE,
                                nFolds_glm = 3L,
                                cache_path = NULL,
                                cache_overwrite = FALSE,
                                cache_compress = TRUE) {
  if (!is.null(cache_path)) {
    cache_path <- as.character(cache_path)
    if (length(cache_path) != 1L || !nzchar(cache_path)) {
      stop("'cache_path' must be a non-empty character path.", call. = FALSE)
    }
  }
  cs2step_validate_binary_outcome(Y)
  if (missing(W) || is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  if (nrow(W) != length(Y)) {
    stop(sprintf("Dimension mismatch: Y has %d elements but W has %d rows.",
                 length(Y), nrow(W)),
         call. = FALSE)
  }

  model <- match.arg(model)
  mode <- match.arg(mode)
  mode_use <- mode
  if (identical(mode, "auto")) {
    mode_use <- if (!is.null(pair_id)) "pairwise" else "single"
  }
  if (identical(mode_use, "pairwise")) {
    cs2step_validate_pairwise_ids(pair_id, nrow(W))
  }

  W_df <- as.data.frame(W)
  if (is.null(colnames(W_df))) {
    colnames(W_df) <- paste0("V", seq_len(ncol(W_df)))
  }

  names_list <- cs2step_build_names_list(W_df)
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))

  # Training encoding (no unknown padding).
  W_idx_train <- cs2step_encode_W_indices(W_df, names_list, unknown = "error", pad_unknown = 0L)

  diff <- identical(mode_use, "pairwise")

  fit_control <- neural_mcmc_control
  if (identical(model, "neural") && !is.null(cache_path)) {
    fit_control <- fit_control %||% list()
    checkpoint_path <- fit_control$checkpoint_path %||% paste0(cache_path, ".inprogress")
    if (is.null(fit_control$checkpoint_path)) {
      fit_control$checkpoint_path <- checkpoint_path
    }
    if (isTRUE(cache_overwrite) && identical(checkpoint_path, paste0(cache_path, ".inprogress"))) {
      neural_svi_checkpoint_remove_dir(checkpoint_path)
    }
  }

  request_fingerprint <- cs2step_predictor_request_fingerprint(
    Y = Y,
    W = W_df,
    X = X,
    model = model,
    mode = mode_use,
    pair_id = pair_id,
    profile_order = profile_order,
    varcov_cluster_variable = varcov_cluster_variable,
    names_list = names_list,
    factor_levels = factor_levels,
    neural_mcmc_control = fit_control,
    use_regularization = use_regularization,
    nFolds_glm = nFolds_glm
  )

  if (!is.null(cache_path) && !isTRUE(cache_overwrite) && file.exists(cache_path)) {
    bundle <- readRDS(cache_path)
    cs2step_artifact_assert_fingerprint(
      stored = bundle$metadata$fingerprint %||% NULL,
      expected = request_fingerprint,
      context = "strategic_prediction cache"
    )
    return(cs2step_unpack_predictor(
      bundle,
      conda_env = conda_env,
      conda_env_required = conda_env_required
    ))
  }

  fit <- if (identical(model, "glm")) {
    cs2step_eval_outcome_model_glm(
      Y = as.numeric(Y),
      W_idx = W_idx_train,
      factor_levels = factor_levels,
      diff = diff,
      pair_id = pair_id,
      profile_order = profile_order,
      varcov_cluster_variable = varcov_cluster_variable,
      use_regularization = use_regularization,
      nFolds_glm = nFolds_glm
    )
  } else {
    cs2step_eval_outcome_model_neural(
      Y = as.numeric(Y),
      W_idx = W_idx_train,
      names_list = names_list,
      factor_levels = factor_levels,
      diff = diff,
      pair_id = pair_id,
      profile_order = profile_order,
      X = X,
      conda_env = conda_env,
      conda_env_required = conda_env_required,
      neural_mcmc_control = fit_control,
      varcov_cluster_variable = varcov_cluster_variable,
      nFolds_glm = nFolds_glm
    )
  }

  if (identical(model, "neural") && is.null(fit$fit_metrics)) {
    fit$fit_metrics <- fit$neural_model_info$fit_metrics %||% NULL
  }

  out <- structure(
    list(
      model_type = model,
      mode = mode_use,
      encoder = list(
        factor_names = names(names_list),
        names_list = names_list,
        factor_levels = factor_levels,
        unknown_policy = "holdout"
      ),
      fit = fit,
      metadata = list(
        call = match.call(),
        timestamp = Sys.time(),
        conda_env = if (identical(model, "neural")) conda_env else NULL,
        conda_env_required = if (identical(model, "neural")) conda_env_required else NULL,
        fingerprint = request_fingerprint
      )
    ),
    class = "strategic_predictor"
  )
  if (!is.null(cache_path)) {
    save_strategic_predictor(
      out,
      file = cache_path,
      overwrite = TRUE,
      compress = cache_compress
    )
  }
  out
}

cs2step_glm_build_design <- function(W_idx, main_info, interaction_info) {
  W_idx <- as.matrix(W_idx)
  main_info <- as.data.frame(main_info)
  interaction_info <- as.data.frame(interaction_info)

  if (nrow(main_info) > 0) {
    main_dat <- apply(main_info, 1, function(row_) {
      d_ <- as.integer(row_[["d"]])
      l_ <- as.integer(row_[["l"]])
      1L * (W_idx[, d_] == l_)
    })
    if (length(main_dat) == nrow(W_idx)) {
      main_dat <- matrix(main_dat, ncol = 1)
    }
  } else {
    main_dat <- matrix(numeric(0), nrow = nrow(W_idx), ncol = 0)
  }

  if (nrow(interaction_info) > 0) {
    inter_dat <- apply(interaction_info, 1, function(row_) {
      d_ <- as.integer(row_[["d"]])
      l_ <- as.integer(row_[["l"]])
      dp_ <- as.integer(row_[["dp"]])
      lp_ <- as.integer(row_[["lp"]])
      1L * (W_idx[, d_] == l_) * 1L * (W_idx[, dp_] == lp_)
    })
    if (length(inter_dat) == nrow(W_idx)) {
      inter_dat <- matrix(inter_dat, ncol = 1)
    }
  } else {
    inter_dat <- matrix(numeric(0), nrow = nrow(W_idx), ncol = 0)
  }

  cbind(main_dat, inter_dat)
}

cs2step_rmvnorm <- function(n, mu, Sigma) {
  mu <- as.numeric(mu)
  p <- length(mu)
  Sigma <- as.matrix(Sigma)
  if (n == 0L) {
    return(matrix(numeric(0), nrow = 0, ncol = p))
  }
  if (!all(dim(Sigma) == c(p, p))) {
    stop("Sigma has incompatible dimensions.", call. = FALSE)
  }
  Sigma[!is.finite(Sigma)] <- 0

  R <- tryCatch(chol(Sigma), error = function(e) NULL)
  if (is.null(R)) {
    eig <- eigen(Sigma, symmetric = TRUE)
    vals <- pmax(eig$values, 0)
    R <- t(eig$vectors %*% diag(sqrt(vals), nrow = p, ncol = p))
  }
  Z <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  sweep(Z %*% R, 2, mu, `+`)
}

cs2step_glm_predict_internal <- function(object,
                                        W_new,
                                        pair_id = NULL,
                                        profile_order = NULL,
                                        type = c("response", "link"),
                                        interval = c("none", "ci", "draws"),
                                        level = 0.95,
                                        n_draws = 0L,
                                        seed = NULL) {
  type <- match.arg(type)
  interval <- match.arg(interval)
  if (interval != "none" && (is.null(n_draws) || n_draws < 1L)) {
    n_draws <- 500L
  }
  n_draws <- as.integer(n_draws)

  enc <- object$encoder
  W_new <- cs2step_align_W(W_new, enc$factor_names)
  W_idx <- cs2step_encode_W_indices(W_new, enc$names_list, unknown = "holdout", pad_unknown = 0L)

  X <- cs2step_glm_build_design(W_idx, object$fit$main_info, object$fit$interaction_info)

  if (identical(object$mode, "pairwise")) {
    cs2step_validate_pairwise_ids(pair_id, nrow(W_idx))
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W_idx,
      profile_order = profile_order,
      competing_group_variable_candidate = NULL
    )
    pair_mat <- pair_info$pair_mat
    X <- X[pair_mat[, 1], , drop = FALSE] - X[pair_mat[, 2], , drop = FALSE]
  }

  beta <- c(object$fit$intercept, object$fit$coefficients)
  if (ncol(X) != length(object$fit$coefficients)) {
    stop("Prediction design matrix is not aligned with fitted coefficients.", call. = FALSE)
  }
  eta <- as.numeric(beta[1] + X %*% beta[-1])
  pred <- if (type == "link") eta else stats::plogis(eta)

  if (interval == "none") {
    return(pred)
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  V <- object$fit$vcov
  if (is.null(V) || !is.matrix(V) || any(!is.finite(dim(V)))) {
    stop("Fitted GLM object does not contain a usable variance-covariance matrix.", call. = FALSE)
  }

  beta_draws <- cs2step_rmvnorm(n_draws, mu = beta, Sigma = V)
  lin_draws <- X %*% t(beta_draws[, -1, drop = FALSE])
  lin_draws <- sweep(lin_draws, 2, beta_draws[, 1], `+`)
  draw_mat <- if (type == "link") {
    lin_draws
  } else {
    stats::plogis(lin_draws)
  }

  alpha <- (1 - level) / 2
  qs <- c(alpha, 1 - alpha)
  q_mat <- matrixStats::rowQuantiles(draw_mat, probs = qs, drop = FALSE)
  out_df <- data.frame(
    fit = pred,
    lo = q_mat[, 1],
    hi = q_mat[, 2]
  )
  if (interval == "ci") {
    return(out_df)
  }
  list(
    fit = pred,
    interval = out_df,
    draws = draw_mat,
    level = level
  )
}

cs2step_neural_to_index_matrix <- function(x_mat, factor_levels) {
  x_mat <- as.matrix(x_mat)
  x_int <- matrix(as.integer(x_mat), nrow = nrow(x_mat), ncol = ncol(x_mat))
  n_cols <- ncol(x_int)
  n_levels <- as.integer(factor_levels)
  if (length(n_levels) != n_cols) {
    n_levels <- rep_len(n_levels, n_cols)
  }
  for (d_ in seq_len(n_cols)) {
    max_level <- n_levels[[d_]]
    missing_level <- max_level + 1L
    col_vals <- x_int[, d_]
    col_vals[is.na(col_vals) | col_vals < 1L | col_vals > max_level] <- missing_level
    x_int[, d_] <- col_vals - 1L
  }
  x_int
}

cs2step_neural_factor_schema_from_inputs <- function(factor_schema = NULL,
                                                     names_list = NULL,
                                                     p_list = NULL) {
  schema <- list()
  if (!is.null(factor_schema)) {
    if (!is.list(factor_schema)) {
      stop("'factor_schema' must be a list.", call. = FALSE)
    }
    schema_names <- names(factor_schema)
    known_fields <- cs2step_neural_factor_schema_known_fields()
    schema <- if (is.null(schema_names) || !any(schema_names %in% known_fields)) {
      list(names_list = factor_schema)
    } else {
      factor_schema
    }
  }
  if (!is.null(names_list)) {
    schema$names_list <- names_list
  }
  if (!is.null(p_list)) {
    schema$p_list <- p_list
  }
  if (!length(schema)) {
    return(NULL)
  }
  schema
}

cs2step_neural_merge_factor_schema <- function(newdata_schema = NULL,
                                               newdata_names_list = NULL,
                                               newdata_p_list = NULL,
                                               explicit_schema = NULL) {
  base <- cs2step_neural_factor_schema_from_inputs(
    factor_schema = newdata_schema,
    names_list = newdata_names_list,
    p_list = newdata_p_list
  )
  explicit <- cs2step_neural_factor_schema_from_inputs(factor_schema = explicit_schema)
  if (is.null(base)) {
    return(explicit)
  }
  if (is.null(explicit)) {
    return(base)
  }
  modifyList(base, explicit)
}

cs2step_neural_schema_text_dim <- function(model_info, params = NULL) {
  text_dim <- suppressWarnings(as.integer(model_info$text_semantic_dim %||% NA_integer_))
  if (length(text_dim) == 1L && !is.na(text_dim) && text_dim > 0L) {
    return(text_dim)
  }
  for (name in c("W_factor_name_text", "W_level_name_text")) {
    mat <- params[[name]] %||% NULL
    if (!is.null(mat)) {
      arr <- tryCatch(cs2step_neural_to_r_array(mat), error = function(e) NULL)
      if (!is.null(arr) && length(dim(arr)) >= 2L && dim(arr)[[1L]] > 0L) {
        return(as.integer(dim(arr)[[1L]]))
      }
    }
  }
  0L
}

cs2step_neural_reorder_schema_matrix <- function(x,
                                                 row_names,
                                                 n_cols,
                                                 field,
                                                 allow_missing_rows = FALSE) {
  mat <- as.matrix(x)
  storage.mode(mat) <- "double"
  if (ncol(mat) != n_cols) {
    stop(
      sprintf(
        "'factor_schema$%s' has width %d but the fitted text semantic width is %d.",
        field,
        ncol(mat),
        n_cols
      ),
      call. = FALSE
    )
  }
  row_names <- as.character(row_names)
  mat_rows <- rownames(mat)
  if (!is.null(mat_rows) && all(row_names %in% mat_rows)) {
    mat <- mat[row_names, , drop = FALSE]
  } else if (nrow(mat) == length(row_names)) {
    rownames(mat) <- row_names
  } else if (isTRUE(allow_missing_rows)) {
    out <- matrix(0, nrow = length(row_names), ncol = n_cols)
    rownames(out) <- row_names
    colnames(out) <- colnames(mat)
    if (!is.null(mat_rows)) {
      ok <- match(intersect(row_names, mat_rows), row_names)
      src <- match(row_names[ok], mat_rows)
      out[ok, ] <- mat[src, , drop = FALSE]
      return(out)
    }
    stop(
      sprintf(
        "'factor_schema$%s' must have %d row(s).",
        field,
        length(row_names)
      ),
      call. = FALSE
    )
  } else {
    stop(
      sprintf(
        "'factor_schema$%s' must have %d row(s).",
        field,
        length(row_names)
      ),
      call. = FALSE
    )
  }
  mat
}

cs2step_neural_schema_factor_text <- function(schema,
                                              factor_names,
                                              text_dim,
                                              text_embedding_fn = NULL) {
  supplied <- schema$factor_name_text %||% NULL
  if (!is.null(supplied)) {
    return(cs2step_neural_reorder_schema_matrix(
      supplied,
      row_names = factor_names,
      n_cols = text_dim,
      field = "factor_name_text"
    ))
  }
  if (is.null(text_embedding_fn) || !is.function(text_embedding_fn)) {
    stop(
      "Prediction-time factor_schema requires 'text_embedding_fn' or 'factor_schema$factor_name_text'.",
      call. = FALSE
    )
  }
  out <- cs_foundation_text_embed(text_embedding_fn, factor_names)
  if (ncol(out) != text_dim) {
    stop(
      "text_embedding_fn returned factor-name embeddings with incompatible width.",
      call. = FALSE
    )
  }
  rownames(out) <- factor_names
  out
}

cs2step_neural_schema_level_text_one <- function(x,
                                                 levels_here,
                                                 text_dim,
                                                 field) {
  row_names <- c(as.character(levels_here), "__holdout__")
  mat <- as.matrix(x)
  storage.mode(mat) <- "double"
  if (ncol(mat) != text_dim) {
    stop(
      sprintf(
        "'factor_schema$level_name_text[[%s]]' has width %d but the fitted text semantic width is %d.",
        field,
        ncol(mat),
        text_dim
      ),
      call. = FALSE
    )
  }
  mat_rows <- rownames(mat)
  out <- matrix(0, nrow = length(row_names), ncol = text_dim)
  rownames(out) <- row_names
  colnames(out) <- colnames(mat)
  if (!is.null(mat_rows) && all(levels_here %in% mat_rows)) {
    out[seq_along(levels_here), ] <- mat[as.character(levels_here), , drop = FALSE]
    if ("__holdout__" %in% mat_rows) {
      out[nrow(out), ] <- mat["__holdout__", , drop = FALSE]
    }
    return(out)
  }
  if (nrow(mat) == length(levels_here)) {
    out[seq_along(levels_here), ] <- mat
    return(out)
  }
  if (nrow(mat) == length(row_names)) {
    rownames(mat) <- row_names
    return(mat)
  }
  stop(
    sprintf(
      "'factor_schema$level_name_text[[%s]]' must have %d or %d row(s).",
      field,
      length(levels_here),
      length(row_names)
    ),
    call. = FALSE
  )
}

cs2step_neural_schema_level_text <- function(schema,
                                             names_list,
                                             text_dim,
                                             text_embedding_fn = NULL) {
  factor_names <- names(names_list)
  supplied <- schema$level_name_text %||% NULL
  if (!is.null(supplied)) {
    if (is.matrix(supplied) && length(factor_names) == 1L) {
      supplied <- setNames(list(supplied), factor_names)
    }
    if (!is.list(supplied)) {
      stop("'factor_schema$level_name_text' must be a list of matrices.", call. = FALSE)
    }
    return(setNames(lapply(seq_along(factor_names), function(i) {
      factor_name <- factor_names[[i]]
      x <- if (!is.null(names(supplied)) && factor_name %in% names(supplied)) {
        supplied[[factor_name]]
      } else {
        supplied[[i]]
      }
      if (is.null(x)) {
        stop(
          sprintf("Missing level_name_text for factor '%s'.", factor_name),
          call. = FALSE
        )
      }
      cs2step_neural_schema_level_text_one(
        x,
        levels_here = names_list[[factor_name]][[1]],
        text_dim = text_dim,
        field = factor_name
      )
    }), factor_names))
  }
  if (is.null(text_embedding_fn) || !is.function(text_embedding_fn)) {
    stop(
      "Prediction-time factor_schema requires 'text_embedding_fn' or 'factor_schema$level_name_text'.",
      call. = FALSE
    )
  }
  setNames(lapply(factor_names, function(factor_name) {
    levels_here <- as.character(names_list[[factor_name]][[1]])
    emb <- if (length(levels_here) > 0L) {
      cs_foundation_text_embed(text_embedding_fn, levels_here)
    } else {
      matrix(numeric(0), nrow = 0L, ncol = text_dim)
    }
    if (ncol(emb) != text_dim) {
      stop(
        "text_embedding_fn returned level-name embeddings with incompatible width.",
        call. = FALSE
      )
    }
    out <- matrix(0, nrow = length(levels_here) + 1L, ncol = text_dim)
    rownames(out) <- c(levels_here, "__holdout__")
    colnames(out) <- colnames(emb)
    if (length(levels_here) > 0L) {
      out[seq_along(levels_here), ] <- emb
    }
    out
  }), factor_names)
}

cs2step_neural_struct_feature_names <- function(model_info,
                                                kind = c("factor", "level")) {
  kind <- match.arg(kind)
  fallback <- if (identical(kind, "factor")) {
    neural_fused_default_factor_struct_feature_names()
  } else {
    neural_fused_default_level_struct_feature_names()
  }
  field <- paste0(kind, "_struct_feature_names")
  dim_field <- paste0(kind, "_struct_dim")
  features <- as.character(model_info[[field]] %||% character(0))
  dim_use <- suppressWarnings(as.integer(model_info[[dim_field]] %||% length(features)))
  if (length(dim_use) != 1L || is.na(dim_use) || dim_use < 1L) {
    dim_use <- length(features)
  }
  if (length(features) < 1L) {
    features <- fallback
  }
  if (dim_use > length(features)) {
    features <- c(features, sprintf("%s_struct_extra_%d", kind, seq_len(dim_use - length(features))))
  }
  if (dim_use > 0L && length(features) > dim_use) {
    features <- features[seq_len(dim_use)]
  }
  features
}

cs2step_neural_align_struct_matrix <- function(x,
                                               row_names,
                                               feature_names,
                                               field) {
  mat <- as.matrix(x)
  storage.mode(mat) <- "double"
  row_names <- as.character(row_names)
  out <- matrix(0, nrow = length(row_names), ncol = length(feature_names))
  rownames(out) <- row_names
  colnames(out) <- feature_names
  if (nrow(mat) != length(row_names)) {
    mat_rows <- rownames(mat)
    if (!is.null(mat_rows) && all(row_names %in% mat_rows)) {
      mat <- mat[row_names, , drop = FALSE]
    } else {
      stop(
        sprintf("'factor_schema$%s' must have %d row(s).", field, length(row_names)),
        call. = FALSE
      )
    }
  }
  mat_cols <- colnames(mat)
  if (!is.null(mat_cols)) {
    ok <- match(intersect(feature_names, mat_cols), feature_names)
    src <- match(feature_names[ok], mat_cols)
    out[, ok] <- mat[, src, drop = FALSE]
  } else if (ncol(mat) > 0L) {
    n_copy <- min(ncol(mat), ncol(out))
    out[, seq_len(n_copy)] <- mat[, seq_len(n_copy), drop = FALSE]
  }
  out
}

cs2step_neural_default_factor_struct_matrix <- function(names_list, feature_names) {
  factor_names <- names(names_list)
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))
  out <- matrix(0, nrow = length(factor_names), ncol = length(feature_names))
  rownames(out) <- factor_names
  colnames(out) <- feature_names
  put <- function(name, value) {
    if (name %in% colnames(out)) {
      out[, name] <<- value
    }
  }
  put("type_categorical", 1)
  put("cardinality_log", log1p(factor_levels))
  put("has_numeric_levels", vapply(names_list, function(x) {
    any(is.finite(suppressWarnings(as.numeric(x[[1]]))))
  }, logical(1)))
  out
}

cs2step_neural_default_level_struct_matrices <- function(names_list, feature_names) {
  factor_names <- names(names_list)
  setNames(lapply(factor_names, function(factor_name) {
    levels_here <- as.character(names_list[[factor_name]][[1]])
    row_names <- c(levels_here, "__holdout__")
    out <- matrix(0, nrow = length(row_names), ncol = length(feature_names))
    rownames(out) <- row_names
    colnames(out) <- feature_names
    n_levels <- length(levels_here)
    if (n_levels > 0L) {
      rank01 <- if (n_levels > 1L) (seq_len(n_levels) - 1) / (n_levels - 1) else rep(0, n_levels)
      numeric_levels <- suppressWarnings(as.numeric(levels_here))
      finite_numeric <- is.finite(numeric_levels)
      if ("has_raw_value" %in% colnames(out)) {
        out[seq_len(n_levels), "has_raw_value"] <- as.numeric(finite_numeric)
      }
      if ("raw_value_log1p_signed" %in% colnames(out)) {
        out[seq_len(n_levels), "raw_value_log1p_signed"] <-
          ifelse(finite_numeric, sign(numeric_levels) * log1p(abs(numeric_levels)), 0)
      }
      if ("level_rank01" %in% colnames(out)) {
        out[seq_len(n_levels), "level_rank01"] <- rank01
      }
      if ("level_quantile" %in% colnames(out)) {
        out[seq_len(n_levels), "level_quantile"] <- rank01
      }
      if ("level_z_score" %in% colnames(out)) {
        out[seq_len(n_levels), "level_z_score"] <- (2 * rank01) - 1
      }
    }
    if ("is_holdout" %in% colnames(out)) {
      out[nrow(out), "is_holdout"] <- 1
    }
    out
  }), factor_names)
}

cs2step_neural_schema_structural_info <- function(schema, names_list, model_info) {
  factor_features <- cs2step_neural_struct_feature_names(model_info, "factor")
  level_features <- cs2step_neural_struct_feature_names(model_info, "level")
  factor_names <- names(names_list)
  factor_mat <- if (!is.null(schema$factor_struct_matrix)) {
    cs2step_neural_align_struct_matrix(
      schema$factor_struct_matrix,
      row_names = factor_names,
      feature_names = factor_features,
      field = "factor_struct_matrix"
    )
  } else {
    cs2step_neural_default_factor_struct_matrix(names_list, factor_features)
  }
  level_input <- schema$level_struct_matrices %||% NULL
  level_mats <- if (!is.null(level_input)) {
    if (is.matrix(level_input) && length(factor_names) == 1L) {
      level_input <- setNames(list(level_input), factor_names)
    }
    if (!is.list(level_input)) {
      stop("'factor_schema$level_struct_matrices' must be a list of matrices.", call. = FALSE)
    }
    setNames(lapply(seq_along(factor_names), function(i) {
      factor_name <- factor_names[[i]]
      level_rows <- c(as.character(names_list[[factor_name]][[1]]), "__holdout__")
      x <- if (!is.null(names(level_input)) && factor_name %in% names(level_input)) {
        level_input[[factor_name]]
      } else {
        level_input[[i]]
      }
      if (is.null(x)) {
        stop(
          sprintf("Missing level_struct_matrices for factor '%s'.", factor_name),
          call. = FALSE
        )
      }
      x <- as.matrix(x)
      storage.mode(x) <- "double"
      if (nrow(x) == length(level_rows) - 1L) {
        holdout <- matrix(0, nrow = 1L, ncol = ncol(x))
        colnames(holdout) <- colnames(x)
        x <- rbind(x, holdout)
        rownames(x) <- level_rows
      }
      cs2step_neural_align_struct_matrix(
        x,
        row_names = level_rows,
        feature_names = level_features,
        field = sprintf("level_struct_matrices[[%s]]", factor_name)
      )
    }), factor_names)
  } else {
    cs2step_neural_default_level_struct_matrices(names_list, level_features)
  }
  list(
    factor_struct_matrix = factor_mat,
    factor_struct_feature_names = factor_features,
    level_struct_matrices = level_mats,
    level_struct_feature_names = level_features
  )
}

cs2step_neural_prediction_factor_index_list <- function(factor_levels, implicit = FALSE) {
  factor_levels <- as.integer(factor_levels)
  out <- vector("list", length(factor_levels))
  offset <- 0L
  for (i in seq_along(factor_levels)) {
    n_i <- max(0L, factor_levels[[i]] - as.integer(isTRUE(implicit)))
    out[[i]] <- if (n_i > 0L) as.integer(offset + seq_len(n_i) - 1L) else integer(0)
    offset <- offset + n_i
  }
  out
}

cs2step_neural_prepare_factor_schema_prediction <- function(object,
                                                            W,
                                                            model_info,
                                                            params,
                                                            factor_schema = NULL,
                                                            text_embedding_fn = NULL) {
  if (is.null(factor_schema)) {
    return(NULL)
  }
  if (!identical(neural_factor_tokenization(model_info), "fused")) {
    stop("Prediction-time factor_schema requires factor_tokenization = 'fused'.", call. = FALSE)
  }
  text_dim <- cs2step_neural_schema_text_dim(model_info, params = params)
  if (text_dim < 1L) {
    stop("Prediction-time factor_schema requires a predictor trained with text semantics.", call. = FALSE)
  }
  if (is.null(params$W_factor_name_text) || is.null(params$W_level_name_text)) {
    stop(
      "Prediction-time factor_schema requires learned factor and level text projection parameters.",
      call. = FALSE
    )
  }

  schema <- factor_schema
  W_df <- as.data.frame(W, check.names = FALSE)
  if (ncol(W_df) < 1L) {
    stop("'W' must contain at least one factor column.", call. = FALSE)
  }
  names_list <- cs_foundation_normalize_names_list_local(
    names_list = schema$names_list %||% NULL,
    W = W_df,
    p_list = schema$p_list %||% NULL
  )
  factor_names <- names(names_list)
  if (is.null(factor_names) || any(is.na(factor_names) | !nzchar(factor_names))) {
    stop("'factor_schema' must define non-empty factor names.", call. = FALSE)
  }
  W_cols <- colnames(W_df)
  has_w_names <- !is.null(W_cols) &&
    length(W_cols) > 0L &&
    any(!is.na(W_cols) & nzchar(W_cols))
  if (!isTRUE(has_w_names)) {
    if (ncol(W_df) != length(factor_names)) {
      stop("Unnamed prediction-time W must match factor_schema width exactly.", call. = FALSE)
    }
    colnames(W_df) <- factor_names
  }
  missing_cols <- setdiff(factor_names, colnames(W_df))
  if (length(missing_cols) > 0L) {
    stop(
      "Missing factor columns in newdata for factor_schema: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  W_use <- W_df[, factor_names, drop = FALSE]
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))
  neural_validate_factor_token_budget(
    n_factors = length(factor_names),
    max_factor_tokens = model_info$max_factor_tokens %||% NULL,
    context = "Prediction-time factor_schema"
  )

  text_fn <- text_embedding_fn %||%
    schema$text_embedding_fn %||%
    object$metadata$text_embedding_fn %||%
    NULL
  factor_name_text <- cs2step_neural_schema_factor_text(
    schema,
    factor_names = factor_names,
    text_dim = text_dim,
    text_embedding_fn = text_fn
  )
  level_name_text <- cs2step_neural_schema_level_text(
    schema,
    names_list = names_list,
    text_dim = text_dim,
    text_embedding_fn = text_fn
  )
  structural_info <- cs2step_neural_schema_structural_info(
    schema,
    names_list = names_list,
    model_info = model_info
  )
  neural_validate_fused_structural_info(
    factor_struct_matrix = structural_info$factor_struct_matrix,
    level_struct_matrices = structural_info$level_struct_matrices,
    factor_struct_feature_names = structural_info$factor_struct_feature_names,
    level_struct_feature_names = structural_info$level_struct_feature_names,
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    context = "prediction-time factor_schema"
  )

  model_info_new <- model_info
  model_info_new$factor_levels <- as.integer(factor_levels)
  model_info_new$n_factors <- as.integer(length(factor_names))
  model_info_new$factor_index_list <- cs2step_neural_prediction_factor_index_list(
    factor_levels,
    implicit = isTRUE(model_info$implicit)
  )
  model_info_new$factor_name_text <- factor_name_text
  model_info_new$level_name_text <- level_name_text
  model_info_new$factor_struct_matrix <- structural_info$factor_struct_matrix
  model_info_new$factor_struct_feature_names <- structural_info$factor_struct_feature_names
  model_info_new$factor_struct_dim <- as.integer(ncol(structural_info$factor_struct_matrix))
  model_info_new$level_struct_matrices <- structural_info$level_struct_matrices
  model_info_new$level_struct_feature_names <- structural_info$level_struct_feature_names
  model_info_new$level_struct_dim <- as.integer(ncol(structural_info$level_struct_matrices[[1L]]))
  model_info_new$factor_schema_supplied <- TRUE
  model_info_new$default_factor_order <- seq.int(0L, length(factor_names) - 1L)
  model_info_new$factor_order_by_experiment <- NULL
  model_info_new$n_candidate_tokens <- neural_active_candidate_token_budget(model_info_new)
  model_info_new$text_semantic_dim <- as.integer(text_dim)

  list(
    W = W_use,
    W_idx = cs2step_encode_W_indices(
      W_use,
      names_list,
      unknown = "holdout",
      pad_unknown = 1L
    ),
    model_info = model_info_new,
    names_list = names_list,
    factor_names = factor_names,
    factor_order_new = NULL
  )
}

cs2step_neural_covariate_aliases <- function(model_info) {
  schema <- model_info$foundation_prediction_schema %||% list()
  aliases <- model_info$covariate_aliases %||%
    schema$covariate_raw_to_internal %||%
    character(0)
  alias_names <- names(aliases)
  aliases <- as.character(aliases)
  if (is.null(alias_names)) {
    alias_names <- rep.int("", length(aliases))
  }
  names(aliases) <- alias_names
  keep <- nzchar(alias_names) & !is.na(aliases) & nzchar(aliases)
  aliases <- aliases[keep]
  aliases[!duplicated(names(aliases))]
}

cs2step_neural_strict_prediction_covariate_schema <- function(model_info) {
  schema <- model_info$foundation_prediction_schema %||% list()
  isTRUE(model_info$strict_prediction_covariate_schema) ||
    isTRUE(schema$strict_covariate_schema)
}

cs2step_neural_format_covariate_names <- function(x) {
  paste(dQuote(as.character(x)), collapse = ", ")
}

cs2step_neural_conflicting_covariate_values <- function(a, b) {
  a_num <- suppressWarnings(as.numeric(a))
  b_num <- suppressWarnings(as.numeric(b))
  if (length(a_num) != length(b_num)) {
    return(TRUE)
  }
  same <- (is.na(a_num) & is.na(b_num)) |
    (!is.na(a_num) & !is.na(b_num) & a_num == b_num)
  !all(same)
}

cs2step_neural_map_resp_cov_columns <- function(resp_cov_new,
                                                covariate_names,
                                                model_info) {
  input_names <- colnames(resp_cov_new)
  aliases <- cs2step_neural_covariate_aliases(model_info)
  strict <- cs2step_neural_strict_prediction_covariate_schema(model_info)

  direct_idx <- match(input_names, covariate_names)
  alias_targets <- aliases[input_names]
  alias_idx <- match(as.character(alias_targets), covariate_names)
  ambiguous <- which(!is.na(direct_idx) & !is.na(alias_idx) & direct_idx != alias_idx)
  if (length(ambiguous) > 0L) {
    stop(
      sprintf(
        "Prediction-time X column %s is ambiguous because it matches both a fitted covariate and a raw covariate alias.",
        cs2step_neural_format_covariate_names(input_names[ambiguous])
      ),
      call. = FALSE
    )
  }

  target_idx <- direct_idx
  use_alias <- is.na(target_idx) & !is.na(alias_idx)
  target_idx[use_alias] <- alias_idx[use_alias]

  unknown <- input_names[is.na(target_idx)]
  if (strict && length(unknown) > 0L) {
    stop(
      sprintf(
        "Unknown prediction-time X columns for foundation predictor: %s.",
        cs2step_neural_format_covariate_names(unknown)
      ),
      call. = FALSE
    )
  }

  ok <- which(!is.na(target_idx))
  if (length(ok) < 1L) {
    return(list(data = resp_cov_new[, integer(0), drop = FALSE], idx = integer(0)))
  }

  keep_cols <- integer(0)
  keep_idx <- integer(0)
  for (k in ok) {
    existing <- match(target_idx[[k]], keep_idx)
    if (is.na(existing)) {
      keep_cols <- c(keep_cols, k)
      keep_idx <- c(keep_idx, target_idx[[k]])
      next
    }
    if (cs2step_neural_conflicting_covariate_values(
      resp_cov_new[[keep_cols[[existing]]]],
      resp_cov_new[[k]]
    )) {
      stop(
        sprintf(
          "Conflicting prediction-time X columns for fitted covariate %s.",
          dQuote(covariate_names[[target_idx[[k]]]])
        ),
        call. = FALSE
      )
    }
  }

  mapped <- resp_cov_new[, keep_cols, drop = FALSE]
  colnames(mapped) <- covariate_names[keep_idx]
  list(data = mapped, idx = keep_idx)
}

cs2step_neural_prepare_resp_cov <- function(resp_cov_new,
                                            model_info,
                                            n_rows,
                                            experiment_idx = NULL) {
  covariate_names <- as.character(model_info$covariate_names %||% character(0))
  n_covariates <- length(covariate_names)
  encoding <- neural_covariate_value_encoding(model_info)

  if (n_covariates < 1L) {
    return(list(
      values = matrix(0, nrow = n_rows, ncol = 0L),
      present = matrix(0, nrow = n_rows, ncol = 0L),
      order = NULL
    ))
  }

  values <- matrix(0, nrow = n_rows, ncol = n_covariates)
  present <- matrix(0, nrow = n_rows, ncol = n_covariates)
  colnames(values) <- covariate_names
  colnames(present) <- covariate_names
  order_idx <- if (identical(encoding, "shared_projection")) {
    as.integer(
      model_info$default_covariate_order %||%
        seq.int(0L, n_covariates - 1L)
    )
  } else {
    NULL
  }
  order_matrix <- function(order_idx_use = order_idx) {
    if (is.null(order_idx_use)) {
      return(NULL)
    }
    neural_build_default_covariate_order_matrix(
      order_idx = order_idx_use,
      n_rows = n_rows,
      max_covariate_tokens = model_info$max_covariate_tokens %||% NULL
    )
  }

  if (is.null(resp_cov_new)) {
    return(list(
      values = values,
      present = present,
      order = order_matrix()
    ))
  }

  if (is.atomic(resp_cov_new) && !is.matrix(resp_cov_new) && !is.data.frame(resp_cov_new)) {
    if (n_covariates != 1L) {
      stop(
        "Unnamed prediction-time X can only be used when the fitted model has exactly one covariate.",
        call. = FALSE
      )
    }
    resp_cov_new <- data.frame(
      stats::setNames(list(as.numeric(resp_cov_new)), covariate_names),
      check.names = FALSE
    )
  } else {
    resp_cov_new <- as.data.frame(resp_cov_new, check.names = FALSE)
  }

  if (nrow(resp_cov_new) == 1L && n_rows > 1L) {
    resp_cov_new <- resp_cov_new[rep.int(1L, n_rows), , drop = FALSE]
    rownames(resp_cov_new) <- NULL
  }
  if (nrow(resp_cov_new) != n_rows) {
    stop(
      "Prediction-time X must have either one row or the same number of rows as the prediction data.",
      call. = FALSE
    )
  }

  if (ncol(resp_cov_new) < 1L) {
    return(list(values = values, present = present, order = order_matrix()))
  }

  if (is.null(colnames(resp_cov_new))) {
    if (ncol(resp_cov_new) != n_covariates) {
      stop(
        "Unnamed prediction-time X must match the fitted covariate width exactly.",
        call. = FALSE
      )
    }
    colnames(resp_cov_new) <- covariate_names
  }

  mapped_cov <- cs2step_neural_map_resp_cov_columns(
    resp_cov_new = resp_cov_new,
    covariate_names = covariate_names,
    model_info = model_info
  )
  resp_cov_new <- mapped_cov$data
  idx <- mapped_cov$idx
  if (length(idx) < 1L) {
    return(list(
      values = values,
      present = present,
      order = order_matrix()
    ))
  }

  for (k in seq_along(idx)) {
    raw_vals <- resp_cov_new[[k]]
    raw_missing <- is.na(raw_vals)
    if (is.factor(raw_vals)) {
      raw_vals <- as.character(raw_vals)
    }
    col_vals <- suppressWarnings(as.numeric(raw_vals))
    coerced_missing <- is.na(col_vals)
    if (any(coerced_missing & !raw_missing, na.rm = TRUE)) {
      stop(
        "Prediction-time X must contain numeric values or NA for fitted covariates.",
        call. = FALSE
      )
    }
    if (any(is.infinite(col_vals), na.rm = TRUE)) {
      stop(
        "Prediction-time X contains Inf values for fitted covariates.",
        call. = FALSE
      )
    }
    observed <- !is.na(col_vals)
    tgt <- idx[[k]]
    values[observed, tgt] <- col_vals[observed]
    values[!observed, tgt] <- 0
    present[observed, tgt] <- 1
    present[!observed, tgt] <- 0
  }

  list(
    values = values,
    present = present,
    order = order_matrix()
  )
}

cs2step_neural_prepare_factor_order <- function(factor_order_new, model_info, n_rows) {
  if (!identical(neural_factor_tokenization(model_info), "fused")) {
    return(NULL)
  }
  if (is.null(factor_order_new)) {
    return(NULL)
  }

  max_token_slots <- neural_max_factor_token_slots(model_info = model_info)
  if (is.data.frame(factor_order_new)) {
    factor_order_new <- as.matrix(factor_order_new)
  }

  if (is.matrix(factor_order_new)) {
    order_mat <- matrix(
      as.integer(factor_order_new),
      nrow = nrow(factor_order_new),
      ncol = ncol(factor_order_new)
    )
    if (nrow(order_mat) == 1L && n_rows > 1L) {
      order_mat <- order_mat[rep.int(1L, n_rows), , drop = FALSE]
    }
    if (nrow(order_mat) != n_rows) {
      stop(
        "Prediction-time factor_order must have one row or the same number of rows as the prediction data.",
        call. = FALSE
      )
    }
    if (ncol(order_mat) > max_token_slots) {
      stop(
        sprintf(
          "Prediction-time factor_order has %d columns but max_factor_tokens=%d only supports %d fused factor tokens.",
          ncol(order_mat),
          model_info$max_factor_tokens %||% neural_default_max_factor_tokens(),
          max_token_slots
        ),
        call. = FALSE
      )
    }
    out <- matrix(-1L, nrow = n_rows, ncol = max_token_slots)
    if (ncol(order_mat) > 0L) {
      order_mat[is.na(order_mat) | order_mat < 0L] <- -1L
      out[, seq_len(ncol(order_mat))] <- order_mat
    }
    return(out)
  }

  neural_build_default_factor_order_matrix(
    order_idx = as.integer(factor_order_new),
    n_rows = n_rows,
    max_factor_tokens = model_info$max_factor_tokens %||% NULL
  )
}

cs2step_neural_prepare_experiment_index <- function(experiment_id, model_info, n_rows) {
  experiment_levels <- as.character(model_info$experiment_levels %||% character(0))
  default_experiment_index <- model_info$default_experiment_index %||% NULL
  if (!is.null(default_experiment_index)) {
    default_experiment_index <- as.integer(default_experiment_index)
  }

  if (is.null(experiment_id)) {
    if (is.null(default_experiment_index) || is.na(default_experiment_index)) {
      return(NULL)
    }
    return(rep.int(default_experiment_index, n_rows))
  }

  experiment_id <- as.character(experiment_id)
  if (length(experiment_id) == 1L && n_rows > 1L) {
    experiment_id <- rep.int(experiment_id, n_rows)
  }
  if (length(experiment_id) != n_rows) {
    stop(
      "Prediction-time experiment_id must have length one or the same number of rows as the prediction data.",
      call. = FALSE
    )
  }
  if (length(experiment_levels) < 1L) {
    return(NULL)
  }

  matched <- match(experiment_id, experiment_levels)
  if (all(is.na(matched))) {
    return(NULL)
  }
  if (any(is.na(matched))) {
    stop(
      "Prediction-time experiment_id must either all match pooled experiment levels or be omitted.",
      call. = FALSE
    )
  }
  as.integer(matched - 1L)
}

cs2step_neural_normalize_experiment_year <- function(experiment_year,
                                                     n_rows,
                                                     arg = "experiment_year") {
  if (is.null(experiment_year)) {
    return(NULL)
  }
  n_rows <- as.integer(n_rows)
  year <- experiment_year
  if (length(year) == 1L && n_rows > 1L) {
    year <- rep.int(year, n_rows)
  }
  if (length(year) != n_rows) {
    stop(
      sprintf(
        "Prediction-time %s must have length one or the same number of rows as the prediction data.",
        arg
      ),
      call. = FALSE
    )
  }
  if (is.factor(year)) {
    year <- as.character(year)
  }
  out <- rep(NA_integer_, n_rows)
  if (n_rows < 1L) {
    return(out)
  }
  is_missing <- is.na(year)
  if (any(!is_missing)) {
    year_num <- suppressWarnings(as.numeric(year[!is_missing]))
    if (any(is.na(year_num) | !is.finite(year_num))) {
      stop(sprintf("Prediction-time %s must be integer-like or NA.", arg), call. = FALSE)
    }
    if (any(abs(year_num - round(year_num)) > 1e-8)) {
      stop(sprintf("Prediction-time %s must be integer-like or NA.", arg), call. = FALSE)
    }
    out[!is_missing] <- as.integer(round(year_num))
  }
  out
}

cs2step_neural_prepare_time_embedding <- function(experiment_year,
                                                  model_info,
                                                  n_rows) {
  years <- cs2step_neural_normalize_experiment_year(experiment_year, n_rows)
  if (is.null(years)) {
    return(NULL)
  }
  emb <- do.call(rbind, lapply(years, function(year_i) {
    neural_encode_time_context(
      year = year_i,
      present = !is.na(year_i)
    )
  }))
  feature_names <- as.character(model_info$time_feature_names %||% neural_time_feature_names())
  if (!is.null(feature_names) && length(feature_names) > 0L) {
    missing_cols <- setdiff(feature_names, colnames(emb))
    if (length(missing_cols) > 0L) {
      extra <- matrix(0, nrow = nrow(emb), ncol = length(missing_cols))
      colnames(extra) <- missing_cols
      emb <- cbind(emb, extra)
    }
    emb <- emb[, feature_names, drop = FALSE]
  }
  emb
}

cs2step_neural_normalize_experiment_country <- function(experiment_country,
                                                        n_rows,
                                                        arg = "experiment_country") {
  if (is.null(experiment_country)) {
    return(NULL)
  }
  n_rows <- as.integer(n_rows)
  country <- experiment_country
  if (is.factor(country)) {
    country <- as.character(country)
  }
  if (length(country) == 1L && n_rows > 1L) {
    country <- rep.int(country, n_rows)
  }
  if (length(country) != n_rows) {
    stop(
      sprintf(
        "Prediction-time %s must have length one or the same number of rows as the prediction data.",
        arg
      ),
      call. = FALSE
    )
  }
  lapply(seq_len(n_rows), function(i) {
    cs2step_normalize_country(
      country[[i]],
      arg = sprintf("%s[%d]", arg, i)
    )
  })
}

cs2step_neural_prepare_place_embedding <- function(experiment_country,
                                                   model_info,
                                                   n_rows) {
  countries <- cs2step_neural_normalize_experiment_country(
    experiment_country,
    n_rows
  )
  if (is.null(countries)) {
    return(NULL)
  }
  feature_names_default <- neural_place_feature_names()
  if (length(countries) < 1L) {
    return(matrix(
      numeric(0),
      nrow = 0L,
      ncol = length(feature_names_default),
      dimnames = list(NULL, feature_names_default)
    ))
  }
  emb <- do.call(rbind, lapply(countries, function(country_i) {
    neural_encode_place_context(
      latitude = country_i$country_latitude,
      longitude = country_i$country_longitude,
      present = isTRUE(country_i$country_present)
    )
  }))
  feature_names <- as.character(model_info$place_feature_names %||% feature_names_default)
  if (!is.null(feature_names) && length(feature_names) > 0L) {
    missing_cols <- setdiff(feature_names, colnames(emb))
    if (length(missing_cols) > 0L) {
      extra <- matrix(0, nrow = nrow(emb), ncol = length(missing_cols))
      colnames(extra) <- missing_cols
      emb <- cbind(emb, extra)
    }
    emb <- emb[, feature_names, drop = FALSE]
  }
  emb
}

cs2step_neural_validate_place_context_request <- function(experiment_country,
                                                          model_info,
                                                          params = NULL) {
  if (is.null(experiment_country)) {
    return(invisible(TRUE))
  }
  if (!isTRUE(neural_place_context_enabled(model_info))) {
    stop(
      "Prediction-time experiment_country requires a predictor trained with place context. Retrain the predictor with experiment_country/place context enabled.",
      call. = FALSE
    )
  }
  if (!is.null(params) && is.null(params$W_place_context)) {
    stop(
      "Prediction-time experiment_country requires learned W_place_context parameters. Retrain the predictor with place context enabled.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

cs2step_neural_embed_experiment_description <- function(experiment_description,
                                                        model_info,
                                                        n_rows,
                                                        text_embedding_fn = NULL) {
  if (is.null(experiment_description)) {
    return(NULL)
  }
  if (is.null(text_embedding_fn) || !is.function(text_embedding_fn)) {
    stop(
      "Prediction-time experiment_description requires an available text_embedding_fn.",
      call. = FALSE
    )
  }
  text_dim <- as.integer(model_info$text_semantic_dim %||% 0L)
  if (text_dim < 1L) {
    stop(
      "Prediction-time experiment_description is unavailable because this model has no text semantics.",
      call. = FALSE
    )
  }
  desc_use <- as.character(experiment_description)
  if (length(desc_use) == 1L && n_rows > 1L) {
    desc_use <- rep.int(desc_use, n_rows)
  }
  if (!length(desc_use) %in% c(1L, n_rows)) {
    stop(
      "experiment_description must have length one or the same number of prediction rows.",
      call. = FALSE
    )
  }
  emb <- cs_foundation_text_embed(text_embedding_fn, desc_use)
  if (ncol(emb) != text_dim) {
    stop(
      "text_embedding_fn returned an experiment description embedding with incompatible width.",
      call. = FALSE
    )
  }
  if (nrow(emb) > 1L) {
    emb <- matrix(colMeans(emb), nrow = 1L)
  }
  emb
}

cs2step_neural_apply_experiment_description <- function(model_info,
                                                        experiment_description,
                                                        n_rows,
                                                        text_embedding_fn = NULL) {
  if (is.null(experiment_description) ||
      !identical(neural_experiment_token_mode(model_info), "description") &&
      !identical(neural_experiment_token_mode(model_info), "hybrid")) {
    return(model_info)
  }
  model_info$default_experiment_text <- cs2step_neural_embed_experiment_description(
    experiment_description = experiment_description,
    model_info = model_info,
    n_rows = n_rows,
    text_embedding_fn = text_embedding_fn
  )
  model_info$default_experiment_text_present <- TRUE
  model_info$default_experiment_index <- NULL
  model_info
}

cs2step_neural_prepare_prediction_data <- function(W_idx,
                                                   model_info,
                                                   competing_group_variable_candidate = NULL,
                                                   competing_group_variable_respondent = NULL,
                                                   resp_cov_new = NULL,
                                                   factor_order_new = NULL,
                                                   experiment_id = NULL,
                                                   experiment_country = NULL,
                                                   experiment_year = NULL,
                                                   pair_id = NULL,
                                                   profile_order = NULL,
                                                   mode = c("pairwise", "single")) {
  mode <- match.arg(mode)
  W_idx <- as.matrix(W_idx)
  party_missing_label <- model_info$party_missing_label %||%
    neural_missing_group_label("candidate")
  resp_party_missing_label <- model_info$resp_party_missing_label %||%
    neural_missing_group_label("respondent")

  if (!is.null(competing_group_variable_candidate) &&
      length(competing_group_variable_candidate) != nrow(W_idx)) {
    stop(
      sprintf(
        "competing_group_variable_candidate has %d elements but W has %d rows.",
        length(competing_group_variable_candidate),
        nrow(W_idx)
      ),
      call. = FALSE
    )
  }
  if (!is.null(competing_group_variable_respondent) &&
      length(competing_group_variable_respondent) != nrow(W_idx)) {
    stop(
      sprintf(
        "competing_group_variable_respondent has %d elements but W has %d rows.",
        length(competing_group_variable_respondent),
        nrow(W_idx)
      ),
      call. = FALSE
    )
  }

  pairwise <- identical(mode, "pairwise")
  if (pairwise) {
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W_idx,
      profile_order = profile_order,
      competing_group_variable_candidate = competing_group_variable_candidate
    )
    pair_mat <- pair_info$pair_mat
    X_left_raw <- W_idx[pair_mat[, 1], , drop = FALSE]
    X_right_raw <- W_idx[pair_mat[, 2], , drop = FALSE]
    n_rows <- nrow(X_left_raw)
    party_left_new <- if (is.null(competing_group_variable_candidate)) {
      NULL
    } else {
      competing_group_variable_candidate[pair_mat[, 1]]
    }
    party_right_new <- if (is.null(competing_group_variable_candidate)) {
      NULL
    } else {
      competing_group_variable_candidate[pair_mat[, 2]]
    }
    resp_party_new <- if (is.null(competing_group_variable_respondent)) {
      NULL
    } else {
      competing_group_variable_respondent[pair_mat[, 1]]
    }
    if (!is.null(resp_cov_new)) {
      if (is.data.frame(resp_cov_new) || is.matrix(resp_cov_new)) {
        if (nrow(resp_cov_new) == nrow(W_idx)) {
          resp_cov_new <- resp_cov_new[pair_mat[, 1], , drop = FALSE]
        }
      } else if (length(resp_cov_new) == nrow(W_idx)) {
        resp_cov_new <- resp_cov_new[pair_mat[, 1]]
      }
    }
    if (!is.null(experiment_id) && length(experiment_id) == nrow(W_idx)) {
      experiment_id <- experiment_id[pair_mat[, 1]]
    }
    if (!is.null(experiment_country) && length(experiment_country) == nrow(W_idx)) {
      experiment_country <- experiment_country[pair_mat[, 1]]
    }
    if (!is.null(experiment_year) && length(experiment_year) == nrow(W_idx)) {
      experiment_year <- experiment_year[pair_mat[, 1]]
    }
    if (!is.null(factor_order_new) && (is.matrix(factor_order_new) || is.data.frame(factor_order_new))) {
      if (nrow(factor_order_new) == nrow(W_idx)) {
        factor_order_new <- factor_order_new[pair_mat[, 1], , drop = FALSE]
      }
    }
    X_left <- strenv$jnp$array(
      cs2step_neural_to_index_matrix(X_left_raw, model_info$factor_levels)
    )$astype(strenv$jnp$int32)
    X_right <- strenv$jnp$array(
      cs2step_neural_to_index_matrix(X_right_raw, model_info$factor_levels)
    )$astype(strenv$jnp$int32)
    party_left <- strenv$jnp$array(neural_coerce_group_index_base(
      values = party_left_new,
      n_rows = n_rows,
      levels = model_info$party_levels,
      missing_label = party_missing_label
    ))$astype(strenv$jnp$int32)
    party_right <- strenv$jnp$array(neural_coerce_group_index_base(
      values = party_right_new,
      n_rows = n_rows,
      levels = model_info$party_levels,
      missing_label = party_missing_label
    ))$astype(strenv$jnp$int32)
    resp_party <- strenv$jnp$array(neural_coerce_group_index_base(
      values = resp_party_new,
      n_rows = n_rows,
      levels = model_info$resp_party_levels,
      missing_label = resp_party_missing_label
    ))$astype(strenv$jnp$int32)
    experiment_idx <- cs2step_neural_prepare_experiment_index(experiment_id, model_info, n_rows)
    experiment_idx_jnp <- if (is.null(experiment_idx)) {
      NULL
    } else {
      strenv$jnp$array(as.integer(experiment_idx))$astype(strenv$jnp$int32)
    }
    cs2step_neural_validate_place_context_request(
      experiment_country,
      model_info
    )
    place_embedding <- cs2step_neural_prepare_place_embedding(
      experiment_country,
      model_info,
      n_rows
    )
    place_embedding_jnp <- if (is.null(place_embedding)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(place_embedding))$astype(strenv$dtj)
    }
    time_embedding <- cs2step_neural_prepare_time_embedding(
      experiment_year,
      model_info,
      n_rows
    )
    time_embedding_jnp <- if (is.null(time_embedding)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(time_embedding))$astype(strenv$dtj)
    }
    resp_cov <- cs2step_neural_prepare_resp_cov(
      resp_cov_new,
      model_info,
      n_rows,
      experiment_idx = experiment_idx
    )
    resp_cov_jnp <- strenv$jnp$array(as.matrix(resp_cov$values))$astype(strenv$dtj)
    resp_cov_present_jnp <- strenv$jnp$array(as.matrix(resp_cov$present))$astype(strenv$dtj)
    resp_cov_order_jnp <- if (is.null(resp_cov$order)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(resp_cov$order))$astype(strenv$jnp$int32)
    }
    factor_order <- cs2step_neural_prepare_factor_order(factor_order_new, model_info, n_rows)
    factor_order_jnp <- if (is.null(factor_order)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(factor_order))$astype(strenv$jnp$int32)
    }

    return(list(
      pairwise = TRUE,
      X_left = X_left,
      X_right = X_right,
      party_left = party_left,
      party_right = party_right,
      resp_party = resp_party,
      resp_cov = resp_cov_jnp,
      resp_cov_present = resp_cov_present_jnp,
      resp_cov_order = resp_cov_order_jnp,
      experiment_idx = experiment_idx_jnp,
      place_embedding = place_embedding_jnp,
      time_embedding = time_embedding_jnp,
      factor_order = factor_order_jnp
    ))
  }

  n_rows <- nrow(W_idx)
  X_single <- strenv$jnp$array(
    cs2step_neural_to_index_matrix(W_idx, model_info$factor_levels)
  )$astype(strenv$jnp$int32)
  party_single <- strenv$jnp$array(neural_coerce_group_index_base(
    values = competing_group_variable_candidate,
    n_rows = n_rows,
    levels = model_info$party_levels,
    missing_label = party_missing_label
  ))$astype(strenv$jnp$int32)
  resp_party <- strenv$jnp$array(neural_coerce_group_index_base(
    values = competing_group_variable_respondent,
    n_rows = n_rows,
    levels = model_info$resp_party_levels,
    missing_label = resp_party_missing_label
  ))$astype(strenv$jnp$int32)
  experiment_idx <- cs2step_neural_prepare_experiment_index(experiment_id, model_info, n_rows)
  experiment_idx_jnp <- if (is.null(experiment_idx)) {
    NULL
  } else {
    strenv$jnp$array(as.integer(experiment_idx))$astype(strenv$jnp$int32)
  }
  cs2step_neural_validate_place_context_request(
    experiment_country,
    model_info
  )
  place_embedding <- cs2step_neural_prepare_place_embedding(
    experiment_country,
    model_info,
    n_rows
  )
  place_embedding_jnp <- if (is.null(place_embedding)) {
    NULL
  } else {
    strenv$jnp$array(as.matrix(place_embedding))$astype(strenv$dtj)
  }
  time_embedding <- cs2step_neural_prepare_time_embedding(
    experiment_year,
    model_info,
    n_rows
  )
  time_embedding_jnp <- if (is.null(time_embedding)) {
    NULL
  } else {
    strenv$jnp$array(as.matrix(time_embedding))$astype(strenv$dtj)
  }
  resp_cov <- cs2step_neural_prepare_resp_cov(
    resp_cov_new,
    model_info,
    n_rows,
    experiment_idx = experiment_idx
  )
  resp_cov_jnp <- strenv$jnp$array(as.matrix(resp_cov$values))$astype(strenv$dtj)
  resp_cov_present_jnp <- strenv$jnp$array(as.matrix(resp_cov$present))$astype(strenv$dtj)
  resp_cov_order_jnp <- if (is.null(resp_cov$order)) {
    NULL
  } else {
    strenv$jnp$array(as.matrix(resp_cov$order))$astype(strenv$jnp$int32)
  }
  factor_order <- cs2step_neural_prepare_factor_order(factor_order_new, model_info, n_rows)
  factor_order_jnp <- if (is.null(factor_order)) {
    NULL
  } else {
    strenv$jnp$array(as.matrix(factor_order))$astype(strenv$jnp$int32)
  }

  list(
    pairwise = FALSE,
    X_single = X_single,
    party_single = party_single,
    resp_party = resp_party,
    resp_cov = resp_cov_jnp,
    resp_cov_present = resp_cov_present_jnp,
    resp_cov_order = resp_cov_order_jnp,
    experiment_idx = experiment_idx_jnp,
    place_embedding = place_embedding_jnp,
    time_embedding = time_embedding_jnp,
    factor_order = factor_order_jnp
  )
}

cs2step_neural_prepare_params <- function(object,
                                          conda_env = NULL,
                                          conda_env_required = TRUE) {
  if (!"jnp" %in% ls(envir = strenv)) {
    env_use <- conda_env %||% object$metadata$conda_env %||% "strategize_env"
    req_use <- if (is.null(conda_env_required)) TRUE else conda_env_required
    initialize_jax(conda_env = env_use, conda_env_required = req_use)
  }

  model_info <- object$fit$neural_model_info
  if (is.null(model_info)) {
    stop("Neural predictor is missing model metadata.", call. = FALSE)
  }
  model_info <- cs2step_neural_upgrade_model_info(model_info)
  object$fit$neural_model_info <- model_info
  if (identical(neural_covariate_value_encoding(model_info), "shared_projection") &&
      cs2step_neural_has_resp_covariates(model_info) &&
      !isTRUE(model_info$has_covariate_fused_tokens)) {
    stop(
      "This shared_projection predictor uses the pre-fused covariate encoder. Refit under the updated architecture.",
      call. = FALSE
    )
  }
  neural_validate_full_attn_compatibility(
    model_info = model_info,
    context = "Neural predictor"
  )
  neural_validate_cross_attn_compatibility(
    model_info = model_info,
    context = "Neural predictor"
  )

  if (!is.null(object$fit$params)) {
    params <- object$fit$params
    if (is.list(params) && length(params) > 0L) {
      needs_cast <- !(cs2step_has_reticulate() && reticulate::is_py_object(params[[1]]))
      if (isTRUE(needs_cast)) {
        params <- lapply(params, function(x) strenv$jnp$array(x)$astype(strenv$dtj))
      }
    }
    neural_validate_full_attn_compatibility(
      model_info = model_info,
      params = params,
      context = "Neural predictor"
    )
    neural_validate_cross_attn_compatibility(
      model_info = model_info,
      params = params,
      context = "Neural predictor"
    )
    return(list(params = params, model_info = model_info))
  }

  cache_id <- object$metadata$cache_id %||% NULL
  cached_params <- cs2step_neural_cache_get(cache_id)
  if (!is.null(cached_params)) {
    neural_validate_full_attn_compatibility(
      model_info = model_info,
      params = cached_params,
      context = "Neural predictor"
    )
    neural_validate_cross_attn_compatibility(
      model_info = model_info,
      params = cached_params,
      context = "Neural predictor"
    )
    return(list(params = cached_params, model_info = model_info))
  }

  theta_mean <- object$fit$theta_mean
  if (is.null(theta_mean) || !length(theta_mean)) {
    if (!is.null(model_info$params)) {
      params <- model_info$params
      if (is.list(params) && length(params) > 0L) {
        needs_cast <- !(cs2step_has_reticulate() && reticulate::is_py_object(params[[1]]))
        if (isTRUE(needs_cast)) {
          params <- lapply(params, function(x) strenv$jnp$array(x)$astype(strenv$dtj))
        }
      }
      neural_validate_full_attn_compatibility(
        model_info = model_info,
        params = params,
        context = "Neural predictor"
      )
      neural_validate_cross_attn_compatibility(
        model_info = model_info,
        params = params,
        context = "Neural predictor"
      )
      return(list(params = params, model_info = model_info))
    }
    stop("Neural predictor is missing fitted parameters.", call. = FALSE)
  }
  theta_jnp <- strenv$jnp$array(as.numeric(theta_mean))$astype(strenv$dtj)
  params <- neural_params_from_theta(theta_jnp, model_info)
  neural_validate_full_attn_compatibility(
    model_info = model_info,
    params = params,
    context = "Neural predictor"
  )
  neural_validate_cross_attn_compatibility(
    model_info = model_info,
    params = params,
    context = "Neural predictor"
  )
  cs2step_neural_cache_set(cache_id, params)
  list(params = params, model_info = model_info)
}

cs2step_neural_to_r_array <- function(x) {
  cs2step_py_to_r(x, context = "neural prediction array")
}

cs2step_ordinal_thresholds_from_raw <- function(raw) {
  raw <- as.matrix(raw)
  if (!length(raw)) {
    return(raw)
  }
  out <- raw
  if (ncol(raw) > 1L) {
    inc <- log1p(exp(-abs(raw[, -1L, drop = FALSE]))) +
      pmax(raw[, -1L, drop = FALSE], 0) +
      1e-4
    out[, -1L] <- inc
  }
  t(apply(out, 1L, cumsum))
}

cs2step_ordinal_prob_matrix <- function(eta,
                                        n_outcomes_obs,
                                        experiment_index = NULL,
                                        ordinal_thresholds = NULL,
                                        ordinal_threshold_raw = NULL) {
  eta <- as.numeric(eta)
  n <- length(eta)
  k_vec <- as.integer(n_outcomes_obs)
  if (length(k_vec) == 1L && n > 1L) {
    k_vec <- rep.int(k_vec, n)
  }
  if (length(k_vec) != n) {
    fallback_k <- if (length(k_vec)) k_vec[[1L]] else 2L
    k_vec <- rep.int(max(2L, suppressWarnings(as.integer(fallback_k %||% 2L))), n)
  }
  k_vec[is.na(k_vec) | k_vec < 2L] <- 2L
  k_max <- max(2L, k_vec, na.rm = TRUE)
  probs <- matrix(0, nrow = n, ncol = k_max)
  thresholds <- ordinal_thresholds
  if (is.null(thresholds) && !is.null(ordinal_threshold_raw)) {
    thresholds <- cs2step_ordinal_thresholds_from_raw(ordinal_threshold_raw)
  }
  if (is.null(thresholds)) {
    thresholds <- matrix(seq(-1, 1, length.out = max(1L, k_max - 1L)), nrow = 1L)
  }
  thresholds <- as.matrix(thresholds)
  exp_idx <- if (is.null(experiment_index)) {
    rep(0L, n)
  } else {
    as.integer(cs2step_neural_to_r_array(experiment_index))
  }
  if (length(exp_idx) == 1L && n > 1L) {
    exp_idx <- rep.int(exp_idx, n)
  }
  if (length(exp_idx) != n) {
    exp_idx <- rep.int(0L, n)
  }
  exp_idx[is.na(exp_idx) | exp_idx < 0L] <- 0L
  exp_idx <- pmin(exp_idx + 1L, nrow(thresholds))
  for (i in seq_len(n)) {
    k_i <- min(as.integer(k_vec[[i]]), k_max)
    if (!is.finite(k_i) || k_i < 2L || !is.finite(eta[[i]])) {
      next
    }
    cut <- as.numeric(thresholds[
      exp_idx[[i]],
      seq_len(min(k_i - 1L, ncol(thresholds))),
      drop = TRUE
    ])
    if (length(cut) < k_i - 1L) {
      cut <- c(cut, tail(cut, 1L) + seq_len(k_i - 1L - length(cut)))
    }
    cut <- cummax(cut)
    cdf <- stats::plogis(cut - eta[[i]])
    p <- c(cdf[[1L]], diff(cdf), 1 - cdf[[length(cdf)]])
    p <- pmax(p, 1e-12)
    probs[i, seq_len(k_i)] <- p / sum(p)
  }
  probs
}

cs2step_neural_coerce_prediction_output <- function(pred,
                                                    likelihood,
                                                    target_likelihood = NULL,
                                                    target_n_outcomes = NULL,
                                                    target_experiment_index = NULL,
                                                    ordinal_thresholds = NULL,
                                                    ordinal_threshold_raw = NULL,
                                                    sigma = NULL,
                                                    model_info = NULL,
                                                    pairwise_prediction = FALSE) {
  if (identical(likelihood, "mixed")) {
    logits <- if (is.list(pred) && !is.null(pred$logits)) {
      pred$logits
    } else {
      pred
    }
    logits <- as.matrix(cs2step_neural_to_r_array(logits))
    if (ncol(logits) < 1L) {
      stop("Mixed-family prediction requires at least one output logit.", call. = FALSE)
    }
    target_likelihood <- tolower(as.character(target_likelihood %||% "bernoulli"))
    if (length(target_likelihood) != 1L || is.na(target_likelihood) || !nzchar(target_likelihood)) {
      target_likelihood <- "bernoulli"
    }
    target_n_outcomes <- suppressWarnings(as.integer(target_n_outcomes %||% 1L))
    if (length(target_n_outcomes) != 1L || is.na(target_n_outcomes) || target_n_outcomes < 1L) {
      target_n_outcomes <- 1L
    }
    # Train/serve parity: the jitted mixed cores skip the calibration
    # temperature for likelihood == "mixed"; training applies it to
    # class-family rows only. Re-apply here (mirrors
    # coerce_mixed_prediction_output in the training file); normal rows
    # are never scaled.
    if (identical(target_likelihood, "bernoulli")) {
      if (isTRUE(pairwise_prediction)) {
        logits[, 1L] <- neural_apply_pairwise_bernoulli_logit_adjustment_r(
          logits[, 1L],
          model_info
        )
      }
      logits[, 1L] <- neural_apply_classification_logit_calibration_r(
        logits[, 1L],
        model_info
      )
      return(stats::plogis(logits[, 1L]))
    }
    if (identical(target_likelihood, "categorical")) {
      k <- max(2L, min(as.integer(target_n_outcomes), ncol(logits)))
      z <- logits[, seq_len(k), drop = FALSE]
      z <- neural_apply_classification_logit_calibration_r(z, model_info)
      z <- sweep(z, 1L, apply(z, 1L, max), "-")
      p <- exp(z)
      return(sweep(p, 1L, rowSums(p), "/"))
    }
    if (target_likelihood %in% c("ordinal", "ordered", "ordered_logit", "ordered-logit", "ordinal_single")) {
      return(cs2step_ordinal_prob_matrix(
        eta = neural_apply_classification_logit_calibration_r(
          logits[, 1L],
          model_info
        ),
        n_outcomes_obs = rep.int(as.integer(target_n_outcomes), nrow(logits)),
        experiment_index = target_experiment_index,
        ordinal_thresholds = cs2step_neural_to_r_array(ordinal_thresholds),
        ordinal_threshold_raw = cs2step_neural_to_r_array(ordinal_threshold_raw)
      ))
    }
    if (identical(target_likelihood, "normal")) {
      sigma_source <- if (is.list(pred) && !is.null(pred$sigma)) pred$sigma else sigma
      sigma_value <- as.numeric(cs2step_neural_to_r_array(sigma_source %||% 1))
      if (!length(sigma_value)) {
        sigma_value <- 1
      }
      return(list(
        mu = as.numeric(logits[, 1L]),
        sigma = rep_len(sigma_value, nrow(logits))
      ))
    }
    stop(
      sprintf("Unsupported target likelihood for mixed-family prediction: %s", target_likelihood),
      call. = FALSE
    )
  }
  if (likelihood == "bernoulli") {
    return(as.numeric(cs2step_neural_to_r_array(pred)))
  }
  if (likelihood == "categorical") {
    return(as.matrix(cs2step_neural_to_r_array(pred)))
  }
  if (likelihood == "normal") {
    return(list(
      mu = as.numeric(cs2step_neural_to_r_array(pred$mu)),
      sigma = as.numeric(cs2step_neural_to_r_array(pred$sigma))
    ))
  }
  if (likelihood == "ordinal") {
    logits <- if (is.list(pred) && !is.null(pred$logits)) {
      pred$logits
    } else {
      pred
    }
    logits <- as.matrix(cs2step_neural_to_r_array(logits))
    return(cs2step_ordinal_prob_matrix(
      eta = logits[, 1L],
      n_outcomes_obs = rep.int(
        as.integer(target_n_outcomes %||% model_info$n_outcomes %||% model_info$nOutcomes %||% 2L),
        nrow(logits)
      ),
      experiment_index = target_experiment_index,
      ordinal_thresholds = cs2step_neural_to_r_array(ordinal_thresholds),
      ordinal_threshold_raw = cs2step_neural_to_r_array(ordinal_threshold_raw)
    ))
  }
  pred
}

cs2step_neural_has_resp_covariates <- function(model_info) {
  n_cov <- suppressWarnings(as.integer(model_info$n_resp_covariates %||% NA_integer_))
  if (length(n_cov) == 1L && !is.na(n_cov)) {
    return(n_cov > 0L)
  }
  length(model_info$covariate_names %||% character(0)) > 0L
}

cs2step_neural_predict_pair_prepared <- function(params,
                                                 model_info,
                                                 prep,
                                                 return_logits = FALSE,
                                                 target_likelihood = NULL,
                                                 target_n_outcomes = NULL) {
  if (!isTRUE(prep$pairwise)) {
    stop("Pairwise prepared prediction requires pairwise prep data.", call. = FALSE)
  }
  pred <- neural_predict_prepared_jitted(
    params = params,
    model_info = model_info,
    prep = prep,
    return_logits = isTRUE(return_logits) || identical(model_info$likelihood, "mixed")
  )
  strategize_jax_block_until_ready(pred)
  if (isTRUE(return_logits) || !identical(model_info$likelihood, "mixed")) {
    return(pred)
  }
  cs2step_neural_coerce_prediction_output(
    pred = pred,
    likelihood = model_info$likelihood,
    target_likelihood = target_likelihood,
    target_n_outcomes = target_n_outcomes,
    target_experiment_index = prep$experiment_idx %||% NULL,
    ordinal_thresholds = params$ordinal_thresholds %||% NULL,
    ordinal_threshold_raw = params$ordinal_threshold_raw %||% NULL,
    sigma = params$sigma %||% NULL,
    model_info = model_info,
    pairwise_prediction = TRUE
  )
}

cs2step_neural_predict_single_prepared <- function(params,
                                                   model_info,
                                                   prep,
                                                   return_logits = FALSE,
                                                   target_likelihood = NULL,
                                                   target_n_outcomes = NULL) {
  if (isTRUE(prep$pairwise)) {
    stop("Single prepared prediction requires single-mode prep data.", call. = FALSE)
  }
  pred <- neural_predict_prepared_jitted(
    params = params,
    model_info = model_info,
    prep = prep,
    return_logits = isTRUE(return_logits) || identical(model_info$likelihood, "mixed")
  )
  strategize_jax_block_until_ready(pred)
  if (isTRUE(return_logits) || !identical(model_info$likelihood, "mixed")) {
    return(pred)
  }
  cs2step_neural_coerce_prediction_output(
    pred = pred,
    likelihood = model_info$likelihood,
    target_likelihood = target_likelihood,
    target_n_outcomes = target_n_outcomes,
    target_experiment_index = prep$experiment_idx %||% NULL,
    ordinal_thresholds = params$ordinal_thresholds %||% NULL,
    ordinal_threshold_raw = params$ordinal_threshold_raw %||% NULL,
    sigma = params$sigma %||% NULL,
    model_info = model_info,
    pairwise_prediction = FALSE
  )
}

cs2step_neural_predict_prepared <- function(params,
                                            model_info,
                                            prep,
                                            return_logits = FALSE,
                                            target_likelihood = NULL,
                                            target_n_outcomes = NULL) {
  if (isTRUE(prep$pairwise)) {
    return(cs2step_neural_predict_pair_prepared(
      params,
      model_info,
      prep,
      return_logits = return_logits,
      target_likelihood = target_likelihood,
      target_n_outcomes = target_n_outcomes
    ))
  }
  cs2step_neural_predict_single_prepared(
    params,
    model_info,
    prep,
    return_logits = return_logits,
    target_likelihood = target_likelihood,
    target_n_outcomes = target_n_outcomes
  )
}

cs2step_neural_predict_internal <- function(object,
                                           W_new,
                                           X_new = NULL,
                                           competing_group_variable_candidate = NULL,
                                           competing_group_variable_respondent = NULL,
                                           experiment_id = NULL,
                                           experiment_description = NULL,
                                           experiment_country = NULL,
                                           experiment_year = NULL,
                                           pair_id = NULL,
                                           profile_order = NULL,
                                           type = c("response", "link"),
                                           interval = c("none", "ci", "draws"),
                                           level = 0.95,
                                           n_draws = 0L,
                                           seed = NULL,
                                           factor_schema = NULL,
                                           text_embedding_fn = NULL) {
  type <- match.arg(type)
  interval <- match.arg(interval)
  if (interval != "none" && (is.null(n_draws) || n_draws < 1L)) {
    n_draws <- 200L
  }
  n_draws <- as.integer(n_draws)

  enc <- object$encoder
  model_info <- object$fit$neural_model_info
  factor_order_new <- NULL
  prep_params <- NULL
  schema_prediction <- NULL
  if (!is.null(factor_schema)) {
    prep_params <- cs2step_neural_prepare_params(object)
    model_info <- prep_params$model_info
    schema_prediction <- cs2step_neural_prepare_factor_schema_prediction(
      object = object,
      W = W_new,
      model_info = model_info,
      params = prep_params$params,
      factor_schema = factor_schema,
      text_embedding_fn = text_embedding_fn
    )
    W_new <- schema_prediction$W
    W_idx <- schema_prediction$W_idx
    model_info <- schema_prediction$model_info
    factor_order_new <- schema_prediction$factor_order_new
  } else {
    W_new <- if (identical(neural_factor_tokenization(model_info), "fused")) {
      cs2step_neural_prepare_W_for_prediction(W_new, enc$factor_names)
    } else {
      cs2step_align_W(W_new, enc$factor_names)
    }
    W_idx <- cs2step_encode_W_indices(W_new, enc$names_list, unknown = "holdout", pad_unknown = 1L)
  }
  if (!is.null(competing_group_variable_candidate) &&
      length(competing_group_variable_candidate) != nrow(W_idx)) {
    stop(
      sprintf(
        "competing_group_variable_candidate has %d elements but W has %d rows.",
        length(competing_group_variable_candidate),
        nrow(W_idx)
      ),
      call. = FALSE
    )
  }
  if (!is.null(competing_group_variable_respondent) &&
      length(competing_group_variable_respondent) != nrow(W_idx)) {
    stop(
      sprintf(
        "competing_group_variable_respondent has %d elements but W has %d rows.",
        length(competing_group_variable_respondent),
        nrow(W_idx)
      ),
      call. = FALSE
    )
  }

  use_internal <- is.null(schema_prediction) &&
    is.function(object$fit$my_model) &&
    is.null(experiment_description) &&
    is.null(experiment_country) &&
    is.null(experiment_year)
  prep <- NULL
  prediction_mode <- if (identical(object$mode, "pairwise") ||
                         (identical(object$mode, "universal") && !is.null(pair_id))) {
    "pairwise"
  } else {
    "single"
  }
  target_likelihood <- tolower(as.character(
    object$metadata$target_likelihood %||%
      object$metadata$likelihood %||%
      if (identical(model_info$likelihood, "mixed")) "bernoulli" else model_info$likelihood
  ))
  if (length(target_likelihood) != 1L || is.na(target_likelihood) || !nzchar(target_likelihood)) {
    target_likelihood <- if (identical(model_info$likelihood, "mixed")) "bernoulli" else model_info$likelihood
  }
  target_n_outcomes <- suppressWarnings(as.integer(
    object$metadata$target_n_outcomes %||%
      object$metadata$n_outcomes %||%
      model_info$global_out_dim %||%
      1L
  ))
  if (length(target_n_outcomes) != 1L || is.na(target_n_outcomes) || target_n_outcomes < 1L) {
    target_n_outcomes <- 1L
  }

  if (identical(prediction_mode, "pairwise")) {
    cs2step_validate_pairwise_ids(pair_id, nrow(W_idx))
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id,
      W = W_idx,
      profile_order = profile_order,
      competing_group_variable_candidate = competing_group_variable_candidate
    )
    pair_mat <- pair_info$pair_mat
    X_left <- W_idx[pair_mat[, 1], , drop = FALSE]
    X_right <- W_idx[pair_mat[, 2], , drop = FALSE]
    party_left_new <- if (is.null(competing_group_variable_candidate)) {
      NULL
    } else {
      competing_group_variable_candidate[pair_mat[, 1]]
    }
    party_right_new <- if (is.null(competing_group_variable_candidate)) {
      NULL
    } else {
      competing_group_variable_candidate[pair_mat[, 2]]
    }
    resp_party_new <- if (is.null(competing_group_variable_respondent)) {
      NULL
    } else {
      competing_group_variable_respondent[pair_mat[, 1]]
    }
    if (!is.null(X_new)) {
      if (is.data.frame(X_new) || is.matrix(X_new)) {
        if (nrow(X_new) == nrow(W_idx)) {
          X_new <- X_new[pair_mat[, 1], , drop = FALSE]
        }
      } else if (length(X_new) == nrow(W_idx)) {
        X_new <- X_new[pair_mat[, 1]]
      }
    }
    if (!is.null(experiment_id) && length(experiment_id) == nrow(W_idx)) {
      experiment_id <- experiment_id[pair_mat[, 1]]
    }
    if (!is.null(experiment_description) && length(experiment_description) == nrow(W_idx)) {
      experiment_description <- experiment_description[pair_mat[, 1]]
    }
    if (!is.null(experiment_country) && length(experiment_country) == nrow(W_idx)) {
      experiment_country <- experiment_country[pair_mat[, 1]]
    }
    if (!is.null(experiment_year) && length(experiment_year) == nrow(W_idx)) {
      experiment_year <- experiment_year[pair_mat[, 1]]
    }
    experiment_idx <- cs2step_neural_prepare_experiment_index(experiment_id, model_info, nrow(X_left))
    resp_cov_prepped <- cs2step_neural_prepare_resp_cov(
      X_new,
      model_info,
      nrow(X_left),
      experiment_idx = experiment_idx
    )
    factor_order_internal <- if (isTRUE(use_internal)) {
      cs2step_neural_prepare_factor_order(
        factor_order_new = factor_order_new,
        model_info = model_info,
        n_rows = nrow(X_left)
      )
    } else {
      factor_order_new
    }
    if (isTRUE(use_internal)) {
      p <- object$fit$my_model(
        X_left_new = X_left,
        X_right_new = X_right,
        party_left_new = party_left_new,
        party_right_new = party_right_new,
        resp_party_new = resp_party_new,
        resp_cov_new = resp_cov_prepped$values,
        resp_cov_present_new = resp_cov_prepped$present,
        resp_cov_order_new = resp_cov_prepped$order,
        experiment_idx_new = experiment_idx,
        factor_order_new = factor_order_internal,
        mode = "pairwise",
        target_likelihood = target_likelihood,
        target_n_outcomes = target_n_outcomes
      )
    } else {
      if (is.null(prep_params)) {
        prep_params <- cs2step_neural_prepare_params(object)
        model_info <- prep_params$model_info
      }
      cs2step_neural_validate_place_context_request(
        experiment_country,
        model_info,
        prep_params$params
      )
      model_info <- cs2step_neural_apply_experiment_description(
        model_info = model_info,
        experiment_description = experiment_description,
        n_rows = nrow(X_left),
        text_embedding_fn = text_embedding_fn %||% object$metadata$text_embedding_fn %||% NULL
      )
      prep <- cs2step_neural_prepare_prediction_data(
        W_idx = W_idx,
        model_info = model_info,
        competing_group_variable_candidate = competing_group_variable_candidate,
        competing_group_variable_respondent = competing_group_variable_respondent,
        resp_cov_new = X_new,
        factor_order_new = factor_order_new,
        experiment_id = experiment_id,
        experiment_country = experiment_country,
        experiment_year = experiment_year,
        pair_id = pair_id,
        profile_order = profile_order,
        mode = "pairwise"
      )
      p <- cs2step_neural_predict_prepared(
        params = prep_params$params,
        model_info = model_info,
        prep = prep,
        return_logits = identical(type, "link"),
        target_likelihood = target_likelihood,
        target_n_outcomes = target_n_outcomes
      )
    }
  } else {
    experiment_idx <- cs2step_neural_prepare_experiment_index(experiment_id, model_info, nrow(W_idx))
    resp_cov_prepped <- cs2step_neural_prepare_resp_cov(
      X_new,
      model_info,
      nrow(W_idx),
      experiment_idx = experiment_idx
    )
    factor_order_internal <- if (isTRUE(use_internal)) {
      cs2step_neural_prepare_factor_order(
        factor_order_new = factor_order_new,
        model_info = model_info,
        n_rows = nrow(W_idx)
      )
    } else {
      factor_order_new
    }
    if (isTRUE(use_internal)) {
      p <- object$fit$my_model(
        X_new = W_idx,
        party_new = competing_group_variable_candidate,
        resp_party_new = competing_group_variable_respondent,
        resp_cov_new = resp_cov_prepped$values,
        resp_cov_present_new = resp_cov_prepped$present,
        resp_cov_order_new = resp_cov_prepped$order,
        experiment_idx_new = experiment_idx,
        factor_order_new = factor_order_internal,
        mode = "single",
        target_likelihood = target_likelihood,
        target_n_outcomes = target_n_outcomes
      )
    } else {
      if (is.null(prep_params)) {
        prep_params <- cs2step_neural_prepare_params(object)
        model_info <- prep_params$model_info
      }
      cs2step_neural_validate_place_context_request(
        experiment_country,
        model_info,
        prep_params$params
      )
      model_info <- cs2step_neural_apply_experiment_description(
        model_info = model_info,
        experiment_description = experiment_description,
        n_rows = nrow(W_idx),
        text_embedding_fn = text_embedding_fn %||% object$metadata$text_embedding_fn %||% NULL
      )
      prep <- cs2step_neural_prepare_prediction_data(
        W_idx = W_idx,
        model_info = model_info,
        competing_group_variable_candidate = competing_group_variable_candidate,
        competing_group_variable_respondent = competing_group_variable_respondent,
        resp_cov_new = X_new,
        factor_order_new = factor_order_new,
        experiment_id = experiment_id,
        experiment_country = experiment_country,
        experiment_year = experiment_year,
        mode = "single"
      )
      p <- cs2step_neural_predict_prepared(
        params = prep_params$params,
        model_info = model_info,
        prep = prep,
        return_logits = identical(type, "link"),
        target_likelihood = target_likelihood,
        target_n_outcomes = target_n_outcomes
      )
    }
  }

  if (isTRUE(use_internal)) {
    if (type == "link") {
      if (is.numeric(p) && is.null(dim(p))) {
        eps <- .Machine$double.eps
        p <- pmin(pmax(p, eps), 1 - eps)
        pred <- stats::qlogis(p)
      } else {
        pred <- p
      }
    } else {
      pred <- if (is.numeric(p) && is.null(dim(p))) as.numeric(p) else p
    }
  } else if (identical(type, "link")) {
    link_mat <- cs2step_neural_to_r_array(p)
    link_mat <- if (is.null(dim(link_mat))) {
      matrix(as.numeric(link_mat), ncol = 1L)
    } else {
      as.matrix(link_mat)
    }
    # For multi-logit (mixed/categorical-pooled) models, preserve the n x k
    # logit matrix -- as.numeric() silently returned a column-major flattened
    # vector of length n*k. Single-logit models keep the numeric-vector
    # contract. Raw logits intentionally exclude the serve-time pairwise/
    # calibration scale adjustments (documented in ?predict): plogis(link)
    # equals type = "response" only when those adjustments are disabled.
    pred <- if (ncol(link_mat) == 1L) as.numeric(link_mat[, 1L]) else link_mat
    pred <- p
  } else {
    pred <- cs2step_neural_coerce_prediction_output(
      pred = p,
      likelihood = model_info$likelihood,
      target_likelihood = target_likelihood,
      target_n_outcomes = target_n_outcomes,
      target_experiment_index = if (!is.null(prep)) prep$experiment_idx %||% NULL else NULL,
      ordinal_thresholds = if (exists("prep_params", inherits = FALSE)) {
        prep_params$params$ordinal_thresholds %||% NULL
      } else {
        NULL
      },
      ordinal_threshold_raw = if (exists("prep_params", inherits = FALSE)) {
        prep_params$params$ordinal_threshold_raw %||% NULL
      } else {
        NULL
      },
      sigma = if (exists("prep_params", inherits = FALSE)) prep_params$params$sigma %||% NULL else NULL,
      model_info = model_info
    )
  }

  if (interval == "none") {
    return(pred)
  }

  theta_mean <- object$fit$theta_mean
  theta_var <- object$fit$theta_var
  if (length(theta_mean) != length(theta_var) || length(theta_mean) == 0L) {
    stop("Neural predictor does not contain parameter uncertainty information for draws.",
         call. = FALSE)
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }
  theta_sd <- sqrt(pmax(theta_var, 0))
  theta_draws <- matrix(stats::rnorm(n_draws * length(theta_mean)), nrow = n_draws)
  theta_draws <- sweep(theta_draws, 2, theta_sd, `*`)
  theta_draws <- sweep(theta_draws, 2, theta_mean, `+`)

  if (is.null(model_info)) {
    model_info <- object$fit$neural_model_info
  }
  pred_length <- if (is.list(pred) && !is.null(pred$mu)) length(pred$mu) else length(pred)
  if (is.null(prep)) {
    prep <- cs2step_neural_prepare_prediction_data(
      W_idx = W_idx,
      model_info = model_info,
      competing_group_variable_candidate = competing_group_variable_candidate,
      competing_group_variable_respondent = competing_group_variable_respondent,
      resp_cov_new = X_new,
      factor_order_new = factor_order_new,
      experiment_id = experiment_id,
      experiment_country = experiment_country,
      experiment_year = experiment_year,
      pair_id = pair_id,
      profile_order = profile_order,
      mode = prediction_mode
    )
  }

  theta_draws_jnp <- strenv$jnp$array(theta_draws)$astype(strenv$dtj)
  batched_pred <- neural_predict_from_theta_prepared_jitted(
    theta_batch = theta_draws_jnp,
    model_info = model_info,
    prep = prep,
    return_logits = identical(type, "link")
  )
  strategize_jax_block_until_ready(batched_pred)
  batched_pred_r <- cs2step_neural_to_r_array(batched_pred)
  if (is.list(batched_pred_r) && !is.null(batched_pred_r$mu)) {
    batched_pred_r <- batched_pred_r$mu
  }
  draw_array <- as.array(batched_pred_r)
  draw_dims <- dim(draw_array)
  if (is.null(draw_dims) || length(draw_dims) <= 1L) {
    draw_pred <- matrix(as.numeric(draw_array), nrow = pred_length, ncol = n_draws)
  } else {
    perm <- c(seq.int(2L, length(draw_dims)), 1L)
    draw_pred <- matrix(
      as.numeric(aperm(draw_array, perm)),
      ncol = draw_dims[[1L]]
    )
  }

  if (type == "response") {
    draw_pred <- pmin(pmax(draw_pred, 0), 1)
  }

  alpha <- (1 - level) / 2
  qs <- c(alpha, 1 - alpha)
  q_mat <- matrixStats::rowQuantiles(draw_pred, probs = qs, drop = FALSE)
  out_df <- data.frame(
    fit = pred,
    lo = q_mat[, 1],
    hi = q_mat[, 2]
  )
  if (interval == "ci") {
    return(out_df)
  }
  list(
    fit = pred,
    interval = out_df,
    draws = draw_pred,
    level = level
  )
}

#' Predict from a fitted strategic predictor
#'
#' @param object A fitted \code{strategic_predictor}.
#' @param newdata New data. For \code{mode="single"}, a data.frame/matrix of factor columns.
#'   For \code{mode="pairwise"}, either:
#'   \itemize{
#'     \item a data.frame containing factor columns plus \code{pair_id} (and optionally \code{profile_order}), or
#'     \item a list with elements \code{W}, \code{pair_id}, and optional \code{profile_order}.
#'   }
#'   Neural foundation predictors also accept optional \code{experiment_country}
#'   and \code{experiment_year} as scalar or row-aligned vectors; in pairwise
#'   mode row-aligned values are taken from the left/profile-level prediction
#'   row.
#' @param type \code{"response"} (probability) or \code{"link"} (logit / linear predictor).
#' @param interval \code{"none"} (default), \code{"ci"}, or \code{"draws"}.
#' @param level Credible interval level for draws.
#' @param n_draws Number of posterior draws when \code{interval!="none"}.
#' @param seed Optional seed for draws.
#' @param factor_schema Optional explicit prediction-time factor schema for fused
#'   neural predictors. Supply a list with \code{names_list} or \code{p_list},
#'   and optionally precomputed factor/level text or structural matrices.
#' @param text_embedding_fn Optional text embedding function used with
#'   \code{factor_schema} and prediction-time \code{experiment_description}.
#' @param ... Unused.
#' @export
#' @method predict strategic_predictor
predict.strategic_predictor <- function(object,
                                        newdata,
                                        type = c("response", "link"),
                                        interval = c("none", "ci", "draws"),
                                        level = 0.95,
                                        n_draws = 0L,
                                        seed = NULL,
                                        factor_schema = NULL,
                                        text_embedding_fn = NULL,
                                        ...) {
  type <- match.arg(type)
  interval <- match.arg(interval)
  if (!inherits(object, "strategic_predictor")) {
    stop("predict.strategic_predictor requires a strategic_predictor object.", call. = FALSE)
  }

  unpack_schema <- cs2step_neural_factor_schema_from_inputs(factor_schema)
  unpacked <- cs2step_unpack_newdata(
    newdata,
    object$encoder$factor_names,
    object$mode,
    factor_schema = unpack_schema
  )
  factor_schema_use <- cs2step_neural_merge_factor_schema(
    newdata_schema = unpacked$factor_schema,
    newdata_names_list = unpacked$names_list,
    newdata_p_list = unpacked$p_list,
    explicit_schema = factor_schema
  )
  W_new <- unpacked$W
  X_new <- unpacked$X
  pair_id <- unpacked$pair_id
  profile_order <- unpacked$profile_order
  competing_group_variable_candidate <- unpacked$competing_group_variable_candidate
  competing_group_variable_respondent <- unpacked$competing_group_variable_respondent
  experiment_id <- unpacked$experiment_id
  experiment_description <- unpacked$experiment_description
  experiment_country <- unpacked$experiment_country
  experiment_year <- unpacked$experiment_year

  if (identical(object$model_type, "glm")) {
    if (!is.null(factor_schema_use)) {
      stop("Prediction-time factor_schema is only supported for neural predictors.", call. = FALSE)
    }
    if (!is.null(experiment_country)) {
      stop(
        "Prediction-time experiment_country requires a neural predictor trained with place context.",
        call. = FALSE
      )
    }
    return(cs2step_glm_predict_internal(
      object = object,
      W_new = W_new,
      pair_id = pair_id,
      profile_order = profile_order,
      type = type,
      interval = interval,
      level = level,
      n_draws = n_draws,
      seed = seed
    ))
  }
  cs2step_neural_predict_internal(
    object = object,
    W_new = W_new,
    X_new = X_new,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_variable_respondent = competing_group_variable_respondent,
    experiment_id = experiment_id,
    experiment_description = experiment_description,
    experiment_country = experiment_country,
    experiment_year = experiment_year,
    pair_id = pair_id,
    profile_order = profile_order,
    type = type,
    interval = interval,
    level = level,
    n_draws = n_draws,
    seed = seed,
    factor_schema = factor_schema_use,
    text_embedding_fn = text_embedding_fn
  )
}

cs2step_predict_pair_stage_aware <- function(fit) {
  model_info <- fit$fit$neural_model_info %||% NULL
  if (is.null(model_info) || !is.list(model_info)) {
    return(FALSE)
  }
  context_mode <- model_info$pairwise_context_mode %||% NULL
  if (!is.null(context_mode) &&
      any(as.character(context_mode) == "stage_aware", na.rm = TRUE)) {
    return(TRUE)
  }
  isTRUE(model_info$has_stage_context) ||
    isTRUE(model_info$has_matchup_context) ||
    isTRUE(model_info$has_stage_token) ||
    isTRUE(model_info$has_matchup_token)
}

#' Predict on wide-format pairwise data
#'
#' @param fit A fitted stage-free \code{strategic_predictor} with
#'   \code{mode="pairwise"}. Stage-aware pairwise predictors require
#'   long-format \code{\link[stats]{predict}()} input because candidate and
#'   respondent group metadata are row-aligned.
#' @param W_left Data frame/matrix of left profiles.
#' @param W_right Data frame/matrix of right profiles.
#' @param type \code{"response"} or \code{"link"}.
#' @param interval \code{"none"}, \code{"ci"}, or \code{"draws"}.
#' @param level Credible interval level for draws.
#' @param n_draws Number of posterior draws when \code{interval!="none"}.
#' @param seed Optional seed for draws.
#' @param ... Unused.
#' @return Predictions for each row-pair.
#' @export
predict_pair <- function(fit,
                         W_left,
                         W_right,
                         type = c("response", "link"),
                         interval = c("none", "ci", "draws"),
                         level = 0.95,
                         n_draws = 0L,
                         seed = NULL,
                         ...) {
  if (!inherits(fit, "strategic_predictor")) {
    stop("'fit' must be a strategic_predictor.", call. = FALSE)
  }
  if (!identical(fit$mode, "pairwise")) {
    stop("predict_pair() requires a pairwise strategic_predictor (mode='pairwise').", call. = FALSE)
  }
  if (isTRUE(cs2step_predict_pair_stage_aware(fit))) {
    stop(
      paste(
        "predict_pair() does not support stage-aware pairwise predictors.",
        "Use predict() with long-format newdata containing W, pair_id,",
        "profile_order, competing_group_variable_candidate, and",
        "competing_group_variable_respondent."
      ),
      call. = FALSE
    )
  }
  W_left <- as.data.frame(W_left)
  W_right <- as.data.frame(W_right)
  if (nrow(W_left) != nrow(W_right)) {
    stop("W_left and W_right must have the same number of rows.", call. = FALSE)
  }
  n_pairs <- nrow(W_left)
  W_long <- rbind(W_left, W_right)
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))
  predict(
    fit,
    newdata = list(W = W_long, pair_id = pair_id, profile_order = profile_order),
    type = type,
    interval = interval,
    level = level,
    n_draws = n_draws,
    seed = seed,
    ...
  )
}

#' Convert a fitted predictor to a scoring function
#'
#' @param fit A fitted \code{strategic_predictor}.
#' @return A closure that calls \code{predict()} (or \code{predict_pair()} for pairwise).
#' @export
as_function <- function(fit) {
  if (!inherits(fit, "strategic_predictor")) {
    stop("'fit' must be a strategic_predictor.", call. = FALSE)
  }
  if (identical(fit$mode, "pairwise")) {
    function(W_left, W_right, ...) {
      predict_pair(fit, W_left = W_left, W_right = W_right, ...)
    }
  } else {
    function(W, ...) {
      predict(fit, newdata = W, ...)
    }
  }
}

cs2step_neural_pack_model_info <- function(model_info, drop_params = TRUE) {
  if (is.null(model_info)) {
    return(NULL)
  }
  out <- model_info
  out$jit_cache_key <- NULL

  if (is.null(out$pairwise_bernoulli_logit_scale) &&
      !is.null(out$params$log_pairwise_bernoulli_logit_scale)) {
    out$pairwise_bernoulli_logit_scale <- exp(
      as.numeric(cs2step_neural_to_r_array(out$params$log_pairwise_bernoulli_logit_scale))[[1L]]
    )
  }
  if (is.null(out$calibration_scale) &&
      !is.null(out$params$log_calibration_scale)) {
    out$calibration_scale <- exp(
      as.numeric(cs2step_neural_to_r_array(out$params$log_calibration_scale))[[1L]]
    )
  }
  has_additive_param <- !is.null((out$params %||% list())$W_add_out)

  if (!is.null(out$params)) {
    out$params <- if (isTRUE(drop_params)) {
      NULL
    } else {
      lapply(out$params, cs2step_neural_to_r_array)
    }
  }
  if (!is.null(out$factor_index_list)) {
    out$factor_index_list <- lapply(out$factor_index_list, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }
  if (!is.null(out$cand_party_to_resp_idx)) {
    out$cand_party_to_resp_idx <- as.integer(cs2step_neural_to_r_array(out$cand_party_to_resp_idx))
  }
  if (!is.null(out$resp_cov_mean)) {
    out$resp_cov_mean <- as.numeric(cs2step_neural_to_r_array(out$resp_cov_mean))
  }
  if (!is.null(out$resp_cov_scale)) {
    out$resp_cov_scale <- as.numeric(cs2step_neural_to_r_array(out$resp_cov_scale))
  }
  if (!is.null(out$resp_cov_default_present)) {
    out$resp_cov_default_present <- as.numeric(cs2step_neural_to_r_array(out$resp_cov_default_present))
  }
  if (!is.null(out$shared_projection_value_encoder)) {
    out$shared_projection_value_encoder <- as.character(out$shared_projection_value_encoder)
  }
  if (!is.null(out$factor_name_text)) {
    out$factor_name_text <- as.matrix(cs2step_neural_to_r_array(out$factor_name_text))
  }
  if (!is.null(out$level_name_text)) {
    out$level_name_text <- lapply(out$level_name_text, function(x) {
      as.matrix(cs2step_neural_to_r_array(x))
    })
  }
  if (!is.null(out$factor_struct_matrix)) {
    out$factor_struct_matrix <- as.matrix(
      cs2step_neural_to_r_array(out$factor_struct_matrix)
    )
  }
  if (!is.null(out$level_struct_matrices)) {
    out$level_struct_matrices <- lapply(out$level_struct_matrices, function(x) {
      as.matrix(cs2step_neural_to_r_array(x))
    })
  }
  if (!is.null(out$factor_struct_feature_names)) {
    out$factor_struct_feature_names <- as.character(out$factor_struct_feature_names)
  }
  if (!is.null(out$level_struct_feature_names)) {
    out$level_struct_feature_names <- as.character(out$level_struct_feature_names)
  }
  if (!is.null(out$covariate_name_text)) {
    out$covariate_name_text <- as.matrix(cs2step_neural_to_r_array(out$covariate_name_text))
  }
  if (!is.null(out$factor_order_by_experiment)) {
    out$factor_order_by_experiment <- lapply(
      out$factor_order_by_experiment,
      function(x) as.integer(cs2step_neural_to_r_array(x))
    )
  }
  if (!is.null(out$default_factor_order)) {
    out$default_factor_order <- as.integer(cs2step_neural_to_r_array(out$default_factor_order))
  }
  if (!is.null(out$covariate_order_by_experiment)) {
    out$covariate_order_by_experiment <- lapply(
      out$covariate_order_by_experiment,
      function(x) as.integer(cs2step_neural_to_r_array(x))
    )
  }
  if (!is.null(out$default_covariate_order)) {
    out$default_covariate_order <- as.integer(cs2step_neural_to_r_array(out$default_covariate_order))
  }
  if (!is.null(out$covariate_value_stats_by_experiment)) {
    out$covariate_value_stats_by_experiment <- lapply(
      out$covariate_value_stats_by_experiment,
      function(x) if (is.null(x)) NULL else as.matrix(cs2step_neural_to_r_array(x))
    )
  }
  if (!is.null(out$default_covariate_value_stats)) {
    out$default_covariate_value_stats <- as.matrix(
      cs2step_neural_to_r_array(out$default_covariate_value_stats)
    )
  }
  if (!is.null(out$covariate_value_metadata_by_experiment)) {
    out$covariate_value_metadata_by_experiment <- lapply(
      out$covariate_value_metadata_by_experiment,
      function(x) if (is.null(x)) NULL else as.matrix(cs2step_neural_to_r_array(x))
    )
  }
  if (!is.null(out$default_covariate_value_metadata)) {
    out$default_covariate_value_metadata <- as.matrix(
      cs2step_neural_to_r_array(out$default_covariate_value_metadata)
    )
  }
  if (!is.null(out$covariate_value_text)) {
    out$covariate_value_text <- cs2step_neural_to_r_array(out$covariate_value_text)
  }
  if (!is.null(out$covariate_value_text_present)) {
    out$covariate_value_text_present <- as.matrix(
      cs2step_neural_to_r_array(out$covariate_value_text_present)
    )
  }
  if (!is.null(out$covariate_value_type)) {
    out$covariate_value_type <- as.integer(cs2step_neural_to_r_array(out$covariate_value_type))
  }
  if (!is.null(out$experiment_description_text)) {
    out$experiment_description_text <- as.matrix(
      cs2step_neural_to_r_array(out$experiment_description_text)
    )
  }
  if (!is.null(out$default_experiment_text)) {
    out$default_experiment_text <- as.matrix(
      cs2step_neural_to_r_array(out$default_experiment_text)
    )
  }
  if (!is.null(out$experiment_description_present)) {
    out$experiment_description_present <- as.logical(
      cs2step_neural_to_r_array(out$experiment_description_present)
    )
  }
  if (!is.null(out$place_embedding)) {
    out$place_embedding <- as.matrix(
      cs2step_neural_to_r_array(out$place_embedding)
    )
  }
  if (!is.null(out$default_place_embedding)) {
    out$default_place_embedding <- as.matrix(
      cs2step_neural_to_r_array(out$default_place_embedding)
    )
  }
  if (!is.null(out$place_present)) {
    out$place_present <- as.logical(
      cs2step_neural_to_r_array(out$place_present)
    )
  }
  if (!is.null(out$place_feature_names)) {
    out$place_feature_names <- as.character(out$place_feature_names)
  }
  if (!is.null(out$place_context_enabled)) {
    out$place_context_enabled <- isTRUE(out$place_context_enabled)
  }
  if (!is.null(out$default_place_present)) {
    out$default_place_present <- isTRUE(out$default_place_present)
  }
  if (!is.null(out$time_embedding)) {
    out$time_embedding <- as.matrix(
      cs2step_neural_to_r_array(out$time_embedding)
    )
  }
  if (!is.null(out$default_time_embedding)) {
    out$default_time_embedding <- as.matrix(
      cs2step_neural_to_r_array(out$default_time_embedding)
    )
  }
  if (!is.null(out$time_present)) {
    out$time_present <- as.logical(
      cs2step_neural_to_r_array(out$time_present)
    )
  }
  if (!is.null(out$time_feature_names)) {
    out$time_feature_names <- as.character(out$time_feature_names)
  }
  if (!is.null(out$time_context_enabled)) {
    out$time_context_enabled <- isTRUE(out$time_context_enabled)
  }
  if (!is.null(out$default_time_present)) {
    out$default_time_present <- isTRUE(out$default_time_present)
  }
  if (!is.null(out$default_experiment_text_present)) {
    out$default_experiment_text_present <- isTRUE(out$default_experiment_text_present)
  }
  if (!is.null(out$covariate_names)) {
    out$covariate_names <- as.character(out$covariate_names)
  }
  if (!is.null(out$experiment_levels)) {
    out$experiment_levels <- as.character(out$experiment_levels)
  }
  if (!is.null(out$token_family_levels)) {
    out$token_family_levels <- as.character(out$token_family_levels)
  }
  if (!is.null(out$readout_embedding_families)) {
    out$readout_embedding_families <- as.character(out$readout_embedding_families)
  }
  if (!is.null(out$experiment_token_mode)) {
    out$experiment_token_mode <- as.character(out$experiment_token_mode)
  }
  if (!is.null(out$factor_tokenization)) {
    out$factor_tokenization <- as.character(out$factor_tokenization)
  }
  if (!is.null(out$covariate_value_encoding)) {
    out$covariate_value_encoding <- as.character(out$covariate_value_encoding)
  }
  if (!is.null(out$schema_dropout)) {
    out$schema_dropout <- neural_resolve_schema_dropout(out$schema_dropout)
  }
  if (!is.null(out$low_rank_interaction_rank)) {
    out$low_rank_interaction_rank <- neural_resolve_low_rank_interaction_rank(
      out$low_rank_interaction_rank
    )
  }
  if (!is.null(out$low_rank_logit_transform)) {
    out$low_rank_logit_transform <- neural_normalize_low_rank_logit_transform(
      out$low_rank_logit_transform
    )
  }
  if (!is.null(out$low_rank_logit_bound)) {
    out$low_rank_logit_bound <- as.numeric(out$low_rank_logit_bound)
  }
  if (!is.null(out$low_rank_logit_softness)) {
    out$low_rank_logit_softness <- as.numeric(out$low_rank_logit_softness)
  }
  if (!is.null(out$low_rank_logit_normalization)) {
    out$low_rank_logit_normalization <- neural_normalize_low_rank_logit_normalization(
      out$low_rank_logit_normalization
    )
  }
  if (!is.null(out$low_rank_head_weight_target_rms)) {
    out$low_rank_head_weight_target_rms <- as.numeric(out$low_rank_head_weight_target_rms)
  }
  if (!is.null(out$low_rank_rc_out_target_rms)) {
    out$low_rank_rc_out_target_rms <- as.numeric(out$low_rank_rc_out_target_rms)
  }
  if (is.null(out$learned_pairwise_bernoulli_logit_scale)) {
    out$learned_pairwise_bernoulli_logit_scale <- FALSE
  } else {
    out$learned_pairwise_bernoulli_logit_scale <-
      neural_resolve_learned_pairwise_bernoulli_logit_scale(
        out$learned_pairwise_bernoulli_logit_scale
      )
  }
  if (isTRUE(out$learned_pairwise_bernoulli_logit_scale)) {
    out$pairwise_bernoulli_logit_scale_prior_sd <-
      neural_resolve_pairwise_bernoulli_logit_scale_prior_sd(
        out$pairwise_bernoulli_logit_scale_prior_sd %||% NULL,
        enabled = TRUE
      )
    scale <- suppressWarnings(as.numeric(out$pairwise_bernoulli_logit_scale %||% NA_real_))
    if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
      scale <- 1.0
    }
    out$pairwise_bernoulli_logit_scale <- as.numeric(scale)
  } else {
    out$pairwise_bernoulli_logit_scale_prior_sd <- NULL
    out$pairwise_bernoulli_logit_scale <- 1.0
  }
  out$pairwise_antisymmetry <- neural_resolve_pairwise_antisymmetry(
    out$pairwise_antisymmetry %||% "legacy"
  )
  out$additive_utility_mode <- neural_resolve_additive_utility_mode(
    out$additive_utility_mode %||% "off",
    factor_tokenization = out$factor_tokenization %||% NULL
  )
  out <- neural_coerce_additive_utility_metadata(
    out,
    legacy_missing_normalization = "none"
  )
  out$has_additive_utility <- if (is.null(out$has_additive_utility)) {
    isTRUE(has_additive_param)
  } else {
    isTRUE(out$has_additive_utility)
  }
  out$calibration_enabled <- isTRUE(out$calibration_enabled)
  out$calibration_method <- if (isTRUE(out$calibration_enabled)) {
    neural_resolve_calibration_method(out$calibration_method %||% "logit_scale")
  } else {
    "none"
  }
  scale <- suppressWarnings(as.numeric(out$calibration_scale %||% 1.0))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    scale <- 1.0
  }
  out$calibration_scale <- as.numeric(scale)
  if (!is.null(out$attention_backend)) {
    out$attention_backend <- neural_normalize_attention_backend(out$attention_backend)
  }
  if (!is.null(out$attention_dtype)) {
    out$attention_dtype <- neural_normalize_attention_dtype(out$attention_dtype)
  }
  if (!is.null(out$attention_resolved_backend)) {
    out$attention_resolved_backend <- as.character(out$attention_resolved_backend)
  }
  if (!is.null(out$attention_fallback_reason)) {
    out$attention_fallback_reason <- as.character(out$attention_fallback_reason)
  }
  if (!is.null(out$factor_levels)) {
    out$factor_levels <- as.integer(out$factor_levels)
  }
  if (!is.null(out$param_shapes)) {
    out$param_shapes <- lapply(out$param_shapes, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }
  if (!is.null(out$param_sizes)) {
    out$param_sizes <- lapply(out$param_sizes, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }
  if (!is.null(out$param_offsets)) {
    out$param_offsets <- lapply(out$param_offsets, function(x) as.integer(cs2step_neural_to_r_array(x)))
  }

  int_fields <- c("n_params", "n_factors", "n_candidate_tokens", "n_party_levels",
                  "n_matchup_levels", "n_resp_party_levels", "party_missing_index",
                  "resp_party_missing_index", "n_resp_covariates", "model_dims", "model_depth",
                  "n_heads", "head_dim", "choice_token_index", "n_experiment_levels",
                  "default_experiment_index", "text_semantic_dim", "factor_struct_dim",
                  "level_struct_dim", "place_context_dim", "time_context_dim", "max_factor_tokens",
                  "max_covariate_tokens", "attention_padding_multiple", "low_rank_interaction_rank")
  for (field in int_fields) {
    if (!is.null(out[[field]])) {
      out[[field]] <- as.integer(out[[field]])
    }
  }

  out$low_rank_interaction_rank <- neural_resolve_low_rank_interaction_rank(
    out$low_rank_interaction_rank %||% 0L
  )
  if (is.null(out$low_rank_logit_transform)) {
    out$low_rank_logit_transform <- "none"
  } else {
    out$low_rank_logit_transform <- neural_normalize_low_rank_logit_transform(
      out$low_rank_logit_transform
    )
    if (is.na(out$low_rank_logit_transform)) {
      out$low_rank_logit_transform <- "none"
    }
  }
  if (out$low_rank_interaction_rank <= 0L ||
      !identical(out$low_rank_logit_transform, "softclip") ||
      is.null(out$low_rank_logit_bound) ||
      is.null(out$low_rank_logit_softness) ||
      !is.finite(as.numeric(out$low_rank_logit_bound)) ||
      !is.finite(as.numeric(out$low_rank_logit_softness)) ||
      as.numeric(out$low_rank_logit_bound) <= 0 ||
      as.numeric(out$low_rank_logit_softness) <= 0) {
    out$low_rank_logit_transform <- "none"
    out$low_rank_logit_bound <- NULL
    out$low_rank_logit_softness <- NULL
  }
  if (is.null(out$low_rank_logit_normalization)) {
    out$low_rank_logit_normalization <- "none"
  } else {
    out$low_rank_logit_normalization <- neural_normalize_low_rank_logit_normalization(
      out$low_rank_logit_normalization
    )
    if (is.na(out$low_rank_logit_normalization)) {
      out$low_rank_logit_normalization <- "none"
    }
  }
  if (out$low_rank_interaction_rank <= 0L ||
      !identical(out$low_rank_logit_normalization, "rms") ||
      is.null(out$low_rank_head_weight_target_rms) ||
      is.null(out$low_rank_rc_out_target_rms) ||
      !is.finite(as.numeric(out$low_rank_head_weight_target_rms)) ||
      !is.finite(as.numeric(out$low_rank_rc_out_target_rms)) ||
      as.numeric(out$low_rank_head_weight_target_rms) <= 0 ||
      as.numeric(out$low_rank_rc_out_target_rms) <= 0) {
    out$low_rank_logit_normalization <- "none"
    out$low_rank_head_weight_target_rms <- NULL
    out$low_rank_rc_out_target_rms <- NULL
  } else {
    out$low_rank_head_weight_target_rms <- as.numeric(out$low_rank_head_weight_target_rms)
    out$low_rank_rc_out_target_rms <- as.numeric(out$low_rank_rc_out_target_rms)
  }
  if (is.null(out$learned_pairwise_bernoulli_logit_scale)) {
    out$learned_pairwise_bernoulli_logit_scale <- FALSE
  } else {
    out$learned_pairwise_bernoulli_logit_scale <-
      neural_resolve_learned_pairwise_bernoulli_logit_scale(
        out$learned_pairwise_bernoulli_logit_scale
      )
  }
  if (isTRUE(out$learned_pairwise_bernoulli_logit_scale)) {
    out$pairwise_bernoulli_logit_scale_prior_sd <-
      neural_resolve_pairwise_bernoulli_logit_scale_prior_sd(
        out$pairwise_bernoulli_logit_scale_prior_sd %||% NULL,
        enabled = TRUE
      )
    scale <- suppressWarnings(as.numeric(out$pairwise_bernoulli_logit_scale %||% NA_real_))
    if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
      scale <- 1.0
    }
    out$pairwise_bernoulli_logit_scale <- as.numeric(scale)
  } else {
    out$pairwise_bernoulli_logit_scale_prior_sd <- NULL
    out$pairwise_bernoulli_logit_scale <- 1.0
  }
  if (!is.null(out$token_family_levels) && out$low_rank_interaction_rank <= 0L) {
    out$token_family_levels <- setdiff(
      as.character(out$token_family_levels),
      c("respondent_cls", "candidate_cls")
    )
  }
  if (out$low_rank_interaction_rank <= 0L) {
    out$has_respondent_cls <- FALSE
    out$has_candidate_cls <- FALSE
    out$has_low_rank_interaction <- FALSE
    out$readout_embedding_families <- neural_readout_embedding_families(
      low_rank_interaction_rank = 0L
    )
  } else {
    if (is.null(out$has_respondent_cls)) {
      out$has_respondent_cls <- !is.null((out$params %||% list())$E_respondent_cls)
    }
    if (is.null(out$has_candidate_cls)) {
      out$has_candidate_cls <- !is.null((out$params %||% list())$E_candidate_cls)
    }
    if (is.null(out$has_low_rank_interaction)) {
      out$has_low_rank_interaction <- isTRUE(neural_has_low_rank_interaction(
        out$params %||% list(),
        out
      ))
    }
    if (is.null(out$readout_embedding_families)) {
      out$readout_embedding_families <- neural_readout_embedding_families(
        low_rank_interaction_rank = out$low_rank_interaction_rank
      )
    }
  }

  out
}

cs2step_neural_upgrade_model_info <- function(model_info) {
  if (is.null(model_info)) {
    return(NULL)
  }
  out <- model_info
  out$jit_cache_key <- NULL
  if (is.null(out$pairwise_bernoulli_logit_scale) &&
      !is.null(out$params$log_pairwise_bernoulli_logit_scale)) {
    out$pairwise_bernoulli_logit_scale <- exp(
      as.numeric(cs2step_neural_to_r_array(out$params$log_pairwise_bernoulli_logit_scale))[[1L]]
    )
  }
  if (is.null(out$calibration_scale) &&
      !is.null(out$params$log_calibration_scale)) {
    out$calibration_scale <- exp(
      as.numeric(cs2step_neural_to_r_array(out$params$log_calibration_scale))[[1L]]
    )
    if (is.null(out$calibration_enabled)) {
      out$calibration_enabled <- TRUE
    }
  }
  if (is.null(out$pairwise_context_mode)) {
    out$pairwise_context_mode <- "stage_free"
  }
  if (is.null(out$has_candidate_group_context)) {
    out$has_candidate_group_context <- FALSE
  }
  if (is.null(out$has_respondent_group_context)) {
    out$has_respondent_group_context <- FALSE
  }
  if (is.null(out$has_relation_token_context)) {
    out$has_relation_token_context <- FALSE
  }
  if (is.null(out$has_stage_context)) {
    out$has_stage_context <- FALSE
  }
  if (is.null(out$has_matchup_context)) {
    out$has_matchup_context <- FALSE
  }
  if (is.null(out$place_context_enabled)) {
    out$place_context_enabled <- FALSE
  }
  if (is.null(out$has_place_context)) {
    out$has_place_context <- FALSE
  }
  if (is.null(out$place_feature_names)) {
    out$place_feature_names <- neural_place_feature_names()
  }
  if (is.null(out$place_context_dim)) {
    out$place_context_dim <- if (!is.null(out$place_embedding)) {
      ncol(as.matrix(out$place_embedding))
    } else if (!is.null(out$default_place_embedding)) {
      ncol(as.matrix(out$default_place_embedding))
    } else {
      0L
    }
  }
  if (is.null(out$place_present)) {
    out$place_present <- logical(0)
  }
  if (is.null(out$default_place_present)) {
    out$default_place_present <- FALSE
  }
  if (is.null(out$time_context_enabled)) {
    out$time_context_enabled <- FALSE
  }
  if (is.null(out$has_time_context)) {
    out$has_time_context <- FALSE
  }
  if (is.null(out$time_feature_names)) {
    out$time_feature_names <- neural_time_feature_names()
  }
  if (is.null(out$time_context_dim)) {
    out$time_context_dim <- if (!is.null(out$time_embedding)) {
      ncol(as.matrix(out$time_embedding))
    } else if (!is.null(out$default_time_embedding)) {
      ncol(as.matrix(out$default_time_embedding))
    } else {
      0L
    }
  }
  if (is.null(out$time_present)) {
    out$time_present <- logical(0)
  }
  if (is.null(out$default_time_present)) {
    out$default_time_present <- FALSE
  }
  out$factor_tokenization <- neural_factor_tokenization(
    mode = out$factor_tokenization %||% "fused"
  )
  out$factor_schema_supplied <- neural_model_info_factor_schema_supplied(out)
  out$covariate_value_encoding <- neural_resolve_covariate_value_encoding(
    out$covariate_value_encoding %||% "shared_projection"
  )
  old_factor_params <- c("E_factor_start", "E_factor_end", "E_factor_role", "E_feature_id")
  old_covariate_params <- c(
    "E_covariate_start",
    "E_covariate_end",
    "E_covariate_role",
    "E_covariate_id",
    "E_covariate_present",
    "V_covariate_value"
  )
  if (!is.null(out$params) &&
      any(c(old_factor_params, old_covariate_params) %in% names(out$params))) {
    stop(
      "This neural model uses span or legacy attribute parameters. Refit under fused attribute tokenization.",
      call. = FALSE
    )
  }
  if (isTRUE(out$has_factor_span_tokens) || isTRUE(out$has_feature_id_embedding)) {
    stop(
      "This neural model uses span or legacy factor tokens. Refit under fused attribute tokenization.",
      call. = FALSE
    )
  }
  if (isTRUE(out$has_covariate_span_tokens)) {
    stop(
      "This neural model uses span covariate tokens. Refit under fused attribute tokenization.",
      call. = FALSE
    )
  }
  out$has_factor_fused_tokens <- isTRUE(out$has_factor_fused_tokens) ||
    (!is.null(out$params) && !is.null(out$params$E_factor_fused_base))
  out$has_factor_span_tokens <- FALSE
  out$has_covariate_fused_tokens <- isTRUE(out$has_covariate_fused_tokens) ||
    (!is.null(out$params) && !is.null(out$params$E_covariate_fused_base))
  out$has_covariate_tokens <- isTRUE(out$has_covariate_fused_tokens)
  out$has_covariate_span_tokens <- FALSE
  if (is.null(out$has_covariate_missing_token)) {
    out$has_covariate_missing_token <- FALSE
  }
  if (is.null(out$has_covariate_value_text_projection)) {
    out$has_covariate_value_text_projection <- FALSE
  }
  if (is.null(out$has_factor_struct_projection)) {
    out$has_factor_struct_projection <- FALSE
  }
  if (is.null(out$has_level_struct_projection)) {
    out$has_level_struct_projection <- FALSE
  }
  if (is.null(out$factor_struct_dim) && !is.null(out$factor_struct_matrix)) {
    out$factor_struct_dim <- ncol(as.matrix(out$factor_struct_matrix))
  }
  if (is.null(out$level_struct_dim) &&
      !is.null(out$level_struct_matrices) &&
      length(out$level_struct_matrices) > 0L) {
    out$level_struct_dim <- ncol(as.matrix(out$level_struct_matrices[[1L]]))
  }
  if (is.null(out$party_missing_label)) {
    out$party_missing_label <- neural_missing_group_label("candidate")
  }
  if (is.null(out$resp_party_missing_label)) {
    out$resp_party_missing_label <- neural_missing_group_label("respondent")
  }
  if (is.null(out$party_missing_index)) {
    out$party_missing_index <- neural_model_party_missing_index(out)
  }
  if (is.null(out$resp_party_missing_index)) {
    out$resp_party_missing_index <- neural_model_resp_party_missing_index(out)
  }
  if (is.null(out$context_present_masking)) {
    out$context_present_masking <- TRUE
  }
  if (is.null(out$schema_dropout)) {
    out$schema_dropout <- neural_schema_dropout_zero()
  } else {
    out$schema_dropout <- neural_resolve_schema_dropout(out$schema_dropout)
  }
  if (is.null(out$low_rank_interaction_rank)) {
    out$low_rank_interaction_rank <- 0L
  } else {
    out$low_rank_interaction_rank <- neural_resolve_low_rank_interaction_rank(
      out$low_rank_interaction_rank
    )
  }
  if (is.null(out$low_rank_logit_transform)) {
    out$low_rank_logit_transform <- "none"
  } else {
    out$low_rank_logit_transform <- neural_normalize_low_rank_logit_transform(
      out$low_rank_logit_transform
    )
    if (is.na(out$low_rank_logit_transform)) {
      out$low_rank_logit_transform <- "none"
    }
  }
  if (out$low_rank_interaction_rank <= 0L ||
      !identical(out$low_rank_logit_transform, "softclip") ||
      is.null(out$low_rank_logit_bound) ||
      is.null(out$low_rank_logit_softness) ||
      !is.finite(as.numeric(out$low_rank_logit_bound)) ||
      !is.finite(as.numeric(out$low_rank_logit_softness)) ||
      as.numeric(out$low_rank_logit_bound) <= 0 ||
      as.numeric(out$low_rank_logit_softness) <= 0) {
    out$low_rank_logit_transform <- "none"
    out$low_rank_logit_bound <- NULL
    out$low_rank_logit_softness <- NULL
  } else {
    out$low_rank_logit_bound <- as.numeric(out$low_rank_logit_bound)
    out$low_rank_logit_softness <- as.numeric(out$low_rank_logit_softness)
  }
  if (is.null(out$low_rank_logit_normalization)) {
    out$low_rank_logit_normalization <- "none"
  } else {
    out$low_rank_logit_normalization <- neural_normalize_low_rank_logit_normalization(
      out$low_rank_logit_normalization
    )
    if (is.na(out$low_rank_logit_normalization)) {
      out$low_rank_logit_normalization <- "none"
    }
  }
  if (out$low_rank_interaction_rank <= 0L ||
      !identical(out$low_rank_logit_normalization, "rms") ||
      is.null(out$low_rank_head_weight_target_rms) ||
      is.null(out$low_rank_rc_out_target_rms) ||
      !is.finite(as.numeric(out$low_rank_head_weight_target_rms)) ||
      !is.finite(as.numeric(out$low_rank_rc_out_target_rms)) ||
      as.numeric(out$low_rank_head_weight_target_rms) <= 0 ||
      as.numeric(out$low_rank_rc_out_target_rms) <= 0) {
    out$low_rank_logit_normalization <- "none"
    out$low_rank_head_weight_target_rms <- NULL
    out$low_rank_rc_out_target_rms <- NULL
  } else {
    out$low_rank_head_weight_target_rms <- as.numeric(out$low_rank_head_weight_target_rms)
    out$low_rank_rc_out_target_rms <- as.numeric(out$low_rank_rc_out_target_rms)
  }
  if (is.null(out$learned_pairwise_bernoulli_logit_scale)) {
    out$learned_pairwise_bernoulli_logit_scale <- FALSE
  } else {
    out$learned_pairwise_bernoulli_logit_scale <-
      neural_resolve_learned_pairwise_bernoulli_logit_scale(
        out$learned_pairwise_bernoulli_logit_scale
      )
  }
  if (isTRUE(out$learned_pairwise_bernoulli_logit_scale)) {
    out$pairwise_bernoulli_logit_scale_prior_sd <-
      neural_resolve_pairwise_bernoulli_logit_scale_prior_sd(
        out$pairwise_bernoulli_logit_scale_prior_sd %||% NULL,
        enabled = TRUE
      )
    scale <- suppressWarnings(as.numeric(out$pairwise_bernoulli_logit_scale %||% NA_real_))
    if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
      scale <- 1.0
    }
    out$pairwise_bernoulli_logit_scale <- as.numeric(scale)
  } else {
    out$pairwise_bernoulli_logit_scale_prior_sd <- NULL
    out$pairwise_bernoulli_logit_scale <- 1.0
  }
  out$pairwise_antisymmetry <- neural_resolve_pairwise_antisymmetry(
    out$pairwise_antisymmetry %||% "legacy"
  )
  out$additive_utility_mode <- neural_resolve_additive_utility_mode(
    out$additive_utility_mode %||% "off",
    factor_tokenization = out$factor_tokenization %||% NULL
  )
  out <- neural_coerce_additive_utility_metadata(
    out,
    legacy_missing_normalization = "none"
  )
  if (is.null(out$has_additive_utility)) {
    out$has_additive_utility <- !is.null((out$params %||% list())$W_add_out)
  } else {
    out$has_additive_utility <- isTRUE(out$has_additive_utility)
  }
  out$calibration_enabled <- isTRUE(out$calibration_enabled)
  out$calibration_method <- if (isTRUE(out$calibration_enabled)) {
    neural_resolve_calibration_method(out$calibration_method %||% "logit_scale")
  } else {
    "none"
  }
  scale <- suppressWarnings(as.numeric(out$calibration_scale %||% 1.0))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    scale <- 1.0
  }
  out$calibration_scale <- as.numeric(scale)
  if (!is.null(out$calibration_prior_sd)) {
    prior_sd <- suppressWarnings(as.numeric(out$calibration_prior_sd))
    out$calibration_prior_sd <- if (length(prior_sd) == 1L &&
                                    is.finite(prior_sd) &&
                                    prior_sd > 0) {
      as.numeric(prior_sd)
    } else {
      NULL
    }
  }
  if (!is.null(out$token_family_levels) && out$low_rank_interaction_rank <= 0L) {
    out$token_family_levels <- setdiff(
      as.character(out$token_family_levels),
      c("respondent_cls", "candidate_cls")
    )
  }
  if (out$low_rank_interaction_rank <= 0L) {
    out$has_respondent_cls <- FALSE
    out$has_candidate_cls <- FALSE
    out$has_low_rank_interaction <- FALSE
    out$readout_embedding_families <- neural_readout_embedding_families(
      low_rank_interaction_rank = 0L
    )
  } else {
    if (is.null(out$has_respondent_cls)) {
      out$has_respondent_cls <- !is.null((out$params %||% list())$E_respondent_cls)
    }
    if (is.null(out$has_candidate_cls)) {
      out$has_candidate_cls <- !is.null((out$params %||% list())$E_candidate_cls)
    }
    if (is.null(out$has_low_rank_interaction)) {
      out$has_low_rank_interaction <- isTRUE(neural_has_low_rank_interaction(
        out$params %||% list(),
        out
      ))
    }
    if (is.null(out$readout_embedding_families)) {
      out$readout_embedding_families <- neural_readout_embedding_families(
        low_rank_interaction_rank = out$low_rank_interaction_rank
      )
    }
  }
  if (is.null(out$attention_backend)) {
    out$attention_backend <- "xla"
  } else {
    out$attention_backend <- neural_normalize_attention_backend(out$attention_backend)
  }
  if (is.null(out$attention_dtype)) {
    out$attention_dtype <- "float32"
  } else {
    out$attention_dtype <- neural_normalize_attention_dtype(out$attention_dtype)
  }
  if (is.null(out$attention_padding_multiple)) {
    out$attention_padding_multiple <- 8L
  } else {
    out$attention_padding_multiple <- as.integer(out$attention_padding_multiple)
  }
  if (is.null(out$attention_resolved_backend)) {
    out$attention_resolved_backend <- out$attention_backend
  }
  if (is.null(out$attention_fallback_reason)) {
    out$attention_fallback_reason <- NA_character_
  }
  if (is.null(out$has_stacked_transformer_layers)) {
    out$has_stacked_transformer_layers <- isTRUE(
      !is.null(out$param_names) &&
        any(as.character(out$param_names) %in% neural_standard_transformer_stack_names())
    )
  }
  if (is.null(out$n_resp_party_levels) && !is.null(out$resp_party_levels)) {
    out$n_resp_party_levels <- as.integer(length(out$resp_party_levels))
  }
  out
}

cs2step_pack_predictor <- function(object, include_metrics = TRUE) {
  if (!inherits(object, "strategic_predictor")) {
    stop("Can only save objects of class 'strategic_predictor'.", call. = FALSE)
  }

  metadata <- object$metadata %||% list()
  text_meta <- cs2step_capture_text_embedding_metadata(
    text_embedding_fn = metadata$text_embedding_fn %||% NULL,
    text_embedding_backend = metadata$text_embedding_backend %||% NULL
  )
  metadata$text_embedding_fn <- text_meta$text_embedding_fn
  metadata$text_embedding_backend <- text_meta$text_embedding_backend

  packed <- list(
    schema_version = 1L,
    model_type = object$model_type,
    mode = object$mode,
    encoder = object$encoder,
    metadata = metadata
  )

  if (identical(object$model_type, "glm")) {
    fit <- object$fit
    packed$fit <- list(
      intercept = fit$intercept,
      coefficients = fit$coefficients,
      vcov = fit$vcov,
      main_info = fit$main_info,
      interaction_info = fit$interaction_info,
      family = fit$family
    )
    if (isTRUE(include_metrics) && !is.null(fit$fit_metrics)) {
      packed$fit$fit_metrics <- fit$fit_metrics
    }
  } else {
    fit <- object$fit
    drop_params <- !is.null(fit$theta_mean) && length(fit$theta_mean) > 0L
    model_info <- cs2step_neural_pack_model_info(fit$neural_model_info, drop_params = drop_params)
    packed$fit <- list(
      theta_mean = if (!is.null(fit$theta_mean)) as.numeric(fit$theta_mean) else NULL,
      theta_var = if (!is.null(fit$theta_var)) as.numeric(fit$theta_var) else NULL,
      neural_model_info = model_info
    )
    if (isTRUE(include_metrics)) {
      metrics <- fit$fit_metrics %||% (if (!is.null(model_info)) model_info$fit_metrics else NULL)
      if (!is.null(metrics)) {
        packed$fit$fit_metrics <- metrics
      }
    }
  }

  class(packed) <- c("strategic_predictor_bundle", "list")
  packed
}

cs2step_unpack_predictor <- function(bundle,
                                     conda_env = "strategize_env",
                                     conda_env_required = TRUE,
                                     preload_params = TRUE) {
  if (inherits(bundle, "strategic_predictor")) {
    return(bundle)
  }
  if (!is.list(bundle) || is.null(bundle$model_type)) {
    stop("Unrecognized predictor cache format.", call. = FALSE)
  }

  if (identical(bundle$model_type, "glm")) {
    fit <- list(
      intercept = bundle$fit$intercept,
      coefficients = bundle$fit$coefficients,
      vcov = bundle$fit$vcov,
      main_info = bundle$fit$main_info,
      interaction_info = bundle$fit$interaction_info,
      family = bundle$fit$family,
      fit_metrics = bundle$fit$fit_metrics %||% NULL
    )
    return(structure(
      list(
        model_type = bundle$model_type,
        mode = bundle$mode,
        encoder = bundle$encoder,
        fit = fit,
        metadata = bundle$metadata
      ),
      class = "strategic_predictor"
    ))
  }

  if (!cs2step_has_reticulate()) {
    stop("Loading neural predictors requires the 'reticulate' package.", call. = FALSE)
  }
  upgraded_model_info <- cs2step_neural_upgrade_model_info(bundle$fit$neural_model_info)
  if (identical(
    neural_covariate_value_encoding(upgraded_model_info),
    "shared_projection"
  ) && cs2step_neural_has_resp_covariates(upgraded_model_info) &&
      !isTRUE(upgraded_model_info$has_covariate_fused_tokens)) {
    stop(
      "This shared_projection bundle uses the pre-fused covariate encoder. Refit the model under the updated architecture.",
      call. = FALSE
    )
  }
  neural_validate_full_attn_compatibility(
    model_info = upgraded_model_info,
    context = "Neural predictor bundle"
  )
  neural_validate_cross_attn_compatibility(
    model_info = upgraded_model_info,
    context = "Neural predictor bundle"
  )
  bundle$fit$neural_model_info <- upgraded_model_info

  fit <- list(
    my_model = NULL,
    predict_pair = NULL,
    predict_single = NULL,
    neural_model_info = bundle$fit$neural_model_info,
    theta_mean = bundle$fit$theta_mean,
    theta_var = bundle$fit$theta_var,
    fit_metrics = bundle$fit$fit_metrics %||% (if (!is.null(bundle$fit$neural_model_info)) {
      bundle$fit$neural_model_info$fit_metrics
    } else {
      NULL
    })
  )

  metadata <- bundle$metadata %||% list()
  env_use <- if (!is.null(conda_env) && nzchar(conda_env)) {
    conda_env
  } else {
    metadata$conda_env
  }
  metadata$conda_env <- env_use %||% "strategize_env"
  metadata$conda_env_required <- if (is.null(conda_env_required)) {
    metadata$conda_env_required %||% TRUE
  } else {
    conda_env_required
  }
  metadata <- cs2step_restore_text_embedding_metadata(
    metadata,
    conda_env = metadata$conda_env,
    required = FALSE
  )

  out <- structure(
    list(
      model_type = bundle$model_type,
      mode = bundle$mode,
      encoder = bundle$encoder,
      fit = fit,
      metadata = metadata
    ),
    class = "strategic_predictor"
  )

  if (isTRUE(preload_params)) {
    prep <- cs2step_neural_prepare_params(out,
                                          conda_env = metadata$conda_env,
                                          conda_env_required = metadata$conda_env_required)
    out$fit$params <- prep$params
  }
  out
}

#' Save a strategic predictor to disk
#'
#' @param fit A fitted \code{strategic_predictor}.
#' @param file Path to save the cache (typically ending in \code{.rds}).
#' @param overwrite Logical; overwrite an existing file.
#' @param compress Compression passed to \code{saveRDS()}.
#' @param include_metrics Logical; include out-of-sample fit metrics when present.
#' @return The cache path (invisibly).
#' @export
save_strategic_predictor <- function(fit,
                                     file,
                                     overwrite = FALSE,
                                     compress = TRUE,
                                     include_metrics = TRUE) {
  if (!inherits(fit, "strategic_predictor")) {
    stop("'fit' must be a strategic_predictor.", call. = FALSE)
  }
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (file.exists(file) && !isTRUE(overwrite)) {
    stop("Cache file already exists; set overwrite = TRUE to replace it.", call. = FALSE)
  }
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  bundle <- cs2step_pack_predictor(fit, include_metrics = include_metrics)
  saveRDS(bundle, file = file, compress = compress)
  neural_svi_checkpoint_remove_dir(paste0(file, ".inprogress"))
  invisible(file)
}

#' Load a strategic predictor from disk
#'
#' @param file Path to a cached predictor created by \code{save_strategic_predictor()}.
#' @param conda_env Conda env name for neural predictors. Use \code{NULL} to
#'   defer to the cached metadata.
#' @param conda_env_required Require conda env to exist for neural predictors.
#' @return A \code{strategic_predictor} ready for \code{predict()}.
#' @export
load_strategic_predictor <- function(file,
                                     conda_env = "strategize_env",
                                     conda_env_required = TRUE) {
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (!file.exists(file)) {
    stop("Cache file does not exist.", call. = FALSE)
  }
  bundle <- readRDS(file)
  cs2step_unpack_predictor(bundle,
                           conda_env = conda_env,
                           conda_env_required = conda_env_required)
}

cs2step_normalize_names_list <- function(names_list) {
  if (is.null(names_list) || length(names_list) == 0L) {
    return(NULL)
  }
  out <- lapply(names_list, function(x) {
    if (is.list(x) && length(x) == 1L && is.atomic(x[[1]])) {
      return(as.character(x[[1]]))
    }
    if (is.atomic(x)) {
      return(as.character(x))
    }
    as.character(unlist(x))
  })
  if (is.null(names(out))) {
    names(out) <- names(names_list)
  }
  out
}

cs2step_build_neural_outcome_bundle <- function(theta_mean,
                                                theta_var = NULL,
                                                neural_model_info,
                                                names_list,
                                                factor_levels = NULL,
                                                mode = c("auto", "pairwise", "single"),
                                                fit_metrics = NULL,
                                                conda_env = "strategize_env",
                                                conda_env_required = TRUE,
                                                metadata = NULL) {
  if (is.null(neural_model_info)) {
    stop("'neural_model_info' is required.", call. = FALSE)
  }
  if (is.null(theta_mean)) {
    stop("'theta_mean' is required.", call. = FALSE)
  }
  names_list_norm <- cs2step_normalize_names_list(names_list)
  if (is.null(names_list_norm) || length(names_list_norm) == 0L) {
    stop("'names_list' must be provided to build a portable bundle.", call. = FALSE)
  }
  if (is.null(names(names_list_norm))) {
    names(names_list_norm) <- paste0("Factor", seq_len(length(names_list_norm)))
  }

  if (is.null(factor_levels)) {
    factor_levels <- vapply(names_list_norm, length, integer(1))
  }

  theta_mean_num <- as.numeric(cs2step_neural_to_r_array(theta_mean))
  theta_var_num <- if (!is.null(theta_var)) {
    as.numeric(cs2step_neural_to_r_array(theta_var))
  } else {
    NULL
  }

  mode <- match.arg(mode)
  if (identical(mode, "auto")) {
    mode <- if (!is.null(neural_model_info$pairwise_mode) &&
                isTRUE(neural_model_info$pairwise_mode)) {
      "pairwise"
    } else {
      "single"
    }
  }

  packed_info <- cs2step_neural_pack_model_info(neural_model_info, drop_params = TRUE)
  fit_metrics <- fit_metrics %||% packed_info$fit_metrics %||% NULL

  encoder <- list(
    factor_names = names(names_list_norm),
    names_list = lapply(names_list_norm, function(x) list(x)),
    factor_levels = as.integer(factor_levels),
    unknown_policy = "holdout"
  )

  meta_default <- list(
    created_at = Sys.time(),
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )
  meta <- modifyList(meta_default, metadata %||% list())

  bundle <- list(
    schema_version = 1L,
    model_type = "neural",
    mode = mode,
    encoder = encoder,
    fit = list(
      theta_mean = theta_mean_num,
      theta_var = theta_var_num,
      neural_model_info = packed_info,
      fit_metrics = fit_metrics
    ),
    metadata = meta
  )
  bundle$theta_mean <- theta_mean_num
  bundle$theta_var <- theta_var_num
  bundle$neural_model_info <- packed_info
  bundle$fit_metrics <- fit_metrics
  class(bundle) <- c("strategic_predictor_bundle", "list")
  bundle
}

#' Save a portable neural outcome bundle
#'
#' @param file Path to save the bundle (typically ending in \code{.rds}).
#' @param theta_mean Numeric vector of posterior means for neural parameters.
#' @param theta_var Optional numeric vector of posterior variances.
#' @param neural_model_info Neural model metadata (can include reticulate objects).
#' @param names_list Optional list of factor level names (see \code{cs_prepare_W_encoding}).
#' @param p_list Optional \code{p_list} to derive factor level names when \code{names_list} is missing.
#' @param factor_levels Optional integer vector of factor levels (derived from \code{names_list} by default).
#' @param mode \code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#' @param fit_metrics Optional fit metrics to include.
#' @param conda_env Conda env name for neural backend (stored in metadata).
#' @param conda_env_required Require conda env to exist (stored in metadata).
#' @param overwrite Logical; overwrite existing file.
#' @param compress Compression passed to \code{saveRDS()}.
#' @param metadata Optional list of extra metadata to include.
#' @return The bundle path (invisibly).
#' @export
save_neural_outcome_bundle <- function(file,
                                       theta_mean,
                                       theta_var = NULL,
                                       neural_model_info,
                                       names_list = NULL,
                                       p_list = NULL,
                                       factor_levels = NULL,
                                       mode = c("auto", "pairwise", "single"),
                                       fit_metrics = NULL,
                                       conda_env = "strategize_env",
                                       conda_env_required = TRUE,
                                       overwrite = FALSE,
                                       compress = TRUE,
                                       metadata = NULL) {
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (file.exists(file) && !isTRUE(overwrite)) {
    stop("Bundle file already exists; set overwrite = TRUE to replace it.", call. = FALSE)
  }

  if (is.null(names_list)) {
    if (!is.null(p_list) && length(p_list) > 0L) {
      names_list <- lapply(p_list, function(zer) {
        levs <- names(zer)
        if (is.null(levs)) {
          levs <- as.character(seq_len(length(zer)))
        }
        list(levs)
      })
      if (!is.null(names(p_list))) {
        names(names_list) <- names(p_list)
      }
    }
  }

  bundle <- cs2step_build_neural_outcome_bundle(
    theta_mean = theta_mean,
    theta_var = theta_var,
    neural_model_info = neural_model_info,
    names_list = names_list,
    factor_levels = factor_levels,
    mode = mode,
    fit_metrics = fit_metrics,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    metadata = metadata
  )

  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(bundle, file = file, compress = compress)
  neural_svi_checkpoint_remove_dir(paste0(file, ".inprogress"))
  invisible(file)
}

#' Load a portable neural outcome bundle
#'
#' @param file Path to a bundle created by \code{save_neural_outcome_bundle()}.
#' @param conda_env Conda env name for neural backend. Use \code{NULL} to defer to metadata.
#' @param conda_env_required Require conda env to exist for neural backend.
#' @param preload_params Logical; if TRUE, reconstruct neural params immediately.
#' @return A \code{strategic_predictor} ready for \code{predict()} / \code{predict_pair()}.
#' @export
load_neural_outcome_bundle <- function(file,
                                       conda_env = "strategize_env",
                                       conda_env_required = TRUE,
                                       preload_params = FALSE) {
  file <- as.character(file)
  if (length(file) != 1L || !nzchar(file)) {
    stop("'file' must be a non-empty path.", call. = FALSE)
  }
  if (!file.exists(file)) {
    stop("Bundle file does not exist.", call. = FALSE)
  }

  bundle <- readRDS(file)
  if (!is.list(bundle)) {
    stop("Unrecognized bundle format.", call. = FALSE)
  }
  if (is.null(bundle$fit)) {
    bundle$fit <- list(
      theta_mean = bundle$theta_mean %||% NULL,
      theta_var = bundle$theta_var %||% NULL,
      neural_model_info = bundle$neural_model_info %||% NULL,
      fit_metrics = bundle$fit_metrics %||% NULL
    )
  } else {
    if (is.null(bundle$fit$theta_mean) && !is.null(bundle$theta_mean)) {
      bundle$fit$theta_mean <- bundle$theta_mean
    }
    if (is.null(bundle$fit$theta_var) && !is.null(bundle$theta_var)) {
      bundle$fit$theta_var <- bundle$theta_var
    }
    if (is.null(bundle$fit$neural_model_info) && !is.null(bundle$neural_model_info)) {
      bundle$fit$neural_model_info <- bundle$neural_model_info
    }
    if (is.null(bundle$fit$fit_metrics) && !is.null(bundle$fit_metrics)) {
      bundle$fit$fit_metrics <- bundle$fit_metrics
    }
  }
  if (is.null(bundle$fit$neural_model_info)) {
    stop("Bundle is missing neural_model_info.", call. = FALSE)
  }
  if (is.null(bundle$model_type)) {
    bundle$model_type <- "neural"
  }
  if (is.null(bundle$mode) && !is.null(bundle$fit$neural_model_info$pairwise_mode)) {
    bundle$mode <- if (isTRUE(bundle$fit$neural_model_info$pairwise_mode)) {
      "pairwise"
    } else {
      "single"
    }
  }
  if (is.null(bundle$mode)) {
    bundle$mode <- "single"
  }

  if (is.null(bundle$encoder) || is.null(bundle$encoder$names_list)) {
    factor_levels <- bundle$fit$neural_model_info$factor_levels
    if (is.null(factor_levels)) {
      stop("Bundle is missing encoder metadata.", call. = FALSE)
    }
    n_factors <- length(factor_levels)
    names_list <- lapply(seq_len(n_factors), function(j) {
      list(as.character(seq_len(factor_levels[[j]])))
    })
    names(names_list) <- paste0("Factor", seq_len(n_factors))
    bundle$encoder <- list(
      factor_names = names(names_list),
      names_list = names_list,
      factor_levels = as.integer(factor_levels),
      unknown_policy = "holdout"
    )
  }

  out <- cs2step_unpack_predictor(
    bundle,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    preload_params = preload_params
  )
  if (inherits(out, "strategic_predictor")) {
    cache_hash <- out$metadata$fingerprint$hash %||% digest::digest(
      list(file = normalizePath(file, winslash = "/", mustWork = FALSE)),
      algo = "xxhash64",
      serialize = TRUE
    )
    out$metadata$cache_id <- paste0("neural_cache_", cache_hash)
  }
  out
}
