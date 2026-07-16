cs2step_artifact_normalize_value <- function(x) {
  if (inherits(x, "formula")) {
    return(as.character(x))
  }
  if (inherits(x, "POSIXt")) {
    return(format(as.POSIXct(x), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"))
  }
  if (is.environment(x) || is.function(x)) {
    return(NULL)
  }

  x_r <- tryCatch(cs2step_neural_to_r_array(x), error = function(e) x)
  if (is.factor(x_r)) {
    return(as.character(x_r))
  }
  if (is.data.frame(x_r)) {
    out <- lapply(x_r, cs2step_artifact_normalize_value)
    attr(out, "row.names") <- row.names(x_r)
    attr(out, "class") <- "data.frame"
    return(out)
  }
  if (is.list(x_r) && is.null(class(x_r))) {
    nms <- names(x_r)
    if (!is.null(nms) && length(nms) == length(x_r) && all(nzchar(nms))) {
      x_r <- x_r[order(nms)]
    }
    return(lapply(x_r, cs2step_artifact_normalize_value))
  }
  x_r
}

cs2step_artifact_strip_runtime_control <- function(control) {
  out <- control %||% list()
  out$checkpoint_path <- NULL
  out$checkpoint_resume <- NULL
  out$checkpoint_n_checks <- NULL
  out$checkpoint_compress <- NULL
  out
}

cs2step_artifact_fingerprint <- function(kind, fields) {
  fields <- fields %||% list()
  normalized <- lapply(fields, cs2step_artifact_normalize_value)
  list(
    schema_version = 1L,
    kind = kind,
    hash = digest::digest(normalized, algo = "xxhash64", serialize = TRUE),
    fields = names(fields)
  )
}

cs2step_artifact_assert_fingerprint <- function(stored,
                                                expected,
                                                context = "Cached artifact") {
  if (is.null(stored) || is.null(stored$hash) || is.null(stored$kind)) {
    stop(
      context,
      " is missing an artifact fingerprint. Refit with cache_overwrite = TRUE.",
      call. = FALSE
    )
  }
  if (!identical(as.integer(stored$schema_version %||% NA_integer_),
                 as.integer(expected$schema_version %||% NA_integer_)) ||
      !identical(stored$kind, expected$kind) ||
      !identical(stored$hash, expected$hash)) {
    stop(
      context,
      " fingerprint mismatch. The cached data, schema, mode, or model controls ",
      "do not match this request. Refit with cache_overwrite = TRUE.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

cs2step_predictor_request_fingerprint <- function(Y,
                                                  W,
                                                  X = NULL,
                                                  model,
                                                  mode,
                                                  pair_id = NULL,
                                                  profile_order = NULL,
                                                  varcov_cluster_variable = NULL,
                                                  names_list = NULL,
                                                  factor_levels = NULL,
                                                  neural_mcmc_control = NULL,
                                                  use_regularization = TRUE,
                                                  nFolds_glm = 3L) {
  cs2step_artifact_fingerprint(
    kind = "strategic_prediction_request",
    fields = list(
      data = list(
        Y = as.numeric(Y),
        W = as.data.frame(W),
        X = X,
        pair_id = pair_id,
        profile_order = profile_order,
        varcov_cluster_variable = varcov_cluster_variable
      ),
      schema = list(
        names_list = names_list,
        factor_levels = factor_levels
      ),
      model = list(
        model = model,
        mode = mode,
        use_regularization = isTRUE(use_regularization),
        nFolds_glm = as.integer(nFolds_glm),
        neural_mcmc_control = cs2step_artifact_strip_runtime_control(neural_mcmc_control)
      )
    )
  )
}

cs2step_neural_outcome_request_fingerprint <- function(Y,
                                                       W,
                                                       X = NULL,
                                                       X_present = NULL,
                                                       names_list = NULL,
                                                       factor_levels = NULL,
                                                       diff = FALSE,
                                                       pair_id = NULL,
                                                       profile_order = NULL,
                                                       p_list = NULL,
                                                       competing_group_variable_candidate = NULL,
                                                       competing_group_variable_respondent = NULL,
                                                       respondent_id = NULL,
                                                       respondent_task_id = NULL,
                                                       outcome_model_key = NULL,
                                                       group = NULL,
                                                       round = NULL,
                                                       adversarial = FALSE,
                                                       adversarial_model_strategy = NULL,
                                                       mcmc_control = NULL,
                                                       neural_token_info = NULL) {
  cs2step_artifact_fingerprint(
    kind = "neural_outcome_bundle_request",
    fields = list(
      data = list(
        Y = as.numeric(Y),
        W = W,
        X = X,
        X_present = X_present,
        pair_id = pair_id,
        profile_order = profile_order,
        competing_group_variable_candidate = competing_group_variable_candidate,
        competing_group_variable_respondent = competing_group_variable_respondent,
        respondent_id = respondent_id,
        respondent_task_id = respondent_task_id
      ),
      schema = list(
        names_list = names_list,
        factor_levels = factor_levels,
        p_list = p_list,
        neural_token_info = neural_token_info
      ),
      model = list(
        diff = isTRUE(diff),
        outcome_model_key = outcome_model_key,
        group = group,
        round = round,
        adversarial = isTRUE(adversarial),
        adversarial_model_strategy = adversarial_model_strategy,
        mcmc_control = cs2step_artifact_strip_runtime_control(mcmc_control)
      )
    )
  )
}

neural_resolve_training_seed <- function(mcmc_control,
                                         default = 123L,
                                         supplied = NULL) {
  seed_supplied <- supplied
  if (is.null(seed_supplied)) {
    seed_supplied <- !is.null(mcmc_control$seed)
  }
  seed <- mcmc_control$seed %||% default
  seed_num <- suppressWarnings(as.numeric(seed))
  if (length(seed_num) != 1L ||
      is.na(seed_num) ||
      !is.finite(seed_num) ||
      seed_num < 0 ||
      seed_num != floor(seed_num) ||
      seed_num > .Machine$integer.max) {
    stop("'neural_mcmc_control$seed' must be one non-negative integer.", call. = FALSE)
  }
  list(seed = as.integer(seed_num), supplied = isTRUE(seed_supplied))
}

neural_derive_training_seed <- function(seed, stream = 0L) {
  stream <- suppressWarnings(as.integer(stream))
  if (length(stream) != 1L || is.na(stream)) {
    stream <- 0L
  }
  seed_num <- as.numeric(seed) + as.numeric(stream) * 1000003
  as.integer(seed_num %% .Machine$integer.max)
}

neural_training_prng_key <- function(seed, stream = 0L) {
  strenv$jax$random$PRNGKey(ai(neural_derive_training_seed(seed, stream)))
}

neural_runtime_provenance <- function(mcmc_control = NULL,
                                      conda_env = NULL) {
  list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    conda_env = conda_env %||% NULL,
    backend = mcmc_control$backend %||% NULL,
    jax_backend = tryCatch(neural_attention_jax_backend(), error = function(e) NA_character_),
    jax_devices = tryCatch({
      devices <- strenv$jax$devices()
      vapply(devices, function(x) as.character(x), character(1))
    }, error = function(e) character(0)),
    python = tryCatch(reticulate::py_config()$python, error = function(e) NA_character_),
    r_version = as.character(getRversion()),
    package_versions = list(
      strategize = tryCatch(as.character(utils::packageVersion("strategize")), error = function(e) NA_character_),
      reticulate = tryCatch(as.character(utils::packageVersion("reticulate")), error = function(e) NA_character_),
      digest = tryCatch(as.character(utils::packageVersion("digest")), error = function(e) NA_character_)
    )
  )
}
