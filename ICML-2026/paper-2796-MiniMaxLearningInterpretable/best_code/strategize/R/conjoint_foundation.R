#' Conjoint Foundation Models
#'
#' @description
#' Load pooled conjoint foundation-model artifacts produced by
#' \pkg{preference.fm}.
#'
#' @details
#' Foundation-model work now follows a split workflow:
#'
#' \enumerate{
#'   \item Train pooled foundation models with
#'   \code{preference.fm::fit_conjoint_foundation_model()}.
#'   \item Save them as checkpoint directories with
#'   \code{preference.fm::save_conjoint_foundation_bundle()}.
#'   \item Load bundles and adapt semantic foundation models with
#'   \code{preference.fm::adapt_conjoint_foundation_model()}.
#'   Use \pkg{strategize} for embedding extraction and prediction from loaded
#'   artifacts.
#' }
#'
#' Pooled training does not force every experiment into one universal output
#' head. Experiments are grouped into compatible neural families defined by
#' \code{(mode, likelihood, n_outcomes)}. Each compatible family shares one
#' pooled schema-aware encoder and one neural fit, while incompatible families
#' are stored as separate internal groups in the returned
#' \code{conjoint_foundation_model}.
#'
#' Cross-study sharing is driven by explicit canonical ids. Raw label equality
#' alone does not force two studies to share factor or level identities.
#' Optional text embeddings now enter the FM encoder directly: each candidate
#' attribute is represented by one fused factor/value token, and each respondent
#' covariate is represented by one fused covariate/value token. Factor and
#' level information is fused by a two-layer SwiGLU MLP. Canonical ids still
#' determine schema sharing across studies.
#'
#' \code{\link{fit_conjoint_foundation_model}()} and
#' \code{\link{save_conjoint_foundation_bundle}()} are retained as migration
#' stubs that point to \pkg{preference.fm}. Use
#' \code{preference.fm::adapt_conjoint_foundation_model()} for semantic
#' target-study adaptation.
#'
#' @name conjoint-foundation
NULL

cs_foundation_default_control <- function() {
  list(
    add_experiment_indicators = TRUE,
    add_text_semantics = TRUE,
    text_embedding_fn = NULL,
    experiment_token_mode = "description",
    factor_tokenization = "fused",
    max_factor_tokens = 256L,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "name_dist_moe",
    max_covariate_tokens = 512L,
    neural_mcmc_control = list(
      subsample_method = "batch_vi",
      uncertainty_scope = "output",
      optimizer = "muon",
      low_rank_interaction_rank = 16L,
      low_rank_logit_normalization = "rms",
      low_rank_logit_transform = "none",
      additive_utility = "off",
      calibration = list(enabled = FALSE),
      svi_lr_schedule = "warmup_cosine"
    )
  )
}

cs_foundation_default_adaptation_control <- function() {
  list(
    strict_schema_match = FALSE,
    allow_extra_covariates = TRUE,
    use_text_semantics = TRUE,
    text_embedding_fn = NULL,
    experiment_token_mode = "description",
    factor_tokenization = "fused",
    max_factor_tokens = 256L,
    covariate_value_encoding = "shared_projection",
    shared_projection_value_encoder = "name_dist_moe",
    max_covariate_tokens = 512L
  )
}

cs_foundation_resolve_adaptation_control <- function(group, adaptation_control = NULL) {
  control <- modifyList(
    cs_foundation_default_adaptation_control(),
    adaptation_control %||% list()
  )
  token_control <- group$token_control %||% list()
  if (is.null(adaptation_control) || !"experiment_token_mode" %in% names(adaptation_control)) {
    control$experiment_token_mode <- token_control$experiment_token_mode %||%
      control$experiment_token_mode
  }
  if (is.null(adaptation_control) || !"factor_tokenization" %in% names(adaptation_control)) {
    control$factor_tokenization <- token_control$factor_tokenization %||%
      control$factor_tokenization
  }
  if (is.null(adaptation_control) || !"max_factor_tokens" %in% names(adaptation_control)) {
    control$max_factor_tokens <- token_control$max_factor_tokens %||%
      control$max_factor_tokens
  }
  if (is.null(adaptation_control) || !"covariate_value_encoding" %in% names(adaptation_control)) {
    control$covariate_value_encoding <- token_control$covariate_value_encoding %||%
      control$covariate_value_encoding
  }
  if (is.null(adaptation_control) || !"shared_projection_value_encoder" %in% names(adaptation_control)) {
    control$shared_projection_value_encoder <- token_control$shared_projection_value_encoder %||%
      control$shared_projection_value_encoder
  }
  if (is.null(adaptation_control) || !"max_covariate_tokens" %in% names(adaptation_control)) {
    control$max_covariate_tokens <- token_control$max_covariate_tokens %||%
      control$max_covariate_tokens
  }
  control$experiment_token_mode <- cs_foundation_experiment_token_mode(
    control$experiment_token_mode
  )
  control$factor_tokenization <- cs_foundation_factor_tokenization(
    control$factor_tokenization
  )
  control$max_factor_tokens <- neural_resolve_max_factor_tokens(
    control$max_factor_tokens %||% NULL
  )
  control$covariate_value_encoding <- cs_foundation_covariate_value_encoding(
    control$covariate_value_encoding
  )
  control$shared_projection_value_encoder <- cs_foundation_shared_projection_value_encoder(
    control$shared_projection_value_encoder %||% NULL
  )
  control$max_covariate_tokens <- neural_resolve_max_covariate_tokens(
    control$max_covariate_tokens %||% NULL
  )
  control
}

cs_foundation_experiment_token_mode <- function(mode) {
  mode_use <- tolower(as.character(mode %||% "description"))
  if (!mode_use %in% c("description", "hybrid", "legacy_id")) {
    stop(
      "'experiment_token_mode' must be one of 'description', 'hybrid', or 'legacy_id'.",
      call. = FALSE
    )
  }
  mode_use
}

cs_foundation_covariate_value_encoding <- function(mode) {
  mode_use <- tolower(as.character(mode %||% "shared_projection"))
  if (!identical(mode_use, "shared_projection")) {
    stop(
      "'covariate_value_encoding' must be 'shared_projection'.",
      call. = FALSE
    )
  }
  mode_use
}

cs_foundation_shared_projection_value_encoder <- function(mode) {
  neural_resolve_shared_projection_value_encoder(mode %||% "name_dist_moe")
}

cs_foundation_factor_tokenization <- function(mode) {
  mode_use <- tolower(as.character(mode %||% "fused"))
  if (!identical(mode_use, "fused")) {
    stop(
      "'factor_tokenization' must be 'fused'.",
      call. = FALSE
    )
  }
  mode_use
}

cs_foundation_normalize_optional_text <- function(x, arg) {
  if (is.null(x)) {
    return(NULL)
  }
  x_use <- as.character(x)
  if (length(x_use) != 1L) {
    stop(sprintf("'%s' must be length 1 when supplied.", arg), call. = FALSE)
  }
  if (!nzchar(x_use)) {
    return(NULL)
  }
  x_use
}

cs_foundation_mode <- function(mode, pair_id) {
  if (!is.null(mode)) {
    mode_use <- tolower(as.character(mode))
    if (!mode_use %in% c("single", "pairwise", "auto")) {
      stop(
        "'mode' must be one of 'auto', 'single', or 'pairwise'.",
        call. = FALSE
      )
    }
    if (!identical(mode_use, "auto")) {
      return(mode_use)
    }
  }
  if (!is.null(pair_id)) "pairwise" else "single"
}

cs_foundation_infer_likelihood <- function(Y, W, likelihood = NULL, n_outcomes = NULL) {
  if (!is.null(likelihood)) {
    like <- tolower(as.character(likelihood))
    if (!like %in% c("bernoulli", "categorical", "normal", "auto")) {
      stop(
        "'likelihood' must be one of 'auto', 'bernoulli', 'categorical', or 'normal'.",
        call. = FALSE
      )
    }
    if (!identical(like, "auto")) {
      return(list(likelihood = like, n_outcomes = n_outcomes))
    }
  }

  y_raw <- if (is.factor(Y) || is.character(Y)) as.character(Y) else Y
  y_num <- suppressWarnings(as.numeric(y_raw))
  observed <- !is.na(y_raw)
  if (!any(observed)) {
    stop("Cannot infer likelihood: outcome Y must contain at least one non-missing value.", call. = FALSE)
  }
  if (any(observed & is.na(y_num))) {
    stop(
      "Cannot infer likelihood for non-numeric outcomes. Supply likelihood='categorical' for categorical Y.",
      call. = FALSE
    )
  }

  vals <- unique(stats::na.omit(y_num))
  is_binary <- length(vals) > 0L && length(vals) <= 2L && all(vals %in% c(0, 1))
  is_intvec <- length(y_num) > 0L &&
    all(!is.na(y_num)) &&
    all(is.finite(y_num)) &&
    all(abs(y_num - round(y_num)) < 1e-8)
  k_classes <- if (is_intvec) length(unique(as.integer(y_num))) else NA_integer_

  if (is_binary) {
    return(list(likelihood = "bernoulli", n_outcomes = 1L))
  }
  if (!is.na(k_classes) && k_classes >= 2L && k_classes <= max(50L, ncol(W) + 1L)) {
    return(list(likelihood = "categorical", n_outcomes = k_classes))
  }
  list(likelihood = "normal", n_outcomes = 1L)
}

cs_foundation_numeric_outcome <- function(Y, experiment_id, likelihood) {
  y_raw <- if (is.factor(Y) || is.character(Y)) as.character(Y) else Y
  y_num <- suppressWarnings(as.numeric(y_raw))
  if (length(y_num) < 1L || any(!is.finite(y_num))) {
    stop(
      sprintf(
        "Experiment '%s' declared %s but Y contains missing, non-finite, or non-numeric values.",
        experiment_id,
        likelihood
      ),
      call. = FALSE
    )
  }
  y_num
}

cs_foundation_normalize_categorical_y <- function(Y, n_outcomes = NULL) {
  y_num <- suppressWarnings(as.numeric(Y))
  if (anyNA(y_num)) {
    stop("Categorical outcomes cannot contain NA values.", call. = FALSE)
  }
  levels_obs <- sort(unique(as.integer(y_num)))
  y_map <- match(as.integer(y_num), levels_obs) - 1L
  n_obs <- length(levels_obs)
  n_outcomes_use <- as.integer(n_outcomes %||% n_obs)
  if (n_outcomes_use != n_obs) {
    stop(
      "For categorical outcomes, 'n_outcomes' must match the number of observed classes in v1.",
      call. = FALSE
    )
  }
  list(
    Y = as.integer(y_map),
    outcome_levels = levels_obs,
    n_outcomes = n_outcomes_use
  )
}

cs_foundation_normalize_names_list_local <- function(names_list, W, p_list = NULL) {
  if (is.null(names_list)) {
    return(cs_build_names_list(W = W, p_list = p_list))
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
    names(out) <- colnames(as.data.frame(W))
  }
  lapply(out, function(x) list(as.character(x)))
}

cs_foundation_validate_numeric_matrix <- function(X, n, arg = "X") {
  if (is.null(X)) {
    return(NULL)
  }
  X_df <- as.data.frame(X)
  if (nrow(X_df) != n) {
    stop(
      sprintf("'%s' has %d rows but expected %d.", arg, nrow(X_df), n),
      call. = FALSE
    )
  }
  is_num <- vapply(X_df, function(col) is.numeric(col) || is.integer(col), logical(1))
  if (!all(is_num)) {
    bad <- names(X_df)[!is_num]
    stop(
      sprintf("'%s' must contain only numeric columns. Bad columns: %s",
              arg, paste(bad, collapse = ", ")),
      call. = FALSE
    )
  }
  X_mat <- as.matrix(X_df)
  storage.mode(X_mat) <- "double"
  if (is.null(colnames(X_mat))) {
    colnames(X_mat) <- paste0("X", seq_len(ncol(X_mat)))
  }
  X_mat
}

cs_foundation_normalize_factor_ids <- function(values, factor_names, arg = "canonical_factor_id") {
  out <- rep(NA_character_, length(factor_names))
  names(out) <- factor_names
  if (is.null(values)) {
    return(out)
  }
  if (is.list(values) && !is.atomic(values)) {
    if (!is.null(names(values))) {
      values <- values[factor_names]
    }
    values <- unlist(values, recursive = FALSE, use.names = TRUE)
  }
  values <- as.character(values)
  if (!is.null(names(values))) {
    idx <- match(names(values), factor_names)
    ok <- which(!is.na(idx))
    out[idx[ok]] <- values[ok]
  } else {
    if (length(values) != length(factor_names)) {
      stop(
        sprintf("'%s' must have length %d when unnamed.", arg, length(factor_names)),
        call. = FALSE
      )
    }
    out[] <- values
  }
  out
}

cs_foundation_normalize_level_ids <- function(values, factor_names, names_list) {
  out <- setNames(vector("list", length(factor_names)), factor_names)
  for (factor_name in factor_names) {
    levels_here <- names_list[[factor_name]][[1]]
    empty <- rep(NA_character_, length(levels_here))
    names(empty) <- levels_here
    out[[factor_name]] <- empty
  }
  if (is.null(values)) {
    return(out)
  }
  if (!is.list(values)) {
    stop("'canonical_level_id' must be a named list when provided.", call. = FALSE)
  }
  for (factor_name in intersect(names(values), factor_names)) {
    levs <- names_list[[factor_name]][[1]]
    raw <- values[[factor_name]]
    if (is.list(raw) && length(raw) == 1L && is.atomic(raw[[1]])) {
      raw <- raw[[1]]
    }
    raw <- as.character(raw)
    mapped <- rep(NA_character_, length(levs))
    names(mapped) <- levs
    if (!is.null(names(raw))) {
      idx <- match(names(raw), levs)
      ok <- which(!is.na(idx))
      mapped[idx[ok]] <- raw[ok]
    } else {
      if (length(raw) != length(levs)) {
        stop(
          sprintf(
            "canonical_level_id[['%s']] must have length %d when unnamed.",
            factor_name,
            length(levs)
          ),
          call. = FALSE
        )
      }
      mapped[] <- raw
    }
    out[[factor_name]] <- mapped
  }
  out
}

cs_foundation_make_factor_key <- function(experiment_id, factor_name, canonical_factor_id = NULL) {
  canon <- canonical_factor_id %||% NA_character_
  if (!is.na(canon) && nzchar(canon)) {
    paste0("canon::", canon)
  } else {
    paste0("local::", experiment_id, "::", factor_name)
  }
}

cs_foundation_make_level_key <- function(experiment_id,
                                         factor_key,
                                         factor_name,
                                         level_name,
                                         canonical_level_id = NULL) {
  canon <- canonical_level_id %||% NA_character_
  if (!is.na(canon) && nzchar(canon)) {
    paste0("canon::", canon)
  } else {
    paste0("local::", experiment_id, "::", factor_name, "::", level_name)
  }
}

# All level-key formats a canonical level id may be registered under, in
# priority order. preference.fm writes factor-scoped keys
# ("<factor_key>::level::<id>", its default) or globally scoped keys
# ("canon::level::<id>"); older strategize-trained artifacts hold
# "canon::<id>". Resolving a request against a single format silently mapped
# canonical levels from preference.fm-trained artifacts to holdout tokens --
# match against every known format instead, then fall back to the local key.
cs_foundation_candidate_level_keys <- function(experiment_id,
                                               factor_key,
                                               factor_name,
                                               level_name,
                                               canonical_level_id = NULL) {
  canon <- canonical_level_id %||% NA_character_
  keys <- character(0)
  if (!is.na(canon) && nzchar(canon)) {
    keys <- c(
      paste0(factor_key, "::level::", canon),
      paste0("canon::level::", canon),
      paste0("canon::", canon)
    )
  }
  c(keys, paste0("local::", experiment_id, "::", factor_name, "::", level_name))
}

cs_foundation_group_key <- function(mode,
                                    likelihood,
                                    n_outcomes,
                                    pairwise_context_mode = NULL) {
  if (identical(mode, "pairwise")) {
    context_mode <- tolower(as.character(pairwise_context_mode %||% "stage_free"))
    if (!context_mode %in% c("stage_free", "stage_aware")) {
      context_mode <- "stage_free"
    }
    return(paste(mode, context_mode, likelihood, as.integer(n_outcomes %||% 1L), sep = "::"))
  }
  paste(mode, likelihood, as.integer(n_outcomes %||% 1L), sep = "::")
}

cs_foundation_universal_group_key <- function() {
  "universal::mixed::v1"
}

cs_foundation_is_current_semantic_model <- function(foundation_model) {
  groups <- foundation_model$groups %||% list()
  if (cs_foundation_universal_group_key() %in% names(groups)) {
    return(TRUE)
  }
  any(vapply(groups, function(group) {
    token_info <- group$token_info %||% list()
    token_control <- group$token_control %||% list()
    x_schema <- group$x_schema %||% list()
    identical(group$transfer_mode %||% NULL, "semantic_zero_overlap") ||
      identical(token_info$transfer_mode %||% NULL, "semantic_zero_overlap") ||
      identical(token_control$transfer_mode %||% NULL, "semantic_zero_overlap") ||
      identical(x_schema$transfer_mode %||% NULL, "semantic_zero_overlap")
  }, logical(1)))
}

cs_foundation_group_aliases <- function(mode,
                                        likelihood,
                                        n_outcomes,
                                        pairwise_context_mode = NULL) {
  aliases <- cs_foundation_group_key(
    mode = mode,
    likelihood = likelihood,
    n_outcomes = n_outcomes,
    pairwise_context_mode = pairwise_context_mode %||% NULL
  )
  if (identical(mode, "pairwise")) {
    aliases <- c(
      aliases,
      paste(mode, likelihood, as.integer(n_outcomes %||% 1L), sep = "::")
    )
  }
  unique(aliases)
}

cs_foundation_normalize_group_variable <- function(values, n, arg) {
  if (is.null(values)) {
    return(NULL)
  }
  if (length(values) != n) {
    stop(
      sprintf("%s has length %d but W has %d rows.", arg, length(values), n),
      call. = FALSE
    )
  }
  out <- trimws(as.character(values))
  out[!nzchar(out)] <- NA_character_
  out
}

cs_foundation_pairwise_context_mode <- function(mode,
                                                W,
                                                pair_id,
                                                profile_order,
                                                competing_group_variable_candidate,
                                                competing_group_variable_respondent) {
  if (!identical(mode, "pairwise")) {
    return(NULL)
  }
  if (is.null(competing_group_variable_candidate) ||
      is.null(competing_group_variable_respondent)) {
    return("stage_free")
  }
  cand <- as.character(competing_group_variable_candidate)
  resp <- as.character(competing_group_variable_respondent)
  cand[!nzchar(trimws(cand))] <- NA_character_
  resp[!nzchar(trimws(resp))] <- NA_character_
  if (all(is.na(cand)) || all(is.na(resp))) {
    return("stage_free")
  }
  pair_info <- cs2step_build_pair_mat(
    pair_id = pair_id,
    W = W,
    profile_order = profile_order,
    competing_group_variable_candidate = cand
  )
  if (is.null(pair_info) ||
      is.null(pair_info$pair_sizes) ||
      !all(pair_info$pair_sizes == 2L)) {
    return("stage_free")
  }
  pair_mat <- pair_info$pair_mat
  if (is.null(pair_mat) || nrow(pair_mat) < 1L) {
    return("stage_free")
  }
  stage_is_primary <- cand[pair_mat[, 1L]] == cand[pair_mat[, 2L]]
  known_stage <- !is.na(stage_is_primary)
  has_primary <- any(stage_is_primary[known_stage], na.rm = TRUE)
  has_general <- any(!stage_is_primary[known_stage], na.rm = TRUE)
  if (isTRUE(has_primary) && isTRUE(has_general)) "stage_aware" else "stage_free"
}

cs_foundation_normalize_experiment <- function(experiment, index) {
  if (!is.list(experiment)) {
    stop("Each experiment must be supplied as a list.", call. = FALSE)
  }
  experiment_id <- as.character(experiment$experiment_id %||% sprintf("experiment_%03d", index))
  if (length(experiment_id) != 1L || !nzchar(experiment_id)) {
    stop("Each experiment must have a non-empty 'experiment_id'.", call. = FALSE)
  }
  if (is.null(experiment$Y)) {
    stop(sprintf("Experiment '%s' is missing 'Y'.", experiment_id), call. = FALSE)
  }
  if (is.null(experiment$W)) {
    stop(sprintf("Experiment '%s' is missing 'W'.", experiment_id), call. = FALSE)
  }

  W_df <- as.data.frame(experiment$W)
  if (ncol(W_df) < 1L) {
    stop(sprintf("Experiment '%s' must have at least one factor column.", experiment_id), call. = FALSE)
  }
  if (is.null(colnames(W_df))) {
    colnames(W_df) <- paste0("V", seq_len(ncol(W_df)))
  }
  Y_raw <- experiment$Y
  if (length(Y_raw) != nrow(W_df)) {
    stop(
      sprintf("Experiment '%s' has %d outcomes but %d profile rows.",
              experiment_id, length(Y_raw), nrow(W_df)),
      call. = FALSE
    )
  }

  mode <- cs_foundation_mode(experiment$mode %||% NULL, experiment$pair_id %||% NULL)
  if (identical(mode, "pairwise")) {
    cs2step_validate_pairwise_ids(experiment$pair_id, nrow(W_df))
  }

  names_list <- cs_foundation_normalize_names_list_local(
    names_list = experiment$names_list %||% NULL,
    W = W_df,
    p_list = experiment$p_list %||% NULL
  )
  factor_names <- names(names_list)
  factor_levels <- vapply(names_list, function(x) length(x[[1]]), integer(1))

  like_info <- cs_foundation_infer_likelihood(
    Y = Y_raw,
    W = W_df,
    likelihood = experiment$likelihood %||% NULL,
    n_outcomes = experiment$n_outcomes %||% NULL
  )
  likelihood <- like_info$likelihood
  n_outcomes <- as.integer(like_info$n_outcomes %||% 1L)
  outcome_levels <- NULL
  if (identical(likelihood, "bernoulli")) {
    Y_use <- cs_foundation_numeric_outcome(Y_raw, experiment_id, "bernoulli")
    vals <- unique(Y_use)
    if (!all(vals %in% c(0, 1))) {
      stop(
        sprintf("Experiment '%s' declared bernoulli but Y is not binary 0/1.", experiment_id),
        call. = FALSE
      )
    }
  } else if (identical(likelihood, "categorical")) {
    cat_info <- cs_foundation_normalize_categorical_y(Y_raw, n_outcomes = n_outcomes)
    Y_use <- cat_info$Y
    outcome_levels <- cat_info$outcome_levels
    n_outcomes <- cat_info$n_outcomes
  } else {
    Y_use <- cs_foundation_numeric_outcome(Y_raw, experiment_id, "normal")
  }

  X_use <- cs_foundation_validate_numeric_matrix(
    X = experiment$X %||% NULL,
    n = nrow(W_df),
    arg = sprintf("experiments[['%s']]$X", experiment_id)
  )

  respondent_id <- experiment$respondent_id %||% NULL
  if (!is.null(respondent_id) && length(respondent_id) != nrow(W_df)) {
    stop(
      sprintf("Experiment '%s' has respondent_id length %d but %d rows.",
              experiment_id, length(respondent_id), nrow(W_df)),
      call. = FALSE
    )
  }
  respondent_task_id <- experiment$respondent_task_id %||% NULL
  if (!is.null(respondent_task_id) && length(respondent_task_id) != nrow(W_df)) {
    stop(
      sprintf("Experiment '%s' has respondent_task_id length %d but %d rows.",
              experiment_id, length(respondent_task_id), nrow(W_df)),
      call. = FALSE
    )
  }
  competing_group_variable_candidate <- cs_foundation_normalize_group_variable(
    values = experiment$competing_group_variable_candidate %||% NULL,
    n = nrow(W_df),
    arg = sprintf("experiments[['%s']]$competing_group_variable_candidate", experiment_id)
  )
  competing_group_variable_respondent <- cs_foundation_normalize_group_variable(
    values = experiment$competing_group_variable_respondent %||% NULL,
    n = nrow(W_df),
    arg = sprintf("experiments[['%s']]$competing_group_variable_respondent", experiment_id)
  )
  pairwise_context_mode <- cs_foundation_pairwise_context_mode(
    mode = mode,
    W = W_df,
    pair_id = experiment$pair_id %||% NULL,
    profile_order = experiment$profile_order %||% NULL,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_variable_respondent = competing_group_variable_respondent
  )

  canonical_factor_id <- cs_foundation_normalize_factor_ids(
    values = experiment$canonical_factor_id %||% NULL,
    factor_names = factor_names
  )
  canonical_level_id <- cs_foundation_normalize_level_ids(
    values = experiment$canonical_level_id %||% NULL,
    factor_names = factor_names,
    names_list = names_list
  )
  experiment_description <- cs_foundation_normalize_optional_text(
    experiment$experiment_description %||% NULL,
    arg = "experiment_description"
  )

  list(
    experiment_id = experiment_id,
    experiment_label = as.character(experiment$experiment_label %||% experiment_id),
    experiment_description = experiment_description,
    Y = Y_use,
    Y_raw = Y_raw,
    W = W_df,
    names_list = names_list,
    factor_names = factor_names,
    factor_levels = factor_levels,
    mode = mode,
    likelihood = likelihood,
    n_outcomes = as.integer(n_outcomes),
    outcome_levels = outcome_levels,
    pair_id = experiment$pair_id %||% NULL,
    profile_order = experiment$profile_order %||% NULL,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_variable_respondent = competing_group_variable_respondent,
    pairwise_context_mode = pairwise_context_mode,
    X = X_use,
    canonical_factor_id = canonical_factor_id,
    canonical_level_id = canonical_level_id
  )
}

cs_foundation_text_embed <- function(text_embedding_fn, texts) {
  texts <- as.character(texts)
  if (!length(texts)) {
    return(matrix(numeric(0), nrow = 0L, ncol = 0L))
  }
  out <- tryCatch(text_embedding_fn(texts), error = function(e) NULL)
  if (!is.null(out)) {
    out <- as.matrix(out)
    storage.mode(out) <- "double"
    if (nrow(out) == length(texts)) {
      return(out)
    }
  }
  out_rows <- lapply(texts, function(txt) {
    emb <- text_embedding_fn(txt)
    emb <- as.numeric(emb)
    if (!length(emb)) {
      stop("text_embedding_fn must return a numeric vector or matrix.", call. = FALSE)
    }
    emb
  })
  dims <- unique(vapply(out_rows, length, integer(1)))
  if (length(dims) != 1L) {
    stop("text_embedding_fn returned inconsistent embedding lengths.", call. = FALSE)
  }
  mat <- do.call(rbind, out_rows)
  storage.mode(mat) <- "double"
  mat
}

cs_foundation_collect_experiment_texts <- function(experiments) {
  ids <- vapply(experiments, `[[`, character(1), "experiment_id")
  desc <- vapply(experiments, function(exp) {
    desc_use <- exp$experiment_description %||% NULL
    if (is.null(desc_use)) {
      return(NA_character_)
    }
    as.character(desc_use)
  }, character(1))
  names(desc) <- ids
  desc[!is.na(desc) & nzchar(desc)]
}

cs_foundation_build_group_registry <- function(experiments) {
  slot_key_to_name <- character(0)
  slot_names <- character(0)
  slot_key_to_index <- integer(0)
  slot_level_keys <- list()
  slot_level_labels <- list()
  slot_factor_labels <- character(0)
  experiment_maps <- list()

  for (exp in experiments) {
    factor_map <- vector("list", length(exp$factor_names))
    names(factor_map) <- exp$factor_names
    level_map <- setNames(vector("list", length(exp$factor_names)), exp$factor_names)

    for (j in seq_along(exp$factor_names)) {
      factor_name <- exp$factor_names[[j]]
      factor_key <- cs_foundation_make_factor_key(
        experiment_id = exp$experiment_id,
        factor_name = factor_name,
        canonical_factor_id = exp$canonical_factor_id[[factor_name]]
      )
      if (!factor_key %in% names(slot_key_to_name)) {
        slot_name <- sprintf("slot_%03d", length(slot_names) + 1L)
        slot_key_to_name[[factor_key]] <- slot_name
        slot_names <- c(slot_names, slot_name)
        slot_key_to_index[[factor_key]] <- length(slot_names)
        slot_level_keys[[slot_name]] <- character(0)
        slot_level_labels[[slot_name]] <- character(0)
        slot_factor_labels[[slot_name]] <- factor_name
      }
      slot_name <- slot_key_to_name[[factor_key]]
      factor_map[[factor_name]] <- list(
        slot_key = factor_key,
        slot_name = slot_name,
        slot_index = as.integer(slot_key_to_index[[factor_key]])
      )

      levels_here <- exp$names_list[[factor_name]][[1]]
      level_key_map <- character(length(levels_here))
      names(level_key_map) <- levels_here
      for (lvl in levels_here) {
        level_key <- cs_foundation_make_level_key(
          experiment_id = exp$experiment_id,
          factor_key = factor_key,
          factor_name = factor_name,
          level_name = lvl,
          canonical_level_id = exp$canonical_level_id[[factor_name]][[lvl]]
        )
        level_key_map[[lvl]] <- level_key
        if (!level_key %in% slot_level_keys[[slot_name]]) {
          slot_level_keys[[slot_name]] <- c(slot_level_keys[[slot_name]], level_key)
          slot_level_labels[[slot_name]] <- c(slot_level_labels[[slot_name]], lvl)
        }
      }
      level_map[[factor_name]] <- level_key_map
    }

    experiment_maps[[exp$experiment_id]] <- list(
      factor_map = factor_map,
      level_map = level_map
    )
  }

  pooled_names_list <- lapply(slot_names, function(slot_name) {
    list(slot_level_keys[[slot_name]])
  })
  names(pooled_names_list) <- slot_names
  slot_keys_ordered <- vapply(slot_names, function(slot_name) {
    names(slot_key_to_name)[match(slot_name, unname(slot_key_to_name))]
  }, character(1))

  slot_table <- data.frame(
    slot_name = slot_names,
    slot_key = slot_keys_ordered,
    display_label = unname(slot_factor_labels[slot_names]),
    stringsAsFactors = FALSE
  )

  list(
    slot_table = slot_table,
    pooled_names_list = pooled_names_list,
    slot_level_keys = slot_level_keys,
    slot_level_labels = slot_level_labels,
    experiment_maps = experiment_maps
  )
}

cs_foundation_build_text_registry <- function(experiments,
                                             registry,
                                             text_embedding_fn,
                                             x_feature_names = character(0),
                                             experiment_texts = NULL) {
  if (is.null(text_embedding_fn)) {
    return(NULL)
  }
  slot_keys <- registry$slot_table$slot_key
  slot_texts <- registry$slot_table$display_label
  slot_emb <- cs_foundation_text_embed(text_embedding_fn, slot_texts)
  rownames(slot_emb) <- slot_keys

  level_keys <- unlist(registry$slot_level_keys, use.names = FALSE)
  level_labels <- unlist(registry$slot_level_labels, use.names = FALSE)
  level_emb <- cs_foundation_text_embed(text_embedding_fn, level_labels)
  rownames(level_emb) <- level_keys

  x_feature_names <- as.character(x_feature_names %||% character(0))
  x_feature_emb <- if (length(x_feature_names) > 0L) {
    emb <- cs_foundation_text_embed(text_embedding_fn, x_feature_names)
    rownames(emb) <- x_feature_names
    emb
  } else {
    NULL
  }
  experiment_texts <- experiment_texts %||% character(0)
  experiment_keys <- names(experiment_texts) %||% character(0)
  experiment_texts <- as.character(unname(experiment_texts))
  keep_experiment <- !is.na(experiment_texts) &
    nzchar(experiment_texts) &
    !is.na(experiment_keys) &
    nzchar(experiment_keys)
  experiment_texts <- experiment_texts[keep_experiment]
  experiment_keys <- experiment_keys[keep_experiment]
  experiment_emb <- if (length(experiment_texts) > 0L) {
    emb <- cs_foundation_text_embed(text_embedding_fn, experiment_texts)
    rownames(emb) <- experiment_keys
    emb
  } else {
    NULL
  }

  dims <- unique(c(
    if (ncol(slot_emb) > 0L) ncol(slot_emb) else integer(0),
    if (ncol(level_emb) > 0L) ncol(level_emb) else integer(0),
    if (!is.null(x_feature_emb) && ncol(x_feature_emb) > 0L) ncol(x_feature_emb) else integer(0),
    if (!is.null(experiment_emb) && ncol(experiment_emb) > 0L) ncol(experiment_emb) else integer(0)
  ))
  if (length(dims) > 1L) {
    stop("text_embedding_fn returned inconsistent embedding lengths.", call. = FALSE)
  }
  dim_use <- if (length(dims) > 0L) as.integer(dims[[1]]) else 0L
  if (is.null(x_feature_emb)) {
    x_feature_emb <- matrix(numeric(0), nrow = 0L, ncol = dim_use)
    rownames(x_feature_emb) <- character(0)
  }
  if (is.null(experiment_emb)) {
    experiment_emb <- matrix(numeric(0), nrow = 0L, ncol = dim_use)
    rownames(experiment_emb) <- character(0)
  }

  list(
    factor_embedding = slot_emb,
    level_embedding = level_emb,
    x_feature_embedding = x_feature_emb,
    experiment_embedding = experiment_emb,
    x_feature_names = x_feature_names,
    dim = dim_use
  )
}

cs_foundation_semantic_feature_names <- function(text_registry) {
  if (is.null(text_registry)) {
    return(character(0))
  }
  dim_use <- as.integer(text_registry$dim %||% 0L)
  if (dim_use < 1L) {
    return(character(0))
  }
  out <- c(
    paste0("semantic_factor_", seq_len(dim_use)),
    paste0("semantic_level_", seq_len(dim_use))
  )
  if (length(text_registry$x_feature_names %||% character(0)) > 0L) {
    out <- c(out, paste0("semantic_x_", seq_len(dim_use)))
  }
  out
}

cs_foundation_get_embedding_rows <- function(emb_matrix, keys, return_present = FALSE) {
  keys <- as.character(keys)
  present <- rep(FALSE, length(keys))
  if (is.null(emb_matrix) || !length(keys)) {
    rows <- matrix(numeric(0), nrow = length(keys), ncol = 0L)
    return(if (isTRUE(return_present)) list(rows = rows, present = present) else rows)
  }
  out <- matrix(0, nrow = length(keys), ncol = ncol(emb_matrix))
  rownames(out) <- keys
  colnames(out) <- colnames(emb_matrix)
  matched <- match(keys, rownames(emb_matrix))
  ok <- which(!is.na(matched))
  if (length(ok) > 0L) {
    out[ok, ] <- emb_matrix[matched[ok], , drop = FALSE]
    present[ok] <- TRUE
  }
  if (isTRUE(return_present)) list(rows = out, present = present) else out
}

cs_foundation_align_covariate_block <- function(schema_names,
                                                X_mat,
                                                n_rows) {
  schema_names <- as.character(schema_names %||% character(0))
  values <- matrix(0, nrow = n_rows, ncol = length(schema_names))
  colnames(values) <- schema_names
  present <- matrix(0, nrow = n_rows, ncol = length(schema_names))
  colnames(present) <- schema_names

  if (length(schema_names) < 1L || is.null(X_mat) || ncol(X_mat) < 1L) {
    return(list(values = values, present = present))
  }

  idx <- match(colnames(X_mat), schema_names)
  ok <- which(!is.na(idx))
  if (length(ok) > 0L) {
    values[, idx[ok]] <- X_mat[, ok, drop = FALSE]
    present[, idx[ok]] <- 1
  }
  list(values = values, present = present)
}

cs_foundation_build_registry_slot_maps <- function(registry) {
  slot_names <- registry$slot_table$slot_name
  factor_map <- setNames(lapply(seq_along(slot_names), function(i) {
    list(
      slot_key = registry$slot_table$slot_key[[i]],
      slot_name = slot_names[[i]],
      slot_index = as.integer(i)
    )
  }), slot_names)
  level_map <- setNames(lapply(slot_names, function(slot_name) {
    lvl_keys <- registry$slot_level_keys[[slot_name]] %||% character(0)
    stats::setNames(lvl_keys, lvl_keys)
  }), slot_names)
  list(
    factor_map = factor_map,
    level_map = level_map
  )
}

cs_foundation_build_token_info <- function(names_list,
                                           factor_map,
                                           level_map,
                                           text_registry,
                                           factor_order_by_experiment = NULL,
                                           default_factor_order = NULL,
                                           covariate_names = character(0),
                                           covariate_order_by_experiment = NULL,
                                           default_covariate_order = NULL,
                                           experiment_levels = character(0),
                                           experiment_index = NULL,
                                           experiment_token_mode = "description",
                                           factor_tokenization = "fused",
                                           max_factor_tokens = 256L,
                                           covariate_value_encoding = "shared_projection",
                                           shared_projection_value_encoder = "name_dist_moe",
                                           max_covariate_tokens = 512L,
                                           default_experiment_key = NULL) {
  factor_names <- names(names_list)
  dim_use <- if (is.null(text_registry)) {
    0L
  } else {
    as.integer(text_registry$dim %||% 0L)
  }

  factor_name_text <- NULL
  level_name_text <- NULL
  factor_name_text_present <- rep(FALSE, length(factor_names))
  names(factor_name_text_present) <- factor_names
  level_name_text_present <- setNames(vector("list", length(factor_names)), factor_names)
  covariate_name_text <- NULL
  experiment_description_text <- NULL
  experiment_description_present <- NULL
  default_experiment_text <- NULL
  default_experiment_text_present <- FALSE
  if (dim_use > 0L) {
    factor_keys <- vapply(factor_map[factor_names], function(x) {
      x$slot_key
    }, character(1))
    factor_name_text_present <- !is.na(match(
      factor_keys,
      rownames(text_registry$factor_embedding %||% matrix(numeric(0), nrow = 0L, ncol = dim_use))
    ))
    factor_name_text <- cs_foundation_get_embedding_rows(
      emb_matrix = text_registry$factor_embedding,
      keys = factor_keys
    )
    rownames(factor_name_text) <- factor_names

    level_name_text_info <- setNames(lapply(factor_names, function(factor_name) {
      levels_here <- names_list[[factor_name]][[1]]
      out <- matrix(0, nrow = length(levels_here) + 1L, ncol = dim_use)
      rownames(out) <- c(levels_here, "__holdout__")
      level_keys <- unname(level_map[[factor_name]][levels_here])
      level_present <- !is.na(match(
        level_keys,
        rownames(text_registry$level_embedding %||% matrix(numeric(0), nrow = 0L, ncol = dim_use))
      ))
      if (length(level_keys) > 0L) {
        out[seq_along(levels_here), ] <- cs_foundation_get_embedding_rows(
          emb_matrix = text_registry$level_embedding,
          keys = level_keys
        )
      }
      list(text = out, present = level_present)
    }), factor_names)
    level_name_text <- lapply(level_name_text_info, `[[`, "text")
    level_name_text_present <- lapply(level_name_text_info, `[[`, "present")

    covariate_names <- as.character(covariate_names %||% character(0))
    if (length(covariate_names) > 0L) {
      covariate_name_text <- cs_foundation_get_embedding_rows(
        emb_matrix = text_registry$x_feature_embedding,
        keys = covariate_names
      )
      rownames(covariate_name_text) <- covariate_names
    }

    experiment_levels <- as.character(experiment_levels %||% character(0))
    if (length(experiment_levels) > 0L) {
      experiment_description_text <- cs_foundation_get_embedding_rows(
        emb_matrix = text_registry$experiment_embedding,
        keys = experiment_levels
      )
      rownames(experiment_description_text) <- experiment_levels
      experiment_description_present <- !is.na(match(
        experiment_levels,
        rownames(text_registry$experiment_embedding %||% matrix(numeric(0), nrow = 0L, ncol = dim_use))
      ))
    }
    default_experiment_key <- as.character(default_experiment_key %||% character(0))
    if (length(default_experiment_key) > 0L) {
      default_experiment_text <- cs_foundation_get_embedding_rows(
        emb_matrix = text_registry$experiment_embedding,
        keys = default_experiment_key[[1L]]
      )
      rownames(default_experiment_text) <- default_experiment_key[[1L]]
      default_experiment_text_present <- !is.na(match(
        default_experiment_key[[1L]],
        rownames(text_registry$experiment_embedding %||% matrix(numeric(0), nrow = 0L, ncol = dim_use))
      ))
    }
  }

  experiment_levels <- as.character(experiment_levels %||% character(0))
  experiment_index <- if (is.null(experiment_index)) {
    NULL
  } else {
    as.integer(experiment_index)
  }
  default_experiment_index <- NA_integer_
  if (!is.null(experiment_index)) {
    idx_ok <- unique(experiment_index[!is.na(experiment_index)])
    if (length(idx_ok) == 1L) {
      default_experiment_index <- as.integer(idx_ok[[1]])
    }
  }
  covariate_order_by_experiment <- lapply(
    covariate_order_by_experiment %||% list(),
    as.integer
  )
  default_covariate_order <- as.integer(default_covariate_order %||% integer(0))
  factor_order_by_experiment <- lapply(
    factor_order_by_experiment %||% list(),
    as.integer
  )
  default_factor_order <- as.integer(default_factor_order %||% integer(0))
  factor_tokenization <- cs_foundation_factor_tokenization(factor_tokenization)
  max_factor_tokens <- neural_resolve_max_factor_tokens(max_factor_tokens)
  shared_projection_value_encoder <- cs_foundation_shared_projection_value_encoder(
    shared_projection_value_encoder
  )
  max_covariate_tokens <- neural_resolve_max_covariate_tokens(max_covariate_tokens)
  structural_level_names <- setNames(lapply(factor_names, function(factor_name) {
    c(as.character(names_list[[factor_name]][[1]] %||% character(0)), "__holdout__")
  }), factor_names)
  structural_token_info <- neural_make_default_fused_structural_info(
    names_list = NULL,
    factor_names = factor_names,
    factor_levels = vapply(structural_level_names, length, integer(1)),
    level_names_list = structural_level_names
  )

  list(
    text_dim = as.integer(dim_use),
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    factor_struct_matrix = structural_token_info$factor_struct_matrix,
    factor_struct_feature_names = structural_token_info$factor_struct_feature_names,
    level_struct_matrices = structural_token_info$level_struct_matrices,
    level_struct_feature_names = structural_token_info$level_struct_feature_names,
    factor_name_text_present = factor_name_text_present,
    level_name_text_present = level_name_text_present,
    factor_order_by_experiment = factor_order_by_experiment,
    default_factor_order = default_factor_order,
    factor_tokenization = factor_tokenization,
    max_factor_tokens = max_factor_tokens,
    covariate_name_text = covariate_name_text,
    covariate_names = as.character(covariate_names %||% character(0)),
    covariate_order_by_experiment = covariate_order_by_experiment,
    default_covariate_order = default_covariate_order,
    experiment_description_text = experiment_description_text,
    experiment_description_present = experiment_description_present,
    default_experiment_text = default_experiment_text,
    default_experiment_text_present = isTRUE(default_experiment_text_present),
    experiment_levels = experiment_levels,
    experiment_index = experiment_index,
    default_experiment_index = default_experiment_index,
    experiment_token_mode = cs_foundation_experiment_token_mode(experiment_token_mode),
    covariate_value_encoding = cs_foundation_covariate_value_encoding(covariate_value_encoding),
    shared_projection_value_encoder = shared_projection_value_encoder,
    max_covariate_tokens = max_covariate_tokens
  )
}

cs_foundation_validate_fused_text <- function(token_info,
                                              context = "Foundation model") {
  if (!identical(token_info$factor_tokenization %||% NULL, "fused")) {
    return(invisible(token_info))
  }
  if (as.integer(token_info$text_dim %||% 0L) < 1L ||
      is.null(token_info$factor_name_text) ||
      is.null(token_info$level_name_text)) {
    stop(
      sprintf(
        "%s requires token-native factor and level text embeddings under factor_tokenization='fused'.",
        context
      ),
      call. = FALSE
    )
  }
  factor_ok <- all(token_info$factor_name_text_present %||% FALSE)
  level_ok <- all(vapply(
    token_info$level_name_text_present %||% list(),
    function(x) all(as.logical(x %||% FALSE)),
    logical(1)
  ))
  if (!isTRUE(factor_ok) || !isTRUE(level_ok)) {
    stop(
      sprintf(
        "%s requires complete factor and level text coverage under factor_tokenization='fused'. Supply a compatible text_embedding_fn.",
        context
      ),
      call. = FALSE
    )
  }
  invisible(token_info)
}

cs_foundation_row_semantics <- function(W_df, exp_map, text_registry, X_mat = NULL) {
  if (is.null(text_registry)) {
    return(NULL)
  }
  dim_use <- as.integer(text_registry$dim)
  if (dim_use < 1L) {
    return(NULL)
  }

  out_blocks <- list()
  if (!is.null(text_registry$factor_embedding) && !is.null(text_registry$level_embedding)) {
    n <- nrow(W_df)
    factor_sum <- matrix(0, nrow = n, ncol = dim_use)
    level_sum <- matrix(0, nrow = n, ncol = dim_use)
    counts <- integer(n)

    for (factor_name in names(exp_map$factor_map)) {
      factor_meta <- exp_map$factor_map[[factor_name]]
      level_map <- exp_map$level_map[[factor_name]]
      factor_info <- cs_foundation_get_embedding_rows(
        emb_matrix = text_registry$factor_embedding,
        keys = factor_meta$slot_key,
        return_present = TRUE
      )
      if (!isTRUE(factor_info$present[[1L]])) {
        next
      }
      vals <- as.character(W_df[[factor_name]])
      lvl_keys <- unname(level_map[vals])
      good <- !is.na(lvl_keys)
      if (!any(good)) {
        next
      }
      good_idx <- which(good)
      level_info <- cs_foundation_get_embedding_rows(
        emb_matrix = text_registry$level_embedding,
        keys = lvl_keys[good_idx],
        return_present = TRUE
      )
      matched_idx <- good_idx[level_info$present]
      if (!length(matched_idx)) {
        next
      }
      factor_sum[matched_idx, ] <- factor_sum[matched_idx, , drop = FALSE] +
        matrix(rep(factor_info$rows[1, ], each = length(matched_idx)), nrow = length(matched_idx))
      level_sum[matched_idx, ] <- level_sum[matched_idx, , drop = FALSE] +
        level_info$rows[level_info$present, , drop = FALSE]
      counts[matched_idx] <- counts[matched_idx] + 1L
    }

    counts[counts < 1L] <- 1L
    factor_mean <- factor_sum / counts
    level_mean <- level_sum / counts
    factor_level_out <- cbind(factor_mean, level_mean)
    colnames(factor_level_out) <- c(
      paste0("semantic_factor_", seq_len(dim_use)),
      paste0("semantic_level_", seq_len(dim_use))
    )
    out_blocks[["factor_level"]] <- factor_level_out
  }

  x_feature_emb <- text_registry$x_feature_embedding %||% NULL
  if (!is.null(X_mat) && !is.null(x_feature_emb) && ncol(X_mat) > 0L && nrow(x_feature_emb) > 0L) {
    X_use <- as.matrix(X_mat)
    storage.mode(X_use) <- "double"
    x_info <- cs_foundation_get_embedding_rows(
      emb_matrix = x_feature_emb,
      keys = colnames(X_use),
      return_present = TRUE
    )
    X_weight <- X_use
    if (length(x_info$present) > 0L && any(!x_info$present)) {
      X_weight[, !x_info$present] <- 0
    }
    semantic_x <- X_weight %*% x_info$rows
    denom <- rowSums(abs(X_weight))
    ok <- denom > 0
    if (any(ok)) {
      semantic_x[ok, ] <- semantic_x[ok, , drop = FALSE] / denom[ok]
    }
    if (any(!ok)) {
      semantic_x[!ok, ] <- 0
    }
    colnames(semantic_x) <- paste0("semantic_x_", seq_len(dim_use))
    out_blocks[["x"]] <- semantic_x
  }

  if (!length(out_blocks)) {
    return(NULL)
  }
  do.call(cbind, out_blocks)
}

cs_foundation_stack_base_x <- function(experiments) {
  base_x_names <- unique(unlist(lapply(experiments, function(exp) {
    if (is.null(exp$X)) {
      character(0)
    } else {
      colnames(exp$X)
    }
  }), use.names = FALSE))
  list(
    base_x_names = base_x_names
  )
}

cs_foundation_build_group_training_data <- function(experiments, registry, control) {
  slot_names <- registry$slot_table$slot_name
  x_schema <- cs_foundation_stack_base_x(experiments)
  experiment_token_mode <- cs_foundation_experiment_token_mode(
    control$experiment_token_mode %||% "description"
  )
  factor_tokenization <- cs_foundation_factor_tokenization(
    control$factor_tokenization %||% "fused"
  )
  max_factor_tokens <- neural_resolve_max_factor_tokens(
    control$max_factor_tokens %||% NULL
  )
  covariate_value_encoding <- cs_foundation_covariate_value_encoding(
    control$covariate_value_encoding %||% "shared_projection"
  )
  shared_projection_value_encoder <- cs_foundation_shared_projection_value_encoder(
    control$shared_projection_value_encoder %||% "name_dist_moe"
  )
  text_registry <- if (isTRUE(control$add_text_semantics)) {
    cs_foundation_build_text_registry(
      experiments = experiments,
      registry = registry,
      text_embedding_fn = control$text_embedding_fn %||% NULL,
      x_feature_names = x_schema$base_x_names,
      experiment_texts = cs_foundation_collect_experiment_texts(experiments)
    )
  } else {
    NULL
  }
  experiment_levels <- vapply(experiments, `[[`, character(1), "experiment_id")
  registry_slot_maps <- cs_foundation_build_registry_slot_maps(registry)

  W_all <- vector("list", length(experiments))
  Y_all <- vector("list", length(experiments))
  pair_all <- vector("list", length(experiments))
  profile_all <- vector("list", length(experiments))
  respondent_all <- vector("list", length(experiments))
  task_all <- vector("list", length(experiments))
  candidate_group_all <- vector("list", length(experiments))
  respondent_group_all <- vector("list", length(experiments))
  X_all <- vector("list", length(experiments))
  X_present_all <- vector("list", length(experiments))
  experiment_index_all <- vector("list", length(experiments))
  task_mode_all <- vector("list", length(experiments))
  likelihood_all <- vector("list", length(experiments))
  n_outcomes_all <- vector("list", length(experiments))
  factor_order_by_experiment <- vector("list", length(experiments))
  covariate_order_by_experiment <- vector("list", length(experiments))
  context_modes <- unique(vapply(experiments, function(exp) {
    if (!identical(exp$mode, "pairwise")) {
      return(NA_character_)
    }
    as.character(exp$pairwise_context_mode %||% "stage_free")
  }, character(1)))
  context_modes <- context_modes[!is.na(context_modes)]
  supported_modes <- unique(vapply(experiments, `[[`, character(1), "mode"))
  supported_likelihoods <- unique(vapply(experiments, `[[`, character(1), "likelihood"))
  max_n_outcomes <- max(1L, vapply(experiments, function(exp) {
    as.integer(exp$n_outcomes %||% 1L)
  }, integer(1)))
  single_null_candidate_label <- "__fm_single_null__"

  for (i in seq_along(experiments)) {
    exp <- experiments[[i]]
    exp_map <- registry$experiment_maps[[exp$experiment_id]]

    pooled_W <- matrix(NA_character_, nrow = nrow(exp$W), ncol = length(slot_names))
    colnames(pooled_W) <- slot_names
    pooled_W <- as.data.frame(pooled_W, stringsAsFactors = FALSE)

    for (factor_name in exp$factor_names) {
      factor_meta <- exp_map$factor_map[[factor_name]]
      level_map <- exp_map$level_map[[factor_name]]
      pooled_W[[factor_meta$slot_name]] <- unname(level_map[as.character(exp$W[[factor_name]])])
    }
    factor_order_by_experiment[[i]] <- neural_factor_order_from_names(
      vapply(exp$factor_names, function(factor_name) {
        exp_map$factor_map[[factor_name]]$slot_name
      }, character(1)),
      slot_names
    )
    if (identical(factor_tokenization, "fused")) {
      neural_validate_factor_token_budget(
        n_factors = length(factor_order_by_experiment[[i]]),
        max_factor_tokens = max_factor_tokens,
        context = sprintf("Foundation experiment '%s'", exp$experiment_id)
      )
    }

    covariate_block <- cs_foundation_align_covariate_block(
      schema_names = x_schema$base_x_names,
      X_mat = exp$X,
      n_rows = nrow(exp$W)
    )
    covariate_order_by_experiment[[i]] <- neural_covariate_order_from_names(
      colnames(exp$X %||% matrix(numeric(0), nrow = 0L, ncol = 0L)),
      x_schema$base_x_names
    )
    if (identical(covariate_value_encoding, "shared_projection")) {
      neural_validate_covariate_token_budget(
        n_covariates = length(covariate_order_by_experiment[[i]]),
        max_covariate_tokens = control$max_covariate_tokens %||% NULL,
        context = sprintf("Foundation experiment '%s'", exp$experiment_id)
      )
    }

    n_rows <- nrow(exp$W)
    task_mode_label <- as.character(exp$mode %||% "single")
    likelihood_label <- as.character(exp$likelihood %||% "bernoulli")
    n_outcomes_label <- as.integer(exp$n_outcomes %||% 1L)

    if (identical(task_mode_label, "pairwise")) {
      X_all[[i]] <- covariate_block$values
      X_present_all[[i]] <- covariate_block$present
      W_all[[i]] <- pooled_W
      Y_all[[i]] <- exp$Y
      pair_all[[i]] <- if (!is.null(exp$pair_id)) {
        paste(exp$experiment_id, exp$pair_id, sep = "::")
      } else {
        NULL
      }
      profile_all[[i]] <- exp$profile_order %||% NULL
      respondent_all[[i]] <- if (!is.null(exp$respondent_id)) {
        paste(exp$experiment_id, exp$respondent_id, sep = "::")
      } else {
        NULL
      }
      task_all[[i]] <- if (!is.null(exp$respondent_task_id)) {
        paste(exp$experiment_id, exp$respondent_task_id, sep = "::")
      } else {
        NULL
      }
      candidate_group_all[[i]] <- exp$competing_group_variable_candidate %||% NULL
      respondent_group_all[[i]] <- exp$competing_group_variable_respondent %||% NULL
      experiment_index_all[[i]] <- rep.int(as.integer(i - 1L), n_rows)
      task_mode_all[[i]] <- rep.int(task_mode_label, n_rows)
      likelihood_all[[i]] <- rep.int(likelihood_label, n_rows)
      n_outcomes_all[[i]] <- rep.int(n_outcomes_label, n_rows)
    } else {
      null_W <- pooled_W[rep.int(1L, n_rows), , drop = FALSE]
      for (slot_name in slot_names) {
        null_W[[slot_name]] <- NA_character_
      }
      W_all[[i]] <- rbind(pooled_W, null_W)
      Y_all[[i]] <- c(exp$Y, exp$Y)
      pair_ids <- paste(exp$experiment_id, sprintf("single_%06d", seq_len(n_rows)), sep = "::")
      pair_all[[i]] <- c(pair_ids, pair_ids)
      profile_all[[i]] <- c(rep.int(1L, n_rows), rep.int(2L, n_rows))
      respondent_all[[i]] <- if (!is.null(exp$respondent_id)) {
        resp_ids <- paste(exp$experiment_id, exp$respondent_id, sep = "::")
        c(resp_ids, resp_ids)
      } else {
        NULL
      }
      task_all[[i]] <- if (!is.null(exp$respondent_task_id)) {
        task_ids <- paste(exp$experiment_id, exp$respondent_task_id, sep = "::")
        c(task_ids, task_ids)
      } else {
        NULL
      }
      candidate_group_all[[i]] <- c(
        rep(NA_character_, n_rows),
        rep(single_null_candidate_label, n_rows)
      )
      respondent_group_all[[i]] <- rep(NA_character_, 2L * n_rows)
      if (length(x_schema$base_x_names) > 0L) {
        X_all[[i]] <- rbind(
          covariate_block$values,
          matrix(
            0,
            nrow = n_rows,
            ncol = ncol(covariate_block$values),
            dimnames = list(NULL, colnames(covariate_block$values))
          )
        )
        X_present_all[[i]] <- rbind(
          covariate_block$present,
          matrix(
            0,
            nrow = n_rows,
            ncol = ncol(covariate_block$present),
            dimnames = list(NULL, colnames(covariate_block$present))
          )
        )
      } else {
        X_all[[i]] <- NULL
        X_present_all[[i]] <- NULL
      }
      experiment_index_all[[i]] <- rep.int(as.integer(i - 1L), 2L * n_rows)
      task_mode_all[[i]] <- rep.int(task_mode_label, 2L * n_rows)
      likelihood_all[[i]] <- rep.int(likelihood_label, 2L * n_rows)
      n_outcomes_all[[i]] <- rep.int(n_outcomes_label, 2L * n_rows)
    }
  }

  token_info <- cs_foundation_build_token_info(
    names_list = registry$pooled_names_list,
    factor_map = registry_slot_maps$factor_map,
    level_map = registry_slot_maps$level_map,
    text_registry = text_registry,
    factor_order_by_experiment = factor_order_by_experiment,
    covariate_names = x_schema$base_x_names,
    covariate_order_by_experiment = covariate_order_by_experiment,
    experiment_levels = experiment_levels,
    experiment_index = unlist(experiment_index_all, use.names = FALSE),
    experiment_token_mode = experiment_token_mode,
    factor_tokenization = factor_tokenization,
    max_factor_tokens = max_factor_tokens,
    covariate_value_encoding = covariate_value_encoding,
    shared_projection_value_encoder = shared_projection_value_encoder,
    max_covariate_tokens = control$max_covariate_tokens %||% NULL
  )
  cs_foundation_validate_fused_text(
    token_info = token_info,
    context = "Foundation pooled training"
  )
  token_info$foundation_universal_training <- list(
    enabled = TRUE,
    task_mode_by_row = unlist(task_mode_all, use.names = FALSE),
    likelihood_by_row = unlist(likelihood_all, use.names = FALSE),
    n_outcomes_by_row = as.integer(unlist(n_outcomes_all, use.names = FALSE)),
    global_out_dim = as.integer(max_n_outcomes),
    supported_modes = supported_modes,
    supported_likelihoods = supported_likelihoods,
    supported_pairwise_context_modes = if (length(context_modes) > 0L) context_modes else "stage_free",
    synthetic_single_null_candidate_label = single_null_candidate_label
  )

  list(
    Y = unlist(Y_all, use.names = FALSE),
    W = do.call(rbind, W_all),
    X = if (length(x_schema$base_x_names) > 0L) do.call(rbind, X_all) else NULL,
    X_present = if (length(x_schema$base_x_names) > 0L) do.call(rbind, X_present_all) else NULL,
    pair_id = if (all(vapply(pair_all, is.null, logical(1)))) NULL else unlist(pair_all, use.names = FALSE),
    profile_order = if (all(vapply(profile_all, is.null, logical(1)))) NULL else unlist(profile_all, use.names = FALSE),
    respondent_id = if (all(vapply(respondent_all, is.null, logical(1)))) NULL else unlist(respondent_all, use.names = FALSE),
    respondent_task_id = if (all(vapply(task_all, is.null, logical(1)))) NULL else unlist(task_all, use.names = FALSE),
    competing_group_variable_candidate = if (all(vapply(candidate_group_all, is.null, logical(1)))) {
      NULL
    } else {
      unlist(lapply(seq_along(candidate_group_all), function(i) {
        vals <- candidate_group_all[[i]]
        if (is.null(vals)) {
          rep(NA_character_, nrow(W_all[[i]]))
        } else {
          as.character(vals)
        }
      }), use.names = FALSE)
    },
    competing_group_variable_respondent = if (all(vapply(respondent_group_all, is.null, logical(1)))) {
      NULL
    } else {
      unlist(lapply(seq_along(respondent_group_all), function(i) {
        vals <- respondent_group_all[[i]]
        if (is.null(vals)) {
          rep(NA_character_, nrow(W_all[[i]]))
        } else {
          as.character(vals)
        }
      }), use.names = FALSE)
    },
    pairwise_context_mode = if (length(context_modes) > 0L &&
                                any(context_modes == "stage_aware")) {
      "stage_aware"
    } else {
      "stage_free"
    },
    names_list = registry$pooled_names_list,
    factor_levels = vapply(registry$pooled_names_list, function(x) length(x[[1]]), integer(1)),
    x_feature_names = x_schema$base_x_names,
    x_schema = list(
      base_x_names = x_schema$base_x_names,
      experiment_indicator_names = character(0),
      semantic_feature_names = character(0),
      experiment_token_levels = experiment_levels
    ),
    text_registry = text_registry,
    token_info = token_info,
    token_control = list(
      experiment_token_mode = experiment_token_mode,
      factor_tokenization = factor_tokenization,
      max_factor_tokens = max_factor_tokens,
      covariate_value_encoding = covariate_value_encoding,
      shared_projection_value_encoder = shared_projection_value_encoder,
      max_covariate_tokens = neural_resolve_max_covariate_tokens(
        control$max_covariate_tokens %||% NULL
      )
    ),
    supported_modes = supported_modes,
    supported_likelihoods = supported_likelihoods,
    supported_pairwise_context_modes = if (length(context_modes) > 0L) context_modes else "stage_free",
    max_n_outcomes = as.integer(max_n_outcomes)
  )
}

cs_foundation_prepare_group_fit <- function(group,
                                            conda_env = "strategize_env",
                                            conda_env_required = TRUE) {
  if (!is.null(group$fit$neural_model_info$params)) {
    return(group)
  }
  if (!"jnp" %in% ls(envir = strenv) || !"np" %in% ls(envir = strenv)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
  }
  theta_jnp <- strenv$jnp$array(as.numeric(group$fit$theta_mean))$astype(strenv$dtj)
  group$fit$neural_model_info$params <- neural_params_from_theta(theta_jnp, group$fit$neural_model_info)
  group
}

cs_foundation_locscale_tau_name <- function(name) {
  if (grepl("^W_(q|k|v|o)_l[0-9]+$", name) || grepl("^W_ff(1|2)_l[0-9]+$", name)) {
    layer_id <- sub("^.*_l([0-9]+)$", "\\1", name)
    return(paste0("tau_w_", layer_id))
  }
  if (name %in% c("W_q_cross", "W_k_cross", "W_v_cross", "W_o_cross")) {
    return("tau_cross_attn")
  }
  if (identical(name, "W_out")) {
    return("tau_w_out")
  }
  if (identical(name, "b_out")) {
    return("tau_b")
  }
  if (identical(name, "M_cross_raw")) {
    return("tau_cross")
  }
  NULL
}

cs_foundation_add_init_value <- function(init_values, name, value) {
  if (is.null(value)) {
    return(init_values)
  }
  tau_name <- cs_foundation_locscale_tau_name(name)
  init_values[[name]] <- value
  if (!is.null(tau_name)) {
    init_values[[tau_name]] <- 1
    init_values[[paste0(name, "_z")]] <- value
  }
  init_values
}

cs_foundation_build_local_factor_map <- function(experiment) {
  factor_map <- vector("list", length(experiment$factor_names))
  names(factor_map) <- experiment$factor_names
  level_map <- setNames(vector("list", length(experiment$factor_names)), experiment$factor_names)
  for (j in seq_along(experiment$factor_names)) {
    factor_name <- experiment$factor_names[[j]]
    factor_key <- cs_foundation_make_factor_key(
      experiment_id = experiment$experiment_id,
      factor_name = factor_name,
      canonical_factor_id = experiment$canonical_factor_id[[factor_name]]
    )
    factor_map[[factor_name]] <- list(slot_key = factor_key)
    levels_here <- experiment$names_list[[factor_name]][[1]]
    level_key_map <- character(length(levels_here))
    names(level_key_map) <- levels_here
    level_candidates <- setNames(vector("list", length(levels_here)), levels_here)
    for (lvl in levels_here) {
      level_key_map[[lvl]] <- cs_foundation_make_level_key(
        experiment_id = experiment$experiment_id,
        factor_key = factor_key,
        factor_name = factor_name,
        level_name = lvl,
        canonical_level_id = experiment$canonical_level_id[[factor_name]][[lvl]]
      )
      level_candidates[[lvl]] <- cs_foundation_candidate_level_keys(
        experiment_id = experiment$experiment_id,
        factor_key = factor_key,
        factor_name = factor_name,
        level_name = lvl,
        canonical_level_id = experiment$canonical_level_id[[factor_name]][[lvl]]
      )
    }
    level_map[[factor_name]] <- level_key_map
    if (is.null(factor_map[[factor_name]]$level_candidates)) {
      factor_map[[factor_name]]$level_candidates <- level_candidates
    }
  }
  list(factor_map = factor_map, level_map = level_map)
}

cs_foundation_build_adaptation_x <- function(group,
                                             experiment,
                                             exp_map,
                                             adaptation_control) {
  adaptation_control <- cs_foundation_resolve_adaptation_control(
    group = group,
    adaptation_control = adaptation_control
  )
  x_schema <- group$x_schema %||% list(
    base_x_names = character(0),
    experiment_indicator_names = character(0),
    semantic_feature_names = character(0),
    experiment_token_levels = character(0)
  )
  base_names <- x_schema$base_x_names %||% character(0)
  base_block <- cs_foundation_align_covariate_block(
    schema_names = base_names,
    X_mat = experiment$X,
    n_rows = nrow(experiment$W)
  )

  text_registry <- NULL
  use_text <- isTRUE(adaptation_control$use_text_semantics) &&
    !is.null(group$text_registry)
  extra_names <- if (isTRUE(adaptation_control$allow_extra_covariates) && !is.null(experiment$X)) {
    setdiff(colnames(experiment$X), base_names)
  } else {
    character(0)
  }
  local_covariate_names <- c(base_names, extra_names)
  if (use_text) {
    if (!is.null(adaptation_control$text_embedding_fn)) {
      text_registry <- cs_foundation_build_text_registry(
        experiments = list(experiment),
        registry = list(
          slot_table = data.frame(
            slot_name = names(exp_map$factor_map),
            slot_key = vapply(exp_map$factor_map, function(x) x$slot_key, character(1)),
            display_label = names(exp_map$factor_map),
            stringsAsFactors = FALSE
          ),
          slot_level_keys = lapply(exp_map$level_map, unname),
          slot_level_labels = lapply(exp_map$level_map, names)
        ),
        text_embedding_fn = adaptation_control$text_embedding_fn,
        x_feature_names = local_covariate_names,
        experiment_texts = cs_foundation_collect_experiment_texts(list(experiment))
      )
      if (!identical(as.integer(text_registry$dim %||% 0L), as.integer(group$text_registry$dim %||% 0L))) {
        stop(
          "Adaptation text embeddings must match the pooled foundation text embedding width.",
          call. = FALSE
        )
      }
    } else {
      text_registry <- group$text_registry
    }
  }

  extra_x <- matrix(0, nrow = nrow(experiment$W), ncol = length(extra_names))
  colnames(extra_x) <- extra_names
  extra_present <- matrix(0, nrow = nrow(experiment$W), ncol = length(extra_names))
  colnames(extra_present) <- extra_names
  if (length(extra_names) > 0L) {
    extra_x[, extra_names] <- experiment$X[, extra_names, drop = FALSE]
    extra_present[, extra_names] <- 1
  }

  out <- cbind(base_block$values, extra_x)
  out_present <- cbind(base_block$present, extra_present)

  experiment_levels <- x_schema$experiment_token_levels %||% character(0)
  experiment_index <- rep.int(NA_integer_, nrow(experiment$W))
  match_idx <- match(experiment$experiment_id, experiment_levels)
  if (!is.na(match_idx)) {
    experiment_index[] <- as.integer(match_idx - 1L)
  }
  token_info <- cs_foundation_build_token_info(
    names_list = experiment$names_list,
    factor_map = exp_map$factor_map,
    level_map = exp_map$level_map,
    text_registry = text_registry,
    default_factor_order = neural_factor_order_from_names(
      experiment$factor_names,
      experiment$factor_names
    ),
    covariate_names = colnames(out),
    default_covariate_order = neural_covariate_order_from_names(
      colnames(experiment$X %||% matrix(numeric(0), nrow = 0L, ncol = 0L)),
      colnames(out)
    ),
    experiment_levels = experiment_levels,
    experiment_index = experiment_index,
    experiment_token_mode = adaptation_control$experiment_token_mode,
    factor_tokenization = adaptation_control$factor_tokenization,
    max_factor_tokens = adaptation_control$max_factor_tokens %||% NULL,
    covariate_value_encoding = adaptation_control$covariate_value_encoding,
    shared_projection_value_encoder = adaptation_control$shared_projection_value_encoder,
    max_covariate_tokens = adaptation_control$max_covariate_tokens %||% NULL,
    default_experiment_key = experiment$experiment_id
  )
  if (identical(adaptation_control$factor_tokenization, "fused")) {
    neural_validate_factor_token_budget(
      n_factors = length(experiment$factor_names),
      max_factor_tokens = adaptation_control$max_factor_tokens %||% NULL,
      context = sprintf("Adaptation experiment '%s'", experiment$experiment_id)
    )
  }
  cs_foundation_validate_fused_text(
    token_info = token_info,
    context = sprintf("Adaptation experiment '%s'", experiment$experiment_id)
  )
  if (identical(adaptation_control$covariate_value_encoding, "shared_projection")) {
    neural_validate_covariate_token_budget(
      n_covariates = length(token_info$default_covariate_order %||% integer(0)),
      max_covariate_tokens = adaptation_control$max_covariate_tokens %||% NULL,
      context = sprintf("Adaptation experiment '%s'", experiment$experiment_id)
    )
  }

  attr(out, "resp_cov_present") <- out_present
  attr(out, "token_info") <- token_info
  out
}

cs_foundation_build_init_site_values <- function(group,
                                                 experiment,
                                                 local_x_feature_names,
                                                 strict_schema_match = FALSE,
                                                 conda_env = "strategize_env",
                                                 conda_env_required = TRUE) {
  group_prepped <- cs_foundation_prepare_group_fit(
    group = group,
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )
  params <- group_prepped$fit$neural_model_info$params
  registry <- group$schema_registry
  exp_map <- cs_foundation_build_local_factor_map(experiment)
  slot_table <- registry$slot_table %||% data.frame(
    slot_name = character(0),
    slot_key = character(0),
    stringsAsFactors = FALSE
  )
  slot_lookup <- setNames(
    as.character(slot_table$slot_name %||% character(0)),
    as.character(slot_table$slot_key %||% character(0))
  )
  lookup_slot <- function(slot_key) {
    if (is.null(slot_key) || is.na(slot_key) || !nzchar(slot_key)) {
      return(NULL)
    }
    slot_name <- unname(slot_lookup[slot_key])
    if (length(slot_name) != 1L || is.na(slot_name) || !nzchar(slot_name)) {
      return(NULL)
    }
    slot_name
  }
  if (isTRUE(strict_schema_match)) {
    matched_slot <- vapply(experiment$factor_names, function(factor_name) {
      slot_key <- exp_map$factor_map[[factor_name]]$slot_key %||% NA_character_
      lookup_slot(slot_key) %||% NA_character_
    }, character(1))
    unmatched <- experiment$factor_names[is.na(matched_slot) | !nzchar(matched_slot)]
    if (length(unmatched) > 0L) {
      stop(
        sprintf(
          "No shared slot found for factor(s) during adaptation: %s",
          paste(unmatched, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }
  init_values <- list()

  direct_names <- setdiff(names(params), c(
    "W_out", "b_out", "sigma", "E_segment", "M_cross"
  ))
  for (name in direct_names) {
    init_values <- cs_foundation_add_init_value(
      init_values = init_values,
      name = name,
      value = cs2step_neural_to_r_array(params[[name]])
    )
  }

  if (!is.null(params$E_segment)) {
    seg_mat <- cs2step_neural_to_r_array(params$E_segment)
    if (is.matrix(seg_mat) && nrow(seg_mat) >= 2L) {
      init_values[["E_segment_delta"]] <- as.numeric(seg_mat[2, ] - seg_mat[1, ])
    }
  }

  if (!is.null(params$M_cross)) {
    init_values <- cs_foundation_add_init_value(
      init_values = init_values,
      name = "M_cross_raw",
      value = cs2step_neural_to_r_array(params$M_cross)
    )
  }

  output_dim <- if (identical(experiment$likelihood, "categorical")) {
    as.integer(experiment$n_outcomes)
  } else {
    1L
  }
  if (!is.null(params$W_out)) {
    w_out_src <- cs2step_neural_to_r_array(params$W_out)
    if (is.matrix(w_out_src) && ncol(w_out_src) >= 1L) {
      init_values <- cs_foundation_add_init_value(
        init_values = init_values,
        name = "W_out",
        value = w_out_src[, seq_len(min(output_dim, ncol(w_out_src))), drop = FALSE]
      )
    }
  }
  if (!is.null(params$b_out)) {
    b_out_src <- as.numeric(cs2step_neural_to_r_array(params$b_out))
    if (length(b_out_src) >= 1L) {
      init_values <- cs_foundation_add_init_value(
        init_values = init_values,
        name = "b_out",
        value = b_out_src[seq_len(min(output_dim, length(b_out_src)))]
      )
    }
  }
  if (identical(experiment$likelihood, "normal") && !is.null(params$sigma)) {
    init_values <- cs_foundation_add_init_value(
      init_values = init_values,
      name = "sigma",
      value = as.numeric(cs2step_neural_to_r_array(params$sigma))
    )
  }

  init_values
}

cs_foundation_build_predictor <- function(fit,
                                          mode,
                                          names_list,
                                          factor_levels,
                                          metadata = NULL) {
  structure(
    list(
      model_type = "neural",
      mode = mode,
      encoder = list(
        factor_names = names(names_list),
        names_list = names_list,
        factor_levels = factor_levels,
        unknown_policy = "holdout"
      ),
      fit = fit,
      metadata = modifyList(
        list(
          timestamp = Sys.time(),
          cache_id = sprintf("foundation_adapt_%d", as.integer(stats::runif(1, 1, 1e9)))
        ),
        metadata %||% list()
      )
    ),
    class = "strategic_predictor"
  )
}

cs_foundation_pack_group <- function(group) {
  fit <- group$fit
  group$fit <- list(
    my_model = NULL,
    predict_pair = NULL,
    predict_single = NULL,
    theta_mean = if (!is.null(fit$theta_mean)) as.numeric(fit$theta_mean) else NULL,
    theta_var = if (!is.null(fit$theta_var)) as.numeric(fit$theta_var) else NULL,
    neural_model_info = cs2step_neural_pack_model_info(fit$neural_model_info, drop_params = TRUE),
    fit_metrics = fit$fit_metrics %||% fit$neural_model_info$fit_metrics %||% NULL
  )
  group
}

cs_foundation_unpack_group <- function(group,
                                       conda_env = "strategize_env",
                                       conda_env_required = TRUE,
                                       preload_params = FALSE) {
  bundle <- list(
    model_type = "neural",
    mode = group$mode,
    encoder = group$encoder,
    fit = list(
      theta_mean = group$fit$theta_mean,
      theta_var = group$fit$theta_var,
      neural_model_info = group$fit$neural_model_info,
      fit_metrics = group$fit$fit_metrics
    ),
    metadata = list(
      conda_env = conda_env,
      conda_env_required = conda_env_required
    )
  )
  predictor <- cs2step_unpack_predictor(
    bundle = bundle,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    preload_params = preload_params
  )
  group$fit <- predictor$fit
  group
}

#' Pooled conjoint foundation-model training moved to preference.fm
#'
#' @param experiments List of experiment specifications. Each element must be a
#'   named list with at least \code{experiment_id}, \code{Y}, and \code{W}.
#' @param foundation_control Optional list controlling pooled training. Supported
#'   keys are \code{add_experiment_indicators}, \code{add_text_semantics},
#'   \code{text_embedding_fn}, and \code{neural_mcmc_control}.
#' @param conda_env Conda env name for the neural backend.
#' @param conda_env_required Require the conda env to exist.
#' @param cache_path Optional path to a cached foundation bundle.
#' @param cache_overwrite Logical; overwrite any existing cache at \code{cache_path}.
#' @param cache_compress Retained for API compatibility.
#' @return An object of class \code{conjoint_foundation_model}.
#'
#' @details
#' This exported name is retained for migration guidance. Pooled training is
#' now owned by \pkg{preference.fm}. Use
#' \code{preference.fm::fit_conjoint_foundation_model()} to train pooled
#' models and \code{preference.fm::adapt_conjoint_foundation_model()} for
#' semantic adaptation. Use \pkg{strategize} for embedding extraction and
#' prediction from loaded artifacts.
#'
#' Each element of \code{experiments} is a per-study specification with the
#' following contract.
#'
#' Required fields:
#' \describe{
#'   \item{\code{experiment_id}}{A unique study identifier.}
#'   \item{\code{Y}}{Outcome vector with \code{length(Y) == nrow(W)}.}
#'   \item{\code{W}}{A data frame or matrix of factor columns.}
#' }
#'
#' Optional fields:
#' \describe{
#'   \item{\code{mode}}{\code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#'   Defaults to \code{"pairwise"} when \code{pair_id} is supplied and
#'   \code{"single"} otherwise.}
#'   \item{\code{pair_id}}{Required for pairwise studies. Each pair id must
#'   appear exactly twice.}
#'   \item{\code{profile_order}}{Optional within-pair ordering, typically
#'   \code{1}/\code{2}.}
#'   \item{\code{X}}{Optional numeric covariates aligned row-wise to
#'   \code{W}. Non-numeric columns are rejected. Raw numeric values remain
#'   unchanged at the API boundary. In the default FM covariate path
#'   (\code{covariate_value_encoding = "shared_projection"} with
#'   \code{shared_projection_value_encoder = "name_dist_moe"}), each present
#'   covariate is emitted as one fused covariate/value token. The fused token can
#'   include projected \code{X}-name text embeddings and combines the
#'   standardized numeric value with local study-specific distribution summaries
#'   and covariate-name semantics.
#'   Absent covariates are masked structurally rather than represented by a
#'   separate learned presence embedding.}
#'   \item{\code{respondent_id}, \code{respondent_task_id}}{Optional row-aligned
#'   respondent/task identifiers used when available for clustering and
#'   evaluation logic.}
#'   \item{\code{competing_group_variable_candidate},
#'   \code{competing_group_variable_respondent}}{Optional row-aligned
#'   competition-group labels for pairwise studies. When both are supplied and
#'   the candidate labels span both same-group and cross-group pairs, the pooled
#'   FM trains in a stage-aware pairwise mode. Otherwise the pairwise FM remains
#'   stage-free.}
#'   \item{\code{likelihood}}{\code{"auto"}, \code{"bernoulli"},
#'   \code{"categorical"}, or \code{"normal"}.}
#'   \item{\code{n_outcomes}}{Required only when forcing a categorical
#'   likelihood. In v1 it must match the number of observed classes in the
#'   study. Categorical outcomes are internally normalized to zero-based class
#'   ids before neural fitting.}
#'   \item{\code{names_list}, \code{p_list}}{Optional factor-level metadata used
#'   to define the local level universe for each factor.}
#'   \item{\code{canonical_factor_id}}{Optional named vector or list that forces
#'   cross-study sharing of factor identities when explicitly provided.}
#'   \item{\code{canonical_level_id}}{Optional named list that forces
#'   cross-study sharing of level identities when explicitly provided.}
#' }
#'
#' Pooled training groups experiments by backend-compatible task mode. Pairwise
#' context mode is a supported capability of a pairwise group rather than a
#' split key: stage-free and stage-aware pairwise studies can share one group.
#' If any pooled pairwise study requires stage-aware context, the shared
#' pairwise group is promoted to stage-aware-capable and records all supported
#' pairwise context modes. The returned object may therefore hold multiple
#' internal foundation groups when the input studies mix pairwise and single
#' designs.
#'
#' Schema sharing rules are conservative:
#' \itemize{
#'   \item explicit canonical ids force sharing;
#'   \item absent factors in a pooled schema are routed to holdout rows during
#'   training;
#'   \item raw text equality alone does not merge schema elements.
#' }
#'
#' The \code{foundation_control} list supports:
#' \describe{
#'   \item{\code{add_experiment_indicators}}{Whether to append experiment
#'   indicators. This field is retained for compatibility, but the foundation
#'   neural path now always represents experiment identity as an explicit
#'   experiment token rather than one-hot covariates.}
#'   \item{\code{add_text_semantics}}{Whether to add token-native text
#'   semantics when \code{text_embedding_fn} is supplied. This covers projected
#'   factor-name embeddings, level-label embeddings, and pooled \code{X}-name
#'   embeddings inside the emitted tokens. Default \code{TRUE}.}
#'   \item{\code{text_embedding_fn}}{Optional function that maps character input
#'   to numeric embeddings. It may accept a character vector and return a matrix
#'   with matching rows, or accept one string at a time and return a fixed-width
#'   numeric vector. These embeddings are projected into FM factor-name,
#'   factor-level, and pooled \code{X}-name token components. Canonical ids
#'   still govern cross-study pooling and schema sharing.}
#'   \item{\code{experiment_token_mode}}{Experiment token mode for FM training.
#'   \code{"description"} uses experiment-description text, \code{"hybrid"}
#'   combines description text and learned pooled experiment ids, and
#'   \code{"legacy_id"} uses only learned pooled experiment ids. Default
#'   \code{"description"}.}
#'   \item{\code{factor_tokenization}}{FM factor tokenizer. The required
#'   \code{"fused"} mode emits one fused factor/value token per candidate
#'   attribute. Factor and level information is fused by a two-layer SwiGLU
#'   MLP.}
#'   \item{\code{max_factor_tokens}}{Total factor-token budget used by the FM
#'   fused factor encoder. The default \code{256L} supports up to 256 factor
#'   attributes per profile.}
#'   \item{\code{covariate_value_encoding}}{FM covariate encoder family.
#'   The required \code{"shared_projection"} mode emits one fused covariate/value
#'   token per respondent covariate.}
#'   \item{\code{shared_projection_value_encoder}}{Value-token generator used
#'   inside \code{"shared_projection"}. The default \code{"name_dist_moe"}
#'   conditions each covariate value token on covariate-name semantics and
#'   local distribution summaries. \code{"legacy_scalar"} falls back to the
#'   older standardized scalar projection.}
#'   \item{\code{max_covariate_tokens}}{Total covariate-token budget used by
#'   the FM fused covariate encoder. The default \code{512L} supports up to 512
#'   covariates per row.}
#'   \item{\code{neural_mcmc_control}}{Optional list passed to the existing
#'   neural outcome backend. Defaults to a pooled SVI configuration using
#'   output-only uncertainty.}
#' }
#'
#' @examples
#' \dontrun{
#' library(strategize)
#'
#' build_backend(conda_env = "strategize_env")
#'
#' text_embedding_fn <- function(x) {
#'   x <- as.character(x)
#'   cbind(nchar = nchar(x), has_space = as.numeric(grepl(" ", x)))
#' }
#'
#' study_a <- list(
#'   experiment_id = "study_a",
#'   experiment_description = "Pairwise policy-message study",
#'   Y = c(1, 0, 0, 1),
#'   W = data.frame(
#'     price = c("Low", "Low", "High", "High"),
#'     message = c("Jobs", "Taxes", "Jobs", "Taxes")
#'   ),
#'   pair_id = c(1, 1, 2, 2),
#'   profile_order = c(1, 2, 1, 2),
#'   canonical_factor_id = c(price = "price", message = "message")
#' )
#'
#' study_b <- list(
#'   experiment_id = "study_b",
#'   experiment_description = "Pairwise messenger study",
#'   Y = c(0, 1, 1, 0),
#'   W = data.frame(
#'     price = c("Low", "High", "Low", "High"),
#'     message = c("Jobs", "Jobs", "Taxes", "Taxes"),
#'     messenger = c("Local", "Local", "National", "National")
#'   ),
#'   pair_id = c(1, 1, 2, 2),
#'   profile_order = c(1, 2, 1, 2),
#'   canonical_factor_id = c(
#'     price = "price",
#'     message = "message",
#'     messenger = "messenger"
#'   )
#' )
#'
#' foundation_fit <- preference.fm::fit_conjoint_foundation_model(
#'   experiments = list(study_a, study_b),
#'   foundation_control = list(
#'     text_embedding_fn = text_embedding_fn,
#'     neural_mcmc_control = list(
#'       ModelDims = 32L,
#'       ModelDepth = 1L,
#'       subsample_method = "batch_vi",
#'       uncertainty_scope = "output",
#'       svi_steps = 100L
#'     )
#'   )
#' )
#' }
#'
#' @seealso \code{\link{adapt_conjoint_foundation_model}()},
#'   \code{\link{save_conjoint_foundation_bundle}()},
#'   \code{\link{load_conjoint_foundation_bundle}()},
#'   \code{\link{build_backend}()}
#' @export
fit_conjoint_foundation_model <- function(experiments,
                                          foundation_control = NULL,
                                          conda_env = "strategize_env",
                                          conda_env_required = TRUE,
                                          cache_path = NULL,
                                          cache_overwrite = FALSE,
                                          cache_compress = TRUE) {
  stop(
    "Pooled conjoint foundation-model training has moved to preference.fm.\n",
    "Use preference.fm::fit_conjoint_foundation_model() to train pooled models, ",
    "preference.fm::adapt_conjoint_foundation_model() for semantic adaptation, ",
    "preference.fm::load_conjoint_foundation_bundle() to load trained checkpoints, ",
    "and strategize::extract_embeddings() for runtime embedding extraction.",
    call. = FALSE
  )
}

cs_foundation_match_group <- function(foundation_model,
                                      mode,
                                      likelihood,
                                      n_outcomes,
                                      pairwise_context_mode = NULL) {
  universal_key <- cs_foundation_universal_group_key()
  universal_group <- foundation_model$groups[[universal_key]] %||% NULL
  if (!is.null(universal_group)) {
    supported_modes <- universal_group$supported_modes %||% universal_group$mode %||% character(0)
    supported_likelihoods <- universal_group$supported_likelihoods %||% universal_group$likelihood %||% character(0)
    supported_pairwise_context_modes <- universal_group$supported_pairwise_context_modes %||%
      universal_group$pairwise_context_mode %||% "stage_free"
    if (!mode %in% supported_modes) {
      stop(
        sprintf(
          "Universal foundation group does not support mode='%s'. Supported modes: %s",
          mode,
          paste(supported_modes, collapse = ", ")
        ),
        call. = FALSE
      )
    }
    if (!likelihood %in% supported_likelihoods) {
      stop(
        sprintf(
          "Universal foundation group does not support likelihood='%s'. Supported likelihoods: %s",
          likelihood,
          paste(supported_likelihoods, collapse = ", ")
        ),
        call. = FALSE
      )
    }
    if (identical(likelihood, "categorical") &&
        as.integer(n_outcomes) > as.integer(universal_group$n_outcomes %||% 1L)) {
      stop(
        sprintf(
          "Universal foundation group supports at most %d categorical outcomes, but target requested %d.",
          as.integer(universal_group$n_outcomes %||% 1L),
          as.integer(n_outcomes)
        ),
        call. = FALSE
      )
    }
    if (identical(mode, "pairwise")) {
      context_use <- as.character(pairwise_context_mode %||% "stage_free")
      if (!context_use %in% supported_pairwise_context_modes) {
        stop(
          sprintf(
            "Universal foundation group does not support pairwise_context_mode='%s'. Supported values: %s",
            context_use,
            paste(supported_pairwise_context_modes, collapse = ", ")
          ),
          call. = FALSE
        )
      }
    }
    return(universal_group)
  }
  key <- cs_foundation_group_key(
    mode,
    likelihood,
    n_outcomes,
    pairwise_context_mode = pairwise_context_mode %||% NULL
  )
  group <- foundation_model$groups[[key]] %||% NULL
  if (is.null(group) && identical(mode, "pairwise") &&
      identical(pairwise_context_mode %||% "stage_free", "stage_free")) {
    legacy_key <- paste(mode, likelihood, as.integer(n_outcomes), sep = "::")
    group <- foundation_model$groups[[legacy_key]] %||% NULL
  }
  if (is.null(group)) {
    stop(
      sprintf(
        "No compatible foundation group found for mode='%s', pairwise_context_mode='%s', likelihood='%s', n_outcomes=%d.",
        mode,
        as.character(pairwise_context_mode %||% if (identical(mode, "pairwise")) "stage_free" else NA_character_),
        likelihood,
        as.integer(n_outcomes)
      ),
      call. = FALSE
    )
  }
  group
}

#' Adapt a pooled conjoint foundation model to a single study
#'
#' @param foundation_model A fitted \code{conjoint_foundation_model}.
#' @param Y Outcome vector.
#' @param W Factor matrix/data.frame.
#' @param X Optional numeric covariates.
#' @param mode \code{"auto"}, \code{"pairwise"}, or \code{"single"}.
#' @param pair_id Optional pair identifiers.
#' @param profile_order Optional within-pair ordering.
#' @param experiment_id Optional experiment identifier for the adaptation study.
#' @param experiment_description Optional text description for the adaptation
#'   study, used when text-aware experiment tokens are enabled.
#' @param names_list Optional factor level names.
#' @param p_list Optional \code{p_list}.
#' @param respondent_id Optional respondent identifiers.
#' @param respondent_task_id Optional respondent-task identifiers.
#' @param competing_group_variable_candidate Optional candidate competition-group
#'   labels for pairwise FM adaptation.
#' @param competing_group_variable_respondent Optional respondent
#'   competition-group labels for pairwise FM adaptation.
#' @param likelihood Optional likelihood override.
#' @param n_outcomes Optional categorical outcome count.
#' @param canonical_factor_id Optional factor-level sharing ids.
#' @param canonical_level_id Optional level-level sharing ids.
#' @param neural_mcmc_control Optional list passed to the Bayesian neural backend.
#' @param foundation_adaptation_control Optional list controlling adaptation.
#'   Supported keys are \code{strict_schema_match}, \code{allow_extra_covariates},
#'   \code{use_text_semantics}, \code{text_embedding_fn},
#'   \code{experiment_token_mode}, \code{factor_tokenization},
#'   \code{max_factor_tokens}, \code{covariate_value_encoding},
#'   \code{shared_projection_value_encoder}, and
#'   \code{max_covariate_tokens}.
#' @param conda_env Conda env name for the neural backend.
#' @param conda_env_required Require the conda env to exist.
#' @param cache_path Optional predictor cache path.
#' @param cache_overwrite Logical; overwrite any existing cache at \code{cache_path}.
#' @param cache_compress Compression passed to \code{saveRDS()}.
#' @return A \code{strategic_predictor}.
#'
#' @details
#' Current semantic zero-overlap foundation models are adapted canonically with
#' \code{preference.fm::adapt_conjoint_foundation_model()}. This helper is
#' retained for legacy runtime compatibility and errors when called on current
#' semantic foundation objects. For legacy objects, it reuses the package's
#' existing Bayesian neural outcome model rather than fitting a separate
#' downstream architecture. Internally, this function:
#'
#' \enumerate{
#'   \item normalizes the target study into the same local schema representation
#'   used by the neural outcome model,
#'   \item finds the compatible foundation group matching
#'   \code{(mode, likelihood, n_outcomes)},
#'   \item builds warm-start values from the pooled foundation parameters, and
#'   \item runs the current Bayesian neural fit with those warm starts.
#' }
#'
#' Group matching is exact. Adaptation fails if the foundation object does not
#' contain a compatible internal family for the requested target study.
#'
#' The target-study data contract mirrors the per-experiment contract used by
#' \code{\link{fit_conjoint_foundation_model}()}:
#' \itemize{
#'   \item \code{length(Y) == nrow(W)};
#'   \item pairwise studies require \code{pair_id};
#'   \item \code{X}, when supplied, must be numeric and row-aligned to
#'   \code{W}; raw numeric values remain unchanged during adaptation. Under the
#'   default FM covariate path, shared fused covariate/value tokens are aligned
#'   by pooled \code{X} name, optionally receive rebuilt \code{X}-name text
#'   embeddings, and are rebuilt from standardized numeric values plus local
#'   adaptation-study distribution summaries. Absent covariates are masked
#'   structurally rather than represented with a separate presence embedding;
#'   \item categorical adaptation follows the same \code{n_outcomes} rule as
#'   pooled training.
#' }
#'
#' Schema reuse is partial by design:
#' \itemize{
#'   \item matched factors and levels inherit warm-started embeddings from the
#'   foundation fit;
#'   \item unmatched local schema elements fall back to local initialization;
#'   \item if \code{foundation_adaptation_control$strict_schema_match = TRUE},
#'   unmatched factors cause an error instead of falling back.
#' }
#'
#' The \code{foundation_adaptation_control} list supports:
#' \describe{
#'   \item{\code{strict_schema_match}}{Require every local factor to match a
#'   pooled foundation slot. Default \code{FALSE}.}
#'   \item{\code{allow_extra_covariates}}{Allow local covariates that were not
#'   present during pooled training. Extra local covariate tokens are appended
#'   after the shared pooled covariate vocabulary. Default \code{TRUE}.}
#'   \item{\code{use_text_semantics}}{Reuse token-native text semantics during
#'   adaptation when the foundation object includes them, including factor-name,
#'   level-label, and pooled \code{X}-name token components. Default
#'   \code{TRUE}.}
#'   \item{\code{text_embedding_fn}}{Optional embedding function used to rebuild
#'   token-native text information for the target study. When omitted,
#'   adaptation reuses the stored pooled text registry when available. Any
#'   supplied function must return the same embedding width as the pooled
#'   foundation text registry.}
#'   \item{\code{experiment_token_mode}}{Experiment token mode for the adapted
#'   fit. Defaults to the pooled group's FM setting.}
#'   \item{\code{factor_tokenization}}{Factor tokenizer used during adaptation.
#'   Defaults to the pooled group's FM setting.}
#'   \item{\code{max_factor_tokens}}{Total factor-token budget used during
#'   adaptation. Defaults to the pooled group's FM setting.}
#'   \item{\code{covariate_value_encoding}}{Covariate encoder family used during
#'   adaptation. Defaults to the pooled group's FM setting.}
#'   \item{\code{shared_projection_value_encoder}}{Value-token generator used
#'   inside \code{"shared_projection"} during adaptation. Defaults to the
#'   pooled group's FM setting.}
#'   \item{\code{max_covariate_tokens}}{Total covariate-token budget used during
#'   adaptation. Defaults to the pooled group's FM setting.}
#' }
#'
#' @examples
#' \dontrun{
#' library(strategize)
#'
#' build_backend(conda_env = "strategize_env")
#'
#' text_embedding_fn <- function(x) {
#'   x <- as.character(x)
#'   cbind(nchar = nchar(x), has_space = as.numeric(grepl(" ", x)))
#' }
#'
#' foundation_fit <- preference.fm::fit_conjoint_foundation_model(
#'   experiments = list(
#'     list(
#'       experiment_id = "study_a",
#'       experiment_description = "Source policy-message study",
#'       Y = c(1, 0, 0, 1),
#'       W = data.frame(
#'         price = c("Low", "Low", "High", "High"),
#'         message = c("Jobs", "Taxes", "Jobs", "Taxes")
#'       ),
#'       pair_id = c(1, 1, 2, 2),
#'       profile_order = c(1, 2, 1, 2),
#'       canonical_factor_id = c(price = "price", message = "message")
#'     )
#'   ),
#'   foundation_control = list(text_embedding_fn = text_embedding_fn)
#' )
#'
#' adapted_fit <- preference.fm::adapt_conjoint_foundation_model(
#'   foundation_model = foundation_fit,
#'   Y = c(1, 0, 0, 1),
#'   W = data.frame(
#'     price = c("Low", "High", "Low", "High"),
#'     message = c("Jobs", "Jobs", "Taxes", "Taxes")
#'   ),
#'   mode = "pairwise",
#'   pair_id = c(1, 1, 2, 2),
#'   profile_order = c(1, 2, 1, 2),
#'   experiment_id = "target_study",
#'   experiment_description = "Target policy-message study",
#'   canonical_factor_id = c(price = "price", message = "message"),
#'   foundation_adaptation_control = list(text_embedding_fn = text_embedding_fn)
#' )
#'
#' predict(
#'   adapted_fit,
#'   newdata = list(
#'     W = data.frame(
#'       price = c("Low", "High"),
#'       message = c("Jobs", "Taxes")
#'     ),
#'     pair_id = c(1, 1),
#'     profile_order = c(1, 2)
#'   )
#' )
#' }
#'
#' @seealso \code{\link{fit_conjoint_foundation_model}()},
#'   \code{\link{build_backend}()}
#' @export
adapt_conjoint_foundation_model <- function(foundation_model,
                                            Y,
                                            W,
                                            X = NULL,
                                            mode = c("auto", "pairwise", "single"),
                                            pair_id = NULL,
                                            profile_order = NULL,
                                            experiment_id = "adaptation_target",
                                            experiment_description = NULL,
                                            names_list = NULL,
                                            p_list = NULL,
                                            respondent_id = NULL,
                                            respondent_task_id = NULL,
                                            competing_group_variable_candidate = NULL,
                                            competing_group_variable_respondent = NULL,
                                            likelihood = NULL,
                                            n_outcomes = NULL,
                                            canonical_factor_id = NULL,
                                            canonical_level_id = NULL,
                                            neural_mcmc_control = NULL,
                                            foundation_adaptation_control = NULL,
                                            conda_env = "strategize_env",
                                            conda_env_required = TRUE,
                                            cache_path = NULL,
                                            cache_overwrite = FALSE,
                                            cache_compress = TRUE) {
  if (!inherits(foundation_model, "conjoint_foundation_model")) {
    stop("'foundation_model' must be a conjoint_foundation_model.", call. = FALSE)
  }
  if (!is.null(cache_path)) {
    cache_path <- as.character(cache_path)
    if (length(cache_path) != 1L || !nzchar(cache_path)) {
      stop("'cache_path' must be a non-empty character path.", call. = FALSE)
    }
    if (!isTRUE(cache_overwrite) && file.exists(cache_path)) {
      return(load_strategic_predictor(
        cache_path,
        conda_env = conda_env,
        conda_env_required = conda_env_required
      ))
    }
  }
  if (cs_foundation_is_current_semantic_model(foundation_model)) {
    stop(
      "Current semantic zero-overlap conjoint foundation models must be adapted with ",
      "preference.fm::adapt_conjoint_foundation_model(). Use ",
      "preference.fm::load_conjoint_foundation_bundle() to load bundles and ",
      "strategize to run saved adapted predictors or extract embeddings.",
      call. = FALSE
    )
  }

  experiment <- cs_foundation_normalize_experiment(
    experiment = list(
      experiment_id = experiment_id,
      experiment_description = experiment_description,
      Y = Y,
      W = W,
      X = X,
      mode = match.arg(mode),
      pair_id = pair_id,
      profile_order = profile_order,
      names_list = names_list,
      p_list = p_list,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      competing_group_variable_candidate = competing_group_variable_candidate,
      competing_group_variable_respondent = competing_group_variable_respondent,
      likelihood = likelihood,
      n_outcomes = n_outcomes,
      canonical_factor_id = canonical_factor_id,
      canonical_level_id = canonical_level_id
    ),
    index = 1L
  )
  group <- cs_foundation_match_group(
    foundation_model = foundation_model,
    mode = experiment$mode,
    pairwise_context_mode = experiment$pairwise_context_mode %||% NULL,
    likelihood = experiment$likelihood,
    n_outcomes = experiment$n_outcomes
  )

  adaptation_control <- cs_foundation_resolve_adaptation_control(
    group = group,
    adaptation_control = foundation_adaptation_control %||% list()
  )
  adaptation_control$text_embedding_fn <- adaptation_control$text_embedding_fn %||%
    foundation_model$metadata$text_embedding_fn %||% NULL
  local_map <- cs_foundation_build_local_factor_map(experiment)
  X_aug <- cs_foundation_build_adaptation_x(
    group = group,
    experiment = experiment,
    exp_map = local_map,
    adaptation_control = adaptation_control
  )
  X_present_aug <- attr(X_aug, "resp_cov_present", exact = TRUE)
  token_info <- attr(X_aug, "token_info", exact = TRUE)
  local_names_list <- experiment$names_list
  enc <- cs_encode_W_indices(
    W = experiment$W,
    names_list = local_names_list,
    unknown = "error",
    align = "by_name"
  )
  local_x_feature_names <- if (!is.null(X_aug)) colnames(X_aug) else character(0)
  init_site_values <- cs_foundation_build_init_site_values(
    group = group,
    experiment = experiment,
    local_x_feature_names = local_x_feature_names,
    strict_schema_match = isTRUE(adaptation_control$strict_schema_match),
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )

  fit_control <- neural_mcmc_control %||% list()
  fit_control <- modifyList(list(init_site_values = init_site_values), fit_control)

  fit <- cs2step_eval_outcome_model_neural(
    Y = experiment$Y,
    W_idx = enc$W_idx,
    names_list = local_names_list,
    factor_levels = experiment$factor_levels,
    diff = identical(experiment$mode, "pairwise"),
    pair_id = experiment$pair_id,
    profile_order = experiment$profile_order,
    competing_group_variable_candidate = experiment$competing_group_variable_candidate,
    competing_group_variable_respondent = experiment$competing_group_variable_respondent,
    X = X_aug,
    X_present = X_present_aug,
    respondent_id = experiment$respondent_id,
    respondent_task_id = experiment$respondent_task_id,
    neural_token_info = token_info,
    likelihood_override = experiment$likelihood,
    n_outcomes_override = if (identical(experiment$likelihood, "categorical")) experiment$n_outcomes else NULL,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    neural_mcmc_control = fit_control
  )

  out <- cs_foundation_build_predictor(
    fit = fit,
    mode = experiment$mode,
    names_list = local_names_list,
    factor_levels = experiment$factor_levels,
    metadata = {
      text_fn <- adaptation_control$text_embedding_fn %||%
        foundation_model$metadata$text_embedding_fn %||% NULL
      text_meta <- cs2step_capture_text_embedding_metadata(
        text_embedding_fn = text_fn,
        text_embedding_backend = adaptation_control$text_embedding_backend %||%
          foundation_model$metadata$text_embedding_backend %||% NULL
      )
      list(
      call = match.call(),
      conda_env = conda_env,
      conda_env_required = conda_env_required,
      text_embedding_fn = text_meta$text_embedding_fn,
      text_embedding_backend = text_meta$text_embedding_backend,
      foundation_group_key = group$group_key,
      foundation_experiment_ids = group$experiment_ids,
      adaptation_experiment_id = experiment$experiment_id,
      adaptation_experiment_description = experiment$experiment_description %||% NULL
    )
    }
  )
  if (!is.null(cache_path)) {
    save_strategic_predictor(
      fit = out,
      file = cache_path,
      overwrite = TRUE,
      compress = cache_compress
    )
  }
  out
}

#' Conjoint foundation bundle writing moved to preference.fm
#'
#' @param file Path to save the bundle.
#' @param foundation_model A fitted \code{conjoint_foundation_model}.
#' @param overwrite Logical; overwrite any existing file.
#' @param compress Retained for API compatibility.
#' @return The bundle path (invisibly).
#'
#' @details
#' This exported name is retained as a compatibility shim. New code should call
#' \code{preference.fm::save_conjoint_foundation_bundle()} directly.
#'
#' @examples
#' \dontrun{
#' preference.fm::save_conjoint_foundation_bundle(
#'   tempfile(),
#'   foundation_model = foundation_fit,
#'   overwrite = TRUE
#' )
#' }
#'
#' @seealso \code{\link{load_conjoint_foundation_bundle}()}
#' @export
save_conjoint_foundation_bundle <- function(file,
                                            foundation_model,
                                            overwrite = FALSE,
                                            compress = TRUE) {
  warning(
    "strategize::save_conjoint_foundation_bundle() is deprecated; ",
    "use preference.fm::save_conjoint_foundation_bundle().",
    call. = FALSE
  )
  save_fn <- cs_foundation_preference_fm_export("save_conjoint_foundation_bundle")
  save_fn(
    file = file,
    foundation_model = foundation_model,
    overwrite = overwrite,
    compress = compress
  )
}

#' Load a conjoint foundation bundle
#'
#' @param file Path to a checkpoint directory created by
#'   \code{preference.fm::save_conjoint_foundation_bundle()} or to a legacy
#'   \code{.rds} bundle.
#' @param conda_env Conda env name for the neural backend.
#' @param conda_env_required Require the conda env to exist.
#' @param preload_params Logical; reconstruct neural params immediately.
#' @return A \code{conjoint_foundation_model}.
#'
#' @details
#' This exported name is retained as a compatibility shim. New code should call
#' \code{preference.fm::load_conjoint_foundation_bundle()} directly.
#'
#' @examples
#' \dontrun{
#' foundation_fit <- preference.fm::load_conjoint_foundation_bundle("foundation_bundle")
#' }
#'
#' @seealso \code{\link{save_conjoint_foundation_bundle}()}
#' @export
load_conjoint_foundation_bundle <- function(file,
                                            conda_env = "strategize_env",
                                            conda_env_required = TRUE,
                                            preload_params = FALSE) {
  warning(
    "strategize::load_conjoint_foundation_bundle() is deprecated; ",
    "use preference.fm::load_conjoint_foundation_bundle().",
    call. = FALSE
  )
  load_fn <- cs_foundation_preference_fm_export("load_conjoint_foundation_bundle")
  load_fn(
    file = file,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    preload_params = preload_params
  )
}

cs_foundation_preference_fm_export <- function(name) {
  if (!requireNamespace("preference.fm", quietly = TRUE)) {
    stop(
      sprintf(
        "strategize::%s() is a compatibility shim. Install preference.fm and call preference.fm::%s() directly.",
        name,
        name
      ),
      call. = FALSE
    )
  }
  getExportedValue("preference.fm", name)
}
