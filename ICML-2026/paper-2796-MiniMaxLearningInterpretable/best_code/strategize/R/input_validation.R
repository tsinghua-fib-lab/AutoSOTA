#' Input Validation for strategize Functions
#'
#' @description
#' Internal validation functions that provide clear, actionable error messages
#' for common input mistakes. These functions are called at the entry points
#' of main package functions to catch errors early.
#'
#' @name input-validation
#' @keywords internal
NULL

#' Validate inputs for strategize()
#'
#' @param Y Outcome vector
#' @param W Factor matrix
#' @param X Optional covariate matrix
#' @param lambda Regularization parameter
#' @param p_list Baseline probability list
#' @param K Number of clusters
#' @param pair_id Pair identifier
#' @param profile_order Profile order indicator for paired choices
#' @param adversarial Adversarial mode flag
#' @param adversarial_model_strategy Character string indicating whether to use
#'   "four" models (primary + general for each group) or "two" models (one per group
#'   reused for primary and general) in adversarial mode.
#' @param partial_pooling Logical indicating whether to partially pool (shrink) group-specific
#'   outcome model coefficients toward a shared average when using the "two" strategy.
#' @param partial_pooling_strength Numeric scalar controlling the amount of shrinkage used for
#'   partial pooling in the two-strategy adversarial case.
#' @param competing_group_variable_respondent Respondent group variable
#' @param competing_group_variable_candidate Candidate group variable
#' @param competing_group_competition_variable_candidate Candidate competition variable
#' @param outcome_model_type Outcome model type
#' @param penalty_type Penalty type
#' @param neural_mcmc_control Optional list overriding neural MCMC defaults. In adversarial
#'   neural mode, set \code{neural_mcmc_control$n_bayesian_models = 2} to fit separate AST/DAG
#'   models (default is 1 for a single differential model). Set
#'   \code{neural_mcmc_control$ModelDims} and \code{neural_mcmc_control$ModelDepth} to override
#'   the Transformer hidden width and depth. Pairwise neural fits default to
#'   \code{neural_mcmc_control$cross_candidate_encoder = "term"} when unspecified,
#'   except rank-positive low-rank respondent-candidate interaction defaults to
#'   \code{"none"} to avoid duplicating pairwise interaction capacity. Set
#'   \code{neural_mcmc_control$cross_candidate_encoder = "term"} (or \code{TRUE}) to include
#'   the opponent-dependent cross-candidate term in pairwise mode, set
#'   \code{neural_mcmc_control$cross_candidate_encoder = "attn"} to add a lightweight
#'   cross-attention step. When combined with
#'   \code{neural_mcmc_control$residual_mode = "full_attn"}, that step consumes the
#'   depth-attended candidate-token readout. Set
#'   \code{neural_mcmc_control$cross_candidate_encoder = "full"}
#'   to enable a full cross-encoder that jointly encodes both candidates. Use
#'   \code{"none"} (or \code{FALSE}) to disable. Set
#'   \code{neural_mcmc_control$qk_norm = TRUE} (default) to apply RMSNorm to projected
#'   queries and keys before attention logits are formed. Set
#'   \code{neural_mcmc_control$residual_mode = "full_attn"} to replace the
#'   default additive/ReZero residual path with full depth-wise attention over
#'   all prior layer outputs plus a final depth-attentive readout; use
#'   \code{"standard"} (default) to keep the
#'   existing residual formulation. Rank-positive low-rank Bernoulli pairwise
#'   logits use RMS logit normalization by default; set
#'   \code{neural_mcmc_control$low_rank_logit_normalization = "none"} to
#'   disable it. Explicit \code{low_rank_logit_transform = "softclip"} remains
#'   supported for compatibility. When additive utility is enabled, its output
#'   head uses RMS normalization by default; set
#'   \code{neural_mcmc_control$additive_utility_normalization = "none"} to
#'   restore the raw additive head.
#'   For variational inference (subsample_method = "batch_vi"), set
#'   \code{neural_mcmc_control$optimizer} to \code{"muon"} (default when \code{optax.contrib.muon} is available),
#'   \code{"adam"} (numpyro.optim), \code{"adamw"} (AdamW), or \code{"adabelief"} (optax).
#'   Muon is only applied to matrix-valued model-weight locations when the guide preserves
#'   matrix structure; with \code{vi_guide = "auto_diagonal"}, it falls back to AdamW/Adam.
#'   Learning-rate decay is controlled by
#'   \code{neural_mcmc_control$svi_lr_schedule} (default \code{"warmup_cosine"}), with optional
#'   \code{svi_lr_warmup_frac} and \code{svi_lr_end_factor}. Validation-based
#'   early stopping is enabled by default for SVI-backed neural fits; set
#'   \code{neural_mcmc_control$early_stopping = FALSE} to force the full
#'   \code{svi_steps} budget. Use
#'   \code{neural_mcmc_control$early_stopping_n_checks} (default \code{10}) to
#'   request an approximate number of validation checks during SVI early
#'   stopping; the cadence is derived as
#'   \code{ceiling(svi_steps / early_stopping_n_checks)}. Use
#'   \code{neural_mcmc_control$early_stopping_patience} (default \code{3}) to
#'   control how many consecutive non-improving validation checks are tolerated
#'   before stopping. For compact streaming SVI, \code{early_stopping = TRUE}
#'   enables validation best-checkpoint selection but still runs the full
#'   \code{svi_steps} budget. Set
#'   \code{neural_mcmc_control$early_stopping_validation_frac} (default \code{0.05})
#'   to retain approximately that fraction of evaluable observations in the
#'   held-out validation split, and optionally
#'   \code{neural_mcmc_control$early_stopping_validation_max_n} (default
#'   \code{2048}) to cap the retained validation size after fraction-based
#'   sizing. Set \code{early_stopping_validation_max_n = NULL} to disable that
#'   cap. Checkpoint gradient diagnostics are enabled by default for SVI fits
#'   and are evaluated at the early-stopping checkpoint cadence; set
#'   \code{neural_mcmc_control$gradient_diagnostics = FALSE} to disable them.
#'   Universal mixed-likelihood fits use
#'   \code{neural_mcmc_control$universal_loss_weighting = "empirical"} by
#'   default so objective mass follows SVI observation counts; set
#'   \code{"balanced_cell"} to reproduce clipped inverse-frequency weighting
#'   across task-mode/likelihood cells.
#'   Set \code{neural_mcmc_control$seed} (default \code{123}) to make SVI,
#'   posterior sampling, and MCMC JAX PRNG streams reproducible.
#'   Advanced restart checkpoints can be enabled with
#'   \code{neural_mcmc_control$checkpoint_path}; rerunning with the same data and
#'   controls resumes from \code{latest.rds} and promotes the best validation
#'   checkpoint when present. \code{checkpoint_resume} defaults to \code{TRUE}
#'   when a checkpoint path is present, \code{checkpoint_n_checks} defaults to
#'   \code{10} for non-early-stopping checkpoint cadence, and
#'   \code{checkpoint_compress} defaults to \code{FALSE}.
#' @param diff Difference mode flag
#' @param primary_pushforward Primary-stage push-forward estimator
#' @param primary_strength Scalar controlling the decisiveness of primaries
#' @param primary_n_entrants Number of entrant candidates sampled per party in multi-candidate primaries
#' @param primary_n_field Number of field candidates sampled per party in multi-candidate primaries
#' @param rain_lambda Numeric scalar giving the base RAIN regularization scale \eqn{\lambda}.
#' @param rain_gamma Non-negative numeric scalar for the RAIN anchor-growth parameter \eqn{\gamma}.
#'   If not supplied, defaults are auto-scaled downward when \code{nSGD} exceeds 100.
#' @param rain_L Optional numeric scalar for the Lipschitz constant \eqn{L} used by RAIN.
#' @param rain_eta Optional numeric scalar step size \eqn{\eta} for RAIN. Defaults to
#'   \code{0.001} and is auto-scaled downward when \code{nSGD} exceeds 100 if not supplied.
#' @param rain_variant Character string specifying the RAIN variant.
#' @param rain_output Character string controlling the RAIN stage output.
#' @return TRUE invisibly if validation passes; stops with error otherwise
#' @keywords internal
validate_strategize_inputs <- function(Y, W, X = NULL, lambda,
                                       p_list = NULL, K = 1,
                                       pair_id = NULL,
                                       profile_order = NULL,
                                       adversarial = FALSE,
                                       adversarial_model_strategy = "four",
                                       partial_pooling = NULL,
                                       partial_pooling_strength = 50,
                                       competing_group_variable_respondent = NULL,
                                       competing_group_variable_candidate = NULL,
                                       competing_group_competition_variable_candidate = NULL,
                                       outcome_model_type = "glm",
                                       penalty_type = "KL",
                                       neural_mcmc_control = NULL,
                                       diff = FALSE,
                                       primary_pushforward = "mc",
                                       primary_strength = 1.0,
                                       primary_n_entrants = 1L,
                                       primary_n_field = 1L,
                                       rain_lambda = 1,
                                       rain_gamma = 0.01,
                                       rain_L = NULL,
                                       rain_eta = 0.001,
                                       rain_variant = "alg10_staged",
                                       rain_output = "last") {

  # ---- Y validation ----
  if (missing(Y) || is.null(Y)) {
    stop(
      "'Y' is required: provide a numeric outcome vector.\n",
      "  Y should be the observed outcomes (e.g., 0/1 for binary choice).",
      call. = FALSE
    )
  }
  if (!is.numeric(Y) && !is.integer(Y)) {
    stop(
      "'Y' must be numeric. Got: ", class(Y)[1], "\n",
      "  Convert Y to numeric with as.numeric() if needed.",
      call. = FALSE
    )
  }
  if (length(Y) == 0) {
    stop("'Y' cannot be empty.", call. = FALSE)
  }
  n_na <- sum(is.na(Y))
  if (n_na > 0) {
    warning(
      sprintf("'Y' contains %d NA values (%.1f%%). ", n_na, 100 * n_na / length(Y)),
      "These may affect results.",
      call. = FALSE
    )
  }

  # ---- W validation ----
  if (missing(W) || is.null(W)) {
    stop(
      "'W' is required: provide a factor matrix (data.frame or matrix).\n",
      "  W should have one column per conjoint factor.",
      call. = FALSE
    )
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop(
      "'W' must be a data.frame or matrix. Got: ", class(W)[1], "\n",
      "  Convert W with as.data.frame() or as.matrix().",
      call. = FALSE
    )
  }
  if (nrow(W) != length(Y)) {
    stop(
      sprintf("Dimension mismatch: Y has %d elements but W has %d rows.\n", length(Y), nrow(W)),
      "  Ensure each row of W corresponds to one element of Y.\n",
      "  For paired forced-choice, each profile appears as a separate row.",
      call. = FALSE
    )
  }
  if (ncol(W) < 1) {
    stop("'W' must have at least 1 column (factor).", call. = FALSE)
  }
  if (anyNA(W)) {
    W_df_missing <- as.data.frame(W)
    factor_names_missing <- colnames(W_df_missing)
    if (is.null(factor_names_missing)) {
      factor_names_missing <- paste0("Column ", seq_len(ncol(W_df_missing)))
    }
    missing_cols <- factor_names_missing[vapply(W_df_missing, anyNA, logical(1))]
    stop(
      "'W' contains missing factor values in column(s): ",
      paste(missing_cols, collapse = ", "),
      ".\n  Impute, remove, or encode missing factor values as an explicit level before calling strategize().",
      call. = FALSE
    )
  }

  # ---- lambda validation ----
  if (missing(lambda) || is.null(lambda)) {
    stop(
      "'lambda' is required: provide a non-negative regularization value.\n",
      "  Tip: use cv_strategize() to select lambda via cross-validation.\n",
      "  Common values range from 0.01 to 1.0.",
      call. = FALSE
    )
  }
  if (!is.numeric(lambda) || length(lambda) != 1) {
    stop(
      "'lambda' must be a single numeric value. Got: ",
      paste(head(lambda, 3), collapse = ", "),
      if (length(lambda) > 3) "..." else "",
      call. = FALSE
    )
  }
  if (lambda < 0) {
    stop(
      "'lambda' must be non-negative. Got: ", lambda, "\n",
      "  Use lambda = 0 for no regularization (may overfit).",
      call. = FALSE
    )
  }

  # ---- p_list validation ----
  if (!is.null(p_list)) {
    if (!is.list(p_list)) {
      stop(
        "'p_list' must be a list of named probability vectors.\n",
        "  Each element should correspond to a factor in W.\n",
        "  Example: list(Gender = c(Male = 0.5, Female = 0.5))",
        call. = FALSE
      )
    }
    if (length(p_list) != ncol(W)) {
      stop(
        sprintf("'p_list' has %d elements but W has %d columns.\n", length(p_list), ncol(W)),
        "  Each factor in W needs a corresponding probability vector in p_list.\n",
        "  Tip: use create_p_list(W) to auto-generate from data.",
        call. = FALSE
      )
    }

    # Check level alignment for each factor
    for (i in seq_along(p_list)) {
      factor_name <- if (!is.null(colnames(W))) colnames(W)[i] else paste0("Column ", i)
      factor_levels <- sort(unique(as.character(W[, i])))
      p_levels <- sort(names(p_list[[i]]))

      if (is.null(names(p_list[[i]]))) {
        stop(
          sprintf("p_list[[%d]] ('%s') must have named elements.\n", i, factor_name),
          "  Example: c(Male = 0.5, Female = 0.5)",
          call. = FALSE
        )
      }

      if (!setequal(factor_levels, p_levels)) {
        missing_in_p <- setdiff(factor_levels, p_levels)
        extra_in_p <- setdiff(p_levels, factor_levels)
        msg <- sprintf("Factor '%s' (column %d) level mismatch:", factor_name, i)
        if (length(missing_in_p) > 0) {
          msg <- paste0(msg, "\n  Missing in p_list: ", paste(missing_in_p, collapse = ", "))
        }
        if (length(extra_in_p) > 0) {
          msg <- paste0(msg, "\n  Extra in p_list (not in W): ", paste(extra_in_p, collapse = ", "))
        }
        msg <- paste0(msg, "\n  Tip: use create_p_list(W) to auto-generate aligned p_list.")
        stop(msg, call. = FALSE)
      }

      probs <- suppressWarnings(as.numeric(p_list[[i]]))
      if (length(probs) != length(p_list[[i]]) || any(!is.finite(probs))) {
        stop(
          sprintf("p_list[[%d]] ('%s') must contain finite, non-missing probabilities.\n", i, factor_name),
          "  Use numeric probabilities with no NA, NaN, Inf, or -Inf values.",
          call. = FALSE
        )
      }

      # Check for negative probabilities
      if (any(probs < 0)) {
        stop(
          sprintf("p_list[[%d]] ('%s') contains negative probabilities.\n", i, factor_name),
          "  All probabilities must be >= 0.",
          call. = FALSE
        )
      }

      # Check probabilities sum to 1
      prob_sum <- sum(probs)
      if (abs(prob_sum - 1) > 1e-4) {
        stop(
          sprintf("p_list[[%d]] ('%s') probabilities sum to %.4f, not 1.0.\n", i, factor_name, prob_sum),
          "  Probabilities should sum to 1 for valid distribution.",
          call. = FALSE
        )
      }
    }
  }

  # ---- RAIN hyperparameters ----
  if (!is.null(rain_lambda)) {
    if (!is.numeric(rain_lambda) || length(rain_lambda) != 1) {
      stop(
        "'rain_lambda' must be a single non-negative numeric value. Got: ",
        paste(head(rain_lambda, 3), collapse = ", "),
        if (length(rain_lambda) > 3) "..." else "",
        call. = FALSE
      )
    }
    if (!is.finite(rain_lambda) || rain_lambda < 0) {
      stop("'rain_lambda' must be finite and non-negative.", call. = FALSE)
    }
  }
  if (!is.null(rain_gamma)) {
    if (!is.numeric(rain_gamma) || length(rain_gamma) != 1) {
      stop(
        "'rain_gamma' must be a single non-negative numeric value. Got: ",
        paste(head(rain_gamma, 3), collapse = ", "),
        if (length(rain_gamma) > 3) "..." else "",
        call. = FALSE
      )
    }
    if (!is.finite(rain_gamma) || rain_gamma < 0) {
      stop("'rain_gamma' must be finite and non-negative.", call. = FALSE)
    }
  }
  if (!is.null(rain_L)) {
    if (!is.numeric(rain_L) || length(rain_L) != 1) {
      stop(
        "'rain_L' must be a single positive numeric value or NULL. Got: ",
        paste(head(rain_L, 3), collapse = ", "),
        if (length(rain_L) > 3) "..." else "",
        call. = FALSE
      )
    }
    if (!is.finite(rain_L) || rain_L <= 0) {
      stop("'rain_L' must be finite and positive.", call. = FALSE)
    }
  }
  if (!is.null(rain_eta)) {
    if (!is.numeric(rain_eta) || length(rain_eta) != 1) {
      stop(
        "'rain_eta' must be a single positive numeric value or NULL. Got: ",
        paste(head(rain_eta, 3), collapse = ", "),
        if (length(rain_eta) > 3) "..." else "",
        call. = FALSE
      )
    }
    if (!is.finite(rain_eta) || rain_eta <= 0) {
      stop("'rain_eta' must be finite and positive.", call. = FALSE)
    }
  }
  if (!is.null(rain_variant)) {
    if (!is.character(rain_variant) || length(rain_variant) != 1) {
      stop("'rain_variant' must be a single character string.", call. = FALSE)
    }
    if (is.na(rain_variant) || !rain_variant %in% c("alg10_staged", "alg9_single_loop")) {
      stop("'rain_variant' must be one of: 'alg10_staged', 'alg9_single_loop'.", call. = FALSE)
    }
  }
  if (!is.null(rain_output)) {
    if (!is.character(rain_output) || length(rain_output) != 1) {
      stop("'rain_output' must be a single character string.", call. = FALSE)
    }
    if (is.na(rain_output) || !rain_output %in% c("uniform_half", "last")) {
      stop("'rain_output' must be one of: 'uniform_half', 'last'.", call. = FALSE)
    }
  }

  # ---- adversarial mode validation ----
  if (isTRUE(adversarial)) {
    if (is.null(competing_group_variable_respondent)) {
      stop(
        "adversarial=TRUE requires 'competing_group_variable_respondent'.\n",
        "  This should be a vector indicating each respondent's group (e.g., party).\n",
        "  Length must match number of rows in W.",
        call. = FALSE
      )
    }
    if (is.null(competing_group_variable_candidate)) {
      stop(
        "adversarial=TRUE requires 'competing_group_variable_candidate'.\n",
        "  This should be a vector indicating each candidate profile's group.\n",
        "  Length must match number of rows in W.",
        call. = FALSE
      )
    }
    n_resp_groups <- length(unique(competing_group_variable_respondent))
    if (n_resp_groups != 2) {
      stop(
        sprintf(
          "Adversarial mode requires exactly 2 groups in competing_group_variable_respondent.\n"
        ),
        sprintf("  Found %d groups: %s\n", n_resp_groups,
                paste(head(unique(competing_group_variable_respondent), 5), collapse = ", ")),
        "  Adversarial mode models a two-player zero-sum game.",
        call. = FALSE
      )
    }

    if (length(competing_group_variable_respondent) != nrow(W)) {
      stop(
        sprintf(
          "competing_group_variable_respondent has %d elements but W has %d rows.\n",
          length(competing_group_variable_respondent), nrow(W)
        ),
        "  Lengths must match.",
        call. = FALSE
      )
    }
    if (length(competing_group_variable_candidate) != nrow(W)) {
      stop(
        sprintf(
          "competing_group_variable_candidate has %d elements but W has %d rows.\n",
          length(competing_group_variable_candidate), nrow(W)
        ),
        "  Lengths must match.",
        call. = FALSE
      )
    }
    if (anyNA(competing_group_variable_respondent) ||
        anyNA(competing_group_variable_candidate)) {
      stop(
        "Adversarial group variables cannot contain missing values.",
        call. = FALSE
      )
    }
    cand_groups <- unique(competing_group_variable_candidate)
    if (length(cand_groups) != 2) {
      stop(
        sprintf(
          "Adversarial mode requires exactly 2 groups in competing_group_variable_candidate.\n"
        ),
        sprintf("  Found %d groups: %s\n", length(cand_groups),
                paste(head(cand_groups, 5), collapse = ", ")),
        "  Adversarial mode models a two-player zero-sum game.",
        call. = FALSE
      )
    }
    if (!setequal(as.character(unique(competing_group_variable_respondent)),
                  as.character(cand_groups))) {
      stop(
        "Adversarial respondent and candidate group variables must contain matching group labels.",
        call. = FALSE
      )
    }
    if (!is.null(competing_group_competition_variable_candidate)) {
      if (length(competing_group_competition_variable_candidate) != nrow(W)) {
        stop(
          sprintf(
            "competing_group_competition_variable_candidate has %d elements but W has %d rows.\n",
            length(competing_group_competition_variable_candidate), nrow(W)
          ),
          "  Lengths must match.",
          call. = FALSE
        )
      }
      if (any(is.na(competing_group_competition_variable_candidate))) {
        stop(
          "competing_group_competition_variable_candidate contains NA values.\n",
          "  Use explicit \"Same\"/\"Different\" labels for all rows.",
          call. = FALSE
        )
      }
      valid_comp_labels <- c("Same", "Different")
      if (!all(competing_group_competition_variable_candidate %in% valid_comp_labels)) {
        stop(
          "competing_group_competition_variable_candidate must contain only \"Same\" or \"Different\".\n",
          "  Check for typos or unexpected labels.",
          call. = FALSE
        )
      }
    }
  }

  # ---- adversarial_model_strategy validation ----
  valid_model_strategies <- c("two", "four", "neural")
  if (!is.character(adversarial_model_strategy) || length(adversarial_model_strategy) != 1) {
    stop(
      "'adversarial_model_strategy' must be a single character string.\n",
      "  Valid options: ", paste(valid_model_strategies, collapse = ", "),
      call. = FALSE
    )
  }
  if (!tolower(adversarial_model_strategy) %in% valid_model_strategies) {
    stop(
      sprintf("'adversarial_model_strategy' must be one of: %s.\n",
              paste(valid_model_strategies, collapse = ", ")),
      sprintf("  Got: '%s'", adversarial_model_strategy),
      call. = FALSE
    )
  }

  # ---- partial_pooling validation ----
  if (!is.null(partial_pooling)) {
    if (!is.logical(partial_pooling) || length(partial_pooling) != 1) {
      stop(
        "'partial_pooling' must be a single logical value (TRUE/FALSE).\n",
        call. = FALSE
      )
    }
  }
  if (!is.numeric(partial_pooling_strength) || length(partial_pooling_strength) != 1 ||
      !is.finite(partial_pooling_strength) || partial_pooling_strength < 0) {
    stop(
      "'partial_pooling_strength' must be a single non-negative numeric value.\n",
      call. = FALSE
    )
  }

  # ---- diff mode validation ----
  neural_diff_strict <- isTRUE(diff) && identical(tolower(as.character(outcome_model_type)), "neural")
  if (isTRUE(diff) && is.null(pair_id)) {
    msg <- paste0(
      "diff=TRUE typically requires 'pair_id' to identify forced-choice pairs.\n",
      "  Without pair_id, the function may not correctly identify paired profiles."
    )
    if (isTRUE(neural_diff_strict)) {
      stop(msg, call. = FALSE)
    }
    warning(msg, call. = FALSE)
  }
  if (isTRUE(diff) && !is.null(pair_id)) {
    if (length(pair_id) != nrow(W)) {
      stop(
        sprintf("pair_id has %d elements but W has %d rows.\n",
                length(pair_id), nrow(W)),
        "  Lengths must match.",
        call. = FALSE
      )
    }
    if (any(is.na(pair_id))) {
      msg <- paste0(
        "pair_id contains NA values.\n",
        "  Missing pair identifiers can lead to ambiguous pair differences."
      )
      if (isTRUE(neural_diff_strict)) {
        stop(msg, call. = FALSE)
      }
      warning(msg, call. = FALSE)
    }
    if (is.null(profile_order)) {
      msg <- paste0(
        "diff=TRUE without 'profile_order'.\n",
        "  Provide profile_order to ensure consistent within-pair ordering."
      )
      if (isTRUE(neural_diff_strict)) {
        stop(msg, call. = FALSE)
      }
      warning(msg, call. = FALSE)
    } else {
      if (length(profile_order) != nrow(W)) {
        stop(
          sprintf("profile_order has %d elements but W has %d rows.\n",
                  length(profile_order), nrow(W)),
          "  Lengths must match.",
          call. = FALSE
        )
      }
      if (any(is.na(profile_order))) {
        msg <- paste0(
          "profile_order contains NA values.\n",
          "  Missing ordering can lead to ambiguous pair differences."
        )
        if (isTRUE(neural_diff_strict)) {
          stop(msg, call. = FALSE)
        }
        warning(msg, call. = FALSE)
      }
      pair_sizes <- table(pair_id)
      if (any(pair_sizes != 2L)) {
        msg <- paste0(
          "diff=TRUE expects exactly two rows per pair_id.\n",
          "  Found pairs with sizes other than 2; ordering may be unreliable."
        )
        if (isTRUE(neural_diff_strict)) {
          stop(msg, call. = FALSE)
        }
        warning(msg, call. = FALSE)
      }
      order_ok <- tapply(profile_order, pair_id, function(x) length(unique(x)) == 2L)
      if (any(!order_ok)) {
        msg <- paste0(
          "profile_order is inconsistent within one or more pairs.\n",
          "  Each pair_id should have two distinct order values (e.g., 1 and 2)."
        )
        if (isTRUE(neural_diff_strict)) {
          stop(msg, call. = FALSE)
        }
        warning(msg, call. = FALSE)
      }
      if (isTRUE(neural_diff_strict)) {
        profile_by_pair <- split(profile_order, pair_id)
        order_12 <- vapply(profile_by_pair, function(x) {
          identical(sort(unique(as.integer(x))), c(1L, 2L))
        }, logical(1))
        if (!all(order_12)) {
          stop(
            "Neural diff=TRUE requires profile_order values 1 and 2 within each pair_id.",
            call. = FALSE
          )
        }
      }
    }
  }

  # ---- K and X validation ----
  if (!is.numeric(K) || length(K) != 1 || K < 1 || K != round(K)) {
    stop(
      "'K' must be a positive integer. Got: ", K,
      call. = FALSE
    )
  }
  if (K > 1 && is.null(X)) {
    warning(
      "K > 1 (multi-cluster) typically requires 'X' (respondent covariates).\n",
      "  Without X, cluster identification may be unreliable.",
      call. = FALSE
    )
  }

  # ---- outcome_model_type validation ----
  valid_model_types <- c("glm", "neural")
  if (!outcome_model_type %in% valid_model_types) {
    stop(
      sprintf("'outcome_model_type' must be one of: %s.\n",
              paste(valid_model_types, collapse = ", ")),
      sprintf("  Got: '%s'", outcome_model_type),
      call. = FALSE
    )
  }
  if (tolower(adversarial_model_strategy) == "neural" && outcome_model_type != "neural") {
    stop(
      "adversarial_model_strategy = \"neural\" requires outcome_model_type = \"neural\".\n",
      "  Set outcome_model_type=\"neural\" to fit Bayesian Transformer models.",
      call. = FALSE
    )
  }
  if (!is.null(neural_mcmc_control) && !is.list(neural_mcmc_control)) {
    stop(
      "'neural_mcmc_control' must be a list when provided.",
      call. = FALSE
    )
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$seed)) {
    seed_val <- suppressWarnings(as.numeric(neural_mcmc_control$seed))
    if (length(seed_val) != 1L ||
        is.na(seed_val) ||
        !is.finite(seed_val) ||
        seed_val < 0 ||
        seed_val != floor(seed_val) ||
        seed_val > .Machine$integer.max) {
      stop(
        "'neural_mcmc_control$seed' must be one non-negative integer.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$uncertainty_scope)) {
    scope_val <- tolower(as.character(neural_mcmc_control$uncertainty_scope))
    if (!scope_val %in% c("all", "output")) {
      stop(
        "'neural_mcmc_control$uncertainty_scope' must be 'all' or 'output'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$n_bayesian_models)) {
    n_models <- neural_mcmc_control$n_bayesian_models
    if (!is.numeric(n_models) || length(n_models) != 1 || !is.finite(n_models)) {
      stop(
        "'neural_mcmc_control$n_bayesian_models' must be a single finite numeric value.",
        call. = FALSE
      )
    }
    if (n_models != round(n_models)) {
      stop(
        "'neural_mcmc_control$n_bayesian_models' must be an integer (1 or 2).",
        call. = FALSE
      )
    }
    n_models <- as.integer(n_models)
    if (!n_models %in% c(1L, 2L)) {
      stop(
        "'neural_mcmc_control$n_bayesian_models' must be 1 or 2.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$ModelDims)) {
    model_dims <- neural_mcmc_control$ModelDims
    if (!is.numeric(model_dims) || length(model_dims) != 1 || !is.finite(model_dims)) {
      stop(
        "'neural_mcmc_control$ModelDims' must be a single finite numeric value.",
        call. = FALSE
      )
    }
    if (model_dims != round(model_dims) || model_dims < 1) {
      stop(
        "'neural_mcmc_control$ModelDims' must be an integer >= 1.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$ModelDepth)) {
    model_depth <- neural_mcmc_control$ModelDepth
    if (!is.numeric(model_depth) || length(model_depth) != 1 || !is.finite(model_depth)) {
      stop(
        "'neural_mcmc_control$ModelDepth' must be a single finite numeric value.",
        call. = FALSE
      )
    }
    if (model_depth != round(model_depth) || model_depth < 1) {
      stop(
        "'neural_mcmc_control$ModelDepth' must be an integer >= 1.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$qk_norm)) {
    qk_norm <- neural_mcmc_control$qk_norm
    if (is.logical(qk_norm)) {
      if (length(qk_norm) != 1L || is.na(qk_norm)) {
        stop(
          "'neural_mcmc_control$qk_norm' must be TRUE/FALSE.",
          call. = FALSE
        )
      }
    } else if (is.character(qk_norm)) {
      qk_norm_tag <- tolower(as.character(qk_norm))
      if (length(qk_norm_tag) != 1L || is.na(qk_norm_tag) ||
          !qk_norm_tag %in% c("true", "false")) {
        stop(
          "'neural_mcmc_control$qk_norm' must be TRUE/FALSE.",
          call. = FALSE
        )
      }
    } else {
      stop(
        "'neural_mcmc_control$qk_norm' must be TRUE/FALSE.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$residual_mode)) {
    residual_mode <- neural_mcmc_control$residual_mode
    if (is.logical(residual_mode)) {
      if (length(residual_mode) != 1L || is.na(residual_mode)) {
        stop(
          "'neural_mcmc_control$residual_mode' must be one of 'standard' or 'full_attn'.",
          call. = FALSE
        )
      }
    } else if (is.character(residual_mode)) {
      mode <- tolower(as.character(residual_mode))
      if (length(mode) != 1L || is.na(mode) || !nzchar(mode) ||
          !mode %in% c(
            "standard", "add", "additive", "rezero",
            "full_attn", "full-attn", "full_attention",
            "attnres", "full_attnres", "attention_residual",
            "attention-residual", "full_attention_residual",
            "full-attention-residual"
          )) {
        stop(
          "'neural_mcmc_control$residual_mode' must be one of 'standard' or 'full_attn'.",
          call. = FALSE
        )
      }
    } else {
      stop(
        "'neural_mcmc_control$residual_mode' must be one of 'standard' or 'full_attn'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$cross_candidate_encoder)) {
    cross_encoder <- neural_mcmc_control$cross_candidate_encoder
    if (is.logical(cross_encoder)) {
      if (length(cross_encoder) != 1L || is.na(cross_encoder)) {
        stop(
          "'neural_mcmc_control$cross_candidate_encoder' must be TRUE/FALSE or one of ",
          "'none', 'term', 'attn', or 'full'.",
          call. = FALSE
        )
      }
    } else if (is.character(cross_encoder)) {
      mode <- tolower(as.character(cross_encoder))
      if (length(mode) != 1L || is.na(mode) || !nzchar(mode) ||
          !mode %in% c("none", "term", "attn", "lite", "cross_attn", "cls_attn",
                       "cross-attn", "cls-attn", "full", "true", "false")) {
        stop(
          "'neural_mcmc_control$cross_candidate_encoder' must be TRUE/FALSE or one of ",
          "'none', 'term', 'attn', or 'full'.",
          call. = FALSE
        )
      }
    } else {
      stop(
      "'neural_mcmc_control$cross_candidate_encoder' must be TRUE/FALSE or one of ",
      "'none', 'term', 'attn', or 'full'.",
      call. = FALSE
    )
  }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$low_rank_logit_transform)) {
    transform <- neural_mcmc_control$low_rank_logit_transform
    mode <- neural_normalize_low_rank_logit_transform(transform)
    if (length(mode) != 1L || is.na(mode) || !mode %in% c("softclip", "none")) {
      stop(
        "'neural_mcmc_control$low_rank_logit_transform' must be 'softclip' or 'none'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$low_rank_logit_bound)) {
    bound <- suppressWarnings(as.numeric(neural_mcmc_control$low_rank_logit_bound))
    if (length(bound) != 1L || is.na(bound)) {
      stop(
        "'neural_mcmc_control$low_rank_logit_bound' must be a scalar numeric value.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$low_rank_logit_softness)) {
    softness <- suppressWarnings(as.numeric(neural_mcmc_control$low_rank_logit_softness))
    if (length(softness) != 1L || is.na(softness) ||
        !is.finite(softness) || softness <= 0) {
      stop(
        "'neural_mcmc_control$low_rank_logit_softness' must be a positive scalar numeric value.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$low_rank_logit_normalization)) {
    normalization <- neural_normalize_low_rank_logit_normalization(
      neural_mcmc_control$low_rank_logit_normalization
    )
    if (length(normalization) != 1L ||
        is.na(normalization) ||
        !normalization %in% c("rms", "none")) {
      stop(
        "'neural_mcmc_control$low_rank_logit_normalization' must be 'rms' or 'none'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$low_rank_head_weight_target_rms)) {
    target <- suppressWarnings(as.numeric(neural_mcmc_control$low_rank_head_weight_target_rms))
    if (length(target) != 1L || is.na(target) ||
        !is.finite(target) || target <= 0) {
      stop(
        "'neural_mcmc_control$low_rank_head_weight_target_rms' must be a positive scalar numeric value.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$low_rank_rc_out_target_rms)) {
    target <- suppressWarnings(as.numeric(neural_mcmc_control$low_rank_rc_out_target_rms))
    if (length(target) != 1L || is.na(target) ||
        !is.finite(target) || target <= 0) {
      stop(
        "'neural_mcmc_control$low_rank_rc_out_target_rms' must be a positive scalar numeric value.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$additive_utility_normalization)) {
    normalization <- neural_normalize_additive_utility_normalization(
      neural_mcmc_control$additive_utility_normalization
    )
    if (length(normalization) != 1L ||
        is.na(normalization) ||
        !normalization %in% c("rms", "none")) {
      stop(
        "'neural_mcmc_control$additive_utility_normalization' must be 'rms' or 'none'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$additive_head_weight_target_rms)) {
    target <- suppressWarnings(as.numeric(neural_mcmc_control$additive_head_weight_target_rms))
    if (length(target) != 1L || is.na(target) ||
        !is.finite(target) || target <= 0) {
      stop(
        "'neural_mcmc_control$additive_head_weight_target_rms' must be a positive scalar numeric value.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$optimizer)) {
    optimizer_val <- tolower(as.character(neural_mcmc_control$optimizer))
    if (length(optimizer_val) != 1L || is.na(optimizer_val) ||
        !optimizer_val %in% c("adam", "adamw", "adabelief", "muon")) {
      stop(
        "'neural_mcmc_control$optimizer' must be 'adam', 'adamw', 'adabelief', or 'muon'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$svi_steps)) {
    steps_val <- neural_mcmc_control$svi_steps
    if (is.character(steps_val)) {
      steps_tag <- tolower(as.character(steps_val))
      if (length(steps_tag) != 1L || is.na(steps_tag) || !nzchar(steps_tag) ||
          !steps_tag %in% c("optimal")) {
        stop(
          "'neural_mcmc_control$svi_steps' must be a positive integer or 'optimal'.",
          call. = FALSE
        )
      }
    } else if (is.numeric(steps_val)) {
      if (length(steps_val) != 1L || !is.finite(steps_val) ||
          steps_val < 1 || steps_val != round(steps_val)) {
        stop(
          "'neural_mcmc_control$svi_steps' must be a positive integer or 'optimal'.",
          call. = FALSE
        )
      }
    } else {
      stop(
        "'neural_mcmc_control$svi_steps' must be a positive integer or 'optimal'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$svi_lr_schedule)) {
    schedule_val <- tolower(as.character(neural_mcmc_control$svi_lr_schedule))
    if (length(schedule_val) != 1L || is.na(schedule_val) ||
        !schedule_val %in% c("none", "constant", "cosine", "warmup_cosine")) {
      stop(
        "'neural_mcmc_control$svi_lr_schedule' must be one of 'none', 'constant', ",
        "'cosine', or 'warmup_cosine'.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$svi_lr_warmup_frac)) {
    warmup_frac <- neural_mcmc_control$svi_lr_warmup_frac
    if (!is.numeric(warmup_frac) || length(warmup_frac) != 1L ||
        !is.finite(warmup_frac) || warmup_frac < 0) {
      stop(
        "'neural_mcmc_control$svi_lr_warmup_frac' must be a single non-negative number.",
        call. = FALSE
      )
    }
  }
  if (!is.null(neural_mcmc_control) &&
      !is.null(neural_mcmc_control$svi_lr_end_factor)) {
    end_factor <- neural_mcmc_control$svi_lr_end_factor
    if (!is.numeric(end_factor) || length(end_factor) != 1L ||
        !is.finite(end_factor) || end_factor < 0) {
      stop(
        "'neural_mcmc_control$svi_lr_end_factor' must be a single non-negative number.",
        call. = FALSE
      )
    }
  }
  early_stopping <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["early_stopping"]]
  } else {
    NULL
  }
  if (!is.null(early_stopping)) {
    if (!is.logical(early_stopping) || length(early_stopping) != 1L || is.na(early_stopping)) {
      stop(
        "'neural_mcmc_control$early_stopping' must be TRUE or FALSE.",
        call. = FALSE
      )
    }
  }
  gradient_diagnostics <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["gradient_diagnostics"]]
  } else {
    NULL
  }
  if (!is.null(gradient_diagnostics)) {
    if (!is.logical(gradient_diagnostics) ||
        length(gradient_diagnostics) != 1L ||
        is.na(gradient_diagnostics)) {
      stop(
        "'neural_mcmc_control$gradient_diagnostics' must be TRUE or FALSE.",
        call. = FALSE
      )
    }
  }
  universal_loss_weighting <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["universal_loss_weighting"]]
  } else {
    NULL
  }
  if (!is.null(universal_loss_weighting)) {
    weighting_mode <- tolower(trimws(as.character(universal_loss_weighting)))
    if (length(weighting_mode) != 1L ||
        is.na(weighting_mode) ||
        !nzchar(weighting_mode) ||
        !weighting_mode %in% c("empirical", "balanced_cell")) {
      stop(
        "'neural_mcmc_control$universal_loss_weighting' must be one of 'empirical' or 'balanced_cell'.",
        call. = FALSE
      )
    }
  }
  universal_loss_weight_clip <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["universal_loss_weight_clip"]]
  } else {
    NULL
  }
  if (!is.null(universal_loss_weight_clip)) {
    clip <- suppressWarnings(as.numeric(universal_loss_weight_clip))
    if (length(clip) != 2L ||
        any(!is.finite(clip)) ||
        clip[[1L]] <= 0 ||
        clip[[2L]] < clip[[1L]]) {
      stop(
        "'neural_mcmc_control$universal_loss_weight_clip' must be two positive finite numbers with min <= max.",
        call. = FALSE
      )
    }
  }
  checkpoint_path <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["checkpoint_path"]]
  } else {
    NULL
  }
  if (!is.null(checkpoint_path)) {
    checkpoint_path <- as.character(checkpoint_path)
    if (length(checkpoint_path) != 1L || is.na(checkpoint_path) || !nzchar(checkpoint_path)) {
      stop(
        "'neural_mcmc_control$checkpoint_path' must be a non-empty path.",
        call. = FALSE
      )
    }
  }
  checkpoint_resume <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["checkpoint_resume"]]
  } else {
    NULL
  }
  if (!is.null(checkpoint_resume)) {
    if (!is.logical(checkpoint_resume) ||
        length(checkpoint_resume) != 1L ||
        is.na(checkpoint_resume)) {
      stop(
        "'neural_mcmc_control$checkpoint_resume' must be TRUE or FALSE.",
        call. = FALSE
      )
    }
  }
  checkpoint_n_checks <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["checkpoint_n_checks"]]
  } else {
    NULL
  }
  if (!is.null(checkpoint_n_checks)) {
    if (!is.numeric(checkpoint_n_checks) ||
        length(checkpoint_n_checks) != 1L ||
        !is.finite(checkpoint_n_checks) ||
        checkpoint_n_checks < 1 ||
        checkpoint_n_checks != round(checkpoint_n_checks)) {
      stop(
        "'neural_mcmc_control$checkpoint_n_checks' must be an integer >= 1.",
        call. = FALSE
      )
    }
  }
  checkpoint_compress <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["checkpoint_compress"]]
  } else {
    NULL
  }
  if (!is.null(checkpoint_compress)) {
    if (!is.logical(checkpoint_compress) ||
        length(checkpoint_compress) != 1L ||
        is.na(checkpoint_compress)) {
      stop(
        "'neural_mcmc_control$checkpoint_compress' must be TRUE or FALSE.",
        call. = FALSE
      )
    }
  }
  n_checks <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["early_stopping_n_checks"]]
  } else {
    NULL
  }
  if (!is.null(n_checks)) {
    if (!is.numeric(n_checks) || length(n_checks) != 1L ||
        !is.finite(n_checks) || n_checks < 1 ||
        n_checks != round(n_checks)) {
      stop(
        "'neural_mcmc_control$early_stopping_n_checks' must be an integer >= 1.",
        call. = FALSE
      )
    }
  }
  patience <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["early_stopping_patience"]]
  } else {
    NULL
  }
  if (!is.null(patience)) {
    if (!is.numeric(patience) || length(patience) != 1L ||
        !is.finite(patience) || patience < 1 ||
        patience != round(patience)) {
      stop(
        "'neural_mcmc_control$early_stopping_patience' must be an integer >= 1.",
        call. = FALSE
      )
    }
  }
  validation_frac <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["early_stopping_validation_frac"]]
  } else {
    NULL
  }
  if (!is.null(validation_frac)) {
    if (!is.numeric(validation_frac) || length(validation_frac) != 1L ||
        !is.finite(validation_frac) || validation_frac <= 0 || validation_frac > 1) {
      stop(
        "'neural_mcmc_control$early_stopping_validation_frac' must be a single number in (0, 1].",
        call. = FALSE
      )
    }
  }
  validation_max_n <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["early_stopping_validation_max_n"]]
  } else {
    NULL
  }
  if (!is.null(validation_max_n)) {
    if (!is.numeric(validation_max_n) || length(validation_max_n) != 1L ||
        !is.finite(validation_max_n) || validation_max_n < 1 ||
        validation_max_n != round(validation_max_n)) {
      stop(
        "'neural_mcmc_control$early_stopping_validation_max_n' must be NULL or an integer >= 1.",
        call. = FALSE
      )
    }
  }
  validation_batch_size <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["early_stopping_validation_batch_size"]]
  } else {
    NULL
  }
  if (!is.null(validation_batch_size)) {
    if (!is.numeric(validation_batch_size) || length(validation_batch_size) != 1L ||
        !is.finite(validation_batch_size) || validation_batch_size < 1 ||
        validation_batch_size != round(validation_batch_size)) {
      stop(
        "'neural_mcmc_control$early_stopping_validation_batch_size' must be NULL or an integer >= 1.",
        call. = FALSE
      )
    }
  }
  compact_update_chunk_size <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["compact_update_chunk_size"]]
  } else {
    NULL
  }
  if (!is.null(compact_update_chunk_size)) {
    if (!is.numeric(compact_update_chunk_size) ||
        length(compact_update_chunk_size) != 1L ||
        !is.finite(compact_update_chunk_size) ||
        compact_update_chunk_size < 1 ||
        compact_update_chunk_size != round(compact_update_chunk_size)) {
      stop(
        "'neural_mcmc_control$compact_update_chunk_size' must be an integer >= 1.",
        call. = FALSE
      )
    }
  }
  compact_update_scan <- if (!is.null(neural_mcmc_control)) {
    neural_mcmc_control[["compact_update_scan"]]
  } else {
    NULL
  }
  if (!is.null(compact_update_scan)) {
    compact_update_scan <- tolower(as.character(compact_update_scan))
    if (length(compact_update_scan) != 1L ||
        is.na(compact_update_scan) ||
        !compact_update_scan %in% c("required", "fallback")) {
      stop(
        "'neural_mcmc_control$compact_update_scan' must be 'required' or 'fallback'.",
        call. = FALSE
      )
    }
  }

  # ---- penalty_type validation ----
  valid_penalty_types <- c("KL", "L2", "LogMaxProb")
  if (!penalty_type %in% valid_penalty_types) {
    stop(
      sprintf("'penalty_type' must be one of: %s.\n",
              paste(valid_penalty_types, collapse = ", ")),
      sprintf("  Got: '%s'\n", penalty_type),
      "  KL = Kullback-Leibler divergence (default)\n",
      "  L2 = Euclidean distance\n",
      "  LogMaxProb = Log maximum probability",
      call. = FALSE
    )
  }

  # ---- primary_pushforward validation ----
  valid_pushforward <- c("mc", "linearized", "multi")
  if (!is.character(primary_pushforward) || length(primary_pushforward) != 1) {
    stop(
      "'primary_pushforward' must be a single character string.\n",
      "  Valid options: ", paste(valid_pushforward, collapse = ", "),
      call. = FALSE
    )
  }
  if (!tolower(primary_pushforward) %in% valid_pushforward) {
    stop(
      sprintf("'primary_pushforward' must be one of: %s.\n",
              paste(valid_pushforward, collapse = ", ")),
      sprintf("  Got: '%s'", primary_pushforward),
      call. = FALSE
    )
  }

  # ---- primary_strength validation ----
  if (!is.numeric(primary_strength) || length(primary_strength) != 1 || is.na(primary_strength)) {
    stop(
      "'primary_strength' must be a single numeric value.\n",
      "  Use 1.0 for neutral scaling, >1 for stronger primaries.",
      call. = FALSE
    )
  }
  if (primary_strength < 0) {
    stop(
      "'primary_strength' must be non-negative.\n",
      "  Use 0 to neutralize primaries or >1 to strengthen them.",
      call. = FALSE
    )
  }

  # ---- multi-candidate primary validation ----
  if (!is.numeric(primary_n_entrants) || length(primary_n_entrants) != 1 ||
      primary_n_entrants < 1 || primary_n_entrants != round(primary_n_entrants)) {
    stop(
      "'primary_n_entrants' must be a positive integer.\n",
      "  Example: primary_n_entrants = 3.",
      call. = FALSE
    )
  }
  if (!is.numeric(primary_n_field) || length(primary_n_field) != 1 ||
      primary_n_field < 1 || primary_n_field != round(primary_n_field)) {
    stop(
      "'primary_n_field' must be a positive integer.\n",
      "  Example: primary_n_field = 3.",
      call. = FALSE
    )
  }
  if ((primary_n_entrants > 1 || primary_n_field > 1) &&
      tolower(primary_pushforward) != "multi") {
    stop(
      "'primary_pushforward' must be \"multi\" when primary_n_entrants > 1 ",
      "or primary_n_field > 1.\n",
      "  Set primary_pushforward = \"multi\" or use primary_n_entrants = 1 ",
      "and primary_n_field = 1.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


#' Validate inputs for cv_strategize()
#'
#' @param Y Outcome vector
#' @param W Factor matrix
#' @param lambda_seq Sequence of lambda values to cross-validate
#' @param folds Number of CV folds
#' @return TRUE invisibly if validation passes
#' @keywords internal
validate_cv_strategize_inputs <- function(Y, W, lambda_seq = NULL, folds = 2L) {

  # Basic Y/W validation (reuse from main function)
  if (missing(Y) || is.null(Y)) {
    stop("'Y' is required.", call. = FALSE)
  }
  if (missing(W) || is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }

  # folds validation
  if (!is.numeric(folds) || length(folds) != 1 || folds < 2) {
    stop(
      "'folds' must be an integer >= 2. Got: ", folds, "\n",
      "  Use folds = 2 for quick validation, folds = 5 or 10 for thorough CV.",
      call. = FALSE
    )
  }

  n_obs <- length(Y)
  if (folds > n_obs) {
    stop(
      sprintf("'folds' (%d) cannot exceed number of observations (%d).", folds, n_obs),
      call. = FALSE
    )
  }

  # lambda_seq validation
  if (!is.null(lambda_seq)) {
    if (!is.numeric(lambda_seq)) {
      stop("'lambda_seq' must be a numeric vector.", call. = FALSE)
    }
    if (any(lambda_seq < 0)) {
      stop("'lambda_seq' values must be non-negative.", call. = FALSE)
    }
    if (length(lambda_seq) < 2) {
      warning(
        "lambda_seq has only 1 value. Cross-validation is most useful with multiple values.\n",
        "  Consider: lambda_seq = c(0.01, 0.1, 0.5, 1.0)",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}


#' Check if JAX/conda environment is available
#'
#' @param conda_env Name of conda environment
#' @param required If TRUE, stops with error; if FALSE, returns FALSE
#' @return TRUE if available, FALSE otherwise (or stops if required)
#' @keywords internal
#' @export
check_jax_available <- function(conda_env = "strategize_env", required = FALSE) {
  # Check if strenv exists and has jnp
  if (exists("strenv", envir = .GlobalEnv) || exists("strenv", envir = parent.frame())) {
    env <- if (exists("strenv", envir = .GlobalEnv)) get("strenv", envir = .GlobalEnv)
           else get("strenv", envir = parent.frame())
    if ("jnp" %in% ls(envir = env)) {
      return(TRUE)
    }
  }

  # Check if reticulate can find the environment
  available <- tryCatch({
    envs <- reticulate::conda_list()
    conda_env %in% envs$name
  }, error = function(e) FALSE)

  if (!available && required) {
    stop(
      sprintf("Conda environment '%s' not found or JAX not initialized.\n\n", conda_env),
      "To set up the environment, run:\n",
      sprintf("  strategize::build_backend(conda_env = '%s')\n\n", conda_env),
      "To check available environments:\n",
      "  reticulate::conda_list()\n\n",
      "If you've already set up the environment, ensure conda is in your PATH.",
      call. = FALSE
    )
  }

  available
}
