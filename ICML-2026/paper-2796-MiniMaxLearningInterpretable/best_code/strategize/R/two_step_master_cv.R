cs_prepare_cv_folds <- function(folds,
                                Y,
                                W,
                                respondent_id = NULL,
                                respondent_task_id = NULL,
                                pair_id = NULL) {
  n <- length(Y)
  if (n < 2L) {
    stop("'Y' must contain at least two observations for cross-validation.", call. = FALSE)
  }
  if (is.null(respondent_id)) {
    respondent_id <- seq_len(n)
  }
  if (is.null(respondent_task_id)) {
    respondent_task_id <- rep(1L, n)
  }
  if (length(respondent_id) != n || length(respondent_task_id) != n) {
    stop("'respondent_id' and 'respondent_task_id' must match length(Y).", call. = FALSE)
  }
  if (!is.null(pair_id) && length(pair_id) != n) {
    stop("'pair_id' must match length(Y).", call. = FALSE)
  }

  task_id <- if (!is.null(pair_id)) {
    if (anyNA(pair_id)) {
      stop("'pair_id' cannot contain missing values when used for cross-validation folds.", call. = FALSE)
    }
    paste0("pair_", as.character(pair_id))
  } else {
    paste0(respondent_id, "_", respondent_task_id)
  }
  task_order <- unique(task_id)

  build_split_matrix <- function(fold_id_obs) {
    if (length(fold_id_obs) != n) {
      stop("Internal fold assignment length mismatch.", call. = FALSE)
    }
    if (any(is.na(fold_id_obs))) {
      stop("'folds' contains missing fold IDs.", call. = FALSE)
    }
    fold_labels <- unique(fold_id_obs)
    if (length(fold_labels) < 2L) {
      stop("'folds' must define at least two folds.", call. = FALSE)
    }
    indi_list <- sapply(fold_labels, function(f_) {
      list(which(fold_id_obs != f_), which(fold_id_obs == f_))
    })
    list(
      indi_list = indi_list,
      n_folds = length(fold_labels),
      fold_id = fold_id_obs,
      fold_labels = fold_labels
    )
  }

  is_scalar_count <- (is.numeric(folds) || is.integer(folds)) && length(folds) == 1L
  if (is_scalar_count) {
    n_folds <- as.integer(folds)
    if (is.na(n_folds) || n_folds < 2L || n_folds != as.numeric(folds)) {
      stop("'folds' must be an integer >= 2 or a fold-assignment vector.", call. = FALSE)
    }

    W_fold <- as.data.frame(W)
    all_tabs <- apply(W_fold, 2, table)
    ok_counter <- 0
    ok <- FALSE
    while(!ok){
      ok_counter <- ok_counter + 1

      task_fold <- sample(seq_len(n_folds), size = length(task_order), replace = TRUE)
      names(task_fold) <- task_order
      fold_id_obs <- task_fold[task_id]
      if (length(unique(fold_id_obs)) < n_folds) {
        next
      }

      split_obj <- build_split_matrix(fold_id_obs)
      indi_list <- split_obj$indi_list
      split_tabs_in <- apply(indi_list, 2, function(l_) {
        apply(W_fold[l_[[1]], , drop = FALSE], 2, table)
      })
      if (all(names(unlist(all_tabs)) == names(unlist(split_tabs_in)))) {
        if (all(unlist(split_tabs_in) > 10)) {
          ok <- TRUE
        }
      }
      if (ok_counter > 1000) {
        stop("Stopping: Could not find split with > 10 observations per factor level.")
      }
    }
    return(split_obj)
  }

  if (!is.atomic(folds)) {
    stop("'folds' must be an integer >= 2 or a fold-assignment vector.", call. = FALSE)
  }
  if (length(folds) == n) {
    fold_id_obs <- folds
    per_task_n <- vapply(split(fold_id_obs, task_id), function(x) {
      length(unique(x))
    }, integer(1))
    if (any(per_task_n != 1L)) {
      unit_label <- if (!is.null(pair_id)) "pair_id" else "respondent-task"
      stop(sprintf("'folds' must assign every %s to exactly one fold.", unit_label), call. = FALSE)
    }
    return(build_split_matrix(fold_id_obs))
  }
  if (length(folds) == length(task_order)) {
    fold_id_task <- folds
    names(fold_id_task) <- task_order
    return(build_split_matrix(fold_id_task[task_id]))
  }

  stop(
    "'folds' must be an integer >= 2, a vector with length(Y) entries, ",
    "or a vector with one entry per unique respondent-task.",
    call. = FALSE
  )
}

#' Cross-validation for Optimal Stochastic Interventions in Conjoint Analysis
#'
#' Performs cross-validation to select the regularization parameter \eqn{\lambda} 
#' (and, if desired, other hyperparameters) for the \code{\link{strategize}} function. 
#' This function splits the data by respondent (or user-specified units), trains
#' candidate models under a grid of \eqn{\lambda} values, and evaluates out-of-sample
#' performance, returning the model that maximizes a chosen criterion (e.g., out-of-sample 
#' expected utility or log-likelihood).
#'
#' @param Y A numeric or binary response vector. If binary (e.g., 0–1), it should 
#'   correspond to forced-choice outcomes (1 if candidate \code{A} is chosen; 0 if 
#'   candidate \code{B} is chosen). If numeric, please see details in 
#'   \code{\link{strategize}} for how outcomes are handled.
#' @param W A data frame or matrix representing the randomized conjoint attributes. 
#'   Each column is a factor or character vector indicating attribute levels for a 
#'   particular dimension. Multiple columns can be used if the conjoint has multiple
#'   attributes. 
#' @param X Optional covariate matrix or data frame for modeling systematic 
#'   heterogeneity. If \code{K > 1}, this is typically required for multi-class or 
#'   cluster-based models. Otherwise, set \code{X = NULL}.
#' @param lambda_seq A numeric vector of candidate \eqn{\lambda} values for 
#'   cross-validation. If \code{NULL} and \code{lambda} is also \code{NULL}, 
#'   a sequence of values is automatically generated (e.g., via 
#'   \code{10^seq(-4, 0, length.out = 5) * sd(Y)}).
#' @param lambda A single user-specified \eqn{\lambda} value. If provided, 
#'   cross-validation is effectively disabled unless \code{lambda_seq} is also 
#'   supplied. 
#' @param folds Either an integer number of cross-validation folds, a vector with
#'   one fold ID per observation, or a vector with one fold ID per unique
#'   respondent-task in first-seen order. Defaults to 2. Observation-level fold
#'   vectors must assign all rows from the same respondent-task to one fold.
#' @param crossfit_q Logical. If \code{TRUE}, compute \code{Q_crossfit},
#'   \code{Q_reference_crossfit}, \code{Q_gain_crossfit}, and
#'   \code{Q_crossfit_info} on the final refit after lambda selection. Supported
#'   for non-adversarial pairwise average-case binomial GLM runs and adversarial
#'   pairwise binomial GLM runs with \code{adversarial_model_strategy = "four"}.
#' @param crossfit_q_control Optional list passed to \code{\link{strategize}} to
#'   control cross-fitted Q evaluation, including the optional adversarial
#'   \code{perspective_group} entry.
#' @param varcov_cluster_variable An optional clustering variable for robust standard 
#'   errors. For instance, if the data is from multiple respondents, specify respondent 
#'   IDs here for cluster-robust inference (via sandwich estimation). If \code{NULL}, no 
#'   cluster-based variance correction is used.
#' @param competing_group_variable_respondent Optional vector for multi-round or 
#'   multi-group setups, indicating which respondent belongs to which group. Used for 
#'   advanced or adversarial designs (e.g., dual-party contexts). If \code{NULL}, 
#'   standard usage is assumed.
#' @param competing_group_variable_candidate Similar to 
#'   \code{competing_group_variable_respondent}, but for candidate-level grouping. 
#'   If \code{NULL}, standard usage is assumed.
#' @param competing_group_competition_variable_candidate An optional variable for 
#'   specifying which candidate is in competition with which group. Relevant if 
#'   multi-step adversarial frameworks are used.
#' @param pair_id An optional vector (same length as \code{Y}) identifying which 
#'   rows (candidate pairs) belong to the same forced choice. For example, if each 
#'   respondent evaluates multiple pairs, this ID ensures correct grouping. 
#'   Required only in certain advanced difference-in-differences or paired analyses.
#' @param respondent_id A user-specified ID to denote respondent-level grouping, 
#'   typically used to cluster standard errors or to perform out-of-sample validation 
#'   by respondent. If \code{NULL}, a simple row index is used for splitting.
#' @param respondent_task_id Another optional ID for tasks (e.g., each respondent 
#'   might see multiple tasks). Helps in advanced designs. If \code{NULL}, ignored.
#' @param profile_order An optional vector capturing the ordering of candidate 
#'   profiles within tasks, if multiple profiles are being shown. Used in difference 
#'   or extended hierarchical modeling.
#' @param p_list A list of assignment probabilities for each attribute, if known 
#'   or desired as a baseline. If \code{NULL}, each level is assumed to have 
#'   uniform probability or derived from empirical frequencies in \code{W}.
#' @param slate_list An optional list specifying alternative or restricted sets of 
#'   attribute levels. Used when a subset of attributes is feasible or when bounding 
#'   certain strategies in an adversarial design.
#' @param use_optax Logical. If \code{TRUE}, uses the \pkg{optax} Python library 
#'   (via \pkg{reticulate}) for gradient-based optimization. If \code{FALSE}, uses 
#'   a default gradient-based approach from \pkg{jax}.
#' @param K An integer specifying the number of mixture components or clusters if 
#'   \code{X} is used (e.g., for multi-class analysis). Defaults to 1 (no mixture).
#' @param nSGD An integer number of iterations for gradient-based training. Defaults 
#'   to 100 but can be increased if convergence has not been reached.
#' @param diff Logical indicating whether a difference-based model (e.g., for 
#'   forced-choice or difference-in-outcomes) is used. Defaults to \code{FALSE}, but 
#'   set \code{TRUE} in certain difference-of-utility designs.
#' @param adversarial Logical indicating whether to use a two-party or multi-agent 
#'   \emph{adversarial} approach in the optimization. If \code{TRUE}, a min-max 
#'   (zero-sum) formulation is employed. Defaults to \code{FALSE} (single-agent 
#'   or average-case optimization).
#' @param adversarial_model_strategy Character string indicating whether to estimate
#'   \code{"four"} outcome models (primary + general for each group), \code{"two"} outcome models
#'   (one per group reused for both primary and general), or \code{"neural"} (Bayesian Transformer
#'   models with party tokens; defaults to a single pooled model across groups and stages. Set
#'   \code{neural_mcmc_control$n_bayesian_models = 2} to fit separate AST/DAG models) in adversarial
#'   mode.
#' @param partial_pooling Logical indicating whether to partially pool (shrink) group-specific
#'   outcome model coefficients toward a shared average when using the "two" strategy. When
#'   \code{NULL}, defaults to \code{TRUE} in the two-strategy adversarial case.
#' @param partial_pooling_strength Numeric scalar controlling the amount of shrinkage used for
#'   partial pooling in the two-strategy adversarial case.
#' @param use_regularization Logical; if \code{TRUE}, penalty-based regularization
#'   is used for the outcome model. Usually set to \code{TRUE} for large designs.
#'   Defaults to \code{TRUE}.
#' @param force_gaussian Logical indicating whether a Gaussian family
#'   (\code{lm}-style) is forced for the outcome model, even if \code{Y} is binary.
#'   Defaults to \code{FALSE}.
#' @param force_reinforce Logical indicating whether to force REINFORCE-based optimization for
#'   the neural average-case objective when exact small-support enumeration is available. This
#'   affects the optimization objective only; final reported \code{Q} values still use the
#'   standard report-time evaluation path. Defaults to \code{FALSE}.
#' @param temperature Optional numeric temperature controlling the smoothness of
#'   Gumbel-Softmax sampling when exploring probability vectors. Smaller values
#'   lead to distributions closer to the argmax. Defaults to \code{NULL}, which
#'   allows internal defaults.
#' @param a_init_sd A numeric controlling the random initialization scale for
#'   unconstrained parameters in gradient-based optimization. Defaults to 0.001.
#'   Larger values can help avoid local minima in complex outcome landscapes.
#' @param learning_rate_max Base learning rate for gradient-based optimizers.
#'   Defaults to \code{0.001}.
#' @param penalty_type A character string specifying the type of penalty for the
#'   \emph{optimal stochastic intervention}, e.g., \code{"KL"}, \code{"L2"},
#'   or \code{"LogMaxProb"}. The default is \code{"KL"}.
#' @param outcome_model_type Character string indicating the outcome model to
#'   use, such as \code{"glm"} for generalized linear models or \code{"neural"}
#'   for a neural-network approximation. Defaults to \code{"glm"}.
#' @param neural_mcmc_control Optional list overriding default MCMC settings used when
#'   \code{outcome_model_type = "neural"}. Use
#'   \code{neural_mcmc_control$uncertainty_scope = "output"} to restrict delta-method
#'   uncertainty to output-layer parameters. In adversarial neural mode, set
#'   \code{neural_mcmc_control$n_bayesian_models = 2} to fit separate AST/DAG models
#'   (default is 1 for a single differential model). Use
#'   \code{neural_mcmc_control$ModelDims} and \code{neural_mcmc_control$ModelDepth}
#'   to override the Transformer hidden width and depth. Pairwise neural fits default to
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
#'   \code{neural_mcmc_control$svi_steps} (integer steps, or \code{"optimal"} for
#'   a scaling-law heuristic based on model/data size; for minibatched VI this
#'   also scales with \code{batch_size}) and
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
#' @param compute_se Logical; if \code{TRUE}, attempts to compute standard errors
#'   using M-estimation or the Delta method. Defaults to \code{TRUE}.
#' @param conda_env A character specifying the name of a Conda environment for
#'   the JAX-backed optimization workflow. Defaults to \code{"strategize_env"}.
#' @param conda_env_required Logical. If \code{TRUE}, errors if the specified 
#'   Conda environment \code{conda_env} cannot be found. Otherwise tries to fall 
#'   back gracefully.
#' @param conf_level The confidence level (between 0 and 1) for interval estimation, 
#'   default 0.90.
#' @param nFolds_glm An integer specifying the number of folds in internal 
#'   regression-based cross-validation (if used) for outcome model selection. 
#'   Defaults to 3.
#' @param nMonte_adversarial A positive integer specifying the number of Monte Carlo 
#'   draws for the min-max (adversarial) stage, if \code{adversarial = TRUE}. Defaults 
#'   to 5.
#' @param primary_pushforward Character string controlling the primary-stage push-forward estimator.
#'   Use \code{"mc"} (default) for Monte Carlo sampling with per-draw primary winners, or
#'   \code{"linearized"} for the faster averaged-weight approximation, or
#'   \code{"multi"} for multi-candidate primaries.
#' @param primary_strength Numeric scalar controlling primary decisiveness (see \code{\link{strategize}}).
#' @param primary_n_entrants Integer number of entrant candidates per party in multi-candidate primaries.
#' @param primary_n_field Integer number of field candidates per party in multi-candidate primaries.
#' @param nMonte_Qglm An integer specifying the number of Monte Carlo draws 
#'   for evaluating certain integrals in \code{glm}-based approximations, default 100.
#' @param optim_type A character describing the optimization routine. Typically 
#'   \code{"default"} uses a standard gradient-based approach; set \code{"tryboth"} 
#'   or \code{"SecondOrder"} for testing or advanced routines.
#' @param optimism Character string controlling optimistic / extra-gradient updates for the
#'   gradient optimizer. Options: \code{"extragrad"} (default), \code{"smp"} (stochastic mirror-prox),
#'   \code{"ogda"}, \code{"rain"} (RAIN: Recursive Anchored Iteration with anchored extra-gradient and
#'   increasing quadratic anchor penalties),
#'   or \code{"none"}. Only supported when \code{use_optax = FALSE}.
#' @param optimism_coef Numeric scalar controlling the magnitude of optimism adjustments. For
#'   \code{"ogda"}, this scales the optimistic correction term. Ignored when
#'   \code{optimism = "rain"}.
#' @param rain_lambda Numeric scalar giving the base RAIN regularization scale \eqn{\lambda}.
#'   The staged algorithm uses \eqn{\lambda_0 = \gamma \lambda} and grows by
#'   \eqn{(1+\gamma)} each stage. Only used when \code{optimism = "rain"}.
#' @param rain_gamma Non-negative numeric scalar for the RAIN anchor-growth parameter \eqn{\gamma}.
#'   If not supplied, the default is auto-scaled downward when \code{nSGD} exceeds 100
#'   to keep total anchor growth roughly constant. Only used when \code{optimism = "rain"}.
#' @param rain_L Optional numeric scalar for the Lipschitz constant \eqn{L} used by RAIN.
#'   When supplied and \code{rain_eta} is missing, \code{rain_eta} defaults to
#'   \eqn{1/(8L)} and stage lengths follow \eqn{T_s = 16L/\lambda_s}. If \code{NULL},
#'   \eqn{L} is estimated from \code{rain_eta}. Only used when \code{optimism = "rain"}.
#' @param rain_eta Optional numeric scalar step size \eqn{\eta} for RAIN. Defaults to
#'   \code{0.001} and is auto-scaled downward when \code{nSGD} exceeds 100 if not supplied.
#'   If \code{rain_L} is supplied and \code{rain_eta} is missing, defaults to
#'   \eqn{1/(8L)}. Only used when \code{optimism = "rain"}.
#' @param rain_variant Character string specifying the RAIN variant. Options:
#'   \code{"alg10_staged"} (merged Algorithm 10 with recursive stage anchors; default) or
#'   \code{"alg9_single_loop"} (single-loop variant; not yet implemented). Only used when
#'   \code{optimism = "rain"}.
#' @param rain_output Character string controlling the stage output for RAIN.
#'   \code{"uniform_half"} samples uniformly from half-iterates within the stage (most faithful);
#'   \code{"last"} returns the last iterate (default). Only used when \code{optimism = "rain"}.
#' 
#' @details
#' \code{cv_strategize} implements a cross-validation routine for
#' \code{\link{strategize}}. First, the data is split into \code{folds} parts. 
#' For each fold, we train candidate outcome models and compute out-of-sample 
#' performance. The best-performing \eqn{\lambda} is selected. Finally, a 
#' refit on the full data is done using the chosen hyperparameters, returning 
#' the results of the final \code{\link{strategize}} call with \eqn{\lambda} set 
#' to the best value.
#'
#' The function supports a wide range of conjoints, including forced-choice 
#' (where \code{diff = TRUE}), multi-cluster outcome modeling (where \eqn{K > 1}), 
#' and adversarial designs (where \code{adversarial = TRUE}). Regularization for the 
#' outcome model or for the candidate distribution can be enabled via 
#' \code{use_regularization} and \code{penalty_type}. Cross-validation is particularly 
#' helpful when the data is limited or highly dimensional.
#'
#' @return A named list with components:
#' \describe{
#'   \item{pi_star_point}{The estimated optimal probability distribution(s) over 
#'   candidate profiles (\eqn{\hat{\boldsymbol{\pi}}^*}).}
#'
#'   \item{Q_point}{The primary estimated expected outcome (e.g., vote share)
#'   under the selected optimal distribution.}
#'
#'   \item{Q_se}{The primary standard error for \code{Q_point}, when standard
#'   errors are requested.}
#'
#'   \item{Q_point_mEst, Q_se_mEst}{Backward-compatible aliases for
#'   \code{Q_point} and \code{Q_se}.}
#'
#'   \item{lambda}{The chosen \eqn{\lambda} value from cross-validation (and 
#'   any other relevant hyperparameters).}
#'
#'   \item{CVInfo}{A data frame or matrix summarizing cross-validation results, 
#'   e.g., in-sample and out-of-sample estimates for each candidate \eqn{\lambda}.}
#'
#'   \item{Q_crossfit, Q_reference_crossfit, Q_gain_crossfit, Q_crossfit_info}{Optional
#'   final-refit cross-fitted policy-value diagnostics when \code{crossfit_q = TRUE}.}
#'
#'   \item{Other components}{Various additional objects useful for inference and 
#'   debugging (e.g., final model fits, standard error estimates, weighting 
#'   details).}
#' }
#'
#' @seealso 
#' \code{\link{strategize}} for direct optimization of stochastic interventions 
#' in conjoint analysis, including average and adversarial settings. 
#'
#' @examples
#' \donttest{
#' # ================================================
#' # Cross-validation to select regularization lambda
#' # ================================================
#' set.seed(123)
#' n <- 400  # profiles (200 pairs)
#'
#' # Generate factor matrix
#' W <- data.frame(
#'   Gender = sample(c("Male", "Female"), n, replace = TRUE),
#'   Age = sample(c("35", "50", "65"), n, replace = TRUE),
#'   Party = sample(c("Dem", "Rep"), n, replace = TRUE)
#' )
#'
#' # Simulate outcome with true effects
#' latent <- 0.2 * (W$Gender == "Female") + 0.15 * (W$Age == "35")
#' prob <- plogis(latent)
#'
#' # Create paired forced-choice structure
#' pair_id <- rep(1:(n/2), each = 2)
#' Y <- numeric(n)
#' for (p in unique(pair_id)) {
#'   idx <- which(pair_id == p)
#'   winner <- sample(idx, 1, prob = prob[idx])
#'   Y[idx] <- as.integer(seq_along(idx) == which(idx == winner))
#' }
#' profile_order <- rep(1:2, n/2)
#'
#' # Cross-validate over lambda values
#' # Lower lambda = less regularization = further from baseline
#' cv_result <- cv_strategize(
#'   Y = Y,
#'   W = W,
#'   lambda_seq = c(0.01, 0.1, 0.5, 1.0),
#'   folds = 2,
#'   pair_id = pair_id,
#'   respondent_id = pair_id,
#'   profile_order = profile_order,
#'   diff = TRUE,
#'   nSGD = 50,
#'   compute_se = FALSE
#' )
#'
#' # View CV results and selected lambda
#' print(cv_result$lambda)       # Optimal lambda
#' print(cv_result$CVInfo)       # Performance at each lambda
#' print(cv_result$pi_star_point) # Optimal distribution
#' print(cv_result$Q_point)       # Expected outcome
#' }
#'
#' @export

cv_strategize       <-          function(
                                            Y,
                                            W,
                                            X = NULL,
                                            lambda_seq = NULL,
                                            lambda = NULL,
                                            folds = 2L,
                                            crossfit_q = FALSE,
                                            crossfit_q_control = NULL,
                                            varcov_cluster_variable = NULL,
                                            competing_group_variable_respondent = NULL,
                                            competing_group_variable_candidate = NULL,
                                            competing_group_competition_variable_candidate = NULL,
                                            pair_id = NULL,
                                            respondent_id = NULL,
                                            respondent_task_id = NULL,
                                            profile_order = NULL,
                                            p_list = NULL,
                                            slate_list = NULL,
                                            use_optax = F,
                                            K = 1,
                                            nSGD = 100,
                                            diff = F,
                                            adversarial = F,
                                            adversarial_model_strategy = "four",
                                            partial_pooling = NULL,
                                            partial_pooling_strength = 50,
                                            use_regularization = TRUE,
                                            force_gaussian = F,
                                            force_reinforce = FALSE,
                                            temperature = NULL,
                                            a_init_sd = 0.001,
                                            learning_rate_max = 0.001, 
                                            penalty_type = "KL",
                                            outcome_model_type = "glm",
                                            neural_mcmc_control = NULL,
                                            compute_se = T,
                                            conda_env = "strategize_env",
                                            conda_env_required = F,
                                            conf_level = 0.90,
                                            nFolds_glm = 3L,
                                            nMonte_adversarial = 5L,
                                            primary_pushforward = "mc",
                                            primary_strength = 1.0,
                                            primary_n_entrants = 1L,
                                            primary_n_field = 1L,
                                            nMonte_Qglm = 100L,
                                            optim_type = "gd",
                                            optimism = "extragrad",
                                            optimism_coef = 1,
                                            rain_lambda = 1,
                                            rain_gamma = 0.01,
                                            rain_L = NULL,
                                            rain_eta = 0.001,
                                            rain_variant = "alg10_staged",
                                            rain_output = "last"){
  optimism <- match.arg(optimism, c("none", "ogda", "extragrad", "smp", "rain"))
  if (optimism == "rain") {
    rain_variant <- match.arg(rain_variant, c("alg10_staged", "alg9_single_loop"))
    rain_output <- match.arg(rain_output, c("uniform_half", "last"))
    if (!missing(optimism_coef) && !is.null(optimism_coef)) {
      warning("'optimism_coef' is ignored when optimism = \"rain\"; use rain_lambda instead.", call. = FALSE)
    }
  }
  if (use_optax && optimism != "none") {
    stop("Optimistic / extra-gradient updates are only available when use_optax = FALSE.")
  }
  if (!is.logical(force_reinforce) || length(force_reinforce) != 1L || is.na(force_reinforce)) {
    stop("'force_reinforce' must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(crossfit_q) || length(crossfit_q) != 1L || is.na(crossfit_q)) {
    stop("'crossfit_q' must be TRUE or FALSE.", call. = FALSE)
  }

  autoscale_rain_gamma <- missing(rain_gamma)
  autoscale_rain_eta <- missing(rain_eta)
  if (missing(rain_eta) && !is.null(rain_L)) {
    rain_eta <- 1 / (8 * as.numeric(rain_L))
    autoscale_rain_eta <- FALSE
  }
  if (!is.null(rain_L)) {
    rain_L <- as.numeric(rain_L)
  }
  # initialize environment
  if(!"jnp" %in% ls(envir = strenv)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required) 
  }

  # setup lambda
  if(is.null(lambda_seq) & is.null(lambda)){
    lambda_seq <- 10^seq(-4, 0, length.out = 5) * sd(Y, na.rm = T)
  }
  if(is.null(lambda_seq) & !is.null(lambda)){ lambda_seq <- lambda }

  if(is.null(respondent_id)){ respondent_id <- 1:length(Y) }
  if(is.null(respondent_task_id)){ respondent_task_id <- rep(1L, length(Y)) }
  pair_id_for_folds <- if (isTRUE(diff) && !is.null(pair_id)) pair_id else NULL

  subset_X_for_cv <- function(rows = NULL) {
    if (is.null(X)) {
      return(NULL)
    }
    if (is.null(rows)) {
      return(X)
    }
    if (is.null(dim(X))) {
      return(X[rows])
    }
    X[rows, , drop = FALSE]
  }

  # Ensure p_list is computed from full data for consistent dimensions across folds
  # This prevents dimension mismatches when different CV folds see different factor levels
  if(is.null(p_list)){
    p_list <- cs_default_p_list(W = W, threshold = 0.1, warn = TRUE, factor_names = colnames(W))
  }

  # CV sequence
  {
    message("Starting CV sequence...")
    outsamp_results <- insamp_results <- matrix(nrow = 0, ncol = 4, dimnames = list(NULL, c("lambda","Qhat","Qse","selected")))

    # build cv splits - same for all lambda
    cv_fold_obj <- cs_prepare_cv_folds(
      folds = folds,
      Y = Y,
      W = W,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      pair_id = pair_id_for_folds
    )
    indi_list <- cv_fold_obj$indi_list
    folds_use <- cv_fold_obj$n_folds
    
    lambda_counter <- 0; for(lambda__ in lambda_seq){
      lambda_counter <- lambda_counter + 1
      Qoptimized__ <- replicate(n = folds_use, list())
      message(sprintf("At lambda %s of %s...", lambda_counter, length(lambda_seq)))

      # CV sequence
      q_vec_in <- q_vec_out <- c()
      for(split_ in seq_len(folds_use)){
        message(sprintf("On fold %s",split_))
        for(type_ in c(1,2)){ 
          # in sample optimization of pi*, evaluation on OOS coefficients 
          use_indices <- indi_list[type_,split_][[1]]
          nSGD_use <- ifelse(type_ == 1, yes = nSGD, no = 1L)
          
          if(type_ == 1){type_<-1}
          if(type_ == 2){type_<-2}
          
          # strategize call
          strategize_args <- list(
            # input data
            Y = Y[use_indices],
            W = W[use_indices,],
            X = subset_X_for_cv(use_indices),
            varcov_cluster_variable = varcov_cluster_variable[use_indices],
            pair_id = pair_id[use_indices],
            respondent_id = respondent_id[ use_indices ],
            respondent_task_id = respondent_task_id[ use_indices ],
            profile_order = profile_order[ use_indices ],
            competing_group_variable_respondent = if(!is.null(competing_group_variable_respondent)) competing_group_variable_respondent[use_indices] else NULL,
            competing_group_variable_candidate = if(!is.null(competing_group_variable_candidate)) competing_group_variable_candidate[use_indices] else NULL,
            competing_group_competition_variable_candidate = if(!is.null(competing_group_competition_variable_candidate)) competing_group_competition_variable_candidate[use_indices] else NULL,
            p_list = p_list,
            slate_list = slate_list,
            use_optax = use_optax,
            lambda = lambda__,

            # hyperparameters
            outcome_model_type = outcome_model_type,
            neural_mcmc_control = neural_mcmc_control,
            temperature = temperature,
            compute_se = F,
            nSGD = nSGD_use,
            penalty_type = penalty_type,
            K = K,
            force_gaussian = force_gaussian,
            force_reinforce = force_reinforce,
            use_regularization = use_regularization,
            optim_type = optim_type,
            optimism = optimism,
            a_init_sd = a_init_sd,
            nFolds_glm = nFolds_glm,
            nMonte_adversarial = nMonte_adversarial,
            primary_pushforward = primary_pushforward,
            primary_strength = primary_strength,
            primary_n_entrants = primary_n_entrants,
            primary_n_field = primary_n_field,
            adversarial_model_strategy = adversarial_model_strategy,
            partial_pooling = partial_pooling,
            partial_pooling_strength = partial_pooling_strength,
            diff = diff,
            adversarial = adversarial,
            conda_env = conda_env,
            conda_env_required = conda_env_required
          )
          if (!missing(optimism_coef)) {
            strategize_args$optimism_coef <- optimism_coef
          }
          if (!missing(rain_lambda)) {
            strategize_args$rain_lambda <- rain_lambda
          }
          if (!missing(rain_L) && !is.null(rain_L)) {
            strategize_args$rain_L <- rain_L
          }
          if (!missing(rain_variant)) {
            strategize_args$rain_variant <- rain_variant
          }
          if (!missing(rain_output)) {
            strategize_args$rain_output <- rain_output
          }
          if (!autoscale_rain_gamma) {
            strategize_args$rain_gamma <- rain_gamma
          }
          if (!autoscale_rain_eta) {
            strategize_args$rain_eta <- rain_eta
          }
          Qoptimized__[[split_]][[type_]] <- do.call(strategize, strategize_args)
        }
        
        # out of sample test of pi* on new estimates 
        q_vec_in <- c(q_vec_in, Qoptimized__[[split_]][[1]]$Q_point)
        q_vec_out <- c(q_vec_out, 
          unlist(Qoptimized__[[split_]][[2]]$QFXN(
          "pi_star_ast" = Qoptimized__[[split_]][[1]]$pi_star_red_ast,
          "pi_star_dag" = Qoptimized__[[split_]][[1]]$pi_star_red_dag,
          "EST_INTERCEPT_tf_ast" = Qoptimized__[[split_]][[2]]$est_intercept_jnp,
          "EST_COEFFICIENTS_tf_ast" = Qoptimized__[[split_]][[2]]$est_coefficients_jnp,
          "EST_INTERCEPT_tf_dag" = Qoptimized__[[split_]][[2]]$est_intercept_jnp,
          "EST_COEFFICIENTS_tf_dag" = Qoptimized__[[split_]][[2]]$est_coefficients_jnp
          )$tolist()[[1]])
        )
      }
      outsamp_results <- as.data.frame(rbind(outsamp_results, 
                                             c(lambda__, mean(q_vec_out), se(q_vec_out), 0)))
      insamp_results <- as.data.frame(rbind(insamp_results, 
                                            c(lambda__, mean(q_vec_in), se(q_vec_in), 0)))
    }

    # Use the requested confidence level to build SE-based bounds
    qStar_lambda <- stats::qnorm(1 - (1 - conf_level)/2)
    outsamp_results$l_bound <- outsamp_results$Qhat - qStar_lambda * outsamp_results$Qse
    outsamp_results$u_bound <- outsamp_results$Qhat + qStar_lambda * outsamp_results$Qse
    lambda__ <- lambda_seq[which_selected <- which.max(outsamp_results$Qhat)] # lambda.min rule
    #lambda__ <- lambda_seq[which_selected <- which(outsamp_results$Qhat >= max(outsamp_results$l_bound))[1]] # lambda.se rule
    outsamp_results$selected[which_selected] <- 1 
    message(sprintf("Done with CV sequence & starting final run with log(lambda) of %.2f...",
                    log(f2n(lambda__))))
  }

  # final output
  gc(); strenv$py_gc$collect();
  strategize_args <- list(
    # input data
    Y = Y,
    W = W,
    X = subset_X_for_cv(),
    nSGD = nSGD,
    penalty_type = penalty_type,
    varcov_cluster_variable = varcov_cluster_variable,
    competing_group_variable_respondent = competing_group_variable_respondent,
    competing_group_variable_candidate = competing_group_variable_candidate,
    competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    p_list = p_list,
    slate_list = slate_list, 
    use_optax = use_optax, 
    lambda = lambda__, # this lambda is the one chosen via CV
    crossfit_q = crossfit_q,
    crossfit_q_control = crossfit_q_control,

    # hyperparameters
    outcome_model_type = outcome_model_type,
    neural_mcmc_control = neural_mcmc_control,
    temperature = temperature,
    optim_type = optim_type,
    optimism = optimism,
    force_gaussian = force_gaussian,
    force_reinforce = force_reinforce,
    use_regularization = use_regularization,
    a_init_sd = a_init_sd,
    compute_se = compute_se,
    K = K,
    nMonte_adversarial = nMonte_adversarial,
    primary_pushforward = primary_pushforward,
    primary_strength = primary_strength,
    primary_n_entrants = primary_n_entrants,
    primary_n_field = primary_n_field,
    adversarial_model_strategy = adversarial_model_strategy,
    partial_pooling = partial_pooling,
    partial_pooling_strength = partial_pooling_strength,
    nFolds_glm = nFolds_glm,
    diff = diff,
    adversarial = adversarial,
    conda_env = conda_env,
    conda_env_required = conda_env_required
  )
  if (!missing(optimism_coef)) {
    strategize_args$optimism_coef <- optimism_coef
  }
  if (!missing(rain_lambda)) {
    strategize_args$rain_lambda <- rain_lambda
  }
  if (!missing(rain_L) && !is.null(rain_L)) {
    strategize_args$rain_L <- rain_L
  }
  if (!missing(rain_variant)) {
    strategize_args$rain_variant <- rain_variant
  }
  if (!missing(rain_output)) {
    strategize_args$rain_output <- rain_output
  }
  if (!autoscale_rain_gamma) {
    strategize_args$rain_gamma <- rain_gamma
  }
  if (!autoscale_rain_eta) {
    strategize_args$rain_eta <- rain_eta
  }
  Qoptimized_ <- do.call(strategize, strategize_args)
  message("Done with strategic analysis!")
  return(c(Qoptimized_,
           list(lambda = lambda__,
                qStar_lambda = qStar_lambda,
                CVInfo = outsamp_results)))
}
