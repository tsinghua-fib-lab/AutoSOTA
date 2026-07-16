# Validation of the soft (mean-embedding) policy relaxation against
# hard-profile Monte Carlo evaluation.
#
# BACKGROUND. strategize() optimizes pi by differentiating through a SOFT
# candidate encoding: for each factor, the level embeddings are mixed by pi
# BEFORE the SwiGLU fusion MLP and the transformer, i.e. the objective is
# Q_soft(pi) = NN(tokens(E_pi[embedding])), a mean-embedding relaxation of the
# true policy value Q(pi) = E_{w ~ pi}[NN(tokens(w))]. The relaxation is what
# makes Q differentiable in pi (the reason the soft path exists) and is exact
# at one-hot pi, but at interior points of the simplex -- exactly where
# stochastic interventions live -- it carries a Jensen-type gap through every
# downstream nonlinearity. These helpers quantify that gap at a candidate
# pi (typically pi*) by comparing the soft prediction with an average of HARD
# predictions over profiles sampled from pi. The Monte Carlo evaluation is
# report-time only: it never enters the optimization loop and needs no
# gradients.

# Split a flat soft-path probability vector into per-factor simplex blocks.
neural_split_pi_blocks <- function(pi_num, factor_levels) {
  factor_levels <- as.integer(factor_levels)
  if (length(pi_num) != sum(factor_levels)) {
    stop(sprintf(paste0(
      "pi vector has length %d but the model's factor levels sum to %d. ",
      "Pass the flat probability vector in the soft-path layout ",
      "(contiguous per-factor blocks, full simplex per factor)."
    ), length(pi_num), sum(factor_levels)), call. = FALSE)
  }
  ends <- cumsum(factor_levels)
  starts <- c(1L, head(ends, -1L) + 1L)
  lapply(seq_along(factor_levels), function(d_) {
    block <- pi_num[starts[d_]:ends[d_]]
    block[!is.finite(block) | block < 0] <- 0
    total <- sum(block)
    if (total <= 0) {
      block <- rep(1 / length(block), length(block))
    } else {
      block <- block / total
    }
    block
  })
}

# Sample n_draws hard profiles (0-based level index matrix) from pi blocks.
neural_sample_profiles_from_pi <- function(pi_blocks, n_draws) {
  n_factors <- length(pi_blocks)
  X_idx <- matrix(0L, nrow = n_draws, ncol = n_factors)
  for (d_ in seq_len(n_factors)) {
    block <- pi_blocks[[d_]]
    X_idx[, d_] <- sample.int(length(block), size = n_draws,
                              replace = TRUE, prob = block) - 1L
  }
  X_idx
}

# Core comparator. pi_vec (and pi_vec_dag for pairwise) are flat numeric
# probability vectors in the soft-path layout. Returns soft value, hard MC
# mean, MC standard error, and the gap.
neural_soft_hard_gap <- function(model_info,
                                 pi_vec,
                                 pi_vec_dag = NULL,
                                 party_idx = NULL,
                                 party_idx_dag = NULL,
                                 resp_party_idx = NULL,
                                 resp_cov_vec = NULL,
                                 params = NULL,
                                 n_draws = 200L,
                                 seed = 123L) {
  if (is.null(model_info)) {
    stop("neural_soft_hard_gap requires a neural model_info object.", call. = FALSE)
  }
  if (is.null(params)) {
    params <- model_info$params
  }
  n_draws <- max(2L, as.integer(n_draws))
  factor_levels <- as.integer(model_info$factor_levels)
  pi_num <- as.numeric(reticulate::py_to_r(strenv$np$array(strenv$jnp$asarray(pi_vec))))
  pi_blocks <- neural_split_pi_blocks(pi_num, factor_levels)
  pairwise <- !is.null(pi_vec_dag)
  if (isTRUE(pairwise)) {
    pi_dag_num <- as.numeric(reticulate::py_to_r(strenv$np$array(strenv$jnp$asarray(pi_vec_dag))))
    pi_dag_blocks <- neural_split_pi_blocks(pi_dag_num, factor_levels)
  }
  if (is.null(party_idx)) {
    party_idx <- neural_model_party_missing_index(model_info)
  }
  if (is.null(party_idx_dag)) {
    party_idx_dag <- party_idx
  }
  if (is.null(resp_party_idx)) {
    resp_party_idx <- neural_model_resp_party_missing_index(model_info)
  }

  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))

  rep_vec <- function(v) rep(as.integer(v)[[1L]], n_draws)
  resp_c_use <- NULL
  if (!is.null(resp_cov_vec)) {
    resp_c_num <- as.numeric(reticulate::py_to_r(strenv$np$array(strenv$jnp$asarray(resp_cov_vec))))
    resp_c_use <- matrix(rep(resp_c_num, each = n_draws), nrow = n_draws)
  }

  if (isTRUE(pairwise)) {
    Xl <- neural_sample_profiles_from_pi(pi_blocks, n_draws)
    Xr <- neural_sample_profiles_from_pi(pi_dag_blocks, n_draws)
    hard <- neural_predict_pair_core_prepared(
      params = params,
      model_info = model_info,
      Xl = Xl,
      Xr = Xr,
      pl = rep_vec(party_idx),
      pr = rep_vec(party_idx_dag),
      resp_p = rep_vec(resp_party_idx),
      resp_c = resp_c_use
    )
    soft <- neural_predict_pair_soft(
      strenv$jnp$asarray(pi_num),
      strenv$jnp$asarray(pi_dag_num),
      party_idx, party_idx_dag,
      resp_party_idx,
      model_info,
      resp_cov_vec = resp_cov_vec,
      params = params
    )
  } else {
    Xb <- neural_sample_profiles_from_pi(pi_blocks, n_draws)
    hard <- neural_predict_single_core_prepared(
      params = params,
      model_info = model_info,
      Xb = Xb,
      party_idx = rep_vec(party_idx),
      resp_p = rep_vec(resp_party_idx),
      resp_c = resp_c_use
    )
    soft <- neural_predict_single_soft(
      strenv$jnp$asarray(pi_num),
      party_idx,
      resp_party_idx,
      model_info,
      resp_cov_vec = resp_cov_vec,
      params = params
    )
  }
  hard_num <- as.numeric(reticulate::py_to_r(strenv$np$array(strenv$jnp$asarray(hard))))
  soft_num <- as.numeric(reticulate::py_to_r(strenv$np$array(strenv$jnp$asarray(soft))))[[1L]]
  hard_mean <- mean(hard_num, na.rm = TRUE)
  hard_se <- stats::sd(hard_num, na.rm = TRUE) / sqrt(sum(is.finite(hard_num)))
  list(
    q_soft = soft_num,
    q_hard_mc = hard_mean,
    mc_se = hard_se,
    gap = soft_num - hard_mean,
    n_draws = n_draws,
    pairwise = isTRUE(pairwise)
  )
}

#' Validate the soft policy relaxation at an optimized distribution
#'
#' @description
#' \code{strategize()} optimizes treatment-assignment probabilities by
#' differentiating a SOFT candidate encoding (level embeddings mixed by
#' \code{pi} before the fusion network). That mean-embedding relaxation is
#' what makes the objective differentiable, and it is exact when \code{pi} is
#' one-hot, but at interior points it approximates the true policy value
#' \eqn{E_{w \sim \pi}[\hat{Y}(w)]} with a Jensen-type gap.
#' \code{validate_soft_relaxation()} quantifies the gap at the returned
#' optimum: it samples \code{n_draws} hard profiles from \code{pi_star},
#' averages the model's hard-path predictions, and compares that Monte Carlo
#' estimate with the soft prediction the optimizer used. A gap that is large
#' relative to \code{mc_se} means the reported \code{Q} should be interpreted
#' as the soft surrogate, not the achievable randomized-design value.
#'
#' @param result A \code{strategize_result} object fit with
#'   \code{outcome_model_type = "neural"}.
#' @param n_draws Number of hard profiles to sample per distribution.
#' @param seed Integer seed for profile sampling.
#'
#' @return A list with one entry per available player (\code{ast}, and
#'   \code{dag} in adversarial mode), each containing \code{q_soft},
#'   \code{q_hard_mc}, \code{mc_se}, \code{gap}, and \code{n_draws}.
#' @export
validate_soft_relaxation <- function(result, n_draws = 200L, seed = 123L) {
  if (is.null(result$neural_model_info) ||
      (is.null(result$neural_model_info$ast) && is.null(result$neural_model_info$ast0))) {
    stop(
      "validate_soft_relaxation() requires a strategize result fit with outcome_model_type = 'neural'.",
      call. = FALSE
    )
  }
  model_info <- result$neural_model_info$ast %||% result$neural_model_info$ast0
  pi_star <- result$pi_star_point
  if (is.null(pi_star)) {
    stop("Result does not contain pi_star_point.", call. = FALSE)
  }
  # Flatten a named per-factor probability list in factor order. The factors
  # and levels of pi_star_point are produced from the same names_list ordering
  # the neural encoder was built with, so concatenation reproduces the
  # soft-path layout; the length check inside neural_split_pi_blocks guards
  # against any schema drift.
  flatten_pi <- function(pi_list) {
    unlist(lapply(pi_list, as.numeric), use.names = FALSE)
  }
  adversarial <- length(pi_star) >= 2L &&
    !is.null(names(pi_star)) &&
    all(c("ast", "dag") %in% names(pi_star))
  out <- list()
  if (isTRUE(adversarial)) {
    pi_ast <- flatten_pi(pi_star$ast)
    pi_dag <- flatten_pi(pi_star$dag)
    out$ast <- neural_soft_hard_gap(
      model_info = model_info,
      pi_vec = pi_ast,
      pi_vec_dag = pi_dag,
      n_draws = n_draws,
      seed = seed
    )
  } else {
    pi_use <- if (is.list(pi_star[[1L]]) && !is.numeric(pi_star[[1L]])) {
      pi_star[[1L]]
    } else {
      pi_star
    }
    out$ast <- neural_soft_hard_gap(
      model_info = model_info,
      pi_vec = flatten_pi(pi_use),
      n_draws = n_draws,
      seed = seed
    )
  }
  class(out) <- c("strategize_soft_relaxation", class(out))
  out
}
