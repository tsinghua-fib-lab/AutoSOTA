print2 <- function(text, quiet = F){
  if(!quiet){ print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) ) }
}

f2n <- function(x){as.numeric(as.character(x))}

ess_fxn <- function(wz){ sum(wz)^2 / sum(wz^2)}

# glinternet's C group_lasso routine segfaults ("invalid permissions") when a
# binomial fit is handed a degenerate response (all values identical). This
# predicate gates glinternet screening: TRUE only when the response carries
# usable signal (>= 2 distinct finite values). Family-agnostic -- for binomial
# this means both classes are present; for gaussian it means genuine variation.
glm_response_has_variation <- function(y) {
  y <- y[is.finite(y)]
  length(unique(y)) >= 2L
}

cs_build_names_list <- function(W, p_list = NULL) {
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  W <- as.data.frame(W)
  use_plist <- !is.null(p_list) &&
    length(p_list) > 0 &&
    !is.null(p_list[[1]]) &&
    !is.null(names(p_list[[1]]))
  if (use_plist) {
    names_list <- lapply(p_list, function(zer) list(names(zer)))
    if (!is.null(names(p_list))) {
      names(names_list) <- names(p_list)
    }
  } else {
    names_list <- lapply(seq_len(ncol(W)), function(j) {
      levs <- sort(names(table(as.factor(W[[j]]))), decreasing = FALSE)
      list(levs)
    })
    if (!is.null(colnames(W))) {
      names(names_list) <- colnames(W)
    }
  }
  if (is.null(names(names_list))) {
    names(names_list) <- if (!is.null(colnames(W))) {
      colnames(W)
    } else {
      paste0("V", seq_len(ncol(W)))
    }
  }
  names_list
}

cs_encode_W_indices <- function(W,
                                names_list,
                                unknown = c("na", "holdout", "error"),
                                pad_unknown = 0L,
                                align = c("none", "by_name")) {
  unknown <- match.arg(unknown)
  align <- match.arg(align)
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  if (is.null(names_list) || length(names_list) == 0) {
    stop("'names_list' must be a non-empty list.", call. = FALSE)
  }

  W_df <- as.data.frame(W)
  factor_names <- NULL
  names_list_names <- names(names_list)

  if (align == "by_name") {
    if (is.null(names_list_names)) {
      if (is.null(colnames(W_df))) {
        stop("'W' must have column names to align with names_list.", call. = FALSE)
      }
      names_list_names <- colnames(W_df)
      names(names_list) <- names_list_names
    }
    factor_names <- names_list_names
    if (is.null(colnames(W_df))) {
      stop("'W' must have column names to align with names_list.", call. = FALSE)
    }
    missing_cols <- setdiff(factor_names, colnames(W_df))
    if (length(missing_cols) > 0) {
      stop(
        "Missing factor columns in newdata: ",
        paste(missing_cols, collapse = ", "),
        call. = FALSE
      )
    }
    W_df <- W_df[, factor_names, drop = FALSE]
  } else {
    if (!is.null(colnames(W_df))) {
      factor_names <- colnames(W_df)
    } else if (!is.null(names_list_names)) {
      factor_names <- names_list_names
    } else {
      factor_names <- paste0("V", seq_len(ncol(W_df)))
    }
    if (is.null(names_list_names)) {
      names(names_list) <- factor_names
    }
    if (is.null(colnames(W_df))) {
      colnames(W_df) <- factor_names
    }
  }

  if (ncol(W_df) != length(names_list)) {
    stop(
      sprintf("'W' has %d columns but names_list has %d element(s).",
              ncol(W_df), length(names_list)),
      call. = FALSE
    )
  }

  W_idx <- sapply(seq_along(names_list), function(j) {
    levs <- names_list[[j]]
    if (is.list(levs)) {
      levs <- levs[[1]]
    }
    idx <- match(as.character(W_df[[j]]), levs)
    if (unknown == "error" && any(is.na(idx))) {
      bad <- unique(as.character(W_df[[j]])[is.na(idx)])
      stop(
        sprintf("Unseen levels in factor '%s': %s",
                factor_names[[j]],
                paste(bad, collapse = ", ")),
        call. = FALSE
      )
    }
    if (unknown == "holdout") {
      holdout <- length(levs) + as.integer(pad_unknown)
      idx[is.na(idx)] <- holdout
    }
    as.integer(idx)
  })

  W_idx <- as.matrix(W_idx)
  colnames(W_idx) <- factor_names
  list(W_idx = W_idx, names_list = names_list, factor_names = factor_names, W_df = W_df)
}

cs_prepare_W_encoding <- function(W,
                                  p_list = NULL,
                                  names_list = NULL,
                                  unknown = c("na", "holdout", "error"),
                                  pad_unknown = 0L,
                                  align = c("none", "by_name")) {
  if (is.null(names_list)) {
    names_list <- cs_build_names_list(W = W, p_list = p_list)
  }
  cs_encode_W_indices(
    W = W,
    names_list = names_list,
    unknown = unknown,
    pad_unknown = pad_unknown,
    align = align
  )
}

cs_generate_p_list <- function(W, uniform = FALSE, factor_names = NULL) {
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  W <- as.data.frame(W)
  if (ncol(W) == 0) {
    stop("'W' must have at least one column.", call. = FALSE)
  }

  p_list <- lapply(seq_len(ncol(W)), function(i) {
    tab <- table(W[[i]])
    if (uniform) {
      probs <- rep(1 / length(tab), length(tab))
    } else {
      probs <- as.numeric(prop.table(tab))
    }
    names(probs) <- names(tab)
    probs
  })

  if (!is.null(factor_names)) {
    if (length(factor_names) != ncol(W)) {
      stop(
        sprintf("'factor_names' must have length %d (got %d).", ncol(W), length(factor_names)),
        call. = FALSE
      )
    }
    names(p_list) <- factor_names
  } else if (!is.null(colnames(W))) {
    names(p_list) <- colnames(W)
  } else {
    names(p_list) <- paste0("Factor", seq_len(ncol(W)))
  }

  p_list
}

cs_p_list_uniform_deviation <- function(W) {
  if (is.null(W)) {
    stop("'W' is required.", call. = FALSE)
  }
  if (!is.data.frame(W) && !is.matrix(W)) {
    stop("'W' must be a data.frame or matrix.", call. = FALSE)
  }
  W <- as.data.frame(W)
  if (ncol(W) == 0) {
    return(numeric(0))
  }
  vapply(seq_len(ncol(W)), function(i) {
    tab <- table(W[[i]])
    if (length(tab) == 0) {
      return(0)
    }
    max(abs(prop.table(tab) - 1 / length(tab)))
  }, numeric(1))
}

cs_default_p_list <- function(W, threshold = 0.1, warn = TRUE, factor_names = NULL) {
  threshold <- as.numeric(threshold)
  if (!is.finite(threshold) || threshold < 0) {
    stop("'threshold' must be a finite non-negative numeric value.", call. = FALSE)
  }
  devs <- cs_p_list_uniform_deviation(W)
  use_observed <- length(devs) > 0 && any(devs > threshold)
  p_list <- cs_generate_p_list(W, uniform = !use_observed, factor_names = factor_names)
  if (isTRUE(warn) && isTRUE(use_observed)) {
    warning(
      sprintf("Assignment probabilities deviate from uniform by more than %.2f; using observed frequencies for p_list.", threshold),
      call. = FALSE
    )
  }
  attr(p_list, "p_list_method") <- if (use_observed) "observed" else "uniform"
  p_list
}

toSimplex = function(x){
  x[x>22] <- 22; x[x< -22] <- -22
  sim_x = exp(x)/sum(exp(x))
  if(any(is.nan(sim_x))){
    warning("NaN values encountered in toSimplex; returning uniform distribution")
    sim_x <- rep(1/length(x), length(x))
  }
  return(sim_x)
}

ai <- as.integer

RescaleFxn <- function(x, estMean=NULL, estSD=NULL, center=T){
  return(  x*estSD + ifelse(center, yes = estMean, no = 0) ) 
}

NA20 <- function(zer){zer[is.na(zer)]<-0;zer}

getSE <- function(er){ sqrt( var(er,na.rm=T) /  length(na.omit(er)) )  }

se <- function(.){sqrt(1/length(.) * var(.))}

getMultinomialSamp_R_DEPRECIATED <- function(
                               pi_value, 
                               temperature, 
                               jax_seed, 
                               ParameterizationType,
                               d_locator_use){
  # get t samp
  T_star_samp <- tapply(1:length(d_locator_use), d_locator_use, function(zer){
    pi_selection <- strenv$jnp$take(pi_value, 
                                    strenv$jnp$array(n2int(zer <- ai(  zer  ) - 1L)),0L)

    # add additional entry if implicit t ype
    if(ParameterizationType == "Implicit"){
      if(length(zer) == 1){ pi_selection <- strenv$jnp$expand_dims(pi_selection,0L) }
      pi_implied <- strenv$jnp$expand_dims(
                      strenv$jnp$expand_dims( strenv$jnp$array(1.) - strenv$jnp$sum(pi_selection),0L),0L)
      
      # add holdout 
      # pi_selection <- strenv$jnp$concatenate(list(pi_implied, pi_selection)) # add FIRST entry 
      pi_selection <- strenv$jnp$concatenate(list(pi_selection, pi_implied)) # add LAST entry 
    }

    # Sample from RelaxedOneHotCategorical using oryx
    # TSamp <- strenv$oryx$distributions$RelaxedOneHotCategorical(
      # probs = pi_selection$transpose(),
      #temperature = temperature)$sample(size = 1L, seed = jax_seed)$transpose()
    
    # Sample from RelaxedOneHotCategorical using base JAX
    logits <- strenv$jnp$log(pi_selection$transpose() + 1e-8)
    gumbels <- strenv$jax$random$gumbel(jax_seed, shape = logits$shape)
    TSamp <- strenv$jax$nn$softmax( ((logits + gumbels) / temperature),
                                    axis = 0L)$transpose()

    # if implicit, drop a term to keep correct shapes
    #if(ParameterizationType == "Implicit"){ TSamp <- strenv$jnp$take(TSamp,strenv$jnp$array(ai(1L:length(zer))),axis=0L) } #drop FIRST entry
    if(ParameterizationType == "Implicit"){ 
      TSamp <- strenv$jnp$take(TSamp,strenv$jnp$array(ai(0L:(length(zer)-1L))),axis=0L) } #drop LAST entry
    
    if(length(zer) == 1){TSamp <- strenv$jnp$expand_dims(TSamp, 1L)}
    return (  TSamp   )
  })
  names(T_star_samp) <- NULL # drop name to allow concatenation
  return( T_star_samp <-  strenv$jnp$concatenate(unlist(T_star_samp),0L) ) 
}

scale_rain_params <- function(rain_gamma, rain_eta, nSGD,
                              nSGD_ref = 100L,
                              autoscale_gamma = TRUE,
                              autoscale_eta = TRUE) {
  nSGD_val <- as.numeric(nSGD)
  if (!is.finite(nSGD_val) || nSGD_val <= 0) {
    return(list(rain_gamma = rain_gamma, rain_eta = rain_eta))
  }
  n_ref <- as.numeric(nSGD_ref)
  if (!is.finite(n_ref) || n_ref <= 0) {
    n_ref <- nSGD_val
  }
  if (autoscale_gamma) {
    gamma_base <- as.numeric(rain_gamma)
    if (!is.finite(gamma_base) || gamma_base < 0) {
      gamma_base <- 0
    }
    if (nSGD_val > n_ref) {
      gamma_base <- (1 + gamma_base)^(n_ref / nSGD_val) - 1
    }
    if (!is.finite(gamma_base) || gamma_base < 0) {
      gamma_base <- 0
    }
    rain_gamma <- gamma_base
  }
  if (autoscale_eta) {
    eta_base <- as.numeric(rain_eta)
    if (!is.finite(eta_base) || eta_base <= 0) {
      eta_base <- 1e-8
    }
    if (nSGD_val > n_ref) {
      eta_base <- eta_base * sqrt(n_ref / nSGD_val)
    }
    if (!is.finite(eta_base) || eta_base <= 0) {
      eta_base <- 1e-8
    }
    rain_eta <- eta_base
  }
  list(rain_gamma = rain_gamma, rain_eta = rain_eta)
}

sanitize_q_draw_count <- function(n_draws, default = 1L) {
  n_draws <- suppressWarnings(as.integer(n_draws))
  if (length(n_draws) != 1L || is.na(n_draws) || n_draws < 1L) {
    n_draws <- as.integer(default)
  }
  as.integer(n_draws)
}

estimate_policy_support_size <- function(ParameterizationType,
                                         d_locator_use) {
  if (is.null(ParameterizationType)) {
    return(NA_real_)
  }

  support_counts <- tryCatch(
    resolve_multinomial_group_spec(d_locator_use, ParameterizationType)$n_unique_levels_by_factors,
    error = function(e) NULL
  )
  if (is.null(support_counts) || length(support_counts) < 1L) {
    return(NA_real_)
  }

  support_size <- prod(as.numeric(support_counts))
  if (!is.finite(support_size) || support_size < 1) {
    return(Inf)
  }
  as.numeric(support_size)
}

is_large_policy_support <- function(ParameterizationType,
                                    d_locator_use,
                                    max_support = 2048L) {
  max_support <- sanitize_q_draw_count(max_support, default = 2048L)
  support_size <- estimate_policy_support_size(
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use
  )
  if (is.na(support_size)) {
    return(FALSE)
  }
  !is.finite(support_size) || support_size > max_support
}

# Q evaluation uses the exact hard target whenever it is tractable:
# exact Gaussian GLM Q, or exact hard-support enumeration for small-support
# single-party neural average-case. When neural support is too large to
# enumerate, the objective falls back to REINFORCE on hard samples because
# those draws are not pathwise-differentiable; reporting still targets the
# hard/exact estimand rather than a relaxed surrogate.
resolve_q_eval_spec <- function(phase = c("objective", "report"),
                                adversarial,
                                outcome_model_type,
                                glm_family,
                                nMonte_Qglm,
                                nMonte_adversarial = NULL,
                                ParameterizationType = NULL,
                                d_locator_use = NULL,
                                single_party = FALSE,
                                q_exact_support_max = 2048L,
                                force_reinforce = FALSE) {
  phase <- match.arg(phase)
  support_size <- estimate_policy_support_size(
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use
  )
  is_large_support <- is_large_policy_support(
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use,
    max_support = q_exact_support_max
  )

  if (!isTRUE(adversarial)) {
    if (identical(outcome_model_type, "neural")) {
      exact_support_n <- average_case_q_exact_support_size(
        ParameterizationType = ParameterizationType,
        d_locator_use = d_locator_use,
        single_party = single_party,
        max_support = q_exact_support_max
      )
      use_forced_reinforce <- identical(phase, "objective") &&
        isTRUE(force_reinforce) &&
        !is.null(exact_support_n)
      use_exact_support <- !use_forced_reinforce && !is.null(exact_support_n)
      return(list(
        use_exact_q = FALSE,
        use_exact_support = use_exact_support,
        exact_support_single_party = isTRUE(single_party) && isTRUE(use_exact_support),
        profile_draw_mode = "hard",
        pool_draw_mode = "hard",
        objective_gradient_mode = if (isTRUE(use_exact_support)) {
          "exact"
        } else if (isTRUE(use_forced_reinforce)) {
          "reinforce"
        } else if (identical(phase, "objective") && isTRUE(is_large_support)) {
          "reinforce"
        } else {
          "pathwise"
        },
        support_size = support_size,
        is_large_support = is_large_support,
        n_draws = if (isTRUE(use_exact_support)) {
          as.integer(exact_support_n)
        } else {
          sanitize_q_draw_count(nMonte_Qglm)
        }
      ))
    }
    if (identical(glm_family, "gaussian")) {
      return(list(
        use_exact_q = TRUE,
        use_exact_support = FALSE,
        exact_support_single_party = FALSE,
        profile_draw_mode = "exact",
        pool_draw_mode = "exact",
        objective_gradient_mode = "exact",
        support_size = support_size,
        is_large_support = FALSE,
        n_draws = 1L
      ))
    }
    return(list(
      use_exact_q = FALSE,
      use_exact_support = FALSE,
      exact_support_single_party = FALSE,
      profile_draw_mode = "relaxed",
      pool_draw_mode = "relaxed",
      objective_gradient_mode = "pathwise",
      support_size = support_size,
      is_large_support = FALSE,
      n_draws = sanitize_q_draw_count(nMonte_Qglm)
    ))
  }

  use_reinforce <- identical(outcome_model_type, "neural") &&
    identical(phase, "objective") &&
    isTRUE(is_large_support)

  draw_mode <- if (identical(outcome_model_type, "neural")) {
    if (use_reinforce) {
      "hard"
    } else if (identical(phase, "objective")) {
      "relaxed"
    } else {
      "hard"
    }
  } else {
    "relaxed"
  }
  n_draws <- if (identical(phase, "objective")) {
    nMonte_adversarial
  } else {
    nMonte_Qglm
  }
  list(
    use_exact_q = FALSE,
    use_exact_support = FALSE,
    exact_support_single_party = FALSE,
    profile_draw_mode = draw_mode,
    pool_draw_mode = draw_mode,
    objective_gradient_mode = if (use_reinforce) "reinforce" else "pathwise",
    support_size = support_size,
    is_large_support = is_large_support,
    n_draws = sanitize_q_draw_count(n_draws)
  )
}

resolve_q_policy_sampler <- function(draw_mode,
                                     d_locator_use = NULL,
                                     ParameterizationType = NULL) {
  if (is.null(draw_mode) || identical(draw_mode, "exact")) {
    return(NULL)
  }

  compiled_sampler_matches_locator <- function(d_locator_use,
                                               ParameterizationType) {
    if (is.null(d_locator_use)) {
      return(TRUE)
    }
    if (is.null(ParameterizationType)) {
      return(FALSE)
    }

    locator_spec <- tryCatch(
      resolve_multinomial_group_spec(d_locator_use, ParameterizationType),
      error = function(e) NULL
    )
    has_global_spec <- exists("nUniqueFactors", envir = strenv, inherits = FALSE) &&
      exists("nUniqueLevelsByFactors", envir = strenv, inherits = FALSE)

    if (is.null(locator_spec) || !isTRUE(has_global_spec)) {
      return(FALSE)
    }

    identical(as.integer(locator_spec$n_unique_factors),
              as.integer(strenv$nUniqueFactors)) &&
      identical(as.integer(locator_spec$n_unique_levels_by_factors),
                as.integer(strenv$nUniqueLevelsByFactors))
  }

  sampler_name <- switch(draw_mode,
                         relaxed = "getMultinomialSamp",
                         hard = "getMultinomialSampHard",
                         stop(sprintf("Unknown policy draw mode '%s'.", draw_mode),
                              call. = FALSE))
  if (exists("strenv", inherits = TRUE) &&
      exists(sampler_name, envir = strenv, inherits = FALSE)) {
    # Compiled samplers can only rely on strenv group metadata. If a concrete
    # locator implies a different grouping, prefer the plain R sampler so stale
    # globals do not widen profile draws.
    if (compiled_sampler_matches_locator(d_locator_use, ParameterizationType)) {
      return(get(sampler_name, envir = strenv, inherits = FALSE))
    }
  }

  switch(draw_mode,
         relaxed = getMultinomialSamp_R,
         hard = getMultinomialSampHard_R)
}

sample_multinomial_group <- function(pi_selection,
                                     temperature,
                                     jax_seed,
                                     draw_mode = c("relaxed", "hard")) {
  draw_mode <- match.arg(draw_mode)
  logits <- strenv$jnp$log(pi_selection$transpose() + 1e-8)
  gumbels <- strenv$jax$random$gumbel(key = jax_seed, shape = logits$shape)
  scores <- logits + gumbels
  soft_sample <- strenv$jax$nn$softmax(scores / temperature, axis = -1L)$transpose()
  if (identical(draw_mode, "relaxed")) {
    return(soft_sample)
  }

  # Categories are stored on the first axis of pi_selection. Using the
  # input shape keeps hard draws valid for both 1D vectors and transposed
  # row-matrix intermediates.
  n_categories <- ai(strenv$jnp$shape(pi_selection)[[1L]])
  hard_idx <- strenv$jnp$argmax(scores, axis = -1L)
  hard_sample <- strenv$jax$nn$one_hot(hard_idx, n_categories,
                                       dtype = soft_sample$dtype)$transpose()
  if (identical(draw_mode, "hard")) {
    return(hard_sample)
  }
}

resolve_multinomial_group_spec <- function(d_locator_use, ParameterizationType) {
  d_locator_r <- tryCatch(
    if (is.null(d_locator_use)) {
      NULL
    } else {
      as.integer(reticulate::py_to_r(strenv$np$array(d_locator_use)))
    },
    error = function(e) NULL
  )
  if (!is.null(d_locator_r) && length(d_locator_r)) {
    factor_ids <- sort(unique(d_locator_r))
    group_counts <- vapply(
      factor_ids,
      function(group_id) sum(d_locator_r == group_id),
      integer(1)
    )
    if (identical(ParameterizationType, "Implicit")) {
      group_counts <- group_counts + 1L
    }

    # Direct sampler calls may supply a concrete locator that differs from the
    # current strenv metadata. Prefer the locator to avoid stale state leakage.
    return(list(
      n_unique_factors = as.integer(length(factor_ids)),
      n_unique_levels_by_factors = as.integer(group_counts)
    ))
  }

  has_global_spec <- exists("nUniqueFactors", envir = strenv, inherits = FALSE) &&
    exists("nUniqueLevelsByFactors", envir = strenv, inherits = FALSE)

  if (isTRUE(has_global_spec)) {
    return(list(
      n_unique_factors = as.integer(strenv$nUniqueFactors),
      n_unique_levels_by_factors = as.integer(strenv$nUniqueLevelsByFactors)
    ))
  }

  stop(
    paste(
      "Could not infer multinomial group sizes from d_locator_use.",
      "Initialize factor metadata or provide a concrete locator array."
    ),
    call. = FALSE
  )
}

resolve_multinomial_group_index_spec <- function(d_locator_use, ParameterizationType) {
  group_spec <- resolve_multinomial_group_spec(d_locator_use, ParameterizationType)
  support_counts <- as.integer(group_spec$n_unique_levels_by_factors)
  explicit_counts <- if (identical(ParameterizationType, "Implicit")) {
    pmax(0L, support_counts - 1L)
  } else {
    support_counts
  }

  d_locator_r <- tryCatch(
    if (is.null(d_locator_use)) {
      NULL
    } else {
      as.integer(reticulate::py_to_r(strenv$np$array(d_locator_use)))
    },
    error = function(e) NULL
  )

  if (!is.null(d_locator_r) && length(d_locator_r)) {
    factor_ids <- sort(unique(d_locator_r))
    index_list <- lapply(
      factor_ids,
      function(group_id) as.integer(which(d_locator_r == group_id) - 1L)
    )
  } else {
    idx_stop <- cumsum(explicit_counts)
    idx_start <- idx_stop - explicit_counts + 1L
    index_list <- Map(
      function(start_i, stop_i) {
        if (stop_i < start_i) {
          integer(0)
        } else {
          seq.int(start_i, stop_i) - 1L
        }
      },
      idx_start,
      idx_stop
    )
  }

  group_count <- length(index_list)
  index_lengths <- as.integer(vapply(index_list, length, integer(1)))
  max_explicit_count <- if (group_count > 0L) {
    max(index_lengths)
  } else {
    0L
  }

  index_matrix <- index_mask <- NULL
  if (max_explicit_count > 0L) {
    index_matrix_r <- matrix(0L, nrow = group_count, ncol = max_explicit_count)
    index_mask_r <- matrix(FALSE, nrow = group_count, ncol = max_explicit_count)
    for (g_i in seq_len(group_count)) {
      len_i <- index_lengths[[g_i]]
      if (len_i < 1L) {
        next
      }
      index_matrix_r[g_i, seq_len(len_i)] <- index_list[[g_i]]
      index_mask_r[g_i, seq_len(len_i)] <- TRUE
    }
    index_matrix <- strenv$jnp$array(index_matrix_r, dtype = strenv$jnp$int32)
    index_mask <- strenv$jnp$array(index_mask_r, dtype = strenv$jnp$bool_)
  }

  list(
    support_counts = support_counts,
    explicit_counts = explicit_counts,
    index_list = index_list,
    index_lengths = index_lengths,
    group_count = as.integer(group_count),
    max_explicit_count = as.integer(max_explicit_count),
    index_matrix = index_matrix,
    index_mask = index_mask
  )
}

compute_grouped_policy_prob_matrix <- function(pi_vec,
                                               flat_profiles,
                                               ParameterizationType,
                                               index_spec) {
  n_flat <- ai(flat_profiles$shape[[1L]])
  group_count <- as.integer(index_spec$group_count)
  if (group_count < 1L) {
    return(NULL)
  }

  max_explicit_count <- as.integer(index_spec$max_explicit_count)
  if (max_explicit_count < 1L || is.null(index_spec$index_matrix) || is.null(index_spec$index_mask)) {
    return(strenv$jnp$ones(list(n_flat, group_count), dtype = pi_vec$dtype))
  }

  flat_profiles <- strenv$jnp$reshape(flat_profiles, list(n_flat, -1L))
  pi_flat <- strenv$jnp$reshape(pi_vec, list(-1L))
  mask_float <- strenv$jnp$astype(index_spec$index_mask, pi_flat$dtype)
  mask_float_3d <- strenv$jnp$expand_dims(mask_float, 0L)
  group_profiles <- strenv$jnp$take(flat_profiles, index_spec$index_matrix, axis = 1L)
  group_profiles <- strenv$jnp$reshape(
    group_profiles,
    list(n_flat, group_count, max_explicit_count)
  ) * mask_float_3d
  pi_selection <- strenv$jnp$reshape(
    strenv$jnp$take(pi_flat, index_spec$index_matrix, axis = 0L),
    list(group_count, max_explicit_count)
  ) * mask_float
  active_prob <- strenv$jnp$sum(
    group_profiles * strenv$jnp$reshape(pi_selection, list(1L, group_count, max_explicit_count)),
    axis = 2L
  )

  if (!identical(ParameterizationType, "Implicit")) {
    return(active_prob)
  }

  group_sum <- strenv$jnp$sum(group_profiles, axis = 2L)
  holdout_prob <- strenv$jnp$array(1., dtype = pi_flat$dtype) -
    strenv$jnp$sum(pi_selection, axis = 1L)
  holdout_matrix <- strenv$jnp$broadcast_to(
    strenv$jnp$expand_dims(holdout_prob, 0L),
    list(n_flat, group_count)
  )
  use_holdout <- strenv$jnp$equal(
    group_sum,
    strenv$jnp$array(0., dtype = group_sum$dtype)
  )
  strenv$jnp$where(use_holdout, holdout_matrix, active_prob)
}

average_case_q_exact_support_size <- function(ParameterizationType,
                                              d_locator_use,
                                              single_party = FALSE,
                                              max_support = 2048L) {
  if (!isTRUE(single_party) || is.null(ParameterizationType)) {
    return(NULL)
  }

  support_counts <- tryCatch(
    resolve_multinomial_group_spec(d_locator_use, ParameterizationType)$n_unique_levels_by_factors,
    error = function(e) NULL
  )
  if (is.null(support_counts) || length(support_counts) < 1L) {
    return(NULL)
  }

  max_support <- suppressWarnings(as.integer(max_support))
  if (length(max_support) != 1L || is.na(max_support) || max_support < 1L) {
    max_support <- 2048L
  }

  support_size <- prod(as.numeric(support_counts))
  if (!is.finite(support_size) || support_size < 1 || support_size > max_support) {
    return(NULL)
  }
  as.integer(support_size)
}

enumerate_policy_support_profiles <- function(ParameterizationType,
                                              d_locator_use,
                                              dtype = NULL) {
  index_spec <- resolve_multinomial_group_index_spec(d_locator_use, ParameterizationType)
  explicit_counts <- as.integer(index_spec$explicit_counts)

  build_group_block <- function(explicit_count) {
    explicit_count <- as.integer(explicit_count)
    if (explicit_count < 1L) {
      return(matrix(0, nrow = 1L, ncol = 0L))
    }
    eye <- diag(explicit_count)
    if (identical(ParameterizationType, "Implicit")) {
      rbind(eye, matrix(0, nrow = 1L, ncol = explicit_count))
    } else {
      eye
    }
  }

  group_blocks <- lapply(explicit_counts, build_group_block)
  selector_grid <- expand.grid(
    lapply(group_blocks, function(block) seq_len(nrow(block))),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  n_profiles <- nrow(selector_grid)
  n_params <- sum(explicit_counts)

  if (n_params < 1L) {
    profiles_r <- matrix(numeric(0), nrow = n_profiles, ncol = 0L)
  } else {
    profiles_r <- t(vapply(seq_len(n_profiles), function(i) {
      selectors <- as.integer(unlist(selector_grid[i, ], use.names = FALSE))
      as.numeric(unlist(Map(function(block, idx) {
        block[idx, , drop = TRUE]
      }, group_blocks, selectors), use.names = FALSE))
    }, numeric(n_params)))
    if (is.null(dim(profiles_r))) {
      profiles_r <- matrix(profiles_r, nrow = 1L)
    }
  }

  profiles <- if (is.null(dtype)) {
    strenv$jnp$array(profiles_r)
  } else {
    strenv$jnp$array(profiles_r, dtype = dtype)
  }

  list(
    profiles = profiles,
    n_profiles = as.integer(n_profiles),
    index_spec = index_spec
  )
}

compute_policy_support_weights <- function(pi_vec,
                                           profiles,
                                           ParameterizationType,
                                           d_locator_use) {
  index_spec <- resolve_multinomial_group_index_spec(d_locator_use, ParameterizationType)
  n_profiles <- ai(profiles$shape[[1L]])
  if (as.integer(index_spec$group_count) < 1L) {
    return(strenv$jnp$ones(list(n_profiles), dtype = pi_vec$dtype))
  }

  group_probs <- compute_grouped_policy_prob_matrix(
    pi_vec = pi_vec,
    flat_profiles = profiles,
    ParameterizationType = ParameterizationType,
    index_spec = index_spec
  )
  strenv$jnp$prod(group_probs, axis = 1L)
}

weighted_q_draw_average <- function(q_draws, weights = NULL) {
  if (is.null(weights)) {
    return(q_draws$mean(0L))
  }

  norm_weights <- weights / strenv$jnp$sum(weights)
  weight_dims <- c(ai(norm_weights$shape[[1L]]), rep(1L, max(0L, length(q_draws$shape) - 1L)))
  strenv$jnp$sum(
    q_draws * strenv$jnp$reshape(norm_weights, as.list(weight_dims)),
    axis = 0L
  )
}

# Score-function helper for REINFORCE. It evaluates grouped multinomial
# log-probabilities of hard one-hot draws, supporting rank-2/3 profile
# batches and rank-4 pooled batches from the large-support samplers.
compute_policy_sample_log_probs <- function(pi_vec,
                                            profiles,
                                            ParameterizationType,
                                            d_locator_use = NULL,
                                            index_spec = NULL) {
  if (is.null(index_spec)) {
    index_spec <- resolve_multinomial_group_index_spec(
      d_locator_use = d_locator_use,
      ParameterizationType = ParameterizationType
    )
  }

  profile_rank <- length(profiles$shape)
  if (!(profile_rank %in% c(2L, 3L, 4L))) {
    stop(
      sprintf(
        paste(
          "Policy log-prob scoring expects rank-2/3 profile batches",
          "or rank-4 pooled batches; got rank %d."
        ),
        profile_rank
      ),
      call. = FALSE
    )
  }

  n_outer <- ai(profiles$shape[[1L]])
  if (profile_rank %in% c(2L, 3L)) {
    n_inner <- 1L
    param_dims <- vapply(seq.int(2L, profile_rank), function(idx) {
      ai(profiles$shape[[idx]])
    }, integer(1))
  } else {
    n_inner <- ai(profiles$shape[[2L]])
    param_dims <- vapply(seq.int(3L, profile_rank), function(idx) {
      ai(profiles$shape[[idx]])
    }, integer(1))
  }
  n_params <- as.integer(prod(param_dims))
  flat_profiles <- strenv$jnp$reshape(profiles, list(-1L, n_params))
  n_flat <- ai(flat_profiles$shape[[1L]])
  eps <- strenv$jnp$array(1e-8, dtype = pi_vec$dtype)
  if (as.integer(index_spec$group_count) < 1L) {
    log_probs <- strenv$jnp$zeros(list(n_flat), dtype = pi_vec$dtype)
  } else {
    group_probs <- compute_grouped_policy_prob_matrix(
      pi_vec = pi_vec,
      flat_profiles = flat_profiles,
      ParameterizationType = ParameterizationType,
      index_spec = index_spec
    )
    log_probs <- strenv$jnp$sum(
      strenv$jnp$log(strenv$jnp$clip(group_probs, eps, 1.0)),
      axis = 1L
    )
  }

  if (n_inner == 1L) {
    return(strenv$jnp$reshape(log_probs, list(n_outer)))
  }

  reshaped <- strenv$jnp$reshape(log_probs, list(n_outer, n_inner))
  strenv$jnp$sum(reshaped, axis = 1L)
}

extract_objective_reward_draws <- function(q_draws,
                                           component = 0L) {
  n_draws <- ai(q_draws$shape[[1L]])
  reward_draws <- strenv$jnp$take(q_draws, as.integer(component), axis = 1L)
  reward_draws <- strenv$jnp$reshape(reward_draws, list(n_draws, -1L))
  strenv$jnp$take(reward_draws, 0L, axis = 1L)
}

getMultinomialSamp_generic_R <- function(pi_value,
                                         temperature,
                                         jax_seed,
                                         ParameterizationType,
                                         d_locator_use,
                                         draw_mode = c("relaxed", "hard")) {
  draw_mode <- match.arg(draw_mode)
  group_spec <- resolve_multinomial_group_spec(d_locator_use, ParameterizationType)
  # Ensure d_locator_use is at least 1D, in case it was a scalar
  d_locator_use <- strenv$jnp$atleast_1d(d_locator_use)
  
  # Identify each unique group + the inverse indices
  unique_groups_inverse_indices <- strenv$jnp$unique(d_locator_use,
                                                     return_inverse=TRUE,
                                                     size = group_spec$n_unique_factors)
  unique_groups <- unique_groups_inverse_indices[[1]]
  inverse_indices <- unique_groups_inverse_indices[[2]]
  
  # Also ensure these are at least 1D
  unique_groups <- strenv$jnp$atleast_1d(unique_groups)
  inverse_indices <- strenv$jnp$atleast_1d(inverse_indices)
  
  # Number of unique groups
  groupCount <- strenv$jnp$shape(unique_groups)[[1]]
  
  # Prepare a list for per-group samples
  T_star_samp_list <- vector("list", groupCount)
  
  # Loop over each unique group
  for(g_i in seq_len(groupCount)) {
    g_jax <- g_i - 1L
    
    # Indices belonging to group g_jax
    zer <- strenv$jnp$where(
      strenv$jnp$equal(inverse_indices, 
                       strenv$jnp$array(n2int(g_jax))),
      size = ai(group_spec$n_unique_levels_by_factors[g_i] -
                  1L * (ParameterizationType == "Implicit"))
    )[[1]]
    
    # pi_selection for that group
    pi_selection <- strenv$jnp$take(pi_value, zer, axis=0L)

    # For Implicit parameterizations, add the "holdout" probability
    if(ParameterizationType == "Implicit"){
      pi_implied <- strenv$jnp$expand_dims(
        strenv$jnp$expand_dims(
          strenv$jnp$array(1.) - strenv$jnp$sum(pi_selection),
          0L), 0L)
      
      # Concatenate implied entry last
      pi_selection <- strenv$jnp$concatenate(list(pi_selection, pi_implied), 0L)
    }
    
    TSamp <- sample_multinomial_group(pi_selection, temperature, jax_seed,
                                      draw_mode = draw_mode)
    jax_seed   <- strenv$jax$random$split(jax_seed)[[1L]]
    
    # If Implicit, remove that last extra dimension after sampling
    if(ParameterizationType == "Implicit"){
      group_len <- strenv$jnp$shape(pi_selection)[[1]] - 1L
      TSamp <- strenv$jnp$take(TSamp, strenv$jnp$arange(group_len), axis=0L)
    }
    
    # If the group originally had length 1, restore the shape by expanding axis=1
    #if(strenv$jnp$shape(TSamp)[[1]] == 1) {
    #  TSamp <- strenv$jnp$expand_dims(TSamp, 1L)
    #}
    
    T_star_samp_list[[g_i]] <- TSamp
  }
  
  # Concatenate all group samples along axis=0
  T_star_samp <- strenv$jnp$concatenate(T_star_samp_list, 0L)
  return(T_star_samp)
}

getMultinomialSamp_R <- function(pi_value,
                                 temperature,
                                 jax_seed,
                                 ParameterizationType,
                                 d_locator_use) {
  getMultinomialSamp_generic_R(
    pi_value = pi_value,
    temperature = temperature,
    jax_seed = jax_seed,
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use,
    draw_mode = "relaxed"
  )
}

getMultinomialSampHard_R <- function(pi_value,
                                     temperature,
                                     jax_seed,
                                     ParameterizationType,
                                     d_locator_use) {
  getMultinomialSamp_generic_R(
    pi_value = pi_value,
    temperature = temperature,
    jax_seed = jax_seed,
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use,
    draw_mode = "hard"
  )
}

sample_pool_jax <- function(pi_vec, n_draws, n_pool, seed_in,
                            temperature,
                            ParameterizationType,
                            d_locator_use,
                            sampler = NULL) {
  if (is.null(sampler)) {
    sampler <- strenv$getMultinomialSamp
  }
  n_total <- as.integer(n_draws * n_pool)
  all_keys <- strenv$jax$random$split(seed_in, as.integer(n_total + 1L))
  seed_next <- strenv$jnp$take(all_keys, -1L, axis = 0L)
  seeds <- strenv$jnp$take(all_keys, strenv$jnp$arange(n_total), axis = 0L)
  seeds <- strenv$jnp$reshape(seeds, list(n_draws, n_pool, 2L))
  samples <- strenv$jax$vmap(function(seed_row){
    strenv$jax$vmap(function(seed_cell){
      sampler(pi_vec, temperature, seed_cell, ParameterizationType, d_locator_use)
    }, in_axes = list(0L))(seed_row)
  }, in_axes = list(0L))(seeds)
  list(samples = samples, seed_next = seed_next)
}

draw_profile_samples <- function(pi_vec, n_draws, seed_in,
                                 temperature,
                                 ParameterizationType,
                                 d_locator_use,
                                 sampler = NULL) {
  if (is.null(sampler)) {
    sampler <- strenv$getMultinomialSamp
  }
  samples <- strenv$jax$vmap(function(s_){
    sampler(pi_vec, temperature, s_, ParameterizationType, d_locator_use)
  }, in_axes = list(0L))(strenv$jax$random$split(seed_in, n_draws))
  seed_next <- strenv$jax$random$split(seed_in)[[1L]]
  list(samples = samples, seed_next = seed_next)
}

average_case_q_uses_mc <- function(outcome_model_type,
                                   glm_family,
                                   nMonte_Qglm) {
  n_draws <- sanitize_q_draw_count(nMonte_Qglm)

  identical(outcome_model_type, "neural") ||
    (!identical(glm_family, "gaussian") && (n_draws > 1L))
}

draw_average_case_q_profiles <- function(pi_star_ast,
                                         pi_star_dag,
                                         outcome_model_type,
                                         glm_family,
                                         nMonte_Qglm,
                                         seed_in,
                                         temperature,
                                         ParameterizationType,
                                         d_locator_use,
                                         sampler = NULL,
                                         profile_draw_mode = NULL,
                                         use_exact_support = FALSE,
                                         exact_support_single_party = FALSE) {
  use_mc_q <- if (is.null(profile_draw_mode)) {
    average_case_q_uses_mc(
      outcome_model_type = outcome_model_type,
      glm_family = glm_family,
      nMonte_Qglm = nMonte_Qglm
    )
  } else {
    !identical(profile_draw_mode, "exact")
  }

  n_draws <- sanitize_q_draw_count(nMonte_Qglm)

  if (isTRUE(use_exact_support)) {
    if (!isTRUE(exact_support_single_party)) {
      stop("Exact average-case support enumeration currently requires single-party mode.",
           call. = FALSE)
    }
    support_ast <- enumerate_policy_support_profiles(
      ParameterizationType = ParameterizationType,
      d_locator_use = d_locator_use,
      dtype = pi_star_ast$dtype
    )
    n_profiles <- as.integer(support_ast$n_profiles)
    dag_shape <- as.integer(reticulate::py_to_r(strenv$np$array(pi_star_dag$shape)))
    dag_profiles <- strenv$jnp$broadcast_to(
      strenv$jnp$expand_dims(pi_star_dag, 0L),
      as.list(c(as.integer(n_profiles), dag_shape))
    )
    profile_weights <- compute_policy_support_weights(
      pi_vec = pi_star_ast,
      profiles = support_ast$profiles,
      ParameterizationType = ParameterizationType,
      d_locator_use = d_locator_use
    )
    return(list(
      pi_star_ast_f_all = support_ast$profiles,
      pi_star_dag_f_all = dag_profiles,
      seed_next = seed_in,
      use_mc_q = TRUE,
      n_draws = n_profiles,
      profile_weights = profile_weights,
      exact_support = TRUE
    ))
  }

  if (!use_mc_q) {
    return(list(
      pi_star_ast_f_all = strenv$jnp$expand_dims(pi_star_ast, 0L),
      pi_star_dag_f_all = strenv$jnp$expand_dims(pi_star_dag, 0L),
      seed_next = seed_in,
      use_mc_q = FALSE,
      n_draws = 1L,
      profile_weights = NULL,
      exact_support = FALSE
    ))
  }

  if (identical(outcome_model_type, "neural")) {
    n_draws <- max(1L, n_draws)
  }

  sampler_use <- if (!is.null(sampler)) {
    sampler
  } else {
    resolve_q_policy_sampler(
      if (is.null(profile_draw_mode)) "relaxed" else profile_draw_mode,
      d_locator_use = d_locator_use,
      ParameterizationType = ParameterizationType
    )
  }

  draw_ast <- draw_profile_samples(
    pi_star_ast, n_draws, seed_in,
    temperature, ParameterizationType, d_locator_use,
    sampler = sampler_use
  )
  draw_dag <- draw_profile_samples(
    pi_star_dag, n_draws, draw_ast$seed_next,
    temperature, ParameterizationType, d_locator_use,
    sampler = sampler_use
  )

  list(
    pi_star_ast_f_all = draw_ast$samples,
    pi_star_dag_f_all = draw_dag$samples,
    seed_next = draw_dag$seed_next,
    use_mc_q = TRUE,
    n_draws = n_draws,
    profile_weights = NULL,
    exact_support = FALSE
  )
}

reshape_scalar_q_value <- function(q_val, template) {
  q_dims <- if (length(template$shape) >= 2L) {
    list(1L, 1L)
  } else {
    list(1L)
  }
  strenv$jnp$reshape(q_val, q_dims)
}

evaluate_average_case_q <- function(pi_star_ast,
                                    pi_star_dag,
                                    INTERCEPT_ast_, COEFFICIENTS_ast_,
                                    INTERCEPT_dag_, COEFFICIENTS_dag_,
                                    seed_in,
                                    phase = c("objective", "report"),
                                    outcome_model_type,
                                    glm_family,
                                    nMonte_Qglm,
                                    temperature,
                                    ParameterizationType,
                                    d_locator_use,
                                    q_fxn,
                                    single_party = FALSE,
                                    force_reinforce = FALSE) {
  phase <- match.arg(phase)
  spec <- resolve_q_eval_spec(
    phase = phase,
    adversarial = FALSE,
    outcome_model_type = outcome_model_type,
    glm_family = glm_family,
    nMonte_Qglm = nMonte_Qglm,
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use,
    single_party = single_party,
    force_reinforce = force_reinforce
  )
  q_profile_draws <- draw_average_case_q_profiles(
    pi_star_ast = pi_star_ast,
    pi_star_dag = pi_star_dag,
    outcome_model_type = outcome_model_type,
    glm_family = glm_family,
    nMonte_Qglm = spec$n_draws,
    seed_in = seed_in,
    temperature = temperature,
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use,
    profile_draw_mode = spec$profile_draw_mode,
    use_exact_support = isTRUE(spec$use_exact_support),
    exact_support_single_party = isTRUE(spec$exact_support_single_party)
  )

  if (isTRUE(q_profile_draws$use_mc_q)) {
    q_draws <- strenv$Vectorized_QMonteIter(
      q_profile_draws$pi_star_ast_f_all,
      q_profile_draws$pi_star_dag_f_all,
      INTERCEPT_ast_,
      COEFFICIENTS_ast_,
      INTERCEPT_dag_,
      COEFFICIENTS_dag_
    )
    q_vec <- weighted_q_draw_average(q_draws, q_profile_draws$profile_weights)
  } else {
    q_vec <- q_fxn(
      pi_star_ast = pi_star_ast,
      pi_star_dag = pi_star_dag,
      EST_INTERCEPT_tf_ast = INTERCEPT_ast_,
      EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_,
      EST_INTERCEPT_tf_dag = INTERCEPT_dag_,
      EST_COEFFICIENTS_tf_dag = COEFFICIENTS_dag_
    )
  }

  if (identical(phase, "report") &&
      identical(outcome_model_type, "neural") &&
      exists("neural_average_case_report_adjust_q", mode = "function")) {
    q_vec <- neural_average_case_report_adjust_q(
      q_vec = q_vec,
      pi_star_ast = pi_star_ast,
      EST_COEFFICIENTS_tf_ast = COEFFICIENTS_ast_
    )
  }

  list(
    q_vec = q_vec,
    q_max = strenv$jnp$take(q_vec, 0L),
    seed_next = q_profile_draws$seed_next,
    spec = spec
  )
}

# Objective-only average-case sampler for REINFORCE. It returns hard-sample
# reward draws and sampled profiles for the score-function gradient, not the
# final reported Q estimator.
evaluate_average_case_q_reinforce <- function(pi_star_ast,
                                              pi_star_dag,
                                              INTERCEPT_ast_, COEFFICIENTS_ast_,
                                              INTERCEPT_dag_, COEFFICIENTS_dag_,
                                              seed_in,
                                              outcome_model_type,
                                              glm_family,
                                              nMonte_Qglm,
                                              temperature,
                                              ParameterizationType,
                                              d_locator_use,
                                              q_fxn,
                                              single_party = FALSE,
                                              force_reinforce = FALSE,
                                              player = c("ast", "dag")) {
  player <- match.arg(player)
  spec <- resolve_q_eval_spec(
    phase = "objective",
    adversarial = FALSE,
    outcome_model_type = outcome_model_type,
    glm_family = glm_family,
    nMonte_Qglm = nMonte_Qglm,
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use,
    single_party = single_party,
    force_reinforce = force_reinforce
  )
  if (!identical(spec$objective_gradient_mode, "reinforce")) {
    stop("Average-case REINFORCE evaluation requires objective_gradient_mode='reinforce'.",
         call. = FALSE)
  }

  q_profile_draws <- draw_average_case_q_profiles(
    pi_star_ast = pi_star_ast,
    pi_star_dag = pi_star_dag,
    outcome_model_type = outcome_model_type,
    glm_family = glm_family,
    nMonte_Qglm = spec$n_draws,
    seed_in = seed_in,
    temperature = temperature,
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use,
    profile_draw_mode = "hard",
    use_exact_support = FALSE,
    exact_support_single_party = FALSE
  )

  q_draws <- strenv$Vectorized_QMonteIter(
    q_profile_draws$pi_star_ast_f_all,
    q_profile_draws$pi_star_dag_f_all,
    INTERCEPT_ast_,
    COEFFICIENTS_ast_,
    INTERCEPT_dag_,
    COEFFICIENTS_dag_
  )
  reward_draws <- extract_objective_reward_draws(q_draws, component = 0L)

  list(
    reward_draws = reward_draws,
    reward_mean = reward_draws$mean(),
    policy_samples = if (identical(player, "ast")) {
      q_profile_draws$pi_star_ast_f_all
    } else {
      q_profile_draws$pi_star_dag_f_all
    },
    seed_next = q_profile_draws$seed_next,
    spec = spec
  )
}

evaluate_adversarial_q <- function(pi_star_ast,
                                   pi_star_dag,
                                   a_i_ast,
                                   a_i_dag,
                                   INTERCEPT_ast_, COEFFICIENTS_ast_,
                                   INTERCEPT_dag_, COEFFICIENTS_dag_,
                                   INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                   P_VEC_FULL_ast_, P_VEC_FULL_dag_,
                                   SLATE_VEC_ast_, SLATE_VEC_dag_,
                                   LAMBDA_,
                                   Q_SIGN,
                                   seed_in,
                                   phase = c("objective", "report"),
                                   outcome_model_type,
                                   glm_family,
                                   nMonte_Qglm,
                                   nMonte_adversarial,
                                   primary_pushforward,
                                   primary_n_entrants,
                                   primary_n_field,
                                   temperature,
                                   ParameterizationType,
                                   d_locator_use) {
  phase <- match.arg(phase)
  spec <- resolve_q_eval_spec(
    phase = phase,
    adversarial = TRUE,
    outcome_model_type = outcome_model_type,
    glm_family = glm_family,
    nMonte_Qglm = nMonte_Qglm,
    nMonte_adversarial = nMonte_adversarial
  )
  sampler_profile <- resolve_q_policy_sampler(
    spec$profile_draw_mode,
    d_locator_use = d_locator_use,
    ParameterizationType = ParameterizationType
  )
  sampler_pool <- resolve_q_policy_sampler(
    spec$pool_draw_mode,
    d_locator_use = d_locator_use,
    ParameterizationType = ParameterizationType
  )
  seed_next <- seed_in
  n_q_samp <- spec$n_draws

  if (primary_pushforward == "multi") {
    samp_ast <- sample_pool_jax(
      pi_star_ast, n_q_samp, primary_n_entrants, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_ast_all <- samp_ast$samples
    seed_next <- samp_ast$seed_next

    samp_dag <- sample_pool_jax(
      pi_star_dag, n_q_samp, primary_n_entrants, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_dag_all <- samp_dag$samples
    seed_next <- samp_dag$seed_next

    samp_ast_field <- sample_pool_jax(
      SLATE_VEC_ast_, n_q_samp, primary_n_field, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_ast_PrimaryComp_all <- samp_ast_field$samples
    seed_next <- samp_ast_field$seed_next

    samp_dag_field <- sample_pool_jax(
      SLATE_VEC_dag_, n_q_samp, primary_n_field, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_dag_PrimaryComp_all <- samp_dag_field$samples
    seed_next <- samp_dag_field$seed_next
  } else {
    draw_ast <- draw_profile_samples(
      pi_star_ast, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_ast_all <- draw_ast$samples
    seed_next <- draw_ast$seed_next

    draw_dag <- draw_profile_samples(
      pi_star_dag, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_dag_all <- draw_dag$samples
    seed_next <- draw_dag$seed_next

    draw_ast_field <- draw_profile_samples(
      SLATE_VEC_ast_, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_ast_PrimaryComp_all <- draw_ast_field$samples
    seed_next <- draw_ast_field$seed_next

    draw_dag_field <- draw_profile_samples(
      SLATE_VEC_dag_, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_dag_PrimaryComp_all <- draw_dag_field$samples
    seed_next <- draw_dag_field$seed_next
  }

  QMonteRes <- strenv$Vectorized_QMonteIter_MaxMin(
    TSAMP_ast_all, TSAMP_dag_all,
    TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
    a_i_ast, a_i_dag,
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_, P_VEC_FULL_dag_,
    LAMBDA_, Q_SIGN,
    strenv$jax$random$split(seed_next, n_q_samp)
  )
  q_ast <- QMonteRes$q_ast$mean()
  q_dag <- QMonteRes$q_dag$mean()
  indicator_UseAst <- 0.5 * (1. + Q_SIGN)

  list(
    q_ast = q_ast,
    q_dag = q_dag,
    q_max = indicator_UseAst * q_ast + (1. - indicator_UseAst) * q_dag,
    seed_next = seed_next,
    spec = spec
  )
}

# Objective-only adversarial sampler for REINFORCE. It supplies hard-sample
# reward draws and sampled profiles/pools for the score-function gradient; the
# reported adversarial Q remains on the standard hard-sample report path.
evaluate_adversarial_q_reinforce <- function(pi_star_ast,
                                             pi_star_dag,
                                             a_i_ast,
                                             a_i_dag,
                                             INTERCEPT_ast_, COEFFICIENTS_ast_,
                                             INTERCEPT_dag_, COEFFICIENTS_dag_,
                                             INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                             INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                             P_VEC_FULL_ast_, P_VEC_FULL_dag_,
                                             SLATE_VEC_ast_, SLATE_VEC_dag_,
                                             LAMBDA_,
                                             seed_in,
                                             outcome_model_type,
                                             glm_family,
                                             nMonte_Qglm,
                                             nMonte_adversarial,
                                             primary_pushforward,
                                             primary_n_entrants,
                                             primary_n_field,
                                             temperature,
                                             ParameterizationType,
                                             d_locator_use,
                                             player = c("ast", "dag")) {
  player <- match.arg(player)
  spec <- resolve_q_eval_spec(
    phase = "objective",
    adversarial = TRUE,
    outcome_model_type = outcome_model_type,
    glm_family = glm_family,
    nMonte_Qglm = nMonte_Qglm,
    nMonte_adversarial = nMonte_adversarial,
    ParameterizationType = ParameterizationType,
    d_locator_use = d_locator_use
  )
  if (!identical(spec$objective_gradient_mode, "reinforce")) {
    stop("Adversarial REINFORCE evaluation requires objective_gradient_mode='reinforce'.",
         call. = FALSE)
  }

  sampler_profile <- resolve_q_policy_sampler(
    "hard",
    d_locator_use = d_locator_use,
    ParameterizationType = ParameterizationType
  )
  sampler_pool <- resolve_q_policy_sampler(
    "hard",
    d_locator_use = d_locator_use,
    ParameterizationType = ParameterizationType
  )
  seed_next <- seed_in
  n_q_samp <- spec$n_draws

  if (primary_pushforward == "multi") {
    samp_ast <- sample_pool_jax(
      pi_star_ast, n_q_samp, primary_n_entrants, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_ast_all <- samp_ast$samples
    seed_next <- samp_ast$seed_next

    samp_dag <- sample_pool_jax(
      pi_star_dag, n_q_samp, primary_n_entrants, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_dag_all <- samp_dag$samples
    seed_next <- samp_dag$seed_next

    samp_ast_field <- sample_pool_jax(
      SLATE_VEC_ast_, n_q_samp, primary_n_field, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_ast_PrimaryComp_all <- samp_ast_field$samples
    seed_next <- samp_ast_field$seed_next

    samp_dag_field <- sample_pool_jax(
      SLATE_VEC_dag_, n_q_samp, primary_n_field, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_dag_PrimaryComp_all <- samp_dag_field$samples
    seed_next <- samp_dag_field$seed_next
  } else {
    draw_ast <- draw_profile_samples(
      pi_star_ast, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_ast_all <- draw_ast$samples
    seed_next <- draw_ast$seed_next

    draw_dag <- draw_profile_samples(
      pi_star_dag, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_profile
    )
    TSAMP_dag_all <- draw_dag$samples
    seed_next <- draw_dag$seed_next

    draw_ast_field <- draw_profile_samples(
      SLATE_VEC_ast_, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_ast_PrimaryComp_all <- draw_ast_field$samples
    seed_next <- draw_ast_field$seed_next

    draw_dag_field <- draw_profile_samples(
      SLATE_VEC_dag_, n_q_samp, seed_next,
      temperature, ParameterizationType, d_locator_use,
      sampler = sampler_pool
    )
    TSAMP_dag_PrimaryComp_all <- draw_dag_field$samples
    seed_next <- draw_dag_field$seed_next
  }

  QMonteRes <- strenv$Vectorized_QMonteIter_MaxMin(
    TSAMP_ast_all, TSAMP_dag_all,
    TSAMP_ast_PrimaryComp_all, TSAMP_dag_PrimaryComp_all,
    a_i_ast, a_i_dag,
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_, P_VEC_FULL_dag_,
    LAMBDA_, strenv$jnp$array(if (identical(player, "ast")) 1. else -1.),
    strenv$jax$random$split(seed_next, n_q_samp)
  )
  reward_draws <- if (identical(player, "ast")) {
    QMonteRes$q_ast
  } else {
    QMonteRes$q_dag
  }

  list(
    reward_draws = reward_draws,
    reward_mean = reward_draws$mean(),
    policy_samples = if (identical(player, "ast")) TSAMP_ast_all else TSAMP_dag_all,
    seed_next = seed_next,
    spec = spec
  )
}


getPrettyPi <- function( pi_star_value, 
                         ParameterizationType,
                         d_locator,
                         main_comp_mat,
                         shadow_comp_mat
                         ){
  if( ParameterizationType == "Full" ){
    #pi_star_full <- tapply(1:length(d_locator_full),d_locator_full,function(zer){strenv$jnp$take(pi_star_value,n2int(ai(zer-1L))) })
    pi_star_full <- pi_star_value
  }
  if( ParameterizationType == "Implicit" ){
    d_locator <- strenv$jnp$reshape(strenv$jnp$atleast_1d(d_locator), list(-1L))
    pi_star_value <- strenv$jnp$reshape(pi_star_value, list(-1L, 1L))
    # Ensure d_locator is a JAX array (assumed to be provided as such)
    # Map d_locator values to consecutive integers starting from 0
    unique_groups_inverse_indices <- strenv$jnp$unique( d_locator, 
                                                        return_inverse=TRUE, 
                                                        size = strenv$nUniqueFactors # Needed for JIT 
                                                        )

    if(length(unique_groups_inverse_indices[[2]]$shape) == 0){
      unique_groups_inverse_indices[[2]] <- strenv$jnp$expand_dims(unique_groups_inverse_indices[[2]],0L)
    }
    
    # Compute the sum of pi_star_value for each group
    group_sums <- strenv$jax$ops$segment_sum(
      pi_star_value, 
      unique_groups_inverse_indices[[2]],
      num_segments = strenv$nUniqueFactors
    )
    #group_sums <- strenv$jax$ops$segment_sum(pi_star_value, 
                                             #unique_groups_inverse_indices[[2]]) # fails with JIT 
    
    
    
    # Compute pi_star_impliedTerms for each group
    pi_star_impliedTerms <- strenv$jnp$subtract(
      strenv$jnp$array(1., dtype = strenv$dtj),
      strenv$jnp$reshape(group_sums, list(-1L, 1L))
    )
    # pi_star_impliedTerms - pi_star_impliedTermsOLD
    
    # Old way of computing implied terms  
    #pi_star_impliedTermsOLD <- tapply(1:length(d_locator), d_locator, function(zer){
          #pi_implied <- strenv$OneTf -  strenv$jnp$sum(strenv$jnp$take(pi_star_value, n2int(ai(zer-1L)),0L)) })
    #names(pi_star_impliedTermsOLD) <- NULL
    #pi_star_impliedTermsOLD <- strenv$jnp$concatenate(pi_star_impliedTermsOLD,0L)

    pi_star_full <- strenv$jnp$expand_dims(
                      strenv$jnp$matmul(main_comp_mat, pi_star_value)$flatten() +
                            strenv$jnp$matmul(shadow_comp_mat, pi_star_impliedTerms)$flatten(),1L)
  }

  return( pi_star_full )
}

computeQ_conjoint_internal <- function(FactorsMat_internal,
                                       Yobs_internal,
                                       FactorsMat_internal_mapped = NULL,
                                       hypotheticalProbList_internal,
                                       assignmentProbList_internal,
                                       log_pr_w_internal = NULL,
                                       hajek = T, 
                                       knownNormalizationFactor = NULL,
                                       computeLB  = F){
  if(is.null(log_pr_w_internal)){
    log_pr_w_internal <- log( sapply(1:ncol(FactorsMat_internal),function(ze){
      (assignmentProbList_internal[[ze]][ FactorsMat_internal[,ze] ]) }) )
    if(all(class(log_pr_w_internal) == "numeric")){ log_pr_w_internal <- sum(log_pr_w_internal)}
    if(any(class(log_pr_w_internal) != "numeric")){ log_pr_w_internal = rowSums(log_pr_w_internal)}
  }

  # new probability
  if(!is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped[] <- log(unlist(hypotheticalProbList_internal)[FactorsMat_internal_mapped])}
  if(is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped <- log( sapply(1:ncol(FactorsMat_internal),function(ze){hypotheticalProbList_internal[[ze]][ FactorsMat_internal[,ze] ]  })  )}
  if(any(class(FactorsMat_internal_mapped) != "numeric")){ log_pr_w_new <- rowSums(FactorsMat_internal_mapped)}
  if(all(class(FactorsMat_internal_mapped) == "numeric")){ log_pr_w_new <- sum(FactorsMat_internal_mapped)}
  my_wts <- exp(  log_pr_w_new   - log_pr_w_internal  )
  sum_raw_wts <- sum( my_wts )
  if(hajek == T){
    #NOTE: Hajek == mean(Yobs_internal*my_wts) / mean( my_wts ), with my_wts unnormalized
    if(is.null(knownNormalizationFactor)){  my_wts <- my_wts / sum_raw_wts }
    if(!is.null(knownNormalizationFactor)){  my_wts <- my_wts / knownNormalizationFactor }
    if(computeLB == F){ Qest = sum(Yobs_internal * my_wts )  }
    if(computeLB == T){
      minValue     <- min(Yobs_internal)
      Yobs_nonZero <- Yobs_internal + (abs(minValue) + 1)*(minValue <= 0)
      Qest <- sum(log(Yobs_nonZero)+log(my_wts))
    }
  }
  if(hajek == F){
    if(computeLB == F){ Qest <- mean(Yobs_internal * my_wts )   }
    if(computeLB == T){
      minValue <- min(Yobs_internal)
      Yobs_nonZero <- Yobs_internal + (abs(minValue) + 1)*(minValue <= 0)
      Qest <- mean(log(Yobs_nonZero)+log(my_wts))
    }
  }

  return(list("Qest"=Qest,
              "Q_wts"=my_wts,
              "Yobs"=Yobs_internal,
              "Q_wts_raw_sum" = sum_raw_wts,
              "log_pr_w_new"=log_pr_w_new,
              "log_PrW"=log_pr_w_internal))
}

computeQse_conjoint <- function(FactorsMat, Yobs,
                                pi_list,
                                assignmentProbList,
                                FactorsMat_internal_mapped = NULL,
                                log_pr_w = NULL,
                                hajek = T,
                                knownNormalizationFactor = NULL,
                                knownSigma2 = NULL,
                                hypotheticalN = NULL,
                                returnLog = T,
                                log_treatment_combs=NULL){

  if(is.null(log_treatment_combs)){
    log_treatment_combs  <- sum(log(
      sapply(1:ncol(FactorsMat),function(ze){
        length(assignmentProbList[[ze]]) }) ))
  }

  if(is.null(log_pr_w)){
    log_pr_w <- log(
      sapply(1:ncol(FactorsMat),function(ze){
        (assignmentProbList[[ze]][ FactorsMat[,ze] ]) }) )
    if(all(class(log_pr_w) == "numeric")){ log_pr_w <- sum(log_pr_w)}
    if(any(class(log_pr_w) != "numeric")){ log_pr_w <- rowSums(log_pr_w)}
  }

  # Perform weighting to obtain bound for E_pi[c_t]
  {
    if(!is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped[] <- log(unlist(pi_list)[FactorsMat_internal_mapped])}
    if(is.null(FactorsMat_internal_mapped)){FactorsMat_internal_mapped <- log( sapply(1:ncol(FactorsMat),function(ze){pi_list[[ze]][ FactorsMat[,ze] ]  })  )}
    if(any(class(FactorsMat_internal_mapped) != "numeric")){ log_pr_w_new <- rowSums(FactorsMat_internal_mapped)}
    if(all(class(FactorsMat_internal_mapped) == "numeric")){ log_pr_w_new <- sum(FactorsMat_internal_mapped)}

    my_wts = exp(log_pr_w_new   - log_pr_w  )
    if(hajek == T){
      if(is.null(knownNormalizationFactor)){  my_wts <- my_wts / sum(my_wts)}
      if(!is.null(knownNormalizationFactor)){  my_wts <- my_wts / knownNormalizationFactor}
      scaleFactor = sum(Yobs^2 * my_wts )
    }
    if(hajek == F){ scaleFactor <- mean(Yobs^2 * my_wts )   }
  }

  # Compute max prob (take maximum prob. of each Multinomial)
  log_maxProb <- sum(log(
    sapply(1:ncol(FactorsMat),function(ze){
      max(pi_list[[ze]]) })
  ))

  if(!is.null(knownSigma2)){ sigma2_hat <- knownSigma2 }
  if(is.null(knownSigma2)){ sigma2_hat <- var(Yobs) }

  # Combine terms to get VE and EV
  logN <- ifelse(is.null(hypotheticalN), yes = log(length(Yobs)), no = log(hypotheticalN))
  upperBound_se_VE_log = (log(scaleFactor) + log_treatment_combs + log_maxProb - logN)
  upperBound_se_EV_log = (log(sigma2_hat) + log_treatment_combs + log_maxProb - logN)
  upperBound_se_ <- 0.5*matrixStats::logSumExp(c(upperBound_se_EV_log,upperBound_se_VE_log))#0.5 for sqrt

  # log scale is used in optimization to improve numerical stability
  if(returnLog == F){upperBound_se_ <- exp(upperBound_se_) }
  return( upperBound_se_ )
}

n2int <- function(x){  strenv$jnp$array(x,strenv$jnp$int32)  }
