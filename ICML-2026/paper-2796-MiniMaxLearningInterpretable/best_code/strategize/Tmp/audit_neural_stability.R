`%||%` <- function(x, y) if (is.null(x)) y else x

summarize_vec <- function(x) {
  x <- as.numeric(x)
  c(
    mean = mean(x),
    sd = stats::sd(x),
    rms = sqrt(mean(x^2)),
    q50_abs = unname(stats::quantile(abs(x), 0.50)),
    q90_abs = unname(stats::quantile(abs(x), 0.90)),
    q99_abs = unname(stats::quantile(abs(x), 0.99)),
    max_abs = max(abs(x))
  )
}

clip_prob <- function(p, eps = 1e-8) pmin(pmax(as.numeric(p), eps), 1 - eps)

binary_log_loss <- function(y, logits) {
  p <- clip_prob(stats::plogis(as.numeric(logits)))
  -mean(as.numeric(y) * log(p) + (1 - as.numeric(y)) * log(1 - p))
}

binary_acc <- function(y, logits) {
  p <- stats::plogis(as.numeric(logits))
  mean(as.integer(p >= 0.5) == as.integer(y == 1))
}

extract_model_info <- function(res) {
  res$neural_model_info$ast %||%
    res$neural_model_info$dag %||%
    res$neural_model_info$ast0 %||%
    res$neural_model_info$dag0 %||%
    res$neural_model_info
}

param_scalar <- function(x, default = NA_real_) {
  if (is.null(x)) return(default)
  out <- tryCatch(as.numeric(strategize:::strenv$np$array(x))[[1L]], error = function(e) NA_real_)
  if (is.finite(out)) out else default
}

drop_low_rank <- function(params) {
  params$W_rc_r <- NULL
  params$W_rc_c <- NULL
  params$W_rc_out <- NULL
  params$alpha_rc <- NULL
  params
}

drop_additive <- function(params) {
  params$W_add_out <- NULL
  params
}

drop_calibration <- function(params) {
  params$log_calibration_scale <- NULL
  params$calibration_scale <- NULL
  params
}

predict_logits <- function(params, info, X_left, X_right, zero_i, resp_cov) {
  as.numeric(strategize:::strenv$np$array(
    strategize:::neural_predict_pair_core_prepared(
      params = params,
      model_info = info,
      Xl = X_left,
      Xr = X_right,
      pl = zero_i,
      pr = zero_i,
      resp_p = zero_i,
      resp_c = resp_cov,
      return_logits = TRUE
    )
  ))
}

fit_one <- function(label, overrides, data, p_list, X_left, X_right, zero_i, resp_cov, y_pair) {
  message("\n--- fitting ", label, " ---")
  params <- default_strategize_params(fast = TRUE)
  params$outcome_model_type <- "neural"
  params$neural_mcmc_control <- modifyList(
    params$neural_mcmc_control %||% list(),
    modifyList(
      list(
        subsample_method = "batch_vi",
        batch_size = 32L,
        ModelDims = 16L,
        ModelDepth = 1L,
        cross_candidate_encoder = "none",
        optimizer = "adam",
        svi_steps = 40L,
        svi_num_draws = 1L,
        eval_enabled = FALSE,
        early_stopping = FALSE,
        gradient_diagnostics = FALSE,
        warn_stage_imbalance_pct = 0,
        warn_min_cell_n = 0L
      ),
      overrides
    )
  )

  res <- suppressMessages(suppressWarnings(do.call(strategize, c(
    list(Y = data$Y, W = data$W, p_list = p_list),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))))
  info <- extract_model_info(res)
  stopifnot(!is.null(info), !is.null(info$params))

  full <- predict_logits(info$params, info, X_left, X_right, zero_i, resp_cov)

  info_no_cal <- info
  info_no_cal$calibration_enabled <- FALSE
  info_no_cal$calibration_method <- "none"
  uncal <- predict_logits(
    drop_calibration(info$params),
    info_no_cal,
    X_left,
    X_right,
    zero_i,
    resp_cov
  )

  no_add <- predict_logits(
    drop_additive(info$params),
    info,
    X_left,
    X_right,
    zero_i,
    resp_cov
  )
  no_lr <- predict_logits(
    drop_low_rank(info$params),
    info,
    X_left,
    X_right,
    zero_i,
    resp_cov
  )
  structural <- predict_logits(
    drop_calibration(drop_additive(drop_low_rank(info$params))),
    info_no_cal,
    X_left,
    X_right,
    zero_i,
    resp_cov
  )

  rows <- rbind(
    full = summarize_vec(full),
    uncalibrated = summarize_vec(uncal),
    no_additive = summarize_vec(no_add),
    no_low_rank = summarize_vec(no_lr),
    structural_only = summarize_vec(structural),
    calibration_delta = summarize_vec(full - uncal),
    additive_delta = summarize_vec(full - no_add),
    low_rank_delta = summarize_vec(full - no_lr)
  )
  metrics <- data.frame(
    label = label,
    component = rownames(rows),
    rows,
    log_loss = c(
      binary_log_loss(y_pair, full),
      binary_log_loss(y_pair, uncal),
      binary_log_loss(y_pair, no_add),
      binary_log_loss(y_pair, no_lr),
      binary_log_loss(y_pair, structural),
      NA, NA, NA
    ),
    accuracy = c(
      binary_acc(y_pair, full),
      binary_acc(y_pair, uncal),
      binary_acc(y_pair, no_add),
      binary_acc(y_pair, no_lr),
      binary_acc(y_pair, structural),
      NA, NA, NA
    ),
    row.names = NULL,
    check.names = FALSE
  )
  metadata <- data.frame(
    label = label,
    low_rank_interaction_rank = as.integer(info$low_rank_interaction_rank %||% 0L),
	    low_rank_logit_normalization = as.character(info$low_rank_logit_normalization %||% NA),
	    low_rank_logit_transform = as.character(info$low_rank_logit_transform %||% NA),
	    additive_utility_mode = as.character(info$additive_utility_mode %||% NA),
	    additive_utility_normalization = as.character(info$additive_utility_normalization %||% NA),
	    additive_head_weight_target_rms = as.numeric(info$additive_head_weight_target_rms %||% NA_real_),
	    has_additive_utility = isTRUE(info$has_additive_utility),
    calibration_enabled = isTRUE(info$calibration_enabled),
    calibration_scale = as.numeric(info$calibration_scale %||% param_scalar(info$params$calibration_scale, 1)),
    log_calibration_scale = param_scalar(info$params$log_calibration_scale),
    pairwise_bernoulli_logit_scale = as.numeric(info$pairwise_bernoulli_logit_scale %||% 1),
    W_out_rms = sqrt(mean(as.numeric(strategize:::strenv$np$array(info$params$W_out))^2)),
    W_add_out_rms = if (!is.null(info$params$W_add_out)) {
      sqrt(mean(as.numeric(strategize:::strenv$np$array(info$params$W_add_out))^2))
    } else {
      NA_real_
    },
    W_rc_out_rms = if (!is.null(info$params$W_rc_out)) {
      sqrt(mean(as.numeric(strategize:::strenv$np$array(info$params$W_rc_out))^2))
    } else {
      NA_real_
    },
    alpha_rc = param_scalar(info$params$alpha_rc),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  list(metrics = metrics, metadata = metadata)
}

devtools::load_all(".", quiet = TRUE)
source("tests/testthat/helper-strategize.R")
strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

withr::local_envvar(c(
  STRATEGIZE_NEURAL_FAST_MCMC = "true",
  STRATEGIZE_NEURAL_SKIP_EVAL = "1"
))
withr::local_seed(20260610)

data <- generate_pairwise_performance_test_data(
  n_pairs = 240L,
  n_factors = 3L,
  n_levels = 2L,
  seed = 20260610
)
p_list <- generate_test_p_list(data$W)

W_numeric <- as.matrix(sapply(seq_len(ncol(data$W)), function(d_) {
  match(data$W[, d_], names(p_list[[d_]]))
}))
to_index_matrix <- function(x_mat) {
  x_mat <- as.matrix(x_mat)
  x_int <- matrix(as.integer(x_mat) - 1L, nrow = nrow(x_mat), ncol = ncol(x_mat))
  x_int[x_int < 0L | is.na(x_int)] <- 0L
  x_int
}
idx_left <- which(data$profile_order == 1L)
idx_right <- which(data$profile_order == 2L)
n_obs <- length(idx_left)
X_left <- strategize:::strenv$jnp$array(to_index_matrix(W_numeric[idx_left, , drop = FALSE]))$
  astype(strategize:::strenv$jnp$int32)
X_right <- strategize:::strenv$jnp$array(to_index_matrix(W_numeric[idx_right, , drop = FALSE]))$
  astype(strategize:::strenv$jnp$int32)
zero_i <- strategize:::strenv$jnp$zeros(list(n_obs), dtype = strategize:::strenv$jnp$int32)
resp_cov <- strategize:::strenv$jnp$zeros(list(n_obs, 0L), dtype = strategize:::strenv$jnp$float32)
y_pair <- data$Y[idx_left]

configs <- list(
  stable_rank0_no_add_no_cal = list(
    low_rank_interaction_rank = 0L,
    additive_utility = "off",
    calibration = list(enabled = FALSE)
  ),
  rank0_add_cal_auto = list(
    low_rank_interaction_rank = 0L,
    additive_utility = "auto",
    calibration = list(enabled = "auto")
  ),
	  rank4_rms_no_add_no_cal = list(
	    low_rank_interaction_rank = 4L,
	    additive_utility = "off",
	    calibration = list(enabled = FALSE)
	  ),
	  rank4_rms_add_norm_no_cal = list(
	    low_rank_interaction_rank = 4L,
	    additive_utility = "auto",
	    calibration = list(enabled = FALSE)
	  ),
	  rank4_rms_add_cal_auto = list(
	    low_rank_interaction_rank = 4L,
	    additive_utility = "auto",
	    calibration = list(enabled = "auto")
	  ),
	  rank4_rms_add_raw_cal_auto = list(
	    low_rank_interaction_rank = 4L,
	    additive_utility = "auto",
	    additive_utility_normalization = "none",
	    calibration = list(enabled = "auto")
	  ),
	  rank4_no_norm_add_cal_auto = list(
	    low_rank_interaction_rank = 4L,
	    low_rank_logit_normalization = "none",
	    additive_utility = "auto",
	    additive_utility_normalization = "none",
	    calibration = list(enabled = "auto")
	  )
	)

results <- lapply(names(configs), function(label) {
  fit_one(label, configs[[label]], data, p_list, X_left, X_right, zero_i, resp_cov, y_pair)
})
metrics <- do.call(rbind, lapply(results, `[[`, "metrics"))
metadata <- do.call(rbind, lapply(results, `[[`, "metadata"))

out_prefix <- file.path("Tmp", sprintf("audit_neural_stability_%s", format(Sys.time(), "%Y%m%d_%H%M%S")))
write.csv(metrics, paste0(out_prefix, "_metrics.csv"), row.names = FALSE)
write.csv(metadata, paste0(out_prefix, "_metadata.csv"), row.names = FALSE)
saveRDS(list(metrics = metrics, metadata = metadata), paste0(out_prefix, ".rds"))

cat("\nMetadata:\n")
print(metadata, row.names = FALSE)
cat("\nFull/structural summaries:\n")
print(metrics[metrics$component %in% c("full", "uncalibrated", "structural_only", "calibration_delta", "additive_delta", "low_rank_delta"), ], row.names = FALSE)
cat("\nWrote audit files with prefix: ", out_prefix, "\n", sep = "")
