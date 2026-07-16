`%||%` <- function(x, y) if (is.null(x)) y else x

summarize <- function(x) {
  x <- as.numeric(x)
  c(
    mean = mean(x),
    sd = sd(x),
    rms = sqrt(mean(x^2)),
    q50_abs = unname(quantile(abs(x), 0.50)),
    q90_abs = unname(quantile(abs(x), 0.90)),
    q99_abs = unname(quantile(abs(x), 0.99)),
    max_abs = max(abs(x))
  )
}

devtools::load_all(".", quiet = TRUE)
source("tests/testthat/helper-strategize.R")

strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
strenv <- strategize:::strenv

withr::local_envvar(c(
  STRATEGIZE_NEURAL_FAST_MCMC = "true",
  STRATEGIZE_NEURAL_SKIP_EVAL = "1"
))

data <- generate_pairwise_performance_test_data(
  n_pairs = 300L,
  n_factors = 3L,
  n_levels = 2L,
  seed = 20260327
)
params <- default_strategize_params(fast = TRUE)
params$outcome_model_type <- "neural"
params$neural_mcmc_control <- modifyList(
  params$neural_mcmc_control %||% list(),
  list(
    subsample_method = "batch_vi",
    batch_size = 32L,
    ModelDims = 16L,
    ModelDepth = 1L,
    low_rank_interaction_rank = 4L,
    cross_candidate_encoder = "none",
    optimizer = "adam",
    svi_steps = 80L,
    svi_num_draws = 1L,
    eval_enabled = FALSE,
    early_stopping = FALSE,
    warn_stage_imbalance_pct = 0,
    warn_min_cell_n = 0L
  )
)
p_list <- generate_test_p_list(data$W)

res <- suppressMessages(suppressWarnings(do.call(strategize, c(
  list(Y = data$Y, W = data$W, p_list = p_list),
  data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
  params
))))

info <- res$neural_model_info$ast %||%
  res$neural_model_info$dag %||%
  res$neural_model_info$ast0 %||%
  res$neural_model_info$dag0
stopifnot(!is.null(info), !is.null(info$params))

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
X_left <- strenv$jnp$array(to_index_matrix(W_numeric[idx_left, , drop = FALSE]))$astype(strenv$jnp$int32)
X_right <- strenv$jnp$array(to_index_matrix(W_numeric[idx_right, , drop = FALSE]))$astype(strenv$jnp$int32)
zero_i <- strenv$jnp$zeros(list(n_obs), dtype = strenv$jnp$int32)
resp_cov <- strenv$jnp$zeros(list(n_obs, 0L), dtype = strenv$jnp$float32)

raw_info <- info
raw_info$low_rank_logit_transform <- "none"
raw_info$low_rank_logit_bound <- NULL
raw_info$low_rank_logit_softness <- NULL

base_params <- info$params
base_params$W_rc_r <- NULL
base_params$W_rc_c <- NULL
base_params$W_rc_out <- NULL
base_params$alpha_rc <- NULL

full_raw <- strategize:::neural_predict_pair_core_prepared(
  params = info$params,
  model_info = raw_info,
  Xl = X_left,
  Xr = X_right,
  pl = zero_i,
  pr = zero_i,
  resp_p = zero_i,
  resp_c = resp_cov,
  return_logits = TRUE
)
base_raw <- strategize:::neural_predict_pair_core_prepared(
  params = base_params,
  model_info = raw_info,
  Xl = X_left,
  Xr = X_right,
  pl = zero_i,
  pr = zero_i,
  resp_p = zero_i,
  resp_c = resp_cov,
  return_logits = TRUE
)
bounded <- strategize:::neural_predict_pair_core_prepared(
  params = info$params,
  model_info = info,
  Xl = X_left,
  Xr = X_right,
  pl = zero_i,
  pr = zero_i,
  resp_p = zero_i,
  resp_c = resp_cov,
  return_logits = TRUE
)

full_raw <- as.numeric(strenv$np$array(full_raw))
base_raw <- as.numeric(strenv$np$array(base_raw))
bounded <- as.numeric(strenv$np$array(bounded))
rc_delta <- full_raw - base_raw

cat("\nModel metadata:\n")
print(info[c(
  "low_rank_interaction_rank",
  "cross_candidate_encoder_mode",
  "low_rank_logit_transform",
  "low_rank_logit_bound",
  "low_rank_logit_softness"
)])

cat("\nFitted logit component scale on training pairs:\n")
print(round(rbind(
  base_raw = summarize(base_raw),
  rc_delta = summarize(rc_delta),
  full_raw = summarize(full_raw),
  bounded = summarize(bounded)
), 4))

cat("\nParameter RMS:\n")
param_rms <- function(x) sqrt(mean(as.numeric(strenv$np$array(x))^2))
print(round(c(
  W_out = param_rms(info$params$W_out),
  W_rc_r = param_rms(info$params$W_rc_r),
  W_rc_c = param_rms(info$params$W_rc_c),
  W_rc_out = param_rms(info$params$W_rc_out),
  alpha_rc = as.numeric(strenv$np$array(info$params$alpha_rc))
), 6))

