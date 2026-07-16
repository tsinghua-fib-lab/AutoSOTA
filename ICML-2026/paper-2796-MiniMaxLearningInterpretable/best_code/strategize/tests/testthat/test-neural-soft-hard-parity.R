# =============================================================================
# Soft-path train/serve parity: calibration temperature + additive head
# =============================================================================
# Regression tests for the fix restoring the learned calibration temperature and
# the additive main-effects head to neural_predict_pair_soft -- the exact function
# the adversarial pi* optimizer differentiates. Before the fix both transforms were
# applied in training and hard-prediction but silently dropped from the soft path,
# so the optimizer maximized a different function than was trained and validated.
#
# The scaffold mirrors test-choice-token-gradient.R (a hand-built model_info/params
# for the two-tower pairwise path), plus the fused-tokenization helpers.

neural_parity_build_model <- function() {
  strategize:::initialize_jax()
  strenv <- strategize:::strenv
  model_dims <- 4L
  ff_dim <- 4L

  model_info <- list(
    n_factors = 1L,
    factor_index_list = list(strenv$jnp$array(as.integer(c(0L, 1L)))),
    implicit = FALSE,
    model_dims = model_dims,
    model_depth = 1L,
    n_heads = 1L,
    head_dim = model_dims,
    likelihood = "bernoulli",
    resp_cov_mean = NULL,
    n_resp_covariates = 0L,
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(0L)),
    params = NULL
  )
  model_info <- neural_test_add_fused_factor_schema(model_info, factor_levels = 2L)

  params <- list(
    E_choice = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(c(1, 0, 0, 0,
               0, 1, 0, 0), nrow = 2, byrow = TRUE),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$zeros(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(1L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    W_q_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
    W_k_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
    W_v_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
    W_o_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
    RMS_attn_l1 = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    RMS_ff_l1 = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    RMS_final = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    W_ff1_l1 = strenv$jnp$zeros(list(model_dims, 2L * ff_dim), dtype = strenv$dtj),
    W_ff2_l1 = strenv$jnp$zeros(list(ff_dim, model_dims), dtype = strenv$dtj),
    W_out = strenv$jnp$ones(list(model_dims, 1L), dtype = strenv$dtj),
    b_out = strenv$jnp$zeros(list(1L), dtype = strenv$dtj)
  )
  params <- neural_test_add_fused_factor_params(
    params,
    model_dims = model_dims,
    model_info = model_info
  )

  list(
    model_info = model_info,
    params = params,
    model_dims = model_dims,
    pi_left = strenv$jnp$array(c(1, 0), dtype = strenv$dtj),
    pi_right = strenv$jnp$array(c(0, 1), dtype = strenv$dtj)
  )
}

neural_parity_soft_logit <- function(model_info, params, pi_left, pi_right) {
  out <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 0L,
    model_info = model_info,
    params = params,
    return_logits = TRUE
  )
  as.numeric(strategize:::cs2step_neural_to_r_array(out))[[1L]]
}

test_that("soft pairwise path applies the learned calibration temperature", {
  skip_on_cran()
  skip_if_no_jax()

  b <- neural_parity_build_model()
  strenv <- strategize:::strenv

  s <- 1.4
  params_cal <- b$params
  params_cal$log_calibration_scale <- strenv$jnp$array(log(s), dtype = strenv$dtj)

  mi_off <- b$model_info
  mi_off$calibration_enabled <- FALSE
  mi_off$additive_utility_mode <- "off"

  mi_on <- b$model_info
  mi_on$calibration_enabled <- TRUE
  mi_on$calibration_method <- "logit_scale"
  mi_on$additive_utility_mode <- "off"

  L_off <- neural_parity_soft_logit(mi_off, params_cal, b$pi_left, b$pi_right)
  L_on  <- neural_parity_soft_logit(mi_on,  params_cal, b$pi_left, b$pi_right)

  # A nonzero base logit makes the temperature effect detectable.
  expect_gt(abs(L_off), 1e-6)
  # For a Bernoulli head the calibration temperature multiplies the logit by s, and
  # (post-fix) is applied before the return_logits exit used by the kappa
  # push-forward. Before the fix the soft path never applied it, so L_on == L_off.
  expect_equal(L_on, s * L_off, tolerance = 1e-4)
})

test_that("soft pairwise path includes the additive main-effects head", {
  skip_on_cran()
  skip_if_no_jax()

  b <- neural_parity_build_model()
  strenv <- strategize:::strenv

  mi <- b$model_info
  mi$calibration_enabled <- FALSE
  mi$additive_utility_mode <- "on"

  # W_add_out = 0 -> the additive head contributes nothing; a nonzero head must move
  # the logit. Before the fix the soft path ignored W_add_out entirely (L_add == L0).
  params0 <- b$params
  params0$W_add_out <- strenv$jnp$zeros(list(b$model_dims, 1L), dtype = strenv$dtj)

  params_add <- b$params
  params_add$W_add_out <- strenv$jnp$array(
    matrix(c(0.7, -0.3, 0.5, 0.2), ncol = 1L),
    dtype = strenv$dtj
  )

  L0 <- neural_parity_soft_logit(mi, params0, b$pi_left, b$pi_right)
  L_add <- neural_parity_soft_logit(mi, params_add, b$pi_left, b$pi_right)

  expect_gt(abs(L_add - L0), 1e-4)
})
