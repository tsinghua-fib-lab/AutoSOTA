# =============================================================================
# Choice Token (CLS) Gradient Flow Tests
# =============================================================================

test_that("learned choice token receives gradients and mixes information", {
  skip_on_cran()
  skip_if_no_jax()

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
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 2L
  )

  params_base <- list(
    E_choice = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(c(1, 0, 0, 0,
               0, 1, 0, 0),
             nrow = 2, byrow = TRUE),
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
  params_base <- neural_test_add_fused_factor_params(
    params_base,
    model_dims = model_dims,
    model_info = model_info
  )

  pi_left <- strenv$jnp$array(c(0.8, 0.2), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0.2, 0.8), dtype = strenv$dtj)

  logit_sum <- function(E_choice, pi_left_in, pi_right_in) {
    params <- params_base
    params$E_choice <- E_choice

    logits <- strategize:::neural_predict_pair_soft(
      pi_left = pi_left_in,
      pi_right = pi_right_in,
      party_left_idx = 0L,
      party_right_idx = 0L,
      resp_party_idx = 0L,
      model_info = model_info,
      resp_cov_vec = NULL,
      params = params,
      return_logits = TRUE
    )
    strenv$jnp$sum(logits)
  }

  grad_choice <- strenv$jax$grad(logit_sum, argnums = 0L)
  grad_pi_left <- strenv$jax$grad(logit_sum, argnums = 1L)
  grad_pi_right <- strenv$jax$grad(logit_sum, argnums = 2L)

  g_choice <- as.numeric(strenv$np$array(grad_choice(params_base$E_choice, pi_left, pi_right)))
  g_left <- as.numeric(strenv$np$array(grad_pi_left(params_base$E_choice, pi_left, pi_right)))
  g_right <- as.numeric(strenv$np$array(grad_pi_right(params_base$E_choice, pi_left, pi_right)))

  expect_length(g_choice, model_dims)
  expect_length(g_left, 2L)
  expect_length(g_right, 2L)
  expect_true(all(is.finite(g_choice)))
  expect_true(all(is.finite(g_left)))
  expect_true(all(is.finite(g_right)))
  expect_true(any(abs(g_choice) > 1e-8))
  expect_true(any(abs(g_left) > 1e-8))
  expect_true(any(abs(g_right) > 1e-8))
})
