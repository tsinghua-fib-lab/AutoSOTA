# =============================================================================
# Pairwise Utility Consistency Tests
# =============================================================================

test_that("readout cls token families are gated by low-rank rank", {
  legacy_levels <- strategize:::neural_token_family_levels()
  readout_levels <- strategize:::neural_token_family_levels(include_readout_cls = TRUE)

  expect_false("respondent_cls" %in% legacy_levels)
  expect_false("candidate_cls" %in% legacy_levels)
  expect_true("respondent_cls" %in% readout_levels)
  expect_true("candidate_cls" %in% readout_levels)
  expect_lt(match("candidate_cls", readout_levels), match("choice", readout_levels))
  expect_identical(
    strategize:::neural_readout_embedding_families(low_rank_interaction_rank = 0L),
    "choice"
  )
  expect_identical(
    strategize:::neural_readout_embedding_families(low_rank_interaction_rank = 4L),
    c("choice", "respondent_cls", "candidate_cls")
  )
})

context_head_fused_fixture <- function(cross_mode = "attn",
                                       residual_mode = "standard") {
  strenv <- strategize:::strenv
  jnp <- strenv$jnp
  dtj <- strenv$dtj
  token_levels <- strategize:::neural_token_family_levels()

  factor_text <- matrix(c(1, 0), nrow = 1L, byrow = TRUE)
  rownames(factor_text) <- "alpha"
  level_text <- list(
    alpha = matrix(
      c(10, 0,
        20, 0,
        0, 0),
      nrow = 3L,
      byrow = TRUE
    )
  )
  factor_struct <- matrix(c(0, 0), nrow = 1L, byrow = TRUE)
  colnames(factor_struct) <- c("s1", "s2")
  level_struct <- list(
    alpha = matrix(
      c(0, 0,
        0, 0,
        0, 0),
      nrow = 3L,
      byrow = TRUE
    )
  )

  model_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = 2L,
    factor_name_text = factor_text,
    level_name_text = level_text,
    factor_struct_matrix = factor_struct,
    level_struct_matrices = level_struct,
    factor_struct_feature_names = colnames(factor_struct),
    level_struct_feature_names = c("s1", "s2"),
    default_factor_order = c(0L),
    factor_tokenization = "fused",
    max_factor_tokens = 1L,
    token_family_levels = token_levels
  )
  transformer_info <- strategize:::neural_make_transformer_model_info(
    model_depth = 1L,
    model_dims = 2L,
    n_heads = 1L,
    head_dim = 2L,
    residual_mode = residual_mode
  )
  model_info[names(transformer_info)] <- transformer_info
  model_info$n_factors <- 1L
  model_info$factor_index_list <- list(jnp$array(as.integer(c(0L, 1L))))
  model_info$implicit <- FALSE
  model_info$likelihood <- "bernoulli"
  model_info$resp_cov_mean <- NULL
  model_info$n_resp_covariates <- 0L
  model_info$resp_party_levels <- c("A", "B")
  model_info$cand_party_to_resp_idx <- jnp$array(as.integer(c(0L)))
  model_info$cross_candidate_encoder_mode <- cross_mode
  model_info$n_candidate_tokens <- 1L

  params <- list(
    E_choice = jnp$array(c(1, 2), dtype = dtj),
    E_factor_fused_base = jnp$array(c(0, 0), dtype = dtj),
    W_factor_fuse_1 = jnp$array({
      w1 <- matrix(0, nrow = 8L, ncol = 8L)
      w1[, 5:6] <- rbind(diag(2L), diag(2L), diag(2L), diag(2L))
      w1
    }, dtype = dtj),
    b_factor_fuse_1 = jnp$array(c(rep(neural_test_swiglu_unit_gate(), 4L), rep(0, 4L)), dtype = dtj),
    W_factor_fuse_2 = jnp$array(rbind(diag(2L), matrix(0, nrow = 2L, ncol = 2L)), dtype = dtj),
    b_factor_fuse_2 = jnp$zeros(list(2L), dtype = dtj),
    E_token_family = jnp$zeros(list(length(token_levels), 2L), dtype = dtj),
    E_sep = jnp$zeros(list(2L), dtype = dtj),
    E_segment = jnp$zeros(list(2L, 2L), dtype = dtj),
    W_factor_name_text = jnp$eye(2L, dtype = dtj),
    W_level_name_text = jnp$eye(2L, dtype = dtj),
    W_factor_struct = jnp$zeros(list(2L, 2L), dtype = dtj),
    W_level_struct = jnp$zeros(list(2L, 2L), dtype = dtj),
    E_party = NULL,
    E_rel = NULL,
    E_stage = NULL,
    E_resp_party = NULL,
    E_matchup = NULL,
    W_q_l1 = jnp$zeros(list(2L, 2L), dtype = dtj),
    W_k_l1 = jnp$zeros(list(2L, 2L), dtype = dtj),
    W_v_l1 = jnp$zeros(list(2L, 2L), dtype = dtj),
    W_o_l1 = jnp$zeros(list(2L, 2L), dtype = dtj),
    RMS_attn_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_ff_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_q_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_k_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_final = if (identical(residual_mode, "full_attn")) {
      jnp$ones(list(2L), dtype = dtj)
    } else {
      NULL
    },
    W_ff1_l1 = jnp$zeros(list(2L, 4L), dtype = dtj),
    W_ff2_l1 = jnp$zeros(list(2L, 2L), dtype = dtj),
    W_out = jnp$ones(list(2L, 1L), dtype = dtj),
    b_out = jnp$zeros(list(1L), dtype = dtj)
  )
  if (identical(residual_mode, "full_attn")) {
    params$pseudo_query_attn_l1 <- jnp$zeros(list(2L), dtype = dtj)
    params$pseudo_query_ff_l1 <- jnp$zeros(list(2L), dtype = dtj)
    params$pseudo_query_final <- jnp$zeros(list(2L), dtype = dtj)
  }

  list(
    model_info = model_info,
    params = params,
    pi_left = jnp$array(c(1, 0), dtype = dtj),
    pi_right = jnp$array(c(0, 1), dtype = dtj)
  )
}

test_that("pairwise logits are utility differences and swap invariant", {
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
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    params = NULL
  )
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 2L
  )

  params_base <- list(
    E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(1, nrow = 2, ncol = model_dims),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
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

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  logits_lr <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  phi_pair <- strategize:::neural_encode_pair_soft_batched(
    pi_left, pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    model_info = model_info,
    resp_party_idx = 1L,
    stage_idx = 1L,
    matchup_idx = NULL,
    resp_cov_vec = NULL,
    params = params_base
  )
  u_left <- strenv$jnp$einsum("nm,mo->no", phi_pair$phi_left, params_base$W_out) + params_base$b_out
  u_right <- strenv$jnp$einsum("nm,mo->no", phi_pair$phi_right, params_base$W_out) + params_base$b_out
  expected_logits <- u_left - u_right

  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    as.numeric(strenv$np$array(expected_logits)),
    tolerance = 1e-6
  )

  logits_rl <- strategize:::neural_predict_pair_soft(
    pi_left = pi_right,
    pi_right = pi_left,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    -as.numeric(strenv$np$array(logits_rl)),
    tolerance = 1e-6
  )
})

test_that("pairwise logits include cross-candidate interaction terms when enabled", {
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
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    cross_candidate_encoder = TRUE,
    params = NULL
  )
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 2L
  )

  params_base <- list(
    E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(1, nrow = 2, ncol = model_dims),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
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
    b_out = strenv$jnp$zeros(list(1L), dtype = strenv$dtj),
    M_cross = strenv$jnp$ones(list(model_dims, model_dims), dtype = strenv$dtj),
    W_cross_out = strenv$jnp$array(c(0.25), dtype = strenv$dtj)
  )
  params_base <- neural_test_add_fused_factor_params(
    params_base,
    model_dims = model_dims,
    model_info = model_info
  )

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  phi_pair <- strategize:::neural_encode_pair_soft_batched(
    pi_left, pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    model_info = model_info,
    resp_party_idx = 1L,
    stage_idx = 1L,
    matchup_idx = NULL,
    resp_cov_vec = NULL,
    params = params_base
  )
  phi_left <- phi_pair$phi_left
  phi_right <- phi_pair$phi_right

  u_left <- strenv$jnp$einsum("nm,mo->no", phi_left, params_base$W_out) + params_base$b_out
  u_right <- strenv$jnp$einsum("nm,mo->no", phi_right, params_base$W_out) + params_base$b_out
  base_logits <- u_left - u_right
  cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, params_base$M_cross, phi_right)
  cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
  expected_logits <- strategize:::neural_apply_cross_term(
    base_logits,
    phi_left,
    phi_right,
    params_base$M_cross,
    params_base$W_cross_out,
    out_dim = 1L
  )

  logits_lr <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  cross_numeric <- as.numeric(strenv$np$array(cross_term))
  expect_true(any(abs(cross_numeric) > 1e-8))
  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    as.numeric(strenv$np$array(expected_logits)),
    tolerance = 1e-6
  )
})

test_that("pairwise logits ignore cross-candidate interaction terms when disabled", {
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
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    cross_candidate_encoder = FALSE,
    params = NULL
  )
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 2L
  )

  params_base <- list(
    E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      matrix(1, nrow = 2, ncol = model_dims),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
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
    b_out = strenv$jnp$zeros(list(1L), dtype = strenv$dtj),
    M_cross = strenv$jnp$ones(list(model_dims, model_dims), dtype = strenv$dtj),
    W_cross_out = strenv$jnp$array(c(0.25), dtype = strenv$dtj)
  )
  params_base <- neural_test_add_fused_factor_params(
    params_base,
    model_dims = model_dims,
    model_info = model_info
  )

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  phi_pair <- strategize:::neural_encode_pair_soft_batched(
    pi_left, pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    model_info = model_info,
    resp_party_idx = 1L,
    stage_idx = 1L,
    matchup_idx = NULL,
    resp_cov_vec = NULL,
    params = params_base
  )
  phi_left <- phi_pair$phi_left
  phi_right <- phi_pair$phi_right

  u_left <- strenv$jnp$einsum("nm,mo->no", phi_left, params_base$W_out) + params_base$b_out
  u_right <- strenv$jnp$einsum("nm,mo->no", phi_right, params_base$W_out) + params_base$b_out
  base_logits <- u_left - u_right
  cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, params_base$M_cross, phi_right)
  cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
  cross_out <- strenv$jnp$reshape(params_base$W_cross_out, list(1L, -1L))
  expected_logits <- base_logits + cross_term * cross_out

  logits_lr <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  cross_numeric <- as.numeric(strenv$np$array(cross_term))
  expect_true(any(abs(cross_numeric) > 1e-8))
  expect_equal(
    as.numeric(strenv$np$array(logits_lr)),
    as.numeric(strenv$np$array(base_logits)),
    tolerance = 1e-6
  )
  expect_true(
    any(abs(as.numeric(strenv$np$array(logits_lr)) -
              as.numeric(strenv$np$array(expected_logits))) > 1e-6)
  )
})

test_that("pairwise logits use full cross-candidate encoder when mode is full", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  model_dims <- 2L
  ff_dim <- 2L

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
    resp_party_levels = c("A", "B"),
    cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
    cross_candidate_encoder_mode = "full",
    params = NULL
  )
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 2L
  )

  params_base <- list(
    E_choice = strenv$jnp$array(c(1, 2), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
    E_party = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$zeros(list(3L, model_dims), dtype = strenv$dtj),
    E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
    E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
    E_sep = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    E_segment = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
    W_q_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    W_k_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    W_v_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    W_o_l1 = strenv$jnp$zeros(list(model_dims, model_dims), dtype = strenv$dtj),
    RMS_attn_l1 = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    RMS_ff_l1 = strenv$jnp$zeros(list(model_dims), dtype = strenv$dtj),
    W_ff1_l1 = strenv$jnp$zeros(list(model_dims, 2L * ff_dim), dtype = strenv$dtj),
    W_ff2_l1 = strenv$jnp$zeros(list(ff_dim, model_dims), dtype = strenv$dtj),
    W_out = strenv$jnp$ones(list(model_dims, 1L), dtype = strenv$dtj),
    b_out = strenv$jnp$zeros(list(1L), dtype = strenv$dtj),
    RMS_final = NULL
  )
  params_base <- neural_test_add_fused_factor_params(
    params_base,
    model_dims = model_dims,
    model_info = model_info
  )

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  logits_full <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  model_info_none <- model_info
  model_info_none$cross_candidate_encoder_mode <- "none"
  logits_none <- strategize:::neural_predict_pair_soft(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = 1L,
    model_info = model_info_none,
    resp_cov_vec = NULL,
    params = params_base,
    return_logits = TRUE
  )

  expected <- sum(as.numeric(strenv$np$array(params_base$E_choice)))

  expect_equal(
    as.numeric(strenv$np$array(logits_full)),
    expected,
    tolerance = 1e-6
  )
  expect_equal(
    as.numeric(strenv$np$array(logits_none)),
    0,
    tolerance = 1e-6
  )
  expect_true(
    abs(as.numeric(strenv$np$array(logits_full)) -
          as.numeric(strenv$np$array(logits_none))) > 1e-6
  )
})

test_that("attn mode logits are antisymmetric across likelihoods (soft path)", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  build_attn_fixture <- function(likelihood, n_outcomes) {
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
      likelihood = likelihood,
      resp_cov_mean = NULL,
      n_resp_covariates = 0L,
      resp_party_levels = c("A", "B"),
      cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
      n_candidate_tokens = 3L,
      cross_candidate_encoder_mode = "attn",
      params = NULL
    )
    model_info <- neural_test_add_fused_factor_schema(
      model_info,
      factor_levels = 2L,
      max_factor_tokens = 3L
    )

    W_out <- strenv$jnp$ones(list(model_dims, n_outcomes), dtype = strenv$dtj)
    b_out <- strenv$jnp$zeros(list(n_outcomes), dtype = strenv$dtj)

    params <- list(
      E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
      E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
      E_factor_1 = strenv$jnp$array(
        rbind(c(1, 0, 0, 0), c(0, 1, 0, 0)),
        dtype = strenv$dtj
      ),
      E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
      E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
      E_stage = strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj),
      E_resp_party = strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj),
      W_q_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
      W_k_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
      W_v_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
      W_o_l1 = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
      RMS_attn_l1 = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
      RMS_ff_l1 = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
      RMS_final = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
      W_ff1_l1 = strenv$jnp$zeros(list(model_dims, 2L * ff_dim), dtype = strenv$dtj),
      W_ff2_l1 = strenv$jnp$zeros(list(ff_dim, model_dims), dtype = strenv$dtj),
      W_out = W_out,
      b_out = b_out,
      alpha_cross = strenv$jnp$array(0.5, dtype = strenv$dtj),
      RMS_cross = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
      RMS_merge_cross = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
      W_q_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
      W_k_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
      W_v_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj),
      W_o_cross = strenv$jnp$eye(model_dims, dtype = strenv$dtj)
    )
    params <- neural_test_add_fused_factor_params(
      params,
      model_dims = model_dims,
      model_info = model_info
    )

    list(model_info = model_info, params = params)
  }

  pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
  pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)

  for (likelihood in c("bernoulli", "categorical", "normal")) {
    n_outcomes <- if (likelihood == "categorical") 2L else 1L
    fixture <- build_attn_fixture(likelihood, n_outcomes)
    logits_lr <- strategize:::neural_predict_pair_soft(
      pi_left = pi_left,
      pi_right = pi_right,
      party_left_idx = 0L,
      party_right_idx = 0L,
      resp_party_idx = 1L,
      model_info = fixture$model_info,
      resp_cov_vec = NULL,
      params = fixture$params,
      return_logits = TRUE
    )
    logits_rl <- strategize:::neural_predict_pair_soft(
      pi_left = pi_right,
      pi_right = pi_left,
      party_left_idx = 0L,
      party_right_idx = 0L,
      resp_party_idx = 1L,
      model_info = fixture$model_info,
      resp_cov_vec = NULL,
      params = fixture$params,
      return_logits = TRUE
    )
    lr_num <- as.numeric(strenv$np$array(logits_lr))
    rl_num <- as.numeric(strenv$np$array(logits_rl))
    expect_lt(max(abs(lr_num + rl_num)), 1e-6)
  }
})

test_that("attn candidate token slicing is stable with context tokens", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv

  model_dims <- 4L
  ff_dim <- 4L
  base_params <- list(
    E_choice = strenv$jnp$ones(list(model_dims), dtype = strenv$dtj),
    E_feature_id = strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj),
    E_factor_1 = strenv$jnp$array(
      rbind(c(1, 0, 0, 0), c(0, 1, 0, 0)),
      dtype = strenv$dtj
    ),
    E_party = strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj),
    E_rel = strenv$jnp$ones(list(3L, model_dims), dtype = strenv$dtj),
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
  scenarios <- list(
    list(has_ctx = FALSE, has_matchup = FALSE, has_resp_cov = FALSE),
    list(has_ctx = TRUE, has_matchup = TRUE, has_resp_cov = TRUE)
  )

  for (sc in scenarios) {
    params <- base_params
    if (isTRUE(sc$has_ctx)) {
      params$E_stage <- strenv$jnp$zeros(list(2L, 2L, model_dims), dtype = strenv$dtj)
      params$E_resp_party <- strenv$jnp$zeros(list(2L, model_dims), dtype = strenv$dtj)
    } else {
      params$E_stage <- NULL
      params$E_resp_party <- NULL
    }
    if (isTRUE(sc$has_matchup)) {
      params$E_matchup <- strenv$jnp$zeros(list(1L, model_dims), dtype = strenv$dtj)
    }
    if (isTRUE(sc$has_resp_cov)) {
      params$W_resp_x <- strenv$jnp$ones(list(1L, model_dims), dtype = strenv$dtj)
    }

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
      n_resp_covariates = if (isTRUE(sc$has_resp_cov)) 1L else 0L,
      resp_party_levels = c("A", "B"),
      cand_party_to_resp_idx = strenv$jnp$array(as.integer(c(0L))),
      n_candidate_tokens = 3L,
      cross_candidate_encoder_mode = "attn",
      params = NULL
    )
    model_info <- neural_test_add_fused_factor_schema(
      model_info,
      factor_levels = 2L,
      max_factor_tokens = 3L
    )
    params <- neural_test_add_fused_factor_params(
      params,
      model_dims = model_dims,
      model_info = model_info
    )

    pi_left <- strenv$jnp$array(c(1, 0), dtype = strenv$dtj)
    pi_right <- strenv$jnp$array(c(0, 1), dtype = strenv$dtj)
    resp_cov_vec <- if (isTRUE(sc$has_resp_cov)) strenv$jnp$array(c(0.5), dtype = strenv$dtj) else NULL
    matchup_idx <- if (isTRUE(sc$has_matchup)) 0L else NULL
    stage_idx <- if (isTRUE(sc$has_ctx)) 1L else NULL

    out <- strategize:::neural_encode_pair_soft_batched(
      pi_left, pi_right,
      party_left_idx = 0L,
      party_right_idx = 0L,
      model_info = model_info,
      resp_party_idx = 1L,
      stage_idx = stage_idx,
      matchup_idx = matchup_idx,
      resp_cov_vec = resp_cov_vec,
      params = params,
      return_tokens = TRUE
    )

    expect_equal(as.integer(out$cand_left_out$shape[[2]]), 1L)
    expect_equal(as.integer(out$cand_right_out$shape[[2]]), 1L)
  }
})

test_that("full attention residual attn pair encoding uses readout candidate tokens", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv
  jnp <- strenv$jnp
  np <- strenv$np
  dtj <- strenv$dtj

  model_info <- strategize:::neural_make_transformer_model_info(
    model_depth = 1L,
    model_dims = 1L,
    n_heads = 1L,
    head_dim = 1L,
    residual_mode = "full_attn"
  )
  model_info$n_factors <- 1L
  model_info$factor_index_list <- list(jnp$array(as.integer(c(0L, 1L))))
  model_info$implicit <- FALSE
  model_info$likelihood <- "bernoulli"
  model_info$resp_cov_mean <- NULL
  model_info$n_resp_covariates <- 0L
  model_info$resp_party_levels <- c("A", "B")
  model_info$cand_party_to_resp_idx <- jnp$array(as.integer(c(0L)))
  model_info$n_candidate_tokens <- 3L
  model_info$cross_candidate_encoder_mode <- "attn"
  model_info <- neural_test_add_fused_factor_schema(
    model_info,
    factor_levels = 2L,
    max_factor_tokens = 3L
  )

  params <- list(
    E_choice = jnp$ones(reticulate::tuple(1L), dtype = dtj) * 10,
    E_feature_id = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    E_factor_1 = jnp$array(matrix(c(1, 2), nrow = 2, ncol = 1), dtype = dtj),
    E_party = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    E_rel = jnp$zeros(reticulate::tuple(3L, 1L), dtype = dtj),
    E_stage = jnp$zeros(reticulate::tuple(2L, 2L, 1L), dtype = dtj),
    E_resp_party = jnp$zeros(reticulate::tuple(2L, 1L), dtype = dtj),
    pseudo_query_attn_l1 = jnp$zeros(reticulate::tuple(1L), dtype = dtj),
    pseudo_query_ff_l1 = jnp$zeros(reticulate::tuple(1L), dtype = dtj),
    pseudo_query_final = jnp$zeros(reticulate::tuple(1L), dtype = dtj),
    RMS_attn_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_ff_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_q_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_k_l1 = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_final = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    W_q_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_k_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_v_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_o_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    W_ff1_l1 = jnp$zeros(reticulate::tuple(1L, 2L), dtype = dtj),
    W_ff2_l1 = jnp$zeros(reticulate::tuple(1L, 1L), dtype = dtj),
    alpha_cross = jnp$array(1, dtype = dtj),
    RMS_cross = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    RMS_merge_cross = jnp$ones(reticulate::tuple(1L), dtype = dtj),
    W_q_cross = jnp$ones(reticulate::tuple(1L, 1L), dtype = dtj),
    W_k_cross = jnp$ones(reticulate::tuple(1L, 1L), dtype = dtj),
    W_v_cross = jnp$ones(reticulate::tuple(1L, 1L), dtype = dtj),
    W_o_cross = jnp$ones(reticulate::tuple(1L, 1L), dtype = dtj),
    W_out = jnp$ones(reticulate::tuple(1L, 1L), dtype = dtj),
    b_out = jnp$zeros(reticulate::tuple(1L), dtype = dtj)
  )
  params <- neural_test_add_fused_factor_params(
    params,
    model_dims = 1L,
    model_info = model_info
  )

  pi_left <- jnp$array(c(1, 0), dtype = dtj)
  pi_right <- jnp$array(c(0, 1), dtype = dtj)
  out <- strategize:::neural_encode_pair_soft_batched(
    pi_left = pi_left,
    pi_right = pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    model_info = model_info,
    resp_party_idx = 1L,
    stage_idx = 1L,
    matchup_idx = NULL,
    resp_cov_vec = NULL,
    params = params,
    return_tokens = TRUE
  )

  choice_tok <- strategize:::neural_build_choice_token(model_info, params)
  choice_mask <- jnp$ones(list(1L, 1L), dtype = dtj)
  resp_info <- strategize:::neural_build_context_tokens(
    model_info,
    resp_party_idx = 1L,
    stage_idx = 1L,
    matchup_idx = NULL,
    resp_cov_vec = NULL,
    params = params,
    return_mask = TRUE
  )
  left_info <- strategize:::neural_build_candidate_tokens_soft(
    pi_left,
    party_idx = 0L,
    role_id = 0L,
    model_info = model_info,
    params = params,
    resp_party_idx = 1L,
    return_mask = TRUE
  )
  right_info <- strategize:::neural_build_candidate_tokens_soft(
    pi_right,
    party_idx = 0L,
    role_id = 0L,
    model_info = model_info,
    params = params,
    resp_party_idx = 1L,
    return_mask = TRUE
  )
  left_seq <- strategize:::neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    ctx_tokens = resp_info$tokens,
    ctx_mask = resp_info$mask,
    cand_tokens = left_info$tokens,
    cand_mask = left_info$mask,
    model_info = model_info,
    preserve_candidate_tail = TRUE
  )
  right_seq <- strategize:::neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    ctx_tokens = resp_info$tokens,
    ctx_mask = resp_info$mask,
    cand_tokens = right_info$tokens,
    cand_mask = right_info$mask,
    model_info = model_info,
    preserve_candidate_tail = TRUE
  )
  transformer_out <- strategize:::neural_run_transformer(
    jnp$concatenate(list(left_seq$tokens, right_seq$tokens), axis = 0L),
    model_info,
    params,
    token_mask = jnp$concatenate(list(left_seq$mask, right_seq$mask), axis = 0L),
    return_details = TRUE
  )

  state_tokens <- strategize:::neural_transformer_state_tokens(transformer_out)
  readout_tokens <- strategize:::neural_transformer_readout_tokens(transformer_out)
  t_total <- as.integer(state_tokens$shape[[2]])
  cand_width <- as.integer(left_seq$cand_mask$shape[[2]])
  cand_idx <- jnp$arange(as.integer(t_total - cand_width), as.integer(t_total))
  state_candidates <- jnp$take(state_tokens, cand_idx, axis = 1L)
  readout_candidates <- jnp$take(readout_tokens, cand_idx, axis = 1L)

  out_left <- as.numeric(reticulate::py_to_r(np$array(out$cand_left_out)))
  out_right <- as.numeric(reticulate::py_to_r(np$array(out$cand_right_out)))
  state_left <- as.numeric(reticulate::py_to_r(np$array(jnp$take(state_candidates, jnp$arange(1L), axis = 0L))))
  state_right <- as.numeric(reticulate::py_to_r(np$array(jnp$take(state_candidates, jnp$arange(1L, 2L), axis = 0L))))
  readout_left <- as.numeric(reticulate::py_to_r(np$array(jnp$take(readout_candidates, jnp$arange(1L), axis = 0L))))
  readout_right <- as.numeric(reticulate::py_to_r(np$array(jnp$take(readout_candidates, jnp$arange(1L, 2L), axis = 0L))))

  expect_equal(out_left, readout_left, tolerance = 1e-6)
  expect_equal(out_right, readout_right, tolerance = 1e-6)
  expect_gt(max(abs(readout_left - state_left)), 1e-3)
  expect_gt(max(abs(readout_right - state_right)), 1e-6)
})

test_that("attn mode packs fused candidates before extraction", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  fx <- context_head_fused_fixture(cross_mode = "attn")

  out <- strategize:::neural_encode_pair_soft_batched(
    pi_left = fx$pi_left,
    pi_right = fx$pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    model_info = fx$model_info,
    resp_party_idx = NULL,
    stage_idx = NULL,
    matchup_idx = NULL,
    resp_cov_vec = NULL,
    params = fx$params,
    return_tokens = TRUE
  )

  expect_equal(as.integer(out$cand_left_out$shape[[2]]), 1L)
  expect_equal(as.integer(out$cand_right_out$shape[[2]]), 1L)
  expect_equal(as.integer(out$cand_left_mask$shape[[2]]), 1L)
  expect_equal(as.integer(out$cand_right_mask$shape[[2]]), 1L)
})

test_that("full attention residual attn extracts packed fused readout candidates", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv
  jnp <- strenv$jnp
  np <- strenv$np
  fx <- context_head_fused_fixture(
    cross_mode = "attn",
    residual_mode = "full_attn"
  )

  out <- strategize:::neural_encode_pair_soft_batched(
    pi_left = fx$pi_left,
    pi_right = fx$pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    model_info = fx$model_info,
    resp_party_idx = NULL,
    stage_idx = NULL,
    matchup_idx = NULL,
    resp_cov_vec = NULL,
    params = fx$params,
    return_tokens = TRUE
  )

  choice_tok <- strategize:::neural_build_choice_token(fx$model_info, fx$params)
  choice_mask <- jnp$ones(list(1L, 1L), dtype = strenv$dtj)
  left_info <- strategize:::neural_build_candidate_tokens_soft(
    fx$pi_left,
    party_idx = 0L,
    role_id = 0L,
    model_info = fx$model_info,
    params = fx$params,
    resp_party_idx = NULL,
    return_mask = TRUE
  )
  right_info <- strategize:::neural_build_candidate_tokens_soft(
    fx$pi_right,
    party_idx = 0L,
    role_id = 0L,
    model_info = fx$model_info,
    params = fx$params,
    resp_party_idx = NULL,
    return_mask = TRUE
  )
  left_seq <- strategize:::neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    cand_tokens = left_info$tokens,
    cand_mask = left_info$mask,
    model_info = fx$model_info,
    preserve_candidate_tail = TRUE
  )
  right_seq <- strategize:::neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    cand_tokens = right_info$tokens,
    cand_mask = right_info$mask,
    model_info = fx$model_info,
    preserve_candidate_tail = TRUE
  )
  transformer_out <- strategize:::neural_run_transformer(
    jnp$concatenate(list(left_seq$tokens, right_seq$tokens), axis = 0L),
    fx$model_info,
    fx$params,
    token_mask = jnp$concatenate(list(left_seq$mask, right_seq$mask), axis = 0L),
    return_details = TRUE
  )

  state_tokens <- strategize:::neural_transformer_state_tokens(transformer_out)
  readout_tokens <- strategize:::neural_transformer_readout_tokens(transformer_out)
  cand_width <- as.integer(left_seq$cand_mask$shape[[2]])
  cand_idx <- jnp$arange(
    as.integer(state_tokens$shape[[2]] - cand_width),
    as.integer(state_tokens$shape[[2]])
  )
  state_candidates <- jnp$take(state_tokens, cand_idx, axis = 1L)
  readout_candidates <- jnp$take(readout_tokens, cand_idx, axis = 1L)

  out_left <- as.numeric(reticulate::py_to_r(np$array(out$cand_left_out)))
  out_right <- as.numeric(reticulate::py_to_r(np$array(out$cand_right_out)))
  state_left <- as.numeric(reticulate::py_to_r(np$array(jnp$take(state_candidates, jnp$arange(1L), axis = 0L))))
  state_right <- as.numeric(reticulate::py_to_r(np$array(jnp$take(state_candidates, jnp$arange(1L, 2L), axis = 0L))))
  readout_left <- as.numeric(reticulate::py_to_r(np$array(jnp$take(readout_candidates, jnp$arange(1L), axis = 0L))))
  readout_right <- as.numeric(reticulate::py_to_r(np$array(jnp$take(readout_candidates, jnp$arange(1L, 2L), axis = 0L))))

  expect_equal(out_left, readout_left, tolerance = 1e-6)
  expect_equal(out_right, readout_right, tolerance = 1e-6)
  expect_gt(max(abs(readout_left - state_left)), 1e-6)
  expect_gt(max(abs(readout_right - state_right)), 1e-6)
})

test_that("pairwise full mode handles packed fused candidate blocks", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv
  fx <- context_head_fused_fixture(cross_mode = "full")

  logits_full <- strategize:::neural_predict_pair_soft(
    pi_left = fx$pi_left,
    pi_right = fx$pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = NULL,
    model_info = fx$model_info,
    resp_cov_vec = NULL,
    params = fx$params,
    return_logits = TRUE
  )

  model_info_none <- fx$model_info
  model_info_none$cross_candidate_encoder_mode <- "none"
  logits_none <- strategize:::neural_predict_pair_soft(
    pi_left = fx$pi_left,
    pi_right = fx$pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = NULL,
    model_info = model_info_none,
    resp_cov_vec = NULL,
    params = fx$params,
    return_logits = TRUE
  )

  expect_equal(
    as.numeric(strenv$np$array(logits_full)),
    sum(as.numeric(strenv$np$array(fx$params$E_choice))),
    tolerance = 1e-6
  )
  expect_equal(
    as.numeric(strenv$np$array(logits_none)),
    -15,
    tolerance = 1e-6
  )

  model_info_none_normal <- model_info_none
  model_info_none_normal$likelihood <- "normal"
  logits_none_normal <- strategize:::neural_predict_pair_soft(
    pi_left = fx$pi_left,
    pi_right = fx$pi_right,
    party_left_idx = 0L,
    party_right_idx = 0L,
    resp_party_idx = NULL,
    model_info = model_info_none_normal,
    resp_cov_vec = NULL,
    params = fx$params,
    return_logits = TRUE
  )
  expect_equal(
    as.numeric(strenv$np$array(logits_none_normal)),
    0,
    tolerance = 1e-6
  )
})

test_that("low-rank respondent-candidate interaction produces utility logits", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv
  params <- list(
    alpha_rc = strenv$jnp$array(2, dtype = strenv$dtj),
    W_rc_r = strenv$jnp$eye(2L, dtype = strenv$dtj),
    W_rc_c = strenv$jnp$eye(2L, dtype = strenv$dtj),
    W_rc_out = strenv$jnp$array(matrix(c(2, -1), nrow = 2L), dtype = strenv$dtj)
  )
  respondent <- strenv$jnp$array(rbind(c(1, 0), c(0, 1)), dtype = strenv$dtj)
  candidate <- strenv$jnp$array(rbind(c(1, 0), c(1, 0)), dtype = strenv$dtj)
  utility <- strenv$jnp$zeros(list(2L, 1L), dtype = strenv$dtj)

  logits <- strategize:::neural_low_rank_interaction_logits(
    respondent_final = respondent,
    candidate_final = candidate,
    params = params,
    out_dim = 1L,
    dtype = strenv$dtj
  )
  adjusted <- strategize:::neural_apply_low_rank_interaction(
    utility,
    respondent,
    candidate,
    params
  )

  expect_equal(as.numeric(strenv$np$array(logits)), c(4, 0), tolerance = 1e-6)
  expect_equal(as.numeric(strenv$np$array(adjusted)), c(4, 0), tolerance = 1e-6)
})

test_that("low-rank pairwise logit softclip only bounds Bernoulli classification rows", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  strenv <- strategize:::strenv
  info_active <- list(
    likelihood = "mixed",
    low_rank_interaction_rank = 2L,
    low_rank_logit_transform = "softclip",
    low_rank_logit_bound = 1.5,
    low_rank_logit_softness = 0.25
  )
  logits <- strenv$jnp$array(matrix(c(100, 100), ncol = 1L), dtype = strenv$dtj)
  codes <- strenv$jnp$array(as.integer(c(0L, 2L)))

  bounded <- strategize:::neural_apply_pairwise_classification_logit_transform(
    logits,
    model_info = info_active,
    likelihood_code_obs = codes,
    pairwise_obs = TRUE
  )
  bounded_r <- as.numeric(strenv$np$array(bounded))
  expect_lt(bounded_r[[1L]], 1.6)
  expect_equal(bounded_r[[2L]], 100, tolerance = 1e-6)

  info_rank_zero <- info_active
  info_rank_zero$low_rank_interaction_rank <- 0L
  unchanged <- strategize:::neural_apply_pairwise_classification_logit_transform(
    logits,
    model_info = info_rank_zero,
    likelihood_code_obs = codes,
    pairwise_obs = TRUE
  )
  expect_equal(as.numeric(strenv$np$array(unchanged)), c(100, 100), tolerance = 1e-6)
})
