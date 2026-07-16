test_that("stacked standard transformer scan matches legacy unrolled path", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  jnp <- strategize:::strenv$jnp
  dtj <- strategize:::strenv$dtj

  model_info <- strategize:::neural_make_transformer_model_info(
    model_depth = 2L,
    model_dims = 2L,
    n_heads = 1L,
    head_dim = 2L,
    residual_mode = "standard",
    attention_backend = "xla",
    attention_dtype = "float32"
  )

  legacy_params <- list(
    W_q_l1 = jnp$eye(2L, dtype = dtj),
    W_k_l1 = jnp$eye(2L, dtype = dtj),
    W_v_l1 = jnp$eye(2L, dtype = dtj),
    W_o_l1 = jnp$eye(2L, dtype = dtj),
    W_ff1_l1 = jnp$array(cbind(diag(2L), diag(2L)), dtype = dtj),
    W_ff2_l1 = jnp$eye(2L, dtype = dtj),
    RMS_attn_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_ff_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_q_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_k_l1 = jnp$ones(list(2L), dtype = dtj),
    alpha_attn_l1 = jnp$array(0.25, dtype = dtj),
    alpha_ff_l1 = jnp$array(0.15, dtype = dtj),
    W_q_l2 = jnp$array(matrix(c(1, 0.1, -0.1, 1), 2L), dtype = dtj),
    W_k_l2 = jnp$array(matrix(c(1, -0.2, 0.2, 1), 2L), dtype = dtj),
    W_v_l2 = jnp$eye(2L, dtype = dtj),
    W_o_l2 = jnp$array(matrix(c(0.8, 0.1, 0.0, 0.9), 2L), dtype = dtj),
    W_ff1_l2 = jnp$array(cbind(diag(2L), diag(2L)), dtype = dtj),
    W_ff2_l2 = jnp$eye(2L, dtype = dtj),
    RMS_attn_l2 = jnp$ones(list(2L), dtype = dtj),
    RMS_ff_l2 = jnp$ones(list(2L), dtype = dtj),
    RMS_q_l2 = jnp$ones(list(2L), dtype = dtj),
    RMS_k_l2 = jnp$ones(list(2L), dtype = dtj),
    alpha_attn_l2 = jnp$array(0.2, dtype = dtj),
    alpha_ff_l2 = jnp$array(0.1, dtype = dtj),
    RMS_final = jnp$ones(list(2L), dtype = dtj)
  )
  stacked_params <- strategize:::neural_stack_standard_transformer_layers(
    legacy_params,
    model_depth = 2L,
    drop_legacy = TRUE
  )

  tokens <- jnp$array(
    array(c(0.2, -0.1, 0.4, 0.3, -0.2, 0.5), dim = c(1L, 3L, 2L)),
    dtype = dtj
  )
  token_mask <- jnp$array(matrix(c(1, 1, 0), nrow = 1L), dtype = dtj)

  legacy_out <- strategize:::neural_run_transformer(
    tokens,
    model_info,
    params = legacy_params,
    token_mask = token_mask
  )
  stacked_out <- strategize:::neural_run_transformer(
    tokens,
    model_info,
    params = stacked_params,
    token_mask = token_mask
  )

  expect_equal(
    as.numeric(strategize:::cs2step_neural_to_r_array(stacked_out)),
    as.numeric(strategize:::cs2step_neural_to_r_array(legacy_out)),
    tolerance = 1e-5
  )
})

test_that("stacked transformer params flatten and unflatten as one schema block", {
  skip_on_cran()
  skip_if_no_jax()

  strategize:::initialize_jax()
  jnp <- strategize:::strenv$jnp
  dtj <- strategize:::strenv$dtj

  legacy_params <- list(
    E_factor_1 = jnp$zeros(list(2L, 2L), dtype = dtj),
    E_feature_id = jnp$zeros(list(1L, 2L), dtype = dtj),
    E_party = jnp$zeros(list(1L, 2L), dtype = dtj),
    E_resp_party = jnp$zeros(list(1L, 2L), dtype = dtj),
    E_choice = jnp$zeros(list(2L), dtype = dtj),
    E_token_family = jnp$zeros(list(1L, 2L), dtype = dtj),
    E_experiment = jnp$zeros(list(1L, 2L), dtype = dtj),
    E_sep = jnp$zeros(list(2L), dtype = dtj),
    E_segment = jnp$zeros(list(2L, 2L), dtype = dtj),
    W_q_l1 = jnp$eye(2L, dtype = dtj),
    W_k_l1 = jnp$eye(2L, dtype = dtj),
    W_v_l1 = jnp$eye(2L, dtype = dtj),
    W_o_l1 = jnp$eye(2L, dtype = dtj),
    W_ff1_l1 = jnp$array(cbind(diag(2L), diag(2L)), dtype = dtj),
    W_ff2_l1 = jnp$eye(2L, dtype = dtj),
    RMS_attn_l1 = jnp$ones(list(2L), dtype = dtj),
    RMS_ff_l1 = jnp$ones(list(2L), dtype = dtj),
    alpha_attn_l1 = jnp$array(0.1, dtype = dtj),
    alpha_ff_l1 = jnp$array(0.1, dtype = dtj),
    W_q_l2 = jnp$multiply(2, jnp$eye(2L, dtype = dtj)),
    W_k_l2 = jnp$multiply(2, jnp$eye(2L, dtype = dtj)),
    W_v_l2 = jnp$multiply(2, jnp$eye(2L, dtype = dtj)),
    W_o_l2 = jnp$multiply(2, jnp$eye(2L, dtype = dtj)),
    W_ff1_l2 = jnp$array(cbind(2 * diag(2L), 2 * diag(2L)), dtype = dtj),
    W_ff2_l2 = jnp$multiply(2, jnp$eye(2L, dtype = dtj)),
    RMS_attn_l2 = jnp$ones(list(2L), dtype = dtj),
    RMS_ff_l2 = jnp$ones(list(2L), dtype = dtj),
    alpha_attn_l2 = jnp$array(0.2, dtype = dtj),
    alpha_ff_l2 = jnp$array(0.2, dtype = dtj),
    RMS_final = jnp$ones(list(2L), dtype = dtj),
    W_out = jnp$zeros(list(2L, 1L), dtype = dtj),
    b_out = jnp$zeros(list(1L), dtype = dtj)
  )
  stacked_params <- strategize:::neural_stack_standard_transformer_layers(
    legacy_params,
    model_depth = 2L,
    drop_legacy = TRUE
  )

  schema <- strategize:::neural_build_param_schema(
    params = stacked_params,
    n_factors = 1L,
    model_depth = 2L
  )
  expect_true("W_q_layers" %in% schema$param_names)
  expect_false("W_q_l1" %in% schema$param_names)
  expect_false("alpha_ff_l2" %in% schema$param_names)

  theta <- strategize:::neural_flatten_params(stacked_params, schema, dtype = dtj)
  roundtrip <- strategize:::neural_params_from_theta(
    theta,
    list(
      param_names = schema$param_names,
      param_shapes = schema$param_shapes,
      param_sizes = schema$param_sizes,
      param_offsets = schema$param_offsets,
      n_params = schema$n_params
    )
  )

  expect_equal(
    as.integer(reticulate::py_to_r(roundtrip$W_q_layers$shape)),
    c(2L, 2L, 2L)
  )
  expect_true(strategize:::neural_has_stacked_standard_transformer(roundtrip))
})

test_that("attention backend config is normalized and affects cache keys", {
  info_xla <- strategize:::neural_make_transformer_model_info(
    model_depth = 1L,
    model_dims = 8L,
    n_heads = 2L,
    head_dim = 4L,
    residual_mode = "standard",
    attention_backend = "xla",
    attention_dtype = "float32",
    attention_padding_multiple = 8L
  )
  info_flash <- strategize:::neural_make_transformer_model_info(
    model_depth = 1L,
    model_dims = 8L,
    n_heads = 2L,
    head_dim = 4L,
    residual_mode = "standard",
    attention_backend = "flash",
    attention_dtype = "bf16",
    attention_padding_multiple = 16L
  )

  expect_identical(info_xla$attention_backend, "xla")
  expect_identical(info_xla$attention_dtype, "float32")
  expect_identical(info_flash$attention_backend, "cudnn")
  expect_identical(info_flash$attention_dtype, "bfloat16")
  expect_identical(info_flash$attention_padding_multiple, 16L)
  expect_false(identical(
    strategize:::neural_model_jit_cache_key(info_xla),
    strategize:::neural_model_jit_cache_key(info_flash)
  ))
})
