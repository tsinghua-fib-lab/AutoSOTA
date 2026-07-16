neural_test_projection_matrix <- function(n_features, model_dims) {
  out <- matrix(0, nrow = as.integer(n_features), ncol = as.integer(model_dims))
  if (nrow(out) > 0L && ncol(out) > 0L) {
    for (i in seq_len(min(nrow(out), ncol(out)))) {
      out[i, i] <- 1
    }
  }
  out
}

neural_test_matrix_ncol <- function(x) {
  x_r <- tryCatch(
    strategize:::cs2step_neural_to_r_array(x),
    error = function(e) x
  )
  ncol(as.matrix(x_r))
}

neural_test_add_fused_factor_schema <- function(model_info,
                                                factor_levels = NULL,
                                                factor_names = NULL,
                                                max_factor_tokens = NULL) {
  n_factors <- as.integer(model_info$n_factors %||% length(factor_levels %||% integer(0)))
  if (length(n_factors) != 1L || is.na(n_factors) || n_factors < 1L) {
    n_factors <- 1L
  }
  if (is.null(factor_names)) {
    factor_names <- sprintf("factor_%d", seq_len(n_factors))
  }
  factor_names <- as.character(factor_names)
  if (is.null(factor_levels)) {
    factor_levels <- rep.int(2L, n_factors)
  }
  factor_levels <- as.integer(factor_levels)
  if (length(factor_levels) < n_factors) {
    factor_levels <- rep_len(factor_levels, n_factors)
  }
  factor_levels <- factor_levels[seq_len(n_factors)]
  struct <- strategize:::neural_make_default_fused_structural_info(
    factor_names = factor_names,
    factor_levels = factor_levels
  )
  runtime_info <- strategize:::neural_make_runtime_token_model_info(
    model_dims = model_info$model_dims,
    factor_struct_matrix = struct$factor_struct_matrix,
    level_struct_matrices = struct$level_struct_matrices,
    factor_struct_feature_names = struct$factor_struct_feature_names,
    level_struct_feature_names = struct$level_struct_feature_names,
    default_factor_order = seq.int(0L, n_factors - 1L),
    factor_tokenization = "fused",
    max_factor_tokens = max_factor_tokens %||% model_info$max_factor_tokens %||% n_factors,
    token_family_levels = model_info$token_family_levels %||% strategize:::neural_token_family_levels()
  )
  model_info[names(runtime_info)] <- runtime_info
  model_info
}

neural_test_add_fused_factor_params <- function(params,
                                                model_dims,
                                                model_info = NULL,
                                                factor_struct_dim = NULL,
                                                level_struct_dim = NULL,
                                                token_family_levels = strategize:::neural_token_family_levels()) {
  strenv <- strategize:::strenv
  dims <- as.integer(model_dims)
  hidden_dims <- 2L * dims
  gate_value_dims <- 2L * hidden_dims
  if (is.null(factor_struct_dim)) {
    factor_struct_dim <- if (!is.null(model_info$factor_struct_matrix)) {
      neural_test_matrix_ncol(model_info$factor_struct_matrix)
    } else {
      length(strategize:::neural_fused_default_factor_struct_feature_names()) + 1L
    }
  }
  if (is.null(level_struct_dim)) {
    level_struct_dim <- if (!is.null(model_info$level_struct_matrices) &&
                            length(model_info$level_struct_matrices) > 0L) {
      neural_test_matrix_ncol(model_info$level_struct_matrices[[1L]])
    } else {
      length(strategize:::neural_fused_default_level_struct_feature_names())
    }
  }
  params$E_factor_fused_base <- params$E_factor_fused_base %||%
    strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
  params$W_factor_fuse_1 <- params$W_factor_fuse_1 %||% {
    w1 <- matrix(0, nrow = 4L * dims, ncol = gate_value_dims)
    w1[, (hidden_dims + 1L):(hidden_dims + dims)] <-
      rbind(diag(dims), diag(dims), diag(dims), diag(dims))
    strenv$jnp$array(w1, dtype = strenv$dtj)
  }
  params$b_factor_fuse_1 <- params$b_factor_fuse_1 %||%
    strenv$jnp$array(
      c(
        rep(neural_test_swiglu_unit_gate(), hidden_dims),
        rep(0, hidden_dims)
      ),
      dtype = strenv$dtj
    )
  params$W_factor_fuse_2 <- params$W_factor_fuse_2 %||%
    strenv$jnp$array(rbind(diag(dims), matrix(0, nrow = dims, ncol = dims)), dtype = strenv$dtj)
  params$b_factor_fuse_2 <- params$b_factor_fuse_2 %||%
    strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
  params$W_factor_name_text <- params$W_factor_name_text %||% NULL
  params$W_level_name_text <- params$W_level_name_text %||% NULL
  params$W_factor_struct <- params$W_factor_struct %||%
    strenv$jnp$array(
      neural_test_projection_matrix(factor_struct_dim, dims),
      dtype = strenv$dtj
    )
  params$W_level_struct <- params$W_level_struct %||%
    strenv$jnp$array(
      neural_test_projection_matrix(level_struct_dim, dims),
      dtype = strenv$dtj
    )
  params$E_token_family <- params$E_token_family %||%
    strenv$jnp$zeros(list(length(token_family_levels), dims), dtype = strenv$dtj)
  params
}

neural_test_add_fused_covariate_value_params <- function(params,
                                                         model_dims,
                                                         metadata_dim = length(strategize:::neural_covariate_value_metadata_names()),
                                                         token_family_levels = strategize:::neural_token_family_levels()) {
  strenv <- strategize:::strenv
  dims <- as.integer(model_dims)
  hidden_dims <- 2L * dims
  gate_value_dims <- 2L * hidden_dims
  metadata_dim <- as.integer(metadata_dim)
  params$E_covariate_fused_base <- params$E_covariate_fused_base %||%
    strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
  params$W_covariate_fuse_1 <- params$W_covariate_fuse_1 %||% {
    w_value <- rbind(
      matrix(0, nrow = dims, ncol = dims),
      diag(dims),
      matrix(0, nrow = metadata_dim, ncol = dims)
    )
    w1 <- matrix(0, nrow = 2L * dims + metadata_dim, ncol = gate_value_dims)
    w1[, (hidden_dims + 1L):(hidden_dims + dims)] <- w_value
    strenv$jnp$array(w1, dtype = strenv$dtj)
  }
  params$b_covariate_fuse_1 <- params$b_covariate_fuse_1 %||%
    strenv$jnp$array(
      c(
        rep(neural_test_swiglu_unit_gate(), hidden_dims),
        rep(0, hidden_dims)
      ),
      dtype = strenv$dtj
    )
  params$W_covariate_fuse_2 <- params$W_covariate_fuse_2 %||%
    strenv$jnp$array(rbind(diag(dims), matrix(0, nrow = dims, ncol = dims)), dtype = strenv$dtj)
  params$b_covariate_fuse_2 <- params$b_covariate_fuse_2 %||%
    strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
  params$E_token_family <- params$E_token_family %||%
    strenv$jnp$zeros(list(length(token_family_levels), dims), dtype = strenv$dtj)
  params
}

neural_test_swiglu_unit_gate <- function() {
  1.2784645427610738
}

neural_test_swiglu_value <- function(x) {
  strenv <- strategize:::strenv
  gate <- strenv$jnp$array(
    rep(neural_test_swiglu_unit_gate(), length(as.numeric(x))),
    dtype = strenv$dtj
  )
  value <- strenv$jnp$array(as.numeric(x), dtype = strenv$dtj)
  as.numeric(strenv$np$array(
    strenv$jax$nn$swish(gate) * value
  ))
}
