# =============================================================================
# Muon Optimizer Targeting Tests
# =============================================================================

muon_test_labels <- local({
  initialized <- FALSE

  function(params) {
    skip_on_cran()
    skip_if_no_jax()

    if (!initialized) {
      strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
      strategize:::neural_get_muon_dimension_numbers_callable(force_refresh = TRUE)
      reticulate::py_run_string(
        paste(
          "def _strategize_muon_test_labels(params):",
          "    dimnums_tree = _strategize_muon_dimnums(params)",
          "    out = {}",
          "    for name in params.keys():",
          "        out[str(name)] = 'muon' if dimnums_tree[str(name)] is not None else 'adam'",
          "    return out",
          sep = "\n"
        )
      )
      initialized <<- TRUE
    }

    labels <- reticulate::py_eval("_strategize_muon_test_labels")(params)
    reticulate::py_to_r(labels)
  }
})

muon_test_array <- function(ndim) {
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  shape <- if (identical(as.integer(ndim), 2L)) list(2L, 2L) else list(2L)
  strategize:::strenv$jnp$ones(shape)
}

test_that("muon dimension-number tree hits intended matrix weights and excludes others", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  if (!reticulate::py_has_attr(strategize:::strenv$optax, "contrib") ||
      !reticulate::py_has_attr(strategize:::strenv$optax$contrib, "muon")) {
    skip("optax.contrib.muon not available")
  }

  cases <- list(
    list(name = "W_q_l1", ndim = 2L, want = "muon"),
    list(name = "W_ff2_l3", ndim = 2L, want = "muon"),
    list(name = "W_q_cross", ndim = 2L, want = "muon"),
    list(name = "M_cross_raw", ndim = 2L, want = "muon"),
    # On-critical-path hidden MLPs now orthogonalized alongside the transformer FF
    # (previously fell to Muon's Adam sub-branch despite identical structure).
    list(name = "W_factor_fuse_1", ndim = 2L, want = "muon"),
    list(name = "W_factor_fuse_2", ndim = 2L, want = "muon"),
    list(name = "W_covariate_fuse_1", ndim = 2L, want = "muon"),
    list(name = "W_covariate_value_conditioner_1", ndim = 2L, want = "muon"),
    list(name = "W_rc_r", ndim = 2L, want = "muon"),
    # Output/unembedding heads are excluded from Muon (standard recipe): they need
    # Adam's per-coordinate scaling to calibrate output magnitude.
    list(name = "W_out", ndim = 2L, want = "adam"),
    list(name = "W_rc_out", ndim = 2L, want = "adam"),
    list(name = "b_out", ndim = 1L, want = "adam"),
    list(name = "RMS_attn_l1", ndim = 1L, want = "adam")
  )

  params <- setNames(
    lapply(cases, function(case) muon_test_array(case$ndim)),
    vapply(cases, `[[`, character(1), "name")
  )
  labels <- muon_test_labels(params)

  for (case in cases) {
    expect_identical(
      unname(labels[[case$name]]),
      case$want,
      info = sprintf("Expected %s to map to %s", case$name, case$want)
    )
  }

  expect_false(strategize:::neural_muon_targets_matrix_weight("W_q_l1", ndim = 1L))
})

test_that("muon dimension-number tree handles guide-location aliases but not guide scales", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  if (!reticulate::py_has_attr(strategize:::strenv$optax, "contrib") ||
      !reticulate::py_has_attr(strategize:::strenv$optax$contrib, "muon")) {
    skip("optax.contrib.muon not available")
  }

  cases <- list(
    list(name = "W_q_l1_auto_loc", want = "muon"),
    list(name = "W_ff1_l2_base_auto_loc", want = "muon"),
    list(name = "W_ff2_l2_decentered_auto_loc", want = "muon"),
    list(name = "W_factor_fuse_1_auto_loc", want = "muon"),
    list(name = "M_cross_raw_auto_loc", want = "muon"),
    list(name = "W_out_auto_loc", want = "adam"),
    list(name = "W_q_l1_auto_scale", want = "adam"),
    list(name = "W_ff1_l2_base_auto_scale", want = "adam"),
    list(name = "W_out_auto_scale", want = "adam")
  )

  params <- setNames(
    replicate(length(cases), muon_test_array(2L), simplify = FALSE),
    vapply(cases, `[[`, character(1), "name")
  )
  labels <- muon_test_labels(params)

  for (case in cases) {
    expect_identical(
      unname(labels[[case$name]]),
      case$want,
      info = sprintf("Expected %s to map to %s", case$name, case$want)
    )
  }
})

test_that("muon falls back for auto_diagonal guide because matrix structure is unavailable", {
  skip_on_cran()
  skip_if_no_jax()
  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)

  expected <- strategize:::neural_default_svi_fallback_optimizer()
  expect_warning(
    resolved <- strategize:::neural_resolve_svi_optimizer_tag(
      optimizer_tag = "muon",
      guide_name = "auto_diagonal",
      user_supplied_optimizer = TRUE
    ),
    "auto_diagonal"
  )
  expect_identical(resolved, expected)
})
