# =============================================================================
# Tests for REINFORCE policy probability helpers
# =============================================================================

ensure_test_jax <- function() {
  if (!"jnp" %in% ls(envir = strategize:::strenv)) {
    strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  }
}

test_that("policy support weights match grouped multinomial probabilities", {
  skip_on_cran()
  skip_if_no_jax()
  ensure_test_jax()

  full_locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L), dtype = strategize:::strenv$jnp$int32)
  full_pi <- strategize:::strenv$jnp$array(c(0.2, 0.8, 0.7, 0.3), dtype = strategize:::strenv$dtj)
  full_profiles <- strategize:::strenv$jnp$array(
    rbind(
      c(1, 0, 1, 0),
      c(0, 1, 0, 1),
      c(1, 0, 0, 1)
    ),
    dtype = strategize:::strenv$dtj
  )

  full_weights <- strategize:::compute_policy_support_weights(
    pi_vec = full_pi,
    profiles = full_profiles,
    ParameterizationType = "Full",
    d_locator_use = full_locator
  )

  expect_equal(
    as.numeric(strategize:::strenv$np$array(full_weights)),
    c(0.2 * 0.7, 0.8 * 0.3, 0.2 * 0.3),
    tolerance = 1e-6
  )

  implicit_locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L), dtype = strategize:::strenv$jnp$int32)
  implicit_pi <- strategize:::strenv$jnp$array(c(0.2, 0.3, 0.4), dtype = strategize:::strenv$dtj)
  implicit_profiles <- strategize:::strenv$jnp$array(
    rbind(
      c(1, 0, 1),
      c(0, 1, 0),
      c(0, 0, 1),
      c(0, 0, 0)
    ),
    dtype = strategize:::strenv$dtj
  )

  implicit_weights <- strategize:::compute_policy_support_weights(
    pi_vec = implicit_pi,
    profiles = implicit_profiles,
    ParameterizationType = "Implicit",
    d_locator_use = implicit_locator
  )

  expect_equal(
    as.numeric(strategize:::strenv$np$array(implicit_weights)),
    c(
      0.2 * 0.4,
      0.3 * (1 - 0.4),
      (1 - 0.2 - 0.3) * 0.4,
      (1 - 0.2 - 0.3) * (1 - 0.4)
    ),
    tolerance = 1e-6
  )
})

test_that("getPrettyPi flattens balanced implicit locators before segment sums", {
  skip_on_cran()
  skip_if_no_jax()
  ensure_test_jax()

  assign("nUniqueFactors", as.integer(5L), envir = strategize:::strenv)
  assign(
    "OneTf",
    strategize:::strenv$jnp$array(matrix(1), dtype = strategize:::strenv$dtj),
    envir = strategize:::strenv
  )

  locator <- matrix(rep(seq_len(5L), each = 2L), nrow = 2L)
  locator <- strategize:::strenv$jnp$array(locator, dtype = strategize:::strenv$jnp$int32)
  pi_reduced <- strategize:::strenv$jnp$array(
    matrix(rep(c(0.2, 0.3), times = 5L), ncol = 1L),
    dtype = strategize:::strenv$dtj
  )

  main_comp_mat <- matrix(0, nrow = 15L, ncol = 10L)
  main_comp_mat[c(1L, 2L, 4L, 5L, 7L, 8L, 10L, 11L, 13L, 14L) +
                  15L * (seq_len(10L) - 1L)] <- 1
  shadow_comp_mat <- matrix(0, nrow = 15L, ncol = 5L)
  shadow_comp_mat[c(3L, 6L, 9L, 12L, 15L) + 15L * (seq_len(5L) - 1L)] <- 1

  pretty <- strategize:::getPrettyPi(
    pi_star_value = pi_reduced,
    ParameterizationType = "Implicit",
    d_locator = locator,
    main_comp_mat = strategize:::strenv$jnp$array(main_comp_mat, dtype = strategize:::strenv$dtj),
    shadow_comp_mat = strategize:::strenv$jnp$array(shadow_comp_mat, dtype = strategize:::strenv$dtj)
  )

  expect_equal(
    as.numeric(strategize:::strenv$np$array(pretty)),
    rep(c(0.2, 0.3, 0.5), times = 5L),
    tolerance = 1e-6
  )
})

test_that("policy sample log-prob helper handles rank-4 pooled batches", {
  skip_on_cran()
  skip_if_no_jax()
  ensure_test_jax()

  locator <- strategize:::strenv$jnp$array(c(1L, 1L, 2L, 2L), dtype = strategize:::strenv$jnp$int32)
  pi_vec <- strategize:::strenv$jnp$array(c(0.2, 0.8, 0.7, 0.3), dtype = strategize:::strenv$dtj)
  pooled_profiles <- array(0, dim = c(2L, 2L, 1L, 4L))
  pooled_profiles[1L, 1L, 1L, ] <- c(1, 0, 1, 0)
  pooled_profiles[1L, 2L, 1L, ] <- c(0, 1, 0, 1)
  pooled_profiles[2L, 1L, 1L, ] <- c(1, 0, 0, 1)
  pooled_profiles[2L, 2L, 1L, ] <- c(0, 1, 1, 0)
  pooled_profiles <- strategize:::strenv$jnp$array(pooled_profiles, dtype = strategize:::strenv$dtj)

  log_probs <- strategize:::compute_policy_sample_log_probs(
    pi_vec = pi_vec,
    profiles = pooled_profiles,
    ParameterizationType = "Full",
    d_locator_use = locator
  )

  expected <- c(
    log(0.2 * 0.7) + log(0.8 * 0.3),
    log(0.2 * 0.3) + log(0.8 * 0.7)
  )
  expect_equal(
    as.numeric(strategize:::strenv$np$array(log_probs)),
    expected,
    tolerance = 1e-6
  )
})
