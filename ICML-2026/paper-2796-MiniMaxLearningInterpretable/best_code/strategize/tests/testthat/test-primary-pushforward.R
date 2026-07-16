# =============================================================================
# Primary push-forward tests
# =============================================================================

jax_to_numeric <- function(x) {
  as.numeric(reticulate::py_to_r(strenv$np$array(x)))
}

test_that("primary_pushforward toggles primary estimator in adversarial mode", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 200, seed = 123)
  data <- add_adversarial_structure(data, seed = 123)
  params <- default_strategize_params(fast = TRUE)

  res_mc <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    primary_pushforward = "mc",
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  res_linear <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    primary_pushforward = "linearized",
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  res_multi <- strategize(
    Y = data$Y,
    W = data$W,
    lambda = params$lambda,
    pair_id = data$pair_id,
    respondent_id = data$respondent_id,
    respondent_task_id = data$respondent_task_id,
    profile_order = data$profile_order,
    competing_group_variable_respondent = data$competing_group_variable_respondent,
    competing_group_variable_candidate = data$competing_group_variable_candidate,
    competing_group_competition_variable_candidate = data$competing_group_competition_variable_candidate,
    adversarial = TRUE,
    diff = TRUE,
    nSGD = params$nSGD,
    outcome_model_type = params$outcome_model_type,
    force_gaussian = params$force_gaussian,
    nMonte_adversarial = params$nMonte_adversarial,
    nMonte_Qglm = params$nMonte_Qglm,
    primary_pushforward = "multi",
    primary_n_entrants = 2L,
    primary_n_field = 2L,
    compute_se = params$compute_se,
    conda_env = params$conda_env,
    conda_env_required = params$conda_env_required
  )

  expect_equal(res_mc$convergence_history$primary_pushforward, "mc")
  expect_equal(res_linear$convergence_history$primary_pushforward, "linearized")
  expect_equal(res_multi$convergence_history$primary_pushforward, "multi")
  expect_equal(as.integer(res_multi$convergence_history$primary_n_entrants), 2L)
  expect_equal(as.integer(res_multi$convergence_history$primary_n_field), 2L)
  expect_equal(as.integer(res_multi$primary_n_entrants), 2L)
  expect_equal(as.integer(res_multi$primary_n_field), 2L)
  expect_true(all(is.finite(res_multi$Q_point)))
  pi_vals <- unlist(res_multi$pi_star_point, use.names = FALSE)
  expect_true(all(is.finite(pi_vals)))
  expect_true(all(pi_vals >= -1e-8 & pi_vals <= 1 + 1e-8))
})

# =============================================================================
# Multinomial logit primary nomination probability tests
# =============================================================================

test_that("multinomial logit primary: uniform pairwise probs give uniform nomination probs", {
  skip_on_cran()
  skip_if_no_jax()

  # Initialize JAX environment if needed
  if (!exists("strenv") || is.null(strenv$jnp)) {
    skip("JAX environment not initialized")
  }

  # Create uniform pairwise win probabilities (all 0.5)
  n_candidates <- 4L
  kappa_matrix <- matrix(0.5, n_candidates, n_candidates)
  diag(kappa_matrix) <- 0.5  # Diagonal doesn't matter due to mask

  # Convert to JAX array
  kA <- strenv$jnp$array(kappa_matrix, dtype = strenv$dtj)
  maskA <- strenv$jnp$ones(list(n_candidates, n_candidates), dtype = strenv$dtj) -
           strenv$jnp$eye(n_candidates, dtype = strenv$dtj)

  # Compute log-odds utility (Bradley-Terry)
  eps <- strenv$jnp$array(1e-8, strenv$dtj)
  one_bt <- strenv$jnp$array(1.0, strenv$dtj)
  logoddsA <- strenv$jnp$log(kA + eps) - strenv$jnp$log(one_bt - kA + eps)

  # Mean utility per candidate
  utilityA <- (logoddsA * maskA)$sum(axis = 1L) / strenv$jnp$array(n_candidates, strenv$dtj)

  # Softmax should give uniform probs for uniform utilities
  pA <- strenv$jax$nn$softmax(utilityA)
  pA_r <- jax_to_numeric(pA)

  # Should be uniform (1/n_candidates for each)
  expected <- rep(1/n_candidates, n_candidates)
  expect_equal(pA_r, expected, tolerance = 1e-6)
})

test_that("multinomial logit primary: binary case matches pairwise probability", {
  skip_on_cran()
  skip_if_no_jax()

  if (!exists("strenv") || is.null(strenv$jnp)) {
    skip("JAX environment not initialized")
  }

  n_candidates <- 2L
  p <- 0.73
  kappa_matrix <- matrix(c(0.5, p,
                           1 - p, 0.5), nrow = 2, byrow = TRUE)

  kA <- strenv$jnp$array(kappa_matrix, dtype = strenv$dtj)
  maskA <- strenv$jnp$ones(list(n_candidates, n_candidates), dtype = strenv$dtj) -
           strenv$jnp$eye(n_candidates, dtype = strenv$dtj)

  eps <- strenv$jnp$array(1e-8, strenv$dtj)
  one_bt <- strenv$jnp$array(1.0, strenv$dtj)
  logoddsA <- strenv$jnp$log(kA + eps) - strenv$jnp$log(one_bt - kA + eps)
  utilityA <- (logoddsA * maskA)$sum(axis = 1L) / strenv$jnp$array(n_candidates, strenv$dtj)

  pA <- strenv$jax$nn$softmax(utilityA)
  pA_r <- jax_to_numeric(pA)

  expect_equal(pA_r[1], p, tolerance = 1e-6)
  expect_equal(pA_r[2], 1 - p, tolerance = 1e-6)
})

test_that("multinomial logit primary: dominant candidate gets near-1.0 prob with extreme pairwise win rates", {
  skip_on_cran()
  skip_if_no_jax()

  if (!exists("strenv") || is.null(strenv$jnp)) {
    skip("JAX environment not initialized")
  }

  # Create pairwise probs where candidate 1 almost always beats others (dominant)
  n_candidates <- 4L
  kappa_matrix <- matrix(0.5, n_candidates, n_candidates)
  kappa_matrix[1, ] <- 0.99  # Candidate 1 beats all others with 99% prob
  kappa_matrix[, 1] <- 0.01  # Others beat candidate 1 with only 1% prob
  diag(kappa_matrix) <- 0.5

  kA <- strenv$jnp$array(kappa_matrix, dtype = strenv$dtj)
  maskA <- strenv$jnp$ones(list(n_candidates, n_candidates), dtype = strenv$dtj) -
           strenv$jnp$eye(n_candidates, dtype = strenv$dtj)

  eps <- strenv$jnp$array(1e-8, strenv$dtj)
  one_bt <- strenv$jnp$array(1.0, strenv$dtj)
  logoddsA <- strenv$jnp$log(kA + eps) - strenv$jnp$log(one_bt - kA + eps)
  utilityA <- (logoddsA * maskA)$sum(axis = 1L) / strenv$jnp$array(n_candidates, strenv$dtj)

  pA_dom <- strenv$jax$nn$softmax(utilityA)
  pA_dom_r <- jax_to_numeric(pA_dom)

  # Candidate 1 should have probability very close to 1
  expect_gt(pA_dom_r[1], 0.95)
  # Others should have very low probability
  expect_lt(sum(pA_dom_r[-1]), 0.05)
})

test_that("multinomial logit primary: order permutation preserves relative probabilities", {
  skip_on_cran()
  skip_if_no_jax()

  if (!exists("strenv") || is.null(strenv$jnp)) {
    skip("JAX environment not initialized")
  }

  # Create asymmetric pairwise probs
  n_candidates <- 3L
  kappa_matrix <- matrix(c(0.5, 0.6, 0.7,
                           0.4, 0.5, 0.55,
                           0.3, 0.45, 0.5), nrow = 3, byrow = TRUE)

  kA <- strenv$jnp$array(kappa_matrix, dtype = strenv$dtj)
  maskA <- strenv$jnp$ones(list(n_candidates, n_candidates), dtype = strenv$dtj) -
           strenv$jnp$eye(n_candidates, dtype = strenv$dtj)

  eps <- strenv$jnp$array(1e-8, strenv$dtj)
  one_bt <- strenv$jnp$array(1.0, strenv$dtj)
  logoddsA <- strenv$jnp$log(kA + eps) - strenv$jnp$log(one_bt - kA + eps)
  utilityA <- (logoddsA * maskA)$sum(axis = 1L) / strenv$jnp$array(n_candidates, strenv$dtj)
  pA <- strenv$jax$nn$softmax(utilityA)
  pA_r <- jax_to_numeric(pA)

  # Create permuted version (swap candidates 1 and 3)
  perm <- c(3, 2, 1)
  kappa_perm <- kappa_matrix[perm, perm]

  kA_perm <- strenv$jnp$array(kappa_perm, dtype = strenv$dtj)
  logoddsA_perm <- strenv$jnp$log(kA_perm + eps) - strenv$jnp$log(one_bt - kA_perm + eps)
  utilityA_perm <- (logoddsA_perm * maskA)$sum(axis = 1L) / strenv$jnp$array(n_candidates, strenv$dtj)
  pA_perm <- strenv$jax$nn$softmax(utilityA_perm)
  pA_perm_r <- jax_to_numeric(pA_perm)

  # After permutation, probabilities should be permuted correspondingly
  expect_equal(pA_perm_r, pA_r[perm], tolerance = 1e-6)
})
