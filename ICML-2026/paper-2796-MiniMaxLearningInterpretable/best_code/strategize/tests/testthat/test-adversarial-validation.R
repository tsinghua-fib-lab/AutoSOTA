# Tests for adversarial validation functions

test_that("is_adversarial correctly identifies adversarial results", {
  # Mock non-adversarial result
  non_adv_result <- list(
    convergence_history = list(adversarial = FALSE)
  )
  expect_false(is_adversarial(non_adv_result))

  # Mock adversarial result
  adv_result <- list(
    convergence_history = list(adversarial = TRUE)
  )
  expect_true(is_adversarial(adv_result))

  # Missing convergence history
  no_hist <- list()
  expect_false(is_adversarial(no_hist))
})

test_that("get_final_gradients extracts gradient norms", {
  # Mock result with convergence history
  mock_result <- list(
    convergence_history = list(
      nSGD = 100,
      grad_ast = c(rep(1, 99), 0.001),
      grad_dag = c(rep(1, 99), 0.002)
    )
  )

  grads <- get_final_gradients(mock_result)

  expect_named(grads, c("AST", "DAG"))
  expect_equal(grads["AST"], c(AST = 0.001))
  expect_equal(grads["DAG"], c(DAG = 0.002))
})

test_that("get_final_gradients handles missing history", {
  mock_result <- list()
  expect_error(get_final_gradients(mock_result), "No convergence history")
})

test_that("plot_convergence requires convergence_history", {
  mock_result <- list()
  expect_error(plot_convergence(mock_result), "No convergence history")
})

test_that("plot_convergence validates metrics argument", {
  mock_result <- list(
    convergence_history = list(
      nSGD = 10,
      adversarial = TRUE,
      grad_ast = rep(0.1, 10),
      grad_dag = rep(0.1, 10),
      loss_ast = rep(0.5, 10),
      loss_dag = rep(0.5, 10),
      inv_lr_ast = rep(0.01, 10),
      inv_lr_dag = rep(0.01, 10)
    )
  )

  # Valid metrics
  expect_no_error(plot_convergence(mock_result, metrics = "gradient"))
  expect_no_error(plot_convergence(mock_result, metrics = "loss"))
  expect_no_error(plot_convergence(mock_result, metrics = c("gradient", "loss")))

  # Invalid metric
  expect_error(plot_convergence(mock_result, metrics = "invalid"))
})

test_that("validate_equilibrium requires adversarial result", {
  non_adv <- list(
    convergence_history = list(adversarial = FALSE)
  )

  expect_error(
    validate_equilibrium(non_adv),
    "requires an adversarial strategize result"
  )
})

test_that("plot_quadrant_breakdown requires adversarial result", {
  non_adv <- list(
    convergence_history = list(adversarial = FALSE)
  )

  expect_error(
    plot_quadrant_breakdown(non_adv),
    "requires an adversarial strategize result"
  )
})

test_that("summarize_adversarial requires adversarial result", {
  non_adv <- list(
    convergence_history = list(adversarial = FALSE)
  )

  expect_error(
    summarize_adversarial(non_adv),
    "requires an adversarial strategize result"
  )
})

# Integration tests that require JAX use skip_if_no_jax() from helper-strategize.R

test_that("convergence history is included in strategize output", {
  skip_if_no_jax()

  # Generate minimal test data
  set.seed(42)
  n <- 200

  # Binary outcome
  Y <- sample(0:1, n, replace = TRUE)

  # Create factor columns
  gender <- sample(c("male", "female"), n, replace = TRUE)
  W <- data.frame(gender = gender)

  # Baseline probabilities
  p_list <- list(gender = c(male = 0.5, female = 0.5))

  # Run non-adversarial strategize (simpler)
  result <- strategize(
    Y = Y, W = W, p_list = p_list,
    lambda = 0.1, nSGD = 5L, nMonte_Qglm = 5L
  )

  # Check convergence history exists

  expect_true(!is.null(result$convergence_history))
  expect_true(!is.null(result$convergence_history$nSGD))
  expect_true(!is.null(result$convergence_history$grad_ast))
})

test_that("plot_convergence works with real strategize output", {
  skip_if_no_jax()

  set.seed(42)
  n <- 200
  Y <- sample(0:1, n, replace = TRUE)
  gender <- sample(c("male", "female"), n, replace = TRUE)
  W <- data.frame(gender = gender)
  p_list <- list(gender = c(male = 0.5, female = 0.5))

  result <- strategize(
    Y = Y, W = W, p_list = p_list,
    lambda = 0.1, nSGD = 5L, nMonte_Qglm = 5L
  )

  # Should not error
  expect_no_error(plot_convergence(result, metrics = "gradient"))
})

# ============================================
# Hessian Geometry Analysis Tests
# ============================================

test_that("check_hessian_geometry requires adversarial result", {
  non_adv <- list(
    convergence_history = list(adversarial = FALSE)
  )

  expect_error(
    check_hessian_geometry(non_adv),
    "requires adversarial=TRUE result"
  )
})

test_that("check_hessian_geometry returns NULL when Hessian unavailable", {
  # Mock result with hessian_available = FALSE
  mock_result <- list(
    convergence_history = list(adversarial = TRUE),
    hessian_available = FALSE,
    hessian_skipped_reason = "user_disabled"
  )

  expect_warning(
    result <- check_hessian_geometry(mock_result, verbose = FALSE),
    "not available"
  )
  expect_null(result)
})

test_that("check_hessian_geometry returns NULL for high_dimension skip", {
  mock_result <- list(
    convergence_history = list(adversarial = TRUE),
    hessian_available = FALSE,
    hessian_skipped_reason = "high_dimension"
  )

  expect_warning(
    result <- check_hessian_geometry(mock_result, verbose = FALSE),
    "not available"
  )
  expect_null(result)
})

test_that("print.hessian_analysis produces output", {
  # Create a mock hessian_analysis object
  mock_hess <- list(
    status = "PASS",
    valid_saddle = TRUE,
    eigenvalues_ast = c(-0.1, -0.2, -0.3),
    eigenvectors_ast = diag(3),
    is_negative_definite_ast = TRUE,
    is_negative_semidefinite_ast = TRUE,
    condition_number_ast = 3.0,
    flat_directions_ast = 0,
    eigenvalues_dag = c(0.1, 0.2, 0.3),
    eigenvectors_dag = diag(3),
    is_positive_definite_dag = TRUE,
    is_positive_semidefinite_dag = TRUE,
    condition_number_dag = 3.0,
    flat_directions_dag = 0,
    interpretation = "Valid Nash equilibrium",
    tolerance = 1e-6
  )
  class(mock_hess) <- c("hessian_analysis", "list")

  # Should print without error
  expect_output(print(mock_hess), "Hessian Geometry Analysis")
  expect_output(print(mock_hess), "Status: PASS")
})

test_that("summarize_adversarial includes Hessian when available", {
  # Mock result with Hessian available
  mock_result <- list(
    convergence_history = list(
      adversarial = TRUE,
      nSGD = 100,
      grad_ast = c(rep(0.1, 99), 0.001),
      grad_dag = c(rep(0.1, 99), 0.002)
    ),
    Q_point = 0.52,
    Q_se = 0.01,
    lambda = 0.1,
    penalty_type = "KL",
    pi_star_point = list(
      k1 = list(gender = c(male = 0.6, female = 0.4)),
      k2 = list(gender = c(male = 0.4, female = 0.6))
    ),
    AstProp = 0.5,
    DagProp = 0.5,
    hessian_available = FALSE,
    hessian_skipped_reason = "not_adversarial"
  )

  # Should run without error (Hessian unavailable case)
  summary <- summarize_adversarial(mock_result, validate = FALSE, verbose = FALSE)

  expect_true(is.na(summary$geometry_valid))
  expect_null(summary$hessian_analysis)
})

test_that("hessian_available flag is FALSE for non-adversarial mode", {
  skip_if_no_jax()

  # Use shared test data generators
  set.seed(42)
  n <- 200
  Y <- sample(0:1, n, replace = TRUE)
  gender <- sample(c("male", "female"), n, replace = TRUE)
  W <- data.frame(gender = gender)
  p_list <- list(gender = c(male = 0.5, female = 0.5))

  # Run non-adversarial strategize
  result <- strategize(
    Y = Y, W = W, p_list = p_list,
    lambda = 0.1, nSGD = 5L, nMonte_Qglm = 5L
  )

  # Non-adversarial mode should have hessian_available = FALSE
  expect_false(result$hessian_available)
  expect_equal(result$hessian_skipped_reason, "not_adversarial")
})
