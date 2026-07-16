test_that("validate_equilibrium validates nMonte and seed", {
  skip_on_cran()

  expect_error(validate_equilibrium(list(), nMonte = 0), "nMonte must be")
  expect_error(validate_equilibrium(list(), seed = -1), "seed must be")
})

test_that("validate_equilibrium errors on missing Q function", {
  skip_on_cran()

  mock_result <- list(
    convergence_history = list(adversarial = TRUE),
    FullGetQStar_ = NULL,
    strenv = list(jnp = 1)
  )

  expect_error(validate_equilibrium(mock_result), "Result does not contain Q function")
})

test_that("validate_equilibrium errors when JAX is unavailable", {
  skip_on_cran()

  mock_result <- list(
    convergence_history = list(adversarial = TRUE),
    FullGetQStar_ = function() NULL,
    strenv = list()
  )

  expect_error(validate_equilibrium(mock_result), "JAX environment not available")
})

test_that("validate_equilibrium grid search batches objective evaluations", {
  body_validate <- paste(deparse(body(validate_equilibrium)), collapse = "\n")
  body_grid <- paste(deparse(body(find_best_response_grid)), collapse = "\n")
  body_random <- paste(deparse(body(find_best_response_random)), collapse = "\n")

  expect_true(grepl("strenv\\$jax\\$vmap", body_validate, perl = TRUE))
  expect_true(grepl("batch_ast", body_grid, fixed = TRUE))
  expect_true(grepl("batch_dag", body_grid, fixed = TRUE))
  expect_true(grepl("batch_ast", body_random, fixed = TRUE))
  expect_true(grepl("batch_dag", body_random, fixed = TRUE))
  expect_false(grepl("eval_Q\\(", body_grid, perl = TRUE))
  expect_false(grepl("eval_Q\\(", body_random, perl = TRUE))
  expect_false(grepl("for\\s*\\(", body_grid, perl = TRUE))
  expect_false(grepl("for\\s*\\(", body_random, perl = TRUE))
})
