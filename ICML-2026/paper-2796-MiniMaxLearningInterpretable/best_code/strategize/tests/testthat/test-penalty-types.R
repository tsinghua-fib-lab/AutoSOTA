# =============================================================================
# Penalty Type Tests
# =============================================================================
# Tests for different penalty types: KL, L2, and LogMaxProb.
# =============================================================================

test_that("strategize handles KL penalty type", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$penalty_type <- "KL"

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})

test_that("strategize handles L2 penalty type", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$penalty_type <- "L2"

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})

test_that("strategize handles LogMaxProb penalty type", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$penalty_type <- "LogMaxProb"

  res <- do.call(strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_valid_strategize_output(res)
})
