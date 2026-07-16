# =============================================================================
# Cross-Validation (cv_strategize) Tests
# =============================================================================
# Tests for the cv_strategize() function which performs cross-validation
# for lambda selection.
# =============================================================================

test_that("cs_prepare_cv_folds handles scalar fold counts", {
  withr::local_seed(123)

  Y <- rep(c(0, 1), 100)
  W <- data.frame(
    A = rep(c("x", "y"), 100),
    B = rep(c("m", "n"), each = 100),
    stringsAsFactors = FALSE
  )

  folds <- strategize:::cs_prepare_cv_folds(
    folds = 2L,
    Y = Y,
    W = W,
    respondent_id = seq_along(Y),
    respondent_task_id = rep(1L, length(Y))
  )

  expect_identical(folds$n_folds, 2L)
  expect_equal(length(folds$fold_id), length(Y))
  expect_true(all(table(folds$fold_id) > 0L))
})

test_that("cs_prepare_cv_folds handles observation-level assignments", {
  Y <- rep(c(0, 1), 4)
  W <- data.frame(A = rep(c("x", "y"), 4), stringsAsFactors = FALSE)
  respondent_id <- rep(seq_len(4), each = 2L)
  respondent_task_id <- rep(1L, length(Y))
  fold_id <- c("fold_a", "fold_a", "fold_b", "fold_b",
               "fold_a", "fold_a", "fold_b", "fold_b")

  folds <- strategize:::cs_prepare_cv_folds(
    folds = fold_id,
    Y = Y,
    W = W,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id
  )

  expect_identical(folds$n_folds, 2L)
  expect_identical(as.character(folds$fold_id), fold_id)
  expect_equal(sort(folds$indi_list[[2, 1]]), c(1L, 2L, 5L, 6L))
})

test_that("cs_prepare_cv_folds handles task-level assignments", {
  Y <- rep(c(0, 1), 4)
  W <- data.frame(A = rep(c("x", "y"), 4), stringsAsFactors = FALSE)
  respondent_id <- rep(seq_len(4), each = 2L)
  respondent_task_id <- rep(1L, length(Y))

  folds <- strategize:::cs_prepare_cv_folds(
    folds = c("fold_a", "fold_b", "fold_a", "fold_b"),
    Y = Y,
    W = W,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id
  )

  expect_identical(folds$n_folds, 2L)
  expect_identical(as.character(folds$fold_id),
                   c("fold_a", "fold_a", "fold_b", "fold_b",
                     "fold_a", "fold_a", "fold_b", "fold_b"))
})

test_that("cs_prepare_cv_folds rejects invalid assignments", {
  Y <- rep(c(0, 1), 4)
  W <- data.frame(A = rep(c("x", "y"), 4), stringsAsFactors = FALSE)
  respondent_id <- rep(seq_len(4), each = 2L)
  respondent_task_id <- rep(1L, length(Y))

  expect_error(
    strategize:::cs_prepare_cv_folds(
      folds = c(1, 2, 1, 1, 2, 2, 1, 1),
      Y = Y,
      W = W,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id
    ),
    "respondent-task"
  )
  expect_error(
    strategize:::cs_prepare_cv_folds(
      folds = c(1, NA, 2, 2),
      Y = Y,
      W = W,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id
    ),
    "missing fold IDs"
  )
  expect_error(
    strategize:::cs_prepare_cv_folds(
      folds = c(1, 2, 3),
      Y = Y,
      W = W,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id
    ),
    "length\\(Y\\)"
  )
})

test_that("cs_prepare_cv_folds keeps pair_id rows together", {
  Y <- rep(c(0, 1), 4)
  W <- data.frame(A = rep(c("x", "y"), 4), stringsAsFactors = FALSE)
  pair_id <- rep(seq_len(4), each = 2L)
  respondent_id <- seq_along(Y)
  respondent_task_id <- rep(1L, length(Y))

  expect_error(
    strategize:::cs_prepare_cv_folds(
      folds = c("fold_a", "fold_b", "fold_a", "fold_a",
                "fold_b", "fold_b", "fold_a", "fold_a"),
      Y = Y,
      W = W,
      respondent_id = respondent_id,
      respondent_task_id = respondent_task_id,
      pair_id = pair_id
    ),
    "pair_id"
  )

  folds <- strategize:::cs_prepare_cv_folds(
    folds = c("fold_a", "fold_b", "fold_a", "fold_b"),
    Y = Y,
    W = W,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    pair_id = pair_id
  )
  expect_identical(as.character(folds$fold_id),
                   c("fold_a", "fold_a", "fold_b", "fold_b",
                     "fold_a", "fold_a", "fold_b", "fold_b"))
})

test_that("cv_strategize selects lambda with single value", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_type(res, "list")
  expect_true("lambda" %in% names(res))
  expect_true("CVInfo" %in% names(res))
  expect_true("pi_star_point" %in% names(res))
  expect_true("p_list" %in% names(res))
})

test_that("cv_strategize handles vector of lambda values", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)
  params$lambda <- c(0.01, 0.1, 1.0)

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_type(res, "list")
  expect_true("lambda" %in% names(res))
  expect_true("CVInfo" %in% names(res))
})

test_that("cv_strategize handles K > 1 (multi-cluster)", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_no_factorhet_stack()

  data <- generate_test_data(n = 500, seed = 42)
  data <- add_respondent_covariates(data)
  params <- default_strategize_params(fast = TRUE)
  params$K <- 2

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W, X = data$X),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expect_type(res, "list")
  expect_true("lambda" %in% names(res))
  expect_true("pi_star_point" %in% names(res))
  expect_equal(length(res$pi_star_point), 2)
})

test_that("cv_strategize returns expected output structure", {
  skip_on_cran()
  skip_if_no_jax()

  data <- generate_test_data(n = 500, seed = 42)
  params <- default_strategize_params(fast = TRUE)

  res <- do.call(cv_strategize, c(
    list(Y = data$Y, W = data$W),
    data[c("pair_id", "respondent_id", "respondent_task_id", "profile_order")],
    params
  ))

  expected_fields <- c("lambda", "CVInfo", "pi_star_point", "p_list")
  for (field in expected_fields) {
    expect_true(field %in% names(res), info = paste("Missing field:", field))
  }
})
