# Input validation and R-level coercion at the wrapper / FFI boundary: the
# finiteness and dimension guards, data.frame / integer coercion, NULL column
# names, and the print method.

test_that("non-finite inputs are rejected", {
  d <- make_data(n = 40)

  y_na <- d$y
  y_na[1] <- NA
  expect_error(tsl(d$x, y_na, verbosity = 0L), "NA, NaN, or infinite")

  x_inf <- d$x
  x_inf[1, 1] <- Inf
  expect_error(tsl(x_inf, d$y, verbosity = 0L), "NA, NaN, or infinite")

  x_nan <- d$x
  x_nan[2, 2] <- NaN
  expect_error(tsl(x_nan, d$y, verbosity = 0L), "NA, NaN, or infinite")

  fit <- tsl(d$x, d$y, epochs = 3L, seed = 1L, verbosity = 0L)
  nd <- d$x
  nd[1, 1] <- NA
  expect_error(predict(fit, nd), "NA, NaN, or infinite")
})

test_that("fit rejects a response of the wrong length", {
  d <- make_data(n = 30)
  expect_error(tsl(d$x, d$y[-1], verbosity = 0L), "one row per element")
})

test_that("data.frame input is accepted and keeps its column names", {
  d <- make_data()
  fit <- tsl(as.data.frame(d$x), d$y, epochs = 4L, seed = 1L, verbosity = 0L)
  expect_equal(fit$feature_names, c("a", "b", "c"))

  te <- make_data(n = 20, seed = 3)
  p <- predict(fit, as.data.frame(te$x))
  expect_length(p, 20L)
  expect_true(all(is.finite(p)))
})

test_that("integer matrices are coerced to double", {
  set.seed(5)
  xi <- matrix(sample(-5:5, 120 * 3, replace = TRUE), ncol = 3)
  yi <- as.numeric(xi[, 1] - xi[, 2])
  fit <- tsl(xi, yi, epochs = 4L, seed = 1L, verbosity = 0L)
  expect_s3_class(fit, "tsl")
  expect_true(all(is.finite(predict(fit, xi))))
})

test_that("an unnamed design matrix gives NULL feature names", {
  d <- make_data()
  x <- d$x
  dimnames(x) <- NULL
  fit <- tsl(x, d$y, epochs = 3L, seed = 1L, verbosity = 0L)
  expect_null(fit$feature_names)
  expect_length(predict(fit, x), nrow(x))
})

test_that("print.tsl reports the model summary", {
  d <- make_data(n = 60)
  fit <- tsl(d$x, d$y, epochs = 3L, seed = 1L, verbosity = 0L)
  expect_output(print(fit), "Tensor Separation Learning")
  expect_output(print(fit), "features:")
  expect_output(print(fit), "training rows:")
})
