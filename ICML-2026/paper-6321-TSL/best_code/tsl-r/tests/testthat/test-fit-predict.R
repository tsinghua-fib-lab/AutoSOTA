test_that("tsl() fits and returns a well-formed object", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L)

  expect_s3_class(fit, "tsl")
  expect_equal(fit$n_features, 3L)
  expect_equal(fit$n_obs, 200L)
  expect_equal(fit$feature_names, c("a", "b", "c"))
  expect_length(fit$residuals, 200L)
  expect_length(fit$y_hat, 200L)
  expect_true(is.finite(fit$err))
  expect_true(all(is.finite(fit$y_hat)))
})

test_that("predict() returns finite predictions of the right length", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L)

  te <- make_data(n = 50, seed = 2)
  p <- predict(fit, te$x)

  expect_length(p, 50L)
  expect_true(all(is.finite(p)))
  # The target is largely separable, so predictions should track the truth.
  expect_gt(stats::cor(p, te$y), 0.8)
})

test_that("fit is reproducible for a fixed seed", {
  d <- make_data()
  f1 <- tsl(d$x, d$y, epochs = 8L, seed = 7L, verbosity = 0L)
  f2 <- tsl(d$x, d$y, epochs = 8L, seed = 7L, verbosity = 0L)
  expect_equal(predict(f1, d$x), predict(f2, d$x))
})

test_that("training fit explains most variance", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 10L, seed = 123L, verbosity = 0L)
  yhat <- predict(fit, d$x)
  r2 <- 1 - sum((d$y - yhat)^2) / sum((d$y - mean(d$y))^2)
  expect_gt(r2, 0.7)
})

test_that("invalid strategy names are rejected", {
  d <- make_data(n = 20)
  expect_error(tsl(d$x, d$y, split_strategy = "nope"))
  expect_error(tsl(d$x, d$y, refinement_strategy = "nope"))
})

test_that("dimension mismatch in predict() is caught", {
  d <- make_data(n = 30)
  fit <- tsl(d$x, d$y, epochs = 3L, seed = 1L, verbosity = 0L)
  expect_error(predict(fit, d$x[, 1:2]), "features")
})

test_that("every split strategy fits and tracks the target", {
  d <- make_data()
  te <- make_data(n = 50, seed = 2)
  for (ss in c("random", "best_split", "top_k")) {
    fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L,
               split_strategy = ss)
    p <- predict(fit, te$x)
    expect_true(all(is.finite(p)), info = ss)
    expect_gt(stats::cor(p, te$y), 0.85)
  }
})

test_that("huber refinement fits and tracks the target", {
  d <- make_data()
  te <- make_data(n = 50, seed = 2)
  fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L,
             refinement_strategy = "huber")
  p <- predict(fit, te$x)
  expect_true(all(is.finite(p)))
  expect_gt(stats::cor(p, te$y), 0.85)
})

test_that("stored fitted values match a fresh prediction", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L)
  expect_equal(predict(fit, d$x), fit$y_hat, tolerance = 1e-10)
})

test_that("residuals and fitted values reconstruct the response", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L)
  expect_equal(fit$residuals + fit$y_hat, d$y,
               tolerance = 1e-12, ignore_attr = TRUE)
})

test_that("the model has one stage per epoch", {
  d <- make_data()
  for (ep in c(1L, 3L, 10L)) {
    fit <- tsl(d$x, d$y, epochs = ep, n_trees = 2L, seed = 1L, verbosity = 0L)
    expect_length(tsl_components(fit), ep)
  }
})

test_that("predict handles a single row and a single-feature model", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 5L, seed = 1L, verbosity = 0L)

  # one observation passed as a 1-row matrix
  expect_length(predict(fit, d$x[1, , drop = FALSE]), 1L)
  expect_length(predict(fit, matrix(d$x[1, ], nrow = 1)), 1L)
  # a bare vector is read as one column, so the feature-count guard fires
  expect_error(predict(fit, d$x[1, ]), "features")

  # a single-feature model exercises the product-over-one-axis path
  set.seed(11)
  x1 <- d$x[, 1, drop = FALSE]
  y1 <- 2 * d$x[, 1] + stats::rnorm(nrow(d$x), sd = 0.1)
  fit1 <- tsl(x1, y1, epochs = 8L, seed = 1L, verbosity = 0L)
  expect_equal(fit1$n_features, 1L)

  te <- make_data(n = 50, seed = 9)
  p1 <- predict(fit1, te$x[, 1, drop = FALSE])
  expect_length(p1, 50L)
  expect_gt(stats::cor(p1, 2 * te$x[, 1]), 0.95)
})

test_that("different seeds give different fits", {
  d <- make_data()
  f1 <- tsl(d$x, d$y, epochs = 8L, seed = 1L, verbosity = 0L)
  f2 <- tsl(d$x, d$y, epochs = 8L, seed = 2L, verbosity = 0L)
  expect_false(isTRUE(all.equal(predict(f1, d$x), predict(f2, d$x))))
})
