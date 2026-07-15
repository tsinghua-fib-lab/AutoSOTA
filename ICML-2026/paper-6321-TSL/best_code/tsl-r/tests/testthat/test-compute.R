# The compute layer reconstructs interpretability quantities from the fitted
# components. Because TSL is separable these are exact, so we pin them against
# predict() and the closed-form definitions.

test_that("first-order PD reconstructs predict() on a swept grid", {
  d <- make_data(n = 300)
  fit <- tsl(d$x, d$y, epochs = 5L, n_trees = 6L, seed = 11L, verbosity = 0L)

  X <- d$x
  feat <- 1L
  grid_points <- 40L
  pd <- tsl_pd(fit, X, features = feat, grid_points = grid_points)

  # Sum the per-stage net PD over stages -> total first-order PD on the grid.
  g <- sort(unique(pd$x))
  total <- tapply(pd$net, pd$x, sum)[as.character(g)]

  # Independent ground truth: tile every other feature at its column mean over
  # the data is NOT the PD; the PD marginalises, so average predict() over the
  # background with feature `feat` fixed to each grid value.
  brute <- vapply(g, function(v) {
    Xb <- X
    Xb[, feat] <- v
    mean(predict(fit, Xb))
  }, numeric(1))

  expect_equal(as.numeric(total), as.numeric(brute), tolerance = 1e-8)
})

test_that("tsl_pd pos/neg branches sum to net", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 4L, n_trees = 4L, seed = 3L, verbosity = 0L)
  pd <- tsl_pd(fit, features = c("a", "b"), grid_points = 25L)
  expect_equal(pd$pos + pd$neg, pd$net, tolerance = 1e-12)
  expect_true(all(pd$c_plus >= 0))
  expect_true(all(pd$c_minus >= 0))
})

test_that("2D PD surface matches the outer-product reconstruction", {
  d <- make_data(n = 250)
  fit <- tsl(d$x, d$y, epochs = 4L, n_trees = 5L, seed = 7L, verbosity = 0L)
  surf <- tsl_pd_2d(fit, feature_x = "a", feature_y = "b", grid_points = 15L)

  # Brute force: at each (a, b) grid node, fix those two columns and average
  # predict() over the background (summed over stages == full model).
  X <- d$x
  xg <- attr(surf, "x_vals")
  yg <- attr(surf, "y_vals")
  total <- tapply(surf$value, list(surf$x, surf$y), sum)
  brute <- matrix(0, length(xg), length(yg))
  for (ix in seq_along(xg)) for (iy in seq_along(yg)) {
    Xb <- X
    Xb[, 1] <- xg[ix]
    Xb[, 2] <- yg[iy]
    brute[ix, iy] <- mean(predict(fit, Xb))
  }
  expect_equal(as.numeric(total[as.character(xg), as.character(yg)]),
               as.numeric(brute), tolerance = 1e-7)
})

test_that("local explanation totals to predict() and obeys intercept formulas", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 5L, n_trees = 4L, seed = 21L, verbosity = 0L)
  pt <- d$x[1, ]
  ex <- tsl_local(fit, pt)

  expect_equal(ex$total_prediction, as.numeric(predict(fit, matrix(pt, 1))),
               tolerance = 1e-8)
  expect_equal(sum(ex$stage_contributions), ex$total_prediction,
               tolerance = 1e-12)

  comp <- tsl_components(fit)
  for (s in seq_along(comp)) {
    g <- comp[[s]]$combined_grid_tensor
    sp <- comp[[s]]$scaling_plus
    sm <- comp[[s]]$scaling_minus
    effp <- sp * g$lambda_plus
    effm <- sm * g$lambda_minus
    expect_equal(ex$intercept_backbone[s], sqrt(max(effp * effm, 0)),
                 tolerance = 1e-10)
  }
})

test_that("importance: weights sum to 1 and combined = backbone + gamma*tilt", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 5L, n_trees = 4L, seed = 31L, verbosity = 0L)
  imp <- tsl_importance(fit, gamma = 1.5)

  expect_equal(sum(imp$stage_weights$weight), 1, tolerance = 1e-12)
  expect_equal(imp$global$combined,
               imp$global$backbone + 1.5 * imp$global$tilt, tolerance = 1e-12)
  expect_true(all(imp$per_stage$backbone >= 0))
  expect_true(all(imp$per_stage$tilt >= 0))
})

test_that("per-stage backbone importance is the population var of log-backbone", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 3L, n_trees = 1L, seed = 5L, verbosity = 0L)
  imp <- tsl_importance(fit)
  comp <- tsl_components(fit)
  X <- d$x
  var_pop <- function(v) mean((v - mean(v))^2)

  g <- comp[[1]]$combined_grid_tensor
  j <- 1L
  splits <- g$splits[[j]]
  b <- g$backbone_values[[j]]
  idx <- pmin(findInterval(X[, j], splits), length(b) - 1L) + 1L
  expected <- var_pop(log(pmax(b[idx], 1e-12)))
  got <- imp$per_stage$backbone[imp$per_stage$feature == "a" &
                                  imp$per_stage$stage == "Stage 1"]
  expect_equal(got, expected, tolerance = 1e-12)
})

test_that(".tsl_stage_bt handles the interval boundary and a no-split feature", {
  # Hand-built grid tensor: feature 1 has two intervals split at 0; feature 2
  # has a single interval (no splits).
  gt <- list(
    splits = list(0, numeric(0)),
    backbone_values = list(c(2, 5), 3),
    tilt_values = list(c(-1, 1), 0.4)
  )
  # value below split -> first interval; at/above split -> second (split <= v).
  expect_equal(.tsl_stage_bt(gt, 1, c(-0.5, 0, 0.5))$b, c(2, 5, 5))
  expect_equal(.tsl_stage_bt(gt, 1, c(-0.5, 0, 0.5))$d, c(-1, 1, 1))
  # no-split feature is constant everywhere
  expect_equal(.tsl_stage_bt(gt, 2, c(-9, 0, 9))$b, c(3, 3, 3))
})

test_that("X defaults to the retained background", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 3L, n_trees = 3L, seed = 1L, verbosity = 0L)
  expect_equal(tsl_pd(fit, features = "a"), tsl_pd(fit, fit$x_background,
                                                   features = "a"))
})
