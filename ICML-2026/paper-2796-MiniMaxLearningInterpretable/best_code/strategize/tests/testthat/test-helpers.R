# =============================================================================
# Unit Tests for Internal Helper Functions
# =============================================================================
# Tests for unexported helper functions accessed via strategize:::
# These are fast, isolated unit tests that don't require external dependencies.
# =============================================================================

test_that("toSimplex returns a valid probability vector", {
  x <- c(0.1, -0.2, 0.3)
  s <- strategize:::toSimplex(x)

  expect_equal(sum(s), 1, tolerance = 1e-7)
  expect_true(all(s >= 0))
})

test_that("toSimplex handles extreme positive values", {
  x_large <- c(25, 30, 20)
  s_large <- strategize:::toSimplex(x_large)

  expect_equal(sum(s_large), 1, tolerance = 1e-7)
  expect_true(all(s_large >= 0))
})

test_that("toSimplex handles extreme negative values", {
  x_neg <- c(-25, -30, -20)
  s_neg <- strategize:::toSimplex(x_neg)

  expect_equal(sum(s_neg), 1, tolerance = 1e-7)
  expect_true(all(s_neg >= 0))
})

test_that("toSimplex handles equal values", {
  x_equal <- c(1, 1, 1)
  s_equal <- strategize:::toSimplex(x_equal)

  expect_equal(sum(s_equal), 1, tolerance = 1e-7)
  expect_equal(s_equal, rep(1/3, 3), tolerance = 1e-7)
})

test_that("ess_fxn computes effective sample size for equal weights", {
  w <- c(1, 1, 1, 1)
  expect_equal(strategize:::ess_fxn(w), 4)
})

test_that("ess_fxn computes effective sample size for unequal weights", {
  w <- c(1, 0.5)
  expected <- sum(w)^2 / sum(w^2)
  expect_equal(strategize:::ess_fxn(w), expected)
})

test_that("ess_fxn handles single weight", {
  expect_equal(strategize:::ess_fxn(1), 1)
})

test_that("ess_fxn handles extreme unbalanced weights", {
  w_extreme <- c(1, 0, 0, 0)
  expect_equal(strategize:::ess_fxn(w_extreme), 1)
})

# ---------------------------------------------------------------------------
# glm_response_has_variation: gates glinternet screening to avoid the C-level
# segfault that glinternet's group_lasso triggers on a single-value response.
# ---------------------------------------------------------------------------

test_that("glm_response_has_variation flags a degenerate response", {
  # The exact segfault trigger: a fold whose response is all identical.
  expect_false(strategize:::glm_response_has_variation(c(1, 1, 1)))
  expect_false(strategize:::glm_response_has_variation(c(0, 0, 0, 0)))
  expect_false(strategize:::glm_response_has_variation(5))
})

test_that("glm_response_has_variation accepts a varied response", {
  expect_true(strategize:::glm_response_has_variation(c(1, 0, 1)))
  expect_true(strategize:::glm_response_has_variation(c(0, 1)))
  expect_true(strategize:::glm_response_has_variation(c(-1, 0, 1)))
})

test_that("glm_response_has_variation ignores non-finite values", {
  # NA / Inf must not count as a distinct level that fakes variation.
  expect_false(strategize:::glm_response_has_variation(c(1, 1, NA)))
  expect_false(strategize:::glm_response_has_variation(c(1, NA, Inf)))
  expect_true(strategize:::glm_response_has_variation(c(1, 0, NA)))
  expect_false(strategize:::glm_response_has_variation(c(NA, NA)))
})

test_that("RescaleFxn rescales and recenters correctly", {
  x <- c(-1, 0, 1)
  res <- strategize:::RescaleFxn(x, estMean = 2, estSD = 3)

  expect_equal(res, x * 3 + 2)
})
test_that("RescaleFxn rescales without centering", {
  x <- c(-1, 0, 1)
  res <- strategize:::RescaleFxn(x, estMean = 2, estSD = 3, center = FALSE)

  expect_equal(res, x * 3)
})

test_that("getSE handles missing values correctly", {
  vals <- c(1, 2, 3, NA)
  n_valid <- sum(!is.na(vals))

  expect_equal(
    strategize:::getSE(vals),
    sqrt(var(vals, na.rm = TRUE) / n_valid)
  )
})

test_that("getSE returns NA for all-NA input", {
  vals_na <- c(NA, NA, NA)
  expect_true(is.na(strategize:::getSE(vals_na)))
})
