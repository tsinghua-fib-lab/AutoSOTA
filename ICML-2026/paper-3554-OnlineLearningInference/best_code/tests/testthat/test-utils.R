test_that("prox_forward works for valid inputs", {
  # Test with p = 3, q = 5
  prox <- prox_forward(3, 5)
  t <- seq(0, 1, length.out = 100)
  b1 <- splines2::bpoly(t, degree = 2, intercept = TRUE)
  b2 <- splines2::bpoly(t, degree = 4, intercept = TRUE)
  alpha1 <- rnorm(3)
  alpha2 <- prox %*% alpha1
  y1 <- b1 %*% alpha1
  y2 <- b2 %*% alpha2
  expect_equal(mean((y1 - y2)^2), 0, tolerance = 1e-8)


  expect_true(is.matrix(prox))
  expect_equal(dim(prox), c(5, 3))

  # Test with p = 1, q = 1
  prox <- prox_forward(1, 1)
  expect_true(is.matrix(prox))
  expect_equal(dim(prox), c(1, 1))
  expect_equal(prox, diag(1))
})

test_that("prox_forward throws errors for invalid inputs", {
  # Non-integer inputs
  expect_error(prox_forward(3.5, 5), "p and q must be integers")
  expect_error(prox_forward(3, 5.5), "p and q must be integers")

  # Non-numeric inputs
  expect_error(prox_forward("3", 5), "p and q must be integers")
  expect_error(prox_forward(3, "5"), "p and q must be integers")

  # Negative or zero inputs
  expect_error(prox_forward(-3, 5), "p and q must be positive integers")
  expect_error(prox_forward(3, 0), "p and q must be positive integers")

  # p is greater than q
  expect_error(prox_forward(5, 3), "p must be less than or equal to q")
})


test_that("prox_reverse works for valid inputs", {
  # Test with q = 5, p = 3
  prox1 <- prox_forward(3, 5)
  prox2 <- prox_reverse(5, 3)
  expect_equal(prox2 %*% prox1, diag(3), tolerance = 1e-8)
  expect_true(is.matrix(prox2))
  expect_equal(dim(prox2), c(3, 5))

  # Test with q = 1, p = 1
  prox <- prox_reverse(1, 1)
  expect_true(is.matrix(prox))
  expect_equal(dim(prox), c(1, 1))
  expect_equal(prox, diag(1))
})

test_that("prox_reverse throws errors for invalid inputs", {
  # Non-integer inputs
  expect_error(prox_reverse(5.5, 3), "p and q must be integers")
  expect_error(prox_reverse(5, 3.5), "p and q must be integers")

  # Non-numeric inputs
  expect_error(prox_reverse("5", 3), "p and q must be integers")
  expect_error(prox_reverse(5, "3"), "p and q must be integers")

  # Negative or zero inputs
  expect_error(prox_reverse(-5, 3), "p and q must be positive integers")
  expect_error(prox_reverse(5, 0), "p and q must be positive integers")

  # p is greater than q
  expect_error(prox_reverse(3, 5), "p must be less than or equal to q")
})

test_that("int_basehaz computes cumulative baseline hazard", {
  result <- int_basehaz(1, c(0.5, -0.2, 0.3), 3, c(0, 2))

  f <- function(t) exp(0.5 - 0.7 * t + 0.3 * t^2)
  cbh_true <- integrate(f, 0, 1)
  expect_true(is.list(result))
  expect_true(is.numeric(result$cbh))
  expect_equal(as.vector(result$cbh), cbh_true$value, tolerance = 1e-6)
})

test_that("int_basehaz computes gradient of the cumulative baseline hazard", {
  result <- int_basehaz(1, c(0.5, -0.2, 0.3), 3, c(0, 2))

  f <- function(t) exp(0.5 - 0.7 * t + 0.3 * t^2)
  g1 <- function(t) f(t) * (1 - t / 2)^2
  g2 <- function(t) f(t) * t * (1 - t / 2)
  g3 <- function(t) f(t) * t^2 / 4
  dcbh_true <- c(
    integrate(g1, 0, 1)$value,
    integrate(g2, 0, 1)$value,
    integrate(g3, 0, 1)$value
  )
  expect_true(is.matrix(result$dcbh))
  expect_equal(ncol(result$dcbh), 3)
  expect_equal(as.vector(result$dcbh), as.vector(dcbh_true), tolerance = 1e-6)
})

test_that("int_basehaz computes Hessian of the cumulative baseline hazard", {
  result <- int_basehaz(1, c(0.5, -0.2, 0.3), 3, c(0, 2))

  f <- function(t) exp(0.5 - 0.7 * t + 0.3 * t^2)
  b <- list(
    function(t) (1 - t / 2)^2,
    function(t) t * (1 - t / 2),
    function(t) t^2 / 4
  )
  d2cbh_true <- matrix(0, nrow = 3, ncol = 3)
  for (i in 1:3) {
    for (j in 1:3) {
      g <- function(t) f(t) * b[[i]](t) * b[[j]](t)
      d2cbh_true[i, j] <- integrate(g, 0, 1)$value
    }
  }
  expect_true(is.matrix(result$d2cbh))
  expect_equal(ncol(result$d2cbh), 9)
  expect_equal(matrix(result$d2cbh, nrow = 3), d2cbh_true, tolerance = 1e-6)
})

test_that("int_basehaz handles multiple time points", {
  times <- c(0.5, 1, 1.5)
  result <- int_basehaz(times, c(0.5, -0.2, 0.3), 3, c(0, 2))

  expect_equal(length(result$cbh), 3)
  expect_equal(nrow(result$dcbh), 3)
  expect_equal(nrow(result$d2cbh), 3)
})


test_that("objective computes value, gradient, and hessian correctly", {
  par <- c(0.5, -0.2, 0.3, 0.1, -0.1)
  time <- c(1, 2, 3)
  status <- c(1, 0, 1)
  x <- matrix(c(1, 0, 1, 0, 1, 0), ncol = 2)
  boundary <- c(0, 3)
  theta <- c(0.4, -0.1, 0.2, 0.05, -0.05)
  hessian <- diag(5)

  result <- objective(par, time, status, x, boundary, theta, hessian)

  expect_true(is.numeric(result))
  expect_true("gradient" %in% names(attributes(result)))
  expect_true("hessian" %in% names(attributes(result)))

  expect_true(is.numeric(attr(result, "gradient")))
  expect_true(is.matrix(attr(result, "hessian")))

  expect_equal(length(attr(result, "gradient")), length(par))
  expect_equal(dim(attr(result, "hessian")), c(length(par), length(par)))
})
