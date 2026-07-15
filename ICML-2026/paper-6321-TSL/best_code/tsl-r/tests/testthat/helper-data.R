# A largely separable target on non-square data (n x 3). The non-square shape
# doubles as an orientation check: a transposed design matrix at the FFI
# boundary would either error or produce garbage predictions.
make_data <- function(n = 200, seed = 1) {
  set.seed(seed)
  x <- matrix(stats::runif(n * 3, -2, 2), ncol = 3,
              dimnames = list(NULL, c("a", "b", "c")))
  y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + stats::rnorm(n, sd = 0.1)
  list(x = x, y = y)
}
