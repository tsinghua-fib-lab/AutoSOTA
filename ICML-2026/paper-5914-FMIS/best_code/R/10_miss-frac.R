# Helper function that returns the Dinkelbach solver
get_miss_solver <- function(X, R, sign = 1L, tol = 1e-16, max_iter = 1e3) {
  n <- length(X)
  stopifnot(n == length(R))

  num <- X * R * sign
  den <- X^2
  tot <- sum(den)
  lambda_init <- max(num / (tot - den))

  return(function(k, lambda = lambda_init) {
    S <- integer(0L)

    for (iter in seq_len(max_iter)) {
      scores <- num + lambda * den
      # S_i <- order(scores, decreasing = TRUE)[seq_len(k)]
      S_i <- order_partial(scores, k, decreasing = TRUE)
      num_i <- sum(num[S_i])
      tot_i <- tot - sum(den[S_i])
      if (tot_i <= 0) {
        stop("Non-positive denominator reached – check rank.")
      }
      lambda_i <- num_i / tot_i

      if (identical(S_i, S) || abs(lambda_i - lambda) < tol) {
        return(list(
          k = k,
          best_S = S_i,
          best_value = lambda_i,
          iter = iter
        ))
      }
      lambda <- lambda_i
      S <- S_i
    }
    warning("Did not converge; increase `max_iter` or check data")
    return(list(
      k = k,
      best_S = S_i,
      best_value = lambda_i,
      iter = iter
    ))
  })
}

# Find sets from 1 to K
find_misses <- function(
  model,
  K,
  sign = 1L,
  tol = 1e-16,
  max_iter = 1e3,
  start_warm = TRUE
) {
  xr <- get_lm_xr(model)
  solver <- get_miss_solver(xr$X, xr$R, sign, tol, max_iter)
  out <- vector("list", K)
  out[[1L]] <- solver(1L)
  lambda <- if (isTRUE(start_warm)) out[[1L]][["best_value"]] else 0
  out[[1L]][["best_value"]] <- model[["coefficients"]] -
    out[[1L]][["best_value"]] * sign
  if (K == 1L) {
    return(out)
  }
  for (k in seq.int(2L, K)) {
    out[[k]] <- solver(k, lambda = lambda)
    if (isTRUE(start_warm)) {
      lambda <- out[[k]][["best_value"]]
    }
    out[[k]][["best_value"]] <- model[["coefficients"]] -
      out[[k]][["best_value"]] * sign
  }
  return(out)
}

# Find set for size k
find_miss <- function(model, k, sign = 1L, tol = 1e-16, max_iter = 1e3) {
  xr <- get_lm_xr(model)
  solver <- get_miss_solver(xr$X, xr$R, sign, tol, max_iter)
  out <- solver(k)
  out[["best_value"]] <- coef(model) - out[["best_value"]] * sign
  return(out)
}
