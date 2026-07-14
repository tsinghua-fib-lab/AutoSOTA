# Greedy approximation for finding influential sets
greedy_miss <- function(model, k, sign = 1L) {
  if (!requireNamespace("influence", quietly = TRUE)) {
    stop("Package 'influence' is needed for the greedy algorithm.")
  }
  sens <- influence:::sensitivity_lm(
    model,
    lambda = influence::set_lambda("beta_i", position = 1L, sign = sign),
    options = influence::set_options(n_max = k - 1L),
    verbose = FALSE
  )
  list(
    k = k,
    best_S = sens[["influence"]][["id"]],
    best_value = sens[["influence"]][["lambda"]] * sign,
    initial_S = sens[["initial"]][["id"]],
    initial_value = sens[["initial"]][["lambda"]] * sign
  )
}

# Find greedy sets from 1 to K
greedy_misses <- function(model, K, sign = 1L) {
  greedy_miss(model, K, sign)
}
