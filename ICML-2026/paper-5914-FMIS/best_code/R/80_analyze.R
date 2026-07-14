analyze_variable <- function(
  data,
  formula,
  var_name,
  K,
  # FALSE or function(X, y, variable) -> list(X, y)
  residualize = update_fwl,
  greedy = FALSE,
  start_warm = TRUE
) {
  # Build model frame and extract components ---
  mf <- model.frame(formula, data = data)
  y <- model.response(mf)
  X <- model.matrix(formula, data = data)
  n <- nrow(X)

  # Find variable ---
  var_idx <- which(colnames(X) == var_name)
  if (length(var_idx) == 0L) {
    # Try partial match for transformed variables
    var_idx <- grep(var_name, colnames(X), fixed = TRUE)
  }
  if (length(var_idx) == 0L) {
    stop(
      "Variable '",
      var_name,
      "' not found. Available variables: ",
      paste(colnames(X), collapse = ", ")
    )
  }
  var_idx <- var_idx[1L] # First one if there's multiple matches

  # Estimate ---
  if (!isFALSE(residualize) && ncol(X) > 1L) {
    fwl <- residualize(X, y, var_idx)
    y_res <- fwl$y
    x_res <- fwl$X
  } else if (ncol(X) > 1L) {
    # Skip residualization
    y_res <- y
    x_res <- X[, var_idx]
  } else {
    # It's univariate
    y_res <- y
    x_res <- X[, 1L]
  }
  model <- lm(y_res ~ 0 + x_res)

  # Determine K
  K <- min(K, n - 2L) # Leave at least two observations

  # Run MISS analysis --
  miss_pos <- find_misses(
    model,
    K = K,
    sign = 1L,
    start_warm = isTRUE(start_warm)
  )
  sets_pos <- lapply(miss_pos, \(x) x$best_S)
  values_pos <- sapply(miss_pos, \(x) x$best_value)
  breaks_pos <- find_breaks(sets_pos)

  miss_neg <- find_misses(
    model,
    K = K,
    sign = -1L,
    start_warm = isTRUE(start_warm)
  )
  sets_neg <- lapply(miss_neg, \(x) x$best_S)
  values_neg <- sapply(miss_neg, \(x) x$best_value)
  breaks_neg <- find_breaks(sets_neg)

  out <- list(
    var_name = var_name,
    n = n,
    K = K,
    coef = coef(model),
    # Positive direction
    miss_pos = miss_pos,
    miss_neg = miss_neg
  )

  # Greedy comparison ---
  if (isTRUE(greedy)) {
    out[["greedy_pos"]] <- tryCatch(
      {
        greedy_misses(model, K = K, sign = 1L)
      },
      error = function(e) {
        message("Greedy algorithm failed: ", e$message)
        NA
      }
    )
    out[["greedy_neg"]] <- tryCatch(
      {
        greedy_misses(model, K = K, sign = -1L)
      },
      error = function(e) {
        message("Greedy algorithm failed: ", e$message)
        NA
      }
    )
  }

  return(out)
}
