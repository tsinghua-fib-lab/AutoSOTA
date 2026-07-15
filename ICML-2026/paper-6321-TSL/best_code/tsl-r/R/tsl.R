#' Fit a Tensor Separation Learning model
#'
#' Fits a boosted TSL regression model. TSL is a glass-box model: the fitted
#' model is a sum of stages, each an ordered difference of two non-negative
#' rank-1 products of univariate functions.
#'
#' The hyperparameters mirror the Python `TSLRegressor`, so a model fit in R
#' with the same data and `seed` reproduces the Python results.
#'
#' @param x A numeric matrix (or object coercible via [as.matrix()]) of
#'   predictors with rows = observations and columns = features.
#' @param y A numeric response vector with one entry per row of `x`.
#' @param epochs Number of boosting rounds.
#' @param n_trees Number of trees (bagged grid tensors) per stage.
#' @param n_iter Number of grid-tensor refinement iterations.
#' @param decay Learning-rate decay applied after the first epoch.
#' @param split_try Number of random split candidates (used by the
#'   `"random"` split strategy).
#' @param colsample_bytree Fraction of features sampled per tree.
#' @param alpha Refinement regularisation strength.
#' @param complexity_penalty Per-split complexity penalty.
#' @param min_split_loss Minimum loss reduction required to make a split.
#' @param min_interval_samples Minimum number of samples per interval.
#' @param refinement_strategy Refinement loss, one of `"l2"` or `"huber"`.
#' @param prior_sample_size Prior pseudo-count for refinement shrinkage.
#' @param update_clamp Maximum magnitude of a refinement update.
#' @param tilt_tau,tilt_rho Tilt regularisation parameters.
#' @param split_strategy Split search, one of `"random"`, `"best_split"`,
#'   or `"top_k"`.
#' @param top_k Number of candidate splits kept by the `"top_k"` strategy.
#' @param must_fill_all_k Whether the `"top_k"` strategy must fill all `k`
#'   candidates.
#' @param similarity_threshold Threshold controlling bagged aggregation of
#'   trees within a stage.
#' @param bagged Accepted for parity with the Python API; has no effect.
#' @param seed Random seed.
#' @param verbosity Logging verbosity: `0` off, `1` info, `2` debug, `3` trace.
#'
#' @return An object of class `"tsl"`: a list with the fitted model pointer
#'   (`ptr`), training error (`err`), `residuals`, fitted values (`y_hat`),
#'   the `feature_names`, the matched `call`, and `x_background` -- a copy of the
#'   training design matrix retained so the plotting and interpretability
#'   functions (e.g. [plot_first_order_pd()]) can marginalise over the data
#'   without it being passed again.
#'
#' @seealso [predict.tsl()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, "a"] - x[, "b"] + 0.5 * x[, "c"] + rnorm(150, sd = 0.1)
#'
#' ## Fit a small boosted TSL model and inspect it
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' print(fit)
#' @export
tsl <- function(x, y,
                epochs = 10L,
                n_trees = 10L,
                n_iter = 10L,
                decay = 1.0,
                split_try = 10L,
                colsample_bytree = 0.8,
                alpha = 0.0,
                complexity_penalty = 0.0,
                min_split_loss = 0.0,
                min_interval_samples = 1L,
                refinement_strategy = c("l2", "huber"),
                prior_sample_size = 0.0,
                update_clamp = Inf,
                tilt_tau = 0.01,
                tilt_rho = 0.0,
                split_strategy = c("random", "best_split", "top_k"),
                top_k = 10L,
                must_fill_all_k = TRUE,
                similarity_threshold = 0.0,
                bagged = FALSE,
                seed = 42L,
                verbosity = 1L) {
  refinement_strategy <- match.arg(refinement_strategy)
  split_strategy <- match.arg(split_strategy)

  x <- as.matrix(x)
  storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (!all(is.finite(x))) {
    stop("`x` must not contain NA, NaN, or infinite values.", call. = FALSE)
  }
  if (!all(is.finite(y))) {
    stop("`y` must not contain NA, NaN, or infinite values.", call. = FALSE)
  }
  if (nrow(x) != length(y)) {
    stop("`x` must have one row per element of `y` (got ", nrow(x),
         " rows and ", length(y), " responses).", call. = FALSE)
  }

  res <- tsl_fit(
    x, y,
    as.integer(epochs), as.double(decay),
    as.integer(n_trees), as.integer(n_iter),
    as.integer(split_try), as.double(colsample_bytree),
    as.double(alpha), as.double(complexity_penalty),
    as.double(min_split_loss), as.integer(min_interval_samples),
    as.character(refinement_strategy), as.double(prior_sample_size),
    as.double(update_clamp), as.double(tilt_tau), as.double(tilt_rho),
    as.character(split_strategy), as.integer(top_k),
    as.logical(must_fill_all_k), as.double(similarity_threshold),
    as.logical(bagged), as.double(seed), as.integer(verbosity)
  )

  structure(
    list(
      ptr = res$model,
      err = res$err,
      residuals = res$residuals,
      y_hat = res$y_hat,
      feature_names = colnames(x),
      n_features = ncol(x),
      n_obs = nrow(x),
      x_background = x,
      call = match.call()
    ),
    class = "tsl"
  )
}

#' Predict from a fitted TSL model
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param newdata A numeric matrix of predictors with the same columns as the
#'   training data.
#' @param ... Ignored.
#'
#' @return A numeric vector of predictions, one per row of `newdata`.
#'
#' @seealso [tsl()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, "a"] - x[, "b"] + 0.5 * x[, "c"] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#'
#' ## Predict on new data
#' newx <- matrix(runif(10 * 3, -2, 2), ncol = 3,
#'                dimnames = list(NULL, c("a", "b", "c")))
#' predict(fit, newx)
#' @export
predict.tsl <- function(object, newdata, ...) {
  x <- as.matrix(newdata)
  storage.mode(x) <- "double"
  if (!all(is.finite(x))) {
    stop("`newdata` must not contain NA, NaN, or infinite values.",
         call. = FALSE)
  }
  if (ncol(x) != object$n_features) {
    stop("`newdata` has ", ncol(x), " columns but the model was fit on ",
         object$n_features, " features.", call. = FALSE)
  }
  tsl_predict(object$ptr, x)
}

#' @export
print.tsl <- function(x, ...) {
  cat("<tsl> Tensor Separation Learning model\n")
  cat("  features:       ", x$n_features, "\n", sep = "")
  cat("  training rows:  ", x$n_obs, "\n", sep = "")
  cat("  training error: ", format(x$err, digits = 6), "\n", sep = "")
  invisible(x)
}

#' Extract the fitted components of a TSL model
#'
#' A fitted TSL model is a sum of boosting stages; each stage aggregates a
#' bag of rank-1 separable components (grid tensors). This returns that
#' structure in two-tensor form so the glass-box pieces can be inspected.
#'
#' Each grid tensor stores, per feature, the interval `splits` and the
#' `backbone_values` (the non-negative magnitude \eqn{b \ge 0}) and
#' `tilt_values` (the signed direction \eqn{d \in \mathbb{R}}) on each
#' interval, together with the branch scalars \eqn{\lambda_+, \lambda_- \ge 0}.
#' The component's prediction is the ordered difference
#' \eqn{\lambda_+ \prod_j b_j e^{d_j} - \lambda_- \prod_j b_j e^{-d_j}}.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#'
#' @return A list with one element per stage. Each stage is a list with:
#'   \describe{
#'     \item{`scaling_plus`, `scaling_minus`}{the stage's OLS coefficients on
#'       the \eqn{+} and \eqn{-} branches (the only place scaling is applied).}
#'     \item{`candidate_indices`}{1-based indices of the bagged trees in the
#'       stage (`1:n_trees`).}
#'     \item{`combined_grid_tensor`}{the aggregated representative component.}
#'     \item{`grid_tensors`}{the bag of per-tree components.}
#'   }
#'   Each grid tensor is itself a list of per-feature `splits`,
#'   `backbone_values`, `tilt_values`, and `observation_counts`, plus the
#'   scalars `lambda_plus`, `lambda_minus`, and the legacy `scaling`.
#'
#' @seealso [tsl()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, "a"] - x[, "b"] + 0.5 * x[, "c"] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#'
#' ## Inspect the glass-box structure: one entry per boosting stage
#' comp <- tsl_components(fit)
#' length(comp)
#' str(comp[[1]], max.level = 1)
#' @export
tsl_components <- function(object) {
  if (!inherits(object, "tsl")) {
    stop("`object` must be a fitted model of class \"tsl\".", call. = FALSE)
  }
  tsl_model_structure(object$ptr)
}
