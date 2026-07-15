# Internal reconstruction engine. Every interpretability quantity tensorsl exposes
# is derived in pure R from the fitted components (tsl_components()), exploiting
# TSL's separability. This file owns the piecewise-interval lookup and the
# per-stage product math so the formula lives in exactly one place.
#
# A stage's combined grid tensor `gt` stores, per feature j: `splits[[j]]`,
# `backbone_values[[j]]` (b >= 0), `tilt_values[[j]]` (d), plus scalars
# `lambda_plus` / `lambda_minus`. The interval for value v on feature j is the
# count of splits <= v (clamped to the last interval) -- matching the Rust
# `partition_point` / numpy `searchsorted(side="right")` lookup.

# Piecewise-constant backbone b(x) and tilt d(x) for feature j at values xv.
.tsl_stage_bt <- function(gt, j, xv) {
  b <- gt$backbone_values[[j]]
  d <- gt$tilt_values[[j]]
  idx <- pmin(findInterval(xv, gt$splits[[j]]), length(b) - 1L) + 1L
  list(b = b[idx], d = d[idx])
}

# Unscaled product branches f+ = lambda+ * prod_j b_j e^{d_j} and
# f- = lambda- * prod_j b_j e^{-d_j}, one value per row of Xm.
.tsl_stage_fpm <- function(gt, Xm) {
  n <- nrow(Xm)
  fp <- rep(gt$lambda_plus, n)
  fm <- rep(gt$lambda_minus, n)
  for (j in seq_len(ncol(Xm))) {
    bt <- .tsl_stage_bt(gt, j, Xm[, j])
    fp <- fp * bt$b * exp(bt$d)
    fm <- fm * bt$b * exp(-bt$d)
  }
  list(fp = fp, fm = fm)
}

# Background means of the product over every feature EXCEPT those in `excl`:
#   ap = mean_rows( prod_{k not in excl} b_k e^{ d_k} )
#   am = mean_rows( prod_{k not in excl} b_k e^{-d_k} )
# These are the per-stage constants that the marginalised features contribute to
# partial dependence (scaling/lambda are applied by the caller).
.tsl_bg_means <- function(gt, X, excl) {
  n <- nrow(X)
  ap <- rep(1, n)
  am <- rep(1, n)
  for (j in seq_len(ncol(X))) {
    if (j %in% excl) next
    bt <- .tsl_stage_bt(gt, j, X[, j])
    ap <- ap * bt$b * exp(bt$d)
    am <- am * bt$b * exp(-bt$d)
  }
  c(ap = mean(ap), am = mean(am))
}

# Per-feature m+ = b e^{d} and m- = b e^{-d} for feature j at xv.
.tsl_stage_m <- function(gt, j, xv) {
  bt <- .tsl_stage_bt(gt, j, xv)
  list(mp = bt$b * exp(bt$d), mm = bt$b * exp(-bt$d))
}

# Effective branch scalars (scaling folded into lambda): used by PD/importance.
.tsl_stage_scalars <- function(stage) {
  gt <- stage$combined_grid_tensor
  sp <- if (is.na(stage$scaling_plus)) 1 else stage$scaling_plus
  sm <- if (is.na(stage$scaling_minus)) 0 else stage$scaling_minus
  list(gt = gt, sp = sp, sm = sm,
       eff_plus = sp * gt$lambda_plus, eff_minus = sm * gt$lambda_minus)
}

# ---- input plumbing -------------------------------------------------------

.tsl_check_model <- function(object) {
  if (!inherits(object, "tsl")) {
    stop("`object` must be a fitted model of class \"tsl\".", call. = FALSE)
  }
}

# Feature display names, falling back to V1.. when the fit had no column names.
.tsl_feature_names <- function(object) {
  nm <- object$feature_names
  if (is.null(nm) || !length(nm)) paste0("V", seq_len(object$n_features)) else nm
}

# Resolve the background design matrix: explicit X if supplied, else the copy
# retained by tsl(). Validates shape and finiteness.
.tsl_background <- function(object, X = NULL) {
  if (is.null(X)) {
    X <- object$x_background
    if (is.null(X)) {
      stop("No background data available. Pass `X` (the training design ",
           "matrix), or refit so the model retains `x_background`.",
           call. = FALSE)
    }
    return(X)
  }
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (!all(is.finite(X))) {
    stop("`X` must not contain NA, NaN, or infinite values.", call. = FALSE)
  }
  if (ncol(X) != object$n_features) {
    stop("`X` has ", ncol(X), " columns but the model was fit on ",
         object$n_features, " features.", call. = FALSE)
  }
  X
}

# Resolve features (names or 1-based indices) to integer indices; default all.
.tsl_resolve_features <- function(object, features = NULL) {
  nm <- .tsl_feature_names(object)
  if (is.null(features)) return(seq_len(object$n_features))
  if (is.character(features)) {
    idx <- match(features, nm)
    if (anyNA(idx)) {
      stop("Unknown feature(s): ",
           paste(features[is.na(idx)], collapse = ", "), call. = FALSE)
    }
    return(idx)
  }
  idx <- as.integer(features)
  if (any(idx < 1L) || any(idx > object$n_features)) {
    stop("Feature index out of range [1, ", object$n_features, "].",
         call. = FALSE)
  }
  idx
}

# Resolve stages (1-based) to integer indices; default all.
.tsl_resolve_stages <- function(stages, n_stage) {
  if (is.null(stages)) return(seq_len(n_stage))
  s <- as.integer(stages)
  if (any(s < 1L) || any(s > n_stage)) {
    stop("Stage index out of range [1, ", n_stage, "].", call. = FALSE)
  }
  s
}

# Uniform evaluation grid over the empirical range of feature j.
.tsl_feature_grid <- function(X, j, grid_points) {
  seq(min(X[, j]), max(X[, j]), length.out = grid_points)
}
