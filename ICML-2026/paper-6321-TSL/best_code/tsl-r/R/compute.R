# Public "compute" layer: pure functions that reconstruct TSL's interpretability
# quantities from the fitted components and return tidy data. The plot_*()
# functions build ggplots from these; power users can consume the data directly.
# All maths lives in reconstruct.R; this file only assembles tidy frames.

# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------

#' First-order partial dependence
#'
#' Reconstructs the per-stage first-order partial dependence of the model: for
#' each requested feature and boosting stage, the positive branch `pos`, the
#' (signed-negative) branch `neg`, and their sum `net`, marginalising every other
#' feature over the background data. Because TSL is separable this is exact, not
#' a sampled approximation. Summing `net` over stages gives the total effect.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param X Background design matrix to marginalise over. Defaults to the
#'   training data retained by [tsl()].
#' @param features Features to evaluate, as names or 1-based indices. Default all.
#' @param grid_points Resolution of the per-feature evaluation grid.
#' @param stages Stages to include, as 1-based indices. Default all.
#' @param scale `"raw"` (default) returns prediction-scale branches; `"component"`
#'   divides out each stage's constant to return per-feature m-space shapes.
#' @return A data frame with columns `feature`, `stage`, `x`, `pos`, `neg`,
#'   `net`, `c_plus`, `c_minus`, `backbone`, plus integer `feature_idx`/`stage_idx`.
#' @seealso [plot_first_order_pd()], [pd_difference_plot()], [tsl_pd_2d()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' head(tsl_pd(fit, features = "a"))
#' @export
tsl_pd <- function(object, X = NULL, features = NULL, grid_points = 200L,
                   stages = NULL, scale = c("raw", "component")) {
  .tsl_check_model(object)
  scale <- match.arg(scale)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  feats <- .tsl_resolve_features(object, features)
  stg <- .tsl_resolve_stages(stages, length(comp))
  nm <- .tsl_feature_names(object)
  eps <- 1e-12

  rows <- list()
  for (j in feats) {
    g <- .tsl_feature_grid(X, j, grid_points)
    for (s in stg) {
      sc <- .tsl_stage_scalars(comp[[s]])
      a <- .tsl_bg_means(sc$gt, X, j)
      m <- .tsl_stage_m(sc$gt, j, g)
      c_plus <- sc$eff_plus * a["ap"]
      c_minus <- sc$eff_minus * a["am"]
      if (scale == "raw") {
        pos <- c_plus * m$mp
        neg <- -c_minus * m$mm
      } else {
        pos <- m$mp
        neg <- -m$mm
      }
      bt <- .tsl_stage_bt(sc$gt, j, g)
      rows[[length(rows) + 1L]] <- data.frame(
        feature = nm[j], feature_idx = j,
        stage = paste("Stage", s), stage_idx = s,
        x = g, pos = as.numeric(pos), neg = as.numeric(neg),
        net = as.numeric(pos + neg),
        c_plus = as.numeric(c_plus), c_minus = as.numeric(c_minus),
        backbone = sqrt(pmax(c_plus * c_minus, 0)) * bt$b,
        stringsAsFactors = FALSE
      )
    }
  }
  out <- do.call(rbind, rows)
  out$feature <- factor(out$feature, levels = nm[feats])
  out$stage <- factor(out$stage, levels = paste("Stage", stg))
  rownames(out) <- NULL
  out
}

#' Two-feature partial dependence surface
#'
#' Per-stage signed 2D partial dependence over a pair of features, evaluated on a
#' `grid_points x grid_points` mesh. Returned in long form for `geom_tile()`.
#'
#' @inheritParams tsl_pd
#' @param feature_x,feature_y The two features (names or 1-based indices).
#' @return A data frame with columns `x`, `y`, `stage`, `value` (the signed 2D
#'   PD of that stage). The grid vectors are attached as attributes
#'   `x_vals`/`y_vals`, and the feature names as `feature_x`/`feature_y`.
#' @seealso [plot_2d_pd()], [tsl_backbone_2d()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' d <- tsl_pd_2d(fit, feature_x = "a", feature_y = "b", grid_points = 20L)
#' @export
tsl_pd_2d <- function(object, X = NULL, feature_x, feature_y,
                      grid_points = 50L, stages = NULL) {
  .tsl_check_model(object)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  fx <- .tsl_resolve_features(object, feature_x)
  fy <- .tsl_resolve_features(object, feature_y)
  stg <- .tsl_resolve_stages(stages, length(comp))
  nm <- .tsl_feature_names(object)

  xg <- .tsl_feature_grid(X, fx, grid_points)
  yg <- .tsl_feature_grid(X, fy, grid_points)
  mesh <- expand.grid(x = xg, y = yg)

  rows <- lapply(stg, function(s) {
    sc <- .tsl_stage_scalars(comp[[s]])
    a <- .tsl_bg_means(sc$gt, X, c(fx, fy))
    mx <- .tsl_stage_m(sc$gt, fx, xg)
    my <- .tsl_stage_m(sc$gt, fy, yg)
    z <- sc$eff_plus * a["ap"] * outer(mx$mp, my$mp) -
      sc$eff_minus * a["am"] * outer(mx$mm, my$mm)
    cbind(mesh, value = as.vector(z), stage = paste("Stage", s))
  })
  out <- do.call(rbind, rows)
  out$stage <- factor(out$stage, levels = paste("Stage", stg))
  attr(out, "x_vals") <- xg
  attr(out, "y_vals") <- yg
  attr(out, "feature_x") <- nm[fx]
  attr(out, "feature_y") <- nm[fy]
  out
}

#' Two-feature backbone product and 2D partial dependence
#'
#' Per-stage backbone product `b_x(x) * b_y(y)` (unsigned magnitude) and the
#' signed 2D partial dependence, stacked in one long frame with a `panel` factor.
#'
#' @inheritParams tsl_pd_2d
#' @return A data frame with columns `x`, `y`, `stage`, `panel`
#'   (`"backbone product"` / `"2D partial dependence"`), `value`. Grid vectors
#'   and feature names attached as attributes.
#' @seealso [plot_2d_backbone()]
#' @export
tsl_backbone_2d <- function(object, X = NULL, feature_x, feature_y,
                            grid_points = 100L, stages = NULL) {
  pd <- tsl_pd_2d(object, X, feature_x, feature_y, grid_points, stages)
  .tsl_check_model(object)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  fx <- .tsl_resolve_features(object, feature_x)
  fy <- .tsl_resolve_features(object, feature_y)
  stg <- .tsl_resolve_stages(stages, length(comp))
  xg <- attr(pd, "x_vals")
  yg <- attr(pd, "y_vals")
  mesh <- expand.grid(x = xg, y = yg)

  bb <- do.call(rbind, lapply(stg, function(s) {
    gt <- comp[[s]]$combined_grid_tensor
    bx <- .tsl_stage_bt(gt, fx, xg)$b
    by <- .tsl_stage_bt(gt, fy, yg)$b
    cbind(mesh, value = as.vector(outer(bx, by)), stage = paste("Stage", s))
  }))
  bb$panel <- "backbone product"
  pd2 <- data.frame(x = pd$x, y = pd$y, value = pd$value, stage = pd$stage,
                    panel = "2D partial dependence")
  out <- rbind(
    data.frame(x = bb$x, y = bb$y, stage = bb$stage, panel = bb$panel,
               value = bb$value),
    data.frame(x = pd2$x, y = pd2$y, stage = pd2$stage, panel = pd2$panel,
               value = pd2$value)
  )
  out$stage <- factor(out$stage, levels = paste("Stage", stg))
  out$panel <- factor(out$panel,
                      levels = c("backbone product", "2D partial dependence"))
  attr(out, "x_vals") <- xg
  attr(out, "y_vals") <- yg
  attr(out, "feature_x") <- attr(pd, "feature_x")
  attr(out, "feature_y") <- attr(pd, "feature_y")
  out
}

#' Individual conditional expectation (ICE) curves
#'
#' ICE curves for one feature: each sampled observation's prediction as the
#' feature is swept over its range, holding the observation's other features
#' fixed, plus the partial-dependence mean.
#'
#' @inheritParams tsl_pd
#' @param feature The feature to vary (name or 1-based index).
#' @param n_ice Number of observations sampled for the ICE lines.
#' @param seed Seed for the observation sample.
#' @return A list with `ice` (data frame `ice_id`, `x`, `y`) and `pd`
#'   (data frame `x`, `y`).
#' @seealso [plot_ice()]
#' @export
tsl_ice <- function(object, X = NULL, feature, n_ice = 50L, grid_points = 100L,
                    seed = 0L) {
  .tsl_check_model(object)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  feat <- .tsl_resolve_features(object, feature)
  set.seed(seed)
  obs <- sort(sample(nrow(X), min(n_ice, nrow(X))))
  Xo <- X[obs, , drop = FALSE]
  g <- .tsl_feature_grid(X, feat, grid_points)

  ice <- matrix(0, length(obs), length(g))
  for (s in seq_along(comp)) {
    sc <- .tsl_stage_scalars(comp[[s]])
    # product over all features except `feat`, per sampled observation
    pp <- rep(sc$gt$lambda_plus, length(obs))
    pm <- rep(sc$gt$lambda_minus, length(obs))
    for (k in seq_len(ncol(Xo))) {
      if (k == feat) next
      m <- .tsl_stage_m(sc$gt, k, Xo[, k])
      pp <- pp * m$mp
      pm <- pm * m$mm
    }
    mg <- .tsl_stage_m(sc$gt, feat, g)
    ice <- ice + sc$sp * outer(pp, mg$mp) - sc$sm * outer(pm, mg$mm)
  }

  ice_df <- data.frame(
    ice_id = rep(seq_along(obs), times = length(g)),
    x = rep(g, each = length(obs)),
    y = as.vector(ice)
  )
  pd_df <- data.frame(x = g, y = colMeans(ice))
  list(ice = ice_df, pd = pd_df)
}

# ---------------------------------------------------------------------------
# Tilt
# ---------------------------------------------------------------------------

#' Per-feature tilt curves
#'
#' The piecewise-constant tilt `d_j(x)` (signed direction) for each requested
#' feature and stage.
#'
#' @inheritParams tsl_pd
#' @return A data frame with columns `feature`, `stage`, `x`, `d`.
#' @seealso [plot_tilt_1d()], [tsl_tilt_diagnostics()]
#' @export
tsl_tilt <- function(object, X = NULL, features = NULL, grid_points = 200L,
                     stages = NULL) {
  .tsl_check_model(object)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  feats <- .tsl_resolve_features(object, features)
  stg <- .tsl_resolve_stages(stages, length(comp))
  nm <- .tsl_feature_names(object)

  rows <- list()
  for (j in feats) {
    g <- .tsl_feature_grid(X, j, grid_points)
    for (s in stg) {
      d <- .tsl_stage_bt(comp[[s]]$combined_grid_tensor, j, g)$d
      rows[[length(rows) + 1L]] <- data.frame(
        feature = nm[j], stage = paste("Stage", s), x = g, d = d,
        stringsAsFactors = FALSE
      )
    }
  }
  out <- do.call(rbind, rows)
  out$feature <- factor(out$feature, levels = nm[feats])
  out$stage <- factor(out$stage, levels = paste("Stage", stg))
  rownames(out) <- NULL
  out
}

#' Two-feature tilt product surface
#'
#' Per-stage tilt product `d_x(x) * d_y(y)` on a mesh, in long form.
#'
#' @inheritParams tsl_pd_2d
#' @param grid_points Mesh resolution per axis.
#' @return A data frame with columns `x`, `y`, `stage`, `value`; grid vectors and
#'   feature names attached as attributes.
#' @seealso [plot_2d_tilt()]
#' @export
tsl_tilt_2d <- function(object, X = NULL, feature_x, feature_y,
                        grid_points = 100L, stages = NULL) {
  .tsl_check_model(object)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  fx <- .tsl_resolve_features(object, feature_x)
  fy <- .tsl_resolve_features(object, feature_y)
  stg <- .tsl_resolve_stages(stages, length(comp))
  nm <- .tsl_feature_names(object)

  xg <- .tsl_feature_grid(X, fx, grid_points)
  yg <- .tsl_feature_grid(X, fy, grid_points)
  mesh <- expand.grid(x = xg, y = yg)
  rows <- lapply(stg, function(s) {
    gt <- comp[[s]]$combined_grid_tensor
    dx <- .tsl_stage_bt(gt, fx, xg)$d
    dy <- .tsl_stage_bt(gt, fy, yg)$d
    cbind(mesh, value = as.vector(outer(dx, dy)), stage = paste("Stage", s))
  })
  out <- do.call(rbind, rows)
  out$stage <- factor(out$stage, levels = paste("Stage", stg))
  attr(out, "x_vals") <- xg
  attr(out, "y_vals") <- yg
  attr(out, "feature_x") <- nm[fx]
  attr(out, "feature_y") <- nm[fy]
  out
}

#' Tilt diagnostic curves
#'
#' Four diagnostic curves per feature and stage: `tanh(d)`, `B * tanh(d)`,
#' `tanh(d - mean d)`, and `B * tanh(d - mean d)`, where `B` is the backbone and
#' `d` the tilt, read directly from the stage's combined grid tensor.
#'
#' @inheritParams tsl_pd
#' @return A data frame with columns `feature`, `stage`, `x`, `curve`, `value`.
#' @seealso [plot_tilt_diagnostics()]
#' @export
tsl_tilt_diagnostics <- function(object, X = NULL, features = NULL,
                                 grid_points = 200L, stages = NULL) {
  .tsl_check_model(object)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  feats <- .tsl_resolve_features(object, features)
  stg <- .tsl_resolve_stages(stages, length(comp))
  nm <- .tsl_feature_names(object)
  curve_levels <- c("tanh(d)", "B*tanh(d)", "tanh(d-mean)", "B*tanh(d-mean)")

  rows <- list()
  for (j in feats) {
    g <- .tsl_feature_grid(X, j, grid_points)
    for (s in stg) {
      bt <- .tsl_stage_bt(comp[[s]]$combined_grid_tensor, j, g)
      bb <- bt$b
      d <- bt$d
      dc <- d - mean(d)
      vals <- list("tanh(d)" = tanh(d), "B*tanh(d)" = bb * tanh(d),
                   "tanh(d-mean)" = tanh(dc), "B*tanh(d-mean)" = bb * tanh(dc))
      for (cv in curve_levels) {
        rows[[length(rows) + 1L]] <- data.frame(
          feature = nm[j], stage = paste("Stage", s), x = g,
          curve = cv, value = vals[[cv]], stringsAsFactors = FALSE
        )
      }
    }
  }
  out <- do.call(rbind, rows)
  out$feature <- factor(out$feature, levels = nm[feats])
  out$stage <- factor(out$stage, levels = paste("Stage", stg))
  out$curve <- factor(out$curve, levels = curve_levels)
  rownames(out) <- NULL
  out
}

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

#' Feature importance from the fitted components
#'
#' Per-stage and aggregated feature importance. Per-stage backbone importance is
#' the variance over the data of `log b_j(x_j)` (how strongly feature `j` gates);
#' tilt importance is the variance of `d_j(x_j)` (how strongly it steers). Stages
#' are weighted by their share of prediction energy; the combined score is
#' `backbone + gamma * tilt`.
#'
#' @inheritParams tsl_pd
#' @param gamma Weight on tilt in the combined score.
#' @return A list with `per_stage` (data frame `feature`, `stage`, `backbone`,
#'   `tilt`), `global` (data frame `feature`, `backbone`, `tilt`, `combined`),
#'   `stage_weights` (data frame `stage`, `weight`), and `gamma`.
#' @seealso [plot_feature_importance()]
#' @export
tsl_importance <- function(object, X = NULL, gamma = 1) {
  .tsl_check_model(object)
  X <- .tsl_background(object, X)
  comp <- tsl_components(object)
  n_stage <- length(comp)
  p <- object$n_features
  nm <- .tsl_feature_names(object)
  var_pop <- function(v) mean((v - mean(v))^2)

  bb <- matrix(0, n_stage, p)
  tt <- matrix(0, n_stage, p)
  energy <- numeric(n_stage)
  for (s in seq_len(n_stage)) {
    sc <- .tsl_stage_scalars(comp[[s]])
    for (j in seq_len(p)) {
      bt <- .tsl_stage_bt(sc$gt, j, X[, j])
      bb[s, j] <- var_pop(log(pmax(bt$b, 1e-12)))
      tt[s, j] <- var_pop(bt$d)
    }
    f <- .tsl_stage_fpm(sc$gt, X)
    energy[s] <- mean((sc$sp * f$fp - sc$sm * f$fm)^2)
  }
  weight <- if (sum(energy) > 0) energy / sum(energy) else rep(1 / n_stage, n_stage)
  global_bb <- as.numeric(crossprod(weight, bb))
  global_tt <- as.numeric(crossprod(weight, tt))

  per_stage <- data.frame(
    feature = factor(rep(nm, each = n_stage), levels = nm),
    stage = factor(rep(paste("Stage", seq_len(n_stage)), times = p),
                   levels = paste("Stage", seq_len(n_stage))),
    backbone = as.vector(t(bb)), tilt = as.vector(t(tt))
  )
  global <- data.frame(
    feature = factor(nm, levels = nm),
    backbone = global_bb, tilt = global_tt,
    combined = global_bb + gamma * global_tt
  )
  stage_weights <- data.frame(
    stage = factor(paste("Stage", seq_len(n_stage)),
                   levels = paste("Stage", seq_len(n_stage))),
    weight = weight
  )
  list(per_stage = per_stage, global = global, stage_weights = stage_weights,
       gamma = gamma)
}

# ---------------------------------------------------------------------------
# Local explanation
# ---------------------------------------------------------------------------

#' Local explanation of a single prediction
#'
#' Decomposes the prediction for one point into per-stage positive and negative
#' branch contributions (summing to the prediction), per-feature backbone and
#' tilt values, and the stage intercepts that absorb the OLS scalings.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param x A numeric vector of length `n_features` (or a one-row matrix).
#' @return A list with `stage_contributions`, `f_plus_contributions`,
#'   `f_minus_contributions`, `backbone_magnitudes`, `tilt_sums`,
#'   `feature_backbone` and `feature_tilt` (matrices, stages x features),
#'   `intercept_backbone`, `intercept_tilt`, `total_prediction`, a tidy `stages`
#'   data frame (`stage`, `fpos`, `fneg`, `net`), and `feature_names`.
#' @seealso [plot_local_interpretation()]
#' @export
tsl_local <- function(object, x) {
  .tsl_check_model(object)
  x <- as.numeric(x)
  if (length(x) != object$n_features) {
    stop("`x` must have length ", object$n_features, " (one value per feature).",
         call. = FALSE)
  }
  comp <- tsl_components(object)
  n_stage <- length(comp)
  p <- object$n_features
  nm <- .tsl_feature_names(object)
  xm <- matrix(x, nrow = 1)
  eps <- 1e-12

  stage_contrib <- fpos <- fneg <- bbmag <- tiltsum <- numeric(n_stage)
  feat_bb <- feat_tilt <- matrix(0, n_stage, p)
  icpt_bb <- icpt_tilt <- numeric(n_stage)
  for (s in seq_len(n_stage)) {
    sc <- .tsl_stage_scalars(comp[[s]])
    f <- .tsl_stage_fpm(sc$gt, xm)
    fpos[s] <- sc$sp * f$fp
    fneg[s] <- -sc$sm * f$fm
    stage_contrib[s] <- fpos[s] + fneg[s]
    bprod <- 1
    dsum <- 0
    for (j in seq_len(p)) {
      bt <- .tsl_stage_bt(sc$gt, j, x[j])
      feat_bb[s, j] <- bt$b
      feat_tilt[s, j] <- bt$d
      bprod <- bprod * bt$b
      dsum <- dsum + bt$d
    }
    bbmag[s] <- bprod
    tiltsum[s] <- dsum
    icpt_bb[s] <- sqrt(max(sc$eff_plus * sc$eff_minus, 0))
    icpt_tilt[s] <- 0.5 * log(max(sc$eff_plus, eps) / max(sc$eff_minus, eps))
  }

  stages <- data.frame(
    stage = factor(paste("Stage", seq_len(n_stage)),
                   levels = paste("Stage", seq_len(n_stage))),
    fpos = fpos, fneg = fneg, net = stage_contrib
  )
  list(
    stage_contributions = stage_contrib,
    f_plus_contributions = fpos, f_minus_contributions = fneg,
    backbone_magnitudes = bbmag, tilt_sums = tiltsum,
    feature_backbone = feat_bb, feature_tilt = feat_tilt,
    intercept_backbone = icpt_bb, intercept_tilt = icpt_tilt,
    total_prediction = sum(stage_contrib),
    stages = stages, feature_names = nm
  )
}

# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------

#' Recover the data behind a tensorsl plot
#'
#' The `plot_*()` functions attach the data frame they were built from as an
#' attribute; this returns it so the plot can be rebuilt or extended.
#'
#' @param p A ggplot returned by a `tensorsl` `plot_*()` function.
#' @return The attached data (a data frame or list), or `NULL` if absent.
#' @export
tsl_plot_data <- function(p) attr(p, "tsl_data", exact = TRUE)
