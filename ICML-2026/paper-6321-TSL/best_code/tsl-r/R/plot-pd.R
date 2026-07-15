# ggplot2 partial-dependence and ICE plots for fitted TSL models, in the flat
# aesthetic. Each function reconstructs its data through the compute layer
# (compute.R) and assembles a ggplot using the shared theme and tokens
# (plot-theme.R); the computed data is attached as the "tsl_data" attribute so
# tsl_plot_data() can recover it.

# Sample up to `n` rows of feature j from the background matrix, as a tidy
# `feature`, `x` frame, for the bottom rug in the density overlays.
.tsl_density_rug <- function(object, X, feats, n = 1500L) {
  nm <- .tsl_feature_names(object)
  rows <- if (nrow(X) > n) sort(sample(nrow(X), n)) else seq_len(nrow(X))
  do.call(rbind, lapply(feats, function(j) {
    data.frame(feature = factor(nm[j], levels = nm[feats]),
               x = X[rows, j], stringsAsFactors = FALSE)
  }))
}

#' Plot first-order partial dependence
#'
#' Draws the per-stage, per-feature first-order partial dependence as a faceted
#' grid: the positive branch as the curve `PD+` (orange) and the negative branch
#' as the curve `PD-` (blue), both on the positive scale, with the gap between
#' them shaded -- orange where `PD+ >= PD-`, blue elsewhere. That signed gap is
#' the net effect `PD+ - PD-`. Because TSL is separable this decomposition is
#' exact, not a sampled approximation.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param X Background design matrix to marginalise over. Defaults to the
#'   training data retained by [tsl()].
#' @param features Features to plot, as names or 1-based indices. Default all.
#' @param grid_points Resolution of the per-feature evaluation grid.
#' @param stages Stages to include, as 1-based indices. Default all.
#' @param scale `"raw"` (default) plots prediction-scale branches; `"component"`
#'   divides out each stage's constant to show per-feature m-space shapes.
#' @param show_backbone_overlay If `TRUE` (default), overlay the
#'   `sqrt(C+ C-) * b` backbone as a dotted line per panel.
#' @param show_data_density If `TRUE`, add a bottom rug from a sample of the
#'   background rows showing the marginal data distribution per feature.
#' @return A ggplot object. The data frame from [tsl_pd()] is attached as the
#'   `"tsl_data"` attribute (see [tsl_plot_data()]).
#' @seealso [tsl_pd()], [pd_difference_plot()], [plot_2d_pd()], [plot_ice()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_first_order_pd(fit, x, features = c("a", "b"))
#' @export
plot_first_order_pd <- function(object, X = NULL, features = NULL,
                                grid_points = 200L, stages = NULL,
                                scale = c("raw", "component"),
                                show_backbone_overlay = TRUE,
                                show_data_density = FALSE) {
  scale <- match.arg(scale)
  df <- tsl_pd(object, X, features, grid_points, stages, scale)

  p <- ggplot(df) +
    facet_grid(stage ~ feature, scales = "free") +
    .tsl_zero_ref() +
    .tsl_signed_diff_fill() +
    .tsl_branch_curves(show_backbone_overlay) +
    labs(
      title = "First-order partial dependence",
      subtitle = expression(
        "gap = net effect (PD+ - PD-); dotted backbone:" ~
          sqrt(C^"+" ~ C^"-") %.% italic(b)
      ),
      x = NULL, y = "PD"
    ) +
    theme_flat()

  if (isTRUE(show_data_density)) {
    bg <- .tsl_background(object, X)
    feats <- .tsl_resolve_features(object, features)
    rug <- .tsl_density_rug(object, bg, feats)
    p <- p + geom_rug(data = rug, aes(x), sides = "b",
                      colour = .tsl_tokens$muted, alpha = 0.12,
                      inherit.aes = FALSE)
  }

  attr(p, "tsl_data") <- df
  p
}

#' Plot the partial-dependence difference
#'
#' Same faceted grid as [plot_first_order_pd()]: the positive branch as the curve
#' `PD+` (orange) and the negative branch as `PD-` (blue), with the signed gap
#' between them shaded -- that gap is the stage's net contribution `PD+ - PD-`.
#' The per-feature backbone is optionally overlaid as a dotted line.
#'
#' @inheritParams plot_first_order_pd
#' @param show_backbone_overlay If `TRUE` (default), overlay the
#'   `sqrt(C+ C-) * b` backbone as a dotted line per panel.
#' @return A ggplot object. The data frame from [tsl_pd()] is attached as the
#'   `"tsl_data"` attribute (see [tsl_plot_data()]).
#' @seealso [tsl_pd()], [plot_first_order_pd()], [plot_2d_pd()], [plot_ice()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' pd_difference_plot(fit, x, features = "a")
#' @export
pd_difference_plot <- function(object, X = NULL, features = NULL,
                               grid_points = 200L, stages = NULL,
                               show_backbone_overlay = TRUE,
                               show_data_density = FALSE) {
  df <- tsl_pd(object, X, features, grid_points, stages, "raw")

  p <- ggplot(df) +
    facet_grid(stage ~ feature, scales = "free") +
    .tsl_zero_ref() +
    .tsl_signed_diff_fill() +
    .tsl_branch_curves(show_backbone_overlay) +
    labs(
      title = "Partial-dependence difference",
      subtitle = "the shaded gap between the curves is the signed contribution",
      x = NULL, y = "PD"
    ) +
    theme_flat()

  if (isTRUE(show_data_density)) {
    bg <- .tsl_background(object, X)
    feats <- .tsl_resolve_features(object, features)
    rug <- .tsl_density_rug(object, bg, feats)
    p <- p + geom_rug(data = rug, aes(x), sides = "b",
                      colour = .tsl_tokens$muted, alpha = 0.12,
                      inherit.aes = FALSE)
  }

  attr(p, "tsl_data") <- df
  p
}

#' Plot a two-feature partial-dependence surface
#'
#' Per-stage signed 2D partial dependence over a pair of features, drawn as a
#' tiled heatmap with the blue-orange diverging fill. Each panel is rescaled to
#' its own 98th-percentile range so weaker stages remain legible.
#'
#' @inheritParams plot_first_order_pd
#' @param feature_x,feature_y The two features (names or 1-based indices).
#' @return A ggplot object. The data frame from [tsl_pd_2d()] is attached as the
#'   `"tsl_data"` attribute (see [tsl_plot_data()]).
#' @seealso [tsl_pd_2d()], [plot_first_order_pd()], [pd_difference_plot()],
#'   [plot_ice()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_2d_pd(fit, x, feature_x = "a", feature_y = "b", grid_points = 30L)
#' @export
plot_2d_pd <- function(object, X = NULL, feature_x, feature_y,
                       grid_points = 50L, stages = NULL) {
  df <- tsl_pd_2d(object, X, feature_x, feature_y, grid_points, stages)
  df$z <- ave(df$value, df$stage, FUN = function(v) {
    m <- stats::quantile(abs(v), 0.98)
    m <- if (m > 0) m else 1
    scales::squish(v / m, c(-1, 1))
  })

  p <- ggplot(df) +
    geom_tile(aes(x, y, fill = z)) +
    facet_wrap(~stage) +
    scale_fill_tsl_diverging(name = "2D PD", limits = c(-1, 1)) +
    labs(
      title = "Two-feature partial dependence",
      subtitle = "signed effect (each panel scaled to its own range)",
      x = attr(df, "feature_x"), y = attr(df, "feature_y")
    ) +
    theme_flat()

  attr(p, "tsl_data") <- df
  p
}

#' Plot individual conditional expectation (ICE) curves
#'
#' ICE curves for one feature: each sampled observation's prediction (faint
#' indigo) as the feature is swept over its range, with the partial-dependence
#' mean overlaid as a bold dark line.
#'
#' @inheritParams plot_first_order_pd
#' @param feature The feature to vary (name or 1-based index).
#' @param n_ice Number of observations sampled for the ICE lines.
#' @param seed Seed for the observation sample.
#' @return A ggplot object. The list from [tsl_ice()] is attached as the
#'   `"tsl_data"` attribute (see [tsl_plot_data()]).
#' @seealso [tsl_ice()], [plot_first_order_pd()], [pd_difference_plot()],
#'   [plot_2d_pd()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_ice(fit, x, feature = "a", n_ice = 20L)
#' @export
plot_ice <- function(object, X = NULL, feature, n_ice = 50L,
                     grid_points = 100L, seed = 0L) {
  d <- tsl_ice(object, X, feature, n_ice, grid_points, seed)
  ice <- d$ice
  pd <- d$pd
  feat_name <- .tsl_feature_names(object)[.tsl_resolve_features(object, feature)]

  p <- ggplot() +
    .tsl_zero_ref() +
    geom_line(data = ice, aes(x, y, group = ice_id),
              colour = .tsl_tokens$accent, alpha = 0.15, linewidth = 0.4) +
    geom_line(data = pd, aes(x, y), colour = .tsl_tokens$ink, linewidth = 1.2) +
    labs(title = "Individual conditional expectation",
         x = feat_name, y = "prediction") +
    theme_flat()

  attr(p, "tsl_data") <- d
  p
}
