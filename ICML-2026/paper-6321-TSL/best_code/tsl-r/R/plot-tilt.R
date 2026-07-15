# Tilt visualisations for fitted TSL models. The tilt d_j(x) is the signed
# per-feature direction stored alongside the backbone in each stage's combined
# grid tensor. These plotters build flat-aesthetic ggplots from the tidy frames
# returned by tsl_tilt(), tsl_tilt_2d(), and tsl_tilt_diagnostics().

#' Per-feature tilt curves
#'
#' Plots the piecewise-constant tilt `d_j(x)` (the signed direction) for each
#' requested feature and boosting stage as step curves, one panel per
#' `(stage, feature)` cell. The signed fill carries the sign: orange where the
#' tilt is positive, blue where it is negative.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param X Background design matrix to evaluate over. Defaults to the training
#'   data retained by [tsl()].
#' @param features Features to plot, as names or 1-based indices. Default all.
#' @param grid_points Resolution of the per-feature evaluation grid.
#' @param stages Stages to include, as 1-based indices. Default all.
#' @return A ggplot object.
#' @seealso [tsl_tilt()], [plot_2d_tilt()], [plot_tilt_diagnostics()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_tilt_1d(fit, x, features = c("a", "b"))
#' @export
plot_tilt_1d <- function(object, X = NULL, features = NULL, grid_points = 200L,
                         stages = NULL) {
  d <- tsl_tilt(object, X, features, grid_points, stages)
  p <- ggplot(d, aes(x, d)) +
    .tsl_zero_ref() +
    geom_area(aes(x, pmax(d, 0)), fill = .tsl_tokens$pos, alpha = 0.45) +
    geom_area(aes(x, pmin(d, 0)), fill = .tsl_tokens$neg, alpha = 0.45) +
    geom_step(aes(x, d), colour = .tsl_tokens$accent, linewidth = 0.9) +
    facet_grid(stage ~ feature, scales = "free_x") +
    labs(title = "Per-feature tilt curves",
         subtitle = "tilt d_j(x) by stage (signed direction)",
         x = NULL, y = "tilt d") +
    theme_flat()
  attr(p, "tsl_data") <- d
  p
}

#' Two-feature tilt product surface
#'
#' Plots the per-stage tilt product `d_x(x) * d_y(y)` as a diverging heatmap,
#' one panel per stage. Each panel is rescaled symmetrically to its own range
#' (the 98th percentile of `|value|`), so the diverging colour is anchored at
#' zero and comparable in shape across stages even when their magnitudes differ.
#'
#' @inheritParams plot_tilt_1d
#' @param feature_x,feature_y The two features (names or 1-based indices).
#' @param grid_points Mesh resolution per axis.
#' @return A ggplot object.
#' @seealso [tsl_tilt_2d()], [plot_tilt_1d()], [plot_2d_backbone()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_2d_tilt(fit, x, feature_x = "a", feature_y = "b", grid_points = 40L)
#' @export
plot_2d_tilt <- function(object, X = NULL, feature_x, feature_y,
                         grid_points = 100L, stages = NULL) {
  df <- tsl_tilt_2d(object, X, feature_x, feature_y, grid_points, stages)
  df$z <- ave(df$value, df$stage, FUN = function(v) {
    m <- stats::quantile(abs(v), 0.98)
    m <- if (m > 0) m else 1
    scales::squish(v / m, c(-1, 1))
  })
  p <- ggplot(df, aes(x, y, fill = z)) +
    geom_tile() +
    facet_wrap(~stage) +
    scale_fill_tsl_diverging(name = "2D tilt", limits = c(-1, 1)) +
    labs(title = "Two-feature tilt",
         subtitle = "tilt product (each panel scaled to its own range)",
         x = attr(df, "feature_x"), y = attr(df, "feature_y")) +
    theme_flat()
  attr(p, "tsl_data") <- df
  p
}

#' Tilt diagnostic curves
#'
#' Plots four diagnostic curves per feature and stage: `tanh(d)`, `B*tanh(d)`,
#' `tanh(d - mean d)`, and `B*tanh(d - mean d)`, where `B` is the backbone and
#' `d` the tilt. The four curves run across the columns, features down the rows,
#' and one coloured line per stage in each panel. If many feature x stage
#' combinations make the figure busy, subset via `features`/`stages`.
#'
#' @inheritParams plot_tilt_1d
#' @return A ggplot object.
#' @seealso [tsl_tilt_diagnostics()], [plot_tilt_1d()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_tilt_diagnostics(fit, x, features = "a")
#' @export
plot_tilt_diagnostics <- function(object, X = NULL, features = NULL,
                                  grid_points = 200L, stages = NULL) {
  df <- tsl_tilt_diagnostics(object, X, features, grid_points, stages)
  p <- ggplot(df, aes(x, value, colour = stage)) +
    .tsl_zero_ref() +
    geom_line(linewidth = 0.8) +
    facet_grid(feature ~ curve, scales = "free_y") +
    scale_colour_tsl(name = NULL) +
    labs(title = "Tilt diagnostics",
         subtitle = "tanh squashing and backbone weighting, by stage",
         x = NULL, y = NULL) +
    theme_flat()
  attr(p, "tsl_data") <- df
  p
}
