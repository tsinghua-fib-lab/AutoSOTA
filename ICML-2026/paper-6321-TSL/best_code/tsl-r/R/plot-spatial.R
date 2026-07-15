# Two-feature backbone evolution: the unsigned backbone product b_x * b_y and
# the signed 2D partial dependence, side by side per stage. The two quantities
# need different colour ramps (sequential indigo for magnitude, diverging
# blue-orange for the signed surface), so each is built as its own ggplot and
# the two are composed vertically with patchwork.

#' Two-feature backbone evolution
#'
#' Composes two stacked panels of per-stage heatmaps over a pair of features:
#' the unsigned backbone product `b_x(x) * b_y(y)` on a sequential indigo ramp,
#' and the signed 2D partial dependence on a diverging blue-orange ramp anchored
#' at zero. Each row of panels iterates the stages. The backbone panels are
#' per-stage rescaled to `[0, 1]` (2nd to 98th percentile); the partial
#' dependence panels are per-stage rescaled symmetrically to `[-1, 1]`.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param X Background design matrix to evaluate over. Defaults to the training
#'   data retained by [tsl()].
#' @param feature_x,feature_y The two features (names or 1-based indices).
#' @param grid_points Mesh resolution per axis.
#' @param stages Stages to include, as 1-based indices. Default all.
#' @return A patchwork object (or a list of ggplots if patchwork is not
#'   installed).
#' @seealso [tsl_backbone_2d()], [plot_2d_tilt()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_2d_backbone(fit, x, feature_x = "a", feature_y = "b", grid_points = 40L)
#' @export
plot_2d_backbone <- function(object, X = NULL, feature_x, feature_y,
                             grid_points = 100L, stages = NULL) {
  df <- tsl_backbone_2d(object, X, feature_x, feature_y, grid_points, stages)
  fx_lab <- attr(df, "feature_x")
  fy_lab <- attr(df, "feature_y")

  bb <- df[df$panel == "backbone product", ]
  bb$z <- ave(bb$value, bb$stage, FUN = function(v) {
    lo <- stats::quantile(v, 0.02)
    hi <- stats::quantile(v, 0.98)
    hi <- if (hi > lo) hi else lo + 1e-9
    scales::squish((v - lo) / (hi - lo), c(0, 1))
  })
  p_bb <- ggplot(bb, aes(x, y, fill = z)) +
    geom_tile() +
    facet_wrap(~stage, nrow = 1) +
    scale_fill_tsl_backbone(name = "backbone") +
    labs(subtitle = "backbone product b_x x b_y", x = fx_lab, y = fy_lab) +
    theme_flat()

  pd <- df[df$panel == "2D partial dependence", ]
  pd$z <- ave(pd$value, pd$stage, FUN = function(v) {
    m <- stats::quantile(abs(v), 0.98)
    m <- if (m > 0) m else 1
    scales::squish(v / m, c(-1, 1))
  })
  p_pd <- ggplot(pd, aes(x, y, fill = z)) +
    geom_tile() +
    facet_wrap(~stage, nrow = 1) +
    scale_fill_tsl_diverging(name = "2D PD", limits = c(-1, 1)) +
    labs(subtitle = "signed 2D partial dependence", x = fx_lab, y = fy_lab) +
    theme_flat()

  if (requireNamespace("patchwork", quietly = TRUE)) {
    p <- patchwork::wrap_plots(p_bb, p_pd, ncol = 1) +
      patchwork::plot_annotation(title = "Two-feature backbone evolution")
    attr(p, "tsl_data") <- df
    return(p)
  }
  message("Install 'patchwork' for the composed figure; ",
          "returning a list of component plots.")
  list(backbone = p_bb, pd = p_pd)
}
