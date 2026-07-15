# R-idiomatic entry point: autoplot.tsl() dispatches to the mirrored plot_*()
# functions by `type`, so users can reach every diagnostic from one verb.

#' Quick diagnostic plot for a fitted TSL model
#'
#' A convenience wrapper dispatching to the `tensorsl` `plot_*()` functions by
#' `type`. Extra arguments are forwarded to the underlying function (e.g.
#' `feature_x`/`feature_y` for the 2D plots, `feature` for `"ice"`, `points`
#' for `"local"`).
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param type Which diagnostic to draw. One of `"pd"`, `"pd_difference"`,
#'   `"pd_2d"`, `"ice"`, `"tilt"`, `"tilt_2d"`, `"tilt_diagnostics"`,
#'   `"backbone_2d"`, `"importance"`, `"local"`, `"components"`.
#' @param ... Passed to the dispatched `plot_*()` function.
#' @return A ggplot object (or a patchwork object for the composite types).
#' @seealso [plot_first_order_pd()], [plot_2d_backbone()],
#'   [plot_feature_importance()], [plot_local_interpretation()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#'
#' ggplot2::autoplot(fit, type = "pd", features = "a")
#' ggplot2::autoplot(fit, type = "tilt_2d", feature_x = "a", feature_y = "b")
#' @importFrom ggplot2 autoplot
#' @exportS3Method ggplot2::autoplot
autoplot.tsl <- function(object,
                         type = c("pd", "pd_difference", "pd_2d", "ice",
                                  "tilt", "tilt_2d", "tilt_diagnostics",
                                  "backbone_2d", "importance", "local",
                                  "components"),
                         ...) {
  type <- match.arg(type)
  fn <- switch(
    type,
    pd               = plot_first_order_pd,
    pd_difference    = pd_difference_plot,
    pd_2d            = plot_2d_pd,
    ice              = plot_ice,
    tilt             = plot_tilt_1d,
    tilt_2d          = plot_2d_tilt,
    tilt_diagnostics = plot_tilt_diagnostics,
    backbone_2d      = plot_2d_backbone,
    importance       = plot_feature_importance,
    local            = plot_local_interpretation,
    components       = plot_combined_grid_tensors
  )
  fn(object, ...)
}
