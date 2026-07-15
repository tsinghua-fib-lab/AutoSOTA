# Composite feature-importance dashboard for fitted TSL models, in the flat
# aesthetic. The six component ggplots are built from tsl_importance() and
# composed with patchwork; when patchwork is not installed the components are
# returned as a named list so callers can arrange them themselves.

# Column-scale a vector to [0, 1] by its maximum (guarding a non-positive max).
.tsl_scale01 <- function(v) {
  m <- max(v)
  if (m > 0) v / m else v * 0
}

#' Plot the feature-importance dashboard
#'
#' Composes a six-panel report of per-stage and global feature importance for a
#' fitted TSL model. Per-stage backbone importance is the variance over the data
#' of `log b_j(x_j)` (how strongly feature `j` gates); tilt importance is the
#' variance of `d_j(x_j)` (how strongly it steers). The global bars weight each
#' stage by its share of prediction energy, and the combined score is
#' `backbone + gamma * tilt`.
#'
#' The panels are: a per-stage backbone heatmap and a per-stage tilt heatmap
#' (each cell column-scaled to its peak); a stage-weight histogram; and global
#' bars for tilt, combined, and backbone importance.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param X Background design matrix to estimate the variances over. Defaults to
#'   the training data retained by [tsl()].
#' @param gamma Weight on tilt in the combined score.
#' @return A patchwork object (or a list of ggplots if patchwork is not
#'   installed). The list from [tsl_importance()] is attached as the
#'   `"tsl_data"` attribute (see [tsl_plot_data()]).
#' @seealso [tsl_importance()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_feature_importance(fit, x)
#' @export
plot_feature_importance <- function(object, X = NULL, gamma = 1) {
  .tsl_check_model(object)
  imp <- tsl_importance(object, X, gamma)

  # Largest combined importance at the top: ggplot's y axis runs bottom-up, so
  # the heatmap feature factor is releveled by ascending combined importance.
  ord <- order(imp$global$combined)
  feat_levels <- as.character(imp$global$feature)[ord]
  ps <- imp$per_stage
  ps$feature <- factor(as.character(ps$feature), levels = feat_levels)
  ps$bscaled <- ave(ps$backbone, ps$stage, FUN = .tsl_scale01)
  ps$tscaled <- ave(ps$tilt, ps$stage, FUN = .tsl_scale01)
  ps$bb_txt <- .tsl_ramp_text(ps$bscaled, .tsl_ramp_backbone)
  ps$tilt_txt <- .tsl_ramp_text(ps$tscaled, .tsl_ramp_tilt)

  bb_heat <- ggplot(ps) +
    geom_tile(aes(stage, feature, fill = bscaled),
              colour = "white", linewidth = 0.5) +
    geom_text(aes(stage, feature, label = sprintf("%.2f", bscaled)),
              colour = ps$bb_txt, family = "mono", size = 2.6) +
    scale_fill_tsl_backbone(name = "rel.") +
    scale_x_discrete(labels = function(l) sub("^Stage ", "S", l)) +
    labs(title = "Backbone importance",
         subtitle = "var of log-backbone (scaled)", x = NULL, y = NULL) +
    theme_flat() +
    theme(panel.grid = element_blank())

  tilt_heat <- ggplot(ps) +
    geom_tile(aes(stage, feature, fill = tscaled),
              colour = "white", linewidth = 0.5) +
    geom_text(aes(stage, feature, label = sprintf("%.2f", tscaled)),
              colour = ps$tilt_txt, family = "mono", size = 2.6) +
    scale_fill_tsl_tilt(name = "rel.") +
    scale_x_discrete(labels = function(l) sub("^Stage ", "S", l)) +
    labs(title = "Tilt importance",
         subtitle = "var of tilt (scaled)", x = NULL, y = NULL) +
    theme_flat() +
    theme(panel.grid = element_blank())

  stage_w <- ggplot(imp$stage_weights) +
    geom_col(aes(stage, weight), fill = .tsl_tokens$greys[3], width = 0.7) +
    scale_x_discrete(labels = function(l) sub("^Stage ", "S", l)) +
    labs(title = "Stage weights", subtitle = "share of prediction energy",
         x = NULL, y = NULL) +
    theme_flat()

  tilt_bar <- ggplot(imp$global) +
    geom_col(aes(tilt, reorder(feature, tilt)),
             fill = .tsl_tokens$pos, width = 0.68) +
    labs(title = "Tilt, global", x = "importance", y = NULL) +
    theme_flat() +
    theme(panel.grid.major.y = element_blank())

  combined_bar <- ggplot(imp$global) +
    geom_col(aes(combined, reorder(feature, combined)),
             fill = .tsl_tokens$neg, width = 0.68) +
    labs(title = "Combined importance", x = "importance", y = NULL) +
    theme_flat() +
    theme(panel.grid.major.y = element_blank())

  backbone_bar <- ggplot(imp$global) +
    geom_col(aes(backbone, reorder(feature, backbone)),
             fill = .tsl_tokens$accent, width = 0.68) +
    labs(title = "Backbone, global", x = "importance", y = NULL) +
    theme_flat() +
    theme(panel.grid.major.y = element_blank())

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("Install 'patchwork' for the composed dashboard; ",
            "returning a list of component plots.")
    return(list(backbone_heat = bb_heat, tilt_heat = tilt_heat,
                stage_weights = stage_w, tilt_bar = tilt_bar,
                combined_bar = combined_bar, backbone_bar = backbone_bar))
  }

  p <- patchwork::wrap_plots(bb_heat, tilt_heat, stage_w,
                             tilt_bar, combined_bar, backbone_bar, nrow = 2) +
    patchwork::plot_annotation(title = "Feature importance report")
  attr(p, "tsl_data") <- imp
  p
}
