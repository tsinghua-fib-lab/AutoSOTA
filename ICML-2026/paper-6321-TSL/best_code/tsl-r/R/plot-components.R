# Raw univariate step components of fitted grid tensors, in the flat aesthetic.
# A component's curve on feature j is the per-interval mean factor m = b*cosh(d),
# drawn as a single connected step over the finite interior intervals (the
# unbounded (-Inf, .) and (., +Inf) tails are dropped).

# Connected step vertices for feature j of a single grid tensor `gt`. With
# K = length(splits[[j]]) breakpoints there are K + 1 intervals; the first and
# last are unbounded, so only intervals i = 2 .. K are finite. Returns a frame of
# ordered `x`, `value` vertices for a post-step (`geom_step(direction = "hv")`),
# so the interior intervals join into one continuous curve. Empty when K < 2.
.tsl_component_step <- function(gt, j) {
  splits <- gt$splits[[j]]
  K <- length(splits)
  if (K < 2L) {
    return(data.frame(x = numeric(0), value = numeric(0)))
  }
  # Interval i = 2..K spans (splits[i-1], splits[i]] with value m[i]. A post-step
  # holds the right-hand value at each vertex, with the last value repeated to
  # close the final interval.
  m <- gt$backbone_values[[j]] * cosh(gt$tilt_values[[j]])
  data.frame(x = splits[seq_len(K)], value = c(m[2:K], m[K]))
}

#' Plot the univariate components of a grid tensor
#'
#' Draws the raw per-feature step components of one fitted grid tensor. For
#' feature axis `j` the curve is the per-interval mean factor
#' \eqn{m = b \cosh(d)} (the backbone times the hyperbolic cosine of the tilt),
#' shown as a piecewise-constant step over the finite interior intervals.
#'
#' @param grid_tensor A single grid-tensor list, e.g.
#'   `tsl_components(fit)[[1]]$combined_grid_tensor`, with per-feature `splits`,
#'   `backbone_values`, and `tilt_values`.
#' @param axis Feature axis to draw, as a 1-based index. If `NULL` (default), all
#'   axes are overlaid and coloured by axis.
#' @param feature_names Optional character vector of axis labels; defaults to
#'   `"axis 1"`, `"axis 2"`, ... when not supplied.
#' @return A ggplot object. The assembled data frame is attached as the
#'   `"tsl_data"` attribute.
#' @seealso [tsl_components()], [plot_combined_grid_tensors()],
#'   [plot_epoch_components()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_grid_tensor_components(tsl_components(fit)[[1]]$combined_grid_tensor,
#'                             feature_names = c("a", "b", "c"))
#' @export
plot_grid_tensor_components <- function(grid_tensor, axis = NULL,
                                        feature_names = NULL) {
  n_axes <- length(grid_tensor$splits)
  axes <- if (is.null(axis)) seq_len(n_axes) else as.integer(axis)

  df <- do.call(rbind, lapply(axes, function(j) {
    seg <- .tsl_component_step(grid_tensor, j)
    if (!nrow(seg)) return(NULL)
    label <- if (!is.null(feature_names)) feature_names[[j]] else paste("axis", j)
    seg$axis <- label
    seg
  }))
  if (is.null(df)) {
    df <- data.frame(x = numeric(0), value = numeric(0), axis = character(0))
  }
  df$axis <- factor(df$axis)

  p <- ggplot(df) +
    geom_step(aes(x = x, y = value, colour = axis, group = axis),
              direction = "hv", linewidth = 1) +
    scale_colour_tsl(name = NULL) +
    labs(title = "Grid-tensor components",
         subtitle = "mean factor b*cosh(d)",
         x = "x", y = "mean factor") +
    theme_flat()
  attr(p, "tsl_data") <- df
  p
}

#' Plot the combined grid-tensor components of every stage
#'
#' Draws the raw univariate step components of each boosting stage's combined
#' grid tensor, one facet per stage. As in [plot_grid_tensor_components()] each
#' curve is the per-interval mean factor \eqn{b \cosh(d)} over the finite
#' interior intervals, coloured by feature axis.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param axis Feature axis to draw, as a 1-based index. If `NULL` (default), all
#'   axes are overlaid and coloured by axis.
#' @return A ggplot object. The assembled data frame is attached as the
#'   `"tsl_data"` attribute.
#' @seealso [tsl_components()], [plot_grid_tensor_components()],
#'   [plot_epoch_components()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_combined_grid_tensors(fit)
#' @export
plot_combined_grid_tensors <- function(object, axis = NULL) {
  .tsl_check_model(object)
  nm <- .tsl_feature_names(object)
  comp <- tsl_components(object)

  df <- do.call(rbind, lapply(seq_along(comp), function(s) {
    gt <- comp[[s]]$combined_grid_tensor
    axes <- if (is.null(axis)) seq_len(length(gt$splits)) else as.integer(axis)
    rows <- do.call(rbind, lapply(axes, function(j) {
      seg <- .tsl_component_step(gt, j)
      if (!nrow(seg)) return(NULL)
      seg$axis <- nm[[j]]
      seg
    }))
    if (is.null(rows)) return(NULL)
    rows$stage <- paste("Stage", s)
    rows
  }))
  if (is.null(df)) {
    df <- data.frame(x = numeric(0), value = numeric(0),
                     axis = character(0), stage = character(0))
  }
  df$axis <- factor(df$axis, levels = nm)
  df$stage <- factor(df$stage)

  p <- ggplot(df) +
    geom_step(aes(x = x, y = value, colour = axis,
                  group = interaction(stage, axis)),
              direction = "hv", linewidth = 1) +
    facet_wrap(~stage) +
    scale_colour_tsl(name = NULL) +
    labs(title = "Combined grid-tensor components",
         subtitle = "mean factor b*cosh(d)",
         x = "x", y = "mean factor") +
    theme_flat()
  attr(p, "tsl_data") <- df
  p
}

#' Plot the per-tree components of one boosting stage
#'
#' Draws the raw univariate step components of every tree in a stage's bag, one
#' facet per feature. Each tree is one connected step of the per-interval mean
#' factor \eqn{b \cosh(d)}, coloured along a pale-to-indigo gradient by its total
#' scale \eqn{\lambda^+ + \lambda^-}, so the bag reads as a fan rather than a
#' tangle of distinct colours. The stage's aligned (combined) grid-tensor
#' component is overlaid in ink on top of the bag.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param epoch Boosting stage to draw, as a 1-based index.
#' @return A ggplot object. The assembled data frame is attached as the
#'   `"tsl_data"` attribute; the combined component is attached as
#'   `"tsl_combined"`.
#' @seealso [tsl_components()], [plot_grid_tensor_components()],
#'   [plot_combined_grid_tensors()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(150 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(150, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_epoch_components(fit, 1L)
#' @export
plot_epoch_components <- function(object, epoch) {
  .tsl_check_model(object)
  comp <- tsl_components(object)
  if (epoch < 1L || epoch > length(comp)) {
    stop("`epoch` must be a 1-based stage index in [1, ", length(comp), "].",
         call. = FALSE)
  }
  nm <- .tsl_feature_names(object)
  trees <- comp[[epoch]]$grid_tensors

  df <- do.call(rbind, lapply(seq_along(trees), function(t) {
    gt <- trees[[t]]
    rows <- do.call(rbind, lapply(seq_len(length(gt$splits)), function(j) {
      seg <- .tsl_component_step(gt, j)
      if (!nrow(seg)) return(NULL)
      seg$feature <- nm[[j]]
      seg
    }))
    if (is.null(rows)) return(NULL)
    rows$tree <- paste("tree", t)
    rows$lambda <- gt$lambda_plus + gt$lambda_minus
    rows
  }))
  if (is.null(df)) {
    df <- data.frame(x = numeric(0), value = numeric(0),
                     feature = character(0), tree = character(0),
                     lambda = numeric(0))
  }
  df$feature <- factor(df$feature, levels = nm)
  df$tree <- factor(df$tree)

  # The stage's aligned (combined) grid tensor, overlaid in ink.
  gtc <- comp[[epoch]]$combined_grid_tensor
  combined <- do.call(rbind, lapply(seq_len(length(gtc$splits)), function(j) {
    seg <- .tsl_component_step(gtc, j)
    if (!nrow(seg)) return(NULL)
    seg$feature <- nm[[j]]
    seg
  }))
  if (is.null(combined)) {
    combined <- data.frame(x = numeric(0), value = numeric(0),
                           feature = character(0))
  }
  combined$feature <- factor(combined$feature, levels = nm)

  p <- ggplot() +
    geom_step(data = df,
              aes(x = x, y = value, colour = lambda,
                  group = interaction(feature, tree)),
              direction = "hv", linewidth = 0.5, alpha = 0.55) +
    geom_step(data = combined,
              aes(x = x, y = value, group = feature),
              direction = "hv", colour = .tsl_tokens$ink, linewidth = 1.1) +
    facet_wrap(~feature, scales = "free_x") +
    scale_colour_gradientn(colours = .tsl_ramp_backbone,
                           name = expression(lambda^"+" + lambda^"-")) +
    labs(title = paste0("Stage ", epoch, " per-tree components"),
         subtitle = "mean factor b*cosh(d); trees by total scale, combined in ink",
         x = "x", y = "mean factor") +
    theme_flat()
  attr(p, "tsl_data") <- df
  attr(p, "tsl_combined") <- combined
  p
}
