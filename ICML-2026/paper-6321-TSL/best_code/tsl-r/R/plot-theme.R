# Flat visual theme for the tensorsl plotting layer: white ground, hairline borders,
# a faint grid, monospace tick labels, and the indigo / blue-orange palette used
# by the Python `tsl_py.plot` diagnostics. The look is driven by a single token
# list and three colour ramps so it re-skins from one place.

# Solid material tokens (mirrors tsl_py.plot._theme.TOKENS).
.tsl_tokens <- list(
  bg = "#FFFFFF", card = "#FFFFFF", border = "#E4E4E7", grid = "#F1F1F3",
  ink = "#18181B", muted = "#71717A", faint = "#B4B4BB",
  accent = "#4F46E5",                 # indigo, primary
  neg = "#2563EB", pos = "#F97316",   # blue / orange for signed effects
  greys = c("#C7C7CE", "#A8A8B0", "#8A8A93")
)

# Sequential pale -> indigo ramp for unsigned backbone magnitude.
.tsl_ramp_backbone <- c("#F4F4F5", "#A5A0F0", "#4F46E5", "#312E81")
# Diverging blue -> pale -> orange ramp for signed surfaces, anchored at zero.
.tsl_ramp_diverging <- c("#2563EB", "#9DBDFB", "#F4F4F5", "#FDC089", "#F97316")
# Sequential pale -> orange ramp for tilt magnitude.
.tsl_ramp_tilt <- c("#F4F4F5", "#FBC99A", "#F97316", "#9A3412")
# Calm categorical cycle for overlaid curves (e.g. one colour per stage).
.tsl_cycle <- c("#4F46E5", "#F97316", "#2563EB", "#0D9488", "#9333EA",
                "#D97706", "#8A8A93", "#A8A8B0")

#' Flat ggplot2 theme for tensorsl diagnostics
#'
#' A minimal theme matching the `tsl_py.plot` "flat" aesthetic: white panels,
#' hairline borders, a faint grid, and muted monospace axis labels. Pair it with
#' the [scale_fill_tsl_backbone()] family for the matching colour ramps.
#'
#' @param base_size Base font size in points.
#' @return A ggplot2 theme object, composable with `+`.
#' @examples
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   library(ggplot2)
#'   ggplot(mtcars, aes(wt, mpg)) + geom_point() + theme_flat()
#' }
#' @export
theme_flat <- function(base_size = 11) {
  t <- .tsl_tokens
  theme_minimal(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.grid.major = element_line(colour = t$grid, linewidth = 0.4),
      panel.grid.minor = element_blank(),
      panel.border     = element_rect(fill = NA, colour = t$border,
                                       linewidth = 0.6),
      axis.ticks  = element_blank(),
      axis.text   = element_text(family = "mono", colour = t$muted,
                                 size = rel(0.78)),
      axis.title  = element_text(family = "mono", colour = t$muted,
                                 size = rel(0.85)),
      plot.title    = element_text(colour = t$ink, face = "bold",
                                   size = rel(1.18)),
      plot.subtitle = element_text(colour = t$muted, size = rel(0.9)),
      strip.text       = element_text(colour = t$ink, face = "bold",
                                       size = rel(0.92), hjust = 0),
      strip.background = element_blank(),
      legend.title = element_text(family = "mono", colour = t$muted,
                                  size = rel(0.78)),
      legend.text  = element_text(family = "mono", colour = t$muted,
                                  size = rel(0.74)),
      plot.margin  = margin(12, 16, 12, 12)
    )
}

#' Flat-theme colour scales
#'
#' Fill and colour scales matching [theme_flat()]: a sequential indigo ramp for
#' unsigned backbone magnitude, a sequential orange ramp for tilt magnitude, a
#' blue-orange diverging ramp anchored at zero for signed surfaces, and a calm
#' categorical colour cycle (e.g. one colour per stage).
#'
#' @param name Legend title.
#' @param limits For the diverging scale, the symmetric fill limits (values
#'   outside are squished to the ends). Defaults to `c(-1, 1)`.
#' @param ... Passed to the underlying ggplot2 scale.
#' @return A ggplot2 scale, composable with `+`.
#' @name scale_tsl
#' @examples
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   library(ggplot2)
#'   df <- expand.grid(x = 1:5, y = 1:5)
#'   df$z <- df$x * df$y
#'   ggplot(df, aes(x, y, fill = z)) + geom_tile() +
#'     scale_fill_tsl_backbone() + theme_flat()
#' }
NULL

#' @rdname scale_tsl
#' @export
scale_fill_tsl_backbone <- function(name = "backbone", ...) {
  scale_fill_gradientn(colours = .tsl_ramp_backbone, name = name, ...)
}

#' @rdname scale_tsl
#' @export
scale_fill_tsl_tilt <- function(name = "tilt", ...) {
  scale_fill_gradientn(colours = .tsl_ramp_tilt, name = name, ...)
}

#' @rdname scale_tsl
#' @export
scale_fill_tsl_diverging <- function(name = "value", limits = c(-1, 1), ...) {
  scale_fill_gradientn(colours = .tsl_ramp_diverging, limits = limits,
                       oob = scales::squish, name = name, ...)
}

#' @rdname scale_tsl
#' @export
scale_colour_tsl <- function(name = NULL, ...) {
  scale_colour_manual(values = .tsl_cycle, name = name, ...)
}

# A faint dashed zero reference line, used across the curve plots.
.tsl_zero_ref <- function(yintercept = 0) {
  geom_hline(yintercept = yintercept, colour = .tsl_tokens$faint,
             linetype = "dashed", linewidth = 0.4)
}

# Solid blend of a hex colour toward white by `w` (stays fully opaque), mirroring
# tsl_py.plot._theme.mix; the pale signed-difference fills use it.
.tsl_mix <- function(hex, w = 0.82) {
  ch <- grDevices::col2rgb(hex) / 255
  grDevices::rgb(ch[1] * (1 - w) + w, ch[2] * (1 - w) + w, ch[3] * (1 - w) + w)
}

# Readable label colour (white or ink) for text sitting on a gradient fill: the
# scaled value `v` in [0, 1] is mapped through `ramp`, then handed to
# .tsl_text_on for the per-cell luminance choice. Vectorised over `v`.
.tsl_ramp_text <- function(v, ramp) {
  rgbf <- grDevices::colorRamp(ramp)
  m <- rgbf(pmin(pmax(v, 0), 1))
  cols <- grDevices::rgb(m[, 1], m[, 2], m[, 3], maxColorValue = 255)
  vapply(cols, .tsl_text_on, character(1))
}

# Two pale ribbons shading the signed gap between the PD+ (`pos`) and PD-
# (`-neg`) curves: the orange (positive) token where PD+ >= PD-, the blue
# (negative) token elsewhere. Each ribbon collapses to zero height where the
# branches cross, so the fill flips exactly at the crossing. Mirrors
# tsl_py.plot._theme.signed_fill; expects `x`, `pos`, `neg` columns.
.tsl_signed_diff_fill <- function() {
  list(
    geom_ribbon(aes(x = x, ymin = -neg, ymax = pmax(pos, -neg)),
                fill = .tsl_mix(.tsl_tokens$pos)),
    geom_ribbon(aes(x = x, ymin = pmin(pos, -neg), ymax = -neg),
                fill = .tsl_mix(.tsl_tokens$neg))
  )
}

# The PD+ (orange) and PD- (blue) branch curves drawn on the positive scale,
# with colour and linetype mapped to a shared key so the panels carry a legend.
# When `backbone` is TRUE the dotted sqrt(C+ C-) * b overlay joins the same
# legend. Expects `x`, `pos`, `neg`, and (for the overlay) `backbone` columns.
.tsl_branch_curves <- function(backbone = TRUE) {
  keys <- c("PD+", "PD-")
  layers <- list(
    geom_line(aes(x, pos, colour = "PD+", linetype = "PD+"), linewidth = 1),
    geom_line(aes(x, -neg, colour = "PD-", linetype = "PD-"), linewidth = 1)
  )
  if (isTRUE(backbone)) {
    keys <- c(keys, "backbone")
    layers <- c(layers, list(
      geom_line(aes(x, backbone, colour = "backbone", linetype = "backbone"),
                linewidth = 0.6)
    ))
  }
  c(layers, list(
    scale_colour_manual(
      name = NULL, breaks = keys,
      values = c("PD+" = .tsl_tokens$pos, "PD-" = .tsl_tokens$neg,
                 "backbone" = .tsl_tokens$ink)
    ),
    scale_linetype_manual(
      name = NULL, breaks = keys,
      values = c("PD+" = "solid", "PD-" = "solid", "backbone" = "dotted")
    )
  ))
}

# Column names used inside aes() across the plotting layer; declaring them keeps
# R CMD check from flagging them as undefined globals (non-standard evaluation).
utils::globalVariables(c(
  "feature", "stage", "x", "y", "pos", "neg", "net", "backbone", "d", "curve",
  "value", "z", "panel", "ice_id", "pd", "importance", "metric", "weight",
  "share", "tilt", "contribution", "cumulative", "ymin", "ymax", "xmin", "xmax",
  "label", "point", "fpos", "fneg", "axis", "tree", "lo", "hi", "xend", "yend",
  "feat_label", "combined", "bscaled", "tscaled", "lambda",
  "mid", "name", "pctlab", "labx", "hjust", "y0", "y1"
))
