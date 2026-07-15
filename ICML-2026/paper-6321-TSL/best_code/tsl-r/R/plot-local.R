# Composite local-interpretation dashboard for fitted TSL models, in the flat
# aesthetic. For each query point, three panels are built from tsl_local() and
# composed in a row with patchwork: a signed stage-contribution waterfall, a
# per-stage backbone-share marimekko, and per-stage grouped signed tilts (with
# the constant intercept treated as a zeroth axis). When patchwork is not
# installed the per-point tsl_local() results are returned instead.

# Coerce `points` (a single numeric vector, a list of vectors, or a matrix with
# one point per row) to a list of numeric vectors.
.tsl_as_point_list <- function(points) {
  if (is.list(points)) {
    return(lapply(points, as.numeric))
  }
  if (is.matrix(points)) {
    return(lapply(seq_len(nrow(points)), function(i) as.numeric(points[i, ])))
  }
  list(as.numeric(points))
}

# Signed, comma-grouped value label. Magnitudes >= 1000 read as integers with
# thousands separators (+280,341); smaller ones keep three decimals (+2.803),
# so the same formatter serves both money-scale and unit-scale models.
.tsl_local_value <- function(v) {
  sign <- if (v < 0) "-" else "+"
  if (abs(v) >= 1000) {
    paste0(sign, formatC(abs(v), format = "d", big.mark = ",", digits = 0))
  } else {
    paste0(sign, formatC(abs(v), format = "f", digits = 3))
  }
}

# Readable label colour (white or ink) for text on a fill, by relative
# luminance — mirrors tsl_py.plot._theme._text_on.
.tsl_text_on <- function(hex) {
  ch <- grDevices::col2rgb(hex) / 255
  lum <- 0.2126 * ch[1] + 0.7152 * ch[2] + 0.0722 * ch[3]
  if (lum < 0.6) "#FFFFFF" else .tsl_tokens$ink
}

# Per-axis share of |log b_j| for one stage's backbone vector. Near-unit
# backbones (|log b| <= 1e-4) and zeros are dropped; the rest are normalised to
# sum to 1 and returned sorted descending.
.tsl_backbone_share <- function(bvec) {
  logs <- ifelse(bvec > 1e-15, abs(log(pmax(bvec, 1e-300))), 0)
  logs[logs <= 1e-4] <- 0
  total <- sum(logs)
  if (total <= 0) {
    return(list(index = integer(0), share = numeric(0)))
  }
  ord <- order(logs, decreasing = TRUE)
  ord <- ord[logs[ord] > 0]
  list(index = ord, share = logs[ord] / total)
}

#' Plot the local-interpretation dashboard
#'
#' Composes a per-point "backbone x tilt" decomposition of one or more TSL
#' predictions. Each query point becomes a row of three panels that share one
#' stage ordering (by absolute net contribution, descending, largest at the
#' top):
#'
#' * **Stage contribution** — a signed waterfall over stages: each bar runs from
#'   the running cumulative to the next, orange for a positive net effect and
#'   blue for a negative one, with the cumulative landing on the prediction
#'   (marked by a dotted line and a boxed value). A signed value label sits at
#'   each bar's tip.
#' * **Backbone share** — one percent-stacked bar per stage giving each feature's
#'   share of `|log b_j(x_j)|`. The intercept is excluded here: it carries the
#'   absolute scale and would dominate the gate. Segments run darkest-to-pale in
#'   indigo with a grey `"Other"` tail.
#' * **Signed tilt** — per stage, the signed local effect `d_j` of the
#'   top-`top_k_features` axes (the constant intercept axis `d_0` included),
#'   orange when positive and blue when negative. The value scale comes from the
#'   feature tilts only, so a one-sided stage whose intercept absorbs the whole
#'   `log(lam_+/lam_-)` does not flatten the rest; an off-scale intercept is
#'   clipped to the edge with a bold label.
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param points A single numeric vector of length `n_features`, a list of such
#'   vectors, or a matrix with one point per row.
#' @param titles Per-point titles. Defaults to `"Point 1"`, `"Point 2"`, ...
#'   A single point names the figure title; several points tag each row.
#' @param top_k_features Number of tilt axes kept per stage in the signed-tilt
#'   panel.
#' @return A patchwork object (or a list of ggplots if patchwork is not
#'   installed). The per-point [tsl_local()] results are attached as the
#'   `"tsl_data"` attribute (see [tsl_plot_data()]).
#' @seealso [tsl_local()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_local_interpretation(fit, x[1, ])
#' plot_local_interpretation(fit, x[1:2, ], titles = c("A", "B"))
#' @export
plot_local_interpretation <- function(object, points, titles = NULL,
                                      top_k_features = 3L) {
  .tsl_check_model(object)
  points <- .tsl_as_point_list(points)
  if (is.null(titles)) titles <- paste("Point", seq_along(points))
  top_k_features <- max(1L, as.integer(top_k_features))

  ex_list <- lapply(points, function(pt) tsl_local(object, pt))
  single <- length(ex_list) == 1L

  sub <- Map(function(ex, title) {
    list(
      stage    = .tsl_local_waterfall(ex, single),
      backbone = .tsl_local_backbone(ex, single),
      tilt     = .tsl_local_tilt(ex, top_k_features, single)
    )
  }, ex_list, titles)

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("Install 'patchwork' for the composed dashboard; ",
            "returning the per-point explanations.")
    return(ex_list)
  }

  rows <- Map(function(s, title) {
    row <- patchwork::wrap_plots(s$stage, s$backbone, s$tilt, nrow = 1,
                                 widths = c(1.05, 1.35, 1.5))
    if (single) row else row + patchwork::plot_annotation(title = title)
  }, sub, titles)

  # plot_annotation does not propagate through wrap_plots, so a per-row tag is
  # carried on the row's leftmost panel for the multi-point layout.
  if (!single) {
    rows <- Map(function(s, title) {
      patchwork::wrap_plots(
        s$stage + labs(tag = title),
        s$backbone, s$tilt, nrow = 1, widths = c(1.05, 1.35, 1.5)
      )
    }, sub, titles)
  }

  title_text <- "Local explanation"
  if (single) title_text <- paste0(title_text, "  \u00b7  ", titles[[1]])

  out <- patchwork::wrap_plots(rows, ncol = 1) +
    patchwork::plot_annotation(
      title = title_text,
      subtitle = "TSL / diagnostics",
      theme = theme(
        plot.title = element_text(colour = .tsl_tokens$ink, face = "bold",
                                  size = rel(1.4)),
        plot.subtitle = element_text(family = "mono", colour = .tsl_tokens$muted,
                                     size = rel(0.85))
      )
    )
  attr(out, "tsl_data") <- ex_list
  out
}

# ---------------------------------------------------------------------------
# Panel 1 — stage-contribution waterfall
# ---------------------------------------------------------------------------
.tsl_local_waterfall <- function(ex, single) {
  net <- ex$stage_contributions
  total <- ex$total_prediction
  n <- length(net)
  ord <- order(abs(net), decreasing = TRUE)
  net_o <- net[ord]
  labels <- paste("Stage", ord)

  # Largest |net| at the top: y descends as we go down the panel.
  y <- n - seq_len(n) + 1
  start <- c(0, cumsum(net_o))[seq_len(n)]
  end <- start + net_o
  band <- 0.62

  bars <- data.frame(
    ymin = y - band / 2, ymax = y + band / 2,
    xmin = pmin(start, end), xmax = pmax(start, end),
    pos = net_o >= 0
  )
  fills <- ifelse(bars$pos, .tsl_tokens$pos, .tsl_tokens$neg)

  # Pad both ends generously so the value labels, which sit just past each bar
  # tip, clear the axis on whichever side the cumulative path runs.
  span <- max(max(c(end, 0)) - min(c(end, 0)), abs(total), 1e-9)
  x_lo <- min(c(start, end, 0)) - 0.22 * span
  x_hi <- max(c(start, end, 0)) + 0.22 * span

  # Dashed connectors between consecutive bars at their shared cumulative x.
  conn <- data.frame(x = end[seq_len(n - 1)],
                     y0 = y[seq_len(n - 1)] - band / 2,
                     y1 = y[seq_len(n - 1) + 1] + band / 2)

  # Value at each bar tip, just past the end, coloured and aligned by sign.
  tips <- data.frame(
    x = ifelse(bars$pos, end + 0.015 * span, end - 0.015 * span),
    y = y,
    label = vapply(net_o, .tsl_local_value, character(1)),
    hjust = ifelse(bars$pos, 0, 1),
    colour = fills
  )

  p <- ggplot() +
    geom_vline(xintercept = 0, colour = .tsl_tokens$faint, linewidth = 0.7) +
    geom_segment(data = conn,
                 aes(x = x, xend = x, y = y0, yend = y1),
                 colour = .tsl_tokens$faint, linetype = "dashed",
                 linewidth = 0.4) +
    geom_rect(data = bars,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              fill = fills) +
    geom_vline(xintercept = total, colour = .tsl_tokens$ink,
               linetype = "dotted", linewidth = 0.5) +
    geom_text(data = tips,
              aes(x = x, y = y, label = label, hjust = hjust),
              colour = tips$colour, family = "mono", fontface = "bold",
              size = 3.1) +
    geom_label(aes(x = total, y = n + 0.7,
                   label = vapply(total, .tsl_local_value, character(1))),
               fill = "white", colour = .tsl_tokens$ink,
               linewidth = 0.4, family = "mono", fontface = "bold",
               size = 3.0) +
    scale_y_continuous(breaks = y, labels = labels,
                       limits = c(0.3, n + 1.1)) +
    coord_cartesian(xlim = c(x_lo, x_hi), clip = "off") +
    labs(title = "Stage contribution",
         subtitle = if (single) "Stage effects, summing to the prediction." else NULL,
         x = "Contribution to prediction", y = NULL) +
    theme_flat() +
    theme(panel.grid.major.y = element_blank(),
          panel.border = element_blank(),
          plot.margin = margin(12, 22, 12, 12))
  p
}

# ---------------------------------------------------------------------------
# Panel 2 — backbone share (per-stage percent-stacked marimekko)
# ---------------------------------------------------------------------------
.tsl_local_backbone <- function(ex, single) {
  net <- ex$stage_contributions
  n <- length(net)
  ord <- order(abs(net), decreasing = TRUE)
  fb <- ex$feature_backbone
  fnames <- ex$feature_names
  band <- 0.62
  accent <- .tsl_tokens$accent
  grey_tail <- .tsl_mix(.tsl_tokens$greys[1], 0)

  segs <- list()
  for (r in seq_len(n)) {
    s_idx <- ord[r]
    y <- n - r + 1
    sh <- .tsl_backbone_share(fb[s_idx, ])
    if (length(sh$index) == 0) {
      segs[[length(segs) + 1]] <- data.frame(
        y = y, xmin = 0, xmax = 1, fill = grey_tail, name = "Other",
        pct = 1, named = FALSE, stringsAsFactors = FALSE)
      next
    }
    # Keep the leading segments until the residual tail drops under 10%, or
    # until every contributing axis is shown.
    cum <- cumsum(sh$share)
    k <- which(1 - cum < 0.10 - 1e-9)[1]
    if (is.na(k)) k <- length(sh$share)
    keep_idx <- sh$index[seq_len(k)]
    keep_pct <- sh$share[seq_len(k)]
    tail_pct <- max(0, 1 - sum(keep_pct))

    left <- 0
    n_seg <- max(length(keep_idx), 1)
    for (i in seq_along(keep_idx)) {
      # Earlier (larger) segments read darkest; later ones fade toward white.
      w <- min(0.55, 0.55 * (i - 1) / n_seg)
      fill <- .tsl_mix(accent, w)
      segs[[length(segs) + 1]] <- data.frame(
        y = y, xmin = left, xmax = left + keep_pct[i], fill = fill,
        name = fnames[keep_idx[i]], pct = keep_pct[i], named = TRUE,
        stringsAsFactors = FALSE)
      left <- left + keep_pct[i]
    }
    if (tail_pct > 1e-6) {
      segs[[length(segs) + 1]] <- data.frame(
        y = y, xmin = left, xmax = 1, fill = grey_tail, name = "Other",
        pct = tail_pct, named = FALSE, stringsAsFactors = FALSE)
    }
  }
  df <- do.call(rbind, segs)
  df$ymin <- df$y - band / 2
  df$ymax <- df$y + band / 2
  df$mid <- (df$xmin + df$xmax) / 2
  df$txt <- vapply(df$fill, .tsl_text_on, character(1))
  df$pctlab <- paste0(round(df$pct * 100), "%")

  # Show the feature name where the segment is wide enough; otherwise drop to a
  # percentage only, and on very thin segments label nothing.
  name_df <- df[df$named & df$pct >= 0.12, , drop = FALSE]
  pct_above <- df[df$pct >= 0.06, , drop = FALSE]            # pct shown
  pct_solo  <- pct_above[!(pct_above$named & pct_above$pct >= 0.12), ,
                         drop = FALSE]

  # Breaks run bottom-up (sorted ascending); the row order is largest-first from
  # the top, so the labels are reversed to meet them.
  y_breaks <- sort(df$y[!duplicated(df$y)])
  y_labels <- rev(paste("Stage", ord))

  p <- ggplot() +
    geom_rect(data = df,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              fill = df$fill) +
    geom_segment(data = df[df$xmax < 1 - 1e-9, , drop = FALSE],
                 aes(x = xmax, xend = xmax, y = ymin, yend = ymax),
                 colour = "white", linewidth = 0.5)
  if (nrow(name_df)) {
    p <- p +
      geom_text(data = name_df,
                aes(x = mid, y = y + 0.13, label = name),
                colour = name_df$txt, family = "sans", fontface = "bold",
                size = 2.7) +
      geom_text(data = name_df,
                aes(x = mid, y = y - 0.13, label = pctlab),
                colour = name_df$txt, family = "mono", size = 2.6)
  }
  if (nrow(pct_solo)) {
    p <- p +
      geom_text(data = pct_solo,
                aes(x = mid, y = y, label = pctlab),
                colour = pct_solo$txt, family = "mono", size = 2.6)
  }
  p +
    scale_x_continuous(breaks = c(0, .25, .5, .75, 1),
                       labels = c("0%", "25%", "50%", "75%", "100%"),
                       limits = c(0, 1), expand = expansion(0)) +
    scale_y_continuous(breaks = y_breaks,
                       labels = y_labels,
                       limits = c(0.3, n + 0.7)) +
    labs(title = "Backbone share",
         subtitle = if (single) "Each feature's share of the magnitude gate." else NULL,
         x = NULL, y = NULL) +
    theme_flat() +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.border = element_blank(),
          axis.text.y = element_text(family = "sans", colour = .tsl_tokens$ink))
}

# ---------------------------------------------------------------------------
# Panel 3 — signed tilt (per-stage grouped signed bars)
# ---------------------------------------------------------------------------
.tsl_local_tilt <- function(ex, top_k, single) {
  net <- ex$stage_contributions
  n <- length(net)
  ord <- order(abs(net), decreasing = TRUE)
  fnames <- ex$feature_names
  axis_labels <- c("Intercept", fnames)
  band <- 0.62

  # Select per stage, then size the value axis from feature tilts only — the
  # intercept absorbs all of log(lam_+/lam_-) on a one-sided stage and would
  # otherwise flatten every feature bar.
  per_row <- vector("list", n)
  feat_mags <- numeric(0)
  for (r in seq_len(n)) {
    s_idx <- ord[r]
    tilt_axis <- c(ex$intercept_tilt[s_idx], ex$feature_tilt[s_idx, ])
    mag <- abs(tilt_axis)
    sel <- order(mag, decreasing = TRUE)
    sel <- sel[mag[sel] > 1e-12]
    sel <- utils::head(sel, top_k)
    per_row[[r]] <- sel
    feat_mags <- c(feat_mags, mag[sel[sel != 1L]])
  }
  scale_lim <- if (length(feat_mags)) max(feat_mags) else 1
  lim <- max(scale_lim * 1.30, 1e-6)

  bars <- list()
  labs_left <- list()
  for (r in seq_len(n)) {
    s_idx <- ord[r]
    y <- n - r + 1
    tilt_axis <- c(ex$intercept_tilt[s_idx], ex$feature_tilt[s_idx, ])
    sel <- per_row[[r]]
    if (length(sel) == 0) next
    # Sort selected axes by signed value descending — positive sub-bars on top.
    sel <- sel[order(tilt_axis[sel], decreasing = TRUE)]
    n_sub <- max(top_k, 1L)
    sub_h <- band / n_sub
    for (k in seq_along(sel)) {
      j <- sel[k]
      # ggplot's y-axis runs upward, so the largest tilt (k = 1) takes the
      # highest sub-slot, stacking positive effects above negative ones.
      yy <- (y - band / 2) + sub_h * (n_sub - k + 0.5)
      raw <- tilt_axis[j]
      off <- abs(raw) > lim
      tip <- if (off) sign(raw) * lim else raw
      bars[[length(bars) + 1]] <- data.frame(
        ymin = yy - sub_h * 0.85 / 2, ymax = yy + sub_h * 0.85 / 2,
        xmin = pmin(0, tip), xmax = pmax(0, tip),
        fill = if (raw >= 0) .tsl_tokens$pos else .tsl_tokens$neg,
        tipx = tip, raw = raw,
        label = sprintf("%+.2f", raw),
        hjust = if (raw >= 0) 0 else 1,
        off = off, y = yy, stringsAsFactors = FALSE)
      labs_left[[length(labs_left) + 1]] <- data.frame(
        y = yy, label = axis_labels[j], stringsAsFactors = FALSE)
    }
  }
  bdf <- do.call(rbind, bars)
  ldf <- do.call(rbind, labs_left)
  gap <- 0.03 * lim

  # The off-scale (intercept) bar is clipped to the edge; its label is pulled
  # just inside the axis so the bold number stays on the panel.
  bdf$labx <- ifelse(
    bdf$raw >= 0,
    ifelse(bdf$off, lim - gap, bdf$tipx + gap),
    ifelse(bdf$off, -lim + gap, bdf$tipx - gap)
  )

  p <- ggplot() +
    geom_vline(xintercept = 0, colour = .tsl_tokens$faint, linewidth = 0.7) +
    geom_rect(data = bdf,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              fill = bdf$fill) +
    geom_text(data = bdf,
              aes(x = labx, y = y, label = label, hjust = hjust),
              colour = bdf$fill, family = "mono",
              fontface = ifelse(bdf$off, "bold", "plain"), size = 2.7) +
    geom_text(data = ldf,
              aes(x = -lim, y = y, label = label),
              hjust = 1, colour = .tsl_tokens$ink, family = "sans",
              size = 2.6) +
    scale_y_continuous(limits = c(0.3, n + 0.7)) +
    coord_cartesian(xlim = c(-lim, lim), clip = "off") +
    labs(title = "Signed tilt",
         subtitle = if (single) "Which way each feature tilts the prediction." else NULL,
         x = expression("Signed local effect (tilt " * d[j] * ")"), y = NULL) +
    theme_flat() +
    theme(panel.grid.major.y = element_blank(),
          panel.border = element_blank(),
          axis.text.y = element_blank(),
          plot.margin = margin(12, 16, 12, 70))
  p
}
