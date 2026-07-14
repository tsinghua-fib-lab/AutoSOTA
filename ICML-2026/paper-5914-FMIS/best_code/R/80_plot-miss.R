# Plot coefficent trace from MISS analysis
plot_trace <- function(
  input,
  K_MAX = Inf,
  add_breaks = TRUE,
  add_greedy = TRUE
) {
  values_p <- vapply(input[["miss_pos"]], \(l) l[["best_value"]], numeric(1L))
  values_n <- vapply(input[["miss_neg"]], \(l) l[["best_value"]], numeric(1L))
  x_max <- min(length(values_p), K_MAX)
  y_lim <- quantile(c(values_p[seq(x_max)], values_n[seq(x_max)]), c(.05, .95))

  # Add lines for the greedy algorithm if available + desired
  has_greedy <- !is.null(input[["greedy_pos"]]) ||
    !is.null(input[["greedy_neg"]])
  greedy <- has_greedy && isTRUE(add_greedy)

  # Add dots indicating breaks in nestedness
  if (isTRUE(add_breaks)) {
    set_p <- lapply(input[["miss_pos"]], \(x) x[["best_S"]])
    set_n <- lapply(input[["miss_neg"]], \(x) x[["best_S"]])
    breaks_p <- find_breaks(set_p[seq(x_max)])
    breaks_n <- find_breaks(set_n[seq(x_max)])
  }

  {
    plot.new()
    plot.window(ylim = y_lim, xlim = c(0L, x_max))
    axis(1L)
    axis(
      2L,
      at = c(0, input[["coef"]], y_lim),
      labels = formatC(c(0, input[["coef"]], y_lim))
    )
    grid()
    abline(h = c(0), col = "black", lty = 2)
    if (isTRUE(greedy)) {
      lines(
        seq(0L, x_max),
        c(input[["coef"]], input[["greedy_pos"]][["best_value"]][seq(x_max)]),
        col = "firebrick",
        lty = 2
      )
      lines(
        seq(0L, x_max),
        c(input[["coef"]], input[["greedy_neg"]][["best_value"]][seq(x_max)]),
        col = "steelblue",
        lty = 2
      )
    }
    if (isTRUE(add_breaks)) {
      points(
        seq(x_max)[breaks_p],
        values_p[breaks_p],
        col = "firebrick",
        pch = 20,
        cex = .5
      )
      points(
        seq(x_max)[breaks_n],
        values_n[breaks_n],
        col = "steelblue",
        pch = 20,
        cex = .5
      )
    }
    lines(
      seq(0L, x_max),
      c(input[["coef"]], values_p[seq(x_max)]),
      col = "firebrick"
    )
    lines(
      seq(0L, x_max),
      c(input[["coef"]], values_n[seq(x_max)]),
      col = "steelblue"
    )
    title("MISS Coefficent Traces", xlab = "Removals (k)", ylab = "Coefficent")
    legend(
      "topleft",
      legend = c("Positive", "Negative"),
      lty = 1,
      col = c("steelblue", "firebrick"),
      lwd = 2,
      bty = "n",
      cex = 0.8
    )
  }
}

# Difference between MISS and greedy
plot_error <- function(input, K_MAX = Inf) {
  if (is.null(input[["greedy_pos"]]) || is.null(input[["greedy_neg"]])) {
    stop("Greedy algorithm outputs are needed to compare overlap.")
  }

  values_p <- vapply(input[["miss_pos"]], \(l) l[["best_value"]], numeric(1L))
  values_n <- vapply(input[["miss_neg"]], \(l) l[["best_value"]], numeric(1L))
  diff_p <- values_p - input[["greedy_pos"]][["best_value"]]
  diff_n <- values_n - input[["greedy_neg"]][["best_value"]]
  x_max <- min(length(diff_p), K_MAX)
  y_lim <- c(
    min(diff_p[seq(x_max)], diff_n[seq(x_max)], -1e-3),
    max(diff_p[seq(x_max)], diff_n[seq(x_max)], 1e-3)
  )

  {
    plot.new()
    plot.window(ylim = y_lim, xlim = c(1L, x_max))
    axis(1L)
    axis(
      2L,
      at = c(0, input[["coef"]], y_lim),
      labels = formatC(c(0, input[["coef"]], y_lim))
    )
    grid()
    abline(h = c(0), col = "black", lty = 2)
    lines(
      seq(x_max),
      c(diff_p[seq(x_max)]),
      col = "firebrick"
    )
    lines(
      seq(x_max),
      c(diff_n[seq(x_max)]),
      col = "steelblue"
    )
    title(
      "Coefficent Errors (MISS vs. Greedy)",
      xlab = "Removals (k)",
      ylab = "Difference"
    )
    legend(
      "topleft",
      legend = c("Positive", "Negative"),
      lty = 1,
      col = c("steelblue", "firebrick"),
      lwd = 2,
      bty = "n",
      cex = 0.8
    )
  }
}

# Plot set stability
plot_stability <- function(input, K_MAX = Inf) {
  set_p <- lapply(input[["miss_pos"]], \(x) x[["best_S"]])
  set_n <- lapply(input[["miss_neg"]], \(x) x[["best_S"]])

  x_max <- min(length(set_p) - 1L, K_MAX)
  x_seq <- seq(1L, x_max)
  {
    plot.new()
    plot.window(ylim = c(0, 1), xlim = c(1L, x_max))
    axis(1L)
    axis(2L)
    abline(h = c(.2, .4, .6, .8), col = "gray", lty = 2)
    abline(h = 1, col = "black", lty = 1)
    lines(x_seq, jaccard_consecutive(set_p[seq(x_max + 1L)]), col = "firebrick")
    lines(x_seq, jaccard_consecutive(set_n[seq(x_max + 1L)]), col = "steelblue")
    title(
      "Set Stability",
      xlab = "Removals (k)",
      ylab = "Jaccard Index (k, k+1)"
    )
    legend(
      "bottomright",
      legend = c("Positive", "Negative"),
      lty = 1,
      col = c("steelblue", "firebrick"),
      bty = "n",
      cex = 0.8
    )
  }
}
# Plot nestedness
plot_nestedness <- function(input, K_MAX = Inf) {
  set_p <- lapply(input[["miss_pos"]], \(x) x[["best_S"]])
  set_n <- lapply(input[["miss_neg"]], \(x) x[["best_S"]])

  x_max <- min(length(set_p), K_MAX)
  x_seq <- seq(1L, x_max)
  breaks_p <- find_breaks(set_p[x_seq])
  breaks_n <- find_breaks(set_n[x_seq])
  y_max <- max(length(breaks_p), length(breaks_n))
  {
    plot.new()
    plot.window(ylim = c(0L, y_max * 1.05), xlim = c(1L, x_max))
    axis(1L)
    axis(2L)
    grid()
    lines(x_seq, cumsum(x_seq %in% breaks_p), col = "firebrick")
    lines(x_seq, cumsum(x_seq %in% breaks_n), col = "steelblue")
    title(
      "Nestedness Violations",
      xlab = "Removals (k)",
      ylab = "Cumulative breaks"
    )
    legend(
      "topleft",
      legend = c("Positive", "Negative"),
      lty = 1,
      col = c("steelblue", "firebrick"),
      bty = "n",
      cex = 0.8
    )
  }
}

# Plot set overlap
plot_overlap <- function(input, K_MAX = Inf) {
  if (is.null(input[["greedy_pos"]]) || is.null(input[["greedy_neg"]])) {
    stop("Greedy algorithm outputs are needed to compare overlap.")
  }

  x_max <- min(length(input[["greedy_neg"]][["best_S"]]), K_MAX)
  jacc_p <- vapply(
    seq_len(x_max),
    \(k) {
      greedy_set <- input[["greedy_pos"]][["best_S"]][seq_len(k)]
      miss_set <- input[["miss_pos"]][[k]][["best_S"]]
      jaccard_index(greedy_set, miss_set)
    },
    numeric(1L)
  )
  jacc_n <- vapply(
    seq_len(x_max),
    \(k) {
      greedy_set <- input[["greedy_neg"]][["best_S"]][seq_len(k)]
      miss_set <- input[["miss_neg"]][[k]][["best_S"]]
      jaccard_index(greedy_set, miss_set)
    },
    numeric(1L)
  )

  x_seq <- seq(1L, x_max)
  y_min <- switch(
    # Move up the bottom depending on the minimum
    as.character(floor(min(jacc_n, jacc_p) * 4)),
    "0" = 0,
    "1" = 0.25,
    "2" = 0.5,
    "3" = 0.75,
    0.75
  )
  {
    plot.new()
    plot.window(ylim = c(y_min, 1), xlim = c(1L, x_max))
    axis(1L)
    axis(2L)
    abline(h = c(.2, .4, .6, .8), col = "gray", lty = 2)
    lines(x_seq, jacc_p, col = "firebrick")
    lines(x_seq, jacc_n, col = "steelblue")
    title(
      "Set Overlap (MISS vs. Greedy)",
      xlab = "Removals (k)",
      ylab = "Jaccard Index"
    )
    legend(
      "bottomright",
      legend = c("Positive", "Negative"),
      lty = 1,
      col = c("steelblue", "firebrick"),
      bty = "n",
      cex = 0.8
    )
  }
}

# Plot Dinkelbach iterations versus coefficent difference
plot_dinkelbach2 <- function(input, K_MAX = Inf) {
  iter_p <- vapply(input[["miss_pos"]], \(l) l[["iter"]], integer(1L))
  iter_n <- vapply(input[["miss_neg"]], \(l) l[["iter"]], integer(1L))
  values_p <- vapply(input[["miss_pos"]], \(l) l[["best_value"]], numeric(1L))
  values_n <- vapply(input[["miss_neg"]], \(l) l[["best_value"]], numeric(1L))
  diff_p <- diff(c(input[["coef"]], values_p)) |> abs()
  diff_n <- diff(c(input[["coef"]], values_n)) |> abs()
  reg_p <- lm(iter_p ~ diff_p)
  reg_n <- lm(iter_n ~ diff_n)
  reg_all <- lm(c(iter_n, iter_p) ~ c(diff_n, diff_p))

  x_max <- min(length(iter_p), K_MAX)
  y_max <- max(iter_p[seq(x_max)], iter_n[seq(x_max)])
  x_range <- range(
    c(input[["coef"]], values_p[seq(x_max)]) |> diff() |> abs(),
    c(input[["coef"]], values_n[seq(x_max)]) |> diff() |> abs()
  )

  {
    plot.new()
    plot.window(ylim = c(0L, y_max + .5), xlim = x_range)
    axis(1L)
    axis(2L)
    grid()
    points(diff_n, iter_n + rnorm(length(iter_n), 0, .1), col = "steelblue")
    points(diff_p, iter_p + rnorm(length(iter_p), 0, .1), col = "firebrick")
    abline(reg_n, col = "steelblue")
    abline(reg_p, col = "firebrick")
    abline(reg_all)
    title(
      "Dinkelbach Iterations",
      xlab = "Coefficient change (abs)",
      ylab = "Iterations (+ ε)"
    )
    legend(
      "bottomright",
      legend = c("Positive", "Negative"),
      pch = 1,
      col = c("steelblue", "firebrick"),
      bty = "n",
      cex = 0.8
    )
  }
}

# Plot Dinkelbach iterations
plot_dinkelbach <- function(input, K_MAX = Inf) {
  iter_p <- vapply(input[["miss_pos"]], \(l) l[["iter"]], integer(1L))
  iter_n <- vapply(input[["miss_neg"]], \(l) l[["iter"]], integer(1L))

  x_max <- min(length(iter_p), K_MAX)
  y_max <- max(iter_p[seq(x_max)], iter_n[seq(x_max)])

  {
    plot.new()
    plot.window(ylim = c(0L, y_max + .5), xlim = c(1L, x_max))
    axis(1L)
    axis(2L)
    grid()
    if (x_max < 500L) {
      points(
        seq_len(x_max),
        iter_n[seq(x_max)],
        type = "h",
        lwd = 1.0,
        col = "steelblue"
      )
      points(
        seq_len(x_max) + .3,
        iter_p[seq(x_max)],
        type = "h",
        lwd = 1.0,
        col = "firebrick"
      )
    } else {
      lines(
        seq_len(x_max),
        iter_n[seq(x_max)],
        lwd = 1.0,
        col = "steelblue"
      )
      lines(
        seq_len(x_max),
        iter_p[seq(x_max)],
        lwd = 1.0,
        col = "firebrick"
      )
    }
    title("Dinkelbach Iterations", xlab = "Removals (k)", ylab = "Iterations")
    legend(
      "topleft",
      legend = c("Positive", "Negative"),
      lty = 1,
      col = c("steelblue", "firebrick"),
      lwd = 2,
      bty = "n",
      cex = 0.8
    )
  }
}
