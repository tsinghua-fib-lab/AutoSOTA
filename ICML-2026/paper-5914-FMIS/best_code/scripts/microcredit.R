# Source functions ---
sapply(
  list.files("R", ".R$"),
  function(f) source(paste0("R/", f))
)

if (!dir.exists("outputs")) {
  dir.create("outputs")
}

# Load data ---
microcredits <- readRDS("data/microcredit.rds")

# Precompute FWL + baseline models for all countries ---
fwls <- lapply(microcredits, \(x) {
  mf <- model.frame(y ~ D, data = x)
  y <- model.response(mf)
  X <- model.matrix(mf, data = x)
  update_fwl(X, y, which(colnames(X) == "D"))
})
mdls <- lapply(fwls, \(d) {
  lm(y ~ X - 1L, data = list(y = d$y, X = d$X))
})
results <- lapply(mdls, \(m) {
  n_max <- length(resid(m)) - 2L
  list(
    miss_p = find_misses(m, K = n_max, sign = 1L),
    greedy_p = greedy_misses(m, K = n_max, sign = 1L),
    miss_n = find_misses(m, K = n_max, sign = -1L),
    greedy_n = greedy_misses(m, K = n_max, sign = -1L)
  )
})

# Produce panelled trace plot for all but MON ---
cairo_pdf("outputs/microcredit-panel.pdf", width = 5, height = 6)
op <- par(mfrow = c(3, 2), mar = c(2, 2, .5, .5))

for (country in names(mdls)[-2L]) {
  message("Processing country: ", country)

  mdl <- mdls[[country]]

  n_max <- length(resid(mdl)) - 2L
  x_seq <- seq(0, n_max)

  miss_p <- results[[country]][["miss_p"]]
  miss_n <- results[[country]][["miss_n"]]
  greedy_p <- results[[country]][["greedy_p"]]
  greedy_n <- results[[country]][["greedy_n"]]

  # Get positive/negative values, sets, and nestedness breaks
  values_p <- c(0, coef(mdl) - vapply(miss_p, `[[`, numeric(1), "best_value"))
  values_n <- c(0, coef(mdl) - vapply(miss_n, `[[`, numeric(1), "best_value"))
  sets_p <- lapply(miss_p, `[[`, "best_S")
  sets_n <- lapply(miss_n, `[[`, "best_S")
  breaks_p <- find_breaks(sets_p)
  breaks_n <- find_breaks(sets_n)

  # Plot settings
  y_quant <- quantile(abs(c(values_n[seq(n_max)], values_p[seq(n_max)])), .80)
  y_lim <- y_quant * c(-1, 1)
  y_lim <- c(0, y_quant)

  # Main trace plot
  {
    plot.new()
    plot.window(ylim = y_lim, xlim = c(0, n_max))
    axis(
      1,
      at = c(0, n_max + 2L),
      labels = c(0, paste0("N = ", n_max + 2L))
    ) # 0 to N
    axis(
      2,
      at = c(0, y_quant %/% 100 * 100),
      labels = c(0, paste0("|", y_quant %/% 100 * 100, "|"))
    )
    grid(nx = 3, ny = 3)
    abline(h = 0, col = "darkgray")
    # title(main = country, xlab = "Removals", ylab = "Coefficient")
    # Exact traces
    lines(x_seq, values_p[x_seq + 1L] |> abs(), col = "#800080", lwd = 1.2)
    lines(x_seq, values_n[x_seq + 1L] |> abs(), col = "#008080", lwd = 1.2)
    points(
      breaks_p,
      values_p[breaks_p] |> abs(),
      pch = 18,
      cex = .8,
      col = "#800080"
    )
    points(
      breaks_n,
      values_n[breaks_n] |> abs(),
      pch = 20,
      cex = .8,
      col = "#008080"
    )
    # Greedy traces
    lines(
      x_seq,
      c(0, coef(mdl) - greedy_n$best_value[seq(n_max)]) |> abs(),
      lty = 3,
      col = "#ff8000"
    )
    lines(
      x_seq,
      c(0, coef(mdl) - greedy_p$best_value[seq(n_max)]) |> abs(),
      lty = 3,
      col = "#ff8000"
    )

    if (country == "BIH") {
      legend(
        "topleft",
        title = country,
        title.font = 2L,
        legend = c("Positive", "Negative", "Greedy"),
        col = c("#008080", "#800080", "#ff8000"),
        lty = c(1, 1, 3),
        pch = c(20, 18, NA),
        lwd = c(1.2, 1.2, 1),
        bty = "n"
      )
    } else {
      legend(
        "topleft",
        title = country,
        title.font = 2L,
        legend = "",
        bty = "n"
      )
    }
  }
}
dev.off()

# Trace and set overlap for MON
country <- "MON"
mdl <- mdls[[country]]

n_max <- 600L # Hand-picked
x_seq <- seq(0, 620)

miss_p <- results[[country]][["miss_p"]]
miss_n <- results[[country]][["miss_n"]]
greedy_p <- results[[country]][["greedy_p"]]
greedy_n <- results[[country]][["greedy_n"]]

# Get positive/negative values, sets, and nestedness breaks
values_p <- c(0, coef(mdl) - vapply(miss_p, `[[`, numeric(1), "best_value"))
values_n <- c(0, coef(mdl) - vapply(miss_n, `[[`, numeric(1), "best_value"))
sets_p <- lapply(miss_p, `[[`, "best_S")
sets_n <- lapply(miss_n, `[[`, "best_S")
breaks_p <- find_breaks(sets_p)
breaks_n <- find_breaks(sets_n)

# Plot settings
y_lim <- c(0, 5.1)

cairo_pdf("outputs/microcredit-MON.pdf", width = 5, height = 6)
op <- par(mfrow = c(2, 1), mar = c(2, 2, .5, .5))
# Main trace plot
{
  plot.new()
  plot.window(ylim = y_lim, xlim = c(0, n_max))
  axis(1, at = c(0, 200, 400, n_max), labels = c(NA, NA, NA, NA))
  axis(2, at = c(0, 2.5, 5))
  abline(v = c(200, 400), lty = 3, col = "lightgray")
  abline(h = 2.5, lty = 3, col = "lightgray")
  abline(h = 0, col = "darkgray")
  # title(main = country, xlab = "Removals", ylab = "Coefficient")
  # Exact traces
  lines(x_seq, values_p[x_seq + 1L] |> abs(), col = "#800080", lwd = 1.2)
  lines(x_seq, values_n[x_seq + 1L] |> abs(), col = "#008080", lwd = 1.2)
  points(
    breaks_p,
    values_p[breaks_p] |> abs(),
    pch = 18,
    cex = 1,
    col = "#800080"
  )
  points(
    breaks_n,
    values_n[breaks_n] |> abs(),
    pch = 20,
    cex = 1,
    col = "#008080"
  )
  # Greedy traces
  lines(
    x_seq[seq(n_max + 1L)],
    c(0, coef(mdl) - greedy_n$best_value[seq(n_max)]) |> abs(),
    lty = 3,
    col = "#ff8000"
  )
  lines(
    x_seq[seq(n_max + 1L)],
    c(0, coef(mdl) - greedy_p$best_value[seq(n_max)]) |> abs(),
    lty = 3,
    col = "#ff8000"
  )

  legend(
    "topleft",
    title = country,
    title.font = 2L,
    legend = c("Positive", "Negative", "Greedy"),
    col = c("#008080", "#800080", "#ff8000"),
    lty = c(1, 1, 3),
    pch = c(20, 18, NA),
    lwd = c(1.2, 1.2, 1),
    bty = "n"
  )
}
# Set divergence
jacc_p <- vapply(
  seq_len(n_max),
  \(k) {
    greedy_set <- greedy_p[["best_S"]][seq_len(k)]
    miss_set <- miss_p[[k]][["best_S"]]
    jaccard_index(greedy_set, miss_set)
  },
  numeric(1L)
)
jacc_n <- vapply(
  seq_len(n_max),
  \(k) {
    greedy_set <- greedy_n[["best_S"]][seq_len(k)]
    miss_set <- miss_n[[k]][["best_S"]]
    jaccard_index(greedy_set, miss_set)
  },
  numeric(1L)
)
{
  plot.new()
  plot.window(ylim = c(0, 1), xlim = c(0L, n_max))
  axis(1L, at = c(0, 200, 400, n_max))
  axis(2L, at = c(0, .5, 1))
  abline(v = c(200, 400), lty = 3, col = "lightgray")
  abline(h = .5, lty = 3, col = "lightgray")
  abline(h = 0, col = "darkgray")
  lines(x_seq[seq(2, n_max + 1)], jacc_p, col = "#800080", lwd = 1.2)
  lines(x_seq[seq(2, n_max + 1)], jacc_n, col = "#008080", lwd = 1.2)
  title(
    xlab = "Removals (k)",
    ylab = "Jaccard Index"
  )
}
dev.off()
