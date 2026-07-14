# Simulation exercises ---
# 1. Sanity check vs. enumeration + greedy
#   - Standard and "adversarial" scenario – check Δ and S
# 2. Benchmark for varying N / K
#   - Report runtimes, feasibility, iterations
#   - Indicate comparison to greedy (implementation-dependent)
# 3. Residualization tests

# Load functions ---
sapply(list.files("R", "\\.R$"), \(f) source(paste0("R/", f)))

# 1. Sanity check ---
sanity_runner <- function(K = 3L, dgp, ...) {
  seed <- sample.int(1e9, 1L)
  set.seed(seed)
  m <- lm(y ~ 0 + x, data = dgp(...))
  res <- list(
    frac = find_miss(m, k = K),
    enum = enumerate_miss(m, k = K, verbose = FALSE),
    grdy = greedy_miss(m, k = K)
  ) |>
    lapply(\(z) {
      list(
        set = sort(z$best_S),
        value = tail(z$best_value, 1L)
      )
    })
  oracle_set <- res$enum$set
  oracle_val <- res$enum$value
  c(
    jaccard_frac_v_oracle = jaccard_index(res$frac$set, oracle_set),
    jaccard_grdy_v_oracle = jaccard_index(res$grdy$set, oracle_set),
    perform_frac_v_oracle = (oracle_val - res$frac$value) / oracle_val,
    perform_grdy_v_oracle = (oracle_val - res$grdy$value) / oracle_val,
    seed_used = seed
  )
}

# Helper functions
sanity_check <- function(dgp, K = 3L, reps = 1000L) {
  s <- replicate(reps, sanity_runner(K, dgp = dgp))
  apply(s[1L:4L, , drop = FALSE], 1L, summary)
}

# DGPs ---
dgp_linear <- function(N = 25L, treatment = rnorm, error = rnorm, beta = 1) {
  x <- treatment(N)
  y <- beta * x + error(N)
  data.frame(y = y, x = x)
}
# Heteroskedasticity
dgp_heterosk <- function(N = 25L, treatment = rnorm, error = rnorm, beta = 1) {
  x <- treatment(N)
  y <- beta * x + (1 + abs(x)) * error(N)
  data.frame(y = y, x = x)
}
# Two components
dgp_2comp <- function(beta = 0.8, n_treat = 2L, n_filler = 8L) {
  x_treat <- runif(n_treat, 0, 2)
  y_treat <- beta * x_treat + rnorm(n_treat, 0, .1)
  x_filler <- runif(n_filler, 3, 4)
  y_filler <- rnorm(n_filler, 0, .1)
  x <- c(x_treat, x_filler)
  y <- c(y_treat, y_filler)
  data.frame(y = y, x = x)
}
# Fat tails and a sine function
dgp_sinetails <- function(N = 25L, df = 2L) {
  x <- sin(rt(N, df))
  y <- x + rt(N, df)
  data.frame(y = y, x = x)
}
# M random components
dgp_mixture <- function(N = 25L, M = 10L) {
  comp <- sample.int(M, N, replace = TRUE)
  loc <- rgamma(M, .5, .5)
  scl <- rgamma(M, 2, 2)
  b <- rnorm(M, 2, 5)
  x <- rnorm(N, loc[comp], scl[comp])
  y <- x * b[comp] + rnorm(N)
  data.frame(y = y, x = x)
}
# Nonlinear polynomial with interaction-like curvature
dgp_poly <- function(N = 25L, beta = c(1, 0.5, -0.25), error = rnorm) {
  x <- rnorm(N)
  y <- beta[1L] * x + beta[2L] * x^2 + beta[3L] * x^3 + error(N)
  data.frame(y = y, x = x)
}
# Piecewise linear / threshold (kink) model
dgp_poly <- function(N = 25L, beta = c(-.5, 0.5, 2), error = rnorm) {
  x <- rnorm(N)
  y <- beta[1L] + beta[2L] * x + beta[3L] * pmax(0, x) + error(N)
  data.frame(y = y, x = x)
}
# Logistic
dgp_logistic <- function(N = 25L, beta = 2) {
  x <- rnorm(N)
  y <- plogis(beta * x) + rnorm(N)
  data.frame(y = y, x = x)
}
# Binary outcomes
dgp_poisson <- function(N = 25L, beta = 0.5) {
  x <- rnorm(N)
  y <- rpois(N, exp(beta * x))
  data.frame(y = y, x = x)
}
# Endogeneity
dgp_endog <- function(N = 25L, beta = 1, rho = 0.7) {
  z <- rnorm(N) # latent factor drives both x and y
  x <- z + rnorm(N)
  u <- rho * z + sqrt(1 - rho^2) * rnorm(N)
  y <- beta * x + u
  data.frame(y = y, x = x)
}
# Time series-ish
dgp_ar1 <- function(N = 25L, beta = 1, phi = 0.7, sd = 1) {
  x <- rnorm(N)
  e <- numeric(N)
  e[1] <- rnorm(1, 0, sd / sqrt(1 - phi^2))
  for (t in 2:N) {
    e[t] <- phi * e[t - 1] + rnorm(1, 0, sd)
  }
  y <- beta * x + e
  data.frame(y = y, x = x)
}
# Heterogeneous treatment effect
dgp_hetero <- function(N = 25L, beta = 1, gamma = 1, error = rnorm) {
  w <- rnorm(N)
  x <- rnorm(N)
  y <- (beta + gamma * w) * x + error(N)
  data.frame(y = y, x = x)
}

# Run the sanity checks ---
DGP_names <- ls(pattern = "^dgp_")
DGP_funs <- mget(DGP_names, mode = "function")
sanity_results <- lapply(DGP_funs, sanity_check, reps = 10000L)

# 2. Benchmarks ---

library("microbenchmark")

dgp <- \(N) {
  x <- rnorm(N)
  y <- x + rnorm(N)
  data.frame(y = y, x = x)
}

base <- c(1, 2, 5)
Ns <- as.integer(outer(base, 10^(1:6), `*`))
Ks <- as.integer(outer(base, 10^(0:5), `*`))

timings <- do.call(
  rbind,
  lapply(Ns[Ns <= 1e6], \(N) {
    m <- lm(y ~ 0 + x, data = dgp(N = N))
    do.call(
      rbind,
      lapply(Ks[Ks < N], \(k) {
        data.frame(
          N = N,
          K = k,
          times = microbenchmark(find_miss(m, k = k))$time,
          iterations = find_miss(m, k = k)$iter
        )
      })
    )
  })
)
timings$logchoose <- lchoose(timings$N, timings$K)
timings$logtimes_enum <- log(min(timings$times)) + timings$logchoose
timings$logtimes <- log(timings$times)

saveRDS(timings, "outputs/timings.rds")

# Check greedy for a subset ---
timings_greedy <- do.call(
  rbind,
  lapply(c(100L, 1000L, 10000L), \(N) {
    m <- lm(y ~ 0 + x, data = dgp(N = N))
    do.call(
      rbind,
      lapply(c(10L, 50L, 100L), \(k) {
        data.frame(
          N = N,
          K = k,
          times = microbenchmark(greedy_miss(m, k = k))$time
        )
      })
    )
  })
)

# Check what's feasible on my machine ---
N <- 1e9 # This requires *a lot* of RAM
k <- 1e7 # 1e8 runs for a long time
d <- dgp(N = N)
m <- lm(y ~ 0 + x, data = d)
s <- get_miss_solver(X = d[["x"]], R = resid(m))
# Clean up the items
rm(m, d)
gc()
microbenchmark(
  {
    res <- s(k = k, lambda = 0)
  },
  times = 1L # Increase for sensible bounds
)
res[c("best_value", "iter")] # 3–5 iterations for all
# Here's suggestive results:
# 1e8, 1e6 ~5s
# 1e8, 1e7 ~60s
# 1e9, 1e6 ~10s
# 1e9, 1e7 ~80s
# 1e9, 1e8 ~900s

# Try to mess up convergence ---
N <- 1e8
k <- 1e7
d <- dgp(N = N)
m <- lm(y ~ 0 + x, data = d)
s <- get_miss_solver(X = d[["x"]], R = resid(m))
res <- s(k = k, lambda = 0)
res[c("best_value", "iter")]

# Check convergence speeds for adversarial starts
sapply(c(0, 1e3, 1e6, 1e9, 1e12, 1e18), \(lambda) {
  s(k = k, lambda = lambda)[c("best_value", "iter")]
}) # We pretty much lose an iteration for terrible starts
# Trace the value along iterations
sapply(seq(10L), \(max_iter) {
  get_miss_solver(
    X = d[["x"]],
    R = resid(m),
    max_iter = max_iter
  )(k = k, lambda = 1e18)[c("best_value", "iter")]
}) # Second step gets the first digit, third the fourth

# Plot timings ---
levels <- c(
  "< 1 second",
  "< 1 hour",
  "< 1 year",
  "< 1 earth",
  "geological"
)
df <- timings |>
  dplyr::group_by(N, K) |>
  dplyr::summarise(
    median_ms = median(times / 1e6),
    log_enum_ms = mean(logtimes_enum - log(1e6))
  ) |>
  dplyr::mutate(
    enum_scale = dplyr::case_when(
      !is.finite(log_enum_ms) ~ "non-finite",
      log_enum_ms < log(60e3) ~ "< 1 second",
      log_enum_ms < log(24 * 3600e3) ~ "< 1 hour",
      log_enum_ms < log(365 * 24 * 3600e3) ~ "< 1 year",
      log_enum_ms < log(4.54e9 * 365.25 * 24 * 3600 * 1e3) ~ "< 1 earth",
      TRUE ~ "geological"
    )
  ) |>
  dplyr::mutate(
    enum_scale = factor(
      enum_scale,
      levels = levels,
      ordered = TRUE
    )
  )

shape_vals <- c(
  "< 1 second" = "s",
  "< 1 hour" = "h",
  "< 1 year" = "y",
  "< 1 earth" = "e",
  "geological" = "x"
)
shape_cols <- c(
  "< 1 second" = "gray20",
  "< 1 hour" = "gray30",
  "< 1 year" = "gray40",
  "< 1 earth" = "gray60",
  "geological" = "gray100"
)
breaks <- 1000 * c(0, .001, .005, .01, .05, .1, .2, .5, .8)
colors <- viridisLite::mako(length(breaks), begin = .7) |> rev()
label_axes <- function(x) {
  out <- scales::scientific(as.numeric(x))
  out <- sub("e([+-])00$", "e\\10", out)
  out <- sub("e\\+0+([1-9])", "e+\\1", out)
  out <- sub("e-0+([1-9])", "e-\\1", out)
  out
}

df |>
  dplyr::mutate(
    median_bin = cut(
      median_ms,
      breaks = breaks,
      include.lowest = TRUE,
      right = FALSE
    )
  ) |>
  dplyr::filter(
    substr(K, 1, 1) != 2L & substr(N, 1, 1) != 2L,
    K < 5e5
  ) |>
  ggplot(aes(y = factor(N), x = factor(K))) +
  geom_tile(aes(fill = median_bin), col = "white") +
  scale_fill_manual(values = colors) +
  geom_point(aes(shape = enum_scale, col = enum_scale), size = 3) +
  scale_x_discrete(labels = label_axes) +
  scale_y_discrete(labels = label_axes) +
  scale_shape_manual(values = shape_vals) +
  scale_color_manual(values = shape_cols) +
  labs(
    y = "n (sample size)",
    x = "k (set size)",
    fill = "Runtime (ms)",
    color = "Enumeration",
    shape = "Enumeration"
  ) +
  guides(
    fill = guide_legend(order = 1),
    shape = guide_legend(order = 2),
    color = guide_legend(order = 2)
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major = element_line(
      colour = "lightgray",
      linewidth = .3,
      linetype = "dotted"
    ),
    axis.text.x = element_text(angle = -90)
  )

ggplot2::ggsave(
  "outputs/runtime_heatmap.pdf",
  device = cairo_pdf,
  width = 6.75,
  height = 4.5,
  units = "in"
)
