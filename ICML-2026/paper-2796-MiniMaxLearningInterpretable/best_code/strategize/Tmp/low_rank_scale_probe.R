softplus <- function(x) pmax(x, 0) + log1p(exp(-abs(x)))
softclip <- function(x, bound = 1.5, softness = bound / 6) {
  low <- -bound
  high <- bound
  low + softness * softplus((x - low) / softness) -
    softness * softplus((x - high) / softness)
}

simulate_scale <- function(D = 16L,
                           rank = 4L,
                           n = 200000L,
                           rms_final = 1,
                           seed = 1L) {
  set.seed(seed)
  weight_sd_scale <- sqrt(2) / sqrt(D)
  cross_weight_sd_scale <- weight_sd_scale / sqrt(D)
  gate_sd_scale <- 0.1 * sqrt(2)
  tau_rc <- cross_weight_sd_scale
  rc_out_sd <- 0.1 / sqrt(rank)

  R <- matrix(rnorm(n * D, sd = rms_final), n, D)
  L <- matrix(rnorm(n * D, sd = rms_final), n, D)
  G <- matrix(rnorm(n * D, sd = rms_final), n, D)
  W_r <- matrix(rnorm(D * rank, sd = tau_rc), D, rank)
  W_c <- matrix(rnorm(D * rank, sd = tau_rc), D, rank)
  W_o <- matrix(rnorm(rank, sd = rc_out_sd), rank, 1)
  alpha <- abs(rnorm(1, sd = gate_sd_scale))

  rc_one <- function(C) {
    drop(((R %*% W_r) * (C %*% W_c)) %*% W_o) * alpha
  }
  rc_left <- rc_one(L)
  rc_right <- rc_one(G)
  rc_delta <- rc_left - rc_right

  M_raw <- matrix(rnorm(D * D, sd = tau_rc), D, D)
  M_cross <- 0.5 * (M_raw - t(M_raw))
  W_cross_out <- rnorm(1, sd = 0.25)
  cross <- rowSums((L %*% M_cross) * G) * W_cross_out

  summarize <- function(x) {
    c(
      mean = mean(x),
      sd = sd(x),
      rms = sqrt(mean(x^2)),
      q50 = unname(quantile(abs(x), 0.50)),
      q90 = unname(quantile(abs(x), 0.90)),
      q99 = unname(quantile(abs(x), 0.99)),
      max_abs = max(abs(x))
    )
  }

  rbind(
    rc_left = summarize(rc_left),
    rc_delta = summarize(rc_delta),
    cross_term = summarize(cross),
    softclip_raw_100 = summarize(softclip(rep(100, n), 1.5, 0.25)),
    softclip_rc_delta = summarize(softclip(rc_delta, 1.5, 0.25))
  )
}

configs <- expand.grid(
  D = c(8L, 16L, 32L),
  rank = c(2L, 4L, 16L),
  KEEP.OUT.ATTRS = FALSE
)

for (i in seq_len(nrow(configs))) {
  D <- configs$D[[i]]
  rank <- configs$rank[[i]]
  cat("\nD=", D, " rank=", rank, "\n", sep = "")
  print(round(simulate_scale(D = D, rank = rank, n = 50000L, seed = 100 + i), 6))
}

cat("\nSoftClip forward map at bound=1.5, softness=0.25:\n")
x <- c(-100, -10, -3, -1.5, -1, 0, 1, 1.5, 3, 10, 100)
print(data.frame(raw = x, softclip = round(softclip(x), 6), prob = round(plogis(softclip(x)), 6)))

