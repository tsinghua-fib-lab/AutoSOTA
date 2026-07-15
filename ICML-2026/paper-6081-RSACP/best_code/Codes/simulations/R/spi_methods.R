# True SPI fast-form score threshold.

order_value <- function(x_sorted, idx) {
  if (idx <= 0) return(-Inf)
  if (idx > length(x_sorted)) return(Inf)
  x_sorted[idx]
}

spi_windows <- function(m, N, beta) {
  Rminus <- Rplus <- integer(m + 1)
  k_grid <- 1:(N + 1)
  denom <- lchoose(N + m + 1, m + 1)

  for (r in 1:(m + 1)) {
    logpmf <- lchoose(k_grid + r - 2, r - 1) +
      lchoose(N + m - k_grid - r + 2, m - r + 1) -
      denom

    pmf <- exp(logpmf - max(logpmf))
    pmf <- pmf / sum(pmf)

    cdf <- cumsum(pmf)
    cdf_prev <- c(0, cdf[-length(cdf)])

    Rminus[r] <- max(k_grid[cdf_prev <= beta / 2])
    Rplus[r] <- min(k_grid[cdf >= 1 - beta / 2])
  }

  list(Rminus = Rminus, Rplus = Rplus)
}

get_spi_quantile <- function(S_real, S_syn, alpha, beta) {
  m <- length(S_real)
  N <- length(S_syn)

  S_real_ord <- c(sort(as.numeric(S_real)), Inf)
  S_syn_ord <- sort(as.numeric(S_syn))

  J <- ceiling((1 - alpha) * (N + 1))
  q_syn_prime <- if (J + 1 <= N) S_syn_ord[J + 1] else Inf

  W <- spi_windows(m, N, beta)

  Rtilde_minus <- max(c(which(W$Rminus <= J), 0))
  Rtilde_plus <- max(c(which(W$Rplus <= J), 0))

  max(
    min(q_syn_prime, order_value(S_real_ord, Rtilde_minus)),
    order_value(S_real_ord, Rtilde_plus)
  )
}
