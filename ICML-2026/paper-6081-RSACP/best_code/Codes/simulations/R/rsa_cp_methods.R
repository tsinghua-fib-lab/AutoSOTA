# RSA-CP score-level methods.
# RSA-CP (OT) aligns real calibration scores onto the reference-score scale
# with a one-dimensional barycentric OT map and then applies a candidate-
# specific Beta-Binomial rank-window calibration rule.

rsa_ot_map <- function(S_new, S_source, S_target) {
  S_new <- as.numeric(S_new)
  src <- sort(as.numeric(S_source))
  tgt <- sort(as.numeric(S_target))

  m <- length(src)
  N <- length(tgt)

  if (m == 1) return(rep(mean(tgt), length(S_new)))

  num <- den <- rep(0, m)
  i <- j <- 1
  r_src <- 1 / m
  r_tgt <- 1 / N

  while (i <= m && j <= N) {
    delta <- min(r_src, r_tgt)

    num[i] <- num[i] + delta * tgt[j]
    den[i] <- den[i] + delta

    r_src <- r_src - delta
    r_tgt <- r_tgt - delta

    if (r_src <= 1e-12) {
      i <- i + 1
      if (i <= m) r_src <- 1 / m
    }

    if (r_tgt <= 1e-12) {
      j <- j + 1
      if (j <= N) r_tgt <- 1 / N
    }
  }

  T_src <- cummax(num / den)

  df <- data.frame(src = src, mapped = T_src)
  df <- aggregate(mapped ~ src, data = df, FUN = mean)
  df$mapped <- cummax(df$mapped)

  if (nrow(df) == 1) {
    return(rep(df$mapped[1], length(S_new)))
  }

  approx(x = df$src, y = df$mapped, xout = S_new, rule = 2)$y
}

q_betabin <- function(p, N, a, b) {
  x <- 0:N
  logpmf <- lchoose(N, x) + lbeta(x + a, N - x + b) - lbeta(a, b)
  pmf <- exp(logpmf - max(logpmf))
  pmf <- pmf / sum(pmf)
  x[which(cumsum(pmf) >= p)[1]]
}

.betabin_cache <- new.env(parent = emptyenv())

q_betabin_cached <- function(p, N, a, b) {
  key <- paste(signif(p, 12), N, a, b, sep = "_")
  if (!exists(key, envir = .betabin_cache, inherits = FALSE)) {
    assign(key, q_betabin(p, N, a, b), envir = .betabin_cache)
  }
  get(key, envir = .betabin_cache, inherits = FALSE)
}

A_value <- function(A_sorted, idx) {
  if (idx <= 0) return(-Inf)
  if (idx > length(A_sorted)) return(Inf)
  A_sorted[idx]
}

get_rsacp_decision <- local({
  qfun <- q_betabin_cached

  function(S_new, S_real, S_ref, alpha, beta, use_ot = TRUE) {
    m <- length(S_real)
    N <- length(S_ref)

    if (use_ot) {
      Z_real <- rsa_ot_map(S_real, S_real, S_ref)
      Z_new <- rsa_ot_map(S_new, S_real, S_ref)
    } else {
      Z_real <- as.numeric(S_real)
      Z_new <- as.numeric(S_new)
    }

    A <- sort(c(Z_real, S_ref))
    J <- ceiling((1 - alpha) * (m + N + 1))

    Z_real_ord <- sort(Z_real)
    # Equivalent to k = 1 + sum(Z_real <= Z_new[i]) for each candidate.
    k_vec <- findInterval(Z_new, Z_real_ord, rightmost.closed = TRUE) + 1
    q_by_k <- rep(NA_real_, m + 1)

    for (k in unique(k_vec)) {
      b_minus <- qfun(beta / 2, N, k, m + 2 - k)
      b_plus <- qfun(1 - beta / 2, N, k, m + 2 - k)

      q_by_k[k] <- max(
        min(A_value(A, k + b_plus), A_value(A, J)),
        A_value(A, k + b_minus)
      )
    }

    data.frame(
      score = S_new,
      mapped_score = Z_new,
      rank_k = k_vec,
      q_rsa_mapped = q_by_k[k_vec],
      include = Z_new <= q_by_k[k_vec]
    )
  }
})

get_rsacp_quantile <- function(S_real, S_ref, alpha, beta,
                               use_ot = TRUE, grid_size = 5000,
                               max_expand = 5) {
  upper <- max(c(S_real, S_ref), na.rm = TRUE)
  upper <- max(upper * 1.5, upper + 1e-8)

  for (expand_iter in 1:max_expand) {
    S_grid <- seq(0, upper, length.out = grid_size)

    dec <- get_rsacp_decision(
      S_new = S_grid,
      S_real = S_real,
      S_ref = S_ref,
      alpha = alpha,
      beta = beta,
      use_ot = use_ot
    )

    if (!any(dec$include)) return(0)

    if (!all(dec$include)) {
      return(max(S_grid[dec$include], na.rm = TRUE))
    }

    upper <- upper * 2
  }

  Inf
}
