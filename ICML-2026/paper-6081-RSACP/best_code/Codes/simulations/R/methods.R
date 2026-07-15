get_standard_quantile <- function(S, alpha) {
  S_ord <- sort(as.numeric(S))
  idx <- ceiling((1 - alpha) * (length(S_ord) + 1))
  if (idx <= 0) return(-Inf)
  if (idx > length(S_ord)) return(Inf)
  S_ord[idx]
}

compute_interval_metrics <- function(S_test, q) {
  data.frame(Cov = mean(S_test <= q), Width = 2 * q)
}
