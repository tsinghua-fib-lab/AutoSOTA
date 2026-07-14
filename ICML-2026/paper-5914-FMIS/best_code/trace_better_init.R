set.seed(42)
sapply(list.files("R", "\\.R$"), function(f) source(paste0("R/", f)))

N <- 1e6
x <- rnorm(N)
y <- x + rnorm(N)
d <- data.frame(y=y, x=x)
m <- lm(y ~ 0 + x, data=d)
xr <- get_lm_xr(m)
num <- xr$X * xr$R
den <- xr$X^2
tot <- sum(den)

# Better initialization: top-k by individual ratio, then compute lambda
cat("=== Better init: top-k by num/(tot-den) ratio ===\n")
ratios <- num / (tot - den)
S_init <- order_partial(ratios, 1e5, TRUE)
num_init <- sum(num[S_init])
den_init <- sum(den[S_init])
lambda_better <- num_init / (tot - den_init)
cat(sprintf("Better lambda_init: %.12f (from top-k ratio set)\n", lambda_better))
cat(sprintf("Original lambda_init: %.12f (max ratio)\n", max(ratios)))

# Trace convergence with better init
lambda <- lambda_better
S_prev <- integer(0)
for (iter in 1:10) {
  scores <- num + lambda * den
  S_i <- order_partial(scores, 1e5, TRUE)
  num_i <- sum(num[S_i])
  tot_i <- tot - sum(den[S_i])
  lambda_i <- num_i / tot_i
  
  same <- identical(S_i, S_prev)
  delta <- abs(lambda_i - lambda)
  
  cat(sprintf("  iter %d: lambda=%.12f lambda_new=%.12f delta=%.2e same_set=%s\n",
      iter, lambda, lambda_i, delta, same))
  
  if (same || delta < 1e-16) {
    cat(sprintf("  Converged at iter %d\n", iter))
    break
  }
  lambda <- lambda_i
  S_prev <- S_i
}
