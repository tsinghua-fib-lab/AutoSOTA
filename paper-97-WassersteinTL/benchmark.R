M <- 100
n_source <- 138
n_train <- 14

set.seed(42)
X_tr <- matrix(runif(n_train), ncol=1)
Y_tr <- matrix(runif(n_train * M), nrow=n_train, ncol=M)
X_s <- matrix(runif(n_source), ncol=1)
Y_s <- matrix(runif(n_source * M), nrow=n_source, ncol=M)
X_val <- as.numeric(runif(1))

n_trials <- 10000
times <- numeric(n_trials)

for (t in 1:n_trials) {
  t0 <- proc.time()["elapsed"]

  # f1_hat (Step 1)
  bigF <- rep(0, M); n_total <- 0
  xb <- mean(X_tr); Sg <- var(as.numeric(X_tr)) * (n_train-1)/n_train
  if (Sg < 1e-12) Sg <- 1e-6
  sv <- as.numeric(1 + (X_tr - xb) / Sg * (X_val - xb))
  for (i in 1:n_train) bigF <- bigF + sv[i] * Y_tr[i,]
  n_total <- n_train

  xbs <- mean(X_s); Sgs <- var(as.numeric(X_s)) * (n_source-1)/n_source
  if (Sgs < 1e-12) Sgs <- 1e-6
  svs <- as.numeric(1 + (X_s - xbs) / Sgs * (X_val - xbs))
  for (i in 1:n_source) bigF <- bigF + svs[i] * Y_s[i,]
  n_total <- n_total + n_source
  f1 <- bigF / n_total

  # Bias correction (Step 2)
  f <- f1
  for (iter in 1:200) {
    dm <- -sweep(Y_tr, 2, f, FUN="-")
    wdm <- sweep(dm, 1, sv, FUN="*")
    grad <- colSums(wdm) + 0.25 * (f - f1)
    fn <- f - 0.1 * grad
    if (sqrt(sum((fn-f)^2)) < 1e-3) { f <- fn; break }
    f <- fn
  }

  t1 <- proc.time()["elapsed"]
  times[t] <- (t1 - t0) * 1000
}

cat(sprintf("Mean time per prediction: %.4f ms\n", mean(times)))
cat(sprintf("Median: %.4f ms\n", median(times)))
cat(sprintf("SD: %.4f ms\n", sd(times)))
