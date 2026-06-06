M <- 100
n_train <- 14

set.seed(42)
X_tr <- matrix(runif(n_train), ncol=1)
Y_tr <- matrix(runif(n_train * M), nrow=n_train, ncol=M)
f1 <- runif(M)
X_val <- as.numeric(runif(1))
xb <- mean(X_tr); Sg <- var(as.numeric(X_tr)) * (n_train-1)/n_train
sv <- as.numeric(1 + (X_tr - xb) / Sg * (X_val - xb))

n_trials <- 100000
times <- numeric(n_trials)

for (t in 1:n_trials) {
  t0 <- proc.time()["elapsed"]

  # Only Step 2: bias correction
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

cat(sprintf("Step 2 only - Mean: %.4f ms, Median: %.4f ms\n", mean(times), median(times)))

# Also try Step 1 only
n_source <- 138
X_s <- matrix(runif(n_source), ncol=1)
Y_s <- matrix(runif(n_source * M), nrow=n_source, ncol=M)

n_trials2 <- 10000
times2 <- numeric(n_trials2)

for (t in 1:n_trials2) {
  t0 <- proc.time()["elapsed"]

  # Step 1 only
  bigF <- rep(0, M)
  for (i in 1:n_train) bigF <- bigF + sv[i] * Y_tr[i,]
  n_total <- n_train
  xbs <- mean(X_s); Sgs <- var(as.numeric(X_s)) * (n_source-1)/n_source
  svs <- as.numeric(1 + (X_s - xbs) / Sgs * (X_val - xbs))
  for (i in 1:n_source) bigF <- bigF + svs[i] * Y_s[i,]
  f1_est <- bigF / (n_total + n_source)

  t1 <- proc.time()["elapsed"]
  times2[t] <- (t1 - t0) * 1000
}

cat(sprintf("Step 1 only - Mean: %.4f ms, Median: %.4f ms\n", mean(times2), median(times2)))
cat(sprintf("Step 1+2 total estimate: %.4f ms\n", mean(times) + mean(times2)))
