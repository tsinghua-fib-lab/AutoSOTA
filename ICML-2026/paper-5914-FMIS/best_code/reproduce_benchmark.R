# Reproduction script for paper 5914
# Target: median wall-clock time for n=1e6, k=1e5 with Rcpp/Dinkelbach/size-k heap

set.seed(42)

# Source all R functions
sapply(list.files("R", "\\.R$"), function(f) source(paste0("R/", f)))

library("microbenchmark")

# Create data: univariate regression with n=1e6
N <- 1e6
x <- rnorm(N)
y <- x + rnorm(N)
d <- data.frame(y = y, x = x)

# Fit model
m <- lm(y ~ 0 + x, data = d)

# Run microbenchmark for k=1e5 with 100 runs
k <- 1e5

cat(sprintf("Running benchmark: n=%d, k=%d\n", N, k))
cat(sprintf("Data memory: %s\n", format(object.size(d), units = "auto")))

# Time one warm-up run
cat("Warm-up run...\n")
warmup <- find_miss(m, k = k)
cat(sprintf("Warm-up: %d iterations, best_value = %.6f\n",
            warmup$iter, warmup$best_value))

# Microbenchmark with 100 runs
cat("Running microbenchmark (100 runs)...\n")
bm <- microbenchmark(
  find_miss(m, k = k),
  times = 100L
)

# Compute statistics
times_ms <- bm$time / 1e6  # nanoseconds to milliseconds
median_ms <- median(times_ms)
mean_ms <- mean(times_ms)
sd_ms <- sd(times_ms)

cat(sprintf("\nResults (n=%d, k=%d, runs=100):\n", N, k))
cat(sprintf("  Median: %.2f ms\n", median_ms))
cat(sprintf("  Mean:   %.2f ms\n", mean_ms))
cat(sprintf("  SD:     %.2f ms\n", sd_ms))
cat(sprintf("  Min:    %.2f ms\n", min(times_ms)))
cat(sprintf("  Max:    %.2f ms\n", max(times_ms)))
cat(sprintf("\n  Paper reports: < 200 ms median (Ryzen AI Max+ Pro 395)\n"))

# Print CPU info
cat("\nCPU info:\n")
system("cat /proc/cpuinfo | grep \"model name\" | head -1", intern = FALSE)
cat(sprintf("Number of cores: %s\n",
            system("nproc", intern = TRUE)))

# Print detailed timings for logging
cat("\nIndividual timings (ms):\n")
write.table(data.frame(run = seq_along(times_ms), time_ms = times_ms),
            row.names = FALSE, col.names = TRUE)

cat("\nBenchmark complete.\n")
