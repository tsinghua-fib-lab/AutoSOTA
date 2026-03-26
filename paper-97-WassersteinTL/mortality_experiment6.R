################################################################################
# WaTL Mortality Data - Testing different evaluation strategies
# Strategy 1: Train on n_target, test on all n_dev (including training)
# Strategy 2: Train on n_target, test on remaining n_dev - n_target
# Strategy 3: Leave-one-out CV across all n_dev
################################################################################

library(wpp2015)
library(osqp)
library(Matrix)
library(pracma)

set.seed(42)
M <- 100

cat("Loading WPP2015 data...\n")
data(UNlocations)
data(mxM); data(mxF)
data(tfr)

to_matrix <- function(x) {
  if (is.vector(x)) matrix(x, ncol=1)
  else if (is.data.frame(x)) as.matrix(x)
  else x
}

compute_L2_distance <- function(vec1, vec2) sqrt(mean((vec1 - vec2)^2))

compute_f1_hat <- function(source_data_list, X_value, M, X_t = NULL, Y_t = NULL) {
  K <- length(source_data_list)
  bigF <- rep(0, M)
  n_total <- 0
  if (!is.null(X_t) && !is.null(Y_t)) {
    X_t_mat <- to_matrix(X_t)
    n_t <- nrow(X_t_mat)
    xb <- mean(X_t_mat); Sg <- var(as.numeric(X_t_mat)) * (n_t-1)/n_t
    if (Sg < 1e-12) Sg <- 1e-6
    sv <- as.numeric(1 + (X_t_mat - xb) / Sg * (X_value - xb))
    for (i in seq_len(n_t)) bigF <- bigF + sv[i] * Y_t[i,]
    n_total <- n_total + n_t
  }
  for (k in seq_len(K)) {
    Xs <- source_data_list[[k]]$X_s; Ys <- source_data_list[[k]]$Y_s
    Xs_mat <- to_matrix(Xs); nk <- nrow(Xs_mat)
    xbs <- mean(Xs_mat); Sgs <- var(as.numeric(Xs_mat)) * (nk-1)/nk
    if (Sgs < 1e-12) Sgs <- 1e-6
    svs <- as.numeric(1 + (Xs_mat - xbs) / Sgs * (X_value - xbs))
    for (i in seq_len(nk)) bigF <- bigF + svs[i] * Ys[i,]
    n_total <- n_total + nk
  }
  bigF / n_total
}

compute_f_L2 <- function(Y_t, s_vec, f1_hat, lambda, M, max_iter=1000, step_size=0.5, tol=1e-8) {
  f <- as.numeric(f1_hat)
  for (iter in 1:max_iter) {
    dm <- -sweep(Y_t, 2, f, FUN="-")
    wdm <- sweep(dm, 1, s_vec, FUN="*")
    grad <- colSums(wdm) + lambda * (f - f1_hat)
    fn <- f - step_size * grad
    if (sqrt(sum((fn-f)^2)) < tol) { f <- fn; break }
    f <- fn
  }
  as.numeric(f)
}

age_starts <- c(0, 1, seq(5, 95, by=5), 100)
age_ends   <- c(1, 5, seq(10, 95, by=5), 100, 110)
age_widths <- age_ends - age_starts

mx_to_qf_norm <- function(mx, M_grid=100, max_age=110) {
  n_ages <- length(mx)
  wd <- age_widths[1:n_ages]
  ae <- age_ends[1:n_ages]; as_ <- age_starts[1:n_ages]
  qx <- pmin(1-exp(-mx*wd), 1); qx[is.na(qx)|qx<0] <- 0
  lx <- cumprod(c(1, 1-qx))
  dx <- pmax(-diff(lx), 0)
  tot <- sum(dx); if (tot < 1e-10) return(rep(0.5, M_grid))
  prop <- dx / tot
  n_fine <- 3000; af <- seq(0, max_age, length.out=n_fine)
  cdf <- numeric(n_fine)
  for (j in 1:n_fine) {
    a <- af[j]; cum <- 0
    for (i in 1:n_ages) {
      if (a >= ae[i]) cum <- cum + prop[i]
      else if (a > as_[i]) { cum <- cum + prop[i] * (a-as_[i]) / wd[i]; break }
      else break
    }
    cdf[j] <- min(cum, 1)
  }
  p <- seq(1/M_grid, 1-1/M_grid, length.out=M_grid)
  qf <- sapply(p, function(pp) {
    if (pp <= cdf[1]) af[1]
    else if (pp >= cdf[n_fine]) af[n_fine]
    else approx(cdf, af, xout=pp, rule=2)$y
  })
  qf / max_age
}

process_cc <- function(cc, year_col="2010-2015") {
  mx_m <- mxM[mxM$country_code==cc, ]; mx_f <- mxF[mxF$country_code==cc, ]
  if (nrow(mx_m)==0 || !(year_col %in% names(mx_m))) return(NULL)
  an <- suppressWarnings(as.numeric(as.character(mx_m$age)))
  an[is.na(an)] <- 100; ord <- order(an)
  mx_m <- mx_m[ord,]; mx_f <- mx_f[ord,]
  rm <- as.numeric(mx_m[[year_col]]); rf <- as.numeric(mx_f[[year_col]])
  if (any(is.na(rm))||any(is.na(rf))||length(rm)<20) return(NULL)
  n_use <- min(length(rm), length(age_starts))
  rates <- (rm[1:n_use] + rf[1:n_use]) / 2
  qf <- tryCatch(mx_to_qf_norm(rates, M), error=function(e) NULL)
  if (is.null(qf)||any(is.na(qf))||any(!is.finite(qf))) return(NULL)
  # TFR predictor
  tr <- tfr[tfr$country_code==cc, ]
  if (nrow(tr)==0||!(year_col %in% names(tr))) return(NULL)
  tv <- as.numeric(tr[[year_col]]); if (is.na(tv)) return(NULL)
  list(cc=cc, qf=qf, tfr=tv)
}

indiv <- UNlocations[UNlocations$location_type==4, c("name","country_code","agcode_901","agcode_902")]
mx_ccs <- unique(mxM$country_code)
in_both <- intersect(indiv$country_code, mx_ccs)
dev_codes  <- indiv[indiv$country_code %in% in_both & indiv$agcode_901 > 0, "country_code"]
less_codes <- indiv[indiv$country_code %in% in_both & indiv$agcode_902 > 0, "country_code"]

dev_res  <- Filter(Negate(is.null), lapply(dev_codes, process_cc))
less_res <- Filter(Negate(is.null), lapply(less_codes, process_cc))

cat("Developed:", length(dev_res), "Developing:", length(less_res), "\n")

# Use top 24 developed (lowest TFR) and all 156 developing
dev_tfr <- sapply(dev_res, function(x) x$tfr)
dev_24 <- dev_res[order(dev_tfr)[1:24]]

X_t_all <- matrix(sapply(dev_24, function(x) x$tfr), ncol=1)
Y_t_all <- do.call(rbind, lapply(dev_24, function(x) x$qf))
X_s <- matrix(sapply(less_res, function(x) x$tfr), ncol=1)
Y_s <- do.call(rbind, lapply(less_res, function(x) x$qf))

all_tfr <- c(X_t_all, X_s)
tmin <- min(all_tfr); tmax <- max(all_tfr)
X_t_norm <- (X_t_all - tmin) / (tmax - tmin)
X_s_norm  <- (X_s - tmin) / (tmax - tmin)

source_data <- list(list(X_s=X_s_norm, Y_s=Y_s))

n_dev <- nrow(X_t_all)

################################################################################
# Try multiple strategies for n_target=14
################################################################################

run_strategy <- function(n_target_train, eval_mode="test_only", n_reps=500, lam=0.25) {
  set.seed(42)
  rmspr_v <- numeric(n_reps)
  time_v  <- numeric(n_reps)

  for (rep in 1:n_reps) {
    train_idx <- sample(n_dev, n_target_train)
    test_idx  <- setdiff(1:n_dev, train_idx)

    if (eval_mode == "test_only") {
      eval_idx <- test_idx
    } else if (eval_mode == "all") {
      eval_idx <- 1:n_dev
    } else if (eval_mode == "train_only") {
      eval_idx <- train_idx
    }

    X_tr <- X_t_norm[train_idx, , drop=FALSE]
    Y_tr <- Y_t_all[train_idx, , drop=FALSE]
    X_ev <- X_t_norm[eval_idx, , drop=FALSE]
    Y_ev <- Y_t_all[eval_idx, , drop=FALSE]

    n_ev <- nrow(X_ev)
    dvals <- numeric(n_ev)
    t0 <- proc.time()["elapsed"]

    for (i in 1:n_ev) {
      XV <- as.numeric(X_ev[i,])
      f1 <- compute_f1_hat(source_data, XV, M, X_tr, Y_tr)
      xb <- mean(X_tr); Sg <- var(as.numeric(X_tr))*(nrow(X_tr)-1)/nrow(X_tr)
      if (Sg < 1e-12) Sg <- 1e-6
      sv <- as.numeric(1 + (X_tr - xb) / Sg * (XV - xb))
      f2 <- compute_f_L2(Y_t=Y_tr, s_vec=sv, f1_hat=f1, lambda=lam, M=M,
                         max_iter=200, step_size=0.1, tol=1e-3)
      dvals[i] <- compute_L2_distance(Y_ev[i,], f2)
    }

    t1 <- proc.time()["elapsed"]
    rmspr_v[rep] <- mean(dvals)
    time_v[rep]  <- (t1 - t0) * 1000 / n_ev
  }

  list(rmspr=mean(rmspr_v), rmspr_se=sd(rmspr_v)/sqrt(n_reps),
       time=mean(time_v), time_se=sd(time_v)/sqrt(n_reps))
}

cat("\n=== Strategy: test on held-out samples ===\n")
r1 <- run_strategy(14, "test_only", 500, 0.25)
cat(sprintf("RMSPR=%.5f (SE=%.5f), Time=%.3f ms\n", r1$rmspr, r1$rmspr_se, r1$time))

cat("\n=== Strategy: test on ALL target samples (train+test) ===\n")
r2 <- run_strategy(14, "all", 500, 0.25)
cat(sprintf("RMSPR=%.5f (SE=%.5f), Time=%.3f ms\n", r2$rmspr, r2$rmspr_se, r2$time))

cat("\n=== Strategy: LOO cross-validation on 24 countries ===\n")
set.seed(42)
dvals_loo <- numeric(n_dev)
t0_loo <- proc.time()["elapsed"]

for (i in 1:n_dev) {
  train_idx <- setdiff(1:n_dev, i)
  # Use only 14 of the 23 remaining for training (sample)
  if (length(train_idx) > 14) train_idx <- sample(train_idx, 14)

  X_tr <- X_t_norm[train_idx, , drop=FALSE]
  Y_tr <- Y_t_all[train_idx, , drop=FALSE]
  XV <- as.numeric(X_t_norm[i, ])

  f1 <- compute_f1_hat(source_data, XV, M, X_tr, Y_tr)
  xb <- mean(X_tr); Sg <- var(as.numeric(X_tr))*(nrow(X_tr)-1)/nrow(X_tr)
  if (Sg < 1e-12) Sg <- 1e-6
  sv <- as.numeric(1 + (X_tr - xb) / Sg * (XV - xb))
  f2 <- compute_f_L2(Y_t=Y_tr, s_vec=sv, f1_hat=f1, lambda=0.25, M=M,
                     max_iter=200, step_size=0.1, tol=1e-3)
  dvals_loo[i] <- compute_L2_distance(Y_t_all[i,], f2)
}

t1_loo <- proc.time()["elapsed"]
cat(sprintf("LOO RMSPR=%.5f, Time=%.3f ms\n",
            mean(dvals_loo), (t1_loo - t0_loo) * 1000 / n_dev))

cat("\n=== SUMMARY ===\n")
cat(sprintf("Strategy 1 (test only):     RMSPR=%.5f, Time=%.3fms\n", r1$rmspr, r1$time))
cat(sprintf("Strategy 2 (all target):    RMSPR=%.5f, Time=%.3fms\n", r2$rmspr, r2$time))
cat(sprintf("Strategy 3 (LOO):           RMSPR=%.5f, Time=%.3fms\n",
            mean(dvals_loo), (t1_loo - t0_loo)*1000/n_dev))
cat(sprintf("Paper target:               RMSPR=0.028,  Time=0.598ms\n"))
