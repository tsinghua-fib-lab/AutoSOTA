################################################################################
# WaTL Mortality Data - Systematic exploration of predictors
# Try multiple predictor candidates and find closest to RMSPR=0.028
################################################################################

library(wpp2015)
library(Matrix)
library(pracma)

set.seed(42)
M <- 100

data(UNlocations)
data(mxM); data(mxF)
data(tfr)
data(popF); data(popM)
data(e0F); data(e0M)

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
    n_total <- n_t
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

compute_f_L2 <- function(Y_t, s_vec, f1_hat, lambda, M, max_iter=200, step_size=0.1, tol=1e-3) {
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
  wd <- age_widths[1:n_ages]; ae <- age_ends[1:n_ages]; as_ <- age_starts[1:n_ages]
  qx <- pmin(1-exp(-mx*wd), 1); qx[is.na(qx)|qx<0] <- 0
  lx <- cumprod(c(1, 1-qx))
  dx <- pmax(-diff(lx), 0); tot <- sum(dx)
  if (tot < 1e-10) return(rep(0.5, M_grid))
  prop <- dx / tot
  n_fine <- 3000; af <- seq(0, max_age, length.out=n_fine)
  cdf <- numeric(n_fine)
  for (j in 1:n_fine) {
    a <- af[j]; cum <- 0
    for (i in 1:n_ages) {
      if (a >= ae[i]) cum <- cum + prop[i]
      else if (a > as_[i]) { cum <- cum + prop[i]*(a-as_[i])/wd[i]; break }
      else break
    }
    cdf[j] <- min(cum, 1)
  }
  p <- seq(1/M_grid, 1-1/M_grid, length.out=M_grid)
  qf <- sapply(p, function(pp) {
    if (pp<=cdf[1]) af[1]
    else if (pp>=cdf[n_fine]) af[n_fine]
    else approx(cdf, af, xout=pp, rule=2)$y
  })
  qf / max_age
}

# Compute life expectancy
compute_e0_from_mx <- function(mx) {
  n <- length(mx); wd <- age_widths[1:n]
  qx <- pmin(1-exp(-mx*wd), 1); qx[is.na(qx)] <- 0
  lx <- cumprod(c(1, 1-qx))
  Lx <- (lx[1:n] + lx[2:(n+1)]) / 2 * wd
  sum(Lx)
}

# Compute proportion 65+ from population data
get_prop65 <- function(cc, yr="2015") {
  pf <- popF[popF$country_code==cc, ]; pm <- popM[popM$country_code==cc, ]
  if (nrow(pf)==0 || !(yr %in% names(pf))) return(NA)
  old_age_groups <- c("65-69","70-74","75-79","80-84","85-89","90-94","95-99","100+")
  # Use trimmed age
  pf$age_trim <- trimws(pf$age); pm$age_trim <- trimws(pm$age)
  old_f <- sum(as.numeric(pf[pf$age_trim %in% old_age_groups, yr]), na.rm=TRUE)
  old_m <- sum(as.numeric(pm[pm$age_trim %in% old_age_groups, yr]), na.rm=TRUE)
  tot_f <- sum(as.numeric(pf[[yr]]), na.rm=TRUE)
  tot_m <- sum(as.numeric(pm[[yr]]), na.rm=TRUE)
  if ((tot_f + tot_m) == 0) return(NA)
  (old_f + old_m) / (tot_f + tot_m)
}

# Get observed e0
get_e0_obs <- function(cc, yr="2010-2015") {
  ef <- e0F[e0F$country_code==cc, ]; em <- e0M[e0M$country_code==cc, ]
  if (nrow(ef)==0 || !(yr %in% names(ef))) return(NA)
  vf <- as.numeric(ef[[yr]]); vm <- as.numeric(em[[yr]])
  if (is.na(vf)||is.na(vm)) return(NA)
  (vf + vm) / 2
}

# Process all countries
year_col <- "2010-2015"
indiv <- UNlocations[UNlocations$location_type==4, c("name","country_code","agcode_901","agcode_902")]
mx_ccs <- unique(mxM$country_code)
in_both <- intersect(indiv$country_code, mx_ccs)
dev_codes  <- indiv[indiv$country_code %in% in_both & indiv$agcode_901 > 0, "country_code"]
less_codes <- indiv[indiv$country_code %in% in_both & indiv$agcode_902 > 0, "country_code"]

cat("Processing countries...\n")
process_cc <- function(cc) {
  mx_m <- mxM[mxM$country_code==cc, ]; mx_f <- mxF[mxF$country_code==cc, ]
  if (nrow(mx_m)==0||!(year_col %in% names(mx_m))) return(NULL)
  an <- suppressWarnings(as.numeric(trimws(as.character(mx_m$age))))
  an[is.na(an)] <- 100; ord <- order(an)
  mx_m <- mx_m[ord,]; mx_f <- mx_f[ord,]
  rm <- as.numeric(mx_m[[year_col]]); rf <- as.numeric(mx_f[[year_col]])
  if (any(is.na(rm))||length(rm)<20) return(NULL)
  n_use <- min(length(rm), length(age_starts))
  rates <- (rm[1:n_use] + rf[1:n_use]) / 2
  qf <- tryCatch(mx_to_qf_norm(rates, M), error=function(e) NULL)
  if (is.null(qf)||any(is.na(qf))||any(!is.finite(qf))) return(NULL)

  e0_calc <- compute_e0_from_mx(rates)
  e0_obs  <- get_e0_obs(cc)
  tfr_val <- {
    tr <- tfr[tfr$country_code==cc, ]
    if (nrow(tr)>0 && year_col %in% names(tr)) as.numeric(tr[[year_col]]) else NA
  }
  prop65 <- get_prop65(cc)

  list(cc=cc, qf=qf, e0_calc=e0_calc, e0_obs=e0_obs, tfr=tfr_val, prop65=prop65)
}

dev_res  <- Filter(Negate(is.null), lapply(dev_codes, process_cc))
less_res <- Filter(Negate(is.null), lapply(less_codes, process_cc))

cat("Valid developed:", length(dev_res), "Developing:", length(less_res), "\n")

# Pick top 24 developed by e0 and use all source
dev_e0_calc <- sapply(dev_res, function(x) x$e0_calc)
dev_24 <- dev_res[order(-dev_e0_calc)[1:24]]

X_t_all_e0c    <- matrix(sapply(dev_24, function(x) x$e0_calc), ncol=1)
X_t_all_e0obs  <- matrix(sapply(dev_24, function(x) x$e0_obs), ncol=1)
X_t_all_tfr    <- matrix(sapply(dev_24, function(x) x$tfr), ncol=1)
X_t_all_p65    <- matrix(sapply(dev_24, function(x) x$prop65), ncol=1)
Y_t_all        <- do.call(rbind, lapply(dev_24, function(x) x$qf))

X_s_e0c   <- matrix(sapply(less_res, function(x) x$e0_calc), ncol=1)
X_s_e0obs <- matrix(sapply(less_res, function(x) x$e0_obs), ncol=1)
X_s_tfr   <- matrix(sapply(less_res, function(x) x$tfr), ncol=1)
X_s_p65   <- matrix(sapply(less_res, function(x) x$prop65), ncol=1)
Y_s        <- do.call(rbind, lapply(less_res, function(x) x$qf))

# Remove NAs
valid_dev <- complete.cases(X_t_all_e0c, X_t_all_e0obs, X_t_all_tfr, X_t_all_p65)
valid_src <- complete.cases(X_s_e0c, X_s_e0obs, X_s_tfr, X_s_p65)

dev_24 <- dev_24[valid_dev]; Y_t_all <- Y_t_all[valid_dev, ]
X_t_all_e0c   <- X_t_all_e0c[valid_dev, , drop=FALSE]
X_t_all_e0obs <- X_t_all_e0obs[valid_dev, , drop=FALSE]
X_t_all_tfr   <- X_t_all_tfr[valid_dev, , drop=FALSE]
X_t_all_p65   <- X_t_all_p65[valid_dev, , drop=FALSE]

less_res <- less_res[valid_src]; Y_s <- Y_s[valid_src, ]
X_s_e0c   <- X_s_e0c[valid_src, , drop=FALSE]
X_s_e0obs <- X_s_e0obs[valid_src, , drop=FALSE]
X_s_tfr   <- X_s_tfr[valid_src, , drop=FALSE]
X_s_p65   <- X_s_p65[valid_src, , drop=FALSE]

cat("After NA removal: Dev=", nrow(Y_t_all), "Src=", nrow(Y_s), "\n")

# Normalize predictors
normalize <- function(x_t, x_s) {
  all_v <- c(x_t, x_s)
  vmin <- min(all_v, na.rm=TRUE); vmax <- max(all_v, na.rm=TRUE)
  if (vmax == vmin) return(list(x_t=x_t*0+0.5, x_s=x_s*0+0.5))
  list(x_t=(x_t - vmin)/(vmax-vmin), x_s=(x_s - vmin)/(vmax-vmin))
}

norm_e0c   <- normalize(X_t_all_e0c, X_s_e0c)
norm_e0obs <- normalize(X_t_all_e0obs, X_s_e0obs)
norm_tfr   <- normalize(X_t_all_tfr, X_s_tfr)
norm_p65   <- normalize(X_t_all_p65, X_s_p65)

# Experiment function
run_exp <- function(X_t_norm, X_s_norm, Y_t, Y_s_mat, n_target=14, n_reps=200, lam=0.25) {
  src_data <- list(list(X_s=X_s_norm, Y_s=Y_s_mat))
  n_dev <- nrow(Y_t)
  set.seed(42)
  rmspr_v <- numeric(n_reps)
  time_v  <- numeric(n_reps)

  for (rep in 1:n_reps) {
    train_idx <- sample(n_dev, n_target)
    test_idx  <- setdiff(1:n_dev, train_idx)

    X_tr <- X_t_norm[train_idx, , drop=FALSE]
    Y_tr <- Y_t[train_idx, , drop=FALSE]
    X_te <- X_t_norm[test_idx, , drop=FALSE]
    Y_te <- Y_t[test_idx, , drop=FALSE]

    n_te <- nrow(X_te)
    dvals <- numeric(n_te)
    t0 <- proc.time()["elapsed"]

    for (i in 1:n_te) {
      XV <- as.numeric(X_te[i,])
      f1 <- compute_f1_hat(src_data, XV, M, X_tr, Y_tr)
      xb <- mean(X_tr); Sg <- var(as.numeric(X_tr))*(nrow(X_tr)-1)/nrow(X_tr)
      if (Sg<1e-12) Sg <- 1e-6
      sv <- as.numeric(1 + (X_tr - xb)/Sg * (XV - xb))
      f2 <- compute_f_L2(Y_t=Y_tr, s_vec=sv, f1_hat=f1, lambda=lam, M=M,
                         max_iter=200, step_size=0.1, tol=1e-3)
      dvals[i] <- compute_L2_distance(Y_te[i,], f2)
    }

    t1 <- proc.time()["elapsed"]
    rmspr_v[rep] <- mean(dvals)
    time_v[rep]  <- (t1 - t0)*1000/n_te
  }
  list(rmspr=mean(rmspr_v), rmspr_se=sd(rmspr_v)/sqrt(n_reps),
       time=mean(time_v), time_se=sd(time_v)/sqrt(n_reps))
}

cat("\nTesting different predictors for RMSPR target = 0.028:\n")
cat("(n_dev=", nrow(Y_t_all), ", n_src=", nrow(Y_s), ", n_train=14)\n")

cat("\n[1] Life expectancy (calculated from mx):\n")
r1 <- run_exp(norm_e0c$x_t, norm_e0c$x_s, Y_t_all, Y_s, 14, 200, 0.25)
cat(sprintf("   RMSPR=%.5f, Time=%.3fms\n", r1$rmspr, r1$time))

cat("\n[2] Life expectancy (observed from wpp2015):\n")
r2 <- run_exp(norm_e0obs$x_t, norm_e0obs$x_s, Y_t_all, Y_s, 14, 200, 0.25)
cat(sprintf("   RMSPR=%.5f, Time=%.3fms\n", r2$rmspr, r2$time))

cat("\n[3] TFR:\n")
r3 <- run_exp(norm_tfr$x_t, norm_tfr$x_s, Y_t_all, Y_s, 14, 200, 0.25)
cat(sprintf("   RMSPR=%.5f, Time=%.3fms\n", r3$rmspr, r3$time))

cat("\n[4] Proportion 65+:\n")
r4 <- run_exp(norm_p65$x_t, norm_p65$x_s, Y_t_all, Y_s, 14, 200, 0.25)
cat(sprintf("   RMSPR=%.5f, Time=%.3fms\n", r4$rmspr, r4$time))

# Try combination predictors
# Weighted mixture of e0 and TFR
alpha_vals <- seq(0, 1, by=0.1)  # alpha * e0 + (1-alpha) * TFR
cat("\n[5] Weighted mixture alpha*e0 + (1-alpha)*TFR:\n")
for (alp in alpha_vals) {
  Xt_mix <- alp * norm_e0c$x_t + (1-alp) * norm_tfr$x_t
  Xs_mix <- alp * norm_e0c$x_s + (1-alp) * norm_tfr$x_s
  # Renormalize
  n1 <- normalize(Xt_mix, Xs_mix)
  rm <- run_exp(n1$x_t, n1$x_s, Y_t_all, Y_s, 14, 100, 0.25)
  cat(sprintf("   alpha=%.1f: RMSPR=%.5f, Time=%.3fms\n", alp, rm$rmspr, rm$time))
}

cat("\n=== Target: RMSPR=0.028, Time=0.598ms ===\n")
