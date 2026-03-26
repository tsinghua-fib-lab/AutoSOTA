################################################################################
# WaTL Mortality Data - Find correct 24 developed countries
# Try bottom-24 by e0 (more diverse set including Eastern Europe)
################################################################################

library(wpp2015)
library(Matrix)
library(pracma)

set.seed(42)
M <- 100

data(UNlocations); data(mxM); data(mxF); data(tfr)

to_matrix <- function(x) {
  if (is.vector(x)) matrix(x, ncol=1)
  else if (is.data.frame(x)) as.matrix(x)
  else x
}

compute_L2_distance <- function(v1, v2) sqrt(mean((v1-v2)^2))

compute_f1_hat <- function(src, XV, M, Xt=NULL, Yt=NULL) {
  K <- length(src); bf <- rep(0, M); nt <- 0
  if (!is.null(Xt) && !is.null(Yt)) {
    Xm <- to_matrix(Xt); n <- nrow(Xm)
    xb <- mean(Xm); Sg <- var(as.numeric(Xm))*(n-1)/n
    if (Sg<1e-12) Sg <- 1e-6
    sv <- as.numeric(1+(Xm-xb)/Sg*(XV-xb))
    for (i in 1:n) bf <- bf + sv[i]*Yt[i,]
    nt <- n
  }
  for (k in 1:K) {
    Xs <- src[[k]]$X_s; Ys <- src[[k]]$Y_s
    Xm <- to_matrix(Xs); nk <- nrow(Xm)
    xb <- mean(Xm); Sg <- var(as.numeric(Xm))*(nk-1)/nk
    if (Sg<1e-12) Sg <- 1e-6
    sv <- as.numeric(1+(Xm-xb)/Sg*(XV-xb))
    for (i in 1:nk) bf <- bf + sv[i]*Ys[i,]
    nt <- nt + nk
  }
  bf/nt
}

compute_f_L2 <- function(Yt, sv, f1, lam, M, mi=200, ss=0.1, tol=1e-3) {
  f <- as.numeric(f1)
  for (iter in 1:mi) {
    dm <- -sweep(Yt, 2, f, FUN="-")
    wdm <- sweep(dm, 1, sv, FUN="*")
    grad <- colSums(wdm) + lam*(f-f1)
    fn <- f - ss*grad
    if (sqrt(sum((fn-f)^2))<tol) { f <- fn; break }
    f <- fn
  }
  as.numeric(f)
}

age_starts <- c(0, 1, seq(5, 95, by=5), 100)
age_ends   <- c(1, 5, seq(10, 95, by=5), 100, 110)
age_widths <- age_ends - age_starts

mx_to_qf_norm <- function(mx, Mg=100, ma=110) {
  n <- length(mx); wd <- age_widths[1:n]; ae <- age_ends[1:n]; as_ <- age_starts[1:n]
  qx <- pmin(1-exp(-mx*wd), 1); qx[is.na(qx)|qx<0] <- 0
  lx <- cumprod(c(1, 1-qx)); dx <- pmax(-diff(lx), 0); tot <- sum(dx)
  if (tot<1e-10) return(rep(0.5, Mg))
  prop <- dx/tot; nf <- 3000; af <- seq(0, ma, length.out=nf)
  cdf <- numeric(nf)
  for (j in 1:nf) {
    a <- af[j]; cum <- 0
    for (i in 1:n) {
      if (a>=ae[i]) cum <- cum+prop[i]
      else if (a>as_[i]) { cum <- cum+prop[i]*(a-as_[i])/wd[i]; break }
      else break
    }
    cdf[j] <- min(cum,1)
  }
  p <- seq(1/Mg, 1-1/Mg, length.out=Mg)
  qf <- sapply(p, function(pp) {
    if (pp<=cdf[1]) af[1]
    else if (pp>=cdf[nf]) af[nf]
    else approx(cdf,af,xout=pp,rule=2)$y
  })
  qf/ma
}

compute_e0 <- function(mx) {
  n <- length(mx); wd <- age_widths[1:n]
  qx <- pmin(1-exp(-mx*wd), 1); qx[is.na(qx)] <- 0
  lx <- cumprod(c(1, 1-qx))
  sum((lx[1:n]+lx[2:(n+1)])/2*wd)
}

process_cc <- function(cc, yc="2010-2015") {
  mm <- mxM[mxM$country_code==cc, ]; mf <- mxF[mxF$country_code==cc, ]
  if (nrow(mm)==0||!(yc %in% names(mm))) return(NULL)
  an <- suppressWarnings(as.numeric(trimws(as.character(mm$age)))); an[is.na(an)] <- 100
  ord <- order(an); mm <- mm[ord,]; mf <- mf[ord,]
  rm <- as.numeric(mm[[yc]]); rf <- as.numeric(mf[[yc]])
  if (any(is.na(rm))||any(is.na(rf))||length(rm)<20) return(NULL)
  n <- min(length(rm), length(age_starts))
  rates <- (rm[1:n]+rf[1:n])/2
  qf <- tryCatch(mx_to_qf_norm(rates, M), error=function(e) NULL)
  if (is.null(qf)||any(!is.finite(qf))) return(NULL)
  tr <- tfr[tfr$country_code==cc, ]
  if (nrow(tr)==0||!(yc %in% names(tr))) return(NULL)
  tv <- as.numeric(tr[[yc]]); if (is.na(tv)) return(NULL)
  list(cc=cc, qf=qf, e0=compute_e0(rates), tfr=tv)
}

indiv <- UNlocations[UNlocations$location_type==4, c("country_code","agcode_901","agcode_902")]
mx_ccs <- unique(mxM$country_code)
in_both <- intersect(indiv$country_code, mx_ccs)
dev_codes  <- indiv[indiv$country_code %in% in_both & indiv$agcode_901 > 0, "country_code"]
less_codes <- indiv[indiv$country_code %in% in_both & indiv$agcode_902 > 0, "country_code"]

dev_all <- Filter(Negate(is.null), lapply(dev_codes, process_cc))
less_all <- Filter(Negate(is.null), lapply(less_codes, process_cc))

cat("All valid developed:", length(dev_all), "Developing:", length(less_all), "\n")

# Build source (all developing countries)
Y_s_all <- do.call(rbind, lapply(less_all, function(x) x$qf))
X_s_tfr_all <- matrix(sapply(less_all, function(x) x$tfr), ncol=1)
X_s_e0_all  <- matrix(sapply(less_all, function(x) x$e0), ncol=1)

run_exp <- function(dev_subset, Y_t, X_t_pred, Y_s, X_s_pred, n_tr=14, nr=200, lam=0.25) {
  n_dev <- nrow(Y_t)
  all_pred <- c(X_t_pred, X_s_pred)
  pmin_v <- min(all_pred, na.rm=TRUE); pmax_v <- max(all_pred, na.rm=TRUE)
  if (pmax_v == pmin_v) { pmax_v <- pmin_v + 1 }
  Xt_n <- (X_t_pred - pmin_v)/(pmax_v - pmin_v)
  Xs_n <- (X_s_pred - pmin_v)/(pmax_v - pmin_v)
  src <- list(list(X_s=matrix(Xs_n, ncol=1), Y_s=Y_s))
  set.seed(42)
  rv <- numeric(nr); tv <- numeric(nr)
  for (r in 1:nr) {
    ti <- sample(n_dev, n_tr); ei <- setdiff(1:n_dev, ti)
    Xtr <- matrix(Xt_n[ti], ncol=1); Ytr <- Y_t[ti,]
    Xte <- matrix(Xt_n[ei], ncol=1); Yte <- Y_t[ei,]
    nte <- nrow(Yte); dv <- numeric(nte)
    t0 <- proc.time()["elapsed"]
    for (i in 1:nte) {
      XV <- as.numeric(Xte[i,])
      f1 <- compute_f1_hat(src, XV, M, Xtr, Ytr)
      xb <- mean(Xtr); Sg <- var(as.numeric(Xtr))*(nrow(Xtr)-1)/nrow(Xtr)
      if (Sg<1e-12) Sg <- 1e-6
      sv <- as.numeric(1+(Xtr-xb)/Sg*(XV-xb))
      f2 <- compute_f_L2(Ytr, sv, f1, lam, M, 200, 0.1, 1e-3)
      dv[i] <- compute_L2_distance(Yte[i,], f2)
    }
    t1 <- proc.time()["elapsed"]
    rv[r] <- mean(dv); tv[r] <- (t1-t0)*1000/nte
  }
  list(rmspr=mean(rv), se=sd(rv)/sqrt(nr), time=mean(tv))
}

dev_e0 <- sapply(dev_all, function(x) x$e0)
dev_tfr <- sapply(dev_all, function(x) x$tfr)

# Try different subsets of 24 developed countries
cat("\n=== TOP 24 by e0 (most developed) ===\n")
idx_top24 <- order(-dev_e0)[1:24]
Y_t_top24 <- do.call(rbind, lapply(dev_all[idx_top24], function(x) x$qf))
X_t_top24_tfr <- sapply(dev_all[idx_top24], function(x) x$tfr)
X_t_top24_e0  <- sapply(dev_all[idx_top24], function(x) x$e0)
r_top24_tfr <- run_exp(idx_top24, Y_t_top24, X_t_top24_tfr, Y_s_all, X_s_tfr_all, 14, 200, 0.25)
r_top24_e0  <- run_exp(idx_top24, Y_t_top24, X_t_top24_e0, Y_s_all, X_s_e0_all, 14, 200, 0.25)
cat(sprintf("TFR predictor: RMSPR=%.5f, Time=%.3fms\n", r_top24_tfr$rmspr, r_top24_tfr$time))
cat(sprintf("e0  predictor: RMSPR=%.5f, Time=%.3fms\n", r_top24_e0$rmspr, r_top24_e0$time))

cat("\n=== BOTTOM 24 by e0 (least developed among developed) ===\n")
idx_bot24 <- order(dev_e0)[1:24]
Y_t_bot24 <- do.call(rbind, lapply(dev_all[idx_bot24], function(x) x$qf))
X_t_bot24_tfr <- sapply(dev_all[idx_bot24], function(x) x$tfr)
X_t_bot24_e0  <- sapply(dev_all[idx_bot24], function(x) x$e0)
r_bot24_tfr <- run_exp(idx_bot24, Y_t_bot24, X_t_bot24_tfr, Y_s_all, X_s_tfr_all, 14, 200, 0.25)
r_bot24_e0  <- run_exp(idx_bot24, Y_t_bot24, X_t_bot24_e0, Y_s_all, X_s_e0_all, 14, 200, 0.25)
cat(sprintf("TFR predictor: RMSPR=%.5f, Time=%.3fms\n", r_bot24_tfr$rmspr, r_bot24_tfr$time))
cat(sprintf("e0  predictor: RMSPR=%.5f, Time=%.3fms\n", r_bot24_e0$rmspr, r_bot24_e0$time))

cat("\n=== ALL 45 developed countries (as target) ===\n")
Y_t_all45 <- do.call(rbind, lapply(dev_all, function(x) x$qf))
X_t_all45_tfr <- sapply(dev_all, function(x) x$tfr)
X_t_all45_e0  <- sapply(dev_all, function(x) x$e0)
r_all45_tfr <- run_exp(1:length(dev_all), Y_t_all45, X_t_all45_tfr, Y_s_all, X_s_tfr_all, 14, 200, 0.25)
r_all45_e0  <- run_exp(1:length(dev_all), Y_t_all45, X_t_all45_e0, Y_s_all, X_s_e0_all, 14, 200, 0.25)
cat(sprintf("TFR predictor: RMSPR=%.5f, Time=%.3fms\n", r_all45_tfr$rmspr, r_all45_tfr$time))
cat(sprintf("e0  predictor: RMSPR=%.5f, Time=%.3fms\n", r_all45_e0$rmspr, r_all45_e0$time))

cat("\n=== Summary ===\n")
cat("Paper target: RMSPR=0.028, Time=0.598ms\n")
cat(sprintf("Top24 (TFR): %.5f\n", r_top24_tfr$rmspr))
cat(sprintf("Bot24 (TFR): %.5f\n", r_bot24_tfr$rmspr))
cat(sprintf("All45 (TFR): %.5f\n", r_all45_tfr$rmspr))
cat(sprintf("Bot24 (e0):  %.5f\n", r_bot24_e0$rmspr))
cat(sprintf("All45 (e0):  %.5f\n", r_all45_e0$rmspr))
