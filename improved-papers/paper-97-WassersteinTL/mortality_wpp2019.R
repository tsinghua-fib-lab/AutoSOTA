################################################################################
# WaTL Mortality Data using wpp2019 (2015-2020 period)
# Paper cites WPP 2022, using 2015 data
################################################################################

suppressPackageStartupMessages({
  library(wpp2019)
  library(Matrix)
  library(pracma)
})

set.seed(42)
M <- 100

data(UNlocations)
data(mxM19 <- mxM); rm(mxM)  # rename

# Use wpp2015 for TFR
library(wpp2015)
data(tfr)

to_matrix <- function(x) { if (is.vector(x)) matrix(x, ncol=1) else if (is.data.frame(x)) as.matrix(x) else x }
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

# Need to reload mxM from wpp2019
library(wpp2019)
data(mxM)

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

process_cc19 <- function(cc, yc="2015-2020") {
  mm <- mxM[mxM$country_code==cc, ]
  if (nrow(mm)==0||!(yc %in% names(mm))) return(NULL)
  an <- suppressWarnings(as.numeric(trimws(as.character(mm$age)))); an[is.na(an)] <- 100
  ord <- order(an); mm <- mm[ord,]
  rm <- as.numeric(mm[[yc]])
  if (any(is.na(rm))||length(rm)<20) return(NULL)
  # wpp2019 mxM is for one sex only? Let's check
  n <- min(length(rm), length(age_starts))
  rates <- rm[1:n]  # Use as-is (total or one sex)
  qf <- tryCatch(mx_to_qf_norm(rates, M), error=function(e) NULL)
  if (is.null(qf)||any(!is.finite(qf))) return(NULL)
  # Get TFR from wpp2015
  tr <- tfr[tfr$country_code==cc, ]
  yc_tfr <- "2010-2015"
  if (nrow(tr)==0||!(yc_tfr %in% names(tr))) return(NULL)
  tv <- as.numeric(tr[[yc_tfr]]); if (is.na(tv)) return(NULL)
  list(cc=cc, qf=qf, e0=compute_e0(rates), tfr=tv)
}

# Check wpp2019 mxM structure
cat("wpp2019 mxM: checking if male-only or total...\n")
cat("Countries in mxM19:", length(unique(mxM$country_code)), "\n")
cat("Rows for World (900):", sum(mxM$country_code==900), "\n")
world_mx <- mxM[mxM$country_code==900, ]
cat("World ages:", paste(trimws(world_mx$age)[1:10], collapse=","), "\n")

indiv <- UNlocations[UNlocations$location_type==4, c("country_code","agcode_901000","agcode_902000")]
mx_ccs <- unique(mxM$country_code)
in_both <- intersect(indiv$country_code, mx_ccs)
dev_codes  <- indiv[indiv$country_code %in% in_both & !is.na(indiv$agcode_901000) & indiv$agcode_901000 > 0, "country_code"]
less_codes <- indiv[indiv$country_code %in% in_both & !is.na(indiv$agcode_902000) & indiv$agcode_902000 > 0, "country_code"]
cat("wpp2019: Developed=", length(dev_codes), "Developing=", length(less_codes), "\n")

dev_all  <- Filter(Negate(is.null), lapply(dev_codes, process_cc19))
less_all <- Filter(Negate(is.null), lapply(less_codes, process_cc19))
cat("Valid: Developed=", length(dev_all), "Developing=", length(less_all), "\n")

dev_e0 <- sapply(dev_all, function(x) x$e0)
idx_bot24 <- order(dev_e0)[1:24]
dev_24 <- dev_all[idx_bot24]

Y_t_all <- do.call(rbind, lapply(dev_24, function(x) x$qf))
X_t_tfr <- sapply(dev_24, function(x) x$tfr)
Y_s_all  <- do.call(rbind, lapply(less_all, function(x) x$qf))
X_s_tfr  <- sapply(less_all, function(x) x$tfr)

all_tfr <- c(X_t_tfr, X_s_tfr)
tfr_min <- min(all_tfr); tfr_max <- max(all_tfr)
X_t_n <- (X_t_tfr - tfr_min)/(tfr_max - tfr_min)
X_s_n <- (X_s_tfr - tfr_min)/(tfr_max - tfr_min)

src_data <- list(list(X_s=matrix(X_s_n, ncol=1), Y_s=Y_s_all))

n_dev <- nrow(Y_t_all)
cat("Target:", n_dev, "Source:", nrow(Y_s_all), "\n")

n_reps <- 500; lam <- 0.25; n_tr <- 14
set.seed(42)
rv <- numeric(n_reps); tv <- numeric(n_reps)
for (r in 1:n_reps) {
  ti <- sample(n_dev, n_tr); ei <- setdiff(1:n_dev, ti)
  Xtr <- matrix(X_t_n[ti], ncol=1); Ytr <- Y_t_all[ti,]
  Xte <- matrix(X_t_n[ei], ncol=1); Yte <- Y_t_all[ei,]
  nte <- nrow(Yte); dv <- numeric(nte)
  t0 <- proc.time()["elapsed"]
  for (i in 1:nte) {
    XV <- as.numeric(Xte[i,])
    f1 <- compute_f1_hat(src_data, XV, M, Xtr, Ytr)
    xb <- mean(Xtr); Sg <- var(as.numeric(Xtr))*(nrow(Xtr)-1)/nrow(Xtr)
    if (Sg<1e-12) Sg <- 1e-6
    sv <- as.numeric(1+(Xtr-xb)/Sg*(XV-xb))
    f2 <- compute_f_L2(Ytr, sv, f1, lam, M, 200, 0.1, 1e-3)
    dv[i] <- compute_L2_distance(Yte[i,], f2)
  }
  t1 <- proc.time()["elapsed"]
  rv[r] <- mean(dv); tv[r] <- (t1-t0)*1000/nte
}

cat(sprintf("\nwpp2019 Results: RMSPR=%.5f, Time=%.3fms\n", mean(rv), mean(tv)))
