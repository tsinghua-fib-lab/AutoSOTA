cs_compute_auc <- function(y_true, y_score) {
  y_true <- as.numeric(y_true)
  y_score <- as.numeric(y_score)
  ok <- is.finite(y_true) & is.finite(y_score)
  y_true <- y_true[ok]
  y_score <- y_score[ok]
  if (!length(y_true)) return(NA_real_)
  pos <- y_true == 1
  neg <- y_true == 0
  n_pos <- sum(pos)
  n_neg <- sum(neg)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  ranks <- rank(y_score, ties.method = "average")
  n_pos_d <- as.double(n_pos)
  n_neg_d <- as.double(n_neg)
  (sum(ranks[pos]) - n_pos_d * (n_pos_d + 1) / 2) / (n_pos_d * n_neg_d)
}

cs_make_stratified_folds <- function(n, n_folds, y = NULL, cluster = NULL, seed = 123L) {
  n <- as.integer(n)
  n_folds <- as.integer(n_folds)
  if (n <= 1L || n_folds < 2L) {
    return(NULL)
  }

  restore_rng <- function(old_seed) {
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }

  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv)
  } else {
    NULL
  }
  set.seed(as.integer(seed))
  on.exit(restore_rng(old_seed), add = TRUE)

  if (!is.null(cluster) && length(cluster) == n) {
    cluster_use <- as.character(cluster)
    if (any(is.na(cluster_use))) {
      na_idx <- which(is.na(cluster_use))
      cluster_use[na_idx] <- paste0("__missing__", na_idx)
    }
  } else {
    cluster_use <- paste0("__obs__", seq_len(n))
  }

  group_index <- split(seq_len(n), cluster_use)
  group_ids <- names(group_index)
  group_sizes <- vapply(group_index, length, integer(1))
  k <- min(n_folds, length(group_index))
  if (k < 2L) {
    return(NULL)
  }

  strata <- rep("all", length(group_index))
  if (!is.null(y) && length(y) == n) {
    y_vec <- y
    keep <- !is.na(y_vec)
    y_keep <- y_vec[keep]
    if (length(y_keep) > 0L) {
      y_num <- suppressWarnings(as.numeric(y_keep))
      binary_like <- length(y_num) == length(y_keep) &&
        all(is.finite(y_num)) &&
        all(y_num %in% c(0, 1)) &&
        length(unique(y_num)) >= 2L
      if (isTRUE(binary_like)) {
        pos_rate <- vapply(group_index, function(idx) {
          yy <- y_vec[idx]
          yy <- suppressWarnings(as.numeric(yy))
          yy <- yy[is.finite(yy)]
          if (!length(yy)) {
            return(NA_real_)
          }
          mean(yy == 1)
        }, numeric(1))
        bucket <- cut(
          pos_rate,
          breaks = c(-Inf, 0, 0.25, 0.5, 0.75, 1),
          labels = FALSE,
          right = TRUE
        )
        bucket[is.na(bucket)] <- 0L
        strata <- paste0("bin_", bucket)
      } else {
        y_chr <- as.character(y_keep)
        if (length(unique(y_chr)) >= 2L) {
          majority_class <- vapply(group_index, function(idx) {
            yy <- as.character(y_vec[idx])
            yy <- yy[!is.na(yy)]
            if (!length(yy)) {
              return("__missing__")
            }
            counts <- sort(table(yy), decreasing = TRUE)
            names(counts)[1L]
          }, character(1))
          if (length(unique(majority_class)) >= 2L) {
            strata <- paste0("cat_", majority_class)
          }
        }
      }
    }
  }

  fold_loads <- integer(k)
  fold_by_group <- integer(length(group_index))
  for (stratum in unique(strata)) {
    stratum_idx <- which(strata == stratum)
    stratum_idx <- sample(stratum_idx, length(stratum_idx))
    start_fold <- if (length(fold_loads)) {
      order(fold_loads, seq_len(k))[1L]
    } else {
      1L
    }
    fold_cycle <- ((seq_along(stratum_idx) - 1L + start_fold - 1L) %% k) + 1L
    for (i in seq_along(stratum_idx)) {
      group_pos <- stratum_idx[[i]]
      fold_id <- fold_cycle[[i]]
      fold_by_group[[group_pos]] <- fold_id
      fold_loads[[fold_id]] <- fold_loads[[fold_id]] + group_sizes[[group_pos]]
    }
  }

  fold_map <- setNames(fold_by_group, group_ids)
  list(fold_id = as.integer(fold_map[cluster_use]), n_folds = as.integer(k))
}

cs_compute_binary_log_loss <- function(y_true, y_score, eps = 1e-12) {
  y_true <- as.numeric(y_true)
  y_score <- as.numeric(y_score)
  ok <- is.finite(y_true) & is.finite(y_score)
  y_true <- y_true[ok]
  y_score <- y_score[ok]
  if (!length(y_true)) return(NA_real_)
  p <- pmin(pmax(y_score, eps), 1 - eps)
  -mean(y_true * log(p) + (1 - y_true) * log(1 - p))
}

cs_compute_multiclass_log_loss <- function(y_true, prob_mat, eps = 1e-12) {
  if (is.null(dim(prob_mat))) {
    prob_mat <- matrix(prob_mat, nrow = length(y_true), byrow = TRUE)
  }
  n_eval <- nrow(prob_mat)
  if (length(y_true) != n_eval) return(NA_real_)
  ok <- !is.na(y_true)
  y_true <- y_true[ok]
  prob_mat <- prob_mat[ok, , drop = FALSE]
  if (!length(y_true)) return(NA_real_)
  idx <- cbind(seq_along(y_true), y_true + 1L)
  p <- prob_mat[idx]
  p <- pmin(pmax(p, eps), 1 - eps)
  -mean(log(p))
}

cs_binary_metric_diagnostics <- function(y_true, y_score, threshold = 0.5) {
  y_true <- as.numeric(y_true)
  y_score <- as.numeric(y_score)
  ok <- is.finite(y_true) & is.finite(y_score)
  y_true <- y_true[ok]
  y_score <- y_score[ok]

  quantile_names <- c("p05", "p25", "p50", "p75", "p95")
  empty_quantiles <- setNames(rep(NA_real_, length(quantile_names)), quantile_names)
  empty_confusion <- c(tn = NA_integer_, fp = NA_integer_, fn = NA_integer_, tp = NA_integer_)

  if (!length(y_true)) {
    return(list(
      truth_pred_correlation = NA_real_,
      y_mean = NA_real_,
      pred_mean = NA_real_,
      pred_sd = NA_real_,
      pred_quantiles = empty_quantiles,
      confusion_0_5 = empty_confusion
    ))
  }

  cor_val <- NA_real_
  if (length(y_true) > 1L) {
    y_sd <- suppressWarnings(stats::sd(y_true))
    pred_sd <- suppressWarnings(stats::sd(y_score))
    if (is.finite(y_sd) && y_sd > 0 && is.finite(pred_sd) && pred_sd > 0) {
      cor_val <- suppressWarnings(stats::cor(y_true, y_score))
    }
  }

  pred_quantiles <- stats::quantile(
    y_score,
    probs = c(0.05, 0.25, 0.5, 0.75, 0.95),
    na.rm = TRUE,
    names = FALSE,
    type = 7
  )
  names(pred_quantiles) <- quantile_names

  pred_class <- as.integer(y_score >= threshold)
  confusion_0_5 <- c(
    tn = sum(pred_class == 0L & y_true == 0L),
    fp = sum(pred_class == 1L & y_true == 0L),
    fn = sum(pred_class == 0L & y_true == 1L),
    tp = sum(pred_class == 1L & y_true == 1L)
  )

  list(
    truth_pred_correlation = cor_val,
    y_mean = mean(y_true),
    pred_mean = mean(y_score),
    pred_sd = if (length(y_score) > 1L) stats::sd(y_score) else 0,
    pred_quantiles = pred_quantiles,
    confusion_0_5 = confusion_0_5
  )
}

cs_compute_outcome_metrics <- function(y_eval, pred_eval, likelihood, threshold = 0.5) {
  binary_likelihood <- likelihood %in% c("bernoulli", "binomial")
  if (binary_likelihood) {
    y_eval <- as.numeric(y_eval)
    p <- as.numeric(pred_eval)
    keep <- is.finite(y_eval) & is.finite(p) & (y_eval %in% c(0, 1))
    y_eval <- y_eval[keep]
    p <- p[keep]
    metrics <- list(
      likelihood = likelihood,
      n_eval = length(y_eval),
      auc = cs_compute_auc(y_eval, p),
      log_loss = cs_compute_binary_log_loss(y_eval, p),
      accuracy = if (length(y_eval)) mean((p >= threshold) == y_eval) else NA_real_,
      brier = if (length(y_eval)) mean((p - y_eval) ^ 2) else NA_real_
    )
    return(c(metrics, cs_binary_metric_diagnostics(y_eval, p, threshold = threshold)))
  }

  if (identical(likelihood, "categorical")) {
    y_eval <- as.integer(y_eval)
    prob_mat <- as.matrix(pred_eval)
    keep <- !is.na(y_eval)
    y_eval <- y_eval[keep]
    prob_mat <- prob_mat[keep, , drop = FALSE]
    if (length(y_eval)) {
      log_loss <- cs_compute_multiclass_log_loss(y_eval, prob_mat)
      pred_class <- max.col(prob_mat, ties.method = "first") - 1L
      accuracy <- mean(pred_class == y_eval, na.rm = TRUE)
    } else {
      log_loss <- NA_real_
      accuracy <- NA_real_
    }
    return(list(
      likelihood = likelihood,
      n_eval = length(y_eval),
      log_loss = log_loss,
      accuracy = accuracy
    ))
  }

  y_eval <- as.numeric(y_eval)
  if (is.list(pred_eval) && !is.null(pred_eval$mu)) {
    pred_mu <- as.numeric(pred_eval$mu)
    pred_sigma <- if (!is.null(pred_eval$sigma)) {
      as.numeric(pred_eval$sigma)
    } else {
      rep(NA_real_, length(pred_mu))
    }
  } else {
    pred_mu <- as.numeric(pred_eval)
    pred_sigma <- rep(NA_real_, length(pred_mu))
  }
  keep <- is.finite(y_eval) & is.finite(pred_mu)
  y_eval <- y_eval[keep]
  pred_mu <- pred_mu[keep]
  if (length(pred_sigma) == 1L && length(y_eval) > 1L) {
    pred_sigma <- rep(pred_sigma, length(y_eval))
  } else if (length(pred_sigma) == length(keep)) {
    pred_sigma <- pred_sigma[keep]
  } else if (length(pred_sigma) != length(y_eval)) {
    pred_sigma <- rep(NA_real_, length(y_eval))
  }

  nll <- NA_real_
  if (length(y_eval) &&
      length(pred_sigma) == length(y_eval) &&
      all(is.finite(pred_sigma)) &&
      all(pred_sigma > 0)) {
    nll <- mean(0.5 * log(2 * pi * pred_sigma ^ 2) + (y_eval - pred_mu) ^ 2 / (2 * pred_sigma ^ 2))
  }

  list(
    likelihood = likelihood,
    n_eval = length(y_eval),
    rmse = if (length(y_eval)) sqrt(mean((pred_mu - y_eval) ^ 2)) else NA_real_,
    mae = if (length(y_eval)) mean(abs(pred_mu - y_eval)) else NA_real_,
    nll = nll
  )
}
