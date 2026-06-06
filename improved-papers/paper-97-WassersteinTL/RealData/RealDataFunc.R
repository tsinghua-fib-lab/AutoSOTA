get_realdata <- function(M) {
  
  PAXINTEN_D[, paste0("MIN", 1:1440)] <- PAXINTEN_D[, paste0("MIN", 1:1440)] *  Flags_D[, paste0("MIN", 1:1440)]
  PAXINTEN_D <- PAXINTEN_D[which(PAXINTEN_D$PAXCAL == 1 & PAXINTEN_D$PAXSTAT == 1), ]
  
  PAXINTEN_D <- PAXINTEN_D[, c("SEQN", paste0("MIN", 1:1440))]
  
  PAXINTEN_D[, paste0("MIN", 1:1440)] <- apply(PAXINTEN_D[, paste0("MIN", 1:1440)], 2, function(x) {
    ifelse(x > 1000 | x == 0, NA, x)
  })
  
  cnt_mat <- as.matrix(PAXINTEN_D[, paste0("MIN", 1:1440)])
  
  valid_obs_per_subject <- apply(cnt_mat, 1, function(x) sum(!is.na(x)))
  keep_rows <- valid_obs_per_subject > 100
  PAXINTEN_D <- PAXINTEN_D[keep_rows, ]
  cnt_mat <- cnt_mat[keep_rows, ]
  valid_participants <- unique(PAXINTEN_D$SEQN)
  
  cat("Number of participants after filtering:", length(valid_participants), "\n")
  
  calc_quantile <- function(data, p) {
    valid_data <- data[!is.na(data)]
    if (length(valid_data) == 0) {
      return(NA_real_)
    } else {
      return(stats::quantile(valid_data, probs = p, na.rm = TRUE, type = 7)) 
    }
  }
  
  p_grid <- seq(0, 1, length.out = M)
  
  results <- matrix(NA, nrow = length(valid_participants), ncol = M)
  rownames(results) <- valid_participants
  
  cat("Dimension of The Results:", dim(results), "\n")
  
  for (i in seq_along(valid_participants)) {
    id <- valid_participants[i]
    id_rows <- which(PAXINTEN_D$SEQN == id)
    participant_data <- cnt_mat[id_rows, ]
    
    all_data <- c(participant_data)
    
    results_all_days <- sapply(p_grid, function(p) calc_quantile(all_data, p))
    
    results[i, ] <- results_all_days
    
    if (i %% 100 == 0) {
      cat("Processed", i, "out of", length(valid_participants), "participants\n")
    }
  }
  
  data <- as.data.frame(results)
  colnames(data) <- paste0("quantile_", round(p_grid, 4))
  data$SEQN <- valid_participants
  
  Covariate_D$Age <- Covariate_D$RIDAGEEX / 12
  data <- dplyr::inner_join(data, 
                            Covariate_D[, c("SEQN", "Race", "Gender" ,"BMI", "Age")],
                            by = "SEQN")
  
  data <- data %>%
     dplyr::mutate(Gender = ifelse(Gender == "Male", 1, 0))

  return(data)
}


realdata_process <- function(realdata, M, race.t = "Black"){
  
  Races = unique(realdata$Race)[unique(realdata$Race) != race.t]
  
  X_t = realdata[which(realdata$Race == race.t), c("BMI","Age")]
  Y_t = as.matrix(realdata[which(realdata$Race == race.t), 1:M])
  
  source_data = list()
  for(k in seq_len(length(Races))){
    
    X_s_k = realdata[which(realdata$Race == Races[k]), c("BMI","Age")]
    Y_s_k = as.matrix(realdata[which(realdata$Race == Races[k]), 1:M])
    
    source_data[[k]] <- list(
      X_s = X_s_k,
      Y_s = Y_s_k
    )
  }
  
  return(list(
    X_t = X_t,
    Y_t = Y_t,
    source_data = source_data,
    Races = Races
  ))
}


create_folds <- function(n, k) {
  folds <- cut(seq(1, n), breaks = k, labels = FALSE)
  return(sample(folds))
}

to_matrix <- function(x) {
  if (is.vector(x)) {
    matrix(x, ncol=1)
  } else if (is.data.frame(x)) {
    as.matrix(x)
  } else {
    x
  }
}

cv_search_lambda <- function(Y_t,
                             s_vec,
                             X_t,
                             source_d,
                             M,
                             rate,
                             lambda_grid = NULL,
                             n_folds = 5,
                             max_iter = 200,
                             step_size = 0.5,
                             tol = 1e-6,
                             n_cores = NULL,
                             val_sample_size = 30
)
{
  start_time <- Sys.time()

  if (is.null(n_cores)) {
    n_cores <- max(1, detectCores() - 1)
  }

  n <- nrow(Y_t)
  total_lambdas <- length(lambda_grid)

  cat(sprintf("\nStarting cross-validation at %s\n", format(start_time)))
  cat(sprintf("Total lambdas to test: %d\n", total_lambdas))
  cat(sprintf("Number of cores: %d\n", n_cores))
  cat(sprintf("Validation sample size: %d\n", val_sample_size))
  cat("----------------------------------------\n\n")

  shuffle_idx <- sample.int(n)
  fold_id <- cut(seq_len(n), breaks = n_folds, labels = FALSE)

  cl <- makeCluster(n_cores)
  registerDoSNOW(cl)

  pb <- txtProgressBar(max=total_lambdas, style=3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress=progress)

  clusterExport(
    cl, 
    c("compute_f1_hat", 
      "compute_f_L2",
      "to_matrix",
      "LocalLinWeights",
      "kerFctn",
      "SetBwRange",
      "plcm",
      "lcm",
      "gcd",
      "bwCV",
      "lrem",
      "compute_L2_distance",
      "Y_t",
      "s_vec",
      "X_t",
      "source_d",
      "M",
      "rate",
      "lambda_grid",
      "n_folds",
      "max_iter",
      "step_size",
      "tol",
      "val_sample_size",
      "shuffle_idx",
      "fold_id"
    ),
    envir = environment()
  )

  progress_counter <- 0

  results <- foreach(l_idx = seq_along(lambda_grid),
                     .combine = 'rbind',
                     .options.snow = opts) %dopar% {
                       lambda_val <- lambda_grid[l_idx]
                       fold_errors <- numeric(n_folds)

                       for (k in seq_len(n_folds)) {
                         val_idx_full <- shuffle_idx[fold_id == k]

                         if (length(val_idx_full) > val_sample_size) {
                           val_idx <- sample(val_idx_full, val_sample_size)
                         } else {
                           val_idx <- val_idx_full
                         }

                         train_idx <- shuffle_idx[fold_id != k]

                         Y_train <- Y_t[train_idx, , drop = FALSE]
                         s_vec_train <- s_vec[train_idx]
                         X_train <- X_t[train_idx, ]

                         Y_val <- Y_t[val_idx, , drop = FALSE]
                         s_vec_val <- s_vec[val_idx]
                         X_val <- X_t[val_idx, ]

                         val_errors_fold <- numeric(length(val_idx))

                         for (i in seq_along(val_idx)) {
                           X_value <- as.vector(as.numeric(X_val[i,]))
                           
                          f1_hat_i <- compute_f1_hat(
                            source_data_list = source_d,
                            X_value = X_value,
                            M = M,
                            rate = rate,
                            X_t = X_train,
                            Y_t = Y_train
                          )

                           f_train <- tryCatch({
                             compute_f_L2(
                               Y_t = Y_train,
                               s_vec = s_vec_train,
                               f1_hat = f1_hat_i,
                               lambda = lambda_val,
                               M = M,
                               max_iter = max_iter,
                               step_size = step_size,
                               tol = tol
                             )
                           }, error = function(e) NULL)

                           if (!is.null(f_train)) {
                             val_errors_fold[i] <- mean((f_train - Y_val[i,])^2)
                           } else {
                             val_errors_fold[i] <- NA
                           }
                         }

                         valid_errors <- !is.na(val_errors_fold)
                         if (sum(valid_errors) > 0) {
                           fold_errors[k] <- mean(val_errors_fold[valid_errors])
                         } else {
                           fold_errors[k] <- NA
                         }
                       }

                       c(lambda_val, mean(fold_errors, na.rm = TRUE))
                     }

  stopCluster(cl)
  close(pb)

  end_time <- Sys.time()
  total_time <- difftime(end_time, start_time, units = "mins")

  cat("\n\n----------------------------------------\n")
  cat(sprintf("Cross-validation completed at %s\n", format(end_time)))
  cat(sprintf("Total time elapsed: %.2f minutes\n", as.numeric(total_time)))
  cat("----------------------------------------\n")

  results <- as.data.frame(results)
  colnames(results) <- c("lambda", "cv_error")

  best_lambda <- results$lambda[which.min(results$cv_error)]
  best_error <- min(results$cv_error)

  return(list(
    results = results,
    best_lambda = best_lambda,
    total_time = total_time
  ))
}

compute_f1_hat <- function(source_data_list,
                           X_value,
                           M,
                           rate,
                           X_t = NULL,
                           Y_t = NULL)
{
  # Algorithm 1, Step 1: Weighted auxiliary estimator
  # \widehat{f}(x) = \frac{1}{n_0+n_{\mathcal{A}}}\sum_{k=0}^{K}n_k\widehat{f}^{(k)}(x)
  
  K <- length(source_data_list)
  z_grid <- seq(1 / M, 1 - 1 / M, length.out = M)
  
  # Aggregate weighted estimates from all sources
  total_n <- 0
  weighted_f <- rep(0, M)
  
  # Include target data in Step 1 (k=0)
  if (!is.null(X_t) && !is.null(Y_t)) {
    n_t <- nrow(as.matrix(X_t))
    qin.target <- split(Y_t, row(Y_t))
    qin.target <- lapply(qin.target, as.numeric)
    xin.t <- to_matrix(X_t)
    
    optns.t <- list(kernel = "gauss", 
                    bw = c(diff(range(X_t$BMI))*0.15, diff(range(X_t$Age))*0.15))
    
    f_target <- lrem(qin.target, xin.t, X_value, optns.t)$qp[1,]
    weighted_f <- weighted_f + n_t * f_target
    total_n <- total_n + n_t
  }
  
  # Add source data (k=1,...,K)
  for (k_idx in seq_len(K)) {
    X_s <- source_data_list[[k_idx]]$X_s
    Y_s <- source_data_list[[k_idx]]$Y_s
    n_k <- nrow(as.matrix(X_s))
    
    # Optional: subsample source data if rate < 1
    if (rate < 1) {
      Index_random <- sample(n_k, as.integer(n_k * rate))
      X_s <- X_s[Index_random,]
      Y_s <- Y_s[Index_random,]
      n_k <- nrow(as.matrix(X_s))
    }
    
    qin.source <- split(Y_s, row(Y_s))
    qin.source <- lapply(qin.source, as.numeric)
    xin.s <- to_matrix(X_s)
    
    optns.s <- list(kernel = "gauss", 
                    bw = c(diff(range(X_s$BMI))*0.15, diff(range(X_s$Age))*0.15))
    
    f_source_k <- lrem(qin.source, xin.s, X_value, optns.s)$qp[1,]
    weighted_f <- weighted_f + n_k * f_source_k
    total_n <- total_n + n_k
  }
  
  # Return weighted average
  f1 <- weighted_f / total_n
  
  return(f1)
}

compute_f_L2 <- function(Y_t,
                         s_vec,
                         f1_hat,
                         lambda,
                         M,
                         max_iter = 1000,
                         step_size = 0.5,
                         tol = 1e-8) {
  n0 <- nrow(Y_t)
  if (length(s_vec) != n0) {
    stop("s_vec length must match row number of Y_t.")
  }
  if (length(f1_hat) != M) {
    stop("f1_hat length must be M.")
  }
  
  f <- as.numeric(f1_hat)
  
  for (iter in 1:max_iter) 
  {
    
    diff_mat <- sweep(Y_t, 2, f, FUN = "-") 

    diff_mat <- -diff_mat                 
    
    weighted_diff_mat <- sweep(diff_mat, 1, s_vec, FUN = "*")  
    gradient_target <- colSums(weighted_diff_mat)         
    gradient_reg <- lambda * (f - f1_hat) 
    grad <- gradient_target + gradient_reg
    
    f_new <- f - step_size * grad 
    
    diff_norm <- sqrt(sum((f_new - f)^2))
    if (diff_norm < tol) {
      f <- f_new
      break
    }
    
    f <- f_new
  }

  return(as.numeric(f))
}

Project <- function(qVec, lower = NULL, upper = NULL) {
  M <- length(qVec)
  
  if (M <= 1) {
    return(qVec)
  }
  
  P <- diag(M)
  q <- -qVec
  
  A <- matrix(0, M-1, M)
  for(i in 1:(M-1)) {
    A[i,i] <- -1
    A[i,i+1] <- 1
  }
  
  l <- rep(0, M-1)
  u <- rep(Inf, M-1)
  
  P <- Matrix::Matrix(P, sparse = TRUE)
  A <- Matrix::Matrix(A, sparse = TRUE)
  
  result <- osqp::solve_osqp(P, q, A, l, u, pars = list(verbose = FALSE))
  
  return(result$x)
}

LocalLinWeights <- function(y = NULL, x = NULL, xOut = NULL, optns = list()) {
  if (is.null(y) | is.null(x)) {
    stop("requires the input of both y and x")
  }
  
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector") 
    }
  }
  n <- nrow(x)
  p <- ncol(x)
  
  if (!is.null(xOut)) {
    if (!is.matrix(xOut)) {
      if (is.data.frame(xOut)) {
        xOut <- as.matrix(xOut)
      } else if (is.vector(xOut)) {
        if (p == 1) {
          xOut <- as.matrix(xOut)
        } else {
          xOut <- t(xOut)
        }
      } else {
        stop("xOut must be a matrix or a data frame or a vector")
      }
    }
    if (ncol(xOut) != p) {
      stop("x and xOut must have the same number of columns")
    }
    nOut <- nrow(xOut)
  } else {
    nOut <- 0
  }
  
  if (is.null(optns$kernel)) {
    optns$kernel <- "gauss"
  }
  
  N <- sapply(y, length)
  y <- lapply(1:n, function(i) {
    sort(y[[i]])
  })
  
  M <- min(plcm(N), n * max(N), 5000)
  yM <- t(sapply(1:n, function(i) {
    residual <- M %% N[i]
    if(residual) {
      sort(c(rep(y[[i]], each = M %/% N[i]), sample(y[[i]], residual)))
    } else {
      rep(y[[i]], each = M %/% N[i])
    }
  }))
  
  if (is.null(optns$bw)) {
    optns$bw <- bwCV(
      xin = x,
      qin = yM,
      xout = xOut,
      optns = optns
    )
  } else {
    if (p == 1) {
      if (optns$bw[1] < max(diff(sort(x[, 1]))) &
          !is.null(optns$kernel)) {
        if (optns$kernel %in% c("rect", "quar", "epan")) {
          warning("optns$bw was set too small and is reset to be chosen by CV.")
          optns$bw <- bwCV(
            xin = x,
            qin = yM,
            xout = xOut,
            optns = optns
          )
        }
      }
    } else {
      if (optns$bw[1] < max(diff(sort(x[, 1]))) &
          optns$bw[2] < max(diff(sort(x[, 2]))) & !is.null(optns$kernel)) {
        if (optns$kernel %in% c("rect", "quar", "epan")) {
          warning("optns$bw was set too small and is reset to be chosen by CV.")
          optns$bw <- bwCV(
            xin = x,
            qin = yM,
            xout = xOut,
            optns = optns
          )
        }
      }
    }
  }
  
  Kern <- kerFctn(optns$kernel)
  K <- function(x, h) {
    k <- 1
    for (i in 1:p) {
      k <- k * Kern(x[i] / h[i])
    }
    return(as.numeric(k))
  }
  
  weights <- matrix(0, nrow = nOut, ncol = n)
  
  for (i in 1:nOut) {
    a <- xOut[i, ]
    if (p > 1) {
      mu1 <- rowMeans(apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * (xi - a)
      }))
      mu2 <- matrix(rowMeans(apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
      })), ncol = p)
    } else {
      mu1 <- mean(apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * (xi - a)
      }))
      mu2 <- mean(apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
      }))
    }
    
    wc <- t(mu1) %*% solve(mu2)
    w <- apply(x, 1, function(xi) {
      K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
    })
    
    weights[i,] <- w
  }
  
  return(weights)
}

lrem <- function(y = NULL,
                 x = NULL,
                 xOut = NULL,
                 optns = list()) {
  if (is.null(y) | is.null(x)) {
    stop("requires the input of both y and x")
  }
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector")
    }
  }
  n <- nrow(x) # number of observations
  p <- ncol(x) # number of covariates
  if (!is.list(y)) {
    stop("y must be a list")
  }
  if (length(y) != n) {
    stop("the number of rows in x must be the same as the number of empirical measures in y")
  }
  if (!is.null(xOut)) {
    if (!is.matrix(xOut)) {
      if (is.data.frame(xOut)) {
        xOut <- as.matrix(xOut)
      } else if (is.vector(xOut)) {
        if (p == 1) {
          xOut <- as.matrix(xOut)
        } else {
          xOut <- t(xOut)
        }
      } else {
        stop("xOut must be a matrix or a data frame or a vector")
      }
    }
    if (ncol(xOut) != p) {
      stop("x and xOut must have the same number of columns")
    }
    nOut <- nrow(xOut) # number of predictions
  } else {
    nOut <- 0
  }
  
  if (!is.null(optns$bw)) {
    if (sum(optns$bw <= 0) > 0) {
      stop("bandwidth must be positive")
    }
    if (length(optns$bw) != p) {
      stop("dimension of bandwidth does not agree with x")
    }
  }
  if (!is.null(optns$bwRange)) {
    if (!is.matrix(optns$bwRange) & !is.vector(optns$bwRange)) {
      stop("bwRange must be a matrix or vector")
    }
    if (is.vector(optns$bwRange)) {
      optns$bwRange <- matrix(optns$bwRange, length(optns$bwRange))
      if (ncol(x) > 1) {
        stop("bwRange must be a matrix")
      } else {
        if (nrow(optns$bwRange) != 2) {
          stop("bwRange must have the lower and upper bound for the bandwidth range")
        }
      }
    } else {
      if (ncol(optns$bwRange) != ncol(x)) {
        stop("bwRange must have the same number of columns as x")
      }
      if (nrow(optns$bwRange) != 2) {
        stop("bwRange must have two rows")
      }
    }
  }
  
  if (is.null(optns$kernel)) {
    optns$kernel <- "gauss"
  }
  
  N <- sapply(y, length)
  y <- lapply(1:n, function(i) {
    sort(y[[i]])
  }) # sort observed values
  
  M <- min(plcm(N), n * max(N), 5000) # least common multiple of N_i
  yM <- t(sapply(1:n, function(i) {
    residual <- M %% N[i]
    if(residual) {
      sort(c(rep(y[[i]], each = M %/% N[i]), sample(y[[i]], residual)))
    } else {
      rep(y[[i]], each = M %/% N[i])
    }
  })) # n by M
  
  # initialization of OSQP solver
  A <- cbind(diag(M), rep(0, M)) + cbind(rep(0, M), -diag(M))
  if (!is.null(optns$upper) &
      !is.null(optns$lower)) {
    # if lower & upper are neither NULL
    l <- c(optns$lower, rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$upper)) {
    # if lower is NULL
    A <- A[, -1]
    l <- c(rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$lower)) {
    # if upper is NULL
    A <- A[, -ncol(A)]
    l <- c(optns$lower, rep(0, M - 1))
  } else {
    # if both lower and upper are NULL
    A <- A[, -c(1, ncol(A))]
    l <- rep(0, M - 1)
  }
  # P <- as(diag(M), "sparseMatrix")
  # A <- as(t(A), "sparseMatrix")
  P <- diag(M)
  A <- t(A)
  q <- rep(0, M)
  u <- rep(Inf, length(l))
  model <-
    osqp::osqp(
      P = P,
      q = q,
      A = A,
      l = l,
      u = u,
      osqp::osqpSettings(max_iter = 1e05, eps_abs = 1e-05, eps_rel = 1e-05, verbose = FALSE)
    )
  
  # select kernel
  Kern <- kerFctn(optns$kernel)
  K <- function(x, h) {
    k <- 1
    for (i in 1:p) {
      k <- k * Kern(x[i] / h[i])
    }
    return(as.numeric(k))
  }
  
  if (is.null(optns$bw)) {
    optns$bw <- bwCV(
      xin = x,
      qin = yM,
      xout = xOut,
      optns = optns
    )
  } else {
    if (ncol(x) == 1) {
      if (optns$bw[1] < max(diff(sort(x[, 1]))) &
          !is.null(optns$kernel)) {
        if (optns$kernel %in% c("rect", "quar", "epan")) {
          warning("optns$bw was set too small and is reset to be chosen by CV.")
          optns$bw <-
            bwCV(
              xin = x,
              qin = yM,
              xout = xOut,
              optns = optns
            )
        }
      }
    } else {
      if (optns$bw[1] < max(diff(sort(x[, 1]))) &
          optns$bw[2] < max(diff(sort(x[, 2]))) & !is.null(optns$kernel)) {
        if (optns$kernel %in% c("rect", "quar", "epan")) {
          warning("optns$bw was set too small and is reset to be chosen by CV.")
          optns$bw <-
            bwCV(
              xin = x,
              qin = yM,
              xout = xOut,
              optns = optns
            )
        }
      }
    }
  }
  
  qf <- matrix(nrow = n, ncol = M)
  residuals <- rep.int(0, n)
  for (i in 1:n) {
    a <- x[i, ]
    if (p > 1) {
      mu1 <-
        rowMeans(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * (xi - a)
        }))
      mu2 <-
        matrix(rowMeans(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
        })), ncol = p)
    } else {
      mu1 <-
        mean(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * (xi - a)
        }))
      mu2 <-
        mean(apply(x, 1, function(xi) {
          K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
        }))
    }
    wc <- t(mu1) %*% solve(mu2) # 1 by p
    w <- apply(x, 1, function(xi) {
      K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
    }) # weight
    qNew <- apply(yM, 2, weighted.mean, w) # M
    if (any(w < 0)) {
      # if negative weights exist, project
      model$Update(q = -qNew)
      qNew <- sort(model$Solve()$x)
    }
    if (!is.null(optns$upper)) {
      qNew <- pmin(qNew, optns$upper)
    }
    if (!is.null(optns$lower)) {
      qNew <- pmax(qNew, optns$lower)
    }
    qf[i, ] <- qNew
    residuals[i] <- sqrt(sum((yM[i, ] - qf[i, ])^2) / M)
  }
  qfSupp <- 1:M / M
  
  if (nOut > 0) {
    qp <- matrix(nrow = nOut, ncol = M)
    for (i in 1:nOut) {
      a <- xOut[i, ]
      if (p > 1) {
        mu1 <-
          rowMeans(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * (xi - a)
          }))
        mu2 <-
          matrix(rowMeans(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
          })), ncol = p)
      } else {
        mu1 <-
          mean(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * (xi - a)
          }))
        mu2 <-
          mean(apply(x, 1, function(xi) {
            K(xi - a, optns$bw) * ((xi - a) %*% t(xi - a))
          }))
      }
      wc <- t(mu1) %*% solve(mu2) # 1 by p
      w <- apply(x, 1, function(xi) {
        K(xi - a, optns$bw) * (1 - wc %*% (xi - a))
      }) # weight
      qNew <- apply(yM, 2, weighted.mean, w) # M
      if (any(w < 0)) {
        # if negative weights exist
        model$Update(q = -qNew)
        qNew <- sort(model$Solve()$x)
      }
      if (!is.null(optns$upper)) {
        qNew <- pmin(qNew, optns$upper)
      }
      if (!is.null(optns$lower)) {
        qNew <- pmax(qNew, optns$lower)
      }
      qp[i, ] <- qNew
    }
    qpSupp <- 1:M / M
    
    res <-
      list(
        qf = qf,
        qfSupp = qfSupp,
        qp = qp,
        qpSupp = qpSupp,
        residuals = residuals,
        y = y,
        x = x,
        xOut = xOut,
        optns = optns
      )
  } else {
    res <- list(
      qf = qf,
      qfSupp = qfSupp,
      residuals = residuals,
      y = y,
      x = x,
      optns = optns
    )
  }
  
  class(res) <- "rem"
  res
}

grem <- function(y = NULL,
                 x = NULL,
                 xOut = NULL,
                 optns = list()) {
  if (is.null(y) | is.null(x)) {
    stop("requires the input of both y and x")
  }
  if (!is.matrix(x)) {
    if (is.data.frame(x) | is.vector(x)) {
      x <- as.matrix(x)
    } else {
      stop("x must be a matrix or a data frame or a vector")
    }
  }
  n <- nrow(x) # number of observations
  p <- ncol(x) # number of covariates
  if (!is.list(y)) {
    stop("y must be a list")
  }
  if (length(y) != n) {
    stop("the number of rows in x must be the same as the number of empirical measures in y")
  }
  if (!is.null(xOut)) {
    if (!is.matrix(xOut)) {
      if (is.data.frame(xOut)) {
        xOut <- as.matrix(xOut)
      } else if (is.vector(xOut)) {
        if (p == 1) {
          xOut <- as.matrix(xOut)
        } else {
          xOut <- t(xOut)
        }
      } else {
        stop("xOut must be a matrix or a data frame or a vector")
      }
    }
    if (ncol(xOut) != p) {
      stop("x and xOut must have the same number of columns")
    }
    nOut <- nrow(xOut) # number of predictions
  } else {
    nOut <- 0
  }

  N <- sapply(y, length)
  y <- lapply(1:n, function(i) {
    sort(y[[i]])
  }) # sort observed values

  M <- min(plcm(N), n * max(N), 5000) # least common multiple of N_i
  yM <- t(sapply(1:n, function(i) {
    residual <- M %% N[i]
    if(residual) {
      sort(c(rep(y[[i]], each = M %/% N[i]), sample(y[[i]], residual)))
    } else {
      rep(y[[i]], each = M %/% N[i])
    }
  })) # n by M

  # initialization of OSQP solver
  A <- cbind(diag(M), rep(0, M)) + cbind(rep(0, M), -diag(M))
  if (!is.null(optns$upper) &
      !is.null(optns$lower)) {
    # if lower & upper are neither NULL
    l <- c(optns$lower, rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$upper)) {
    # if lower is NULL
    A <- A[, -1]
    l <- c(rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$lower)) {
    # if upper is NULL
    A <- A[, -ncol(A)]
    l <- c(optns$lower, rep(0, M - 1))
  } else {
    # if both lower and upper are NULL
    A <- A[, -c(1, ncol(A))]
    l <- rep(0, M - 1)
  }
  # P <- as(diag(M), "sparseMatrix")
  # A <- as(t(A), "sparseMatrix")
  P <- diag(M)
  A <- t(A)
  q <- rep(0, M)
  u <- rep(Inf, length(l))
  model <-
    osqp::osqp(
      P = P,
      q = q,
      A = A,
      l = l,
      u = u,
      osqp::osqpSettings(max_iter = 1e05, eps_abs = 1e-05, eps_rel = 1e-05, verbose = FALSE)
    )

  xMean <- colMeans(x)
  invVa <- solve(var(x) * (n - 1) / n)
  wc <-
    t(apply(x, 1, function(xi) {
      t(xi - xMean) %*% invVa
    })) # n by p
  if (nrow(wc) != n) {
    wc <- t(wc)
  } # for p=1

  qf <- matrix(nrow = n, ncol = M)
  residuals <- rep.int(0, n)
  totVa <- sum((scale(yM, scale = FALSE))^2) / M
  for (i in 1:n) {
    w <- apply(wc, 1, function(wci) {
      1 + t(wci) %*% (x[i, ] - xMean)
    })
    qNew <- apply(yM, 2, weighted.mean, w) # M
    if (any(w < 0)) {
      # if negative weights exist, project
      model$Update(q = -qNew)
      qNew <- sort(model$Solve()$x)
    }
    if (!is.null(optns$upper)) {
      qNew <- pmin(qNew, optns$upper)
    }
    if (!is.null(optns$lower)) {
      qNew <- pmax(qNew, optns$lower)
    }
    qf[i, ] <- qNew
    residuals[i] <- sqrt(sum((yM[i, ] - qf[i, ])^2) / M)
  }
  qfSupp <- 1:M / M
  resVa <- sum(residuals^2)
  RSquare <- 1 - resVa / totVa
  adjRSquare <- RSquare - (1 - RSquare) * p / (n - p - 1)

  if (nOut > 0) {
    qp <- matrix(nrow = nOut, ncol = M)
    for (i in 1:nOut) {
      w <- apply(wc, 1, function(wci) {
        1 + t(wci) %*% (xOut[i, ] - xMean)
      })
      qNew <- apply(yM, 2, weighted.mean, w) # M
      if (any(w < 0)) {
        # if negative weights exist
        model$Update(q = -qNew)
        qNew <- sort(model$Solve()$x)
      }
      if (!is.null(optns$upper)) {
        qNew <- pmin(qNew, optns$upper)
      }
      if (!is.null(optns$lower)) {
        qNew <- pmax(qNew, optns$lower)
      }
      qp[i, ] <- qNew
    }
    qpSupp <- 1:M / M

    res <-
      list(
        qf = qf,
        qfSupp = qfSupp,
        qp = qp,
        qpSupp = qpSupp,
        RSquare = RSquare,
        adjRSquare = adjRSquare,
        residuals = residuals,
        y = y,
        x = x,
        xOut = xOut,
        optns = optns
      )
  } else {
    res <- list(
      qf = qf,
      qfSupp = qfSupp,
      RSquare = RSquare,
      adjRSquare = adjRSquare,
      residuals = residuals,
      y = y,
      x = x,
      optns = optns
    )
  }

  class(res) <- "rem"
  res
}

# 
plcm <- function(x) {
  stopifnot(is.numeric(x))
  # if (any(floor(x) != ceiling(x)) || length(x) < 2)
  #   stop("Argument 'x' must be an integer vector of length >= 2.")

  x <- x[x != 0]
  n <- length(x)
  if (n == 0) {
    l <- 0
  } else if (n == 1) {
    l <- x
  } else if (n == 2) {
    l <- lcm(x[1], x[2])
  } else {
    l <- lcm(x[1], x[2])
    for (i in 3:n) {
      l <- lcm(l, x[i])
    }
  }
  return(l)
}

lcm <- function(n, m) {
  stopifnot(is.numeric(n), is.numeric(m))
  if (length(n) != 1 || floor(n) != ceiling(n) ||
      length(m) != 1 || floor(m) != ceiling(m)) {
    stop("Arguments 'n', 'm' must be integer scalars.")
  }
  if (n == 0 && m == 0) {
    return(0)
  }

  return(n / gcd(n, m) * m)
}
# 
gcd <- function(n, m) {
  stopifnot(is.numeric(n), is.numeric(m))
  if (length(n) != 1 || floor(n) != ceiling(n) ||
      length(m) != 1 || floor(m) != ceiling(m)) {
    stop("Arguments 'n', 'm' must be integer scalars.")
  }
  if (n == 0 && m == 0) {
    return(0)
  }

  n <- abs(n)
  m <- abs(m)
  if (m > n) {
    t <- n
    n <- m
    m <- t
  }
  while (m > 0) {
    t <- n
    n <- m
    m <- t %% m
  }
  return(n)
}

compute_L2_distance <- function(vec1, vec2){
  sqrt(mean( (vec1 - vec2)^2 ))
}


kerFctn <- function(kernel_type){
  if (kernel_type=='gauss'){
    ker <- function(x){
      dnorm(x) #exp(-x^2 / 2) / sqrt(2*pi)
    }
  } else if(kernel_type=='rect'){
    ker <- function(x){
      as.numeric((x<=1) & (x>=-1))
    }
  } else if(kernel_type=='epan'){
    ker <- function(x){
      n <- 1
      (2*n+1) / (4*n) * (1-x^(2*n)) * (abs(x)<=1)
    }
  } else if(kernel_type=='gausvar'){
    ker <- function(x) {
      dnorm(x)*(1.25-0.25*x^2)
    }
  } else if(kernel_type=='quar'){
    ker <- function(x) {
      (15/16)*(1-x^2)^2 * (abs(x)<=1)
    }
  } else {
    stop('Unavailable kernel')
  }
  return(ker)
}

SetBwRange <- function(xin, xout, kernel_type) {
  xinSt <- unique(sort(xin))
  bw.min <- max(diff(xinSt), xinSt[2] - min(xout), max(xout) -
                  xinSt[length(xinSt) - 1]) * 1.1 / (ifelse(kernel_type == "gauss", 3, 1) *
                                                       ifelse(kernel_type == "gausvar", 2.5, 1))
  bw.max <- diff(range(xin)) / 3
  if (bw.max < bw.min) {
    if (bw.min > bw.max * 3 / 2) {
      # warning("Data is too sparse.")
      bw.max <- bw.min * 1.01
    } else {
      bw.max <- bw.max * 3 / 2
    }
  }
  return(list(min = bw.min, max = bw.max))
}

# bandwidth selection via cross validation
bwCV <- function(xin, qin, xout, optns) {
  if(is.null(xout)) {
    xout <- xin
  }
  n <- nrow(xin)
  p <- ncol(xin)
  # initialization of OSQP solver
  M <- ncol(qin)
  A <- cbind(diag(M), rep(0, M)) + cbind(rep(0, M), -diag(M))
  if (!is.null(optns$upper) &
      !is.null(optns$lower)) {
    # if lower & upper are neither NULL
    l <- c(optns$lower, rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$upper)) {
    # if lower is NULL
    A <- A[, -1]
    l <- c(rep(0, M - 1), -optns$upper)
  } else if (!is.null(optns$lower)) {
    # if upper is NULL
    A <- A[, -ncol(A)]
    l <- c(optns$lower, rep(0, M - 1))
  } else {
    # if both lower and upper are NULL
    A <- A[, -c(1, ncol(A))]
    l <- rep(0, M - 1)
  }
  # P <- as(diag(M), "sparseMatrix")
  # A <- as(t(A), "sparseMatrix")
  P <- diag(M)
  A <- t(A)
  q <- rep(0, M)
  u <- rep(Inf, length(l))
  model <-
    osqp::osqp(
      P = P,
      q = q,
      A = A,
      l = l,
      u = u,
      osqp::osqpSettings(verbose = FALSE)
    )
  
  # select kernel
  Kern <- kerFctn(optns$kernel)
  K <- function(x, h) {
    k <- 1
    for (i in 1:p) {
      k <- k * Kern(x[i] / h[i])
    }
    return(as.numeric(k))
  }
  
  # k-fold
  objFctn <- function(h) {
    numFolds <- ifelse(n > 30, 10, n)# leave-one-out or 10-fold cross-validation
    folds <- sample(c(rep.int(1:numFolds, n%/%numFolds), seq_len(n%%numFolds)))
    
    cv <- 0
    for (foldidx in seq_len(numFolds)) {
      # nn by M
      testidx <- which(folds == foldidx)
      for (j in testidx) {
        a <- xin[j, ]
        if (p > 1) {
          mu1 <-
            rowMeans(apply(xin[-testidx, ], 1, function(xi)
              K(xi - a, h) * (xi - a)))
          mu2 <-
            matrix(rowMeans(apply(xin[-testidx, ], 1, function(xi)
              K(xi - a, h) * ((xi - a) %*% t(xi - a)))), ncol = p)
        } else {
          mu1 <-
            mean(sapply(xin[-testidx, ], function(xi)
              K(xi - a, h) * (xi - a)))
          mu2 <-
            mean(sapply(xin[-testidx, ], function(xi)
              K(xi - a, h) * (xi - a)^2))
        }
        wc <- t(mu1) %*% solve(mu2) # 1 by p
        w <- apply(as.matrix(xin[-testidx, ]), 1, function(xi) {
          K(xi - a, h) * (1 - wc %*% (xi - a))
        }) # weight
        qNew <- apply(qin[-testidx,], 2, weighted.mean, w) # N
        if (any(w < 0)) {
          # if negative weights exist
          model$Update(q = -qNew)
          cv <- cv + sum((qin[j, ] - sort(model$Solve()$x))^2) / (n * M)
        } else {
          cv <- cv + sum((qin[j, ] - qNew)^2) / (n * M)
        }
      }
    }
    cv
  }
  
  if (p == 1) {
    aux <-
      SetBwRange(xin = xin[, 1],
                 xout = xout[, 1],
                 kernel_type = optns$kernel)
    bwRange <- matrix(c(aux$min, aux$max), nrow = 2, ncol = 1)
  } else {
    aux <-
      SetBwRange(xin = xin[, 1],
                 xout = xout[, 1],
                 kernel_type = optns$kernel)
    aux2 <-
      SetBwRange(xin = xin[, 2],
                 xout = xout[, 2],
                 kernel_type = optns$kernel)
    bwRange <-
      as.matrix(cbind(c(aux$min, aux$max), c(aux2$min, aux2$max)))
  }
  if (!is.null(optns$bw)) {
    if (p == 1) {
      if (min(optns$bw) < min(bwRange)) {
        message("Minimum bandwidth is too small and has been reset.")
      } else {
        bwRange[1, 1] <- min(optns$bw)
      }
      if (max(optns$bw) > min(bwRange)) {
        bwRange[2, 1] <- max(optns$bw)
      } else {
        message("Maximum bandwidth is too small and has been reset.")
      }
    } else {
      # Check for first dimension of the predictor
      if (min(optns$bw[, 1]) < min(bwRange[, 1])) {
        message("Minimum bandwidth of first predictor dimension is too small and has been reset.")
      } else {
        bwRange[1, 1] <- min(optns$bw[, 1])
      }
      if (max(optns$bw[, 1]) > min(bwRange[, 1])) {
        bwRange[2, 1] <- max(optns$bw[, 1])
      } else {
        message("Maximum bandwidth of first predictor dimension is too small and has been reset.")
      }
      # Check for second dimension of the predictor
      if (min(optns$bw[, 2]) < min(bwRange[, 2])) {
        message("Minimum bandwidth of second predictor dimension is too small and has been reset.")
      } else {
        bwRange[1, 2] <- min(optns$bw[, 2])
      }
      if (max(optns$bw[, 2]) > min(bwRange[, 2])) {
        bwRange[2, 2] <- max(optns$bw[, 2])
      } else {
        message("Maximum bandwidth of second predictor dimension is too small and has been reset.")
      }
    }
  }
  if (p == 1) {
    res <- optimize(f = objFctn, interval = bwRange[, 1])$minimum
  } else {
    res <-
      optim(
        par = colMeans(bwRange),
        fn = objFctn,
        lower = bwRange[1, ],
        upper = bwRange[2, ],
        method = "L-BFGS-B"
      )$par
  }
  res
}