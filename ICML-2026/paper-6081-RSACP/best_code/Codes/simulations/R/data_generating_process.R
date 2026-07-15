get_data_lognormal_true <- function(n) {
  X1 <- rlnorm(n, 0, 0.5)
  X2 <- rt(n, 3)
  X3 <- runif(n, -2, 2)
  X4 <- rnorm(n)
  X5 <- 0.6 * X4 + 0.8 * rnorm(n)
  X <- cbind(X1, X2, X3, X4, X5)
  colnames(X) <- paste0("X", 1:5)

  mu <- 1.5 * log(X1) + abs(X2) + X3 * X4 + 0.5 * X5^2
  sigma <- 0.5 + 0.5 * X1
  Y <- mu + sigma * rnorm(n)

  list(X = X, Y = Y)
}

get_data_student_true <- function(n) {
  X1 <- runif(n, -2, 2)
  X2 <- runif(n, -2, 2)
  X3 <- rnorm(n)
  X4 <- rnorm(n)
  X5 <- rnorm(n)
  X <- cbind(X1, X2, X3, X4, X5)
  colnames(X) <- paste0("X", 1:5)

  mu <- 2 * X1 + 1.5 * X2 - X3
  Y <- mu + rt(n, df = 3)

  list(X = X, Y = Y)
}

train_models <- function(X_tr, Y_tr) {
  df_tr <- data.frame(y = Y_tr, X_tr)
  fit_main <- lm(y ~ ., data = df_tr)
  abs_resid <- abs(fit_main$residuals)
  fit_resid <- lm(abs_resid ~ ., data = data.frame(abs_resid, X_tr))
  list(main = fit_main, resid_model = fit_resid)
}

get_scores <- function(model, X, Y) {
  abs(Y - predict(model, newdata = data.frame(X)))
}

get_m3_syn_lognormal_data <- function(n, model) {
  X1 <- rlnorm(n, 0, 0.5)
  X2 <- rt(n, 3)
  X3 <- runif(n, -2, 2)
  X4 <- rnorm(n)
  X5 <- 0.6 * X4 + 0.8 * rnorm(n)
  X_syn <- cbind(X1, X2, X3, X4, X5)
  colnames(X_syn) <- paste0("X", 1:5)

  pred_y <- predict(model$main, newdata = data.frame(X_syn))
  pred_score <- predict(model$resid_model, newdata = data.frame(X_syn))
  shock <- rbinom(n, 1, 0.05) * rnorm(n, 8, 2)
  S_syn <- pmax(0, pred_score + rnorm(n, 0, 1.0) + shock)
  Y_syn <- pred_y + sample(c(-1, 1), n, replace = TRUE) * S_syn

  list(X = X_syn, Y = Y_syn)
}

get_m3_syn_student_data <- function(n, model) {
  X1 <- runif(n, -2, 2)
  X2 <- runif(n, -2, 2)
  X3 <- rnorm(n)
  X4 <- rnorm(n)
  X5 <- rnorm(n)
  X_syn <- cbind(X1, X2, X3, X4, X5)
  colnames(X_syn) <- paste0("X", 1:5)

  mu <- 2 * X1 + 1.5 * X2 - X3
  epsilon_noisy <- rt(n, df = 3) + rnorm(n, 0, 0.5)
  shock <- rbinom(n, 1, 0.05) * runif(n, 5, 10)
  Y_syn <- mu + epsilon_noisy + shock

  list(X = X_syn, Y = Y_syn)
}

get_m3_syn_lognormal_var_noise_data <- function(n, model, noise_scale) {
  X1 <- rlnorm(n, 0, 0.5)
  X2 <- rt(n, 3)
  X3 <- runif(n, -2, 2)
  X4 <- rnorm(n)
  X5 <- 0.6 * X4 + 0.8 * rnorm(n)
  X_syn <- cbind(X1, X2, X3, X4, X5)
  colnames(X_syn) <- paste0("X", 1:5)

  pred_y <- predict(model$main, newdata = data.frame(X_syn))
  pred_score <- predict(model$resid_model, newdata = data.frame(X_syn))
  S_syn <- pmax(0, pred_score + rnorm(n, 0, noise_scale))
  Y_syn <- pred_y + sample(c(-1, 1), n, replace = TRUE) * S_syn

  list(X = X_syn, Y = Y_syn)
}

get_m3_syn_student_var_noise_data <- function(n, model, noise_scale) {
  X1 <- runif(n, -2, 2)
  X2 <- runif(n, -2, 2)
  X3 <- rnorm(n)
  X4 <- rnorm(n)
  X5 <- rnorm(n)
  X_syn <- cbind(X1, X2, X3, X4, X5)
  colnames(X_syn) <- paste0("X", 1:5)

  mu <- 2 * X1 + 1.5 * X2 - X3
  Y_syn <- mu + rt(n, df = 3) + rnorm(n, 0, noise_scale)

  list(X = X_syn, Y = Y_syn)
}

get_synthetic_data <- function(dataset_name, n, model_bundle) {
  if (dataset_name == "LogNormal") {
    get_m3_syn_lognormal_data(n, model_bundle)
  } else {
    get_m3_syn_student_data(n, model_bundle)
  }
}

get_synthetic_data_with_noise <- function(dataset_name, n, model_bundle,
                                          noise_scale) {
  if (dataset_name == "LogNormal") {
    get_m3_syn_lognormal_var_noise_data(n, model_bundle, noise_scale)
  } else {
    get_m3_syn_student_var_noise_data(n, model_bundle, noise_scale)
  }
}

gen_gamma_reference_scores <- function(n, scores) {
  mu <- mean(scores)
  va <- var(scores)
  if (va < 1e-9) va <- 1e-9
  if (mu < 1e-9) mu <- 1e-9
  rgamma(n, shape = mu^2 / va, scale = va / mu)
}

gen_normal_reference_scores <- function(n, scores) {
  mu <- mean(scores)
  sd_val <- max(sd(scores), 1e-8)
  pmax(0, rnorm(n, mean = mu, sd = sd_val))
}

gen_beta_reference_scores <- function(n, scores) {
  upper_bound <- max(scores, na.rm = TRUE) * 1.2
  if (upper_bound < 1e-6) upper_bound <- 1.0

  scaled_scores <- pmin(pmax(scores / upper_bound, 1e-6), 1 - 1e-6)
  m <- mean(scaled_scores)
  v <- var(scaled_scores)

  if (is.finite(v) && v > 1e-10 && v < m * (1 - m)) {
    alpha <- m * ((m * (1 - m)) / v - 1)
    beta_param <- (1 - m) * ((m * (1 - m)) / v - 1)
  } else {
    alpha <- 1
    beta_param <- 1
  }

  rbeta(n, shape1 = alpha, shape2 = beta_param) * upper_bound
}

apply_reference_tail_shock <- function(S_ref_clean, shock_prob, shock_scale) {
  if (shock_prob <= 0) return(S_ref_clean)

  shocked <- rbinom(length(S_ref_clean), 1, shock_prob) == 1
  if (!any(shocked)) return(S_ref_clean)

  tail_sd <- max(sd(S_ref_clean), 1e-8)
  S_ref_clean[shocked] <- S_ref_clean[shocked] +
    shock_scale * tail_sd +
    shock_scale * abs(rt(sum(shocked), df = 3)) * tail_sd

  S_ref_clean
}
