required_packages <- c("ggplot2", "dplyr", "tidyr", "patchwork")

load_simulation_packages <- function() {
  missing <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]

  if (length(missing) > 0) {
    stop(
      "Missing required R packages: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(patchwork)
  })
}

source_simulation_code <- function() {
  source(file.path("R", "methods.R"))
  source(file.path("R", "spi_methods.R"))
  source(file.path("R", "rsa_cp_methods.R"))
  source(file.path("R", "data_generating_process.R"))
  source(file.path("R", "plotting_helpers.R"))
}

set_simulation_root <- function(anchor = NULL) {
  if (!is.null(anchor)) {
    root <- normalizePath(anchor, mustWork = TRUE)
  } else if (file.exists(file.path("R", "simulation_helpers.R"))) {
    root <- normalizePath(".", mustWork = TRUE)
  } else if (file.exists(file.path("..", "R", "simulation_helpers.R"))) {
    root <- normalizePath("..", mustWork = TRUE)
  } else {
    stop("Run from simulations/ or one of its immediate child folders.")
  }

  setwd(root)
  invisible(root)
}

simulation_config <- function() {
  list(
    seed = 20260529,
    alpha = 0.05,
    beta = 0.4,
    n_trials = 100,
    n_train = 1500,
    n_pool = 50000,
    n_test = 2000,
    n_cal_fixed = 40,
    n_grid = c(50, 100, 250, 500, 1000, 2000, 3000),
    ncal_grid = c(20, 30, 40, 80, 160),
    n_ref_fixed = 3000,
    noise_grid = c(0.25, 0.5, 1, 1.5, 2, 3),
    shock_probs = c(0, 0.1, 0.2, 0.3, 0.4, 0.5),
    n_cal_shock = 20,
    n_ref_shock = 1000,
    shock_scale = 10,
    rsa_grid_size = 1200
  )
}

method_levels <- c("SCP", "RSA-CP (OT) (Ours)", "SPI", "Synthetic-only")
shock_method_levels <- c("SCP", "RSA-CP (OT) (Ours)")

ensure_output_dirs <- function() {
  dirs <- c(
    file.path("outputs", "raw"),
    file.path("outputs", "summary"),
    file.path("outputs", "figures"),
    file.path("outputs", "logs")
  )
  invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))
}

log_message <- function(..., file = file.path("outputs", "logs", "run_log.txt")) {
  ensure_output_dirs()
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = ""))
  cat(msg, "\n")
  cat(msg, "\n", file = file, append = TRUE)
}

time_it <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- eval.parent(substitute(expr))
  seconds <- as.numeric(proc.time()[["elapsed"]] - start)
  list(value = value, seconds = seconds)
}

make_dataset_bundle <- function(dataset_name, dp, test, idx_tr, idx_ca,
                                idx_ref_pool = NULL) {
  model_bundle <- train_models(dp$X[idx_tr, ], dp$Y[idx_tr])
  S_real <- get_scores(model_bundle$main, dp$X[idx_ca, ], dp$Y[idx_ca])
  S_test <- get_scores(model_bundle$main, test$X, test$Y)

  X_ref_pool <- Y_ref_pool <- NULL
  if (!is.null(idx_ref_pool)) {
    X_ref_pool <- dp$X[idx_ref_pool, ]
    Y_ref_pool <- dp$Y[idx_ref_pool]
  }

  list(
    dataset = dataset_name,
    model_bundle = model_bundle,
    S_real = S_real,
    S_test = S_test,
    X_ref_pool = X_ref_pool,
    Y_ref_pool = Y_ref_pool
  )
}

get_reference_scores_for_bundle <- function(bundle, n_ref) {
  if (!is.null(bundle$X_ref_pool) && nrow(bundle$X_ref_pool) >= n_ref) {
    get_scores(
      bundle$model_bundle$main,
      bundle$X_ref_pool[seq_len(n_ref), , drop = FALSE],
      bundle$Y_ref_pool[seq_len(n_ref)]
    )
  } else {
    gen_gamma_reference_scores(n_ref, bundle$S_real)
  }
}

evaluate_methods_for_size <- function(bundle, n_ref, x_name, x_value, trial_id,
                                      config) {
  S_real <- bundle$S_real
  S_test <- bundle$S_test
  model_main <- bundle$model_bundle$main

  scp_timed <- time_it({
    q_scp <- get_standard_quantile(S_real, config$alpha)
    list(q = q_scp, metrics = compute_interval_metrics(S_test, q_scp))
  })

  rsa_timed <- time_it({
    S_ref <- get_reference_scores_for_bundle(bundle, n_ref)
    q_rsa <- get_rsacp_quantile(
      S_real = S_real,
      S_ref = S_ref,
      alpha = config$alpha,
      beta = config$beta,
      use_ot = TRUE,
      grid_size = config$rsa_grid_size
    )
    list(q = q_rsa, metrics = compute_interval_metrics(S_test, q_rsa))
  })

  syn_score_timed <- time_it({
    syn <- get_synthetic_data(bundle$dataset, n_ref, bundle$model_bundle)
    get_scores(model_main, syn$X, syn$Y)
  })
  S_syn <- syn_score_timed$value

  spi_timed <- time_it({
    q_spi <- get_spi_quantile(
      S_real = S_real,
      S_syn = S_syn,
      alpha = config$alpha,
      beta = config$beta
    )
    list(q = q_spi, metrics = compute_interval_metrics(S_test, q_spi))
  })

  synthetic_only_timed <- time_it({
    q_syn <- get_standard_quantile(S_syn, config$alpha)
    list(q = q_syn, metrics = compute_interval_metrics(S_test, q_syn))
  })

  out <- dplyr::bind_rows(
    cbind(Method = "SCP", scp_timed$value$metrics, Time = scp_timed$seconds),
    cbind(Method = "RSA-CP (OT) (Ours)", rsa_timed$value$metrics, Time = rsa_timed$seconds),
    cbind(Method = "SPI", spi_timed$value$metrics, Time = syn_score_timed$seconds + spi_timed$seconds),
    cbind(Method = "Synthetic-only", synthetic_only_timed$value$metrics, Time = syn_score_timed$seconds + synthetic_only_timed$seconds)
  ) |>
    dplyr::mutate(
      Dataset = bundle$dataset,
      Trial = trial_id,
      N = n_ref,
      Method = factor(Method, levels = method_levels)
    )

  out[[x_name]] <- x_value
  out
}

summarise_by_method <- function(results, group_vars) {
  results |>
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_vars, "Method")))) |>
    dplyr::summarise(
      Mean_Cov = mean(Cov),
      SD_Cov = sd(Cov),
      Mean_Width = mean(Width),
      SD_Width = sd(Width),
      Total_Time = sum(Time, na.rm = TRUE),
      Mean_Time = mean(Time, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(Method = factor(Method, levels = method_levels))
}

write_result_pair <- function(raw, summary, raw_name, summary_name) {
  ensure_output_dirs()
  write.csv(raw, file.path("outputs", "raw", raw_name), row.names = FALSE)
  write.csv(summary, file.path("outputs", "summary", summary_name), row.names = FALSE)
}
