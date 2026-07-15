helper_path <- if (file.exists(file.path("R", "simulation_helpers.R"))) {
  file.path("R", "simulation_helpers.R")
} else {
  file.path("..", "R", "simulation_helpers.R")
}
source(helper_path)
set_simulation_root()
load_simulation_packages()
source_simulation_code()
ensure_output_dirs()

config <- simulation_config()
set.seed(config$seed + 7)

run_figure7_calibration_size <- function(config = simulation_config()) {
  results <- list()

  dp_ln <- get_data_lognormal_true(config$n_pool)
  test_ln <- get_data_lognormal_true(config$n_test)
  dp_st <- get_data_student_true(config$n_pool)
  test_st <- get_data_student_true(config$n_test)

  log_message("Figure 7 left: calibration-size experiment started.")
  counter <- 1

  for (n_cal in config$ncal_grid) {
    log_message("Figure 7 calibration size m = ", n_cal)

    for (trial_id in seq_len(config$n_trials)) {
      idx_tr <- seq_len(config$n_train)
      idx_ca <- sample((config$n_train + 1):config$n_pool, n_cal)
      idx_ref_pool <- sample(
        setdiff((config$n_train + 1):config$n_pool, idx_ca),
        config$n_ref_fixed,
        replace = FALSE
      )

      bundles <- list(
        make_dataset_bundle("LogNormal", dp_ln, test_ln, idx_tr, idx_ca, idx_ref_pool),
        make_dataset_bundle("Student-t", dp_st, test_st, idx_tr, idx_ca, idx_ref_pool)
      )

      for (bundle in bundles) {
        results[[counter]] <- evaluate_methods_for_size(
          bundle = bundle,
          n_ref = config$n_ref_fixed,
          x_name = "N_cal",
          x_value = n_cal,
          trial_id = trial_id,
          config = config
        )
        counter <- counter + 1
      }
    }
  }

  raw <- dplyr::bind_rows(results)
  summary <- summarise_by_method(raw, c("Dataset", "N_cal"))
  write_result_pair(
    raw,
    summary,
    "figure7_calibration_size_raw.csv",
    "figure7_calibration_size_summary.csv"
  )

  list(raw = raw, summary = summary)
}

run_figure7_score_distribution <- function(config = simulation_config()) {
  n_cal <- config$n_cal_fixed
  n_ref <- config$n_ref_fixed
  results <- list()

  dp_ln <- get_data_lognormal_true(config$n_pool)
  test_ln <- get_data_lognormal_true(config$n_test)
  dp_st <- get_data_student_true(config$n_pool)
  test_st <- get_data_student_true(config$n_test)

  log_message("Figure 7 right: score-distribution boxplot experiment started.")
  counter <- 1

  for (trial_id in seq_len(config$n_trials)) {
    if (trial_id %% 10 == 0) log_message("Figure 7 distribution trial ", trial_id, " of ", config$n_trials)

    idx_tr <- seq_len(config$n_train)
    idx_ca <- sample((config$n_train + 1):config$n_pool, n_cal)

    bundles <- list(
      make_dataset_bundle("LogNormal", dp_ln, test_ln, idx_tr, idx_ca),
      make_dataset_bundle("Student-t", dp_st, test_st, idx_tr, idx_ca)
    )

    for (bundle in bundles) {
      q_scp <- get_standard_quantile(bundle$S_real, config$alpha)
      dist_specs <- list(
        "SCP" = NULL,
        "RSA-CP Gamma" = gen_gamma_reference_scores(n_ref, bundle$S_real),
        "RSA-CP Normal" = gen_normal_reference_scores(n_ref, bundle$S_real),
        "RSA-CP Beta" = gen_beta_reference_scores(n_ref, bundle$S_real)
      )

      for (method_name in names(dist_specs)) {
        if (method_name == "SCP") {
          q <- q_scp
        } else {
          q <- get_rsacp_quantile(
            S_real = bundle$S_real,
            S_ref = dist_specs[[method_name]],
            alpha = config$alpha,
            beta = config$beta,
            use_ot = TRUE,
            grid_size = config$rsa_grid_size
          )
        }

        results[[counter]] <- compute_interval_metrics(bundle$S_test, q) |>
          dplyr::mutate(
            Dataset = bundle$dataset,
            Trial = trial_id,
            Score_Distribution = method_name
          )
        counter <- counter + 1
      }
    }
  }

  raw <- dplyr::bind_rows(results)
  summary <- raw |>
    dplyr::group_by(Dataset, Score_Distribution) |>
    dplyr::summarise(
      Mean_Cov = mean(Cov),
      SD_Cov = sd(Cov),
      Mean_Width = mean(Width),
      SD_Width = sd(Width),
      .groups = "drop"
    )

  write_result_pair(
    raw,
    summary,
    "figure7_score_distribution_boxplot_raw.csv",
    "figure7_score_distribution_boxplot_summary.csv"
  )

  list(raw = raw, summary = summary)
}

calibration <- run_figure7_calibration_size(config)
distribution <- run_figure7_score_distribution(config)
fig <- plot_figure7_combined(calibration$summary, distribution$raw, config)
save_plot_pdf_png(fig, "figure7_calibration_size_and_score_distribution", width = 15, height = 6.6)
log_message("Figure 7: combined outputs written.")
