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
set.seed(config$seed + 8)

run_shock_probability_experiment <- function(config = simulation_config()) {
  n_cal <- config$n_cal_shock
  n_ref <- config$n_ref_shock
  results <- list()

  dp_ln <- get_data_lognormal_true(config$n_pool)
  test_ln <- get_data_lognormal_true(config$n_test)
  dp_st <- get_data_student_true(config$n_pool)
  test_st <- get_data_student_true(config$n_test)

  log_message("Shock probability experiment started.")
  log_message(
    "Shock design: p = 0 uses matched real/reference score distributions; ",
    "p > 0 corrupts only reference scores with positive tail shocks."
  )
  counter <- 1

  for (trial_id in seq_len(config$n_trials)) {
    if (trial_id %% 10 == 0) log_message("Shock trial ", trial_id, " of ", config$n_trials)

    idx_tr <- seq_len(config$n_train)
    idx_ca <- sample((config$n_train + 1):config$n_pool, n_cal)
    idx_ref_pool <- sample(
      setdiff((config$n_train + 1):config$n_pool, idx_ca),
      n_ref,
      replace = FALSE
    )

    bundles <- list(
      list(
        bundle = make_dataset_bundle("LogNormal", dp_ln, test_ln, idx_tr, idx_ca),
        X_ref = dp_ln$X[idx_ref_pool, ],
        Y_ref = dp_ln$Y[idx_ref_pool]
      ),
      list(
        bundle = make_dataset_bundle("Student-t", dp_st, test_st, idx_tr, idx_ca),
        X_ref = dp_st$X[idx_ref_pool, ],
        Y_ref = dp_st$Y[idx_ref_pool]
      )
    )

    for (item in bundles) {
      bundle <- item$bundle
      scp_timed <- time_it({
        q_scp <- get_standard_quantile(bundle$S_real, config$alpha)
        compute_interval_metrics(bundle$S_test, q_scp)
      })

      for (shock_prob in config$shock_probs) {
        rsa_timed <- time_it({
          S_ref_clean <- get_scores(bundle$model_bundle$main, item$X_ref, item$Y_ref)
          S_ref <- apply_reference_tail_shock(
            S_ref_clean,
            shock_prob = shock_prob,
            shock_scale = config$shock_scale
          )
          q_rsa <- get_rsacp_quantile(
            S_real = bundle$S_real,
            S_ref = S_ref,
            alpha = config$alpha,
            beta = config$beta,
            use_ot = TRUE,
            grid_size = config$rsa_grid_size
          )
          compute_interval_metrics(bundle$S_test, q_rsa)
        })

        results[[counter]] <- dplyr::bind_rows(
          data.frame(Method = "SCP", scp_timed$value, Time = scp_timed$seconds),
          data.frame(Method = "RSA-CP (OT) (Ours)", rsa_timed$value, Time = rsa_timed$seconds)
        ) |>
          dplyr::mutate(
            Dataset = bundle$dataset,
            Trial = trial_id,
            Shock_Probability = shock_prob,
            Method = factor(Method, levels = shock_method_levels)
          )
        counter <- counter + 1
      }
    }
  }

  raw <- dplyr::bind_rows(results)
  summary <- raw |>
    dplyr::group_by(Dataset, Shock_Probability, Method) |>
    dplyr::summarise(
      Mean_Cov = mean(Cov),
      SD_Cov = sd(Cov),
      Mean_Width = mean(Width),
      SD_Width = sd(Width),
      Total_Time = sum(Time, na.rm = TRUE),
      Mean_Time = mean(Time, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(Method = factor(Method, levels = shock_method_levels))

  write_result_pair(raw, summary, "shock_probability_raw.csv", "shock_probability_summary.csv")
  fig <- plot_shock_probability(summary, config)
  save_plot_pdf_png(fig, "shock_probability", width = 10, height = 6.1)
  log_message("Shock probability: outputs written.")

  invisible(list(raw = raw, summary = summary, figure = fig))
}

run_shock_probability_experiment(config)
