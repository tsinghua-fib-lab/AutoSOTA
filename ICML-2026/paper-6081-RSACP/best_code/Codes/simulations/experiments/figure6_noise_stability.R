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
set.seed(config$seed + 6)

run_figure6_noise_stability <- function(config = simulation_config()) {
  n_cal <- config$n_cal_fixed
  n_ref <- config$n_ref_fixed
  results <- list()

  dp_ln <- get_data_lognormal_true(config$n_pool)
  test_ln <- get_data_lognormal_true(config$n_test)
  dp_st <- get_data_student_true(config$n_pool)
  test_st <- get_data_student_true(config$n_test)

  log_message("Figure 6: generator-noise stability experiment started.")
  counter <- 1

  for (trial_id in seq_len(config$n_trials)) {
    if (trial_id %% 10 == 0) log_message("Figure 6 trial ", trial_id, " of ", config$n_trials)

    idx_tr <- seq_len(config$n_train)
    idx_ca <- sample((config$n_train + 1):config$n_pool, n_cal)

    bundles <- list(
      make_dataset_bundle("LogNormal", dp_ln, test_ln, idx_tr, idx_ca),
      make_dataset_bundle("Student-t", dp_st, test_st, idx_tr, idx_ca)
    )

    for (bundle in bundles) {
      scp_timed <- time_it({
        q_scp <- get_standard_quantile(bundle$S_real, config$alpha)
        compute_interval_metrics(bundle$S_test, q_scp)
      })

      for (noise_scale in config$noise_grid) {
        syn_score_timed <- time_it({
          syn <- get_synthetic_data_with_noise(
            bundle$dataset,
            n_ref,
            bundle$model_bundle,
            noise_scale
          )
          get_scores(bundle$model_bundle$main, syn$X, syn$Y)
        })
        S_syn <- syn_score_timed$value

        rsa_timed <- time_it({
          S_ref <- as.numeric(S_syn)
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

        spi_timed <- time_it({
          q_spi <- get_spi_quantile(bundle$S_real, S_syn, config$alpha, config$beta)
          compute_interval_metrics(bundle$S_test, q_spi)
        })

        synthetic_only_timed <- time_it({
          q_syn <- get_standard_quantile(S_syn, config$alpha)
          compute_interval_metrics(bundle$S_test, q_syn)
        })

        results[[counter]] <- dplyr::bind_rows(
          data.frame(Method = "SCP", scp_timed$value, Time = scp_timed$seconds),
          data.frame(Method = "RSA-CP (OT) (Ours)", rsa_timed$value, Time = rsa_timed$seconds),
          data.frame(Method = "SPI", spi_timed$value, Time = syn_score_timed$seconds + spi_timed$seconds),
          data.frame(Method = "Synthetic-only", synthetic_only_timed$value, Time = syn_score_timed$seconds + synthetic_only_timed$seconds)
        ) |>
          dplyr::mutate(
            Dataset = bundle$dataset,
            Trial = trial_id,
            Noise = noise_scale,
            Method = factor(Method, levels = method_levels)
          )
        counter <- counter + 1
      }
    }
  }

  raw <- dplyr::bind_rows(results)
  summary <- summarise_by_method(raw, c("Dataset", "Noise"))
  write_result_pair(
    raw,
    summary,
    "figure6_noise_stability_raw.csv",
    "figure6_noise_stability_summary.csv"
  )

  fig <- plot_figure6_noise_stability(summary, config)
  save_plot_pdf_png(fig, "figure6_noise_stability", width = 10.5, height = 6.3)
  log_message("Figure 6: outputs written.")

  invisible(list(raw = raw, summary = summary, figure = fig))
}

run_figure6_noise_stability(config)
