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
set.seed(config$seed + 3)

run_figure3_reference_score_size <- function(config = simulation_config()) {
  n_cal <- config$n_cal_fixed
  results <- list()

  dp_ln <- get_data_lognormal_true(config$n_pool)
  test_ln <- get_data_lognormal_true(config$n_test)
  dp_st <- get_data_student_true(config$n_pool)
  test_st <- get_data_student_true(config$n_test)

  log_message("Figure 3: reference/synthetic-size experiment started.")
  counter <- 1

  for (trial_id in seq_len(config$n_trials)) {
    if (trial_id %% 10 == 0) log_message("Figure 3 trial ", trial_id, " of ", config$n_trials)

    idx_tr <- seq_len(config$n_train)
    idx_ca <- sample((config$n_train + 1):config$n_pool, n_cal)
    idx_ref_pool <- sample(
      setdiff((config$n_train + 1):config$n_pool, idx_ca),
      max(config$n_grid),
      replace = FALSE
    )

    bundles <- list(
      make_dataset_bundle("LogNormal", dp_ln, test_ln, idx_tr, idx_ca, idx_ref_pool),
      make_dataset_bundle("Student-t", dp_st, test_st, idx_tr, idx_ca, idx_ref_pool)
    )

    for (bundle in bundles) {
      for (n_ref in config$n_grid) {
        results[[counter]] <- evaluate_methods_for_size(
          bundle = bundle,
          n_ref = n_ref,
          x_name = "N_syn",
          x_value = n_ref,
          trial_id = trial_id,
          config = config
        )
        counter <- counter + 1
      }
    }
  }

  raw <- dplyr::bind_rows(results)
  summary <- summarise_by_method(raw, c("Dataset", "N_syn"))
  write_result_pair(
    raw,
    summary,
    "figure3_reference_score_size_raw.csv",
    "figure3_reference_score_size_summary.csv"
  )

  fig <- plot_figure3_reference_size(summary, config)
  save_plot_pdf_png(fig, "figure3_reference_score_size", width = 13, height = 6.3)
  log_message("Figure 3: outputs written.")

  invisible(list(raw = raw, summary = summary, figure = fig))
}

run_figure3_reference_score_size(config)
