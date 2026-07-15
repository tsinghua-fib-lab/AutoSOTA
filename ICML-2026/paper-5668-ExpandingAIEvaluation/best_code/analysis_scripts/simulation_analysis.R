# Run: Rscript analysis_scripts/simulation_analysis.R
# Inputs: simulations/simulation_results/
# Outputs: 
#   figures/simulations/
#     Figure2_simulation_summary.pdf
#     FigureC1_distribution_a.pdf
#     FigureC1_distribution_b.pdf
#     FigureC1_distribution_c.pdf
#     FigureC1_distribution_d.pdf
#     FigureC1_distribution_e.pdf
#     FigureC1_legend.pdf
#   tables/
#     TableC5_simulation_settings_summary.csv
#     TableC6_SettingA_summary.csv

source("src/common_paths.R")
source("src/constants.R")
library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)
library(cowplot)

set.seed(1001)

SIMULATION_DIR <- repo_path("simulations", "simulation_results", "")

DISTRIBUTION_CONFIG <- data.frame(
  scenario = c("baseline", "n10", "n20", "t4", "t6"),
  filename = c(
    "simulation_estimates_baseline.csv",
    "simulation_estimates_10items.csv",
    "simulation_estimates_20items.csv",
    "simulation_estimates_4trials.csv",
    "simulation_estimates_6trials.csv"
  ),
  output = c(
    "FigureC1_distribution_a.pdf",
    "FigureC1_distribution_b.pdf",
    "FigureC1_distribution_c.pdf",
    "FigureC1_distribution_d.pdf",
    "FigureC1_distribution_e.pdf"
  ),
  binwidth = c(0.001, 0.0033, 0.002, 0.001, 0.001),
  xmax = c(0.25, 0.75, 0.5, 0.25, 0.25),
  x_interval = c(0.1, 0.3, 0.2, 0.1, 0.1),
  ymax = c(175, 175, 175, 175, 175),
  stringsAsFactors = FALSE
)

SWEEP_CONFIG <- list(
  trial_results = list(
    x_column = "t_trials",
    baseline_value = 8,
    baseline_index = 4,
    baseline_file = "simulation_estimates_baseline.csv",
    sweeps = data.frame(
      filename = c(
        "simulation_estimates_2trials.csv",
        "simulation_estimates_4trials.csv",
        "simulation_estimates_6trials.csv",
        "simulation_estimates_12trials.csv",
        "simulation_estimates_16trials.csv"
      ),
      x_value = c(2, 4, 6, 12, 16)
    )
  ),
  item_results = list(
    x_column = "n_items",
    baseline_value = 40,
    baseline_index = 3,
    baseline_file = "simulation_estimates_baseline.csv",
    sweeps = data.frame(
      filename = c(
        "simulation_estimates_10items.csv",
        "simulation_estimates_20items.csv",
        "simulation_estimates_80items.csv",
        "simulation_estimates_120items.csv",
        "simulation_estimates_160items.csv"
      ),
      x_value = c(10, 20, 80, 120, 160)
    )
  ),
  sigma_results = list(
    x_column = "item_sigma",
    baseline_value = 1.3,
    baseline_index = 4,
    baseline_file = "simulation_estimates_baseline.csv",
    sweeps = data.frame(
      filename = c(
        "simulation_estimates_point13itemsigma.csv",
        "simulation_estimates_point65itemsigma.csv",
        "simulation_estimates_2point6itemsigma.csv",
        "simulation_estimates_3point9itemsigma.csv"
      ),
      x_value = c(0.13, 0.65, 2.6, 3.9)
    )
  )
)

SUMMARY_PANEL_CONFIG <- data.frame(
  result_name = c(
    "item_results", "trial_results", "sigma_results",
    "item_results", "trial_results", "sigma_results"
  ),
  metric = c("marginal_rmse", "marginal_rmse", "marginal_rmse", "marginal_coverage", "marginal_coverage", "marginal_coverage"),
  panel = c(
    "Varying Items", "Varying Trials", "Varying Item Sigma",
    "Varying Items Row 2", "Varying Trials Row 2", "Varying Item Sigma Row 2"
  ),
  stringsAsFactors = FALSE
)

SUMMARY_PANEL_LAYOUT <- data.frame(
  panel = c(
    "Varying Items", "Varying Trials", "Varying Item Sigma",
    "Varying Items Row 2", "Varying Trials Row 2", "Varying Item Sigma Row 2"
  ),
  title = c("Varying # Items", "Varying Trials", "Varying Difficulty", NA, NA, NA),
  x_label_key = c(NA, NA, NA, "items", "trials", "sigma"),
  y_label = c("RMSE (probability)", NA, NA, "Coverage", NA, NA),
  limits_min = c(0, 0, 0, 0.9, 0.9, 0.9),
  limits_max = c(0.1, 0.1, 0.1, 1, 1, 1),
  show_x = c(FALSE, FALSE, FALSE, TRUE, TRUE, TRUE),
  show_y = c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE),
  add_ref = c(FALSE, FALSE, FALSE, TRUE, TRUE, TRUE),
  stringsAsFactors = FALSE
)

TABLE_C6_CONFIG <- data.frame(
  Condition = c(
    "Baseline",
    "6 LLMs",
    "9 LLMs",
    "10 items",
    "20 items",
    "80 items",
    "120 items",
    "160 items",
    "2 trials",
    "4 trials",
    "6 trials",
    "12 trials",
    "16 trials",
    "Sigma 0.13",
    "Sigma 0.65",
    "Sigma 2.6",
    "Sigma 3.9"
  ),
  filename = c(
    "simulation_estimates_baseline.csv",
    "simulation_estimates_6llms.csv",
    "simulation_estimates_9llms.csv",
    "simulation_estimates_10items.csv",
    "simulation_estimates_20items.csv",
    "simulation_estimates_80items.csv",
    "simulation_estimates_120items.csv",
    "simulation_estimates_160items.csv",
    "simulation_estimates_2trials.csv",
    "simulation_estimates_4trials.csv",
    "simulation_estimates_6trials.csv",
    "simulation_estimates_12trials.csv",
    "simulation_estimates_16trials.csv",
    "simulation_estimates_point13itemsigma.csv",
    "simulation_estimates_point65itemsigma.csv",
    "simulation_estimates_2point6itemsigma.csv",
    "simulation_estimates_3point9itemsigma.csv"
  ),
  stringsAsFactors = FALSE
)

TABLE_C5_CONFIG <- data.frame(
  Setting = c(
    "Setting A (well-specified)",
    "Setting B (linear)",
    "Setting C (no structure)"
  ),
  filename = c(
    "simulation_estimates_baseline.csv",
    "simulation_estimates_plinear_baseline.csv",
    "simulation_estimates_pindependent_baseline.csv"
  ),
  stringsAsFactors = FALSE
)

coerce_coverage <- function(values) {
  if (is.logical(values)) {
    return(as.numeric(values))
  }
  if (is.numeric(values)) {
    return(values)
  }
  normalized <- trimws(tolower(as.character(values)))
  as.numeric(normalized %in% c("true", "1"))
}

normalize_method_columns <- function(data) {
  rename_map <- c(
    marginal_estimate_clt = "marginal_estimate_rf",
    marginal_lower_clt = "marginal_lower_rf",
    marginal_upper_clt = "marginal_upper_rf",
    marginal_coverage_clt = "marginal_coverage_rf"
  )

  for (source_name in names(rename_map)) {
    target_name <- rename_map[[source_name]]
    if (!(target_name %in% names(data)) && source_name %in% names(data)) {
      names(data)[names(data) == source_name] <- target_name
    }
  }

  data
}

validate_input_file <- function(filename) {
  file_path <- file.path(SIMULATION_DIR, filename)
  if (!file.exists(file_path)) {
    stop(sprintf("Missing simulation input: %s", filename))
  }
  file_path
}

extract_results_with_id <- function(filename, column_name, value) {
  result <- extract_results(load_simulation_data(filename))
  result[[column_name]] <- value
  result
}

prep_data <- function(data) {
  data <- normalize_method_columns(data)
  required_cols <- c(
    "marginal_estimate_glmm",
    "marginal_upper_rf",
    "marginal_lower_rf",
    "marginal_upper_glmm",
    "marginal_lower_glmm",
    "marginal_coverage_rf",
    "marginal_coverage_glmm",
    "marginal_estimate_rf",
    "marginal_true_value"
  )
  stopifnot(all(required_cols %in% names(data)))
  data <- data[!is.na(data$marginal_estimate_glmm), ]
  data$interval_width_rf <- data$marginal_upper_rf - data$marginal_lower_rf
  data$interval_width_glmm <- data$marginal_upper_glmm - data$marginal_lower_glmm
  data$marginal_coverage_rf <- coerce_coverage(data$marginal_coverage_rf)
  data$marginal_coverage_glmm <- coerce_coverage(data$marginal_coverage_glmm)
  data
}

load_simulation_data <- function(filename) {
  read.csv(validate_input_file(filename)) %>%
    prep_data()
}

extract_results <- function(data) {
  data %>%
    summarize(
      marginal_coverage_rf = mean(marginal_coverage_rf),
      marginal_coverage_glmm = mean(marginal_coverage_glmm),
      marginal_rmse_rf = sqrt(mean((marginal_estimate_rf - marginal_true_value)^2)),
      marginal_rmse_glmm = sqrt(mean((marginal_estimate_glmm - marginal_true_value)^2))
    )
}

round_summary_row <- function(summary_row, digits = 3) {
  numeric_cols <- vapply(summary_row, is.numeric, logical(1))
  summary_row[numeric_cols] <- lapply(summary_row[numeric_cols], signif, digits = digits)
  summary_row
}

summarize_simulation_condition <- function(data, label_column, label_value) {
  stopifnot(
    is.data.frame(data),
    is.character(label_column),
    length(label_column) == 1,
    nzchar(label_column),
    is.character(label_value),
    length(label_value) == 1,
    nzchar(label_value)
  )

  data.frame(
    label_value,
    `RF Bias` = mean(data$marginal_estimate_rf - data$marginal_true_value),
    `GLMM Bias` = mean(data$marginal_estimate_glmm - data$marginal_true_value),
    `RF RMSE` = sqrt(mean((data$marginal_estimate_rf - data$marginal_true_value)^2)),
    `GLMM RMSE` = sqrt(mean((data$marginal_estimate_glmm - data$marginal_true_value)^2)),
    `RF Coverage` = mean(data$marginal_coverage_rf),
    `GLMM Coverage` = mean(data$marginal_coverage_glmm),
    check.names = FALSE,
    stringsAsFactors = FALSE
  ) %>%
    stats::setNames(c(label_column, "RF Bias", "GLMM Bias", "RF RMSE", "GLMM RMSE", "RF Coverage", "GLMM Coverage")) %>%
    round_summary_row()
}

summarize_table_c5_condition <- function(filename, setting_label) {
  data <- load_simulation_data(filename)
  summarize_simulation_condition(data, "Setting", setting_label)
}

summarize_table_c6_condition <- function(filename, condition_label) {
  data <- load_simulation_data(filename)

  summary_row <- summarize_simulation_condition(data, "Condition", condition_label) %>%
    mutate(
      `RF CI Width` = mean(data$interval_width_rf),
      `GLMM CI Width` = mean(data$interval_width_glmm),
      `RF CI Width SD` = stats::sd(data$interval_width_rf),
      `GLMM CI Width SD` = stats::sd(data$interval_width_glmm)
    )

  round_summary_row(summary_row)
}

write_table_c5 <- function(config = TABLE_C5_CONFIG) {
  stopifnot(all(c("Setting", "filename") %in% names(config)))

  table_c5 <- bind_rows(lapply(seq_len(nrow(config)), function(row_index) {
    summarize_table_c5_condition(config$filename[[row_index]], config$Setting[[row_index]])
  }))

  write.csv(
    table_c5,
    file = repo_path("tables", "TableC5_simulation_settings_summary.csv"),
    row.names = FALSE
  )

  invisible(table_c5)
}

write_table_c6 <- function(config = TABLE_C6_CONFIG) {
  stopifnot(all(c("Condition", "filename") %in% names(config)))

  table_c6 <- bind_rows(lapply(seq_len(nrow(config)), function(row_index) {
    summarize_table_c6_condition(config$filename[[row_index]], config$Condition[[row_index]])
  }))

  write.csv(
    table_c6,
    file = repo_path("tables", "TableC6_SettingA_summary.csv"),
    row.names = FALSE
  )

  invisible(table_c6)
}

build_distribution_data <- function(data) {
  pivot_longer(
    data %>% select(interval_width_rf, interval_width_glmm),
    cols = everything(),
    names_to = "method",
    values_to = "ci_width"
  )
}

build_distribution_legend <- function(data) {
  legend_source <- ggplot(
    build_distribution_data(data),
    aes(x = ci_width, fill = method)
  ) +
    geom_histogram(position = "identity", alpha = 0.6, binwidth = 0.001) +
    scale_fill_manual(
      values = c("interval_width_rf" = REGRESSION_FREE_COLOR, "interval_width_glmm" = GLMM_COLOR),
      labels = c("Regression-Free", "GLMM")
    ) +
    theme_minimal() +
    theme(legend.position = "bottom", legend.title = element_blank())
  cowplot::get_legend(legend_source)
}

plot_distribution <- function(data, binwidth = 0.001, xmax = 0.25, x_interval = 0.1, ymax = 175) {
  distribution_data <- build_distribution_data(data)
  distribution_means <- aggregate(ci_width ~ method, data = distribution_data, FUN = mean, na.rm = TRUE)
  colors <- c("interval_width_rf" = REGRESSION_FREE_COLOR, "interval_width_glmm" = GLMM_COLOR)

  ggplot(distribution_data, aes(x = ci_width, fill = method)) +
    geom_histogram(position = "identity", alpha = 0.6, binwidth = binwidth) +
    scale_x_continuous(breaks = seq(0, xmax, by = x_interval)) +
    geom_vline(data = distribution_means, aes(xintercept = ci_width, color = method), linewidth = 3) +
    scale_fill_manual(values = colors, labels = c("Regression-Free", "GLMM")) +
    scale_color_manual(values = colors, guide = "none") +
    labs(x = "Generalized Accuracy CI Width", y = "Density") +
    coord_cartesian(xlim = c(0, xmax), ylim = c(0, ymax)) +
    theme_minimal(base_size = 50) +
    theme(
      axis.title = element_blank(),
      legend.title = element_blank(),
      legend.position = "none",
      panel.grid.minor = element_blank()
    )
}

load_distribution_scenarios <- function(config) {
  scenarios <- lapply(config$filename, load_simulation_data)
  names(scenarios) <- config$scenario
  scenarios
}

write_distribution_figures <- function(config, scenarios) {  # Figure C.1
  legend <- build_distribution_legend(scenarios$baseline)
  ggsave(repo_path("figures", "simulations", "FigureC1_legend.pdf"), legend, height = 1, width = 2)

  for (row_index in seq_len(nrow(config))) {
    row <- config[row_index, ]
    plot_obj <- plot_distribution(
      scenarios[[row$scenario]],
      binwidth = row$binwidth,
      xmax = row$xmax,
      x_interval = row$x_interval,
      ymax = row$ymax
    )
    ggsave(repo_path("figures", "simulations", row$output), plot_obj, height = 5)
  }
}

build_summary_sweep <- function(config) {
  baseline <- extract_results(load_simulation_data(config$baseline_file))
  baseline[[config$x_column]] <- config$baseline_value

  sweep_rows <- lapply(seq_len(nrow(config$sweeps)), function(row_index) {
    sweep_row <- config$sweeps[row_index, ]
    extract_results_with_id(sweep_row$filename, config$x_column, sweep_row$x_value)
  })

  bind_rows(
    sweep_rows[seq_len(config$baseline_index - 1)],
    list(baseline),
    sweep_rows[seq(config$baseline_index, length(sweep_rows))]
  )
}

build_summary_data <- function(config) {
  lapply(config, build_summary_sweep)
}

build_metric_frame <- function(summary_data, result_name, metric_prefix, method, panel_name) {
  result_frame <- summary_data[[result_name]]
  x_column <- setdiff(names(result_frame), c(
    "marginal_coverage_rf",
    "marginal_coverage_glmm",
    "marginal_rmse_rf",
    "marginal_rmse_glmm"
  ))
  stopifnot(length(x_column) == 1)

  data.frame(
    x_val = result_frame[[x_column]],
    y_val = result_frame[[paste0(metric_prefix, "_", method)]],
    series = ifelse(method == "glmm", "GLMM", "Regression-Free"),
    plot_group = panel_name
  )
}

build_toy_plot_data <- function(summary_data) {
  panel_frames <- lapply(seq_len(nrow(SUMMARY_PANEL_CONFIG)), function(row_index) {
    panel_row <- SUMMARY_PANEL_CONFIG[row_index, ]
    metric_prefix <- panel_row$metric

    bind_rows(
      build_metric_frame(summary_data, panel_row$result_name, metric_prefix, "glmm", panel_row$panel),
      build_metric_frame(summary_data, panel_row$result_name, metric_prefix, "rf", panel_row$panel)
    )
  })

  bind_rows(panel_frames)
}

make_toy_panel <- function(plot_data, layout_row, common_theme, x_label = NULL) {
  plot_obj <- ggplot(
    subset(plot_data, plot_group == layout_row$panel),
    aes(x = x_val, y = y_val, color = series, shape = series)
  ) +
    geom_point(size = 1, alpha = 0.8, stroke = 2) +
    scale_color_manual(values = c("Regression-Free" = REGRESSION_FREE_COLOR, "GLMM" = GLMM_COLOR)) +
    scale_shape_manual(values = c("Regression-Free" = 1, "GLMM" = 3)) +
    coord_cartesian(ylim = c(layout_row$limits_min, layout_row$limits_max)) +
    common_theme +
    labs(
      title = normalize_plot_label(layout_row$title),
      x = normalize_plot_label(x_label),
      y = normalize_plot_label(layout_row$y_label)
    )

  if (layout_row$add_ref) {
    plot_obj <- plot_obj + geom_hline(yintercept = 0.95, color = "black", linetype = "dashed", linewidth = 0.5)
  }
  if (!layout_row$show_x) {
    plot_obj <- plot_obj + theme(axis.title.x = element_blank(), axis.text.x = element_blank())
  }
  if (!layout_row$show_y) {
    plot_obj <- plot_obj + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  }

  plot_obj
}

normalize_plot_label <- function(label) {
  if (length(label) == 0 || (length(label) == 1 && is.atomic(label) && is.na(label))) {
    return(NULL)
  }
  label
}

resolve_x_label <- function(label_key) {
  if (is.na(label_key)) {
    return(NULL)
  }
  switch(
    label_key,
    "items" = bquote("Number of items " * italic(n)),
    "trials" = bquote("Number of trials " * italic(t)),
    "sigma" = bquote("Item sigma " * sigma[j]^"*"),
    label_key
  )
}

write_toy_summary_figure <- function(plot_data) {  # Figure 2
  common_theme <- list(
    theme_minimal(),
    theme(
      axis.title = element_text(size = 12),
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.background = element_rect(fill = "grey95", color = NA)
    )
  )

  panels <- lapply(seq_len(nrow(SUMMARY_PANEL_LAYOUT)), function(row_index) {
    layout_row <- SUMMARY_PANEL_LAYOUT[row_index, ]
    make_toy_panel(
      plot_data,
      layout_row,
      common_theme,
      x_label = resolve_x_label(layout_row$x_label_key)
    )
  })

  combined_plot <- ((panels[[1]] + panels[[2]] + panels[[3]]) / (panels[[4]] + panels[[5]] + panels[[6]])) +
    plot_layout(guides = "collect") &
    theme(legend.position = "right", legend.text = element_text(size = 12))

  ggsave(repo_path("figures", "simulations", "Figure2_simulation_summary.pdf"), combined_plot, height = 4, width = 8)
}

main <- function() {
  distribution_scenarios <- load_distribution_scenarios(DISTRIBUTION_CONFIG)
  write_distribution_figures(DISTRIBUTION_CONFIG, distribution_scenarios)  # Figure C.1

  summary_data <- build_summary_data(SWEEP_CONFIG)
  toy_plot_data <- build_toy_plot_data(summary_data)
  write_toy_summary_figure(toy_plot_data) # Figure 2
  write_table_c5()
  write_table_c6()
}

main()
