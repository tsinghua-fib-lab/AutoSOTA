# Run: Rscript analysis_scripts/gpqa_analysis.R
# Inputs: benchmark_results/gpqa.csv, benchmark_results/gpqa_info.csv
# Outputs: 
#   figures/gpqa/
#     Figure1a_gpqa_bar_chart_summary_benchmark.pdf
#     Figure1b_gpqa_bar_chart_summary_generalized.pdf
#     Figure1_legend.pdf
#     Figure4a_gpqa_items_by_domain.pdf
#     Figure4b_gpqa_items_by_difficulty.pdf
#     FigureC2_gpqa_bar_chart_full.pdf
#     FigureC4a_gpqa_diagnostic.pdf
#     FigureC4b_gpqa_dispersion.pdf
#     FigureC4c_gpqa_LLM_residuals.pdf
#     FigureC4d_gpqa_item_residuals.pdf
#   tables/TableC9_gpqa_model_comparison.csv

# ----------------------------
# Imports
# ----------------------------

source("src/analysis_functions.R")
library(ggplot2)
library(dplyr)
library(ggtext)
library(patchwork)
library(cowplot)

# ----------------------------
# Plotting functions
# ----------------------------

METHOD_DISPLAY_LABELS <- c(
  "avg_single_epoch" = "Simple average (1 trial)",
  "Regression-Free" = "Regression-free (8 trials)",
  "GLMM" = "GLMM (8 trials)"
)

METHOD_DISPLAY_LABELS_WRAPPED <- c(
  "avg_single_epoch" = "Simple average\n(1 trial only)",
  "Regression-Free" = "Regression-\nFree",
  "GLMM" = "GLMM"
)

method_values <- function(method_ids, values) {
  unname(values[method_ids])
}

align_estimate_series <- function(df, order_levels) {
  df %>%
    mutate(
      LLM_id = factor(LLM_id, levels = order_levels),
      ci_width = upper - lower
    )
}

filter_featured_LLMs <- function(df) {
  df |>
    dplyr::filter(LLM_id %in% names(FEATURED_LLM_LABELS)) |>
    dplyr::mutate(LLM_id = factor(as.character(LLM_id), levels = names(FEATURED_LLM_LABELS)))
}

prepare_plot_data <- function(glmm_estimates, data) {
  order_levels <- levels(glmm_estimates$LLM_id)

  list(
    generalized_glmm = align_estimate_series(glmm_estimates, order_levels),
    generalized_rf = align_estimate_series(compute_marginal_accuracy(data), order_levels),
    benchmark_rf = align_estimate_series(compute_conditional_accuracy(data), order_levels),
    simple = align_estimate_series(compute_simple_accuracy(filter(data, trial_id == 1)), order_levels)
  )
}

plot_bars <- function(
  df,
  xlabel_angle = 60,
  bar_width = 1,
  alpha = 1,
  labels = NULL,
  x_labels = LLM_LABELS,
  y_label = NULL
) {
  if (is.null(labels)) {
    labels <- METHOD_DISPLAY_LABELS[c("Regression-Free", "GLMM")]
  }

  ggplot(df, aes(x = LLM_id, y = prob, fill = source)) +
    scale_fill_manual(
      values = scales::alpha(method_values(names(labels), PLOT_COLORS), alpha),
      breaks = names(labels),
      labels = labels
    ) +
    scale_x_discrete(labels = x_labels) +
    geom_col(position = position_dodge(width = 0.9), width = bar_width) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      position = position_dodge(width = 0.9),
      width = 0.25,
      color = "black"
    ) +
    geom_point(
      aes(shape = source),
      position = position_dodge(width = 0.9),
      color = "black",
      size = 2,
      stroke = 0.4,
      show.legend = FALSE
    ) +
    scale_shape_manual(
      values = method_values(names(labels), PLOT_SHAPES),
      breaks = names(labels),
      labels = labels
    ) +
    labs(y = y_label) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      axis.title.y = element_markdown(),
      axis.title.x = element_blank(),
      axis.text.x = element_text(angle = xlabel_angle, hjust = 1)
    )
}

plot_widths <- function(df, x_labels = LLM_LABELS) {
  methods <- levels(df$source)

  ggplot(df, aes(x = LLM_id, y = ci_width, color = source, shape = source)) +
    geom_point(size = 2.4, stroke = 0.4) +
    scale_color_manual(
      values = method_values(methods, PLOT_COLORS),
      breaks = methods,
      labels = METHOD_DISPLAY_LABELS[methods]
    ) +
    scale_shape_manual(
      values = method_values(methods, PLOT_SHAPES),
      breaks = methods,
      labels = METHOD_DISPLAY_LABELS[methods]
    ) +
    scale_x_discrete(labels = x_labels) +
    scale_y_continuous(limits = c(0, 0.15), breaks = c(0, 0.05, 0.10)) +
    labs(y = "CI Width", x = NULL) +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid.minor = element_blank()
    )
}

build_panel_data <- function(plot_data, series_names, source_levels, featured_only = FALSE) {
  panel_data <- bind_rows(plot_data[series_names])
  if (featured_only) {
    panel_data <- filter_featured_LLMs(panel_data)
  }
  panel_data %>%
    mutate(source = forcats::fct_relevel(source, source_levels))
}

save_bar_panel <- function(
  df,
  output_name,
  heights,
  width,
  height,
  x_labels = LLM_LABELS,
  xlabel_angle = 60,
  bar_width = 1,
  labels = NULL,
  y_label = NULL,
  hide_legend = FALSE
) {
  panel_plot <- plot_widths(df, x_labels = x_labels) /
    plot_bars(
      df,
      xlabel_angle = xlabel_angle,
      bar_width = bar_width,
      labels = labels,
      x_labels = x_labels,
      y_label = y_label
    ) +
    plot_layout(heights = heights)

  if (hide_legend) {
    panel_plot <- panel_plot & theme(legend.position = "none")
  }

  ggsave(repo_path("figures", "gpqa", output_name), panel_plot, height = height, width = width)
}

write_gpqa_bar_figures <- function(plot_data) {
  save_bar_panel(  # Figure C.2
    build_panel_data(plot_data, c("generalized_rf", "generalized_glmm"), c("Regression-Free", "GLMM")),
    "FigureC2_gpqa_bar_chart_full.pdf",
    heights = c(1, 2.5),
    width = 6,
    height = 4.5,
    y_label = "Generalized Accuracy",
  )

  save_bar_panel(  # Figure 1a
    build_panel_data(plot_data, c("simple", "benchmark_rf"), c("avg_single_epoch", "Regression-Free"), featured_only = TRUE),
    "Figure1a_gpqa_bar_chart_summary_benchmark.pdf",
    heights = c(1, 1.5),
    width = 3.75,
    height = 3.5,
    x_labels = FEATURED_LLM_LABELS,
    xlabel_angle = 20,
    bar_width = 0.7,
    labels = METHOD_DISPLAY_LABELS_WRAPPED[c("avg_single_epoch", "Regression-Free")],
    y_label = "**Benchmark** Accuracy",
    hide_legend = TRUE
  )

  save_bar_panel(  # Figure 1b
    build_panel_data(
      plot_data,
      c("simple", "generalized_rf", "generalized_glmm"),
      c("avg_single_epoch", "Regression-Free", "GLMM"),
      featured_only = TRUE
    ),
    "Figure1b_gpqa_bar_chart_summary_generalized.pdf",
    heights = c(1, 1.5),
    width = 3.75,
    height = 3.5,
    x_labels = FEATURED_LLM_LABELS,
    xlabel_angle = 20,
    bar_width = 0.7,
    labels = METHOD_DISPLAY_LABELS_WRAPPED[c("avg_single_epoch", "Regression-Free", "GLMM")],
    y_label = "**Generalized** Accuracy",
    hide_legend = TRUE
  )
}

write_shared_legend <- function() {
  dummy_data <- data.frame(
    source = factor(names(METHOD_DISPLAY_LABELS), levels = names(METHOD_DISPLAY_LABELS)),
    x = seq_along(METHOD_DISPLAY_LABELS),
    y = 1
  )

  dummy_plot <- ggplot(dummy_data, aes(x = x, y = y, color = source, shape = source)) +
    geom_point(size = 3, stroke = 0.4) +
    scale_color_manual(
      values = method_values(names(METHOD_DISPLAY_LABELS), PLOT_COLORS),
      breaks = names(METHOD_DISPLAY_LABELS),
      labels = METHOD_DISPLAY_LABELS
    ) +
    scale_shape_manual(
      values = method_values(names(METHOD_DISPLAY_LABELS), PLOT_SHAPES),
      breaks = names(METHOD_DISPLAY_LABELS),
      labels = METHOD_DISPLAY_LABELS
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_blank(),
      legend.text = element_text(size = 10)
    )

  shared_legend <- get_legend(dummy_plot)
  ggsave(repo_path("figures", "gpqa", "Figure1_legend.pdf"), shared_legend, height = 0.5, width = 8)
}

prepare_difficulty_effects <- function(random_effects) {
  random_effects %>%
    mutate(
      diff = factor(
        writers_difficulty_estimate,
        levels = c("Undergrad - Easy", "Undergrad - Hard", "Grad - Hard", "Expert", "Unrated")
      )
    ) %>%
    filter(diff != "Unrated", diff != "Undergrad - Easy")
}

build_random_effect_plot <- function(data, x_var, y_label, x_label) {
  ggplot(data, aes(x = .data[[x_var]], y = intercept)) +
    geom_violin(aes(color = .data[[x_var]]), width = 1, fill = NA) +
    geom_point(
      aes(color = .data[[x_var]]),
      position = position_jitter(width = 0.05),
      size = 2,
      alpha = 0.2
    ) +
    geom_boxplot(aes(color = .data[[x_var]]), width = 0.25, fill = NA) +
    labs(x = x_label, y = y_label) +
    coord_flip() +
    theme_minimal() +
    theme(legend.position = "none", axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))
}

write_random_effect_figures <- function(glmm, data, dataset) {
  random_effects <- prepare_random_effects(glmm, data, dataset)

  domain_plot <- build_random_effect_plot(  # Figure 4a
    random_effects,
    "High.level.domain",
    "Difficulty Estimates (Random Effects)",
    "Domain"
  )
  ggsave(repo_path("figures", "gpqa", "Figure4a_gpqa_items_by_domain.pdf"), domain_plot, width = 4, height = 2, units = "in", dpi = 300)

  difficulty_plot <- build_random_effect_plot(  # Figure 4b
    prepare_difficulty_effects(random_effects),
    "diff",
    "Difficulty Estimates (Random Effects)",
    "Human Difficulty"
  )
  ggsave(
    repo_path("figures", "gpqa", "Figure4b_gpqa_items_by_difficulty.pdf"),
    difficulty_plot,
    width = 4,
    height = 2,
    units = "in",
    dpi = 300
  )
}

# ----------------------------
# Analysis flow
# ----------------------------

main <- function() {
  
  data <- load_benchmark_data("gpqa.csv")

  # fit GLMM and get estimates
  glmm <- fit_glmm(data)
  table_c9 <- build_model_comparison_table(list(
    "Full GLMM" = glmm,
    "LLM fixed effects only" = fit_llm_fixed_effects_glm(data),
    "Item random effects only" = fit_item_random_effects_only_glmm(data)
  ))
  write.csv(
    table_c9,
    file = repo_path("tables", "TableC9_gpqa_model_comparison.csv"),
    row.names = FALSE
  )
  print_model_performance_table(glmm, "GPQA", observed = data$score)
  write_glmm_diagnostics(
    glmm,
    data,
    "gpqa",
    include_dispersion = TRUE,
    include_residuals = TRUE,
    diagnostic_figure_label = "FigureC4a",
    dispersion_figure_label = "FigureC4b",
    llm_residuals_figure_label = "FigureC4c",
    item_residuals_figure_label = "FigureC4d"
  )  # Figure C.4
  estimates <- compute_glmm_estimates(glmm, data)
  
  # make generalized accuracy and random effects outputs
  plot_data <- prepare_plot_data(estimates, data)
  write_gpqa_bar_figures(plot_data)  # Figure 1, Figure C.2
  write_shared_legend()  # Figure 1
  write_random_effect_figures(  # Figure 4
    glmm, data, 
    read.csv(repo_path("benchmark_results", "gpqa_info.csv"))
    )
    
}

main()
