method_colors <- c(
  "SCP" = "#000000",
  "RSA-CP (OT) (Ours)" = "#0072B2",
  "SPI" = "#D55E00",
  "Synthetic-only" = "#009E73"
)

method_linetypes <- c(
  "SCP" = "dashed",
  "RSA-CP (OT) (Ours)" = "solid",
  "SPI" = "solid",
  "Synthetic-only" = "dotdash"
)

method_linewidths <- c(
  "SCP" = 0.85,
  "RSA-CP (OT) (Ours)" = 1.25,
  "SPI" = 0.95,
  "Synthetic-only" = 0.9
)

paper_theme <- function(base_size = 11) {
  ggplot2::theme_bw(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey65"),
      strip.text = ggplot2::element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = ggplot2::element_blank(),
      legend.box = "horizontal",
      plot.title = ggplot2::element_text(face = "bold", hjust = 0),
      axis.title = ggplot2::element_text(face = "bold")
    )
}

method_scale <- function(levels = method_levels) {
  list(
    ggplot2::scale_color_manual(values = method_colors[levels], breaks = levels, drop = FALSE),
    ggplot2::scale_linetype_manual(values = method_linetypes[levels], breaks = levels, drop = FALSE),
    ggplot2::scale_linewidth_manual(values = method_linewidths[levels], breaks = levels, drop = FALSE)
  )
}

make_metric_panel <- function(dataset, metric, metrics) {
  factor(
    paste(dataset, metric, sep = "\n"),
    levels = c(
      paste("LogNormal", metrics, sep = "\n"),
      paste("Student-t", metrics, sep = "\n")
    )
  )
}

save_plot_pdf_png <- function(plot, base_name, width, height, dpi = 300) {
  ensure_output_dirs()
  ggplot2::ggsave(
    file.path("outputs", "figures", paste0(base_name, ".png")),
    plot,
    width = width,
    height = height,
    dpi = dpi
  )
  ggplot2::ggsave(
    file.path("outputs", "figures", paste0(base_name, ".pdf")),
    plot,
    width = width,
    height = height
  )
}

plot_figure3_reference_size <- function(summary, config) {
  metrics <- c("Coverage", "Average Width", "Computation Time")

  fig_long <- summary |>
    dplyr::transmute(
      Dataset,
      N_syn,
      Method,
      Coverage = Mean_Cov,
      `Average Width` = Mean_Width,
      `Computation Time` = Total_Time
    ) |>
    tidyr::pivot_longer(
      cols = c(Coverage, `Average Width`, `Computation Time`),
      names_to = "Metric",
      values_to = "Value"
    ) |>
    dplyr::mutate(
      Metric = factor(Metric, levels = metrics),
      Panel = make_metric_panel(Dataset, Metric, metrics)
    )

  target <- fig_long |>
    dplyr::filter(Metric == "Coverage") |>
    dplyr::distinct(Panel) |>
    dplyr::mutate(Value = 1 - config$alpha, Method = factor("SCP", levels = method_levels))

  ggplot2::ggplot(
    fig_long,
    ggplot2::aes(
      x = N_syn,
      y = Value,
      color = Method,
      linetype = Method,
      linewidth = Method,
      group = Method
    )
  ) +
    ggplot2::geom_hline(
      data = target,
      ggplot2::aes(yintercept = Value),
      linetype = "dashed",
      color = "grey45"
    ) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 1.7) +
    ggplot2::facet_wrap(~Panel, nrow = 2, scales = "free_y") +
    ggplot2::scale_x_log10(breaks = config$n_grid, labels = config$n_grid) +
    method_scale(method_levels) +
    ggplot2::guides(
      color = ggplot2::guide_legend(nrow = 1),
      linetype = ggplot2::guide_legend(nrow = 1),
      linewidth = ggplot2::guide_legend(nrow = 1)
    ) +
    ggplot2::labs(x = "Reference / synthetic score size N", y = NULL) +
    paper_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}

plot_figure6_noise_stability <- function(summary, config) {
  levels <- method_levels
  metrics <- c("Coverage", "Average Width")

  fig_long <- summary |>
    dplyr::mutate(Method = factor(as.character(Method), levels = levels)) |>
    dplyr::transmute(
      Dataset,
      Noise,
      Method,
      Coverage = Mean_Cov,
      `Average Width` = Mean_Width
    ) |>
    tidyr::pivot_longer(
      cols = c(Coverage, `Average Width`),
      names_to = "Metric",
      values_to = "Value"
    ) |>
    dplyr::mutate(
      Metric = factor(Metric, levels = metrics),
      Panel = make_metric_panel(Dataset, Metric, metrics)
    )

  target <- fig_long |>
    dplyr::filter(Metric == "Coverage") |>
    dplyr::distinct(Panel) |>
    dplyr::mutate(Value = 1 - config$alpha, Method = factor("SCP", levels = levels))

  ggplot2::ggplot(
    fig_long,
    ggplot2::aes(
      x = Noise,
      y = Value,
      color = Method,
      linetype = Method,
      linewidth = Method,
      group = Method
    )
  ) +
    ggplot2::geom_hline(
      data = target,
      ggplot2::aes(yintercept = Value),
      linetype = "dashed",
      color = "grey45"
    ) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 1.7) +
    ggplot2::facet_wrap(~Panel, nrow = 2, scales = "free_y") +
    ggplot2::scale_x_continuous(breaks = config$noise_grid) +
    method_scale(levels) +
    ggplot2::guides(
      color = ggplot2::guide_legend(nrow = 1),
      linetype = ggplot2::guide_legend(nrow = 1),
      linewidth = ggplot2::guide_legend(nrow = 1)
    ) +
    ggplot2::labs(x = "Generator noise level", y = NULL) +
    paper_theme()
}

plot_figure7_combined <- function(cal_summary, dist_raw, config) {
  metrics <- c("Coverage", "Average Width")

  left_long <- cal_summary |>
    dplyr::transmute(
      Dataset,
      N_cal,
      Method,
      Coverage = Mean_Cov,
      `Average Width` = Mean_Width
    ) |>
    tidyr::pivot_longer(
      cols = c(Coverage, `Average Width`),
      names_to = "Metric",
      values_to = "Value"
    ) |>
    dplyr::mutate(
      Metric = factor(Metric, levels = metrics),
      Panel = make_metric_panel(Dataset, Metric, metrics)
    )

  target <- left_long |>
    dplyr::filter(Metric == "Coverage") |>
    dplyr::distinct(Panel) |>
    dplyr::mutate(Value = 1 - config$alpha, Method = factor("SCP", levels = method_levels))

  fig_left <- ggplot2::ggplot(
    left_long,
    ggplot2::aes(
      x = N_cal,
      y = Value,
      color = Method,
      linetype = Method,
      linewidth = Method,
      group = Method
    )
  ) +
    ggplot2::geom_hline(
      data = target,
      ggplot2::aes(yintercept = Value),
      linetype = "dashed",
      color = "grey45"
    ) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 1.7) +
    ggplot2::facet_wrap(~Panel, nrow = 2, scales = "free_y") +
    ggplot2::scale_x_continuous(breaks = config$ncal_grid) +
    method_scale(method_levels) +
    ggplot2::guides(
      color = ggplot2::guide_legend(nrow = 1),
      linetype = ggplot2::guide_legend(nrow = 1),
      linewidth = ggplot2::guide_legend(nrow = 1)
    ) +
    ggplot2::labs(x = "Calibration size m", y = NULL) +
    paper_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

  dist_levels <- c("SCP", "RSA-CP Gamma", "RSA-CP Normal", "RSA-CP Beta")
  dist_cols <- c(
    "SCP" = "gray70",
    "RSA-CP Gamma" = "#6C63FF",
    "RSA-CP Normal" = "#9BD3E5",
    "RSA-CP Beta" = "#B64AD8"
  )

  right_long <- dist_raw |>
    dplyr::mutate(Score_Distribution = factor(Score_Distribution, levels = dist_levels)) |>
    tidyr::pivot_longer(cols = c(Cov, Width), names_to = "Metric", values_to = "Value") |>
    dplyr::mutate(
      Metric = dplyr::recode(Metric, Cov = "Coverage", Width = "Average Width"),
      Metric = factor(Metric, levels = metrics),
      Panel = make_metric_panel(Dataset, Metric, metrics)
    )

  dist_target <- right_long |>
    dplyr::filter(Metric == "Coverage") |>
    dplyr::distinct(Panel) |>
    dplyr::mutate(
      Value = 1 - config$alpha,
      Score_Distribution = factor("SCP", levels = dist_levels)
    )

  fig_right <- ggplot2::ggplot(
    right_long,
    ggplot2::aes(x = Score_Distribution, y = Value, fill = Score_Distribution)
  ) +
    ggplot2::geom_hline(
      data = dist_target,
      ggplot2::aes(yintercept = Value),
      linetype = "dashed",
      color = "grey45"
    ) +
    ggplot2::geom_boxplot(alpha = 0.82, outlier.size = 0.45, linewidth = 0.35) +
    ggplot2::facet_wrap(~Panel, nrow = 2, scales = "free_y") +
    ggplot2::scale_x_discrete(labels = c(
      "SCP" = "SCP",
      "RSA-CP Gamma" = "Gamma",
      "RSA-CP Normal" = "Normal",
      "RSA-CP Beta" = "Beta"
    )) +
    ggplot2::scale_fill_manual(values = dist_cols, drop = FALSE) +
    ggplot2::labs(x = NULL, y = NULL) +
    paper_theme() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 0, hjust = 0.5, face = "bold", size = 8),
      legend.position = "none"
    )

  fig_left + fig_right +
    patchwork::plot_layout(widths = c(1.05, 1.15)) +
    patchwork::plot_annotation(
      title = "Figure 7. Calibration Size and Score-Distribution Sensitivity"
    )
}

plot_shock_probability <- function(summary, config) {
  metrics <- c("Coverage", "Average Width")
  levels <- shock_method_levels

  fig_long <- summary |>
    dplyr::transmute(
      Dataset,
      Shock_Probability,
      Method,
      Coverage = Mean_Cov,
      `Average Width` = Mean_Width
    ) |>
    tidyr::pivot_longer(
      cols = c(Coverage, `Average Width`),
      names_to = "Metric",
      values_to = "Value"
    ) |>
    dplyr::mutate(
      Method = factor(as.character(Method), levels = levels),
      Metric = factor(Metric, levels = metrics),
      Panel = make_metric_panel(Dataset, Metric, metrics)
    )

  target <- fig_long |>
    dplyr::filter(Metric == "Coverage") |>
    dplyr::distinct(Panel) |>
    dplyr::mutate(Value = 1 - config$alpha, Method = factor("SCP", levels = levels))

  ggplot2::ggplot(
    fig_long,
    ggplot2::aes(
      x = Shock_Probability,
      y = Value,
      color = Method,
      linetype = Method,
      linewidth = Method,
      group = Method
    )
  ) +
    ggplot2::geom_hline(
      data = target,
      ggplot2::aes(yintercept = Value),
      linetype = "dashed",
      color = "grey45"
    ) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 1.7) +
    ggplot2::facet_wrap(~Panel, nrow = 2, scales = "free_y") +
    ggplot2::scale_x_continuous(breaks = config$shock_probs) +
    method_scale(levels) +
    ggplot2::guides(
      color = ggplot2::guide_legend(nrow = 1),
      linetype = ggplot2::guide_legend(nrow = 1),
      linewidth = ggplot2::guide_legend(nrow = 1)
    ) +
    ggplot2::labs(x = "Shock probability", y = NULL) +
    paper_theme()
}
