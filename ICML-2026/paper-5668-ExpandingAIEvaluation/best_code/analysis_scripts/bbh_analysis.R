# Run: Rscript analysis_scripts/bbh_analysis.R
# Inputs: benchmark_results/bbh.csv
# Outputs: 
#   figures/bbh/
#     Figure5_bbh_summary.png
#     FigureC5a_bbh_diagnostic.pdf
#   tables/bbh_summary.csv

# ----------------------------
# Imports
# ----------------------------

source("src/analysis_functions.R")
library(ggplot2)
library(forestploter)
library(dplyr)
library(stringr)
set.seed(1001)
BBH_TASK_ORDER <- c(
  "tracking_shuffled_objects_seven_objects",
  "geometric_shapes",
  "disambiguation_qa",
  "salient_translation_error_detection",
  "causal_judgement",
  "logical_deduction_seven_objects",
  "movie_recommendation",
  "date_understanding",
  "ruin_names",
  "snarks",
  "sports_understanding",
  "penguins_in_a_table",
  "formal_fallacies",
  "reasoning_about_colored_objects",
  "hyperbaton",
  "boolean_expressions",
  "navigate",
  "web_of_lies",
  "temporal_sequences"
)

# ----------------------------
# Plotting functions
# ----------------------------

compute_task_order <- function(summary_data) {
  stopifnot(all(summary_data$Task %in% BBH_TASK_ORDER))
  stopifnot(all(BBH_TASK_ORDER %in% summary_data$Task))

  data.frame(
    Task = BBH_TASK_ORDER,
    task_order = seq_along(BBH_TASK_ORDER),
    stringsAsFactors = FALSE
  )
}

format_task_labels <- function(task_names) {
  formatted_names <- task_names %>%
    str_replace_all("_", " ") %>%
    str_to_title()

  dplyr::case_when(
    formatted_names == "Tracking Shuffled Objects Seven Objects" ~ "Tracking Shuffled Objects",
    formatted_names == "Logical Deduction Seven Objects" ~ "Logical Deduction",
    formatted_names == "Disambiguation Qa" ~ "Disambiguation QA",
    formatted_names == "Movie Recommendation" ~ "Movie Rec.",
    TRUE ~ formatted_names
  ) %>%
    str_wrap(width = 16)
}

prepare_bbh_summary_table <- function(summary_data, task_order) {
  summary_data %>%
    mutate(across(-n, \(x) if (is.numeric(x)) signif(x, 2) else x)) %>%
    left_join(task_order, by = "Task") %>%
    arrange(task_order) %>%
    rename(`sigma^2` = `σ²`) %>%
    select(-task_order) %>%
    relocate(Task, n, Score, `sigma^2`, ICC, EST) %>%
    mutate(
      `Variance Ratio\n(Q1, Median, Q3) ` = paste(rep(" ", 20), collapse = " "),
      Task = format_task_labels(Task)
    )
}

build_bbh_forest_plot <- function(plot_data) {  # Figure 5
  plot_theme <- forest_theme(
    core = list(
      fg_params = list(hjust = 0, x = 0),
      padding = unit(c(2, 3), "mm")
    ),
    colhead = list(
      padding = unit(c(0, 3), "mm")
    )
  )

  forest(
    plot_data[, c(1, 3:6, 10)],
    est = as.numeric(plot_data$Median),
    lower = as.numeric(plot_data$t25),
    upper = as.numeric(plot_data$t75),
    sizes = 0.5,
    ref_line = 1,
    ci_column = 6,
    xlim = c(0, 3.5),
    ticks_at = c(0, 1, 2, 3),
    theme = plot_theme,
    footnote = "\nm = 22 LLMs, t = 5 trials. n = 250 items except\nCausal Judgement (187), Snarks (178), and Penguins (146)."
  ) %>%
    edit_plot(
      col = 2:5,
      which = "text",
      hjust = unit(0.5, "npc"),
      x = unit(0.5, "npc")
    ) %>%
    edit_plot(
      col = 2:6,
      which = "text",
      part = "header",
      hjust = unit(0.5, "npc"),
      x = unit(0.5, "npc")
    )
}

# ----------------------------
# Load and analyze BBH
# ----------------------------

data <- load_benchmark_data("bbh.csv") %>%
  apply_archived_factor_ordering() %>%
  mutate(task = str_remove(item_id, "_[^_]+$"))  # gets task from item_id

# Demonstrate variance decomposition for each task subdivision
plot.me <- summarize_benchmark_slices(
  data,
  slice_columns = "task",
  output_columns = "Task",
  context_fn = function(key) sprintf("Task '%s'", key$Task)
)

# ----------------------------
# Format outputs
# ----------------------------

plot.me <- prepare_bbh_summary_table(plot.me, compute_task_order(plot.me))
p <- build_bbh_forest_plot(plot.me)

p_wh <- get_wh(plot = p, unit = "in")
ggsave(  # Figure 5
  repo_path("figures", "bbh", "Figure5_bbh_summary.png"), plot = p, dpi = 300,
  width = p_wh[1]*.93, height = p_wh[2]*.95 # crop margins
)

write.csv(
  plot.me,
  file = repo_path("tables", "bbh_summary.csv"),
  row.names = FALSE
)

glmm <- fit_glmm(data)
print_model_performance_table(glmm, "BIG-Bench Hard", observed = data$score)
write_glmm_diagnostics(
  glmm,
  data,
  "bbh",
  diagnostic_figure_label = "FigureC5a"
)  # Figure C.5a
