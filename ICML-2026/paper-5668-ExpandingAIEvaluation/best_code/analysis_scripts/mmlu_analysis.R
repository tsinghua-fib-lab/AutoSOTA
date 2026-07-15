# Run: Rscript analysis_scripts/mmlu_analysis.R
# Inputs: benchmark_results/mmlu.csv
# Outputs: 
#   figures/mmlu/
#     FigureC3_mmlu_summary.png
#     FigureC5b_mmlu_diagnostic.pdf
#   tables/mmlu_summary.csv

# ----------------------------
# Imports
# ----------------------------

source("src/analysis_functions.R")
library(ggplot2)
library(forestploter)
library(dplyr)
library(stringr)
set.seed(1001)
MMLU_CATEGORY_ORDER <- c("STEM", "Business", "Medical")

# ----------------------------
# Plotting functions
# ----------------------------

prepend_group_headers <- function(data, group_col) {
  split_rows <- split(data, data[[group_col]], drop = TRUE)
  header_rows <- lapply(names(split_rows), function(group_name) {
    header <- split_rows[[group_name]][1, , drop = FALSE]
    header[,] <- NA
    header[[group_col]] <- group_name

    body <- split_rows[[group_name]]
    body[[group_col]] <- ""
    dplyr::bind_rows(header, body)
  })

  dplyr::bind_rows(header_rows)
}

prepare_mmlu_summary_table <- function(summary_data) {
  summary_data %>%
    mutate(across(where(is.numeric), \(x) round(x, 2))) %>%
    filter(Category %in% c("Business", "Medical", "STEM")) %>%
    mutate(Category = factor(Category, levels = MMLU_CATEGORY_ORDER)) %>%
    arrange(Category) %>%
    rename(`sigma^2` = `σ²`) %>%
    prepend_group_headers("Category") %>%
    mutate(`Variance Ratio (Q1, Median, Q3) ` = paste(rep(" ", 10), collapse = " ")) %>%
    relocate(Category, Language) %>%
    select(-n) %>%
    mutate(across(everything(), as.character)) %>%
    mutate(across(everything(), \(x) tidyr::replace_na(x, "")))
}

build_mmlu_forest_plot <- function(plot_data) {  # Figure C.3
  forest(
    plot_data[, c(1:6, 10)],
    est = as.numeric(plot_data$Median),
    lower = as.numeric(plot_data$t25),
    upper = as.numeric(plot_data$t75),
    sizes = 0.5,
    ref_line = 1,
    ci_column = 7,
    xlim = c(0, 5),
    footnote = "m = 22 LLMs, t = 5 trials. n = 46 (STEM), 58 (Business), or 36 (Medical) items."
  )
}

# ----------------------------
# Load and analyze MMLU
# ----------------------------

data <- load_benchmark_data("mmlu.csv") %>%
  apply_archived_factor_ordering() %>%
  mutate(
    item_id = paste(language, item_id, sep = "::"),
    item_id = factor(item_id, levels = unique(item_id))
  )

# Demonstrate variance decomposition for each (category,language) subdivision
plot.me <- summarize_benchmark_slices(
  data,
  slice_columns = c("subject_category", "language"),
  output_columns = c("Category", "Language"),
  context_fn = function(key) sprintf("Language '%s' and category '%s'", key$Language, key$Category)
)
  
# ----------------------------
# Format outputs
# ----------------------------
plot.me <- prepare_mmlu_summary_table(plot.me)

p <- build_mmlu_forest_plot(plot.me)
p_wh <- get_wh(plot = p, unit = "in")
ggsave(
  plot = p,
  repo_path("figures", "mmlu", "FigureC3_mmlu_summary.png"),
  dpi = 300,
  width = p_wh[1],
  height = p_wh[2]
)
p

write.csv(
  plot.me,
  file = repo_path("tables", "mmlu_summary.csv"),
  row.names = FALSE
)

glmm <- fit_glmm(data)
print_model_performance_table(glmm, "Global-MMLU Lite", observed = data$score)
write_glmm_diagnostics(
  glmm,
  data,
  "mmlu",
  diagnostic_figure_label = "FigureC5b"
)  # Figure C.5b
