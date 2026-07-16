#!/usr/bin/env Rscript

if (!require("dplyr", character.only = TRUE, quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
  library(dplyr)
}

process_atlas <- function(file_path) {
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  if ("Theta_ATLAS" %in% colnames(data)) {
    diff <- abs(data$Theta_ATLAS - data$Theta_WLE)
  } else {
    diff <- abs(data$Theta_CAT - data$Theta_WLE)
  }
  data.frame(
    file          = file_path,
    model_count   = nrow(data),
    mae_cat       = mean(diff),
    sd_mae_cat    = sd(diff),
    avg_num_items = mean(data$Num_Items),
    sd_num_items  = sd(data$Num_Items),
    avg_time_cat  = mean(data$Time_Taken_Sec),
    sd_time_cat   = sd(data$Time_Taken_Sec)
  )
}

process_baseline <- function(file_path, ground_truth_path) {
  data  <- read.csv(file_path, stringsAsFactors = FALSE)
  truth <- read.csv(ground_truth_path, stringsAsFactors = FALSE)
  diff  <- abs(data$Theta_WLE - truth$Theta_WLE)
  data.frame(
    file          = file_path,
    model_count   = nrow(data),
    mae_cat       = mean(diff),
    sd_mae_cat    = sd(diff),
    avg_num_items = mean(data$Num_Items),
    sd_num_items  = sd(data$Num_Items),
    avg_time_cat  = mean(data$Time_Taken_Sec),
    sd_time_cat   = sd(data$Time_Taken_Sec)
  )
}

collect_files <- function(path, pattern) {
  files <- list.files(path = path, pattern = pattern, full.names = TRUE)
  cat("Found", length(files), "files in", path, "\n")
  files
}

benchmarks <- c("winogrande", "truthfulqa", "hellaswag", "gsm8k", "arc")

# Ground truth file differs for gsm8k
gt_file <- function(bm) {
  paste0(bm, "/irt_person_scores_WLE_SE_test.csv")
}

results <- list()

# ATLAS results
for (bm in benchmarks) {
  files <- collect_files(paste0(bm, "/atlas_", bm, "_random"), "irt_person_scores.*\\.csv$")
  for (f in files) results[[length(results) + 1]] <- process_atlas(f)
}

# Baseline (EAP) results
for (bm in benchmarks) {
  files <- collect_files("experiments/baseline_compare", paste0(".*", bm, ".*[Ee][Aa][Pp].*\\.csv$"))
  for (f in files) results[[length(results) + 1]] <- process_baseline(f, gt_file(bm))
}

if (length(results) > 0) {
  all_results <- do.call(rbind, results)
  cat("\n=== Summary ===\n")
  print(all_results)
  write.csv(all_results, "rmse_cat_summary.csv", row.names = FALSE)
  cat("\nResults saved to rmse_cat_summary.csv\n")
} else {
  cat("No result files found.\n")
}
