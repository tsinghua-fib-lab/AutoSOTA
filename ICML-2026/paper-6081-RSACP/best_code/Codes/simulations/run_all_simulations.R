args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_idx <- grep(file_arg, args, fixed = TRUE)
if (length(script_idx) > 0) {
  script_path <- normalizePath(sub(file_arg, "", args[script_idx[1]]), mustWork = TRUE)
  setwd(dirname(script_path))
}

source(file.path("R", "simulation_helpers.R"))
set_simulation_root()
load_simulation_packages()
ensure_output_dirs()

run_log <- file.path("outputs", "logs", "run_log.txt")
if (file.exists(run_log)) unlink(run_log)

config <- simulation_config()
log_message("RSA-CP simulation run started.")
log_message("Seed = ", config$seed)
log_message("Trials = ", config$n_trials)
log_message("alpha = ", config$alpha, "; beta = ", config$beta)

source(file.path("checks", "method_check.R"))

source(file.path("experiments", "figure3_reference_score_size.R"))
source(file.path("experiments", "figure6_noise_stability.R"))
source(file.path("experiments", "figure7_calibration_size.R"))
source(file.path("experiments", "shock_probability_experiment.R"))

writeLines(capture.output(sessionInfo()), file.path("outputs", "logs", "session_info.txt"))

summary_files <- c(
  "figure3_reference_score_size_summary.csv",
  "figure6_noise_stability_summary.csv",
  "figure7_calibration_size_summary.csv",
  "figure7_score_distribution_boxplot_summary.csv",
  "shock_probability_summary.csv"
)

summary_labels <- c(
  "Figure 3 reference score size",
  "Figure 6 generator-noise stability",
  "Figure 7 calibration size",
  "Figure 7 score-distribution boxplot",
  "Shock probability drift"
)

combined <- dplyr::bind_rows(lapply(seq_along(summary_files), function(i) {
  path <- file.path("outputs", "summary", summary_files[i])
  if (!file.exists(path)) return(NULL)
  dat <- read.csv(path)
  dat$Experiment <- summary_labels[i]
  dat
}))
write.csv(combined, file.path("outputs", "summary", "all_summary_results.csv"), row.names = FALSE)

source(file.path("checks", "reproduce_check.R"))
log_message("RSA-CP simulation run completed.")
