helper_path <- if (file.exists(file.path("R", "simulation_helpers.R"))) {
  file.path("R", "simulation_helpers.R")
} else {
  file.path("..", "R", "simulation_helpers.R")
}
source(helper_path)
set_simulation_root()
ensure_output_dirs()

expected <- c(
  file.path("outputs", "raw", "figure3_reference_score_size_raw.csv"),
  file.path("outputs", "summary", "figure3_reference_score_size_summary.csv"),
  file.path("outputs", "figures", "figure3_reference_score_size.pdf"),
  file.path("outputs", "figures", "figure3_reference_score_size.png"),
  file.path("outputs", "raw", "figure6_noise_stability_raw.csv"),
  file.path("outputs", "summary", "figure6_noise_stability_summary.csv"),
  file.path("outputs", "figures", "figure6_noise_stability.pdf"),
  file.path("outputs", "figures", "figure6_noise_stability.png"),
  file.path("outputs", "raw", "figure7_calibration_size_raw.csv"),
  file.path("outputs", "summary", "figure7_calibration_size_summary.csv"),
  file.path("outputs", "raw", "figure7_score_distribution_boxplot_raw.csv"),
  file.path("outputs", "summary", "figure7_score_distribution_boxplot_summary.csv"),
  file.path("outputs", "figures", "figure7_calibration_size_and_score_distribution.pdf"),
  file.path("outputs", "figures", "figure7_calibration_size_and_score_distribution.png"),
  file.path("outputs", "raw", "shock_probability_raw.csv"),
  file.path("outputs", "summary", "shock_probability_summary.csv"),
  file.path("outputs", "figures", "shock_probability.pdf"),
  file.path("outputs", "figures", "shock_probability.png"),
  file.path("outputs", "logs", "method_check.txt"),
  file.path("outputs", "logs", "session_info.txt")
)

missing <- expected[!file.exists(expected)]

lines <- c(
  "RSA-CP simulation reproduction check",
  paste0("Expected files: ", length(expected)),
  paste0("Present files: ", length(expected) - length(missing)),
  if (length(missing) == 0) "PASS: all expected outputs are present." else c("FAIL: missing files:", paste0("  ", missing))
)

writeLines(lines, file.path("outputs", "logs", "reproduce_check.txt"))

if (length(missing) > 0) {
  stop("Reproduction check failed. See outputs/logs/reproduce_check.txt", call. = FALSE)
}

message("Reproduction check passed.")
