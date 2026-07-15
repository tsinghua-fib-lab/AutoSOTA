helper_path <- if (file.exists(file.path("R", "simulation_helpers.R"))) {
  file.path("R", "simulation_helpers.R")
} else {
  file.path("..", "R", "simulation_helpers.R")
}
source(helper_path)
set_simulation_root()
ensure_output_dirs()

check_file <- file.path("outputs", "logs", "method_check.txt")

code_files <- list.files(
  c("R", "experiments"),
  pattern = "\\.R$",
  recursive = TRUE,
  full.names = TRUE
)

bad_patterns <- c(
  paste0("apply_", "scale_ot_map"),
  paste0("S_", "mapped_real"),
  paste0("w_", "r <-"),
  paste0("w_", "s <-"),
  paste0("cum_", "weights"),
  paste("weighted", "quantile"),
  "mean(S_target)",
  "mean(S_source)"
)

found <- list()
for (f in code_files) {
  txt <- paste(readLines(f, warn = FALSE), collapse = "\n")
  hits <- bad_patterns[vapply(bad_patterns, grepl, logical(1), txt, fixed = TRUE)]
  if (length(hits) > 0) found[[f]] <- hits
}

rsa_txt <- paste(readLines(file.path("R", "rsa_cp_methods.R"), warn = FALSE), collapse = "\n")
spi_txt <- paste(readLines(file.path("R", "spi_methods.R"), warn = FALSE), collapse = "\n")

positive_checks <- c(
  "RSA-CP barycentric OT map is implemented" =
    grepl("rsa_ot_map", rsa_txt, fixed = TRUE) &&
      grepl("num[i] <- num[i] + delta * tgt[j]", rsa_txt, fixed = TRUE) &&
      grepl("cummax(num / den)", rsa_txt, fixed = TRUE),
  "RSA-CP uses Beta-Binomial rank windows" =
    grepl("q_betabin", rsa_txt, fixed = TRUE) &&
      grepl("lbeta", rsa_txt, fixed = TRUE) &&
      grepl("b_minus", rsa_txt, fixed = TRUE) &&
      grepl("b_plus", rsa_txt, fixed = TRUE),
  "RSA-CP uses candidate ranks after OT alignment" =
    grepl("findInterval(Z_new, Z_real_ord", rsa_txt, fixed = TRUE) &&
      grepl("rank_k", rsa_txt, fixed = TRUE),
  "SPI uses fast-form windows" =
    grepl("spi_windows", spi_txt, fixed = TRUE) &&
      grepl("Rminus", spi_txt, fixed = TRUE) &&
      grepl("Rplus", spi_txt, fixed = TRUE) &&
      grepl("q_syn_prime", spi_txt, fixed = TRUE) &&
      grepl("Rtilde_minus", spi_txt, fixed = TRUE) &&
      grepl("Rtilde_plus", spi_txt, fixed = TRUE)
)

lines <- c(
  "RSA-CP simulation method check",
  paste0("Checked code files: ", length(code_files)),
  "",
  "Old heuristic scan:",
  if (length(found) == 0) {
    "  PASS: no banned old heuristic strings found in R/ or experiments/."
  } else {
    c("  FAIL:", unlist(Map(function(f, hits) paste(" ", f, ":", paste(hits, collapse = ", ")), names(found), found)))
  },
  "",
  "Positive implementation checks:",
  paste0("  ", ifelse(positive_checks, "PASS: ", "FAIL: "), names(positive_checks))
)

writeLines(lines, check_file)

if (length(found) > 0 || any(!positive_checks)) {
  stop("Method check failed. See outputs/logs/method_check.txt", call. = FALSE)
}

message("Method check passed. Log written to ", check_file)
