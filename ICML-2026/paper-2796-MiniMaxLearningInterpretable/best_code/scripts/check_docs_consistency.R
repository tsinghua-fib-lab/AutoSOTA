get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE))
  }
  normalizePath("scripts/check_docs_consistency.R", mustWork = TRUE)
}

read_full_text <- function(path) {
  paste(readLines(path, warn = FALSE), collapse = "\n")
}

read_roxygen_text <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- grep("^#'", lines, value = TRUE)
  paste(sub("^#' ?", "", lines), collapse = "\n")
}

check_absent <- function(text, path, patterns) {
  failures <- character(0)
  for (entry in patterns) {
    if (grepl(entry$pattern, text, perl = TRUE)) {
      failures <- c(
        failures,
        sprintf("%s contains forbidden documentation token: %s", path, entry$label)
      )
    }
  }
  failures
}

check_present <- function(text, path, patterns) {
  failures <- character(0)
  for (entry in patterns) {
    if (!grepl(entry$pattern, text, perl = TRUE)) {
      failures <- c(
        failures,
        sprintf("%s is missing required documentation content: %s", path, entry$label)
      )
    }
  }
  failures
}

script_path <- get_script_path()
repo_root <- dirname(dirname(script_path))

narrative_paths <- c(
  file.path(repo_root, "README.md"),
  file.path(repo_root, "strategize", "vignettes", "QuickStart.Rmd"),
  file.path(repo_root, "strategize", "vignettes", "MainVignette.Rmd"),
  file.path(repo_root, "strategize", "vignettes", "Troubleshooting.Rmd")
)

roxygen_paths <- c(
  file.path(repo_root, "strategize", "R", "build_backend.R"),
  file.path(repo_root, "strategize", "R", "two_step_master.R"),
  file.path(repo_root, "strategize", "R", "two_step_master_cv.R")
)

forbidden_public_terms <- list(
  list(label = "PiStar_point", pattern = "\\bPiStar_point\\b"),
  list(label = "PiStar_se", pattern = "\\bPiStar_se\\b"),
  list(label = "MaxMin", pattern = "\\bMaxMin\\b"),
  list(label = 'conda_env = "strategize"', pattern = 'conda_env\\s*=\\s*"strategize"')
)

forbidden_narrative_terms <- list(
  list(label = "Q_point_mEst", pattern = "\\bQ_point_mEst\\b"),
  list(label = "Q_se_mEst", pattern = "\\bQ_se_mEst\\b"),
  list(label = "res_avg$", pattern = "\\bres_avg\\$"),
  list(label = "my_data_red$profile", pattern = "my_data_red\\$profile\\b")
)

required_patterns <- list(
  "README.md" = list(
    list(label = "canonical backend setup", pattern = 'build_backend\\(conda_env\\s*=\\s*"strategize_env"\\)'),
    list(label = "prediction-only API mention", pattern = "strategic_prediction\\("),
    list(label = "foundation training split mention", pattern = "preference\\.fm::fit_conjoint_foundation_model\\("),
    list(label = "foundation adaptation mention", pattern = "adapt_conjoint_foundation_model\\(")
  ),
  "QuickStart.Rmd" = list(
    list(label = "canonical outcome point field", pattern = "result\\$Q_point\\b"),
    list(label = "canonical outcome SE field", pattern = "result\\$Q_se\\b")
  ),
  "MainVignette.Rmd" = list(
    list(label = "average-case pi field", pattern = "res_avecase\\$pi_star_point\\b"),
    list(label = "average-case outcome field", pattern = "res_avecase\\$Q_point\\b"),
    list(label = "average-case SE field", pattern = "res_avecase\\$Q_se\\b"),
    list(label = "average-case pi SE field", pattern = "res_avecase\\$pi_star_se\\b"),
    list(label = "canonical backend env", pattern = 'conda_env\\s*=\\s*"strategize_env"'),
    list(label = "foundation split mention", pattern = "preference\\.fm")
  ),
  "build_backend.R" = list(
    list(label = "strategize_env example", pattern = 'build_backend\\(conda_env\\s*=\\s*"strategize_env"')
  ),
  "two_step_master.R" = list(
    list(label = "Q_point return docs", pattern = "\\\\item\\{\\\\code\\{Q_point\\}\\}"),
    list(label = "Q_se return docs", pattern = "\\\\item\\{\\\\code\\{Q_se\\}\\}")
  ),
  "two_step_master_cv.R" = list(
    list(label = "Q_point return docs", pattern = "\\\\item\\{Q_point\\}"),
    list(label = "Q_se return docs", pattern = "\\\\item\\{Q_se\\}")
  )
)

failures <- character(0)

for (path in narrative_paths) {
  if (!file.exists(path)) {
    failures <- c(failures, sprintf("Missing narrative documentation file: %s", path))
    next
  }
  text <- read_full_text(path)
  failures <- c(failures, check_absent(text, path, forbidden_public_terms))
  failures <- c(failures, check_absent(text, path, forbidden_narrative_terms))
  key <- basename(path)
  if (!is.null(required_patterns[[key]])) {
    failures <- c(failures, check_present(text, path, required_patterns[[key]]))
  }
}

for (path in roxygen_paths) {
  if (!file.exists(path)) {
    failures <- c(failures, sprintf("Missing roxygen source file: %s", path))
    next
  }
  text <- read_roxygen_text(path)
  failures <- c(failures, check_absent(text, path, forbidden_public_terms))
  key <- basename(path)
  if (!is.null(required_patterns[[key]])) {
    failures <- c(failures, check_present(text, path, required_patterns[[key]]))
  }
}

if (length(failures) > 0L) {
  writeLines(c("Documentation consistency checks failed:", failures), con = stderr())
  quit(status = 1L)
}

cat("Documentation consistency checks passed.\n")
