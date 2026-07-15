repo_root <- normalizePath(getwd(), mustWork = TRUE)

repo_path <- function(...) {
  file.path(repo_root, ...)
}

ensure_output_dirs <- function() {
  for (dir_path in c(
    repo_path("figures"),
    repo_path("tables")
  )) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  }
}
