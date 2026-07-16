{
  rm(list = ls())
  options(error = NULL)

  install.packages( "~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F);
  # strategize::build_backend()
  
  # Package configuration (change these for different packages)
  package_name <- "strategize"
  base_path <- sprintf("~/Documents/%s-software", package_name)

  run_initial_tests <- function(path) {
    old_max_fails <- Sys.getenv("TESTTHAT_MAX_FAILS", unset = NA_character_)
    on.exit({
      if (is.na(old_max_fails)) {
        Sys.unsetenv("TESTTHAT_MAX_FAILS")
      } else {
        Sys.setenv(TESTTHAT_MAX_FAILS = old_max_fails)
      }
    }, add = TRUE)

    Sys.setenv(TESTTHAT_MAX_FAILS = "Inf")
    devtools::test(path)
  }

  run_doc_checks <- function(repo_path) {
    script_path <- file.path(repo_path, "scripts", "check_docs_consistency.R")
    if (!file.exists(script_path)) {
      stop("Documentation consistency script is missing: ", script_path)
    }

    status <- system2(
      command = file.path(R.home("bin"), "Rscript"),
      args = script_path
    )

    if (!identical(status, 0L)) {
      stop("Documentation consistency checks failed.")
    }

    invisible(TRUE)
  }

  maybe_linearize_pdf <- function(pdf_path) {
    qpdf_bin <- Sys.which("qpdf")
    if (!nzchar(qpdf_bin)) {
      message("qpdf not found on PATH; skipping PDF linearization.")
      return(invisible(FALSE))
    }

    if (!file.exists(pdf_path)) {
      stop("Cannot linearize missing PDF: ", pdf_path)
    }

    tmp_pdf <- tempfile(pattern = "strategize-linearized-", fileext = ".pdf")
    status <- system2(
      command = qpdf_bin,
      args = c("--linearize", normalizePath(pdf_path), tmp_pdf)
    )

    if (!identical(status, 0L) || !file.exists(tmp_pdf)) {
      stop("qpdf linearization failed for: ", pdf_path)
    }

    if (file.exists(pdf_path)) {
      file.remove(pdf_path)
    }
    if (!file.rename(tmp_pdf, pdf_path)) {
      stop("Failed to replace PDF after linearization: ", pdf_path)
    }

    message("Linearized PDF: ", pdf_path)
    invisible(TRUE)
  }

  setwd(base_path)
  package_path <- file.path(base_path, package_name)

  # Get version number from DESCRIPTION
  versionNumber <- read.dcf(file.path(package_path, "DESCRIPTION"), fields = "Version")[1, 1]
  cat(sprintf("Documenting %s version %s\n", package_name, versionNumber))

  # Generate datalist for package data (if any)
  try(tools::add_datalist(package_path, force = TRUE, small.size = 1L), silent = TRUE)

  # Build vignettes
  devtools::build_vignettes(package_path)

  # Document package (generates .Rd files and NAMESPACE)
  devtools::document(package_path)

  # Validate README, vignettes, and roxygen docs before building artifacts
  run_doc_checks(base_path)

  # Remove old PDF manual
  pdf_file <- sprintf("./%s.pdf", package_name)
  try(file.remove(pdf_file), silent = TRUE)

  # Create new PDF manual
  system(sprintf("R CMD Rd2pdf %s", shQuote(package_path)))
  maybe_linearize_pdf(pdf_file)

  # Run tests (stop on failure)
  test_results <- run_initial_tests(package_path)
  if (any(as.data.frame(test_results)$failed > 0)) {
    stop("Tests failed! Stopping build process.")
  }
  cat("\n\U2713 Done with tests...\n\n")

  # Show object sizes in environment (for debugging memory usage)
  log(sort(sapply(ls(), function(l_) { object.size(eval(parse(text = l_))) })))

  # Check package to ensure it meets CRAN standards
  devtools::check(package_path)

  # Build tarball
  system(paste(
    shQuote(file.path(R.home("bin"), "R")),
    "CMD build --resave-data",
    shQuote(package_path)
  ))

  # Check as CRAN
  tarball <- sprintf("%s_%s.tar.gz", package_name, versionNumber)
  system(paste(
    shQuote(file.path(R.home("bin"), "R")),
    "CMD check --as-cran",
    shQuote(tarball)
  ))


  # Manual commands for reference:
  # R CMD build --resave-data ~/Documents/strategize-software/strategize
  # R CMD check --as-cran ~/Documents/strategize-software/strategize_0.0.1.tar.gz

  # Install from local source
  install.packages(package_path, repos = NULL, type = "source", force = FALSE)
  # install.packages( "~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F);
  # devtools::install_github("cjerzak/strategize-software/strategize", dependencies = TRUE) # install from github
  # strategize::build_backend()
}
