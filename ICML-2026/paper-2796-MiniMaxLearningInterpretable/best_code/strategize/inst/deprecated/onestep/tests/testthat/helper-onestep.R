# Archived helper subset for the deprecated one-step estimator tests.
# These helpers are retained with the archived tests so the test fixtures stay
# close to the deprecated implementation.

skip_if_no_conda <- function(conda_env = "strategize_env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    skip("reticulate package not available")
  }

  conda_list <- tryCatch(
    reticulate::conda_list(),
    error = function(e) NULL
  )

  if (is.null(conda_list) || !conda_env %in% conda_list$name) {
    skip(paste0("Conda environment '", conda_env, "' not available"))
  }
}

skip_if_no_jax <- function(conda_env = "strategize_env") {
  skip_if_no_conda(conda_env)

  jax_available <- tryCatch({
    reticulate::use_condaenv(conda_env, required = TRUE)
    reticulate::py_module_available("jax")
  }, error = function(e) FALSE)

  if (!isTRUE(jax_available)) {
    skip("JAX not available in conda environment")
  }
}

skip_if_slow <- function() {
  if (!identical(Sys.getenv("STRATEGIZE_RUN_SLOW_TESTS"), "true")) {
    skip("Slow test (set STRATEGIZE_RUN_SLOW_TESTS=true to run)")
  }
}

skip_onestep_tests <- function() {
  skip("One-step estimator tests require larger datasets (set STRATEGIZE_RUN_ONESTEP_TESTS=true to run)")
}

generate_test_data <- function(n = 1000, n_factors = 3, n_levels = 2, seed = 1234321) {
  withr::local_seed(seed)

  levels <- LETTERS[seq_len(n_levels)]
  W <- matrix(
    sample(levels, n * n_factors, replace = TRUE),
    nrow = n,
    ncol = n_factors
  )
  colnames(W) <- paste0("V", seq_len(n_factors))

  n_pairs <- n / 2
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))

  effects <- seq(0.4, 0.2, length.out = n_factors)
  latent_utility <- drop((W == "B") %*% effects)

  Y <- as.numeric(ave(
    latent_utility,
    respondent_task_id,
    FUN = function(g) rank(g, ties.method = "random") == length(g)
  ))

  list(
    Y = Y,
    W = W,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order
  )
}

generate_test_p_list <- function(W) {
  suppressMessages(create_p_list(W, uniform = TRUE))
}
