# ----------------------------

# This script includes functions to run simulations under different 
# data generating processes and compare the results of GLMM and regression-free
# marginal estimates (including via RMSE, bias, CI width, and coverage).

# Three simulation scenarios or settings are available as described in the paper:
# Setting A ("wellspecified"); Setting B ("plinear"), and Setting C 
# ("pindependent"). See paper's Appendix C.1 for details.

# Inputs: caller-supplied simulation parameters
# Outputs: in-memory data frames or CSVs in simulations/simulation_results
# Run: source("simulations/run_simulations.R")

source("src/simulation_functions.R")

# ----------------------------

normalize_progress_every <- function(progress_every) {
  if (
    is.null(progress_every) ||
    identical(progress_every, FALSE) ||
    (length(progress_every) == 1 && is.atomic(progress_every) && is.na(progress_every))
  ) {
    return(NULL)
  }

  if (!is.numeric(progress_every) || length(progress_every) != 1 || !is.finite(progress_every)) {
    stop("progress_every must be NULL, FALSE, or a single finite number.", call. = FALSE)
  }

  progress_every <- as.integer(progress_every)
  if (progress_every <= 0) {
    return(NULL)
  }

  progress_every
}

resolve_progress_label <- function(progress_label, default_label) {
  if (is.null(progress_label)) {
    return(default_label)
  }

  if (!is.character(progress_label) || length(progress_label) != 1 || !nzchar(progress_label)) {
    stop("progress_label must be NULL or a non-empty string.", call. = FALSE)
  }

  progress_label
}

maybe_report_progress <- function(i, replications, progress_every, progress_label) {
  if (is.null(progress_every)) {
    return(invisible(NULL))
  }

  if (!(i == 1 || i == replications || i %% progress_every == 0)) {
    return(invisible(NULL))
  }

  message(sprintf("%s: %d/%d (%.1f%%)", progress_label, i, replications, 100 * i / replications))
  invisible(NULL)
}

simulate_wellspecified <- function(
  #' Setting A (well-specified) simulation function
    replications = 2000,
    m_LLMs = 4,
    n_items = 40,
    t_trials = 8,
    LLM_capability_mean = 1,
    LLM_capability_sigma = 0.8,
    item_sigma = 1.3,
    mu_sims = 10^5,
    filename = NULL,  # when not null, writes to simulations/simulation_results/filename.csv
    progress_every = NULL,
    progress_label = NULL
) {
  llm_ids <- make_llm_ids(m_LLMs)
  item_ids <- make_item_ids(n_items)
  output <- vector("list", replications)
  progress_every <- normalize_progress_every(progress_every)
  progress_label <- resolve_progress_label(progress_label, "Setting A")

  for (i in seq_len(replications)) {
    llm_capabilities <- stats::rnorm(m_LLMs, mean = LLM_capability_mean, sd = LLM_capability_sigma)
    item_effects <- stats::rnorm(n_items, mean = 0, sd = item_sigma)
    logodds <- llm_capabilities + rep(item_effects, each = m_LLMs)
    dim(logodds) <- c(m_LLMs, n_items)
    p <- plogis(logodds)

    true_marginals <- matrix(NA_real_, nrow = mu_sims, ncol = m_LLMs)
    for (j in seq_len(mu_sims)) {
      sampled_effects <- stats::rnorm(n_items, mean = 0, sd = item_sigma)
      sampled_logodds <- llm_capabilities + rep(sampled_effects, each = m_LLMs)
      dim(sampled_logodds) <- c(m_LLMs, n_items)
      true_marginals[j, ] <- rowMeans(plogis(sampled_logodds))
    }
    true_marginals <- colMeans(true_marginals)

    estimates <- run_benchmark(p, t_trials, llm_ids, item_ids)
    output[[i]] <- calculate_stats(estimates, true_marginals, llm_ids, i - 1)
    maybe_report_progress(i, replications, progress_every, progress_label)
  }

  output <- bind_rows(output)

  if (!is.null(filename)) {
    write_simulation_results(output, filename)
  }

  output
}

simulate_plinear <- function(  # Setting B
  #' Setting B (linear) simulation function
    replications = 2000,
    m_LLMs = 4,
    n_items = 40,
    t_trials = 8,
    LLM_distribution = SETTINGB_BASELINE_LLM_DISTRIBUTION,  # defaults to unif[0.4,0.8]
    item_distribution = SETTINGB_BASELINE_ITEM_DISTRIBUTION,  # defaults to unif[-0.2,0.2]
    filename = NULL,  # when not null, writes to simulations/simulation_results/filename.csv
    progress_every = NULL,
    progress_label = NULL
) {
  llm_ids <- make_llm_ids(m_LLMs)
  item_ids <- make_item_ids(n_items)
  output <- vector("list", replications)
  progress_every <- normalize_progress_every(progress_every)
  progress_label <- resolve_progress_label(progress_label, "Setting B")

  for (i in seq_len(replications)) {
    llm_capabilities <- LLM_distribution(m_LLMs)
    item_effects <- item_distribution(n_items)
    p <- llm_capabilities + rep(item_effects, each = m_LLMs)
    dim(p) <- c(m_LLMs, n_items)
    estimates <- run_benchmark(p, t_trials, llm_ids, item_ids)
    output[[i]] <- calculate_stats(estimates, llm_capabilities, llm_ids, i - 1)
    maybe_report_progress(i, replications, progress_every, progress_label)
  }

  output <- bind_rows(output)

  if (!is.null(filename)) {
    write_simulation_results(output, filename)
  }

  output
}

simulate_pindependent <- function(  # Setting C
  #' Setting C (no structure) simulation function
    replications = 2000,
    m_LLMs = 4,
    n_items = 40,
    t_trials = 8,
    p_distribution = list(SETTINGC_BASELINE_P_DISTRIBUTION, SETTINGC_BASELINE_MU),  # defaults to unif[0.2,1]
    filename = NULL,  # when not null, writes to simulations/simulation_results/filename.csv
    progress_every = NULL,
    progress_label = NULL
) {
  llm_ids <- make_llm_ids(m_LLMs)
  item_ids <- make_item_ids(n_items)
  output <- vector("list", replications)
  progress_every <- normalize_progress_every(progress_every)
  progress_label <- resolve_progress_label(progress_label, "Setting C")
  distribution_spec <- p_distribution
  p_distribution <- distribution_spec[[1]]
  mu <- distribution_spec[[2]]
  true_marginals <- rep(mu, m_LLMs)
  
  for (i in seq_len(replications)) {
    p <- p_distribution(m_LLMs, n_items)
    estimates <- run_benchmark(p, t_trials, llm_ids, item_ids)
    output[[i]] <- calculate_stats(estimates, true_marginals, llm_ids, i - 1)
    maybe_report_progress(i, replications, progress_every, progress_label)
  }
  
  output <- bind_rows(output)
  
  if (!is.null(filename)) {
    write_simulation_results(output, filename)
  }
  
  output
}
