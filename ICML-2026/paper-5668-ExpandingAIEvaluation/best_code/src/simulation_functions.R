# Defines helper functions for simulation script

# ----------------------------
# Imports and constants
# ----------------------------

source("src/analysis_functions.R")
ensure_output_dirs()

library(dplyr)
library(GLMMadaptive)

ESTIMATE_COLS <- c(
  "LLM_id",
  "marginal_estimate_rf",
  "marginal_lower_rf",
  "marginal_upper_rf"
)

MARGINAL_RENAMER <- c(
  marginal_estimate_glmm = "pred",
  marginal_lower_glmm = "low",
  marginal_upper_glmm = "upp"
)

# ----------------------------
# Simulation Defaults
# ----------------------------

SETTINGB_BASELINE_LLM_DISTRIBUTION <- function(m_llms) {
  stats::runif(m_llms, min = 0.4, max = 0.8)
}

SETTINGB_BASELINE_ITEM_DISTRIBUTION <- function(n_items) {
  stats::runif(n_items, min = -0.2, max = 0.2)
}

SETTINGC_BASELINE_P_DISTRIBUTION <- function(m_llms,n_items){
  matrix(
    stats::runif(m_llms * n_items, min = 0.2, max = 1),
    nrow = m_llms,
    ncol = n_items
  )
}
SETTINGC_BASELINE_MU <- 0.6

# ----------------------------
# Helper functions
# ----------------------------

int_to_alpha <- function(index) {
  letters_out <- character()
  while (index > 0) {
    remainder <- (index - 1) %% 26
    letters_out <- c(letters[remainder + 1], letters_out)
    index <- (index - remainder - 1) %/% 26
  }
  paste0(letters_out, collapse = "")
}

make_llm_ids <- function(m_llms) {
  paste0("LLM_", vapply(seq_len(m_llms), int_to_alpha, character(1)))
}

make_item_ids <- function(n_items) {
  paste0("item_", seq.int(0, n_items - 1))
}

simulation_results_path <- function(output_stem) {
  repo_path("simulations", "simulation_results", paste0(output_stem, ".csv"))
}

sweep_results_path <- function(output_stem) {
  repo_path("simulations", "sweep_results", paste0(output_stem, ".csv"))
}

write_simulation_results <- function(data, output_stem) {
  #' Writes results of a simulation set to 
  output_path <- simulation_results_path(output_stem)
  write.csv(data, output_path, row.names = FALSE)
  invisible(output_path)
}

write_sweep_results <- function(data, output_stem) {
  output_path <- sweep_results_path(output_stem)
  write.csv(data, output_path, row.names = FALSE)
  invisible(output_path)
}

stats_to_row <- function(stats_table) {
  as.data.frame(t(stats_table), stringsAsFactors = FALSE)
}

warn_glmm_failure <- function(stage, err) {
  warning(
    sprintf("GLMM %s failed: %s", stage, conditionMessage(err)),
    call. = FALSE
  )
}

format_results <- function(results, model_ids, item_ids) {
  stopifnot(length(dim(results)) == 3)
  t_trials <- dim(results)[1]
  m_llms <- dim(results)[2]
  n_items <- dim(results)[3]
  stopifnot(length(model_ids) == m_llms, length(item_ids) == n_items)
  
  index <- expand.grid(
    item_index = seq_len(n_items),
    model_index = seq_len(m_llms),
    trial = seq.int(0, t_trials - 1),
    KEEP.OUT.ATTRS = FALSE
  )
  
  data.frame(
    score = as.vector(aperm(results, c(3, 2, 1))),
    item_id = item_ids[index$item_index],
    LLM_id = model_ids[index$model_index],
    trial = index$trial
  )
}

estimate_glmm <- function(results) {
  glmm <- tryCatch(
    fit_glmm(results),
    error = function(err) {
      warn_glmm_failure("fit", err)
      NULL
    }
  )
  
  if (is.null(glmm)) {
    return(
      data.frame(
        LLM_id = unique(results$LLM_id),
        marginal_estimate_glmm = NA_real_,
        marginal_lower_glmm = NA_real_,
        marginal_upper_glmm = NA_real_
      )
    )
  }
  
  marginal_estimates <- tryCatch(
    get_glmm_estimates(glmm, data = results),
    error = function(err) {
      warn_glmm_failure("marginal response-scale prediction", err)
      NULL
    }
  )
  
  if (is.null(marginal_estimates)) {
    return(
      data.frame(
        LLM_id = unique(results$LLM_id),
        marginal_estimate_glmm = NA_real_,
        marginal_lower_glmm = NA_real_,
        marginal_upper_glmm = NA_real_
      )
    )
  }
  
  marginal_estimates %>%
    rename(any_of(MARGINAL_RENAMER)) %>%
    select(any_of(c("LLM_id", "marginal_estimate_glmm", "marginal_lower_glmm", "marginal_upper_glmm")))
}

estimate_rf <- function(results) {
  t_trials <- dplyr::n_distinct(results$trial)
  n_items <- dplyr::n_distinct(results$item_id)
  stopifnot(n_items >= 2, t_trials >= 2)

  compute_marginal_accuracy_(results) %>%
    rename(
      marginal_estimate_rf = prob,
      marginal_lower_rf = lower,
      marginal_upper_rf = upper
    ) %>%
    select(any_of(ESTIMATE_COLS))
}

# ----------------------------
# Simulation functions
# ----------------------------

draw_results <- function(p, t_trials) {
  stopifnot(is.matrix(p), t_trials >= 1, all(is.finite(p)), all(p >= 0), all(p <= 1))
  probs <- rep(as.vector(p), each = t_trials)
  array(
    stats::rbinom(length(probs), size = 1, prob = probs),
    dim = c(t_trials, nrow(p), ncol(p))
  )
}

run_benchmark <- function(
    p,
    t_trials,
    model_ids,
    item_ids
) {
  results <- format_results(draw_results(p, t_trials), model_ids, item_ids)

  estimate_rf(results) %>%
    left_join(estimate_glmm(results), by = "LLM_id")
}

calculate_stats <- function(
    estimates,
    true_marginals,
    model_ids,
    run_id = NULL
) {
  stopifnot(length(true_marginals) == length(model_ids))
  
  stats <- data.frame(
    LLM_id = model_ids,
    marginal_true_value = true_marginals
  ) %>%
    left_join(estimates, by = "LLM_id")

  stats$marginal_error_rf <- stats$marginal_estimate_rf - stats$marginal_true_value
  stats$marginal_error_glmm <- stats$marginal_estimate_glmm - stats$marginal_true_value
  stats$marginal_coverage_rf <- (stats$marginal_lower_rf <= stats$marginal_true_value) &
    (stats$marginal_true_value <= stats$marginal_upper_rf)
  stats$marginal_coverage_glmm <- (stats$marginal_lower_glmm <= stats$marginal_true_value) &
    (stats$marginal_true_value <= stats$marginal_upper_glmm)
  
  if (!is.null(run_id)) {
    stats$run_id <- run_id
  }
  
  stats
}