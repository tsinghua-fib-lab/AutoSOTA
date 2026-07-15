# Defines analysis and helper functions for plotting scripts

source("src/common_paths.R")
source("src/constants.R")
ensure_output_dirs()

active_levels <- function(values) {
  # 'Gets set of used levels
  if (is.factor(values)) {
    return(levels(droplevels(values)))
  }
  unique(as.character(values))
}

load_benchmark_data <- function(filename) {
  #' Reads benchmark results file.
  
  data <- utils::read.csv(
    repo_path("benchmark_results", filename))
  data$LLM_id <- factor(data$LLM_id, levels = unique(data$LLM_id))
  data$item_id <- factor(data$item_id, levels = unique(data$item_id))
  data$score <- as.numeric(data$score)
  data$trial_id <- as.integer(data$trial_id)
  
  return(data)
}

apply_archived_factor_ordering <- function(data) {
  #' Restores archived alphabetical factor ordering for benchmark identifiers.

  stopifnot(
    is.data.frame(data),
    all(c("LLM_id", "item_id") %in% names(data))
  )

  data$LLM_id <- factor(as.character(data$LLM_id), levels = sort(unique(as.character(data$LLM_id))))
  data$item_id <- factor(as.character(data$item_id), levels = sort(unique(as.character(data$item_id))))
  data
}

fit_glmm <- function(data) {
  #' Fits GLMM under paper specification via maximum likelihood with adaptive Gaussian quadrature.
  stopifnot(
    is.data.frame(data),
    all(c("score", "LLM_id", "item_id") %in% names(data))
  )
  
  GLMMadaptive::mixed_model(
    fixed = score ~ LLM_id,
    random = ~1 | item_id,
    data = data,
    family = stats::binomial()
  )
}

fit_llm_fixed_effects_glm <- function(data) {
  #' Fits benchmark model with LLM fixed effects only.
  stopifnot(
    is.data.frame(data),
    all(c("score", "LLM_id") %in% names(data))
  )

  stats::glm(
    score ~ LLM_id,
    data = data,
    family = stats::binomial()
  )
}

fit_item_random_effects_only_glmm <- function(data) {
  #' Fits benchmark model with item random effects only.
  stopifnot(
    is.data.frame(data),
    all(c("score", "item_id") %in% names(data))
  )

  GLMMadaptive::mixed_model(
    fixed = score ~ 1,
    random = ~1 | item_id,
    data = data,
    family = stats::binomial()
  )
}

round_numeric_columns <- function(data, digits = 4) {
  numeric_cols <- vapply(data, is.numeric, logical(1))
  data[numeric_cols] <- lapply(data[numeric_cols], signif, digits = digits)
  data
}

clip_probabilities <- function(probabilities, epsilon = 1e-15) {
  stopifnot(is.numeric(probabilities), all(is.finite(probabilities)))
  pmin(pmax(probabilities, epsilon), 1 - epsilon)
}

compute_manual_binary_metrics <- function(observed, predicted) {
  stopifnot(
    is.numeric(observed),
    is.numeric(predicted),
    length(observed) == length(predicted),
    all(is.finite(observed)),
    all(is.finite(predicted))
  )

  predicted <- clip_probabilities(predicted)

  data.frame(
    RMSE = sqrt(mean((observed - predicted)^2)),
    Log_loss = mean(-(observed * log(predicted) + (1 - observed) * log(1 - predicted))),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

compute_manual_model_metrics <- function(model, observed = NULL) {
  if (is.null(observed)) {
    observed <- insight::get_response(model, verbose = FALSE)
  }
  if (is.data.frame(observed)) {
    stopifnot(ncol(observed) == 1)
    observed <- observed[[1]]
  }
  observed <- as.numeric(observed)
  predicted <- as.numeric(stats::fitted(model))
  compute_manual_binary_metrics(observed, predicted)
}

build_model_comparison_table <- function(models) {
  #' Creates a model-comparison summary for Table C.9.
  stopifnot(length(models) >= 1, !is.null(names(models)), all(nzchar(names(models))))

  round_numeric_columns(
    data.frame(
      Model = names(models),
      AIC = vapply(models, function(model) as.numeric(stats::AIC(model)), numeric(1)),
      BIC = vapply(models, function(model) as.numeric(stats::BIC(model)), numeric(1)),
      logLik = vapply(models, function(model) as.numeric(stats::logLik(model)), numeric(1)),
      check.names = FALSE,
      stringsAsFactors = FALSE
    ),
    digits = 5
  )
}

build_model_performance_table <- function(model, benchmark_label, observed = NULL) {
  #' Creates a C.10-style performance summary for a fitted benchmark model.
  metric_map <- c(
    "AIC" = "AIC",
    "AICc" = "AICc",
    "BIC" = "BIC",
    "R2 (cond)" = "R2_conditional",
    "R2 (marg)" = "R2_marginal",
    "ICC" = "ICC",
    "Sigma" = "Sigma"
  )

  performance_row <- as.data.frame(
    performance::model_performance(
      model,
      metrics = c("AIC", "AICc", "BIC", "R2", "ICC", "SIGMA"),
      verbose = FALSE
    )
  )
  stopifnot(all(unname(metric_map) %in% names(performance_row)))
  manual_metrics <- compute_manual_model_metrics(model, observed = observed)

  summary_row <- data.frame(
    Benchmark = benchmark_label,
    as.list(performance_row[1, unname(metric_map), drop = FALSE]),
    RMSE = manual_metrics$RMSE,
    `Log loss` = manual_metrics$Log_loss,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  names(summary_row)[2:(length(metric_map) + 1)] <- names(metric_map)

  round_numeric_columns(summary_row)
}

print_model_performance_table <- function(model, benchmark_label, observed = NULL) {
  #' Prints C.10-style performance metrics for a fitted benchmark model.
  cat(sprintf("\nTable C.10 metrics: %s\n", benchmark_label))
  print(build_model_performance_table(model, benchmark_label, observed = observed), row.names = FALSE)
}

diagnostic_output_path <- function(stem, suffix, figure_label = NULL) {
  if (is.null(figure_label)) {
    filename <- paste0(stem, "_", suffix, ".pdf")
  } else {
    filename <- paste0(figure_label, "_", stem, "_", suffix, ".pdf")
  }
  repo_path("figures", stem, filename)
}

write_glmm_diagnostics <- function(
  glmm,
  data,
  stem,
  include_dispersion = FALSE,
  include_residuals = FALSE,
  diagnostic_figure_label = NULL,
  dispersion_figure_label = NULL,
  llm_residuals_figure_label = NULL,
  item_residuals_figure_label = NULL
) {
  #' Performs goodness-of-fit checks for fitted GLMM. 

  sim_res <- DHARMa::simulateResiduals(fittedModel = glmm, n = 1000)

  grDevices::pdf(diagnostic_output_path(stem, "diagnostic", diagnostic_figure_label))
  plot(sim_res)
  grDevices::dev.off()

  if (include_dispersion) {
    grDevices::pdf(diagnostic_output_path(stem, "dispersion", dispersion_figure_label))
    DHARMa::testDispersion(sim_res)
    grDevices::dev.off()
  }

  if (include_residuals) {
    grDevices::pdf(diagnostic_output_path(stem, "LLM_residuals", llm_residuals_figure_label))
    DHARMa::plotResiduals(sim_res, data$LLM_id)
    grDevices::dev.off()

    grDevices::pdf(diagnostic_output_path(stem, "item_residuals", item_residuals_figure_label))
    DHARMa::plotResiduals(sim_res, data$item_id)
    grDevices::dev.off()
  }

  invisible(
    performance::model_performance(
      glmm,
      metrics = c("AIC", "AICc", "BIC", "R2", "ICC", "SIGMA"),
      verbose = FALSE
    )
  )
}

transform_effect_plot <- function(effect_df) {
  #' Transforms link-scale estimates to response scale
  
  effect_df <- as.data.frame(effect_df)
  for (col_name in intersect(c("pred", "low", "upp"), names(effect_df))) {
    effect_df[[col_name]] <- plogis(effect_df[[col_name]])
  }
  effect_df
}

t_critical <- function(sample_size, conf_level = 0.95) {
  stats::qt((1 + conf_level) / 2, pmax(sample_size - 1, 1))
}

get_glmm_estimates <- function(glmm, data) {
  #' Computes response-scale marginal estimates from a fitted GLMM.

  stopifnot(!is.null(data), "LLM_id" %in% names(data))
  llm_levels <- active_levels(data$LLM_id)
  new_data <- data.frame(LLM_id = factor(llm_levels, levels = llm_levels))

  GLMMadaptive::effectPlotData(glmm, new_data, marginal = TRUE) |>
    transform_effect_plot()
}

compute_glmm_estimates <- function(glmm, data) {
  #' Computes generalized accuracy (marginal) estimates from fitted GLMM via Monte Carlo integration.
  
  glmm_marginal <- get_glmm_estimates(glmm, data) |>
    dplyr::rename(prob = pred, lower = low, upper = upp) |>
    dplyr::mutate(source = "GLMM")

  order_levels <- glmm_marginal |>
    dplyr::arrange(prob) |>
    dplyr::pull(LLM_id)

  glmm_marginal |>
    dplyr::mutate(LLM_id = factor(LLM_id, levels = order_levels))
}

compute_conditional_accuracy <- function(data) {
  #' Computes benchmark accuracy (conditional) estimates via regression-free approach.
  
  data |>
    dplyr::group_by(LLM_id, item_id) |>
    dplyr::summarise(
      avg = mean(score),
      var_z = avg * (1 - avg),
      trials = dplyr::n(),
      .groups = "drop"
    ) |>
    dplyr::group_by(LLM_id) |>
    dplyr::summarise(
      n_items = dplyr::n(),
      prob = mean(avg),
      var_s = sum(var_z / (n_items^2 * (mean(trials) - 1))),
      se = sqrt(var_s),
      critical = t_critical(n_items),
      lower = prob - critical * se,
      upper = prob + critical * se,
      .groups = "drop"
    ) |>
    dplyr::select(-n_items, -se, -critical, -var_s) |>
    dplyr::mutate(source = "Regression-Free")
}

compute_marginal_accuracy <- function(data) {
  #' Computes generalized accuracy (marginal) estimates via regression-free approach.

  compute_marginal_accuracy_(data) |>
    dplyr::mutate(source = "Regression-Free")
}

compute_marginal_accuracy_ <- function(data) {

  data |>
    dplyr::group_by(LLM_id, item_id) |>
    dplyr::summarise(avg = mean(score), .groups = "drop") |>
    dplyr::group_by(LLM_id) |>
    dplyr::summarise(
      n = dplyr::n(),
      prob = mean(avg),
      se = stats::sd(avg) / sqrt(n),
      critical = t_critical(n),
      lower = prob - critical * se,
      upper = prob + critical * se,
      .groups = "drop"
    ) |>
    dplyr::select(-n, -se, -critical)
}

compute_simple_accuracy <- function(data) {
  #' Computes accuracy with CIs calculated via simple standard deviation across all items and trials.

  data |>
    dplyr::group_by(LLM_id) |>
    dplyr::summarise(
      n = dplyr::n(),
      prob = mean(score),
      se = stats::sd(score) / sqrt(n),
      critical = t_critical(n),
      lower = prob - critical * se,
      upper = prob + critical * se,
      .groups = "drop"
    ) |>
    dplyr::select(-n, -se, -critical) |>
    dplyr::mutate(source = "avg_single_epoch")
}

filter_LLMs <- function(data, context_label = NULL) {
  #' Filters data to exclude LLMs that achieve 100% or 0% score across all items and trials.
  
  valid_llms <- data |>
    dplyr::group_by(LLM_id) |>
    dplyr::summarise(avg_score = mean(score), .groups = "drop") |>
    dplyr::filter(avg_score > 0, avg_score < 1) |>
    dplyr::select(LLM_id)

  if (nrow(valid_llms) < 2) {
    if (is.null(context_label)) {
      context_label <- "This benchmark slice"
    }
    stop(sprintf("%s has fewer than two non-degenerate LLMs after filtering.", context_label))
  }

  filtered_data <- data |>
    dplyr::semi_join(valid_llms, by = "LLM_id")
  filtered_data$LLM_id <- factor(filtered_data$LLM_id, levels = active_levels(filtered_data$LLM_id))
  filtered_data$item_id <- factor(filtered_data$item_id, levels = active_levels(filtered_data$item_id))
  filtered_data
}

eff.ss <- function(glmm, n_total, k_cluster) {
  #' Estimates effective sample size from fitted GLMM ICC estimate.

  model.icc <- performance::icc(glmm)$ICC_adjusted
  design_effect <- 1 + (k_cluster - 1) * model.icc
  as.numeric(n_total / design_effect)
}

prepare_random_effects <- function(glmm, data, dataset) {
  #' Extracts difficulty estimates from fitted GLMM.
  
  random_effects <- as.data.frame(GLMMadaptive::ranef(glmm)) %>%
    tibble::rownames_to_column("item_id") %>%
    rename(intercept = `(Intercept)`)
  random_effects$intercept <- -1 * random_effects$intercept  # define such that higher value = harder difficulty
  random_effects <- merge(dataset, random_effects, by = "item_id")
  merge(
    random_effects,
    data %>% group_by(item_id) %>% summarize(score = mean(score), .groups = "drop"),
    by = "item_id"
  )
}

summarize_benchmark_slices <- function(data, slice_columns, output_columns = slice_columns, context_fn = NULL) {
  #' For a benchmark subdivision, fits GLMM and calculates variance statistics.

  slice_keys <- unique(data[, slice_columns, drop = FALSE])
  summary_rows <- vector("list", nrow(slice_keys))

  for (i in seq_len(nrow(slice_keys))) {
    key_values <- slice_keys[i, , drop = FALSE]
    slice_data <- data
    for (col_name in slice_columns) {
      slice_data <- slice_data[
        as.character(slice_data[[col_name]]) == as.character(key_values[[col_name]][[1]]),
        ,
        drop = FALSE
      ]
    }

    output_key <- as.data.frame(key_values, stringsAsFactors = FALSE)
    names(output_key) <- output_columns
    context_label <- if (is.null(context_fn)) {
      paste(sprintf("%s '%s'", output_columns, unlist(output_key, use.names = FALSE)), collapse = ", ")
    } else {
      context_fn(as.list(output_key))
    }

    filtered_data <- filter_LLMs(slice_data, context_label = context_label)
    glmm <- fit_glmm(filtered_data)
    glm <- stats::glm(score ~ LLM_id, family = stats::binomial(), data = filtered_data)

    k_cluster <- filtered_data |>
      dplyr::group_by(LLM_id, item_id) |>
      dplyr::summarise(trials = dplyr::n(), .groups = "drop") |>
      dplyr::pull(trials) |>
      mean()
    n_items <- length(unique(filtered_data$item_id))
    eff <- eff.ss(glmm, n_items, k_cluster)

    llm_levels <- active_levels(filtered_data$LLM_id)
    new_data <- data.frame(LLM_id = factor(llm_levels, levels = llm_levels))
    glmm_ci <- GLMMadaptive::effectPlotData(glmm, new_data, marginal = TRUE) |>
      transform_effect_plot() |>
      dplyr::transmute(LLM_id, ci_width_glmm = upp - low)

    preds <- stats::predict(glm, newdata = new_data, type = "link", se.fit = TRUE)
    critical <- t_critical(n_items)
    glm_ci <- data.frame(
      LLM_id = new_data$LLM_id,
      ci_width_glm = glm$family$linkinv(preds$fit + (critical * preds$se.fit)) -
        glm$family$linkinv(preds$fit - (critical * preds$se.fit))
    )

    joined_ci <- dplyr::inner_join(glmm_ci, glm_ci, by = "LLM_id")
    joined_ci$ratio <- (joined_ci$ci_width_glmm / joined_ci$ci_width_glm)^2

    metric_row <- data.frame(
      n = n_items,
      Score = mean(slice_data$score),
      ICC = as.numeric(performance::icc(glmm)$ICC_adjusted),
      EST = as.numeric(eff / n_items),
      Median = as.numeric(median(joined_ci$ratio)),
      t25 = as.numeric(stats::quantile(joined_ci$ratio, probs = 0.25)),
      t75 = as.numeric(stats::quantile(joined_ci$ratio, probs = 0.75)),
      check.names = FALSE
    )
    metric_row[["σ²"]] <- as.numeric(insight::get_variance(glmm)$var.random)
    metric_row <- metric_row[, c("n", "Score", "σ²", "ICC", "EST", "Median", "t25", "t75")]

    summary_rows[[i]] <- dplyr::bind_cols(output_key, metric_row)
  }

  dplyr::bind_rows(summary_rows)
}
