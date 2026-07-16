neural_get_index <- function(model_info,
                             party_label = NULL,
                             levels_field,
                             map_field) {
  missing_field <- if (identical(levels_field, "party_levels")) {
    "party_missing_label"
  } else if (identical(levels_field, "resp_party_levels")) {
    "resp_party_missing_label"
  } else {
    NULL
  }
  if (is.null(model_info) || is.null(model_info[[levels_field]])) {
    return(ai(0L))
  }
  missing_idx <- neural_missing_group_index(
    model_info[[levels_field]],
    model_info[[missing_field]] %||% neural_missing_group_label("candidate")
  )
  if (!is.null(model_info[[map_field]]) && !is.null(party_label)) {
    key <- as.character(party_label)
    if (key %in% names(model_info[[map_field]])) {
      return(ai(model_info[[map_field]][[key]]))
    }
  }
  if (is.null(party_label)) {
    return(ai(missing_idx))
  }
  idx <- match(as.character(party_label), model_info[[levels_field]]) - 1L
  if (is.na(idx)) ai(missing_idx) else ai(idx)
}

neural_get_party_index <- function(model_info, party_label = NULL){
  neural_get_index(model_info,
                   party_label = party_label,
                   levels_field = "party_levels",
                   map_field = "party_index_map")
}

neural_get_resp_party_index <- function(model_info, party_label = NULL){
  neural_get_index(model_info,
                   party_label = party_label,
                   levels_field = "resp_party_levels",
                   map_field = "resp_party_index_map")
}

neural_pairwise_context_mode <- function(model_info = NULL) {
  mode <- tolower(as.character(model_info$pairwise_context_mode %||% "stage_free"))
  if (!mode %in% c("stage_free", "stage_aware")) {
    return("stage_free")
  }
  mode
}

neural_context_flag <- function(model_info, field, default = FALSE) {
  value <- tryCatch(model_info[[field]], error = function(e) NULL)
  if (is.null(value)) {
    return(isTRUE(default))
  }
  isTRUE(value)
}

neural_candidate_group_context_enabled <- function(model_info = NULL) {
  neural_context_flag(model_info, "has_candidate_group_context", default = FALSE)
}

neural_respondent_group_context_enabled <- function(model_info = NULL) {
  neural_context_flag(model_info, "has_respondent_group_context", default = FALSE)
}

neural_relation_context_enabled <- function(model_info = NULL) {
  neural_context_flag(model_info, "has_relation_token_context", default = FALSE)
}

neural_stage_context_enabled <- function(model_info = NULL) {
  if (!identical(neural_pairwise_context_mode(model_info), "stage_aware")) {
    return(FALSE)
  }
  neural_context_flag(model_info, "has_stage_context", default = TRUE)
}

neural_matchup_context_enabled <- function(model_info = NULL) {
  if (!neural_stage_context_enabled(model_info)) {
    return(FALSE)
  }
  neural_context_flag(model_info, "has_matchup_context", default = TRUE)
}

neural_place_context_enabled <- function(model_info = NULL) {
  isTRUE(neural_context_flag(model_info, "place_context_enabled", default = FALSE)) ||
    isTRUE(neural_context_flag(model_info, "has_place_context", default = FALSE))
}

neural_place_feature_names <- function() {
  frequencies <- c(1L, 2L, 4L, 8L)
  c(
    "sphere_x",
    "sphere_y",
    "sphere_z",
    unlist(lapply(frequencies, function(freq) {
      paste0(
        c("sin_x_", "cos_x_", "sin_y_", "cos_y_", "sin_z_", "cos_z_"),
        freq
      )
    }), use.names = FALSE),
    "missing_country"
  )
}

neural_encode_place_context <- function(latitude, longitude, present = TRUE) {
  feature_names <- neural_place_feature_names()
  out <- stats::setNames(rep(0, length(feature_names)), feature_names)
  lat <- suppressWarnings(as.numeric(latitude))
  lon <- suppressWarnings(as.numeric(longitude))
  if (!isTRUE(present) || length(lat) != 1L || length(lon) != 1L ||
      is.na(lat) || is.na(lon) || !is.finite(lat) || !is.finite(lon)) {
    out[["missing_country"]] <- 1
    return(out)
  }
  lat_rad <- lat * pi / 180
  lon_rad <- lon * pi / 180
  sphere <- c(
    sphere_x = cos(lat_rad) * cos(lon_rad),
    sphere_y = cos(lat_rad) * sin(lon_rad),
    sphere_z = sin(lat_rad)
  )
  out[names(sphere)] <- sphere
  for (freq in c(1L, 2L, 4L, 8L)) {
    angle <- freq * pi * sphere
    out[[paste0("sin_x_", freq)]] <- sin(angle[["sphere_x"]])
    out[[paste0("cos_x_", freq)]] <- cos(angle[["sphere_x"]])
    out[[paste0("sin_y_", freq)]] <- sin(angle[["sphere_y"]])
    out[[paste0("cos_y_", freq)]] <- cos(angle[["sphere_y"]])
    out[[paste0("sin_z_", freq)]] <- sin(angle[["sphere_z"]])
    out[[paste0("cos_z_", freq)]] <- cos(angle[["sphere_z"]])
  }
  out
}

neural_default_place_context_matrix <- function(key = "__missing_country__") {
  matrix(
    neural_encode_place_context(NA_real_, NA_real_, present = FALSE),
    nrow = 1L,
    dimnames = list(as.character(key %||% "__missing_country__"), neural_place_feature_names())
  )
}

neural_time_context_enabled <- function(model_info = NULL) {
  isTRUE(neural_context_flag(model_info, "time_context_enabled", default = FALSE)) ||
    isTRUE(neural_context_flag(model_info, "has_time_context", default = FALSE))
}

neural_time_feature_names <- function() {
  periods <- c(2L, 5L, 10L, 20L, 50L, 100L)
  c(
    "linear_year_2000_25",
    unlist(lapply(periods, function(period) {
      paste0(c("sin_period_", "cos_period_"), period)
    }), use.names = FALSE),
    "missing_year"
  )
}

neural_encode_time_context <- function(year, present = TRUE) {
  feature_names <- neural_time_feature_names()
  out <- stats::setNames(rep(0, length(feature_names)), feature_names)
  year_num <- suppressWarnings(as.numeric(year))
  if (!isTRUE(present) || length(year_num) != 1L ||
      is.na(year_num) || !is.finite(year_num)) {
    out[["missing_year"]] <- 1
    return(out)
  }
  centered <- year_num - 2000
  out[["linear_year_2000_25"]] <- centered / 25
  for (period in c(2L, 5L, 10L, 20L, 50L, 100L)) {
    angle <- 2 * pi * centered / period
    out[[paste0("sin_period_", period)]] <- sin(angle)
    out[[paste0("cos_period_", period)]] <- cos(angle)
  }
  out
}

neural_default_time_context_matrix <- function(key = "__missing_year__") {
  matrix(
    neural_encode_time_context(NA_real_, present = FALSE),
    nrow = 1L,
    dimnames = list(as.character(key %||% "__missing_year__"), neural_time_feature_names())
  )
}

neural_missing_group_label <- function(kind = c("candidate", "respondent")) {
  kind <- match.arg(kind)
  if (identical(kind, "candidate")) {
    "__fm_missing_candidate__"
  } else {
    "__fm_missing_respondent__"
  }
}

neural_prepare_group_levels <- function(values,
                                        override = NULL,
                                        missing_label) {
  if (!is.null(override)) {
    levels <- as.character(override)
  } else {
    values_chr <- as.character(values %||% character(0))
    levels <- sort(unique(values_chr[!is.na(values_chr) & nzchar(values_chr)]))
  }
  levels <- levels[!is.na(levels) & nzchar(levels)]
  if (!missing_label %in% levels) {
    levels <- c(levels, missing_label)
  }
  levels
}

neural_missing_group_index <- function(levels, missing_label) {
  idx <- match(as.character(missing_label), as.character(levels))
  if (is.na(idx)) {
    return(0L)
  }
  as.integer(idx - 1L)
}

neural_coerce_group_index_base <- function(values,
                                           n_rows,
                                           levels,
                                           missing_label) {
  levels <- as.character(levels %||% character(0))
  missing_idx <- neural_missing_group_index(levels, missing_label)
  if (length(levels) < 1L) {
    return(rep(0L, as.integer(n_rows)))
  }
  if (is.null(values)) {
    return(rep(missing_idx, as.integer(n_rows)))
  }
  if (is.numeric(values)) {
    vals_num <- as.numeric(values)
    finite_vals <- vals_num[is.finite(vals_num)]
    if (length(finite_vals) > 0L && any(abs(finite_vals - round(finite_vals)) > 1e-8)) {
      stop(
        "Numeric group values must be whole-number codes or match the group labels; ",
        "received fractional values. Supply character labels instead.",
        call. = FALSE
      )
    }
    # Numeric labels first: if every observed value literally matches a level
    # label (e.g., levels "1"/"2"), treat values as labels, not positions.
    vals_chr <- as.character(values)
    observed <- vals_chr[!is.na(vals_chr) & nzchar(vals_chr)]
    if (length(observed) > 0L && all(observed %in% levels)) {
      idx <- match(vals_chr, levels) - 1L
      idx[is.na(idx)] <- missing_idx
      return(as.integer(idx))
    }
    # Positional codes: infer the base only from definitive evidence. A value
    # of 0 can only be 0-based; a value of length(levels) can only be 1-based.
    # The old heuristic ("shift iff some value >= length(levels)") silently
    # misread 1-based codes whenever the batch happened to omit the top level,
    # shifting every group by one -- never guess silently.
    idx <- as.integer(values)
    has_zero <- any(idx == 0L, na.rm = TRUE)
    has_top <- any(idx == length(levels), na.rm = TRUE)
    if (has_zero && has_top) {
      stop(
        sprintf(paste0(
          "Numeric group codes contain both 0 and %d, which no single ",
          "0-based or 1-based indexing of the %d group levels can produce. ",
          "Supply character labels instead."
        ), length(levels), length(levels)),
        call. = FALSE
      )
    }
    if (has_top) {
      idx <- idx - 1L
    } else if (!has_zero && any(!is.na(idx))) {
      warning(
        paste0(
          "Numeric group codes are ambiguous (no 0 and no top-level code ",
          "observed); interpreting them as 0-based indices into the group ",
          "levels. Pass character labels to remove the ambiguity."
        ),
        call. = FALSE
      )
    }
    idx[is.na(idx) | idx < 0L | idx >= length(levels)] <- missing_idx
    return(idx)
  }
  values_chr <- as.character(values)
  values_chr[is.na(values_chr) | !nzchar(values_chr)] <- missing_label
  idx <- match(values_chr, levels) - 1L
  idx[is.na(idx)] <- missing_idx
  as.integer(idx)
}

neural_has_shape <- function(x) {
  if (is.null(x)) {
    return(FALSE)
  }
  has_shape <- tryCatch(!is.null(x$shape), error = function(e) FALSE)
  if (isTRUE(has_shape)) {
    return(TRUE)
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(FALSE)
  }
  tryCatch(reticulate::is_py_object(x), error = function(e) FALSE)
}

neural_as_jnp_array <- function(x, dtype = NULL) {
  if (is.null(x)) {
    return(NULL)
  }
  has_array_shape <- tryCatch(!is.null(x$shape), error = function(e) FALSE)
  arr <- if (isTRUE(has_array_shape)) {
    x
  } else {
    strenv$jnp$array(x)
  }
  if (!is.null(dtype)) {
    arr <- strenv$jnp$astype(arr, dtype)
  }
  arr
}

neural_as_jnp_vector <- function(x, dtype = NULL) {
  arr <- neural_as_jnp_array(x)
  if (is.null(arr)) {
    return(NULL)
  }
  arr <- strenv$jnp$atleast_1d(arr)
  if (!is.null(dtype)) {
    arr <- strenv$jnp$astype(arr, dtype)
  }
  arr
}

neural_as_jnp_matrix <- function(x, dtype = NULL) {
  arr <- neural_as_jnp_array(x)
  if (is.null(arr)) {
    return(NULL)
  }
  arr <- strenv$jnp$atleast_2d(arr)
  if (!is.null(dtype)) {
    arr <- strenv$jnp$astype(arr, dtype)
  }
  arr
}

neural_batch_vector_jnp <- function(x, dtype = NULL) {
  neural_as_jnp_vector(x, dtype = dtype)
}

neural_batch_matrix_jnp <- function(x, dtype = NULL) {
  neural_as_jnp_matrix(x, dtype = dtype)
}

neural_format_svi_elbo_plot_title <- function(svi_loss_curve,
                                              n_tail = 20L,
                                              digits = 4L) {
  title_base <- "SVI ELBO Loss"
  if (is.null(svi_loss_curve) || length(svi_loss_curve) < 1L) {
    return(title_base)
  }

  finite_losses <- as.numeric(svi_loss_curve)
  finite_losses <- finite_losses[is.finite(finite_losses)]
  if (length(finite_losses) < 1L) {
    return(title_base)
  }

  n_tail <- as.integer(n_tail)
  if (length(n_tail) != 1L || is.na(n_tail) || n_tail < 1L) {
    n_tail <- 20L
  }
  digits <- as.integer(digits)
  if (length(digits) != 1L || is.na(digits) || digits < 0L) {
    digits <- 4L
  }

  tail_mean <- mean(tail(finite_losses, n_tail))
  if (!is.finite(tail_mean)) {
    return(title_base)
  }

  sprintf("%s [%.*f]", title_base, digits, tail_mean)
}

neural_diagnostic_numeric <- function(x) {
  if (is.null(x)) {
    return(numeric(0))
  }
  out <- tryCatch(
    as.numeric(cs2step_neural_to_r_array(x)),
    error = function(e) {
      tryCatch(as.numeric(x), error = function(e2) numeric(0))
    }
  )
  out
}

neural_build_parameter_diagnostics <- function(params) {
  empty <- list(
    parameter_status = "unavailable",
    n_parameters = NA_integer_,
    n_finite = NA_integer_,
    n_nonfinite = NA_integer_,
    global_l2_norm = NA_real_,
    global_max_abs = NA_real_,
    parameter_summaries = data.frame(),
    global_histogram = list(breaks = numeric(0), counts = integer(0))
  )
  if (is.null(params) || !is.list(params) || length(params) < 1L) {
    return(empty)
  }

  param_names <- names(params)
  if (is.null(param_names)) {
    param_names <- paste0("param_", seq_along(params))
  }
  values <- lapply(params, neural_diagnostic_numeric)
  summaries <- lapply(seq_along(values), function(i) {
    v <- values[[i]]
    finite_v <- v[is.finite(v)]
    data.frame(
      name = param_names[[i]],
      size = length(v),
      n_finite = length(finite_v),
      n_nonfinite = sum(!is.finite(v)),
      min = if (length(finite_v)) min(finite_v) else NA_real_,
      max = if (length(finite_v)) max(finite_v) else NA_real_,
      mean = if (length(finite_v)) mean(finite_v) else NA_real_,
      sd = if (length(finite_v) > 1L) stats::sd(finite_v) else NA_real_,
      l2_norm = if (length(finite_v)) sqrt(sum(finite_v ^ 2)) else NA_real_,
      max_abs = if (length(finite_v)) max(abs(finite_v)) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  summary_df <- do.call(rbind, summaries)
  all_values <- unlist(values, use.names = FALSE)
  finite_values <- all_values[is.finite(all_values)]
  hist_info <- if (length(finite_values) > 1L) {
    hist_out <- tryCatch(
      graphics::hist(finite_values, breaks = "FD", plot = FALSE),
      error = function(e) NULL
    )
    if (is.null(hist_out) || length(hist_out$counts) > 24L) {
      hist_out <- tryCatch(
        graphics::hist(finite_values, breaks = 12L, plot = FALSE),
        error = function(e) NULL
      )
    }
    if (is.null(hist_out)) {
      list(breaks = numeric(0), counts = integer(0))
    } else {
      list(
        breaks = as.numeric(hist_out$breaks),
        counts = as.integer(hist_out$counts)
      )
    }
  } else {
    list(breaks = finite_values, counts = as.integer(length(finite_values)))
  }

  list(
    parameter_status = "ok",
    n_parameters = length(all_values),
    n_finite = length(finite_values),
    n_nonfinite = sum(!is.finite(all_values)),
    global_l2_norm = if (length(finite_values)) sqrt(sum(finite_values ^ 2)) else NA_real_,
    global_max_abs = if (length(finite_values)) max(abs(finite_values)) else NA_real_,
    parameter_summaries = summary_df,
    global_histogram = hist_info
  )
}

neural_build_gradient_diagnostics <- function(status = "ok",
                                              source = NA_character_,
                                              notes = character(0)) {
  list(
    gradient_status = status,
    checkpoint_steps = integer(0),
    checkpoint_global_l2_norm = numeric(0),
    checkpoint_global_rms = numeric(0),
    checkpoint_global_max_abs = numeric(0),
    checkpoint_n_nonfinite = integer(0),
    checkpoint_n_elements = integer(0),
    global_l2_norm = NA_real_,
    global_rms = NA_real_,
    global_max_abs = NA_real_,
    source = source,
    notes = notes
  )
}

neural_append_gradient_checkpoint <- function(diagnostics, step, checkpoint) {
  if (is.null(diagnostics) || !is.list(diagnostics)) {
    diagnostics <- neural_build_gradient_diagnostics(
      status = "ok",
      source = "checkpoint_value_and_grad"
    )
  }

  step <- suppressWarnings(as.integer(step))
  if (length(step) != 1L || is.na(step)) {
    step <- NA_integer_
  }
  grad_l2 <- suppressWarnings(as.numeric(checkpoint$grad_l2 %||% NA_real_))
  grad_rms <- suppressWarnings(as.numeric(checkpoint$grad_rms %||% NA_real_))
  grad_max_abs <- suppressWarnings(as.numeric(checkpoint$grad_max_abs %||% NA_real_))
  grad_n_nonfinite <- suppressWarnings(as.integer(checkpoint$grad_n_nonfinite %||% NA_integer_))
  grad_n_elements <- suppressWarnings(as.integer(checkpoint$grad_n_elements %||% NA_integer_))

  diagnostics$checkpoint_steps <- c(
    as.integer(diagnostics$checkpoint_steps %||% integer(0)),
    step
  )
  diagnostics$checkpoint_global_l2_norm <- c(
    as.numeric(diagnostics$checkpoint_global_l2_norm %||% numeric(0)),
    grad_l2
  )
  diagnostics$checkpoint_global_rms <- c(
    as.numeric(diagnostics$checkpoint_global_rms %||% numeric(0)),
    grad_rms
  )
  diagnostics$checkpoint_global_max_abs <- c(
    as.numeric(diagnostics$checkpoint_global_max_abs %||% numeric(0)),
    grad_max_abs
  )
  diagnostics$checkpoint_n_nonfinite <- c(
    as.integer(diagnostics$checkpoint_n_nonfinite %||% integer(0)),
    grad_n_nonfinite
  )
  diagnostics$checkpoint_n_elements <- c(
    as.integer(diagnostics$checkpoint_n_elements %||% integer(0)),
    grad_n_elements
  )
  diagnostics$global_l2_norm <- grad_l2
  diagnostics$global_rms <- grad_rms
  diagnostics$global_max_abs <- grad_max_abs
  diagnostics$source <- "checkpoint_value_and_grad"
  if (!identical(diagnostics$gradient_status, "failed")) {
    diagnostics$gradient_status <- "ok"
  }
  diagnostics
}

neural_mark_gradient_diagnostics_failed <- function(diagnostics, error_message = NULL) {
  if (is.null(diagnostics) || !is.list(diagnostics)) {
    diagnostics <- neural_build_gradient_diagnostics(
      status = "failed",
      source = "checkpoint_value_and_grad"
    )
  }
  diagnostics$gradient_status <- "failed"
  diagnostics$source <- "checkpoint_value_and_grad"
  if (!is.null(error_message) && nzchar(error_message)) {
    note <- sprintf("Checkpoint gradient diagnostics failed: %s", error_message)
    diagnostics$notes <- unique(c(as.character(diagnostics$notes %||% character(0)), note))
  }
  diagnostics
}

neural_svi_gradient_helper <- local({
  helper <- NULL
  function() {
    if (!is.null(helper)) {
      return(helper)
    }
    if (!requireNamespace("reticulate", quietly = TRUE)) {
      stop("reticulate is required for SVI gradient diagnostics.", call. = FALSE)
    }
    strategize_register_jax_svi_helpers()
    helper <<- strenv$jax_svi_gradient_diagnostics
    if (is.null(helper)) {
      stop("JAX SVI gradient diagnostic helper is unavailable.", call. = FALSE)
    }
    helper
  }
})

neural_py_kwargs_from_list <- function(args) {
  kwargs <- reticulate::dict()
  arg_names <- names(args)
  if (is.null(arg_names) || length(arg_names) != length(args)) {
    arg_names <- rep("", length(args))
  }
  for (i in seq_along(args)) {
    if (!nzchar(arg_names[[i]])) {
      next
    }
    kwargs[[arg_names[[i]]]] <- if (is.null(args[[i]])) {
      reticulate::py_none()
    } else {
      args[[i]]
    }
  }
  kwargs
}

neural_compute_svi_gradient_checkpoint <- function(svi, svi_state, model_args) {
  helper <- neural_svi_gradient_helper()
  kwargs <- neural_py_kwargs_from_list(model_args %||% list())
  helper(svi, svi_state, kwargs)
}

neural_extract_lr_trace <- function(lr_schedule, steps_completed, fallback_lr = NA_real_) {
  steps_completed <- suppressWarnings(as.integer(steps_completed))
  if (length(steps_completed) != 1L || is.na(steps_completed) || steps_completed < 1L) {
    return(list(lr_trace = numeric(0), lr_trace_status = "ok"))
  }
  fallback_lr <- suppressWarnings(as.numeric(fallback_lr))
  if (length(fallback_lr) != 1L || !is.finite(fallback_lr)) {
    fallback_lr <- NA_real_
  }
  if (is.null(lr_schedule) || is.numeric(lr_schedule)) {
    lr_value <- suppressWarnings(as.numeric(lr_schedule %||% fallback_lr))
    if (length(lr_value) != 1L || !is.finite(lr_value)) {
      return(list(
        lr_trace = rep(NA_real_, steps_completed),
        lr_trace_status = "unavailable"
      ))
    }
    return(list(lr_trace = rep(lr_value, steps_completed), lr_trace_status = "ok"))
  }

  # Evaluate the whole schedule in ONE vmapped device call: the per-step loop
  # cost one reticulate round-trip per completed step (40k+ at teardown of a
  # long run). Fall back to the loop for schedules vmap cannot trace.
  trace <- tryCatch({
    steps_arr <- strenv$jnp$arange(ai(steps_completed))
    values <- strenv$jax$vmap(lr_schedule)(steps_arr)
    as.numeric(strenv$np$array(values))
  }, error = function(e) NULL)
  if (is.null(trace) || length(trace) != steps_completed) {
    trace <- tryCatch(
      vapply(seq_len(steps_completed) - 1L, function(step) {
        value <- lr_schedule(ai(step))
        as.numeric(strenv$np$array(value))
      }, numeric(1)),
      error = function(e) NULL
    )
  }
  if (is.null(trace) || length(trace) != steps_completed) {
    return(list(
      lr_trace = rep(NA_real_, steps_completed),
      lr_trace_status = "unavailable"
    ))
  }
  trace[!is.finite(trace)] <- NA_real_
  list(lr_trace = as.numeric(trace), lr_trace_status = "ok")
}

neural_detect_plateau <- function(loss_curve) {
  loss_curve <- as.numeric(loss_curve %||% numeric(0))
  finite_loss <- loss_curve[is.finite(loss_curve)]
  if (length(finite_loss) < 20L) {
    return(FALSE)
  }
  window <- max(5L, floor(length(finite_loss) * 0.2))
  if (length(finite_loss) < (2L * window)) {
    return(FALSE)
  }
  prev <- finite_loss[(length(finite_loss) - 2L * window + 1L):(length(finite_loss) - window)]
  last <- tail(finite_loss, window)
  prev_mean <- mean(prev)
  last_mean <- mean(last)
  if (!is.finite(prev_mean) || !is.finite(last_mean)) {
    return(FALSE)
  }
  abs(last_mean - prev_mean) <= 1e-4 * max(1, abs(prev_mean))
}

neural_metric_value <- function(metrics, names) {
  if (is.null(metrics) || !is.list(metrics)) {
    return(NA_real_)
  }
  for (name in names) {
    value <- suppressWarnings(as.numeric(metrics[[name]] %||% NA_real_))
    if (length(value) == 1L && is.finite(value)) {
      return(value)
    }
  }
  NA_real_
}

neural_detect_overfit <- function(fit_metrics) {
  if (is.null(fit_metrics) || is.null(fit_metrics$in_sample_metrics)) {
    return(FALSE)
  }
  in_sample <- fit_metrics$in_sample_metrics
  oos_loss <- neural_metric_value(fit_metrics, c("log_loss", "nll"))
  in_loss <- neural_metric_value(in_sample, c("log_loss", "nll"))
  if (is.finite(oos_loss) && is.finite(in_loss) && (oos_loss - in_loss) >= 0.05) {
    return(TRUE)
  }
  oos_auc <- neural_metric_value(fit_metrics, "auc")
  in_auc <- neural_metric_value(in_sample, "auc")
  if (is.finite(oos_auc) && is.finite(in_auc) && (in_auc - oos_auc) >= 0.05) {
    return(TRUE)
  }
  FALSE
}

neural_build_convergence_diagnostics <- function(parameter_diagnostics = NULL,
                                                 svi_loss_curve = NULL,
                                                 early_stopping = NULL,
                                                 fit_metrics = NULL,
                                                 steps_completed = NULL,
                                                 steps_planned = NULL,
                                                 use_svi = TRUE) {
  steps_completed <- suppressWarnings(as.integer(steps_completed %||% NA_integer_))
  steps_planned <- suppressWarnings(as.integer(steps_planned %||% NA_integer_))
  losses <- as.numeric(svi_loss_curve %||% numeric(0))
  has_losses <- length(losses) > 0L
  finite_losses <- losses[is.finite(losses)]
  loss_nonfinite <- if (has_losses) any(!is.finite(losses)) else NA
  all_losses_nonfinite <- isTRUE(has_losses && length(finite_losses) < 1L)
  final_loss <- if (length(finite_losses)) tail(finite_losses, 1L) else NA_real_
  param_nonfinite <- suppressWarnings(as.integer(parameter_diagnostics$n_nonfinite %||% NA_integer_))
  params_failed <- !is.na(param_nonfinite) && param_nonfinite > 0L
  es_reason <- as.character(early_stopping$reason %||% NA_character_)
  skipped_updates <- suppressWarnings(as.integer(
    early_stopping$compact_update_nonfinite_count %||%
      early_stopping$nonfinite_update_count %||%
      0L
  ))
  if (length(skipped_updates) != 1L || is.na(skipped_updates) || skipped_updates < 0L) {
    skipped_updates <- 0L
  }
  validation_failed <- es_reason %in% c("validation_error", "metric_failed")
  update_failed <- es_reason %in% c("update_failed")
  n_failed_folds <- suppressWarnings(as.integer(fit_metrics$n_failed_folds %||% NA_integer_))
  n_folds <- suppressWarnings(as.integer(fit_metrics$n_folds %||% NA_integer_))
  n_eval_success <- suppressWarnings(as.integer(fit_metrics$n_eval_success %||% NA_integer_))
  n_eval_failed <- suppressWarnings(as.integer(fit_metrics$n_eval_failed %||% NA_integer_))
  all_folds_failed <- (
    !is.na(n_failed_folds) && !is.na(n_folds) && n_folds > 0L && n_failed_folds >= n_folds
  ) || (
    !is.na(n_eval_success) && !is.na(n_eval_failed) && n_eval_success < 1L && n_eval_failed > 0L
  )
  any_failed_folds <- !is.na(n_failed_folds) && n_failed_folds > 0L
  plateaued <- neural_detect_plateau(losses)
  overfit_suspected <- neural_detect_overfit(fit_metrics)
  best_metric <- suppressWarnings(as.numeric(early_stopping$best_metric %||% NA_real_))
  final_metric <- suppressWarnings(as.numeric(early_stopping$final_metric %||% NA_real_))
  if (!is.finite(final_metric)) {
    final_metric <- neural_metric_value(fit_metrics, c("log_loss", "nll", "rmse", "mae", "brier"))
  }

  failed_reason <- NA_character_
  if (isTRUE(params_failed)) {
    failed_reason <- "nonfinite_parameters"
  } else if (isTRUE(all_losses_nonfinite)) {
    failed_reason <- "loss_nonfinite_all"
  } else if (isTRUE(update_failed)) {
    failed_reason <- "backend_update_failed"
  } else if (isTRUE(validation_failed)) {
    failed_reason <- paste0("validation_", es_reason)
  } else if (isTRUE(all_folds_failed)) {
    failed_reason <- "prediction_failed_all_folds"
  }

  enough_for_converged <- isTRUE(use_svi) &&
    !is.na(steps_completed) &&
    !is.na(steps_planned) &&
    (
      is.finite(best_metric) ||
        (steps_completed >= steps_planned && is.finite(final_loss))
    )
  verdict <- "unknown"
  converged <- NA
  if (!is.na(failed_reason)) {
    verdict <- "failed"
    converged <- FALSE
  } else if (isTRUE(loss_nonfinite) || isTRUE(any_failed_folds) ||
             isTRUE(plateaued) || isTRUE(overfit_suspected)) {
    verdict <- "warning"
    converged <- FALSE
  } else if (isTRUE(enough_for_converged) && !isTRUE(params_failed) &&
             !isTRUE(loss_nonfinite) && !isTRUE(any_failed_folds)) {
    verdict <- "converged"
    converged <- TRUE
  }

  notes <- character(0)
  if (isTRUE(loss_nonfinite)) notes <- c(notes, "Some SVI losses were nonfinite.")
  if (skipped_updates > 0L) {
    notes <- c(
      notes,
      sprintf("%d compact SVI update attempt(s) were skipped after non-finite losses.", skipped_updates)
    )
  }
  if (isTRUE(any_failed_folds)) notes <- c(notes, "One or more OOS folds failed.")
  if (isTRUE(plateaued)) notes <- c(notes, "SVI loss plateau detected.")
  if (isTRUE(overfit_suspected)) notes <- c(notes, "OOS metrics are worse than in-sample metrics.")
  if (!is.na(failed_reason)) notes <- c(notes, sprintf("Failure reason: %s.", failed_reason))

  list(
    verdict = verdict,
    converged = converged,
    failed_reason = failed_reason,
    loss_nonfinite = if (is.na(loss_nonfinite)) NA else isTRUE(loss_nonfinite),
    plateaued = isTRUE(plateaued),
    overfit_suspected = isTRUE(overfit_suspected),
    steps_completed = if (is.na(steps_completed)) NA_integer_ else steps_completed,
    steps_planned = if (is.na(steps_planned)) NA_integer_ else steps_planned,
    best_metric = if (length(best_metric) == 1L && is.finite(best_metric)) best_metric else NA_real_,
    final_metric = if (length(final_metric) == 1L && is.finite(final_metric)) final_metric else NA_real_,
    final_loss = if (is.finite(final_loss)) final_loss else NA_real_,
    notes = notes
  )
}

neural_resolve_early_stopping_validation_target_n <- function(n_eval,
                                                              n_validation_available,
                                                              validation_frac = 0.05,
                                                              validation_max_n = 2048L,
                                                              validation_min_n = 32L) {
  n_eval <- suppressWarnings(as.integer(n_eval))
  n_validation_available <- suppressWarnings(as.integer(n_validation_available))
  validation_frac <- as.numeric(validation_frac)
  validation_min_n <- suppressWarnings(as.integer(validation_min_n))
  validation_max_n_use <- if (is.null(validation_max_n)) {
    NULL
  } else {
    suppressWarnings(as.integer(validation_max_n))
  }

  if (length(n_eval) != 1L || is.na(n_eval) || n_eval < 1L ||
      length(n_validation_available) != 1L || is.na(n_validation_available) ||
      n_validation_available < 1L) {
    return(NA_integer_)
  }
  if (length(validation_frac) != 1L || is.na(validation_frac) ||
      !is.finite(validation_frac) || validation_frac <= 0) {
    validation_frac <- 0.05
  }
  validation_frac <- min(validation_frac, 1)
  if (length(validation_min_n) != 1L || is.na(validation_min_n) ||
      validation_min_n < 1L) {
    validation_min_n <- 1L
  }

  target_n <- as.integer(max(1L, ceiling(n_eval * validation_frac)))
  if (n_validation_available >= validation_min_n) {
    target_n <- max(target_n, validation_min_n)
  }
  target_n <- min(target_n, n_validation_available)

  if (!is.null(validation_max_n_use)) {
    if (length(validation_max_n_use) != 1L || is.na(validation_max_n_use) ||
        validation_max_n_use < 1L) {
      validation_max_n_use <- 2048L
    }
    target_n <- min(target_n, validation_max_n_use)
  }

  as.integer(max(1L, target_n))
}

neural_resolve_early_stopping_validation_batch_size <- function(validation_target_n,
                                                                validation_batch_size = 128L) {
  validation_target_n <- suppressWarnings(as.integer(validation_target_n))
  if (length(validation_target_n) != 1L ||
      is.na(validation_target_n) ||
      validation_target_n < 1L) {
    return(NA_integer_)
  }

  if (is.null(validation_batch_size)) {
    return(as.integer(validation_target_n))
  }

  validation_batch_size <- suppressWarnings(as.integer(validation_batch_size))
  if (length(validation_batch_size) != 1L ||
      is.na(validation_batch_size) ||
      validation_batch_size < 1L) {
    validation_batch_size <- 128L
  }

  as.integer(max(1L, min(validation_target_n, validation_batch_size)))
}

neural_resolve_compact_update_scan <- function(compact_training,
                                               compact_update_chunk_size,
                                               compact_update_scan = NULL) {
  chunk_size <- suppressWarnings(as.integer(compact_update_chunk_size))
  if (length(chunk_size) != 1L || is.na(chunk_size) || chunk_size < 1L) {
    chunk_size <- 1L
  }
  if (!isTRUE(compact_training) || chunk_size <= 1L) {
    return("fallback")
  }
  if (is.null(compact_update_scan)) {
    return("required")
  }
  mode <- tolower(as.character(compact_update_scan))
  if (length(mode) != 1L || is.na(mode) || !mode %in% c("required", "fallback")) {
    stop(
      "'neural_mcmc_control$compact_update_scan' must be 'required' or 'fallback'.",
      call. = FALSE
    )
  }
  mode
}

neural_compact_effective_scan_steps <- function(svi_steps,
                                                completed_steps = 0L,
                                                chunk_size,
                                                scan_available = TRUE) {
  requested <- suppressWarnings(as.integer(svi_steps))
  completed <- suppressWarnings(as.integer(completed_steps %||% 0L))
  chunk_size <- suppressWarnings(as.integer(chunk_size))
  if (length(requested) != 1L || is.na(requested) || requested < 1L) {
    return(0L)
  }
  if (!isTRUE(scan_available) ||
      length(chunk_size) != 1L ||
      is.na(chunk_size) ||
      chunk_size <= 1L) {
    return(as.integer(requested))
  }
  if (length(completed) != 1L || is.na(completed) || completed < 0L) {
    completed <- 0L
  }
  if (completed >= requested) {
    return(as.integer(requested))
  }
  remaining <- as.integer(requested) - as.integer(completed)
  tail_steps <- remaining %% as.integer(chunk_size)
  if (tail_steps == 0L) {
    return(as.integer(requested))
  }
  as.integer(requested + (as.integer(chunk_size) - tail_steps))
}

neural_resolve_positive_int <- function(value, fallback = 1L) {
  out <- suppressWarnings(as.integer(value))
  if (length(out) != 1L || is.na(out) || !is.finite(out) || out < 1L) {
    out <- suppressWarnings(as.integer(fallback))
  }
  if (length(out) != 1L || is.na(out) || !is.finite(out) || out < 1L) {
    out <- 1L
  }
  as.integer(out)
}

neural_compact_chunk_boundary_checks <- function(svi_steps,
                                                 n_checks,
                                                 chunk_size) {
  svi_steps <- suppressWarnings(as.integer(svi_steps))
  n_checks <- suppressWarnings(as.integer(n_checks))
  chunk_size <- suppressWarnings(as.integer(chunk_size))
  if (length(svi_steps) != 1L || is.na(svi_steps) || svi_steps < 1L) {
    return(integer(0))
  }
  if (length(n_checks) != 1L || is.na(n_checks) || n_checks < 1L) {
    n_checks <- 1L
  }
  if (length(chunk_size) != 1L || is.na(chunk_size) || chunk_size < 1L) {
    chunk_size <- 1L
  }

  eval_every <- as.integer(max(1L, ceiling(svi_steps / n_checks)))
  chunk_boundaries <- if (chunk_size <= svi_steps) {
    seq.int(chunk_size, svi_steps, by = chunk_size)
  } else {
    integer(0)
  }
  boundaries <- unique(c(chunk_boundaries, svi_steps))
  next_target <- eval_every
  checks <- integer(0)
  for (boundary in boundaries) {
    if (boundary >= next_target || boundary == svi_steps) {
      checks <- c(checks, as.integer(boundary))
      while (next_target <= boundary) {
        next_target <- next_target + eval_every
      }
    }
  }
  unique(as.integer(checks))
}

neural_stop_compact_scan_required <- function(reason = NULL) {
  reason <- as.character(reason %||% "unknown scan failure")
  if (length(reason) != 1L || is.na(reason) || !nzchar(reason)) {
    reason <- "unknown scan failure"
  }
  stop(
    paste0(
      "Compact streaming SVI requires scanned JAX updates because ",
      "neural_mcmc_control$compact_update_scan = 'required', but the scan path failed: ",
      reason,
      ". Set neural_mcmc_control$compact_update_scan = 'fallback' or ",
      "compact_update_chunk_size = 1 to use single-step updates."
    ),
    call. = FALSE
  )
}

neural_stop_compact_jit_required <- function(reason = NULL) {
  reason <- as.character(reason %||% "unknown JIT update failure")
  if (length(reason) != 1L || is.na(reason) || !nzchar(reason)) {
    reason <- "unknown JIT update failure"
  }
  stop(
    paste0(
      "Compact streaming SVI requires cached jitted JAX updates, but the jitted update path failed: ",
      reason,
      ". Compact SVI has no unjitted update fallback."
    ),
    call. = FALSE
  )
}

neural_stage_index <- function(party_left_idx, party_right_idx, model_info = NULL) {
  if (!neural_stage_context_enabled(model_info)) {
    return(NULL)
  }
  mode <- NULL
  if (!is.null(model_info) && !is.null(model_info$stage_mode)) {
    mode <- tolower(as.character(model_info$stage_mode))
  }
  if (length(mode) != 1L || is.na(mode) || !nzchar(mode)) {
    mode <- "same"
  }

  if (neural_has_shape(party_left_idx) || neural_has_shape(party_right_idx)) {
    pl <- strenv$jnp$array(party_left_idx)
    pr <- strenv$jnp$array(party_right_idx)
    same <- strenv$jnp$equal(pl, pr)
    stage <- if (identical(mode, "different")) {
      strenv$jnp$logical_not(same)
    } else {
      same
    }
    return(strenv$jnp$astype(stage, strenv$jnp$int32))
  }

  same <- ai(party_left_idx) == ai(party_right_idx)
  if (identical(mode, "different")) {
    return(ifelse(same, 0L, 1L))
  }
  ifelse(same, 1L, 0L)
}

neural_matchup_index <- function(party_left_idx, party_right_idx, model_info){
  if (!neural_matchup_context_enabled(model_info)) {
    return(NULL)
  }
  if (is.null(model_info)) {
    return(NULL)
  }
  n_party_levels <- if (!is.null(model_info$n_party_levels)) {
    as.integer(model_info$n_party_levels)
  } else if (!is.null(model_info$party_levels)) {
    length(model_info$party_levels)
  } else {
    NA_integer_
  }
  if (is.na(n_party_levels) || n_party_levels < 1L) {
    return(NULL)
  }
  pl <- strenv$jnp$array(party_left_idx)
  pr <- strenv$jnp$array(party_right_idx)
  p_min <- strenv$jnp$minimum(pl, pr)
  p_max <- strenv$jnp$maximum(pl, pr)
  half_term <- strenv$jnp$floor_divide(p_min * (p_min - 1L), ai(2L))
  idx <- p_min * ai(n_party_levels) - half_term + (p_max - p_min)
  strenv$jnp$astype(idx, strenv$jnp$int32)
}

neural_model_party_missing_index <- function(model_info = NULL) {
  if (!is.null(model_info$party_missing_index)) {
    return(as.integer(model_info$party_missing_index))
  }
  party_levels <- model_info$party_levels %||% NULL
  party_missing_label <- model_info$party_missing_label %||%
    neural_missing_group_label("candidate")
  if (!is.null(party_levels)) {
    return(neural_missing_group_index(party_levels, party_missing_label))
  }
  0L
}

neural_model_resp_party_missing_index <- function(model_info = NULL) {
  if (!is.null(model_info$resp_party_missing_index)) {
    return(as.integer(model_info$resp_party_missing_index))
  }
  resp_party_levels <- model_info$resp_party_levels %||% NULL
  resp_party_missing_label <- model_info$resp_party_missing_label %||%
    neural_missing_group_label("respondent")
  if (!is.null(resp_party_levels)) {
    return(neural_missing_group_index(resp_party_levels, resp_party_missing_label))
  }
  0L
}

neural_context_present_mask <- function(party_idx = NULL,
                                        other_party_idx = NULL,
                                        resp_party_idx = NULL,
                                        context_present = NULL,
                                        model_info = NULL) {
  if (!is.null(model_info) && identical(model_info$context_present_masking, FALSE)) {
    return(NULL)
  }
  if (!is.null(context_present)) {
    return(strenv$jnp$array(context_present) > 0L)
  }

  checks <- list()
  if (!is.null(party_idx)) {
    party_missing_index <- ai(neural_model_party_missing_index(model_info))
    checks[[length(checks) + 1L]] <- strenv$jnp$not_equal(
      strenv$jnp$array(party_idx),
      party_missing_index
    )
  }
  if (!is.null(other_party_idx)) {
    party_missing_index <- ai(neural_model_party_missing_index(model_info))
    checks[[length(checks) + 1L]] <- strenv$jnp$not_equal(
      strenv$jnp$array(other_party_idx),
      party_missing_index
    )
  }
  if (!is.null(resp_party_idx)) {
    resp_party_missing_index <- ai(neural_model_resp_party_missing_index(model_info))
    checks[[length(checks) + 1L]] <- strenv$jnp$not_equal(
      strenv$jnp$array(resp_party_idx),
      resp_party_missing_index
    )
  }
  if (!length(checks)) {
    return(NULL)
  }

  present <- checks[[1L]]
  if (length(checks) > 1L) {
    for (i in 2L:length(checks)) {
      present <- strenv$jnp$logical_and(present, checks[[i]])
    }
  }
  present
}

neural_resolve_balanced_sampling <- function(config = NULL) {
  if (is.null(config) || identical(config, FALSE)) {
    return(list(enabled = FALSE))
  }
  if (isTRUE(config)) {
    config <- list(enabled = TRUE)
  } else if (is.character(config)) {
    config <- list(enabled = TRUE, scheme = config[[1L]])
  }
  if (!is.list(config)) {
    stop("'neural_mcmc_control$balanced_sampling' must be TRUE, FALSE, a scheme string, or a list.",
         call. = FALSE)
  }
  enabled <- isTRUE(config$enabled %||% TRUE)
  if (!isTRUE(enabled)) {
    return(list(enabled = FALSE))
  }
  scheme <- tolower(as.character(config$scheme %||% "study_equal_respondent"))
  if (length(scheme) != 1L || is.na(scheme) || !identical(scheme, "study_equal_respondent")) {
    stop("'neural_mcmc_control$balanced_sampling$scheme' must be 'study_equal_respondent'.",
         call. = FALSE)
  }
  within_respondent <- tolower(as.character(
    config$within_respondent %||% "uniform_observation"
  ))
  if (length(within_respondent) != 1L || is.na(within_respondent) ||
      !identical(within_respondent, "uniform_observation")) {
    stop("'neural_mcmc_control$balanced_sampling$within_respondent' must be 'uniform_observation'.",
         call. = FALSE)
  }
  replacement <- isTRUE(config$replacement %||% TRUE)
  if (!isTRUE(replacement)) {
    stop("'neural_mcmc_control$balanced_sampling$replacement' must be TRUE for compact SVI.",
         call. = FALSE)
  }
  effective_likelihood_mass <- tolower(as.character(
    config$effective_likelihood_mass %||% "training_observation_count"
  ))
  if (length(effective_likelihood_mass) != 1L ||
      is.na(effective_likelihood_mass) ||
      !identical(effective_likelihood_mass, "training_observation_count")) {
    stop(
      "'neural_mcmc_control$balanced_sampling$effective_likelihood_mass' must be 'training_observation_count'.",
      call. = FALSE
    )
  }
  list(
    enabled = TRUE,
    scheme = scheme,
    within_respondent = within_respondent,
    replacement = TRUE,
    require_respondent_id = isTRUE(config$require_respondent_id %||% TRUE),
    effective_likelihood_mass = effective_likelihood_mass
  )
}

neural_resolve_universal_loss_weighting <- function(value = NULL) {
  if (is.null(value)) {
    return("empirical")
  }
  mode <- tolower(trimws(as.character(value)))
  if (length(mode) != 1L || is.na(mode) || !nzchar(mode) ||
      !mode %in% c("empirical", "balanced_cell")) {
    stop(
      "'neural_mcmc_control$universal_loss_weighting' must be one of 'empirical' or 'balanced_cell'.",
      call. = FALSE
    )
  }
  mode
}

neural_resolve_universal_loss_weight_clip <- function(value = NULL) {
  clip <- suppressWarnings(as.numeric(value %||% c(0.25, 4.0)))
  if (length(clip) < 2L || any(!is.finite(clip[seq_len(2L)])) ||
      clip[[1L]] <= 0 || clip[[2L]] < clip[[1L]]) {
    return(c(0.25, 4.0))
  }
  as.numeric(clip[seq_len(2L)])
}

neural_universal_loss_weighting <- function(task_mode,
                                            likelihood_code,
                                            policy = NULL,
                                            clip = NULL) {
  policy <- neural_resolve_universal_loss_weighting(policy)
  task_mode <- as.character(task_mode)
  likelihood_code <- as.character(likelihood_code)
  if (length(task_mode) != length(likelihood_code)) {
    stop("Universal loss weighting requires row-aligned task modes and likelihood codes.",
         call. = FALSE)
  }
  n_obs <- length(task_mode)
  if (n_obs < 1L) {
    return(list(
      policy = policy,
      weights = NULL,
      observation_weights = numeric(0),
      diagnostics = list(
        policy = policy,
        n_observations = 0L,
        n_cells = 0L,
        clip = NULL,
        cells = data.frame()
      )
    ))
  }
  if (any(is.na(task_mode) | is.na(likelihood_code))) {
    stop("Universal loss weighting requires non-missing task modes and likelihood codes.",
         call. = FALSE)
  }

  cell_label <- paste(task_mode, likelihood_code, sep = "::")
  cell_n <- table(cell_label)
  weights <- rep.int(1, n_obs)
  raw_weight <- rep.int(1, n_obs)
  clip_use <- NULL
  if (identical(policy, "balanced_cell")) {
    clip_use <- neural_resolve_universal_loss_weight_clip(clip)
    raw_weight <- as.numeric(n_obs) /
      (as.numeric(length(cell_n)) * as.numeric(cell_n[cell_label]))
    weights <- pmin(pmax(raw_weight, clip_use[[1L]]), clip_use[[2L]])
    weights <- weights / mean(weights)
  }

  cell_counts <- as.integer(cell_n)
  cell_names <- names(cell_n)
  cell_task <- vapply(strsplit(cell_names, "::", fixed = TRUE), `[[`, character(1), 1L)
  cell_likelihood_code <- vapply(strsplit(cell_names, "::", fixed = TRUE), `[[`, character(1), 2L)
  cell_weight_mean <- as.numeric(tapply(weights, cell_label, mean)[cell_names])
  cell_weight_raw <- as.numeric(tapply(raw_weight, cell_label, mean)[cell_names])
  cell_effective_mass <- as.numeric(tapply(weights, cell_label, sum)[cell_names])
  effective_total <- sum(cell_effective_mass)
  cells <- data.frame(
    cell = cell_names,
    task_mode = cell_task,
    likelihood_code = cell_likelihood_code,
    n_observations = cell_counts,
    raw_observation_share = as.numeric(cell_counts) / as.numeric(n_obs),
    raw_inverse_cell_weight = cell_weight_raw,
    effective_observation_weight = cell_weight_mean,
    effective_weighted_mass = cell_effective_mass,
    effective_weighted_share = if (effective_total > 0) {
      cell_effective_mass / effective_total
    } else {
      rep.int(NA_real_, length(cell_effective_mass))
    },
    stringsAsFactors = FALSE
  )

  list(
    policy = policy,
    weights = if (identical(policy, "balanced_cell")) weights else NULL,
    observation_weights = weights,
    diagnostics = list(
      policy = policy,
      n_observations = as.integer(n_obs),
      n_cells = as.integer(length(cell_n)),
      clip = clip_use,
      cells = cells
    )
  )
}

neural_compact_mixed_branch_loss_scales <- function(n_pair,
                                                    n_single,
                                                    pair_batch_n,
                                                    single_batch_n,
                                                    total_n = NULL,
                                                    batch_n = NULL,
                                                    balanced_sampling = FALSE) {
  n_pair <- as.integer(n_pair %||% 0L)
  n_single <- as.integer(n_single %||% 0L)
  pair_batch_n <- as.integer(pair_batch_n %||% 0L)
  single_batch_n <- as.integer(single_batch_n %||% 0L)
  if (isTRUE(balanced_sampling)) {
    total_n <- as.integer(total_n %||% (n_pair + n_single))
    batch_n <- as.integer(batch_n %||% (pair_batch_n + single_batch_n))
    scale <- if (!is.na(batch_n) && batch_n > 0L) {
      as.numeric(total_n) / as.numeric(batch_n)
    } else {
      0
    }
    return(list(pair = scale, single = scale))
  }
  list(
    pair = if (!is.na(pair_batch_n) && pair_batch_n > 0L) {
      as.numeric(n_pair) / as.numeric(pair_batch_n)
    } else {
      0
    },
    single = if (!is.na(single_batch_n) && single_batch_n > 0L) {
      as.numeric(n_single) / as.numeric(single_batch_n)
    } else {
      0
    }
  )
}

neural_sample_one <- function(x) {
  if (length(x) < 1L) {
    stop("Cannot sample from an empty vector.", call. = FALSE)
  }
  if (length(x) == 1L) {
    return(x[[1L]])
  }
  sample(x, size = 1L)
}

neural_build_balanced_sampling_state <- function(obs_idx,
                                                 study_index,
                                                 respondent_id,
                                                 config,
                                                 context = "compact SVI") {
  config <- neural_resolve_balanced_sampling(config)
  if (!isTRUE(config$enabled)) {
    return(NULL)
  }
  obs_idx <- as.integer(obs_idx)
  obs_idx <- obs_idx[!is.na(obs_idx) & obs_idx >= 1L]
  if (length(obs_idx) < 1L) {
    stop(sprintf("%s balanced sampling received no observations.", context), call. = FALSE)
  }
  max_idx <- max(obs_idx)
  if (is.null(study_index) || length(study_index) < max_idx) {
    stop(sprintf(
      "%s balanced sampling requires row-aligned experiment/study indices.",
      context
    ), call. = FALSE)
  }
  study_index <- as.character(study_index[obs_idx])
  if (any(is.na(study_index) | !nzchar(study_index))) {
    stop(sprintf(
      "%s balanced sampling requires non-missing experiment/study indices.",
      context
    ), call. = FALSE)
  }
  if (is.null(respondent_id) || length(respondent_id) < max_idx) {
    if (isTRUE(config$require_respondent_id)) {
      stop(sprintf(
        "%s balanced sampling requires row-aligned respondent_id values.",
        context
      ), call. = FALSE)
    }
    respondent_id <- as.character(seq_len(max_idx))
  }
  respondent_id <- as.character(respondent_id[obs_idx])
  if (any(is.na(respondent_id) | !nzchar(respondent_id))) {
    if (isTRUE(config$require_respondent_id)) {
      stop(sprintf(
        "%s balanced sampling requires non-missing respondent_id values.",
        context
      ), call. = FALSE)
    }
    missing <- is.na(respondent_id) | !nzchar(respondent_id)
    respondent_id[missing] <- paste0("row::", obs_idx[missing])
  }

  unit_key <- paste(study_index, respondent_id, sep = "\r")
  unit_levels <- unique(unit_key)
  unit_index <- match(unit_key, unit_levels)
  obs_by_unit <- split(obs_idx, unit_index)
  unit_study <- vapply(seq_along(obs_by_unit), function(i) {
    study_index[match(i, unit_index)]
  }, character(1))
  units_by_study <- split(seq_along(obs_by_unit), unit_study)
  study_levels <- names(units_by_study)
  respondent_counts <- vapply(units_by_study, length, integer(1))
  obs_counts <- vapply(units_by_study, function(units) {
    sum(vapply(obs_by_unit[units], length, integer(1)))
  }, integer(1))
  list(
    enabled = TRUE,
    scheme = config$scheme,
    within_respondent = config$within_respondent,
    replacement = TRUE,
    effective_likelihood_mass = config$effective_likelihood_mass,
    study_levels = study_levels,
    units_by_study = units_by_study,
    obs_by_unit = obs_by_unit,
    n_observations = length(obs_idx),
    n_studies = length(study_levels),
    n_respondent_units = length(obs_by_unit),
    respondent_counts_by_study = respondent_counts,
    observation_counts_by_study = obs_counts
  )
}

neural_sample_balanced_obs_idx <- function(state, batch_size) {
  if (is.null(state) || !isTRUE(state$enabled)) {
    stop("Balanced sampling state is not enabled.", call. = FALSE)
  }
  batch_size <- as.integer(batch_size)
  if (length(batch_size) != 1L || is.na(batch_size) || batch_size < 1L) {
    stop("'batch_size' must be a positive integer.", call. = FALSE)
  }
  out <- integer(batch_size)
  for (i in seq_len(batch_size)) {
    study <- neural_sample_one(state$study_levels)
    unit <- neural_sample_one(state$units_by_study[[study]])
    out[[i]] <- neural_sample_one(state$obs_by_unit[[unit]])
  }
  out
}

neural_context_present_float <- function(party_idx = NULL,
                                         other_party_idx = NULL,
                                         resp_party_idx = NULL,
                                         context_present = NULL,
                                         model_info = NULL,
                                         n_batch = NULL) {
  mask <- neural_context_present_mask(
    party_idx = party_idx,
    other_party_idx = other_party_idx,
    resp_party_idx = resp_party_idx,
    context_present = context_present,
    model_info = model_info
  )
  if (is.null(mask)) {
    return(NULL)
  }
  out <- strenv$jnp$atleast_1d(strenv$jnp$astype(mask, strenv$dtj))
  if (!is.null(n_batch) &&
      ai(out$shape[[1]]) == 1L &&
      ai(n_batch) > 1L) {
    out <- out * strenv$jnp$ones(list(ai(n_batch)), dtype = strenv$dtj)
  }
  out
}

neural_pair_context_present <- function(party_left_idx,
                                        party_right_idx,
                                        resp_party_idx,
                                        model_info = NULL) {
  neural_context_present_mask(
    party_idx = party_left_idx,
    other_party_idx = party_right_idx,
    resp_party_idx = resp_party_idx,
    model_info = model_info
  )
}

neural_logits_to_q <- function(logits, likelihood){
  if (likelihood == "bernoulli") {
    prob <- strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L))
    return(strenv$jnp$reshape(prob, list(-1L, 1L)))
  }
  if (likelihood == "categorical") {
    probs <- strenv$jax$nn$softmax(logits, axis = -1L)
    prob <- strenv$jnp$take(probs, 1L, axis = 1L)
    return(strenv$jnp$reshape(prob, list(-1L, 1L)))
  }
  if (likelihood == "mixed") {
    # Universal/mixed models emit (n, global_out_dim) logits; column 0 carries
    # the bernoulli/pairwise logit. The normal-mu fallback below would crash on
    # squeeze(axis = 1) for out_dim > 1 and, worse, return a raw logit as "q".
    # jnp$take keeps the pi-gradient path intact.
    prob <- strenv$jax$nn$sigmoid(strenv$jnp$take(logits, 0L, axis = 1L))
    return(strenv$jnp$reshape(prob, list(-1L, 1L)))
  }
  mu <- strenv$jnp$squeeze(logits, axis = 1L)
  strenv$jnp$reshape(mu, list(-1L, 1L))
}

# Row-level outcome validity under the universal mixed likelihood. Codes:
# 0 = bernoulli (y in {0,1}), 1 = categorical (integer y in [0, n_outcomes)),
# 2 = normal (finite y), 3 = ordinal (integer y in [0, n_outcomes)). A single
# invalid row that reaches the likelihood either NaNs the whole ELBO (NaN y)
# or is index-clamped into a finite-but-wrong log-prob, so training inputs
# must be screened with this predicate before they reach JAX.
neural_mixed_row_is_valid <- function(y,
                                      likelihood_code_obs,
                                      n_outcomes_obs) {
  y_num <- as.numeric(y)
  code <- as.integer(likelihood_code_obs)
  n_outcomes_obs <- as.integer(n_outcomes_obs)
  ok <- is.finite(y_num) & is.finite(code)
  bern_idx <- ok & code == 0L
  if (any(bern_idx)) {
    ok[bern_idx] <- y_num[bern_idx] %in% c(0, 1)
  }
  cat_idx <- ok & code == 1L
  if (any(cat_idx)) {
    y_cat <- suppressWarnings(as.integer(round(y_num[cat_idx])))
    ok[cat_idx] <- is.finite(y_cat) &
      abs(y_num[cat_idx] - y_cat) < 1e-8 &
      is.finite(n_outcomes_obs[cat_idx]) &
      n_outcomes_obs[cat_idx] >= 2L &
      y_cat >= 0L &
      y_cat < n_outcomes_obs[cat_idx]
  }
  norm_idx <- ok & code == 2L
  if (any(norm_idx)) {
    ok[norm_idx] <- is.finite(y_num[norm_idx])
  }
  ord_idx <- ok & code == 3L
  if (any(ord_idx)) {
    y_ord <- suppressWarnings(as.integer(round(y_num[ord_idx])))
    ok[ord_idx] <- is.finite(y_ord) &
      abs(y_num[ord_idx] - y_ord) < 1e-8 &
      is.finite(n_outcomes_obs[ord_idx]) &
      n_outcomes_obs[ord_idx] >= 2L &
      y_ord >= 0L &
      y_ord < n_outcomes_obs[ord_idx]
  }
  ok
}

apply_implicit_parameterization_jnp <- function(p_sub,
                                                implicit = FALSE,
                                                axis = -1L,
                                                clip = TRUE) {
  p_sub <- strenv$jnp$array(p_sub)
  if (!isTRUE(implicit)) {
    return(p_sub)
  }
  axis_use <- as.integer(axis)
  ndim <- length(p_sub$shape)
  if (axis_use < 0L) {
    axis_use <- axis_use + ndim
  }
  holdout <- strenv$jnp$array(1., dtype = p_sub$dtype) -
    strenv$jnp$sum(p_sub, axis = axis_use)
  if (isTRUE(clip)) {
    holdout <- strenv$jnp$clip(holdout, 0., 1.)
  }
  holdout_exp <- strenv$jnp$expand_dims(holdout, axis = axis_use)
  strenv$jnp$concatenate(list(p_sub, holdout_exp), axis = axis_use)
}

neural_params_from_theta <- function(theta_vec, model_info){
  if (is.null(model_info)) {
    stop("neural_params_from_theta requires a non-null model_info.", call. = FALSE)
  }
  if (is.null(theta_vec)) {
    return(model_info$params)
  }
  schema_missing <- is.null(model_info$param_names) ||
    is.null(model_info$param_offsets) ||
    is.null(model_info$param_sizes) ||
    is.null(model_info$param_shapes)
  if (isTRUE(schema_missing)) {
    if (!is.null(model_info$params) &&
        !is.null(model_info$n_factors) &&
        !is.null(model_info$model_depth)) {
      schema <- neural_build_param_schema(
        params = model_info$params,
        n_factors = model_info$n_factors,
        model_depth = model_info$model_depth,
        factor_tokenization = neural_factor_tokenization(model_info)
      )
      model_info$param_names <- schema$param_names
      model_info$param_shapes <- schema$param_shapes
      model_info$param_sizes <- schema$param_sizes
      model_info$param_offsets <- schema$param_offsets
      if (is.null(model_info$n_params)) {
        model_info$n_params <- schema$n_params
      }
    } else {
      stop("model_info is missing parameter schema; cannot unpack theta_vec.", call. = FALSE)
    }
  }
  theta_vec <- strenv$jnp$reshape(theta_vec, list(-1L))
  offsets_num <- as.integer(unlist(model_info$param_offsets))
  sizes_num <- as.integer(unlist(model_info$param_sizes))
  if (length(offsets_num) != length(sizes_num)) {
    stop("param_offsets and param_sizes length mismatch in model_info.", call. = FALSE)
  }
  theta_len <- as.integer(theta_vec$shape[[1]])
  max_end <- if (length(sizes_num) > 0L) max(offsets_num + sizes_num) else 0L
  expected_total <- if (!is.null(model_info$n_params)) as.integer(model_info$n_params) else max_end
  if (is.finite(theta_len) && is.finite(expected_total) && theta_len != expected_total) {
    stop(sprintf("theta_vec length (%d) does not match expected parameter total (%d).",
                 theta_len, expected_total),
         call. = FALSE)
  }
  if (is.finite(theta_len) && is.finite(max_end) && theta_len < max_end) {
    stop(sprintf("theta_vec length (%d) is shorter than required (%d).",
                 theta_len, max_end),
         call. = FALSE)
  }
  params <- list()
  param_names <- model_info$param_names
  param_offsets <- model_info$param_offsets
  param_sizes <- model_info$param_sizes
  param_shapes <- model_info$param_shapes
  for (i_ in seq_along(param_names)) {
    start <- as.integer(param_offsets[[i_]])
    size <- as.integer(param_sizes[[i_]])
    if (is.finite(theta_len) && is.finite(start) && is.finite(size) &&
        (start < 0L || (start + size) > theta_len)) {
      stop(sprintf("theta_vec slice for '%s' (%d..%d) exceeds length (%d).",
                   param_names[[i_]], start, start + size - 1L, theta_len),
           call. = FALSE)
    }
    idx <- strenv$jnp$arange(ai(start), ai(start + size))
    slice <- strenv$jnp$take(theta_vec, idx, axis = 0L)
    shape_use <- param_shapes[[i_]]
    if (length(shape_use) == 0L) {
      shape_use <- c(1L)
    }
    shape_size <- as.integer(prod(shape_use))
    if (is.finite(size) && is.finite(shape_size) && size != shape_size) {
      stop(sprintf("Param '%s' size (%d) does not match shape product (%d).",
                   param_names[[i_]], size, shape_size),
           call. = FALSE)
    }
    params[[param_names[[i_]]]] <- strenv$jnp$reshape(slice, as.integer(shape_use))
  }
  if (!is.null(params$log_pairwise_bernoulli_logit_scale)) {
    params$pairwise_bernoulli_logit_scale <- strenv$jnp$exp(
      params$log_pairwise_bernoulli_logit_scale
    )
  }
  if (!is.null(params$log_calibration_scale)) {
    params$calibration_scale <- strenv$jnp$exp(params$log_calibration_scale)
  }
  params
}

neural_build_param_schema <- function(params,
                                      n_factors,
                                      model_depth,
                                      factor_tokenization = NULL) {
  if (is.null(params) || !is.list(params)) {
    stop("neural_build_param_schema requires a params list.", call. = FALSE)
  }
  n_factors <- as.integer(n_factors)
  model_depth <- as.integer(model_depth)
  if (is.na(n_factors) || n_factors < 1L) {
    stop("neural_build_param_schema requires n_factors >= 1.", call. = FALSE)
  }
  if (is.na(model_depth) || model_depth < 1L) {
    stop("neural_build_param_schema requires model_depth >= 1.", call. = FALSE)
  }

  factor_mode <- neural_factor_tokenization(mode = factor_tokenization %||% params$factor_tokenization %||% "fused")
  if (!is.null(params$E_factor_start) || !is.null(params$E_factor_end) ||
      !is.null(params$E_factor_role)) {
    stop("Span factor parameters are incompatible with fused factor tokenization. Refit the model.", call. = FALSE)
  }
  if (!identical(factor_mode, "fused")) {
    stop("'factor_tokenization' must be 'fused'.", call. = FALSE)
  }

  param_names <- c(c(
                     "E_factor_fused_base",
                     "W_factor_fuse_1",
                     "b_factor_fuse_1",
                     "W_factor_fuse_2",
                     "b_factor_fuse_2"
                   ),
                   "E_party", "E_resp_party", "E_choice",
                   "E_respondent_cls", "E_candidate_cls",
                   "E_token_family", "E_experiment",
                   "E_sep", "E_segment")
  if (!is.null(params$E_stage)) {
    param_names <- c(param_names, "E_stage")
  }
  if (!is.null(params$E_matchup)) {
    param_names <- c(param_names, "E_matchup")
  }
  if (!is.null(params$E_rel)) {
    param_names <- c(param_names, "E_rel")
  }
  if (!is.null(params$E_covariate_start) || !is.null(params$E_covariate_end) ||
      !is.null(params$E_covariate_role)) {
    stop("Span covariate parameters are incompatible with fused covariate tokenization. Refit the model.", call. = FALSE)
  }
  if (!is.null(params$E_covariate_fused_base)) {
    param_names <- c(
      param_names,
      "E_covariate_fused_base",
      "E_covariate_missing",
      "W_covariate_fuse_1",
      "b_covariate_fuse_1",
      "W_covariate_fuse_2",
      "b_covariate_fuse_2"
    )
    if (!is.null(params$W_covariate_value_text)) {
      param_names <- c(param_names, "W_covariate_value_text")
    }
    if (!is.null(params$W_covariate_value_shared)) {
      param_names <- c(param_names, "W_covariate_value_shared")
    }
    if (!is.null(params$W_covariate_value_basis)) {
      param_names <- c(
        param_names,
        "W_covariate_value_basis",
        "W_covariate_value_conditioner_1",
        "b_covariate_value_conditioner_1",
        "W_covariate_value_conditioner_2",
        "b_covariate_value_conditioner_2"
      )
    }
  } else if (!is.null(params$E_covariate_id) ||
             !is.null(params$E_covariate_present) ||
             !is.null(params$V_covariate_value)) {
    stop("Legacy covariate parameters are incompatible with fused covariate tokenization. Refit the model.", call. = FALSE)
  }
  if (!is.null(params$W_factor_name_text)) {
    param_names <- c(
      param_names,
      "W_factor_name_text",
      "W_level_name_text",
      "W_covariate_name_text"
    )
  }
  if (!is.null(params$W_factor_struct)) {
    param_names <- c(param_names, "W_factor_struct")
  }
  if (!is.null(params$W_level_struct)) {
    param_names <- c(param_names, "W_level_struct")
  }
  if (!is.null(params$W_experiment_text)) {
    param_names <- c(param_names, "W_experiment_text")
  }
  if (!is.null(params$W_place_context)) {
    param_names <- c(param_names, "W_place_context")
  }
  if (!is.null(params$W_time_context)) {
    param_names <- c(param_names, "W_time_context")
  }
  if (!is.null(params$M_cross)) {
    param_names <- c(param_names, "M_cross")
  }
  if (!is.null(params$W_cross_out)) {
    param_names <- c(param_names, "W_cross_out")
  }
  if (!is.null(params$W_rc_r) || !is.null(params$W_rc_c) || !is.null(params$W_rc_out)) {
    param_names <- c(param_names, "alpha_rc", "W_rc_r", "W_rc_c", "W_rc_out")
  }
  if (!is.null(params$log_pairwise_bernoulli_logit_scale)) {
    param_names <- c(param_names, "log_pairwise_bernoulli_logit_scale")
  }
  if (!is.null(params$log_calibration_scale)) {
    param_names <- c(param_names, "log_calibration_scale")
  }
  if (isTRUE(neural_has_stacked_standard_transformer(params))) {
    param_names <- c(param_names, neural_standard_transformer_stack_names(params))
  } else {
    for (l_ in 1L:model_depth) {
      param_names <- c(param_names,
                       paste0("pseudo_query_attn_l", l_),
                       paste0("pseudo_query_ff_l", l_),
                       paste0("alpha_attn_l", l_),
                       paste0("alpha_ff_l", l_),
                       paste0("RMS_attn_l", l_),
                       paste0("RMS_q_l", l_),
                       paste0("RMS_k_l", l_),
                       paste0("RMS_ff_l", l_),
                       paste0("W_q_l", l_),
                       paste0("W_k_l", l_),
                       paste0("W_v_l", l_),
                       paste0("W_o_l", l_),
                       paste0("W_ff1_l", l_),
                       paste0("W_ff2_l", l_))
    }
  }
  param_names <- c(param_names,
                   "alpha_cross",
                   "RMS_cross",
                   "RMS_merge_cross",
                   "RMS_q_cross",
                   "RMS_k_cross",
                   "W_q_cross",
                   "W_k_cross",
                   "W_v_cross",
                   "W_o_cross")
  param_names <- c(param_names, "pseudo_query_final", "RMS_final", "W_add_out", "W_out", "b_out")
  if (!is.null(params$sigma)) {
    param_names <- c(param_names, "sigma")
  }
  param_names <- param_names[param_names %in% names(params)]

  param_shapes <- lapply(param_names, function(name) {
    shape <- tryCatch(reticulate::py_to_r(params[[name]]$shape), error = function(e) NULL)
    if (is.null(shape)) integer(0) else as.integer(shape)
  })
  param_sizes <- vapply(param_shapes, function(shape) {
    if (length(shape) == 0L) {
      1L
    } else {
      as.integer(prod(shape))
    }
  }, integer(1))
  param_offsets <- as.integer(cumsum(c(0L, param_sizes))[seq_len(length(param_sizes))])
  param_total <- sum(param_sizes)

  list(
    param_names = param_names,
    param_shapes = param_shapes,
    param_sizes = param_sizes,
    param_offsets = param_offsets,
    n_params = ai(param_total)
  )
}

neural_flatten_params <- function(params, schema, dtype = NULL) {
  if (is.null(dtype)) {
    dtype <- strenv$jnp$float32
  }
  parts <- lapply(schema$param_names, function(name) {
    strenv$jnp$ravel(params[[name]])
  })
  if (length(parts) == 0L) {
    return(strenv$jnp$array(numeric(0), dtype = dtype))
  }
  strenv$jnp$concatenate(parts, axis = 0L)
}

neural_standard_transformer_stack_map <- function() {
  c(
    W_q_layers = "W_q_l",
    W_k_layers = "W_k_l",
    W_v_layers = "W_v_l",
    W_o_layers = "W_o_l",
    W_ff1_layers = "W_ff1_l",
    W_ff2_layers = "W_ff2_l",
    RMS_attn_layers = "RMS_attn_l",
    RMS_ff_layers = "RMS_ff_l",
    RMS_q_layers = "RMS_q_l",
    RMS_k_layers = "RMS_k_l",
    alpha_attn_layers = "alpha_attn_l",
    alpha_ff_layers = "alpha_ff_l"
  )
}

neural_standard_transformer_stack_names <- function(params = NULL) {
  names_use <- names(neural_standard_transformer_stack_map())
  if (is.null(params)) {
    return(names_use)
  }
  names_use[names_use %in% names(params)]
}

neural_has_stacked_standard_transformer <- function(params) {
  if (is.null(params) || !is.list(params)) {
    return(FALSE)
  }
  required <- c(
    "W_q_layers", "W_k_layers", "W_v_layers", "W_o_layers",
    "W_ff1_layers", "W_ff2_layers",
    "RMS_attn_layers", "RMS_ff_layers",
    "alpha_attn_layers", "alpha_ff_layers"
  )
  all(required %in% names(params))
}

neural_stack_standard_transformer_layers <- function(params,
                                                     model_depth,
                                                     drop_legacy = FALSE) {
  if (is.null(params) || !is.list(params)) {
    return(params)
  }
  model_depth <- as.integer(model_depth)
  if (is.na(model_depth) || model_depth < 1L) {
    return(params)
  }
  out <- params
  map <- neural_standard_transformer_stack_map()
  for (stack_name in names(map)) {
    legacy_base <- unname(map[[stack_name]])
    legacy_names <- paste0(legacy_base, seq_len(model_depth))
    if (!all(legacy_names %in% names(params))) {
      next
    }
    values <- lapply(legacy_names, function(name) params[[name]])
    if (any(vapply(values, is.null, logical(1)))) {
      next
    }
    out[[stack_name]] <- strenv$jnp$stack(values, axis = 0L)
    if (isTRUE(drop_legacy)) {
      out[legacy_names] <- NULL
    }
  }
  out
}

neural_normalize_cross_encoder_mode <- function(value) {
  if (is.null(value)) {
    return(NULL)
  }
  if (isTRUE(value)) {
    return("term")
  }
  if (identical(value, FALSE)) {
    return("none")
  }
  if (is.character(value)) {
    mode <- tolower(as.character(value))
    if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      if (mode %in% c("none", "term", "full")) {
        return(mode)
      }
      if (mode %in% c("attn", "lite", "cross_attn", "cls_attn", "cross-attn", "cls-attn")) {
        return("attn")
      }
      if (mode %in% c("true", "false")) {
        return(ifelse(mode == "true", "term", "none"))
      }
    }
  }
  NA_character_
}

neural_cross_encoder_mode <- function(model_info) {
  mode <- neural_normalize_cross_encoder_mode(model_info$cross_candidate_encoder_mode)
  if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
    return(mode)
  }
  if (isTRUE(model_info$cross_candidate_encoder)) {
    return("term")
  }
  "none"
}

neural_normalize_residual_mode <- function(value) {
  if (is.null(value)) {
    return("standard")
  }
  if (isTRUE(value)) {
    return("full_attn")
  }
  if (identical(value, FALSE)) {
    return("standard")
  }
  if (is.character(value)) {
    mode <- tolower(as.character(value))
    if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      if (mode %in% c("standard", "add", "additive", "rezero")) {
        return("standard")
      }
      if (mode %in% c(
        "full_attn", "full-attn", "full_attention",
        "attnres", "full_attnres", "attention_residual",
        "attention-residual", "full_attention_residual",
        "full-attention-residual"
      )) {
        return("full_attn")
      }
    }
  }
  NA_character_
}

neural_transformer_residual_mode <- function(model_info) {
  mode <- neural_normalize_residual_mode(model_info$residual_mode)
  if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
    return(mode)
  }
  "standard"
}

neural_normalize_attention_backend <- function(value) {
  if (is.null(value)) {
    return("auto")
  }
  mode <- tolower(as.character(value))
  if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
    if (mode %in% c("auto", "default")) {
      return("auto")
    }
    if (mode %in% c("xla", "jax", "standard")) {
      return("xla")
    }
    if (mode %in% c("cudnn", "cuda", "flash", "flash_attention", "flash-attention")) {
      return("cudnn")
    }
  }
  NA_character_
}

neural_normalize_attention_dtype <- function(value) {
  if (is.null(value)) {
    return("auto")
  }
  mode <- tolower(as.character(value))
  if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
    if (mode %in% c("auto", "default")) {
      return("auto")
    }
    if (mode %in% c("float32", "fp32", "f32")) {
      return("float32")
    }
    if (mode %in% c("bfloat16", "bf16")) {
      return("bfloat16")
    }
    if (mode %in% c("float16", "fp16", "f16", "half")) {
      return("float16")
    }
  }
  NA_character_
}

neural_attention_backend <- function(model_info = NULL) {
  mode <- neural_normalize_attention_backend(model_info$attention_backend %||% "auto")
  if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
    return(mode)
  }
  "auto"
}

neural_attention_dtype_mode <- function(model_info = NULL) {
  mode <- neural_normalize_attention_dtype(model_info$attention_dtype %||% "auto")
  if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
    return(mode)
  }
  "auto"
}

neural_attention_padding_multiple <- function(model_info = NULL) {
  value <- suppressWarnings(as.integer(model_info$attention_padding_multiple %||% 8L))
  if (length(value) != 1L || is.na(value) || value < 1L) {
    return(8L)
  }
  value
}

neural_attention_has_dpa <- function() {
  tryCatch(
    reticulate::py_has_attr(strenv$jax$nn, "dot_product_attention"),
    error = function(e) FALSE
  )
}

neural_attention_jax_backend <- function() {
  tryCatch(as.character(strenv$jax$default_backend()), error = function(e) NA_character_)
}

neural_attention_cuda_available <- function() {
  backend <- neural_attention_jax_backend()
  if (identical(backend, "gpu")) {
    return(TRUE)
  }
  devices <- tryCatch(strenv$jax$devices(), error = function(e) NULL)
  if (is.null(devices) || length(devices) < 1L) {
    return(FALSE)
  }
  any(vapply(devices, function(device) {
    grepl("cuda", tolower(as.character(device)), fixed = TRUE)
  }, logical(1)))
}

neural_attention_dtype_object <- function(dtype_mode, prefer_cudnn = FALSE) {
  mode <- neural_normalize_attention_dtype(dtype_mode)
  if (!is.character(mode) || length(mode) != 1L || is.na(mode) || !nzchar(mode)) {
    mode <- "auto"
  }
  if (identical(mode, "auto")) {
    mode <- if (isTRUE(prefer_cudnn)) "bfloat16" else "float32"
  }
  if (identical(mode, "bfloat16")) {
    return(list(dtype = strenv$jnp$bfloat16, label = "bfloat16"))
  }
  if (identical(mode, "float16")) {
    return(list(dtype = strenv$jnp$float16, label = "float16"))
  }
  list(dtype = strenv$jnp$float32, label = "float32")
}

neural_attention_resolve_backend <- function(model_info = NULL,
                                             role = c("self", "cross"),
                                             fail_on_forced = TRUE) {
  role <- match.arg(role)
  requested <- neural_attention_backend(model_info)
  has_dpa <- neural_attention_has_dpa()
  cuda_available <- neural_attention_cuda_available()
  if (!isTRUE(has_dpa)) {
    if (identical(requested, "cudnn") && isTRUE(fail_on_forced)) {
      stop("attention_backend='cudnn' requires jax.nn.dot_product_attention.", call. = FALSE)
    }
    return(list(
      requested = requested,
      backend = "dense",
      fallback_reason = "jax_dot_product_attention_unavailable",
      cuda_available = isTRUE(cuda_available)
    ))
  }
  if (identical(requested, "xla")) {
    return(list(
      requested = requested,
      backend = "xla",
      fallback_reason = NA_character_,
      cuda_available = isTRUE(cuda_available)
    ))
  }
  if (identical(requested, "cudnn")) {
    if (!isTRUE(cuda_available)) {
      if (isTRUE(fail_on_forced)) {
        stop("attention_backend='cudnn' requires a CUDA-backed JAX device.", call. = FALSE)
      }
      return(list(
        requested = requested,
        backend = "xla",
        fallback_reason = "cuda_unavailable",
        cuda_available = FALSE
      ))
    }
    if (identical(neural_attention_dtype_mode(model_info), "float32")) {
      if (isTRUE(fail_on_forced)) {
        stop("attention_backend='cudnn' requires attention_dtype='auto', 'bfloat16', or 'float16'.", call. = FALSE)
      }
      return(list(
        requested = requested,
        backend = "xla",
        fallback_reason = "cudnn_requires_fp16_or_bf16",
        cuda_available = TRUE
      ))
    }
    if (!identical(role, "self")) {
      return(list(
        requested = requested,
        backend = "xla",
        fallback_reason = "cudnn_cross_attention_disabled",
        cuda_available = TRUE
      ))
    }
    return(list(
      requested = requested,
      backend = "cudnn",
      fallback_reason = NA_character_,
      cuda_available = TRUE
    ))
  }
  if (isTRUE(cuda_available) && identical(role, "self")) {
    if (identical(neural_attention_dtype_mode(model_info), "float32")) {
      return(list(
        requested = requested,
        backend = "xla",
        fallback_reason = "cudnn_requires_fp16_or_bf16",
        cuda_available = TRUE
      ))
    }
    return(list(
      requested = requested,
      backend = "cudnn",
      fallback_reason = NA_character_,
      cuda_available = TRUE
    ))
  }
  list(
    requested = requested,
    backend = "xla",
    fallback_reason = if (isTRUE(cuda_available)) "non_self_attention" else "cuda_unavailable",
    cuda_available = isTRUE(cuda_available)
  )
}

neural_next_multiple <- function(x, multiple) {
  x <- ai(x)
  multiple <- ai(multiple)
  if (multiple <= 1L) {
    return(x)
  }
  as.integer(ceiling(x / multiple) * multiple)
}

neural_attention_mask_value <- function(scores) {
  dtype <- scores$dtype
  strenv$jnp$array(strenv$jnp$finfo(dtype)$min, dtype = dtype)
}

neural_floor_sigma <- function(sigma, dtype = NULL) {
  if (is.null(sigma)) {
    return(NULL)
  }
  dtype_use <- dtype %||% tryCatch(sigma$dtype, error = function(e) strenv$dtj)
  strenv$jnp$maximum(sigma, strenv$jnp$array(1e-3, dtype = dtype_use))
}

neural_self_attention_context <- function(Qh, Kh, Vh, token_mask, model_info) {
  resolve <- neural_attention_resolve_backend(model_info, role = "self", fail_on_forced = TRUE)
  if (identical(resolve$backend, "dense")) {
    scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(ai(model_info$head_dim))))
    scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
    if (!is.null(token_mask)) {
      mask_use <- strenv$jnp$reshape(
        strenv$jnp$astype(token_mask > 0, scores$dtype),
        list(token_mask$shape[[1]], 1L, 1L, token_mask$shape[[2]])
      )
      # Choice/CLS tokens are always emitted by the packer, so every row has at least one valid key.
      large_neg <- neural_attention_mask_value(scores)
      scores <- strenv$jnp$where(mask_use > 0, scores, large_neg)
    }
    attn <- strenv$jax$nn$softmax(scores, axis = -1L)
    return(strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh))
  }

  original_dtype <- Qh$dtype
  n_batch <- ai(Qh$shape[[1]])
  seq_len <- ai(Qh$shape[[2]])
  n_heads <- ai(model_info$n_heads)
  head_dim <- ai(model_info$head_dim)
  backend <- resolve$backend
  dtype_info <- neural_attention_dtype_object(
    neural_attention_dtype_mode(model_info),
    prefer_cudnn = identical(backend, "cudnn")
  )
  Q_use <- strenv$jnp$astype(Qh, dtype_info$dtype)
  K_use <- strenv$jnp$astype(Kh, dtype_info$dtype)
  V_use <- strenv$jnp$astype(Vh, dtype_info$dtype)

  mask_use <- token_mask
  if (is.null(mask_use)) {
    mask_use <- strenv$jnp$ones(list(n_batch, seq_len), dtype = strenv$dtj)
  }

  seq_use <- seq_len
  if (identical(backend, "cudnn")) {
    pad_to <- neural_next_multiple(seq_len, neural_attention_padding_multiple(model_info))
    pad_n <- ai(pad_to - seq_len)
    if (pad_n > 0L) {
      q_pad <- strenv$jnp$zeros(list(n_batch, pad_n, n_heads, head_dim), dtype = Q_use$dtype)
      k_pad <- strenv$jnp$zeros(list(n_batch, pad_n, n_heads, head_dim), dtype = K_use$dtype)
      v_pad <- strenv$jnp$zeros(list(n_batch, pad_n, n_heads, head_dim), dtype = V_use$dtype)
      Q_use <- strenv$jnp$concatenate(list(Q_use, q_pad), axis = 1L)
      K_use <- strenv$jnp$concatenate(list(K_use, k_pad), axis = 1L)
      V_use <- strenv$jnp$concatenate(list(V_use, v_pad), axis = 1L)
      mask_pad <- strenv$jnp$zeros(list(n_batch, pad_n), dtype = mask_use$dtype)
      mask_use <- strenv$jnp$concatenate(list(mask_use, mask_pad), axis = 1L)
      seq_use <- pad_to
    }
  }

  key_mask <- strenv$jnp$reshape(mask_use > 0, list(n_batch, 1L, 1L, seq_use))
  if (identical(backend, "cudnn")) {
    attn_mask <- strenv$jnp$broadcast_to(key_mask, list(n_batch, 1L, seq_use, seq_use))
  } else {
    attn_mask <- key_mask
  }
  context_h <- strenv$jax$nn$dot_product_attention(
    Q_use,
    K_use,
    V_use,
    mask = attn_mask,
    implementation = backend
  )
  if (seq_use != seq_len) {
    keep_idx <- strenv$jnp$arange(seq_len)
    context_h <- strenv$jnp$take(context_h, keep_idx, axis = 1L)
  }
  strenv$jnp$astype(context_h, original_dtype)
}

neural_make_transformer_model_info <- function(model_depth,
                                               model_dims,
                                               n_heads,
                                               head_dim,
                                               residual_mode = "standard",
                                               attention_backend = "auto",
                                               attention_dtype = "auto",
                                               attention_padding_multiple = 8L,
                                               attention_resolved_backend = NULL,
                                               attention_fallback_reason = NULL) {
  list(
    model_depth = model_depth,
    model_dims = model_dims,
    n_heads = n_heads,
    head_dim = head_dim,
    transformer_ffn = "swiglu",
    residual_mode = neural_normalize_residual_mode(residual_mode),
    attention_backend = neural_normalize_attention_backend(attention_backend),
    attention_dtype = neural_normalize_attention_dtype(attention_dtype),
    attention_padding_multiple = neural_attention_padding_multiple(
      list(attention_padding_multiple = attention_padding_multiple)
    ),
    attention_resolved_backend = attention_resolved_backend,
    attention_fallback_reason = attention_fallback_reason
  )
}

neural_factor_schema_supplied <- function(factor_name_text = NULL,
                                          level_name_text = NULL,
                                          factor_struct_matrix = NULL,
                                          level_struct_matrices = NULL,
                                          default_factor_order = NULL,
                                          factor_order_by_experiment = NULL) {
  !is.null(factor_name_text) ||
    !is.null(level_name_text) ||
    !is.null(factor_struct_matrix) ||
    !is.null(level_struct_matrices) ||
    length(default_factor_order %||% integer(0)) > 0L ||
    length(factor_order_by_experiment %||% list()) > 0L
}

neural_model_info_factor_schema_supplied <- function(model_info) {
  if (is.null(model_info)) {
    return(FALSE)
  }
  isTRUE(model_info$factor_schema_supplied) ||
    neural_factor_schema_supplied(
      factor_name_text = model_info$factor_name_text %||% NULL,
      level_name_text = model_info$level_name_text %||% NULL,
      factor_struct_matrix = model_info$factor_struct_matrix %||% NULL,
      level_struct_matrices = model_info$level_struct_matrices %||% NULL,
      default_factor_order = model_info$default_factor_order %||% NULL,
      factor_order_by_experiment = model_info$factor_order_by_experiment %||% NULL
    )
}

neural_fused_default_factor_struct_feature_names <- function() {
  c(
    "type_categorical",
    "type_ordinal",
    "type_numeric",
    "type_logical",
    "type_datetime",
    "ordered",
    "cardinality_log",
    "has_numeric_levels",
    "has_unit",
    "monotonic_candidate"
  )
}

neural_fused_default_level_struct_feature_names <- function() {
  c(
    "has_raw_value",
    "raw_value_log1p_signed",
    "level_rank01",
    "level_z_score",
    "level_quantile",
    "monotonic_candidate",
    "is_holdout"
  )
}

neural_default_level_names <- function(level_names = NULL, n_levels = NULL) {
  n_levels <- suppressWarnings(as.integer(n_levels %||% length(level_names %||% character(0))))
  if (length(n_levels) != 1L || is.na(n_levels) || n_levels < 0L) {
    n_levels <- 0L
  }
  names_use <- as.character(level_names %||% character(0))
  names_use <- names_use[nzchar(names_use)]
  if (length(names_use) >= n_levels) {
    return(names_use[seq_len(n_levels)])
  }
  if (n_levels < 1L) {
    return(character(0))
  }
  c(names_use, sprintf("__level_%d__", seq.int(length(names_use) + 1L, n_levels)))
}

neural_make_default_fused_structural_info <- function(names_list = NULL,
                                                     factor_levels = NULL,
                                                     factor_names = NULL,
                                                     level_names_list = NULL) {
  if (!is.null(names_list) && length(names_list) > 0L) {
    inferred_factor_names <- names(names_list)
    if (is.null(inferred_factor_names) || any(!nzchar(inferred_factor_names))) {
      inferred_factor_names <- sprintf("factor_%d", seq_along(names_list))
    }
    if (is.null(factor_names)) {
      factor_names <- inferred_factor_names
    }
    if (is.null(level_names_list)) {
      level_names_list <- lapply(names_list, function(x) {
        if (is.list(x) && length(x) > 0L) {
          return(as.character(x[[1L]] %||% character(0)))
        }
        as.character(x %||% character(0))
      })
    }
  }

  if (is.null(factor_names)) {
    n_factors <- max(length(factor_levels %||% integer(0)), length(level_names_list %||% list()))
    factor_names <- sprintf("factor_%d", seq_len(n_factors))
  }
  factor_names <- as.character(factor_names)
  n_factors <- length(factor_names)
  if (n_factors < 1L) {
    return(list(
      factor_struct_matrix = matrix(
        numeric(0),
        nrow = 0L,
        ncol = length(neural_fused_default_factor_struct_feature_names()),
        dimnames = list(NULL, neural_fused_default_factor_struct_feature_names())
      ),
      factor_struct_feature_names = neural_fused_default_factor_struct_feature_names(),
      level_struct_matrices = list(),
      level_struct_feature_names = neural_fused_default_level_struct_feature_names()
    ))
  }

  factor_levels <- as.integer(factor_levels %||% integer(0))
  level_names_container <- level_names_list %||% list()
  if (length(factor_levels) < n_factors) {
    inferred_counts <- vapply(seq_len(n_factors), function(i) {
      if (length(level_names_container) >= i) {
        length(level_names_container[[i]] %||% character(0))
      } else {
        0L
      }
    }, integer(1))
    factor_levels <- c(factor_levels, inferred_counts[(length(factor_levels) + 1L):n_factors])
  }
  factor_levels <- factor_levels[seq_len(n_factors)]
  factor_levels[is.na(factor_levels) | factor_levels < 1L] <- 1L

  factor_features <- c(
    neural_fused_default_factor_struct_feature_names(),
    sprintf("factor_identity_%d", seq_len(n_factors))
  )
  level_features <- c(
    neural_fused_default_level_struct_feature_names(),
    sprintf("level_factor_identity_%d", seq_len(n_factors))
  )
  factor_mat <- matrix(
    0,
    nrow = n_factors,
    ncol = length(factor_features),
    dimnames = list(factor_names, factor_features)
  )
  factor_mat[, "type_categorical"] <- 1
  factor_mat[, "cardinality_log"] <- log1p(factor_levels)
  factor_mat[, sprintf("factor_identity_%d", seq_len(n_factors))] <- diag(n_factors)

  level_mats <- setNames(vector("list", n_factors), factor_names)
  for (i in seq_len(n_factors)) {
    level_names <- neural_default_level_names(
      level_names = if (length(level_names_container) >= i) level_names_container[[i]] else NULL,
      n_levels = factor_levels[[i]]
    )
    level_mat <- matrix(
      0,
      nrow = factor_levels[[i]],
      ncol = length(level_features),
      dimnames = list(level_names, level_features)
    )
    if (factor_levels[[i]] > 1L) {
      rank01 <- (seq_len(factor_levels[[i]]) - 1) / (factor_levels[[i]] - 1)
      level_mat[, "level_rank01"] <- rank01
      level_mat[, "level_quantile"] <- rank01
      centered_rank <- (2 * rank01) - 1
      level_mat[, "level_z_score"] <- centered_rank
      level_mat[, sprintf("level_factor_identity_%d", i)] <- centered_rank
    }
    if (factor_levels[[i]] > 0L) {
      level_mat[, "is_holdout"] <- as.numeric(level_names %in% c("__holdout__", ".holdout", "holdout"))
    }
    level_mats[[i]] <- level_mat
  }

  list(
    factor_struct_matrix = factor_mat,
    factor_struct_feature_names = factor_features,
    level_struct_matrices = level_mats,
    level_struct_feature_names = level_features
  )
}

neural_make_prepared_prediction_model_info <- function(model_depth,
                                                       model_dims,
                                                       n_heads,
                                                       head_dim,
                                                       residual_mode = "standard",
                                                       cand_party_to_resp_idx = NULL,
                                                       n_party_levels = NULL,
                                                       n_candidate_tokens = NULL,
                                                       factor_tokenization = NULL,
                                                       max_factor_tokens = NULL,
                                                       factor_name_text = NULL,
                                                       level_name_text = NULL,
                                                       factor_order_by_experiment = NULL,
                                                       default_factor_order = NULL,
                                                       factor_struct_matrix = NULL,
                                                       level_struct_matrices = NULL,
                                                       factor_struct_feature_names = NULL,
                                                       level_struct_feature_names = NULL,
                                                       cross_candidate_encoder_mode = "none",
                                                       cross_candidate_encoder = FALSE,
                                                       likelihood = "bernoulli",
                                                       shared_projection_value_encoder = NULL,
                                                       covariate_value_stats_by_experiment = NULL,
                                                       default_covariate_value_stats = NULL,
                                                       covariate_value_metadata_by_experiment = NULL,
                                                       default_covariate_value_metadata = NULL,
                                                       covariate_value_text = NULL,
                                                       covariate_value_text_present = NULL,
                                                       covariate_value_type = NULL,
                                                       low_rank_interaction_rank = 0L,
                                                       low_rank_logit_transform = "none",
                                                       low_rank_logit_bound = NULL,
                                                       low_rank_logit_softness = NULL,
                                                       low_rank_logit_normalization = "none",
                                                       low_rank_head_weight_target_rms = NULL,
                                                       low_rank_rc_out_target_rms = NULL,
                                                       pairwise_antisymmetry = NULL,
                                                       additive_utility_mode = NULL,
                                                       additive_utility_normalization = NULL,
                                                       additive_head_weight_target_rms = NULL,
                                                       calibration_enabled = NULL,
                                                       calibration_method = NULL,
                                                       calibration_scale = NULL,
                                                       stage_mode = NULL,
                                                       attention_backend = "auto",
                                                       attention_dtype = "auto",
                                                       attention_padding_multiple = 8L,
                                                       attention_resolved_backend = NULL,
                                                       attention_fallback_reason = NULL,
                                                       jit_cache_key = NULL) {
  factor_tokenization <- neural_factor_tokenization(mode = factor_tokenization)
  factor_schema_supplied <- neural_factor_schema_supplied(
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    factor_struct_matrix = factor_struct_matrix,
    level_struct_matrices = level_struct_matrices,
    default_factor_order = default_factor_order,
    factor_order_by_experiment = factor_order_by_experiment
  )
  if (identical(factor_tokenization, "fused") && isTRUE(factor_schema_supplied)) {
    neural_validate_fused_structural_info(
      factor_struct_matrix = factor_struct_matrix,
      level_struct_matrices = level_struct_matrices,
      factor_struct_feature_names = factor_struct_feature_names,
      level_struct_feature_names = level_struct_feature_names,
      factor_name_text = factor_name_text,
      level_name_text = level_name_text,
      context = "prepared fused prediction model info"
    )
  }
  info <- neural_make_transformer_model_info(
    model_depth = model_depth,
    model_dims = model_dims,
    n_heads = n_heads,
    head_dim = head_dim,
    residual_mode = residual_mode,
    attention_backend = attention_backend,
    attention_dtype = attention_dtype,
    attention_padding_multiple = attention_padding_multiple,
    attention_resolved_backend = attention_resolved_backend,
    attention_fallback_reason = attention_fallback_reason
  )
  info$cand_party_to_resp_idx <- cand_party_to_resp_idx
  info$n_party_levels <- n_party_levels
  info$n_candidate_tokens <- n_candidate_tokens
  info$factor_tokenization <- factor_tokenization
  info$fused_token_mlp <- "swiglu"
  info$max_factor_tokens <- max_factor_tokens
  info$factor_name_text <- factor_name_text
  info$level_name_text <- level_name_text
  info$factor_schema_supplied <- isTRUE(factor_schema_supplied)
  info$factor_order_by_experiment <- factor_order_by_experiment
  info$default_factor_order <- default_factor_order
  info$factor_struct_matrix <- factor_struct_matrix
  info$level_struct_matrices <- level_struct_matrices
  info$factor_struct_feature_names <- factor_struct_feature_names
  info$level_struct_feature_names <- level_struct_feature_names
  info$cross_candidate_encoder_mode <- cross_candidate_encoder_mode
  info$cross_candidate_encoder <- cross_candidate_encoder
  info$likelihood <- likelihood
  info$shared_projection_value_encoder <- shared_projection_value_encoder
  info$covariate_value_stats_by_experiment <- covariate_value_stats_by_experiment
  info$default_covariate_value_stats <- default_covariate_value_stats
  info$covariate_value_metadata_by_experiment <- covariate_value_metadata_by_experiment
  info$default_covariate_value_metadata <- default_covariate_value_metadata
  info$covariate_value_text <- covariate_value_text
  info$covariate_value_text_present <- covariate_value_text_present
  info$covariate_value_type <- covariate_value_type
  info$low_rank_interaction_rank <- neural_resolve_low_rank_interaction_rank(
    low_rank_interaction_rank
  )
  info$low_rank_logit_transform <- neural_normalize_low_rank_logit_transform(
    low_rank_logit_transform %||% "none"
  )
  info$low_rank_logit_bound <- low_rank_logit_bound
  info$low_rank_logit_softness <- low_rank_logit_softness
  info$low_rank_logit_normalization <- neural_normalize_low_rank_logit_normalization(
    low_rank_logit_normalization %||% "none"
  )
  info$low_rank_head_weight_target_rms <- low_rank_head_weight_target_rms
  info$low_rank_rc_out_target_rms <- low_rank_rc_out_target_rms
  if (!is.null(pairwise_antisymmetry)) {
    info$pairwise_antisymmetry <- neural_resolve_pairwise_antisymmetry(pairwise_antisymmetry)
  }
  if (!is.null(additive_utility_mode)) {
    info$additive_utility_mode <- neural_resolve_additive_utility_mode(
      additive_utility_mode,
      factor_tokenization = factor_tokenization
    )
    info$additive_utility_normalization <- neural_resolve_additive_utility_normalization(
      value = additive_utility_normalization,
      additive_utility_mode = info$additive_utility_mode,
      supplied = !is.null(additive_utility_normalization)
    )
    info$additive_head_weight_target_rms <- neural_resolve_additive_head_weight_target_rms(
      value = additive_head_weight_target_rms,
      model_dims = model_dims,
      normalization = info$additive_utility_normalization,
      supplied = !is.null(additive_head_weight_target_rms)
    )
  }
  if (!is.null(calibration_enabled)) {
    info$calibration_enabled <- isTRUE(calibration_enabled)
  }
  if (!is.null(calibration_method)) {
    info$calibration_method <- neural_resolve_calibration_method(calibration_method)
  }
  if (!is.null(calibration_scale)) {
    info$calibration_scale <- calibration_scale
  }
  if (!is.null(stage_mode)) {
    info$stage_mode <- stage_mode
  }
  if (!is.null(jit_cache_key)) {
    info$jit_cache_key <- as.character(jit_cache_key)
  }
  info
}

neural_dim2 <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (neural_has_shape(x)) {
    return(as.integer(c(ai(x$shape[[1]]), ai(x$shape[[2]]))))
  }
  dim(as.matrix(x))
}

neural_validate_fused_structural_info <- function(factor_struct_matrix,
                                                  level_struct_matrices,
                                                  factor_struct_feature_names,
                                                  level_struct_feature_names,
                                                  factor_name_text = NULL,
                                                  level_name_text = NULL,
                                                  context = "fused tokenization") {
  required_missing <- c(
    if (is.null(factor_struct_matrix)) "factor_struct_matrix" else character(0),
    if (is.null(level_struct_matrices)) "level_struct_matrices" else character(0),
    if (is.null(factor_struct_feature_names)) "factor_struct_feature_names" else character(0),
    if (is.null(level_struct_feature_names)) "level_struct_feature_names" else character(0)
  )
  if (length(required_missing) > 0L) {
    stop(
      sprintf("%s requires structural token_info fields: %s.",
              context, paste(required_missing, collapse = ", ")),
      call. = FALSE
    )
  }

  factor_dims <- neural_dim2(factor_struct_matrix)
  if (length(factor_dims) != 2L ||
      any(is.na(factor_dims)) ||
      factor_dims[[1L]] < 1L ||
      factor_dims[[2L]] < 1L) {
    stop(
      sprintf("%s requires a non-empty factor_struct_matrix.", context),
      call. = FALSE
    )
  }
  factor_feature_names <- as.character(factor_struct_feature_names)
  if (length(factor_feature_names) != factor_dims[[2L]]) {
    stop(
      sprintf(
        "%s requires factor_struct_feature_names length (%d) to match factor_struct_matrix columns (%d).",
        context,
        length(factor_feature_names),
        factor_dims[[2L]]
      ),
      call. = FALSE
    )
  }

  n_factor_text <- {
    dims <- neural_dim2(factor_name_text)
    if (!is.null(dims)) dims[[1L]] else NA_integer_
  }
  if (!is.na(n_factor_text) && n_factor_text > 0L && n_factor_text != factor_dims[[1L]]) {
    stop(
      sprintf(
        "%s requires factor_struct_matrix rows (%d) to match factor_name_text rows (%d).",
        context,
        factor_dims[[1L]],
        n_factor_text
      ),
      call. = FALSE
    )
  }

  if (!is.list(level_struct_matrices) || length(level_struct_matrices) != factor_dims[[1L]]) {
    stop(
      sprintf(
        "%s requires level_struct_matrices to contain one matrix per factor (%d).",
        context,
        factor_dims[[1L]]
      ),
      call. = FALSE
    )
  }
  level_feature_names <- as.character(level_struct_feature_names)
  if (length(level_feature_names) < 1L) {
    stop(
      sprintf("%s requires non-empty level_struct_feature_names.", context),
      call. = FALSE
    )
  }
  for (d_ in seq_along(level_struct_matrices)) {
    level_dims <- neural_dim2(level_struct_matrices[[d_]])
    if (length(level_dims) != 2L ||
        any(is.na(level_dims)) ||
        level_dims[[1L]] < 1L ||
        level_dims[[2L]] != length(level_feature_names)) {
      stop(
        sprintf(
          "%s requires level_struct_matrices[[%d]] to have %d columns and at least one row.",
          context,
          d_,
          length(level_feature_names)
        ),
        call. = FALSE
      )
    }
    if (!is.null(level_name_text) && length(level_name_text) >= d_) {
      text_dims <- neural_dim2(level_name_text[[d_]])
      if (!is.null(text_dims) && text_dims[[1L]] > 0L && text_dims[[1L]] != level_dims[[1L]]) {
        stop(
          sprintf(
            "%s requires level_struct_matrices[[%d]] rows (%d) to match level_name_text[[%d]] rows (%d).",
            context,
            d_,
            level_dims[[1L]],
            d_,
            text_dims[[1L]]
          ),
          call. = FALSE
        )
      }
    }
  }
  invisible(TRUE)
}

neural_make_runtime_token_model_info <- function(model_dims,
                                                 cand_party_to_resp_idx = NULL,
                                                 n_party_levels = NULL,
                                                 factor_name_text = NULL,
                                                 level_name_text = NULL,
                                                 factor_struct_matrix = NULL,
                                                 level_struct_matrices = NULL,
                                                 factor_struct_feature_names = NULL,
                                                 level_struct_feature_names = NULL,
                                                 factor_order_by_experiment = NULL,
                                                 default_factor_order = NULL,
                                                 factor_tokenization = NULL,
                                                 max_factor_tokens = NULL,
                                                 covariate_name_text = NULL,
                                                 covariate_names = NULL,
                                                 resp_cov_mean = NULL,
                                                 resp_cov_scale = NULL,
                                                 resp_cov_default_present = NULL,
                                                 covariate_order_by_experiment = NULL,
                                                 default_covariate_order = NULL,
                                                 covariate_value_stats_by_experiment = NULL,
                                                 default_covariate_value_stats = NULL,
                                                 covariate_value_metadata_by_experiment = NULL,
                                                 default_covariate_value_metadata = NULL,
                                                 covariate_value_text = NULL,
                                                 covariate_value_text_present = NULL,
                                                 covariate_value_type = NULL,
                                                 max_covariate_tokens = NULL,
                                                 default_experiment_index = NULL,
                                                 token_family_levels = NULL,
                                                 experiment_token_mode = NULL,
                                                 covariate_value_encoding = NULL,
                                                 shared_projection_value_encoder = NULL,
                                                 experiment_description_text = NULL,
                                                 experiment_description_present = NULL,
                                                 default_experiment_text = NULL,
                                                 default_experiment_text_present = FALSE,
                                                 place_embedding = NULL,
                                                 place_present = NULL,
                                                 place_context_enabled = FALSE,
                                                 place_feature_names = NULL,
                                                 default_place_embedding = NULL,
                                                 default_place_present = FALSE,
                                                 time_embedding = NULL,
                                                 time_present = NULL,
                                                 time_context_enabled = FALSE,
                                                 time_feature_names = NULL,
                                                 default_time_embedding = NULL,
                                                 default_time_present = FALSE,
                                                 low_rank_interaction_rank = 0L,
                                                 low_rank_logit_transform = "none",
                                                 low_rank_logit_bound = NULL,
                                                 low_rank_logit_softness = NULL,
                                                 low_rank_logit_normalization = "none",
                                                 low_rank_head_weight_target_rms = NULL,
                                                 low_rank_rc_out_target_rms = NULL,
                                                 pairwise_antisymmetry = NULL,
                                                 additive_utility_mode = NULL,
                                                 additive_utility_normalization = NULL,
                                                 additive_head_weight_target_rms = NULL,
                                                 calibration_enabled = NULL,
                                                 calibration_method = NULL,
                                                 calibration_scale = NULL,
                                                 calibration_prior_sd = NULL,
                                                 likelihood = "bernoulli",
                                                 schema_dropout = NULL) {
  text_matrix <- function(x) neural_as_jnp_matrix(x, dtype = strenv$dtj)
  text_matrix_list <- function(x) {
    if (is.null(x)) {
      return(NULL)
    }
    lapply(x, text_matrix)
  }
  covariate_matrix_list <- function(x) {
    if (is.null(x)) {
      return(NULL)
    }
    lapply(x, function(x_i) neural_as_jnp_matrix(x_i, dtype = strenv$dtj))
  }
  factor_tokenization <- neural_factor_tokenization(mode = factor_tokenization)
  factor_schema_supplied <- neural_factor_schema_supplied(
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    factor_struct_matrix = factor_struct_matrix,
    level_struct_matrices = level_struct_matrices,
    default_factor_order = default_factor_order,
    factor_order_by_experiment = factor_order_by_experiment
  )
  if (identical(factor_tokenization, "fused") && isTRUE(factor_schema_supplied)) {
    neural_validate_fused_structural_info(
      factor_struct_matrix = factor_struct_matrix,
      level_struct_matrices = level_struct_matrices,
      factor_struct_feature_names = factor_struct_feature_names,
      level_struct_feature_names = level_struct_feature_names,
      factor_name_text = factor_name_text,
      level_name_text = level_name_text
    )
  }
  additive_utility_mode_resolved <- neural_resolve_additive_utility_mode(
    additive_utility_mode %||% "off",
    factor_tokenization = factor_tokenization
  )
  additive_utility_normalization_resolved <- neural_resolve_additive_utility_normalization(
    value = additive_utility_normalization,
    additive_utility_mode = additive_utility_mode_resolved,
    supplied = !is.null(additive_utility_normalization)
  )
  additive_head_weight_target_rms_resolved <- neural_resolve_additive_head_weight_target_rms(
    value = additive_head_weight_target_rms,
    model_dims = model_dims,
    normalization = additive_utility_normalization_resolved,
    supplied = !is.null(additive_head_weight_target_rms)
  )

  list(
    model_dims = model_dims,
    cand_party_to_resp_idx = cand_party_to_resp_idx,
    n_party_levels = n_party_levels,
    factor_name_text = text_matrix(factor_name_text),
    level_name_text = text_matrix_list(level_name_text),
    factor_struct_matrix = text_matrix(factor_struct_matrix),
    factor_struct_feature_names = as.character(factor_struct_feature_names %||% character(0)),
    level_struct_matrices = text_matrix_list(level_struct_matrices),
    level_struct_feature_names = as.character(level_struct_feature_names %||% character(0)),
    factor_schema_supplied = isTRUE(factor_schema_supplied),
    factor_order_by_experiment = factor_order_by_experiment,
    default_factor_order = if (is.null(default_factor_order)) {
      NULL
    } else {
      as.integer(default_factor_order)
    },
    factor_tokenization = factor_tokenization,
    fused_token_mlp = "swiglu",
    max_factor_tokens = if (is.null(max_factor_tokens)) {
      NULL
    } else {
      as.integer(max_factor_tokens)
    },
    covariate_name_text = text_matrix(covariate_name_text),
    covariate_names = as.character(covariate_names %||% character(0)),
    resp_cov_mean = neural_as_jnp_vector(resp_cov_mean, dtype = strenv$dtj),
    resp_cov_scale = neural_as_jnp_vector(resp_cov_scale, dtype = strenv$dtj),
    resp_cov_default_present = neural_as_jnp_vector(resp_cov_default_present, dtype = strenv$dtj),
    covariate_order_by_experiment = covariate_order_by_experiment,
    default_covariate_order = if (is.null(default_covariate_order)) {
      NULL
    } else {
      as.integer(default_covariate_order)
    },
    covariate_value_stats_by_experiment = covariate_matrix_list(covariate_value_stats_by_experiment),
    default_covariate_value_stats = neural_as_jnp_matrix(default_covariate_value_stats, dtype = strenv$dtj),
    covariate_value_metadata_by_experiment = covariate_matrix_list(covariate_value_metadata_by_experiment),
    default_covariate_value_metadata = neural_as_jnp_matrix(default_covariate_value_metadata, dtype = strenv$dtj),
    covariate_value_text = neural_as_jnp_array(covariate_value_text, dtype = strenv$dtj),
    covariate_value_text_present = neural_as_jnp_matrix(covariate_value_text_present, dtype = strenv$dtj),
    covariate_value_type = neural_as_jnp_vector(covariate_value_type, dtype = strenv$jnp$int32),
    max_covariate_tokens = if (is.null(max_covariate_tokens)) {
      NULL
    } else {
      as.integer(max_covariate_tokens)
    },
    default_experiment_index = if (is.null(default_experiment_index) || is.na(default_experiment_index)) {
      NULL
    } else {
      as.integer(default_experiment_index)
    },
    token_family_levels = token_family_levels,
    experiment_token_mode = experiment_token_mode,
    covariate_value_encoding = covariate_value_encoding,
    shared_projection_value_encoder = shared_projection_value_encoder,
    experiment_description_text = text_matrix(experiment_description_text),
    experiment_description_present = neural_as_jnp_vector(experiment_description_present, dtype = strenv$dtj),
    default_experiment_text = text_matrix(default_experiment_text),
    default_experiment_text_present = isTRUE(default_experiment_text_present),
    place_embedding = text_matrix(place_embedding),
    place_present = neural_as_jnp_vector(place_present, dtype = strenv$dtj),
    place_context_enabled = isTRUE(place_context_enabled),
    place_feature_names = as.character(place_feature_names %||% neural_place_feature_names()),
    default_place_embedding = text_matrix(default_place_embedding),
    default_place_present = isTRUE(default_place_present),
    time_embedding = text_matrix(time_embedding),
    time_present = neural_as_jnp_vector(time_present, dtype = strenv$dtj),
    time_context_enabled = isTRUE(time_context_enabled),
    time_feature_names = as.character(time_feature_names %||% neural_time_feature_names()),
    default_time_embedding = text_matrix(default_time_embedding),
    default_time_present = isTRUE(default_time_present),
    low_rank_interaction_rank = neural_resolve_low_rank_interaction_rank(
      low_rank_interaction_rank
    ),
    low_rank_logit_transform = neural_normalize_low_rank_logit_transform(
      low_rank_logit_transform %||% "none"
    ),
    low_rank_logit_bound = low_rank_logit_bound,
    low_rank_logit_softness = low_rank_logit_softness,
    low_rank_logit_normalization = neural_normalize_low_rank_logit_normalization(
      low_rank_logit_normalization %||% "none"
    ),
    low_rank_head_weight_target_rms = low_rank_head_weight_target_rms,
    low_rank_rc_out_target_rms = low_rank_rc_out_target_rms,
    pairwise_antisymmetry = neural_resolve_pairwise_antisymmetry(pairwise_antisymmetry %||% "legacy"),
    additive_utility_mode = additive_utility_mode_resolved,
    additive_utility_normalization = additive_utility_normalization_resolved,
    additive_head_weight_target_rms = additive_head_weight_target_rms_resolved,
    calibration_enabled = isTRUE(calibration_enabled),
    calibration_method = neural_resolve_calibration_method(
      calibration_method %||% if (isTRUE(calibration_enabled)) "logit_scale" else "none"
    ),
    calibration_scale = calibration_scale,
    calibration_prior_sd = calibration_prior_sd,
    likelihood = tolower(as.character(likelihood %||% "bernoulli")),
    schema_dropout = neural_resolve_schema_dropout(schema_dropout)
  )
}

neural_set_pairwise_context_model_info <- function(info,
                                                   pairwise_context_mode = "stage_free",
                                                   has_candidate_group_context = FALSE,
                                                   has_respondent_group_context = FALSE,
                                                   has_relation_token_context = FALSE,
                                                   has_stage_context = FALSE,
                                                   has_matchup_context = FALSE,
                                                   party_missing_label = neural_missing_group_label("candidate"),
                                                   resp_party_missing_label = neural_missing_group_label("respondent"),
                                                   n_resp_party_levels = NULL,
                                                   party_missing_index = NULL,
                                                   resp_party_missing_index = NULL,
                                                   context_present_masking = TRUE) {
  info$pairwise_context_mode <- neural_pairwise_context_mode(
    list(pairwise_context_mode = pairwise_context_mode)
  )
  info$has_candidate_group_context <- isTRUE(has_candidate_group_context)
  info$has_respondent_group_context <- isTRUE(has_respondent_group_context)
  info$has_relation_token_context <- isTRUE(has_relation_token_context)
  info$has_stage_context <- isTRUE(has_stage_context)
  info$has_matchup_context <- isTRUE(has_matchup_context)
  info$party_missing_label <- as.character(party_missing_label %||%
    neural_missing_group_label("candidate"))
  info$resp_party_missing_label <- as.character(resp_party_missing_label %||%
    neural_missing_group_label("respondent"))
  if (!is.null(n_resp_party_levels)) {
    info$n_resp_party_levels <- as.integer(n_resp_party_levels)
  }
  if (!is.null(party_missing_index)) {
    info$party_missing_index <- as.integer(party_missing_index)
  }
  if (!is.null(resp_party_missing_index)) {
    info$resp_party_missing_index <- as.integer(resp_party_missing_index)
  }
  info$context_present_masking <- isTRUE(context_present_masking)
  info
}

neural_resolve_covariate_value_encoding <- function(mode = NULL) {
  mode_use <- tolower(as.character(mode %||% "shared_projection"))
  if (!identical(mode_use, "shared_projection")) {
    stop(
      "'covariate_value_encoding' must be 'shared_projection'.",
      call. = FALSE
    )
  }
  mode_use
}

neural_resolve_shared_projection_value_encoder <- function(mode = NULL) {
  mode_use <- tolower(as.character(mode %||% "name_dist_moe"))
  if (!mode_use %in% c("name_dist_moe", "legacy_scalar")) {
    stop(
      "'shared_projection_value_encoder' must be one of 'name_dist_moe' or 'legacy_scalar'.",
      call. = FALSE
    )
  }
  mode_use
}

neural_resolve_low_rank_interaction_rank <- function(value = NULL) {
  if (is.null(value) || identical(value, FALSE)) {
    return(0L)
  }
  if (identical(value, TRUE)) {
    return(16L)
  }
  rank <- suppressWarnings(as.integer(value[[1L]]))
  if (is.na(rank) || rank < 0L) {
    stop(
      "'low_rank_interaction_rank' must be a non-negative integer.",
      call. = FALSE
    )
  }
  as.integer(rank)
}

neural_resolve_pairwise_antisymmetry <- function(value = NULL) {
  if (is.null(value)) {
    return("strict")
  }
  if (isTRUE(value)) {
    return("strict")
  }
  if (identical(value, FALSE)) {
    return("legacy")
  }
  if (is.character(value)) {
    mode <- tolower(trimws(as.character(value[[1L]])))
    if (mode %in% c("strict", "on", "true", "yes", "antisymmetric", "antisym")) {
      return("strict")
    }
    if (mode %in% c("legacy", "off", "false", "no", "none")) {
      return("legacy")
    }
  }
  stop(
    "'pairwise_antisymmetry' must be TRUE/FALSE or one of 'strict'/'legacy'.",
    call. = FALSE
  )
}

neural_pairwise_antisymmetry_strict <- function(model_info = NULL) {
  if (is.null(model_info)) {
    return(FALSE)
  }
  identical(
    neural_resolve_pairwise_antisymmetry(model_info$pairwise_antisymmetry %||% "legacy"),
    "strict"
  )
}

neural_resolve_additive_utility_mode <- function(value = NULL,
                                                 factor_tokenization = NULL,
                                                 pairwise_mode = FALSE) {
  if (is.null(value)) {
    value <- "auto"
  }
  if (isTRUE(value)) {
    return("on")
  }
  if (identical(value, FALSE)) {
    return("off")
  }
  if (is.character(value)) {
    mode <- tolower(trimws(as.character(value[[1L]])))
    if (mode %in% c("on", "true", "yes", "additive", "main_effects", "main-effects")) {
      return("on")
    }
    if (mode %in% c("off", "false", "no", "none")) {
      return("off")
    }
    if (mode %in% c("auto", "default")) {
      factor_mode <- tolower(as.character(factor_tokenization %||% "fused"))
      return(if (identical(factor_mode, "fused")) "on" else "off")
    }
  }
  stop(
    "'additive_utility' must be TRUE/FALSE or one of 'auto', 'on', or 'off'.",
    call. = FALSE
  )
}

neural_additive_utility_enabled <- function(model_info = NULL, params = NULL) {
  if (is.null(model_info)) {
    return(FALSE)
  }
  mode <- neural_resolve_additive_utility_mode(
    model_info$additive_utility_mode %||% "off",
    factor_tokenization = model_info$factor_tokenization %||% NULL
  )
  identical(mode, "on") &&
    !is.null(params) &&
    !is.null(params$W_add_out)
}

neural_normalize_additive_utility_normalization <- function(value = NULL) {
  if (is.null(value)) {
    return("none")
  }
  if (isTRUE(value)) {
    return("rms")
  }
  if (identical(value, FALSE)) {
    return("none")
  }
  if (is.character(value)) {
    mode <- tolower(as.character(value))
    if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      if (mode %in% c("rms", "rmsnorm", "rms_norm", "rms-normalize", "rms_normalize")) {
        return("rms")
      }
      if (mode %in% c("none", "identity", "off", "false")) {
        return("none")
      }
    }
  }
  NA_character_
}

neural_resolve_additive_utility_normalization <- function(value = NULL,
                                                          additive_utility_mode = "off",
                                                          supplied = FALSE) {
  mode <- neural_resolve_additive_utility_mode(additive_utility_mode %||% "off")
  if (!identical(mode, "on")) {
    return("none")
  }
  normalization <- if (isTRUE(supplied)) {
    neural_normalize_additive_utility_normalization(value)
  } else {
    "rms"
  }
  if (length(normalization) != 1L ||
      is.na(normalization) ||
      !normalization %in% c("rms", "none")) {
    stop(
      "'neural_mcmc_control$additive_utility_normalization' must be 'rms' or 'none'.",
      call. = FALSE
    )
  }
  normalization
}

neural_resolve_additive_head_weight_target_rms <- function(value = NULL,
                                                           model_dims = NULL,
                                                           normalization = "none",
                                                           supplied = FALSE) {
  if (!identical(normalization, "rms")) {
    return(NULL)
  }
  if (!isTRUE(supplied) || is.null(value)) {
    dims <- suppressWarnings(as.numeric(model_dims))
    if (length(dims) != 1L || !is.finite(dims) || dims <= 0) {
      dims <- 1
    }
    return(1 / sqrt(dims))
  }
  if (length(value) != 1L) {
    stop("'additive_head_weight_target_rms' must be a positive scalar numeric value.", call. = FALSE)
  }
  target <- suppressWarnings(as.numeric(value))
  if (is.na(target) || !is.finite(target) || target <= 0) {
    stop("'additive_head_weight_target_rms' must be a positive scalar numeric value.", call. = FALSE)
  }
  as.numeric(target)
}

neural_additive_utility_normalization_enabled <- function(model_info = NULL) {
  if (is.null(model_info)) {
    return(FALSE)
  }
  mode <- neural_resolve_additive_utility_mode(
    model_info$additive_utility_mode %||% "off",
    factor_tokenization = model_info$factor_tokenization %||% NULL
  )
  normalization <- neural_normalize_additive_utility_normalization(
    model_info$additive_utility_normalization %||% "none"
  )
  target <- suppressWarnings(as.numeric(model_info$additive_head_weight_target_rms %||% NA_real_))
  identical(mode, "on") &&
    identical(normalization, "rms") &&
    length(target) == 1L && is.finite(target) && target > 0
}

neural_coerce_additive_utility_metadata <- function(model_info,
                                                    legacy_missing_normalization = "none") {
  if (is.null(model_info)) {
    return(NULL)
  }
  out <- model_info
  out$additive_utility_mode <- neural_resolve_additive_utility_mode(
    out$additive_utility_mode %||% "off",
    factor_tokenization = out$factor_tokenization %||% NULL
  )
  normalization_supplied <- !is.null(out$additive_utility_normalization)
  normalization <- if (isTRUE(normalization_supplied)) {
    neural_normalize_additive_utility_normalization(out$additive_utility_normalization)
  } else {
    neural_normalize_additive_utility_normalization(legacy_missing_normalization)
  }
  if (length(normalization) != 1L ||
      is.na(normalization) ||
      !normalization %in% c("rms", "none")) {
    normalization <- "none"
  }
  if (!identical(out$additive_utility_mode, "on")) {
    normalization <- "none"
  }
  target <- NULL
  if (identical(normalization, "rms")) {
    target_supplied <- !is.null(out$additive_head_weight_target_rms)
    target <- tryCatch(
      neural_resolve_additive_head_weight_target_rms(
        value = out$additive_head_weight_target_rms,
        model_dims = out$model_dims %||% NULL,
        normalization = normalization,
        supplied = target_supplied
      ),
      error = function(e) NULL
    )
    if (is.null(target)) {
      normalization <- "none"
    }
  }
  out$additive_utility_normalization <- normalization
  out$additive_head_weight_target_rms <- target
  out
}

neural_resolve_calibration_method <- function(value = NULL) {
  if (is.null(value)) {
    return("logit_scale")
  }
  if (isTRUE(value)) {
    return("logit_scale")
  }
  if (identical(value, FALSE)) {
    return("none")
  }
  if (is.character(value)) {
    mode <- tolower(trimws(as.character(value[[1L]])))
    if (mode %in% c("logit_scale", "logit-scale", "temperature", "scale")) {
      return("logit_scale")
    }
    if (mode %in% c("none", "off", "false")) {
      return("none")
    }
  }
  stop(
    "'calibration$method' must be 'logit_scale' or 'none'.",
    call. = FALSE
  )
}

neural_resolve_calibration_control <- function(value = NULL,
                                               likelihood = NULL,
                                               legacy_pairwise_scale = FALSE,
                                               prior_sd_fallback = 0.5) {
  value <- value %||% list()
  if (isTRUE(value)) {
    value <- list(enabled = TRUE)
  } else if (identical(value, FALSE)) {
    value <- list(enabled = FALSE)
  } else if (!is.list(value)) {
    stop("'neural_mcmc_control$calibration' must be TRUE/FALSE or a list.", call. = FALSE)
  }
  likelihood_use <- tolower(as.character(likelihood %||% "bernoulli"))
  eligible <- likelihood_use %in% c("bernoulli", "categorical", "ordinal", "mixed")
  enabled_raw <- value$enabled %||% "auto"
  enabled <- if (is.character(enabled_raw)) {
    tag <- tolower(trimws(as.character(enabled_raw[[1L]])))
    if (tag %in% c("auto", "default")) {
      eligible || isTRUE(legacy_pairwise_scale)
    } else if (tag %in% c("true", "t", "yes", "y", "1", "on", "learned")) {
      TRUE
    } else if (tag %in% c("false", "f", "no", "n", "0", "off", "none")) {
      FALSE
    } else {
      stop("'calibration$enabled' must be TRUE/FALSE or 'auto'.", call. = FALSE)
    }
  } else {
    isTRUE(enabled_raw)
  }
  method <- neural_resolve_calibration_method(value$method %||% "logit_scale")
  if (!isTRUE(enabled) || identical(method, "none") || !isTRUE(eligible || legacy_pairwise_scale)) {
    return(list(enabled = FALSE, method = "none", prior_sd = NULL))
  }
  prior_sd <- suppressWarnings(as.numeric(value$prior_sd %||% prior_sd_fallback))
  if (length(prior_sd) != 1L || is.na(prior_sd) || !is.finite(prior_sd) || prior_sd <= 0) {
    stop("'calibration$prior_sd' must be a positive finite scalar.", call. = FALSE)
  }
  list(enabled = TRUE, method = method, prior_sd = as.numeric(prior_sd))
}

neural_validate_calibration_temperature_compatibility <- function(learned_pairwise_bernoulli_logit_scale,
                                                                  calibration_control) {
  if (isTRUE(learned_pairwise_bernoulli_logit_scale) &&
      isTRUE(calibration_control$enabled) &&
      identical(calibration_control$method %||% NULL, "logit_scale")) {
    stop(
      "neural_mcmc_control cannot learn both pairwise_bernoulli_logit_scale and calibration logit_scale; disable one.",
      call. = FALSE
    )
  }
  invisible(calibration_control)
}

neural_resolve_learned_pairwise_bernoulli_logit_scale <- function(value = NULL) {
  if (is.null(value)) {
    return(FALSE)
  }
  if (is.logical(value)) {
    if (length(value) != 1L || is.na(value)) {
      stop(
        "'learned_pairwise_bernoulli_logit_scale' must be TRUE or FALSE.",
        call. = FALSE
      )
    }
    return(isTRUE(value))
  }
  if (is.character(value)) {
    mode <- tolower(trimws(as.character(value[[1L]])))
    if (mode %in% c("true", "t", "yes", "y", "1", "learned", "on")) {
      return(TRUE)
    }
    if (mode %in% c("false", "f", "no", "n", "0", "none", "off")) {
      return(FALSE)
    }
  }
  stop(
    "'learned_pairwise_bernoulli_logit_scale' must be TRUE/FALSE or 'learned'/'off'.",
    call. = FALSE
  )
}

neural_resolve_pairwise_bernoulli_logit_scale_prior_sd <- function(value = NULL,
                                                                   enabled = FALSE) {
  if (!isTRUE(enabled)) {
    return(NULL)
  }
  prior_sd <- if (is.null(value)) 0.5 else suppressWarnings(as.numeric(value[[1L]]))
  if (length(prior_sd) != 1L || is.na(prior_sd) || !is.finite(prior_sd) || prior_sd <= 0) {
    stop(
      "'pairwise_bernoulli_logit_scale_prior_sd' must be a positive finite scalar.",
      call. = FALSE
    )
  }
  as.numeric(prior_sd)
}

neural_normalize_low_rank_logit_transform <- function(value = NULL) {
  if (is.null(value)) {
    return("none")
  }
  if (isTRUE(value)) {
    return("softclip")
  }
  if (identical(value, FALSE)) {
    return("none")
  }
  if (is.character(value)) {
    mode <- tolower(as.character(value))
    if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      if (mode %in% c("softclip", "soft_clip", "soft-clip")) {
        return("softclip")
      }
      if (mode %in% c("none", "identity", "off", "false")) {
        return("none")
      }
    }
  }
  NA_character_
}

neural_resolve_low_rank_logit_bound <- function(value = NULL,
                                                low_rank_interaction_rank = 0L,
                                                supplied = FALSE,
                                                transform = "softclip") {
  transform <- neural_normalize_low_rank_logit_transform(transform)
  if (!identical(transform, "softclip")) {
    return(NULL)
  }
  rank <- neural_resolve_low_rank_interaction_rank(low_rank_interaction_rank)
  if (rank <= 0L) {
    return(NULL)
  }
  if (!isTRUE(supplied)) {
    return(1.5)
  }
  if (is.null(value)) {
    return(NULL)
  }
  if (length(value) != 1L) {
    stop("'low_rank_logit_bound' must be a scalar numeric value.", call. = FALSE)
  }
  bound <- suppressWarnings(as.numeric(value))
  if (is.na(bound)) {
    stop("'low_rank_logit_bound' must be a scalar numeric value.", call. = FALSE)
  }
  if (!is.finite(bound) || bound <= 0) {
    return(NULL)
  }
  as.numeric(bound)
}

neural_resolve_low_rank_logit_softness <- function(value = NULL,
                                                   bound = NULL,
                                                   supplied = FALSE) {
  if (is.null(bound)) {
    return(NULL)
  }
  if (!isTRUE(supplied) || is.null(value)) {
    return(as.numeric(bound) / 6)
  }
  if (length(value) != 1L) {
    stop("'low_rank_logit_softness' must be a positive scalar numeric value.", call. = FALSE)
  }
  softness <- suppressWarnings(as.numeric(value))
  if (is.na(softness) || !is.finite(softness) || softness <= 0) {
    stop("'low_rank_logit_softness' must be a positive scalar numeric value.", call. = FALSE)
  }
  as.numeric(softness)
}

neural_normalize_low_rank_logit_normalization <- function(value = NULL) {
  if (is.null(value)) {
    return("none")
  }
  if (isTRUE(value)) {
    return("rms")
  }
  if (identical(value, FALSE)) {
    return("none")
  }
  if (is.character(value)) {
    mode <- tolower(as.character(value))
    if (length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      if (mode %in% c("rms", "rmsnorm", "rms_norm", "rms-normalize", "rms_normalize")) {
        return("rms")
      }
      if (mode %in% c("none", "identity", "off", "false")) {
        return("none")
      }
    }
  }
  NA_character_
}

neural_resolve_low_rank_logit_normalization <- function(value = NULL,
                                                        supplied = FALSE,
                                                        low_rank_interaction_rank = 0L,
                                                        pairwise_mode = FALSE,
                                                        likelihood = "bernoulli") {
  rank <- neural_resolve_low_rank_interaction_rank(low_rank_interaction_rank)
  likelihood_use <- tolower(as.character(likelihood %||% "bernoulli"))
  eligible <- rank > 0L &&
    isTRUE(pairwise_mode) &&
    likelihood_use %in% c("bernoulli", "mixed")
  mode <- if (isTRUE(supplied)) {
    neural_normalize_low_rank_logit_normalization(value)
  } else if (isTRUE(eligible)) {
    "rms"
  } else {
    "none"
  }
  if (length(mode) != 1L || is.na(mode) || !mode %in% c("rms", "none")) {
    stop(
      "'neural_mcmc_control$low_rank_logit_normalization' must be 'rms' or 'none'.",
      call. = FALSE
    )
  }
  if (!isTRUE(eligible)) {
    return("none")
  }
  mode
}

neural_resolve_low_rank_head_weight_target_rms <- function(value = NULL,
                                                           model_dims = NULL,
                                                           normalization = "none",
                                                           supplied = FALSE) {
  if (!identical(normalization, "rms")) {
    return(NULL)
  }
  if (!isTRUE(supplied) || is.null(value)) {
    dims <- suppressWarnings(as.numeric(model_dims))
    if (length(dims) != 1L || !is.finite(dims) || dims <= 0) {
      dims <- 1
    }
    return(1 / (sqrt(2) * dims))
  }
  if (length(value) != 1L) {
    stop("'low_rank_head_weight_target_rms' must be a positive scalar numeric value.", call. = FALSE)
  }
  target <- suppressWarnings(as.numeric(value))
  if (is.na(target) || !is.finite(target) || target <= 0) {
    stop("'low_rank_head_weight_target_rms' must be a positive scalar numeric value.", call. = FALSE)
  }
  as.numeric(target)
}

neural_resolve_low_rank_rc_out_target_rms <- function(value = NULL,
                                                      low_rank_interaction_rank = 0L,
                                                      normalization = "none",
                                                      supplied = FALSE) {
  if (!identical(normalization, "rms")) {
    return(NULL)
  }
  if (!isTRUE(supplied) || is.null(value)) {
    rank <- max(1L, neural_resolve_low_rank_interaction_rank(low_rank_interaction_rank))
    return(1 / (sqrt(2) * rank))
  }
  if (length(value) != 1L) {
    stop("'low_rank_rc_out_target_rms' must be a positive scalar numeric value.", call. = FALSE)
  }
  target <- suppressWarnings(as.numeric(value))
  if (is.na(target) || !is.finite(target) || target <= 0) {
    stop("'low_rank_rc_out_target_rms' must be a positive scalar numeric value.", call. = FALSE)
  }
  as.numeric(target)
}

neural_low_rank_logit_transform_enabled <- function(model_info = NULL) {
  if (is.null(model_info)) {
    return(FALSE)
  }
  rank <- neural_low_rank_interaction_rank(model_info)
  transform <- neural_normalize_low_rank_logit_transform(
    model_info$low_rank_logit_transform %||% "none"
  )
  bound <- suppressWarnings(as.numeric(model_info$low_rank_logit_bound %||% NA_real_))
  softness <- suppressWarnings(as.numeric(model_info$low_rank_logit_softness %||% NA_real_))
  rank > 0L &&
    identical(transform, "softclip") &&
    length(bound) == 1L && is.finite(bound) && bound > 0 &&
    length(softness) == 1L && is.finite(softness) && softness > 0
}

neural_low_rank_logit_normalization_enabled <- function(model_info = NULL,
                                                        pairwise_obs = FALSE) {
  if (is.null(model_info) || !isTRUE(pairwise_obs)) {
    return(FALSE)
  }
  rank <- neural_low_rank_interaction_rank(model_info)
  normalization <- neural_normalize_low_rank_logit_normalization(
    model_info$low_rank_logit_normalization %||% "none"
  )
  likelihood <- tolower(as.character(model_info$likelihood %||% "bernoulli"))
  head_target <- suppressWarnings(as.numeric(model_info$low_rank_head_weight_target_rms %||% NA_real_))
  rc_target <- suppressWarnings(as.numeric(model_info$low_rank_rc_out_target_rms %||% NA_real_))
  rank > 0L &&
    identical(normalization, "rms") &&
    likelihood %in% c("bernoulli", "mixed") &&
    length(head_target) == 1L && is.finite(head_target) && head_target > 0 &&
    length(rc_target) == 1L && is.finite(rc_target) && rc_target > 0
}

neural_resolve_rms_scale <- function(value = NULL) {
  scale <- suppressWarnings(as.numeric(value %||% 0.25))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    stop("'neural_mcmc_control$RMS_scale' must be a positive finite scalar.", call. = FALSE)
  }
  as.numeric(scale)
}

neural_resolve_qk_rms_scale <- function(value = NULL) {
  scale <- suppressWarnings(as.numeric(value %||% 0.1))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    stop("'neural_mcmc_control$qk_rms_scale' must be a positive finite scalar.", call. = FALSE)
  }
  as.numeric(scale)
}

neural_resolve_moe_load_balance_lambda <- function(value = NULL) {
  lambda <- suppressWarnings(as.numeric(value %||% 0.01))
  if (length(lambda) != 1L || is.na(lambda) || !is.finite(lambda) || lambda < 0) {
    return(0.01)
  }
  as.numeric(lambda)
}

# Emit the MoE load-balancing penalty collected during this model trace as ONE
# numpyro.factor at model-body level (outside any data plate), scaled by
# n_obs_total * lambda so the penalty tracks the likelihood magnitude
# identically under plate subsampling and compact (obs_scale) training. The
# mean over collected terms keeps the weight independent of how many times the
# covariate encoder ran within the trace (forward/reversed pass, low-rank
# tower). Clears the collector; a no-op when training is inactive.
neural_emit_moe_load_balance_factor <- function(n_obs_total) {
  pending <- strenv$moe_aux_pending
  strenv$moe_aux_pending <- list()
  if (!isTRUE(strenv$moe_aux_active) ||
      is.null(pending) || length(pending) < 1L) {
    return(invisible(NULL))
  }
  moe_lambda <- suppressWarnings(as.numeric(strenv$moe_load_balance_lambda %||% 0))
  if (length(moe_lambda) != 1L || !is.finite(moe_lambda) || moe_lambda <= 0) {
    return(invisible(NULL))
  }
  n_scale <- suppressWarnings(as.numeric(n_obs_total))
  if (length(n_scale) != 1L || !is.finite(n_scale) || n_scale < 1) {
    n_scale <- 1
  }
  aux_total <- Reduce(`+`, pending)
  if (length(pending) > 1L) {
    aux_total <- aux_total / as.numeric(length(pending))
  }
  scale_val <- strenv$jnp$array(moe_lambda * n_scale, dtype = strenv$dtj)
  strenv$numpyro$factor(
    "moe_load_balance",
    strenv$jnp$negative(scale_val * aux_total)
  )
  invisible(NULL)
}

neural_resolve_residual_weight_depth_scale <- function(value = NULL) {
  mode <- tolower(as.character(value %||% "floor_one"))
  mode <- gsub("-", "_", mode, fixed = TRUE)
  if (length(mode) != 1L || is.na(mode) || !nzchar(mode) ||
      !mode %in% c("floor_one", "legacy", "none")) {
    stop(
      "'neural_mcmc_control$residual_weight_depth_scale' must be one of ",
      "'floor_one', 'legacy', or 'none'.",
      call. = FALSE
    )
  }
  mode
}

neural_resolve_init_policy <- function(model_depth,
                                       model_dims,
                                       RMS_scale = NULL,
                                       residual_weight_depth_scale = NULL,
                                       qk_rms_scale = NULL) {
  depth <- suppressWarnings(as.numeric(model_depth))
  dims <- suppressWarnings(as.numeric(model_dims))
  if (length(depth) != 1L || is.na(depth) || !is.finite(depth) || depth < 1) {
    stop("'neural_mcmc_control$ModelDepth' must be an integer >= 1.", call. = FALSE)
  }
  if (length(dims) != 1L || is.na(dims) || !is.finite(dims) || dims < 1) {
    stop("'neural_mcmc_control$ModelDims' must be an integer >= 1.", call. = FALSE)
  }
  rms_scale <- neural_resolve_rms_scale(RMS_scale)
  qk_rms_scale_resolved <- neural_resolve_qk_rms_scale(qk_rms_scale)
  residual_policy <- neural_resolve_residual_weight_depth_scale(residual_weight_depth_scale)
  depth_prior_scale <- sqrt(2) / sqrt(depth)
  residual_weight_multiplier <- switch(
    residual_policy,
    floor_one = max(1, depth_prior_scale),
    legacy = depth_prior_scale,
    none = 1
  )
  weight_sd_scale <- sqrt(2) / sqrt(dims)
  list(
    RMS_scale = rms_scale,
    qk_rms_scale = qk_rms_scale_resolved,
    residual_weight_depth_scale = residual_policy,
    depth_prior_scale = as.numeric(depth_prior_scale),
    residual_weight_multiplier = as.numeric(residual_weight_multiplier),
    gate_depth_multiplier = as.numeric(depth_prior_scale),
    weight_sd_scale = as.numeric(weight_sd_scale),
    residual_weight_sd_scale = as.numeric(weight_sd_scale * residual_weight_multiplier),
    gate_sd_scale = as.numeric(0.1 * depth_prior_scale)
  )
}

neural_pairwise_bernoulli_logit_scale_enabled <- function(model_info = NULL) {
  if (is.null(model_info)) {
    return(FALSE)
  }
  isTRUE(model_info$learned_pairwise_bernoulli_logit_scale)
}

neural_pairwise_bernoulli_logit_scale_from_params <- function(params = NULL,
                                                              model_info = NULL) {
  if (!isTRUE(neural_pairwise_bernoulli_logit_scale_enabled(model_info))) {
    return(NULL)
  }
  if (!is.null(params$pairwise_bernoulli_logit_scale)) {
    return(params$pairwise_bernoulli_logit_scale)
  }
  if (!is.null(params$log_pairwise_bernoulli_logit_scale)) {
    return(strenv$jnp$exp(params$log_pairwise_bernoulli_logit_scale))
  }
  scale <- suppressWarnings(as.numeric(model_info$pairwise_bernoulli_logit_scale %||% NA_real_))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    return(NULL)
  }
  strenv$jnp$array(scale, dtype = strenv$dtj)
}

neural_calibration_scale_from_params <- function(params = NULL,
                                                 model_info = NULL) {
  if (is.null(model_info) ||
      !isTRUE(model_info$calibration_enabled) ||
      !identical(model_info$calibration_method %||% NULL, "logit_scale")) {
    return(NULL)
  }
  if (!is.null(params$calibration_scale)) {
    return(params$calibration_scale)
  }
  if (!is.null(params$log_calibration_scale)) {
    return(strenv$jnp$exp(params$log_calibration_scale))
  }
  scale <- suppressWarnings(as.numeric(model_info$calibration_scale %||% NA_real_))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    return(NULL)
  }
  strenv$jnp$array(scale, dtype = strenv$dtj)
}

neural_apply_classification_logit_calibration <- function(logits,
                                                          model_info = NULL,
                                                          params = NULL,
                                                          likelihood_code_obs = NULL) {
  scale <- neural_calibration_scale_from_params(params = params, model_info = model_info)
  if (is.null(scale)) {
    return(logits)
  }
  scale <- strenv$jnp$array(scale, dtype = logits$dtype %||% strenv$dtj)
  likelihood <- tolower(as.character(model_info$likelihood %||% "bernoulli"))
  if (likelihood %in% c("bernoulli", "categorical", "ordinal")) {
    return(logits * scale)
  }
  if (!identical(likelihood, "mixed") || is.null(likelihood_code_obs)) {
    return(logits)
  }
  class_mask <- strenv$jnp$logical_or(
    strenv$jnp$equal(likelihood_code_obs, ai(0L)),
    strenv$jnp$logical_or(
      strenv$jnp$equal(likelihood_code_obs, ai(1L)),
      strenv$jnp$equal(likelihood_code_obs, ai(3L))
    )
  )
  strenv$jnp$where(
    strenv$jnp$reshape(class_mask, list(-1L, 1L)),
    logits * scale,
    logits
  )
}

neural_softclip_jnp <- function(x, low, high, softness, straight_through = TRUE) {
  s <- strenv$jnp$array(as.numeric(softness), dtype = x$dtype %||% strenv$dtj)
  low_j <- strenv$jnp$array(as.numeric(low), dtype = x$dtype %||% strenv$dtj)
  high_j <- strenv$jnp$array(as.numeric(high), dtype = x$dtype %||% strenv$dtj)
  clipped <- low_j + s * strenv$jax$nn$softplus((x - low_j) / s) -
    s * strenv$jax$nn$softplus((x - high_j) / s)
  if (!isTRUE(straight_through)) {
    # True-gradient path (serve / pi*-differentiated): forward value is identical
    # to the STE below, but the softclip's own vanishing tail gradient IS the
    # correct signal. Returning it keeps the optimizer's gradient consistent with
    # the bounded Q surface, instead of the STE's phantom identity push that
    # drives pi* toward degenerate extremes once the logit saturates.
    return(clipped)
  }
  # Straight-through estimator: bounded forward value with identity gradient,
  # used during training to avoid near-zero tail gradients starving the trunk.
  x + strenv$jax$lax$stop_gradient(clipped - x)
}

neural_apply_low_rank_logit_transform <- function(logits, model_info = NULL,
                                                  straight_through = TRUE) {
  if (!isTRUE(neural_low_rank_logit_transform_enabled(model_info))) {
    return(logits)
  }
  bound <- as.numeric(model_info$low_rank_logit_bound)
  softness <- as.numeric(model_info$low_rank_logit_softness)
  neural_softclip_jnp(
    logits,
    low = -bound,
    high = bound,
    softness = softness,
    straight_through = straight_through
  )
}

neural_apply_pairwise_classification_logit_transform <- function(logits,
                                                                 model_info = NULL,
                                                                 likelihood_code_obs = NULL,
                                                                 pairwise_obs = FALSE,
                                                                 straight_through = TRUE) {
  if (!isTRUE(pairwise_obs) ||
      !isTRUE(neural_low_rank_logit_transform_enabled(model_info))) {
    return(logits)
  }
  likelihood <- tolower(as.character(model_info$likelihood %||% "bernoulli"))
  if (identical(likelihood, "bernoulli")) {
    return(neural_apply_low_rank_logit_transform(logits, model_info,
                                                 straight_through = straight_through))
  }
  if (!identical(likelihood, "mixed") || is.null(likelihood_code_obs)) {
    return(logits)
  }

  out_dim <- ai(logits$shape[[2]])
  if (out_dim < 1L) {
    return(logits)
  }
  first <- strenv$jnp$take(logits, ai(0L), axis = 1L)
  first_soft <- neural_apply_low_rank_logit_transform(first, model_info,
                                                      straight_through = straight_through)
  bern_mask <- strenv$jnp$equal(likelihood_code_obs, ai(0L))
  first_use <- strenv$jnp$where(bern_mask, first_soft, first)
  first_col <- strenv$jnp$reshape(first_use, list(-1L, 1L))
  if (out_dim == 1L) {
    return(first_col)
  }
  rest <- strenv$jnp$take(logits, strenv$jnp$arange(ai(1L), ai(out_dim)), axis = 1L)
  strenv$jnp$concatenate(list(first_col, rest), axis = 1L)
}

neural_apply_pairwise_bernoulli_logit_scale <- function(logits,
                                                        model_info = NULL,
                                                        scale = NULL,
                                                        likelihood_code_obs = NULL,
                                                        pairwise_obs = FALSE) {
  if (!isTRUE(pairwise_obs) ||
      !isTRUE(neural_pairwise_bernoulli_logit_scale_enabled(model_info))) {
    return(logits)
  }
  scale <- scale %||% neural_pairwise_bernoulli_logit_scale_from_params(
    params = NULL,
    model_info = model_info
  )
  if (is.null(scale)) {
    return(logits)
  }
  scale <- strenv$jnp$array(scale, dtype = logits$dtype %||% strenv$dtj)
  likelihood <- tolower(as.character(model_info$likelihood %||% "bernoulli"))
  if (identical(likelihood, "bernoulli")) {
    return(logits * scale)
  }
  if (!identical(likelihood, "mixed") || is.null(likelihood_code_obs)) {
    return(logits)
  }

  out_dim <- ai(logits$shape[[2]])
  if (out_dim < 1L) {
    return(logits)
  }
  first <- strenv$jnp$take(logits, ai(0L), axis = 1L)
  first_scaled <- first * scale
  bern_mask <- strenv$jnp$equal(likelihood_code_obs, ai(0L))
  first_use <- strenv$jnp$where(bern_mask, first_scaled, first)
  first_col <- strenv$jnp$reshape(first_use, list(-1L, 1L))
  if (out_dim == 1L) {
    return(first_col)
  }
  rest <- strenv$jnp$take(logits, strenv$jnp$arange(ai(1L), ai(out_dim)), axis = 1L)
  strenv$jnp$concatenate(list(first_col, rest), axis = 1L)
}

neural_softplus_numeric <- function(x) {
  pmax(x, 0) + log1p(exp(-abs(x)))
}

neural_apply_low_rank_logit_transform_r <- function(logits, model_info = NULL) {
  if (!isTRUE(neural_low_rank_logit_transform_enabled(model_info))) {
    return(logits)
  }
  bound <- as.numeric(model_info$low_rank_logit_bound)
  softness <- as.numeric(model_info$low_rank_logit_softness)
  low <- -bound
  high <- bound
  low + softness * neural_softplus_numeric((logits - low) / softness) -
    softness * neural_softplus_numeric((logits - high) / softness)
}

neural_pairwise_bernoulli_logit_scale_r <- function(model_info = NULL,
                                                    scale = NULL) {
  if (!isTRUE(neural_pairwise_bernoulli_logit_scale_enabled(model_info))) {
    return(NULL)
  }
  if (is.null(scale)) {
    scale <- model_info$pairwise_bernoulli_logit_scale %||% NULL
  }
  if (is.null(scale) && !is.null(model_info$params$log_pairwise_bernoulli_logit_scale)) {
    scale <- exp(as.numeric(model_info$params$log_pairwise_bernoulli_logit_scale)[[1L]])
  }
  scale <- suppressWarnings(as.numeric(scale[[1L]]))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    return(NULL)
  }
  as.numeric(scale)
}

neural_apply_pairwise_bernoulli_logit_scale_r <- function(logits,
                                                          model_info = NULL,
                                                          scale = NULL) {
  scale <- neural_pairwise_bernoulli_logit_scale_r(
    model_info = model_info,
    scale = scale
  )
  if (is.null(scale)) {
    return(logits)
  }
  as.numeric(logits) * scale
}

neural_apply_pairwise_bernoulli_logit_adjustment_r <- function(logits,
                                                               model_info = NULL) {
  logits <- neural_apply_low_rank_logit_transform_r(logits, model_info)
  neural_apply_pairwise_bernoulli_logit_scale_r(logits, model_info)
}

# R-side calibration temperature (mirrors
# neural_apply_classification_logit_calibration). The jitted mixed-model cores
# call the jnp version without likelihood_code_obs, which makes it a no-op for
# likelihood == "mixed" -- so mixed models trained with
# calibration_method = "logit_scale" must have the temperature re-applied here
# on the class-family rows before logits are converted to probabilities.
neural_calibration_scale_numeric_r <- function(model_info = NULL) {
  if (is.null(model_info) ||
      !isTRUE(model_info$calibration_enabled) ||
      !identical(model_info$calibration_method %||% NULL, "logit_scale")) {
    return(NULL)
  }
  scale <- model_info$calibration_scale %||% NULL
  if (is.null(scale) && !is.null(model_info$params$calibration_scale)) {
    scale <- tryCatch(
      as.numeric(reticulate::py_to_r(strenv$np$array(model_info$params$calibration_scale))),
      error = function(e) NULL
    )
  }
  if (is.null(scale) && !is.null(model_info$params$log_calibration_scale)) {
    scale <- tryCatch(
      exp(as.numeric(reticulate::py_to_r(
        strenv$np$array(model_info$params$log_calibration_scale)
      ))[[1L]]),
      error = function(e) NULL
    )
  }
  scale <- suppressWarnings(as.numeric(scale[[1L]]))
  if (length(scale) != 1L || is.na(scale) || !is.finite(scale) || scale <= 0) {
    return(NULL)
  }
  as.numeric(scale)
}

neural_apply_classification_logit_calibration_r <- function(logits,
                                                            model_info = NULL) {
  scale <- neural_calibration_scale_numeric_r(model_info)
  if (is.null(scale)) {
    return(logits)
  }
  logits * scale
}

neural_low_rank_interaction_rank <- function(model_info = NULL) {
  neural_resolve_low_rank_interaction_rank(
    model_info$low_rank_interaction_rank %||% 0L
  )
}

neural_has_low_rank_interaction <- function(params = NULL, model_info = NULL) {
  rank <- neural_low_rank_interaction_rank(model_info)
  rank > 0L &&
    !is.null(params$W_rc_r) &&
    !is.null(params$W_rc_c) &&
    !is.null(params$W_rc_out)
}

neural_has_readout_cls <- function(model_info = NULL, low_rank_interaction_rank = NULL) {
  rank <- if (is.null(low_rank_interaction_rank)) {
    neural_low_rank_interaction_rank(model_info)
  } else {
    neural_resolve_low_rank_interaction_rank(low_rank_interaction_rank)
  }
  rank > 0L
}

neural_readout_embedding_families <- function(model_info = NULL,
                                              low_rank_interaction_rank = NULL) {
  if (isTRUE(neural_has_readout_cls(
    model_info = model_info,
    low_rank_interaction_rank = low_rank_interaction_rank
  ))) {
    return(c("choice", "respondent_cls", "candidate_cls"))
  }
  "choice"
}

neural_shared_projection_value_encoder <- function(model_info = NULL, mode = NULL) {
  neural_resolve_shared_projection_value_encoder(
    mode %||% model_info$shared_projection_value_encoder %||% "name_dist_moe"
  )
}

neural_covariate_value_stat_names <- function() {
  c("mean", "scale", "min", "q05", "q25", "q50", "q75", "q95", "max")
}

neural_covariate_value_metadata_names <- function() {
  c(
    "n_present_log", "missing_rate", "unique_count_log", "unique_ratio",
    "mean_signlog", "sd_log", "min_z", "max_z",
    "q05_z", "q25_z", "q50_z", "q75_z", "q95_z",
    "integer_like", "binary_like"
  )
}

neural_covariate_value_basis_dim <- function() {
  7L
}

neural_covariate_value_experts <- function() {
  4L
}

neural_default_covariate_value_stats_row <- function() {
  out <- c(
    mean = 0,
    scale = 1,
    min = 0,
    q05 = 0,
    q25 = 0,
    q50 = 0,
    q75 = 0,
    q95 = 0,
    max = 0
  )
  names(out) <- neural_covariate_value_stat_names()
  out
}

neural_default_covariate_value_metadata_row <- function() {
  out <- c(
    n_present_log = 0,
    missing_rate = 1,
    unique_count_log = 0,
    unique_ratio = 0,
    mean_signlog = 0,
    sd_log = 0,
    min_z = 0,
    max_z = 0,
    q05_z = 0,
    q25_z = 0,
    q50_z = 0,
    q75_z = 0,
    q95_z = 0,
    integer_like = 0,
    binary_like = 0
  )
  names(out) <- neural_covariate_value_metadata_names()
  out
}

neural_covariate_distribution_summary <- function(x_col, present_col = NULL) {
  default_stats <- neural_default_covariate_value_stats_row()
  default_meta <- neural_default_covariate_value_metadata_row()
  x_num <- suppressWarnings(as.numeric(x_col))
  n_total <- length(x_num)
  if (is.null(present_col)) {
    present_idx <- which(is.finite(x_num))
  } else {
    present_idx <- which(as.numeric(present_col) > 0 & is.finite(x_num))
  }
  n_present <- length(present_idx)
  if (n_present < 1L) {
    return(list(stats = default_stats, metadata = default_meta))
  }

  x_use <- x_num[present_idx]
  mean_raw <- mean(x_use)
  scale_raw <- if (n_present > 1L) stats::sd(x_use) else 0
  safe_scale <- if (is.finite(scale_raw) && scale_raw >= 1e-6) scale_raw else 1
  probs <- c(0.05, 0.25, 0.50, 0.75, 0.95)
  q_vals <- as.numeric(stats::quantile(x_use, probs = probs, na.rm = TRUE, names = FALSE, type = 7))
  min_raw <- min(x_use)
  max_raw <- max(x_use)
  unique_count <- length(unique(signif(x_use, digits = 12L)))
  unique_ratio <- unique_count / max(n_present, 1L)
  integer_like <- all(abs(x_use - round(x_use)) <= 1e-8)
  binary_like <- unique_count <= 2L

  stats_row <- c(
    mean = mean_raw,
    scale = safe_scale,
    min = min_raw,
    q05 = q_vals[[1]],
    q25 = q_vals[[2]],
    q50 = q_vals[[3]],
    q75 = q_vals[[4]],
    q95 = q_vals[[5]],
    max = max_raw
  )
  stats_row[!is.finite(stats_row)] <- default_stats[names(stats_row[!is.finite(stats_row)])]

  meta_row <- c(
    n_present_log = log1p(n_present),
    missing_rate = max(0, min(1, 1 - (n_present / max(n_total, 1L)))),
    unique_count_log = log1p(unique_count),
    unique_ratio = max(0, min(1, unique_ratio)),
    mean_signlog = sign(mean_raw) * log1p(abs(mean_raw)),
    sd_log = log1p(max(scale_raw, 0)),
    min_z = (min_raw - mean_raw) / safe_scale,
    max_z = (max_raw - mean_raw) / safe_scale,
    q05_z = (q_vals[[1]] - mean_raw) / safe_scale,
    q25_z = (q_vals[[2]] - mean_raw) / safe_scale,
    q50_z = (q_vals[[3]] - mean_raw) / safe_scale,
    q75_z = (q_vals[[4]] - mean_raw) / safe_scale,
    q95_z = (q_vals[[5]] - mean_raw) / safe_scale,
    integer_like = as.numeric(integer_like),
    binary_like = as.numeric(binary_like)
  )
  meta_row[!is.finite(meta_row)] <- default_meta[names(meta_row[!is.finite(meta_row)])]

  list(
    stats = stats_row[neural_covariate_value_stat_names()],
    metadata = meta_row[neural_covariate_value_metadata_names()]
  )
}

neural_empty_covariate_distribution_matrix <- function(n_covariates,
                                                       colnames_use,
                                                       default_row) {
  out <- matrix(
    rep(default_row, each = max(1L, as.integer(n_covariates))),
    nrow = max(1L, as.integer(n_covariates)),
    byrow = FALSE
  )
  out <- out[seq_len(as.integer(n_covariates)), , drop = FALSE]
  rownames(out) <- as.character(colnames_use %||% character(0))
  colnames(out) <- names(default_row)
  storage.mode(out) <- "double"
  out
}

neural_build_covariate_distribution_profiles <- function(X_mat,
                                                         X_present_mat = NULL,
                                                         experiment_index = NULL,
                                                         covariate_names = NULL,
                                                         default_experiment_index = NULL) {
  covariate_names <- as.character(covariate_names %||% colnames(X_mat) %||% character(0))
  n_covariates <- length(covariate_names)
  default_stats_row <- neural_default_covariate_value_stats_row()
  default_meta_row <- neural_default_covariate_value_metadata_row()
  if (n_covariates < 1L) {
    return(list(
      by_experiment = list(),
      metadata_by_experiment = list(),
      default_stats = matrix(0, nrow = 0L, ncol = length(default_stats_row)),
      default_metadata = matrix(0, nrow = 0L, ncol = length(default_meta_row))
    ))
  }

  X_use <- as.matrix(X_mat)
  storage.mode(X_use) <- "double"
  if (is.null(colnames(X_use))) {
    colnames(X_use) <- covariate_names
  }
  X_present_use <- if (is.null(X_present_mat)) {
    matrix(1, nrow = nrow(X_use), ncol = ncol(X_use))
  } else {
    as.matrix(X_present_mat)
  }
  storage.mode(X_present_use) <- "double"

  build_for_rows <- function(row_idx) {
    stats_mat <- neural_empty_covariate_distribution_matrix(
      n_covariates = n_covariates,
      colnames_use = covariate_names,
      default_row = default_stats_row
    )
    meta_mat <- neural_empty_covariate_distribution_matrix(
      n_covariates = n_covariates,
      colnames_use = covariate_names,
      default_row = default_meta_row
    )
    if (length(row_idx) < 1L) {
      return(list(stats = stats_mat, metadata = meta_mat))
    }
    for (j in seq_len(n_covariates)) {
      summary_j <- neural_covariate_distribution_summary(
        x_col = X_use[row_idx, j],
        present_col = X_present_use[row_idx, j]
      )
      stats_mat[j, ] <- summary_j$stats
      meta_mat[j, ] <- summary_j$metadata
    }
    list(stats = stats_mat, metadata = meta_mat)
  }

  global_summary <- build_for_rows(seq_len(nrow(X_use)))
  by_experiment <- list()
  metadata_by_experiment <- list()
  if (!is.null(experiment_index) &&
      length(experiment_index) == nrow(X_use) &&
      any(!is.na(experiment_index))) {
    exp_idx <- as.integer(experiment_index)
    max_idx <- max(exp_idx[!is.na(exp_idx)], na.rm = TRUE)
    by_experiment <- vector("list", max_idx + 1L)
    metadata_by_experiment <- vector("list", max_idx + 1L)
    for (idx in seq.int(0L, max_idx)) {
      rows_idx <- which(exp_idx == idx)
      sum_idx <- build_for_rows(rows_idx)
      by_experiment[[idx + 1L]] <- sum_idx$stats
      metadata_by_experiment[[idx + 1L]] <- sum_idx$metadata
    }
  }

  default_stats <- global_summary$stats
  default_metadata <- global_summary$metadata
  if (!is.null(default_experiment_index) &&
      length(default_experiment_index) == 1L &&
      !is.na(default_experiment_index) &&
      length(by_experiment) >= (default_experiment_index + 1L)) {
    stats_idx <- by_experiment[[default_experiment_index + 1L]]
    meta_idx <- metadata_by_experiment[[default_experiment_index + 1L]]
    if (!is.null(stats_idx)) {
      default_stats <- stats_idx
    }
    if (!is.null(meta_idx)) {
      default_metadata <- meta_idx
    }
  }

  list(
    by_experiment = by_experiment,
    metadata_by_experiment = metadata_by_experiment,
    default_stats = default_stats,
    default_metadata = default_metadata
  )
}

neural_prepare_covariate_lookup_array <- function(mat_list,
                                                  default_mat,
                                                  experiment_idx = NULL,
                                                  n_batch = 1L) {
  mat_list <- mat_list %||% list()
  default_use <- if (!is.null(default_mat)) {
    neural_as_jnp_matrix(default_mat, dtype = strenv$dtj)
  } else if (length(mat_list %||% list()) > 0L) {
    first_ok <- which(vapply(mat_list, Negate(is.null), logical(1)))
    if (length(first_ok) > 0L) {
      neural_as_jnp_matrix(mat_list[[first_ok[[1L]]]], dtype = strenv$dtj)
    } else {
      NULL
    }
  } else {
    NULL
  }
  if (is.null(default_use)) {
    return(NULL)
  }
  n_rows <- ai(default_use$shape[[1]])
  n_cols <- ai(default_use$shape[[2]])
  if (n_rows < 1L || n_cols < 1L) {
    return(NULL)
  }
  if (!is.null(experiment_idx) && length(mat_list) > 0L) {
    n_lookup <- length(mat_list)
    arr_list <- vector("list", n_lookup)
    for (i in seq_len(n_lookup)) {
      mat_i <- mat_list[[i]]
      arr_list[[i]] <- if (is.null(mat_i)) {
        default_use
      } else {
        neural_as_jnp_matrix(mat_i, dtype = strenv$dtj)
      }
    }
    arr_jnp <- strenv$jnp$stack(arr_list, axis = 0L)
    exp_idx <- neural_as_jnp_vector(experiment_idx, dtype = strenv$jnp$int32)
    if (ai(exp_idx$shape[[1]]) == 1L && n_batch > 1L) {
      exp_idx <- exp_idx * strenv$jnp$ones(list(n_batch), dtype = strenv$jnp$int32)
    }
    exp_idx <- strenv$jnp$clip(exp_idx, ai(0L), ai(n_lookup - 1L))
    return(strenv$jnp$take(arr_jnp, exp_idx, axis = 0L))
  }
  strenv$jnp$reshape(default_use, list(1L, n_rows, n_cols)) *
    strenv$jnp$ones(list(n_batch, 1L, 1L), dtype = strenv$dtj)
}

neural_resolve_default_resp_cov_values <- function(model_info,
                                                   n_rows,
                                                   experiment_idx = NULL) {
  n_covariates <- length(as.character(model_info$covariate_names %||% character(0)))
  if (n_covariates < 1L) {
    return(matrix(0, nrow = n_rows, ncol = 0L))
  }
  encoder_mode <- neural_shared_projection_value_encoder(model_info)
  if (!identical(neural_covariate_value_encoding(model_info), "shared_projection") ||
      !identical(encoder_mode, "name_dist_moe")) {
    base_mean <- as.numeric(cs2step_neural_to_r_array(model_info$resp_cov_mean))
    if (length(base_mean) != n_covariates) {
      base_mean <- if (length(base_mean) < 1L) rep(0, n_covariates) else rep_len(base_mean, n_covariates)
    }
    return(matrix(rep(base_mean, each = n_rows), nrow = n_rows, ncol = n_covariates))
  }
  default_stats <- model_info$default_covariate_value_stats %||% NULL
  stats_by_experiment <- model_info$covariate_value_stats_by_experiment %||% list()
  if (!is.null(experiment_idx) && length(stats_by_experiment) > 0L) {
    exp_idx <- as.integer(experiment_idx)
    if (length(exp_idx) == 1L && n_rows > 1L) {
      exp_idx <- rep.int(exp_idx, n_rows)
    }
    out <- matrix(0, nrow = n_rows, ncol = n_covariates)
    for (i in seq_len(n_rows)) {
      idx <- exp_idx[[min(i, length(exp_idx))]]
      stats_i <- if (!is.na(idx) && length(stats_by_experiment) >= (idx + 1L)) {
        stats_by_experiment[[idx + 1L]] %||% default_stats
      } else {
        default_stats
      }
      if (is.null(stats_i)) {
        next
      }
      out[i, ] <- as.matrix(stats_i)[, "mean", drop = TRUE]
    }
    colnames(out) <- as.character(model_info$covariate_names %||% character(0))
    return(out)
  }
  base_stats <- if (!is.null(default_stats)) as.matrix(default_stats) else NULL
  if (is.null(base_stats) || ncol(base_stats) < 1L) {
    base_mean <- as.numeric(cs2step_neural_to_r_array(model_info$resp_cov_mean))
    if (length(base_mean) != n_covariates) {
      base_mean <- if (length(base_mean) < 1L) rep(0, n_covariates) else rep_len(base_mean, n_covariates)
    }
    return(matrix(rep(base_mean, each = n_rows), nrow = n_rows, ncol = n_covariates))
  }
  base_mean <- base_stats[, "mean", drop = TRUE]
  out <- matrix(rep(base_mean, each = n_rows), nrow = n_rows, ncol = n_covariates)
  colnames(out) <- as.character(model_info$covariate_names %||% character(0))
  out
}

neural_schema_dropout_keys <- function() {
  c(
    "experiment_token",
    "schema_text",
    "structural_metadata",
    "context_token",
    "factor_token",
    "covariate_token"
  )
}

neural_schema_dropout_defaults <- function() {
  list(
    experiment_token = 0.25,
    schema_text = 0.10,
    structural_metadata = 0.05,
    context_token = 0.10,
    factor_token = 0.03,
    covariate_token = 0.05
  )
}

neural_schema_dropout_zero <- function() {
  stats::setNames(
    as.list(rep(0, length(neural_schema_dropout_keys()))),
    neural_schema_dropout_keys()
  )
}

neural_schema_dropout_defaults_flag <- function(value) {
  if (is.logical(value) && length(value) == 1L && !is.na(value)) {
    return(isTRUE(value))
  }
  if (is.numeric(value) && length(value) == 1L &&
      !is.na(value) && is.finite(value) && value %in% c(0, 1)) {
    return(as.logical(value))
  }
  stop("'schema_dropout$defaults' must be TRUE or FALSE.", call. = FALSE)
}

neural_validate_schema_dropout_rate <- function(value, name) {
  if (!is.numeric(value) || length(value) != 1L ||
      is.na(value) || !is.finite(value) || value < 0 || value >= 1) {
    stop(
      sprintf("'schema_dropout$%s' must be a scalar numeric value in [0, 1).", name),
      call. = FALSE
    )
  }
  as.numeric(value)
}

neural_resolve_schema_dropout <- function(schema_dropout = NULL) {
  if (is.null(schema_dropout) || identical(schema_dropout, FALSE)) {
    return(neural_schema_dropout_zero())
  }
  if (identical(schema_dropout, TRUE)) {
    return(neural_schema_dropout_defaults())
  }
  if (!is.list(schema_dropout) && !is.atomic(schema_dropout)) {
    stop(
      "'schema_dropout' must be TRUE, FALSE, NULL, or a named list/vector of rates.",
      call. = FALSE
    )
  }

  values <- as.list(schema_dropout)
  value_names <- names(values)
  if (is.null(value_names) || any(!nzchar(value_names))) {
    stop("'schema_dropout' overrides must be named.", call. = FALSE)
  }

  use_defaults <- FALSE
  if ("defaults" %in% value_names) {
    use_defaults <- neural_schema_dropout_defaults_flag(values[["defaults"]])
    values[["defaults"]] <- NULL
    value_names <- names(values)
  }

  allowed <- neural_schema_dropout_keys()
  unknown <- setdiff(value_names, allowed)
  if (length(unknown) > 0L) {
    stop(
      sprintf("Unknown schema_dropout rate(s): %s.", paste(unknown, collapse = ", ")),
      call. = FALSE
    )
  }

  resolved <- if (isTRUE(use_defaults)) {
    neural_schema_dropout_defaults()
  } else {
    neural_schema_dropout_zero()
  }
  for (rate_name in value_names) {
    resolved[[rate_name]] <- neural_validate_schema_dropout_rate(
      values[[rate_name]],
      rate_name
    )
  }
  resolved[allowed]
}

neural_schema_dropout_active <- function(schema_dropout = NULL) {
  rates <- unlist(neural_resolve_schema_dropout(schema_dropout), use.names = FALSE)
  any(rates > 0)
}

neural_resolve_token_runtime_config <- function(neural_token_info = NULL,
                                                mcmc_control = NULL) {
  resolved <- neural_token_info %||% list()
  resolved$factor_tokenization <- neural_factor_tokenization(
    mode = resolved$factor_tokenization %||%
      mcmc_control$factor_tokenization %||%
      "fused"
  )
  resolved$max_factor_tokens <- neural_resolve_max_factor_tokens(
    resolved$max_factor_tokens %||%
      mcmc_control$max_factor_tokens %||%
      NULL
  )
  resolved$covariate_value_encoding <- neural_resolve_covariate_value_encoding(
    resolved$covariate_value_encoding %||%
      mcmc_control$covariate_value_encoding %||%
      "shared_projection"
  )
  resolved$shared_projection_value_encoder <- neural_resolve_shared_projection_value_encoder(
    resolved$shared_projection_value_encoder %||%
      mcmc_control$shared_projection_value_encoder %||%
      "name_dist_moe"
  )
  resolved$max_covariate_tokens <- neural_resolve_max_covariate_tokens(
    resolved$max_covariate_tokens %||%
      mcmc_control$max_covariate_tokens %||%
      NULL
  )
  resolved$schema_dropout <- neural_resolve_schema_dropout(
    resolved$schema_dropout %||%
      mcmc_control$schema_dropout %||%
      NULL
  )
  resolved$low_rank_interaction_rank <- neural_resolve_low_rank_interaction_rank(
    resolved$low_rank_interaction_rank %||%
      mcmc_control$low_rank_interaction_rank %||%
      mcmc_control$respondent_candidate_interaction_rank %||%
      0L
  )
  resolved
}

neural_prediction_jit_cache <- new.env(parent = emptyenv())

neural_model_jit_cache_key <- function(model_info) {
  if (is.null(model_info)) {
    return("stable:v1:null")
  }

  snapshot <- tryCatch({
    snapshot_out <- cs2step_neural_pack_model_info(
      cs2step_neural_upgrade_model_info(model_info),
      drop_params = TRUE
    )
    snapshot_out$fit_metrics <- NULL
    snapshot_out$jit_cache_key <- NULL
    snapshot_out
  }, error = function(e) NULL)

  if (!is.null(snapshot)) {
    return(paste0(
      "stable:v1:",
      digest::digest(snapshot, algo = "xxhash64", serialize = TRUE)
    ))
  }

  map_token <- "none"
  cand_map <- tryCatch(model_info$cand_party_to_resp_idx, error = function(e) NULL)
  if (!is.null(cand_map)) {
    map_vals <- tryCatch(
      as.integer(reticulate::py_to_r(strenv$np$array(cand_map))),
      error = function(e) tryCatch(as.integer(cand_map), error = function(e2) integer(0))
    )
    map_token <- paste(map_vals, collapse = ",")
    if (!nzchar(map_token)) {
      map_token <- "empty"
    }
  }

  digest_field <- function(x) {
    value <- tryCatch(x, error = function(e) NULL)
    if (is.null(value)) {
      return("null")
    }
    value <- tryCatch(cs2step_neural_to_r_array(value), error = function(e) value)
    digest::digest(value, algo = "xxhash64", serialize = TRUE)
  }

  fields <- c(
    tryCatch(as.character(model_info$model_depth), error = function(e) "na"),
    tryCatch(as.character(model_info$model_dims), error = function(e) "na"),
    tryCatch(as.character(model_info$n_heads), error = function(e) "na"),
    tryCatch(as.character(model_info$head_dim), error = function(e) "na"),
    neural_transformer_residual_mode(model_info),
    neural_attention_backend(model_info),
    neural_attention_dtype_mode(model_info),
    tryCatch(as.character(neural_attention_padding_multiple(model_info)), error = function(e) "na"),
    tryCatch(as.character(model_info$attention_resolved_backend), error = function(e) "na"),
    neural_cross_encoder_mode(model_info),
    tryCatch(as.character(model_info$likelihood), error = function(e) "na"),
    tryCatch(as.character(model_info$experiment_token_mode), error = function(e) "na"),
    tryCatch(as.character(model_info$place_context_enabled), error = function(e) "na"),
    tryCatch(as.character(model_info$place_context_dim), error = function(e) "na"),
    tryCatch(as.character(model_info$time_context_enabled), error = function(e) "na"),
    tryCatch(as.character(model_info$time_context_dim), error = function(e) "na"),
    tryCatch(as.character(model_info$factor_tokenization), error = function(e) "na"),
    tryCatch(as.character(model_info$max_factor_tokens), error = function(e) "na"),
    tryCatch(paste(dim(as.matrix(model_info$factor_struct_matrix)), collapse = "x"), error = function(e) "na"),
    digest_field(model_info$factor_struct_matrix),
    tryCatch(paste(as.character(model_info$factor_struct_feature_names %||% character(0)), collapse = "|"), error = function(e) "na"),
    tryCatch(paste(vapply(model_info$level_struct_matrices %||% list(), function(x) {
      paste(dim(as.matrix(x)), collapse = "x")
    }, character(1)), collapse = ","), error = function(e) "na"),
    digest_field(model_info$level_struct_matrices %||% list()),
    tryCatch(paste(as.character(model_info$level_struct_feature_names %||% character(0)), collapse = "|"), error = function(e) "na"),
    tryCatch(as.character(model_info$covariate_value_encoding), error = function(e) "na"),
    tryCatch(as.character(model_info$shared_projection_value_encoder), error = function(e) "na"),
    tryCatch(as.character(neural_low_rank_interaction_rank(model_info)), error = function(e) "na"),
    tryCatch(as.character(model_info$low_rank_logit_transform %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$low_rank_logit_bound %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$low_rank_logit_softness %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$low_rank_logit_normalization %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$low_rank_head_weight_target_rms %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$low_rank_rc_out_target_rms %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$additive_utility_mode %||% "off"), error = function(e) "na"),
    tryCatch(as.character(model_info$additive_utility_normalization %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$additive_head_weight_target_rms %||% "none"), error = function(e) "na"),
    tryCatch(as.character(model_info$max_covariate_tokens), error = function(e) "na"),
    tryCatch(as.character(model_info$n_candidate_tokens), error = function(e) "na"),
    tryCatch(as.character(model_info$n_party_levels), error = function(e) "na"),
    digest_field(model_info$experiment_description_text),
    digest_field(model_info$experiment_description_present),
    digest_field(model_info$default_experiment_text),
    tryCatch(as.character(isTRUE(model_info$default_experiment_text_present %||% FALSE)), error = function(e) "na"),
    digest_field(model_info$place_embedding),
    digest_field(model_info$default_place_embedding),
    digest_field(model_info$time_embedding),
    digest_field(model_info$default_time_embedding),
    map_token
  )
  paste0("stable:v1:fallback::", paste(fields, collapse = "::"))
}

neural_rms_norm <- function(x, g, model_dims, eps = 1e-6) {
  if (is.null(g)) {
    return(x)
  }
  x <- strenv$jnp$array(x)
  g_use <- strenv$jnp$array(g, dtype = x$dtype)
  mean_sq <- strenv$jnp$mean(x * x, axis = -1L, keepdims = TRUE)
  inv_rms <- strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps))
  x * inv_rms * g_use
}

neural_rms_norm_no_scale <- function(x, eps = 1e-6) {
  x <- strenv$jnp$array(x)
  mean_sq <- strenv$jnp$mean(x * x, axis = -1L, keepdims = TRUE)
  inv_rms <- strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps))
  x * inv_rms
}

neural_column_rms_normalize <- function(W, target_rms, eps = 1e-6) {
  W <- strenv$jnp$array(W)
  target <- strenv$jnp$array(as.numeric(target_rms), dtype = W$dtype)
  mean_sq <- strenv$jnp$mean(W * W, axis = 0L, keepdims = TRUE)
  W * strenv$jnp$reciprocal(strenv$jnp$sqrt(mean_sq + eps)) * target
}

neural_full_attn_residual_core <- function(sources,
                                           pseudo_query = NULL,
                                           model_dims = NULL,
                                           eps = 1e-6) {
  if (is.null(model_dims)) {
    model_dims <- tryCatch(ai(sources$shape[[4]]), error = function(e) NULL)
  }
  if (is.null(pseudo_query)) {
    pseudo_query <- strenv$jnp$zeros(
      list(ai(model_dims)),
      dtype = tryCatch(sources$dtype, error = function(e) strenv$dtj)
    )
  } else {
    pseudo_query <- strenv$jnp$array(
      pseudo_query,
      dtype = tryCatch(sources$dtype, error = function(e) strenv$dtj)
    )
  }
  keys <- neural_rms_norm_no_scale(sources, eps = eps)
  logits <- strenv$jnp$einsum("d,nbtd->nbt", pseudo_query, keys)
  weights <- strenv$jax$nn$softmax(logits, axis = 0L)
  strenv$jnp$einsum("nbt,nbtd->btd", weights, sources)
}

neural_full_attn_residual <- function(sources,
                                      pseudo_query = NULL,
                                      model_dims = NULL,
                                      eps = 1e-6) {
  if (is.list(sources)) {
    sources <- strenv$jnp$stack(sources, axis = 0L)
  } else {
    sources <- strenv$jnp$array(sources)
  }
  neural_full_attn_residual_core(
    sources = sources,
    pseudo_query = pseudo_query,
    model_dims = model_dims,
    eps = eps
  )
}

neural_init_residual_history <- function(tokens) {
  strenv$jnp$expand_dims(strenv$jnp$array(tokens), axis = 0L)
}

neural_append_residual_history <- function(history, x) {
  x_expanded <- strenv$jnp$expand_dims(strenv$jnp$array(x), axis = 0L)
  strenv$jnp$concatenate(list(history, x_expanded), axis = 0L)
}

neural_full_attn_residual_from_history <- function(history,
                                                   pseudo_query = NULL,
                                                   model_dims = NULL,
                                                   n_used = NULL,
                                                   eps = 1e-6) {
  history_use <- strenv$jnp$array(history)
  if (!is.null(n_used)) {
    idx <- strenv$jnp$arange(ai(n_used))
    history_use <- strenv$jnp$take(history_use, idx, axis = 0L)
  }
  neural_full_attn_residual_core(
    sources = history_use,
    pseudo_query = pseudo_query,
    model_dims = model_dims,
    eps = eps
  )
}

neural_validate_full_attn_compatibility <- function(model_info,
                                                    params = NULL,
                                                    context = "Neural model") {
  if (is.null(model_info)) {
    return(invisible(NULL))
  }
  if (!identical(neural_transformer_residual_mode(model_info), "full_attn")) {
    return(invisible(NULL))
  }

  has_final_query <- FALSE
  if (!is.null(params)) {
    has_final_query <- !is.null(params[["pseudo_query_final"]])
  }
  if (!isTRUE(has_final_query) && !is.null(model_info$params)) {
    has_final_query <- !is.null(model_info$params[["pseudo_query_final"]])
  }
  if (!isTRUE(has_final_query) && !is.null(model_info$param_names)) {
    has_final_query <- "pseudo_query_final" %in% model_info$param_names
  }

  if (!isTRUE(has_final_query)) {
    stop(
      sprintf(
        "%s uses legacy 'full_attn' parameters without 'pseudo_query_final'. Refit and re-export under the updated architecture.",
        context
      ),
      call. = FALSE
    )
  }

  invisible(NULL)
}

neural_required_cross_attn_param_names <- function() {
  c(
    "RMS_merge_cross",
    "RMS_q_cross",
    "RMS_k_cross",
    "W_q_cross",
    "W_k_cross",
    "W_v_cross",
    "W_o_cross"
  )
}

neural_missing_required_param_names <- function(required_names,
                                                model_info,
                                                params = NULL) {
  missing_names <- character(0)
  for (name in required_names) {
    has_name <- FALSE
    if (!is.null(params)) {
      has_name <- !is.null(params[[name]])
    }
    if (!isTRUE(has_name) && !is.null(model_info$params)) {
      has_name <- !is.null(model_info$params[[name]])
    }
    if (!isTRUE(has_name) && !is.null(model_info$param_names)) {
      has_name <- name %in% model_info$param_names
    }
    if (!isTRUE(has_name)) {
      missing_names <- c(missing_names, name)
    }
  }
  unique(missing_names)
}

neural_validate_cross_attn_compatibility <- function(model_info,
                                                     params = NULL,
                                                     context = "Neural model") {
  if (is.null(model_info)) {
    return(invisible(NULL))
  }
  if (!identical(neural_cross_encoder_mode(model_info), "attn")) {
    return(invisible(NULL))
  }

  missing_names <- neural_missing_required_param_names(
    required_names = neural_required_cross_attn_param_names(),
    model_info = model_info,
    params = params
  )
  if (length(missing_names) < 1L) {
    return(invisible(NULL))
  }

  stop(
    sprintf(
      paste0(
        "%s uses legacy cross-candidate attention parameters without required ",
        "sites: %s. Refit and re-export under the updated architecture."
      ),
      context,
      paste(missing_names, collapse = ", ")
    ),
    call. = FALSE
  )
}

neural_transformer_state_tokens <- function(transformer_out) {
  if (is.list(transformer_out) && !is.null(transformer_out$tokens)) {
    return(transformer_out$tokens)
  }
  transformer_out
}

neural_transformer_readout_tokens <- function(transformer_out) {
  if (is.list(transformer_out) && !is.null(transformer_out$readout_tokens)) {
    return(transformer_out$readout_tokens)
  }
  transformer_out
}

neural_extract_candidate_tokens <- function(transformer_out,
                                            model_info,
                                            n_candidate_tokens = NULL) {
  tokens_use <- if (identical(neural_transformer_residual_mode(model_info), "full_attn")) {
    neural_transformer_readout_tokens(transformer_out)
  } else {
    neural_transformer_state_tokens(transformer_out)
  }

  t_cand <- if (is.null(n_candidate_tokens)) {
    tryCatch(ai(model_info$n_candidate_tokens), error = function(e) NULL)
  } else {
    ai(n_candidate_tokens)
  }
  if (length(t_cand) != 1L || is.na(t_cand) || t_cand < 1L) {
    stop(
      "n_candidate_tokens must be a positive integer for candidate-token extraction.",
      call. = FALSE
    )
  }

  t_total <- ai(tokens_use$shape[[2]])
  cand_idx <- strenv$jnp$arange(ai(t_total - t_cand), ai(t_total))
  strenv$jnp$take(tokens_use, cand_idx, axis = 1L)
}

neural_candidate_token_count_from_mask <- function(token_mask) {
  if (is.null(token_mask)) {
    return(NULL)
  }
  tryCatch(ai(token_mask$shape[[2]]), error = function(e) NULL)
}

neural_max_order_length <- function(order_list = NULL, default_order = NULL) {
  lengths_use <- integer(0)
  if (!is.null(default_order)) {
    lengths_use <- c(lengths_use, length(default_order %||% integer(0)))
  }
  if (length(order_list %||% list()) > 0L) {
    lengths_use <- c(
      lengths_use,
      vapply(order_list, function(x) length(x %||% integer(0)), integer(1))
    )
  }
  if (length(lengths_use) < 1L) {
    return(0L)
  }
  as.integer(max(lengths_use, 0L, na.rm = TRUE))
}

neural_active_candidate_token_budget <- function(model_info) {
  aux_tokens <- as.integer(neural_candidate_group_context_enabled(model_info)) +
    as.integer(neural_relation_context_enabled(model_info))
  n_tokens <- neural_max_order_length(
    order_list = model_info$factor_order_by_experiment %||% NULL,
    default_order = model_info$default_factor_order %||% NULL
  )
  if (length(n_tokens) != 1L || is.na(n_tokens) || n_tokens < 1L) {
    n_tokens <- tryCatch(ai(model_info$n_factors), error = function(e) 0L)
  }
  as.integer(n_tokens + aux_tokens)
}

neural_active_context_token_budget <- function(model_info) {
  base_tokens <- as.integer(isTRUE(model_info$has_experiment_token)) +
    as.integer(neural_place_context_enabled(model_info)) +
    as.integer(neural_time_context_enabled(model_info)) +
    as.integer(isTRUE(model_info$has_stage_token)) +
    as.integer(neural_respondent_group_context_enabled(model_info)) +
    as.integer(isTRUE(model_info$has_matchup_token))
  covariate_tokens <- 0L
  if (isTRUE(model_info$has_covariate_fused_tokens) || isTRUE(model_info$has_covariate_tokens)) {
    n_tokens <- neural_max_order_length(
      order_list = model_info$covariate_order_by_experiment %||% NULL,
      default_order = model_info$default_covariate_order %||% NULL
    )
    if (length(n_tokens) != 1L || is.na(n_tokens) || n_tokens < 1L) {
      n_tokens <- tryCatch(ai(model_info$n_resp_covariates), error = function(e) 0L)
      if (length(n_tokens) != 1L || is.null(n_tokens) || is.na(n_tokens) || n_tokens < 1L) {
        n_tokens <- length(model_info$covariate_names %||% character(0))
      }
    }
    covariate_tokens <- as.integer(n_tokens)
  }
  as.integer(base_tokens + covariate_tokens)
}

neural_pack_token_block <- function(tokens,
                                    token_mask,
                                    trim_tokens = NULL) {
  if (is.null(tokens) || is.null(token_mask)) {
    return(list(tokens = tokens, mask = token_mask))
  }
  width <- tryCatch(ai(token_mask$shape[[2]]), error = function(e) NULL)
  if (is.null(width) || is.na(width) || width < 1L) {
    return(list(tokens = tokens, mask = token_mask))
  }
  trim_use <- if (is.null(trim_tokens)) {
    ai(width)
  } else {
    min(ai(trim_tokens), ai(width))
  }
  if (is.na(trim_use) || trim_use < 0L) {
    trim_use <- ai(width)
  }

  base_idx <- strenv$jnp$reshape(
    strenv$jnp$arange(ai(width), dtype = strenv$jnp$int32),
    list(1L, ai(width))
  )
  sort_keys <- strenv$jnp$where(
    token_mask > 0,
    base_idx,
    base_idx + ai(width)
  )
  order <- strenv$jnp$argsort(sort_keys, axis = 1L)
  dims <- ai(tokens$shape[[3]])
  gather_idx <- strenv$jnp[["repeat"]](
    strenv$jnp$expand_dims(order, axis = 2L),
    repeats = ai(dims),
    axis = 2L
  )
  packed_tokens <- strenv$jnp$take_along_axis(tokens, gather_idx, axis = 1L)
  packed_mask <- strenv$jnp$take_along_axis(token_mask, order, axis = 1L)
  if (trim_use < width) {
    keep_idx <- strenv$jnp$arange(ai(trim_use))
    packed_tokens <- strenv$jnp$take(packed_tokens, keep_idx, axis = 1L)
    packed_mask <- strenv$jnp$take(packed_mask, keep_idx, axis = 1L)
  }
  list(tokens = packed_tokens, mask = packed_mask)
}

neural_fused_classification_summary_enabled <- function(model_info) {
  if (!identical(model_info$factor_tokenization %||% NULL, "fused")) {
    return(FALSE)
  }
  likelihood <- tolower(as.character(model_info$likelihood %||% ""))
  likelihood %in% c("bernoulli", "categorical", "ordinal", "mixed")
}

neural_pack_candidate_sequence <- function(choice_tok,
                                           choice_mask,
                                           ctx_tokens = NULL,
                                           ctx_mask = NULL,
                                           cand_tokens,
                                           cand_mask,
                                           model_info,
                                           preserve_candidate_tail = FALSE) {
  if (isTRUE(neural_fused_classification_summary_enabled(model_info)) &&
      !is.null(cand_tokens) &&
      !is.null(cand_mask)) {
    cand_mask_float <- strenv$jnp$astype(cand_mask, strenv$dtj)
    cand_mask_expanded <- strenv$jnp$reshape(
      cand_mask_float,
      list(cand_mask_float$shape[[1]], cand_mask_float$shape[[2]], 1L)
    )
    cand_count <- strenv$jnp$maximum(
      strenv$jnp$sum(cand_mask_float, axis = 1L, keepdims = TRUE),
      strenv$jnp$array(1., dtype = strenv$dtj)
    )
    cand_count <- strenv$jnp$reshape(cand_count, list(cand_count$shape[[1]], 1L, 1L))
    cand_summary <- strenv$jnp$sum(
      cand_tokens * cand_mask_expanded,
      axis = 1L,
      keepdims = TRUE
    ) / cand_count
    choice_tok <- choice_tok + cand_summary
  }

  ctx_trim <- neural_active_context_token_budget(model_info)
  cand_trim <- neural_active_candidate_token_budget(model_info)
  ctx_width <- if (is.null(ctx_mask)) {
    0L
  } else {
    tryCatch(ai(ctx_mask$shape[[2]]), error = function(e) 0L)
  }
  cand_width <- tryCatch(ai(cand_mask$shape[[2]]), error = function(e) 0L)
  if (length(ctx_trim) != 1L || is.na(ctx_trim) || ctx_trim < 0L) {
    ctx_trim <- ctx_width
  }
  if (length(cand_trim) != 1L || is.na(cand_trim) || cand_trim < 1L) {
    cand_trim <- cand_width
  }
  if (isTRUE(preserve_candidate_tail)) {
    ctx_packed <- if (is.null(ctx_tokens)) {
      list(tokens = NULL, mask = NULL)
    } else {
      neural_pack_token_block(ctx_tokens, ctx_mask, trim_tokens = ctx_trim)
    }
    cand_packed <- neural_pack_token_block(cand_tokens, cand_mask, trim_tokens = cand_trim)
    token_parts <- list(choice_tok)
    mask_parts <- list(choice_mask)
    if (!is.null(ctx_packed$tokens)) {
      token_parts <- c(token_parts, list(ctx_packed$tokens))
      mask_parts <- c(mask_parts, list(ctx_packed$mask))
    }
    token_parts <- c(token_parts, list(cand_packed$tokens))
    mask_parts <- c(mask_parts, list(cand_packed$mask))
    return(list(
      tokens = strenv$jnp$concatenate(token_parts, axis = 1L),
      mask = strenv$jnp$concatenate(mask_parts, axis = 1L),
      cand_mask = cand_packed$mask
    ))
  }

  tail_tokens <- if (is.null(ctx_tokens)) {
    cand_tokens
  } else {
    strenv$jnp$concatenate(list(ctx_tokens, cand_tokens), axis = 1L)
  }
  tail_mask <- if (is.null(ctx_mask)) {
    cand_mask
  } else {
    strenv$jnp$concatenate(list(ctx_mask, cand_mask), axis = 1L)
  }
  tail_trim <- as.integer(ctx_trim + cand_trim)
  tail_packed <- neural_pack_token_block(
    tail_tokens,
    tail_mask,
    trim_tokens = tail_trim
  )
  list(
    tokens = strenv$jnp$concatenate(list(choice_tok, tail_packed$tokens), axis = 1L),
    mask = strenv$jnp$concatenate(list(choice_mask, tail_packed$mask), axis = 1L),
    cand_mask = cand_mask
  )
}

neural_pack_full_cross_sequence <- function(choice_tok,
                                            choice_mask,
                                            sep_tok,
                                            sep_mask,
                                            left_tokens,
                                            left_mask,
                                            right_tokens,
                                            right_mask,
                                            model_info,
                                            ctx_tokens = NULL,
                                            ctx_mask = NULL) {
  ctx_trim <- neural_active_context_token_budget(model_info)
  cand_trim <- neural_active_candidate_token_budget(model_info)
  ctx_width <- if (is.null(ctx_mask)) {
    0L
  } else {
    tryCatch(ai(ctx_mask$shape[[2]]), error = function(e) 0L)
  }
  left_width <- tryCatch(ai(left_mask$shape[[2]]), error = function(e) 0L)
  if (length(ctx_trim) != 1L || is.na(ctx_trim) || ctx_trim < 0L) {
    ctx_trim <- ctx_width
  }
  if (length(cand_trim) != 1L || is.na(cand_trim) || cand_trim < 1L) {
    cand_trim <- left_width
  }
  ctx_packed <- if (is.null(ctx_tokens)) {
    list(tokens = NULL, mask = NULL)
  } else {
    neural_pack_token_block(ctx_tokens, ctx_mask, trim_tokens = ctx_trim)
  }
  left_packed <- neural_pack_token_block(left_tokens, left_mask, trim_tokens = cand_trim)
  right_packed <- neural_pack_token_block(right_tokens, right_mask, trim_tokens = cand_trim)
  token_parts <- list(choice_tok)
  mask_parts <- list(choice_mask)
  if (!is.null(ctx_packed$tokens)) {
    token_parts <- c(token_parts, list(ctx_packed$tokens))
    mask_parts <- c(mask_parts, list(ctx_packed$mask))
  }
  token_parts <- c(token_parts, list(sep_tok, left_packed$tokens, sep_tok, right_packed$tokens))
  mask_parts <- c(mask_parts, list(sep_mask, left_packed$mask, sep_mask, right_packed$mask))
  list(
    tokens = strenv$jnp$concatenate(token_parts, axis = 1L),
    mask = strenv$jnp$concatenate(mask_parts, axis = 1L),
    left_mask = left_packed$mask,
    right_mask = right_packed$mask
  )
}

neural_extract_choice_representation <- function(transformer_out,
                                                 token_mask = NULL,
                                                 include_tail_summary = FALSE,
                                                 tail_summary_scale = 0.5) {
  readout_tokens <- neural_transformer_readout_tokens(transformer_out)
  choice_out <- strenv$jnp$take(readout_tokens, strenv$jnp$arange(1L), axis = 1L)
  choice_out <- strenv$jnp$squeeze(choice_out, axis = 1L)
  if (!isTRUE(include_tail_summary)) {
    return(choice_out)
  }
  n_tokens <- ai(readout_tokens$shape[[2]])
  if (n_tokens <= 1L) {
    return(choice_out)
  }
  tail_idx <- strenv$jnp$arange(ai(1L), ai(n_tokens))
  tail_tokens <- strenv$jnp$take(readout_tokens, tail_idx, axis = 1L)
  if (is.null(token_mask)) {
    tail_mask <- strenv$jnp$ones(
      list(ai(tail_tokens$shape[[1]]), ai(tail_tokens$shape[[2]])),
      dtype = strenv$dtj
    )
  } else {
    tail_mask <- strenv$jnp$take(token_mask, tail_idx, axis = 1L)
    tail_mask <- strenv$jnp$astype(tail_mask, strenv$dtj)
  }
  tail_mask_expanded <- strenv$jnp$reshape(
    tail_mask,
    list(tail_mask$shape[[1]], tail_mask$shape[[2]], 1L)
  )
  tail_count <- strenv$jnp$maximum(
    strenv$jnp$sum(tail_mask, axis = 1L, keepdims = TRUE),
    strenv$jnp$array(1., dtype = strenv$dtj)
  )
  tail_summary <- strenv$jnp$sum(
    tail_tokens * tail_mask_expanded,
    axis = 1L
  ) / tail_count
  choice_out + as.numeric(tail_summary_scale) * tail_summary
}

neural_center_token_rows <- function(x) {
  x <- strenv$jnp$array(x)
  ndim <- length(x$shape)
  if (ndim < 2L) {
    return(x)
  }
  row_axis <- ai(ndim - 2L)
  x - strenv$jnp$mean(x, axis = row_axis, keepdims = TRUE)
}

neural_build_symmetric_segment_embeddings <- function(delta) {
  delta <- strenv$jnp$array(delta)
  ndim <- length(delta$shape)
  axis_use <- if (ndim <= 1L) 0L else ai(ndim - 1L)
  half_delta <- 0.5 * delta
  strenv$jnp$stack(list(-half_delta, half_delta), axis = axis_use)
}

neural_param_or_default <- function(params, name, default) {
  val <- params[[name]]
  if (is.null(val)) {
    return(default)
  }
  val
}

neural_build_output_site_init_values <- function(Y,
                                                 likelihood,
                                                 nOutcomes = 1L,
                                                 model_dims = NULL,
                                                 output_dim = NULL,
                                                 b_out_site_name = "b_out",
                                                 tau_b_scale = 0.5,
                                                 sigma_floor = 1e-3) {
  init_values <- list()
  model_dims <- suppressWarnings(as.integer(model_dims %||% NA_integer_))
  output_dim <- suppressWarnings(as.integer(output_dim %||% nOutcomes %||% NA_integer_))
  if (length(model_dims) == 1L && is.finite(model_dims) && model_dims > 0L &&
      length(output_dim) == 1L && is.finite(output_dim) && output_dim > 0L) {
    init_values$W_out <- matrix(0, nrow = model_dims, ncol = output_dim)
  }
  if (!identical(likelihood, "normal") ||
      length(nOutcomes) != 1L ||
      is.na(nOutcomes) ||
      as.integer(nOutcomes) != 1L) {
    return(init_values)
  }

  y_numeric <- suppressWarnings(as.numeric(Y))
  finite_y <- y_numeric[is.finite(y_numeric)]
  mean_y <- if (length(finite_y)) {
    mean(finite_y)
  } else {
    0.0
  }
  sigma0 <- suppressWarnings(stats::mad(finite_y, na.rm = TRUE))
  if (!is.finite(sigma0) || sigma0 <= 0) {
    sigma0 <- suppressWarnings(stats::sd(finite_y, na.rm = TRUE))
  }
  if (!is.finite(sigma0) || sigma0 <= 0) {
    sigma0 <- 1.0
  }
  sigma0 <- max(as.numeric(sigma0), as.numeric(sigma_floor))
  tau_b0 <- max(abs(as.numeric(mean_y)), as.numeric(tau_b_scale), as.numeric(sigma_floor))
  b_out0 <- rep(as.numeric(mean_y), times = as.integer(nOutcomes))

  init_values$tau_b <- as.numeric(tau_b0)
  init_values$sigma <- as.numeric(sigma0)
  if (is.character(b_out_site_name) && length(b_out_site_name) == 1L && nzchar(b_out_site_name)) {
    if (identical(b_out_site_name, "b_out")) {
      init_values[[b_out_site_name]] <- b_out0
    } else {
      init_values[[b_out_site_name]] <- b_out0 / tau_b0
    }
  }

  init_values
}

neural_get_init_to_value <- function() {
  if (is.null(strenv$numpyro) || is.null(strenv$numpyro$infer)) {
    return(NULL)
  }
  if (reticulate::py_has_attr(strenv$numpyro$infer, "initialization") &&
      reticulate::py_has_attr(strenv$numpyro$infer$initialization, "init_to_value")) {
    return(strenv$numpyro$infer$initialization$init_to_value)
  }
  if (reticulate::py_has_attr(strenv$numpyro$infer, "init_to_value")) {
    return(strenv$numpyro$infer$init_to_value)
  }
  NULL
}

neural_can_use_adamw_optimizer <- function() {
  (reticulate::py_has_attr(strenv$numpyro$optim, "AdamW") ||
     reticulate::py_has_attr(strenv$optax, "adamw"))
}

neural_default_svi_fallback_optimizer <- function() {
  if (isTRUE(neural_can_use_adamw_optimizer())) "adamw" else "adam"
}

neural_resolve_svi_optimizer_tag <- function(optimizer_tag,
                                             guide_name = NULL,
                                             user_supplied_optimizer = FALSE) {
  if (identical(optimizer_tag, "muon") &&
      identical(guide_name, "auto_diagonal")) {
    optimizer_tag <- neural_default_svi_fallback_optimizer()
    warning(
      sprintf(
        "optimizer='muon' is incompatible with vi_guide='auto_diagonal'; falling back to '%s'.",
        optimizer_tag
      ),
      call. = FALSE
    )
    return(optimizer_tag)
  }
  muon_available <- reticulate::py_has_attr(strenv$optax, "contrib") &&
    reticulate::py_has_attr(strenv$optax$contrib, "muon")
  if (identical(optimizer_tag, "muon") &&
      !isTRUE(user_supplied_optimizer) &&
      !isTRUE(muon_available)) {
    optimizer_tag <- neural_default_svi_fallback_optimizer()
    warning(
      sprintf(
        "Default optimizer 'muon' is unavailable; falling back to '%s'.",
        optimizer_tag
      ),
      call. = FALSE
    )
  }
  optimizer_tag
}

neural_muon_target_name_regex <- function() {
  # Single source of truth for Muon-target parameter names, consumed by both the R
  # selector (neural_muon_targets_matrix_weight) and the Python MuonDimensionNumbers
  # callable (which injects this string), so the mask and the dimension-numbers tree
  # stay in sync. Combined with the ndim==2 gate, this is effectively "every hidden
  # weight matrix EXCEPT embeddings, the readout/interaction output heads, and gains".
  # Added: the always-on fused-tokenization SwiGLU MLP (W_factor_fuse_1/2) and the
  # covariate fusion / value-encoder MLPs, which are on the pi* hot path and were
  # previously routed to Muon's Adam sub-branch while the identical transformer FF
  # (W_ff1/2) was orthogonalized. Dropped: the readout head W_out and interaction
  # output head W_rc_out, which benefit from Adam's per-coordinate scaling to
  # calibrate output magnitude (standard Muon recipes exclude the output head).
  # W_covariate_value_shared is excluded: it is shaped (1, ModelDims), so it
  # passes the ndim==2 gate but is effectively a vector -- Newton-Schulz
  # orthogonalization of a rank-1 matrix degenerates to direction
  # normalization (Muon-on-vectors is discouraged); Adam handles it.
  paste0(
    "^(",
    "W_(q|k|v|o)_l\\d+|W_ff(1|2)_l\\d+|W_(q|k|v|o)_cross",
    "|W_factor_struct|W_level_struct",
    "|W_factor_fuse_(1|2)|W_covariate_fuse_(1|2)",
    "|W_covariate_value_(conditioner_1|conditioner_2|basis)",
    "|M_cross_raw|W_rc_(r|c)",
    ")$"
  )
}

neural_swiglu <- function(x) {
  ndim <- length(x$shape)
  width <- ai(x$shape[[ndim]])
  hidden <- ai(width / 2L)
  gate_idx <- strenv$jnp$arange(ai(0L), hidden)
  value_idx <- strenv$jnp$arange(hidden, width)
  axis <- ai(ndim - 1L)
  gate <- strenv$jnp$take(x, gate_idx, axis = axis)
  value <- strenv$jnp$take(x, value_idx, axis = axis)
  strenv$jax$nn$swish(gate) * value
}

neural_muon_normalize_param_name <- function(name) {
  if (length(name) != 1L || is.na(name) || !nzchar(name)) {
    return(NULL)
  }
  normalized_name <- as.character(name)
  if (grepl("_auto_scale$", normalized_name)) {
    return(NULL)
  }
  normalized_name <- sub("_auto_loc$", "", normalized_name)
  repeat {
    stripped_name <- sub("(_decentered|_base|_z)$", "", normalized_name)
    if (identical(stripped_name, normalized_name)) {
      break
    }
    normalized_name <- stripped_name
  }
  normalized_name
}

neural_muon_targets_matrix_weight <- function(name, ndim = NULL) {
  normalized_name <- neural_muon_normalize_param_name(name)
  if (is.null(normalized_name)) {
    return(FALSE)
  }
  if (!is.null(ndim)) {
    ndim <- suppressWarnings(as.integer(ndim)[1L])
    if (is.na(ndim) || ndim != 2L) {
      return(FALSE)
    }
  }
  grepl(neural_muon_target_name_regex(), normalized_name)
}

neural_get_muon_dimension_numbers_callable <- local({
  cached_callable <- NULL
  cached_regex <- NULL

  function(force_refresh = FALSE) {
    if (!reticulate::py_has_attr(strenv$optax, "contrib") ||
        !reticulate::py_has_attr(strenv$optax$contrib, "MuonDimensionNumbers")) {
      return(NULL)
    }

    regex <- neural_muon_target_name_regex()
    if (!isTRUE(force_refresh) &&
        !is.null(cached_callable) &&
        identical(cached_regex, regex)) {
      return(cached_callable)
    }

    regex_py <- gsub("'", "\\\\'", regex, fixed = TRUE)
    reticulate::py_run_string(sprintf(
      paste(
        "import re",
        "import jax",
        "import optax",
        "",
        "_STRATEGIZE_MUON_KEY_RE = re.compile(r'%s')",
        "",
        "def _strategize_muon_normalize_name(name):",
        "    if not name:",
        "        return None",
        "    if name.endswith('_auto_scale'):",
        "        return None",
        "    if name.endswith('_auto_loc'):",
        "        name = name[:-len('_auto_loc')]",
        "    while True:",
        "        stripped = re.sub(r'(_decentered|_base|_z)$', '', name)",
        "        if stripped == name:",
        "            break",
        "        name = stripped",
        "    return name",
        "",
        "def _strategize_muon_dimnums(params):",
        "    tree_util = jax.tree_util",
        "",
        "    if hasattr(tree_util, 'tree_flatten_with_path'):",
        "        path_leaves, treedef = tree_util.tree_flatten_with_path(params)",
        "        out_leaves = []",
        "        for path, value in path_leaves:",
        "            name = ''",
        "            for entry in reversed(path):",
        "                if hasattr(entry, 'key'):",
        "                    name = str(entry.key)",
        "                    break",
        "",
        "            normalized_name = _strategize_muon_normalize_name(name)",
        "            use_muon = False",
        "            try:",
        "                ndim = getattr(value, 'ndim', None)",
        "                if ndim == 2 and normalized_name and _STRATEGIZE_MUON_KEY_RE.match(normalized_name):",
        "                    use_muon = True",
        "            except Exception:",
        "                use_muon = False",
        "",
        "            out_leaves.append(optax.contrib.MuonDimensionNumbers() if use_muon else None)",
        "        return tree_util.tree_unflatten(treedef, out_leaves)",
        "",
        "    if hasattr(params, 'items'):",
        "        out = {}",
        "        for k, v in params.items():",
        "            name = str(k)",
        "            normalized_name = _strategize_muon_normalize_name(name)",
        "            use_muon = (",
        "                getattr(v, 'ndim', None) == 2",
        "                and normalized_name is not None",
        "                and _STRATEGIZE_MUON_KEY_RE.match(normalized_name)",
        "            )",
        "            out[k] = optax.contrib.MuonDimensionNumbers() if use_muon else None",
        "        try:",
        "            return params.__class__(out)",
        "        except Exception:",
        "            return out",
        "",
        "    return tree_util.tree_map(lambda _: None, params)",
        sep = "\n"
      ),
      regex_py
    ))

    cached_regex <<- regex
    cached_callable <<- reticulate::py_eval("_strategize_muon_dimnums")
    cached_callable
  }
})

neural_build_muon_dimension_numbers_tree <- function(params) {
  muon_callable <- neural_get_muon_dimension_numbers_callable()
  if (is.null(muon_callable)) {
    return(NULL)
  }
  muon_callable(params)
}

neural_linear_head <- function(phi,
                               W_out,
                               b_out = NULL,
                               dtype = NULL,
                               model_info = NULL,
                               pairwise_obs = FALSE) {
  W_use <- W_out
  if (isTRUE(neural_low_rank_logit_normalization_enabled(model_info, pairwise_obs = pairwise_obs))) {
    phi <- neural_rms_norm_no_scale(phi)
    W_use <- neural_column_rms_normalize(
      W_out,
      target_rms = as.numeric(model_info$low_rank_head_weight_target_rms)
    )
  }
  logits <- strenv$jnp$einsum("nm,mo->no", phi, W_use)
  if (is.null(b_out)) {
    dtype_use <- dtype
    if (is.null(dtype_use)) {
      dtype_use <- tryCatch(W_use$dtype, error = function(e) NULL)
    }
    if (is.null(dtype_use)) {
      dtype_use <- strenv$dtj
    }
    b_out <- strenv$jnp$zeros(list(ai(W_use$shape[[2]])), dtype = dtype_use)
  }
  logits + b_out
}

neural_additive_linear_head <- function(pooled,
                                        W_add_out,
                                        dtype = NULL,
                                        model_info = NULL) {
  W_use <- W_add_out
  if (isTRUE(neural_additive_utility_normalization_enabled(model_info))) {
    pooled <- neural_rms_norm_no_scale(pooled)
    W_use <- neural_column_rms_normalize(
      W_add_out,
      target_rms = as.numeric(model_info$additive_head_weight_target_rms)
    )
  }
  logits <- strenv$jnp$einsum("nm,mo->no", pooled, W_use)
  dtype_use <- dtype
  if (is.null(dtype_use)) {
    dtype_use <- tryCatch(W_use$dtype, error = function(e) NULL)
  }
  if (is.null(dtype_use)) {
    dtype_use <- strenv$dtj
  }
  logits + strenv$jnp$zeros(list(ai(W_use$shape[[2]])), dtype = dtype_use)
}

neural_additive_logits_from_candidate_tokens <- function(tokens,
                                                         token_mask,
                                                         params,
                                                         model_info,
                                                         out_dim = NULL,
                                                         dtype = NULL) {
  if (!isTRUE(neural_additive_utility_enabled(model_info, params))) {
    n_batch <- tryCatch(ai(tokens$shape[[1]]), error = function(e) 1L)
    out_dim_use <- out_dim %||% tryCatch(ai(params$W_out$shape[[2]]), error = function(e) 1L)
    return(strenv$jnp$zeros(list(ai(n_batch), ai(out_dim_use)), dtype = dtype %||% strenv$dtj))
  }
  if (is.null(tokens) || is.null(token_mask)) {
    out_dim_use <- out_dim %||% tryCatch(ai(params$W_add_out$shape[[2]]), error = function(e) 1L)
    return(strenv$jnp$zeros(list(1L, ai(out_dim_use)), dtype = dtype %||% strenv$dtj))
  }
  mask <- strenv$jnp$astype(token_mask > 0, tokens$dtype %||% strenv$dtj)
  mask_exp <- strenv$jnp$reshape(mask, list(mask$shape[[1]], mask$shape[[2]], 1L))
  denom <- strenv$jnp$maximum(
    strenv$jnp$sum(mask, axis = 1L, keepdims = TRUE),
    strenv$jnp$array(1., dtype = tokens$dtype %||% strenv$dtj)
  )
  pooled <- strenv$jnp$sum(tokens * mask_exp, axis = 1L) / denom
  neural_additive_linear_head(
    pooled,
    params$W_add_out,
    dtype = dtype,
    model_info = model_info
  )
}

neural_additive_single_logits_prepared <- function(params,
                                                   model_info,
                                                   X_idx,
                                                   party_idx,
                                                   resp_party_idx,
                                                   experiment_idx = NULL,
                                                   factor_order = NULL,
                                                   context_present = NULL,
                                                   schema_dropout_masks = NULL,
                                                   out_dim = NULL,
                                                   dtype = NULL) {
  if (!isTRUE(neural_additive_utility_enabled(model_info, params))) {
    n_batch <- tryCatch(ai(X_idx$shape[[1]]), error = function(e) 1L)
    return(strenv$jnp$zeros(list(ai(n_batch), ai(out_dim %||% 1L)), dtype = dtype %||% strenv$dtj))
  }
  cand_info <- neural_build_candidate_tokens_hard(
    X_idx,
    party_idx,
    model_info = model_info,
    resp_party_idx = resp_party_idx,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    params = params,
    return_mask = TRUE,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_masks
  )
  neural_additive_logits_from_candidate_tokens(
    tokens = cand_info$tokens,
    token_mask = cand_info$mask,
    params = params,
    model_info = model_info,
    out_dim = out_dim,
    dtype = dtype
  )
}

neural_additive_pair_delta_prepared <- function(params,
                                                model_info,
                                                Xl,
                                                Xr,
                                                pl,
                                                pr,
                                                resp_p,
                                                experiment_idx = NULL,
                                                factor_order = NULL,
                                                context_present = NULL,
                                                schema_dropout_left = NULL,
                                                schema_dropout_right = NULL,
                                                out_dim = NULL,
                                                dtype = NULL) {
  if (!isTRUE(neural_additive_utility_enabled(model_info, params))) {
    n_batch <- tryCatch(ai(Xl$shape[[1]]), error = function(e) 1L)
    return(strenv$jnp$zeros(list(ai(n_batch), ai(out_dim %||% 1L)), dtype = dtype %||% strenv$dtj))
  }
  left <- neural_additive_single_logits_prepared(
    params = params,
    model_info = model_info,
    X_idx = Xl,
    party_idx = pl,
    resp_party_idx = resp_p,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_left,
    out_dim = out_dim,
    dtype = dtype
  )
  right <- neural_additive_single_logits_prepared(
    params = params,
    model_info = model_info,
    X_idx = Xr,
    party_idx = pr,
    resp_party_idx = resp_p,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_right,
    out_dim = out_dim,
    dtype = dtype
  )
  left - right
}

neural_apply_cross_term <- function(logits, phi_left, phi_right,
                                    M_cross, W_cross_out = NULL,
                                    out_dim = NULL, dtype = NULL) {
  if (is.null(M_cross)) {
    return(logits)
  }
  cross_term <- strenv$jnp$einsum("nm,mp,np->n", phi_left, M_cross, phi_right)
  cross_term <- strenv$jnp$reshape(cross_term, list(-1L, 1L))
  if (is.null(W_cross_out)) {
    # M_cross without its output projection can only come from a corrupted or
    # legacy artifact. Zero-filling here silently removed the bilinear term
    # from predictions; fail loudly like the sibling artifact checks.
    stop(
      "Model params contain M_cross but no W_cross_out; the cross-term output ",
      "projection is missing. The artifact is corrupted or from an ",
      "incompatible strategize version -- refit or re-export it.",
      call. = FALSE
    )
  }
  cross_out <- strenv$jnp$reshape(W_cross_out, list(1L, -1L))
  logits + cross_term * cross_out
}

neural_add_segment_embedding <- function(tokens, segment_idx, model_info, params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  if (is.null(params$E_segment)) {
    return(tokens)
  }
  seg_vec <- strenv$jnp$take(params$E_segment, ai(segment_idx), axis = 0L)
  seg_tok <- strenv$jnp$reshape(seg_vec, list(1L, 1L, ai(model_info$model_dims)))
  tokens + seg_tok
}

neural_token_family_levels <- function(include_candidate_group = TRUE,
                                       include_relation = TRUE,
                                       include_stage = TRUE,
                                       include_respondent_group = TRUE,
                                       include_matchup = TRUE,
                                       include_place = FALSE,
                                       include_time = FALSE,
                                       include_readout_cls = FALSE) {
  levels <- c(
    "factor_fused",
    "covariate_fused",
    "experiment",
    "choice",
    "separator"
  )
  if (isTRUE(include_candidate_group)) {
    levels <- append(levels, "party", after = 1L)
  }
  if (isTRUE(include_relation)) {
    insert_after <- if (isTRUE(include_candidate_group)) 2L else 1L
    levels <- append(levels, "relation", after = insert_after)
  }
  if (isTRUE(include_stage)) {
    levels <- append(levels, "stage", after = match("experiment", levels))
  }
  if (isTRUE(include_respondent_group)) {
    insert_after <- if (isTRUE(include_stage)) {
      match("stage", levels)
    } else {
      match("experiment", levels)
    }
    levels <- append(levels, "resp_party", after = insert_after)
  }
  if (isTRUE(include_matchup)) {
    insert_after <- if (isTRUE(include_respondent_group)) {
      match("resp_party", levels)
    } else if (isTRUE(include_stage)) {
      match("stage", levels)
    } else {
      match("experiment", levels)
    }
    levels <- append(levels, "matchup", after = insert_after)
  }
  if (isTRUE(include_place)) {
    levels <- append(levels, "place", after = match("experiment", levels))
  }
  if (isTRUE(include_time)) {
    insert_after <- if (isTRUE(include_place)) {
      match("place", levels)
    } else {
      match("experiment", levels)
    }
    levels <- append(levels, "time", after = insert_after)
  }
  if (isTRUE(include_readout_cls)) {
    levels <- append(
      levels,
      c("respondent_cls", "candidate_cls"),
      after = match("choice", levels) - 1L
    )
  }
  levels
}

neural_factor_tokenization <- function(model_info = NULL, mode = NULL) {
  mode_use <- tolower(as.character(mode %||% model_info$factor_tokenization %||% "fused"))
  if (!identical(mode_use, "fused")) {
    stop(
      "This neural model uses legacy factor tokenization. Refit under fused attribute tokenization.",
      call. = FALSE
    )
  }
  mode_use
}

neural_default_max_factor_tokens <- function() {
  256L
}

neural_factor_fused_token_width <- function() {
  1L
}

neural_resolve_max_factor_tokens <- function(value = NULL) {
  value_use <- if (is.null(value)) {
    neural_default_max_factor_tokens()
  } else {
    value
  }
  value_use <- suppressWarnings(as.integer(value_use)[[1L]])
  if (is.na(value_use) || value_use < 0L) {
    stop("'max_factor_tokens' must be a single non-negative integer.", call. = FALSE)
  }
  value_use
}

neural_max_factor_token_slots <- function(model_info = NULL,
                                    max_factor_tokens = NULL) {
  token_budget <- if (is.null(max_factor_tokens)) {
    model_info$max_factor_tokens %||% neural_default_max_factor_tokens()
  } else {
    max_factor_tokens
  }
  token_budget <- neural_resolve_max_factor_tokens(token_budget)
  as.integer(floor(token_budget / neural_factor_fused_token_width()))
}

neural_validate_factor_token_budget <- function(n_factors,
                                                max_factor_tokens = NULL,
                                                context = "FM fused factor encoder") {
  n_factors <- as.integer(n_factors %||% 0L)
  max_token_slots <- neural_max_factor_token_slots(
    max_factor_tokens = max_factor_tokens
  )
  if (n_factors > max_token_slots) {
    stop(
      sprintf(
        "%s received %d factors but max_factor_tokens=%d only supports %d fused factor tokens.",
        context,
        n_factors,
        neural_resolve_max_factor_tokens(max_factor_tokens),
        max_token_slots
      ),
      call. = FALSE
    )
  }
  invisible(n_factors)
}

neural_factor_order_from_names <- function(order_names,
                                           factor_names,
                                           error_on_partial = FALSE) {
  factor_names <- as.character(factor_names %||% character(0))
  order_names <- as.character(order_names %||% character(0))
  if (length(order_names) < 1L || length(factor_names) < 1L) {
    return(integer(0))
  }
  idx <- match(order_names, factor_names)
  matched <- !is.na(idx)
  if (isTRUE(error_on_partial) && any(matched) && !all(matched)) {
    stop(
      "Prediction-time factor names must either all match the fitted factor schema or none of them may match.",
      call. = FALSE
    )
  }
  if (isTRUE(error_on_partial) && anyDuplicated(order_names[matched]) > 0L) {
    stop(
      "Prediction-time factor names must not contain duplicates.",
      call. = FALSE
    )
  }
  as.integer(idx[matched] - 1L)
}

neural_factor_order_lookup_matrix <- function(order_list,
                                              max_factor_tokens = NULL) {
  order_list <- order_list %||% list()
  max_token_slots <- neural_max_factor_token_slots(
    max_factor_tokens = max_factor_tokens
  )
  if (length(order_list) < 1L) {
    return(matrix(-1L, nrow = 0L, ncol = max_token_slots))
  }
  out <- matrix(-1L, nrow = length(order_list), ncol = max_token_slots)
  for (i in seq_along(order_list)) {
    idx <- as.integer(order_list[[i]] %||% integer(0))
    if (length(idx) > max_token_slots) {
      stop(
        sprintf(
          "Factor order %d contains %d factors but max_factor_tokens=%d only supports %d fused factor tokens.",
          i,
          length(idx),
          neural_resolve_max_factor_tokens(max_factor_tokens),
          max_token_slots
        ),
        call. = FALSE
      )
    }
    if (length(idx) > 0L) {
      out[i, seq_along(idx)] <- idx
    }
  }
  out
}

neural_build_default_factor_order_matrix <- function(order_idx,
                                                     n_rows,
                                                     max_factor_tokens = NULL) {
  order_idx <- as.integer(order_idx %||% integer(0))
  max_token_slots <- neural_max_factor_token_slots(
    max_factor_tokens = max_factor_tokens
  )
  if (length(order_idx) > max_token_slots) {
    stop(
      sprintf(
        "Default factor order contains %d factors but max_factor_tokens=%d only supports %d fused factor tokens.",
        length(order_idx),
        neural_resolve_max_factor_tokens(max_factor_tokens),
        max_token_slots
      ),
      call. = FALSE
    )
  }
  out <- matrix(-1L, nrow = as.integer(n_rows), ncol = max_token_slots)
  if (length(order_idx) > 0L) {
    out[, seq_along(order_idx)] <- rep(order_idx, each = as.integer(n_rows))
  }
  out
}

neural_default_max_covariate_tokens <- function() {
  512L
}

neural_covariate_fused_token_width <- function() {
  1L
}

neural_resolve_max_covariate_tokens <- function(value = NULL) {
  out <- if (is.null(value)) {
    neural_default_max_covariate_tokens()
  } else {
    as.integer(value)
  }
  if (length(out) != 1L || is.na(out) || out < 0L) {
    stop("'max_covariate_tokens' must be a single non-negative integer.", call. = FALSE)
  }
  out
}

neural_max_covariate_token_slots <- function(model_info = NULL,
                                       max_covariate_tokens = NULL) {
  token_budget <- if (is.null(max_covariate_tokens)) {
    model_info$max_covariate_tokens %||% neural_default_max_covariate_tokens()
  } else {
    max_covariate_tokens
  }
  token_budget <- neural_resolve_max_covariate_tokens(token_budget)
  as.integer(floor(token_budget / neural_covariate_fused_token_width()))
}

neural_validate_covariate_token_budget <- function(n_covariates,
                                                   max_covariate_tokens = NULL,
                                                   context = "Covariate encoder") {
  n_covariates <- as.integer(n_covariates %||% 0L)
  max_token_slots <- neural_max_covariate_token_slots(
    max_covariate_tokens = max_covariate_tokens
  )
  if (n_covariates > max_token_slots) {
    stop(
      sprintf(
        "%s received %d covariates but max_covariate_tokens=%d only supports %d fused covariate tokens.",
        context,
        n_covariates,
        neural_resolve_max_covariate_tokens(max_covariate_tokens),
        max_token_slots
      ),
      call. = FALSE
    )
  }
  invisible(max_token_slots)
}

neural_covariate_order_from_names <- function(order_names, covariate_names) {
  covariate_names <- as.character(covariate_names %||% character(0))
  order_names <- as.character(order_names %||% character(0))
  if (length(order_names) < 1L || length(covariate_names) < 1L) {
    return(integer(0))
  }
  idx <- match(order_names, covariate_names) - 1L
  unmatched <- order_names[is.na(idx)]
  if (length(unmatched) > 0L) {
    # A typo'd covariate name at adaptation/prediction time previously removed
    # the token silently; mirror the factor-order helper's strictness.
    warning(sprintf(paste0(
      "Covariate order name(s) not found in the model's covariate schema and ",
      "dropped: %s. Available covariates: %s."
    ),
    paste(unmatched, collapse = ", "),
    paste(utils::head(covariate_names, 20L), collapse = ", ")
    ), call. = FALSE)
  }
  idx <- idx[!is.na(idx) & idx >= 0L]
  as.integer(idx)
}

neural_covariate_order_lookup_matrix <- function(order_list,
                                                 max_covariate_tokens = NULL) {
  order_list <- order_list %||% list()
  max_token_slots <- neural_max_covariate_token_slots(
    max_covariate_tokens = max_covariate_tokens
  )
  if (length(order_list) < 1L) {
    return(NULL)
  }
  out <- matrix(-1L, nrow = length(order_list), ncol = max_token_slots)
  for (i in seq_along(order_list)) {
    ord <- as.integer(order_list[[i]] %||% integer(0))
    if (length(ord) > max_token_slots) {
      stop(
        sprintf(
          "Covariate order %d contains %d covariates but max_covariate_tokens=%d only supports %d fused covariate tokens.",
          i,
          length(ord),
          neural_resolve_max_covariate_tokens(max_covariate_tokens),
          max_token_slots
        ),
        call. = FALSE
      )
    }
    if (length(ord) > 0L) {
      out[i, seq_along(ord)] <- ord
    }
  }
  out
}

neural_build_default_covariate_order_matrix <- function(order_idx,
                                                        n_rows,
                                                        max_covariate_tokens = NULL) {
  order_idx <- as.integer(order_idx %||% integer(0))
  max_token_slots <- neural_max_covariate_token_slots(
    max_covariate_tokens = max_covariate_tokens
  )
  if (length(order_idx) > max_token_slots) {
    stop(
      sprintf(
        "Default covariate order contains %d covariates but max_covariate_tokens=%d only supports %d fused covariate tokens.",
        length(order_idx),
        neural_resolve_max_covariate_tokens(max_covariate_tokens),
        max_token_slots
      ),
      call. = FALSE
    )
  }
  out <- matrix(-1L, nrow = as.integer(n_rows), ncol = max_token_slots)
  if (length(order_idx) > 0L) {
    out[, seq_along(order_idx)] <- matrix(
      rep(order_idx, each = as.integer(n_rows)),
      nrow = as.integer(n_rows)
    )
  }
  out
}

neural_resolve_factor_order_jnp <- function(model_info,
                                            factor_order = NULL,
                                            experiment_idx = NULL,
                                            n_batch = 1L,
                                            n_factors = NULL) {
  if (!identical(neural_factor_tokenization(model_info), "fused")) {
    return(NULL)
  }
  max_token_slots <- neural_max_factor_token_slots(
    model_info = model_info,
    max_factor_tokens = model_info$max_factor_tokens %||% NULL
  )
  if (max_token_slots < 1L) {
    return(NULL)
  }
  n_factors_use <- if (is.null(n_factors)) {
    factor_text <- model_info$factor_name_text %||% NULL
    if (!is.null(factor_text) && neural_has_shape(factor_text)) {
      ai(factor_text$shape[[1]])
    } else {
      length(factor_text %||% list())
    }
  } else {
    as.integer(n_factors)
  }
  if (n_factors_use < 1L) {
    return(NULL)
  }
  if (!is.null(factor_order)) {
    order_mat <- neural_as_jnp_matrix(factor_order, dtype = strenv$jnp$int32)
    if (ai(order_mat$shape[[2]]) != max_token_slots) {
      stop(
        sprintf(
          "factor_order must have width %d under max_factor_tokens=%d.",
          max_token_slots,
          neural_resolve_max_factor_tokens(model_info$max_factor_tokens %||% NULL)
        ),
        call. = FALSE
      )
    }
    if (ai(order_mat$shape[[1]]) == 1L && n_batch > 1L) {
      order_mat <- order_mat * strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$jnp$int32)
    }
    return(order_mat)
  }
  if (!is.null(experiment_idx) && length(model_info$factor_order_by_experiment %||% list()) > 0L) {
    lookup <- neural_factor_order_lookup_matrix(
      order_list = model_info$factor_order_by_experiment %||% list(),
      max_factor_tokens = model_info$max_factor_tokens %||% NULL
    )
    if (!is.null(lookup) && nrow(lookup) > 0L) {
      lookup_jnp <- strenv$jnp$array(lookup)$astype(strenv$jnp$int32)
      exp_idx <- neural_as_jnp_vector(experiment_idx, dtype = strenv$jnp$int32)
      if (ai(exp_idx$shape[[1]]) == 1L && n_batch > 1L) {
        exp_idx <- exp_idx * strenv$jnp$ones(list(n_batch), dtype = strenv$jnp$int32)
      }
      exp_idx <- strenv$jnp$clip(exp_idx, ai(0L), ai(nrow(lookup) - 1L))
      return(strenv$jnp$take(lookup_jnp, exp_idx, axis = 0L))
    }
  }
  default_mat <- neural_build_default_factor_order_matrix(
    order_idx = model_info$default_factor_order %||% integer(0),
    n_rows = n_batch,
    max_factor_tokens = model_info$max_factor_tokens %||% NULL
  )
  strenv$jnp$array(default_mat)$astype(strenv$jnp$int32)
}

neural_token_family_index <- function(model_info, family_name) {
  levels <- as.character(model_info$token_family_levels %||% neural_token_family_levels())
  idx <- match(as.character(family_name), levels)
  if (is.na(idx)) {
    return(NULL)
  }
  as.integer(idx - 1L)
}

neural_numpyro_prng_key <- function() {
  if (reticulate::py_has_attr(strenv$numpyro, "prng_key")) {
    return(strenv$numpyro$prng_key())
  }
  primitives <- tryCatch(strenv$numpyro$primitives, error = function(e) NULL)
  if (!is.null(primitives) && reticulate::py_has_attr(primitives, "prng_key")) {
    return(primitives$prng_key())
  }
  stop("Installed numpyro does not expose prng_key(); schema_dropout requires NumPyro model RNG access.",
       call. = FALSE)
}

neural_schema_dropout_random_keep <- function(key,
                                              rate,
                                              n_batch,
                                              n_units,
                                              inverted = FALSE) {
  rate <- as.numeric(rate %||% 0)
  n_batch <- ai(n_batch)
  n_units <- ai(n_units)
  if (!is.finite(rate) || rate <= 0 || n_batch < 1L || n_units < 1L) {
    return(NULL)
  }
  if (rate >= 1) {
    return(strenv$jnp$zeros(reticulate::tuple(n_batch, n_units), dtype = strenv$dtj))
  }
  u <- strenv$jax$random$uniform(
    key,
    shape = reticulate::tuple(n_batch, n_units),
    dtype = strenv$dtj
  )
  keep <- (u >= strenv$jnp$array(rate, dtype = strenv$dtj))$astype(strenv$dtj)
  if (!isTRUE(inverted)) {
    return(keep)
  }
  keep / strenv$jnp$array(1 - rate, dtype = strenv$dtj)
}

neural_sample_schema_dropout_masks <- function(model_info,
                                               n_batch = 1L) {
  rates <- neural_resolve_schema_dropout(model_info$schema_dropout %||% NULL)
  if (!any(unlist(rates, use.names = FALSE) > 0)) {
    return(NULL)
  }
  n_batch <- ai(n_batch)
  n_factor_tokens <- neural_max_factor_token_slots(model_info = model_info)
  n_covariate_tokens <- neural_max_covariate_token_slots(model_info = model_info)
  key_count <- 11L
  keys <- strenv$jax$random$split(neural_numpyro_prng_key(), ai(key_count))
  key_at <- function(i) strenv$jnp$take(keys, ai(i - 1L), axis = 0L)
  out <- list(
    experiment_token = neural_schema_dropout_random_keep(
      key_at(1L), rates$experiment_token, n_batch, 1L
    ),
    context_token = neural_schema_dropout_random_keep(
      key_at(2L), rates$context_token, n_batch, 1L
    ),
    factor_token = neural_schema_dropout_random_keep(
      key_at(3L), rates$factor_token, n_batch, n_factor_tokens
    ),
    covariate_token = neural_schema_dropout_random_keep(
      key_at(4L), rates$covariate_token, n_batch, n_covariate_tokens
    ),
    schema_text_factor = neural_schema_dropout_random_keep(
      key_at(5L), rates$schema_text, n_batch, n_factor_tokens, inverted = TRUE
    ),
    schema_text_level = neural_schema_dropout_random_keep(
      key_at(6L), rates$schema_text, n_batch, n_factor_tokens, inverted = TRUE
    ),
    schema_text_covariate = neural_schema_dropout_random_keep(
      key_at(7L), rates$schema_text, n_batch, n_covariate_tokens, inverted = TRUE
    ),
    schema_text_covariate_value = neural_schema_dropout_random_keep(
      key_at(8L), rates$schema_text, n_batch, n_covariate_tokens, inverted = TRUE
    ),
    structural_factor = neural_schema_dropout_random_keep(
      key_at(9L), rates$structural_metadata, n_batch, n_factor_tokens, inverted = TRUE
    ),
    structural_level = neural_schema_dropout_random_keep(
      key_at(10L), rates$structural_metadata, n_batch, n_factor_tokens, inverted = TRUE
    ),
    structural_covariate_metadata = neural_schema_dropout_random_keep(
      key_at(11L), rates$structural_metadata, n_batch, n_covariate_tokens, inverted = TRUE
    )
  )
  out <- Filter(Negate(is.null), out)
  if (!length(out)) NULL else out
}

neural_concat_schema_dropout_masks <- function(left_masks, right_masks) {
  if (is.null(left_masks) && is.null(right_masks)) {
    return(NULL)
  }
  mask_names <- union(names(left_masks %||% list()), names(right_masks %||% list()))
  out <- setNames(vector("list", length(mask_names)), mask_names)
  for (mask_name in mask_names) {
    left <- left_masks[[mask_name]] %||% NULL
    right <- right_masks[[mask_name]] %||% NULL
    out[[mask_name]] <- if (is.null(left)) {
      right
    } else if (is.null(right)) {
      left
    } else {
      strenv$jnp$concatenate(list(left, right), axis = 0L)
    }
  }
  Filter(Negate(is.null), out)
}

neural_schema_dropout_keep <- function(schema_dropout_masks, name) {
  if (is.null(schema_dropout_masks)) {
    return(NULL)
  }
  schema_dropout_masks[[name]] %||% NULL
}

neural_schema_dropout_apply_unit <- function(tokens,
                                             schema_dropout_masks,
                                             name) {
  keep <- neural_schema_dropout_keep(schema_dropout_masks, name)
  if (is.null(tokens) || is.null(keep)) {
    return(tokens)
  }
  tokens * strenv$jnp$reshape(
    keep,
    list(tokens$shape[[1]], tokens$shape[[2]], 1L)
  )
}

neural_schema_dropout_apply_token_mask <- function(token_mask,
                                                  schema_dropout_masks,
                                                  name,
                                                  preserve_one = FALSE) {
  keep <- neural_schema_dropout_keep(schema_dropout_masks, name)
  if (is.null(token_mask) || is.null(keep)) {
    return(token_mask)
  }
  keep_bool <- keep > 0
  dropped <- token_mask & keep_bool
  if (!isTRUE(preserve_one)) {
    return(dropped)
  }
  any_valid <- strenv$jnp$any(token_mask, axis = 1L, keepdims = TRUE)
  any_kept <- strenv$jnp$any(dropped, axis = 1L, keepdims = TRUE)
  first_idx <- strenv$jnp$argmax(
    strenv$jnp$astype(token_mask, strenv$jnp$int32),
    axis = 1L
  )
  token_axis <- strenv$jnp$reshape(
    strenv$jnp$arange(ai(token_mask$shape[[2]])),
    list(1L, ai(token_mask$shape[[2]]))
  )
  first_keep <- (token_axis == strenv$jnp$reshape(first_idx, list(-1L, 1L))) & token_mask
  use_dropped <- strenv$jnp$logical_or(
    any_kept,
    strenv$jnp$logical_not(any_valid)
  )
  strenv$jnp$where(use_dropped, dropped, first_keep)
}

neural_text_matrix_jnp <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  x_jnp <- neural_as_jnp_matrix(x, dtype = strenv$dtj)
  if (is.null(x_jnp) ||
      ai(x_jnp$shape[[1]]) < 1L ||
      ai(x_jnp$shape[[2]]) < 1L) {
    return(NULL)
  }
  x_jnp
}

neural_project_text_matrix <- function(text_matrix, projection) {
  text_jnp <- neural_text_matrix_jnp(text_matrix)
  if (is.null(text_jnp) || is.null(projection)) {
    return(NULL)
  }
  strenv$jnp$einsum("td,dm->tm", text_jnp, projection)
}

neural_experiment_token_mode <- function(model_info) {
  mode <- tolower(as.character(model_info$experiment_token_mode %||% "legacy_id"))
  if (!mode %in% c("description", "hybrid", "legacy_id")) {
    return("legacy_id")
  }
  mode
}

neural_covariate_value_encoding <- function(model_info) {
  mode <- tolower(as.character(model_info$covariate_value_encoding %||% "shared_projection"))
  if (!identical(mode, "shared_projection")) {
    stop("'covariate_value_encoding' must be 'shared_projection'.", call. = FALSE)
  }
  mode
}

neural_prepare_default_text_matrix <- function(text_matrix, n_batch = 1L) {
  text_jnp <- neural_text_matrix_jnp(text_matrix)
  if (is.null(text_jnp)) {
    return(NULL)
  }
  n_batch <- as.integer(n_batch %||% 1L)
  if (ai(text_jnp$shape[[1]]) == 1L && n_batch > 1L) {
    text_jnp <- text_jnp * strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
  }
  text_jnp
}

neural_project_experiment_text <- function(model_info,
                                           params,
                                           experiment_idx = NULL,
                                           n_batch = 1L) {
  if (is.null(params$W_experiment_text)) {
    return(NULL)
  }
  if (!is.null(experiment_idx) && !is.null(model_info$experiment_description_text)) {
    exp_text_proj <- neural_project_text_matrix(
      model_info$experiment_description_text,
      params$W_experiment_text
    )
    if (!is.null(exp_text_proj)) {
      return(strenv$jnp$take(exp_text_proj, experiment_idx, axis = 0L))
    }
  }
  if (!is.null(model_info$default_experiment_text) &&
      isTRUE(model_info$default_experiment_text_present %||% FALSE)) {
    default_text <- neural_prepare_default_text_matrix(
      model_info$default_experiment_text,
      n_batch = n_batch
    )
    if (!is.null(default_text)) {
      return(strenv$jnp$einsum("td,dm->tm", default_text, params$W_experiment_text))
    }
  }
  NULL
}

neural_prepare_context_matrix <- function(context_matrix, n_batch = 1L) {
  context_jnp <- neural_text_matrix_jnp(context_matrix)
  if (is.null(context_jnp)) {
    return(NULL)
  }
  n_batch <- as.integer(n_batch %||% 1L)
  if (ai(context_jnp$shape[[1]]) == 1L && n_batch > 1L) {
    context_jnp <- context_jnp * strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
  }
  context_jnp
}

neural_project_place_context <- function(model_info,
                                         params,
                                         experiment_idx = NULL,
                                         place_embedding = NULL,
                                         n_batch = 1L) {
  if (!isTRUE(neural_place_context_enabled(model_info)) ||
      is.null(params$W_place_context)) {
    return(NULL)
  }
  place_mat <- NULL
  if (!is.null(place_embedding)) {
    place_mat <- neural_prepare_context_matrix(place_embedding, n_batch = n_batch)
  } else if (!is.null(experiment_idx) && !is.null(model_info$place_embedding)) {
    place_proj_all <- neural_project_text_matrix(
      model_info$place_embedding,
      params$W_place_context
    )
    if (!is.null(place_proj_all)) {
      return(strenv$jnp$take(place_proj_all, experiment_idx, axis = 0L))
    }
  }
  if (is.null(place_mat) && !is.null(model_info$default_place_embedding)) {
    place_mat <- neural_prepare_context_matrix(
      model_info$default_place_embedding,
      n_batch = n_batch
    )
  }
  if (is.null(place_mat)) {
    place_mat <- neural_prepare_context_matrix(
      neural_default_place_context_matrix(),
      n_batch = n_batch
    )
  }
  if (is.null(place_mat)) {
    return(NULL)
  }
  strenv$jnp$einsum("td,dm->tm", place_mat, params$W_place_context)
}

neural_project_time_context <- function(model_info,
                                        params,
                                        experiment_idx = NULL,
                                        time_embedding = NULL,
                                        n_batch = 1L) {
  if (!isTRUE(neural_time_context_enabled(model_info)) ||
      is.null(params$W_time_context)) {
    return(NULL)
  }
  time_mat <- NULL
  if (!is.null(time_embedding)) {
    time_mat <- neural_prepare_context_matrix(time_embedding, n_batch = n_batch)
  } else if (!is.null(experiment_idx) && !is.null(model_info$time_embedding)) {
    time_proj_all <- neural_project_text_matrix(
      model_info$time_embedding,
      params$W_time_context
    )
    if (!is.null(time_proj_all)) {
      return(strenv$jnp$take(time_proj_all, experiment_idx, axis = 0L))
    }
  }
  if (is.null(time_mat) && !is.null(model_info$default_time_embedding)) {
    time_mat <- neural_prepare_context_matrix(
      model_info$default_time_embedding,
      n_batch = n_batch
    )
  }
  if (is.null(time_mat)) {
    time_mat <- neural_prepare_context_matrix(
      neural_default_time_context_matrix(),
      n_batch = n_batch
    )
  }
  if (is.null(time_mat)) {
    return(NULL)
  }
  strenv$jnp$einsum("td,dm->tm", time_mat, params$W_time_context)
}

neural_covariate_basis <- function(resp_cov_mat,
                                   resp_cov_present_mat,
                                   model_info) {
  if (is.null(resp_cov_mat) || is.null(resp_cov_present_mat)) {
    return(NULL)
  }
  cov_mean <- model_info$resp_cov_mean %||% NULL
  cov_scale <- model_info$resp_cov_scale %||% NULL
  n_covariates <- ai(resp_cov_mat$shape[[2]])
  mean_mat <- if (!is.null(cov_mean)) {
    neural_as_jnp_matrix(cov_mean, dtype = strenv$dtj)
  } else {
    strenv$jnp$zeros(list(1L, n_covariates), dtype = strenv$dtj)
  }
  scale_mat <- if (!is.null(cov_scale)) {
    neural_as_jnp_matrix(cov_scale, dtype = strenv$dtj)
  } else {
    strenv$jnp$ones(list(1L, n_covariates), dtype = strenv$dtj)
  }
  safe_scale <- strenv$jnp$maximum(scale_mat, strenv$jnp$array(1e-6, dtype = strenv$dtj))
  z <- (resp_cov_mat - mean_mat) / safe_scale
  z <- z * resp_cov_present_mat
  z_sq <- z * z
  strenv$jnp$stack(list(z, z_sq, resp_cov_present_mat), axis = 2L)
}

neural_covariate_z_scores <- function(resp_cov_mat, model_info) {
  if (is.null(resp_cov_mat)) {
    return(NULL)
  }
  cov_mean <- model_info$resp_cov_mean %||% NULL
  cov_scale <- model_info$resp_cov_scale %||% NULL
  n_covariates <- ai(resp_cov_mat$shape[[2]])
  mean_mat <- if (!is.null(cov_mean)) {
    neural_as_jnp_matrix(cov_mean, dtype = strenv$dtj)
  } else {
    strenv$jnp$zeros(list(1L, n_covariates), dtype = strenv$dtj)
  }
  scale_mat <- if (!is.null(cov_scale)) {
    neural_as_jnp_matrix(cov_scale, dtype = strenv$dtj)
  } else {
    strenv$jnp$ones(list(1L, n_covariates), dtype = strenv$dtj)
  }
  safe_scale <- strenv$jnp$maximum(scale_mat, strenv$jnp$array(1e-6, dtype = strenv$dtj))
  (resp_cov_mat - mean_mat) / safe_scale
}

neural_resolve_covariate_distribution_jnp <- function(model_info,
                                                      experiment_idx = NULL,
                                                      n_batch = 1L,
                                                      kind = c("stats", "metadata")) {
  kind <- match.arg(kind)
  if (kind == "stats") {
    mat_list <- model_info$covariate_value_stats_by_experiment %||% list()
    default_mat <- model_info$default_covariate_value_stats %||% NULL
  } else {
    mat_list <- model_info$covariate_value_metadata_by_experiment %||% list()
    default_mat <- model_info$default_covariate_value_metadata %||% NULL
  }
  neural_prepare_covariate_lookup_array(
    mat_list = mat_list,
    default_mat = default_mat,
    experiment_idx = experiment_idx,
    n_batch = n_batch
  )
}

neural_covariate_rank_from_breaks <- function(values, breakpoints) {
  left <- strenv$jnp$take(
    breakpoints,
    strenv$jnp$array(as.integer(0L:5L))$astype(strenv$jnp$int32),
    axis = 2L
  )
  right <- strenv$jnp$take(
    breakpoints,
    strenv$jnp$array(as.integer(1L:6L))$astype(strenv$jnp$int32),
    axis = 2L
  )
  values_expanded <- strenv$jnp$expand_dims(values, axis = 2L)
  seg_idx <- strenv$jnp$sum(values_expanded >= left, axis = 2L) - ai(1L)
  seg_idx <- strenv$jnp$clip(seg_idx, ai(0L), ai(5L))
  seg_idx_expanded <- strenv$jnp$expand_dims(seg_idx, axis = 2L)
  x0 <- strenv$jnp$squeeze(
    strenv$jnp$take_along_axis(left, seg_idx_expanded, axis = 2L),
    axis = 2L
  )
  x1 <- strenv$jnp$squeeze(
    strenv$jnp$take_along_axis(right, seg_idx_expanded, axis = 2L),
    axis = 2L
  )
  rank_knots <- strenv$jnp$array(c(0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0))$astype(strenv$dtj)
  r0 <- strenv$jnp$take(rank_knots, seg_idx, axis = 0L)
  r1 <- strenv$jnp$take(rank_knots, seg_idx + ai(1L), axis = 0L)
  denom <- strenv$jnp$maximum(x1 - x0, strenv$jnp$array(1e-6, dtype = strenv$dtj))
  t <- strenv$jnp$clip((values - x0) / denom, ai(0.), ai(1.))
  strenv$jnp$clip(r0 + t * (r1 - r0), ai(0.), ai(1.))
}

neural_covariate_name_dist_basis <- function(values, stats_tensor) {
  mean_vals <- strenv$jnp$take(stats_tensor, ai(0L), axis = 2L)
  scale_vals <- strenv$jnp$maximum(
    strenv$jnp$take(stats_tensor, ai(1L), axis = 2L),
    strenv$jnp$array(1e-6, dtype = strenv$dtj)
  )
  z <- strenv$jnp$clip((values - mean_vals) / scale_vals, ai(-6.), ai(6.))
  breaks <- strenv$jnp$take(
    stats_tensor,
    strenv$jnp$array(as.integer(2L:8L))$astype(strenv$jnp$int32),
    axis = 2L
  )
  rank_vals <- neural_covariate_rank_from_breaks(values, breaks)
  strenv$jnp$stack(
    list(
      z,
      rank_vals,
      strenv$jnp$sin(ai(pi) * rank_vals),
      strenv$jnp$cos(ai(pi) * rank_vals),
      strenv$jnp$maximum(z, ai(0.)),
      strenv$jnp$maximum(-z, ai(0.)),
      strenv$jnp$ones(values$shape, dtype = strenv$dtj)
    ),
    axis = 2L
  )
}

neural_resolve_covariate_order_jnp <- function(model_info,
                                               resp_cov_order = NULL,
                                               experiment_idx = NULL,
                                               n_batch = 1L,
                                               n_covariates = 0L) {
  max_token_slots <- neural_max_covariate_token_slots(model_info = model_info)
  if (max_token_slots < 1L || n_covariates < 1L) {
    return(NULL)
  }

  if (!is.null(resp_cov_order)) {
    order_arr <- neural_as_jnp_matrix(resp_cov_order, dtype = strenv$jnp$int32)
    if (ai(order_arr$shape[[2]]) != max_token_slots) {
      stop(
        sprintf(
          "resp_cov_order must have width %d under max_covariate_tokens=%d.",
          max_token_slots,
          neural_resolve_max_covariate_tokens(model_info$max_covariate_tokens %||% NULL)
        ),
        call. = FALSE
      )
    }
    if (ai(order_arr$shape[[1]]) == 1L && n_batch > 1L) {
      order_arr <- order_arr * strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$jnp$int32)
    }
    return(order_arr)
  }

  lookup <- neural_covariate_order_lookup_matrix(
    order_list = model_info$covariate_order_by_experiment %||% NULL,
    max_covariate_tokens = model_info$max_covariate_tokens %||% NULL
  )
  if (!is.null(lookup) && !is.null(experiment_idx)) {
    lookup_jnp <- strenv$jnp$array(lookup)$astype(strenv$jnp$int32)
    exp_idx <- neural_as_jnp_vector(experiment_idx, dtype = strenv$jnp$int32)
    if (ai(exp_idx$shape[[1]]) == 1L && n_batch > 1L) {
      exp_idx <- exp_idx * strenv$jnp$ones(list(n_batch), dtype = strenv$jnp$int32)
    }
    return(strenv$jnp$take(lookup_jnp, exp_idx, axis = 0L))
  }

  default_order <- as.integer(
    model_info$default_covariate_order %||%
      if (n_covariates > 0L) seq.int(0L, n_covariates - 1L) else integer(0)
  )
  default_matrix <- neural_build_default_covariate_order_matrix(
    order_idx = default_order,
    n_rows = n_batch,
    max_covariate_tokens = model_info$max_covariate_tokens %||% NULL
  )
  strenv$jnp$array(default_matrix)$astype(strenv$jnp$int32)
}

neural_fuse_attribute_token <- function(fusion_input,
                                        token_mask,
                                        model_info,
                                        params,
                                        prefix,
                                        base_name,
                                        family_name) {
  required <- c(
    base_name,
    paste0("W_", prefix, "_fuse_1"),
    paste0("b_", prefix, "_fuse_1"),
    paste0("W_", prefix, "_fuse_2"),
    paste0("b_", prefix, "_fuse_2")
  )
  missing <- required[vapply(required, function(nm) is.null(params[[nm]]), logical(1))]
  if (length(missing)) {
    stop(
      sprintf(
        "Fused %s tokenization is missing required parameter(s): %s.",
        prefix,
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  hidden_pre <- strenv$jnp$einsum(
    "nsi,ih->nsh",
    fusion_input,
    params[[paste0("W_", prefix, "_fuse_1")]]
  ) + strenv$jnp$reshape(
    params[[paste0("b_", prefix, "_fuse_1")]],
    list(1L, 1L, -1L)
  )
  hidden <- neural_swiglu(hidden_pre)
  fused <- strenv$jnp$einsum(
    "nsh,hm->nsm",
    hidden,
    params[[paste0("W_", prefix, "_fuse_2")]]
  ) + strenv$jnp$reshape(
    params[[paste0("b_", prefix, "_fuse_2")]],
    list(1L, 1L, -1L)
  )
  fused <- fused + strenv$jnp$reshape(
    params[[base_name]],
    list(1L, 1L, ai(model_info$model_dims))
  )
  if (identical(prefix, "factor") &&
      isTRUE(neural_fused_classification_summary_enabled(model_info))) {
    dims <- ai(model_info$model_dims)
    factor_struct_idx <- strenv$jnp$arange(ai(dims), ai(2L * dims))
    level_struct_idx <- strenv$jnp$arange(ai(3L * dims), ai(4L * dims))
    factor_struct_direct <- strenv$jnp$take(fusion_input, factor_struct_idx, axis = 2L)
    level_struct_direct <- strenv$jnp$take(fusion_input, level_struct_idx, axis = 2L)
    fused <- fused + 0.5 * (factor_struct_direct + level_struct_direct)
  }
  fused <- neural_add_token_family_embedding(fused, family_name, model_info, params)
  fused * strenv$jnp$reshape(
    strenv$jnp$astype(token_mask, strenv$dtj),
    list(token_mask$shape[[1]], token_mask$shape[[2]], 1L)
  )
}

neural_build_covariate_fused_tokens <- function(model_info,
                                                params,
                                                resp_cov_mat,
                                                resp_cov_present_mat = NULL,
                                                resp_cov_order = NULL,
                                                experiment_idx = NULL,
                                                n_batch = 1L,
                                                schema_dropout_masks = NULL) {
  n_covariates <- length(model_info$covariate_names %||% character(0))
  max_tokens <- neural_max_covariate_token_slots(model_info = model_info)
  if (n_covariates < 1L || max_tokens < 1L) {
    return(list(tokens = NULL, mask = NULL))
  }

  order_idx <- neural_resolve_covariate_order_jnp(
    model_info = model_info,
    resp_cov_order = resp_cov_order,
    experiment_idx = experiment_idx,
    n_batch = n_batch,
    n_covariates = n_covariates
  )
  if (is.null(order_idx)) {
    return(list(tokens = NULL, mask = NULL))
  }

  if (is.null(resp_cov_mat)) {
    resp_cov_mat <- strenv$jnp$zeros(list(n_batch, n_covariates), dtype = strenv$dtj)
  }
  if (is.null(resp_cov_present_mat)) {
    resp_cov_present_mat <- strenv$jnp$ones(resp_cov_mat$shape, dtype = strenv$dtj)
  }

  idx_valid <- order_idx >= 0L
  idx_safe <- strenv$jnp$maximum(order_idx, ai(0L))
  gathered_present <- strenv$jnp$take_along_axis(
    resp_cov_present_mat,
    idx_safe,
    axis = 1L
  ) * strenv$jnp$astype(idx_valid, strenv$dtj)
  token_mask <- neural_schema_dropout_apply_token_mask(
    idx_valid,
    schema_dropout_masks,
    "covariate_token",
    preserve_one = FALSE
  )
  observed_mask <- gathered_present > 0

  z_all <- neural_covariate_z_scores(resp_cov_mat, model_info)
  gathered_z <- strenv$jnp$take_along_axis(
    z_all,
    idx_safe,
    axis = 1L
  )

  dims <- ai(model_info$model_dims)
  observed_mask_expanded <- strenv$jnp$expand_dims(
    strenv$jnp$astype(observed_mask, strenv$dtj),
    axis = 2L
  )

  cov_text_proj <- neural_project_text_matrix(
    model_info$covariate_name_text,
    params$W_covariate_name_text
  )
  name_tok_base <- strenv$jnp$zeros(list(n_batch, max_tokens, dims), dtype = strenv$dtj)
  if (!is.null(cov_text_proj)) {
    name_tok_base <- strenv$jnp$take(cov_text_proj, idx_safe, axis = 0L)
  }
  name_tok_base <- neural_schema_dropout_apply_unit(
    name_tok_base,
    schema_dropout_masks,
    "schema_text_covariate"
  )
  metadata_dim <- length(neural_covariate_value_metadata_names())
  metadata_tok <- strenv$jnp$zeros(
    list(n_batch, max_tokens, metadata_dim),
    dtype = strenv$dtj
  )

  encoder_mode <- neural_shared_projection_value_encoder(model_info)
  value_tok <- NULL
  if (identical(encoder_mode, "name_dist_moe") &&
      !is.null(params$W_covariate_value_basis) &&
      !is.null(params$W_covariate_value_conditioner_1) &&
      !is.null(params$b_covariate_value_conditioner_1) &&
      !is.null(params$W_covariate_value_conditioner_2) &&
      !is.null(params$b_covariate_value_conditioner_2)) {
    stats_lookup <- neural_resolve_covariate_distribution_jnp(
      model_info = model_info,
      experiment_idx = experiment_idx,
      n_batch = n_batch,
      kind = "stats"
    )
    meta_lookup <- neural_resolve_covariate_distribution_jnp(
      model_info = model_info,
      experiment_idx = experiment_idx,
      n_batch = n_batch,
      kind = "metadata"
    )
    if (!is.null(stats_lookup) && !is.null(meta_lookup)) {
      gathered_values <- strenv$jnp$take_along_axis(
        resp_cov_mat,
        idx_safe,
        axis = 1L
      )
      n_stats <- ai(stats_lookup$shape[[3]])
      n_meta <- ai(meta_lookup$shape[[3]])
      stats_idx <- strenv$jnp$expand_dims(idx_safe, axis = 2L) *
        strenv$jnp$ones(list(n_batch, max_tokens, n_stats), dtype = strenv$jnp$int32)
      meta_idx <- strenv$jnp$expand_dims(idx_safe, axis = 2L) *
        strenv$jnp$ones(list(n_batch, max_tokens, n_meta), dtype = strenv$jnp$int32)
      gathered_stats <- strenv$jnp$take_along_axis(
        stats_lookup,
        stats_idx,
        axis = 1L
      )
      gathered_meta <- strenv$jnp$take_along_axis(
        meta_lookup,
        meta_idx,
        axis = 1L
      )
      metadata_tok <- neural_schema_dropout_apply_unit(
        gathered_meta,
        schema_dropout_masks,
        "structural_covariate_metadata"
      )
      phi <- neural_covariate_name_dist_basis(gathered_values, gathered_stats)
      cond_in <- strenv$jnp$concatenate(list(name_tok_base, metadata_tok), axis = 2L)
      hidden_pre <- strenv$jnp$einsum(
        "nsi,ih->nsh",
        cond_in,
        params$W_covariate_value_conditioner_1
      ) + strenv$jnp$reshape(params$b_covariate_value_conditioner_1, list(1L, 1L, -1L))
      hidden <- strenv$jax$nn$gelu(hidden_pre)
      weight_logits <- strenv$jnp$einsum(
        "nsh,hm->nsm",
        hidden,
        params$W_covariate_value_conditioner_2
      ) + strenv$jnp$reshape(params$b_covariate_value_conditioner_2, list(1L, 1L, -1L))
      mix_weights <- strenv$jax$nn$softmax(weight_logits, axis = -1L)
      basis_proj <- strenv$jnp$einsum(
        "nsk,mkd->nsmd",
        phi,
        params$W_covariate_value_basis
      )
      value_tok <- strenv$jnp$einsum("nsm,nsmd->nsd", mix_weights, basis_proj)
      # Issue 5: Switch-style load-balancing penalty. Push the (token-mask-weighted)
      # mean routing mass toward uniform so no single expert absorbs all weight while
      # the others stop receiving gradients. The deviation term is COLLECTED here
      # and emitted once per model trace by neural_emit_moe_load_balance_factor(),
      # outside the data plate with an explicit N * lambda scale -- emitting the
      # factor from inside this encoder made its effective weight depend on the
      # plate scaling mode (N*lambda under plate subsampling vs only B*lambda
      # under compact training) and on how many times the encoder ran per trace.
      moe_lambda <- suppressWarnings(as.numeric(strenv$moe_load_balance_lambda %||% 0))
      if (isTRUE(strenv$moe_aux_active) && length(moe_lambda) == 1L &&
          is.finite(moe_lambda) && moe_lambda > 0) {
        n_experts_moe <- ai(mix_weights$shape[[3]])
        route_mask <- strenv$jnp$astype(token_mask, strenv$dtj)
        route_denom <- strenv$jnp$maximum(
          strenv$jnp$sum(route_mask),
          strenv$jnp$array(1., dtype = strenv$dtj)
        )
        route_weighted <- mix_weights * strenv$jnp$expand_dims(route_mask, axis = 2L)
        mean_route <- strenv$jnp$sum(
          strenv$jnp$sum(route_weighted, axis = 1L),
          axis = 0L
        ) / route_denom
        uniform_target <- strenv$jnp$array(1. / as.numeric(n_experts_moe), dtype = strenv$dtj)
        moe_aux <- strenv$jnp$sum(strenv$jnp$square(mean_route - uniform_target))
        pending <- strenv$moe_aux_pending
        if (is.null(pending)) pending <- list()
        pending[[length(pending) + 1L]] <- moe_aux
        strenv$moe_aux_pending <- pending
      }
    }
  }
  if (is.null(value_tok)) {
    if (!is.null(params$W_covariate_value_shared)) {
      value_proj <- strenv$jnp$reshape(
        params$W_covariate_value_shared,
        list(1L, 1L, dims)
      )
      value_tok <- strenv$jnp$expand_dims(gathered_z, axis = 2L) * value_proj
    } else {
      value_tok <- strenv$jnp$expand_dims(gathered_z, axis = 2L) *
        strenv$jnp$ones(list(1L, 1L, dims), dtype = strenv$dtj)
    }
  }
  if (!is.null(params$W_covariate_value_text) &&
      !is.null(model_info$covariate_value_text) &&
      !is.null(model_info$covariate_value_text_present)) {
    value_text_tensor <- neural_as_jnp_array(model_info$covariate_value_text, dtype = strenv$dtj)
    value_text_present_lookup <- neural_as_jnp_matrix(
      model_info$covariate_value_text_present,
      dtype = strenv$dtj
    )
    tensor_shape_len <- tryCatch(length(value_text_tensor$shape), error = function(e) 0L)
    if (!is.null(value_text_tensor) &&
        !is.null(value_text_present_lookup) &&
        tensor_shape_len == 3L &&
        ai(value_text_tensor$shape[[1]]) == n_covariates &&
        ai(value_text_present_lookup$shape[[1]]) == n_covariates) {
      n_value_codes <- ai(value_text_tensor$shape[[2]])
      text_dim <- ai(value_text_tensor$shape[[3]])
      if (n_value_codes > 0L &&
          text_dim > 0L &&
          ai(value_text_present_lookup$shape[[2]]) == n_value_codes) {
        gathered_values_for_text <- strenv$jnp$take_along_axis(
          resp_cov_mat,
          idx_safe,
          axis = 1L
        )
        code_idx <- strenv$jnp$floor(
          gathered_values_for_text + strenv$jnp$array(0.5, dtype = strenv$dtj)
        )$astype(strenv$jnp$int32)
        code_idx <- strenv$jnp$clip(code_idx, ai(0L), ai(n_value_codes - 1L))
        flat_idx <- idx_safe * ai(n_value_codes) + code_idx
        flat_text <- strenv$jnp$reshape(
          value_text_tensor,
          list(ai(n_covariates * n_value_codes), text_dim)
        )
        gathered_text <- strenv$jnp$take(flat_text, flat_idx, axis = 0L)
        flat_present <- strenv$jnp$reshape(
          value_text_present_lookup,
          list(ai(n_covariates * n_value_codes))
        )
        gathered_text_present <- strenv$jnp$take(flat_present, flat_idx, axis = 0L) > 0
        value_text_tok <- strenv$jnp$einsum(
          "nsd,dm->nsm",
          gathered_text,
          params$W_covariate_value_text
        )
        value_text_tok <- neural_schema_dropout_apply_unit(
          value_text_tok,
          schema_dropout_masks,
          "schema_text_covariate_value"
        )
        type_lookup <- if (!is.null(model_info$covariate_value_type)) {
          neural_as_jnp_vector(model_info$covariate_value_type, dtype = strenv$jnp$int32)
        } else {
          strenv$jnp$zeros(list(n_covariates), dtype = strenv$jnp$int32)
        }
        gathered_type <- strenv$jnp$take(type_lookup, idx_safe, axis = 0L)
        use_value_text <- gathered_text_present & observed_mask & idx_valid
        nominal_text <- strenv$jnp$expand_dims(use_value_text & (gathered_type == ai(1L)), axis = 2L)
        ordered_text <- strenv$jnp$expand_dims(use_value_text & (gathered_type == ai(2L)), axis = 2L)
        value_tok <- strenv$jnp$where(nominal_text, value_text_tok, value_tok)
        value_tok <- strenv$jnp$where(ordered_text, value_tok + value_text_tok, value_tok)
      }
    }
  }
  missing_value_tok <- if (!is.null(params$E_covariate_missing)) {
    strenv$jnp$reshape(params$E_covariate_missing, list(1L, 1L, dims))
  } else {
    strenv$jnp$zeros(list(1L, 1L, dims), dtype = strenv$dtj)
  }
  value_tok <- strenv$jnp$where(observed_mask_expanded > 0, value_tok, missing_value_tok)

  fusion_input <- strenv$jnp$concatenate(
    list(name_tok_base, value_tok, metadata_tok),
    axis = 2L
  )
  fused_tokens <- neural_fuse_attribute_token(
    fusion_input = fusion_input,
    token_mask = token_mask,
    model_info = model_info,
    params = params,
    prefix = "covariate",
    base_name = "E_covariate_fused_base",
    family_name = "covariate_fused"
  )

  list(tokens = fused_tokens, mask = strenv$jnp$astype(token_mask, strenv$dtj))
}

neural_add_token_family_embedding <- function(tokens,
                                              family_name,
                                              model_info,
                                              params = NULL) {
  if (is.null(params)) {
    params <- model_info$params
  }
  if (is.null(tokens) || is.null(params$E_token_family)) {
    return(tokens)
  }
  family_idx <- neural_token_family_index(model_info, family_name)
  if (is.null(family_idx)) {
    return(tokens)
  }
  family_vec <- strenv$jnp$take(params$E_token_family, ai(family_idx), axis = 0L)
  ndim <- length(tokens$shape)
  if (ndim == 3L) {
    family_tok <- strenv$jnp$reshape(
      family_vec,
      list(1L, 1L, ai(model_info$model_dims))
    )
  } else {
    family_tok <- strenv$jnp$reshape(
      family_vec,
      list(1L, ai(model_info$model_dims))
    )
  }
  tokens + family_tok
}

neural_build_sep_token <- function(model_info, n_batch = NULL, params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  sep_vec <- if (!is.null(params$E_sep)) {
    params$E_sep
  } else {
    strenv$jnp$zeros(list(ai(model_info$model_dims)), dtype = strenv$dtj)
  }
  sep_tok <- strenv$jnp$reshape(sep_vec, list(1L, 1L, ai(model_info$model_dims)))
  sep_tok <- neural_add_token_family_embedding(sep_tok, "separator", model_info, params)
  if (is.null(n_batch)) {
    return(sep_tok)
  }
  sep_tok * strenv$jnp$ones(list(ai(n_batch), 1L, 1L))
}

add_party_rel_tokens <- function(tokens,
                                 party_idx,
                                 model_info,
                                 resp_party_idx = NULL,
                                 params = NULL,
                                 use_role = FALSE,
                                 role_id = NULL,
                                 require_party = NULL,
                                 context_present = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  tok_ndim <- length(tokens$shape)
  if (is.null(require_party)) {
    require_party <- neural_candidate_group_context_enabled(model_info)
  }

  if (isTRUE(require_party) && is.null(params$E_party)) {
    stop("E_party is required for party/rel tokens.")
  }

  party_idx_arr <- strenv$jnp$array(party_idx)
  party_idx_arr <- strenv$jnp$astype(party_idx_arr, strenv$jnp$int32)
  n_batch <- if (tok_ndim == 3L) ai(tokens$shape[[1]]) else 1L
  context_weight <- neural_context_present_float(
    party_idx = party_idx_arr,
    resp_party_idx = resp_party_idx,
    context_present = context_present,
    model_info = model_info,
    n_batch = n_batch
  )

  party_vec <- NULL
  if (!is.null(params$E_party)) {
    party_vec <- strenv$jnp$take(params$E_party, party_idx_arr, axis = 0L)
  }

  if (isTRUE(use_role) && !is.null(params$E_role)) {
    role_id_use <- if (is.null(role_id)) 0L else role_id
    n_roles <- ai(params$E_role$shape[[1]])
    role_use <- if (ai(role_id_use) >= n_roles) 0L else ai(role_id_use)
    role_vec <- strenv$jnp$take(params$E_role, strenv$jnp$array(ai(role_use)), axis = 0L)

    dims <- ai(model_info$model_dims)
    role_add <- if (tok_ndim == 3L) {
      strenv$jnp$reshape(role_vec, list(1L, 1L, dims))
    } else {
      strenv$jnp$reshape(role_vec, list(1L, dims))
    }
    tokens <- tokens + role_add

    if (!is.null(party_vec)) {
      party_vec <- party_vec + strenv$jnp$reshape(role_vec, list(dims))
    }
  }

  if (!is.null(party_vec)) {
    dims <- ai(model_info$model_dims)
    party_tok <- if (tok_ndim == 3L) {
      strenv$jnp$reshape(party_vec, list(tokens$shape[[1]], 1L, dims))
    } else {
      strenv$jnp$reshape(party_vec, list(1L, dims))
    }
    party_tok <- neural_add_token_family_embedding(party_tok, "party", model_info, params)
    if (!is.null(context_weight)) {
      party_scale <- if (tok_ndim == 3L) {
        strenv$jnp$reshape(context_weight, list(tokens$shape[[1]], 1L, 1L))
      } else {
        strenv$jnp$reshape(context_weight, list(1L, 1L))
      }
      party_tok <- party_tok * party_scale
    }
    tokens <- strenv$jnp$concatenate(list(tokens, party_tok),
                                     axis = if (tok_ndim == 3L) 1L else 0L)
  }

  if (!is.null(params$E_rel)) {
    rel_idx <- if (is.null(model_info$cand_party_to_resp_idx) || is.null(resp_party_idx)) {
      if (tok_ndim == 3L) {
        strenv$jnp$full(list(tokens$shape[[1]]), ai(2L))
      } else {
        ai(2L)
      }
    } else {
      cand_map <- strenv$jnp$atleast_1d(
        strenv$jnp$array(model_info$cand_party_to_resp_idx)
      )$astype(strenv$jnp$int32)
      cand_resp_idx <- strenv$jnp$take(cand_map, party_idx_arr, axis = 0L)
      is_known <- cand_resp_idx >= 0L
      resp_arr <- strenv$jnp$array(resp_party_idx)
      is_match <- strenv$jnp$equal(cand_resp_idx, resp_arr)
      strenv$jnp$where(is_match, ai(0L),
                       strenv$jnp$where(is_known, ai(1L), ai(2L)))
    }
    rel_idx <- strenv$jnp$astype(strenv$jnp$array(rel_idx), strenv$jnp$int32)
    rel_vec <- strenv$jnp$take(params$E_rel, rel_idx, axis = 0L)
    dims <- ai(model_info$model_dims)
    rel_tok <- if (tok_ndim == 3L) {
      strenv$jnp$reshape(rel_vec, list(tokens$shape[[1]], 1L, dims))
    } else {
      strenv$jnp$reshape(rel_vec, list(1L, dims))
    }
    rel_tok <- neural_add_token_family_embedding(rel_tok, "relation", model_info, params)
    if (!is.null(context_weight)) {
      rel_scale <- if (tok_ndim == 3L) {
        strenv$jnp$reshape(context_weight, list(tokens$shape[[1]], 1L, 1L))
      } else {
        strenv$jnp$reshape(context_weight, list(1L, 1L))
      }
      rel_tok <- rel_tok * rel_scale
    }
    tokens <- strenv$jnp$concatenate(list(tokens, rel_tok),
                                     axis = if (tok_ndim == 3L) 1L else 0L)
  }

  tokens
}

add_context_tokens <- function(model_info,
                               resp_party_idx,
                               stage_idx = NULL,
                               matchup_idx = NULL,
                               resp_cov = NULL,
                               resp_cov_present = NULL,
                               resp_cov_order = NULL,
                               experiment_idx = NULL,
                               place_embedding = NULL,
                               time_embedding = NULL,
                               params = NULL,
                               batch = FALSE,
                               return_mask = FALSE,
                               context_present = NULL,
                               schema_dropout_masks = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  token_list <- list()
  token_mask_list <- list()
  dims <- ai(model_info$model_dims)
  is_batch <- isTRUE(batch)

  resp_party_idx_use <- if (is.null(resp_party_idx)) 0L else resp_party_idx
  if (!is_batch) {
    resp_party_idx_use <- ai(resp_party_idx_use)
    resp_party_idx_use <- strenv$jnp$atleast_1d(strenv$jnp$array(resp_party_idx_use))
  } else {
    resp_party_idx_use <- strenv$jnp$atleast_1d(resp_party_idx_use)
  }

  stage_idx_use <- NULL
  if (!is.null(stage_idx)) {
    if (!is_batch) {
      if (neural_has_shape(stage_idx)) {
        stage_use <- strenv$jnp$maximum(strenv$jnp$array(stage_idx), ai(0L))
        stage_idx_use <- strenv$jnp$atleast_1d(stage_use)
      } else {
        stage_use <- ai(stage_idx)
        if (ai(stage_use) < 0L) stage_use <- 0L
        stage_idx_use <- strenv$jnp$atleast_1d(strenv$jnp$array(stage_use))
      }
    } else {
      stage_idx_use <- strenv$jnp$atleast_1d(stage_idx)
    }
    stage_idx_use <- strenv$jnp$maximum(
      strenv$jnp$astype(stage_idx_use, strenv$jnp$int32),
      ai(0L)
    )
  }

  matchup_idx_use <- NULL
  if (!is.null(matchup_idx)) {
    matchup_idx_use <- strenv$jnp$atleast_1d(matchup_idx)
  }

  N_batch <- 1L
  if (is_batch) {
    N_batch <- tryCatch(ai(resp_party_idx_use$shape[[1L]]), error = function(e) 1L)
  }
  context_weight <- neural_context_present_float(
    resp_party_idx = resp_party_idx_use,
    context_present = context_present,
    model_info = model_info,
    n_batch = N_batch
  )
  context_token_mask <- if (is.null(context_weight)) {
    strenv$jnp$ones(list(N_batch, 1L), dtype = strenv$dtj)
  } else {
    strenv$jnp$reshape(context_weight, list(N_batch, 1L))
  }

  experiment_idx_use <- experiment_idx
  if (is.null(experiment_idx_use) &&
      !is.null(model_info$default_experiment_index) &&
      !is.na(model_info$default_experiment_index)) {
    experiment_idx_use <- as.integer(model_info$default_experiment_index)
  }
  if (!is.null(experiment_idx_use)) {
    if (!is_batch) {
      experiment_idx_use <- strenv$jnp$atleast_1d(
        strenv$jnp$array(as.integer(experiment_idx_use))
      )$astype(strenv$jnp$int32)
    } else {
      experiment_idx_use <- strenv$jnp$atleast_1d(experiment_idx_use)
      experiment_idx_use <- strenv$jnp$astype(experiment_idx_use, strenv$jnp$int32)
      if (ai(experiment_idx_use$shape[[1]]) == 1L && N_batch > 1L) {
        experiment_idx_use <- experiment_idx_use * strenv$jnp$ones(list(N_batch), dtype = strenv$jnp$int32)
      }
    }
  }

  resp_cov_mat <- NULL
  if (is.null(resp_cov) && !is.null(model_info$resp_cov_mean)) {
    resp_cov <- model_info$resp_cov_mean
  }
  if (!is.null(resp_cov)) {
    resp_cov_mat <- neural_as_jnp_matrix(resp_cov, dtype = strenv$dtj)
    if (ai(resp_cov_mat$shape[[1]]) == 1L && is_batch && N_batch > 1L) {
      resp_cov_mat <- resp_cov_mat * strenv$jnp$ones(list(N_batch, 1L))
    }
  }

  resp_cov_present_mat <- NULL
  if (is.null(resp_cov_present) && !is.null(model_info$resp_cov_default_present)) {
    resp_cov_present <- model_info$resp_cov_default_present
  }
  if (!is.null(resp_cov_present)) {
    resp_cov_present_mat <- neural_as_jnp_matrix(resp_cov_present, dtype = strenv$dtj)
    if (ai(resp_cov_present_mat$shape[[1]]) == 1L && is_batch && N_batch > 1L) {
      resp_cov_present_mat <- resp_cov_present_mat * strenv$jnp$ones(list(N_batch, 1L))
    }
  } else if (!is.null(resp_cov_mat)) {
    resp_cov_present_mat <- strenv$jnp$ones(resp_cov_mat$shape, dtype = strenv$dtj)
  }

  experiment_tok <- NULL
  experiment_token_mode <- neural_experiment_token_mode(model_info)
  if (experiment_token_mode %in% c("legacy_id", "hybrid") &&
      !is.null(params$E_experiment) &&
      !is.null(experiment_idx_use)) {
    exp_vec <- strenv$jnp$take(params$E_experiment, experiment_idx_use, axis = 0L)
    experiment_tok <- strenv$jnp$reshape(exp_vec, list(-1L, 1L, dims))
  }
  if (experiment_token_mode %in% c("description", "hybrid")) {
    exp_text_proj <- neural_project_experiment_text(
      model_info = model_info,
      params = params,
      experiment_idx = experiment_idx_use,
      n_batch = N_batch
    )
    if (!is.null(exp_text_proj)) {
      exp_text_tok <- strenv$jnp$reshape(exp_text_proj, list(-1L, 1L, dims))
      if (is.null(experiment_tok)) {
        experiment_tok <- exp_text_tok
      } else {
        experiment_tok <- experiment_tok + exp_text_tok
      }
    }
  }
  if (!is.null(experiment_tok)) {
    experiment_tok <- neural_add_token_family_embedding(
      experiment_tok,
      "experiment",
      model_info,
      params
    )
    experiment_tok <- neural_schema_dropout_apply_unit(
      experiment_tok,
      schema_dropout_masks,
      "experiment_token"
    )
    token_list[[length(token_list) + 1L]] <- experiment_tok
    experiment_keep <- neural_schema_dropout_keep(
      schema_dropout_masks,
      "experiment_token"
    )
    token_mask_list[[length(token_mask_list) + 1L]] <- if (is.null(experiment_keep)) {
      strenv$jnp$ones(
        list(N_batch, 1L),
        dtype = strenv$dtj
      )
    } else {
      experiment_keep
    }
  }
  place_proj <- neural_project_place_context(
    model_info = model_info,
    params = params,
    experiment_idx = experiment_idx_use,
    place_embedding = place_embedding,
    n_batch = N_batch
  )
  if (!is.null(place_proj)) {
    place_tok <- strenv$jnp$reshape(place_proj, list(-1L, 1L, dims))
    place_tok <- neural_add_token_family_embedding(
      place_tok,
      "place",
      model_info,
      params
    )
    place_tok <- neural_schema_dropout_apply_unit(
      place_tok,
      schema_dropout_masks,
      "context_token"
    )
    token_list[[length(token_list) + 1L]] <- place_tok
    context_keep <- neural_schema_dropout_keep(schema_dropout_masks, "context_token")
    token_mask_list[[length(token_mask_list) + 1L]] <- if (is.null(context_keep)) {
      strenv$jnp$ones(list(N_batch, 1L), dtype = strenv$dtj)
    } else {
      context_keep
    }
  }
  time_proj <- neural_project_time_context(
    model_info = model_info,
    params = params,
    experiment_idx = experiment_idx_use,
    time_embedding = time_embedding,
    n_batch = N_batch
  )
  if (!is.null(time_proj)) {
    time_tok <- strenv$jnp$reshape(time_proj, list(-1L, 1L, dims))
    time_tok <- neural_add_token_family_embedding(
      time_tok,
      "time",
      model_info,
      params
    )
    time_tok <- neural_schema_dropout_apply_unit(
      time_tok,
      schema_dropout_masks,
      "context_token"
    )
    token_list[[length(token_list) + 1L]] <- time_tok
    context_keep <- neural_schema_dropout_keep(schema_dropout_masks, "context_token")
    token_mask_list[[length(token_mask_list) + 1L]] <- if (is.null(context_keep)) {
      strenv$jnp$ones(list(N_batch, 1L), dtype = strenv$dtj)
    } else {
      context_keep
    }
  }
  if (!is.null(params$E_stage) && !is.null(stage_idx_use)) {
    stage_vec <- params$E_stage[resp_party_idx_use, stage_idx_use]
    stage_tok <- strenv$jnp$reshape(stage_vec, list(-1L, 1L, dims))
    stage_tok <- neural_add_token_family_embedding(stage_tok, "stage", model_info, params)
    stage_tok <- stage_tok * strenv$jnp$reshape(context_token_mask, list(N_batch, 1L, 1L))
    token_list[[length(token_list) + 1L]] <- stage_tok
    token_mask_list[[length(token_mask_list) + 1L]] <- context_token_mask
  }
  if (!is.null(params$E_resp_party)) {
    resp_vec <- strenv$jnp$take(params$E_resp_party, resp_party_idx_use, axis = 0L)
    resp_tok <- strenv$jnp$reshape(resp_vec, list(-1L, 1L, dims))
    resp_tok <- neural_add_token_family_embedding(resp_tok, "resp_party", model_info, params)
    resp_tok <- resp_tok * strenv$jnp$reshape(context_token_mask, list(N_batch, 1L, 1L))
    token_list[[length(token_list) + 1L]] <- resp_tok
    token_mask_list[[length(token_mask_list) + 1L]] <- context_token_mask
  }
  if (!is.null(params$E_matchup) && !is.null(matchup_idx_use)) {
    matchup_vec <- strenv$jnp$take(params$E_matchup, matchup_idx_use, axis = 0L)
    matchup_tok <- strenv$jnp$reshape(matchup_vec, list(-1L, 1L, dims))
    matchup_tok <- neural_add_token_family_embedding(matchup_tok, "matchup", model_info, params)
    matchup_tok <- matchup_tok * strenv$jnp$reshape(context_token_mask, list(N_batch, 1L, 1L))
    token_list[[length(token_list) + 1L]] <- matchup_tok
    token_mask_list[[length(token_mask_list) + 1L]] <- context_token_mask
  }
  if (identical(neural_covariate_value_encoding(model_info), "shared_projection") &&
      !is.null(params$E_covariate_fused_base)) {
    cov_tokens <- neural_build_covariate_fused_tokens(
      model_info = model_info,
      params = params,
      resp_cov_mat = resp_cov_mat,
      resp_cov_present_mat = resp_cov_present_mat,
      resp_cov_order = resp_cov_order,
      experiment_idx = experiment_idx_use,
      n_batch = N_batch,
      schema_dropout_masks = schema_dropout_masks
    )
    if (!is.null(cov_tokens$tokens)) {
      token_list[[length(token_list) + 1L]] <- cov_tokens$tokens
      token_mask_list[[length(token_mask_list) + 1L]] <- cov_tokens$mask
    }
  } else if (!is.null(params$E_covariate_id) ||
             !is.null(params$E_covariate_present) ||
             !is.null(params$V_covariate_value) ||
             !is.null(params$E_covariate_start)) {
    stop("Legacy covariate token parameters are incompatible with fused covariate tokenization. Refit the model.", call. = FALSE)
  }

  if (length(token_list) == 0L) {
    if (!isTRUE(return_mask)) {
      return(NULL)
    }
    return(list(tokens = NULL, mask = NULL))
  }
  out_tokens <- strenv$jnp$concatenate(token_list, axis = 1L)
  if (!isTRUE(return_mask)) {
    return(out_tokens)
  }
  out_mask <- if (length(token_mask_list) > 0L) {
    strenv$jnp$concatenate(token_mask_list, axis = 1L)
  } else {
    NULL
  }
  list(tokens = out_tokens, mask = out_mask)
}

neural_build_factor_fused_tokens_hard <- function(X_idx,
                                                  model_info,
                                                  params,
                                                  factor_order = NULL,
                                                  experiment_idx = NULL,
                                                  schema_dropout_masks = NULL) {
  D_local <- ai(X_idx$shape[[2]])
  n_batch <- ai(X_idx$shape[[1]])
  max_tokens <- neural_max_factor_token_slots(model_info = model_info)
  if (D_local < 1L || max_tokens < 1L) {
    return(list(tokens = NULL, mask = NULL))
  }
  if (!isTRUE(neural_model_info_factor_schema_supplied(model_info))) {
    stop(
      "Fused factor tokenization requires factor schema metadata to build candidate tokens.",
      call. = FALSE
    )
  }

  order_idx <- neural_resolve_factor_order_jnp(
    model_info = model_info,
    factor_order = factor_order,
    experiment_idx = experiment_idx,
    n_batch = n_batch,
    n_factors = D_local
  )
  if (is.null(order_idx)) {
    return(list(tokens = NULL, mask = NULL))
  }

  idx_valid <- order_idx >= 0L
  idx_safe <- strenv$jnp$maximum(order_idx, ai(0L))
  token_mask <- neural_schema_dropout_apply_token_mask(
    idx_valid,
    schema_dropout_masks,
    "factor_token",
    preserve_one = TRUE
  )
  dims <- ai(model_info$model_dims)

  gather_idx <- strenv$jnp[["repeat"]](
    strenv$jnp$expand_dims(idx_safe, axis = 2L),
    repeats = ai(dims),
    axis = 2L
  )

  factor_text_tok_all <- strenv$jnp$zeros(list(n_batch, D_local, dims), dtype = strenv$dtj)
  factor_text_proj <- neural_project_text_matrix(
    model_info$factor_name_text,
    params$W_factor_name_text
  )
  if (!is.null(factor_text_proj)) {
    factor_text_tok_all <- strenv$jnp$reshape(
      factor_text_proj,
      list(1L, D_local, dims)
    ) * strenv$jnp$ones(list(n_batch, 1L, 1L), dtype = strenv$dtj)
  }
  factor_struct_tok_all <- strenv$jnp$zeros(list(n_batch, D_local, dims), dtype = strenv$dtj)
  factor_struct_proj <- neural_project_text_matrix(
    model_info$factor_struct_matrix,
    params$W_factor_struct
  )
  if (!is.null(factor_struct_proj)) {
    factor_struct_tok_all <- strenv$jnp$reshape(
      factor_struct_proj,
      list(1L, D_local, dims)
    ) * strenv$jnp$ones(list(n_batch, 1L, 1L), dtype = strenv$dtj)
  }
  factor_text_tok <- strenv$jnp$take_along_axis(
    factor_text_tok_all,
    gather_idx,
    axis = 1L
  )
  factor_struct_tok <- strenv$jnp$take_along_axis(
    factor_struct_tok_all,
    gather_idx,
    axis = 1L
  )
  factor_text_tok <- neural_schema_dropout_apply_unit(
    factor_text_tok,
    schema_dropout_masks,
    "schema_text_factor"
  )
  factor_struct_tok <- neural_schema_dropout_apply_unit(
    factor_struct_tok,
    schema_dropout_masks,
    "structural_factor"
  )

  level_text_token_list <- vector("list", D_local)
  level_struct_token_list <- vector("list", D_local)
  for (d_ in seq_len(D_local)) {
    level_text_d <- strenv$jnp$zeros(list(n_batch, dims), dtype = strenv$dtj)
    if (!is.null(params$W_level_name_text) &&
        !is.null(model_info$level_name_text) &&
        length(model_info$level_name_text) >= d_) {
      level_text_proj <- neural_project_text_matrix(
        model_info$level_name_text[[d_]],
        params$W_level_name_text
      )
      if (!is.null(level_text_proj)) {
        idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
        # Clamp: bare-fit default schemas carry no explicit __holdout__ row, so
        # an unseen-level code (== n_levels) would otherwise gather out of
        # bounds -- jnp.take's OOB fill yields NaN tokens that silently poison
        # every downstream logit. FM/prediction schemas carry the +1 row and
        # are unaffected by the clamp.
        n_rows_text <- ai(level_text_proj$shape[[1]])
        idx_d <- strenv$jnp$clip(idx_d, 0L, ai(n_rows_text - 1L))
        level_text_d <- strenv$jnp$take(level_text_proj, idx_d, axis = 0L)
      }
    }
    level_struct_d <- strenv$jnp$zeros(list(n_batch, dims), dtype = strenv$dtj)
    if (!is.null(params$W_level_struct) &&
        !is.null(model_info$level_struct_matrices) &&
        length(model_info$level_struct_matrices) >= d_) {
      level_struct_proj <- neural_project_text_matrix(
        model_info$level_struct_matrices[[d_]],
        params$W_level_struct
      )
      if (!is.null(level_struct_proj)) {
        idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
        n_rows_struct <- ai(level_struct_proj$shape[[1]])
        idx_d <- strenv$jnp$clip(idx_d, 0L, ai(n_rows_struct - 1L))
        level_struct_d <- strenv$jnp$take(level_struct_proj, idx_d, axis = 0L)
      }
    }
    level_text_token_list[[d_]] <- level_text_d
    level_struct_token_list[[d_]] <- level_struct_d
  }
  level_text_tok_all <- strenv$jnp$stack(level_text_token_list, axis = 1L)
  level_struct_tok_all <- strenv$jnp$stack(level_struct_token_list, axis = 1L)
  level_text_tok <- strenv$jnp$take_along_axis(
    level_text_tok_all,
    gather_idx,
    axis = 1L
  )
  level_struct_tok <- strenv$jnp$take_along_axis(
    level_struct_tok_all,
    gather_idx,
    axis = 1L
  )
  level_text_tok <- neural_schema_dropout_apply_unit(
    level_text_tok,
    schema_dropout_masks,
    "schema_text_level"
  )
  level_struct_tok <- neural_schema_dropout_apply_unit(
    level_struct_tok,
    schema_dropout_masks,
    "structural_level"
  )

  fusion_input <- strenv$jnp$concatenate(
    list(factor_text_tok, factor_struct_tok, level_text_tok, level_struct_tok),
    axis = 2L
  )
  fused_tokens <- neural_fuse_attribute_token(
    fusion_input = fusion_input,
    token_mask = token_mask,
    model_info = model_info,
    params = params,
    prefix = "factor",
    base_name = "E_factor_fused_base",
    family_name = "factor_fused"
  )

  list(tokens = fused_tokens, mask = strenv$jnp$astype(token_mask, strenv$dtj))
}

neural_build_candidate_tokens_hard <- function(X_idx, party_idx, model_info,
                                               resp_party_idx = NULL,
                                               experiment_idx = NULL,
                                               factor_order = NULL,
                                               params = NULL,
                                               return_mask = FALSE,
                                               context_present = NULL,
                                               schema_dropout_masks = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  n_batch <- ai(X_idx$shape[[1]])
  cand_info <- neural_build_factor_fused_tokens_hard(
    X_idx = X_idx,
    model_info = model_info,
    params = params,
    factor_order = factor_order,
    experiment_idx = experiment_idx,
    schema_dropout_masks = schema_dropout_masks
  )
  tokens <- cand_info$tokens %||% strenv$jnp$zeros(
    list(n_batch, 0L, ai(model_info$model_dims)),
    dtype = strenv$dtj
  )
  token_mask <- cand_info$mask %||% strenv$jnp$zeros(list(n_batch, 0L), dtype = strenv$dtj)
  width_before_aux <- ai(tokens$shape[[2]])
  tokens <- add_party_rel_tokens(tokens,
                                 party_idx = party_idx,
                                 resp_party_idx = resp_party_idx,
                                 model_info = model_info,
                                 params = params,
                                 require_party = FALSE,
                                 context_present = context_present)
  if (!isTRUE(return_mask)) {
    return(tokens)
  }
  width_after_aux <- ai(tokens$shape[[2]])
  if (width_after_aux > width_before_aux) {
    aux_present <- neural_context_present_float(
      party_idx = party_idx,
      resp_party_idx = resp_party_idx,
      context_present = context_present,
      model_info = model_info,
      n_batch = n_batch
    )
    aux_mask <- strenv$jnp$ones(
      list(n_batch, ai(width_after_aux - width_before_aux)),
      dtype = strenv$dtj
    )
    if (!is.null(aux_present)) {
      aux_mask <- aux_mask * strenv$jnp$reshape(aux_present, list(n_batch, 1L))
    }
    token_mask <- strenv$jnp$concatenate(list(token_mask, aux_mask), axis = 1L)
  }
  list(tokens = tokens, mask = token_mask)
}

neural_build_context_tokens_batch <- function(model_info,
                                              resp_party_idx,
                                              stage_idx = NULL,
                                              matchup_idx = NULL,
                                              resp_cov = NULL,
                                              resp_cov_present = NULL,
                                              resp_cov_order = NULL,
                                              experiment_idx = NULL,
                                              place_embedding = NULL,
                                              time_embedding = NULL,
                                              params = NULL,
                                              return_mask = FALSE,
                                              context_present = NULL,
                                              schema_dropout_masks = NULL){
  add_context_tokens(model_info = model_info,
                     resp_party_idx = resp_party_idx,
                     stage_idx = stage_idx,
                     matchup_idx = matchup_idx,
                     resp_cov = resp_cov,
                     resp_cov_present = resp_cov_present,
                     resp_cov_order = resp_cov_order,
                     experiment_idx = experiment_idx,
                     place_embedding = place_embedding,
                     time_embedding = time_embedding,
                     params = params,
                     batch = TRUE,
                     return_mask = return_mask,
                     context_present = context_present,
                     schema_dropout_masks = schema_dropout_masks)
}

# SOFT (mean-embedding) relaxation -- the differentiable policy path.
# Each factor's level embeddings are mixed by pi BEFORE the SwiGLU fusion MLP
# and the transformer, so downstream code computes NN(tokens(E_pi[embedding])),
# i.e. f(E[x]) rather than E[f(x)]. This is deliberate: mixing before the
# network is what keeps Q(pi) differentiable in pi for gradient-based policy
# optimization, and it is exact at one-hot pi. At interior points of the
# simplex (stochastic interventions) it is a Jensen-type surrogate of the true
# expected outcome E_{w~pi}[NN(tokens(w))] -- mixing after the fusion MLP
# would remove only the within-token gap, not the sequence-level one, at a
# levels^factors cost. Quantify the surrogate gap at a returned optimum with
# validate_soft_relaxation() (hard-profile Monte Carlo, report-time only).
neural_build_factor_fused_tokens_soft <- function(pi_vec,
                                                  model_info,
                                                  params,
                                                  factor_order = NULL) {
  n_factors <- ai(model_info$n_factors)
  max_tokens <- neural_max_factor_token_slots(model_info = model_info)
  if (n_factors < 1L || max_tokens < 1L) {
    return(list(tokens = NULL, mask = NULL))
  }
  if (!isTRUE(neural_model_info_factor_schema_supplied(model_info))) {
    stop(
      "Fused factor tokenization requires factor schema metadata to build candidate tokens.",
      call. = FALSE
    )
  }

  order_idx <- neural_resolve_factor_order_jnp(
    model_info = model_info,
    factor_order = factor_order,
    n_batch = 1L,
    n_factors = n_factors
  )
  if (is.null(order_idx)) {
    return(list(tokens = NULL, mask = NULL))
  }

  pi_vec <- strenv$jnp$reshape(pi_vec, list(-1L))
  dims <- ai(model_info$model_dims)
  factor_text_token_list <- vector("list", n_factors)
  factor_struct_token_list <- vector("list", n_factors)
  level_text_token_list <- vector("list", n_factors)
  level_struct_token_list <- vector("list", n_factors)
  factor_text_proj <- neural_project_text_matrix(
    model_info$factor_name_text,
    params$W_factor_name_text
  )
  factor_struct_proj <- neural_project_text_matrix(
    model_info$factor_struct_matrix,
    params$W_factor_struct
  )
  match_probability_width <- function(p_full, n_target) {
    n_p <- ai(p_full$shape[[1]])
    n_e <- ai(n_target)
    if (!is.na(n_p) && !is.na(n_e) && n_p != n_e) {
      if (n_e > n_p) {
        # Expected case: the level-embedding table carries one extra
        # __holdout__ row beyond the pi simplex; zero-pad so the holdout row
        # gets no probability mass.
        pad_n <- ai(n_e - n_p)
        pad <- strenv$jnp$zeros(list(pad_n), dtype = strenv$dtj)
        return(strenv$jnp$concatenate(list(p_full, pad), axis = 0L))
      }
      # pi longer than the embedding table can only arise from a schema/pi
      # misalignment; truncation would silently drop probability mass and
      # shrink the mixed token toward zero -- fail loudly instead.
      stop(sprintf(paste0(
        "Soft policy vector has %d entries for a factor whose level-embedding ",
        "table has only %d rows; the policy simplex and the model schema are ",
        "misaligned (was the model trained under a different factor/level set?)."
      ), as.integer(n_p), as.integer(n_e)), call. = FALSE)
    }
    p_full
  }
  for (d_ in seq_len(n_factors)) {
    factor_text_d <- if (!is.null(factor_text_proj)) {
      strenv$jnp$take(factor_text_proj, ai(d_ - 1L), axis = 0L)
    } else {
      strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
    }
    factor_struct_d <- if (!is.null(factor_struct_proj)) {
      strenv$jnp$take(factor_struct_proj, ai(d_ - 1L), axis = 0L)
    } else {
      strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
    }
    factor_text_token_list[[d_]] <- factor_text_d
    factor_struct_token_list[[d_]] <- factor_struct_d

    idx <- model_info$factor_index_list[[d_]]
    idx <- strenv$jnp$atleast_1d(idx)
    p_sub <- strenv$jnp$take(pi_vec, idx, axis = 0L)
    p_full <- apply_implicit_parameterization_jnp(
      p_sub,
      implicit = isTRUE(model_info$implicit),
      axis = 0L,
      clip = TRUE
    )
    level_text_d <- strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
    if (!is.null(params$W_level_name_text) &&
        !is.null(model_info$level_name_text) &&
        length(model_info$level_name_text) >= d_) {
      level_text_proj <- neural_project_text_matrix(
        model_info$level_name_text[[d_]],
        params$W_level_name_text
      )
      if (!is.null(level_text_proj)) {
        p_text <- match_probability_width(p_full, level_text_proj$shape[[1]])
        level_text_d <- strenv$jnp$einsum("l,lm->m", p_text, level_text_proj)
      }
    }
    level_struct_d <- strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
    if (!is.null(params$W_level_struct) &&
        !is.null(model_info$level_struct_matrices) &&
        length(model_info$level_struct_matrices) >= d_) {
      level_struct_proj <- neural_project_text_matrix(
        model_info$level_struct_matrices[[d_]],
        params$W_level_struct
      )
      if (!is.null(level_struct_proj)) {
        p_struct <- match_probability_width(p_full, level_struct_proj$shape[[1]])
        level_struct_d <- strenv$jnp$einsum("l,lm->m", p_struct, level_struct_proj)
      }
    }
    level_text_token_list[[d_]] <- level_text_d
    level_struct_token_list[[d_]] <- level_struct_d
  }

  factor_text_tok_all <- strenv$jnp$reshape(
    strenv$jnp$stack(factor_text_token_list, axis = 0L),
    list(1L, n_factors, dims)
  )
  factor_struct_tok_all <- strenv$jnp$reshape(
    strenv$jnp$stack(factor_struct_token_list, axis = 0L),
    list(1L, n_factors, dims)
  )
  level_text_tok_all <- strenv$jnp$reshape(
    strenv$jnp$stack(level_text_token_list, axis = 0L),
    list(1L, n_factors, dims)
  )
  level_struct_tok_all <- strenv$jnp$reshape(
    strenv$jnp$stack(level_struct_token_list, axis = 0L),
    list(1L, n_factors, dims)
  )
  idx_valid <- order_idx >= 0L
  idx_safe <- strenv$jnp$maximum(order_idx, ai(0L))
  gather_idx <- strenv$jnp[["repeat"]](
    strenv$jnp$expand_dims(idx_safe, axis = 2L),
    repeats = ai(dims),
    axis = 2L
  )
  token_mask <- idx_valid
  factor_text_tok <- strenv$jnp$take_along_axis(factor_text_tok_all, gather_idx, axis = 1L)
  factor_struct_tok <- strenv$jnp$take_along_axis(factor_struct_tok_all, gather_idx, axis = 1L)
  level_text_tok <- strenv$jnp$take_along_axis(level_text_tok_all, gather_idx, axis = 1L)
  level_struct_tok <- strenv$jnp$take_along_axis(level_struct_tok_all, gather_idx, axis = 1L)
  fusion_input <- strenv$jnp$concatenate(
    list(factor_text_tok, factor_struct_tok, level_text_tok, level_struct_tok),
    axis = 2L
  )
  fused_tokens <- neural_fuse_attribute_token(
    fusion_input = fusion_input,
    token_mask = token_mask,
    model_info = model_info,
    params = params,
    prefix = "factor",
    base_name = "E_factor_fused_base",
    family_name = "factor_fused"
  )
  list(tokens = fused_tokens, mask = strenv$jnp$astype(token_mask, strenv$dtj))
}

neural_build_candidate_tokens_soft <- function(pi_vec, party_idx, role_id, model_info, params = NULL,
                                               use_role = FALSE, resp_party_idx = NULL,
                                               factor_order = NULL, return_mask = FALSE,
                                               context_present = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  cand_info <- neural_build_factor_fused_tokens_soft(
    pi_vec = pi_vec,
    model_info = model_info,
    params = params,
    factor_order = factor_order
  )
  tokens <- cand_info$tokens %||% strenv$jnp$zeros(
    list(1L, 0L, ai(model_info$model_dims)),
    dtype = strenv$dtj
  )
  token_mask <- cand_info$mask %||% strenv$jnp$zeros(list(1L, 0L), dtype = strenv$dtj)
  width_before_aux <- ai(tokens$shape[[2]])
  tokens <- add_party_rel_tokens(tokens,
                                 party_idx = party_idx,
                                 role_id = role_id,
                                 use_role = use_role,
                                 resp_party_idx = resp_party_idx,
                                 model_info = model_info,
                                 params = params,
                                 context_present = context_present)
  if (!isTRUE(return_mask)) {
    return(tokens)
  }
  width_after_aux <- ai(tokens$shape[[2]])
  if (width_after_aux > width_before_aux) {
    aux_present <- neural_context_present_float(
      party_idx = party_idx,
      resp_party_idx = resp_party_idx,
      context_present = context_present,
      model_info = model_info,
      n_batch = 1L
    )
    aux_mask <- strenv$jnp$ones(list(1L, ai(width_after_aux - width_before_aux)), dtype = strenv$dtj)
    if (!is.null(aux_present)) {
      aux_mask <- aux_mask * strenv$jnp$reshape(aux_present, list(1L, 1L))
    }
    token_mask <- strenv$jnp$concatenate(list(token_mask, aux_mask), axis = 1L)
  }
  list(tokens = tokens, mask = token_mask)
}

neural_build_context_tokens <- function(model_info,
                                        resp_party_idx = NULL,
                                        stage_idx = NULL,
                                        matchup_idx = NULL,
                                        resp_cov_vec = NULL,
                                        resp_cov_present_vec = NULL,
                                        resp_cov_order = NULL,
                                        experiment_idx = NULL,
                                        place_embedding = NULL,
                                        time_embedding = NULL,
                                        params = NULL,
                                        return_mask = FALSE,
                                        context_present = NULL,
                                        schema_dropout_masks = NULL){
  add_context_tokens(model_info = model_info,
                     resp_party_idx = resp_party_idx,
                     stage_idx = stage_idx,
                     matchup_idx = matchup_idx,
                     resp_cov = resp_cov_vec,
                     resp_cov_present = resp_cov_present_vec,
                     resp_cov_order = resp_cov_order,
                     experiment_idx = experiment_idx,
                     place_embedding = place_embedding,
                     time_embedding = time_embedding,
                     params = params,
                     batch = FALSE,
                     return_mask = return_mask,
                     context_present = context_present,
                     schema_dropout_masks = schema_dropout_masks)
}

neural_build_choice_token <- function(model_info, params = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  if (!is.null(params$E_choice)) {
    choice_vec <- params$E_choice
  } else {
    choice_vec <- strenv$jnp$zeros(list(ai(model_info$model_dims)), dtype = strenv$dtj)
  }
  choice_tok <- strenv$jnp$reshape(choice_vec, list(1L, 1L, ai(model_info$model_dims)))
  neural_add_token_family_embedding(choice_tok, "choice", model_info, params)
}

neural_build_cls_token <- function(model_info,
                                   params = NULL,
                                   param_name,
                                   family_name,
                                   n_batch = 1L) {
  if (is.null(params)) {
    params <- model_info$params
  }
  dims <- ai(model_info$model_dims)
  cls_vec <- params[[param_name]]
  if (is.null(cls_vec)) {
    cls_vec <- strenv$jnp$zeros(list(dims), dtype = strenv$dtj)
  }
  cls_tok <- strenv$jnp$reshape(cls_vec, list(1L, 1L, dims))
  cls_tok <- neural_add_token_family_embedding(cls_tok, family_name, model_info, params)
  cls_tok * strenv$jnp$ones(list(ai(n_batch), 1L, 1L), dtype = strenv$dtj)
}

neural_build_respondent_cls_token <- function(model_info, params = NULL, n_batch = 1L) {
  neural_build_cls_token(
    model_info = model_info,
    params = params,
    param_name = "E_respondent_cls",
    family_name = "respondent_cls",
    n_batch = n_batch
  )
}

neural_build_candidate_cls_token <- function(model_info, params = NULL, n_batch = 1L) {
  neural_build_cls_token(
    model_info = model_info,
    params = params,
    param_name = "E_candidate_cls",
    family_name = "candidate_cls",
    n_batch = n_batch
  )
}

neural_masked_mean_pool <- function(tokens, token_mask = NULL, model_dims = NULL) {
  if (is.null(tokens)) {
    return(NULL)
  }
  n_batch <- ai(tokens$shape[[1]])
  n_tokens <- ai(tokens$shape[[2]])
  dims <- ai(model_dims %||% tokens$shape[[3]])
  if (n_tokens < 1L) {
    return(strenv$jnp$zeros(list(n_batch, dims), dtype = strenv$dtj))
  }
  if (is.null(token_mask)) {
    token_mask <- strenv$jnp$ones(list(n_batch, n_tokens), dtype = strenv$dtj)
  }
  mask <- strenv$jnp$astype(token_mask > 0, tokens$dtype)
  mask_expanded <- strenv$jnp$expand_dims(mask, axis = 2L)
  denom <- strenv$jnp$maximum(
    strenv$jnp$sum(mask, axis = 1L, keepdims = TRUE),
    strenv$jnp$array(1., dtype = tokens$dtype)
  )
  strenv$jnp$sum(tokens * mask_expanded, axis = 1L) / denom
}

neural_readout_bundle_from_transformer <- function(transformer_out,
                                                   token_mask = NULL,
                                                   model_info,
                                                   params = NULL) {
  if (is.null(params)) {
    params <- model_info$params
  }
  readout_tokens <- neural_transformer_readout_tokens(transformer_out)
  state_tokens <- neural_transformer_state_tokens(transformer_out)
  n_batch <- ai(readout_tokens$shape[[1]])
  seq_len <- ai(readout_tokens$shape[[2]])
  dims <- ai(model_info$model_dims)
  cls <- strenv$jnp$squeeze(
    strenv$jnp$take(readout_tokens, strenv$jnp$arange(1L), axis = 1L),
    axis = 1L
  )
  if (seq_len > 1L) {
    tail_idx <- strenv$jnp$arange(ai(1L), ai(seq_len))
    tail_tokens <- strenv$jnp$take(readout_tokens, tail_idx, axis = 1L)
    tail_state <- strenv$jnp$take(state_tokens, tail_idx, axis = 1L)
    tail_mask <- if (is.null(token_mask)) {
      strenv$jnp$ones(list(n_batch, ai(seq_len - 1L)), dtype = strenv$dtj)
    } else {
      strenv$jnp$take(token_mask, tail_idx, axis = 1L)
    }
    mean <- neural_masked_mean_pool(tail_tokens, tail_mask, model_dims = dims)
    pool <- neural_masked_mean_pool(tail_state, tail_mask, model_dims = dims)
  } else {
    mean <- strenv$jnp$zeros(list(n_batch, dims), dtype = strenv$dtj)
    pool <- mean
  }
  final <- neural_rms_norm(cls + pool + mean, params$RMS_final, dims)
  list(
    cls = cls,
    pool = pool,
    mean = mean,
    final = final
  )
}

neural_encode_respondent_tower_prepared <- function(params,
                                                    model_info,
                                                    resp_party_idx,
                                                    resp_cov = NULL,
                                                    resp_cov_present = NULL,
                                                    resp_cov_order = NULL,
                                                    experiment_idx = NULL,
                                                    place_embedding = NULL,
                                                    time_embedding = NULL,
                                                    stage_idx = NULL,
                                                    matchup_idx = NULL,
                                                    context_present = NULL,
                                                    schema_dropout_masks = NULL,
                                                    transformer_model_info = NULL) {
  if (is.null(transformer_model_info)) {
    transformer_model_info <- model_info
  }
  if (is.null(resp_party_idx)) {
    n_batch_seed <- 1L
    if (!is.null(resp_cov)) {
      n_batch_seed <- ai(neural_batch_matrix_jnp(resp_cov, dtype = strenv$dtj)$shape[[1]])
    } else if (!is.null(experiment_idx)) {
      n_batch_seed <- ai(neural_batch_vector_jnp(experiment_idx, dtype = strenv$jnp$int32)$shape[[1]])
    } else if (!is.null(context_present)) {
      n_batch_seed <- ai(neural_batch_vector_jnp(context_present, dtype = strenv$dtj)$shape[[1]])
    }
    resp_party_idx <- strenv$jnp$zeros(list(ai(n_batch_seed)), dtype = strenv$jnp$int32)
  } else {
    resp_party_idx <- neural_batch_vector_jnp(resp_party_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(resp_cov)) {
    resp_cov <- neural_batch_matrix_jnp(resp_cov, dtype = strenv$dtj)
  }
  if (!is.null(resp_cov_present)) {
    resp_cov_present <- neural_batch_matrix_jnp(resp_cov_present, dtype = strenv$dtj)
  }
  if (!is.null(resp_cov_order)) {
    resp_cov_order <- neural_batch_matrix_jnp(resp_cov_order, dtype = strenv$jnp$int32)
  }
  if (!is.null(experiment_idx)) {
    experiment_idx <- neural_batch_vector_jnp(experiment_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(place_embedding)) {
    place_embedding <- neural_batch_matrix_jnp(place_embedding, dtype = strenv$dtj)
  }
  if (!is.null(time_embedding)) {
    time_embedding <- neural_batch_matrix_jnp(time_embedding, dtype = strenv$dtj)
  }
  if (!is.null(stage_idx)) {
    stage_idx <- neural_batch_vector_jnp(stage_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(matchup_idx)) {
    matchup_idx <- neural_batch_vector_jnp(matchup_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(context_present)) {
    context_present <- neural_batch_vector_jnp(context_present, dtype = strenv$dtj)
  }
  n_batch <- ai(resp_party_idx$shape[[1]])
  ctx_info <- neural_build_context_tokens_batch(
    model_info = model_info,
    resp_party_idx = resp_party_idx,
    stage_idx = stage_idx,
    matchup_idx = matchup_idx,
    resp_cov = resp_cov,
    resp_cov_present = resp_cov_present,
    resp_cov_order = resp_cov_order,
    experiment_idx = experiment_idx,
    place_embedding = place_embedding,
    time_embedding = time_embedding,
    params = params,
    return_mask = TRUE,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_masks
  )
  r_cls <- neural_build_respondent_cls_token(model_info, params, n_batch = n_batch)
  cls_mask <- strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
  token_parts <- list(r_cls)
  mask_parts <- list(cls_mask)
  if (!is.null(ctx_info$tokens)) {
    ctx_packed <- neural_pack_token_block(
      tokens = ctx_info$tokens,
      token_mask = ctx_info$mask,
      trim_tokens = neural_active_context_token_budget(model_info)
    )
    token_parts <- c(token_parts, list(ctx_packed$tokens))
    mask_parts <- c(mask_parts, list(ctx_packed$mask))
  }
  tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
  token_mask <- strenv$jnp$concatenate(mask_parts, axis = 1L)
  transformer_out <- neural_run_transformer(
    tokens,
    transformer_model_info,
    params,
    token_mask = token_mask,
    return_details = TRUE
  )
  neural_readout_bundle_from_transformer(
    transformer_out,
    token_mask = token_mask,
    model_info = transformer_model_info,
    params = params
  )
}

neural_encode_candidate_profile_tower_hard <- function(params,
                                                       model_info,
                                                       X_idx,
                                                       party_idx,
                                                       resp_party_idx = NULL,
                                                       experiment_idx = NULL,
                                                       factor_order = NULL,
                                                       context_present = NULL,
                                                       schema_dropout_masks = NULL,
                                                       transformer_model_info = NULL) {
  if (is.null(transformer_model_info)) {
    transformer_model_info <- model_info
  }
  X_idx <- neural_batch_matrix_jnp(X_idx, dtype = strenv$jnp$int32)
  party_idx <- neural_batch_vector_jnp(party_idx, dtype = strenv$jnp$int32)
  if (!is.null(resp_party_idx)) {
    resp_party_idx <- neural_batch_vector_jnp(resp_party_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(experiment_idx)) {
    experiment_idx <- neural_batch_vector_jnp(experiment_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(factor_order)) {
    factor_order <- neural_batch_matrix_jnp(factor_order, dtype = strenv$jnp$int32)
  }
  if (!is.null(context_present)) {
    context_present <- neural_batch_vector_jnp(context_present, dtype = strenv$dtj)
  }
  n_batch <- ai(X_idx$shape[[1]])
  cand_info <- neural_build_candidate_tokens_hard(
    X_idx,
    party_idx,
    model_info = model_info,
    resp_party_idx = resp_party_idx,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    params = params,
    return_mask = TRUE,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_masks
  )
  p_cls <- neural_build_candidate_cls_token(model_info, params, n_batch = n_batch)
  cls_mask <- strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
  seq_info <- neural_pack_candidate_sequence(
    choice_tok = p_cls,
    choice_mask = cls_mask,
    cand_tokens = cand_info$tokens,
    cand_mask = cand_info$mask,
    model_info = model_info,
    preserve_candidate_tail = TRUE
  )
  transformer_out <- neural_run_transformer(
    seq_info$tokens,
    transformer_model_info,
    params,
    token_mask = seq_info$mask,
    return_details = TRUE
  )
  neural_readout_bundle_from_transformer(
    transformer_out,
    token_mask = seq_info$mask,
    model_info = transformer_model_info,
    params = params
  )
}

neural_encode_candidate_profile_tower_soft <- function(params,
                                                       model_info,
                                                       pi_vec,
                                                       party_idx,
                                                       resp_party_idx = NULL,
                                                       factor_order = NULL,
                                                       context_present = NULL,
                                                       transformer_model_info = NULL) {
  if (is.null(transformer_model_info)) {
    transformer_model_info <- model_info
  }
  cand_info <- neural_build_candidate_tokens_soft(
    pi_vec,
    party_idx,
    role_id = 0L,
    model_info = model_info,
    params = params,
    use_role = FALSE,
    resp_party_idx = resp_party_idx,
    factor_order = factor_order,
    return_mask = TRUE,
    context_present = context_present
  )
  p_cls <- neural_build_candidate_cls_token(model_info, params, n_batch = 1L)
  cls_mask <- strenv$jnp$ones(list(1L, 1L), dtype = strenv$dtj)
  seq_info <- neural_pack_candidate_sequence(
    choice_tok = p_cls,
    choice_mask = cls_mask,
    cand_tokens = cand_info$tokens,
    cand_mask = cand_info$mask,
    model_info = model_info,
    preserve_candidate_tail = TRUE
  )
  transformer_out <- neural_run_transformer(
    seq_info$tokens,
    transformer_model_info,
    params,
    token_mask = seq_info$mask,
    return_details = TRUE
  )
  neural_readout_bundle_from_transformer(
    transformer_out,
    token_mask = seq_info$mask,
    model_info = transformer_model_info,
    params = params
  )
}

neural_low_rank_interaction_logits <- function(respondent_final,
                                               candidate_final,
                                               params,
                                               out_dim = NULL,
                                               dtype = NULL,
                                               model_info = NULL,
                                               pairwise_obs = FALSE) {
  if (is.null(params$W_rc_r) ||
      is.null(params$W_rc_c) ||
      is.null(params$W_rc_out)) {
    dtype_use <- dtype %||% strenv$dtj
    n_batch <- ai(candidate_final$shape[[1]])
    out_dim <- out_dim %||% 1L
    return(strenv$jnp$zeros(list(n_batch, ai(out_dim)), dtype = dtype_use))
  }
  r_proj <- strenv$jnp$einsum("nm,mk->nk", respondent_final, params$W_rc_r)
  c_proj <- strenv$jnp$einsum("nm,mk->nk", candidate_final, params$W_rc_c)
  prod <- r_proj * c_proj
  W_rc_out_use <- params$W_rc_out
  if (isTRUE(neural_low_rank_logit_normalization_enabled(model_info, pairwise_obs = pairwise_obs))) {
    r_proj <- neural_rms_norm_no_scale(r_proj)
    c_proj <- neural_rms_norm_no_scale(c_proj)
    prod <- neural_rms_norm_no_scale(r_proj * c_proj)
    W_rc_out_use <- neural_column_rms_normalize(
      params$W_rc_out,
      target_rms = as.numeric(model_info$low_rank_rc_out_target_rms)
    )
  }
  raw <- strenv$jnp$einsum("nk,ko->no", prod, W_rc_out_use)
  alpha <- neural_param_or_default(params, "alpha_rc", 1.0)
  alpha * raw
}

neural_apply_low_rank_interaction <- function(utility,
                                              respondent_final,
                                              candidate_final,
                                              params) {
  if (is.null(params$W_rc_r) ||
      is.null(params$W_rc_c) ||
      is.null(params$W_rc_out)) {
    return(utility)
  }
  utility + neural_low_rank_interaction_logits(
    respondent_final = respondent_final,
    candidate_final = candidate_final,
    params = params,
    out_dim = ai(utility$shape[[2]]),
    dtype = utility$dtype
  )
}

neural_low_rank_pair_delta_prepared <- function(params,
                                                model_info,
                                                Xl,
                                                Xr,
                                                pl,
                                                pr,
                                                resp_p,
                                                resp_c = NULL,
                                                resp_c_present = NULL,
                                                resp_c_order = NULL,
                                                experiment_idx = NULL,
                                                place_embedding = NULL,
                                                time_embedding = NULL,
                                                factor_order = NULL,
                                                stage_idx = NULL,
                                                matchup_idx = NULL,
                                                context_present = NULL,
                                                schema_dropout_context = NULL,
                                                schema_dropout_left = NULL,
                                                schema_dropout_right = NULL,
                                                transformer_model_info = NULL,
                                                out_dim = NULL,
                                                dtype = NULL) {
  if (!isTRUE(neural_has_low_rank_interaction(params, model_info))) {
    n_batch <- ai(Xl$shape[[1]])
    out_dim_use <- out_dim %||% 1L
    return(strenv$jnp$zeros(list(n_batch, ai(out_dim_use)), dtype = dtype %||% strenv$dtj))
  }
  if (is.null(transformer_model_info)) {
    transformer_model_info <- model_info
  }
  n_batch <- ai(Xl$shape[[1]])
  X_all <- strenv$jnp$concatenate(list(Xl, Xr), axis = 0L)
  p_all <- strenv$jnp$concatenate(list(pl, pr), axis = 0L)
  resp_p_all <- strenv$jnp$concatenate(list(resp_p, resp_p), axis = 0L)
  experiment_idx_all <- if (is.null(experiment_idx)) NULL else {
    strenv$jnp$concatenate(list(experiment_idx, experiment_idx), axis = 0L)
  }
  factor_order_all <- if (is.null(factor_order)) NULL else {
    strenv$jnp$concatenate(list(factor_order, factor_order), axis = 0L)
  }
  context_present_all <- if (is.null(context_present)) NULL else {
    strenv$jnp$concatenate(list(context_present, context_present), axis = 0L)
  }
  schema_dropout_all <- neural_concat_schema_dropout_masks(
    schema_dropout_left,
    schema_dropout_right
  )

  resp_readout <- neural_encode_respondent_tower_prepared(
    params = params,
    model_info = model_info,
    resp_party_idx = resp_p,
    resp_cov = resp_c,
    resp_cov_present = resp_c_present,
    resp_cov_order = resp_c_order,
    experiment_idx = experiment_idx,
    place_embedding = place_embedding,
    time_embedding = time_embedding,
    stage_idx = stage_idx,
    matchup_idx = matchup_idx,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_context,
    transformer_model_info = transformer_model_info
  )
  cand_readout <- neural_encode_candidate_profile_tower_hard(
    params = params,
    model_info = model_info,
    X_idx = X_all,
    party_idx = p_all,
    resp_party_idx = resp_p_all,
    experiment_idx = experiment_idx_all,
    factor_order = factor_order_all,
    context_present = context_present_all,
    schema_dropout_masks = schema_dropout_all,
    transformer_model_info = transformer_model_info
  )
  idx_left <- strenv$jnp$arange(n_batch)
  idx_right <- strenv$jnp$arange(n_batch, ai(2L * n_batch))
  cand_left_final <- strenv$jnp$take(cand_readout$final, idx_left, axis = 0L)
  cand_right_final <- strenv$jnp$take(cand_readout$final, idx_right, axis = 0L)
  out_dim_use <- out_dim %||% ai(params$W_rc_out$shape[[2]])
  left_logits <- neural_low_rank_interaction_logits(
    respondent_final = resp_readout$final,
    candidate_final = cand_left_final,
    params = params,
    out_dim = out_dim_use,
    dtype = dtype,
    model_info = model_info,
    pairwise_obs = TRUE
  )
  right_logits <- neural_low_rank_interaction_logits(
    respondent_final = resp_readout$final,
    candidate_final = cand_right_final,
    params = params,
    out_dim = out_dim_use,
    dtype = dtype,
    model_info = model_info,
    pairwise_obs = TRUE
  )
  left_logits - right_logits
}

neural_low_rank_single_utility_prepared <- function(params,
                                                    model_info,
                                                    X_idx,
                                                    party_idx,
                                                    resp_party_idx,
                                                    resp_cov = NULL,
                                                    resp_cov_present = NULL,
                                                    resp_cov_order = NULL,
                                                    experiment_idx = NULL,
                                                    place_embedding = NULL,
                                                    time_embedding = NULL,
                                                    factor_order = NULL,
                                                    stage_idx = NULL,
                                                    matchup_idx = NULL,
                                                    context_present = NULL,
                                                    schema_dropout_context = NULL,
                                                    schema_dropout_candidate = NULL,
                                                    transformer_model_info = NULL,
                                                    out_dim = NULL,
                                                    dtype = NULL) {
  if (!isTRUE(neural_has_low_rank_interaction(params, model_info))) {
    n_batch <- ai(X_idx$shape[[1]])
    out_dim_use <- out_dim %||% 1L
    return(strenv$jnp$zeros(list(n_batch, ai(out_dim_use)), dtype = dtype %||% strenv$dtj))
  }
  if (is.null(transformer_model_info)) {
    transformer_model_info <- model_info
  }
  resp_readout <- neural_encode_respondent_tower_prepared(
    params = params,
    model_info = model_info,
    resp_party_idx = resp_party_idx,
    resp_cov = resp_cov,
    resp_cov_present = resp_cov_present,
    resp_cov_order = resp_cov_order,
    experiment_idx = experiment_idx,
    place_embedding = place_embedding,
    time_embedding = time_embedding,
    stage_idx = stage_idx,
    matchup_idx = matchup_idx,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_context,
    transformer_model_info = transformer_model_info
  )
  cand_readout <- neural_encode_candidate_profile_tower_hard(
    params = params,
    model_info = model_info,
    X_idx = X_idx,
    party_idx = party_idx,
    resp_party_idx = resp_party_idx,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    context_present = context_present,
    schema_dropout_masks = schema_dropout_candidate,
    transformer_model_info = transformer_model_info
  )
  neural_low_rank_interaction_logits(
    respondent_final = resp_readout$final,
    candidate_final = cand_readout$final,
    params = params,
    out_dim = out_dim %||% ai(params$W_rc_out$shape[[2]]),
    dtype = dtype
  )
}

neural_run_transformer_scan_standard <- function(tokens,
                                                 model_info,
                                                 params,
                                                 token_mask = NULL) {
  if (!isTRUE(neural_has_stacked_standard_transformer(params))) {
    return(NULL)
  }
  if (is.null(strenv$jax_transformer_scan_standard)) {
    registered <- tryCatch({
      strategize_register_jax_transformer_helpers()
      TRUE
    }, error = function(e) FALSE)
    if (!isTRUE(registered) || is.null(strenv$jax_transformer_scan_standard)) {
      return(NULL)
    }
  }
  attention_resolve <- neural_attention_resolve_backend(
    model_info,
    role = "self",
    fail_on_forced = TRUE
  )
  strenv$jax_transformer_scan_standard(
    tokens,
    token_mask,
    params$W_q_layers,
    params$W_k_layers,
    params$W_v_layers,
    params$W_o_layers,
    params$W_ff1_layers,
    params$W_ff2_layers,
    params$RMS_attn_layers,
    params$RMS_ff_layers,
    params$RMS_q_layers %||% NULL,
    params$RMS_k_layers %||% NULL,
    params$alpha_attn_layers,
    params$alpha_ff_layers,
    params$RMS_final,
    ai(model_info$model_dims),
    ai(model_info$n_heads),
    ai(model_info$head_dim),
    as.character(attention_resolve$backend),
    neural_attention_dtype_mode(model_info),
    ai(neural_attention_padding_multiple(model_info))
  )
}

neural_run_transformer <- function(tokens,
                                   model_info,
                                   params = NULL,
                                   token_mask = NULL,
                                   return_details = FALSE){
  if (is.null(params)) {
    params <- model_info$params
  }
  residual_mode <- neural_transformer_residual_mode(model_info)
  use_full_attn_residual <- identical(residual_mode, "full_attn")
  if (isTRUE(use_full_attn_residual)) {
    neural_validate_full_attn_compatibility(
      model_info = model_info,
      params = params,
      context = "Neural transformer"
    )
  }
  if (!isTRUE(use_full_attn_residual) &&
      isTRUE(neural_has_stacked_standard_transformer(params))) {
    tokens_final <- neural_run_transformer_scan_standard(
      tokens = tokens,
      model_info = model_info,
      params = params,
      token_mask = token_mask
    )
    if (!is.null(tokens_final)) {
      if (isTRUE(return_details)) {
        return(list(tokens = tokens_final, readout_tokens = tokens_final))
      }
      return(tokens_final)
    }
    # Stacked params usually drop the per-layer W_q_l1... entries
    # (drop_legacy = TRUE), in which case the R loop below cannot run -- it
    # would dereference NULL weights and die in reticulate with an
    # inscrutable error. Fail with the real cause; fall through only when the
    # legacy per-layer entries were kept alongside the stack.
    if (is.null(params[["W_q_l1"]])) {
      stop(
        "The scanned transformer path is unavailable (Python scan helper not ",
        "registered or scan failed), and this model's parameters are stacked ",
        "for scanning, so the per-layer fallback loop cannot run. Re-run ",
        "strategize::initialize_jax() (which registers the scan helpers) or ",
        "reinstall the Python backend.",
        call. = FALSE
      )
    }
  }
  residual_history <- if (isTRUE(use_full_attn_residual)) {
    neural_init_residual_history(tokens)
  } else {
    NULL
  }
  for (l_ in 1L:ai(model_info$model_depth)) {
    Wq <- params[[paste0("W_q_l", l_)]]
    Wk <- params[[paste0("W_k_l", l_)]]
    Wv <- params[[paste0("W_v_l", l_)]]
    Wo <- params[[paste0("W_o_l", l_)]]
    Wff1 <- params[[paste0("W_ff1_l", l_)]]
    Wff2 <- params[[paste0("W_ff2_l", l_)]]
    RMS_attn <- params[[paste0("RMS_attn_l", l_)]]
    RMS_ff <- params[[paste0("RMS_ff_l", l_)]]
    RMS_q <- params[[paste0("RMS_q_l", l_)]]
    RMS_k <- params[[paste0("RMS_k_l", l_)]]
    if (isTRUE(use_full_attn_residual)) {
      h_attn <- neural_full_attn_residual_from_history(
        residual_history,
        pseudo_query = params[[paste0("pseudo_query_attn_l", l_)]],
        model_dims = model_info$model_dims
      )
      tokens_norm <- neural_rms_norm(h_attn, RMS_attn, model_info$model_dims)
    } else {
      alpha_attn <- neural_param_or_default(params, paste0("alpha_attn_l", l_), 1.0)
      alpha_ff <- neural_param_or_default(params, paste0("alpha_ff_l", l_), 1.0)
      tokens_norm <- neural_rms_norm(tokens, RMS_attn, model_info$model_dims)
    }

    Q <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wq)
    K <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wk)
    V <- strenv$jnp$einsum("ntm,mk->ntk", tokens_norm, Wv)

    Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], Q$shape[[2]],
                                    ai(model_info$n_heads), ai(model_info$head_dim)))
    Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]],
                                    ai(model_info$n_heads), ai(model_info$head_dim)))
    Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]],
                                    ai(model_info$n_heads), ai(model_info$head_dim)))
    Qh <- neural_rms_norm(Qh, RMS_q, model_info$head_dim)
    Kh <- neural_rms_norm(Kh, RMS_k, model_info$head_dim)
    context_h <- neural_self_attention_context(Qh, Kh, Vh, token_mask, model_info)
    context <- strenv$jnp$reshape(context_h, list(context_h$shape[[1]],
                                                  context_h$shape[[2]],
                                                  ai(model_info$model_dims)))
    attn_out <- strenv$jnp$einsum("ntm,mk->ntk", context, Wo)

    if (isTRUE(use_full_attn_residual)) {
      residual_history <- neural_append_residual_history(residual_history, attn_out)
      h_ff <- neural_full_attn_residual_from_history(
        residual_history,
        pseudo_query = params[[paste0("pseudo_query_ff_l", l_)]],
        model_dims = model_info$model_dims
      )
      h_ff_norm <- neural_rms_norm(h_ff, RMS_ff, model_info$model_dims)
      ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h_ff_norm, Wff1)
    } else {
      h1 <- tokens + alpha_attn * attn_out
      h1_norm <- neural_rms_norm(h1, RMS_ff, model_info$model_dims)
      ff_pre <- strenv$jnp$einsum("ntm,mf->ntf", h1_norm, Wff1)
    }
    ff_act <- neural_swiglu(ff_pre)
    ff_out <- strenv$jnp$einsum("ntf,fm->ntm", ff_act, Wff2)
    if (isTRUE(use_full_attn_residual)) {
      residual_history <- neural_append_residual_history(residual_history, ff_out)
      tokens <- ff_out
    } else {
      tokens <- h1 + alpha_ff * ff_out
    }
  }
  tokens_final <- neural_rms_norm(tokens, params$RMS_final, model_info$model_dims)
  readout_tokens <- tokens_final
  if (isTRUE(use_full_attn_residual)) {
    h_out <- neural_full_attn_residual_from_history(
      residual_history,
      pseudo_query = params[["pseudo_query_final"]],
      model_dims = model_info$model_dims
    )
    readout_tokens <- neural_rms_norm(h_out, params$RMS_final, model_info$model_dims)
  }
  if (isTRUE(return_details)) {
    return(list(tokens = tokens_final, readout_tokens = readout_tokens))
  }
  tokens_final
}

neural_cross_attend_cls_to_tokens <- function(q_vec, kv_tokens, model_info,
                                              params = NULL, kv_token_mask = NULL,
                                              return_attn = FALSE) {
  if (is.null(params)) {
    params <- model_info$params
  }
  q_use <- q_vec
  kv_use <- kv_tokens
  if (!is.null(params$RMS_cross)) {
    q_reshaped <- strenv$jnp$reshape(q_vec, list(q_vec$shape[[1]], 1L, ai(model_info$model_dims)))
    q_reshaped <- neural_rms_norm(q_reshaped, params$RMS_cross, model_info$model_dims)
    q_use <- strenv$jnp$squeeze(q_reshaped, axis = 1L)
    kv_use <- neural_rms_norm(kv_tokens, params$RMS_cross, model_info$model_dims)
  }

  Q <- strenv$jnp$einsum("nd,dk->nk", q_use, params$W_q_cross)
  K <- strenv$jnp$einsum("ntd,dk->ntk", kv_use, params$W_k_cross)
  V <- strenv$jnp$einsum("ntd,dk->ntk", kv_use, params$W_v_cross)

  Qh <- strenv$jnp$reshape(Q, list(Q$shape[[1]], 1L,
                                  ai(model_info$n_heads), ai(model_info$head_dim)))
  Kh <- strenv$jnp$reshape(K, list(K$shape[[1]], K$shape[[2]],
                                  ai(model_info$n_heads), ai(model_info$head_dim)))
  Vh <- strenv$jnp$reshape(V, list(V$shape[[1]], V$shape[[2]],
                                  ai(model_info$n_heads), ai(model_info$head_dim)))
  Qh <- neural_rms_norm(Qh, params$RMS_q_cross, model_info$head_dim)
  Kh <- neural_rms_norm(Kh, params$RMS_k_cross, model_info$head_dim)
  scale_ <- strenv$jnp$sqrt(strenv$jnp$array(as.numeric(ai(model_info$head_dim))))
  scores <- strenv$jnp$einsum("nqhd,nkhd->nhqk", Qh, Kh) / scale_
  if (!is.null(kv_token_mask)) {
    kv_mask <- strenv$jnp$reshape(
      strenv$jnp$astype(kv_token_mask > 0, scores$dtype),
      list(kv_token_mask$shape[[1]], 1L, 1L, kv_token_mask$shape[[2]])
    )
    neg_inf <- neural_attention_mask_value(scores)
    scores <- strenv$jnp$where(kv_mask > 0, scores, neg_inf)
  }
  attn <- strenv$jax$nn$softmax(scores, axis = -1L)
  ctx_h <- strenv$jnp$einsum("nhqk,nkhd->nqhd", attn, Vh)
  ctx <- strenv$jnp$reshape(ctx_h, list(ctx_h$shape[[1]], 1L, ai(model_info$model_dims)))
  ctx <- strenv$jnp$squeeze(ctx, axis = 1L)

  ctx_out <- strenv$jnp$einsum("nd,dk->nk", ctx, params$W_o_cross)
  if (isTRUE(return_attn)) {
    return(list(ctx = ctx_out, attn = attn))
  }
  ctx_out
}

neural_merge_cross_attn_representation <- function(phi, ctx, params, model_dims) {
  alpha_cross <- neural_param_or_default(params, "alpha_cross", 1.0)
  RMS_merge_cross <- params[["RMS_merge_cross"]]
  if (is.null(RMS_merge_cross)) {
    stop(
      "Pairwise neural models with cross_candidate_encoder='attn' now require ",
      "'RMS_merge_cross'. Refit the neural model under the updated architecture.",
      call. = FALSE
    )
  }
  merged <- phi + alpha_cross * ctx
  neural_rms_norm(merged, RMS_merge_cross, model_dims)
}

neural_prepare_choice_token_batch <- function(model_info, params, n_batch) {
  choice_tok <- neural_build_choice_token(model_info, params)
  choice_tok * strenv$jnp$ones(list(ai(n_batch), 1L, 1L))
}

neural_encode_candidate_core_prepared <- function(params,
                                                  model_info,
                                                  X_idx,
                                                  party_idx,
                                                  resp_party_idx,
                                                  resp_cov = NULL,
                                                  resp_cov_present = NULL,
                                                  resp_cov_order = NULL,
                                                  experiment_idx = NULL,
                                                  place_embedding = NULL,
                                                  time_embedding = NULL,
                                                  factor_order = NULL,
                                                  stage_idx = NULL,
                                                  matchup_idx = NULL,
                                                  context_present = NULL,
                                                  return_tokens = FALSE) {
  X_idx <- neural_batch_matrix_jnp(X_idx, dtype = strenv$jnp$int32)
  party_idx <- neural_batch_vector_jnp(party_idx, dtype = strenv$jnp$int32)
  resp_party_idx <- neural_batch_vector_jnp(resp_party_idx, dtype = strenv$jnp$int32)
  if (!is.null(resp_cov)) {
    resp_cov <- neural_batch_matrix_jnp(resp_cov, dtype = strenv$dtj)
  }
  if (!is.null(resp_cov_present)) {
    resp_cov_present <- neural_batch_matrix_jnp(resp_cov_present, dtype = strenv$dtj)
  }
  if (!is.null(resp_cov_order)) {
    resp_cov_order <- neural_batch_matrix_jnp(resp_cov_order, dtype = strenv$jnp$int32)
  }
  if (!is.null(experiment_idx)) {
    experiment_idx <- neural_batch_vector_jnp(experiment_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(place_embedding)) {
    place_embedding <- neural_batch_matrix_jnp(place_embedding, dtype = strenv$dtj)
  }
  if (!is.null(time_embedding)) {
    time_embedding <- neural_batch_matrix_jnp(time_embedding, dtype = strenv$dtj)
  }
  if (!is.null(factor_order)) {
    factor_order <- neural_batch_matrix_jnp(factor_order, dtype = strenv$jnp$int32)
  }
  if (!is.null(stage_idx)) {
    stage_idx <- neural_batch_vector_jnp(stage_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(matchup_idx)) {
    matchup_idx <- neural_batch_vector_jnp(matchup_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(context_present)) {
    context_present <- neural_batch_vector_jnp(context_present, dtype = strenv$dtj)
  }
  n_batch <- ai(X_idx$shape[[1]])
  choice_tok <- neural_prepare_choice_token_batch(model_info, params, n_batch)
  choice_mask <- strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
  ctx_info <- neural_build_context_tokens_batch(
    model_info = model_info,
    resp_party_idx = resp_party_idx,
    stage_idx = stage_idx,
    matchup_idx = matchup_idx,
    resp_cov = resp_cov,
    resp_cov_present = resp_cov_present,
    resp_cov_order = resp_cov_order,
    experiment_idx = experiment_idx,
    place_embedding = place_embedding,
    time_embedding = time_embedding,
    params = params,
    return_mask = TRUE,
    context_present = context_present
  )
  ctx_tokens <- ctx_info$tokens %||% NULL
  ctx_mask <- ctx_info$mask %||% NULL
  cand_info <- neural_build_candidate_tokens_hard(
    X_idx,
    party_idx,
    model_info = model_info,
    resp_party_idx = resp_party_idx,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    params = params,
    return_mask = TRUE,
    context_present = context_present
  )
  cand_tokens <- cand_info$tokens
  cand_mask <- cand_info$mask
  seq_info <- neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    ctx_tokens = ctx_tokens,
    ctx_mask = ctx_mask,
    cand_tokens = cand_tokens,
    cand_mask = cand_mask,
    model_info = model_info,
    preserve_candidate_tail = isTRUE(return_tokens)
  )
  tokens <- seq_info$tokens
  token_mask <- seq_info$mask
  transformer_out <- neural_run_transformer(
    tokens,
    model_info,
    params,
    token_mask = token_mask,
    return_details = TRUE
  )
  phi <- neural_extract_choice_representation(
    transformer_out,
    token_mask = token_mask,
    include_tail_summary = neural_fused_classification_summary_enabled(model_info)
  )
  if (!isTRUE(return_tokens)) {
    return(phi)
  }
  cand_out <- neural_extract_candidate_tokens(
    transformer_out,
    model_info,
    n_candidate_tokens = neural_candidate_token_count_from_mask(seq_info$cand_mask)
  )
  list(phi = phi, cand_tokens_out = cand_out, cand_token_mask = seq_info$cand_mask)
}

neural_predict_pair_cross_core_prepared <- function(params,
                                                    model_info,
                                                    Xl,
                                                    Xr,
                                                    pl,
                                                    pr,
                                                    resp_p,
                                                    resp_c = NULL,
                                                    resp_c_present = NULL,
                                                    resp_c_order = NULL,
                                                    experiment_idx = NULL,
                                                    place_embedding = NULL,
                                                    time_embedding = NULL,
                                                    factor_order = NULL,
                                                    stage_idx,
                                                    matchup_idx = NULL,
                                                    context_present = NULL) {
  Xl <- neural_batch_matrix_jnp(Xl, dtype = strenv$jnp$int32)
  Xr <- neural_batch_matrix_jnp(Xr, dtype = strenv$jnp$int32)
  pl <- neural_batch_vector_jnp(pl, dtype = strenv$jnp$int32)
  pr <- neural_batch_vector_jnp(pr, dtype = strenv$jnp$int32)
  resp_p <- neural_batch_vector_jnp(resp_p, dtype = strenv$jnp$int32)
  if (!is.null(resp_c)) {
    resp_c <- neural_batch_matrix_jnp(resp_c, dtype = strenv$dtj)
  }
  if (!is.null(resp_c_present)) {
    resp_c_present <- neural_batch_matrix_jnp(resp_c_present, dtype = strenv$dtj)
  }
  if (!is.null(resp_c_order)) {
    resp_c_order <- neural_batch_matrix_jnp(resp_c_order, dtype = strenv$jnp$int32)
  }
  if (!is.null(experiment_idx)) {
    experiment_idx <- neural_batch_vector_jnp(experiment_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(place_embedding)) {
    place_embedding <- neural_batch_matrix_jnp(place_embedding, dtype = strenv$dtj)
  }
  if (!is.null(time_embedding)) {
    time_embedding <- neural_batch_matrix_jnp(time_embedding, dtype = strenv$dtj)
  }
  if (!is.null(factor_order)) {
    factor_order <- neural_batch_matrix_jnp(factor_order, dtype = strenv$jnp$int32)
  }
  stage_idx <- neural_batch_vector_jnp(stage_idx, dtype = strenv$jnp$int32)
  if (!is.null(matchup_idx)) {
    matchup_idx <- neural_batch_vector_jnp(matchup_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(context_present)) {
    context_present <- neural_batch_vector_jnp(context_present, dtype = strenv$dtj)
  }
  n_batch <- ai(Xl$shape[[1]])
  choice_tok <- neural_prepare_choice_token_batch(model_info, params, n_batch)
  choice_mask <- strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
  ctx_info <- neural_build_context_tokens_batch(
    model_info = model_info,
    resp_party_idx = resp_p,
    stage_idx = stage_idx,
    matchup_idx = matchup_idx,
    resp_cov = resp_c,
    resp_cov_present = resp_c_present,
    resp_cov_order = resp_c_order,
    experiment_idx = experiment_idx,
    place_embedding = place_embedding,
    time_embedding = time_embedding,
    params = params,
    return_mask = TRUE,
    context_present = context_present
  )
  ctx_tokens <- ctx_info$tokens %||% NULL
  ctx_mask <- ctx_info$mask %||% NULL
  left_info <- neural_build_candidate_tokens_hard(
    Xl,
    pl,
    model_info = model_info,
    resp_party_idx = resp_p,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    params = params,
    return_mask = TRUE,
    context_present = context_present
  )
  left_tokens <- neural_add_segment_embedding(
    left_info$tokens,
    0L,
    model_info = model_info,
    params = params
  )
  right_info <- neural_build_candidate_tokens_hard(
    Xr,
    pr,
    model_info = model_info,
    resp_party_idx = resp_p,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    params = params,
    return_mask = TRUE,
    context_present = context_present
  )
  right_tokens <- neural_add_segment_embedding(
    right_info$tokens,
    1L,
    model_info = model_info,
    params = params
  )
  sep_tok <- neural_build_sep_token(model_info, n_batch = n_batch, params = params)
  sep_mask <- strenv$jnp$ones(list(n_batch, 1L), dtype = strenv$dtj)
  seq_info <- neural_pack_full_cross_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    sep_tok = sep_tok,
    sep_mask = sep_mask,
    left_tokens = left_tokens,
    left_mask = left_info$mask,
    right_tokens = right_tokens,
    right_mask = right_info$mask,
    model_info = model_info,
    ctx_tokens = ctx_tokens,
    ctx_mask = ctx_mask
  )
  tokens <- seq_info$tokens
  token_mask <- seq_info$mask
  transformer_out <- neural_run_transformer(
    tokens,
    model_info,
    params,
    token_mask = token_mask,
    return_details = TRUE
  )
  cls_out <- neural_extract_choice_representation(transformer_out)
  neural_linear_head(
    cls_out,
    params$W_out,
    params$b_out,
    model_info = model_info,
    pairwise_obs = TRUE
  )
}

neural_predict_pair_core_prepared <- function(params,
                                              model_info,
                                              Xl,
                                              Xr,
                                              pl,
                                              pr,
                                              resp_p,
                                              resp_c = NULL,
                                              resp_c_present = NULL,
                                              resp_c_order = NULL,
                                              experiment_idx = NULL,
                                              place_embedding = NULL,
                                              time_embedding = NULL,
                                              factor_order = NULL,
                                              return_logits = FALSE) {
  Xl <- neural_batch_matrix_jnp(Xl, dtype = strenv$jnp$int32)
  Xr <- neural_batch_matrix_jnp(Xr, dtype = strenv$jnp$int32)
  pl <- neural_batch_vector_jnp(pl, dtype = strenv$jnp$int32)
  pr <- neural_batch_vector_jnp(pr, dtype = strenv$jnp$int32)
  resp_p <- neural_batch_vector_jnp(resp_p, dtype = strenv$jnp$int32)
  if (!is.null(resp_c)) {
    resp_c <- neural_batch_matrix_jnp(resp_c, dtype = strenv$dtj)
  }
  if (!is.null(resp_c_present)) {
    resp_c_present <- neural_batch_matrix_jnp(resp_c_present, dtype = strenv$dtj)
  }
  if (!is.null(resp_c_order)) {
    resp_c_order <- neural_batch_matrix_jnp(resp_c_order, dtype = strenv$jnp$int32)
  }
  if (!is.null(experiment_idx)) {
    experiment_idx <- neural_batch_vector_jnp(experiment_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(place_embedding)) {
    place_embedding <- neural_batch_matrix_jnp(place_embedding, dtype = strenv$dtj)
  }
  if (!is.null(time_embedding)) {
    time_embedding <- neural_batch_matrix_jnp(time_embedding, dtype = strenv$dtj)
  }
  if (!is.null(factor_order)) {
    factor_order <- neural_batch_matrix_jnp(factor_order, dtype = strenv$jnp$int32)
  }
  mode <- neural_cross_encoder_mode(model_info)
  use_cross_encoder <- identical(mode, "full")
  use_cross_term <- identical(mode, "term")
  use_cross_attn <- identical(mode, "attn")

  stage_idx <- neural_stage_index(pl, pr, model_info)
  matchup_idx <- NULL
  if (!is.null(params$E_matchup)) {
    matchup_idx <- neural_matchup_index(pl, pr, model_info)
  }
  context_present <- neural_pair_context_present(pl, pr, resp_p, model_info)

  if (isTRUE(use_cross_encoder)) {
    logits <- neural_predict_pair_cross_core_prepared(
      params = params,
      model_info = model_info,
      Xl = Xl,
      Xr = Xr,
      pl = pl,
      pr = pr,
      resp_p = resp_p,
      resp_c = resp_c,
      resp_c_present = resp_c_present,
      resp_c_order = resp_c_order,
      experiment_idx = experiment_idx,
      place_embedding = place_embedding,
      time_embedding = time_embedding,
      factor_order = factor_order,
      stage_idx = stage_idx,
      matchup_idx = matchup_idx,
      context_present = context_present
    )
    if (isTRUE(neural_pairwise_antisymmetry_strict(model_info))) {
      stage_idx_rev <- neural_stage_index(pr, pl, model_info)
      matchup_idx_rev <- NULL
      if (!is.null(params$E_matchup)) {
        matchup_idx_rev <- neural_matchup_index(pr, pl, model_info)
      }
      context_present_rev <- neural_pair_context_present(pr, pl, resp_p, model_info)
      logits_rev <- neural_predict_pair_cross_core_prepared(
        params = params,
        model_info = model_info,
        Xl = Xr,
        Xr = Xl,
        pl = pr,
        pr = pl,
        resp_p = resp_p,
        resp_c = resp_c,
        resp_c_present = resp_c_present,
        resp_c_order = resp_c_order,
        experiment_idx = experiment_idx,
        place_embedding = place_embedding,
        time_embedding = time_embedding,
        factor_order = factor_order,
        stage_idx = stage_idx_rev,
        matchup_idx = matchup_idx_rev,
        context_present = context_present_rev
      )
      logits <- 0.5 * (logits - logits_rev)
    }
    if (isTRUE(neural_has_low_rank_interaction(params, model_info))) {
      logits <- logits + neural_low_rank_pair_delta_prepared(
        params = params,
        model_info = model_info,
        Xl = Xl,
        Xr = Xr,
        pl = pl,
        pr = pr,
        resp_p = resp_p,
        resp_c = resp_c,
        resp_c_present = resp_c_present,
        resp_c_order = resp_c_order,
        experiment_idx = experiment_idx,
        place_embedding = place_embedding,
        time_embedding = time_embedding,
        factor_order = factor_order,
        stage_idx = stage_idx,
        matchup_idx = matchup_idx,
        context_present = context_present,
        out_dim = ai(logits$shape[[2]]),
        dtype = logits$dtype
      )
    }
    logits <- logits + neural_additive_pair_delta_prepared(
      params = params,
      model_info = model_info,
      Xl = Xl,
      Xr = Xr,
      pl = pl,
      pr = pr,
      resp_p = resp_p,
      experiment_idx = experiment_idx,
      factor_order = factor_order,
      context_present = context_present,
      out_dim = ai(logits$shape[[2]]),
      dtype = logits$dtype
    )
  } else {
    n_batch <- ai(Xl$shape[[1]])
    X_all <- strenv$jnp$concatenate(list(Xl, Xr), axis = 0L)
    p_all <- strenv$jnp$concatenate(list(pl, pr), axis = 0L)
    resp_p_all <- strenv$jnp$concatenate(list(resp_p, resp_p), axis = 0L)
    resp_c_all <- if (is.null(resp_c)) NULL else {
      strenv$jnp$concatenate(list(resp_c, resp_c), axis = 0L)
    }
    resp_c_present_all <- if (is.null(resp_c_present)) NULL else {
      strenv$jnp$concatenate(list(resp_c_present, resp_c_present), axis = 0L)
    }
    resp_c_order_all <- if (is.null(resp_c_order)) NULL else {
      strenv$jnp$concatenate(list(resp_c_order, resp_c_order), axis = 0L)
    }
    factor_order_all <- if (is.null(factor_order)) NULL else {
      strenv$jnp$concatenate(list(factor_order, factor_order), axis = 0L)
    }
    experiment_idx_all <- if (is.null(experiment_idx)) NULL else {
      strenv$jnp$concatenate(list(experiment_idx, experiment_idx), axis = 0L)
    }
    place_embedding_all <- if (is.null(place_embedding)) NULL else {
      strenv$jnp$concatenate(list(place_embedding, place_embedding), axis = 0L)
    }
    time_embedding_all <- if (is.null(time_embedding)) NULL else {
      strenv$jnp$concatenate(list(time_embedding, time_embedding), axis = 0L)
    }
    stage_all <- if (is.null(stage_idx)) NULL else {
      strenv$jnp$concatenate(list(stage_idx, stage_idx), axis = 0L)
    }
    matchup_all <- if (is.null(matchup_idx)) NULL else {
      strenv$jnp$concatenate(list(matchup_idx, matchup_idx), axis = 0L)
    }
    context_present_all <- strenv$jnp$concatenate(
      list(context_present, context_present),
      axis = 0L
    )

    if (isTRUE(use_cross_attn)) {
      enc_all <- neural_encode_candidate_core_prepared(
        params = params,
        model_info = model_info,
        X_idx = X_all,
        party_idx = p_all,
        resp_party_idx = resp_p_all,
        resp_cov = resp_c_all,
        resp_cov_present = resp_c_present_all,
        resp_cov_order = resp_c_order_all,
        experiment_idx = experiment_idx_all,
        place_embedding = place_embedding_all,
        time_embedding = time_embedding_all,
        factor_order = factor_order_all,
        stage_idx = stage_all,
        matchup_idx = matchup_all,
        context_present = context_present_all,
        return_tokens = TRUE
      )
      phi_all <- enc_all$phi
      cand_all <- enc_all$cand_tokens_out
      cand_mask_all <- enc_all$cand_token_mask
    } else {
      phi_all <- neural_encode_candidate_core_prepared(
        params = params,
        model_info = model_info,
        X_idx = X_all,
        party_idx = p_all,
        resp_party_idx = resp_p_all,
        resp_cov = resp_c_all,
        resp_cov_present = resp_c_present_all,
        resp_cov_order = resp_c_order_all,
        experiment_idx = experiment_idx_all,
        place_embedding = place_embedding_all,
        time_embedding = time_embedding_all,
        factor_order = factor_order_all,
        stage_idx = stage_all,
        matchup_idx = matchup_all,
        context_present = context_present_all,
        return_tokens = FALSE
      )
      cand_all <- NULL
      cand_mask_all <- NULL
    }

    idx_left <- strenv$jnp$arange(n_batch)
    idx_right <- strenv$jnp$arange(n_batch, 2L * n_batch)
    phi_l <- strenv$jnp$take(phi_all, idx_left, axis = 0L)
    phi_r <- strenv$jnp$take(phi_all, idx_right, axis = 0L)

    if (isTRUE(use_cross_attn)) {
      cand_left_out <- strenv$jnp$take(cand_all, idx_left, axis = 0L)
      cand_right_out <- strenv$jnp$take(cand_all, idx_right, axis = 0L)
      cand_left_mask <- strenv$jnp$take(cand_mask_all, idx_left, axis = 0L)
      cand_right_mask <- strenv$jnp$take(cand_mask_all, idx_right, axis = 0L)
      ctx_left <- neural_cross_attend_cls_to_tokens(
        phi_l,
        cand_right_out,
        model_info = model_info,
        params = params,
        kv_token_mask = cand_right_mask
      )
      ctx_right <- neural_cross_attend_cls_to_tokens(
        phi_r,
        cand_left_out,
        model_info = model_info,
        params = params,
        kv_token_mask = cand_left_mask
      )
      phi_l <- neural_merge_cross_attn_representation(
        phi_l,
        ctx_left,
        params,
        model_info$model_dims
      )
      phi_r <- neural_merge_cross_attn_representation(
        phi_r,
        ctx_right,
        params,
        model_info$model_dims
      )
    }

    u_l <- neural_linear_head(
      phi_l,
      params$W_out,
      params$b_out,
      model_info = model_info,
      pairwise_obs = TRUE
    )
    u_r <- neural_linear_head(
      phi_r,
      params$W_out,
      params$b_out,
      model_info = model_info,
      pairwise_obs = TRUE
    )
    logits <- u_l - u_r
    if (isTRUE(neural_has_low_rank_interaction(params, model_info))) {
      logits <- logits + neural_low_rank_pair_delta_prepared(
        params = params,
        model_info = model_info,
        Xl = Xl,
        Xr = Xr,
        pl = pl,
        pr = pr,
        resp_p = resp_p,
        resp_c = resp_c,
        resp_c_present = resp_c_present,
        resp_c_order = resp_c_order,
        experiment_idx = experiment_idx,
        place_embedding = place_embedding,
        time_embedding = time_embedding,
        factor_order = factor_order,
        stage_idx = stage_idx,
        matchup_idx = matchup_idx,
        context_present = context_present,
        out_dim = ai(logits$shape[[2]]),
        dtype = logits$dtype
      )
    }
    if (isTRUE(use_cross_term)) {
      logits <- neural_apply_cross_term(
        logits,
        phi_l,
        phi_r,
        params$M_cross,
        params$W_cross_out,
        out_dim = ai(params$W_out$shape[[2]])
      )
    }
    logits <- logits + neural_additive_pair_delta_prepared(
      params = params,
      model_info = model_info,
      Xl = Xl,
      Xr = Xr,
      pl = pl,
      pr = pr,
      resp_p = resp_p,
      experiment_idx = experiment_idx,
      factor_order = factor_order,
      context_present = context_present,
      out_dim = ai(logits$shape[[2]]),
      dtype = logits$dtype
    )
  }

  logits <- neural_apply_pairwise_classification_logit_transform(
    logits,
    model_info = model_info,
    pairwise_obs = TRUE
  )
  logits <- neural_apply_pairwise_bernoulli_logit_scale(
    logits,
    model_info = model_info,
    scale = neural_pairwise_bernoulli_logit_scale_from_params(
      params = params,
      model_info = model_info
    ),
    pairwise_obs = TRUE
  )
  logits <- neural_apply_classification_logit_calibration(
    logits,
    model_info = model_info,
    params = params
  )
  if (isTRUE(return_logits)) {
    return(logits)
  }
  if (model_info$likelihood == "bernoulli") {
    return(strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L)))
  }
  if (model_info$likelihood == "categorical") {
    return(strenv$jax$nn$softmax(logits, axis = -1L))
  }
  if (model_info$likelihood == "ordinal") {
    return(logits)
  }
  list(
    mu = strenv$jnp$squeeze(logits, axis = 1L),
    sigma = if (!is.null(params$sigma)) {
      neural_floor_sigma(params$sigma, dtype = logits$dtype)
    } else {
      strenv$jnp$array(1., dtype = logits$dtype)
    }
  )
}

neural_predict_single_core_prepared <- function(params,
                                                model_info,
                                                Xb,
                                                party_idx,
                                                resp_p,
                                                resp_c = NULL,
                                                resp_c_present = NULL,
                                                resp_c_order = NULL,
                                                experiment_idx = NULL,
                                                place_embedding = NULL,
                                                time_embedding = NULL,
                                                factor_order = NULL,
                                                return_logits = FALSE) {
  Xb <- neural_batch_matrix_jnp(Xb, dtype = strenv$jnp$int32)
  party_idx <- neural_batch_vector_jnp(party_idx, dtype = strenv$jnp$int32)
  resp_p <- neural_batch_vector_jnp(resp_p, dtype = strenv$jnp$int32)
  if (!is.null(resp_c)) {
    resp_c <- neural_batch_matrix_jnp(resp_c, dtype = strenv$dtj)
  }
  if (!is.null(resp_c_present)) {
    resp_c_present <- neural_batch_matrix_jnp(resp_c_present, dtype = strenv$dtj)
  }
  if (!is.null(resp_c_order)) {
    resp_c_order <- neural_batch_matrix_jnp(resp_c_order, dtype = strenv$jnp$int32)
  }
  if (!is.null(experiment_idx)) {
    experiment_idx <- neural_batch_vector_jnp(experiment_idx, dtype = strenv$jnp$int32)
  }
  if (!is.null(place_embedding)) {
    place_embedding <- neural_batch_matrix_jnp(place_embedding, dtype = strenv$dtj)
  }
  if (!is.null(time_embedding)) {
    time_embedding <- neural_batch_matrix_jnp(time_embedding, dtype = strenv$dtj)
  }
  if (!is.null(factor_order)) {
    factor_order <- neural_batch_matrix_jnp(factor_order, dtype = strenv$jnp$int32)
  }
  choice_out <- neural_encode_candidate_core_prepared(
    params = params,
    model_info = model_info,
    X_idx = Xb,
    party_idx = party_idx,
    resp_party_idx = resp_p,
    resp_cov = resp_c,
    resp_cov_present = resp_c_present,
    resp_cov_order = resp_c_order,
    experiment_idx = experiment_idx,
    place_embedding = place_embedding,
    time_embedding = time_embedding,
    factor_order = factor_order,
    return_tokens = FALSE
  )
  logits <- neural_linear_head(choice_out, params$W_out, params$b_out)
  if (isTRUE(neural_has_low_rank_interaction(params, model_info))) {
    logits <- logits + neural_low_rank_single_utility_prepared(
      params = params,
      model_info = model_info,
      X_idx = Xb,
      party_idx = party_idx,
      resp_party_idx = resp_p,
      resp_cov = resp_c,
      resp_cov_present = resp_c_present,
      resp_cov_order = resp_c_order,
      experiment_idx = experiment_idx,
      place_embedding = place_embedding,
      time_embedding = time_embedding,
      factor_order = factor_order,
      out_dim = ai(logits$shape[[2]]),
      dtype = logits$dtype
    )
  }
  logits <- logits + neural_additive_single_logits_prepared(
    params = params,
    model_info = model_info,
    X_idx = Xb,
    party_idx = party_idx,
    resp_party_idx = resp_p,
    experiment_idx = experiment_idx,
    factor_order = factor_order,
    out_dim = ai(logits$shape[[2]]),
    dtype = logits$dtype
  )
  logits <- neural_apply_classification_logit_calibration(
    logits,
    model_info = model_info,
    params = params
  )

  if (isTRUE(return_logits)) {
    return(logits)
  }
  if (model_info$likelihood == "bernoulli") {
    return(strenv$jax$nn$sigmoid(strenv$jnp$squeeze(logits, axis = 1L)))
  }
  if (model_info$likelihood == "categorical") {
    return(strenv$jax$nn$softmax(logits, axis = -1L))
  }
  if (model_info$likelihood == "ordinal") {
    return(logits)
  }
  list(
    mu = strenv$jnp$squeeze(logits, axis = 1L),
    sigma = if (!is.null(params$sigma)) {
      neural_floor_sigma(params$sigma, dtype = logits$dtype)
    } else {
      strenv$jnp$array(1., dtype = logits$dtype)
    }
  )
}

neural_predict_prepared <- function(params,
                                    model_info,
                                    prep,
                                    return_logits = FALSE) {
  if (isTRUE(prep$pairwise)) {
    return(neural_predict_pair_core_prepared(
      params = params,
      model_info = model_info,
      Xl = prep$X_left,
      Xr = prep$X_right,
      pl = prep$party_left,
      pr = prep$party_right,
      resp_p = prep$resp_party,
      resp_c = prep$resp_cov,
      resp_c_present = prep$resp_cov_present %||% NULL,
      resp_c_order = prep$resp_cov_order %||% NULL,
      experiment_idx = prep$experiment_idx %||% NULL,
      place_embedding = prep$place_embedding %||% NULL,
      time_embedding = prep$time_embedding %||% NULL,
      factor_order = prep$factor_order %||% NULL,
      return_logits = return_logits
    ))
  }
  neural_predict_single_core_prepared(
    params = params,
    model_info = model_info,
    Xb = prep$X_single,
    party_idx = prep$party_single,
    resp_p = prep$resp_party,
    resp_c = prep$resp_cov,
    resp_c_present = prep$resp_cov_present %||% NULL,
    resp_c_order = prep$resp_cov_order %||% NULL,
    experiment_idx = prep$experiment_idx %||% NULL,
    place_embedding = prep$place_embedding %||% NULL,
    time_embedding = prep$time_embedding %||% NULL,
    factor_order = prep$factor_order %||% NULL,
    return_logits = return_logits
  )
}

neural_predict_from_theta_prepared <- function(theta_vec,
                                               model_info,
                                               prep,
                                               return_logits = FALSE) {
  params <- neural_params_from_theta(theta_vec, model_info)
  neural_predict_prepared(
    params = params,
    model_info = model_info,
    prep = prep,
    return_logits = return_logits
  )
}

neural_get_predict_jit <- function(model_info,
                                   pairwise,
                                   return_logits = FALSE) {
  cache_key <- paste(
    "predict",
    neural_model_jit_cache_key(model_info),
    if (isTRUE(pairwise)) "pairwise" else "single",
    if (isTRUE(return_logits)) "logits" else "response",
    sep = "::"
  )
  if (!exists(cache_key, envir = neural_prediction_jit_cache, inherits = FALSE)) {
    compiled <- if (isTRUE(pairwise)) {
      strenv$jax$jit(function(params, Xl, Xr, pl, pr, resp_p, resp_c, resp_c_present, resp_c_order, experiment_idx, place_embedding, time_embedding, factor_order) {
        neural_predict_pair_core_prepared(
          params = params,
          model_info = model_info,
          Xl = Xl,
          Xr = Xr,
          pl = pl,
          pr = pr,
          resp_p = resp_p,
          resp_c = resp_c,
          resp_c_present = resp_c_present,
          resp_c_order = resp_c_order,
          experiment_idx = experiment_idx,
          place_embedding = place_embedding,
          time_embedding = time_embedding,
          factor_order = factor_order,
          return_logits = return_logits
        )
      })
    } else {
      strenv$jax$jit(function(params, Xb, party_idx, resp_p, resp_c, resp_c_present, resp_c_order, experiment_idx, place_embedding, time_embedding, factor_order) {
        neural_predict_single_core_prepared(
          params = params,
          model_info = model_info,
          Xb = Xb,
          party_idx = party_idx,
          resp_p = resp_p,
          resp_c = resp_c,
          resp_c_present = resp_c_present,
          resp_c_order = resp_c_order,
          experiment_idx = experiment_idx,
          place_embedding = place_embedding,
          time_embedding = time_embedding,
          factor_order = factor_order,
          return_logits = return_logits
        )
      })
    }
    assign(cache_key, compiled, envir = neural_prediction_jit_cache)
  }
  get(cache_key, envir = neural_prediction_jit_cache, inherits = FALSE)
}

neural_get_predict_from_theta_jit <- function(model_info,
                                              pairwise,
                                              return_logits = FALSE) {
  cache_key <- paste(
    "predict_from_theta",
    neural_model_jit_cache_key(model_info),
    if (isTRUE(pairwise)) "pairwise" else "single",
    if (isTRUE(return_logits)) "logits" else "response",
    sep = "::"
  )
  if (!exists(cache_key, envir = neural_prediction_jit_cache, inherits = FALSE)) {
    compiled <- if (isTRUE(pairwise)) {
      predict_one <- function(theta_vec, Xl, Xr, pl, pr, resp_p, resp_c, resp_c_present, resp_c_order, experiment_idx, place_embedding, time_embedding, factor_order) {
        params <- neural_params_from_theta(theta_vec, model_info)
        neural_predict_pair_core_prepared(
          params = params,
          model_info = model_info,
          Xl = Xl,
          Xr = Xr,
          pl = pl,
          pr = pr,
          resp_p = resp_p,
          resp_c = resp_c,
          resp_c_present = resp_c_present,
          resp_c_order = resp_c_order,
          experiment_idx = experiment_idx,
          place_embedding = place_embedding,
          time_embedding = time_embedding,
          factor_order = factor_order,
          return_logits = return_logits
        )
      }
      vmapped <- strenv$jax$vmap(
        predict_one,
        in_axes = list(0L, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
      )
      strenv$jax$jit(vmapped)
    } else {
      predict_one <- function(theta_vec, Xb, party_idx, resp_p, resp_c, resp_c_present, resp_c_order, experiment_idx, place_embedding, time_embedding, factor_order) {
        params <- neural_params_from_theta(theta_vec, model_info)
        neural_predict_single_core_prepared(
          params = params,
          model_info = model_info,
          Xb = Xb,
          party_idx = party_idx,
          resp_p = resp_p,
          resp_c = resp_c,
          resp_c_present = resp_c_present,
          resp_c_order = resp_c_order,
          experiment_idx = experiment_idx,
          place_embedding = place_embedding,
          time_embedding = time_embedding,
          factor_order = factor_order,
          return_logits = return_logits
        )
      }
      vmapped <- strenv$jax$vmap(
        predict_one,
        in_axes = list(0L, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
      )
      strenv$jax$jit(vmapped)
    }
    assign(cache_key, compiled, envir = neural_prediction_jit_cache)
  }
  get(cache_key, envir = neural_prediction_jit_cache, inherits = FALSE)
}

neural_predict_prepared_jitted <- function(params,
                                           model_info,
                                           prep,
                                           return_logits = FALSE) {
  predict_fn <- neural_get_predict_jit(
    model_info = model_info,
    pairwise = isTRUE(prep$pairwise),
    return_logits = return_logits
  )
  if (isTRUE(prep$pairwise)) {
    return(predict_fn(
      params,
      prep$X_left,
      prep$X_right,
      prep$party_left,
      prep$party_right,
      prep$resp_party,
      prep$resp_cov,
      prep$resp_cov_present %||% NULL,
      prep$resp_cov_order %||% NULL,
      prep$experiment_idx %||% NULL,
      prep$place_embedding %||% NULL,
      prep$time_embedding %||% NULL,
      prep$factor_order %||% NULL
    ))
  }
  predict_fn(
    params,
    prep$X_single,
    prep$party_single,
    prep$resp_party,
    prep$resp_cov,
    prep$resp_cov_present %||% NULL,
    prep$resp_cov_order %||% NULL,
    prep$experiment_idx %||% NULL,
    prep$place_embedding %||% NULL,
    prep$time_embedding %||% NULL,
    prep$factor_order %||% NULL
  )
}

neural_predict_from_theta_prepared_jitted <- function(theta_batch,
                                                      model_info,
                                                      prep,
                                                      return_logits = FALSE) {
  predict_fn <- neural_get_predict_from_theta_jit(
    model_info = model_info,
    pairwise = isTRUE(prep$pairwise),
    return_logits = return_logits
  )
  if (isTRUE(prep$pairwise)) {
    return(predict_fn(
      theta_batch,
      prep$X_left,
      prep$X_right,
      prep$party_left,
      prep$party_right,
      prep$resp_party,
      prep$resp_cov,
      prep$resp_cov_present %||% NULL,
      prep$resp_cov_order %||% NULL,
      prep$experiment_idx %||% NULL,
      prep$place_embedding %||% NULL,
      prep$time_embedding %||% NULL,
      prep$factor_order %||% NULL
    ))
  }
  predict_fn(
    theta_batch,
    prep$X_single,
    prep$party_single,
    prep$resp_party,
    prep$resp_cov,
    prep$resp_cov_present %||% NULL,
    prep$resp_cov_order %||% NULL,
    prep$experiment_idx %||% NULL,
    prep$place_embedding %||% NULL,
    prep$time_embedding %||% NULL,
    prep$factor_order %||% NULL
  )
}

neural_encode_candidate_soft <- function(pi_vec, party_idx, model_info,
                                         resp_party_idx = NULL,
                                         stage_idx = NULL,
                                         matchup_idx = NULL,
                                         resp_cov_vec = NULL,
                                         params = NULL, use_role = FALSE,
                                         context_present = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  choice_tok <- neural_build_choice_token(model_info, params)
  choice_mask <- strenv$jnp$ones(list(1L, 1L), dtype = strenv$dtj)
  resp_info <- neural_build_context_tokens(model_info,
                                           resp_party_idx = resp_party_idx,
                                           stage_idx = stage_idx,
                                           matchup_idx = matchup_idx,
                                           resp_cov_vec = resp_cov_vec,
                                           params = params,
                                           return_mask = TRUE,
                                           context_present = context_present)
  resp_tokens <- resp_info$tokens %||% NULL
  resp_mask <- resp_info$mask %||% NULL
  cand_info <- neural_build_candidate_tokens_soft(pi_vec, party_idx, 0L, model_info, params,
                                                  use_role = use_role,
                                                  resp_party_idx = resp_party_idx,
                                                  return_mask = TRUE,
                                                  context_present = context_present)
  cand_tokens <- cand_info$tokens
  cand_mask <- cand_info$mask
  seq_info <- neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    ctx_tokens = resp_tokens,
    ctx_mask = resp_mask,
    cand_tokens = cand_tokens,
    cand_mask = cand_mask,
    model_info = model_info,
    preserve_candidate_tail = FALSE
  )
  tokens <- seq_info$tokens
  token_mask <- seq_info$mask
  transformer_out <- neural_run_transformer(
    tokens,
    model_info,
    params,
    token_mask = token_mask,
    return_details = TRUE
  )
  neural_extract_choice_representation(
    transformer_out,
    token_mask = token_mask,
    include_tail_summary = neural_fused_classification_summary_enabled(model_info)
  )
}

neural_encode_pair_soft_batched <- function(pi_left, pi_right,
                                            party_left_idx, party_right_idx,
                                            model_info,
                                            resp_party_idx = NULL,
                                            stage_idx = NULL,
                                            matchup_idx = NULL,
                                            resp_cov_vec = NULL,
                                            params = NULL,
                                            use_role = FALSE,
                                            return_tokens = FALSE,
                                            context_present = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  if (is.null(context_present)) {
    context_present <- neural_pair_context_present(
      party_left_idx,
      party_right_idx,
      resp_party_idx,
      model_info
    )
  }
  choice_tok <- neural_build_choice_token(model_info, params)
  choice_mask <- strenv$jnp$ones(list(1L, 1L), dtype = strenv$dtj)
  resp_info <- neural_build_context_tokens(model_info,
                                           resp_party_idx = resp_party_idx,
                                           stage_idx = stage_idx,
                                           matchup_idx = matchup_idx,
                                           resp_cov_vec = resp_cov_vec,
                                           params = params,
                                           return_mask = TRUE,
                                           context_present = context_present)
  resp_tokens <- resp_info$tokens %||% NULL
  resp_mask <- resp_info$mask %||% NULL
  left_info <- neural_build_candidate_tokens_soft(pi_left, party_left_idx, 0L, model_info, params,
                                                  use_role = use_role,
                                                  resp_party_idx = resp_party_idx,
                                                  return_mask = TRUE,
                                                  context_present = context_present)
  right_info <- neural_build_candidate_tokens_soft(pi_right, party_right_idx, 0L, model_info, params,
                                                   use_role = use_role,
                                                   resp_party_idx = resp_party_idx,
                                                   return_mask = TRUE,
                                                   context_present = context_present)
  left_tokens <- left_info$tokens
  right_tokens <- right_info$tokens
  left_mask <- left_info$mask
  right_mask <- right_info$mask
  left_seq <- neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    ctx_tokens = resp_tokens,
    ctx_mask = resp_mask,
    cand_tokens = left_tokens,
    cand_mask = left_mask,
    model_info = model_info,
    preserve_candidate_tail = isTRUE(return_tokens)
  )
  right_seq <- neural_pack_candidate_sequence(
    choice_tok = choice_tok,
    choice_mask = choice_mask,
    ctx_tokens = resp_tokens,
    ctx_mask = resp_mask,
    cand_tokens = right_tokens,
    cand_mask = right_mask,
    model_info = model_info,
    preserve_candidate_tail = isTRUE(return_tokens)
  )
  tokens_left <- left_seq$tokens
  tokens_right <- right_seq$tokens
  token_mask_left <- left_seq$mask
  token_mask_right <- right_seq$mask
  tokens <- strenv$jnp$concatenate(list(tokens_left, tokens_right), axis = 0L)
  token_mask <- strenv$jnp$concatenate(list(token_mask_left, token_mask_right), axis = 0L)
  transformer_out <- neural_run_transformer(
    tokens,
    model_info,
    params,
    token_mask = token_mask,
    return_details = TRUE
  )
  phi_all <- neural_extract_choice_representation(
    transformer_out,
    token_mask = token_mask,
    include_tail_summary = neural_fused_classification_summary_enabled(model_info)
  )
  idx_left <- strenv$jnp$arange(1L)
  idx_right <- strenv$jnp$arange(1L, 2L)
  out <- list(
    phi_left = strenv$jnp$take(phi_all, idx_left, axis = 0L),
    phi_right = strenv$jnp$take(phi_all, idx_right, axis = 0L)
  )
  if (isTRUE(return_tokens)) {
    cand_tokens <- neural_extract_candidate_tokens(
      transformer_out,
      model_info,
      n_candidate_tokens = neural_candidate_token_count_from_mask(left_seq$cand_mask)
    )
    out$cand_left_out <- strenv$jnp$take(cand_tokens, idx_left, axis = 0L)
    out$cand_right_out <- strenv$jnp$take(cand_tokens, idx_right, axis = 0L)
    out$cand_left_mask <- left_seq$cand_mask
    out$cand_right_mask <- right_seq$cand_mask
  }
  out
}

neural_candidate_utility_soft <- function(pi_vec, party_idx,
                                          resp_party_idx, stage_idx,
                                          model_info,
                                          resp_cov_vec = NULL,
                                          params = NULL,
                                          matchup_idx = NULL,
                                          context_present = NULL){
  if (is.null(params)) {
    params <- model_info$params
  }
  phi <- neural_encode_candidate_soft(pi_vec, party_idx, model_info,
                                      resp_party_idx = resp_party_idx,
                                      stage_idx = stage_idx,
                                      matchup_idx = matchup_idx,
                                      resp_cov_vec = resp_cov_vec,
                                      params = params,
                                      context_present = context_present)
  # For a pairwise-trained model, apply the same head/low-rank RMS
  # normalization the pair logit path uses (pairwise_obs = TRUE at 9988-10001):
  # under low_rank_logit_normalization = "rms" the raw W_out column scales are
  # unidentified (only the normalized head enters the training loss), so
  # un-normalized utilities here would feed the Bradley-Terry nomination
  # softmax at an arbitrary temperature. All ops are jnp -- the utility stays
  # differentiable in pi_vec.
  utility_pairwise_obs <- isTRUE(model_info$pairwise_mode)
  utility <- neural_linear_head(
    phi,
    params$W_out,
    params$b_out,
    model_info = model_info,
    pairwise_obs = utility_pairwise_obs
  )
  if (isTRUE(neural_has_low_rank_interaction(params, model_info))) {
    resp_readout <- neural_encode_respondent_tower_prepared(
      params = params,
      model_info = model_info,
      resp_party_idx = resp_party_idx,
      resp_cov = resp_cov_vec,
      stage_idx = stage_idx,
      matchup_idx = matchup_idx,
      context_present = context_present
    )
    cand_readout <- neural_encode_candidate_profile_tower_soft(
      params = params,
      model_info = model_info,
      pi_vec = pi_vec,
      party_idx = party_idx,
      resp_party_idx = resp_party_idx,
      context_present = context_present
    )
    utility <- utility + neural_low_rank_interaction_logits(
      respondent_final = resp_readout$final,
      candidate_final = cand_readout$final,
      params = params,
      out_dim = ai(utility$shape[[2]]),
      dtype = utility$dtype,
      model_info = model_info,
      pairwise_obs = utility_pairwise_obs
    )
  }
  # Additive main-effects head + calibration temperature (train/serve parity with
  # neural_predict_single_core_prepared at 9094/9105). neural_predict_single_soft
  # and the utility-based primary nomination both consume this utility, so both
  # inherit the fix. Each step self-gates and is a no-op when disabled.
  if (isTRUE(neural_additive_utility_enabled(model_info, params))) {
    add_info <- neural_build_candidate_tokens_soft(
      pi_vec, party_idx, 0L, model_info, params,
      resp_party_idx = resp_party_idx, return_mask = TRUE,
      context_present = context_present
    )
    utility <- utility +
      neural_additive_logits_from_candidate_tokens(
        tokens = add_info$tokens, token_mask = add_info$mask,
        params = params, model_info = model_info,
        out_dim = ai(utility$shape[[2]]), dtype = utility$dtype
      )
  }
  utility <- neural_apply_classification_logit_calibration(
    utility,
    model_info = model_info,
    params = params
  )
  utility
}

neural_predict_pair_soft <- function(pi_left, pi_right,
                                     party_left_idx, party_right_idx,
                                     resp_party_idx, model_info,
                                     resp_cov_vec = NULL,
                                     params = NULL,
                                     return_logits = FALSE){
  if (is.null(params)) {
    params <- model_info$params
  }
  mode <- neural_cross_encoder_mode(model_info)
  use_cross_encoder <- identical(mode, "full")
  use_cross_term <- identical(mode, "term")
  use_cross_attn <- identical(mode, "attn")
  stage_idx <- neural_stage_index(party_left_idx, party_right_idx, model_info)
  matchup_idx <- NULL
  if (!is.null(params$E_matchup)) {
    matchup_idx <- neural_matchup_index(party_left_idx, party_right_idx, model_info)
  }
  context_present <- neural_pair_context_present(
    party_left_idx,
    party_right_idx,
    resp_party_idx,
    model_info
  )
  if (isTRUE(use_cross_encoder)) {
    choice_tok <- neural_build_choice_token(model_info, params)
    choice_mask <- strenv$jnp$ones(list(1L, 1L), dtype = strenv$dtj)
    # Cross-encoder CLS logits for a given candidate ordering (a = left/segment0,
    # b = right/segment1). Factored into a closure so strict antisymmetry can
    # average the forward and reversed orderings, matching the hard core at
    # 8746-8772: the raw cross-encoder logit is not antisymmetric in (left, right),
    # so serving it un-averaged yields a self-inconsistent Q(l,r)+Q(r,l) != 1
    # surface that distorts both the general-election Q and the kappa push-forward.
    cross_cls_logits <- function(pi_a, party_a, pi_b, party_b,
                                 stage_idx_use, matchup_idx_use, context_present_use) {
      ctx_info <- neural_build_context_tokens(model_info,
                                              resp_party_idx = resp_party_idx,
                                              stage_idx = stage_idx_use,
                                              matchup_idx = matchup_idx_use,
                                              resp_cov_vec = resp_cov_vec,
                                              params = params,
                                              return_mask = TRUE,
                                              context_present = context_present_use)
      a_info <- neural_build_candidate_tokens_soft(pi_a, party_a, 0L, model_info, params,
                                                   resp_party_idx = resp_party_idx,
                                                   return_mask = TRUE,
                                                   context_present = context_present_use)
      b_info <- neural_build_candidate_tokens_soft(pi_b, party_b, 1L, model_info, params,
                                                   resp_party_idx = resp_party_idx,
                                                   return_mask = TRUE,
                                                   context_present = context_present_use)
      a_tokens <- neural_add_segment_embedding(a_info$tokens, 0L, model_info = model_info, params = params)
      b_tokens <- neural_add_segment_embedding(b_info$tokens, 1L, model_info = model_info, params = params)
      sep_tok <- neural_build_sep_token(model_info, params = params)
      sep_mask <- strenv$jnp$ones(list(1L, 1L), dtype = strenv$dtj)
      seq_info <- neural_pack_full_cross_sequence(
        choice_tok = choice_tok,
        choice_mask = choice_mask,
        sep_tok = sep_tok,
        sep_mask = sep_mask,
        left_tokens = a_tokens,
        left_mask = a_info$mask,
        right_tokens = b_tokens,
        right_mask = b_info$mask,
        model_info = model_info,
        ctx_tokens = ctx_info$tokens %||% NULL,
        ctx_mask = ctx_info$mask %||% NULL
      )
      transformer_out <- neural_run_transformer(
        seq_info$tokens,
        model_info,
        params,
        token_mask = seq_info$mask,
        return_details = TRUE
      )
      cls_out <- neural_extract_choice_representation(transformer_out)
      neural_linear_head(
        cls_out,
        params$W_out,
        params$b_out,
        model_info = model_info,
        pairwise_obs = TRUE
      )
    }
    logits <- cross_cls_logits(pi_left, party_left_idx, pi_right, party_right_idx,
                               stage_idx, matchup_idx, context_present)
    if (isTRUE(neural_pairwise_antisymmetry_strict(model_info))) {
      stage_idx_rev <- neural_stage_index(party_right_idx, party_left_idx, model_info)
      matchup_idx_rev <- if (!is.null(params$E_matchup)) {
        neural_matchup_index(party_right_idx, party_left_idx, model_info)
      } else {
        NULL
      }
      context_present_rev <- neural_pair_context_present(
        party_right_idx, party_left_idx, resp_party_idx, model_info
      )
      logits_rev <- cross_cls_logits(pi_right, party_right_idx, pi_left, party_left_idx,
                                     stage_idx_rev, matchup_idx_rev, context_present_rev)
      logits <- 0.5 * (logits - logits_rev)
    }
    if (isTRUE(neural_has_low_rank_interaction(params, model_info))) {
      resp_readout <- neural_encode_respondent_tower_prepared(
        params = params,
        model_info = model_info,
        resp_party_idx = resp_party_idx,
        resp_cov = resp_cov_vec,
        stage_idx = stage_idx,
        matchup_idx = matchup_idx,
        context_present = context_present
      )
      left_readout <- neural_encode_candidate_profile_tower_soft(
        params = params,
        model_info = model_info,
        pi_vec = pi_left,
        party_idx = party_left_idx,
        resp_party_idx = resp_party_idx,
        context_present = context_present
      )
      right_readout <- neural_encode_candidate_profile_tower_soft(
        params = params,
        model_info = model_info,
        pi_vec = pi_right,
        party_idx = party_right_idx,
        resp_party_idx = resp_party_idx,
        context_present = context_present
      )
      logits <- logits +
        neural_low_rank_interaction_logits(
          respondent_final = resp_readout$final,
          candidate_final = left_readout$final,
          params = params,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype,
          model_info = model_info,
          pairwise_obs = TRUE
        ) -
        neural_low_rank_interaction_logits(
          respondent_final = resp_readout$final,
          candidate_final = right_readout$final,
          params = params,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype,
          model_info = model_info,
          pairwise_obs = TRUE
        )
    }
    # Additive main-effects head (parity with the hard cross-encoder core at 8797),
    # added after the antisymmetry averaging and the low-rank delta.
    if (isTRUE(neural_additive_utility_enabled(model_info, params))) {
      add_out_dim <- ai(logits$shape[[2]])
      left_add_info <- neural_build_candidate_tokens_soft(
        pi_left, party_left_idx, 0L, model_info, params,
        resp_party_idx = resp_party_idx, return_mask = TRUE,
        context_present = context_present
      )
      right_add_info <- neural_build_candidate_tokens_soft(
        pi_right, party_right_idx, 1L, model_info, params,
        resp_party_idx = resp_party_idx, return_mask = TRUE,
        context_present = context_present
      )
      logits <- logits +
        neural_additive_logits_from_candidate_tokens(
          tokens = left_add_info$tokens, token_mask = left_add_info$mask,
          params = params, model_info = model_info,
          out_dim = add_out_dim, dtype = logits$dtype
        ) -
        neural_additive_logits_from_candidate_tokens(
          tokens = right_add_info$tokens, token_mask = right_add_info$mask,
          params = params, model_info = model_info,
          out_dim = add_out_dim, dtype = logits$dtype
        )
    }
  } else {
    phi_pair <- neural_encode_pair_soft_batched(pi_left, pi_right,
                                                party_left_idx, party_right_idx,
                                                model_info,
                                                resp_party_idx = resp_party_idx,
                                                stage_idx = stage_idx,
                                                matchup_idx = matchup_idx,
                                                resp_cov_vec = resp_cov_vec,
                                                params = params,
                                                context_present = context_present,
                                                return_tokens = isTRUE(use_cross_attn))
    phi_left <- phi_pair$phi_left
    phi_right <- phi_pair$phi_right
    if (isTRUE(use_cross_attn)) {
      ctx_left <- neural_cross_attend_cls_to_tokens(phi_left, phi_pair$cand_right_out,
                                                    model_info = model_info,
                                                    params = params,
                                                    kv_token_mask = phi_pair$cand_right_mask)
      ctx_right <- neural_cross_attend_cls_to_tokens(phi_right, phi_pair$cand_left_out,
                                                     model_info = model_info,
                                                     params = params,
                                                     kv_token_mask = phi_pair$cand_left_mask)
      phi_left <- neural_merge_cross_attn_representation(
        phi_left, ctx_left, params, model_info$model_dims
      )
      phi_right <- neural_merge_cross_attn_representation(
        phi_right, ctx_right, params, model_info$model_dims
      )
    }
    u_left <- neural_linear_head(
      phi_left,
      params$W_out,
      params$b_out,
      model_info = model_info,
      pairwise_obs = TRUE
    )
    u_right <- neural_linear_head(
      phi_right,
      params$W_out,
      params$b_out,
      model_info = model_info,
      pairwise_obs = TRUE
    )
    logits <- u_left - u_right
    if (isTRUE(neural_has_low_rank_interaction(params, model_info))) {
      resp_readout <- neural_encode_respondent_tower_prepared(
        params = params,
        model_info = model_info,
        resp_party_idx = resp_party_idx,
        resp_cov = resp_cov_vec,
        stage_idx = stage_idx,
        matchup_idx = matchup_idx,
        context_present = context_present
      )
      left_readout <- neural_encode_candidate_profile_tower_soft(
        params = params,
        model_info = model_info,
        pi_vec = pi_left,
        party_idx = party_left_idx,
        resp_party_idx = resp_party_idx,
        context_present = context_present
      )
      right_readout <- neural_encode_candidate_profile_tower_soft(
        params = params,
        model_info = model_info,
        pi_vec = pi_right,
        party_idx = party_right_idx,
        resp_party_idx = resp_party_idx,
        context_present = context_present
      )
      logits <- logits +
        neural_low_rank_interaction_logits(
          respondent_final = resp_readout$final,
          candidate_final = left_readout$final,
          params = params,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype,
          model_info = model_info,
          pairwise_obs = TRUE
        ) -
        neural_low_rank_interaction_logits(
          respondent_final = resp_readout$final,
          candidate_final = right_readout$final,
          params = params,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype,
          model_info = model_info,
          pairwise_obs = TRUE
        )
    }
    if (isTRUE(use_cross_term)) {
      logits <- neural_apply_cross_term(logits, phi_left, phi_right,
                                        params$M_cross, params$W_cross_out,
                                        out_dim = ai(params$W_out$shape[[2]]))
    }
    # Additive main-effects head (train/serve parity with the hard core at 8966):
    # applied only in the two-tower branch, as the last logit contribution before
    # the shared transform/scale/calibration. Self-gates via
    # neural_additive_utility_enabled, so it is a no-op when the additive head is off.
    if (isTRUE(neural_additive_utility_enabled(model_info, params))) {
      add_out_dim <- ai(logits$shape[[2]])
      left_add_info <- neural_build_candidate_tokens_soft(
        pi_left, party_left_idx, 0L, model_info, params,
        resp_party_idx = resp_party_idx, return_mask = TRUE,
        context_present = context_present
      )
      right_add_info <- neural_build_candidate_tokens_soft(
        pi_right, party_right_idx, 1L, model_info, params,
        resp_party_idx = resp_party_idx, return_mask = TRUE,
        context_present = context_present
      )
      logits <- logits +
        neural_additive_logits_from_candidate_tokens(
          tokens = left_add_info$tokens, token_mask = left_add_info$mask,
          params = params, model_info = model_info,
          out_dim = add_out_dim, dtype = logits$dtype
        ) -
        neural_additive_logits_from_candidate_tokens(
          tokens = right_add_info$tokens, token_mask = right_add_info$mask,
          params = params, model_info = model_info,
          out_dim = add_out_dim, dtype = logits$dtype
        )
    }
  }
  logits <- neural_apply_pairwise_classification_logit_transform(
    logits,
    model_info = model_info,
    pairwise_obs = TRUE,
    straight_through = FALSE
  )
  logits <- neural_apply_pairwise_bernoulli_logit_scale(
    logits,
    model_info = model_info,
    scale = neural_pairwise_bernoulli_logit_scale_from_params(
      params = params,
      model_info = model_info
    ),
    pairwise_obs = TRUE
  )
  # Learned calibration temperature (train/serve parity with the hard core at 8996):
  # apply before the return_logits exit so the kappa push-forward sees the same
  # calibrated logits training did, prior to the separate push-forward temperature.
  logits <- neural_apply_classification_logit_calibration(
    logits,
    model_info = model_info,
    params = params
  )
  if (return_logits) {
    return(logits)
  }
  neural_logits_to_q(logits, model_info$likelihood)
}

neural_predict_single_soft <- function(pi_vec,
                                       party_idx,
                                       resp_party_idx,
                                       model_info,
                                       resp_cov_vec = NULL,
                                       params = NULL){
  logits <- neural_candidate_utility_soft(pi_vec, party_idx,
                                          resp_party_idx, stage_idx = NULL,
                                          model_info = model_info,
                                          resp_cov_vec = resp_cov_vec,
                                          params = params)
  neural_logits_to_q(logits, model_info$likelihood)
}

neural_build_average_case_linear_q_calibration <- function(W_idx,
                                                           Y,
                                                           factor_levels,
                                                           holdout_indicator = 1L) {
  if (is.null(W_idx) || is.null(Y)) {
    return(NULL)
  }
  W_idx <- as.matrix(W_idx)
  Y <- as.numeric(Y)
  factor_levels <- as.integer(factor_levels %||% integer(0))
  if (nrow(W_idx) != length(Y) ||
      ncol(W_idx) != length(factor_levels) ||
      length(Y) < 2L ||
      any(!is.finite(Y))) {
    return(NULL)
  }
  explicit_counts <- pmax(0L, factor_levels - as.integer(holdout_indicator))
  if (sum(explicit_counts) < 1L) {
    return(NULL)
  }

  main_cols <- vector("list", sum(explicit_counts))
  main_factor <- integer(sum(explicit_counts))
  main_level <- integer(sum(explicit_counts))
  param_idx <- 0L
  out_idx <- 0L
  for (d_ in seq_along(factor_levels)) {
    n_exp <- explicit_counts[[d_]]
    if (n_exp < 1L) {
      next
    }
    for (l_ in seq_len(n_exp)) {
      out_idx <- out_idx + 1L
      main_cols[[out_idx]] <- as.numeric(W_idx[, d_] == l_)
      main_factor[[out_idx]] <- d_
      main_level[[out_idx]] <- l_
      param_idx <- param_idx + 1L
    }
  }
  main_mat <- do.call(cbind, main_cols)
  storage.mode(main_mat) <- "double"

  inter_left <- integer(0)
  inter_right <- integer(0)
  inter_cols <- list()
  if (ncol(main_mat) > 1L) {
    pairs <- utils::combn(seq_len(ncol(main_mat)), 2L)
    keep <- main_factor[pairs[1L, ]] != main_factor[pairs[2L, ]]
    pairs <- pairs[, keep, drop = FALSE]
    if (ncol(pairs) > 0L) {
      inter_left <- as.integer(pairs[1L, ] - 1L)
      inter_right <- as.integer(pairs[2L, ] - 1L)
      inter_cols <- lapply(seq_len(ncol(pairs)), function(i_) {
        main_mat[, pairs[1L, i_]] * main_mat[, pairs[2L, i_]]
      })
    }
  }
  inter_mat <- if (length(inter_cols) > 0L) {
    do.call(cbind, inter_cols)
  } else {
    matrix(numeric(0), nrow = nrow(main_mat), ncol = 0L)
  }
  storage.mode(inter_mat) <- "double"

  design <- cbind("(Intercept)" = 1, main_mat, inter_mat)
  fit <- tryCatch(stats::lm.fit(x = design, y = Y), error = function(e) NULL)
  if (is.null(fit) || is.null(fit$coefficients)) {
    return(NULL)
  }
  coef <- as.numeric(fit$coefficients)
  coef[!is.finite(coef)] <- 0
  n_main <- ncol(main_mat)
  n_inter <- ncol(inter_mat)
  list(
    intercept = coef[[1L]],
    main_coef = coef[seq_len(n_main) + 1L],
    inter_coef = if (n_inter > 0L) coef[seq_len(n_inter) + 1L + n_main] else numeric(0),
    inter_left = inter_left,
    inter_right = inter_right
  )
}

neural_average_case_linear_q_from_policy <- function(pi_vec, model_info) {
  cal <- model_info$average_case_linear_q %||% NULL
  if (is.null(cal) || is.null(cal$main_coef) || is.null(cal$inter_coef)) {
    return(NULL)
  }
  pi_flat <- strenv$jnp$reshape(pi_vec, list(-1L))
  main_coef <- cal$main_coef
  if (ai(main_coef$shape[[1L]]) != ai(pi_flat$shape[[1L]])) {
    return(NULL)
  }
  q_val <- cal$intercept +
    strenv$jnp$sum(pi_flat * main_coef)
  if (!is.null(cal$inter_left) &&
      !is.null(cal$inter_right) &&
      ai(cal$inter_coef$shape[[1L]]) > 0L) {
    pi_left <- strenv$jnp$take(pi_flat, cal$inter_left, axis = 0L)
    pi_right <- strenv$jnp$take(pi_flat, cal$inter_right, axis = 0L)
    q_val <- q_val + strenv$jnp$sum(pi_left * pi_right * cal$inter_coef)
  }
  q_val
}

neural_policy_log_prob_observed_profiles <- function(pi_vec,
                                                     X_idx,
                                                     model_info,
                                                     baseline_level_probs = NULL) {
  X_idx <- neural_batch_matrix_jnp(X_idx, dtype = strenv$jnp$int32)
  n_obs <- ai(X_idx$shape[[1]])
  n_factors <- ai(model_info$n_factors)
  eps <- strenv$jnp$array(1e-8, dtype = strenv$dtj)
  log_prob <- strenv$jnp$zeros(list(n_obs), dtype = strenv$dtj)

  for (d_ in seq_len(n_factors)) {
    idx_d <- strenv$jnp$take(X_idx, ai(d_ - 1L), axis = 1L)
    if (is.null(baseline_level_probs)) {
      idx_param <- strenv$jnp$atleast_1d(model_info$factor_index_list[[d_]])
      p_sub <- strenv$jnp$take(strenv$jnp$reshape(pi_vec, list(-1L)), idx_param, axis = 0L)
      p_full <- apply_implicit_parameterization_jnp(
        p_sub,
        implicit = isTRUE(model_info$implicit),
        axis = 0L,
        clip = TRUE
      )
    } else {
      p_full <- baseline_level_probs[[d_]]
    }
    p_full <- strenv$jnp$reshape(p_full, list(-1L))
    n_levels <- ai(p_full$shape[[1L]])
    idx_safe <- strenv$jnp$minimum(
      strenv$jnp$maximum(idx_d, ai(0L)),
      ai(max(0L, n_levels - 1L))
    )
    p_d <- strenv$jnp$take(p_full, idx_safe, axis = 0L)
    valid <- (idx_d >= ai(0L)) & (idx_d < ai(n_levels))
    p_d <- strenv$jnp$where(valid, p_d, eps)
    log_prob <- log_prob + strenv$jnp$log(strenv$jnp$clip(p_d, eps, 1.0))
  }

  log_prob
}

neural_average_case_report_adjust_q <- function(q_vec,
                                                pi_star_ast,
                                                EST_COEFFICIENTS_tf_ast) {
  model_ast <- neural_resolve_model_info("neural_model_info_ast_jnp")
  if (is.null(model_ast) ||
      !identical(model_ast$likelihood, "normal") ||
      isTRUE(model_ast$pairwise_mode) ||
      isTRUE(model_ast$universal_foundation_training)) {
    return(q_vec)
  }
  q_linear <- neural_average_case_linear_q_from_policy(pi_star_ast, model_ast)
  if (!is.null(q_linear)) {
    return((q_vec * strenv$jnp$array(0., dtype = q_vec$dtype)) + q_linear)
  }

  dr <- model_ast$average_case_dr_training %||% NULL
  if (is.null(dr) || is.null(dr$X_single) || is.null(dr$Y_single) ||
      is.null(dr$baseline_level_probs)) {
    return(q_vec)
  }

  params_ast <- neural_params_from_theta(EST_COEFFICIENTS_tf_ast, model_ast)
  pred_obs <- neural_predict_single_core_prepared(
    params = params_ast,
    model_info = model_ast,
    Xb = dr$X_single,
    party_idx = dr$party_single,
    resp_p = dr$resp_party,
    resp_c = dr$resp_cov %||% NULL,
    resp_c_present = dr$resp_cov_present %||% NULL,
    experiment_idx = dr$experiment_idx %||% NULL,
    return_logits = FALSE
  )
  if (!is.list(pred_obs) || is.null(pred_obs$mu)) {
    return(q_vec)
  }

  log_target <- neural_policy_log_prob_observed_profiles(
    pi_vec = pi_star_ast,
    X_idx = dr$X_single,
    model_info = model_ast,
    baseline_level_probs = NULL
  )
  log_baseline <- neural_policy_log_prob_observed_profiles(
    pi_vec = pi_star_ast,
    X_idx = dr$X_single,
    model_info = model_ast,
    baseline_level_probs = dr$baseline_level_probs
  )
  w <- strenv$jnp$exp(log_target - log_baseline)
  denom <- strenv$jnp$sum(w)
  eps <- strenv$jnp$array(1e-8, dtype = w$dtype)
  # jnp$where evaluates BOTH branches, so dividing by the raw denom (which the
  # condition guards against being ~0) poisons the reverse pass with NaN when a
  # degenerate pi* underflows the importance weights. Divide by a floored
  # denominator so both branches stay finite; the where still selects 0.
  safe_denom <- strenv$jnp$maximum(denom, eps)
  residual <- strenv$jnp$reshape(dr$Y_single, list(-1L)) -
    strenv$jnp$reshape(pred_obs$mu, list(-1L))
  correction <- strenv$jnp$where(
    denom > eps,
    strenv$jnp$sum(w * residual) / safe_denom,
    strenv$jnp$array(0., dtype = w$dtype)
  )
  q_vec + correction
}

neural_resolve_model_info <- function(name) {
  if (exists(name, inherits = TRUE)) {
    return(get(name, inherits = TRUE))
  }
  if (exists("strenv", inherits = TRUE)) {
    model_env <- tryCatch(strenv$neural_model_env, error = function(e) NULL)
    if (is.environment(model_env) &&
        exists(name, envir = model_env, inherits = TRUE)) {
      return(get(name, envir = model_env, inherits = TRUE))
    }
  }
  NULL
}

# The soft policy builders never receive a per-row experiment index or
# per-experiment factor order: policy optimization always evaluates under
# model_info's default_experiment_index / default_factor_order. That is
# correct for single-study fits and for adapted FM predictors whose defaults
# were pinned to the target study -- but for a multi-experiment pooled model
# with no pinned default, pi* would be optimized against the wrong (global
# fallback) study context with nothing visible to the user. Warn once.
neural_warn_soft_default_context <- function(model_info, label = "policy optimization") {
  if (is.null(model_info)) {
    return(invisible(NULL))
  }
  n_exp <- suppressWarnings(as.integer(model_info$n_experiment_levels %||%
                                         length(model_info$experiment_levels %||% character(0))))
  n_exp <- n_exp[1L]
  if (is.na(n_exp) || n_exp <= 1L) {
    return(invisible(NULL))
  }
  default_idx <- model_info$default_experiment_index %||% NULL
  pinned <- !is.null(default_idx) && length(default_idx) == 1L && !is.na(default_idx)
  if (isTRUE(pinned)) {
    return(invisible(NULL))
  }
  if (isTRUE(strenv$soft_default_context_warned)) {
    return(invisible(NULL))
  }
  strenv$soft_default_context_warned <- TRUE
  warning(sprintf(paste0(
    "The neural model pools %d experiments but no default_experiment_index ",
    "is pinned; %s will evaluate the soft policy under the global fallback ",
    "context rather than a specific study. Adapt the foundation model to the ",
    "target study (which pins the default) before optimizing."
  ), n_exp, label), call. = FALSE)
  invisible(NULL)
}

neural_getQStar_single <- function(pi_star_ast,
                                   EST_COEFFICIENTS_tf_ast) {
  model_ast <- neural_resolve_model_info("neural_model_info_ast_jnp")
  if (is.null(model_ast)) {
    stop("neural_getQStar_single requires neural_model_info_ast_jnp.", call. = FALSE)
  }
  neural_warn_soft_default_context(model_ast)
  party_label <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    GroupsPool[1]
  } else {
    NULL
  }
  party_idx <- neural_get_party_index(model_ast, party_label)
  resp_idx <- neural_get_resp_party_index(model_ast, party_label)
  params_ast <- neural_params_from_theta(EST_COEFFICIENTS_tf_ast, model_ast)
  q_linear <- neural_average_case_linear_q_from_policy(pi_star_ast, model_ast)
  Qhat <- if (!is.null(q_linear)) {
    strenv$jnp$reshape(q_linear, list(1L, 1L))
  } else {
    neural_predict_single_soft(pi_star_ast, party_idx, resp_idx, model_ast,
                               params = params_ast)
  }
  strenv$jnp$concatenate(list(Qhat, Qhat, Qhat), 0L)
}

neural_getQStar_diff_BASE <- function(pi_star_ast, pi_star_dag,
                                      EST_COEFFICIENTS_tf_ast,
                                      EST_COEFFICIENTS_tf_dag) {
  model_ast <- neural_resolve_model_info("neural_model_info_ast_jnp")
  if (is.null(model_ast)) {
    stop("neural_getQStar_diff_BASE requires neural_model_info_ast_jnp.", call. = FALSE)
  }
  neural_warn_soft_default_context(model_ast)
  model_dag <- neural_resolve_model_info("neural_model_info_dag_jnp")
  if (is.null(model_dag)) {
    model_dag <- model_ast
  }

  party_label_ast <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    GroupsPool[1]
  } else {
    NULL
  }
  party_label_dag <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 1) {
    GroupsPool[2]
  } else {
    party_label_ast
  }

  party_idx_ast <- neural_get_party_index(model_ast, party_label_ast)
  party_idx_dag <- neural_get_party_index(model_dag, party_label_dag)
  resp_idx_ast <- neural_get_resp_party_index(model_ast, party_label_ast)
  resp_idx_dag <- neural_get_resp_party_index(model_dag, party_label_dag)
  params_ast <- neural_params_from_theta(EST_COEFFICIENTS_tf_ast, model_ast)
  params_dag <- neural_params_from_theta(EST_COEFFICIENTS_tf_dag, model_dag)

  disaggregate <- if (exists("Q_DISAGGREGATE", inherits = TRUE)) {
    isTRUE(Q_DISAGGREGATE)
  } else if (exists("adversarial", inherits = TRUE)) {
    isTRUE(adversarial)
  } else {
    FALSE
  }
  if (!isTRUE(disaggregate)) {
    resp_idx_dag <- resp_idx_ast
    params_dag <- params_ast
  }

  Qhat_ast_among_ast <- neural_predict_pair_soft(
    pi_star_ast, pi_star_dag,
    party_idx_ast, party_idx_dag,
    resp_idx_ast, model_ast,
    params = params_ast
  )

  if (!isTRUE(disaggregate)) {
    Qhat_population <- Qhat_ast_among_dag <- Qhat_ast_among_ast
  } else {
    Qhat_ast_among_dag <- neural_predict_pair_soft(
      pi_star_ast, pi_star_dag,
      party_idx_ast, party_idx_dag,
      resp_idx_dag, model_dag,
      params = params_dag
    )
    Qhat_population <- Qhat_ast_among_ast * strenv$jnp$array(strenv$AstProp) +
      Qhat_ast_among_dag * strenv$jnp$array(strenv$DagProp)
  }

  strenv$jnp$concatenate(list(Qhat_population,
                              Qhat_ast_among_ast,
                              Qhat_ast_among_dag), 0L)
}

cs2step_build_pair_mat <- function(pair_id,
                                   W,
                                   n_rows = NULL,
                                   row_hash = NULL,
                                   profile_order = NULL,
                                   competing_group_variable_candidate = NULL) {
  if (is.null(pair_id) || !length(pair_id)) {
    return(NULL)
  }
  pair_id <- as.vector(pair_id)
  if (is.null(n_rows)) {
    if (is.null(W) || !length(W)) {
      stop("cs2step_build_pair_mat requires a non-empty W or n_rows.", call. = FALSE)
    }
    W <- as.matrix(W)
    n_rows <- nrow(W)
  } else {
    n_rows <- as.integer(n_rows)
  }
  if (length(pair_id) != n_rows) {
    stop(sprintf("pair_id has %d elements but W has %d rows.",
                 length(pair_id), n_rows),
         call. = FALSE)
  }

  pair_key <- as.character(pair_id)
  pair_indices_list <- split(
    seq_along(pair_id),
    factor(pair_key, levels = unique(pair_key)),
    drop = TRUE
  )
  profile_order_present <- !is.null(profile_order) &&
    length(profile_order) == length(pair_id)

  if (is.null(row_hash)) {
    row_hash <- if (!is.null(W) && length(W)) {
      row_key <- apply(W, 1, function(row) {
        paste(ifelse(is.na(row), "NA", as.character(row)), collapse = "|")
      })
      vapply(row_key, function(key) {
        ints <- utf8ToInt(key)
        if (!length(ints)) {
          return(0)
        }
        sum(as.numeric(ints) * seq_along(ints)) %% 2147483647
      }, numeric(1))
    } else {
      seq_len(n_rows)
    }
  }

  pair_mat <- do.call(rbind, lapply(pair_indices_list, function(idx){
    order_by_profile <- profile_order_present &&
      length(idx) == 2L &&
      length(unique(profile_order[idx])) == 2L &&
      !any(is.na(profile_order[idx]))
    if (!is.null(competing_group_variable_candidate)) {
      if (order_by_profile) {
        idx[order(competing_group_variable_candidate[idx],
                  profile_order[idx],
                  row_hash[idx],
                  idx)]
      } else {
        idx[order(competing_group_variable_candidate[idx],
                  row_hash[idx],
                  idx)]
      }
    } else if (order_by_profile) {
      idx[order(profile_order[idx],
                row_hash[idx],
                idx)]
    } else {
      idx[order(row_hash[idx], idx)]
    }
  }))

  list(
    pair_mat = pair_mat,
    pair_sizes = lengths(pair_indices_list),
    profile_order_present = profile_order_present
  )
}

cs2step_compact_n_rows <- function(x) {
  if (is.null(x)) {
    return(0L)
  }
  as.integer(x$n_rows %||% sum(vapply(x$blocks %||% list(), function(block) {
    as.integer(block$n_rows %||% 0L)
  }, integer(1))))
}

cs2step_compact_n_cols <- function(x, field = c("n_factors", "n_covariates")) {
  field <- match.arg(field)
  if (is.null(x)) {
    return(0L)
  }
  as.integer(x[[field]] %||% {
    blocks <- x$blocks %||% list()
    if (length(blocks)) {
      blocks[[1L]][[field]] %||% 0L
    } else {
      0L
    }
  })
}

cs2step_compact_block_ranges <- function(x) {
  blocks <- x$blocks %||% list()
  n_by_block <- vapply(blocks, function(block) as.integer(block$n_rows %||% 0L), integer(1))
  starts <- cumsum(c(1L, head(n_by_block, -1L)))
  ends <- cumsum(n_by_block)
  data.frame(block = seq_along(blocks), start = starts, end = ends)
}

cs2step_compact_rows_by_block <- function(x, rows = NULL) {
  total_rows <- cs2step_compact_n_rows(x)
  if (is.null(rows)) {
    rows <- seq_len(total_rows)
  }
  rows <- as.integer(rows)
  if (length(rows) < 1L) {
    return(list(rows = rows, split = list()))
  }
  if (any(is.na(rows) | rows < 1L | rows > total_rows)) {
    stop("Compact row index out of bounds.", call. = FALSE)
  }
  ranges <- cs2step_compact_block_ranges(x)
  block_id <- findInterval(rows, ranges$start)
  block_id[block_id < 1L] <- 1L
  block_id[block_id > nrow(ranges)] <- nrow(ranges)
  split_idx <- split(seq_along(rows), block_id)
  list(rows = rows, ranges = ranges, split = split_idx)
}

cs2step_compact_present_cols <- function(block, n_cols, context) {
  present_cols <- as.integer(block$present_cols %||% integer(0))
  if (length(present_cols) > 0L &&
      any(is.na(present_cols) | present_cols < 1L | present_cols > n_cols)) {
    stop(sprintf("Compact %s block present_cols are out of bounds.", context), call. = FALSE)
  }
  present_cols
}

cs2step_compact_block_values <- function(block, present_cols, context, storage = c("double", "integer")) {
  storage <- match.arg(storage)
  values <- as.matrix(block$values)
  if (ncol(values) != length(present_cols)) {
    stop(
      sprintf("Compact %s block values must have one column per present column.", context),
      call. = FALSE
    )
  }
  storage.mode(values) <- storage
  values
}

cs2step_validate_w_idx_compact <- function(x, factor_levels) {
  if (is.null(x) || is.matrix(x) || is.data.frame(x)) {
    return(invisible(x))
  }
  factor_levels <- as.integer(factor_levels %||% integer(0))
  n_cols <- cs2step_compact_n_cols(x, "n_factors")
  if (length(factor_levels) != n_cols) {
    stop("Compact factor index metadata must align with factor_levels.", call. = FALSE)
  }
  expected_holdout <- factor_levels + 1L
  holdout_codes <- as.integer(x$holdout_codes %||% integer(0))
  if (length(holdout_codes) != n_cols || any(holdout_codes != expected_holdout)) {
    stop(
      "Compact factor holdout_codes must equal factor_levels + 1 for the explicit holdout row.",
      call. = FALSE
    )
  }
  for (block in x$blocks %||% list()) {
    block_holdout <- as.integer(block$holdout_codes %||% holdout_codes)
    if (length(block_holdout) != n_cols || any(block_holdout != expected_holdout)) {
      stop(
        "Compact factor block holdout_codes must equal factor_levels + 1 for the explicit holdout row.",
        call. = FALSE
      )
    }
    present_cols <- cs2step_compact_present_cols(block, n_cols, "factor")
    if (length(present_cols) < 1L) {
      next
    }
    vals <- cs2step_compact_block_values(block, present_cols, "factor", storage = "integer")
    for (j in seq_along(present_cols)) {
      col_idx <- present_cols[[j]]
      bad <- vals[, j]
      bad <- bad[!is.na(bad) & (bad < 1L | bad > expected_holdout[[col_idx]])]
      if (length(bad) > 0L) {
        stop("Compact factor block values contain an out-of-range level code.", call. = FALSE)
      }
    }
  }
  invisible(x)
}

cs2step_materialize_w_idx_compact <- function(x, rows = NULL) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.matrix(x) || is.data.frame(x)) {
    out <- as.matrix(x)
    storage.mode(out) <- "integer"
    if (!is.null(rows)) {
      out <- out[rows, , drop = FALSE]
    }
    return(out)
  }
  n_cols <- cs2step_compact_n_cols(x, "n_factors")
  holdout_codes <- as.integer(x$holdout_codes %||% integer(0))
  if (length(holdout_codes) != n_cols) {
    stop("Compact factor index metadata must include one holdout code per factor.", call. = FALSE)
  }
  rows_info <- cs2step_compact_rows_by_block(x, rows)
  out <- matrix(
    rep.int(holdout_codes, rep.int(length(rows_info$rows), n_cols)),
    nrow = length(rows_info$rows),
    ncol = n_cols
  )
  colnames(out) <- x$factor_names %||% character(n_cols)
  for (block_name in names(rows_info$split)) {
    block_idx <- as.integer(block_name)
    block <- x$blocks[[block_idx]]
    out_rows <- rows_info$split[[block_name]]
    local_rows <- rows_info$rows[out_rows] - rows_info$ranges$start[[block_idx]] + 1L
    present_cols <- cs2step_compact_present_cols(block, n_cols, "factor")
    if (length(present_cols) > 0L) {
      vals <- cs2step_compact_block_values(block, present_cols, "factor", storage = "integer")
      vals <- vals[local_rows, , drop = FALSE]
      out[out_rows, present_cols] <- vals
    }
  }
  storage.mode(out) <- "integer"
  out
}

cs2step_compact_missing_rows <- function(block, present_cols, n_rows) {
  out <- setNames(vector("list", length(present_cols)), as.character(present_cols))
  missing_rows <- block$missing_rows_by_col %||% list()
  for (col_name in names(out)) {
    vals <- as.integer(missing_rows[[col_name]] %||% integer(0))
    vals <- vals[!is.na(vals) & vals >= 1L & vals <= n_rows]
    out[[col_name]] <- vals
  }
  out
}

cs2step_materialize_x_compact <- function(x, rows = NULL) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.matrix(x) || is.data.frame(x)) {
    out <- as.matrix(x)
    storage.mode(out) <- "double"
    if (!is.null(rows)) {
      out <- out[rows, , drop = FALSE]
    }
    return(out)
  }
  n_cols <- cs2step_compact_n_cols(x, "n_covariates")
  rows_info <- cs2step_compact_rows_by_block(x, rows)
  out <- matrix(0, nrow = length(rows_info$rows), ncol = n_cols)
  colnames(out) <- x$covariate_names %||% character(n_cols)
  for (block_name in names(rows_info$split)) {
    block_idx <- as.integer(block_name)
    block <- x$blocks[[block_idx]]
    out_rows <- rows_info$split[[block_name]]
    local_rows <- rows_info$rows[out_rows] - rows_info$ranges$start[[block_idx]] + 1L
    present_cols <- cs2step_compact_present_cols(block, n_cols, "covariate")
    if (length(present_cols) > 0L) {
      vals <- cs2step_compact_block_values(block, present_cols, "covariate", storage = "double")
      vals <- vals[local_rows, , drop = FALSE]
      vals[!is.finite(vals)] <- 0
      out[out_rows, present_cols] <- vals
    }
  }
  storage.mode(out) <- "double"
  out
}

cs2step_materialize_x_present_compact <- function(x, rows = NULL) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.matrix(x) || is.data.frame(x)) {
    out <- as.matrix(x)
    storage.mode(out) <- "double"
    if (!is.null(rows)) {
      out <- out[rows, , drop = FALSE]
    }
    return(out)
  }
  n_cols <- cs2step_compact_n_cols(x, "n_covariates")
  rows_info <- cs2step_compact_rows_by_block(x, rows)
  out <- matrix(0, nrow = length(rows_info$rows), ncol = n_cols)
  colnames(out) <- x$covariate_names %||% character(n_cols)
  for (block_name in names(rows_info$split)) {
    block_idx <- as.integer(block_name)
    block <- x$blocks[[block_idx]]
    out_rows <- rows_info$split[[block_name]]
    local_rows <- rows_info$rows[out_rows] - rows_info$ranges$start[[block_idx]] + 1L
    present_cols <- cs2step_compact_present_cols(block, n_cols, "covariate")
    if (length(present_cols) > 0L) {
      out[out_rows, present_cols] <- 1
      missing_by_col <- cs2step_compact_missing_rows(block, present_cols, as.integer(block$n_rows %||% 0L))
      for (col_name in names(missing_by_col)) {
        missing_local <- missing_by_col[[col_name]]
        if (length(missing_local) > 0L) {
          hit <- match(missing_local, local_rows, nomatch = 0L)
          hit <- hit[hit > 0L]
          if (length(hit) > 0L) {
            out[out_rows[hit], as.integer(col_name)] <- 0
          }
        }
      }
    }
  }
  storage.mode(out) <- "double"
  out
}

cs2step_compact_covariate_profiles <- function(x,
                                               rows = NULL,
                                               experiment_index = NULL,
                                               covariate_names = NULL,
                                               default_experiment_index = NULL) {
  covariate_names <- as.character(covariate_names %||% x$covariate_names %||% character(0))
  n_covariates <- length(covariate_names)
  default_stats_row <- neural_default_covariate_value_stats_row()
  default_meta_row <- neural_default_covariate_value_metadata_row()
  if (n_covariates < 1L || is.null(x)) {
    empty <- neural_build_covariate_distribution_profiles(
      X_mat = matrix(numeric(0), nrow = 0L, ncol = 0L),
      covariate_names = covariate_names
    )
    return(list(
      mean = NULL,
      scale = NULL,
      default_present = NULL,
      distribution_profiles = empty
    ))
  }
  rows_info <- cs2step_compact_rows_by_block(x, rows)
  n_total <- length(rows_info$rows)
  values_by_col <- setNames(vector("list", n_covariates), as.character(seq_len(n_covariates)))
  values_by_exp <- list()
  n_by_exp <- integer(0)
  for (block_name in names(rows_info$split)) {
    block_idx <- as.integer(block_name)
    block <- x$blocks[[block_idx]]
    exp_idx <- as.integer(block$experiment_index %||% (block_idx - 1L))
    exp_key <- as.character(exp_idx)
    out_rows <- rows_info$split[[block_name]]
    local_rows <- rows_info$rows[out_rows] - rows_info$ranges$start[[block_idx]] + 1L
    current_n <- n_by_exp[exp_key]
    if (length(current_n) != 1L || is.na(current_n)) {
      current_n <- 0L
    }
    n_by_exp[exp_key] <- as.integer(current_n) + length(local_rows)
    if (is.null(values_by_exp[[exp_key]])) {
      values_by_exp[[exp_key]] <- setNames(vector("list", n_covariates), as.character(seq_len(n_covariates)))
    }
    present_cols <- cs2step_compact_present_cols(block, n_covariates, "covariate")
    if (length(present_cols) < 1L) {
      next
    }
    vals_mat <- cs2step_compact_block_values(block, present_cols, "covariate", storage = "double")
    vals_mat <- vals_mat[local_rows, , drop = FALSE]
    for (j in seq_along(present_cols)) {
      col_idx <- present_cols[[j]]
      vals <- vals_mat[, j]
      vals <- vals[is.finite(vals)]
      if (length(vals) > 0L) {
        key <- as.character(col_idx)
        values_by_col[[key]] <- c(values_by_col[[key]], vals)
        values_by_exp[[exp_key]][[key]] <- c(values_by_exp[[exp_key]][[key]], vals)
      }
    }
  }
  build_summary_mats <- function(values_list, rows_n) {
    stats_mat <- neural_empty_covariate_distribution_matrix(
      n_covariates = n_covariates,
      colnames_use = covariate_names,
      default_row = default_stats_row
    )
    meta_mat <- neural_empty_covariate_distribution_matrix(
      n_covariates = n_covariates,
      colnames_use = covariate_names,
      default_row = default_meta_row
    )
    for (j in seq_len(n_covariates)) {
      vals <- values_list[[as.character(j)]] %||% numeric(0)
      present <- c(rep(1, length(vals)), rep(0, max(0L, rows_n - length(vals))))
      x_col <- c(vals, rep(0, max(0L, rows_n - length(vals))))
      summary_j <- neural_covariate_distribution_summary(x_col = x_col, present_col = present)
      stats_mat[j, ] <- summary_j$stats
      meta_mat[j, ] <- summary_j$metadata
    }
    list(stats = stats_mat, metadata = meta_mat)
  }
  global_summary <- build_summary_mats(values_by_col, n_total)
  max_exp <- if (length(n_by_exp)) max(as.integer(names(n_by_exp)), na.rm = TRUE) else -1L
  by_experiment <- if (is.finite(max_exp) && max_exp >= 0L) vector("list", max_exp + 1L) else list()
  metadata_by_experiment <- if (is.finite(max_exp) && max_exp >= 0L) vector("list", max_exp + 1L) else list()
  for (exp_key in names(values_by_exp)) {
    exp_idx <- as.integer(exp_key)
    summary_i <- build_summary_mats(values_by_exp[[exp_key]], as.integer(n_by_exp[[exp_key]] %||% 0L))
    by_experiment[[exp_idx + 1L]] <- summary_i$stats
    metadata_by_experiment[[exp_idx + 1L]] <- summary_i$metadata
  }
  default_stats <- global_summary$stats
  default_metadata <- global_summary$metadata
  if (!is.null(default_experiment_index) && !is.na(default_experiment_index) &&
      length(by_experiment) >= (as.integer(default_experiment_index) + 1L)) {
    idx <- as.integer(default_experiment_index) + 1L
    default_stats <- by_experiment[[idx]] %||% default_stats
    default_metadata <- metadata_by_experiment[[idx]] %||% default_metadata
  }
  list(
    mean = as.numeric(global_summary$stats[, "mean"]),
    scale = as.numeric(global_summary$stats[, "scale"]),
    default_present = as.numeric(vapply(values_by_col, function(vals) length(vals) > 0L, logical(1))),
    distribution_profiles = list(
      by_experiment = by_experiment,
      metadata_by_experiment = metadata_by_experiment,
      default_stats = default_stats,
      default_metadata = default_metadata
    )
  )
}

generate_ModelOutcome_neural <- function(){
  message("Defining MCMC parameters in generate_ModelOutcome_neural...")
  mcmc_control <- list(
    backend = "numpyro",
    n_samples_warmup = 500L,
    n_samples_mcmc   = 1000L,
    batch_size = 512L,
    chain_method = "parallel",
    subsample_method = "full",
    n_thin_by = 1L,
    n_chains = 2L,
    svi_steps = "optimal",
    svi_lr = 0.01,
    svi_num_particles = 1L,
    svi_num_draws = 50L,
    vi_guide = "auto_normal",
    optimizer = "muon",
    svi_clip_global_norm = 10,
    early_stopping = TRUE,
    early_stopping_n_checks = 10L,
    early_stopping_patience = 3L,
    early_stopping_validation_frac = 0.05,
    early_stopping_validation_max_n = 2048L,
    early_stopping_validation_batch_size = 128L,
    gradient_diagnostics = TRUE,
    svi_lr_schedule = "warmup_cosine",
    svi_lr_warmup_frac = 0.1,
    svi_lr_end_factor = 0.01,
    compact_update_chunk_size = 8L,
    compact_update_scan = "required",
    checkpoint_path = NULL,
    checkpoint_resume = NULL,
    checkpoint_n_checks = 10L,
    checkpoint_compress = FALSE,
    attention_backend = "auto",
    attention_dtype = "auto",
    attention_padding_multiple = 8L,
    learned_pairwise_bernoulli_logit_scale = FALSE,
    pairwise_bernoulli_logit_scale_prior_sd = 0.5,
    pairwise_antisymmetry = "strict",
    additive_utility = "auto",
    additive_utility_normalization = NULL,
    additive_head_weight_target_rms = NULL,
    calibration = list(
      enabled = "auto",
      method = "logit_scale",
      prior_sd = 0.5
    ),
    universal_loss_weighting = "empirical",
    universal_family_logprob_normalize = FALSE,
    balanced_sampling = NULL,
    RMS_scale = 0.25,
    qk_rms_scale = 0.1,
    moe_load_balance_lambda = 0.01,
    residual_weight_depth_scale = "floor_one",
    svi_lr_plate_rescale = TRUE,
    # For single-candidate normal outcomes, optimize/report the average-case
    # linear (main + 2-way OLS) Q surrogate instead of the neural soft
    # prediction. Announced via message when active; set FALSE to optimize the
    # neural prediction directly.
    average_case_linear_q = TRUE,
    seed = 123L
  )
  UsedRegularization <- FALSE
  uncertainty_scope <- "all"
  mcmc_overrides <- NULL
  eval_control <- list(enabled = TRUE, max_n = NULL, seed = 123L, n_folds = NULL)
  model_dims <- 128L
  model_depth <- 2L
  qk_norm_enabled <- TRUE
  residual_mode <- "standard"
  attention_backend <- "auto"
  attention_dtype <- "auto"
  attention_padding_multiple <- 8L
  cross_candidate_encoder_mode <- "none"
  cross_candidate_encoder_supplied <- FALSE
  cross_candidate_encoder_note <- NULL
  low_rank_logit_transform_control <- NULL
  low_rank_logit_transform_supplied <- FALSE
  low_rank_logit_bound_control <- NULL
  low_rank_logit_bound_supplied <- FALSE
  low_rank_logit_softness_control <- NULL
  low_rank_logit_softness_supplied <- FALSE
  low_rank_logit_normalization_control <- NULL
  low_rank_logit_normalization_supplied <- FALSE
  low_rank_head_weight_target_rms_control <- NULL
  low_rank_head_weight_target_rms_supplied <- FALSE
  low_rank_rc_out_target_rms_control <- NULL
  low_rank_rc_out_target_rms_supplied <- FALSE
  low_rank_logit_normalization <- "none"
  low_rank_head_weight_target_rms <- NULL
  low_rank_rc_out_target_rms <- NULL
  pairwise_antisymmetry_control <- mcmc_control$pairwise_antisymmetry
  pairwise_antisymmetry_mode <- "strict"
  additive_utility_control <- mcmc_control$additive_utility
  additive_utility_mode <- "off"
  additive_utility_normalization_control <- NULL
  additive_utility_normalization_supplied <- FALSE
  additive_head_weight_target_rms_control <- NULL
  additive_head_weight_target_rms_supplied <- FALSE
  additive_utility_normalization <- "none"
  additive_head_weight_target_rms <- NULL
  calibration_control_raw <- mcmc_control$calibration
  calibration_control_supplied <- FALSE
  calibration_control <- list(enabled = FALSE, method = "none", prior_sd = NULL)
  warn_stage_imbalance_pct <- 0.10
  warn_min_cell_n <- 50L
  neural_oos_eval_internal_flag <- exists("neural_oos_eval_internal", inherits = TRUE) &&
    isTRUE(get("neural_oos_eval_internal", inherits = TRUE))

  file_suffix <- if (!is.null(outcome_model_key)) {
    sprintf("%s_%s_%s", GroupsPool[GroupCounter], Round_, outcome_model_key)
  } else {
    sprintf("%s_%s", GroupsPool[GroupCounter], Round_)
  }
  if (isTRUE(adversarial) && adversarial_model_strategy == "two") {
    file_suffix <- sprintf("%s_two", file_suffix)
  }
  bundle_path <- sprintf("./StrategizeInternals/neural_bundle_%s.rds", file_suffix)

  normalize_cross_candidate_encoder <- function(value) {
    if (is.null(value)) {
      return("none")
    }
    mode <- neural_normalize_cross_encoder_mode(value)
    if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      return(mode)
    }
    stop(
      "'neural_mcmc_control$cross_candidate_encoder' must be TRUE/FALSE or one of ",
      "'none', 'term', 'attn', or 'full'.",
      call. = FALSE
    )
  }
  normalize_residual_mode <- function(value) {
    mode <- neural_normalize_residual_mode(value)
    if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      return(mode)
    }
    stop(
      "'neural_mcmc_control$residual_mode' must be one of ",
      "'standard' or 'full_attn'.",
      call. = FALSE
    )
  }
  normalize_attention_backend <- function(value) {
    mode <- neural_normalize_attention_backend(value)
    if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      return(mode)
    }
    stop(
      "'neural_mcmc_control$attention_backend' must be one of ",
      "'auto', 'xla', or 'cudnn'.",
      call. = FALSE
    )
  }
  normalize_attention_dtype <- function(value) {
    mode <- neural_normalize_attention_dtype(value)
    if (is.character(mode) && length(mode) == 1L && !is.na(mode) && nzchar(mode)) {
      return(mode)
    }
    stop(
      "'neural_mcmc_control$attention_dtype' must be one of ",
      "'auto', 'float32', 'bfloat16', or 'float16'.",
      call. = FALSE
    )
  }
  if (exists("neural_mcmc_control", inherits = TRUE) &&
      !is.null(neural_mcmc_control)) {
    if (!is.list(neural_mcmc_control)) {
      stop("'neural_mcmc_control' must be a list.", call. = FALSE)
    }
    if (!is.null(neural_mcmc_control$uncertainty_scope)) {
      uncertainty_scope <- tolower(as.character(neural_mcmc_control$uncertainty_scope))
    }
    if (!is.null(neural_mcmc_control$eval_enabled)) {
      eval_control$enabled <- isTRUE(neural_mcmc_control$eval_enabled)
    }
    if (!is.null(neural_mcmc_control$eval_max_n)) {
      eval_control$max_n <- as.integer(neural_mcmc_control$eval_max_n)
    }
    if (!is.null(neural_mcmc_control$eval_seed)) {
      eval_control$seed <- as.integer(neural_mcmc_control$eval_seed)
    }
    if (!is.null(neural_mcmc_control$eval_n_folds)) {
      eval_control$n_folds <- as.integer(neural_mcmc_control$eval_n_folds)
    } else if (!is.null(neural_mcmc_control$eval_folds)) {
      eval_control$n_folds <- as.integer(neural_mcmc_control$eval_folds)
    }
    if (!is.null(neural_mcmc_control$ModelDims)) {
      model_dims <- neural_mcmc_control$ModelDims
    }
    if (!is.null(neural_mcmc_control$ModelDepth)) {
      model_depth <- neural_mcmc_control$ModelDepth
    }
    if (!is.null(neural_mcmc_control$qk_norm)) {
      qk_norm_value <- neural_mcmc_control$qk_norm
      if (is.logical(qk_norm_value) && length(qk_norm_value) == 1L && !is.na(qk_norm_value)) {
        qk_norm_enabled <- isTRUE(qk_norm_value)
      } else if (is.character(qk_norm_value)) {
        qk_norm_tag <- tolower(as.character(qk_norm_value))
        if (length(qk_norm_tag) == 1L && !is.na(qk_norm_tag) &&
            qk_norm_tag %in% c("true", "false")) {
          qk_norm_enabled <- identical(qk_norm_tag, "true")
        } else {
          stop("'neural_mcmc_control$qk_norm' must be TRUE/FALSE.", call. = FALSE)
        }
      } else {
        stop("'neural_mcmc_control$qk_norm' must be TRUE/FALSE.", call. = FALSE)
      }
    }
    if (!is.null(neural_mcmc_control$residual_mode)) {
      residual_mode <- normalize_residual_mode(neural_mcmc_control$residual_mode)
    }
    if (!is.null(neural_mcmc_control$attention_backend)) {
      attention_backend <- normalize_attention_backend(neural_mcmc_control$attention_backend)
    }
    if (!is.null(neural_mcmc_control$attention_dtype)) {
      attention_dtype <- normalize_attention_dtype(neural_mcmc_control$attention_dtype)
    }
    if (!is.null(neural_mcmc_control$attention_padding_multiple)) {
      attention_padding_multiple <- suppressWarnings(as.integer(neural_mcmc_control$attention_padding_multiple))
      if (length(attention_padding_multiple) != 1L ||
          is.na(attention_padding_multiple) ||
          attention_padding_multiple < 1L) {
        stop("'neural_mcmc_control$attention_padding_multiple' must be an integer >= 1.", call. = FALSE)
      }
    }
    if (!is.null(neural_mcmc_control$cross_candidate_encoder)) {
      cross_candidate_encoder_supplied <- TRUE
      cross_candidate_encoder_mode <- normalize_cross_candidate_encoder(
        neural_mcmc_control$cross_candidate_encoder
      )
    }
    if ("pairwise_antisymmetry" %in% names(neural_mcmc_control)) {
      pairwise_antisymmetry_control <- neural_mcmc_control$pairwise_antisymmetry
    }
    if ("additive_utility" %in% names(neural_mcmc_control)) {
      additive_utility_control <- neural_mcmc_control$additive_utility
    }
    if ("additive_utility_normalization" %in% names(neural_mcmc_control)) {
      additive_utility_normalization_supplied <- TRUE
      additive_utility_normalization_control <- neural_mcmc_control$additive_utility_normalization
    }
    if ("additive_head_weight_target_rms" %in% names(neural_mcmc_control)) {
      additive_head_weight_target_rms_supplied <- TRUE
      additive_head_weight_target_rms_control <- neural_mcmc_control$additive_head_weight_target_rms
    }
    if ("calibration" %in% names(neural_mcmc_control)) {
      calibration_control_supplied <- TRUE
      calibration_control_raw <- neural_mcmc_control$calibration
    }
    if ("low_rank_logit_transform" %in% names(neural_mcmc_control)) {
      low_rank_logit_transform_supplied <- TRUE
      low_rank_logit_transform_control <- neural_mcmc_control$low_rank_logit_transform
    }
    if ("low_rank_logit_bound" %in% names(neural_mcmc_control)) {
      low_rank_logit_bound_supplied <- TRUE
      low_rank_logit_bound_control <- neural_mcmc_control$low_rank_logit_bound
    }
    if ("low_rank_logit_softness" %in% names(neural_mcmc_control)) {
      low_rank_logit_softness_supplied <- TRUE
      low_rank_logit_softness_control <- neural_mcmc_control$low_rank_logit_softness
    }
    if ("low_rank_logit_normalization" %in% names(neural_mcmc_control)) {
      low_rank_logit_normalization_supplied <- TRUE
      low_rank_logit_normalization_control <- neural_mcmc_control$low_rank_logit_normalization
    }
    if ("low_rank_head_weight_target_rms" %in% names(neural_mcmc_control)) {
      low_rank_head_weight_target_rms_supplied <- TRUE
      low_rank_head_weight_target_rms_control <- neural_mcmc_control$low_rank_head_weight_target_rms
    }
    if ("low_rank_rc_out_target_rms" %in% names(neural_mcmc_control)) {
      low_rank_rc_out_target_rms_supplied <- TRUE
      low_rank_rc_out_target_rms_control <- neural_mcmc_control$low_rank_rc_out_target_rms
    }
    if (!is.null(neural_mcmc_control$warn_stage_imbalance_pct)) {
      warn_stage_imbalance_pct <- as.numeric(neural_mcmc_control$warn_stage_imbalance_pct)
    }
    if (!is.null(neural_mcmc_control$warn_min_cell_n)) {
      warn_min_cell_n <- as.integer(neural_mcmc_control$warn_min_cell_n)
    }
    mcmc_overrides <- neural_mcmc_control
    mcmc_overrides$uncertainty_scope <- NULL
    mcmc_overrides$n_bayesian_models <- NULL
    mcmc_overrides$ModelDims <- NULL
    mcmc_overrides$ModelDepth <- NULL
    mcmc_overrides$qk_norm <- NULL
    mcmc_overrides$residual_mode <- NULL
    mcmc_overrides$attention_backend <- NULL
    mcmc_overrides$attention_dtype <- NULL
    mcmc_overrides$attention_padding_multiple <- NULL
    mcmc_overrides$cross_candidate_encoder <- NULL
    mcmc_overrides$pairwise_antisymmetry <- NULL
    mcmc_overrides$additive_utility <- NULL
    mcmc_overrides$additive_utility_normalization <- NULL
    mcmc_overrides$additive_head_weight_target_rms <- NULL
    mcmc_overrides$calibration <- NULL
    mcmc_overrides$low_rank_logit_transform <- NULL
    mcmc_overrides$low_rank_logit_bound <- NULL
    mcmc_overrides$low_rank_logit_softness <- NULL
    mcmc_overrides$low_rank_logit_normalization <- NULL
    mcmc_overrides$low_rank_head_weight_target_rms <- NULL
    mcmc_overrides$low_rank_rc_out_target_rms <- NULL
  }
  uncertainty_scope_env <- Sys.getenv("STRATEGIZE_NEURAL_UNCERTAINTY_SCOPE")
  if (nzchar(uncertainty_scope_env)) {
    uncertainty_scope <- tolower(as.character(uncertainty_scope_env))
  }
  fast_mcmc_flag <- tolower(Sys.getenv("STRATEGIZE_NEURAL_FAST_MCMC")) %in%
    c("1", "true", "yes")
  if (isTRUE(fast_mcmc_flag)) {
    mcmc_control$n_samples_warmup <- 50L
    mcmc_control$n_samples_mcmc <- 50L
    mcmc_control$batch_size <- 128L
    mcmc_control$n_chains <- 1L
    mcmc_control$chain_method <- "sequential"
    mcmc_control$svi_steps <- 200L
    mcmc_control$svi_num_draws <- 100L
  }
  if (!is.null(mcmc_overrides) && length(mcmc_overrides) > 0) {
    mcmc_control <- modifyList(mcmc_control, mcmc_overrides)
    if ("early_stopping_validation_max_n" %in% names(mcmc_overrides) &&
        is.null(mcmc_overrides$early_stopping_validation_max_n)) {
      mcmc_control$early_stopping_validation_max_n <- NULL
    }
    if ("early_stopping_validation_batch_size" %in% names(mcmc_overrides) &&
        is.null(mcmc_overrides$early_stopping_validation_batch_size)) {
      mcmc_control$early_stopping_validation_batch_size <- NULL
    }
  }
  if (isTRUE(neural_oos_eval_internal_flag)) {
    mcmc_control$checkpoint_path <- NULL
    mcmc_control$checkpoint_resume <- FALSE
  }
  if (isTRUE(save_outcome_model) &&
      !isTRUE(neural_oos_eval_internal_flag) &&
      is.null(mcmc_control$checkpoint_path)) {
    mcmc_control$checkpoint_path <- paste0(bundle_path, ".inprogress")
  }
  mcmc_control$RMS_scale <- neural_resolve_rms_scale(mcmc_control$RMS_scale)
  mcmc_control$qk_rms_scale <- neural_resolve_qk_rms_scale(mcmc_control$qk_rms_scale)
  mcmc_control$moe_load_balance_lambda <- neural_resolve_moe_load_balance_lambda(
    mcmc_control$moe_load_balance_lambda
  )
  mcmc_control$universal_family_logprob_normalize <- isTRUE(
    mcmc_control$universal_family_logprob_normalize
  )
  mcmc_control$svi_lr_plate_rescale <- isTRUE(mcmc_control$svi_lr_plate_rescale %||% TRUE)
  mcmc_control$residual_weight_depth_scale <- neural_resolve_residual_weight_depth_scale(
    mcmc_control$residual_weight_depth_scale
  )
  training_seed_supplied <- !is.null(mcmc_overrides) &&
    "seed" %in% names(mcmc_overrides) &&
    !is.null(mcmc_overrides$seed)
  training_seed_info <- neural_resolve_training_seed(
    mcmc_control,
    supplied = training_seed_supplied
  )
  mcmc_control$seed <- training_seed_info$seed
  neural_training_seed <- training_seed_info$seed
  neural_training_seed_supplied <- isTRUE(training_seed_info$supplied)
  neural_runtime_info <- neural_runtime_provenance(
    mcmc_control = mcmc_control,
    conda_env = conda_env
  )
  neural_outcome_fingerprint <- cs2step_neural_outcome_request_fingerprint(
    Y = Y_,
    W = W_,
    X = if (exists("X_", inherits = TRUE)) X_ else NULL,
    X_present = if (exists("X_present_", inherits = TRUE)) X_present_ else NULL,
    names_list = if (exists("names_list", inherits = TRUE)) names_list else NULL,
    factor_levels = factor_levels,
    diff = diff,
    pair_id = pair_id_ %||% NULL,
    profile_order = profile_order_ %||% NULL,
    p_list = if (exists("p_list", inherits = TRUE)) p_list else NULL,
    competing_group_variable_candidate = competing_group_variable_candidate_ %||% NULL,
    competing_group_variable_respondent = competing_group_variable_respondent_ %||% NULL,
    respondent_id = if (exists("respondent_id", inherits = TRUE)) respondent_id else NULL,
    respondent_task_id = if (exists("respondent_task_id", inherits = TRUE)) respondent_task_id else NULL,
    outcome_model_key = outcome_model_key,
    group = GroupsPool[GroupCounter],
    round = Round_,
    adversarial = adversarial,
    adversarial_model_strategy = adversarial_model_strategy,
    mcmc_control = mcmc_control,
    neural_token_info = NULL
  )

  if (!isTRUE(neural_oos_eval_internal_flag) &&
      isTRUE(presaved_outcome_model) && file.exists(bundle_path)) {
    bundle_raw <- tryCatch(readRDS(bundle_path), error = function(e) NULL)
    if (is.list(bundle_raw) && !is.null(bundle_raw$fit) &&
        !is.null(bundle_raw$fit$theta_mean) &&
        !is.null(bundle_raw$fit$neural_model_info)) {
      cs2step_artifact_assert_fingerprint(
        stored = bundle_raw$metadata$fingerprint %||% NULL,
        expected = neural_outcome_fingerprint,
        context = "Cached neural outcome bundle"
      )
      cached_predictor <- load_neural_outcome_bundle(
        bundle_path,
        conda_env = conda_env,
        conda_env_required = conda_env_required,
        preload_params = FALSE
      )
      message(sprintf("Loading cached neural outcome bundle: %s", bundle_path))

      for(nrp in 1:2){
        main_info <- do.call(rbind, sapply(1:length(factor_levels), function(d_){
          list(data.frame(
            "d" = d_,
            "l" = 1:max(1, factor_levels[d_] - ifelse(nrp == 1, yes = 1, no = holdout_indicator))
          ))
        }))
        main_info <- cbind(main_info, "d_index" = 1:nrow(main_info))
        if(nrp == 1){ a_structure <- main_info }
      }
      if(holdout_indicator == 0){
        a_structure_leftoutLdminus1 <- main_info[which(c(base::diff(main_info$d),1)==0),]
        a_structure_leftoutLdminus1$d_index <- 1:nrow(a_structure_leftoutLdminus1)
      }
      interaction_info <- data.frame()
      interaction_info_PreRegularization <- interaction_info
      regularization_adjust_hash <- main_info$d
      names(regularization_adjust_hash) <- main_info$d
      main_dat <- matrix(0, nrow = 0L, ncol = 0L)

      theta_mean_num <- as.numeric(cached_predictor$fit$theta_mean)
      theta_var_num <- cached_predictor$fit$theta_var
      my_mean <- numeric(0)
      my_mean_full <- NULL
      vcov_OutcomeModel_by_k <- NULL
      vcov_OutcomeModel <- if (!is.null(theta_var_num)) {
        c(0, as.numeric(theta_var_num))
      } else {
        c(0, rep(0, length(theta_mean_num)))
      }

      EST_INTERCEPT_tf <- strenv$jnp$array(matrix(0, nrow = 1L, ncol = 1L), dtype = strenv$dtj)
      EST_COEFFICIENTS_tf <- strenv$jnp$reshape(
        strenv$jnp$array(theta_mean_num, dtype = strenv$dtj),
        list(-1L, 1L)
      )

      if (exists("K", inherits = TRUE) && is.numeric(K) && K > 1) {
        base_vec <- c(0, theta_mean_num)
        my_mean_full <- matrix(rep(base_vec, K), ncol = K)
        vcov_OutcomeModel_by_k <- replicate(K, vcov_OutcomeModel, simplify = FALSE)
      }

      neural_model_info <- cached_predictor$fit$neural_model_info
      neural_validate_full_attn_compatibility(
        model_info = neural_model_info,
        context = "Cached neural outcome bundle"
      )
      fit_metrics <- cached_predictor$fit$fit_metrics %||% neural_model_info$fit_metrics
      my_model <- NULL

      return(invisible(NULL))
    }
  }
  user_supplied_svi_steps <- !is.null(mcmc_overrides) && !is.null(mcmc_overrides$svi_steps)
  user_supplied_svi_num_draws <- !is.null(mcmc_overrides) &&
    !is.null(mcmc_overrides$svi_num_draws)
  skip_eval_flag <- tolower(Sys.getenv("STRATEGIZE_NEURAL_SKIP_EVAL")) %in%
    c("1", "true", "yes")
  if (isTRUE(skip_eval_flag)) {
    eval_control$enabled <- FALSE
  }
  eval_max_env <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_NEURAL_EVAL_MAX")))
  if (!is.na(eval_max_env) && eval_max_env > 0L) {
    eval_control$max_n <- eval_max_env
  }
  eval_seed_env <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_NEURAL_EVAL_SEED")))
  if (!is.na(eval_seed_env) && eval_seed_env > 0L) {
    eval_control$seed <- eval_seed_env
  }
  eval_folds_env <- suppressWarnings(as.integer(Sys.getenv("STRATEGIZE_NEURAL_EVAL_FOLDS")))
  if (!is.na(eval_folds_env) && eval_folds_env > 0L) {
    eval_control$n_folds <- eval_folds_env
  }
  if (is.null(eval_control$n_folds) &&
      exists("nFolds_glm", inherits = TRUE) &&
      is.numeric(nFolds_glm) &&
      length(nFolds_glm) == 1L &&
      is.finite(nFolds_glm)) {
    eval_control$n_folds <- as.integer(nFolds_glm)
  }
  if (!uncertainty_scope %in% c("all", "output")) {
    stop("'neural_mcmc_control$uncertainty_scope' must be 'all' or 'output'.",
         call. = FALSE)
  }
  subsample_method <- if (!is.null(mcmc_control$subsample_method)) {
    tolower(as.character(mcmc_control$subsample_method))
  } else {
    "full"
  }
  if (length(subsample_method) != 1L || is.na(subsample_method) || !nzchar(subsample_method)) {
    subsample_method <- "full"
  }
  mcmc_control$subsample_method <- subsample_method
  W_idx_compact_use <- if (exists("W_idx_compact", inherits = TRUE) &&
                           !is.null(W_idx_compact)) {
    get("W_idx_compact", inherits = TRUE)
  } else {
    NULL
  }
  X_compact_use <- if (exists("X_compact", inherits = TRUE) &&
                       !is.null(X_compact)) {
    get("X_compact", inherits = TRUE)
  } else {
    NULL
  }
  X_present_compact_use <- if (exists("X_present_compact", inherits = TRUE) &&
                               !is.null(X_present_compact)) {
    get("X_present_compact", inherits = TRUE)
  } else {
    NULL
  }
  compact_training <- !is.null(W_idx_compact_use) || !is.null(X_compact_use)
  subsample_method_model <- if (isTRUE(compact_training)) "full" else subsample_method
  compact_update_chunk_size <- if (isTRUE(compact_training)) {
    chunk_size <- suppressWarnings(as.integer(mcmc_control$compact_update_chunk_size))
    if (length(chunk_size) != 1L || is.na(chunk_size) || chunk_size < 1L) {
      stop(
        "'neural_mcmc_control$compact_update_chunk_size' must be an integer >= 1.",
        call. = FALSE
      )
    }
    chunk_size
  } else {
    1L
  }
  mcmc_control$compact_update_chunk_size <- as.integer(compact_update_chunk_size)
  compact_update_scan <- neural_resolve_compact_update_scan(
    compact_training = compact_training,
    compact_update_chunk_size = compact_update_chunk_size,
    compact_update_scan = mcmc_control$compact_update_scan %||% NULL
  )
  mcmc_control$compact_update_scan <- compact_update_scan
  if (isTRUE(compact_training)) {
    if (is.null(W_idx_compact_use)) {
      stop("Compact neural training requires W_idx_compact.", call. = FALSE)
    }
    if (!identical(subsample_method, "batch_vi")) {
      stop(
        "Compact neural training currently requires neural_mcmc_control$subsample_method = 'batch_vi'.",
        call. = FALSE
      )
    }
    if (isTRUE(eval_control$enabled)) {
      message("Disabling neural OOS evaluation for compact streaming training.")
      eval_control$enabled <- FALSE
    }
  }

  if (!is.numeric(model_dims) || length(model_dims) != 1L || !is.finite(model_dims)) {
    stop("'neural_mcmc_control$ModelDims' must be a single finite numeric value.",
         call. = FALSE)
  }
  if (model_dims != round(model_dims) || model_dims < 1L) {
    stop("'neural_mcmc_control$ModelDims' must be an integer >= 1.",
         call. = FALSE)
  }
  if (!is.numeric(model_depth) || length(model_depth) != 1L || !is.finite(model_depth)) {
    stop("'neural_mcmc_control$ModelDepth' must be a single finite numeric value.",
         call. = FALSE)
  }
  if (model_depth != round(model_depth) || model_depth < 1L) {
    stop("'neural_mcmc_control$ModelDepth' must be an integer >= 1.",
         call. = FALSE)
  }
  # Hyperparameters
  ModelDims  <- ai(model_dims)
  ModelDepth <- ai(model_depth)
  WideMultiplicationFactor <- 3.75
  MD_int <- ai(ModelDims)
  cand_heads <- (1:MD_int)[(MD_int %% (1:MD_int)) == 0L]
  TransformerHeads <- ai(cand_heads[which.min(abs(cand_heads - 8L))])
  head_dim <- ai(ai(MD_int / TransformerHeads))
  attention_config_probe <- list(
    attention_backend = attention_backend,
    attention_dtype = attention_dtype,
    attention_padding_multiple = attention_padding_multiple
  )
  attention_resolve <- neural_attention_resolve_backend(
    attention_config_probe,
    role = "self",
    fail_on_forced = TRUE
  )
  attention_resolved_backend <- attention_resolve$backend
  attention_fallback_reason <- attention_resolve$fallback_reason
  attention_dtype_label <- neural_attention_dtype_object(
    attention_dtype,
    prefer_cudnn = identical(attention_resolved_backend, "cudnn")
  )$label
  message(sprintf(
    paste0(
      "Neural attention backend: requested=%s; resolved=%s; dtype=%s; ",
      "jax_backend=%s; cuda_available=%s; padding_multiple=%s; fallback=%s."
    ),
    attention_backend,
    attention_resolved_backend,
    attention_dtype_label,
    neural_attention_jax_backend(),
    if (isTRUE(attention_resolve$cuda_available)) "TRUE" else "FALSE",
    as.integer(attention_padding_multiple),
    if (is.na(attention_fallback_reason) || !nzchar(attention_fallback_reason)) {
      "none"
    } else {
      attention_fallback_reason
    }
  ))
  FFDim <- ai(ai(round(MD_int * WideMultiplicationFactor)))
  init_policy <- neural_resolve_init_policy(
    model_depth = ModelDepth,
    model_dims = ModelDims,
    RMS_scale = mcmc_control$RMS_scale,
    residual_weight_depth_scale = mcmc_control$residual_weight_depth_scale,
    qk_rms_scale = mcmc_control$qk_rms_scale
  )
  RMS_scale <- init_policy$RMS_scale
  # QK-norm gets its own (tighter) prior scale, distinct from the general RMS_scale
  # (Issue 6): a loose QK-norm scale lets attention logits amplify at init, causing
  # early attention-sink behavior and sparse gradients through the softmax.
  qk_rms_scale <- init_policy$qk_rms_scale
  # Per-family log-prob normalization (Issue 3): when enabled, each likelihood
  # family's contribution to the mixed-task log-prob is divided by an in-batch scale
  # so a small-sigma Normal head cannot dominate the shared trunk's gradient. Off by
  # default to preserve existing (empirical) behavior.
  universal_family_logprob_normalize <- isTRUE(mcmc_control$universal_family_logprob_normalize)
  # MoE covariate-value encoder load balancing (Issue 5): a Switch-style auxiliary
  # penalty keeps routing mass off a single expert. The penalty is emitted as a
  # numpyro.factor from the encoder only while `moe_aux_active` is TRUE (set around
  # SVI training below); the counter yields a unique factor-site name per encoder
  # invocation within a single model trace, avoiding duplicate-site collisions.
  moe_load_balance_lambda <- neural_resolve_moe_load_balance_lambda(mcmc_control$moe_load_balance_lambda)
  strenv$moe_load_balance_lambda <- moe_load_balance_lambda
  strenv$moe_aux_active <- FALSE
  strenv$moe_aux_counter <- 0L
  strenv$moe_aux_pending <- list()
  weight_sd_scale <- init_policy$weight_sd_scale
  #weight_sd_scale <- sqrt(2 * log(1 + ModelDims/2))/sqrt(ModelDims)

  # Depth-aware scaling for residual priors and ReZero-style residual gates.
  depth_prior_scale <- init_policy$depth_prior_scale
  residual_weight_sd_scale <- init_policy$residual_weight_sd_scale
  gate_sd_scale <- init_policy$gate_sd_scale
  embed_sd_scale <- 4 * weight_sd_scale
  context_embed_sd_scale <- embed_sd_scale
  choice_embed_sd_scale <- 0.25 * context_embed_sd_scale
  sep_embed_sd_scale <- 0.10 * context_embed_sd_scale
  segment_embed_sd_scale <- 0.10 * context_embed_sd_scale
  tau_b_scale <- 0.5
  
  # shrink M_cross more (initialize interactionto smaller than main temr)
  # cross_out does NOT need to shrink with ModelDims (doesn't scale with that)
  cross_weight_sd_scale <- weight_sd_scale / sqrt(as.numeric(ModelDims))
  cross_out_sd_scale    <- 0.5  # or 0.25 if you want it more conservative

  # Pairwise mode for forced-choice
  pairwise_mode <- isTRUE(diff) && !is.null(pair_id_) && length(pair_id_) > 0
  pairwise_antisymmetry_mode <- if (isTRUE(pairwise_mode)) {
    neural_resolve_pairwise_antisymmetry(pairwise_antisymmetry_control)
  } else {
    "legacy"
  }
  mcmc_control$pairwise_antisymmetry <- pairwise_antisymmetry_mode
  if (!isTRUE(pairwise_mode)) {
    cross_candidate_encoder_mode <- "none"
  }
  learned_pairwise_bernoulli_logit_scale_requested <-
    neural_resolve_learned_pairwise_bernoulli_logit_scale(
      mcmc_control$learned_pairwise_bernoulli_logit_scale %||% FALSE
    )
  use_full_attn_residual <- identical(residual_mode, "full_attn")
  use_matchup_token <- FALSE

  # Main-info structure for downstream compatibility
  for(nrp in 1:2){
    main_info <- do.call(rbind, sapply(1:length(factor_levels), function(d_){
      list(data.frame(
        "d" = d_,
        "l" = 1:max(1, factor_levels[d_] - ifelse(nrp == 1, yes = 1, no = holdout_indicator))
      ))
    }))
    main_info <- cbind(main_info, "d_index" = 1:nrow(main_info))
    if(nrp == 1){ a_structure <- main_info }
  }
  if(holdout_indicator == 0){
    a_structure_leftoutLdminus1 <- main_info[which(c(base::diff(main_info$d),1)==0),]
    a_structure_leftoutLdminus1$d_index <- 1:nrow(a_structure_leftoutLdminus1)
  }

  interaction_info <- data.frame()
  interaction_info_PreRegularization <- interaction_info
  regularization_adjust_hash <- main_info$d
  names(regularization_adjust_hash) <- main_info$d

  factor_levels_int <- as.integer(factor_levels)
  cs2step_validate_w_idx_compact(W_idx_compact_use, factor_levels_int)
  factor_index_list <- vector("list", length(factor_levels))
  offset <- 0L
  for (d_ in seq_along(factor_levels)) {
    n_levels_use <- ai(factor_levels[d_] - holdout_indicator)
    idx <- if (n_levels_use > 0L) {
      as.integer(offset + seq_len(n_levels_use) - 1L)
    } else {
      integer(0)
    }
    factor_index_list[[d_]] <- strenv$jnp$array(idx)$astype(strenv$jnp$int32)
    offset <- offset + n_levels_use
  }
  n_rel_levels <- ai(3L)

  neural_token_info_use <- if (exists("neural_token_info", inherits = TRUE) &&
                               !is.null(neural_token_info)) {
    neural_token_info
  } else {
    list()
  }
  neural_token_info_use <- neural_resolve_token_runtime_config(
    neural_token_info = neural_token_info_use,
    mcmc_control = mcmc_control
  )
  group_context_schema <- neural_token_info_use$group_context_schema %||% list()
  group_context_tokenization <- tolower(as.character(
    group_context_schema$tokenization %||%
      neural_token_info_use$group_context_tokenization %||%
      "auto"
  ))
  unified_group_context <- isTRUE(group_context_schema$enabled) ||
    group_context_tokenization %in% c(
      "unified",
      "unified_role_embedding",
      "unified_role_embeddings"
    )

  # Candidate/respondent group context
  party_missing_label <- if (exists("party_missing_label_fixed", inherits = TRUE)) {
    as.character(get("party_missing_label_fixed", inherits = TRUE))
  } else if (!is.null(group_context_schema$candidate_missing_label)) {
    as.character(group_context_schema$candidate_missing_label)
  } else if (!is.null(neural_token_info_use$candidate_group_missing_label)) {
    as.character(neural_token_info_use$candidate_group_missing_label)
  } else {
    neural_missing_group_label("candidate")
  }
  resp_party_missing_label <- if (exists("resp_party_missing_label_fixed", inherits = TRUE)) {
    as.character(get("resp_party_missing_label_fixed", inherits = TRUE))
  } else if (!is.null(group_context_schema$respondent_missing_label)) {
    as.character(group_context_schema$respondent_missing_label)
  } else if (!is.null(neural_token_info_use$respondent_group_missing_label)) {
    as.character(neural_token_info_use$respondent_group_missing_label)
  } else {
    neural_missing_group_label("respondent")
  }
  candidate_group_input <- if (!is.null(competing_group_variable_candidate_)) {
    as.character(competing_group_variable_candidate_)
  } else {
    NULL
  }
  respondent_group_input <- if (!is.null(competing_group_variable_respondent_)) {
    as.character(competing_group_variable_respondent_)
  } else {
    NULL
  }
  has_candidate_group_context <- !is.null(candidate_group_input) &&
    any(!is.na(candidate_group_input) & nzchar(candidate_group_input))
  has_respondent_group_context <- !is.null(respondent_group_input) &&
    any(!is.na(respondent_group_input) & nzchar(respondent_group_input))
  has_candidate_group_context <- isTRUE(has_candidate_group_context) ||
    isTRUE(unified_group_context) ||
    isTRUE(group_context_schema$include_candidate_group)
  has_respondent_group_context <- isTRUE(has_respondent_group_context) ||
    isTRUE(unified_group_context) ||
    isTRUE(group_context_schema$include_respondent_group)
  has_relation_context <- (isTRUE(has_candidate_group_context) &&
    isTRUE(has_respondent_group_context)) ||
    isTRUE(group_context_schema$include_relation)

  force_stage_context <- isTRUE(unified_group_context) ||
    isTRUE(group_context_schema$include_stage)
  force_matchup_context <- isTRUE(unified_group_context) ||
    isTRUE(group_context_schema$include_matchup)

  has_relation_context <- isTRUE(has_relation_context) &&
    isTRUE(has_candidate_group_context) &&
    isTRUE(has_respondent_group_context)

  # Candidate group mapping
  party_levels_override <- NULL
  if (exists("party_levels_fixed", inherits = TRUE)) {
    party_levels_override <- get("party_levels_fixed", inherits = TRUE)
  } else if (!is.null(group_context_schema$candidate_group_levels)) {
    party_levels_override <- group_context_schema$candidate_group_levels
  } else if (!is.null(neural_token_info_use$candidate_group_levels)) {
    party_levels_override <- neural_token_info_use$candidate_group_levels
  }
  party_levels <- neural_prepare_group_levels(
    values = candidate_group_input,
    override = party_levels_override,
    missing_label = party_missing_label
  )
  n_party_levels <- max(1L, length(party_levels))
  party_index <- neural_coerce_group_index_base(
    values = candidate_group_input,
    n_rows = length(Y_),
    levels = party_levels,
    missing_label = party_missing_label
  )
  party_missing_index <- neural_missing_group_index(party_levels, party_missing_label)

  # Respondent group mapping
  resp_party_levels_override <- NULL
  if (exists("resp_party_levels_fixed", inherits = TRUE)) {
    resp_party_levels_override <- get("resp_party_levels_fixed", inherits = TRUE)
  } else if (!is.null(group_context_schema$respondent_group_levels)) {
    resp_party_levels_override <- group_context_schema$respondent_group_levels
  } else if (!is.null(neural_token_info_use$respondent_group_levels)) {
    resp_party_levels_override <- neural_token_info_use$respondent_group_levels
  }
  resp_party_levels <- neural_prepare_group_levels(
    values = respondent_group_input,
    override = resp_party_levels_override,
    missing_label = resp_party_missing_label
  )
  n_resp_party_levels <- max(1L, length(resp_party_levels))
  resp_party_index <- neural_coerce_group_index_base(
    values = respondent_group_input,
    n_rows = length(Y_),
    levels = resp_party_levels,
    missing_label = resp_party_missing_label
  )
  resp_party_missing_index <- neural_missing_group_index(
    resp_party_levels,
    resp_party_missing_label
  )

  cand_party_to_resp_idx <- vapply(party_levels, function(party_label) {
    if (!isTRUE(has_relation_context) ||
        identical(as.character(party_label), party_missing_label)) {
      return(-1L)
    }
    idx <- match(as.character(party_label), resp_party_levels)
    if (is.na(idx)) {
      resp_party_missing_index
    } else {
      as.integer(idx - 1L)
    }
  }, integer(1))
  cand_party_to_resp_idx_jnp <- strenv$jnp$array(as.integer(cand_party_to_resp_idx))
  cand_party_to_resp_idx_jnp <- strenv$jnp$atleast_1d(cand_party_to_resp_idx_jnp)$astype(strenv$jnp$int32)

  # Respondent covariates (optional)
  X_use <- NULL
  X_ <- NULL
  if (!isTRUE(compact_training) && exists("X", inherits = TRUE) && !is.null(X)) {
    X_ <- as.matrix(X[indi_, , drop = FALSE])
  }
  X_present_ <- NULL
  if (!isTRUE(compact_training) && exists("X_present", inherits = TRUE) && !is.null(X_present)) {
    X_present_ <- as.matrix(X_present[indi_, , drop = FALSE])
  } else if (!is.null(X_)) {
    X_present_ <- matrix(1, nrow = nrow(X_), ncol = ncol(X_))
    colnames(X_present_) <- colnames(X_)
  }
  token_family_levels <- NULL
  factor_tokenization <- neural_factor_tokenization(
    mode = neural_token_info_use$factor_tokenization %||% "fused"
  )
  additive_utility_mode <- neural_resolve_additive_utility_mode(
    additive_utility_control,
    factor_tokenization = factor_tokenization,
    pairwise_mode = pairwise_mode
  )
  additive_utility_normalization <- neural_resolve_additive_utility_normalization(
    value = additive_utility_normalization_control,
    additive_utility_mode = additive_utility_mode,
    supplied = additive_utility_normalization_supplied
  )
  additive_head_weight_target_rms <- neural_resolve_additive_head_weight_target_rms(
    value = additive_head_weight_target_rms_control,
    model_dims = ModelDims,
    normalization = additive_utility_normalization,
    supplied = additive_head_weight_target_rms_supplied
  )
  mcmc_control$additive_utility <- additive_utility_mode
  mcmc_control$additive_utility_normalization <- additive_utility_normalization
  mcmc_control$additive_head_weight_target_rms <- additive_head_weight_target_rms
  factor_order_by_experiment <- lapply(
    neural_token_info_use$factor_order_by_experiment %||% list(),
    as.integer
  )
  default_factor_order <- as.integer(
    neural_token_info_use$default_factor_order %||% integer(0)
  )
  max_factor_tokens <- as.integer(neural_token_info_use$max_factor_tokens)
  max_covariate_tokens <- as.integer(neural_token_info_use$max_covariate_tokens)
  factor_name_text <- neural_token_info_use$factor_name_text %||% NULL
  level_name_text <- neural_token_info_use$level_name_text %||% NULL
  factor_struct_matrix <- neural_token_info_use$factor_struct_matrix %||% NULL
  level_struct_matrices <- neural_token_info_use$level_struct_matrices %||% NULL
  factor_struct_feature_names <- as.character(
    neural_token_info_use$factor_struct_feature_names %||% character(0)
  )
  level_struct_feature_names <- as.character(
    neural_token_info_use$level_struct_feature_names %||% character(0)
  )
  factor_schema_supplied <- neural_factor_schema_supplied(
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    factor_struct_matrix = factor_struct_matrix,
    level_struct_matrices = level_struct_matrices,
    default_factor_order = default_factor_order,
    factor_order_by_experiment = factor_order_by_experiment
  )
  if (identical(factor_tokenization, "fused") && !isTRUE(factor_schema_supplied)) {
    default_structural_info <- neural_make_default_fused_structural_info(
      names_list = if (exists("names_list", inherits = TRUE)) names_list else NULL,
      factor_levels = factor_levels_int
    )
    factor_struct_matrix <- default_structural_info$factor_struct_matrix
    level_struct_matrices <- default_structural_info$level_struct_matrices
    factor_struct_feature_names <- default_structural_info$factor_struct_feature_names
    level_struct_feature_names <- default_structural_info$level_struct_feature_names
    neural_token_info_use$factor_struct_matrix <- factor_struct_matrix
    neural_token_info_use$level_struct_matrices <- level_struct_matrices
    neural_token_info_use$factor_struct_feature_names <- factor_struct_feature_names
    neural_token_info_use$level_struct_feature_names <- level_struct_feature_names
  }
  neural_validate_fused_structural_info(
    factor_struct_matrix = factor_struct_matrix,
    level_struct_matrices = level_struct_matrices,
    factor_struct_feature_names = factor_struct_feature_names,
    level_struct_feature_names = level_struct_feature_names,
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    context = "neural_token_info fused"
  )
  covariate_name_text <- neural_token_info_use$covariate_name_text %||% NULL
  covariate_value_text <- neural_token_info_use$covariate_value_text %||% NULL
  covariate_value_text_present <- neural_token_info_use$covariate_value_text_present %||% NULL
  covariate_value_type <- neural_token_info_use$covariate_value_type %||% NULL
  covariate_order_by_experiment <- lapply(
    neural_token_info_use$covariate_order_by_experiment %||% list(),
    as.integer
  )
  default_covariate_order <- as.integer(
    neural_token_info_use$default_covariate_order %||% integer(0)
  )
  experiment_description_text <- neural_token_info_use$experiment_description_text %||% NULL
  experiment_description_present <- neural_token_info_use$experiment_description_present %||% NULL
  default_experiment_text <- neural_token_info_use$default_experiment_text %||% NULL
  default_experiment_text_present <- isTRUE(
    neural_token_info_use$default_experiment_text_present %||% FALSE
  )
  place_feature_names <- as.character(
    neural_token_info_use$place_feature_names %||% neural_place_feature_names()
  )
  place_embedding <- neural_token_info_use$place_embedding %||% NULL
  place_present <- neural_token_info_use$place_present %||% NULL
  default_place_embedding <- neural_token_info_use$default_place_embedding %||% NULL
  default_place_present <- isTRUE(neural_token_info_use$default_place_present %||% FALSE)
  place_context_enabled <- isTRUE(neural_token_info_use$place_context_enabled %||% FALSE)
  if (isTRUE(place_context_enabled)) {
    if (is.null(place_embedding) && length(experiment_levels_override <- as.character(
      neural_token_info_use$experiment_levels %||% character(0)
    )) > 0L) {
      place_embedding <- matrix(
        rep(neural_encode_place_context(NA_real_, NA_real_, present = FALSE), times = length(experiment_levels_override)),
        nrow = length(experiment_levels_override),
        byrow = TRUE,
        dimnames = list(experiment_levels_override, place_feature_names)
      )
    }
    if (is.null(default_place_embedding)) {
      default_place_embedding <- neural_default_place_context_matrix()
    }
  }
  time_feature_names <- as.character(
    neural_token_info_use$time_feature_names %||% neural_time_feature_names()
  )
  time_embedding <- neural_token_info_use$time_embedding %||% NULL
  time_present <- neural_token_info_use$time_present %||% NULL
  default_time_embedding <- neural_token_info_use$default_time_embedding %||% NULL
  default_time_present <- isTRUE(neural_token_info_use$default_time_present %||% FALSE)
  time_context_enabled <- isTRUE(neural_token_info_use$time_context_enabled %||% FALSE)
  if (isTRUE(time_context_enabled)) {
    if (is.null(time_embedding) && length(experiment_levels_override <- as.character(
      neural_token_info_use$experiment_levels %||% character(0)
    )) > 0L) {
      time_embedding <- matrix(
        rep(neural_encode_time_context(NA_real_, present = FALSE), times = length(experiment_levels_override)),
        nrow = length(experiment_levels_override),
        byrow = TRUE,
        dimnames = list(experiment_levels_override, time_feature_names)
      )
    }
    if (is.null(default_time_embedding)) {
      default_time_embedding <- neural_default_time_context_matrix()
    }
  }
  context_present_masking <- isTRUE(neural_token_info_use$context_present_masking %||% TRUE)
  experiment_token_mode <- tolower(as.character(
    neural_token_info_use$experiment_token_mode %||% "legacy_id"
  ))
  covariate_value_encoding <- neural_resolve_covariate_value_encoding(
    neural_token_info_use$covariate_value_encoding %||% "shared_projection"
  )
  shared_projection_value_encoder <- neural_shared_projection_value_encoder(
    mode = neural_token_info_use$shared_projection_value_encoder %||%
      mcmc_control$shared_projection_value_encoder %||%
      "name_dist_moe"
  )
  compact_covariate_names <- as.character(
    X_compact_use$covariate_names %||%
      X_present_compact_use$covariate_names %||%
      character(0)
  )
  covariate_names_override <- as.character(
    neural_token_info_use$covariate_names %||%
      colnames(X_) %||%
      compact_covariate_names %||%
      character(0)
  )
  neural_token_info_use$covariate_names <- covariate_names_override
  if (length(default_factor_order) < 1L && length(factor_levels_int) > 0L) {
    default_factor_order <- seq.int(0L, length(factor_levels_int) - 1L)
  }
  if (length(default_covariate_order) < 1L && length(covariate_names_override) > 0L) {
    default_covariate_order <- seq.int(0L, length(covariate_names_override) - 1L)
  }
  neural_token_info_use$default_factor_order <- default_factor_order
  neural_token_info_use$default_covariate_order <- default_covariate_order
  experiment_levels_override <- as.character(
    neural_token_info_use$experiment_levels %||% character(0)
  )
  experiment_index_all <- neural_token_info_use$experiment_index %||% NULL
  if (!is.null(experiment_index_all)) {
    experiment_index_all <- as.integer(experiment_index_all)
    if (length(experiment_index_all) != length(Y_) || all(is.na(experiment_index_all))) {
      experiment_index_all <- NULL
    }
  }
  if (!is.null(experiment_index_all)) {
    # The contract is 0-based codes into experiment_levels. Downstream gathers
    # clamp out-of-range values, so a 1-based caller would silently shift every
    # experiment's embedding and per-experiment ordinal thresholds -- assert
    # the range here instead.
    exp_idx_finite <- experiment_index_all[!is.na(experiment_index_all)]
    if (length(exp_idx_finite) > 0L && any(exp_idx_finite < 0L)) {
      stop(
        "neural_token_info$experiment_index must contain 0-based experiment codes; ",
        "found negative values.",
        call. = FALSE
      )
    }
    if (length(experiment_levels_override) > 0L &&
        length(exp_idx_finite) > 0L &&
        any(exp_idx_finite >= length(experiment_levels_override))) {
      stop(sprintf(paste0(
        "neural_token_info$experiment_index contains code(s) >= %d, the number ",
        "of experiment levels; expected 0-based codes in [0, %d]. A 1-based ",
        "index would silently shift every experiment embedding."
      ), length(experiment_levels_override),
      length(experiment_levels_override) - 1L), call. = FALSE)
    }
  }
  respondent_id_all <- if (exists("respondent_id", inherits = TRUE) &&
                           !is.null(respondent_id)) {
    as.character(respondent_id)
  } else {
    NULL
  }
  if (!is.null(respondent_id_all) &&
      length(respondent_id_all) != length(Y_)) {
    respondent_id_all <- NULL
  }
  respondent_task_id_all <- if (exists("respondent_task_id", inherits = TRUE) &&
                                !is.null(respondent_task_id)) {
    as.character(respondent_task_id)
  } else {
    NULL
  }
  if (!is.null(respondent_task_id_all) &&
      length(respondent_task_id_all) != length(Y_)) {
    respondent_task_id_all <- NULL
  }
  default_experiment_index <- if (!is.null(neural_token_info_use$default_experiment_index) &&
                                  !is.na(neural_token_info_use$default_experiment_index)) {
    as.integer(neural_token_info_use$default_experiment_index)
  } else {
    NA_integer_
  }
  schema_dropout <- neural_resolve_schema_dropout(
    neural_token_info_use$schema_dropout %||% NULL
  )
  low_rank_interaction_rank <- neural_resolve_low_rank_interaction_rank(
    neural_token_info_use$low_rank_interaction_rank %||%
      mcmc_control$low_rank_interaction_rank %||%
      mcmc_control$respondent_candidate_interaction_rank %||%
      0L
  )
  neural_token_info_use$low_rank_interaction_rank <- low_rank_interaction_rank
  low_rank_logit_transform <- if (isTRUE(low_rank_logit_transform_supplied)) {
    neural_normalize_low_rank_logit_transform(low_rank_logit_transform_control)
  } else {
    "none"
  }
  if (length(low_rank_logit_transform) != 1L ||
      is.na(low_rank_logit_transform) ||
      !low_rank_logit_transform %in% c("softclip", "none")) {
    stop(
      "'neural_mcmc_control$low_rank_logit_transform' must be 'softclip' or 'none'.",
      call. = FALSE
    )
  }
  low_rank_logit_bound <- neural_resolve_low_rank_logit_bound(
    value = low_rank_logit_bound_control,
    low_rank_interaction_rank = low_rank_interaction_rank,
    supplied = low_rank_logit_bound_supplied,
    transform = low_rank_logit_transform
  )
  low_rank_logit_softness <- neural_resolve_low_rank_logit_softness(
    value = low_rank_logit_softness_control,
    bound = low_rank_logit_bound,
    supplied = low_rank_logit_softness_supplied
  )
  if (is.null(low_rank_logit_bound)) {
    low_rank_logit_transform <- "none"
    low_rank_logit_softness <- NULL
  }
  low_rank_logit_model_info <- list(
    likelihood = "bernoulli",
    low_rank_interaction_rank = low_rank_interaction_rank,
    low_rank_logit_transform = low_rank_logit_transform,
    low_rank_logit_bound = low_rank_logit_bound,
    low_rank_logit_softness = low_rank_logit_softness,
    low_rank_logit_normalization = low_rank_logit_normalization,
    low_rank_head_weight_target_rms = low_rank_head_weight_target_rms,
    low_rank_rc_out_target_rms = low_rank_rc_out_target_rms,
    pairwise_antisymmetry = pairwise_antisymmetry_mode,
    additive_utility_mode = additive_utility_mode,
    additive_utility_normalization = additive_utility_normalization,
    additive_head_weight_target_rms = additive_head_weight_target_rms,
    calibration_enabled = isTRUE(calibration_control$enabled),
    calibration_method = calibration_control$method %||% "none",
    calibration_scale = NULL
  )
  universal_training <- neural_token_info_use$foundation_universal_training %||% NULL
  universal_enabled <- isTRUE(universal_training$enabled)
  universal_task_mode_all <- as.character(universal_training$task_mode_by_row %||% character(0))
  universal_likelihood_all <- tolower(as.character(
    universal_training$likelihood_by_row %||% character(0)
  ))
  universal_n_outcomes_all <- as.integer(
    universal_training$n_outcomes_by_row %||% integer(0)
  )
  universal_global_out_dim <- as.integer(universal_training$global_out_dim %||% 1L)
  if (isTRUE(universal_enabled)) {
    if (length(universal_task_mode_all) != length(Y_) ||
        length(universal_likelihood_all) != length(Y_) ||
        length(universal_n_outcomes_all) != length(Y_)) {
      stop(
        "foundation_universal_training metadata must align with the training rows passed to the neural backend.",
        call. = FALSE
      )
    }
    if (length(universal_likelihood_all) < 1L ||
        !all(universal_likelihood_all %in% c("bernoulli", "categorical", "normal", "ordinal"))) {
      stop(
        "foundation_universal_training$likelihood_by_row must only contain 'bernoulli', 'categorical', 'normal', or 'ordinal'.",
        call. = FALSE
      )
    }
    if (is.na(universal_global_out_dim) || universal_global_out_dim < 1L) {
      universal_global_out_dim <- max(1L, max(universal_n_outcomes_all, na.rm = TRUE))
    }
    # Screen every pooled training row against its declared outcome family
    # BEFORE anything reaches JAX: one NaN outcome NaNs the entire ELBO and
    # its gradients, while an out-of-range categorical/ordinal index would be
    # clamped into a finite-but-wrong likelihood term -- both silent at FM
    # scale. Fail loudly with the offending rows instead.
    universal_family_codes_all <- match(
      universal_likelihood_all,
      c("bernoulli", "categorical", "normal", "ordinal")
    ) - 1L
    universal_rows_valid <- neural_mixed_row_is_valid(
      y = Y_,
      likelihood_code_obs = universal_family_codes_all,
      n_outcomes_obs = universal_n_outcomes_all
    )
    if (!all(universal_rows_valid)) {
      bad_rows <- which(!universal_rows_valid)
      stop(sprintf(paste0(
        "Universal foundation training received %d invalid outcome row(s) ",
        "(non-finite Y, or categorical/ordinal Y outside [0, n_outcomes)). ",
        "First offending rows: %s. Clean or drop these rows before training."
      ),
      length(bad_rows),
      paste(utils::head(bad_rows, 10L), collapse = ", ")
      ), call. = FALSE)
    }
  }
  universal_task_mode_levels <- unique(universal_task_mode_all)
  universal_mixed_mode <- isTRUE(universal_enabled) &&
    "pairwise" %in% universal_task_mode_levels &&
    "single" %in% universal_task_mode_levels
  text_semantic_dim <- as.integer(neural_token_info_use$text_dim %||% 0L)
  factor_struct_dim <- if (!is.null(factor_struct_matrix)) {
    ncol(as.matrix(factor_struct_matrix))
  } else {
    0L
  }
  level_struct_dim <- if (!is.null(level_struct_matrices) && length(level_struct_matrices) > 0L) {
    ncol(as.matrix(level_struct_matrices[[1L]]))
  } else {
    0L
  }
  place_context_dim <- if (!is.null(place_embedding)) {
    ncol(as.matrix(place_embedding))
  } else if (!is.null(default_place_embedding)) {
    ncol(as.matrix(default_place_embedding))
  } else {
    length(place_feature_names)
  }
  if (!isTRUE(place_context_enabled)) {
    place_context_dim <- 0L
  }
  time_context_dim <- if (!is.null(time_embedding)) {
    ncol(as.matrix(time_embedding))
  } else if (!is.null(default_time_embedding)) {
    ncol(as.matrix(default_time_embedding))
  } else {
    length(time_feature_names)
  }
  if (!isTRUE(time_context_enabled)) {
    time_context_dim <- 0L
  }
  n_candidate_factor_tokens <- ai(max_factor_tokens)
  n_candidate_tokens <- ai(
    n_candidate_factor_tokens +
      as.integer(isTRUE(has_candidate_group_context)) +
      as.integer(isTRUE(has_relation_context))
  )

  # Helper to sanitize integer indices (adds explicit missing/OOV level per factor)
  to_index_matrix <- function(x_mat){
    x_mat <- as.matrix(x_mat)
    x_int <- matrix(as.integer(x_mat), nrow = nrow(x_mat), ncol = ncol(x_mat))
    n_cols <- ncol(x_int)
    n_levels <- factor_levels_int
    if (length(n_levels) != n_cols) {
      n_levels <- rep_len(n_levels, n_cols)
    }
    for (d_ in seq_len(n_cols)) {
      max_level <- n_levels[[d_]]
      missing_level <- max_level + 1L
      col_vals <- x_int[, d_]
      col_vals[is.na(col_vals) | col_vals < 1L | col_vals > max_level] <- missing_level
      x_int[, d_] <- col_vals - 1L
    }
    x_int
  }

  # Build pairwise, single-candidate, or mixed universal observation data.
  pair_mat <- NULL
  universal_pair_rows <- integer(0)
  universal_single_rows <- integer(0)
  universal_pair_obs_rows <- integer(0)
  n_universal_pair_obs <- 0L
  n_universal_single_obs <- 0L
  Y_pair_use <- NULL
  Y_single_use <- NULL
  resp_party_pair_use <- NULL
  resp_party_single_use <- NULL
  X_pair_use <- NULL
  X_single_cov_use <- NULL
  X_present_pair_use <- NULL
  X_present_single_use <- NULL
  experiment_index_pair_use <- NULL
  experiment_index_single_use <- NULL
  universal_likelihood_pair_use <- NULL
  universal_likelihood_single_use <- NULL
  universal_n_outcomes_pair_use <- NULL
  universal_n_outcomes_single_use <- NULL

  neural_pairwise_assert_pair_constant <- function(values, pair_mat, label) {
    if (is.null(values) || is.null(pair_mat) || nrow(pair_mat) < 1L) {
      return(invisible(TRUE))
    }
    if (!is.null(dim(values))) {
      values_df <- as.data.frame(values)
      row_sig <- vapply(seq_len(nrow(values_df)), function(i_) {
        digest::digest(as.list(values_df[i_, , drop = FALSE]), algo = "xxhash64", serialize = TRUE)
      }, character(1))
    } else {
      row_sig <- vapply(seq_along(values), function(i_) {
        digest::digest(values[[i_]], algo = "xxhash64", serialize = TRUE)
      }, character(1))
    }
    bad <- row_sig[pair_mat[, 1L]] != row_sig[pair_mat[, 2L]]
    if (any(bad, na.rm = TRUE)) {
      stop(
        sprintf("Pairwise neural mode requires %s to be constant within each pair_id.", label),
        call. = FALSE
      )
    }
    invisible(TRUE)
  }

  if (isTRUE(universal_mixed_mode)) {
    universal_pair_rows <- which(universal_task_mode_all == "pairwise")
    universal_single_rows <- which(universal_task_mode_all == "single")
    if (length(universal_pair_rows) < 2L || length(universal_single_rows) < 1L) {
      stop("Universal mixed-mode foundation training requires pairwise and single observation rows.", call. = FALSE)
    }
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id_[universal_pair_rows],
      W = if (isTRUE(compact_training)) NULL else W_[universal_pair_rows, , drop = FALSE],
      n_rows = length(universal_pair_rows),
      profile_order = if (!is.null(profile_order_)) profile_order_[universal_pair_rows] else NULL,
      competing_group_variable_candidate = if (!is.null(competing_group_variable_candidate_)) {
        competing_group_variable_candidate_[universal_pair_rows]
      } else {
        NULL
      }
    )
    if (is.null(pair_info) || is.null(pair_info$pair_sizes) ||
        !all(pair_info$pair_sizes == 2L)) {
      stop("Universal mixed-mode pairwise rows must define exactly 2 rows per pair_id.", call. = FALSE)
    }
    pair_mat <- matrix(universal_pair_rows[as.integer(pair_info$pair_mat)], ncol = 2L)
    n_universal_pair_obs <- nrow(pair_mat)
    n_universal_single_obs <- length(universal_single_rows)
    universal_pair_obs_rows <- pair_mat[, 1L]
    neural_pairwise_assert_pair_constant(resp_party_index, pair_mat, "respondent group metadata")
    neural_pairwise_assert_pair_constant(X_, pair_mat, "respondent covariates")
    neural_pairwise_assert_pair_constant(X_present_, pair_mat, "respondent covariate presence metadata")
    neural_pairwise_assert_pair_constant(experiment_index_all, pair_mat, "experiment metadata")
    neural_pairwise_assert_pair_constant(respondent_id_all, pair_mat, "respondent_id")
    neural_pairwise_assert_pair_constant(respondent_task_id_all, pair_mat, "respondent_task_id")

    if (isTRUE(compact_training)) {
      X_left <- NULL
      X_right <- NULL
      X_single <- NULL
    } else {
      X_left <- W_[pair_mat[, 1L], , drop = FALSE]
      X_right <- W_[pair_mat[, 2L], , drop = FALSE]
      X_single <- W_[universal_single_rows, , drop = FALSE]
    }
    Y_pair_use <- Y_[universal_pair_obs_rows]
    Y_single_use <- Y_[universal_single_rows]
    Y_use <- c(Y_pair_use, Y_single_use)
    party_left <- party_index[pair_mat[, 1L]]
    party_right <- party_index[pair_mat[, 2L]]
    party_single <- party_index[universal_single_rows]
    resp_party_pair_use <- resp_party_index[universal_pair_obs_rows]
    resp_party_single_use <- resp_party_index[universal_single_rows]
    resp_party_use <- c(resp_party_pair_use, resp_party_single_use)
    if (!is.null(X_)) {
      X_pair_use <- X_[universal_pair_obs_rows, , drop = FALSE]
      X_single_cov_use <- X_[universal_single_rows, , drop = FALSE]
      X_use <- rbind(X_pair_use, X_single_cov_use)
    }
    if (!is.null(X_present_)) {
      X_present_pair_use <- X_present_[universal_pair_obs_rows, , drop = FALSE]
      X_present_single_use <- X_present_[universal_single_rows, , drop = FALSE]
      X_present_use <- rbind(X_present_pair_use, X_present_single_use)
    } else {
      X_present_use <- NULL
    }
    if (!is.null(experiment_index_all)) {
      experiment_index_pair_use <- experiment_index_all[universal_pair_obs_rows]
      experiment_index_single_use <- experiment_index_all[universal_single_rows]
      experiment_index_use <- c(experiment_index_pair_use, experiment_index_single_use)
    } else {
      experiment_index_use <- NULL
    }
  } else if (pairwise_mode) {
    pair_info <- cs2step_build_pair_mat(
      pair_id = pair_id_,
      W = if (isTRUE(compact_training)) NULL else W_,
      n_rows = if (isTRUE(compact_training)) length(Y_) else NULL,
      profile_order = profile_order_,
      competing_group_variable_candidate = competing_group_variable_candidate_
    )
    if (is.null(pair_info) || is.null(pair_info$pair_sizes) ||
        !all(pair_info$pair_sizes == 2L)) {
      stop("Pairwise neural mode requires pair_id to define exactly 2 rows per pair.", call. = FALSE)
    } else {
      pair_mat <- pair_info$pair_mat
    }
    if (pairwise_mode) {
      neural_pairwise_assert_pair_constant(resp_party_index, pair_mat, "respondent group metadata")
      neural_pairwise_assert_pair_constant(X_, pair_mat, "respondent covariates")
      neural_pairwise_assert_pair_constant(X_present_, pair_mat, "respondent covariate presence metadata")
      neural_pairwise_assert_pair_constant(experiment_index_all, pair_mat, "experiment metadata")
      neural_pairwise_assert_pair_constant(respondent_id_all, pair_mat, "respondent_id")
      neural_pairwise_assert_pair_constant(respondent_task_id_all, pair_mat, "respondent_task_id")
      if (isTRUE(compact_training)) {
        X_left <- NULL
        X_right <- NULL
      } else {
        X_left <- W_[pair_mat[,1], , drop = FALSE]
        X_right <- W_[pair_mat[,2], , drop = FALSE]
      }
      Y_use <- Y_[pair_mat[,1]]
      party_left <- party_index[pair_mat[,1]]
      party_right <- party_index[pair_mat[,2]]
      resp_party_use <- resp_party_index[pair_mat[,1]]
      resp_party_pair_use <- resp_party_use
      if (!is.null(X_)) {
        X_use <- X_[pair_mat[,1], , drop = FALSE]
        X_pair_use <- X_use
      }
      if (!is.null(X_present_)) {
        X_present_use <- X_present_[pair_mat[,1], , drop = FALSE]
        X_present_pair_use <- X_present_use
      } else {
        X_present_use <- NULL
      }
      if (!is.null(experiment_index_all)) {
        experiment_index_use <- experiment_index_all[pair_mat[,1]]
        experiment_index_pair_use <- experiment_index_use
      } else {
        experiment_index_use <- NULL
      }
    }
  }

  if (!pairwise_mode) {
    X_single <- if (isTRUE(compact_training)) NULL else W_
    Y_use <- Y_
    party_single <- party_index
    resp_party_use <- resp_party_index
    resp_party_single_use <- resp_party_use
    X_use <- X_
    X_single_cov_use <- X_use
    X_present_use <- X_present_
    X_present_single_use <- X_present_use
    experiment_index_use <- experiment_index_all
    experiment_index_single_use <- experiment_index_use
  }

  if (!isTRUE(pairwise_mode)) {
    cross_candidate_encoder_mode <- "none"
  } else if (!isTRUE(cross_candidate_encoder_supplied)) {
    cross_candidate_encoder_mode <- if (low_rank_interaction_rank > 0L) "none" else "term"
  } else if (identical(cross_candidate_encoder_mode, "term") &&
             low_rank_interaction_rank > 0L) {
    cross_candidate_encoder_note <- paste(
      "Explicit cross_candidate_encoder='term' was honored with",
      "low_rank_interaction_rank > 0; this combination can be unstable."
    )
    if (!isTRUE(neural_oos_eval_internal_flag)) {
      message(cross_candidate_encoder_note)
    }
  }
  use_cross_term <- identical(cross_candidate_encoder_mode, "term")
  use_cross_attn <- identical(cross_candidate_encoder_mode, "attn")
  use_cross_encoder <- identical(cross_candidate_encoder_mode, "full")

  universal_task_mode_use <- NULL
  universal_likelihood_use <- NULL
  universal_n_outcomes_use <- NULL
  if (isTRUE(universal_enabled)) {
    if (isTRUE(universal_mixed_mode)) {
      universal_task_mode_use <- c(rep.int("pairwise", n_universal_pair_obs), rep.int("single", n_universal_single_obs))
      universal_likelihood_pair_use <- universal_likelihood_all[universal_pair_obs_rows]
      universal_likelihood_single_use <- universal_likelihood_all[universal_single_rows]
      universal_n_outcomes_pair_use <- universal_n_outcomes_all[universal_pair_obs_rows]
      universal_n_outcomes_single_use <- universal_n_outcomes_all[universal_single_rows]
      universal_likelihood_use <- c(universal_likelihood_pair_use, universal_likelihood_single_use)
      universal_n_outcomes_use <- c(universal_n_outcomes_pair_use, universal_n_outcomes_single_use)
    } else if (pairwise_mode) {
      universal_task_mode_use <- universal_task_mode_all[pair_mat[,1]]
      universal_likelihood_use <- universal_likelihood_all[pair_mat[,1]]
      universal_n_outcomes_use <- universal_n_outcomes_all[pair_mat[,1]]
      universal_likelihood_pair_use <- universal_likelihood_use
      universal_n_outcomes_pair_use <- universal_n_outcomes_use
    } else {
      universal_task_mode_use <- universal_task_mode_all
      universal_likelihood_use <- universal_likelihood_all
      universal_n_outcomes_use <- universal_n_outcomes_all
      universal_likelihood_single_use <- universal_likelihood_use
      universal_n_outcomes_single_use <- universal_n_outcomes_use
    }
  }
  respondent_id_use <- NULL
  respondent_task_id_use <- NULL
  if (!is.null(respondent_id_all)) {
    respondent_id_use <- if (isTRUE(universal_enabled) && isTRUE(universal_mixed_mode)) {
      c(respondent_id_all[universal_pair_obs_rows], respondent_id_all[universal_single_rows])
    } else if (isTRUE(pairwise_mode)) {
      respondent_id_all[pair_mat[, 1L]]
    } else {
      respondent_id_all
    }
  }
  if (!is.null(respondent_task_id_all)) {
    respondent_task_id_use <- if (isTRUE(universal_enabled) && isTRUE(universal_mixed_mode)) {
      c(respondent_task_id_all[universal_pair_obs_rows], respondent_task_id_all[universal_single_rows])
    } else if (isTRUE(pairwise_mode)) {
      respondent_task_id_all[pair_mat[, 1L]]
    } else {
      respondent_task_id_all
    }
  }
  if (!isTRUE(compact_training) &&
      identical(covariate_value_encoding, "shared_projection") &&
      !is.null(X_use) &&
      ncol(X_use) > 0L) {
    present_mask <- if (!is.null(X_present_use) && ncol(X_present_use) == ncol(X_use)) {
      X_present_use > 0
    } else {
      matrix(TRUE, nrow = nrow(X_use), ncol = ncol(X_use))
    }
    bad_mask <- present_mask & !is.finite(X_use)
    if (any(bad_mask, na.rm = TRUE)) {
      bad_cols <- unique(colnames(X_use)[col(bad_mask)[bad_mask]])
      stop(
        sprintf(
          "shared_projection does not support NA/Inf covariate values. Offending covariates: %s",
          paste(bad_cols, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }

  pairwise_context_mode <- "stage_free"
  stage_context_enabled <- FALSE
  stage_diagnostics <- list(
    pairwise_context_mode = pairwise_context_mode,
    stage_context_policy = "masked_optional",
    candidate_group_context = isTRUE(has_candidate_group_context),
    respondent_group_context = isTRUE(has_respondent_group_context),
    relation_context = isTRUE(has_relation_context),
    stage_context_enabled = FALSE,
    stage_context_reason = if (!isTRUE(pairwise_mode)) {
      "not_pairwise"
    } else if (!isTRUE(has_candidate_group_context) || !isTRUE(has_respondent_group_context)) {
      "missing_group_metadata"
    } else {
      "pending"
    }
  )
  if (pairwise_mode) {
    stage_is_primary <- party_left == party_right
    pairwise_rows_use <- if (isTRUE(universal_mixed_mode)) {
      rep(TRUE, length(stage_is_primary))
    } else if (isTRUE(universal_enabled) && !is.null(universal_task_mode_use)) {
      universal_task_mode_use == "pairwise"
    } else {
      rep(TRUE, length(stage_is_primary))
    }
    resp_party_stage_use <- resp_party_pair_use %||% resp_party_use
    context_present_pair <- !is.na(party_left) &
      !is.na(party_right) &
      !is.na(resp_party_stage_use) &
      party_left != party_missing_index &
      party_right != party_missing_index &
      resp_party_stage_use != resp_party_missing_index
    context_present_pair[is.na(context_present_pair)] <- FALSE
    n_context_present_pairs <- sum(pairwise_rows_use & context_present_pair, na.rm = TRUE)
    n_context_absent_pairs <- sum(pairwise_rows_use & !context_present_pair, na.rm = TRUE)
    stage_rows_use <- pairwise_rows_use & context_present_pair
    n_total <- sum(stage_rows_use, na.rm = TRUE)
    n_primary <- if (n_total > 0L) {
      sum(stage_is_primary[stage_rows_use], na.rm = TRUE)
    } else {
      0L
    }
    n_general <- if (n_total > 0L) {
      sum(!stage_is_primary[stage_rows_use], na.rm = TRUE)
    } else {
      0L
    }
    pct_primary <- if (n_total > 0L) n_primary / n_total else NA_real_
    stage_label <- ifelse(stage_is_primary[stage_rows_use], "primary", "general")
    resp_party_label <- if (!is.null(resp_party_levels)) {
      idx <- as.integer(resp_party_stage_use) + 1L
      idx[idx < 1L | idx > length(resp_party_levels)] <- NA_integer_
      resp_party_levels[idx][stage_rows_use]
    } else {
      as.character(resp_party_stage_use)[stage_rows_use]
    }
    stage_table <- table(resp_party_label, stage_label)
    cell_counts <- as.integer(stage_table)
    min_cell_n <- if (length(cell_counts) > 0L) min(cell_counts) else NA_integer_
    single_stage_only <- !isTRUE(has_relation_context) || isTRUE(pct_primary == 0 || pct_primary == 1)
    warn_stage_imbalance <- isTRUE(has_relation_context) &&
      is.finite(pct_primary) &&
      (pct_primary < warn_stage_imbalance_pct || pct_primary > (1 - warn_stage_imbalance_pct))
    warn_sparse_cells <- isTRUE(has_relation_context) &&
      !is.na(min_cell_n) && min_cell_n < warn_min_cell_n
    observed_stage_context <- isTRUE(has_relation_context) &&
      is.finite(pct_primary) &&
      n_primary > 0L &&
      n_general > 0L
    stage_context_enabled <- isTRUE(pairwise_mode) &&
      (isTRUE(force_stage_context) || isTRUE(observed_stage_context))
    pairwise_context_mode <- if (isTRUE(stage_context_enabled)) {
      "stage_aware"
    } else {
      "stage_free"
    }
    stage_context_reason <- if (!isTRUE(has_candidate_group_context) ||
                                !isTRUE(has_respondent_group_context)) {
      "missing_group_metadata"
    } else if (isTRUE(force_stage_context) && !isTRUE(observed_stage_context)) {
      "forced_unified"
    } else if (n_primary < 1L || n_general < 1L) {
      "single_stage_only"
    } else {
      "enabled"
    }

    if (isTRUE(observed_stage_context) && isTRUE(warn_stage_imbalance)) {
      warning(
        sprintf("Stage imbalance detected in neural training data (pct_primary=%.3f).", pct_primary),
        call. = FALSE
      )
    }
    if (isTRUE(observed_stage_context) && isTRUE(warn_sparse_cells)) {
      warning(
        sprintf("Sparse stage/resp-party cells detected (min cell n=%d).", min_cell_n),
        call. = FALSE
      )
    }
    if (isTRUE(has_relation_context) &&
        !isTRUE(stage_context_enabled) &&
        !isTRUE(force_stage_context)) {
      warning(
        "Stage indicator has no variation; fitting pairwise FM in stage-free mode.",
        call. = FALSE
      )
    }

    stage_updates <- list(
      stage_context_policy = "masked_optional",
      n_primary = as.integer(n_primary),
      n_general = as.integer(n_general),
      n_context_present_pairs = as.integer(n_context_present_pairs),
      n_context_absent_pairs = as.integer(n_context_absent_pairs),
      pct_primary = pct_primary,
      resp_party_stage_table = stage_table,
      single_stage_only = single_stage_only,
      sparse_cells = warn_sparse_cells,
      min_cell_n = min_cell_n,
      warn_stage_imbalance_pct = warn_stage_imbalance_pct,
      warn_min_cell_n = warn_min_cell_n,
      stage_context_enabled = stage_context_enabled,
      stage_context_reason = stage_context_reason,
      pairwise_context_mode = pairwise_context_mode
    )
    stage_diagnostics[names(stage_updates)] <- stage_updates
  }
  use_matchup_token <- isTRUE(pairwise_mode) &&
    isTRUE(stage_context_enabled) &&
    (isTRUE(force_matchup_context) || !identical(cross_candidate_encoder_mode, "none"))
  n_matchup_levels <- if (isTRUE(use_matchup_token)) {
    as.integer(n_party_levels * (n_party_levels + 1L) / 2L)
  } else {
    0L
  }
  allowed_token_family_levels <- neural_token_family_levels(
    include_candidate_group = has_candidate_group_context,
    include_relation = has_relation_context,
    include_stage = stage_context_enabled,
    include_respondent_group = has_respondent_group_context,
    include_matchup = use_matchup_token,
    include_place = place_context_enabled,
    include_time = time_context_enabled,
    include_readout_cls = low_rank_interaction_rank > 0L
  )
  token_family_override <- neural_token_info_use$token_family_levels %||% NULL
  token_family_levels <- if (is.null(token_family_override)) {
    allowed_token_family_levels
  } else {
    override_chr <- as.character(token_family_override)
    override_chr <- override_chr[override_chr %in% allowed_token_family_levels]
    c(override_chr, setdiff(allowed_token_family_levels, override_chr))
  }

  n_resp_covariates <- if (isTRUE(compact_training)) {
    ai(length(covariate_names_override))
  } else if (!is.null(X_use)) {
    ai(ncol(X_use))
  } else {
    ai(0L)
  }
  compact_covariate_rows <- if (isTRUE(compact_training) && n_resp_covariates > 0L) {
    if (isTRUE(pairwise_mode)) {
      pair_mat[, 1L]
    } else {
      seq_along(Y_)
    }
  } else {
    NULL
  }
  compact_covariate_profiles <- if (isTRUE(compact_training) && n_resp_covariates > 0L) {
    cs2step_compact_covariate_profiles(
      x = X_compact_use,
      rows = compact_covariate_rows,
      experiment_index = experiment_index_use,
      covariate_names = covariate_names_override,
      default_experiment_index = default_experiment_index
    )
  } else {
    NULL
  }
  resp_cov_mean <- if (!is.null(compact_covariate_profiles)) {
    compact_covariate_profiles$mean
  } else if (!is.null(X_use) && n_resp_covariates > 0L) {
    if (!is.null(X_present_use) && ncol(X_present_use) == ncol(X_use)) {
      means <- numeric(ncol(X_use))
      for (j in seq_len(ncol(X_use))) {
        present_idx <- which(X_present_use[, j] > 0)
        means[[j]] <- if (length(present_idx) > 0L) {
          mean(X_use[present_idx, j])
        } else {
          0
        }
      }
      means
    } else {
      as.numeric(colMeans(X_use))
    }
  } else {
    NULL
  }
  resp_cov_scale <- if (!is.null(compact_covariate_profiles)) {
    compact_covariate_profiles$scale
  } else if (!is.null(X_use) && n_resp_covariates > 0L) {
    if (!is.null(X_present_use) && ncol(X_present_use) == ncol(X_use)) {
      sds <- numeric(ncol(X_use))
      for (j in seq_len(ncol(X_use))) {
        present_idx <- which(X_present_use[, j] > 0)
        sds[[j]] <- if (length(present_idx) > 1L) {
          stats::sd(X_use[present_idx, j])
        } else {
          0
        }
      }
      sds[!is.finite(sds) | sds < 1e-6] <- 1
      sds
    } else {
      sds <- as.numeric(apply(X_use, 2L, stats::sd))
      sds[!is.finite(sds) | sds < 1e-6] <- 1
      sds
    }
  } else {
    NULL
  }
  resp_cov_default_present <- if (!is.null(compact_covariate_profiles)) {
    compact_covariate_profiles$default_present
  } else if (!is.null(X_present_use) && n_resp_covariates > 0L) {
    as.numeric(colMeans(X_present_use > 0, na.rm = TRUE) > 0)
  } else {
    NULL
  }
  covariate_distribution_profiles <- if (!is.null(compact_covariate_profiles)) {
    compact_covariate_profiles$distribution_profiles
  } else {
    neural_build_covariate_distribution_profiles(
      X_mat = X_use,
      X_present_mat = X_present_use,
      experiment_index = experiment_index_use,
      covariate_names = covariate_names_override,
      default_experiment_index = default_experiment_index
    )
  }
  covariate_value_stats_by_experiment <- covariate_distribution_profiles$by_experiment %||% list()
  default_covariate_value_stats <- covariate_distribution_profiles$default_stats %||% NULL
  covariate_value_metadata_by_experiment <- covariate_distribution_profiles$metadata_by_experiment %||% list()
  default_covariate_value_metadata <- covariate_distribution_profiles$default_metadata %||% NULL
  n_experiment_levels <- length(experiment_levels_override)

  # Placeholder to keep model chunks consistent (no surrogate regression)
  main_dat <- matrix(0, nrow = 0L, ncol = 0L)

  # Likelihood selection (allow overrides for cross-fit evaluation runs).
  likelihood_override <- NULL
  nOutcomes_override <- NULL
  if (exists("neural_likelihood_override", inherits = TRUE)) {
    likelihood_override <- get("neural_likelihood_override", inherits = TRUE)
  }
  if (exists("neural_nOutcomes_override", inherits = TRUE)) {
    nOutcomes_override <- get("neural_nOutcomes_override", inherits = TRUE)
  }

  is_binary <- all(unique(na.omit(as.numeric(Y_use))) %in% c(0, 1)) &&
    length(unique(na.omit(Y_use))) <= 2
  is_intvec <- all(!is.na(Y_use)) && all(abs(Y_use - round(Y_use)) < 1e-8)
  K_classes <- if (is_intvec) length(unique(ai(Y_use))) else NA_integer_
  universal_has_multiple_likelihoods <- isTRUE(universal_enabled) &&
    length(unique(universal_likelihood_use)) > 1L
  universal_has_normal <- isTRUE(universal_enabled) &&
    any(universal_likelihood_use == "normal")
  universal_has_ordinal <- isTRUE(universal_enabled) &&
    any(universal_likelihood_use == "ordinal")
  universal_likelihood_levels <- c("bernoulli", "categorical", "normal", "ordinal")

  if (!is.null(likelihood_override)) {
    likelihood <- tolower(as.character(likelihood_override))
    if (likelihood %in% c("ordered", "ordered_logit", "ordered-logit", "ordinal_single")) {
      likelihood <- "ordinal"
    }
    if (!likelihood %in% c("bernoulli", "categorical", "normal", "ordinal")) {
      stop("neural_likelihood_override must be one of 'bernoulli', 'categorical', 'normal', or 'ordinal'.",
           call. = FALSE)
    }
    if (!is.null(nOutcomes_override)) {
      nOutcomes <- ai(nOutcomes_override)
    } else if (likelihood %in% c("categorical", "ordinal")) {
      stop("neural_nOutcomes_override must be provided when forcing categorical or ordinal likelihood.",
           call. = FALSE)
    } else {
      nOutcomes <- ai(1L)
    }
  } else if (isTRUE(universal_enabled)) {
    likelihood <- if (isTRUE(universal_has_multiple_likelihoods)) {
      "mixed"
    } else {
      unique(universal_likelihood_use)[[1L]]
    }
    nOutcomes <- ai(max(1L, universal_global_out_dim))
  } else if (is_binary) {
    likelihood <- "bernoulli"; nOutcomes <- ai(1L)
  } else if (!is.na(K_classes) && K_classes >= 2L &&
             K_classes <= max(50L, length(factor_levels_int) + 1L)) {
    likelihood <- "categorical"; nOutcomes <- ai(K_classes)
  } else {
    likelihood <- "normal"; nOutcomes <- ai(1L)
  }
  has_ordinal_likelihood <- identical(likelihood, "ordinal") ||
    isTRUE(universal_has_ordinal)
  ordinal_n_threshold_groups <- if (isTRUE(has_ordinal_likelihood)) {
    max(1L, as.integer(n_experiment_levels))
  } else {
    0L
  }
  ordinal_max_n_outcomes <- if (isTRUE(has_ordinal_likelihood)) {
    if (isTRUE(universal_enabled) && any(universal_likelihood_use == "ordinal")) {
      max(2L, max(as.integer(universal_n_outcomes_use[universal_likelihood_use == "ordinal"]), na.rm = TRUE))
    } else {
      max(2L, as.integer(nOutcomes))
    }
  } else {
    0L
  }
  ordinal_max_thresholds <- if (isTRUE(has_ordinal_likelihood)) {
    max(1L, as.integer(ordinal_max_n_outcomes) - 1L)
  } else {
    0L
  }
  low_rank_logit_model_info$likelihood <- likelihood
  low_rank_logit_normalization <- neural_resolve_low_rank_logit_normalization(
    value = low_rank_logit_normalization_control,
    supplied = low_rank_logit_normalization_supplied,
    low_rank_interaction_rank = low_rank_interaction_rank,
    pairwise_mode = pairwise_mode,
    likelihood = likelihood
  )
  low_rank_head_weight_target_rms <- neural_resolve_low_rank_head_weight_target_rms(
    value = low_rank_head_weight_target_rms_control,
    model_dims = ModelDims,
    normalization = low_rank_logit_normalization,
    supplied = low_rank_head_weight_target_rms_supplied
  )
  low_rank_rc_out_target_rms <- neural_resolve_low_rank_rc_out_target_rms(
    value = low_rank_rc_out_target_rms_control,
    low_rank_interaction_rank = low_rank_interaction_rank,
    normalization = low_rank_logit_normalization,
    supplied = low_rank_rc_out_target_rms_supplied
  )
  learned_pairwise_bernoulli_logit_scale <- isTRUE(
    learned_pairwise_bernoulli_logit_scale_requested
  ) && isTRUE(pairwise_mode) && likelihood %in% c("bernoulli", "mixed")
  calibration_control_resolve <- calibration_control_raw
  if (isTRUE(learned_pairwise_bernoulli_logit_scale) &&
      !isTRUE(calibration_control_supplied)) {
    calibration_control_resolve <- list(enabled = FALSE)
  }
  calibration_control <- neural_resolve_calibration_control(
    value = calibration_control_resolve,
    likelihood = likelihood,
    legacy_pairwise_scale = FALSE,
    prior_sd_fallback = 0.5
  )
  neural_validate_calibration_temperature_compatibility(
    learned_pairwise_bernoulli_logit_scale = learned_pairwise_bernoulli_logit_scale,
    calibration_control = calibration_control
  )
  mcmc_control$calibration <- calibration_control
  pairwise_bernoulli_logit_scale_prior_sd <-
    neural_resolve_pairwise_bernoulli_logit_scale_prior_sd(
      value = mcmc_control$pairwise_bernoulli_logit_scale_prior_sd %||% NULL,
      enabled = learned_pairwise_bernoulli_logit_scale
    )
  if (identical(low_rank_logit_normalization, "rms") &&
      identical(low_rank_logit_transform, "softclip") &&
      isTRUE(low_rank_logit_transform_supplied) &&
      !isTRUE(neural_oos_eval_internal_flag)) {
    message(
      "neural_mcmc_control$low_rank_logit_transform='softclip' was honored, ",
      "but low_rank_logit_normalization='rms' already normalizes low-rank ",
      "pairwise Bernoulli logits."
    )
  }
  low_rank_logit_model_info$low_rank_logit_normalization <- low_rank_logit_normalization
  low_rank_logit_model_info$low_rank_head_weight_target_rms <- low_rank_head_weight_target_rms
  low_rank_logit_model_info$low_rank_rc_out_target_rms <- low_rank_rc_out_target_rms
  low_rank_logit_model_info$learned_pairwise_bernoulli_logit_scale <-
    learned_pairwise_bernoulli_logit_scale
  low_rank_logit_model_info$pairwise_bernoulli_logit_scale_prior_sd <-
    pairwise_bernoulli_logit_scale_prior_sd
  low_rank_logit_model_info$pairwise_antisymmetry <- pairwise_antisymmetry_mode
  low_rank_logit_model_info$additive_utility_mode <- additive_utility_mode
  low_rank_logit_model_info$additive_utility_normalization <- additive_utility_normalization
  low_rank_logit_model_info$additive_head_weight_target_rms <- additive_head_weight_target_rms
  low_rank_logit_model_info$calibration_enabled <- isTRUE(calibration_control$enabled)
  low_rank_logit_model_info$calibration_method <- calibration_control$method %||% "none"
  low_rank_logit_model_info$calibration_prior_sd <- calibration_control$prior_sd %||% NULL
  sigma_prior_scale <- 1.0
  if (likelihood == "normal" || isTRUE(universal_has_normal)) {
    y_numeric <- if (isTRUE(universal_enabled) && any(universal_likelihood_use == "normal")) {
      as.numeric(Y_use[universal_likelihood_use == "normal"])
    } else {
      as.numeric(Y_use)
    }
    y_mad <- suppressWarnings(stats::mad(y_numeric, na.rm = TRUE))
    y_sd <- suppressWarnings(stats::sd(y_numeric, na.rm = TRUE))
    sigma_prior_scale <- if (is.finite(y_mad) && y_mad > 0) {
      y_mad
    } else if (is.finite(y_sd) && y_sd > 0) {
      y_sd
    } else {
      1.0
    }
  }

  universal_likelihood_code_use <- if (isTRUE(universal_enabled) &&
                                       !is.null(universal_likelihood_use)) {
    as.integer(match(universal_likelihood_use, universal_likelihood_levels) - 1L)
  } else {
    NULL
  }
  universal_n_outcomes_use_int <- if (isTRUE(universal_enabled) &&
                                      !is.null(universal_n_outcomes_use)) {
    as.integer(universal_n_outcomes_use)
  } else {
    NULL
  }
  universal_loss_weights <- NULL
  universal_loss_weights_pair <- NULL
  universal_loss_weights_single <- NULL
  universal_loss_weighting_info <- NULL
  universal_loss_weighting_diagnostics <- NULL
  if (isTRUE(universal_enabled) &&
      !is.null(universal_task_mode_use) &&
      !is.null(universal_likelihood_code_use)) {
    universal_loss_weighting_info <- neural_universal_loss_weighting(
      task_mode = universal_task_mode_use,
      likelihood_code = universal_likelihood_code_use,
      policy = mcmc_control$universal_loss_weighting %||% NULL,
      clip = mcmc_control$universal_loss_weight_clip %||% NULL
    )
    universal_loss_weights <- universal_loss_weighting_info$weights
    universal_loss_weighting_diagnostics <- universal_loss_weighting_info$diagnostics
    if (!is.null(universal_loss_weights)) {
      if (isTRUE(universal_mixed_mode)) {
        universal_loss_weights_pair <- universal_loss_weights[seq_len(n_universal_pair_obs)]
        universal_loss_weights_single <- universal_loss_weights[
          n_universal_pair_obs + seq_len(n_universal_single_obs)
        ]
      } else if (pairwise_mode) {
        universal_loss_weights_pair <- universal_loss_weights
      } else {
        universal_loss_weights_single <- universal_loss_weights
      }
    }
  }
  low_rank_logit_model_info$universal_loss_weighting <- universal_loss_weighting_diagnostics

  mixed_row_is_valid_r <- neural_mixed_row_is_valid

  mixed_eval_strata_r <- function(y,
                                  likelihood_code_obs,
                                  experiment_index = NULL,
                                  n_outcomes_obs = NULL) {
    n <- length(y)
    exp_label <- if (!is.null(experiment_index) && length(experiment_index) == n) {
      paste0("exp", as.integer(experiment_index))
    } else {
      rep("exp_all", n)
    }
    fam_label <- rep("unknown", n)
    fam_label[likelihood_code_obs == 0L] <- "bernoulli"
    fam_label[likelihood_code_obs == 1L] <- "categorical"
    fam_label[likelihood_code_obs == 2L] <- "normal"
    fam_label[likelihood_code_obs == 3L] <- "ordinal"
    strata <- paste(fam_label, exp_label, sep = "::")
    class_idx <- likelihood_code_obs %in% c(0L, 1L, 3L) & is.finite(as.numeric(y))
    if (any(class_idx)) {
      y_class <- suppressWarnings(as.integer(round(as.numeric(y[class_idx]))))
      strata[class_idx] <- paste(strata[class_idx], paste0("class", y_class), sep = "::")
    }
    if (!is.null(n_outcomes_obs) && length(n_outcomes_obs) == n) {
      cat_idx <- likelihood_code_obs == 1L & is.finite(as.numeric(n_outcomes_obs))
      if (any(cat_idx)) {
        strata[cat_idx] <- paste(
          strata[cat_idx],
          paste0("k", as.integer(n_outcomes_obs[cat_idx])),
          sep = "::"
        )
      }
      ord_idx <- likelihood_code_obs == 3L & is.finite(as.numeric(n_outcomes_obs))
      if (any(ord_idx)) {
        strata[ord_idx] <- paste(
          strata[ord_idx],
          paste0("k", as.integer(n_outcomes_obs[ord_idx])),
          sep = "::"
        )
      }
    }
    strata
  }

  mixed_softmax_prob_matrix_r <- function(logits, n_outcomes_obs) {
    logits <- as.matrix(logits)
    if (!length(logits)) {
      return(matrix(numeric(0), nrow = 0L, ncol = ncol(logits)))
    }
    n <- nrow(logits)
    k_max <- ncol(logits)
    probs <- matrix(0, nrow = n, ncol = k_max)
    for (i in seq_len(n)) {
      k_i <- as.integer(n_outcomes_obs[[i]])
      if (!is.finite(k_i) || k_i < 1L) {
        next
      }
      k_i <- min(k_i, k_max)
      row_logits <- as.numeric(logits[i, seq_len(k_i), drop = TRUE])
      if (!all(is.finite(row_logits))) {
        probs[i, seq_len(k_i)] <- rep(1 / k_i, k_i)
        next
      }
      max_logit <- max(row_logits)
      weights <- exp(row_logits - max_logit)
      denom <- sum(weights)
      if (!is.finite(denom) || denom <= 0) {
        probs[i, seq_len(k_i)] <- rep(1 / k_i, k_i)
      } else {
        probs[i, seq_len(k_i)] <- weights / denom
      }
    }
    probs
  }

  ordinal_thresholds_from_raw_r <- function(raw) {
    raw <- as.matrix(raw)
    if (!length(raw)) {
      return(raw)
    }
    out <- raw
    if (ncol(raw) > 1L) {
      inc <- log1p(exp(-abs(raw[, -1L, drop = FALSE]))) +
        pmax(raw[, -1L, drop = FALSE], 0) +
        1e-4
      out[, -1L] <- inc
    }
    t(apply(out, 1L, cumsum))
  }

  ordinal_prob_matrix_r <- function(eta,
                                    n_outcomes_obs,
                                    experiment_index = NULL,
                                    ordinal_thresholds = NULL,
                                    ordinal_threshold_raw = NULL) {
    eta <- as.numeric(eta)
    n <- length(eta)
    k_vec <- as.integer(n_outcomes_obs)
    k_max <- max(1L, k_vec[is.finite(k_vec)], na.rm = TRUE)
    if (!is.finite(k_max) || k_max < 1L) {
      k_max <- 1L
    }
    probs <- matrix(0, nrow = n, ncol = k_max)
    thresholds <- ordinal_thresholds
    if (is.null(thresholds) && !is.null(ordinal_threshold_raw)) {
      thresholds <- ordinal_thresholds_from_raw_r(ordinal_threshold_raw)
    }
    if (is.null(thresholds)) {
      thresholds <- matrix(seq(-1, 1, length.out = max(1L, k_max - 1L)), nrow = 1L)
    }
    thresholds <- as.matrix(thresholds)
    exp_idx <- as.integer(experiment_index %||% rep(0L, n))
    if (length(exp_idx) == 1L && n > 1L) {
      exp_idx <- rep.int(exp_idx, n)
    }
    if (length(exp_idx) != n) {
      exp_idx <- rep.int(0L, n)
    }
    exp_idx[is.na(exp_idx) | exp_idx < 0L] <- 0L
    exp_idx <- pmin(exp_idx + 1L, nrow(thresholds))
    for (i in seq_len(n)) {
      k_i <- as.integer(k_vec[[i]])
      if (!is.finite(k_i) || k_i < 2L || !is.finite(eta[[i]])) {
        next
      }
      k_i <- min(k_i, k_max)
      cut <- as.numeric(thresholds[exp_idx[[i]], seq_len(min(k_i - 1L, ncol(thresholds))), drop = TRUE])
      if (length(cut) < k_i - 1L) {
        cut <- c(cut, tail(cut, 1L) + seq_len(k_i - 1L - length(cut)))
      }
      cut <- cummax(cut)
      cdf <- stats::plogis(cut - eta[[i]])
      p <- c(cdf[[1L]], diff(cdf), 1 - cdf[[length(cdf)]])
      p <- pmax(p, 1e-12)
      probs[i, seq_len(k_i)] <- p / sum(p)
    }
    probs
  }

  ordinal_metrics_r <- function(y_eval, prob_mat, n_outcomes_obs = NULL) {
    y_int <- suppressWarnings(as.integer(round(as.numeric(y_eval))))
    prob_mat <- as.matrix(prob_mat)
    n <- length(y_int)
    k_vec <- as.integer(n_outcomes_obs %||% rep(ncol(prob_mat), n))
    if (length(k_vec) == 1L && n > 1L) {
      k_vec <- rep.int(k_vec, n)
    }
    keep <- is.finite(y_int) & y_int >= 0L & seq_along(y_int) <= nrow(prob_mat)
    keep <- keep & is.finite(k_vec) & k_vec >= 2L & y_int < k_vec
    if (!any(keep)) {
      return(list(likelihood = "ordinal", n_eval = 0L, n_obs = 0L,
                  log_loss = NA_real_, nll = NA_real_,
                  accuracy = NA_real_, mae = NA_real_,
                  rmse = NA_real_, ranked_probability_score = NA_real_))
    }
    row_idx <- which(keep)
    p_y <- mapply(function(i, y) {
      prob_mat[i, y + 1L]
    }, row_idx, y_int[keep])
    p_y <- pmin(pmax(as.numeric(p_y), 1e-12), 1)
    expected <- rowSums(sweep(prob_mat[row_idx, , drop = FALSE], 2L, seq_len(ncol(prob_mat)) - 1L, `*`))
    denom <- pmax(k_vec[keep] - 1L, 1L)
    err <- (expected - y_int[keep]) / denom
    rps <- numeric(length(row_idx))
    for (j in seq_along(row_idx)) {
      i <- row_idx[[j]]
      k_i <- min(k_vec[keep][[j]], ncol(prob_mat))
      pred_cdf <- cumsum(prob_mat[i, seq_len(k_i)])
      obs_cdf <- as.numeric(seq.int(0L, k_i - 1L) >= y_int[keep][[j]])
      rps[[j]] <- sum((pred_cdf - obs_cdf)^2) / max(1L, k_i - 1L)
    }
    list(
      likelihood = "ordinal",
      n_eval = length(row_idx),
      n_obs = length(row_idx),
      log_loss = -mean(log(p_y)),
      nll = -mean(log(p_y)),
      accuracy = mean(max.col(prob_mat[row_idx, , drop = FALSE], ties.method = "first") - 1L == y_int[keep]),
      mae = mean(abs(err)),
      rmse = sqrt(mean(err^2)),
      ranked_probability_score = mean(rps)
    )
  }

  mixed_sigma_vector_r <- function(sigma, n) {
    if (is.null(sigma)) {
      return(rep(NA_real_, n))
    }
    sigma_vec <- as.numeric(sigma)
    if (length(sigma_vec) == 1L && n > 1L) {
      sigma_vec <- rep(sigma_vec, n)
    }
    if (length(sigma_vec) != n) {
      sigma_vec <- rep(NA_real_, n)
    }
    sigma_vec
  }

  mixed_pairwise_bernoulli_logit_r <- function(logits, row_idx, task_mode_obs = NULL) {
    z <- as.numeric(logits[row_idx, 1L])
    if (!length(z)) {
      return(z)
    }
    pairwise_idx <- if (is.null(task_mode_obs)) {
      rep(isTRUE(pairwise_mode), length(row_idx))
    } else {
      task_mode_chr <- tolower(as.character(task_mode_obs))
      if (length(task_mode_chr) == nrow(logits)) {
        task_mode_chr <- task_mode_chr[row_idx]
      }
      rep_len(task_mode_chr == "pairwise", length(row_idx))
    }
    pairwise_idx[is.na(pairwise_idx)] <- FALSE
    if (any(pairwise_idx)) {
      z[pairwise_idx] <- neural_apply_pairwise_bernoulli_logit_adjustment_r(
        z[pairwise_idx],
        low_rank_logit_model_info
      )
    }
    z
  }

  mixed_row_log_prob_r <- function(logits,
                                   y,
                                   likelihood_code_obs,
                                   n_outcomes_obs,
                                   sigma = NULL,
                                   task_mode_obs = NULL,
                                   experiment_index = NULL,
                                   ordinal_thresholds = NULL,
                                   ordinal_threshold_raw = NULL) {
    logits <- as.matrix(logits)
    n <- nrow(logits)
    if (n < 1L) {
      return(numeric(0))
    }
    y_num <- as.numeric(y)
    code <- as.integer(likelihood_code_obs)
    n_outcomes_obs <- as.integer(n_outcomes_obs)
    sigma_vec <- mixed_sigma_vector_r(sigma, n)
    out <- rep(NA_real_, n)

    bern_idx <- which(code == 0L)
    if (length(bern_idx) > 0L) {
      z <- mixed_pairwise_bernoulli_logit_r(
        logits,
        bern_idx,
        task_mode_obs = task_mode_obs
      )
      yb <- y_num[bern_idx]
      keep <- is.finite(z) & is.finite(yb) & yb %in% c(0, 1)
      if (any(keep)) {
        p <- stats::plogis(z[keep])
        p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
        out_bern <- ifelse(yb[keep] >= 0.5, log(p), log1p(-p))
        out[bern_idx[keep]] <- out_bern
      }
    }

    cat_idx <- which(code == 1L)
    if (length(cat_idx) > 0L) {
      for (j in seq_along(cat_idx)) {
        row_idx <- cat_idx[[j]]
        k_i <- as.integer(n_outcomes_obs[[row_idx]])
        y_i <- suppressWarnings(as.integer(round(y_num[[row_idx]])))
        if (!is.finite(k_i) || k_i < 2L || !is.finite(y_i) || y_i < 0L) {
          next
        }
        k_i <- min(k_i, ncol(logits))
        if (y_i >= k_i) {
          next
        }
        row_logits <- as.numeric(logits[row_idx, seq_len(k_i), drop = TRUE])
        if (!all(is.finite(row_logits))) {
          next
        }
        max_logit <- max(row_logits)
        log_denom <- max_logit + log(sum(exp(row_logits - max_logit)))
        out[[row_idx]] <- row_logits[[y_i + 1L]] - log_denom
      }
    }

    norm_idx <- which(code == 2L)
    if (length(norm_idx) > 0L) {
      mu <- as.numeric(logits[norm_idx, 1L])
      y_norm <- y_num[norm_idx]
      sigma_norm <- sigma_vec[norm_idx]
      keep <- is.finite(mu) & is.finite(y_norm) & is.finite(sigma_norm) & sigma_norm > 0
      if (any(keep)) {
        out_norm <- stats::dnorm(
          y_norm[keep],
          mean = mu[keep],
          sd = sigma_norm[keep],
          log = TRUE
        )
        out[norm_idx[keep]] <- out_norm
      }
    }

    ord_idx <- which(code == 3L)
    if (length(ord_idx) > 0L) {
      exp_idx_ord <- if (!is.null(experiment_index) && length(experiment_index) == n) {
        as.integer(experiment_index[ord_idx])
      } else {
        rep(0L, length(ord_idx))
      }
      prob_ord <- ordinal_prob_matrix_r(
        eta = as.numeric(logits[ord_idx, 1L]),
        n_outcomes_obs = n_outcomes_obs[ord_idx],
        experiment_index = exp_idx_ord,
        ordinal_thresholds = ordinal_thresholds,
        ordinal_threshold_raw = ordinal_threshold_raw
      )
      y_ord <- suppressWarnings(as.integer(round(y_num[ord_idx])))
      keep <- is.finite(y_ord) &
        y_ord >= 0L &
        y_ord < as.integer(n_outcomes_obs[ord_idx]) &
        (y_ord + 1L) <= ncol(prob_ord)
      if (any(keep)) {
        p <- prob_ord[cbind(which(keep), y_ord[keep] + 1L)]
        p <- pmin(pmax(p, 1e-12), 1)
        out[ord_idx[keep]] <- log(p)
      }
    }

    out
  }

  compute_mixed_outcome_metrics_r <- function(y_eval,
                                              pred_eval,
                                              likelihood_code_obs,
                                              n_outcomes_obs,
                                              task_mode_obs = NULL,
                                              experiment_index = NULL,
                                              ordinal_thresholds = NULL,
                                              ordinal_threshold_raw = NULL,
                                              threshold = 0.5) {
    logits <- as.matrix(pred_eval$logits %||% pred_eval)
    sigma_vec <- mixed_sigma_vector_r(pred_eval$sigma %||% NULL, nrow(logits))
    code <- as.integer(likelihood_code_obs)
    n_outcomes_obs <- as.integer(n_outcomes_obs)
    y_num <- as.numeric(y_eval)
    log_prob <- mixed_row_log_prob_r(
      logits = logits,
      y = y_num,
      likelihood_code_obs = code,
      n_outcomes_obs = n_outcomes_obs,
      sigma = sigma_vec,
      task_mode_obs = task_mode_obs,
      experiment_index = experiment_index,
      ordinal_thresholds = ordinal_thresholds,
      ordinal_threshold_raw = ordinal_threshold_raw
    )
    keep <- is.finite(log_prob)
    out <- list(
      likelihood = "mixed",
      n_eval = sum(keep),
      n_obs = sum(keep),
      nll = if (any(keep)) -mean(log_prob[keep]) else NA_real_,
      by_family = list()
    )

    bern_idx <- which(code == 0L)
    if (length(bern_idx) > 0L) {
      prob <- stats::plogis(mixed_pairwise_bernoulli_logit_r(
        logits,
        bern_idx,
        task_mode_obs = task_mode_obs
      ))
      metrics_bern <- cs_compute_outcome_metrics(
        y_eval = y_num[bern_idx],
        pred_eval = prob,
        likelihood = "bernoulli",
        threshold = threshold
      )
      metrics_bern$n_obs <- metrics_bern$n_eval %||% length(bern_idx)
      out$by_family$bernoulli <- metrics_bern
    }

    cat_idx <- which(code == 1L)
    if (length(cat_idx) > 0L) {
      prob_mat <- mixed_softmax_prob_matrix_r(logits[cat_idx, , drop = FALSE], n_outcomes_obs[cat_idx])
      metrics_cat <- cs_compute_outcome_metrics(
        y_eval = suppressWarnings(as.integer(round(y_num[cat_idx]))),
        pred_eval = prob_mat,
        likelihood = "categorical"
      )
      metrics_cat$n_obs <- metrics_cat$n_eval %||% length(cat_idx)
      out$by_family$categorical <- metrics_cat
    }

    norm_idx <- which(code == 2L)
    if (length(norm_idx) > 0L) {
      metrics_norm <- cs_compute_outcome_metrics(
        y_eval = y_num[norm_idx],
        pred_eval = list(
          mu = as.numeric(logits[norm_idx, 1L]),
          sigma = sigma_vec[norm_idx]
        ),
        likelihood = "normal"
      )
      metrics_norm$n_obs <- metrics_norm$n_eval %||% length(norm_idx)
      out$by_family$normal <- metrics_norm
    }

    ord_idx <- which(code == 3L)
    if (length(ord_idx) > 0L) {
      exp_idx_ord <- if (!is.null(experiment_index) && length(experiment_index) == length(y_num)) {
        as.integer(experiment_index[ord_idx])
      } else {
        rep(0L, length(ord_idx))
      }
      prob_ord <- ordinal_prob_matrix_r(
        eta = as.numeric(logits[ord_idx, 1L]),
        n_outcomes_obs = n_outcomes_obs[ord_idx],
        experiment_index = exp_idx_ord,
        ordinal_thresholds = ordinal_thresholds,
        ordinal_threshold_raw = ordinal_threshold_raw
      )
      metrics_ord <- ordinal_metrics_r(
        y_eval = y_num[ord_idx],
        prob_mat = prob_ord,
        n_outcomes_obs = n_outcomes_obs[ord_idx]
      )
      metrics_ord$n_obs <- metrics_ord$n_eval %||% length(ord_idx)
      out$by_family$ordinal <- metrics_ord
    }

    out
  }

  pdtype_ <- ddtype_ <- strenv$jnp$float32
  manual_noncentered_loc_scale <- FALSE
  p2d_eps <- 1e-6
  p2d_hash31 <- function(x, mod = 2147483647) {
    bytes <- utf8ToInt(as.character(x))
    h <- 0
    for (b in bytes) {
      h <- (h * 31 + b) %% mod
    }
    if (!is.finite(h) || is.na(h) || h < 0) {
      h <- 0
    }
    as.integer(h)
  }
  p2d_draw_fixed_normal <- function(name, scale, shape_tuple, dtype = ddtype_) {
    seed <- p2d_hash31(paste0("p2d:", name))
    key <- strenv$jax$random$PRNGKey(ai(seed))
    draw <- strenv$jax$random$normal(key, shape = shape_tuple, dtype = dtype)
    as.numeric(scale) * draw
  }
  p2d_init_normal <- function(name, scale, shape_tuple, dtype = ddtype_) {
    p2d_draw_fixed_normal(paste0(name, ":init"), scale, shape_tuple, dtype = dtype)
  }
  p2d_init_halfnormal <- function(name, scale, shape_tuple, dtype = ddtype_) {
    strenv$jnp$abs(p2d_draw_fixed_normal(paste0(name, ":init"), scale, shape_tuple, dtype = dtype)) + p2d_eps
  }
  p2d_init_lognormal <- function(name, sd, shape_tuple, dtype = ddtype_) {
    strenv$jnp$exp(p2d_draw_fixed_normal(paste0(name, ":init_log"), sd, shape_tuple, dtype = dtype)) + p2d_eps
  }
  p2d_constraint_positive <- NULL
  if (!is.null(strenv$numpyro$distributions) &&
      reticulate::py_has_attr(strenv$numpyro$distributions, "constraints")) {
    constraints_mod <- strenv$numpyro$distributions$constraints
    if (!is.null(constraints_mod) && reticulate::py_has_attr(constraints_mod, "positive")) {
      p2d_constraint_positive <- constraints_mod$positive
    }
  }
  # Warm-start values for point-estimated (numpyro.param) trunk sites under
  # uncertainty_scope = "output". init_to_value only seeds *sample* sites, so
  # without this lookup a foundation-model warm start supplied through
  # mcmc_control$init_site_values was silently discarded for every trunk
  # parameter and the trunk retrained from random init. Populated after
  # mcmc_control$init_site_values is validated (same frame, before tracing).
  p2d_warm_start_values <- list()
  p2d_warm_start_warned <- new.env(parent = emptyenv())
  p2d_shape_of <- function(x) {
    tryCatch(
      as.integer(unlist(reticulate::py_to_r(x$shape))),
      error = function(e) NULL
    )
  }
  p2d <- function(name,
                  sample_fxn,
                  init_fxn,
                  constraint = NULL,
                  uncertainty_scope_arg = uncertainty_scope,
                  is_output_layer = FALSE) {
    scope <- tolower(as.character(uncertainty_scope_arg))
    if (identical(scope, "output") && !isTRUE(is_output_layer)) {
      init_val <- init_fxn()
      warm <- p2d_warm_start_values[[name]]
      if (!is.null(warm)) {
        warm_val <- tryCatch(
          strenv$jnp$asarray(warm)$astype(ddtype_),
          error = function(e) NULL
        )
        warm_shape <- if (is.null(warm_val)) NULL else p2d_shape_of(warm_val)
        init_shape <- p2d_shape_of(init_val)
        if (!is.null(warm_val) && !is.null(warm_shape) &&
            !is.null(init_shape) && identical(warm_shape, init_shape)) {
          init_val <- warm_val
        } else if (!isTRUE(get0(name, envir = p2d_warm_start_warned,
                                ifnotfound = FALSE))) {
          assign(name, TRUE, envir = p2d_warm_start_warned)
          warning(sprintf(paste0(
            "init_site_values['%s'] does not match the expected parameter ",
            "shape (%s vs %s); ignoring this warm-start value."
          ), name,
          paste(warm_shape %||% "?", collapse = "x"),
          paste(init_shape %||% "?", collapse = "x")
          ), call. = FALSE)
        }
      }
      if (!is.null(constraint)) {
        return(strenv$numpyro$param(name, init_val, constraint = constraint))
      }
      return(strenv$numpyro$param(name, init_val))
    }
    sample_fxn()
  }
  # ReZero residual gates (Issue 1): treat as deterministic learned scalars rather
  # than HalfNormal latents. Sampling the gate from HalfNormal(gate_sd_scale) left
  # the gates at random positive init and, for small gate_sd_scale, let the
  # variational posterior collapse toward zero -- starving gradients through the
  # residual stream. A plain numpyro.param initialized to a small positive constant
  # restores standard-ReZero semantics (gate starts small, grows toward ~1) and
  # removes the posterior-collapse failure mode. Gates are nuisance scalars, so the
  # loss of posterior uncertainty on them is acceptable.
  #
  # Init from the depth-aware policy scale (gate_sd_scale = 0.1 * sqrt(2/depth));
  # this equals the historical hardcoded 0.1 at the default depth 2 but restores
  # the intended attenuation of residual-branch injections for deeper models,
  # keeping residual-stream variance bounded with depth.
  p2d_gate_init_value <- as.numeric(gate_sd_scale)
  if (length(p2d_gate_init_value) != 1L ||
      is.na(p2d_gate_init_value) ||
      !is.finite(p2d_gate_init_value) ||
      p2d_gate_init_value <= 0) {
    p2d_gate_init_value <- 0.1
  }
  p2d_fixed_gate_value <- function(init_value = p2d_gate_init_value) {
    strenv$jnp$array(as.numeric(init_value), dtype = ddtype_)
  }
  p2d_deterministic_gate <- function(name, init_value = p2d_gate_init_value) {
    init_val <- p2d_fixed_gate_value(init_value)
    if (!is.null(p2d_constraint_positive)) {
      return(strenv$numpyro$param(name, init_val, constraint = p2d_constraint_positive))
    }
    strenv$numpyro$param(name, init_val)
  }
  sample_loc_scale <- function(name, scale, shape_tuple) {
    if (isTRUE(manual_noncentered_loc_scale)) {
      z <- strenv$numpyro$sample(
        paste0(name, "_z"),
        strenv$numpyro$distributions$Normal(0., 1.)$expand(shape_tuple)
      )
      strenv$numpyro$deterministic(name, scale * z)
    } else {
      strenv$numpyro$sample(
        name,
        strenv$numpyro$distributions$Normal(0., scale)$expand(shape_tuple)
      )
    }
  }

  ordinal_thresholds_from_raw_jnp <- function(raw) {
    if (is.null(raw)) {
      return(NULL)
    }
    n_thr <- ai(raw$shape[[2]])
    if (n_thr <= 1L) {
      return(raw)
    }
    first <- strenv$jnp$take(raw, ai(0L), axis = 1L)
    rest <- strenv$jnp$take(raw, strenv$jnp$arange(ai(1L), n_thr), axis = 1L)
    increments <- strenv$jax$nn$softplus(rest) +
      strenv$jnp$array(1e-4, dtype = raw$dtype %||% ddtype_)
    values <- strenv$jnp$concatenate(
      list(strenv$jnp$reshape(first, list(-1L, 1L)), increments),
      axis = 1L
    )
    strenv$jnp$cumsum(values, axis = 1L)
  }

  sample_shared_transformer_params <- function(D_local, pairwise = FALSE) {
    output_only_mode <- identical(tolower(as.character(uncertainty_scope)), "output")

    tau_context <- if (isTRUE(output_only_mode)) {
      as.numeric(context_embed_sd_scale)
    } else {
      strenv$numpyro$sample(
        "tau_context",
        strenv$numpyro$distributions$HalfNormal(as.numeric(context_embed_sd_scale))
      )
    }
    tau_choice <- tau_context * (choice_embed_sd_scale / context_embed_sd_scale)
    tau_sep <- tau_context * (sep_embed_sd_scale / context_embed_sd_scale)
    tau_segment <- tau_context * (segment_embed_sd_scale / context_embed_sd_scale)

    E_factor_fused_base <- NULL
    W_factor_fuse_1 <- NULL
    b_factor_fuse_1 <- NULL
    W_factor_fuse_2 <- NULL
    b_factor_fuse_2 <- NULL
    if (identical(factor_tokenization, "fused")) {
      factor_base_shape <- reticulate::tuple(ModelDims)
      factor_fuse_input_dim <- ai(4L * as.integer(ModelDims))
      factor_fuse_hidden_dim <- ai(2L * as.integer(ModelDims))
      factor_fuse_gate_dim <- ai(2L * factor_fuse_hidden_dim)
      factor_fuse_w1_shape <- reticulate::tuple(factor_fuse_input_dim, factor_fuse_gate_dim)
      factor_fuse_b1_shape <- reticulate::tuple(factor_fuse_gate_dim)
      factor_fuse_w2_shape <- reticulate::tuple(factor_fuse_hidden_dim, ModelDims)
      factor_fuse_b2_shape <- reticulate::tuple(ModelDims)
      factor_fuse_w1_sd <- tau_context / sqrt(as.numeric(factor_fuse_input_dim))
      factor_fuse_w2_sd <- tau_context / sqrt(as.numeric(factor_fuse_hidden_dim))
      factor_fuse_bias_sd <- 0.05 * tau_context
      E_factor_fused_base <- p2d(
        name = "E_factor_fused_base",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_factor_fused_base",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = factor_base_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_factor_fused_base", tau_context, factor_base_shape)
        }
      )
      W_factor_fuse_1 <- p2d(
        name = "W_factor_fuse_1",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_factor_fuse_1",
            strenv$numpyro$distributions$Normal(0., factor_fuse_w1_sd),
            sample_shape = factor_fuse_w1_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_factor_fuse_1", factor_fuse_w1_sd, factor_fuse_w1_shape)
        }
      )
      b_factor_fuse_1 <- p2d(
        name = "b_factor_fuse_1",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "b_factor_fuse_1",
            strenv$numpyro$distributions$Normal(0., factor_fuse_bias_sd),
            sample_shape = factor_fuse_b1_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("b_factor_fuse_1", 0., factor_fuse_b1_shape)
        }
      )
      W_factor_fuse_2 <- p2d(
        name = "W_factor_fuse_2",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_factor_fuse_2",
            strenv$numpyro$distributions$Normal(0., factor_fuse_w2_sd),
            sample_shape = factor_fuse_w2_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_factor_fuse_2", factor_fuse_w2_sd, factor_fuse_w2_shape)
        }
      )
      b_factor_fuse_2 <- p2d(
        name = "b_factor_fuse_2",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "b_factor_fuse_2",
            strenv$numpyro$distributions$Normal(0., factor_fuse_bias_sd),
            sample_shape = factor_fuse_b2_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("b_factor_fuse_2", 0., factor_fuse_b2_shape)
        }
      )
    }

    E_party <- NULL
    if (isTRUE(has_candidate_group_context)) {
      E_party_shape <- reticulate::tuple(ai(n_party_levels), ModelDims)
      E_party <- p2d(
        name = "E_party",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_party",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_party_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_party", tau_context, E_party_shape)
        }
      )
    }

    E_rel <- NULL
    if (isTRUE(has_relation_context)) {
      E_rel_shape <- reticulate::tuple(ai(n_rel_levels), ModelDims)
      E_rel <- p2d(
        name = "E_rel",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_rel",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_rel_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_rel", tau_context, E_rel_shape)
        }
      )
    }

    E_resp_party <- NULL
    if (isTRUE(has_respondent_group_context)) {
      E_resp_party_shape <- reticulate::tuple(ai(n_resp_party_levels), ModelDims)
      E_resp_party <- p2d(
        name = "E_resp_party",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_resp_party",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_resp_party_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_resp_party", tau_context, E_resp_party_shape)
        }
      )
    }

    E_token_family_shape <- reticulate::tuple(ai(length(token_family_levels)), ModelDims)
    E_token_family <- p2d(
      name = "E_token_family",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "E_token_family",
          strenv$numpyro$distributions$Normal(0., tau_context),
          sample_shape = E_token_family_shape
        )
      },
      init_fxn = function() {
        p2d_init_normal("E_token_family", tau_context, E_token_family_shape)
      }
    )

    E_experiment <- NULL
    if (n_experiment_levels > 0L &&
        experiment_token_mode %in% c("legacy_id", "hybrid")) {
      E_experiment_shape <- reticulate::tuple(ai(n_experiment_levels), ModelDims)
      E_experiment <- p2d(
        name = "E_experiment",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_experiment",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_experiment_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_experiment", tau_context, E_experiment_shape)
        }
      )
    }

    E_stage <- NULL
    E_matchup <- NULL
    if (isTRUE(pairwise) && isTRUE(stage_context_enabled)) {
      E_stage_shape <- reticulate::tuple(ai(n_resp_party_levels), ai(2L), ModelDims)
      E_stage <- p2d(
        name = "E_stage",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_stage",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = E_stage_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_stage", tau_context, E_stage_shape)
        }
      )
      if (isTRUE(use_matchup_token)) {
        E_matchup_shape <- reticulate::tuple(ai(n_matchup_levels), ModelDims)
        E_matchup <- p2d(
          name = "E_matchup",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "E_matchup",
              strenv$numpyro$distributions$Normal(0., tau_context),
              sample_shape = E_matchup_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal("E_matchup", tau_context, E_matchup_shape)
          }
        )
      }
    }

    E_choice_shape <- reticulate::tuple(ModelDims)
    E_choice <- p2d(
      name = "E_choice",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "E_choice",
          strenv$numpyro$distributions$Normal(0., tau_choice),
          sample_shape = E_choice_shape
        )
      },
      init_fxn = function() {
        p2d_init_normal("E_choice", 0., E_choice_shape)
      }
    )

    E_respondent_cls <- NULL
    E_candidate_cls <- NULL
    if (low_rank_interaction_rank > 0L) {
      E_respondent_cls <- p2d(
        name = "E_respondent_cls",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_respondent_cls",
            strenv$numpyro$distributions$Normal(0., tau_choice),
            sample_shape = E_choice_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_respondent_cls", 0., E_choice_shape)
        }
      )
      E_candidate_cls <- p2d(
        name = "E_candidate_cls",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_candidate_cls",
            strenv$numpyro$distributions$Normal(0., tau_choice),
            sample_shape = E_choice_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_candidate_cls", 0., E_choice_shape)
        }
      )
    }

    E_sep <- NULL
    E_segment <- NULL
    if (isTRUE(pairwise) && isTRUE(use_cross_encoder)) {
      E_sep_shape <- reticulate::tuple(ModelDims)
      E_sep <- p2d(
        name = "E_sep",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_sep",
            strenv$numpyro$distributions$Normal(0., tau_sep),
            sample_shape = E_sep_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_sep", 0., E_sep_shape)
        }
      )

      E_segment_delta_shape <- reticulate::tuple(ModelDims)
      E_segment_delta <- p2d(
        name = "E_segment_delta",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_segment_delta",
            strenv$numpyro$distributions$Normal(0., tau_segment),
            sample_shape = E_segment_delta_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_segment_delta", tau_segment, E_segment_delta_shape)
        }
      )
      E_segment <- strenv$numpyro$deterministic(
        "E_segment",
        neural_build_symmetric_segment_embeddings(E_segment_delta)
      )
    }

    W_factor_name_text <- NULL
    W_level_name_text <- NULL
    W_factor_struct <- NULL
    W_level_struct <- NULL
    W_covariate_name_text <- NULL
    W_experiment_text <- NULL
    W_place_context <- NULL
    W_time_context <- NULL
    if (identical(factor_tokenization, "fused") && factor_struct_dim > 0L) {
      factor_struct_shape <- reticulate::tuple(ai(factor_struct_dim), ModelDims)
      factor_struct_sd <- tau_context / sqrt(as.numeric(factor_struct_dim))
      W_factor_struct <- p2d(
        name = "W_factor_struct",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_factor_struct",
            strenv$numpyro$distributions$Normal(0., factor_struct_sd),
            sample_shape = factor_struct_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_factor_struct", factor_struct_sd, factor_struct_shape)
        }
      )
    }
    if (identical(factor_tokenization, "fused") && level_struct_dim > 0L) {
      level_struct_shape <- reticulate::tuple(ai(level_struct_dim), ModelDims)
      level_struct_sd <- tau_context / sqrt(as.numeric(level_struct_dim))
      W_level_struct <- p2d(
        name = "W_level_struct",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_level_struct",
            strenv$numpyro$distributions$Normal(0., level_struct_sd),
            sample_shape = level_struct_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_level_struct", level_struct_sd, level_struct_shape)
        }
      )
    }
    if (text_semantic_dim > 0L) {
      text_proj_shape <- reticulate::tuple(ai(text_semantic_dim), ModelDims)
      text_proj_sd <- tau_context / sqrt(as.numeric(text_semantic_dim))
      W_factor_name_text <- p2d(
        name = "W_factor_name_text",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_factor_name_text",
            strenv$numpyro$distributions$Normal(0., text_proj_sd),
            sample_shape = text_proj_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_factor_name_text", text_proj_sd, text_proj_shape)
        }
      )
      W_level_name_text <- p2d(
        name = "W_level_name_text",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_level_name_text",
            strenv$numpyro$distributions$Normal(0., text_proj_sd),
            sample_shape = text_proj_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_level_name_text", text_proj_sd, text_proj_shape)
        }
      )
      W_covariate_name_text <- p2d(
        name = "W_covariate_name_text",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_covariate_name_text",
            strenv$numpyro$distributions$Normal(0., text_proj_sd),
            sample_shape = text_proj_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_covariate_name_text", text_proj_sd, text_proj_shape)
        }
      )
      if (experiment_token_mode %in% c("description", "hybrid") &&
          !is.null(experiment_description_text)) {
        W_experiment_text <- p2d(
          name = "W_experiment_text",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "W_experiment_text",
              strenv$numpyro$distributions$Normal(0., text_proj_sd),
              sample_shape = text_proj_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal("W_experiment_text", text_proj_sd, text_proj_shape)
          }
        )
      }
    }
    if (isTRUE(time_context_enabled) && time_context_dim > 0L) {
      time_context_shape <- reticulate::tuple(ai(time_context_dim), ModelDims)
      time_context_sd <- tau_context / sqrt(as.numeric(time_context_dim))
      W_time_context <- p2d(
        name = "W_time_context",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_time_context",
            strenv$numpyro$distributions$Normal(0., time_context_sd),
            sample_shape = time_context_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_time_context", time_context_sd, time_context_shape)
        }
      )
    }
    if (isTRUE(place_context_enabled) && place_context_dim > 0L) {
      place_context_shape <- reticulate::tuple(ai(place_context_dim), ModelDims)
      place_context_sd <- tau_context / sqrt(as.numeric(place_context_dim))
      W_place_context <- p2d(
        name = "W_place_context",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_place_context",
            strenv$numpyro$distributions$Normal(0., place_context_sd),
            sample_shape = place_context_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_place_context", place_context_sd, place_context_shape)
        }
      )
    }

    E_covariate_fused_base <- NULL
    E_covariate_missing <- NULL
    W_covariate_fuse_1 <- NULL
    b_covariate_fuse_1 <- NULL
    W_covariate_fuse_2 <- NULL
    b_covariate_fuse_2 <- NULL
    W_covariate_value_text <- NULL
    W_covariate_value_shared <- NULL
    W_covariate_value_basis <- NULL
    W_covariate_value_conditioner_1 <- NULL
    b_covariate_value_conditioner_1 <- NULL
    W_covariate_value_conditioner_2 <- NULL
    b_covariate_value_conditioner_2 <- NULL
    if (n_resp_covariates > 0L) {
      if (!identical(covariate_value_encoding, "shared_projection")) {
        stop("'covariate_value_encoding' must be 'shared_projection'.", call. = FALSE)
      }
      covariate_base_shape <- reticulate::tuple(ModelDims)
      covariate_fuse_input_dim <- ai(
        2L * as.integer(ModelDims) + length(neural_covariate_value_metadata_names())
      )
      covariate_fuse_hidden_dim <- ai(2L * as.integer(ModelDims))
      covariate_fuse_gate_dim <- ai(2L * covariate_fuse_hidden_dim)
      covariate_fuse_w1_shape <- reticulate::tuple(
        covariate_fuse_input_dim,
        covariate_fuse_gate_dim
      )
      covariate_fuse_b1_shape <- reticulate::tuple(covariate_fuse_gate_dim)
      covariate_fuse_w2_shape <- reticulate::tuple(covariate_fuse_hidden_dim, ModelDims)
      covariate_fuse_b2_shape <- reticulate::tuple(ModelDims)
      covariate_fuse_w1_sd <- tau_context / sqrt(as.numeric(covariate_fuse_input_dim))
      covariate_fuse_w2_sd <- tau_context / sqrt(as.numeric(covariate_fuse_hidden_dim))
      covariate_fuse_bias_sd <- 0.05 * tau_context

      E_covariate_fused_base <- p2d(
        name = "E_covariate_fused_base",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_covariate_fused_base",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = covariate_base_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_covariate_fused_base", tau_context, covariate_base_shape)
        }
      )
      E_covariate_missing <- p2d(
        name = "E_covariate_missing",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "E_covariate_missing",
            strenv$numpyro$distributions$Normal(0., tau_context),
            sample_shape = covariate_base_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("E_covariate_missing", tau_context, covariate_base_shape)
        }
      )
      W_covariate_fuse_1 <- p2d(
        name = "W_covariate_fuse_1",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_covariate_fuse_1",
            strenv$numpyro$distributions$Normal(0., covariate_fuse_w1_sd),
            sample_shape = covariate_fuse_w1_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_covariate_fuse_1", covariate_fuse_w1_sd, covariate_fuse_w1_shape)
        }
      )
      b_covariate_fuse_1 <- p2d(
        name = "b_covariate_fuse_1",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "b_covariate_fuse_1",
            strenv$numpyro$distributions$Normal(0., covariate_fuse_bias_sd),
            sample_shape = covariate_fuse_b1_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("b_covariate_fuse_1", 0., covariate_fuse_b1_shape)
        }
      )
      W_covariate_fuse_2 <- p2d(
        name = "W_covariate_fuse_2",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_covariate_fuse_2",
            strenv$numpyro$distributions$Normal(0., covariate_fuse_w2_sd),
            sample_shape = covariate_fuse_w2_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_covariate_fuse_2", covariate_fuse_w2_sd, covariate_fuse_w2_shape)
        }
      )
      b_covariate_fuse_2 <- p2d(
        name = "b_covariate_fuse_2",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "b_covariate_fuse_2",
            strenv$numpyro$distributions$Normal(0., covariate_fuse_bias_sd),
            sample_shape = covariate_fuse_b2_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("b_covariate_fuse_2", 0., covariate_fuse_b2_shape)
        }
      )
      if (text_semantic_dim > 0L && !is.null(covariate_value_text)) {
        value_text_proj_shape <- reticulate::tuple(ai(text_semantic_dim), ModelDims)
        value_text_proj_sd <- tau_context / sqrt(as.numeric(max(1L, text_semantic_dim)))
        W_covariate_value_text <- p2d(
          name = "W_covariate_value_text",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "W_covariate_value_text",
              strenv$numpyro$distributions$Normal(0., value_text_proj_sd),
              sample_shape = value_text_proj_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal("W_covariate_value_text", value_text_proj_sd, value_text_proj_shape)
          }
        )
      }
      if (identical(shared_projection_value_encoder, "name_dist_moe")) {
        basis_shape <- reticulate::tuple(
          ai(neural_covariate_value_experts()),
          ai(neural_covariate_value_basis_dim()),
          ModelDims
        )
        conditioner_input_dim <- ai(ModelDims + length(neural_covariate_value_metadata_names()))
        conditioner_hidden_dim <- ai(min(as.integer(ModelDims), 64L))
        conditioner_w1_shape <- reticulate::tuple(conditioner_input_dim, conditioner_hidden_dim)
        conditioner_b1_shape <- reticulate::tuple(conditioner_hidden_dim)
        conditioner_w2_shape <- reticulate::tuple(
          conditioner_hidden_dim,
          ai(neural_covariate_value_experts())
        )
        conditioner_b2_shape <- reticulate::tuple(ai(neural_covariate_value_experts()))
        basis_sd <- tau_context / sqrt(as.numeric(neural_covariate_value_basis_dim()))
        conditioner_w1_sd <- tau_context / sqrt(as.numeric(conditioner_input_dim))
        conditioner_w2_sd <- tau_context / sqrt(as.numeric(conditioner_hidden_dim))
        conditioner_bias_sd <- 0.05 * tau_context

        W_covariate_value_basis <- p2d(
          name = "W_covariate_value_basis",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "W_covariate_value_basis",
              strenv$numpyro$distributions$Normal(0., basis_sd),
              sample_shape = basis_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal("W_covariate_value_basis", basis_sd, basis_shape)
          }
        )
        W_covariate_value_conditioner_1 <- p2d(
          name = "W_covariate_value_conditioner_1",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "W_covariate_value_conditioner_1",
              strenv$numpyro$distributions$Normal(0., conditioner_w1_sd),
              sample_shape = conditioner_w1_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal(
              "W_covariate_value_conditioner_1",
              conditioner_w1_sd,
              conditioner_w1_shape
            )
          }
        )
        b_covariate_value_conditioner_1 <- p2d(
          name = "b_covariate_value_conditioner_1",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "b_covariate_value_conditioner_1",
              strenv$numpyro$distributions$Normal(0., conditioner_bias_sd),
              sample_shape = conditioner_b1_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal(
              "b_covariate_value_conditioner_1",
              0.,
              conditioner_b1_shape
            )
          }
        )
        W_covariate_value_conditioner_2 <- p2d(
          name = "W_covariate_value_conditioner_2",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "W_covariate_value_conditioner_2",
              strenv$numpyro$distributions$Normal(0., conditioner_w2_sd),
              sample_shape = conditioner_w2_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal(
              "W_covariate_value_conditioner_2",
              conditioner_w2_sd,
              conditioner_w2_shape
            )
          }
        )
        b_covariate_value_conditioner_2 <- p2d(
          name = "b_covariate_value_conditioner_2",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "b_covariate_value_conditioner_2",
              strenv$numpyro$distributions$Normal(0., conditioner_bias_sd),
              sample_shape = conditioner_b2_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal(
              "b_covariate_value_conditioner_2",
              0.,
              conditioner_b2_shape
            )
          }
        )
      } else {
        value_proj_shape <- reticulate::tuple(ai(1L), ModelDims)
        value_proj_sd <- tau_context
        W_covariate_value_shared <- p2d(
          name = "W_covariate_value_shared",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "W_covariate_value_shared",
              strenv$numpyro$distributions$Normal(0., value_proj_sd),
              sample_shape = value_proj_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal("W_covariate_value_shared", value_proj_sd, value_proj_shape)
          }
        )
      }
    }

    layer_params <- list()
    attnres_query_sd_scale <- as.numeric(residual_weight_sd_scale)
    for (l_ in 1L:ModelDepth) {
      tau_w_prior <- as.numeric(residual_weight_sd_scale)
      tau_w_l <- if (isTRUE(output_only_mode)) {
        tau_w_prior
      } else {
        strenv$numpyro$sample(
          paste0("tau_w_", l_),
          strenv$numpyro$distributions$HalfNormal(tau_w_prior)
        )
      }

      pseudo_query_attn_l <- NULL
      pseudo_query_ff_l <- NULL
      alpha_attn_l <- NULL
      alpha_ff_l <- NULL
      if (isTRUE(use_full_attn_residual)) {
        pseudo_query_attn_name <- paste0("pseudo_query_attn_l", l_)
        pseudo_query_shape <- reticulate::tuple(ModelDims)
        pseudo_query_attn_l <- p2d(
          name = pseudo_query_attn_name,
          sample_fxn = function() {
            strenv$numpyro$sample(
              pseudo_query_attn_name,
              strenv$numpyro$distributions$Normal(0., attnres_query_sd_scale),
              sample_shape = pseudo_query_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal(pseudo_query_attn_name, 0., pseudo_query_shape)
          }
        )

        pseudo_query_ff_name <- paste0("pseudo_query_ff_l", l_)
        pseudo_query_ff_l <- p2d(
          name = pseudo_query_ff_name,
          sample_fxn = function() {
            strenv$numpyro$sample(
              pseudo_query_ff_name,
              strenv$numpyro$distributions$Normal(0., attnres_query_sd_scale),
              sample_shape = pseudo_query_shape
            )
          },
          init_fxn = function() {
            p2d_init_normal(pseudo_query_ff_name, 0., pseudo_query_shape)
          }
        )
      } else {
        alpha_attn_name <- paste0("alpha_attn_l", l_)
        alpha_attn_l <- p2d_deterministic_gate(alpha_attn_name)

        alpha_ff_name <- paste0("alpha_ff_l", l_)
        alpha_ff_l <- p2d_deterministic_gate(alpha_ff_name)
      }

      RMS_attn_name <- paste0("RMS_attn_l", l_)
      RMS_attn_shape <- reticulate::tuple(ModelDims)
      RMS_attn_l <- p2d(
        name = RMS_attn_name,
        sample_fxn = function() {
          strenv$numpyro$sample(
            RMS_attn_name,
            strenv$numpyro$distributions$LogNormal(0., RMS_scale),
            sample_shape = RMS_attn_shape
          )
        },
        init_fxn = function() {
          p2d_init_lognormal(RMS_attn_name, RMS_scale, RMS_attn_shape)
        },
        constraint = p2d_constraint_positive
      )

      RMS_ff_name <- paste0("RMS_ff_l", l_)
      RMS_ff_shape <- reticulate::tuple(ModelDims)
      RMS_ff_l <- p2d(
        name = RMS_ff_name,
        sample_fxn = function() {
          strenv$numpyro$sample(
            RMS_ff_name,
            strenv$numpyro$distributions$LogNormal(0., RMS_scale),
            sample_shape = RMS_ff_shape
          )
        },
        init_fxn = function() {
          p2d_init_lognormal(RMS_ff_name, RMS_scale, RMS_ff_shape)
        },
        constraint = p2d_constraint_positive
      )

      RMS_q_l <- NULL
      RMS_k_l <- NULL
      if (isTRUE(qk_norm_enabled)) {
        RMS_q_name <- paste0("RMS_q_l", l_)
        RMS_head_shape <- reticulate::tuple(head_dim)
        RMS_q_l <- p2d(
          name = RMS_q_name,
          sample_fxn = function() {
            strenv$numpyro$sample(
              RMS_q_name,
              strenv$numpyro$distributions$LogNormal(0., qk_rms_scale),
              sample_shape = RMS_head_shape
            )
          },
          init_fxn = function() {
            p2d_init_lognormal(RMS_q_name, qk_rms_scale, RMS_head_shape)
          },
          constraint = p2d_constraint_positive
        )

        RMS_k_name <- paste0("RMS_k_l", l_)
        RMS_k_l <- p2d(
          name = RMS_k_name,
          sample_fxn = function() {
            strenv$numpyro$sample(
              RMS_k_name,
              strenv$numpyro$distributions$LogNormal(0., qk_rms_scale),
              sample_shape = RMS_head_shape
            )
          },
          init_fxn = function() {
            p2d_init_lognormal(RMS_k_name, qk_rms_scale, RMS_head_shape)
          },
          constraint = p2d_constraint_positive
        )
      }

      W_q_name <- paste0("W_q_l", l_)
      W_q_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_q_l <- p2d(
        name = W_q_name,
        sample_fxn = function() {
          sample_loc_scale(W_q_name, tau_w_l, W_q_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_q_name, tau_w_l, W_q_shape)
        }
      )

      W_k_name <- paste0("W_k_l", l_)
      W_k_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_k_l <- p2d(
        name = W_k_name,
        sample_fxn = function() {
          sample_loc_scale(W_k_name, tau_w_l, W_k_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_k_name, tau_w_l, W_k_shape)
        }
      )

      W_v_name <- paste0("W_v_l", l_)
      W_v_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_v_l <- p2d(
        name = W_v_name,
        sample_fxn = function() {
          sample_loc_scale(W_v_name, tau_w_l, W_v_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_v_name, tau_w_l, W_v_shape)
        }
      )

      W_o_name <- paste0("W_o_l", l_)
      W_o_shape <- reticulate::tuple(ModelDims, ModelDims)
      W_o_l <- p2d(
        name = W_o_name,
        sample_fxn = function() {
          sample_loc_scale(W_o_name, tau_w_l, W_o_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_o_name, tau_w_l, W_o_shape)
        }
      )

      W_ff1_name <- paste0("W_ff1_l", l_)
      W_ff1_shape <- reticulate::tuple(ModelDims, ai(2L * FFDim))
      W_ff1_l <- p2d(
        name = W_ff1_name,
        sample_fxn = function() {
          sample_loc_scale(W_ff1_name, tau_w_l, W_ff1_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_ff1_name, tau_w_l, W_ff1_shape)
        }
      )

      W_ff2_name <- paste0("W_ff2_l", l_)
      W_ff2_shape <- reticulate::tuple(FFDim, ModelDims)
      ff2_fan_scale <- as.numeric(sqrt(as.numeric(ModelDims) / max(1, as.numeric(FFDim))))
      tau_w_ff2_l <- tau_w_l * ff2_fan_scale
      W_ff2_l <- p2d(
        name = W_ff2_name,
        sample_fxn = function() {
          sample_loc_scale(W_ff2_name, tau_w_ff2_l, W_ff2_shape)
        },
        init_fxn = function() {
          p2d_init_normal(W_ff2_name, tau_w_ff2_l, W_ff2_shape)
        }
      )

      layer_params[[paste0("W_q_l", l_)]] <- W_q_l
      layer_params[[paste0("W_k_l", l_)]] <- W_k_l
      layer_params[[paste0("W_v_l", l_)]] <- W_v_l
      layer_params[[paste0("W_o_l", l_)]] <- W_o_l
      layer_params[[paste0("W_ff1_l", l_)]] <- W_ff1_l
      layer_params[[paste0("W_ff2_l", l_)]] <- W_ff2_l
      layer_params[[paste0("RMS_attn_l", l_)]] <- RMS_attn_l
      if (!is.null(pseudo_query_attn_l)) {
        layer_params[[paste0("pseudo_query_attn_l", l_)]] <- pseudo_query_attn_l
      }
      if (!is.null(pseudo_query_ff_l)) {
        layer_params[[paste0("pseudo_query_ff_l", l_)]] <- pseudo_query_ff_l
      }
      if (!is.null(RMS_q_l)) {
        layer_params[[paste0("RMS_q_l", l_)]] <- RMS_q_l
      }
      if (!is.null(RMS_k_l)) {
        layer_params[[paste0("RMS_k_l", l_)]] <- RMS_k_l
      }
      layer_params[[paste0("RMS_ff_l", l_)]] <- RMS_ff_l
      if (!is.null(alpha_attn_l)) {
        layer_params[[paste0("alpha_attn_l", l_)]] <- alpha_attn_l
      }
      if (!is.null(alpha_ff_l)) {
        layer_params[[paste0("alpha_ff_l", l_)]] <- alpha_ff_l
      }
    }
    if (!isTRUE(use_full_attn_residual)) {
      layer_params <- neural_stack_standard_transformer_layers(
        layer_params,
        model_depth = ModelDepth,
        drop_legacy = TRUE
      )
    }

    alpha_cross <- NULL
    RMS_cross <- NULL
    RMS_merge_cross <- NULL
    RMS_q_cross <- NULL
    RMS_k_cross <- NULL
    W_q_cross <- NULL
    W_k_cross <- NULL
    W_v_cross <- NULL
    W_o_cross <- NULL
    if (isTRUE(pairwise) && isTRUE(use_cross_attn)) {
      tau_cross_attn <- if (isTRUE(output_only_mode)) {
        as.numeric(residual_weight_sd_scale)
      } else {
        strenv$numpyro$sample(
          "tau_cross_attn",
          strenv$numpyro$distributions$HalfNormal(as.numeric(residual_weight_sd_scale))
        )
      }

      alpha_cross <- p2d_deterministic_gate("alpha_cross")

      RMS_cross <- p2d(
        name = "RMS_cross",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "RMS_cross",
            strenv$numpyro$distributions$LogNormal(0., RMS_scale),
            sample_shape = reticulate::tuple(ModelDims)
          )
        },
        init_fxn = function() {
          p2d_init_lognormal("RMS_cross", RMS_scale, reticulate::tuple(ModelDims))
        },
        constraint = p2d_constraint_positive
      )

      RMS_merge_cross <- p2d(
        name = "RMS_merge_cross",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "RMS_merge_cross",
            strenv$numpyro$distributions$LogNormal(0., RMS_scale),
            sample_shape = reticulate::tuple(ModelDims)
          )
        },
        init_fxn = function() {
          p2d_init_lognormal("RMS_merge_cross", RMS_scale, reticulate::tuple(ModelDims))
        },
        constraint = p2d_constraint_positive
      )

      if (isTRUE(qk_norm_enabled)) {
        RMS_q_cross <- p2d(
          name = "RMS_q_cross",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "RMS_q_cross",
              strenv$numpyro$distributions$LogNormal(0., qk_rms_scale),
              sample_shape = reticulate::tuple(head_dim)
            )
          },
          init_fxn = function() {
            p2d_init_lognormal("RMS_q_cross", qk_rms_scale, reticulate::tuple(head_dim))
          },
          constraint = p2d_constraint_positive
        )
        RMS_k_cross <- p2d(
          name = "RMS_k_cross",
          sample_fxn = function() {
            strenv$numpyro$sample(
              "RMS_k_cross",
              strenv$numpyro$distributions$LogNormal(0., qk_rms_scale),
              sample_shape = reticulate::tuple(head_dim)
            )
          },
          init_fxn = function() {
            p2d_init_lognormal("RMS_k_cross", qk_rms_scale, reticulate::tuple(head_dim))
          },
          constraint = p2d_constraint_positive
        )
      }

      W_q_cross <- p2d(
        name = "W_q_cross",
        sample_fxn = function() {
          sample_loc_scale("W_q_cross", tau_cross_attn,
                           reticulate::tuple(ModelDims, ModelDims))
        },
        init_fxn = function() {
          p2d_init_normal("W_q_cross", tau_cross_attn,
                          reticulate::tuple(ModelDims, ModelDims))
        }
      )
      W_k_cross <- p2d(
        name = "W_k_cross",
        sample_fxn = function() {
          sample_loc_scale("W_k_cross", tau_cross_attn,
                           reticulate::tuple(ModelDims, ModelDims))
        },
        init_fxn = function() {
          p2d_init_normal("W_k_cross", tau_cross_attn,
                          reticulate::tuple(ModelDims, ModelDims))
        }
      )
      W_v_cross <- p2d(
        name = "W_v_cross",
        sample_fxn = function() {
          sample_loc_scale("W_v_cross", tau_cross_attn,
                           reticulate::tuple(ModelDims, ModelDims))
        },
        init_fxn = function() {
          p2d_init_normal("W_v_cross", tau_cross_attn,
                          reticulate::tuple(ModelDims, ModelDims))
        }
      )
      W_o_cross <- p2d(
        name = "W_o_cross",
        sample_fxn = function() {
          sample_loc_scale("W_o_cross", tau_cross_attn,
                           reticulate::tuple(ModelDims, ModelDims))
        },
        init_fxn = function() {
          p2d_init_normal("W_o_cross", tau_cross_attn,
                          reticulate::tuple(ModelDims, ModelDims))
        }
      )
    }

    pseudo_query_final <- NULL
    if (isTRUE(use_full_attn_residual)) {
      pseudo_query_final_shape <- reticulate::tuple(ModelDims)
      pseudo_query_final <- p2d(
        name = "pseudo_query_final",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "pseudo_query_final",
            strenv$numpyro$distributions$Normal(0., attnres_query_sd_scale),
            sample_shape = pseudo_query_final_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("pseudo_query_final", 0., pseudo_query_final_shape)
        }
      )
    }

    RMS_final_shape <- reticulate::tuple(ModelDims)
    RMS_final <- p2d(
      name = "RMS_final",
      sample_fxn = function() {
        strenv$numpyro$sample(
          "RMS_final",
          strenv$numpyro$distributions$LogNormal(0., RMS_scale),
          sample_shape = RMS_final_shape
        )
      },
      init_fxn = function() {
        p2d_init_lognormal("RMS_final", RMS_scale, RMS_final_shape)
      },
      constraint = p2d_constraint_positive
    )
    output_dim <- if (isTRUE(universal_enabled)) ai(universal_global_out_dim) else nOutcomes
    tau_w_out <- strenv$numpyro$sample(
      "tau_w_out",
      strenv$numpyro$distributions$HalfNormal(as.numeric(weight_sd_scale))
    )
    W_out <- sample_loc_scale("W_out", tau_w_out,
                              reticulate::tuple(ModelDims, output_dim))
    W_add_out <- NULL
    if (identical(additive_utility_mode, "on")) {
      W_add_out <- sample_loc_scale(
        "W_add_out",
        tau_w_out,
        reticulate::tuple(ModelDims, output_dim)
      )
    }
    tau_b <- strenv$numpyro$sample(
      "tau_b",
      strenv$numpyro$distributions$HalfNormal(as.numeric(tau_b_scale))
    )
    b_out <- sample_loc_scale("b_out", tau_b, reticulate::tuple(output_dim))

    ordinal_threshold_raw <- NULL
    ordinal_thresholds <- NULL
    if (isTRUE(has_ordinal_likelihood)) {
      ordinal_threshold_shape <- reticulate::tuple(
        ai(ordinal_n_threshold_groups),
        ai(ordinal_max_thresholds)
      )
      ordinal_threshold_raw <- p2d(
        name = "ordinal_threshold_raw",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "ordinal_threshold_raw",
            strenv$numpyro$distributions$Normal(0., 1.)$expand(ordinal_threshold_shape)
          )
        },
        init_fxn = function() {
          p2d_init_normal("ordinal_threshold_raw", 0., ordinal_threshold_shape)
        },
        is_output_layer = TRUE
      )
      ordinal_thresholds <- strenv$numpyro$deterministic(
        "ordinal_thresholds",
        ordinal_thresholds_from_raw_jnp(ordinal_threshold_raw)
      )
    }

    M_cross <- NULL
    W_cross_out <- NULL
    if (isTRUE(pairwise) && isTRUE(use_cross_term)) {
      # Antisymmetric bilinear term enables opponent-dependent matchups.
      tau_cross <- if (isTRUE(output_only_mode)) {
        as.numeric(cross_weight_sd_scale)
      } else {
        strenv$numpyro$sample(
          "tau_cross",
          strenv$numpyro$distributions$HalfNormal(as.numeric(cross_weight_sd_scale))
        )
      }
      M_cross_shape <- reticulate::tuple(ModelDims, ModelDims)
      M_cross_raw <- p2d(
        name = "M_cross_raw",
        sample_fxn = function() {
          sample_loc_scale("M_cross_raw", tau_cross, M_cross_shape)
        },
        init_fxn = function() {
          p2d_init_normal("M_cross_raw", tau_cross, M_cross_shape)
        }
      )
      M_cross <- 0.5 * (M_cross_raw - strenv$jnp$transpose(M_cross_raw))
      M_cross <- strenv$numpyro$deterministic("M_cross", M_cross)
      W_cross_out <- strenv$numpyro$sample(
        "W_cross_out",
        strenv$numpyro$distributions$Normal(0., 0.25),
        sample_shape = reticulate::tuple(output_dim)
      )
    }

    alpha_rc <- NULL
    W_rc_r <- NULL
    W_rc_c <- NULL
    W_rc_out <- NULL
    if (low_rank_interaction_rank > 0L) {
      rc_rank <- ai(low_rank_interaction_rank)
      tau_rc <- if (isTRUE(output_only_mode)) {
        as.numeric(cross_weight_sd_scale)
      } else {
        strenv$numpyro$sample(
          "tau_rc",
          strenv$numpyro$distributions$HalfNormal(as.numeric(cross_weight_sd_scale))
        )
      }
      rc_projection_shape <- reticulate::tuple(ModelDims, rc_rank)
      W_rc_r <- p2d(
        name = "W_rc_r",
        sample_fxn = function() {
          sample_loc_scale("W_rc_r", tau_rc, rc_projection_shape)
        },
        init_fxn = function() {
          p2d_init_normal("W_rc_r", tau_rc, rc_projection_shape)
        }
      )
      W_rc_c <- p2d(
        name = "W_rc_c",
        sample_fxn = function() {
          sample_loc_scale("W_rc_c", tau_rc, rc_projection_shape)
        },
        init_fxn = function() {
          p2d_init_normal("W_rc_c", tau_rc, rc_projection_shape)
        }
      )
      rc_out_sd <- as.numeric(0.1 / sqrt(max(1L, low_rank_interaction_rank)))
      rc_out_shape <- reticulate::tuple(rc_rank, output_dim)
      W_rc_out <- p2d(
        name = "W_rc_out",
        sample_fxn = function() {
          strenv$numpyro$sample(
            "W_rc_out",
            strenv$numpyro$distributions$Normal(0., rc_out_sd),
            sample_shape = rc_out_shape
          )
        },
        init_fxn = function() {
          p2d_init_normal("W_rc_out", rc_out_sd, rc_out_shape)
        },
        is_output_layer = TRUE
      )
      alpha_rc <- p2d_deterministic_gate("alpha_rc")
    }

    sigma <- NULL
    if (likelihood == "normal" || isTRUE(universal_has_normal)) {
      sigma <- strenv$numpyro$sample(
        "sigma",
        strenv$numpyro$distributions$HalfNormal(as.numeric(sigma_prior_scale))
      )
      sigma <- neural_floor_sigma(sigma, dtype = ddtype_)
    }
    log_pairwise_bernoulli_logit_scale <- NULL
    if (isTRUE(learned_pairwise_bernoulli_logit_scale) &&
        isTRUE(pairwise) &&
        likelihood %in% c("bernoulli", "mixed")) {
      log_pairwise_bernoulli_logit_scale <- strenv$numpyro$sample(
        "log_pairwise_bernoulli_logit_scale",
        strenv$numpyro$distributions$Normal(
          0.,
          as.numeric(pairwise_bernoulli_logit_scale_prior_sd)
        )
      )
    }
    log_calibration_scale <- NULL
    if (isTRUE(calibration_control$enabled) &&
        identical(calibration_control$method %||% NULL, "logit_scale")) {
      log_calibration_scale <- strenv$numpyro$sample(
        "log_calibration_scale",
        strenv$numpyro$distributions$Normal(
          0.,
          as.numeric(calibration_control$prior_sd %||% 0.5)
        )
      )
    }

    params_view <- if (isTRUE(pairwise)) {
      list(
        E_party = E_party,
        E_rel = E_rel,
        E_resp_party = E_resp_party,
        E_token_family = E_token_family,
        E_stage = E_stage,
        E_choice = E_choice,
        E_sep = E_sep,
        E_segment = E_segment
      )
    } else {
      list(
        E_party = E_party,
        E_rel = E_rel,
        E_resp_party = E_resp_party,
        E_token_family = E_token_family,
        E_choice = E_choice
      )
    }
    if (!is.null(E_respondent_cls)) {
      params_view$E_respondent_cls <- E_respondent_cls
    }
    if (!is.null(E_candidate_cls)) {
      params_view$E_candidate_cls <- E_candidate_cls
    }
    if (!is.null(E_factor_fused_base)) {
      params_view$E_factor_fused_base <- E_factor_fused_base
      params_view$W_factor_fuse_1 <- W_factor_fuse_1
      params_view$b_factor_fuse_1 <- b_factor_fuse_1
      params_view$W_factor_fuse_2 <- W_factor_fuse_2
      params_view$b_factor_fuse_2 <- b_factor_fuse_2
    }
    if (!is.null(E_experiment)) {
      params_view$E_experiment <- E_experiment
    }
    if (isTRUE(pairwise) && isTRUE(use_matchup_token)) {
      params_view$E_matchup <- E_matchup
    }
    if (!is.null(W_factor_name_text)) {
      params_view$W_factor_name_text <- W_factor_name_text
    }
    if (!is.null(W_level_name_text)) {
      params_view$W_level_name_text <- W_level_name_text
    }
    if (!is.null(W_factor_struct)) {
      params_view$W_factor_struct <- W_factor_struct
    }
    if (!is.null(W_level_struct)) {
      params_view$W_level_struct <- W_level_struct
    }
    if (!is.null(W_covariate_name_text)) {
      params_view$W_covariate_name_text <- W_covariate_name_text
    }
    if (!is.null(W_experiment_text)) {
      params_view$W_experiment_text <- W_experiment_text
    }
    if (!is.null(W_place_context)) {
      params_view$W_place_context <- W_place_context
    }
    if (!is.null(W_time_context)) {
      params_view$W_time_context <- W_time_context
    }
    if (n_resp_covariates > 0L) {
      if (!is.null(E_covariate_fused_base)) {
        params_view$E_covariate_fused_base <- E_covariate_fused_base
        params_view$E_covariate_missing <- E_covariate_missing
        params_view$W_covariate_fuse_1 <- W_covariate_fuse_1
        params_view$b_covariate_fuse_1 <- b_covariate_fuse_1
        params_view$W_covariate_fuse_2 <- W_covariate_fuse_2
        params_view$b_covariate_fuse_2 <- b_covariate_fuse_2
      }
      if (!is.null(W_covariate_value_shared)) {
        params_view$W_covariate_value_shared <- W_covariate_value_shared
      }
      if (!is.null(W_covariate_value_text)) {
        params_view$W_covariate_value_text <- W_covariate_value_text
      }
      if (!is.null(W_covariate_value_basis)) {
        params_view$W_covariate_value_basis <- W_covariate_value_basis
        params_view$W_covariate_value_conditioner_1 <- W_covariate_value_conditioner_1
        params_view$b_covariate_value_conditioner_1 <- b_covariate_value_conditioner_1
        params_view$W_covariate_value_conditioner_2 <- W_covariate_value_conditioner_2
        params_view$b_covariate_value_conditioner_2 <- b_covariate_value_conditioner_2
      }
    }
    params_view <- c(params_view, layer_params)
    if (isTRUE(pairwise) && isTRUE(use_cross_attn)) {
      params_view$alpha_cross <- alpha_cross
      params_view$RMS_cross <- RMS_cross
      params_view$RMS_merge_cross <- RMS_merge_cross
      params_view$RMS_q_cross <- RMS_q_cross
      params_view$RMS_k_cross <- RMS_k_cross
      params_view$W_q_cross <- W_q_cross
      params_view$W_k_cross <- W_k_cross
      params_view$W_v_cross <- W_v_cross
      params_view$W_o_cross <- W_o_cross
    }
    if (!is.null(pseudo_query_final)) {
      params_view$pseudo_query_final <- pseudo_query_final
    }
    params_view$RMS_final <- RMS_final
    if (!is.null(W_add_out)) {
      params_view$W_add_out <- W_add_out
    }
    params_view$W_out <- W_out
    params_view$b_out <- b_out
    if (!is.null(ordinal_threshold_raw)) {
      params_view$ordinal_threshold_raw <- ordinal_threshold_raw
      params_view$ordinal_thresholds <- ordinal_thresholds
    }
    if (isTRUE(pairwise) && isTRUE(use_cross_term)) {
      params_view$M_cross <- M_cross
      params_view$W_cross_out <- W_cross_out
    }
    if (low_rank_interaction_rank > 0L) {
      params_view$alpha_rc <- alpha_rc
      params_view$W_rc_r <- W_rc_r
      params_view$W_rc_c <- W_rc_c
      params_view$W_rc_out <- W_rc_out
    }
    if (!is.null(log_pairwise_bernoulli_logit_scale)) {
      params_view$log_pairwise_bernoulli_logit_scale <- log_pairwise_bernoulli_logit_scale
      params_view$pairwise_bernoulli_logit_scale <- strenv$jnp$exp(
        log_pairwise_bernoulli_logit_scale
      )
    }
    if (!is.null(log_calibration_scale)) {
      params_view$log_calibration_scale <- log_calibration_scale
      params_view$calibration_scale <- strenv$jnp$exp(log_calibration_scale)
    }

    list(
      params_view = params_view,
      E_choice = E_choice,
      E_respondent_cls = E_respondent_cls,
      E_candidate_cls = E_candidate_cls,
      W_add_out = W_add_out,
      W_out = W_out,
      b_out = b_out,
      ordinal_threshold_raw = ordinal_threshold_raw,
      ordinal_thresholds = ordinal_thresholds,
      sigma = sigma,
      M_cross = M_cross,
      W_cross_out = W_cross_out,
      log_pairwise_bernoulli_logit_scale = log_pairwise_bernoulli_logit_scale,
      log_calibration_scale = log_calibration_scale
    )
  }

  universal_row_log_prob_jnp <- function(logits,
                                         Yb,
                                         likelihood_code_obs,
                                         n_outcomes_obs,
                                         sigma = NULL,
                                         experiment_index_obs = NULL,
                                         ordinal_thresholds = NULL) {
    out_dim <- ai(logits$shape[[2]])
    class_axis <- strenv$jnp$arange(out_dim)$astype(strenv$jnp$int32)
    class_mask <- strenv$jnp$less(
      strenv$jnp$reshape(class_axis, list(1L, out_dim)),
      strenv$jnp$reshape(n_outcomes_obs, list(-1L, 1L))
    )
    neg_large <- strenv$jnp$full(
      logits$shape,
      strenv$jnp$array(-1e9, dtype = ddtype_),
      dtype = ddtype_
    )
    masked_logits <- strenv$jnp$where(class_mask, logits, neg_large)
    y_numeric <- Yb$astype(ddtype_)
    y_int <- strenv$jnp$astype(strenv$jnp$round(y_numeric), strenv$jnp$int32)
    like_bern_mask <- strenv$jnp$equal(likelihood_code_obs, ai(0L))
    like_cat_mask <- strenv$jnp$equal(likelihood_code_obs, ai(1L))
    like_norm_mask <- strenv$jnp$equal(likelihood_code_obs, ai(2L))
    like_ord_mask <- strenv$jnp$equal(likelihood_code_obs, ai(3L))
    like_bern <- like_bern_mask$astype(ddtype_)
    like_cat <- like_cat_mask$astype(ddtype_)
    like_norm <- like_norm_mask$astype(ddtype_)
    like_ord <- like_ord_mask$astype(ddtype_)
    zeros_y <- strenv$jnp$zeros_like(y_numeric)
    zeros_i <- strenv$jnp$zeros_like(y_int)
    y_bern <- strenv$jnp$where(like_bern_mask, y_numeric, zeros_y)
    y_cat <- strenv$jnp$where(like_cat_mask, y_int, zeros_i)
    y_norm <- strenv$jnp$where(like_norm_mask, y_numeric, zeros_y)
    bern_logits <- strenv$jnp$take(logits, ai(0L), axis = 1L)
    bern_logp <- strenv$numpyro$distributions$Bernoulli(logits = bern_logits)$log_prob(y_bern)
    cat_logp <- strenv$numpyro$distributions$Categorical(logits = masked_logits)$log_prob(y_cat)
    mu <- strenv$jnp$take(logits, ai(0L), axis = 1L)
    sigma_use <- if (is.null(sigma)) {
      strenv$jnp$array(1., dtype = ddtype_)
    } else {
      neural_floor_sigma(sigma, dtype = ddtype_)
    }
    norm_logp <- strenv$numpyro$distributions$Normal(mu, sigma_use)$log_prob(y_norm)
    ord_logp <- strenv$jnp$zeros_like(y_numeric)
    if (!is.null(ordinal_thresholds)) {
      exp_idx <- if (is.null(experiment_index_obs)) {
        strenv$jnp$zeros_like(y_int)
      } else {
        experiment_index_obs
      }
      exp_idx <- strenv$jnp$maximum(exp_idx, ai(0L))
      n_threshold_groups <- ai(ordinal_thresholds$shape[[1]])
      exp_idx <- strenv$jnp$minimum(exp_idx, n_threshold_groups - ai(1L))
      cutpoints <- strenv$jnp$take(ordinal_thresholds, exp_idx, axis = 0L)
      eta <- strenv$jnp$take(logits, ai(0L), axis = 1L)
      cdf <- strenv$jax$nn$sigmoid(cutpoints - strenv$jnp$reshape(eta, list(-1L, 1L)))
      max_thr <- ai(cdf$shape[[2]])
      y_ord <- strenv$jnp$where(like_ord_mask, y_int, zeros_i)
      prev_idx <- strenv$jnp$maximum(y_ord - ai(1L), ai(0L))
      curr_idx <- strenv$jnp$minimum(y_ord, max_thr - ai(1L))
      prev_cdf <- strenv$jnp$take_along_axis(
        cdf,
        strenv$jnp$reshape(prev_idx, list(-1L, 1L)),
        axis = 1L
      )
      curr_cdf <- strenv$jnp$take_along_axis(
        cdf,
        strenv$jnp$reshape(curr_idx, list(-1L, 1L)),
        axis = 1L
      )
      prev_cdf <- strenv$jnp$reshape(prev_cdf, list(-1L))
      curr_cdf <- strenv$jnp$reshape(curr_cdf, list(-1L))
      first_class <- strenv$jnp$equal(y_ord, ai(0L))
      last_class <- strenv$jnp$greater_equal(y_ord, n_outcomes_obs - ai(1L))
      prev_cdf <- strenv$jnp$where(first_class, zeros_y, prev_cdf)
      curr_cdf <- strenv$jnp$where(last_class, strenv$jnp$ones_like(y_numeric), curr_cdf)
      ord_prob <- strenv$jnp$maximum(
        curr_cdf - prev_cdf,
        strenv$jnp$array(1e-12, dtype = ddtype_)
      )
      ord_logp <- strenv$jnp$log(ord_prob)
    }
    bern_term <- like_bern * bern_logp
    cat_term <- like_cat * cat_logp
    norm_term <- like_norm * norm_logp
    ord_term <- like_ord * ord_logp
    if (isTRUE(universal_family_logprob_normalize)) {
      # Normalize each family's per-row contribution by an in-batch scale
      # s = max(1, mean|log-prob| over that family's rows), held constant w.r.t.
      # gradients. Families whose log-probs are already O(1) (Bernoulli/Categorical)
      # are left unchanged; only families with large-magnitude log-probs (e.g. a
      # Normal head with small sigma) get downscaled, equalizing gradient magnitude
      # across families on the shared transformer trunk.
      one_ddt <- strenv$jnp$array(1., dtype = ddtype_)
      family_logp_scale <- function(mask, term) {
        count <- strenv$jnp$maximum(strenv$jnp$sum(mask), one_ddt)
        mean_abs <- strenv$jnp$sum(strenv$jnp$abs(term)) / count
        strenv$jax$lax$stop_gradient(strenv$jnp$maximum(mean_abs, one_ddt))
      }
      s_bern <- family_logp_scale(like_bern, bern_term)
      s_cat <- family_logp_scale(like_cat, cat_term)
      s_norm <- family_logp_scale(like_norm, norm_term)
      s_ord <- family_logp_scale(like_ord, ord_term)
      bern_term / s_bern + cat_term / s_cat + norm_term / s_norm + ord_term / s_ord
    } else {
      bern_term + cat_term + norm_term + ord_term
    }
  }

  apply_observation_likelihood <- function(logits,
                                           Yb,
                                           likelihood_code_obs = NULL,
                                           n_outcomes_obs = NULL,
                                           sigma = NULL,
                                           site_name = "obs",
                                           obs_scale = NULL,
                                           pairwise_obs = FALSE,
                                           pairwise_logit_scale = NULL,
                                           experiment_index_obs = NULL,
                                           ordinal_thresholds = NULL,
                                           params = NULL) {
    logits <- neural_apply_pairwise_classification_logit_transform(
      logits,
      model_info = low_rank_logit_model_info,
      likelihood_code_obs = likelihood_code_obs,
      pairwise_obs = pairwise_obs
    )
    logits <- neural_apply_pairwise_bernoulli_logit_scale(
      logits,
      model_info = low_rank_logit_model_info,
      scale = pairwise_logit_scale,
      likelihood_code_obs = likelihood_code_obs,
      pairwise_obs = pairwise_obs
    )
    logits <- neural_apply_classification_logit_calibration(
      logits,
      model_info = low_rank_logit_model_info,
      params = params,
      likelihood_code_obs = likelihood_code_obs
    )
    scaled_observation_factor <- function(distribution, obs) {
      if (is.null(obs_scale)) {
        strenv$numpyro$sample(site_name, distribution, obs = obs)
      } else {
        scale_use <- strenv$jnp$array(obs_scale, dtype = ddtype_)
        strenv$numpyro$factor(site_name, distribution$log_prob(obs) * scale_use)
      }
      invisible(NULL)
    }
    if (!isTRUE(universal_enabled)) {
      if (likelihood == "bernoulli") {
        logits_vec <- strenv$jnp$take(logits, ai(0L), axis = 1L)
        scaled_observation_factor(
          strenv$numpyro$distributions$Bernoulli(logits = logits_vec),
          Yb
        )
        return(invisible(NULL))
      }
      if (likelihood == "categorical") {
        scaled_observation_factor(
          strenv$numpyro$distributions$Categorical(logits = logits),
          Yb
        )
        return(invisible(NULL))
      }
      if (likelihood == "ordinal") {
        total_logp <- universal_row_log_prob_jnp(
          logits = logits,
          Yb = Yb,
          likelihood_code_obs = strenv$jnp$full(Yb$shape, ai(3L), dtype = strenv$jnp$int32),
          n_outcomes_obs = strenv$jnp$full(Yb$shape, ai(nOutcomes), dtype = strenv$jnp$int32),
          sigma = sigma,
          experiment_index_obs = experiment_index_obs,
          ordinal_thresholds = ordinal_thresholds
        )
        if (!is.null(obs_scale)) {
          total_logp <- total_logp * strenv$jnp$array(obs_scale, dtype = ddtype_)
        }
        strenv$numpyro$factor(site_name, total_logp)
        return(invisible(NULL))
      }
      mu <- strenv$jnp$take(logits, ai(0L), axis = 1L)
      sigma_use <- neural_floor_sigma(sigma, dtype = ddtype_)
      scaled_observation_factor(
        strenv$numpyro$distributions$Normal(mu, sigma_use),
        Yb
      )
      return(invisible(NULL))
    }

    if (is.null(likelihood_code_obs) || is.null(n_outcomes_obs)) {
      stop("Universal foundation training requires likelihood_code_obs and n_outcomes_obs.", call. = FALSE)
    }
    total_logp <- universal_row_log_prob_jnp(
      logits = logits,
      Yb = Yb,
      likelihood_code_obs = likelihood_code_obs,
      n_outcomes_obs = n_outcomes_obs,
      sigma = sigma,
      experiment_index_obs = experiment_index_obs,
      ordinal_thresholds = ordinal_thresholds
    )
    if (!is.null(obs_scale)) {
      total_logp <- total_logp * strenv$jnp$array(obs_scale, dtype = ddtype_)
    }
    strenv$numpyro$factor(site_name, total_logp)
    invisible(NULL)
  }

  normalize_resp_cov_present_for_model <- function(resp_cov,
                                                   resp_cov_present = NULL,
                                                   n_rows = 1L) {
    n_rows <- ai(n_rows)
    if (!is.null(resp_cov_present)) {
      return(neural_as_jnp_matrix(resp_cov_present, dtype = ddtype_))
    }
    if (!is.null(resp_cov_default_present)) {
      default_present <- matrix(as.numeric(resp_cov_default_present), nrow = 1L)
      if (n_rows > 1L) {
        default_present <- matrix(
          rep(default_present[1L, ], each = n_rows),
          nrow = n_rows
        )
      }
      return(neural_as_jnp_matrix(default_present, dtype = ddtype_))
    }
    if (!is.null(resp_cov)) {
      resp_cov_mat <- neural_as_jnp_matrix(resp_cov, dtype = ddtype_)
      return(strenv$jnp$ones(resp_cov_mat$shape, dtype = ddtype_))
    }
    if (n_resp_covariates > 0L) {
      return(strenv$jnp$ones(list(n_rows, ai(n_resp_covariates)), dtype = ddtype_))
    }
    strenv$jnp$zeros(list(n_rows, ai(0L)), dtype = ddtype_)
  }

  normalize_model_obs_idx <- function(obs_idx) {
    if (is.null(obs_idx)) {
      return(NULL)
    }
    if (reticulate::is_py_object(obs_idx)) {
      return(strenv$jnp$array(obs_idx)$astype(strenv$jnp$int32))
    }
    strenv$jnp$array(as.integer(obs_idx) - 1L)$astype(strenv$jnp$int32)
  }
  subset_model_rows <- function(x, obs_idx) {
    if (is.null(x) || is.null(obs_idx)) {
      return(x)
    }
    strenv$jnp$take(x, obs_idx, axis = 0L)
  }
  BayesianPairTransformerModel <- function(X_left, X_right, party_left, party_right,
                                           resp_party, resp_cov, resp_cov_present = NULL,
                                           experiment_index = NULL,
                                           likelihood_code = NULL,
                                           n_outcomes_obs = NULL,
                                           Y_obs,
                                           obs_idx = NULL,
                                           obs_scale = NULL,
                                           X_single = NULL,
                                           party_single = NULL,
                                           resp_party_single = NULL,
                                           resp_cov_single = NULL,
                                           resp_cov_present_single = NULL,
                                           experiment_index_single = NULL,
                                           likelihood_code_single = NULL,
                                           n_outcomes_single = NULL,
                                           Y_single_obs = NULL,
                                           obs_scale_single = NULL) {
    strenv$moe_aux_counter <- 0L
    strenv$moe_aux_pending <- list()
    obs_idx <- normalize_model_obs_idx(obs_idx)
    if (!is.null(obs_idx)) {
      X_left <- subset_model_rows(X_left, obs_idx)
      X_right <- subset_model_rows(X_right, obs_idx)
      party_left <- subset_model_rows(party_left, obs_idx)
      party_right <- subset_model_rows(party_right, obs_idx)
      resp_party <- subset_model_rows(resp_party, obs_idx)
      resp_cov <- subset_model_rows(resp_cov, obs_idx)
      resp_cov_present <- subset_model_rows(resp_cov_present, obs_idx)
      experiment_index <- subset_model_rows(experiment_index, obs_idx)
      likelihood_code <- subset_model_rows(likelihood_code, obs_idx)
      n_outcomes_obs <- subset_model_rows(n_outcomes_obs, obs_idx)
      Y_obs <- subset_model_rows(Y_obs, obs_idx)
    }
    N_local <- ai(X_left$shape[[1]])
    D_local <- ai(X_left$shape[[2]])
    resp_cov_present <- normalize_resp_cov_present_for_model(
      resp_cov = resp_cov,
      resp_cov_present = resp_cov_present,
      n_rows = N_local
    )

    shared_params <- sample_shared_transformer_params(D_local = D_local, pairwise = TRUE)
    params_view <- shared_params$params_view
    E_choice <- shared_params$E_choice
    W_out <- shared_params$W_out
    b_out <- shared_params$b_out
    sigma <- shared_params$sigma
    M_cross <- shared_params$M_cross
    W_cross_out <- shared_params$W_cross_out

    transformer_model_info <- neural_make_transformer_model_info(
      model_depth = ModelDepth,
      model_dims = ModelDims,
      n_heads = TransformerHeads,
      head_dim = head_dim,
      residual_mode = residual_mode,
      attention_backend = attention_backend,
      attention_dtype = attention_dtype,
      attention_padding_multiple = attention_padding_multiple,
      attention_resolved_backend = attention_resolved_backend,
      attention_fallback_reason = attention_fallback_reason
    )
    model_info_local <- neural_make_runtime_token_model_info(
      model_dims = ModelDims,
      cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
      n_party_levels = ai(n_party_levels),
      factor_name_text = factor_name_text,
      level_name_text = level_name_text,
      factor_struct_matrix = factor_struct_matrix,
      level_struct_matrices = level_struct_matrices,
      factor_struct_feature_names = factor_struct_feature_names,
      level_struct_feature_names = level_struct_feature_names,
      factor_order_by_experiment = factor_order_by_experiment,
      default_factor_order = default_factor_order,
      factor_tokenization = factor_tokenization,
      max_factor_tokens = max_factor_tokens,
      covariate_name_text = covariate_name_text,
      covariate_names = covariate_names_override,
      resp_cov_mean = resp_cov_mean,
      resp_cov_scale = resp_cov_scale,
      resp_cov_default_present = resp_cov_default_present,
      covariate_order_by_experiment = covariate_order_by_experiment,
      default_covariate_order = default_covariate_order,
      covariate_value_stats_by_experiment = covariate_value_stats_by_experiment,
      default_covariate_value_stats = default_covariate_value_stats,
      covariate_value_metadata_by_experiment = covariate_value_metadata_by_experiment,
      default_covariate_value_metadata = default_covariate_value_metadata,
      covariate_value_text = covariate_value_text,
      covariate_value_text_present = covariate_value_text_present,
      covariate_value_type = covariate_value_type,
      max_covariate_tokens = max_covariate_tokens,
      default_experiment_index = default_experiment_index,
      token_family_levels = token_family_levels,
      experiment_token_mode = experiment_token_mode,
      covariate_value_encoding = covariate_value_encoding,
      shared_projection_value_encoder = shared_projection_value_encoder,
      experiment_description_text = experiment_description_text,
      experiment_description_present = experiment_description_present,
      default_experiment_text = default_experiment_text,
      default_experiment_text_present = default_experiment_text_present,
      place_embedding = place_embedding,
      place_present = place_present,
      place_context_enabled = place_context_enabled,
      place_feature_names = place_feature_names,
      default_place_embedding = default_place_embedding,
      default_place_present = default_place_present,
      time_embedding = time_embedding,
      time_present = time_present,
      time_context_enabled = time_context_enabled,
      time_feature_names = time_feature_names,
      default_time_embedding = default_time_embedding,
      default_time_present = default_time_present,
      low_rank_interaction_rank = low_rank_interaction_rank,
      low_rank_logit_transform = low_rank_logit_transform,
      low_rank_logit_bound = low_rank_logit_bound,
      low_rank_logit_softness = low_rank_logit_softness,
      low_rank_logit_normalization = low_rank_logit_normalization,
      low_rank_head_weight_target_rms = low_rank_head_weight_target_rms,
      low_rank_rc_out_target_rms = low_rank_rc_out_target_rms,
      pairwise_antisymmetry = pairwise_antisymmetry_mode,
      additive_utility_mode = additive_utility_mode,
      additive_utility_normalization = additive_utility_normalization,
      additive_head_weight_target_rms = additive_head_weight_target_rms,
      calibration_enabled = isTRUE(calibration_control$enabled),
      calibration_method = calibration_control$method %||% "none",
      calibration_prior_sd = calibration_control$prior_sd %||% NULL,
      likelihood = likelihood,
      schema_dropout = schema_dropout
    )
    model_info_local <- neural_set_pairwise_context_model_info(
      info = model_info_local,
      pairwise_context_mode = pairwise_context_mode,
      has_candidate_group_context = has_candidate_group_context,
      has_respondent_group_context = has_respondent_group_context,
      has_relation_token_context = has_relation_context,
      has_stage_context = stage_context_enabled,
      has_matchup_context = use_matchup_token,
      party_missing_label = party_missing_label,
      resp_party_missing_label = resp_party_missing_label,
      n_resp_party_levels = n_resp_party_levels,
      party_missing_index = party_missing_index,
      resp_party_missing_index = resp_party_missing_index,
      context_present_masking = context_present_masking
    )

    embed_candidate <- function(X_idx, party_idx, resp_p, experiment_idx = NULL,
                                return_mask = FALSE, context_present = NULL,
                                schema_dropout_masks = NULL) {
      neural_build_candidate_tokens_hard(X_idx, party_idx,
                                         model_info = model_info_local,
                                         resp_party_idx = resp_p,
                                         experiment_idx = experiment_idx,
                                         params = params_view,
                                         return_mask = return_mask,
                                         context_present = context_present,
                                         schema_dropout_masks = schema_dropout_masks)
    }

    add_segment_embedding <- function(tokens, segment_idx) {
      neural_add_segment_embedding(tokens, segment_idx,
                                   model_info = model_info_local,
                                   params = params_view)
    }

    run_transformer <- function(tokens, token_mask = NULL, return_details = FALSE) {
      neural_run_transformer(tokens,
                             model_info = transformer_model_info,
                             params = params_view,
                             token_mask = token_mask,
                             return_details = return_details)
    }

    compute_matchup_idx <- function(pl, pr) {
      neural_matchup_index(pl, pr, model_info_local)
    }

    build_context_tokens <- function(stage_idx,
                                    resp_p,
                                    resp_c,
                                    resp_c_present = NULL,
                                    experiment_idx = NULL,
                                    matchup_idx = NULL,
                                    return_mask = FALSE,
                                    context_present = NULL,
                                    schema_dropout_masks = NULL) {
      neural_build_context_tokens_batch(model_info = model_info_local,
                                        resp_party_idx = resp_p,
                                        stage_idx = stage_idx,
                                        matchup_idx = matchup_idx,
                                        resp_cov = resp_c,
                                        resp_cov_present = resp_c_present,
                                        experiment_idx = experiment_idx,
                                        params = params_view,
                                        return_mask = return_mask,
                                        context_present = context_present,
                                        schema_dropout_masks = schema_dropout_masks)
    }

    build_sep_token <- function(N_batch) {
      neural_build_sep_token(model_info_local,
                             n_batch = N_batch,
                             params = params_view)
    }

    encode_pair_cross <- function(Xl, Xr, pl, pr, resp_p, resp_c, resp_c_present = NULL,
                                  experiment_idx = NULL, stage_idx, matchup_idx = NULL,
                                  context_present = NULL,
                                  schema_dropout_context = NULL,
                                  schema_dropout_left = NULL,
                                  schema_dropout_right = NULL) {
      N_batch <- ai(Xl$shape[[1]])
      choice_tok <- neural_build_choice_token(model_info_local, params_view)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      choice_mask <- strenv$jnp$ones(list(N_batch, 1L), dtype = ddtype_)
      ctx_info <- build_context_tokens(
        stage_idx,
        resp_p,
        resp_c,
        resp_c_present,
        experiment_idx,
        matchup_idx,
        return_mask = TRUE,
        context_present = context_present,
        schema_dropout_masks = schema_dropout_context
      )
      ctx_tokens <- ctx_info$tokens %||% NULL
      ctx_mask <- ctx_info$mask %||% NULL
      left_info <- embed_candidate(Xl, pl, resp_p, experiment_idx,
                                   return_mask = TRUE,
                                   context_present = context_present,
                                   schema_dropout_masks = schema_dropout_left)
      right_info <- embed_candidate(Xr, pr, resp_p, experiment_idx,
                                    return_mask = TRUE,
                                    context_present = context_present,
                                    schema_dropout_masks = schema_dropout_right)
      left_tokens <- add_segment_embedding(left_info$tokens, 0L)
      right_tokens <- add_segment_embedding(right_info$tokens, 1L)
      sep_tok <- build_sep_token(N_batch)
      sep_mask <- strenv$jnp$ones(list(N_batch, 1L), dtype = ddtype_)
      seq_info <- neural_pack_full_cross_sequence(
        choice_tok = choice_tok,
        choice_mask = choice_mask,
        sep_tok = sep_tok,
        sep_mask = sep_mask,
        left_tokens = left_tokens,
        left_mask = left_info$mask,
        right_tokens = right_tokens,
        right_mask = right_info$mask,
        model_info = model_info_local,
        ctx_tokens = ctx_tokens,
        ctx_mask = ctx_mask
      )
      tokens <- seq_info$tokens
      token_mask <- seq_info$mask
      transformer_out <- run_transformer(tokens, token_mask = token_mask, return_details = TRUE)
      cls_out <- neural_extract_choice_representation(transformer_out)
      neural_linear_head(
        cls_out,
        W_out,
        b_out,
        model_info = model_info_local,
        pairwise_obs = TRUE
      )
    }

    encode_candidate <- function(Xa, pa, resp_p, resp_c, resp_c_present = NULL,
                                 experiment_idx = NULL, stage_idx, matchup_idx = NULL,
                                 return_tokens = FALSE, context_present = NULL,
                                 schema_dropout_context = NULL,
                                 schema_dropout_candidate = NULL) {
      N_batch <- ai(Xa$shape[[1]])
      choice_tok <- neural_build_choice_token(model_info_local, params_view)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      choice_mask <- strenv$jnp$ones(list(N_batch, 1L), dtype = ddtype_)
      ctx_info <- build_context_tokens(
        stage_idx,
        resp_p,
        resp_c,
        resp_c_present,
        experiment_idx,
        matchup_idx,
        return_mask = TRUE,
        context_present = context_present,
        schema_dropout_masks = schema_dropout_context
      )
      ctx_tokens <- ctx_info$tokens %||% NULL
      ctx_mask <- ctx_info$mask %||% NULL
      cand_info <- embed_candidate(Xa, pa, resp_p, experiment_idx,
                                   return_mask = TRUE,
                                   context_present = context_present,
                                   schema_dropout_masks = schema_dropout_candidate)
      cand_tokens <- cand_info$tokens
      cand_mask <- cand_info$mask
      seq_info <- neural_pack_candidate_sequence(
        choice_tok = choice_tok,
        choice_mask = choice_mask,
        ctx_tokens = ctx_tokens,
        ctx_mask = ctx_mask,
        cand_tokens = cand_tokens,
        cand_mask = cand_mask,
        model_info = model_info_local,
        preserve_candidate_tail = isTRUE(return_tokens)
      )
      tokens <- seq_info$tokens
      token_mask <- seq_info$mask
      transformer_out <- run_transformer(tokens, token_mask = token_mask, return_details = TRUE)
      phi <- neural_extract_choice_representation(
        transformer_out,
        token_mask = token_mask,
        include_tail_summary = neural_fused_classification_summary_enabled(model_info_local)
      )
      if (!isTRUE(return_tokens)) {
        return(phi)
      }
      cand_out <- neural_extract_candidate_tokens(
        transformer_out,
        transformer_model_info,
        n_candidate_tokens = neural_candidate_token_count_from_mask(seq_info$cand_mask)
      )
      list(phi = phi, cand_tokens_out = cand_out, cand_token_mask = seq_info$cand_mask)
    }

    encode_candidate_pair <- function(Xl, Xr, pl, pr, resp_p, resp_c, resp_c_present = NULL,
                                      experiment_idx = NULL, stage_idx, matchup_idx = NULL,
                                      context_present = NULL,
                                      schema_dropout_context = NULL,
                                      schema_dropout_left = NULL,
                                      schema_dropout_right = NULL) {
      N_batch <- ai(Xl$shape[[1]])
      X_all <- strenv$jnp$concatenate(list(Xl, Xr), axis = 0L)
      p_all <- strenv$jnp$concatenate(list(pl, pr), axis = 0L)
      resp_p_all <- strenv$jnp$concatenate(list(resp_p, resp_p), axis = 0L)
      resp_c_all <- if (is.null(resp_c)) NULL else strenv$jnp$concatenate(list(resp_c, resp_c), axis = 0L)
      resp_c_present_all <- if (is.null(resp_c_present)) NULL else strenv$jnp$concatenate(list(resp_c_present, resp_c_present), axis = 0L)
      experiment_idx_all <- if (is.null(experiment_idx)) NULL else strenv$jnp$concatenate(list(experiment_idx, experiment_idx), axis = 0L)
      stage_all <- if (is.null(stage_idx)) NULL else strenv$jnp$concatenate(list(stage_idx, stage_idx), axis = 0L)
      matchup_all <- if (is.null(matchup_idx)) NULL else strenv$jnp$concatenate(list(matchup_idx, matchup_idx), axis = 0L)
      context_present_all <- if (is.null(context_present)) NULL else {
        strenv$jnp$concatenate(list(context_present, context_present), axis = 0L)
      }
      schema_dropout_all <- neural_concat_schema_dropout_masks(
        schema_dropout_left,
        schema_dropout_right
      )
      schema_dropout_context_all <- neural_concat_schema_dropout_masks(
        schema_dropout_context,
        schema_dropout_context
      )
      if (isTRUE(use_cross_attn)) {
        enc_all <- encode_candidate(X_all, p_all, resp_p_all, resp_c_all,
                                    resp_c_present_all, experiment_idx_all,
                                    stage_all, matchup_all, return_tokens = TRUE,
                                    context_present = context_present_all,
                                    schema_dropout_context = schema_dropout_context_all,
                                    schema_dropout_candidate = schema_dropout_all)
        phi_all <- enc_all$phi
        cand_all <- enc_all$cand_tokens_out
      } else {
        phi_all <- encode_candidate(X_all, p_all, resp_p_all, resp_c_all,
                                    resp_c_present_all, experiment_idx_all,
                                    stage_all, matchup_all,
                                    context_present = context_present_all,
                                    schema_dropout_context = schema_dropout_context_all,
                                    schema_dropout_candidate = schema_dropout_all)
        cand_all <- NULL
      }
      idx_left <- strenv$jnp$arange(N_batch)
      idx_right <- strenv$jnp$arange(N_batch, 2L * N_batch)
      out <- list(
        phi_left = strenv$jnp$take(phi_all, idx_left, axis = 0L),
        phi_right = strenv$jnp$take(phi_all, idx_right, axis = 0L)
      )
      if (isTRUE(use_cross_attn)) {
        out$cand_left_out <- strenv$jnp$take(cand_all, idx_left, axis = 0L)
        out$cand_right_out <- strenv$jnp$take(cand_all, idx_right, axis = 0L)
        out$cand_left_mask <- strenv$jnp$take(enc_all$cand_token_mask, idx_left, axis = 0L)
        out$cand_right_mask <- strenv$jnp$take(enc_all$cand_token_mask, idx_right, axis = 0L)
      }
      out
    }

    do_forward_and_lik_ <- function(Xl, Xr, pl, pr, resp_p, resp_c,
                                    resp_c_present = NULL, experiment_idx = NULL,
                                    likelihood_code_b = NULL,
                                    n_outcomes_b = NULL,
                                    Yb,
                                    obs_scale_b = obs_scale) {
      stage_idx <- neural_stage_index(pl, pr, model_info_local)
      matchup_idx <- NULL
      if (isTRUE(use_matchup_token)) {
        matchup_idx <- compute_matchup_idx(pl, pr)
      }
      context_present <- neural_pair_context_present(pl, pr, resp_p, model_info_local)
      schema_dropout_context <- neural_sample_schema_dropout_masks(
        model_info_local,
        n_batch = ai(Xl$shape[[1]])
      )
      schema_dropout_left <- neural_sample_schema_dropout_masks(
        model_info_local,
        n_batch = ai(Xl$shape[[1]])
      )
      # Share the schema-availability masks across the two candidates: real
      # test-time schema missingness is task-level (a factor absent for one
      # candidate is absent for both), so independent left/right sampling
      # trained on asymmetric-information pairs that cannot occur at
      # inference.
      schema_dropout_right <- schema_dropout_left
      if (isTRUE(use_cross_encoder)) {
        logits <- encode_pair_cross(Xl, Xr, pl, pr, resp_p, resp_c,
                                    resp_c_present, experiment_idx,
                                    stage_idx, matchup_idx,
                                    context_present = context_present,
                                    schema_dropout_context = schema_dropout_context,
                                    schema_dropout_left = schema_dropout_left,
                                    schema_dropout_right = schema_dropout_right)
        if (isTRUE(neural_pairwise_antisymmetry_strict(model_info_local))) {
          stage_idx_rev <- neural_stage_index(pr, pl, model_info_local)
          matchup_idx_rev <- NULL
          if (isTRUE(use_matchup_token)) {
            matchup_idx_rev <- compute_matchup_idx(pr, pl)
          }
          context_present_rev <- neural_pair_context_present(pr, pl, resp_p, model_info_local)
          logits_rev <- encode_pair_cross(Xr, Xl, pr, pl, resp_p, resp_c,
                                          resp_c_present, experiment_idx,
                                          stage_idx_rev, matchup_idx_rev,
                                          context_present = context_present_rev,
                                          schema_dropout_context = schema_dropout_context,
                                          schema_dropout_left = schema_dropout_right,
                                          schema_dropout_right = schema_dropout_left)
          logits <- 0.5 * (logits - logits_rev)
        }
        if (isTRUE(neural_has_low_rank_interaction(params_view, model_info_local))) {
          logits <- logits + neural_low_rank_pair_delta_prepared(
            params = params_view,
            model_info = model_info_local,
            Xl = Xl,
            Xr = Xr,
            pl = pl,
            pr = pr,
            resp_p = resp_p,
            resp_c = resp_c,
            resp_c_present = resp_c_present,
            experiment_idx = experiment_idx,
            stage_idx = stage_idx,
            matchup_idx = matchup_idx,
            context_present = context_present,
            schema_dropout_context = schema_dropout_context,
            schema_dropout_left = schema_dropout_left,
            schema_dropout_right = schema_dropout_right,
            transformer_model_info = transformer_model_info,
            out_dim = ai(logits$shape[[2]]),
            dtype = logits$dtype
          )
        }
        logits <- logits + neural_additive_pair_delta_prepared(
          params = params_view,
          model_info = model_info_local,
          Xl = Xl,
          Xr = Xr,
          pl = pl,
          pr = pr,
          resp_p = resp_p,
          experiment_idx = experiment_idx,
          context_present = context_present,
          schema_dropout_left = schema_dropout_left,
          schema_dropout_right = schema_dropout_right,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype
        )
      } else {
        phi_pair <- encode_candidate_pair(Xl, Xr, pl, pr, resp_p, resp_c,
                                          resp_c_present, experiment_idx,
                                          stage_idx, matchup_idx,
                                          context_present = context_present,
                                          schema_dropout_context = schema_dropout_context,
                                          schema_dropout_left = schema_dropout_left,
                                          schema_dropout_right = schema_dropout_right)
        phi_l <- phi_pair$phi_left
        phi_r <- phi_pair$phi_right
        if (isTRUE(use_cross_attn)) {
          ctx_left <- neural_cross_attend_cls_to_tokens(phi_l, phi_pair$cand_right_out,
                                                        model_info = transformer_model_info,
                                                        params = params_view,
                                                        kv_token_mask = phi_pair$cand_right_mask)
          ctx_right <- neural_cross_attend_cls_to_tokens(phi_r, phi_pair$cand_left_out,
                                                         model_info = transformer_model_info,
                                                         params = params_view,
                                                         kv_token_mask = phi_pair$cand_left_mask)
          phi_l <- neural_merge_cross_attn_representation(
            phi_l, ctx_left, params_view, transformer_model_info$model_dims
          )
          phi_r <- neural_merge_cross_attn_representation(
            phi_r, ctx_right, params_view, transformer_model_info$model_dims
          )
        }
        u_l <- neural_linear_head(
          phi_l,
          W_out,
          b_out,
          model_info = model_info_local,
          pairwise_obs = TRUE
        )
        u_r <- neural_linear_head(
          phi_r,
          W_out,
          b_out,
          model_info = model_info_local,
          pairwise_obs = TRUE
        )
        logits <- u_l - u_r
        if (isTRUE(neural_has_low_rank_interaction(params_view, model_info_local))) {
          logits <- logits + neural_low_rank_pair_delta_prepared(
            params = params_view,
            model_info = model_info_local,
            Xl = Xl,
            Xr = Xr,
            pl = pl,
            pr = pr,
            resp_p = resp_p,
            resp_c = resp_c,
            resp_c_present = resp_c_present,
            experiment_idx = experiment_idx,
            stage_idx = stage_idx,
            matchup_idx = matchup_idx,
            context_present = context_present,
            schema_dropout_context = schema_dropout_context,
            schema_dropout_left = schema_dropout_left,
            schema_dropout_right = schema_dropout_right,
            transformer_model_info = transformer_model_info,
            out_dim = ai(logits$shape[[2]]),
            dtype = logits$dtype
          )
        }
        if (isTRUE(use_cross_term)) {
          logits <- neural_apply_cross_term(logits, phi_l, phi_r,
                                            M_cross, W_cross_out,
                                            out_dim = ai(W_out$shape[[2]]))
        }
        logits <- logits + neural_additive_pair_delta_prepared(
          params = params_view,
          model_info = model_info_local,
          Xl = Xl,
          Xr = Xr,
          pl = pl,
          pr = pr,
          resp_p = resp_p,
          experiment_idx = experiment_idx,
          context_present = context_present,
          schema_dropout_left = schema_dropout_left,
          schema_dropout_right = schema_dropout_right,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype
        )
      }

      apply_observation_likelihood(
        logits = logits,
        Yb = Yb,
        likelihood_code_obs = likelihood_code_b,
        n_outcomes_obs = n_outcomes_b,
        sigma = sigma,
        site_name = "obs_pair",
        obs_scale = obs_scale_b,
        pairwise_obs = TRUE,
        pairwise_logit_scale = neural_pairwise_bernoulli_logit_scale_from_params(
          params = params_view,
          model_info = low_rank_logit_model_info
        ),
        experiment_index_obs = experiment_idx,
        ordinal_thresholds = params_view$ordinal_thresholds %||% NULL,
        params = params_view
      )
    }

    do_forward_single_and_lik_ <- function(Xb, pb, resp_p, resp_c,
                                           resp_c_present = NULL,
                                           experiment_idx = NULL,
                                           likelihood_code_b = NULL,
                                           n_outcomes_b = NULL,
                                           Yb,
                                           obs_scale_b = NULL) {
      N_batch <- ai(Xb$shape[[1]])
      schema_dropout_context <- neural_sample_schema_dropout_masks(
        model_info_local,
        n_batch = N_batch
      )
      schema_dropout_candidate <- neural_sample_schema_dropout_masks(
        model_info_local,
        n_batch = N_batch
      )
      choice_tok <- neural_build_choice_token(model_info_local, params_view)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      choice_mask <- strenv$jnp$ones(list(N_batch, 1L), dtype = ddtype_)
      ctx_info <- neural_build_context_tokens_batch(
        model_info = model_info_local,
        resp_party_idx = resp_p,
        resp_cov = resp_c,
        resp_cov_present = resp_c_present,
        experiment_idx = experiment_idx,
        params = params_view,
        return_mask = TRUE,
        schema_dropout_masks = schema_dropout_context
      )
      cand_info <- embed_candidate(
        Xb,
        pb,
        resp_p,
        experiment_idx,
        return_mask = TRUE,
        schema_dropout_masks = schema_dropout_candidate
      )
      seq_info <- neural_pack_candidate_sequence(
        choice_tok = choice_tok,
        choice_mask = choice_mask,
        ctx_tokens = ctx_info$tokens %||% NULL,
        ctx_mask = ctx_info$mask %||% NULL,
        cand_tokens = cand_info$tokens,
        cand_mask = cand_info$mask,
        model_info = model_info_local,
        preserve_candidate_tail = FALSE
      )
      transformer_out <- run_transformer(seq_info$tokens, token_mask = seq_info$mask, return_details = TRUE)
      phi <- neural_extract_choice_representation(
        transformer_out,
        token_mask = seq_info$mask,
        include_tail_summary = neural_fused_classification_summary_enabled(model_info_local)
      )
      logits <- neural_linear_head(phi, W_out, b_out)
      if (isTRUE(neural_has_low_rank_interaction(params_view, model_info_local))) {
        logits <- logits + neural_low_rank_single_utility_prepared(
          params = params_view,
          model_info = model_info_local,
          X_idx = Xb,
          party_idx = pb,
          resp_party_idx = resp_p,
          resp_cov = resp_c,
          resp_cov_present = resp_c_present,
          experiment_idx = experiment_idx,
          schema_dropout_context = schema_dropout_context,
          schema_dropout_candidate = schema_dropout_candidate,
          transformer_model_info = transformer_model_info,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype
        )
      }
      logits <- logits + neural_additive_single_logits_prepared(
        params = params_view,
        model_info = model_info_local,
        X_idx = Xb,
        party_idx = pb,
        resp_party_idx = resp_p,
        experiment_idx = experiment_idx,
        context_present = NULL,
        schema_dropout_masks = schema_dropout_candidate,
        out_dim = ai(logits$shape[[2]]),
        dtype = logits$dtype
      )
      apply_observation_likelihood(
        logits = logits,
        Yb = Yb,
        likelihood_code_obs = likelihood_code_b,
        n_outcomes_obs = n_outcomes_b,
        sigma = sigma,
        site_name = "obs_single",
        obs_scale = obs_scale_b,
        pairwise_obs = FALSE,
        experiment_index_obs = experiment_idx,
        ordinal_thresholds = params_view$ordinal_thresholds %||% NULL,
        params = params_view
      )
    }

    local_lik <- function() {
      if (isTRUE(subsample_method_model %in% c("batch", "batch_vi"))) {
        with(strenv$numpyro$plate("data", size = N_local,
                                  # Clamp subsample_size to the available data to avoid
                                  # plate() errors when batch_size > N_local.
                                  subsample_size = ai(min(ai(mcmc_control$batch_size), N_local)),
                                  dim = -1L) %as% "idx", {
                                    Xl_b <- strenv$jnp$take(X_left, idx, axis = 0L)
                                    Xr_b <- strenv$jnp$take(X_right, idx, axis = 0L)
                                    pl_b <- strenv$jnp$take(party_left, idx, axis = 0L)
                                    pr_b <- strenv$jnp$take(party_right, idx, axis = 0L)
                                    resp_p_b <- strenv$jnp$take(resp_party, idx, axis = 0L)
                                    resp_c_b <- strenv$jnp$take(resp_cov, idx, axis = 0L)
                                    resp_c_present_b <- strenv$jnp$take(resp_cov_present, idx, axis = 0L)
                                    experiment_idx_b <- if (is.null(experiment_index)) NULL else {
                                      strenv$jnp$take(experiment_index, idx, axis = 0L)
                                    }
                                    likelihood_code_b <- if (is.null(likelihood_code)) NULL else {
                                      strenv$jnp$take(likelihood_code, idx, axis = 0L)
                                    }
                                    n_outcomes_b <- if (is.null(n_outcomes_obs)) NULL else {
                                      strenv$jnp$take(n_outcomes_obs, idx, axis = 0L)
                                    }
                                    Yb <- if (is.null(Y_obs)) NULL else strenv$jnp$take(Y_obs, idx, axis = 0L)
                                    obs_scale_b <- if (is.null(obs_scale) ||
                                                       (!reticulate::is_py_object(obs_scale) && length(obs_scale) == 1L)) {
                                      obs_scale
                                    } else if (reticulate::is_py_object(obs_scale)) {
                                      strenv$jnp$take(obs_scale, idx, axis = 0L)
                                    } else {
                                      strenv$jnp$take(strenv$jnp$array(obs_scale, dtype = ddtype_), idx, axis = 0L)
                                    }
                                    do_forward_and_lik_(Xl_b, Xr_b, pl_b, pr_b, resp_p_b,
                                                        resp_c_b, resp_c_present_b,
                                                        experiment_idx_b,
                                                        likelihood_code_b,
                                                        n_outcomes_b,
                                                        Yb,
                                                        obs_scale_b = obs_scale_b)
                                  })
      } else {
        with(strenv$numpyro$plate("data", size = N_local, dim = -1L), {
          do_forward_and_lik_(X_left, X_right, party_left, party_right,
                              resp_party, resp_cov, resp_cov_present,
                              experiment_index, likelihood_code, n_outcomes_obs, Y_obs)
        })
      }
    }

    local_lik()
    if (!is.null(X_single)) {
      resp_cov_present_single <- normalize_resp_cov_present_for_model(
        resp_cov = resp_cov_single,
        resp_cov_present = resp_cov_present_single,
        n_rows = ai(X_single$shape[[1]])
      )
      N_single_local <- ai(X_single$shape[[1]])
      if (isTRUE(subsample_method_model %in% c("batch", "batch_vi"))) {
        with(strenv$numpyro$plate("single_data", size = N_single_local,
                                  subsample_size = ai(min(ai(mcmc_control$batch_size), N_single_local)),
                                  dim = -1L) %as% "idx_single", {
                                    Xb <- strenv$jnp$take(X_single, idx_single, axis = 0L)
                                    pb <- strenv$jnp$take(party_single, idx_single, axis = 0L)
                                    resp_p_b <- strenv$jnp$take(resp_party_single, idx_single, axis = 0L)
                                    resp_c_b <- strenv$jnp$take(resp_cov_single, idx_single, axis = 0L)
                                    resp_c_present_b <- strenv$jnp$take(resp_cov_present_single, idx_single, axis = 0L)
                                    experiment_idx_b <- if (is.null(experiment_index_single)) NULL else {
                                      strenv$jnp$take(experiment_index_single, idx_single, axis = 0L)
                                    }
                                    likelihood_code_b <- if (is.null(likelihood_code_single)) NULL else {
                                      strenv$jnp$take(likelihood_code_single, idx_single, axis = 0L)
                                    }
                                    n_outcomes_b <- if (is.null(n_outcomes_single)) NULL else {
                                      strenv$jnp$take(n_outcomes_single, idx_single, axis = 0L)
                                    }
                                    Yb <- if (is.null(Y_single_obs)) NULL else strenv$jnp$take(Y_single_obs, idx_single, axis = 0L)
                                    obs_scale_b <- if (is.null(obs_scale_single) ||
                                                       (!reticulate::is_py_object(obs_scale_single) && length(obs_scale_single) == 1L)) {
                                      obs_scale_single
                                    } else if (reticulate::is_py_object(obs_scale_single)) {
                                      strenv$jnp$take(obs_scale_single, idx_single, axis = 0L)
                                    } else {
                                      strenv$jnp$take(strenv$jnp$array(obs_scale_single, dtype = ddtype_), idx_single, axis = 0L)
                                    }
                                    do_forward_single_and_lik_(
                                      Xb, pb, resp_p_b, resp_c_b, resp_c_present_b,
                                      experiment_idx_b, likelihood_code_b, n_outcomes_b,
                                      Yb, obs_scale_b
                                    )
                                  })
      } else {
        with(strenv$numpyro$plate("single_data", size = N_single_local, dim = -1L), {
          do_forward_single_and_lik_(
            X_single, party_single, resp_party_single, resp_cov_single,
            resp_cov_present_single, experiment_index_single,
            likelihood_code_single, n_outcomes_single, Y_single_obs,
            obs_scale_single
          )
        })
      }
    }
    # Under compact training the model receives one BATCH, so the local plate
    # sizes are B, not the dataset size; use the training-observation total the
    # compact path resolves (compact_model_n_obs) so the penalty tracks the
    # likelihood scale identically in both modes.
    moe_n_total <- if (isTRUE(compact_training)) {
      as.numeric(compact_model_n_obs)
    } else {
      n_tot <- as.numeric(N_local)
      if (!is.null(X_single)) {
        n_tot <- n_tot + as.numeric(ai(X_single$shape[[1]]))
      }
      n_tot
    }
    neural_emit_moe_load_balance_factor(moe_n_total)
  }

  BayesianSingleTransformerModel <- function(X, party, resp_party, resp_cov,
                                             resp_cov_present = NULL,
                                             experiment_index = NULL,
                                             likelihood_code = NULL,
                                             n_outcomes_obs = NULL,
                                             Y_obs,
                                             obs_idx = NULL,
                                             obs_scale = NULL) {
    strenv$moe_aux_counter <- 0L
    strenv$moe_aux_pending <- list()
    obs_idx <- normalize_model_obs_idx(obs_idx)
    if (!is.null(obs_idx)) {
      X <- subset_model_rows(X, obs_idx)
      party <- subset_model_rows(party, obs_idx)
      resp_party <- subset_model_rows(resp_party, obs_idx)
      resp_cov <- subset_model_rows(resp_cov, obs_idx)
      resp_cov_present <- subset_model_rows(resp_cov_present, obs_idx)
      experiment_index <- subset_model_rows(experiment_index, obs_idx)
      likelihood_code <- subset_model_rows(likelihood_code, obs_idx)
      n_outcomes_obs <- subset_model_rows(n_outcomes_obs, obs_idx)
      Y_obs <- subset_model_rows(Y_obs, obs_idx)
    }
    N_local <- ai(X$shape[[1]])
    D_local <- ai(X$shape[[2]])
    resp_cov_present <- normalize_resp_cov_present_for_model(
      resp_cov = resp_cov,
      resp_cov_present = resp_cov_present,
      n_rows = N_local
    )

    shared_params <- sample_shared_transformer_params(D_local = D_local, pairwise = FALSE)
    params_view <- shared_params$params_view
    E_choice <- shared_params$E_choice
    W_out <- shared_params$W_out
    b_out <- shared_params$b_out
    sigma <- shared_params$sigma

    transformer_model_info <- neural_make_transformer_model_info(
      model_depth = ModelDepth,
      model_dims = ModelDims,
      n_heads = TransformerHeads,
      head_dim = head_dim,
      residual_mode = residual_mode,
      attention_backend = attention_backend,
      attention_dtype = attention_dtype,
      attention_padding_multiple = attention_padding_multiple,
      attention_resolved_backend = attention_resolved_backend,
      attention_fallback_reason = attention_fallback_reason
    )
    model_info_local <- neural_make_runtime_token_model_info(
      model_dims = ModelDims,
      cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
      n_party_levels = ai(n_party_levels),
      factor_name_text = factor_name_text,
      level_name_text = level_name_text,
      factor_struct_matrix = factor_struct_matrix,
      level_struct_matrices = level_struct_matrices,
      factor_struct_feature_names = factor_struct_feature_names,
      level_struct_feature_names = level_struct_feature_names,
      factor_order_by_experiment = factor_order_by_experiment,
      default_factor_order = default_factor_order,
      factor_tokenization = factor_tokenization,
      max_factor_tokens = max_factor_tokens,
      covariate_name_text = covariate_name_text,
      covariate_names = covariate_names_override,
      resp_cov_mean = resp_cov_mean,
      resp_cov_scale = resp_cov_scale,
      resp_cov_default_present = resp_cov_default_present,
      covariate_order_by_experiment = covariate_order_by_experiment,
      default_covariate_order = default_covariate_order,
      covariate_value_stats_by_experiment = covariate_value_stats_by_experiment,
      default_covariate_value_stats = default_covariate_value_stats,
      covariate_value_metadata_by_experiment = covariate_value_metadata_by_experiment,
      default_covariate_value_metadata = default_covariate_value_metadata,
      covariate_value_text = covariate_value_text,
      covariate_value_text_present = covariate_value_text_present,
      covariate_value_type = covariate_value_type,
      max_covariate_tokens = max_covariate_tokens,
      default_experiment_index = default_experiment_index,
      token_family_levels = token_family_levels,
      experiment_token_mode = experiment_token_mode,
      covariate_value_encoding = covariate_value_encoding,
      shared_projection_value_encoder = shared_projection_value_encoder,
      experiment_description_text = experiment_description_text,
      experiment_description_present = experiment_description_present,
      default_experiment_text = default_experiment_text,
      default_experiment_text_present = default_experiment_text_present,
      place_embedding = place_embedding,
      place_present = place_present,
      place_context_enabled = place_context_enabled,
      place_feature_names = place_feature_names,
      default_place_embedding = default_place_embedding,
      default_place_present = default_place_present,
      time_embedding = time_embedding,
      time_present = time_present,
      time_context_enabled = time_context_enabled,
      time_feature_names = time_feature_names,
      default_time_embedding = default_time_embedding,
      default_time_present = default_time_present,
      low_rank_interaction_rank = low_rank_interaction_rank,
      low_rank_logit_transform = low_rank_logit_transform,
      low_rank_logit_bound = low_rank_logit_bound,
      low_rank_logit_softness = low_rank_logit_softness,
      low_rank_logit_normalization = low_rank_logit_normalization,
      low_rank_head_weight_target_rms = low_rank_head_weight_target_rms,
      low_rank_rc_out_target_rms = low_rank_rc_out_target_rms,
      pairwise_antisymmetry = pairwise_antisymmetry_mode,
      additive_utility_mode = additive_utility_mode,
      additive_utility_normalization = additive_utility_normalization,
      additive_head_weight_target_rms = additive_head_weight_target_rms,
      calibration_enabled = isTRUE(calibration_control$enabled),
      calibration_method = calibration_control$method %||% "none",
      calibration_prior_sd = calibration_control$prior_sd %||% NULL,
      likelihood = likelihood,
      schema_dropout = schema_dropout
    )
    model_info_local <- neural_set_pairwise_context_model_info(
      info = model_info_local,
      pairwise_context_mode = pairwise_context_mode,
      has_candidate_group_context = has_candidate_group_context,
      has_respondent_group_context = has_respondent_group_context,
      has_relation_token_context = has_relation_context,
      has_stage_context = stage_context_enabled,
      has_matchup_context = use_matchup_token,
      party_missing_label = party_missing_label,
      resp_party_missing_label = resp_party_missing_label,
      n_resp_party_levels = n_resp_party_levels,
      party_missing_index = party_missing_index,
      resp_party_missing_index = resp_party_missing_index,
      context_present_masking = context_present_masking
    )

    embed_candidate <- function(X_idx, party_idx, resp_p, experiment_idx = NULL,
                                return_mask = FALSE, context_present = NULL,
                                schema_dropout_masks = NULL) {
      neural_build_candidate_tokens_hard(X_idx, party_idx,
                                         model_info = model_info_local,
                                         resp_party_idx = resp_p,
                                         experiment_idx = experiment_idx,
                                         params = params_view,
                                         return_mask = return_mask,
                                         context_present = context_present,
                                         schema_dropout_masks = schema_dropout_masks)
    }

    run_transformer <- function(tokens, token_mask = NULL, return_details = FALSE) {
      neural_run_transformer(tokens,
                             model_info = transformer_model_info,
                             params = params_view,
                             token_mask = token_mask,
                             return_details = return_details)
    }

    do_forward_and_lik_ <- function(Xb, pb, resp_p, resp_c,
                                    resp_c_present = NULL, experiment_idx = NULL,
                                    likelihood_code_b = NULL,
                                    n_outcomes_b = NULL,
                                    Yb,
                                    obs_scale_b = obs_scale) {
      N_batch <- ai(Xb$shape[[1]])
      schema_dropout_context <- neural_sample_schema_dropout_masks(
        model_info_local,
        n_batch = N_batch
      )
      schema_dropout_candidate <- neural_sample_schema_dropout_masks(
        model_info_local,
        n_batch = N_batch
      )
      choice_tok <- neural_build_choice_token(model_info_local, params_view)
      choice_tok <- choice_tok * strenv$jnp$ones(list(N_batch, 1L, 1L))
      choice_mask <- strenv$jnp$ones(list(N_batch, 1L), dtype = ddtype_)
      ctx_info <- neural_build_context_tokens_batch(model_info = model_info_local,
                                                    resp_party_idx = resp_p,
                                                    resp_cov = resp_c,
                                                    resp_cov_present = resp_c_present,
                                                    experiment_idx = experiment_idx,
                                                    params = params_view,
                                                    return_mask = TRUE,
                                                    schema_dropout_masks = schema_dropout_context)
      ctx_tokens <- ctx_info$tokens %||% NULL
      ctx_mask <- ctx_info$mask %||% NULL
      cand_info <- embed_candidate(
        Xb,
        pb,
        resp_p,
        experiment_idx,
        return_mask = TRUE,
        schema_dropout_masks = schema_dropout_candidate
      )
      cand_tokens <- cand_info$tokens
      cand_mask <- cand_info$mask
      token_parts <- list(choice_tok)
      if (!is.null(ctx_tokens)) {
        token_parts <- c(token_parts, list(ctx_tokens))
      }
      token_parts <- c(token_parts, list(cand_tokens))
      tokens <- strenv$jnp$concatenate(token_parts, axis = 1L)
      token_mask <- if (!is.null(ctx_tokens)) {
        strenv$jnp$concatenate(list(choice_mask, ctx_mask, cand_mask), axis = 1L)
      } else {
        strenv$jnp$concatenate(list(choice_mask, cand_mask), axis = 1L)
      }
      transformer_out <- run_transformer(tokens, token_mask = token_mask, return_details = TRUE)
      choice_out <- neural_extract_choice_representation(
        transformer_out,
        token_mask = token_mask,
        include_tail_summary = neural_fused_classification_summary_enabled(model_info_local)
      )
      logits <- neural_linear_head(choice_out, W_out, b_out)
      if (isTRUE(neural_has_low_rank_interaction(params_view, model_info_local))) {
        logits <- logits + neural_low_rank_single_utility_prepared(
          params = params_view,
          model_info = model_info_local,
          X_idx = Xb,
          party_idx = pb,
          resp_party_idx = resp_p,
          resp_cov = resp_c,
          resp_cov_present = resp_c_present,
          experiment_idx = experiment_idx,
          schema_dropout_context = schema_dropout_context,
          schema_dropout_candidate = schema_dropout_candidate,
          transformer_model_info = transformer_model_info,
          out_dim = ai(logits$shape[[2]]),
          dtype = logits$dtype
        )
      }
      logits <- logits + neural_additive_single_logits_prepared(
        params = params_view,
        model_info = model_info_local,
        X_idx = Xb,
        party_idx = pb,
        resp_party_idx = resp_p,
        experiment_idx = experiment_idx,
        context_present = NULL,
        schema_dropout_masks = schema_dropout_candidate,
        out_dim = ai(logits$shape[[2]]),
        dtype = logits$dtype
      )

      apply_observation_likelihood(
        logits = logits,
        Yb = Yb,
        likelihood_code_obs = likelihood_code_b,
        n_outcomes_obs = n_outcomes_b,
        sigma = sigma,
        site_name = "obs",
        obs_scale = obs_scale_b,
        experiment_index_obs = experiment_idx,
        ordinal_thresholds = params_view$ordinal_thresholds %||% NULL,
        params = params_view
      )
    }

    local_lik <- function() {
      if (isTRUE(subsample_method_model %in% c("batch", "batch_vi"))) {
        with(strenv$numpyro$plate("data", size = N_local,
                                  # Clamp subsample_size to the available data to avoid
                                  # plate() errors when batch_size > N_local.
                                  subsample_size = ai(min(ai(mcmc_control$batch_size), N_local)),
                                  dim = -1L) %as% "idx", {
                                    Xb <- strenv$jnp$take(X, idx, axis = 0L)
                                    pb <- strenv$jnp$take(party, idx, axis = 0L)
                                    resp_p_b <- strenv$jnp$take(resp_party, idx, axis = 0L)
                                    resp_c_b <- strenv$jnp$take(resp_cov, idx, axis = 0L)
                                    resp_c_present_b <- strenv$jnp$take(resp_cov_present, idx, axis = 0L)
                                    experiment_idx_b <- if (is.null(experiment_index)) NULL else {
                                      strenv$jnp$take(experiment_index, idx, axis = 0L)
                                    }
                                    likelihood_code_b <- if (is.null(likelihood_code)) NULL else {
                                      strenv$jnp$take(likelihood_code, idx, axis = 0L)
                                    }
                                    n_outcomes_b <- if (is.null(n_outcomes_obs)) NULL else {
                                      strenv$jnp$take(n_outcomes_obs, idx, axis = 0L)
                                    }
                                    Yb <- if (is.null(Y_obs)) NULL else strenv$jnp$take(Y_obs, idx, axis = 0L)
                                    obs_scale_b <- if (is.null(obs_scale) ||
                                                       (!reticulate::is_py_object(obs_scale) && length(obs_scale) == 1L)) {
                                      obs_scale
                                    } else if (reticulate::is_py_object(obs_scale)) {
                                      strenv$jnp$take(obs_scale, idx, axis = 0L)
                                    } else {
                                      strenv$jnp$take(strenv$jnp$array(obs_scale, dtype = ddtype_), idx, axis = 0L)
                                    }
                                    do_forward_and_lik_(Xb, pb, resp_p_b, resp_c_b,
                                                        resp_c_present_b, experiment_idx_b,
                                                        likelihood_code_b, n_outcomes_b, Yb,
                                                        obs_scale_b = obs_scale_b)
                                  })
      } else {
        with(strenv$numpyro$plate("data", size = N_local, dim = -1L), {
          do_forward_and_lik_(X, party, resp_party, resp_cov,
                              resp_cov_present, experiment_index,
                              likelihood_code, n_outcomes_obs, Y_obs)
        })
      }
    }

    local_lik()
    neural_emit_moe_load_balance_factor(
      if (isTRUE(compact_training)) {
        as.numeric(compact_model_n_obs)
      } else {
        as.numeric(N_local)
      }
    )
  }

  # Cross-fitted out-of-sample fit metrics (computed before final full-data fit).
  fit_metrics <- NULL
  neural_skip_oos_eval <- isTRUE(neural_oos_eval_internal_flag)
  if (!isTRUE(neural_skip_oos_eval) && isTRUE(eval_control$enabled)) {
    fit_metrics <- local({
      restore_rng_state <- function(old_seed) {
        if (is.null(old_seed)) {
          if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
            rm(".Random.seed", envir = .GlobalEnv)
          }
        } else {
          assign(".Random.seed", old_seed, envir = .GlobalEnv)
        }
      }
      old_seed_state <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
      } else {
        NULL
      }
      on.exit(restore_rng_state(old_seed_state), add = FALSE)

      fit_metrics <- tryCatch({
        fit_metrics <- NULL
        n_total <- length(Y_use)
        if (n_total > 0L) {
        eval_idx <- seq_len(n_total)
        if (!is.null(eval_control$max_n) &&
            is.finite(eval_control$max_n) &&
            eval_control$max_n > 0L &&
            eval_control$max_n < n_total) {
          eval_seed <- eval_control$seed
          if (is.null(eval_seed) || is.na(eval_seed)) {
            eval_seed <- 123L
          }
          rng <- strenv$np$random$default_rng(as.integer(eval_seed))
          idx_py <- rng$choice(as.integer(n_total),
                               size = as.integer(eval_control$max_n),
                               replace = FALSE)
          eval_idx <- as.integer(reticulate::py_to_r(idx_py)) + 1L
        }

        y_all <- NULL
        likelihood_code_all <- NULL
        n_outcomes_all <- NULL
        stratify_y <- NULL
        y_levels_full <- NULL
        ok_rows <- rep(TRUE, n_total)
        if (likelihood == "mixed") {
          y_all <- as.numeric(Y_use)
          likelihood_code_all <- universal_likelihood_code_use
          n_outcomes_all <- universal_n_outcomes_use_int
          ok_rows <- mixed_row_is_valid_r(
            y = y_all,
            likelihood_code_obs = likelihood_code_all,
            n_outcomes_obs = n_outcomes_all
          )
          stratify_y <- mixed_eval_strata_r(
            y = y_all,
            likelihood_code_obs = likelihood_code_all,
            experiment_index = experiment_index_use,
            n_outcomes_obs = n_outcomes_all
          )
        } else if (likelihood == "bernoulli") {
          y_all <- as.numeric(Y_use)
          ok_rows <- is.finite(y_all) & (y_all %in% c(0, 1))
        } else if (likelihood == "categorical") {
          y_levels_full <- levels(as.factor(Y_use))
          y_all <- match(as.character(Y_use), y_levels_full) - 1L
          ok_rows <- !is.na(y_all)
        } else {
          y_all <- as.numeric(Y_use)
          ok_rows <- is.finite(y_all)
        }
        if (!all(ok_rows)) {
          eval_idx <- eval_idx[ok_rows[eval_idx]]
        }

        n_eval_total <- length(eval_idx)
        if (n_eval_total >= 2L) {
          subset_note <- if (n_eval_total < n_total) {
            sprintf("n=%d/%d", n_eval_total, n_total)
          } else {
            sprintf("n=%d", n_eval_total)
          }

	        cluster_obs <- NULL
	        if (exists("varcov_cluster_variable_", inherits = TRUE)) {
	          cluster_raw <- get("varcov_cluster_variable_", inherits = TRUE)
	          if (!is.null(cluster_raw) && length(cluster_raw) > 0L) {
	            if (isTRUE(universal_mixed_mode) &&
	                !is.null(pair_mat) && nrow(pair_mat) > 0) {
	              need <- suppressWarnings(
	                max(c(pair_mat[, 1], universal_single_rows), na.rm = TRUE)
	              )
	              if (is.finite(need) && length(cluster_raw) >= need) {
	                cluster_obs <- c(
	                  cluster_raw[pair_mat[, 1]],
	                  cluster_raw[universal_single_rows]
	                )
	              }
	            } else if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0) {
	              need <- suppressWarnings(max(pair_mat[, 1], na.rm = TRUE))
	              if (is.finite(need) && length(cluster_raw) >= need) {
	                cluster_obs <- cluster_raw[pair_mat[, 1]]
	              }
	            } else if (!pairwise_mode && length(cluster_raw) == n_total) {
	              cluster_obs <- cluster_raw
	            }
	          }
	        }
	        if (is.null(cluster_obs) || length(cluster_obs) != n_total) {
	          cluster_raw <- NULL

	          subset_by_indi <- function(x) {
	            if (is.null(x) || length(x) == 0L) {
	              return(NULL)
	            }
	            x <- as.vector(x)
	            if (!exists("indi_", inherits = TRUE)) {
	              return(x)
	            }
	            idx <- get("indi_", inherits = TRUE)
	            if (is.null(idx) || length(idx) == 0L) {
	              return(x)
	            }
	            if (length(x) == length(Y_)) {
	              return(x)
	            }
	            max_idx <- suppressWarnings(max(as.integer(idx), na.rm = TRUE))
	            if (is.finite(max_idx) && length(x) >= max_idx) {
	              return(x[idx])
	            }
	            x
	          }

	          resp_id <- if (exists("respondent_id", inherits = TRUE)) {
	            subset_by_indi(get("respondent_id", inherits = TRUE))
	          } else {
	            NULL
	          }
	          task_id <- if (exists("respondent_task_id", inherits = TRUE)) {
	            subset_by_indi(get("respondent_task_id", inherits = TRUE))
	          } else {
	            NULL
	          }

	          if (!is.null(resp_id) && length(resp_id) > 0L) {
	            cluster_raw <- resp_id
	          } else if (!is.null(task_id) && length(task_id) > 0L) {
	            cluster_raw <- task_id
	          }

	          if (!is.null(cluster_raw) && length(cluster_raw) > 0L) {
	            if (isTRUE(universal_mixed_mode) &&
	                !is.null(pair_mat) && nrow(pair_mat) > 0) {
	              need <- suppressWarnings(
	                max(c(pair_mat[, 1], universal_single_rows), na.rm = TRUE)
	              )
	              if (is.finite(need) && length(cluster_raw) >= need) {
	                cluster_obs <- c(
	                  cluster_raw[pair_mat[, 1]],
	                  cluster_raw[universal_single_rows]
	                )
	              }
	            } else if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0) {
	              cluster_obs <- cluster_raw[pair_mat[, 1]]
	            } else if (!pairwise_mode && length(cluster_raw) == n_total) {
	              cluster_obs <- cluster_raw
	            }
	          }
	        }
	        cluster_eval <- if (!is.null(cluster_obs) && length(cluster_obs) == n_total) {
	          cluster_obs[eval_idx]
	        } else {
	          NULL
	        }

	        n_folds <- eval_control$n_folds
	        if (is.null(n_folds) || !is.finite(n_folds)) {
	          n_folds <- 3L
        }
        n_folds <- as.integer(n_folds)
        if (n_folds < 2L) {
          n_folds <- 2L
        }

	        fold_y <- if (identical(likelihood, "mixed")) {
	          stratify_y[eval_idx]
	        } else {
	          y_all[eval_idx]
	        }
	        folds_out <- cs_make_stratified_folds(
	          n = n_eval_total,
	          n_folds = n_folds,
	          y = fold_y,
	          cluster = cluster_eval,
	          seed = eval_control$seed
	        )
        if (!is.null(folds_out) && !is.null(folds_out$fold_id)) {
          fold_id <- folds_out$fold_id
          n_folds_use <- as.integer(folds_out$n_folds)

          init_model <- body(generate_ModelOutcome_neural)

          format_metric <- function(label, value, digits = 4) {
            if (is.null(value) || !is.finite(value)) return(NULL)
            fmt <- paste0("%s=%.", digits, "f")
            sprintf(fmt, label, value)
          }

          compute_metrics <- function(y_eval, pred_eval, idx_use = NULL) {
            if (identical(likelihood, "mixed")) {
              return(compute_mixed_outcome_metrics_r(
                y_eval = y_eval,
                pred_eval = pred_eval,
                likelihood_code_obs = likelihood_code_all[idx_use],
                n_outcomes_obs = n_outcomes_all[idx_use],
                task_mode_obs = if (!is.null(universal_task_mode_use)) {
                  universal_task_mode_use[idx_use]
                } else {
                  NULL
                },
                experiment_index = if (!is.null(experiment_index_use)) {
                  experiment_index_use[idx_use]
                } else {
                  NULL
                }
              ))
            }
            if (identical(likelihood, "ordinal")) {
              return(ordinal_metrics_r(
                y_eval = y_eval,
                prob_mat = pred_eval,
                n_outcomes_obs = rep.int(as.integer(nOutcomes), length(y_eval))
              ))
            }
            cs_compute_outcome_metrics(
              y_eval = y_eval,
              pred_eval = pred_eval,
              likelihood = likelihood
            )
          }

          format_mixed_metric_items <- function(metrics) {
            if (is.null(metrics)) {
              return(character(0))
            }
            items <- Filter(Negate(is.null), list(
              format_metric("NLL", metrics$nll, 4)
            ))
            by_family <- metrics$by_family %||% list()
            if (!is.null(by_family$bernoulli)) {
              items <- c(items, Filter(Negate(is.null), list(
                format_metric("BernAUC", by_family$bernoulli$auc, 4),
                format_metric("BernLL", by_family$bernoulli$log_loss, 4),
                format_metric("BernAcc", by_family$bernoulli$accuracy, 3),
                format_metric("BernBrier", by_family$bernoulli$brier, 4)
              )))
            }
            if (!is.null(by_family$categorical)) {
              items <- c(items, Filter(Negate(is.null), list(
                format_metric("CatLL", by_family$categorical$log_loss, 4),
                format_metric("CatAcc", by_family$categorical$accuracy, 3)
              )))
            }
            if (!is.null(by_family$normal)) {
              items <- c(items, Filter(Negate(is.null), list(
                format_metric("NormRMSE", by_family$normal$rmse, 4),
                format_metric("NormMAE", by_family$normal$mae, 4),
                format_metric("NormNLL", by_family$normal$nll, 4)
              )))
            }
            if (!is.null(by_family$ordinal)) {
              items <- c(items, Filter(Negate(is.null), list(
                format_metric("OrdLL", by_family$ordinal$log_loss, 4),
                format_metric("OrdAcc", by_family$ordinal$accuracy, 3),
                format_metric("OrdMAE", by_family$ordinal$mae, 4),
                format_metric("RPS", by_family$ordinal$ranked_probability_score, 4)
              )))
            }
            items
          }

          make_na_metrics <- function(likelihood_value) {
            if (identical(likelihood_value, "mixed")) {
              return(list(
                likelihood = "mixed",
                n_eval = 0L,
                n_obs = 0L,
                nll = NA_real_,
                by_family = list()
              ))
            }
            if (identical(likelihood_value, "bernoulli")) {
              return(list(
                likelihood = likelihood_value,
                n_eval = 0L,
                auc = NA_real_,
                log_loss = NA_real_,
                accuracy = NA_real_,
                brier = NA_real_
              ))
            }
            if (identical(likelihood_value, "categorical")) {
              return(list(
                likelihood = likelihood_value,
                n_eval = 0L,
                log_loss = NA_real_,
                accuracy = NA_real_
              ))
            }
            if (identical(likelihood_value, "ordinal")) {
              return(list(
                likelihood = likelihood_value,
                n_eval = 0L,
                n_obs = 0L,
                log_loss = NA_real_,
                nll = NA_real_,
                accuracy = NA_real_,
                mae = NA_real_,
                rmse = NA_real_,
                ranked_probability_score = NA_real_
              ))
            }
            list(
              likelihood = likelihood_value,
              n_eval = 0L,
              rmse = NA_real_,
              mae = NA_real_,
              nll = NA_real_
            )
          }

          make_failed_fold_metrics <- function(fold, test_idx, train_idx, reason) {
            out <- make_na_metrics(likelihood)
            out$fold <- as.integer(fold)
            out$eval_note <- "oos"
            out$n_test <- length(test_idx)
            out$n_train <- length(train_idx)
            out$prediction_status <- "failed"
            out$failure_reason <- as.character(reason %||% "prediction_failed")
            out$n_eval_success <- 0L
            out$n_eval_failed <- length(test_idx)
            out
          }

          subset_oos_prediction <- function(idx) {
            if (likelihood == "mixed") {
              list(
                logits = pred_oos$logits[idx, , drop = FALSE],
                sigma = pred_oos$sigma[idx]
              )
            } else if (likelihood == "bernoulli") {
              pred_oos[idx]
            } else if (likelihood == "categorical") {
              pred_oos[idx, , drop = FALSE]
            } else if (likelihood == "ordinal") {
              pred_oos[idx, , drop = FALSE]
            } else {
              list(mu = pred_oos$mu[idx],
                   sigma = pred_oos$sigma[idx])
            }
          }

          pred_oos <- NULL
          if (likelihood == "mixed") {
            pred_oos <- list(
              logits = matrix(NA_real_, nrow = n_total, ncol = as.integer(nOutcomes)),
              sigma = rep(NA_real_, n_total)
            )
          } else if (likelihood == "bernoulli") {
            pred_oos <- rep(NA_real_, n_total)
          } else if (likelihood %in% c("categorical", "ordinal")) {
            pred_oos <- matrix(NA_real_, nrow = n_total, ncol = as.integer(nOutcomes))
          } else {
            pred_oos <- list(mu = rep(NA_real_, n_total),
                             sigma = rep(NA_real_, n_total))
          }

          by_fold <- vector("list", n_folds_use)
          successful_eval_idx <- integer(0)
          failed_eval_idx <- integer(0)
          for (fold in seq_len(n_folds_use)) {
            test_pos <- which(fold_id == fold)
            train_pos <- which(fold_id != fold)
            test_idx <- eval_idx[test_pos]
            train_idx <- eval_idx[train_pos]
            if (!length(test_pos) || !length(train_pos)) {
              by_fold[[fold]] <- make_failed_fold_metrics(
                fold,
                test_idx,
                train_idx,
                "empty_train_or_test_fold"
              )
              failed_eval_idx <- c(failed_eval_idx, test_idx)
              next
            }

            fallback <- NULL
            if (likelihood == "mixed") {
              fallback_logits <- matrix(0, nrow = length(test_idx), ncol = as.integer(nOutcomes))
              fallback_sigma <- rep(1, length(test_idx))
              train_code <- likelihood_code_all[train_idx]
              train_n_out <- n_outcomes_all[train_idx]
              train_y <- y_all[train_idx]

              bern_train <- train_y[train_code == 0L]
              bern_p <- mean(bern_train, na.rm = TRUE)
              if (!is.finite(bern_p)) bern_p <- 0.5
              bern_p <- min(max(bern_p, 1e-6), 1 - 1e-6)

              norm_train <- train_y[train_code == 2L]
              norm_mu <- mean(norm_train, na.rm = TRUE)
              if (!is.finite(norm_mu)) norm_mu <- 0
              norm_sigma <- suppressWarnings(stats::sd(norm_train, na.rm = TRUE))
              if (!is.finite(norm_sigma) || norm_sigma <= 0) norm_sigma <- 1

              test_code <- likelihood_code_all[test_idx]
              test_n_out <- n_outcomes_all[test_idx]
              for (row_idx in seq_along(test_idx)) {
                code_i <- test_code[[row_idx]]
                if (identical(code_i, 0L)) {
                  fallback_logits[row_idx, 1L] <- stats::qlogis(bern_p)
                } else if (identical(code_i, 1L)) {
                  k_i <- max(2L, min(as.integer(test_n_out[[row_idx]]), as.integer(nOutcomes)))
                  train_cat <- train_y[train_code == 1L & train_n_out == test_n_out[[row_idx]]]
                  probs <- vapply(0:(k_i - 1L), function(k_) {
                    mean(train_cat == k_, na.rm = TRUE)
                  }, numeric(1))
                  if (!any(is.finite(probs)) || sum(probs, na.rm = TRUE) <= 0) {
                    probs <- rep(1 / k_i, k_i)
                  } else {
                    probs[!is.finite(probs)] <- 0
                    probs_sum <- sum(probs)
                    probs <- if (probs_sum <= 0) rep(1 / k_i, k_i) else probs / probs_sum
                  }
                  fallback_logits[row_idx, seq_len(k_i)] <- log(pmax(probs, 1e-12))
                } else {
                  fallback_logits[row_idx, 1L] <- norm_mu
                  fallback_sigma[[row_idx]] <- norm_sigma
                }
              }
              fallback <- list(
                logits = fallback_logits,
                sigma = fallback_sigma
              )
            } else if (likelihood == "bernoulli") {
              y_train <- y_all[train_idx]
              fallback_val <- mean(y_train, na.rm = TRUE)
              if (!is.finite(fallback_val)) fallback_val <- 0.5
              fallback_val <- min(max(fallback_val, 0), 1)
              fallback <- rep(fallback_val, length(test_idx))
            } else if (likelihood == "categorical") {
              y_train <- y_all[train_idx]
              Kc <- as.integer(nOutcomes)
              probs <- vapply(0:(Kc - 1L), function(k_) mean(y_train == k_, na.rm = TRUE), numeric(1))
              if (!any(is.finite(probs)) || sum(probs, na.rm = TRUE) <= 0) {
                probs <- rep(1 / Kc, Kc)
              } else {
                probs[!is.finite(probs)] <- 0
                s <- sum(probs)
                if (s <= 0) probs <- rep(1 / Kc, Kc) else probs <- probs / s
              }
              fallback <- matrix(rep(probs, each = length(test_idx)),
                                 nrow = length(test_idx),
                                 ncol = Kc)
            } else {
              y_train <- y_all[train_idx]
              mu0 <- mean(y_train, na.rm = TRUE)
              if (!is.finite(mu0)) mu0 <- 0
              sigma0 <- suppressWarnings(stats::sd(y_train, na.rm = TRUE))
              if (!is.finite(sigma0) || sigma0 <= 0) sigma0 <- 1
              fallback <- list(mu = rep(mu0, length(test_idx)),
                               sigma = rep(sigma0, length(test_idx)))
            }

            raw_train <- if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0) {
              as.integer(c(pair_mat[train_idx, 1], pair_mat[train_idx, 2]))
            } else {
              as.integer(train_idx)
            }
            raw_train <- sort(unique(raw_train))
            if (!length(raw_train)) {
              by_fold[[fold]] <- make_failed_fold_metrics(
                fold,
                test_idx,
                train_idx,
                "empty_raw_training_rows"
              )
              failed_eval_idx <- c(failed_eval_idx, test_idx)
              next
            }

            fold_env <- new.env(parent = environment())
            fold_env$neural_oos_eval_internal <- TRUE
            if (!identical(likelihood, "mixed")) {
              fold_env$neural_likelihood_override <- likelihood
              fold_env$neural_nOutcomes_override <- nOutcomes
            }
            fold_env$party_levels_fixed <- party_levels
            fold_env$resp_party_levels_fixed <- resp_party_levels
            if (!is.null(y_levels_full)) {
              fold_env$neural_y_levels_override <- y_levels_full
            }

            fold_env$W_ <- W_[raw_train, , drop = FALSE]
            fold_env$Y_ <- Y_[raw_train]
            fold_env$pair_id_ <- if (!is.null(pair_id_)) pair_id_[raw_train] else NULL
            fold_env$profile_order_ <- if (!is.null(profile_order_)) profile_order_[raw_train] else NULL
            fold_env$X <- if (!is.null(X_)) X_[raw_train, , drop = FALSE] else NULL
            fold_env$X_ <- if (!is.null(X_)) X_[raw_train, , drop = FALSE] else NULL
            fold_env$X_present <- if (!is.null(X_present_)) X_present_[raw_train, , drop = FALSE] else NULL
            fold_env$X_present_ <- if (!is.null(X_present_)) X_present_[raw_train, , drop = FALSE] else NULL
            if (!is.null(neural_token_info_use) && length(neural_token_info_use) > 0L) {
              fold_token_info <- neural_token_info_use
              if (!is.null(experiment_index_all)) {
                fold_token_info$experiment_index <- experiment_index_all[raw_train]
              }
              universal_training_info <- fold_token_info$foundation_universal_training %||% NULL
              if (!is.null(universal_training_info)) {
                if (!is.null(universal_training_info$task_mode_by_row)) {
                  universal_training_info$task_mode_by_row <- universal_training_info$task_mode_by_row[raw_train]
                }
                if (!is.null(universal_training_info$likelihood_by_row)) {
                  universal_training_info$likelihood_by_row <- universal_training_info$likelihood_by_row[raw_train]
                }
                if (!is.null(universal_training_info$n_outcomes_by_row)) {
                  universal_training_info$n_outcomes_by_row <- universal_training_info$n_outcomes_by_row[raw_train]
                }
                fold_token_info$foundation_universal_training <- universal_training_info
              }
              fold_env$neural_token_info <- fold_token_info
            } else {
              fold_env$neural_token_info <- neural_token_info_use
            }
            fold_env$competing_group_competition_variable_candidate_ <- if (!is.null(competing_group_competition_variable_candidate_)) {
              competing_group_competition_variable_candidate_[raw_train]
            } else {
              NULL
            }
            fold_env$competing_group_variable_respondent_ <- if (!is.null(competing_group_variable_respondent_)) {
              competing_group_variable_respondent_[raw_train]
            } else {
              NULL
            }
            fold_env$competing_group_variable_candidate_ <- if (!is.null(competing_group_variable_candidate_)) {
              competing_group_variable_candidate_[raw_train]
            } else {
              NULL
            }
            if (exists("varcov_cluster_variable_", inherits = TRUE)) {
              cluster_full <- get("varcov_cluster_variable_", inherits = TRUE)
              fold_env$varcov_cluster_variable_ <- if (!is.null(cluster_full)) cluster_full[raw_train] else NULL
            } else {
              fold_env$varcov_cluster_variable_ <- NULL
            }
            if (exists("indi_", inherits = TRUE)) {
              indi_full <- get("indi_", inherits = TRUE)
              fold_env$indi_ <- if (!is.null(indi_full)) indi_full[raw_train] else raw_train
            } else {
            fold_env$indi_ <- raw_train
            }

            fold_seed <- eval_control$seed
            if (is.null(fold_seed) || is.na(fold_seed) || !is.finite(fold_seed)) {
              fold_seed <- 123L
            }
            set.seed(as.integer(fold_seed) + as.integer(fold))

            fit_failure_reason <- NULL
            fit_ok <- tryCatch({
              eval(init_model, envir = fold_env)
              is.function(fold_env$my_model)
            }, error = function(e) {
              fit_failure_reason <<- conditionMessage(e)
              FALSE
            })

            pred_fold <- NULL
            if (isTRUE(fit_ok)) {
              prediction_failure_reason <- NULL
              pred_fold <- tryCatch({
                if (pairwise_mode) {
                  X_left_test <- X_left[test_idx, , drop = FALSE]
                  X_right_test <- X_right[test_idx, , drop = FALSE]
                  party_left_test <- party_left[test_idx]
                  party_right_test <- party_right[test_idx]
                  resp_party_test <- resp_party_use[test_idx]
                  resp_cov_test <- if (!is.null(X_use) && n_resp_covariates > 0L) {
                    X_use[test_idx, , drop = FALSE]
                  } else {
                    NULL
                  }
                  resp_cov_present_test <- if (!is.null(X_present_use) && n_resp_covariates > 0L) {
                    X_present_use[test_idx, , drop = FALSE]
                  } else {
                    NULL
                  }
                  experiment_idx_test <- if (!is.null(experiment_index_use)) {
                    experiment_index_use[test_idx]
                  } else {
                    NULL
                  }
                  fold_env$my_model(
                    X_left_new = X_left_test,
                    X_right_new = X_right_test,
                    party_left_new = party_left_test,
                    party_right_new = party_right_test,
                    resp_party_new = resp_party_test,
                    resp_cov_new = resp_cov_test,
                    resp_cov_present_new = resp_cov_present_test,
                    experiment_idx_new = experiment_idx_test,
                    return_logits = identical(likelihood, "mixed")
                  )
                } else {
                  X_test <- X_single[test_idx, , drop = FALSE]
                  party_test <- party_single[test_idx]
                  resp_party_test <- resp_party_use[test_idx]
                  resp_cov_test <- if (!is.null(X_use) && n_resp_covariates > 0L) {
                    X_use[test_idx, , drop = FALSE]
                  } else {
                    NULL
                  }
                  resp_cov_present_test <- if (!is.null(X_present_use) && n_resp_covariates > 0L) {
                    X_present_use[test_idx, , drop = FALSE]
                  } else {
                    NULL
                  }
                  experiment_idx_test <- if (!is.null(experiment_index_use)) {
                    experiment_index_use[test_idx]
                  } else {
                    NULL
                  }
                  fold_env$my_model(
                    X_new = X_test,
                    party_new = party_test,
                    resp_party_new = resp_party_test,
                    resp_cov_new = resp_cov_test,
                    resp_cov_present_new = resp_cov_present_test,
                    experiment_idx_new = experiment_idx_test,
                    return_logits = identical(likelihood, "mixed")
                  )
                }
              }, error = function(e) {
                prediction_failure_reason <<- conditionMessage(e)
                NULL
              })
            } else {
              prediction_failure_reason <- fit_failure_reason %||% "fold_fit_failed"
            }

            if (is.null(pred_fold)) {
              by_fold[[fold]] <- make_failed_fold_metrics(
                fold,
                test_idx,
                train_idx,
                prediction_failure_reason %||% "prediction_failed"
              )
              failed_eval_idx <- c(failed_eval_idx, test_idx)
              next
            }

            prediction_failure_reason <- NULL
            fold_metrics <- tryCatch({
            if (likelihood == "mixed") {
              train_family_present <- unique(likelihood_code_all[train_idx])
              test_code <- likelihood_code_all[test_idx]
              if (!all(test_code %in% train_family_present)) {
                by_fold[[fold]] <- make_failed_fold_metrics(
                  fold,
                  test_idx,
                  train_idx,
                  "missing_training_likelihood_family"
                )
                failed_eval_idx <- c(failed_eval_idx, test_idx)
                next
              }
              pred_oos$logits[test_idx, ] <- as.matrix(pred_fold$logits)
              pred_oos$sigma[test_idx] <- as.numeric(pred_fold$sigma)
              fold_metrics <- compute_metrics(
                y_all[test_idx],
                pred_fold,
                idx_use = test_idx
              )
            } else if (likelihood == "bernoulli") {
              pred_oos[test_idx] <- as.numeric(pred_fold)
              fold_metrics <- compute_metrics(y_all[test_idx], pred_fold)
            } else if (likelihood == "categorical") {
              pred_oos[test_idx, ] <- as.matrix(pred_fold)
              fold_metrics <- compute_metrics(y_all[test_idx], pred_fold)
            } else {
              pred_oos$mu[test_idx] <- as.numeric(pred_fold$mu)
              pred_oos$sigma[test_idx] <- as.numeric(pred_fold$sigma)
              fold_metrics <- compute_metrics(y_all[test_idx], pred_fold)
            }
            fold_metrics
            }, error = function(e) {
              prediction_failure_reason <<- conditionMessage(e)
              NULL
            })
            if (is.null(fold_metrics)) {
              by_fold[[fold]] <- make_failed_fold_metrics(
                fold,
                test_idx,
                train_idx,
                prediction_failure_reason %||% "prediction_metric_failed"
              )
              failed_eval_idx <- c(failed_eval_idx, test_idx)
              next
            }
            fold_metrics$fold <- fold
            fold_metrics$eval_note <- "oos"
            fold_metrics$n_test <- length(test_idx)
            fold_metrics$n_train <- length(train_idx)
            fold_metrics$prediction_status <- "ok"
            fold_metrics$failure_reason <- NA_character_
            fold_metrics$n_eval_success <- as.integer(fold_metrics$n_eval %||% length(test_idx))
            fold_metrics$n_eval_failed <- max(0L, length(test_idx) - fold_metrics$n_eval_success)
            successful_eval_idx <- c(successful_eval_idx, test_idx)
            fold_metric_items <- if (likelihood == "mixed") {
              format_mixed_metric_items(fold_metrics)
            } else if (likelihood == "bernoulli") {
              Filter(Negate(is.null), list(
                format_metric("AUC", fold_metrics$auc, 4),
                format_metric("LogLoss", fold_metrics$log_loss, 4),
                format_metric("Acc", fold_metrics$accuracy, 3),
                format_metric("Brier", fold_metrics$brier, 4)
              ))
            } else if (likelihood == "categorical") {
              Filter(Negate(is.null), list(
                format_metric("LogLoss", fold_metrics$log_loss, 4),
                format_metric("Acc", fold_metrics$accuracy, 3)
              ))
            } else {
              Filter(Negate(is.null), list(
                format_metric("RMSE", fold_metrics$rmse, 4),
                format_metric("MAE", fold_metrics$mae, 4),
                format_metric("NLL", fold_metrics$nll, 4)
              ))
            }
            if (length(fold_metric_items) > 0L) {
              message(sprintf("Neural OOS fold %d/%d (%s): %s",
                              fold,
                              n_folds_use,
                              ifelse(pairwise_mode, "pairwise", "single"),
                              paste(fold_metric_items, collapse = ", ")))
            }
            by_fold[[fold]] <- fold_metrics
          }

          successful_eval_idx <- sort(unique(successful_eval_idx))
          failed_eval_idx <- sort(unique(c(failed_eval_idx, setdiff(eval_idx, successful_eval_idx))))
          failed_fold_ids <- which(vapply(by_fold, function(x) {
            is.list(x) && identical(x$prediction_status, "failed")
          }, logical(1)))
          pred_eval <- NULL
          if (length(successful_eval_idx) > 0L) {
            pred_eval <- subset_oos_prediction(successful_eval_idx)
            overall_metrics <- compute_metrics(
              y_all[successful_eval_idx],
              pred_eval,
              idx_use = successful_eval_idx
            )
          } else {
            overall_metrics <- make_na_metrics(likelihood)
          }
          overall_metrics$eval_note <- sprintf("oos_%dfold", n_folds_use)
          overall_metrics$eval_subset <- subset_note
          overall_metrics$n_folds <- n_folds_use
          overall_metrics$seed <- eval_control$seed
          overall_metrics$eval_index <- successful_eval_idx
          overall_metrics$n_eval_success <- length(successful_eval_idx)
          overall_metrics$n_eval_failed <- length(failed_eval_idx)
          overall_metrics$n_failed_folds <- length(failed_fold_ids)
          overall_metrics$failed_fold_ids <- as.integer(failed_fold_ids)
          overall_metrics$prediction_status <- if (length(failed_fold_ids) < 1L) {
            "ok"
          } else if (length(successful_eval_idx) > 0L) {
            "partial"
          } else {
            "failed"
          }
          overall_metrics$failure_reason <- if (length(failed_fold_ids) < 1L) {
            NA_character_
          } else if (length(successful_eval_idx) > 0L) {
            "prediction_failed_some_folds"
          } else {
            "prediction_failed_all_folds"
          }
          overall_metrics$by_fold <- by_fold

          if (pairwise_mode && length(successful_eval_idx) > 0L) {
            stage_primary <- party_left == party_right
              if (length(stage_primary) == n_total) {
                stage_primary <- stage_primary[successful_eval_idx]
                stage_keep <- context_present_pair[successful_eval_idx]
                stage_keep[is.na(stage_keep)] <- FALSE
                stage_primary[!stage_keep] <- NA
                by_stage <- list()
                if (any(stage_primary, na.rm = TRUE)) {
                  idx0 <- which(stage_primary)
                  pred_stage <- if (likelihood == "bernoulli") {
                    pred_eval[idx0]
                  } else if (likelihood == "mixed") {
                    list(
                      logits = pred_eval$logits[idx0, , drop = FALSE],
                      sigma = pred_eval$sigma[idx0]
                    )
                  } else if (likelihood == "categorical") {
                    pred_eval[idx0, , drop = FALSE]
                  } else {
                    list(mu = pred_eval$mu[idx0],
                         sigma = pred_eval$sigma[idx0])
                  }
                  by_stage$primary <- compute_metrics(
                    y_all[successful_eval_idx][idx0],
                    pred_stage,
                    idx_use = successful_eval_idx[idx0]
                  )
                }
                if (any(!stage_primary, na.rm = TRUE)) {
                  idx1 <- which(!stage_primary)
                  pred_stage <- if (likelihood == "bernoulli") {
                    pred_eval[idx1]
                  } else if (likelihood == "mixed") {
                    list(
                      logits = pred_eval$logits[idx1, , drop = FALSE],
                      sigma = pred_eval$sigma[idx1]
                    )
                  } else if (likelihood == "categorical") {
                    pred_eval[idx1, , drop = FALSE]
                  } else {
                    list(mu = pred_eval$mu[idx1],
                         sigma = pred_eval$sigma[idx1])
                  }
                  by_stage$general <- compute_metrics(
                    y_all[successful_eval_idx][idx1],
                    pred_stage,
                    idx_use = successful_eval_idx[idx1]
                  )
                }
                if (length(by_stage) > 0L) {
                  overall_metrics$by_stage <- by_stage
                }
              }
          }

          fit_metrics <- overall_metrics

          metric_items <- if (likelihood == "mixed") {
            format_mixed_metric_items(fit_metrics)
          } else if (likelihood == "bernoulli") {
            Filter(Negate(is.null), list(
              format_metric("AUC", fit_metrics$auc, 4),
              format_metric("LogLoss", fit_metrics$log_loss, 4),
              format_metric("Acc", fit_metrics$accuracy, 3),
              format_metric("Brier", fit_metrics$brier, 4)
            ))
          } else if (likelihood == "categorical") {
            Filter(Negate(is.null), list(
              format_metric("LogLoss", fit_metrics$log_loss, 4),
              format_metric("Acc", fit_metrics$accuracy, 3)
            ))
          } else {
            Filter(Negate(is.null), list(
              format_metric("RMSE", fit_metrics$rmse, 4),
              format_metric("MAE", fit_metrics$mae, 4),
              format_metric("NLL", fit_metrics$nll, 4)
            ))
          }
          if (!is.null(fit_metrics) && length(metric_items) > 0L) {
            message(sprintf("Neural OOS fit metrics (%s, %s, %s): %s",
                            ifelse(pairwise_mode, "pairwise", "single"),
                            fit_metrics$eval_note,
                            subset_note,
                            paste(metric_items, collapse = ", ")))
          }
          }
        }
      }
      fit_metrics
      }, error = function(e) {
        message(sprintf("Neural OOS evaluation failed: %s", conditionMessage(e)))
        NULL
      })

      fit_metrics
  })
  }

  y_fac <- NULL
  if (likelihood == "categorical") {
    y_levels_override <- NULL
    if (exists("neural_y_levels_override", inherits = TRUE)) {
      y_levels_override <- get("neural_y_levels_override", inherits = TRUE)
    }
    if (!is.null(y_levels_override)) {
      y_fac <- match(as.character(Y_use), as.character(y_levels_override)) - 1L
    } else {
      y_fac <- ai(as.factor(Y_use)) - 1L
    }
    if (any(is.na(y_fac))) {
      stop("Categorical outcome contains unseen/NA levels after applying neural_y_levels_override.",
           call. = FALSE)
    }
    Y_jnp <- if (isTRUE(compact_training)) {
      NULL
    } else {
      strenv$jnp$array(ai(y_fac))$astype(strenv$jnp$int32)
    }
  } else {
    Y_jnp <- if (isTRUE(compact_training)) {
      NULL
    } else {
      strenv$jnp$array(as.numeric(Y_use))$astype(ddtype_)
    }
  }
  if (isTRUE(compact_training)) {
    resp_party_jnp <- NULL
    resp_party_pair_jnp <- NULL
    resp_party_single_jnp <- NULL
    resp_cov_jnp <- NULL
    resp_cov_pair_jnp <- NULL
    resp_cov_single_jnp <- NULL
    resp_cov_present_jnp <- NULL
    resp_cov_present_pair_jnp <- NULL
    resp_cov_present_single_jnp <- NULL
    experiment_index_jnp <- NULL
    experiment_index_pair_jnp <- NULL
    experiment_index_single_jnp <- NULL
    likelihood_code_jnp <- NULL
    likelihood_code_pair_jnp <- NULL
    likelihood_code_single_jnp <- NULL
    n_outcomes_obs_jnp <- NULL
    n_outcomes_pair_jnp <- NULL
    n_outcomes_single_jnp <- NULL
    obs_scale_jnp <- NULL
    obs_scale_pair_jnp <- NULL
    obs_scale_single_jnp <- NULL
    Y_pair_jnp <- NULL
    Y_single_jnp <- NULL
    X_left_jnp <- NULL
    X_right_jnp <- NULL
    party_left_jnp <- NULL
    party_right_jnp <- NULL
    X_single_jnp <- NULL
    party_single_jnp <- NULL
  } else {
    resp_party_jnp <- strenv$jnp$array(as.integer(resp_party_use))$astype(strenv$jnp$int32)
    if (n_resp_covariates > 0L) {
      resp_cov_jnp <- strenv$jnp$array(as.matrix(X_use))$astype(ddtype_)
      resp_cov_present_jnp <- if (!is.null(X_present_use)) {
        strenv$jnp$array(as.matrix(X_present_use))$astype(ddtype_)
      } else {
        strenv$jnp$ones(list(ai(length(Y_use)), ai(n_resp_covariates)), dtype = ddtype_)
      }
    } else {
      resp_cov_jnp <- strenv$jnp$zeros(list(ai(length(Y_use)), ai(0L)), dtype = ddtype_)
      resp_cov_present_jnp <- strenv$jnp$zeros(list(ai(length(Y_use)), ai(0L)), dtype = ddtype_)
    }
    experiment_index_jnp <- if (!is.null(experiment_index_use)) {
      strenv$jnp$array(as.integer(experiment_index_use))$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    likelihood_code_jnp <- if (isTRUE(universal_enabled) && !is.null(universal_likelihood_use)) {
      strenv$jnp$array(
        as.integer(match(universal_likelihood_use, universal_likelihood_levels) - 1L)
      )$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    n_outcomes_obs_jnp <- if (isTRUE(universal_enabled) && !is.null(universal_n_outcomes_use)) {
      strenv$jnp$array(as.integer(universal_n_outcomes_use))$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    obs_scale_jnp <- if (!is.null(universal_loss_weights)) {
      strenv$jnp$array(as.numeric(universal_loss_weights))$astype(ddtype_)
    } else {
      NULL
    }

    if (pairwise_mode) {
      X_left_jnp <- strenv$jnp$array(to_index_matrix(X_left))$astype(strenv$jnp$int32)
      X_right_jnp <- strenv$jnp$array(to_index_matrix(X_right))$astype(strenv$jnp$int32)
      party_left_jnp <- strenv$jnp$array(as.integer(party_left))$astype(strenv$jnp$int32)
      party_right_jnp <- strenv$jnp$array(as.integer(party_right))$astype(strenv$jnp$int32)
      if (isTRUE(universal_mixed_mode)) {
        Y_pair_jnp <- strenv$jnp$array(as.numeric(Y_pair_use))$astype(ddtype_)
        Y_single_jnp <- strenv$jnp$array(as.numeric(Y_single_use))$astype(ddtype_)
        resp_party_pair_jnp <- strenv$jnp$array(as.integer(resp_party_pair_use))$astype(strenv$jnp$int32)
        resp_party_single_jnp <- strenv$jnp$array(as.integer(resp_party_single_use))$astype(strenv$jnp$int32)
        if (n_resp_covariates > 0L) {
          resp_cov_pair_jnp <- strenv$jnp$array(as.matrix(X_pair_use))$astype(ddtype_)
          resp_cov_single_jnp <- strenv$jnp$array(as.matrix(X_single_cov_use))$astype(ddtype_)
          resp_cov_present_pair_jnp <- if (!is.null(X_present_pair_use)) {
            strenv$jnp$array(as.matrix(X_present_pair_use))$astype(ddtype_)
          } else {
            strenv$jnp$ones(list(ai(length(Y_pair_use)), ai(n_resp_covariates)), dtype = ddtype_)
          }
          resp_cov_present_single_jnp <- if (!is.null(X_present_single_use)) {
            strenv$jnp$array(as.matrix(X_present_single_use))$astype(ddtype_)
          } else {
            strenv$jnp$ones(list(ai(length(Y_single_use)), ai(n_resp_covariates)), dtype = ddtype_)
          }
        } else {
          resp_cov_pair_jnp <- strenv$jnp$zeros(list(ai(length(Y_pair_use)), ai(0L)), dtype = ddtype_)
          resp_cov_single_jnp <- strenv$jnp$zeros(list(ai(length(Y_single_use)), ai(0L)), dtype = ddtype_)
          resp_cov_present_pair_jnp <- strenv$jnp$zeros(list(ai(length(Y_pair_use)), ai(0L)), dtype = ddtype_)
          resp_cov_present_single_jnp <- strenv$jnp$zeros(list(ai(length(Y_single_use)), ai(0L)), dtype = ddtype_)
        }
        experiment_index_pair_jnp <- if (!is.null(experiment_index_pair_use)) {
          strenv$jnp$array(as.integer(experiment_index_pair_use))$astype(strenv$jnp$int32)
        } else {
          NULL
        }
        experiment_index_single_jnp <- if (!is.null(experiment_index_single_use)) {
          strenv$jnp$array(as.integer(experiment_index_single_use))$astype(strenv$jnp$int32)
        } else {
          NULL
        }
        likelihood_code_pair_jnp <- if (!is.null(universal_likelihood_pair_use)) {
          strenv$jnp$array(as.integer(match(universal_likelihood_pair_use, universal_likelihood_levels) - 1L))$astype(strenv$jnp$int32)
        } else {
          NULL
        }
        likelihood_code_single_jnp <- if (!is.null(universal_likelihood_single_use)) {
          strenv$jnp$array(as.integer(match(universal_likelihood_single_use, universal_likelihood_levels) - 1L))$astype(strenv$jnp$int32)
        } else {
          NULL
        }
        n_outcomes_pair_jnp <- if (!is.null(universal_n_outcomes_pair_use)) {
          strenv$jnp$array(as.integer(universal_n_outcomes_pair_use))$astype(strenv$jnp$int32)
        } else {
          NULL
        }
        n_outcomes_single_jnp <- if (!is.null(universal_n_outcomes_single_use)) {
          strenv$jnp$array(as.integer(universal_n_outcomes_single_use))$astype(strenv$jnp$int32)
        } else {
          NULL
        }
        obs_scale_pair_jnp <- if (!is.null(universal_loss_weights_pair)) {
          strenv$jnp$array(as.numeric(universal_loss_weights_pair))$astype(ddtype_)
        } else {
          NULL
        }
        obs_scale_single_jnp <- if (!is.null(universal_loss_weights_single)) {
          strenv$jnp$array(as.numeric(universal_loss_weights_single))$astype(ddtype_)
        } else {
          NULL
        }
        X_single_jnp <- strenv$jnp$array(to_index_matrix(X_single))$astype(strenv$jnp$int32)
        party_single_jnp <- strenv$jnp$array(as.integer(party_single))$astype(strenv$jnp$int32)
      }
    } else {
      X_single_jnp <- strenv$jnp$array(to_index_matrix(X_single))$astype(strenv$jnp$int32)
      party_single_jnp <- strenv$jnp$array(as.integer(party_single))$astype(strenv$jnp$int32)
    }
  }

  compact_model_n_obs <- if (isTRUE(compact_training)) {
    as.integer(length(Y_use))
  } else {
    0L
  }
  compact_svi_batch_size <- if (isTRUE(compact_training)) {
    batch_size_use <- suppressWarnings(as.integer(mcmc_control$batch_size))
    if (length(batch_size_use) != 1L || is.na(batch_size_use) || batch_size_use < 1L) {
      batch_size_use <- 1L
    }
    as.integer(min(batch_size_use, compact_model_n_obs))
  } else {
    0L
  }
  compact_sampling_obs_idx <- NULL
  compact_sampling_pool <- function() {
    if (!isTRUE(compact_training)) {
      return(integer(0))
    }
    pool <- compact_sampling_obs_idx
    if (is.null(pool)) {
      pool <- seq_len(compact_model_n_obs)
    }
    pool <- as.integer(pool)
    pool[!is.na(pool) & pool >= 1L & pool <= compact_model_n_obs]
  }
  compact_sampling_n_obs <- function() {
    pool <- compact_sampling_pool()
    as.integer(length(pool))
  }
  compact_balanced_sampling <- neural_resolve_balanced_sampling(
    mcmc_control$balanced_sampling %||% NULL
  )
  if (isTRUE(compact_balanced_sampling$enabled) && !isTRUE(compact_training)) {
    stop("Balanced study/respondent sampling currently requires compact batch_vi training.",
         call. = FALSE)
  }
  if (isTRUE(compact_balanced_sampling$enabled) &&
      identical(tolower(as.character(mcmc_control$universal_loss_weighting %||% "empirical")),
                "balanced_cell")) {
    warning(
      "balanced_sampling and universal_loss_weighting = 'balanced_cell' are ",
      "both enabled: the cell weights are inverse GLOBAL frequencies while ",
      "balanced sampling already reshapes the realized batch distribution, so ",
      "the two corrections stack on overlapping axes (studies aligned with ",
      "task/likelihood cells are effectively double-reweighted). Prefer one ",
      "mechanism unless this compounding is intended.",
      call. = FALSE
    )
  }
  compact_balanced_state <- NULL
  compact_balanced_pool_key <- NULL
  compact_balanced_sampling_state <- function() {
    if (!isTRUE(compact_balanced_sampling$enabled)) {
      return(NULL)
    }
    balanced_respondent_id_use <- respondent_id_use
    respondent_missing <- if (is.null(balanced_respondent_id_use)) {
      rep(TRUE, length(Y_use))
    } else {
      is.na(balanced_respondent_id_use) |
        !nzchar(as.character(balanced_respondent_id_use))
    }
    if (any(respondent_missing) && !is.null(respondent_task_id_use)) {
      if (is.null(balanced_respondent_id_use)) {
        balanced_respondent_id_use <- respondent_task_id_use
      } else {
        balanced_respondent_id_use[respondent_missing] <- respondent_task_id_use[respondent_missing]
      }
    }
    pool <- compact_sampling_pool()
    pool_key <- if (length(pool) < 1L) {
      "empty"
    } else {
      paste(
        length(pool),
        pool[[1L]],
        pool[[length(pool)]],
        format(sum(as.numeric(pool)), scientific = FALSE),
        sep = ":"
      )
    }
    if (is.null(compact_balanced_state) ||
        !identical(compact_balanced_pool_key, pool_key)) {
      compact_balanced_state <<- neural_build_balanced_sampling_state(
        obs_idx = pool,
        study_index = experiment_index_use,
        respondent_id = balanced_respondent_id_use,
        config = compact_balanced_sampling,
        context = "compact SVI"
      )
      compact_balanced_pool_key <<- pool_key
    }
    compact_balanced_state
  }
  compact_balanced_sampling_summary <- function() {
    if (!isTRUE(compact_balanced_sampling$enabled)) {
      return(NULL)
    }
    state <- compact_balanced_sampling_state()
    list(
      enabled = TRUE,
      scheme = state$scheme,
      within_respondent = state$within_respondent,
      replacement = TRUE,
      effective_likelihood_mass = state$effective_likelihood_mass,
      n_studies = as.integer(state$n_studies),
      n_respondent_units = as.integer(state$n_respondent_units),
      n_observations = as.integer(state$n_observations),
      respondent_counts_by_study = state$respondent_counts_by_study,
      observation_counts_by_study = state$observation_counts_by_study
    )
  }
  compact_sample_obs_idx <- function() {
    if (isTRUE(compact_balanced_sampling$enabled)) {
      return(neural_sample_balanced_obs_idx(
        compact_balanced_sampling_state(),
        compact_svi_batch_size
      ))
    }
    if (!isTRUE(universal_mixed_mode)) {
      pool <- compact_sampling_pool()
      if (length(pool) <= compact_svi_batch_size) {
        return(pool)
      }
      return(sample(pool, compact_svi_batch_size, replace = FALSE))
    }
    pool <- compact_sampling_pool()
    pair_pool <- pool[pool <= n_universal_pair_obs]
    single_pool <- pool[pool > n_universal_pair_obs] - n_universal_pair_obs
    if (length(pair_pool) < 1L || length(single_pool) < 1L) {
      stop("Universal mixed-mode compact SVI batches must include pairwise and single rows.", call. = FALSE)
    }
    pool_n <- length(pool)
    pair_target <- max(1L, min(length(pair_pool), as.integer(round(
      compact_svi_batch_size * length(pair_pool) / pool_n
    ))))
    single_target <- max(1L, min(length(single_pool), compact_svi_batch_size - pair_target))
    if (pair_target + single_target > compact_svi_batch_size) {
      if (pair_target > single_target) {
        pair_target <- pair_target - 1L
      } else {
        single_target <- single_target - 1L
      }
    }
    pair_idx <- if (length(pair_pool) <= pair_target) {
      pair_pool
    } else {
      sample(pair_pool, pair_target, replace = FALSE)
    }
    single_idx <- if (length(single_pool) <= single_target) {
      single_pool
    } else {
      sample(single_pool, single_target, replace = FALSE)
    }
    c(pair_idx, n_universal_pair_obs + single_idx)
  }
  compact_batch_args <- function(obs_idx) {
    obs_idx <- as.integer(obs_idx)
    obs_idx <- obs_idx[!is.na(obs_idx) & obs_idx >= 1L & obs_idx <= compact_model_n_obs]
    if (length(obs_idx) < 1L) {
      stop("Compact SVI batch has no valid observation rows.", call. = FALSE)
    }
    jnp_int_vector <- function(x) {
      strenv$jnp$atleast_1d(strenv$jnp$array(as.integer(x))$astype(strenv$jnp$int32))
    }
    jnp_num_vector <- function(x) {
      strenv$jnp$atleast_1d(strenv$jnp$array(as.numeric(x))$astype(ddtype_))
    }
    if (isTRUE(universal_mixed_mode)) {
      obs_idx_n <- length(obs_idx)
      pair_obs_idx_active <- obs_idx[obs_idx <= n_universal_pair_obs]
      single_obs_idx_active <- obs_idx[obs_idx > n_universal_pair_obs] - n_universal_pair_obs
      pair_active <- length(pair_obs_idx_active) > 0L
      single_active <- length(single_obs_idx_active) > 0L
      pad_mixed_branch_idx <- function(active_idx, target_n) {
        active_idx <- as.integer(active_idx)
        dummy_idx <- if (length(active_idx) > 0L) active_idx[[1L]] else 1L
        if (length(active_idx) >= target_n) {
          return(active_idx)
        }
        c(active_idx, rep.int(dummy_idx, target_n - length(active_idx)))
      }
      pair_target_n <- if (isTRUE(compact_balanced_sampling$enabled)) compact_svi_batch_size else max(1L, length(pair_obs_idx_active))
      single_target_n <- if (isTRUE(compact_balanced_sampling$enabled)) compact_svi_batch_size else max(1L, length(single_obs_idx_active))
      pair_obs_idx <- pad_mixed_branch_idx(pair_obs_idx_active, pair_target_n)
      single_obs_idx <- pad_mixed_branch_idx(single_obs_idx_active, single_target_n)
      pair_left_rows <- pair_mat[pair_obs_idx, 1L]
      pair_right_rows <- pair_mat[pair_obs_idx, 2L]
      single_rows <- universal_single_rows[single_obs_idx]
      materialize_cov <- function(rows) {
        if (n_resp_covariates < 1L) {
          return(list(
            values = strenv$jnp$zeros(list(ai(length(rows)), ai(0L)), dtype = ddtype_),
            present = strenv$jnp$zeros(list(ai(length(rows)), ai(0L)), dtype = ddtype_)
          ))
        }
        values <- cs2step_materialize_x_compact(X_compact_use, rows)
        if (is.null(values)) {
          values <- matrix(0, nrow = length(rows), ncol = n_resp_covariates)
        }
        present <- cs2step_materialize_x_present_compact(
          X_present_compact_use %||% X_compact_use,
          rows
        )
        if (is.null(present)) {
          present <- matrix(1, nrow = length(rows), ncol = n_resp_covariates)
        }
        list(
          values = strenv$jnp$array(as.matrix(values))$astype(ddtype_),
          present = strenv$jnp$array(as.matrix(present))$astype(ddtype_)
        )
      }
      cov_pair <- materialize_cov(pair_left_rows)
      cov_single <- materialize_cov(single_rows)
      pair_global_obs <- pair_obs_idx
      single_global_obs <- n_universal_pair_obs + single_obs_idx
      pool <- compact_sampling_pool()
      branch_scales <- neural_compact_mixed_branch_loss_scales(
        n_pair = sum(pool <= n_universal_pair_obs),
        n_single = sum(pool > n_universal_pair_obs),
        pair_batch_n = length(pair_obs_idx_active),
        single_batch_n = length(single_obs_idx_active),
        total_n = compact_sampling_n_obs(),
        batch_n = obs_idx_n,
        balanced_sampling = isTRUE(compact_balanced_sampling$enabled)
      )
      observation_weights <- function(global_obs_idx) {
        if (is.null(universal_loss_weights)) {
          return(rep.int(1, length(global_obs_idx)))
        }
        as.numeric(universal_loss_weights[global_obs_idx])
      }
      pair_obs_scale <- rep(0, length(pair_global_obs))
      if (isTRUE(pair_active)) {
        pair_global_obs_active <- pair_obs_idx_active
        pair_obs_scale[seq_along(pair_global_obs_active)] <-
          branch_scales$pair * observation_weights(pair_global_obs_active)
      }
      single_obs_scale <- rep(0, length(single_global_obs))
      if (isTRUE(single_active)) {
        single_global_obs_active <- n_universal_pair_obs + single_obs_idx_active
        single_obs_scale[seq_along(single_global_obs_active)] <-
          branch_scales$single * observation_weights(single_global_obs_active)
      }
      return(list(
        X_left = strenv$jnp$array(to_index_matrix(
          cs2step_materialize_w_idx_compact(W_idx_compact_use, pair_left_rows)
        ))$astype(strenv$jnp$int32),
        X_right = strenv$jnp$array(to_index_matrix(
          cs2step_materialize_w_idx_compact(W_idx_compact_use, pair_right_rows)
        ))$astype(strenv$jnp$int32),
        party_left = jnp_int_vector(party_left[pair_obs_idx]),
        party_right = jnp_int_vector(party_right[pair_obs_idx]),
        resp_party = jnp_int_vector(resp_party_pair_use[pair_obs_idx]),
        resp_cov = cov_pair$values,
        resp_cov_present = cov_pair$present,
        experiment_index = if (!is.null(experiment_index_pair_use)) {
          jnp_int_vector(experiment_index_pair_use[pair_obs_idx])
        } else {
          NULL
        },
        likelihood_code = jnp_int_vector(universal_likelihood_code_use[pair_global_obs]),
        n_outcomes_obs = jnp_int_vector(universal_n_outcomes_use_int[pair_global_obs]),
        Y_obs = jnp_num_vector(Y_pair_use[pair_obs_idx]),
        obs_scale = jnp_num_vector(pair_obs_scale),
        X_single = strenv$jnp$array(to_index_matrix(
          cs2step_materialize_w_idx_compact(W_idx_compact_use, single_rows)
        ))$astype(strenv$jnp$int32),
        party_single = jnp_int_vector(party_single[single_obs_idx]),
        resp_party_single = jnp_int_vector(resp_party_single_use[single_obs_idx]),
        resp_cov_single = cov_single$values,
        resp_cov_present_single = cov_single$present,
        experiment_index_single = if (!is.null(experiment_index_single_use)) {
          jnp_int_vector(experiment_index_single_use[single_obs_idx])
        } else {
          NULL
        },
        likelihood_code_single = jnp_int_vector(universal_likelihood_code_use[single_global_obs]),
        n_outcomes_single = jnp_int_vector(universal_n_outcomes_use_int[single_global_obs]),
        Y_single_obs = jnp_num_vector(Y_single_use[single_obs_idx]),
        obs_scale_single = jnp_num_vector(single_obs_scale)
      ))
    }
    y_obs <- if (likelihood == "categorical") {
      strenv$jnp$array(as.integer(y_fac[obs_idx]))$astype(strenv$jnp$int32)
    } else {
      strenv$jnp$array(as.numeric(Y_use[obs_idx]))$astype(ddtype_)
    }
    resp_party_obs <- strenv$jnp$array(as.integer(resp_party_use[obs_idx]))$astype(strenv$jnp$int32)
    cov_rows <- if (isTRUE(pairwise_mode)) {
      pair_mat[obs_idx, 1L]
    } else {
      obs_idx
    }
    if (n_resp_covariates > 0L) {
      resp_cov_mat <- cs2step_materialize_x_compact(X_compact_use, cov_rows)
      if (is.null(resp_cov_mat)) {
        resp_cov_mat <- matrix(0, nrow = length(obs_idx), ncol = n_resp_covariates)
      }
      resp_cov_present_mat <- cs2step_materialize_x_present_compact(
        X_present_compact_use %||% X_compact_use,
        cov_rows
      )
      if (is.null(resp_cov_present_mat)) {
        resp_cov_present_mat <- matrix(1, nrow = length(obs_idx), ncol = n_resp_covariates)
      }
      resp_cov <- strenv$jnp$array(as.matrix(resp_cov_mat))$astype(ddtype_)
      resp_cov_present <- strenv$jnp$array(as.matrix(resp_cov_present_mat))$astype(ddtype_)
    } else {
      resp_cov <- strenv$jnp$zeros(list(ai(length(obs_idx)), ai(0L)), dtype = ddtype_)
      resp_cov_present <- strenv$jnp$zeros(list(ai(length(obs_idx)), ai(0L)), dtype = ddtype_)
    }
    experiment_idx <- if (!is.null(experiment_index_use)) {
      strenv$jnp$array(as.integer(experiment_index_use[obs_idx]))$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    likelihood_code_obs <- if (isTRUE(universal_enabled) && !is.null(universal_likelihood_use)) {
      strenv$jnp$array(as.integer(universal_likelihood_code_use[obs_idx]))$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    n_outcomes_obs <- if (isTRUE(universal_enabled) && !is.null(universal_n_outcomes_use_int)) {
      strenv$jnp$array(as.integer(universal_n_outcomes_use_int[obs_idx]))$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    obs_scale <- if (!is.null(universal_loss_weights)) {
      strenv$jnp$array(
        as.numeric(compact_sampling_n_obs()) / as.numeric(length(obs_idx)) *
          as.numeric(universal_loss_weights[obs_idx])
      )$astype(ddtype_)
    } else {
      as.numeric(compact_sampling_n_obs()) / as.numeric(length(obs_idx))
    }
    if (isTRUE(pairwise_mode)) {
      left_rows <- pair_mat[obs_idx, 1L]
      right_rows <- pair_mat[obs_idx, 2L]
      list(
        X_left = strenv$jnp$array(to_index_matrix(
          cs2step_materialize_w_idx_compact(W_idx_compact_use, left_rows)
        ))$astype(strenv$jnp$int32),
        X_right = strenv$jnp$array(to_index_matrix(
          cs2step_materialize_w_idx_compact(W_idx_compact_use, right_rows)
        ))$astype(strenv$jnp$int32),
        party_left = strenv$jnp$array(as.integer(party_left[obs_idx]))$astype(strenv$jnp$int32),
        party_right = strenv$jnp$array(as.integer(party_right[obs_idx]))$astype(strenv$jnp$int32),
        resp_party = resp_party_obs,
        resp_cov = resp_cov,
        resp_cov_present = resp_cov_present,
        experiment_index = experiment_idx,
        likelihood_code = likelihood_code_obs,
        n_outcomes_obs = n_outcomes_obs,
        Y_obs = y_obs,
        obs_scale = obs_scale
      )
    } else {
      list(
        X = strenv$jnp$array(to_index_matrix(
          cs2step_materialize_w_idx_compact(W_idx_compact_use, obs_idx)
        ))$astype(strenv$jnp$int32),
        party = strenv$jnp$array(as.integer(party_single[obs_idx]))$astype(strenv$jnp$int32),
        resp_party = resp_party_obs,
        resp_cov = resp_cov,
        resp_cov_present = resp_cov_present,
        experiment_index = experiment_idx,
        likelihood_code = likelihood_code_obs,
        n_outcomes_obs = n_outcomes_obs,
        Y_obs = y_obs,
        obs_scale = obs_scale
      )
    }
  }

  model_fn_base <- if (pairwise_mode) BayesianPairTransformerModel else BayesianSingleTransformerModel
  model_fn <- model_fn_base

  build_resp_cov_order_new <- function(resp_cov_new, n_rows, experiment_idx_new = NULL) {
    if (!identical(covariate_value_encoding, "shared_projection") ||
        length(covariate_names_override) < 1L) {
      return(NULL)
    }
    if (!is.null(experiment_idx_new) && length(covariate_order_by_experiment) > 0L) {
      use_experiment_lookup <- is.null(resp_cov_new)
      if (!use_experiment_lookup && (is.data.frame(resp_cov_new) || is.matrix(resp_cov_new))) {
        cols <- colnames(resp_cov_new)
        use_experiment_lookup <- is.null(cols) || identical(as.character(cols), covariate_names_override)
      }
      if (isTRUE(use_experiment_lookup)) {
        lookup <- neural_covariate_order_lookup_matrix(
          order_list = covariate_order_by_experiment,
          max_covariate_tokens = max_covariate_tokens
        )
        exp_idx <- as.integer(experiment_idx_new)
        if (length(exp_idx) == 1L && n_rows > 1L) {
          exp_idx <- rep.int(exp_idx, n_rows)
        }
        if (!is.null(lookup) && length(exp_idx) == n_rows && all(!is.na(exp_idx))) {
          return(lookup[exp_idx + 1L, , drop = FALSE])
        }
      }
    }
    order_idx <- default_covariate_order
    if (!is.null(resp_cov_new)) {
      if (is.data.frame(resp_cov_new) || is.matrix(resp_cov_new)) {
        if (!is.null(colnames(resp_cov_new))) {
          order_idx <- neural_covariate_order_from_names(
            colnames(resp_cov_new),
            covariate_names_override
          )
        }
      } else if (length(covariate_names_override) == 1L) {
        order_idx <- 0L
      }
    }
    neural_build_default_covariate_order_matrix(
      order_idx = order_idx,
      n_rows = n_rows,
      max_covariate_tokens = max_covariate_tokens
    )
  }

  build_factor_order_new <- function(W_new, n_rows, experiment_idx_new = NULL) {
    if (!identical(factor_tokenization, "fused")) {
      return(NULL)
    }
    factor_names_use <- names(names_list)
    W_cols <- if (!is.null(W_new) && (is.data.frame(W_new) || is.matrix(W_new))) {
      colnames(W_new)
    } else {
      NULL
    }
    if (!is.null(experiment_idx_new) && length(factor_order_by_experiment) > 0L) {
      use_experiment_lookup <- is.null(W_new)
      if (!use_experiment_lookup) {
        use_experiment_lookup <- is.null(W_cols) ||
          identical(as.character(W_cols), factor_names_use)
      }
      if (isTRUE(use_experiment_lookup)) {
        lookup <- neural_factor_order_lookup_matrix(
          order_list = factor_order_by_experiment,
          max_factor_tokens = max_factor_tokens
        )
        exp_idx <- as.integer(experiment_idx_new)
        if (length(exp_idx) == 1L && n_rows > 1L) {
          exp_idx <- rep.int(exp_idx, n_rows)
        }
        if (!is.null(lookup) && length(exp_idx) == n_rows && all(!is.na(exp_idx))) {
          return(lookup[exp_idx + 1L, , drop = FALSE])
        }
      }
    }
    if (!is.null(W_new)) {
      if (!is.null(W_cols)) {
        order_idx <- neural_factor_order_from_names(
          W_cols,
          factor_names_use,
          error_on_partial = TRUE
        )
        if (length(order_idx) > 0L) {
          return(neural_build_default_factor_order_matrix(
            order_idx = order_idx,
            n_rows = n_rows,
            max_factor_tokens = max_factor_tokens
          ))
        }
      } else if (!is.data.frame(W_new) && !is.matrix(W_new)) {
        W_df <- as.data.frame(W_new, check.names = FALSE)
        order_idx <- neural_factor_order_from_names(
          colnames(W_df),
          factor_names_use,
          error_on_partial = TRUE
        )
        if (length(order_idx) > 0L) {
          return(neural_build_default_factor_order_matrix(
            order_idx = order_idx,
            n_rows = n_rows,
            max_factor_tokens = max_factor_tokens
          ))
        }
      }
    }
    neural_build_default_factor_order_matrix(
      order_idx = default_factor_order,
      n_rows = n_rows,
      max_factor_tokens = max_factor_tokens
    )
  }

  build_params_from_sites_for_svi_validation <- function(param_sites,
                                                         fallback_params = NULL) {
    if (is.null(param_sites) && is.null(fallback_params)) {
      return(NULL)
    }

    prefer_fallback_param_store <- function(name) {
      if (!isTRUE(output_only_mode)) {
        return(FALSE)
      }
      dynamic_names <- c(
        "W_out", "W_out_decentered", "W_out_base", "W_out_z",
        "W_add_out", "W_add_out_decentered", "W_add_out_base", "W_add_out_z",
        "b_out", "b_out_decentered", "b_out_base", "b_out_z",
        "tau_w_out", "tau_b",
        "sigma",
        "W_cross_out",
        "M_cross", "M_cross_raw", "tau_cross",
        "alpha_rc", "W_rc_out",
        "log_pairwise_bernoulli_logit_scale",
        "log_calibration_scale"
      )
      !name %in% dynamic_names
    }

    get_site_value <- function(name) {
      value <- NULL
      if (isTRUE(prefer_fallback_param_store(name)) && !is.null(fallback_params)) {
        value <- tryCatch(fallback_params[[name]], error = function(e) NULL)
      }
      if (is.null(value) && !is.null(param_sites)) {
        value <- tryCatch(param_sites[[name]], error = function(e) NULL)
      }
      if (is.null(value) && !is.null(fallback_params)) {
        value <- tryCatch(fallback_params[[name]], error = function(e) NULL)
      }
      value
    }

    get_loc_scale_site_value <- function(name, scale_name) {
      direct_value <- get_site_value(name)
      if (!is.null(direct_value)) {
        return(direct_value)
      }

      base_names <- c(paste0(name, "_decentered"),
                      paste0(name, "_base"),
                      paste0(name, "_z"))
      base_value <- NULL
      for (base_name in base_names) {
        base_value <- get_site_value(base_name)
        if (!is.null(base_value)) {
          break
        }
      }
      scale_value <- get_site_value(scale_name)
      if (is.null(base_value) || is.null(scale_value)) {
        return(NULL)
      }

      scale_shape <- tryCatch(
        as.integer(reticulate::py_to_r(scale_value$shape)),
        error = function(e) NULL
      )
      base_shape <- tryCatch(
        as.integer(reticulate::py_to_r(base_value$shape)),
        error = function(e) NULL
      )
      if (!is.null(scale_shape) && !is.null(base_shape)) {
        extra_dims <- length(base_shape) - length(scale_shape)
        if (extra_dims > 0L) {
          reshape_dims <- c(scale_shape, rep(1L, extra_dims))
          scale_value <- strenv$jnp$reshape(scale_value, as.list(reshape_dims))
        }
      }
      scale_value * base_value
    }

    get_cross_site_value <- function() {
      value <- get_site_value("M_cross")
      if (!is.null(value)) {
        return(value)
      }
      raw_value <- get_site_value("M_cross_raw")
      if (is.null(raw_value)) {
        raw_value <- get_loc_scale_site_value("M_cross_raw", "tau_cross")
      }
      if (is.null(raw_value)) {
        return(NULL)
      }
      0.5 * (raw_value - strenv$jnp$transpose(raw_value))
    }

    params_out <- list()
    maybe_site <- function(name, assign_as = name, value = NULL) {
      if (is.null(value)) {
        value <- get_site_value(name)
      }
      if (!is.null(value)) {
        params_out[[assign_as]] <<- value
      }
      invisible(value)
    }

    params_out$E_choice <- get_site_value("E_choice")
    if (is.null(params_out$E_choice)) {
      return(NULL)
    }

    maybe_site("E_respondent_cls")
    maybe_site("E_candidate_cls")
    maybe_site("E_party")
    maybe_site("E_resp_party")
    maybe_site("E_sep")
    maybe_site("E_rel")
    maybe_site("E_token_family")
    maybe_site("E_experiment")
    maybe_site("E_stage")
    maybe_site("E_matchup")
    maybe_site("E_factor_fused_base")
    maybe_site("W_factor_fuse_1")
    maybe_site("b_factor_fuse_1")
    maybe_site("W_factor_fuse_2")
    maybe_site("b_factor_fuse_2")
    maybe_site("W_factor_name_text")
    maybe_site("W_level_name_text")
    maybe_site("W_factor_struct")
    maybe_site("W_level_struct")
    maybe_site("W_covariate_name_text")
    maybe_site("W_experiment_text")
    maybe_site("W_place_context")
    maybe_site("W_time_context")
    maybe_site("alpha_cross")
    maybe_site("RMS_cross")
    maybe_site("RMS_merge_cross")
    maybe_site("RMS_q_cross")
    maybe_site("RMS_k_cross")

    segment_value <- get_site_value("E_segment")
    if (is.null(segment_value)) {
      segment_delta <- get_site_value("E_segment_delta")
      if (!is.null(segment_delta)) {
        segment_value <- neural_build_symmetric_segment_embeddings(segment_delta)
      }
    }
    maybe_site("E_segment", value = segment_value)

    params_out$W_out <- get_loc_scale_site_value("W_out", "tau_w_out")
    W_add_out_value <- get_loc_scale_site_value("W_add_out", "tau_w_out")
    maybe_site("W_add_out", value = W_add_out_value)
    params_out$b_out <- get_loc_scale_site_value("b_out", "tau_b")
    if (is.null(params_out$W_out) || is.null(params_out$b_out)) {
      return(NULL)
    }
    ordinal_raw_value <- get_site_value("ordinal_threshold_raw")
    if (!is.null(ordinal_raw_value)) {
      params_out$ordinal_threshold_raw <- ordinal_raw_value
      params_out$ordinal_thresholds <- ordinal_thresholds_from_raw_jnp(ordinal_raw_value)
    } else {
      maybe_site("ordinal_thresholds")
    }

    cross_value <- get_cross_site_value()
    if (!is.null(cross_value)) {
      params_out$M_cross <- cross_value
    }
    maybe_site("W_cross_out")
    maybe_site("alpha_rc")
    W_rc_r_value <- get_loc_scale_site_value("W_rc_r", "tau_rc")
    maybe_site("W_rc_r", value = W_rc_r_value)
    W_rc_c_value <- get_loc_scale_site_value("W_rc_c", "tau_rc")
    maybe_site("W_rc_c", value = W_rc_c_value)
    maybe_site("W_rc_out")
    maybe_site("log_pairwise_bernoulli_logit_scale")
    if (!is.null(params_out$log_pairwise_bernoulli_logit_scale)) {
      params_out$pairwise_bernoulli_logit_scale <- strenv$jnp$exp(
        params_out$log_pairwise_bernoulli_logit_scale
      )
    }
    maybe_site("log_calibration_scale")
    if (!is.null(params_out$log_calibration_scale)) {
      params_out$calibration_scale <- strenv$jnp$exp(params_out$log_calibration_scale)
    }

    if (likelihood == "normal" || isTRUE(universal_has_normal)) {
      maybe_site("sigma")
    }
    if (n_resp_covariates > 0L) {
      maybe_site("E_covariate_fused_base")
      maybe_site("E_covariate_missing")
      maybe_site("W_covariate_fuse_1")
      maybe_site("b_covariate_fuse_1")
      maybe_site("W_covariate_fuse_2")
      maybe_site("b_covariate_fuse_2")
      maybe_site("W_covariate_value_text")
      maybe_site("W_covariate_value_shared")
      maybe_site("W_covariate_value_basis")
      maybe_site("W_covariate_value_conditioner_1")
      maybe_site("b_covariate_value_conditioner_1")
      maybe_site("W_covariate_value_conditioner_2")
      maybe_site("b_covariate_value_conditioner_2")
    }

    for (l_ in 1L:ModelDepth) {
      maybe_site(paste0("pseudo_query_attn_l", l_))
      maybe_site(paste0("pseudo_query_ff_l", l_))
      maybe_site(paste0("alpha_attn_l", l_))
      maybe_site(paste0("alpha_ff_l", l_))
      maybe_site(paste0("RMS_attn_l", l_))
      maybe_site(paste0("RMS_q_l", l_))
      maybe_site(paste0("RMS_k_l", l_))
      maybe_site(paste0("RMS_ff_l", l_))

      tau_name <- paste0("tau_w_", l_)
      for (base in c("W_q_l", "W_k_l", "W_v_l", "W_o_l", "W_ff1_l", "W_ff2_l")) {
        params_out[[paste0(base, l_)]] <- get_loc_scale_site_value(paste0(base, l_), tau_name)
      }
    }
    if (!isTRUE(use_full_attn_residual)) {
      params_out <- neural_stack_standard_transformer_layers(
        params_out,
        model_depth = ModelDepth,
        drop_legacy = TRUE
      )
    }

    for (name in c("W_q_cross", "W_k_cross", "W_v_cross", "W_o_cross")) {
      params_out[[name]] <- get_loc_scale_site_value(name, "tau_cross_attn")
    }
    maybe_site("pseudo_query_final")
    maybe_site("RMS_final")

    params_out
  }

  validation_model_info <- neural_make_prepared_prediction_model_info(
    model_depth = ModelDepth,
    model_dims = ModelDims,
    n_heads = TransformerHeads,
    head_dim = head_dim,
    residual_mode = residual_mode,
    attention_backend = attention_backend,
    attention_dtype = attention_dtype,
    attention_padding_multiple = attention_padding_multiple,
    attention_resolved_backend = attention_resolved_backend,
    attention_fallback_reason = attention_fallback_reason,
    cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
    n_party_levels = ai(n_party_levels),
    n_candidate_tokens = n_candidate_tokens,
    cross_candidate_encoder_mode = cross_candidate_encoder_mode,
    cross_candidate_encoder = !identical(cross_candidate_encoder_mode, "none"),
    likelihood = likelihood,
    shared_projection_value_encoder = shared_projection_value_encoder,
    covariate_value_stats_by_experiment = covariate_value_stats_by_experiment,
    default_covariate_value_stats = default_covariate_value_stats,
    covariate_value_metadata_by_experiment = covariate_value_metadata_by_experiment,
    default_covariate_value_metadata = default_covariate_value_metadata,
    covariate_value_text = covariate_value_text,
    covariate_value_text_present = covariate_value_text_present,
    covariate_value_type = covariate_value_type,
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    factor_order_by_experiment = factor_order_by_experiment,
    default_factor_order = default_factor_order,
    factor_struct_matrix = factor_struct_matrix,
    level_struct_matrices = level_struct_matrices,
    factor_struct_feature_names = factor_struct_feature_names,
    level_struct_feature_names = level_struct_feature_names,
    factor_tokenization = factor_tokenization,
    max_factor_tokens = max_factor_tokens,
    low_rank_interaction_rank = low_rank_interaction_rank,
    low_rank_logit_transform = low_rank_logit_transform,
    low_rank_logit_bound = low_rank_logit_bound,
    low_rank_logit_softness = low_rank_logit_softness,
    low_rank_logit_normalization = low_rank_logit_normalization,
    low_rank_head_weight_target_rms = low_rank_head_weight_target_rms,
    low_rank_rc_out_target_rms = low_rank_rc_out_target_rms,
    pairwise_antisymmetry = pairwise_antisymmetry_mode,
    additive_utility_mode = additive_utility_mode,
    additive_utility_normalization = additive_utility_normalization,
    additive_head_weight_target_rms = additive_head_weight_target_rms,
    calibration_enabled = isTRUE(calibration_control$enabled),
    calibration_method = calibration_control$method %||% "none",
    calibration_scale = NULL
  )
  validation_model_info$learned_pairwise_bernoulli_logit_scale <-
    learned_pairwise_bernoulli_logit_scale
  validation_model_info$pairwise_bernoulli_logit_scale_prior_sd <-
    pairwise_bernoulli_logit_scale_prior_sd
  validation_model_info <- neural_set_pairwise_context_model_info(
    info = validation_model_info,
    pairwise_context_mode = pairwise_context_mode,
    has_candidate_group_context = has_candidate_group_context,
    has_respondent_group_context = has_respondent_group_context,
    has_relation_token_context = has_relation_context,
    has_stage_context = stage_context_enabled,
    has_matchup_context = use_matchup_token,
    party_missing_label = party_missing_label,
    resp_party_missing_label = resp_party_missing_label,
    n_resp_party_levels = n_resp_party_levels,
    party_missing_index = party_missing_index,
    resp_party_missing_index = resp_party_missing_index,
    context_present_masking = context_present_masking
  )
  validation_model_info$init_policy <- init_policy
  validation_model_info$factor_name_text <- factor_name_text
  validation_model_info$level_name_text <- level_name_text
  validation_model_info$factor_struct_matrix <- factor_struct_matrix
  validation_model_info$level_struct_matrices <- level_struct_matrices
  validation_model_info$factor_struct_feature_names <- factor_struct_feature_names
  validation_model_info$level_struct_feature_names <- level_struct_feature_names
  validation_model_info$covariate_name_text <- covariate_name_text
  validation_model_info$covariate_names <- covariate_names_override
  validation_model_info$resp_cov_mean <- resp_cov_mean
  validation_model_info$resp_cov_scale <- resp_cov_scale
  validation_model_info$resp_cov_default_present <- resp_cov_default_present
  validation_model_info$n_resp_covariates <- n_resp_covariates
  validation_model_info$covariate_order_by_experiment <- covariate_order_by_experiment
  validation_model_info$default_covariate_order <- default_covariate_order
  validation_model_info$max_covariate_tokens <- max_covariate_tokens
  validation_model_info$default_experiment_index <- if (is.na(default_experiment_index)) NULL else as.integer(default_experiment_index)
  validation_model_info$token_family_levels <- token_family_levels
  validation_model_info$experiment_token_mode <- experiment_token_mode
  validation_model_info$covariate_value_encoding <- covariate_value_encoding
  validation_model_info$experiment_description_text <- experiment_description_text
  validation_model_info$experiment_description_present <- experiment_description_present
  validation_model_info$default_experiment_text <- default_experiment_text
  validation_model_info$default_experiment_text_present <- default_experiment_text_present
  validation_model_info$place_embedding <- place_embedding
  validation_model_info$place_present <- place_present
  validation_model_info$place_context_enabled <- place_context_enabled
  validation_model_info$place_feature_names <- place_feature_names
  validation_model_info$default_place_embedding <- default_place_embedding
  validation_model_info$default_place_present <- default_place_present
  validation_model_info$place_context_dim <- place_context_dim
  validation_model_info$time_embedding <- time_embedding
  validation_model_info$time_present <- time_present
  validation_model_info$time_context_enabled <- time_context_enabled
  validation_model_info$time_feature_names <- time_feature_names
  validation_model_info$default_time_embedding <- default_time_embedding
  validation_model_info$default_time_present <- default_time_present
  validation_model_info$time_context_dim <- time_context_dim
  validation_return_logits <- identical(likelihood, "mixed")
  validation_predict_pair_jit <- if (isTRUE(pairwise_mode)) {
    neural_get_predict_jit(
      model_info = validation_model_info,
      pairwise = TRUE,
      return_logits = validation_return_logits
    )
  } else {
    NULL
  }
  validation_predict_single_jit <- if (!isTRUE(pairwise_mode) || isTRUE(universal_mixed_mode)) {
    neural_get_predict_jit(
      model_info = validation_model_info,
      pairwise = FALSE,
      return_logits = validation_return_logits
    )
  } else {
    NULL
  }

  svi_validation_predict_chunk <- function(param_sites, idx, fallback_params = NULL) {
    params <- build_params_from_sites_for_svi_validation(
      param_sites,
      fallback_params = fallback_params
    )
    if (is.null(params) || length(idx) < 1L) {
      return(NULL)
    }

    to_r_array_local <- function(x) {
      if (is.null(x) || is.numeric(x)) {
        return(x)
      }
      strategize_jax_block_until_ready(x)
      tryCatch(reticulate::py_to_r(strenv$np$array(x)),
               error = function(e) {
                 tryCatch(reticulate::py_to_r(x), error = function(e2) x)
               })
    }

    coerce_prediction_output_local <- function(pred) {
      if (identical(likelihood, "mixed")) {
        logits <- as.matrix(to_r_array_local(pred))
        sigma_vec <- mixed_sigma_vector_r(
          to_r_array_local(params$sigma %||% NULL),
          nrow(logits)
        )
        return(list(
          logits = logits,
          sigma = sigma_vec
        ))
      }
      if (likelihood == "bernoulli") {
        return(as.numeric(to_r_array_local(pred)))
      }
      if (likelihood == "categorical") {
        return(as.matrix(to_r_array_local(pred)))
      }
      if (likelihood == "normal") {
        return(list(
          mu = as.numeric(to_r_array_local(pred$mu)),
          sigma = as.numeric(to_r_array_local(pred$sigma))
        ))
      }
      pred
    }

    materialize_validation_resp_cov <- function(obs_idx, compact_rows) {
      n_rows <- length(obs_idx)
      if (n_resp_covariates > 0L) {
        if (isTRUE(compact_training)) {
          resp_cov_mat <- if (!is.null(X_compact_use)) {
            cs2step_materialize_x_compact(X_compact_use, compact_rows)
          } else {
            NULL
          }
          if (is.null(resp_cov_mat)) {
            resp_cov_mat <- matrix(0, nrow = n_rows, ncol = n_resp_covariates)
          }
          resp_present_mat <- if (!is.null(X_present_compact_use) || !is.null(X_compact_use)) {
            cs2step_materialize_x_present_compact(
              X_present_compact_use %||% X_compact_use,
              compact_rows
            )
          } else {
            NULL
          }
          if (is.null(resp_present_mat)) {
            resp_present_mat <- matrix(1, nrow = n_rows, ncol = n_resp_covariates)
          }
        } else {
          resp_cov_mat <- if (!is.null(X_use)) {
            X_use[obs_idx, , drop = FALSE]
          } else {
            matrix(0, nrow = n_rows, ncol = n_resp_covariates)
          }
          resp_present_mat <- if (!is.null(X_present_use)) {
            X_present_use[obs_idx, , drop = FALSE]
          } else {
            matrix(1, nrow = n_rows, ncol = n_resp_covariates)
          }
        }
        return(list(
          values = strenv$jnp$array(as.matrix(resp_cov_mat))$astype(ddtype_),
          present = strenv$jnp$array(as.matrix(resp_present_mat))$astype(ddtype_),
          values_r = as.matrix(resp_cov_mat)
        ))
      }
      list(
        values = strenv$jnp$zeros(list(ai(n_rows), ai(0L)), dtype = ddtype_),
        present = strenv$jnp$zeros(list(ai(n_rows), ai(0L)), dtype = ddtype_),
        values_r = NULL
      )
    }

    predict_validation_pair <- function(pair_idx) {
      if (length(pair_idx) < 1L || is.null(validation_predict_pair_jit)) {
        return(NULL)
      }
      left_rows <- pair_mat[pair_idx, 1L]
      right_rows <- pair_mat[pair_idx, 2L]
      Xl_r <- if (isTRUE(compact_training)) {
        cs2step_materialize_w_idx_compact(W_idx_compact_use, left_rows)
      } else {
        X_left[pair_idx, , drop = FALSE]
      }
      Xr_r <- if (isTRUE(compact_training)) {
        cs2step_materialize_w_idx_compact(W_idx_compact_use, right_rows)
      } else {
        X_right[pair_idx, , drop = FALSE]
      }
      if (is.null(Xl_r) || is.null(Xr_r)) {
        return(NULL)
      }
      resp_cov <- materialize_validation_resp_cov(pair_idx, left_rows)
      experiment_values <- if (!is.null(experiment_index_use)) {
        experiment_index_use[pair_idx]
      } else {
        NULL
      }
      resp_c_order <- if (is.null(resp_cov_mean) || n_resp_covariates < 1L) {
        NULL
      } else {
        resp_order_r <- build_resp_cov_order_new(
          resp_cov$values_r,
          length(pair_idx),
          experiment_idx_new = experiment_values
        )
        if (is.null(resp_order_r)) {
          NULL
        } else {
          strenv$jnp$array(as.matrix(resp_order_r))$astype(strenv$jnp$int32)
        }
      }
      factor_order <- if (identical(factor_tokenization, "fused")) {
        strenv$jnp$array(as.matrix(build_factor_order_new(
          Xl_r,
          length(pair_idx),
          experiment_idx_new = experiment_values
        )))$astype(strenv$jnp$int32)
      } else {
        NULL
      }
      pred <- validation_predict_pair_jit(
        params,
        strenv$jnp$array(to_index_matrix(Xl_r))$astype(strenv$jnp$int32),
        strenv$jnp$array(to_index_matrix(Xr_r))$astype(strenv$jnp$int32),
        strenv$jnp$array(as.integer(party_left[pair_idx]))$astype(strenv$jnp$int32),
        strenv$jnp$array(as.integer(party_right[pair_idx]))$astype(strenv$jnp$int32),
        strenv$jnp$array(as.integer(resp_party_use[pair_idx]))$astype(strenv$jnp$int32),
        resp_cov$values,
        resp_cov$present,
        resp_c_order,
        if (!is.null(experiment_values)) {
          strenv$jnp$array(as.integer(experiment_values))$astype(strenv$jnp$int32)
        } else {
          NULL
        },
        NULL,
        NULL,
        factor_order
      )
      strategize_jax_block_until_ready(pred)
      out <- coerce_prediction_output_local(pred)
      if (identical(likelihood, "mixed") &&
          is.list(out) &&
          !is.null(out$logits) &&
          ncol(as.matrix(out$logits)) >= 1L) {
        scale_now <- neural_pairwise_bernoulli_logit_scale_from_params(
          params = params,
          model_info = validation_model_info
        )
        out$logits[, 1L] <- neural_apply_pairwise_bernoulli_logit_scale_r(
          out$logits[, 1L],
          validation_model_info,
          scale = to_r_array_local(scale_now)
        )
      }
      out
    }

    predict_validation_single <- function(obs_idx) {
      if (length(obs_idx) < 1L || is.null(validation_predict_single_jit)) {
        return(NULL)
      }
      single_pos <- if (isTRUE(universal_mixed_mode)) {
        obs_idx - n_universal_pair_obs
      } else {
        obs_idx
      }
      compact_rows <- if (isTRUE(universal_mixed_mode)) {
        universal_single_rows[single_pos]
      } else {
        obs_idx
      }
      Xb_r <- if (isTRUE(compact_training)) {
        cs2step_materialize_w_idx_compact(W_idx_compact_use, compact_rows)
      } else {
        X_single[single_pos, , drop = FALSE]
      }
      if (is.null(Xb_r)) {
        return(NULL)
      }
      resp_cov <- materialize_validation_resp_cov(obs_idx, compact_rows)
      experiment_values <- if (!is.null(experiment_index_use)) {
        experiment_index_use[obs_idx]
      } else {
        NULL
      }
      resp_c_order <- if (is.null(resp_cov_mean) || n_resp_covariates < 1L) {
        NULL
      } else {
        resp_order_r <- build_resp_cov_order_new(
          resp_cov$values_r,
          length(obs_idx),
          experiment_idx_new = experiment_values
        )
        if (is.null(resp_order_r)) {
          NULL
        } else {
          strenv$jnp$array(as.matrix(resp_order_r))$astype(strenv$jnp$int32)
        }
      }
      factor_order <- if (identical(factor_tokenization, "fused")) {
        strenv$jnp$array(as.matrix(build_factor_order_new(
          Xb_r,
          length(obs_idx),
          experiment_idx_new = experiment_values
        )))$astype(strenv$jnp$int32)
      } else {
        NULL
      }
      pred <- validation_predict_single_jit(
        params,
        strenv$jnp$array(to_index_matrix(Xb_r))$astype(strenv$jnp$int32),
        strenv$jnp$array(as.integer(party_single[single_pos]))$astype(strenv$jnp$int32),
        strenv$jnp$array(as.integer(resp_party_use[obs_idx]))$astype(strenv$jnp$int32),
        resp_cov$values,
        resp_cov$present,
        resp_c_order,
        if (!is.null(experiment_values)) {
          strenv$jnp$array(as.integer(experiment_values))$astype(strenv$jnp$int32)
        } else {
          NULL
        },
        NULL,
        NULL,
        factor_order
      )
      strategize_jax_block_until_ready(pred)
      coerce_prediction_output_local(pred)
    }

    idx <- as.integer(idx)
    if (isTRUE(universal_mixed_mode)) {
      pair_pos <- which(idx <= n_universal_pair_obs)
      single_pos <- which(idx > n_universal_pair_obs)
      pred_pair <- if (length(pair_pos) > 0L) {
        predict_validation_pair(idx[pair_pos])
      } else {
        NULL
      }
      pred_single <- if (length(single_pos) > 0L) {
        predict_validation_single(idx[single_pos])
      } else {
        NULL
      }
      if (identical(likelihood, "mixed")) {
        pred_nonnull <- Filter(Negate(is.null), list(pred_pair, pred_single))
        if (length(pred_nonnull) < 1L) {
          return(NULL)
        }
        n_logits <- ncol(as.matrix(pred_nonnull[[1L]]$logits))
        logits <- matrix(NA_real_, nrow = length(idx), ncol = n_logits)
        sigma <- rep(NA_real_, length(idx))
        if (!is.null(pred_pair)) {
          logits[pair_pos, ] <- as.matrix(pred_pair$logits)
          sigma[pair_pos] <- as.numeric(pred_pair$sigma)
        }
        if (!is.null(pred_single)) {
          logits[single_pos, ] <- as.matrix(pred_single$logits)
          sigma[single_pos] <- as.numeric(pred_single$sigma)
        }
        return(list(logits = logits, sigma = sigma))
      }
      return(combine_svi_validation_predictions(list(pred_pair, pred_single)))
    }

    if (isTRUE(pairwise_mode)) {
      return(predict_validation_pair(idx))
    }
    predict_validation_single(idx)
  }
  combine_svi_validation_predictions <- function(pred_chunks) {
    pred_chunks <- Filter(Negate(is.null), pred_chunks)
    if (length(pred_chunks) < 1L) {
      return(NULL)
    }
    if (identical(likelihood, "mixed")) {
      return(list(
        logits = do.call(rbind, lapply(pred_chunks, function(pred) as.matrix(pred$logits))),
        sigma = unlist(lapply(pred_chunks, function(pred) as.numeric(pred$sigma)), use.names = FALSE)
      ))
    }
    if (identical(likelihood, "categorical")) {
      return(do.call(rbind, lapply(pred_chunks, as.matrix)))
    }
    if (identical(likelihood, "normal")) {
      return(list(
        mu = unlist(lapply(pred_chunks, function(pred) as.numeric(pred$mu)), use.names = FALSE),
        sigma = unlist(lapply(pred_chunks, function(pred) as.numeric(pred$sigma)), use.names = FALSE)
      ))
    }
    unlist(lapply(pred_chunks, as.numeric), use.names = FALSE)
  }
  locscale_reparam <- NULL
  if (!is.null(strenv$numpyro$infer) &&
      reticulate::py_has_attr(strenv$numpyro$infer, "reparam")) {
    reparam_mod <- strenv$numpyro$infer$reparam
    if (reticulate::py_has_attr(reparam_mod, "LocScaleReparam")) {
      locscale_reparam <- reparam_mod$LocScaleReparam
    }
  }
  if (is.null(locscale_reparam)) {
    locscale_reparam <- tryCatch(
      reticulate::import("numpyro.infer.reparam", delay_load = TRUE)$LocScaleReparam,
      error = function(e) NULL
    )
  }
  if (!is.null(locscale_reparam) &&
      reticulate::py_has_attr(strenv$numpyro, "handlers")) {
    reparam_config <- list()
    for (l_ in 1L:ModelDepth) {
      for (base in c("W_q_l", "W_k_l", "W_v_l", "W_o_l", "W_ff1_l", "W_ff2_l")) {
        site <- paste0(base, l_)
        reparam_config[[site]] <- locscale_reparam(centered = 0)
      }
    }
    reparam_config[["W_out"]] <- locscale_reparam(centered = 0)
    # Non-centered (Neal's-funnel-avoiding) parameterization for the secondary
    # hierarchical heads, matching the main trunk. Each is a Normal(0, tau)
    # sample_loc_scale site with a HalfNormal tau; under mean-field AutoNormal the
    # centered form induces a funnel that biases the scale posterior and inflates
    # gradient variance. numpyro's reparam handler keys on encountered site names,
    # so listing sites absent from a given model config is a safe no-op.
    for (site in c("W_add_out", "W_rc_r", "W_rc_c",
                   "W_q_cross", "W_k_cross", "W_v_cross", "W_o_cross")) {
      reparam_config[[site]] <- locscale_reparam(centered = 0)
    }
    # b_out is deliberately NOT reparameterized: LocScaleReparam renames the
    # latent to b_out_decentered, which would silently disconnect the
    # data-informed init_to_value warm start built under "b_out" by
    # neural_build_output_site_init_values (base-rate logit bias). The residual
    # funnel on this low-dimensional site is a smaller cost than losing the
    # output-bias warm start; the manual fallback path handles the equivalent
    # trade-off explicitly via b_out_site_name = "b_out_z".
    if (isTRUE(use_cross_term)) {
      reparam_config[["M_cross_raw"]] <- locscale_reparam(centered = 0)
    }
    model_fn <- tryCatch(
      strenv$numpyro$handlers$reparam(fn = model_fn_base, config = reparam_config),
      error = function(e) NULL
    )
    if (is.null(model_fn)) {
      model_fn <- model_fn_base
      manual_noncentered_loc_scale <- TRUE
    } else {
      manual_noncentered_loc_scale <- FALSE
    }
  } else {
    manual_noncentered_loc_scale <- TRUE
  }

  t0_ <- Sys.time()
  output_only_mode <- identical(tolower(as.character(uncertainty_scope)), "output")
  output_site_init_values <- neural_build_output_site_init_values(
    Y = Y_use,
    likelihood = likelihood,
    nOutcomes = nOutcomes,
    model_dims = ModelDims,
    output_dim = if (isTRUE(universal_enabled)) ai(universal_global_out_dim) else nOutcomes,
    b_out_site_name = if (isTRUE(manual_noncentered_loc_scale)) "b_out_z" else "b_out",
    tau_b_scale = tau_b_scale
  )
  user_site_init_values <- mcmc_control$init_site_values
  if (!is.null(user_site_init_values) && !is.list(user_site_init_values)) {
    stop(
      "'neural_mcmc_control$init_site_values' must be a named list when provided.",
      call. = FALSE
    )
  }
  if (!is.null(user_site_init_values) && length(user_site_init_values) > 0L) {
    output_site_init_values <- modifyList(output_site_init_values, user_site_init_values)
    # Under uncertainty_scope = "output" trunk sites are numpyro.param, which
    # init_to_value cannot seed; route the warm start through p2d's lookup so
    # foundation-model adaptation warm-starts the trunk in both scopes.
    p2d_warm_start_values <- user_site_init_values
  }
  init_to_value <- neural_get_init_to_value()
  use_svi <- isTRUE(output_only_mode) || identical(subsample_method, "batch_vi")
  if (isTRUE(universal_mixed_mode) && !isTRUE(use_svi)) {
    stop(
      "Universal mixed-mode foundation training requires SVI/batch_vi so pairwise and single losses share one optimizer run.",
      call. = FALSE
    )
  }
  run_mcmc_after_svi <- isTRUE(output_only_mode) && isTRUE(subsample_method %in% c("batch", "full"))
  emit_transformer_structure_banner <- function() {
    message(sprintf(
      "Bayesian Transformer complete. Pairwise=%s, Heads=%d, Depth=%d, Hidden=%d; likelihood=%s.",
      pairwise_mode,
      TransformerHeads,
      ModelDepth,
      MD_int,
      likelihood
    ))
  }
  format_metric_item <- function(label, value, digits = 4L) {
    if (is.null(value) || length(value) != 1L || !is.finite(value)) {
      return(NULL)
    }
    sprintf(paste0("%s=%.", as.integer(digits), "f"), label, value)
  }
  build_fit_metric_items <- function(metrics, likelihood) {
    if (is.null(metrics)) {
      return(character(0))
    }
    if (identical(likelihood, "mixed")) {
      items <- Filter(Negate(is.null), list(
        format_metric_item("NLL", metrics$nll, 4L)
      ))
      by_family <- metrics$by_family %||% list()
      if (!is.null(by_family$bernoulli)) {
        items <- c(items, Filter(Negate(is.null), list(
          format_metric_item("BernAUC", by_family$bernoulli$auc, 4L),
          format_metric_item("BernLL", by_family$bernoulli$log_loss, 4L),
          format_metric_item("BernAcc", by_family$bernoulli$accuracy, 3L),
          format_metric_item("BernBrier", by_family$bernoulli$brier, 4L)
        )))
      }
      if (!is.null(by_family$categorical)) {
        items <- c(items, Filter(Negate(is.null), list(
          format_metric_item("CatLL", by_family$categorical$log_loss, 4L),
          format_metric_item("CatAcc", by_family$categorical$accuracy, 3L)
        )))
      }
      if (!is.null(by_family$normal)) {
        items <- c(items, Filter(Negate(is.null), list(
          format_metric_item("NormRMSE", by_family$normal$rmse, 4L),
          format_metric_item("NormMAE", by_family$normal$mae, 4L),
          format_metric_item("NormNLL", by_family$normal$nll, 4L)
        )))
      }
      if (!is.null(by_family$ordinal)) {
        items <- c(items, Filter(Negate(is.null), list(
          format_metric_item("OrdLL", by_family$ordinal$log_loss, 4L),
          format_metric_item("OrdAcc", by_family$ordinal$accuracy, 3L),
          format_metric_item("OrdMAE", by_family$ordinal$mae, 4L),
          format_metric_item("RPS", by_family$ordinal$ranked_probability_score, 4L)
        )))
      }
      return(items)
    }
    if (!likelihood %in% c("bernoulli", "categorical", "normal", "ordinal")) {
      return(character(0))
    }
    if (likelihood == "bernoulli") {
      return(Filter(Negate(is.null), list(
        format_metric_item("AUC", metrics$auc, 4L),
        format_metric_item("LogLoss", metrics$log_loss, 4L),
        format_metric_item("Acc", metrics$accuracy, 3L),
        format_metric_item("Brier", metrics$brier, 4L)
      )))
    }
    if (likelihood == "categorical") {
      return(Filter(Negate(is.null), list(
        format_metric_item("LogLoss", metrics$log_loss, 4L),
        format_metric_item("Acc", metrics$accuracy, 3L)
      )))
    }
    if (likelihood == "ordinal") {
      return(Filter(Negate(is.null), list(
        format_metric_item("OrdLL", metrics$log_loss, 4L),
        format_metric_item("OrdAcc", metrics$accuracy, 3L),
        format_metric_item("OrdMAE", metrics$mae, 4L),
        format_metric_item("RPS", metrics$ranked_probability_score, 4L)
      )))
    }
    Filter(Negate(is.null), list(
      format_metric_item("RMSE", metrics$rmse, 4L),
      format_metric_item("MAE", metrics$mae, 4L),
      format_metric_item("NLL", metrics$nll, 4L)
    ))
  }
  emit_svi_fit_summary <- function() {
    completed_steps <- suppressWarnings(as.integer(svi_steps_completed %||% resolved_svi_steps))
    planned_steps <- suppressWarnings(as.integer(resolved_svi_steps))
    if (is.na(completed_steps)) {
      completed_steps <- planned_steps
    }

    summary_line <- NULL
    if (isTRUE(early_stopping_info$active) &&
        identical(early_stopping_info$reason, "validation_error")) {
      error_text <- early_stopping_info$error_message %||% "validation metric execution failed"
      error_text <- trimws(gsub("[\r\n]+", " ", as.character(error_text)))
      summary_line <- sprintf(
        "SVI fit summary: steps=%d/%d; validation error at step %d (%s).",
        completed_steps,
        planned_steps,
        as.integer(early_stopping_info$stop_step %||% completed_steps),
        error_text
      )
    } else if (isTRUE(early_stopping_info$active) &&
               identical(early_stopping_info$reason, "metric_failed")) {
      summary_line <- sprintf(
        "SVI fit summary: steps=%d/%d; validation %s unavailable at step %d.",
        completed_steps,
        planned_steps,
        early_stopping_info$metric %||% "metric",
        as.integer(early_stopping_info$stop_step %||% completed_steps)
      )
    } else if (isTRUE(early_stopping_info$active) &&
               is.finite(early_stopping_info$best_metric) &&
               !is.na(early_stopping_info$best_step)) {
      summary_line <- sprintf(
        "SVI fit summary: steps=%d/%d; best %s=%.6f at step %d.",
        completed_steps,
        planned_steps,
        early_stopping_info$metric,
        early_stopping_info$best_metric,
        as.integer(early_stopping_info$best_step)
      )
    } else {
      finite_losses <- svi_loss_curve[is.finite(svi_loss_curve)]
      final_elbo <- if (length(finite_losses) > 0L) tail(finite_losses, 1L) else NA_real_
      if (is.finite(final_elbo)) {
        summary_line <- sprintf(
          "SVI fit summary: steps=%d/%d; final ELBO=%.6f.",
          completed_steps,
          planned_steps,
          final_elbo
        )
      }
    }
    if (!is.null(summary_line)) {
      message(summary_line)
    }

    metric_items <- build_fit_metric_items(fit_metrics, likelihood)
    if (length(metric_items) > 0L) {
      metric_context <- c(ifelse(pairwise_mode, "pairwise", "single"))
      if (!is.null(fit_metrics$eval_note) && nzchar(fit_metrics$eval_note)) {
        metric_context <- c(metric_context, fit_metrics$eval_note)
      }
      if (!is.null(fit_metrics$eval_subset) && nzchar(fit_metrics$eval_subset)) {
        metric_context <- c(metric_context, fit_metrics$eval_subset)
      }
      message(sprintf(
        "Neural fit metrics (%s): %s",
        paste(metric_context, collapse = ", "),
        paste(metric_items, collapse = ", ")
      ))
    }
  }
  SVIParams <- NULL
  SVIInitValues <- NULL
  SVIPosteriorDraws <- NULL
  # TRUE only when posterior draws degenerate to a single restored point
  # (checkpoint rebuild where guide resampling failed); downstream variance
  # must then be NA, never a silent 0.
  posterior_draws_point_only <- FALSE
  svi_loss_curve <- NULL
  resolved_svi_steps <- NULL
  resolved_svi_num_draws <- NULL
  svi_budget_info <- NULL
  svi_steps_completed <- NULL
  optimizer_tag <- NULL
  user_supplied_optimizer <- FALSE
  svi_lr <- NA_real_
  schedule_tag <- NA_character_
  warmup_frac <- NA_real_
  warmup_steps <- NA_integer_
  decay_steps <- NA_integer_
  end_factor <- NA_real_
  lr_schedule <- NULL
  optimizer_diagnostics <- list(
    optimizer_status = if (isTRUE(use_svi)) "pending" else "not_svi",
    optimizer = NA_character_,
    optimizer_requested = NA_character_,
    user_supplied_optimizer = NA,
    schedule_name = NA_character_,
    svi_lr = NA_real_,
    warmup_frac = NA_real_,
    warmup_steps = NA_integer_,
    decay_steps = NA_integer_,
    end_factor = NA_real_,
    steps_completed = NA_integer_,
    lr_trace = numeric(0),
    lr_trace_status = if (isTRUE(use_svi)) "pending" else "not_svi"
  )
  gradient_diagnostics_enabled <- isTRUE(mcmc_control$gradient_diagnostics)
  gradient_diagnostics <- if (!isTRUE(use_svi)) {
    neural_build_gradient_diagnostics(
      status = "not_svi",
      source = NA_character_,
      notes = "SVI was not used for this fit."
    )
  } else if (!isTRUE(gradient_diagnostics_enabled)) {
    neural_build_gradient_diagnostics(
      status = "disabled",
      source = NA_character_,
      notes = "Checkpoint gradient diagnostics were disabled by neural_mcmc_control$gradient_diagnostics = FALSE."
    )
  } else {
    neural_build_gradient_diagnostics(
      status = "ok",
      source = "checkpoint_value_and_grad"
    )
  }
  gradient_diagnostics_failed <- FALSE
  early_stopping_enabled <- isTRUE(mcmc_control$early_stopping)
  early_stopping_info <- list(
    enabled = early_stopping_enabled,
    active = FALSE,
    stopped_early = FALSE,
    reason = if (isTRUE(early_stopping_enabled)) {
      "not_initialized"
    } else {
      "disabled"
    },
    error_message = NULL,
    metric = NULL,
    n_checks = NA_integer_,
    eval_every = NA_integer_,
    patience = NA_integer_,
    min_delta = NA_real_,
    best_step = NA_integer_,
    stop_step = NA_integer_,
    stop_check = NA_integer_,
    best_metric = NA_real_,
    final_metric = NA_real_,
    n_train = NA_integer_,
    n_validation = NA_integer_,
    validation_frac = NA_real_,
    validation_max_n = NULL,
    validation_batch_size = NA_integer_,
    validation_target_n = NA_integer_,
    validation_prediction_mode = NA_character_,
    validation_n_batches = NA_integer_,
    validation_loss_history = numeric(0)
  )
  if (isTRUE(use_svi)) {
    if (isTRUE(output_only_mode)) {
      message("Enlisting SVI with autoguide for output-only uncertainty...")
    } else {
      message("Enlisting SVI with autoguide for minibatched likelihood...")
    }
    emit_transformer_structure_banner()
    if (!is.null(strenv$numpyro) && reticulate::py_has_attr(strenv$numpyro, "clear_param_store")) {
      tryCatch(strenv$numpyro$clear_param_store(), error = function(e) NULL)
    }
    guide_name <- if (!is.null(mcmc_control$vi_guide)) {
      tolower(as.character(mcmc_control$vi_guide))
    } else {
      "auto_normal"
    }
    if (length(guide_name) != 1L || is.na(guide_name) || !nzchar(guide_name)) {
      guide_name <- "auto_normal"
    }
    guide_init_loc_fn <- NULL
    if (!is.null(init_to_value) && length(output_site_init_values) > 0L) {
      guide_init_loc_fn <- init_to_value(values = output_site_init_values)
    }
    guide <- if (is.null(guide_init_loc_fn)) {
      switch(guide_name,
             auto_delta = strenv$numpyro$infer$autoguide$AutoDelta(model_fn),
             auto_normal = strenv$numpyro$infer$autoguide$AutoNormal(model_fn),
             auto_diagonal = strenv$numpyro$infer$autoguide$AutoDiagonalNormal(model_fn),
             stop(sprintf("Unknown vi_guide '%s' for SVI.", guide_name), call. = FALSE))
    } else {
      switch(guide_name,
             auto_delta = strenv$numpyro$infer$autoguide$AutoDelta(model_fn, init_loc_fn = guide_init_loc_fn),
             auto_normal = strenv$numpyro$infer$autoguide$AutoNormal(model_fn, init_loc_fn = guide_init_loc_fn),
             auto_diagonal = strenv$numpyro$infer$autoguide$AutoDiagonalNormal(model_fn, init_loc_fn = guide_init_loc_fn),
             stop(sprintf("Unknown vi_guide '%s' for SVI.", guide_name), call. = FALSE))
    }
    n_particles <- ai(mcmc_control$svi_num_particles)
    if (length(n_particles) != 1L || is.na(n_particles) || n_particles < 1L) {
      n_particles <- 1L
    }
    optimizer_raw <- if (!is.null(mcmc_control$optimizer)) {
      tolower(as.character(mcmc_control$optimizer))
    } else {
      character(0)
    }
    user_supplied_optimizer <- length(optimizer_raw) == 1L &&
      !is.na(optimizer_raw) &&
      nzchar(optimizer_raw)
    optimizer_tag <- if (isTRUE(user_supplied_optimizer)) {
      optimizer_raw
    } else {
      "muon"
    }
    if (!optimizer_tag %in% c("adam", "adamw", "adabelief", "muon")) {
      stop(
        sprintf("Unknown optimizer '%s' for SVI.", optimizer_tag),
        call. = FALSE
      )
    }
    svi_lr <- as.numeric(mcmc_control$svi_lr)
    if (length(svi_lr) != 1L || is.na(svi_lr) || !is.finite(svi_lr) || svi_lr <= 0) {
      svi_lr <- 0.01
    }
    schedule_tag <- if (!is.null(mcmc_control$svi_lr_schedule)) {
      tolower(as.character(mcmc_control$svi_lr_schedule))
    } else {
      "warmup_cosine"
    }
    if (length(schedule_tag) != 1L || is.na(schedule_tag) || !nzchar(schedule_tag)) {
      schedule_tag <- "warmup_cosine"
    }
    if (!schedule_tag %in% c("none", "constant", "cosine", "warmup_cosine")) {
      stop(
        sprintf("Unknown svi_lr_schedule '%s'.", schedule_tag),
        call. = FALSE
      )
    }
    if (identical(schedule_tag, "warmup_cosine") &&
        isTRUE(user_supplied_svi_steps) &&
        !is.null(mcmc_control$svi_steps)) {
      svi_steps_raw_int <- suppressWarnings(as.integer(mcmc_control$svi_steps))
      if (length(svi_steps_raw_int) == 1L &&
          !is.na(svi_steps_raw_int) &&
          is.finite(svi_steps_raw_int) &&
          svi_steps_raw_int <= 5L) {
        schedule_tag <- "constant"
      }
    }
    n_obs_svi <- length(Y_use)
    pairwise_scaling <- pairwise_mode
    if (isTRUE(universal_mixed_mode)) {
      n_obs_svi <- length(Y_use)
      pairwise_scaling <- TRUE
    } else if (isTRUE(pairwise_mode) && !is.null(pair_mat) && nrow(pair_mat) > 0L) {
      n_obs_svi <- nrow(pair_mat)
      pairwise_scaling <- TRUE
    }
    if ((is.null(pair_mat) || nrow(pair_mat) < 1L) &&
        isTRUE(diff) && !is.null(pair_id_) && length(pair_id_) > 0L) {
      pair_id_use <- pair_id_
      pair_id_use <- pair_id_use[!is.na(pair_id_use)]
      if (length(pair_id_use) > 0L) {
        n_obs_svi <- length(unique(pair_id_use))
        pairwise_scaling <- TRUE
      }
    }
    svi_subsample_method <- if (isTRUE(subsample_method %in% c("batch", "batch_vi"))) {
      "batch_vi"
    } else {
      subsample_method
    }
    # Issue 4 (revised): the plate multiplies the minibatch likelihood by
    # N/subsample_size, inflating the ELBO gradient. The earlier LR rescale by
    # sqrt(subsample/N) cannot achieve its stated goal -- clip_by_global_norm sits
    # UPSTREAM of the learning rate in the optax chain, so scaling the LR never
    # changes how often the clip fires. And the supported optimizers (muon
    # orthogonalization; adam/adamw/adabelief m/sqrt(v)) are invariant to a global
    # gradient scale, so the plate inflation is already normalized away -- the
    # rescale then only shrinks the effective LR (up to ~10x), wasting the step
    # budget. So skip it for scale-invariant optimizers. If clip saturation is ever
    # a real concern, the correct lever is to scale clip_global_norm by N/subsample,
    # not the LR. The gate is retained for any future non-scale-invariant optimizer.
    optimizer_scale_invariant <- optimizer_tag %in% c("adam", "adamw", "adabelief", "muon")
    if (isTRUE(mcmc_control$svi_lr_plate_rescale) &&
        isTRUE(subsample_method_model %in% c("batch", "batch_vi")) &&
        !isTRUE(optimizer_scale_invariant)) {
      plate_batch_size <- min(as.numeric(mcmc_control$batch_size), as.numeric(n_obs_svi))
      if (is.finite(plate_batch_size) && plate_batch_size > 0 &&
          n_obs_svi > 0L && plate_batch_size < as.numeric(n_obs_svi)) {
        svi_lr_plate_factor <- sqrt(plate_batch_size / as.numeric(n_obs_svi))
        svi_lr_pre_rescale <- svi_lr
        svi_lr <- svi_lr * svi_lr_plate_factor
        message(sprintf(
          "SVI plate LR rescale: batch=%d, N=%d -> lr %.5g x %.4f = %.5g",
          as.integer(plate_batch_size), as.integer(n_obs_svi),
          svi_lr_pre_rescale, svi_lr_plate_factor, svi_lr
        ))
      }
    }
    svi_budget_info <- neural_resolve_svi_budget(
      svi_steps_input = mcmc_control$svi_steps,
      svi_num_draws_input = mcmc_control$svi_num_draws,
      user_supplied_svi_steps = user_supplied_svi_steps,
      user_supplied_svi_num_draws = user_supplied_svi_num_draws,
      n_obs = n_obs_svi,
      n_factors = length(factor_levels_int),
      factor_levels = factor_levels_int,
      model_dims = ModelDims,
      model_depth = ModelDepth,
      n_party_levels = n_party_levels,
      n_resp_party_levels = n_resp_party_levels,
      n_resp_covariates = n_resp_covariates,
      n_outcomes = nOutcomes,
      pairwise_mode = pairwise_scaling,
      use_matchup_token = use_matchup_token,
      use_cross_encoder = use_cross_encoder,
      use_cross_term = use_cross_term,
      use_cross_attn = use_cross_attn,
      use_qk_norm = qk_norm_enabled,
      batch_size = mcmc_control$batch_size,
      subsample_method = svi_subsample_method,
      output_only_mode = output_only_mode,
      likelihood = likelihood
    )
    svi_steps <- as.integer(svi_budget_info$svi_steps)
    resolved_svi_steps <- svi_steps
    resolved_svi_num_draws <- as.integer(svi_budget_info$svi_num_draws)
    mcmc_control$svi_steps <- svi_steps
    mcmc_control$svi_num_draws <- resolved_svi_num_draws
    if (isTRUE(svi_budget_info$used_optimal)) {
      message(sprintf("Using svi_steps='optimal' => %d steps.", svi_steps))
    }
    if (isTRUE(svi_budget_info$applied_output_single_normal_batch_vi_floor)) {
      message(sprintf(
        "Applying output-only single-model normal batch_vi SVI floor => %d steps.",
        svi_steps
      ))
    }
    warmup_frac <- if (!is.null(mcmc_control$svi_lr_warmup_frac)) {
      as.numeric(mcmc_control$svi_lr_warmup_frac)
    } else {
      0.1
    }
    if (length(warmup_frac) != 1L || is.na(warmup_frac) || !is.finite(warmup_frac)) {
      warmup_frac <- 0.1
    }
    warmup_frac <- max(0, min(warmup_frac, 0.9))
    decay_steps <- max(2L, svi_steps)
    warmup_steps <- if (schedule_tag == "warmup_cosine") {
      max(1L, min(as.integer(round(svi_steps * warmup_frac)), decay_steps - 1L))
    } else {
      0L
    }
    end_factor <- if (!is.null(mcmc_control$svi_lr_end_factor)) {
      as.numeric(mcmc_control$svi_lr_end_factor)
    } else {
      0.01
    }
    if (length(end_factor) != 1L || is.na(end_factor) || !is.finite(end_factor)) {
      end_factor <- 0.01
    }
    end_factor <- max(0, min(end_factor, 1))
    # Resolve the optimizer tag before the checkpoint fingerprint is computed
    # (the fingerprint includes it, and resolution may change it, e.g. the
    # muon -> adamw fallback).
    optimizer_tag <- neural_resolve_svi_optimizer_tag(
      optimizer_tag = optimizer_tag,
      guide_name = guide_name,
      user_supplied_optimizer = user_supplied_optimizer
    )
    # The checkpoint is restored BEFORE the LR schedule is built so that a
    # partial-run resume can continue the schedule from the completed step
    # (offset below) instead of re-entering warmup at peak LR mid-run.
    svi_checkpoint <- neural_svi_checkpoint_control(mcmc_control)
    svi_checkpoint_fingerprint <- NULL
    svi_checkpoint_latest <- NULL
    svi_checkpoint_best <- NULL
    if (isTRUE(svi_checkpoint$enabled)) {
      svi_checkpoint_fingerprint <- neural_svi_checkpoint_fingerprint(list(
        data = list(
          Y = Y_use,
          W = W_,
          W_idx_compact = if (isTRUE(compact_training)) W_idx_compact_use else NULL,
          X = X_use,
          X_compact = if (isTRUE(compact_training)) X_compact_use else NULL,
          X_present = X_present_use,
          X_present_compact = if (isTRUE(compact_training)) X_present_compact_use else NULL,
          pair_id = pair_id_ %||% NULL,
          profile_order = profile_order_ %||% NULL,
          competing_group_variable_candidate = competing_group_variable_candidate_ %||% NULL,
          competing_group_variable_respondent = competing_group_variable_respondent_ %||% NULL,
          respondent_id = respondent_id %||% NULL,
          respondent_task_id = respondent_task_id %||% NULL
        ),
        model = list(
          likelihood = likelihood,
          n_outcomes = as.integer(nOutcomes),
          factor_levels = factor_levels_int,
          pairwise_mode = isTRUE(pairwise_mode),
          pairwise_context_mode = pairwise_context_mode,
          model_dims = as.integer(ModelDims),
          model_depth = as.integer(ModelDepth),
          residual_mode = residual_mode,
          cross_candidate_encoder_mode = cross_candidate_encoder_mode,
          qk_norm = isTRUE(qk_norm_enabled),
          subsample_method = subsample_method,
          output_only_mode = isTRUE(output_only_mode),
          guide = guide_name,
          optimizer = optimizer_tag,
          svi_lr = svi_lr,
          schedule = schedule_tag,
          warmup_frac = warmup_frac,
          end_factor = end_factor,
          n_particles = as.integer(n_particles),
          svi_budget_info = svi_budget_info
        ),
        control = neural_svi_checkpoint_strip_control(mcmc_control),
        token = neural_token_info_use
      ))
      if (isTRUE(svi_checkpoint$resume)) {
        svi_checkpoint_latest <- neural_svi_checkpoint_restore_latest(
          svi_checkpoint$path,
          svi_checkpoint_fingerprint
        )
        svi_checkpoint_best <- neural_svi_checkpoint_restore_best(
          svi_checkpoint$path,
          svi_checkpoint_fingerprint
        )
        if (!is.null(svi_checkpoint_latest)) {
          message(sprintf(
            "Resuming neural SVI checkpoint from %s at step %d/%d.",
            svi_checkpoint$path,
            as.integer(svi_checkpoint_latest$completed_step %||% 0L),
            as.integer(svi_checkpoint_latest$resolved_svi_steps %||% svi_steps)
          ))
        }
      }
    }
    lr_schedule_step_offset <- 0L
    if (!is.null(svi_checkpoint_latest)) {
      resume_completed_for_lr <- as.integer(svi_checkpoint_latest$completed_step %||% 0L)
      if (!is.na(resume_completed_for_lr) &&
          resume_completed_for_lr > 0L &&
          resume_completed_for_lr < as.integer(svi_steps)) {
        lr_schedule_step_offset <- resume_completed_for_lr
      }
    }
    lr_schedule <- if (schedule_tag == "warmup_cosine") {
      strenv$optax$warmup_cosine_decay_schedule(
        init_value = svi_lr * end_factor,
        peak_value = svi_lr,
        warmup_steps = warmup_steps,
        decay_steps = decay_steps,
        end_value = svi_lr * end_factor
      )
    } else if (schedule_tag == "cosine") {
      strenv$optax$cosine_decay_schedule(
        init_value = svi_lr,
        decay_steps = decay_steps,
        alpha = end_factor
      )
    } else {
      svi_lr
    }
    if (lr_schedule_step_offset > 0L && !is.numeric(lr_schedule) &&
        !is.null(strenv$jax_offset_schedule)) {
      lr_schedule <- strenv$jax_offset_schedule(lr_schedule, ai(lr_schedule_step_offset))
      message(sprintf(
        "Resumed LR schedule continues from step %d (no re-warmup).",
        as.integer(lr_schedule_step_offset)
      ))
    }
    clip_global_norm <- as.numeric(mcmc_control$svi_clip_global_norm %||% 10)
    if (length(clip_global_norm) != 1L ||
        is.na(clip_global_norm) ||
        !is.finite(clip_global_norm) ||
        clip_global_norm <= 0) {
      clip_global_norm <- NA_real_
    }
    optax_clip_available <- reticulate::py_has_attr(strenv$optax, "clip_by_global_norm") &&
      reticulate::py_has_attr(strenv$optax, "chain")
    optax_to_numpyro_available <- reticulate::py_has_attr(strenv$numpyro$optim, "optax_to_numpyro")
    clip_enabled <- isTRUE(optax_clip_available) && is.finite(clip_global_norm)
    wrap_optax_optimizer <- function(optax_optim) {
      if (!isTRUE(clip_enabled)) {
        return(optax_optim)
      }
      strenv$optax$chain(
        strenv$optax$clip_by_global_norm(as.numeric(clip_global_norm)),
        optax_optim
      )
    }
    optax_to_numpyro_optimizer <- function(optax_optim) {
      optax_optim <- wrap_optax_optimizer(optax_optim)
      if (isTRUE(optax_to_numpyro_available)) {
        return(strenv$numpyro$optim$optax_to_numpyro(optax_optim))
      }
      optax_optim
    }
    muon_available <- reticulate::py_has_attr(strenv$optax, "contrib") &&
      reticulate::py_has_attr(strenv$optax$contrib, "muon")
    # optimizer_tag was resolved above, before the checkpoint fingerprint.
    optimizer_diagnostics <- list(
      optimizer_status = "configured",
      optimizer = optimizer_tag,
      optimizer_requested = if (isTRUE(user_supplied_optimizer)) optimizer_raw else NA_character_,
      user_supplied_optimizer = isTRUE(user_supplied_optimizer),
      schedule_name = schedule_tag,
      svi_lr = svi_lr,
      warmup_frac = warmup_frac,
      warmup_steps = as.integer(warmup_steps),
      decay_steps = as.integer(decay_steps),
      end_factor = end_factor,
      clip_global_norm = if (is.finite(clip_global_norm)) clip_global_norm else NA_real_,
      clip_status = if (isTRUE(clip_enabled)) "enabled" else if (!is.finite(clip_global_norm)) "disabled" else "unavailable",
      steps_completed = NA_integer_,
      lr_trace = numeric(0),
      lr_trace_status = "pending"
    )
    svi_optim <- if (optimizer_tag == "adam") {
      if (isTRUE(clip_enabled) &&
          reticulate::py_has_attr(strenv$optax, "adam")) {
        optax_to_numpyro_optimizer(strenv$optax$adam(learning_rate = lr_schedule))
      } else {
        strenv$numpyro$optim$Adam(lr_schedule)
      }
    } else if (optimizer_tag == "adamw") {
      # optax.adamw defaults to weight_decay=1e-4 with mask=None, which applies
      # decoupled decay to EVERY optimized leaf -- including the AutoNormal
      # posterior stddevs (*_auto_scale), RMSNorm/QK gains, and the deterministic
      # ReZero gates (init small) -- decaying gates toward 0 silently undoes the
      # ReZero fix and mis-calibrates the variational posterior. Set weight_decay=0
      # to match the muon default (adam_weight_decay=0). A name-masked nonzero decay
      # restricted to true weight matrices would be the richer alternative.
      if (isTRUE(clip_enabled) &&
          reticulate::py_has_attr(strenv$optax, "adamw")) {
        optax_to_numpyro_optimizer(strenv$optax$adamw(learning_rate = lr_schedule,
                                                      weight_decay = 0))
      } else if (reticulate::py_has_attr(strenv$numpyro$optim, "AdamW")) {
        strenv$numpyro$optim$AdamW(lr_schedule, weight_decay = 0)
      } else if (reticulate::py_has_attr(strenv$optax, "adamw")) {
        optax_optim <- strenv$optax$adamw(learning_rate = lr_schedule, weight_decay = 0)
        optax_to_numpyro_optimizer(optax_optim)
      } else {
        stop(
          "optimizer='adamw' requested, but neither numpyro.optim.AdamW nor optax.adamw is available.",
          call. = FALSE
        )
      }
    } else if (optimizer_tag == "muon") {
      if (isTRUE(muon_available)) {
        muon_dimnums <- tryCatch(
          neural_get_muon_dimension_numbers_callable(),
          error = function(e) NULL
        )
        # Never run Muon without the explicit weight partition: optax's default
        # ('muon' iff ndim == 2) would orthogonalize embedding tables and output
        # heads -- the known-bad Muon usage the partition regex exists to
        # prevent. Fall back to adamw/adam loudly instead of degrading silently.
        muon_adam_fallback <- function(reason) {
          fallback_tag <- neural_default_svi_fallback_optimizer()
          warning(sprintf(paste0(
            "optimizer='muon' cannot be configured safely (%s); ",
            "falling back to '%s'. A partition-less Muon would orthogonalize ",
            "embedding tables and output heads."
          ), reason, fallback_tag), call. = FALSE)
          optimizer_tag <<- fallback_tag
          if (identical(fallback_tag, "adamw") &&
              reticulate::py_has_attr(strenv$optax, "adamw")) {
            strenv$optax$adamw(learning_rate = lr_schedule, weight_decay = 0)
          } else {
            strenv$optax$adam(learning_rate = lr_schedule)
          }
        }
        optax_optim <- if (is.null(muon_dimnums)) {
          muon_adam_fallback("Muon weight dimension numbers are unavailable")
        } else {
          muon_kwargs <- list(
            learning_rate = lr_schedule,
            adam_weight_decay = 0,
            consistent_rms = 0.2,
            muon_weight_dimension_numbers = muon_dimnums
          )
          tryCatch(
            do.call(strenv$optax$contrib$muon, muon_kwargs),
            error = function(e) {
              muon_kwargs_fallback <- list(
                learning_rate = lr_schedule,
                muon_weight_dimension_numbers = muon_dimnums
              )
              tryCatch(
                do.call(strenv$optax$contrib$muon, muon_kwargs_fallback),
                error = function(e2) {
                  muon_adam_fallback(sprintf(
                    "installed optax rejected the Muon weight partition: %s",
                    conditionMessage(e2)
                  ))
                }
              )
            }
          )
        }
        optax_to_numpyro_optimizer(optax_optim)
      } else {
        stop(
          "optimizer='muon' requested, but optax.contrib.muon is unavailable.",
          call. = FALSE
        )
      }
    } else {
      optax_optim <- strenv$optax$adabelief(learning_rate = lr_schedule)
      optax_to_numpyro_optimizer(optax_optim)
    }
    # The muon safety fallback may have downgraded the tag mid-construction.
    optimizer_diagnostics$optimizer <- optimizer_tag
    # With a mean-field AutoNormal/AutoDiagonalNormal guide, every weight latent is
    # Normal-prior / Normal-posterior, so the per-site KL is available in closed
    # form. TraceMeanField_ELBO substitutes that analytic KL for Trace_ELBO's
    # single-sample MC estimate of the entropy/cross-entropy term (num_particles=1
    # by default), removing avoidable gradient variance at no extra cost. Fall back
    # to Trace_ELBO for non-mean-field guides or if the class is unavailable.
    elbo_loss <- local({
      mean_field_guide <- guide_name %in% c("auto_normal", "auto_diagonal")
      if (isTRUE(mean_field_guide) &&
          reticulate::py_has_attr(strenv$numpyro$infer, "TraceMeanField_ELBO")) {
        tryCatch(
          strenv$numpyro$infer$TraceMeanField_ELBO(num_particles = n_particles),
          error = function(e) strenv$numpyro$infer$Trace_ELBO(num_particles = n_particles)
        )
      } else {
        strenv$numpyro$infer$Trace_ELBO(num_particles = n_particles)
      }
    })
    svi <- strenv$numpyro$infer$SVI(
      model = model_fn,
      guide = guide,
      optim = svi_optim,
      loss = elbo_loss
    )
    # Activate the MoE load-balancing auxiliary factor (Issue 5) for the duration of
    # SVI training so it enters the ELBO; deactivated before posterior sampling /
    # prediction so it never perturbs predictive outputs.
    strenv$moe_aux_active <- TRUE
    on.exit(strenv$moe_aux_active <- FALSE, add = TRUE)
    svi_model_args <- if (isTRUE(compact_training)) {
      compact_batch_args(compact_sample_obs_idx())
    } else if (isTRUE(universal_mixed_mode)) {
      list(
        X_left = X_left_jnp,
        X_right = X_right_jnp,
        party_left = party_left_jnp,
        party_right = party_right_jnp,
        resp_party = resp_party_pair_jnp,
        resp_cov = resp_cov_pair_jnp,
        resp_cov_present = resp_cov_present_pair_jnp,
        experiment_index = experiment_index_pair_jnp,
        likelihood_code = likelihood_code_pair_jnp,
        n_outcomes_obs = n_outcomes_pair_jnp,
        Y_obs = Y_pair_jnp,
        obs_scale = obs_scale_pair_jnp,
        X_single = X_single_jnp,
        party_single = party_single_jnp,
        resp_party_single = resp_party_single_jnp,
        resp_cov_single = resp_cov_single_jnp,
        resp_cov_present_single = resp_cov_present_single_jnp,
        experiment_index_single = experiment_index_single_jnp,
        likelihood_code_single = likelihood_code_single_jnp,
        n_outcomes_single = n_outcomes_single_jnp,
        Y_single_obs = Y_single_jnp,
        obs_scale_single = obs_scale_single_jnp
      )
    } else if (pairwise_mode) {
      list(
        X_left = X_left_jnp,
        X_right = X_right_jnp,
        party_left = party_left_jnp,
        party_right = party_right_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        resp_cov_present = resp_cov_present_jnp,
        experiment_index = experiment_index_jnp,
        likelihood_code = likelihood_code_jnp,
        n_outcomes_obs = n_outcomes_obs_jnp,
        Y_obs = Y_jnp,
        obs_scale = obs_scale_jnp
      )
    } else {
      list(
        X = X_single_jnp,
        party = party_single_jnp,
        resp_party = resp_party_jnp,
        resp_cov = resp_cov_jnp,
        resp_cov_present = resp_cov_present_jnp,
        experiment_index = experiment_index_jnp,
        likelihood_code = likelihood_code_jnp,
        n_outcomes_obs = n_outcomes_obs_jnp,
        Y_obs = Y_jnp,
        obs_scale = obs_scale_jnp
      )
    }
    rng_key <- neural_training_prng_key(neural_training_seed, stream = 1L)
    validation_split_reason <- "validation_split_unavailable"
    build_svi_validation_split <- function() {
      validation_split_reason <<- "validation_split_unavailable"
      n_total <- length(Y_use)
      if (n_total < 2L) {
        return(NULL)
      }

      likelihood_code_all <- NULL
      n_outcomes_all <- NULL
      split_y <- NULL
      if (likelihood == "mixed") {
        y_all <- as.numeric(Y_use)
        likelihood_code_all <- universal_likelihood_code_use
        n_outcomes_all <- universal_n_outcomes_use_int
        ok_rows <- mixed_row_is_valid_r(
          y = y_all,
          likelihood_code_obs = likelihood_code_all,
          n_outcomes_obs = n_outcomes_all
        )
        split_y <- mixed_eval_strata_r(
          y = y_all,
          likelihood_code_obs = likelihood_code_all,
          experiment_index = experiment_index_use,
          n_outcomes_obs = n_outcomes_all
        )
      } else if (likelihood == "categorical") {
        y_all <- as.integer(reticulate::py_to_r(strenv$np$array(Y_jnp)))
        ok_rows <- !is.na(y_all)
      } else if (likelihood == "bernoulli") {
        y_all <- as.numeric(Y_use)
        ok_rows <- is.finite(y_all) & (y_all %in% c(0, 1))
      } else {
        y_all <- as.numeric(Y_use)
        ok_rows <- is.finite(y_all)
      }
      eval_idx <- which(ok_rows)
      if (length(eval_idx) < 2L) {
        return(NULL)
      }

      cluster_obs <- NULL
      if (exists("varcov_cluster_variable_", inherits = TRUE)) {
        cluster_raw <- get("varcov_cluster_variable_", inherits = TRUE)
        if (!is.null(cluster_raw) && length(cluster_raw) > 0L) {
          if (isTRUE(universal_mixed_mode) &&
              !is.null(pair_mat) && nrow(pair_mat) > 0L) {
            # Mixed pooled rows are laid out c(pair_obs, single_obs); build the
            # cluster vector in the same order so respondent clustering is not
            # silently discarded by the length check below (which previously
            # let rows from one respondent straddle train/validation).
            need <- suppressWarnings(
              max(c(pair_mat[, 1], universal_single_rows), na.rm = TRUE)
            )
            if (is.finite(need) && length(cluster_raw) >= need) {
              cluster_obs <- c(
                cluster_raw[pair_mat[, 1]],
                cluster_raw[universal_single_rows]
              )
            }
          } else if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0L) {
            need <- suppressWarnings(max(pair_mat[, 1], na.rm = TRUE))
            if (is.finite(need) && length(cluster_raw) >= need) {
              cluster_obs <- cluster_raw[pair_mat[, 1]]
            }
          } else if (!pairwise_mode && length(cluster_raw) == n_total) {
            cluster_obs <- cluster_raw
          }
        }
      }
      if (is.null(cluster_obs) || length(cluster_obs) != n_total) {
        cluster_raw <- NULL

        subset_by_indi <- function(x) {
          if (is.null(x) || length(x) == 0L) {
            return(NULL)
          }
          x <- as.vector(x)
          if (!exists("indi_", inherits = TRUE)) {
            return(x)
          }
          idx <- get("indi_", inherits = TRUE)
          if (is.null(idx) || length(idx) == 0L) {
            return(x)
          }
          if (length(x) == length(Y_)) {
            return(x)
          }
          max_idx <- suppressWarnings(max(as.integer(idx), na.rm = TRUE))
          if (is.finite(max_idx) && length(x) >= max_idx) {
            return(x[idx])
          }
          x
        }

        resp_id <- if (exists("respondent_id", inherits = TRUE)) {
          subset_by_indi(get("respondent_id", inherits = TRUE))
        } else {
          NULL
        }
        task_id <- if (exists("respondent_task_id", inherits = TRUE)) {
          subset_by_indi(get("respondent_task_id", inherits = TRUE))
        } else {
          NULL
        }

        if (!is.null(resp_id) && length(resp_id) > 0L) {
          cluster_raw <- resp_id
        } else if (!is.null(task_id) && length(task_id) > 0L) {
          cluster_raw <- task_id
        }

        if (!is.null(cluster_raw) && length(cluster_raw) > 0L) {
          if (isTRUE(universal_mixed_mode) &&
              !is.null(pair_mat) && nrow(pair_mat) > 0L) {
            need <- suppressWarnings(
              max(c(pair_mat[, 1], universal_single_rows), na.rm = TRUE)
            )
            if (is.finite(need) && length(cluster_raw) >= need) {
              cluster_obs <- c(
                cluster_raw[pair_mat[, 1]],
                cluster_raw[universal_single_rows]
              )
            }
          } else if (pairwise_mode && !is.null(pair_mat) && nrow(pair_mat) > 0L) {
            cluster_obs <- cluster_raw[pair_mat[, 1]]
          } else if (!pairwise_mode && length(cluster_raw) == n_total) {
            cluster_obs <- cluster_raw
          }
        }
      }

      cluster_eval <- if (!is.null(cluster_obs) && length(cluster_obs) == n_total) {
        cluster_obs[eval_idx]
      } else {
        NULL
      }

      split_seed <- eval_control$seed
      if (is.null(split_seed) || is.na(split_seed) || !is.finite(split_seed)) {
        split_seed <- 123L
      }
      n_folds_es <- min(5L, length(eval_idx))
      if (n_folds_es < 2L) {
        return(NULL)
      }
      folds_out <- cs_make_stratified_folds(
        n = length(eval_idx),
        n_folds = as.integer(n_folds_es),
        y = if (identical(likelihood, "mixed")) split_y[eval_idx] else y_all[eval_idx],
        cluster = cluster_eval,
        seed = as.integer(split_seed)
      )
      if (is.null(folds_out) || is.null(folds_out$fold_id)) {
        return(NULL)
      }
      fold_id <- as.integer(folds_out$fold_id)
      available_folds <- sort(unique(fold_id[!is.na(fold_id)]))
      if (length(available_folds) < 2L && length(eval_idx) > 1L) {
        validation_idx <- eval_idx[1L]
        train_idx <- eval_idx[-1L]
      } else {
        validation_fold <- available_folds[[1L]]
        validation_idx <- eval_idx[fold_id == validation_fold]
        train_idx <- eval_idx[fold_id != validation_fold]
      }
      if (length(train_idx) < 1L || length(validation_idx) < 1L) {
        return(NULL)
      }
      if (identical(likelihood, "mixed")) {
        train_families <- sort(unique(likelihood_code_all[train_idx]))
        validation_families <- sort(unique(likelihood_code_all[validation_idx]))
        if (!all(validation_families %in% train_families)) {
          validation_split_reason <<- "mixed_family_validation_split_unavailable"
          return(NULL)
        }
      }
      validation_target_n <- neural_resolve_early_stopping_validation_target_n(
        n_eval = length(eval_idx),
        n_validation_available = length(validation_idx),
        validation_frac = early_stopping_validation_frac,
        validation_max_n = early_stopping_validation_max_n,
        validation_min_n = 32L
      )
      if (!is.finite(validation_target_n) || validation_target_n < 1L) {
        return(NULL)
      }
      if (length(validation_idx) > validation_target_n) {
        # Downsample under a local seed without clobbering the caller's RNG.
        old_seed_split <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
          get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
        } else {
          NULL
        }
        on.exit({
          if (is.null(old_seed_split)) {
            if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
              rm(".Random.seed", envir = .GlobalEnv)
            }
          } else {
            assign(".Random.seed", old_seed_split, envir = .GlobalEnv)
          }
        }, add = TRUE)
        set.seed(as.integer(split_seed) + 1L)
        validation_fold_rows <- validation_idx
        validation_idx <- sort(sample(validation_idx, size = validation_target_n, replace = FALSE))
        # Return the fold rows NOT chosen for validation to the training set.
        # Previously they were dropped from both sets, silently shrinking training
        # to ~1 - 1/n_folds of N (~80%) regardless of validation_frac (and worse for
        # large N, where validation is additionally capped at validation_max_n).
        # train_idx stays disjoint from the downsampled validation_idx, so there is
        # no train/validation leakage.
        returned_to_train <- setdiff(validation_fold_rows, validation_idx)
        if (length(returned_to_train) > 0L) {
          train_idx <- sort(c(train_idx, returned_to_train))
        }
      }
      validation_batch_size <- if (isTRUE(early_stopping_validation_batch_size_supplied)) {
        neural_resolve_early_stopping_validation_batch_size(
          validation_target_n = length(validation_idx),
          validation_batch_size = early_stopping_validation_batch_size
        )
      } else {
        as.integer(length(validation_idx))
      }
      validation_prediction_mode <- if (validation_batch_size < length(validation_idx)) {
        "batched_fallback"
      } else {
        "single_jit_call"
      }
      validation_batches <- if (identical(validation_prediction_mode, "batched_fallback")) {
        split(
          validation_idx,
          ceiling(seq_along(validation_idx) / validation_batch_size)
        )
      } else {
        list(validation_idx)
      }

      list(
        train_idx = as.integer(train_idx),
        train_idx_jnp = normalize_model_obs_idx(train_idx),
        validation_idx = as.integer(validation_idx),
        validation_batches = unname(lapply(validation_batches, as.integer)),
        validation_batch_size = as.integer(validation_batch_size),
        validation_target_n = as.integer(validation_target_n),
        validation_prediction_mode = validation_prediction_mode,
        y_all = y_all,
        likelihood_code_all = likelihood_code_all,
        n_outcomes_all = n_outcomes_all
      )
    }
    extract_svi_param_sites <- function(svi_params_current) {
      param_sites <- NULL
      if (reticulate::py_has_attr(guide, "median")) {
        param_sites <- tryCatch(guide$median(svi_params_current), error = function(e) NULL)
        if (is.null(param_sites)) {
          param_sites <- tryCatch(
            do.call(guide$median, c(list(svi_params_current), svi_model_args)),
            error = function(e) NULL
          )
        }
      }
      if (is.null(param_sites) && reticulate::py_has_attr(guide, "sample_posterior")) {
        fixed_key <- strenv$jax$random$PRNGKey(ai(0L))
        param_sites <- tryCatch(
          do.call(guide$sample_posterior, c(list(fixed_key, svi_params_current), svi_model_args)),
          error = function(e) {
            tryCatch(
              do.call(
                guide$sample_posterior,
                c(list(fixed_key, svi_params_current, sample_shape = reticulate::tuple()), svi_model_args)
              ),
              error = function(e2) NULL
            )
          }
        )
      }
      param_sites
    }
    compute_svi_validation_metric <- function(svi_state_current, validation_split) {
      svi_params_current <- tryCatch(svi$get_params(svi_state_current), error = function(e) NULL)
      if (is.null(svi_params_current)) {
        return(NA_real_)
      }
      param_sites <- extract_svi_param_sites(svi_params_current)
      validation_batches <- validation_split$validation_batches %||% list(validation_split$validation_idx)
      validation_prediction_mode <- validation_split$validation_prediction_mode %||% "batched_fallback"
      pred_eval <- if (identical(validation_prediction_mode, "single_jit_call")) {
        svi_validation_predict_chunk(
          param_sites,
          validation_split$validation_idx,
          fallback_params = svi_params_current
        )
      } else {
        pred_chunks <- vector("list", length(validation_batches))
        for (batch_idx in seq_along(validation_batches)) {
          pred_chunks[[batch_idx]] <- svi_validation_predict_chunk(
            param_sites,
            validation_batches[[batch_idx]],
            fallback_params = svi_params_current
          )
        }
        combine_svi_validation_predictions(pred_chunks)
      }
      if (is.null(pred_eval)) {
        return(NA_real_)
      }
      metrics <- if (identical(likelihood, "mixed")) {
        compute_mixed_outcome_metrics_r(
          y_eval = validation_split$y_all[validation_split$validation_idx],
          pred_eval = pred_eval,
          likelihood_code_obs = validation_split$likelihood_code_all[validation_split$validation_idx],
          n_outcomes_obs = validation_split$n_outcomes_all[validation_split$validation_idx],
          task_mode_obs = if (!is.null(universal_task_mode_use)) {
            universal_task_mode_use[validation_split$validation_idx]
          } else {
            NULL
          },
          experiment_index = if (!is.null(experiment_index_use)) {
            experiment_index_use[validation_split$validation_idx]
          } else {
            NULL
          }
        )
      } else {
        if (identical(likelihood, "ordinal")) {
          ordinal_metrics_r(
            y_eval = validation_split$y_all[validation_split$validation_idx],
            prob_mat = pred_eval,
            n_outcomes_obs = rep.int(as.integer(nOutcomes), length(validation_split$validation_idx))
          )
        } else {
          cs_compute_outcome_metrics(
            y_eval = validation_split$y_all[validation_split$validation_idx],
            pred_eval = pred_eval,
            likelihood = likelihood
          )
        }
      }
      metric_name <- if (likelihood %in% c("normal", "mixed")) "nll" else "log_loss"
      metric_value <- metrics[[metric_name]]
      if (is.null(metric_value)) {
        return(NA_real_)
      }
      as.numeric(metric_value)
    }
    parse_svi_run_result <- function(run_result) {
      strategize_jax_block_until_ready(run_result)
      losses <- tryCatch({
        if (reticulate::py_has_attr(run_result, "losses")) {
          as.numeric(strenv$np$array(run_result$losses))
        } else if (!is.null(run_result$losses)) {
          as.numeric(strenv$np$array(run_result$losses))
        } else {
          NULL
        }
      }, error = function(e) NULL)
      state <- if (!is.null(run_result$state)) {
        run_result$state
      } else if (length(run_result) > 0L) {
        run_result[[1]]
      } else {
        run_result
      }
      list(state = state, losses = losses)
    }
    parse_svi_update_result <- function(update_result) {
      strategize_jax_block_until_ready(update_result)
      parts <- tryCatch(as.list(update_result), error = function(e) NULL)
      if (is.null(parts) || length(parts) < 2L) {
        parts <- list(
          tryCatch(update_result[[1L]], error = function(e) NULL),
          tryCatch(update_result[[2L]], error = function(e) NULL)
        )
      }
      state <- parts[[1L]]
      loss <- tryCatch(
        as.numeric(strenv$np$array(parts[[2L]])),
        error = function(e) {
          tryCatch(as.numeric(reticulate::py_to_r(parts[[2L]])), error = function(e2) NA_real_)
        }
      )
      loss_value <- if (length(loss) > 0L) loss[[1L]] else NA_real_
      list(state = state, loss = loss_value)
    }
    parse_svi_scan_update_result <- function(update_result) {
      strategize_jax_block_until_ready(update_result)
      parts <- tryCatch(as.list(update_result), error = function(e) NULL)
      if (is.null(parts) || length(parts) < 2L) {
        parts <- list(
          tryCatch(update_result[[1L]], error = function(e) NULL),
          tryCatch(update_result[[2L]], error = function(e) NULL)
        )
      }
      losses <- tryCatch(
        as.numeric(strenv$np$array(parts[[2L]])),
        error = function(e) {
          tryCatch(as.numeric(reticulate::py_to_r(parts[[2L]])), error = function(e2) numeric(0))
        }
      )
      list(state = parts[[1L]], losses = losses)
    }
    compact_stack_batch_arg_chunks <- function(batch_args_list) {
      if (length(batch_args_list) < 1L) {
        return(NULL)
      }
      arg_names <- names(batch_args_list[[1L]])
      if (is.null(arg_names) || any(!nzchar(arg_names))) {
        return(NULL)
      }
      out <- list()
      for (arg_name in arg_names) {
        values <- lapply(batch_args_list, function(args) args[[arg_name]])
        all_null <- all(vapply(values, is.null, logical(1)))
        if (isTRUE(all_null)) {
          next
        }
        if (any(vapply(values, is.null, logical(1)))) {
          return(NULL)
        }
        out[[arg_name]] <- strenv$jnp$stack(values, axis = 0L)
      }
      if (length(out) < 1L) {
        return(NULL)
      }
      out
    }
    format_svi_diag_value <- function(value, digits = 8L) {
      if (length(value) != 1L || !is.finite(value)) {
        return("NA")
      }
      sprintf(paste0("%.", as.integer(digits), "f"), as.numeric(value))
    }
    format_svi_grad_value <- function(value) {
      if (length(value) != 1L || !is.finite(value)) {
        return("NA")
      }
      formatC(as.numeric(value), digits = 4L, format = "fg", flag = "#")
    }
    record_svi_gradient_checkpoint <- function(svi_state_current,
                                               model_args_current,
                                               step_current) {
      if (!isTRUE(gradient_diagnostics_enabled) ||
          isTRUE(gradient_diagnostics_failed)) {
        return(NULL)
      }
      checkpoint <- tryCatch(
        neural_compute_svi_gradient_checkpoint(
          svi = svi,
          svi_state = svi_state_current,
          model_args = model_args_current
        ),
        error = function(e) {
          gradient_diagnostics <<- neural_mark_gradient_diagnostics_failed(
            gradient_diagnostics,
            conditionMessage(e)
          )
          gradient_diagnostics_failed <<- TRUE
          NULL
        }
      )
      if (is.null(checkpoint)) {
        return(NULL)
      }
      gradient_diagnostics <<- neural_append_gradient_checkpoint(
        gradient_diagnostics,
        step = step_current,
        checkpoint = checkpoint
      )
      checkpoint
    }
    format_svi_gradient_fields <- function(checkpoint) {
      if (!isTRUE(gradient_diagnostics_enabled)) {
        return("")
      }
      grad_l2 <- suppressWarnings(as.numeric(checkpoint$grad_l2 %||% NA_real_))
      grad_rms <- suppressWarnings(as.numeric(checkpoint$grad_rms %||% NA_real_))
      grad_max_abs <- suppressWarnings(as.numeric(checkpoint$grad_max_abs %||% NA_real_))
      grad_bad <- suppressWarnings(as.integer(checkpoint$grad_n_nonfinite %||% NA_integer_))
      paste0(
        "; grad_l2=", format_svi_grad_value(grad_l2),
        "; grad_rms=", format_svi_grad_value(grad_rms),
        "; grad_max=", format_svi_grad_value(grad_max_abs),
        "; grad_bad=", ifelse(length(grad_bad) == 1L && !is.na(grad_bad), as.character(grad_bad), "NA")
      )
    }
    current_process_rss_mb <- function() {
      if (!requireNamespace("ps", quietly = TRUE)) {
        return(NA_real_)
      }
      mem_info <- tryCatch(ps::ps_memory_info(), error = function(e) NULL)
      rss_bytes <- tryCatch(as.numeric(mem_info[["rss"]]), error = function(e) NA_real_)
      if (length(rss_bytes) != 1L || !is.finite(rss_bytes) || rss_bytes < 0) {
        return(NA_real_)
      }
      rss_bytes / (1024^2)
    }

    validation_split <- NULL
    early_stopping_running <- FALSE
    early_stopping_n_checks <- if (!is.null(mcmc_control$early_stopping_n_checks)) {
      as.integer(mcmc_control$early_stopping_n_checks)
    } else {
      10L
    }
    if (length(early_stopping_n_checks) != 1L ||
        is.na(early_stopping_n_checks) ||
        !is.finite(early_stopping_n_checks) ||
        early_stopping_n_checks < 1L) {
      early_stopping_n_checks <- 10L
    }
    early_stopping_patience <- if (!is.null(mcmc_control$early_stopping_patience)) {
      as.integer(mcmc_control$early_stopping_patience)
    } else {
      3L
    }
    if (length(early_stopping_patience) != 1L ||
        is.na(early_stopping_patience) ||
        !is.finite(early_stopping_patience) ||
        early_stopping_patience < 1L) {
      early_stopping_patience <- 3L
    }
    early_stopping_info$n_checks <- early_stopping_n_checks
    early_stopping_info$patience <- early_stopping_patience
    early_stopping_validation_frac <- if (!is.null(mcmc_control$early_stopping_validation_frac)) {
      as.numeric(mcmc_control$early_stopping_validation_frac)
    } else {
      0.05
    }
    if (length(early_stopping_validation_frac) != 1L ||
        is.na(early_stopping_validation_frac) ||
        !is.finite(early_stopping_validation_frac) ||
        early_stopping_validation_frac <= 0) {
      early_stopping_validation_frac <- 0.05
    }
    early_stopping_validation_frac <- min(early_stopping_validation_frac, 1)
    early_stopping_validation_max_n <- if ("early_stopping_validation_max_n" %in% names(mcmc_control)) {
      mcmc_control$early_stopping_validation_max_n
    } else {
      2048L
    }
    if (!is.null(early_stopping_validation_max_n)) {
      early_stopping_validation_max_n <- as.integer(early_stopping_validation_max_n)
      if (length(early_stopping_validation_max_n) != 1L ||
          is.na(early_stopping_validation_max_n) ||
          !is.finite(early_stopping_validation_max_n) ||
          early_stopping_validation_max_n < 1L) {
        early_stopping_validation_max_n <- 2048L
      }
    }
    early_stopping_info$validation_frac <- as.numeric(early_stopping_validation_frac)
    early_stopping_info$validation_max_n <- if (is.null(early_stopping_validation_max_n)) {
      NULL
    } else {
      as.integer(early_stopping_validation_max_n)
    }
    early_stopping_validation_batch_size <- if ("early_stopping_validation_batch_size" %in% names(mcmc_control)) {
      mcmc_control$early_stopping_validation_batch_size
    } else {
      128L
    }
    if (!is.null(early_stopping_validation_batch_size)) {
      early_stopping_validation_batch_size <- as.integer(early_stopping_validation_batch_size)
      if (length(early_stopping_validation_batch_size) != 1L ||
          is.na(early_stopping_validation_batch_size) ||
          !is.finite(early_stopping_validation_batch_size) ||
          early_stopping_validation_batch_size < 1L) {
        early_stopping_validation_batch_size <- 128L
      }
    }
    early_stopping_validation_batch_size_supplied <- !is.null(mcmc_overrides) &&
      "early_stopping_validation_batch_size" %in% names(mcmc_overrides) &&
      !is.null(mcmc_overrides$early_stopping_validation_batch_size)
    early_stopping_reason <- if (isTRUE(early_stopping_enabled)) {
      "validation_split_unavailable"
    } else {
      "disabled"
    }
    svi_train_model_args <- svi_model_args

    if (isTRUE(early_stopping_enabled)) {
      validation_split <- tryCatch(build_svi_validation_split(), error = function(e) NULL)
      if (!is.null(validation_split) &&
          reticulate::py_has_attr(svi, "init") &&
          reticulate::py_has_attr(svi, "run") &&
          reticulate::py_has_attr(svi, "get_params")) {
        early_stopping_info$active <- TRUE
        early_stopping_info$metric <- if (likelihood %in% c("normal", "mixed")) "nll" else "log_loss"
        early_stopping_info$min_delta <- if (isTRUE(compact_training)) 0 else 1e-4
        early_stopping_info$eval_every <- as.integer(max(
          1L,
          ceiling(svi_steps / early_stopping_n_checks)
        ))
        early_stopping_info$n_train <- length(validation_split$train_idx)
        early_stopping_info$n_validation <- length(validation_split$validation_idx)
        early_stopping_info$validation_batch_size <- as.integer(validation_split$validation_batch_size)
        early_stopping_info$validation_target_n <- as.integer(validation_split$validation_target_n)
        early_stopping_info$validation_prediction_mode <- validation_split$validation_prediction_mode %||%
          "single_jit_call"
        early_stopping_info$validation_n_batches <- length(validation_split$validation_batches %||% list())
        if (isTRUE(compact_training)) {
          compact_sampling_obs_idx <- as.integer(validation_split$train_idx)
        } else {
          svi_train_model_args <- c(
            svi_model_args,
            list(obs_idx = validation_split$train_idx_jnp)
          )
        }
        early_stopping_running <- TRUE
        early_stopping_reason <- "completed_budget"
      } else {
        early_stopping_reason <- if (is.null(validation_split)) {
          validation_split_reason %||% "validation_split_unavailable"
        } else {
          "api_unavailable"
        }
        early_stopping_info$n_train <- length(Y_use)
        early_stopping_info$n_validation <- 0L
        early_stopping_info$validation_batch_size <- NA_integer_
        early_stopping_info$validation_target_n <- NA_integer_
        early_stopping_info$validation_prediction_mode <- NA_character_
        early_stopping_info$validation_n_batches <- NA_integer_
      }
    } else {
      early_stopping_info$n_train <- length(Y_use)
      early_stopping_info$n_validation <- 0L
      early_stopping_info$validation_batch_size <- NA_integer_
      early_stopping_info$validation_target_n <- NA_integer_
      early_stopping_info$validation_prediction_mode <- NA_character_
      early_stopping_info$validation_n_batches <- NA_integer_
    }

    checkpoint_context <- function() {
      list(
        likelihood = likelihood,
        pairwise_mode = isTRUE(pairwise_mode),
        subsample_method = subsample_method,
        training_seed = as.integer(neural_training_seed),
        early_stopping_running = isTRUE(early_stopping_running),
        compact_training = isTRUE(compact_training),
        compact_rng_state = if (isTRUE(compact_training) &&
                                exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
          as.integer(get(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
        } else {
          NULL
        },
        compact_sampling_obs_idx = if (isTRUE(compact_training) &&
                                       !is.null(compact_sampling_obs_idx)) {
          as.integer(compact_sampling_obs_idx)
        } else {
          NULL
        }
      )
    }
    checkpoint_save <- function(type = c("latest", "best"),
                                svi_state_current = NULL,
                                svi_params_current = NULL,
                                prediction_params_current = NULL,
                                step_current = NULL,
                                loss_history_current = NULL,
                                best_metric_current = NA_real_,
                                best_step_current = NA_integer_,
                                no_improve_checks_current = 0L) {
      type <- match.arg(type)
      if (!isTRUE(svi_checkpoint$enabled)) {
        return(NULL)
      }
      if (is.null(svi_params_current) && !is.null(svi_state_current)) {
        svi_params_current <- tryCatch(svi$get_params(svi_state_current), error = function(e) NULL)
      }
      if (is.null(svi_params_current)) {
        return(NULL)
      }
      if (is.null(prediction_params_current)) {
        prediction_params_current <- tryCatch(
          extract_svi_param_sites(svi_params_current),
          error = function(e) NULL
        )
      }
      payload <- neural_svi_checkpoint_make_payload(
        snapshot_type = type,
        fingerprint = svi_checkpoint_fingerprint,
        completed_step = as.integer(step_current %||% svi_steps_completed %||% 0L),
        resolved_svi_steps = resolved_svi_steps,
        svi_params = svi_params_current,
        prediction_params = prediction_params_current,
        loss_history = loss_history_current %||% svi_loss_curve %||% numeric(0),
        validation_history = early_stopping_info$validation_loss_history %||% numeric(0),
        best_metric = best_metric_current,
        best_step = best_step_current,
        no_improve_checks = no_improve_checks_current,
        early_stopping = early_stopping_info,
        optimizer_diagnostics = optimizer_diagnostics,
        svi_budget_info = svi_budget_info,
        checkpoint_context = checkpoint_context()
      )
      neural_svi_checkpoint_save_snapshot(
        svi_checkpoint$path,
        type = type,
        payload = payload,
        compress = isTRUE(svi_checkpoint$compress)
      )
      payload
    }
    checkpoint_run_chunk <- function(chunk_steps,
                                     model_args_current,
                                     init_state_current = NULL,
                                     init_params_current = NULL,
                                     progress_bar = FALSE) {
      run_args <- list(
        rng_key,
        ai(chunk_steps),
        progress_bar = isTRUE(progress_bar)
      )
      if (!is.null(init_state_current)) {
        run_args$init_state <- init_state_current
      } else if (!is.null(init_params_current)) {
        run_args$init_params <- init_params_current
      }
      do.call(svi$run, c(run_args, model_args_current))
    }
    checkpoint_prediction_params_to_draws <- function(prediction_params) {
      if (is.null(prediction_params)) {
        return(NULL)
      }
      params_jax <- neural_svi_checkpoint_params_to_jax(prediction_params)
      params_list <- neural_svi_checkpoint_params_to_list(params_jax)
      if (is.null(params_list) || length(params_list) < 1L) {
        return(NULL)
      }
      out <- lapply(params_list, function(x) {
        strenv$jnp$expand_dims(strenv$jnp$expand_dims(x, 0L), 0L)
      })
      names(out) <- names(params_list)
      out
    }

    checkpoint_resume_params <- NULL
    checkpoint_resume_completed <- 0L
    checkpoint_training_complete <- FALSE
    checkpoint_final_snapshot <- NULL
    if (isTRUE(svi_checkpoint$enabled) && !is.null(svi_checkpoint_latest)) {
      checkpoint_resume_completed <- as.integer(svi_checkpoint_latest$completed_step %||% 0L)
      if (is.na(checkpoint_resume_completed) || checkpoint_resume_completed < 0L) {
        checkpoint_resume_completed <- 0L
      }
      checkpoint_resume_params <- neural_svi_checkpoint_params_to_jax(
        svi_checkpoint_latest$svi_params
      )
      if (!is.null(svi_checkpoint_latest$early_stopping)) {
        early_stopping_info <- modifyList(
          early_stopping_info,
          svi_checkpoint_latest$early_stopping
        )
      }
      if (!is.null(svi_checkpoint_latest$validation_history)) {
        early_stopping_info$validation_loss_history <- as.numeric(
          svi_checkpoint_latest$validation_history
        )
      }
      checkpoint_training_complete <- checkpoint_resume_completed >= as.integer(svi_steps) ||
        isTRUE(svi_checkpoint_latest$early_stopping$stopped_early)
      # A partial-checkpoint resume re-initializes the optimizer from saved params
      # only (svi$init below). The LR schedule now CONTINUES from the completed
      # step (offset applied where lr_schedule is built), so there is no
      # re-warmup LR shock; optax/Muon momentum buffers are still discarded and
      # rebuild over the first few steps. Full continuity would require
      # persisting and restoring the optax optim_state in the checkpoint;
      # until then, surface the remaining discontinuity rather than letting it
      # be silent.
      if (checkpoint_resume_completed > 0L && !isTRUE(checkpoint_training_complete)) {
        warning(sprintf(
          paste0(
            "Neural SVI resume from step %d/%d: the LR schedule continues from ",
            "the completed step, but Muon/optax momentum is discarded and ",
            "rebuilds, so the resumed run is not bitwise identical to an ",
            "uninterrupted fit."
          ),
          as.integer(checkpoint_resume_completed), as.integer(svi_steps)
        ), call. = FALSE)
      }
      if (isTRUE(checkpoint_training_complete)) {
        checkpoint_final_snapshot <- if (!is.null(svi_checkpoint_best) &&
                                         is.finite(svi_checkpoint_best$best_metric %||% NA_real_)) {
          svi_checkpoint_best
        } else {
          svi_checkpoint_latest
        }
        SVIParams <- neural_svi_checkpoint_params_to_jax(checkpoint_final_snapshot$svi_params)
        SVIPosteriorDraws <- checkpoint_prediction_params_to_draws(
          checkpoint_final_snapshot$prediction_params
        )
        svi_loss_curve <- as.numeric(svi_checkpoint_latest$loss_history %||% numeric(0))
        svi_steps_completed <- as.integer(checkpoint_resume_completed)
        early_stopping_info$reason <- early_stopping_info$reason %||% early_stopping_reason
        early_stopping_info$stop_step <- as.integer(svi_steps_completed)
        message(sprintf(
          "Neural SVI checkpoint already reached step %d/%d; rebuilding final fit from saved parameters.",
          as.integer(svi_steps_completed),
          as.integer(svi_steps)
        ))
      }
    }

    if (!isTRUE(checkpoint_training_complete) && isTRUE(compact_training)) {
      if (!reticulate::py_has_attr(svi, "init") ||
          !reticulate::py_has_attr(svi, "update") ||
          !reticulate::py_has_attr(svi, "get_params")) {
        stop("Compact streaming SVI requires numpyro SVI init/update/get_params APIs.", call. = FALSE)
      }
      if (compact_model_n_obs < 1L || compact_svi_batch_size < 1L) {
        stop("Compact streaming SVI received no training observations.", call. = FALSE)
      }
      message(sprintf(
        "Running compact streaming SVI: observations=%d, batch_size=%d, steps=%d, update_chunk_size=%d.",
        as.integer(compact_model_n_obs),
        as.integer(compact_svi_batch_size),
        as.integer(svi_steps),
        as.integer(compact_update_chunk_size)
      ))
      compact_seed <- eval_control$seed
      if (length(compact_seed) != 1L || is.na(compact_seed) || !is.finite(compact_seed)) {
        compact_seed <- 123L
      }
      old_seed_state <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
      } else {
        NULL
      }
      on.exit({
        if (is.null(old_seed_state)) {
          if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
            rm(".Random.seed", envir = .GlobalEnv)
          }
        } else {
          assign(".Random.seed", old_seed_state, envir = .GlobalEnv)
        }
      }, add = TRUE)
      set.seed(as.integer(compact_seed))
      compact_saved_rng_state <- tryCatch(
        as.integer(svi_checkpoint_latest$checkpoint_context$compact_rng_state),
        error = function(e) NULL
      )
      if (!is.null(compact_saved_rng_state) && length(compact_saved_rng_state) > 1L) {
        assign(".Random.seed", compact_saved_rng_state, envir = .GlobalEnv)
      }
      if (checkpoint_resume_completed > 0L && is.null(checkpoint_resume_params)) {
        checkpoint_resume_completed <- 0L
      }
      init_idx <- compact_sample_obs_idx()
      init_args <- c(list(rng_key), compact_batch_args(init_idx))
      if (!is.null(checkpoint_resume_params)) {
        init_args$init_params <- checkpoint_resume_params
      }
      svi_state <- do.call(svi$init, init_args)
      checkpoint_resume_params <- NULL
      if (!is.null(compact_saved_rng_state) && length(compact_saved_rng_state) > 1L) {
        assign(".Random.seed", compact_saved_rng_state, envir = .GlobalEnv)
      }
      svi_loss_curve <- rep(NA_real_, as.integer(svi_steps))
      saved_loss_history <- as.numeric(svi_checkpoint_latest$loss_history %||% numeric(0))
      if (length(saved_loss_history) > 0L) {
        saved_n <- min(length(saved_loss_history), length(svi_loss_curve))
        svi_loss_curve[seq_len(saved_n)] <- saved_loss_history[seq_len(saved_n)]
      }
      svi_steps_completed <- as.integer(checkpoint_resume_completed)
      compact_validation_active <- isTRUE(early_stopping_running)
      compact_validation_eval_every <- neural_resolve_positive_int(
        early_stopping_info$eval_every,
        ceiling(as.integer(svi_steps) / early_stopping_n_checks)
      )
      compact_checkpoint_n_checks <- neural_resolve_positive_int(
        svi_checkpoint$n_checks %||% 10L,
        10L
      )
      compact_checkpoint_eval_every <- neural_resolve_positive_int(
        ceiling(as.integer(svi_steps) / compact_checkpoint_n_checks),
        1L
      )
      progress_every <- if (isTRUE(compact_validation_active)) {
        compact_validation_eval_every
      } else {
        compact_checkpoint_eval_every
      }
      last_gradient_checkpoint <- NULL
      last_progress_step <- as.integer(svi_steps_completed)
      last_progress_elapsed <- as.numeric(difftime(Sys.time(), t0_, units = "secs"))
      window_steps <- 0L
      window_chunks <- 0L
      window_update_elapsed_s <- 0
      window_sampled_train_obs <- 0
      compact_scan_error <- NULL
      compact_jit_error <- NULL
      compact_scan_message_emitted <- FALSE
      compact_scan_status <- if (compact_update_chunk_size > 1L) "pending" else "single_step"
      compact_scan_required <- compact_update_chunk_size > 1L &&
        identical(compact_update_scan, "required")
      compact_update_chunk_size_effective <- 1L
      compact_update_jit_path <- if (compact_update_chunk_size > 1L) "pending" else "single"
      compact_update_jit_status <- "pending"
      compact_update_jit_cache_size <- NA_integer_
      compact_update_jit_compile_count <- NA_integer_
      saved_attempted_steps <- max(0L, min(as.integer(svi_steps_completed), length(svi_loss_curve)))
      saved_attempted_losses <- if (saved_attempted_steps > 0L) {
        as.numeric(svi_loss_curve[seq_len(saved_attempted_steps)])
      } else {
        numeric(0)
      }
      saved_nonfinite <- if (length(saved_attempted_losses) > 0L) {
        !is.finite(saved_attempted_losses)
      } else {
        logical(0)
      }
      compact_update_nonfinite_steps <- which(saved_nonfinite)
      compact_update_nonfinite_losses <- saved_attempted_losses[saved_nonfinite]
      compact_update_nonfinite_paths <- rep("resume", length(compact_update_nonfinite_steps))
      compact_update_committed_steps <- as.integer(sum(is.finite(saved_attempted_losses)))
      compact_update_skipped_steps <- as.integer(length(compact_update_nonfinite_steps))
      compact_update_stable_available <- FALSE
      refresh_compact_skip_diagnostics <- function() {
        n_nonfinite <- as.integer(length(compact_update_nonfinite_steps))
        optimizer_diagnostics$compact_update_stable_available <<- isTRUE(compact_update_stable_available)
        optimizer_diagnostics$compact_update_nonfinite_count <<- n_nonfinite
        optimizer_diagnostics$compact_update_nonfinite_steps <<- as.integer(compact_update_nonfinite_steps)
        optimizer_diagnostics$compact_update_nonfinite_losses <<- as.numeric(compact_update_nonfinite_losses)
        optimizer_diagnostics$compact_update_nonfinite_path <<- as.character(compact_update_nonfinite_paths)
        optimizer_diagnostics$compact_update_committed_steps <<- as.integer(compact_update_committed_steps)
        optimizer_diagnostics$compact_update_skipped_steps <<- as.integer(compact_update_skipped_steps)
        optimizer_diagnostics$compact_update_attempted_steps <<- as.integer(svi_steps_completed)
        early_stopping_info$compact_update_nonfinite_count <<- n_nonfinite
        early_stopping_info$compact_update_nonfinite_steps <<- as.integer(compact_update_nonfinite_steps)
        early_stopping_info$compact_update_skipped_steps <<- as.integer(compact_update_skipped_steps)
        invisible(NULL)
      }
      record_compact_update_attempt <- function(step_range,
                                                losses,
                                                path,
                                                committed_steps) {
        losses <- as.numeric(losses)
        step_range <- as.integer(step_range)
        finite_loss <- is.finite(losses)
        if (length(losses) > 0L && any(!finite_loss)) {
          bad_idx <- which(!finite_loss)
          compact_update_nonfinite_steps <<- c(
            as.integer(compact_update_nonfinite_steps),
            as.integer(step_range[bad_idx])
          )
          compact_update_nonfinite_losses <<- c(
            as.numeric(compact_update_nonfinite_losses),
            as.numeric(losses[bad_idx])
          )
          compact_update_nonfinite_paths <<- c(
            as.character(compact_update_nonfinite_paths),
            rep(as.character(path), length(bad_idx))
          )
          compact_update_skipped_steps <<- as.integer(compact_update_skipped_steps + length(bad_idx))
        }
        committed_steps <- suppressWarnings(as.integer(committed_steps))
        if (length(committed_steps) != 1L || is.na(committed_steps) || committed_steps < 0L) {
          committed_steps <- 0L
        }
        compact_update_committed_steps <<- as.integer(compact_update_committed_steps + committed_steps)
        refresh_compact_skip_diagnostics()
        invisible(NULL)
      }
      read_compact_jit_cache_info <- function() {
        info <- tryCatch(strenv$jax_svi_update_jit_cache_info(), error = function(e) NULL)
        if (is.null(info)) {
          return(list(size = NA_integer_, compile_count = NA_integer_))
        }
        info <- tryCatch(as.list(info), error = function(e) info)
        list(
          size = suppressWarnings(as.integer(info$size %||% NA_integer_)),
          compile_count = suppressWarnings(as.integer(info$compile_count %||% NA_integer_))
        )
      }
      refresh_compact_jit_diagnostics <- function() {
        cache_info <- read_compact_jit_cache_info()
        compact_update_jit_cache_size <<- cache_info$size
        compact_update_jit_compile_count <<- cache_info$compile_count
        optimizer_diagnostics$compact_update_jit_required <<- TRUE
        optimizer_diagnostics$compact_update_jit_status <<- compact_update_jit_status
        optimizer_diagnostics$compact_update_jit_path <<- compact_update_jit_path
        optimizer_diagnostics$compact_update_jit_cache_size <<- compact_update_jit_cache_size
        optimizer_diagnostics$compact_update_jit_compile_count <<- compact_update_jit_compile_count
        optimizer_diagnostics$compact_update_jit_error <<- compact_jit_error
        invisible(NULL)
      }
      compact_jit_available <- tryCatch({
        strategize_register_jax_svi_helpers()
        !is.null(strenv$jax_svi_update)
      }, error = function(e) {
        compact_jit_error <<- conditionMessage(e)
        FALSE
      })
      compact_update_stable_available <- tryCatch(
        isTRUE(strenv$jax_svi_update_stable_available(svi)),
        error = function(e) FALSE
      )
      compact_scan_available <- compact_update_chunk_size > 1L && !is.null(strenv$jax_svi_update_scan)
      if (!isTRUE(compact_jit_available)) {
        compact_update_jit_status <- "unavailable"
        refresh_compact_jit_diagnostics()
        neural_stop_compact_jit_required(compact_jit_error %||% "JAX SVI jitted update helper is unavailable")
      }
      compact_update_jit_status <- "ok"
      if (compact_update_chunk_size > 1L && !isTRUE(compact_scan_available)) {
        compact_scan_status <- if (isTRUE(compact_scan_required)) {
          "helper_unavailable"
        } else {
          "fallback_single_step"
        }
        compact_update_jit_path <- if (isTRUE(compact_scan_required)) {
          "scan"
        } else {
          "scan_to_single_fallback"
        }
        compact_scan_error <- compact_scan_error %||% "JAX SVI jitted scan helper is unavailable"
        if (isTRUE(compact_scan_required)) {
          refresh_compact_jit_diagnostics()
          neural_stop_compact_scan_required(compact_scan_error)
        }
      }
      compact_svi_steps_target <- neural_compact_effective_scan_steps(
        svi_steps = svi_steps,
        completed_steps = svi_steps_completed,
        chunk_size = compact_update_chunk_size,
        scan_available = isTRUE(compact_scan_available)
      )
      if (compact_svi_steps_target > length(svi_loss_curve)) {
        svi_loss_curve <- c(
          svi_loss_curve,
          rep(NA_real_, as.integer(compact_svi_steps_target - length(svi_loss_curve)))
        )
      }
      optimizer_diagnostics$compact_update_chunk_size_requested <- as.integer(compact_update_chunk_size)
      optimizer_diagnostics$compact_update_chunk_size_effective <- as.integer(compact_update_chunk_size_effective)
      optimizer_diagnostics$compact_update_steps_requested <- as.integer(svi_steps)
      optimizer_diagnostics$compact_update_steps_effective <- as.integer(compact_svi_steps_target)
      optimizer_diagnostics$compact_update_steps_overrun <- as.integer(
        max(0L, as.integer(compact_svi_steps_target) - as.integer(svi_steps))
      )
      optimizer_diagnostics$compact_update_scan_mode <- compact_update_scan
      optimizer_diagnostics$compact_update_scan_status <- compact_scan_status
      optimizer_diagnostics$compact_update_scan_error <- compact_scan_error
      refresh_compact_skip_diagnostics()
      refresh_compact_jit_diagnostics()

      best_metric <- if (!is.null(svi_checkpoint_best) &&
                         is.finite(svi_checkpoint_best$best_metric %||% NA_real_)) {
        as.numeric(svi_checkpoint_best$best_metric)
      } else if (is.finite(early_stopping_info$best_metric %||% NA_real_)) {
        as.numeric(early_stopping_info$best_metric)
      } else {
        Inf
      }
      if (is.finite(best_metric)) {
        early_stopping_info$best_metric <- best_metric
        if (!is.null(svi_checkpoint_best)) {
          early_stopping_info$best_step <- as.integer(
            svi_checkpoint_best$best_step %||% early_stopping_info$best_step
          )
        }
      }
      no_improve_checks <- as.integer(svi_checkpoint_latest$no_improve_checks %||% 0L)
      if (is.na(no_improve_checks) || no_improve_checks < 0L) {
        no_improve_checks <- 0L
      }
      compact_best_svi_state <- NULL
      compact_best_svi_params <- NULL
      compact_validation_errors <- 0L
      compact_metric_failures <- 0L
      compact_last_validation_step <- if (length(early_stopping_info$validation_loss_history) > 0L) {
        as.integer(early_stopping_info$stop_step %||% checkpoint_resume_completed)
      } else {
        NA_integer_
      }
      compact_last_checkpoint_step <- if (!is.null(svi_checkpoint_latest)) {
        as.integer(checkpoint_resume_completed)
      } else {
        NA_integer_
      }
      compact_next_validation_step <- compact_validation_eval_every *
        (length(early_stopping_info$validation_loss_history %||% numeric(0)) + 1L)
      while (compact_next_validation_step <= svi_steps_completed) {
        compact_next_validation_step <- compact_next_validation_step + compact_validation_eval_every
      }
      compact_next_checkpoint_step <- compact_checkpoint_eval_every *
        (as.integer(floor(svi_steps_completed / compact_checkpoint_eval_every)) + 1L)
      current_compact_loss_history <- function() {
        if (svi_steps_completed > 0L) {
          return(as.numeric(svi_loss_curve[seq_len(svi_steps_completed)]))
        }
        numeric(0)
      }
      advance_compact_target <- function(next_step, completed_step, every) {
        while (next_step <= completed_step) {
          next_step <- next_step + every
        }
        next_step
      }
      compact_throughput_rate <- function(count, elapsed_s) {
        count <- as.numeric(count)
        elapsed_s <- as.numeric(elapsed_s)
        if (length(count) != 1L || length(elapsed_s) != 1L ||
            !is.finite(count) || !is.finite(elapsed_s) || count < 0 || elapsed_s <= 0) {
          return(NA_real_)
        }
        count / elapsed_s
      }
      compact_validation_check <- function(batch_args_current,
                                           chunk_steps,
                                           chunk_elapsed_s,
                                           chunk_step_per_s) {
        validation_metric_error <- NULL
        validation_started_at <- proc.time()[["elapsed"]]
        metric_value <- tryCatch(
          compute_svi_validation_metric(svi_state, validation_split),
          error = function(e) {
            validation_metric_error <<- conditionMessage(e)
            NA_real_
          }
        )
        strategize_jax_block_until_ready(metric_value)
        validation_elapsed_s <- as.numeric(proc.time()[["elapsed"]] - validation_started_at)
        validation_obs <- length(validation_split$validation_idx)
        validation_obs_per_s <- compact_throughput_rate(validation_obs, validation_elapsed_s)
        previous_metric <- if (length(early_stopping_info$validation_loss_history) > 0L) {
          tail(early_stopping_info$validation_loss_history, 1L)
        } else {
          NA_real_
        }
        early_stopping_info$validation_loss_history <<- c(
          early_stopping_info$validation_loss_history,
          as.numeric(metric_value)
        )
        early_stopping_info$stop_check <<- as.integer(length(early_stopping_info$validation_loss_history))
        early_stopping_info$stop_step <<- as.integer(svi_steps_completed)

        improved_metric <- FALSE
        if (!is.null(validation_metric_error)) {
          compact_validation_errors <<- compact_validation_errors + 1L
          early_stopping_info$error_message <<- validation_metric_error
          early_stopping_info$reason <<- "validation_error"
        } else if (!is.finite(metric_value)) {
          compact_metric_failures <<- compact_metric_failures + 1L
          early_stopping_info$reason <<- "metric_failed"
        } else {
          improved_metric <- !is.finite(best_metric) || metric_value < best_metric
          if (isTRUE(improved_metric)) {
            best_metric <<- metric_value
            compact_best_svi_state <<- svi_state
            compact_best_svi_params <<- tryCatch(svi$get_params(svi_state), error = function(e) NULL)
            early_stopping_info$best_step <<- as.integer(svi_steps_completed)
            early_stopping_info$best_metric <<- metric_value
            no_improve_checks <<- 0L
          } else {
            no_improve_checks <<- no_improve_checks + 1L
          }
          early_stopping_info$reason <<- "completed_budget"
        }

        gradient_checkpoint <- record_svi_gradient_checkpoint(
          svi_state_current = svi_state,
          model_args_current = batch_args_current,
          step_current = svi_steps_completed
        )
        last_gradient_checkpoint <<- gradient_checkpoint
        train_elbo_value <- if (svi_steps_completed > 0L) {
          svi_loss_curve[[svi_steps_completed]]
        } else {
          NA_real_
        }
        rss_mb_value <- current_process_rss_mb()
        elapsed_seconds <- as.numeric(difftime(Sys.time(), t0_, units = "secs"))
        best_metric_for_message <- if (is.finite(early_stopping_info$best_metric)) {
          early_stopping_info$best_metric
        } else {
          metric_value
        }
        best_step_for_message <- if (!is.na(early_stopping_info$best_step)) {
          as.integer(early_stopping_info$best_step)
        } else {
          as.integer(svi_steps_completed)
        }
        delta_prev <- if (is.finite(previous_metric) && is.finite(metric_value)) {
          metric_value - previous_metric
        } else {
          NA_real_
        }
        delta_prev_text <- if (is.finite(delta_prev)) {
          sprintf("%+.6f", delta_prev)
        } else {
          "NA"
        }
        total_checks_planned <- max(
          1L,
          as.integer(length(neural_compact_chunk_boundary_checks(
            svi_steps = svi_steps,
            n_checks = early_stopping_n_checks,
            chunk_size = compact_update_chunk_size_effective
          )))
        )
        message(sprintf(
          paste0(
            "Compact SVI validation check %d/%d: step=%d/%d; validation %s=%s; ",
            "train_elbo=%s; best=%s at step %d; delta_prev=%s; ",
            "chunk_steps=%d; chunk_elapsed_s=%s; chunk_step_per_s=%s; ",
            "validation_obs=%d; validation_elapsed_s=%s; validation_obs_per_s=%s; ",
            "rss_mb=%s; elapsed=%ss%s."
          ),
          as.integer(early_stopping_info$stop_check),
          total_checks_planned,
          as.integer(svi_steps_completed),
          as.integer(svi_steps),
          early_stopping_info$metric,
          format_svi_diag_value(metric_value, digits = 6L),
          format_svi_diag_value(train_elbo_value, digits = 2L),
          format_svi_diag_value(best_metric_for_message, digits = 6L),
          best_step_for_message,
          delta_prev_text,
          as.integer(chunk_steps),
          format_svi_diag_value(chunk_elapsed_s, digits = 3L),
          format_svi_diag_value(chunk_step_per_s, digits = 3L),
          as.integer(validation_obs),
          format_svi_diag_value(validation_elapsed_s, digits = 3L),
          format_svi_diag_value(validation_obs_per_s, digits = 1L),
          format_svi_diag_value(rss_mb_value, digits = 1L),
          format_svi_diag_value(elapsed_seconds, digits = 3L),
          format_svi_gradient_fields(gradient_checkpoint)
        ))

        if (isTRUE(improved_metric)) {
          checkpoint_save(
            type = "best",
            svi_state_current = svi_state,
            svi_params_current = compact_best_svi_params,
            step_current = svi_steps_completed,
            loss_history_current = current_compact_loss_history(),
            best_metric_current = best_metric,
            best_step_current = early_stopping_info$best_step,
            no_improve_checks_current = no_improve_checks
          )
        }
        checkpoint_save(
          type = "latest",
          svi_state_current = svi_state,
          step_current = svi_steps_completed,
          loss_history_current = current_compact_loss_history(),
          best_metric_current = best_metric,
          best_step_current = early_stopping_info$best_step,
          no_improve_checks_current = no_improve_checks
        )
        compact_last_validation_step <<- as.integer(svi_steps_completed)
        invisible(metric_value)
      }
      compact_latest_checkpoint <- function(batch_args_current,
                                            chunk_steps,
                                            chunk_elapsed_s,
                                            chunk_step_per_s) {
        gradient_checkpoint <- record_svi_gradient_checkpoint(
          svi_state_current = svi_state,
          model_args_current = batch_args_current,
          step_current = svi_steps_completed
        )
        last_gradient_checkpoint <<- gradient_checkpoint
        message(sprintf(
          "Compact SVI checkpoint: step=%d/%d; train_elbo=%s; chunk_steps=%d; chunk_elapsed_s=%s; chunk_step_per_s=%s%s.",
          as.integer(svi_steps_completed),
          as.integer(svi_steps),
          format_svi_diag_value(svi_loss_curve[[svi_steps_completed]], digits = 2L),
          as.integer(chunk_steps),
          format_svi_diag_value(chunk_elapsed_s, digits = 3L),
          format_svi_diag_value(chunk_step_per_s, digits = 3L),
          format_svi_gradient_fields(gradient_checkpoint)
        ))
        checkpoint_save(
          type = "latest",
          svi_state_current = svi_state,
          step_current = svi_steps_completed,
          loss_history_current = current_compact_loss_history(),
          best_metric_current = NA_real_,
          best_step_current = NA_integer_,
          no_improve_checks_current = 0L
        )
        compact_last_checkpoint_step <<- as.integer(svi_steps_completed)
        invisible(NULL)
      }

      step_cursor <- as.integer(svi_steps_completed) + 1L
      while (step_cursor <= as.integer(compact_svi_steps_target)) {
        batch_args <- NULL
        chunk_completed <- FALSE
        chunk_n <- min(
          as.integer(compact_update_chunk_size),
          as.integer(compact_svi_steps_target) - as.integer(step_cursor) + 1L
        )
        chunk_start_step <- as.integer(svi_steps_completed)
        chunk_started_at <- proc.time()[["elapsed"]]
        if (chunk_n > 1L && isTRUE(compact_scan_available)) {
          batch_args_list <- tryCatch(
            lapply(seq_len(chunk_n), function(i) compact_batch_args(compact_sample_obs_idx())),
            error = function(e) {
              compact_scan_error <<- conditionMessage(e)
              NULL
            }
          )
          stacked_batch_args <- if (!is.null(batch_args_list)) {
            tryCatch(
              compact_stack_batch_arg_chunks(batch_args_list),
              error = function(e) {
                compact_scan_error <<- conditionMessage(e)
                NULL
              }
            )
          } else {
            NULL
          }
          if (!is.null(stacked_batch_args)) {
            scan_result <- tryCatch(
              strenv$jax_svi_update_scan(svi, svi_state, stacked_batch_args),
              error = function(e) {
                compact_scan_error <<- conditionMessage(e)
                compact_update_jit_status <<- "scan_failed"
                NULL
              }
            )
            if (!is.null(scan_result)) {
              strategize_jax_block_until_ready(scan_result)
              scan_parts <- parse_svi_scan_update_result(scan_result)
              if (!is.null(scan_parts$state)) {
                losses <- as.numeric(scan_parts$losses)
                if (length(losses) < chunk_n) {
                  losses <- c(losses, rep(NA_real_, chunk_n - length(losses)))
                }
                losses <- losses[seq_len(chunk_n)]
                step_range <- seq.int(step_cursor, length.out = chunk_n)
                finite_losses <- is.finite(losses)
                has_nonfinite_loss <- any(!finite_losses)
                scan_state_safe <- !isTRUE(has_nonfinite_loss) || isTRUE(compact_update_stable_available)
                if (isTRUE(scan_state_safe)) {
                  svi_state <- scan_parts$state
                }
                svi_loss_curve[step_range] <- losses
                svi_steps_completed <- as.integer(tail(step_range, 1L))
                record_compact_update_attempt(
                  step_range = step_range,
                  losses = losses,
                  path = "scan",
                  committed_steps = if (isTRUE(scan_state_safe)) sum(finite_losses) else 0L
                )
                batch_args <- batch_args_list[[length(batch_args_list)]]
                compact_scan_status <- "ok"
                compact_update_jit_status <- if (isTRUE(has_nonfinite_loss)) {
                  "ok_with_skipped_updates"
                } else {
                  "ok"
                }
                compact_update_jit_path <- "scan"
                compact_update_chunk_size_effective <- max(
                  as.integer(compact_update_chunk_size_effective),
                  as.integer(chunk_n)
                )
                chunk_completed <- TRUE
              } else if (is.null(compact_scan_error)) {
                compact_scan_error <- "scanned SVI update returned no state"
              }
            }
          } else if (is.null(compact_scan_error)) {
            compact_scan_error <- "could not stack compact mini-batch arguments"
          }
          if (!isTRUE(chunk_completed)) {
            if (isTRUE(compact_scan_required)) {
              neural_stop_compact_scan_required(compact_scan_error)
            }
            compact_scan_available <- FALSE
            compact_scan_status <- "fallback_single_step"
            compact_update_jit_path <- "scan_to_single_fallback"
            compact_update_chunk_size_effective <- 1L
            compact_svi_steps_target <- as.integer(svi_steps)
            if (!isTRUE(compact_scan_message_emitted)) {
              message(sprintf(
                "Compact SVI scanned updates unavailable; falling back to single-step updates%s.",
                if (!is.null(compact_scan_error) && nzchar(compact_scan_error)) {
                  paste0(" (", compact_scan_error, ")")
                } else {
                  ""
                }
              ))
              compact_scan_message_emitted <- TRUE
            }
          }
        } else if (chunk_n > 1L && !isTRUE(compact_scan_available) &&
                   isTRUE(compact_scan_required)) {
          neural_stop_compact_scan_required(compact_scan_error %||% "scanned updates are unavailable")
        } else if (chunk_n <= 1L && identical(compact_scan_status, "pending")) {
          compact_scan_status <- "single_step"
        }
        if (!isTRUE(chunk_completed)) {
          obs_idx <- compact_sample_obs_idx()
          batch_args <- compact_batch_args(obs_idx)
          update_result <- tryCatch(
            strenv$jax_svi_update(svi, svi_state, batch_args),
            error = function(e) {
              compact_jit_error <<- conditionMessage(e)
              compact_update_jit_status <<- "single_failed"
              NULL
            }
          )
          if (is.null(update_result)) {
            refresh_compact_jit_diagnostics()
            neural_stop_compact_jit_required(compact_jit_error)
          }
          strategize_jax_block_until_ready(update_result)
          update_parts <- parse_svi_update_result(update_result)
          if (is.null(update_parts$state)) {
            early_stopping_reason <- "update_failed"
            break
          }
          update_loss <- as.numeric(update_parts$loss)
          if (length(update_loss) != 1L) {
            update_loss <- NA_real_
          }
          single_state_safe <- is.finite(update_loss) || isTRUE(compact_update_stable_available)
          if (isTRUE(single_state_safe)) {
            svi_state <- update_parts$state
          }
          svi_loss_curve[[step_cursor]] <- update_loss
          svi_steps_completed <- as.integer(step_cursor)
          record_compact_update_attempt(
            step_range = step_cursor,
            losses = update_loss,
            path = "single",
            committed_steps = if (is.finite(update_loss)) 1L else 0L
          )
          compact_update_jit_status <- if (is.finite(update_loss)) {
            "ok"
          } else {
            "ok_with_skipped_updates"
          }
          compact_update_jit_path <- if (identical(compact_scan_status, "fallback_single_step") &&
                                          compact_update_chunk_size > 1L) {
            "scan_to_single_fallback"
          } else {
            "single"
          }
        }
        chunk_elapsed_s <- as.numeric(proc.time()[["elapsed"]] - chunk_started_at)
        chunk_steps <- as.integer(svi_steps_completed) - as.integer(chunk_start_step)
        if (length(chunk_steps) != 1L || is.na(chunk_steps) || chunk_steps < 0L) {
          chunk_steps <- 0L
        }
        chunk_step_per_s <- compact_throughput_rate(chunk_steps, chunk_elapsed_s)
        if (chunk_steps > 0L) {
          window_steps <- as.integer(window_steps + chunk_steps)
          window_chunks <- as.integer(window_chunks + 1L)
          window_update_elapsed_s <- as.numeric(window_update_elapsed_s) + as.numeric(chunk_elapsed_s)
          window_sampled_train_obs <- as.numeric(window_sampled_train_obs) +
            as.numeric(chunk_steps) * as.numeric(compact_svi_batch_size)
        }
        optimizer_diagnostics$compact_update_chunk_size_effective <- as.integer(compact_update_chunk_size_effective)
        optimizer_diagnostics$compact_update_steps_effective <- as.integer(compact_svi_steps_target)
        optimizer_diagnostics$compact_update_steps_overrun <- as.integer(
          max(0L, as.integer(compact_svi_steps_target) - as.integer(svi_steps))
        )
        optimizer_diagnostics$compact_update_scan_status <- compact_scan_status
        optimizer_diagnostics$compact_update_scan_error <- compact_scan_error
        refresh_compact_jit_diagnostics()
        validation_due <- isTRUE(compact_validation_active) &&
          (svi_steps_completed >= compact_next_validation_step ||
             svi_steps_completed == as.integer(svi_steps)) &&
          !identical(as.integer(compact_last_validation_step), as.integer(svi_steps_completed))
        if (isTRUE(validation_due)) {
          compact_validation_check(batch_args, chunk_steps, chunk_elapsed_s, chunk_step_per_s)
          compact_next_validation_step <- advance_compact_target(
            compact_next_validation_step,
            svi_steps_completed,
            compact_validation_eval_every
          )
          # Enforce early-stopping patience (previously only the non-compact
          # path stopped; compact runs always consumed the full step budget).
          # Matches the non-compact semantics: mark stopped_early, persist a
          # final latest checkpoint so resume treats the run as complete, and
          # break; best-so-far params are restored by the post-loop block.
          if (no_improve_checks >= early_stopping_info$patience &&
              svi_steps_completed < as.integer(svi_steps) &&
              is.finite(best_metric)) {
            early_stopping_info$stopped_early <- TRUE
            early_stopping_reason <- "patience_exhausted"
            early_stopping_info$reason <- early_stopping_reason
            early_stopping_info$stop_step <- as.integer(svi_steps_completed)
            checkpoint_save(
              type = "latest",
              svi_state_current = svi_state,
              step_current = svi_steps_completed,
              loss_history_current = current_compact_loss_history(),
              best_metric_current = best_metric,
              best_step_current = early_stopping_info$best_step,
              no_improve_checks_current = no_improve_checks
            )
            message(sprintf(
              "Compact SVI early stopping at step %d/%d: no validation improvement in %d consecutive checks (patience=%d).",
              as.integer(svi_steps_completed),
              as.integer(svi_steps),
              as.integer(no_improve_checks),
              as.integer(early_stopping_info$patience)
            ))
            break
          }
        }
        checkpoint_due <- !isTRUE(compact_validation_active) &&
          isTRUE(svi_checkpoint$enabled) &&
          (svi_steps_completed >= compact_next_checkpoint_step ||
             svi_steps_completed == as.integer(svi_steps)) &&
          !identical(as.integer(compact_last_checkpoint_step), as.integer(svi_steps_completed))
        if (isTRUE(checkpoint_due)) {
          compact_latest_checkpoint(batch_args, chunk_steps, chunk_elapsed_s, chunk_step_per_s)
          compact_next_checkpoint_step <- advance_compact_target(
            compact_next_checkpoint_step,
            svi_steps_completed,
            compact_checkpoint_eval_every
          )
        }
        progress_due <- window_steps > 0L &&
          (as.integer(svi_steps_completed) - as.integer(last_progress_step) >= as.integer(progress_every) ||
             svi_steps_completed == as.integer(svi_steps))
        if (isTRUE(progress_due)) {
          if (!isTRUE(validation_due) && !isTRUE(checkpoint_due)) {
            last_gradient_checkpoint <- record_svi_gradient_checkpoint(
              svi_state_current = svi_state,
              model_args_current = batch_args,
              step_current = svi_steps_completed
            )
          }
          rss_mb_value <- current_process_rss_mb()
          elapsed_seconds <- as.numeric(difftime(Sys.time(), t0_, units = "secs"))
          progress_elapsed_s <- elapsed_seconds - as.numeric(last_progress_elapsed)
          step_per_s_value <- compact_throughput_rate(window_steps, progress_elapsed_s)
          sampled_train_obs_per_s_value <- compact_throughput_rate(window_sampled_train_obs, progress_elapsed_s)
          update_step_per_s_value <- compact_throughput_rate(window_steps, window_update_elapsed_s)
          update_sampled_train_obs_per_s_value <- compact_throughput_rate(
            window_sampled_train_obs,
            window_update_elapsed_s
          )
          message(sprintf(
            paste0(
              "Compact SVI progress: step=%d/%d; train_elbo=%s; ",
              "step_per_s=%s; sampled_train_obs_per_s=%s; ",
              "update_step_per_s=%s; update_sampled_train_obs_per_s=%s; ",
              "window_steps=%d; window_chunks=%d; progress_elapsed_s=%s; ",
              "window_update_elapsed_s=%s; rss_mb=%s; elapsed=%ss%s."
            ),
            as.integer(svi_steps_completed),
            as.integer(svi_steps),
            format_svi_diag_value(svi_loss_curve[[svi_steps_completed]], digits = 2L),
            format_svi_diag_value(step_per_s_value, digits = 3L),
            format_svi_diag_value(sampled_train_obs_per_s_value, digits = 3L),
            format_svi_diag_value(update_step_per_s_value, digits = 3L),
            format_svi_diag_value(update_sampled_train_obs_per_s_value, digits = 3L),
            as.integer(window_steps),
            as.integer(window_chunks),
            format_svi_diag_value(progress_elapsed_s, digits = 3L),
            format_svi_diag_value(window_update_elapsed_s, digits = 3L),
            format_svi_diag_value(rss_mb_value, digits = 1L),
            format_svi_diag_value(elapsed_seconds, digits = 3L),
            format_svi_gradient_fields(last_gradient_checkpoint)
          ))
          last_progress_step <- as.integer(svi_steps_completed)
          last_progress_elapsed <- elapsed_seconds
          window_steps <- 0L
          window_chunks <- 0L
          window_update_elapsed_s <- 0
          window_sampled_train_obs <- 0
        }
        step_cursor <- as.integer(svi_steps_completed) + 1L
      }
      if (svi_steps_completed < length(svi_loss_curve)) {
        svi_loss_curve <- if (svi_steps_completed > 0L) {
          svi_loss_curve[seq_len(svi_steps_completed)]
        } else {
          numeric(0)
        }
      }
      if (isTRUE(compact_validation_active) &&
          !identical(early_stopping_reason, "update_failed")) {
        finite_validation_history <- early_stopping_info$validation_loss_history[
          is.finite(early_stopping_info$validation_loss_history)
        ]
        if (length(finite_validation_history) > 0L && is.finite(best_metric)) {
          if (!identical(early_stopping_reason, "patience_exhausted")) {
            early_stopping_reason <- "completed_budget"
          }
          early_stopping_info$reason <- early_stopping_reason
          early_stopping_info$best_metric <- best_metric
          early_stopping_info$final_metric <- best_metric
          if (!is.null(compact_best_svi_params)) {
            SVIParams <- compact_best_svi_params
          } else if (!is.null(compact_best_svi_state)) {
            SVIParams <- tryCatch(svi$get_params(compact_best_svi_state), error = function(e) NULL)
          }
          if (isTRUE(svi_checkpoint$enabled) && is.null(SVIParams)) {
            svi_checkpoint_best <- neural_svi_checkpoint_restore_best(
              svi_checkpoint$path,
              svi_checkpoint_fingerprint
            )
            if (!is.null(svi_checkpoint_best) &&
                is.finite(svi_checkpoint_best$best_metric %||% NA_real_)) {
              SVIParams <- neural_svi_checkpoint_params_to_jax(svi_checkpoint_best$svi_params)
              SVIPosteriorDraws <- checkpoint_prediction_params_to_draws(
                svi_checkpoint_best$prediction_params
              )
            }
          }
        } else {
          early_stopping_reason <- if (compact_validation_errors > 0L) {
            "validation_error"
          } else {
            "metric_failed"
          }
          early_stopping_info$reason <- early_stopping_reason
        }
      } else {
        early_stopping_info$reason <- early_stopping_reason
      }
      early_stopping_info$stop_step <- as.integer(svi_steps_completed)
    } else if (!isTRUE(checkpoint_training_complete) && isTRUE(early_stopping_running)) {
      svi_state <- if (is.null(checkpoint_resume_params)) {
        do.call(svi$init, c(list(rng_key), svi_train_model_args))
      } else {
        NULL
      }
      best_svi_state <- NULL
      best_metric <- if (!is.null(svi_checkpoint_best) &&
                         is.finite(svi_checkpoint_best$best_metric %||% NA_real_)) {
        as.numeric(svi_checkpoint_best$best_metric)
      } else {
        Inf
      }
      if (is.finite(best_metric)) {
        early_stopping_info$best_metric <- best_metric
        early_stopping_info$best_step <- as.integer(svi_checkpoint_best$best_step %||%
                                                      early_stopping_info$best_step)
      }
      last_metric <- if (length(early_stopping_info$validation_loss_history) > 0L) {
        tail(early_stopping_info$validation_loss_history, 1L)
      } else {
        NA_real_
      }
      no_improve_checks <- as.integer(svi_checkpoint_latest$no_improve_checks %||% 0L)
      if (is.na(no_improve_checks) || no_improve_checks < 0L) {
        no_improve_checks <- 0L
      }
      chunk_size <- max(1L, as.integer(early_stopping_info$eval_every %||% svi_steps))
      total_checks_planned <- max(1L, as.integer(ceiling(svi_steps / chunk_size)))
      svi_loss_chunks <- list()
      if (!is.null(svi_checkpoint_latest$loss_history) &&
          length(svi_checkpoint_latest$loss_history) > 0L) {
        svi_loss_chunks[[1L]] <- as.numeric(svi_checkpoint_latest$loss_history)
      }
      steps_completed <- as.integer(checkpoint_resume_completed)
      steps_remaining <- max(0L, as.integer(svi_steps) - steps_completed)

      while (steps_remaining > 0L) {
        chunk_steps <- min(chunk_size, steps_remaining)
        chunk_started_at <- proc.time()[["elapsed"]]
        run_result <- checkpoint_run_chunk(
          chunk_steps = chunk_steps,
          model_args_current = svi_train_model_args,
          init_state_current = svi_state,
          init_params_current = checkpoint_resume_params,
          progress_bar = FALSE
        )
        checkpoint_resume_params <- NULL
        strategize_jax_block_until_ready(run_result)
        chunk_run_seconds <- as.numeric(proc.time()[["elapsed"]] - chunk_started_at)
        run_parts <- parse_svi_run_result(run_result)
        if (is.null(run_parts$state)) {
          early_stopping_reason <- "update_failed"
          break
        }
        svi_state <- run_parts$state
        chunk_losses <- as.numeric(run_parts$losses)
        if (!length(chunk_losses)) {
          chunk_losses <- rep(NA_real_, chunk_steps)
        }
        svi_loss_chunks[[length(svi_loss_chunks) + 1L]] <- chunk_losses
        steps_completed <- steps_completed + as.integer(chunk_steps)
        steps_remaining <- steps_remaining - as.integer(chunk_steps)
        svi_steps_completed <- as.integer(steps_completed)
        current_loss_history <- unlist(svi_loss_chunks, use.names = FALSE)
        gradient_checkpoint <- record_svi_gradient_checkpoint(
          svi_state_current = svi_state,
          model_args_current = svi_train_model_args,
          step_current = steps_completed
        )

        validation_metric_error <- NULL
        metric_value <- tryCatch(
          compute_svi_validation_metric(svi_state, validation_split),
          error = function(e) {
            validation_metric_error <<- conditionMessage(e)
            NA_real_
          }
        )
        if (!is.null(validation_metric_error)) {
          early_stopping_info$error_message <- validation_metric_error
          early_stopping_reason <- "validation_error"
          early_stopping_info$reason <- early_stopping_reason
          early_stopping_info$stop_step <- as.integer(svi_steps_completed)
          checkpoint_save(
            type = "latest",
            svi_state_current = svi_state,
            step_current = steps_completed,
            loss_history_current = current_loss_history,
            best_metric_current = best_metric,
            best_step_current = early_stopping_info$best_step,
            no_improve_checks_current = no_improve_checks
          )
          break
        }
        if (!is.finite(metric_value)) {
          early_stopping_reason <- "metric_failed"
          early_stopping_info$reason <- early_stopping_reason
          early_stopping_info$stop_step <- as.integer(svi_steps_completed)
          checkpoint_save(
            type = "latest",
            svi_state_current = svi_state,
            step_current = steps_completed,
            loss_history_current = current_loss_history,
            best_metric_current = best_metric,
            best_step_current = early_stopping_info$best_step,
            no_improve_checks_current = no_improve_checks
          )
          break
        }

        previous_metric <- if (length(early_stopping_info$validation_loss_history) > 0L) {
          tail(early_stopping_info$validation_loss_history, 1L)
        } else {
          NA_real_
        }
        last_metric <- metric_value
        early_stopping_info$validation_loss_history <- c(
          early_stopping_info$validation_loss_history,
          metric_value
        )
        early_stopping_info$stop_check <- as.integer(length(early_stopping_info$validation_loss_history))
        improved_metric <- !is.finite(best_metric) ||
          metric_value < (best_metric - early_stopping_info$min_delta)
        if (isTRUE(improved_metric)) {
          best_metric <- metric_value
          best_svi_state <- svi_state
          early_stopping_info$best_step <- as.integer(steps_completed)
          early_stopping_info$best_metric <- metric_value
          no_improve_checks <- 0L
        } else {
          no_improve_checks <- no_improve_checks + 1L
        }

        best_metric_for_message <- if (is.finite(early_stopping_info$best_metric)) {
          early_stopping_info$best_metric
        } else {
          metric_value
        }
        best_step_for_message <- if (!is.na(early_stopping_info$best_step)) {
          as.integer(early_stopping_info$best_step)
        } else {
          as.integer(steps_completed)
        }
        delta_prev <- if (is.finite(previous_metric)) {
          metric_value - previous_metric
        } else {
          NA_real_
        }
        delta_prev_text <- if (is.finite(delta_prev)) {
          sprintf("%+.6f", delta_prev)
        } else {
          "NA"
        }
        train_elbo_value <- if (length(chunk_losses) > 0L) {
          tail(chunk_losses, 1L)
        } else {
          NA_real_
        }
        iter_per_s_value <- if (is.finite(chunk_run_seconds) && chunk_run_seconds > 0) {
          as.numeric(chunk_steps) / chunk_run_seconds
        } else {
          NA_real_
        }
        rss_mb_value <- current_process_rss_mb()
        elapsed_seconds <- as.numeric(difftime(Sys.time(), t0_, units = "secs"))
        message(sprintf(
          paste0(
            "SVI early-stop check %d/%d: step=%d/%d; validation %s=%.6f; ",
            "train_elbo=%s; best=%.6f at step %d; delta_prev=%s; iter_per_s=%s; ",
            "rss_mb=%s; elapsed=%ss%s."
          ),
          as.integer(early_stopping_info$stop_check),
          total_checks_planned,
          steps_completed,
          svi_steps,
          early_stopping_info$metric,
          metric_value,
          format_svi_diag_value(train_elbo_value, digits = 2L),
          best_metric_for_message,
          best_step_for_message,
          delta_prev_text,
          format_svi_diag_value(iter_per_s_value, digits = 3L),
          format_svi_diag_value(rss_mb_value, digits = 1L),
          format_svi_diag_value(elapsed_seconds, digits = 3L),
          format_svi_gradient_fields(gradient_checkpoint)
        ))

        if (isTRUE(improved_metric)) {
          checkpoint_save(
            type = "best",
            svi_state_current = svi_state,
            step_current = steps_completed,
            loss_history_current = current_loss_history,
            best_metric_current = best_metric,
            best_step_current = early_stopping_info$best_step,
            no_improve_checks_current = no_improve_checks
          )
        }
        early_stopping_info$reason <- early_stopping_reason
        early_stopping_info$stop_step <- as.integer(svi_steps_completed)
        checkpoint_save(
          type = "latest",
          svi_state_current = svi_state,
          step_current = steps_completed,
          loss_history_current = current_loss_history,
          best_metric_current = best_metric,
          best_step_current = early_stopping_info$best_step,
          no_improve_checks_current = no_improve_checks
        )

        if (no_improve_checks >= early_stopping_info$patience &&
            steps_completed < as.integer(svi_steps)) {
          early_stopping_info$stopped_early <- TRUE
          early_stopping_reason <- "patience_exhausted"
          early_stopping_info$reason <- early_stopping_reason
          early_stopping_info$stop_step <- as.integer(svi_steps_completed)
          checkpoint_save(
            type = "latest",
            svi_state_current = svi_state,
            step_current = steps_completed,
            loss_history_current = current_loss_history,
            best_metric_current = best_metric,
            best_step_current = early_stopping_info$best_step,
            no_improve_checks_current = no_improve_checks
          )
          message(sprintf(
            "SVI early stopping at step %d/%d on validation %s=%.6f.",
            steps_completed,
            svi_steps,
            early_stopping_info$metric,
            metric_value
          ))
          break
        }
      }
      svi_loss_curve <- if (length(svi_loss_chunks) > 0L) {
        unlist(svi_loss_chunks, use.names = FALSE)
      } else {
        NULL
      }

      if (is.finite(best_metric)) {
        early_stopping_info$final_metric <- best_metric
      } else if (is.finite(last_metric)) {
        early_stopping_info$final_metric <- last_metric
      }
      if (!is.null(best_svi_state) &&
          !early_stopping_reason %in% c("metric_failed", "validation_error")) {
        SVIParams <- tryCatch(svi$get_params(best_svi_state), error = function(e) NULL)
        if (!isTRUE(svi_checkpoint$enabled)) {
          svi_state <- best_svi_state
        }
      }
      early_stopping_info$reason <- early_stopping_reason
      early_stopping_info$stop_step <- as.integer(svi_steps_completed %||% svi_steps)
      if (isTRUE(svi_checkpoint$enabled) &&
          is.null(SVIParams) &&
          !early_stopping_reason %in% c("metric_failed", "validation_error")) {
        svi_checkpoint_best <- neural_svi_checkpoint_restore_best(
          svi_checkpoint$path,
          svi_checkpoint_fingerprint
        )
        if (!is.null(svi_checkpoint_best) &&
            is.finite(svi_checkpoint_best$best_metric %||% NA_real_)) {
          SVIParams <- neural_svi_checkpoint_params_to_jax(svi_checkpoint_best$svi_params)
          SVIPosteriorDraws <- checkpoint_prediction_params_to_draws(
            svi_checkpoint_best$prediction_params
          )
        }
      }
    } else if (!isTRUE(checkpoint_training_complete) && isTRUE(svi_checkpoint$enabled)) {
      chunk_size <- max(1L, as.integer(ceiling(svi_steps / svi_checkpoint$n_checks)))
      total_checks_planned <- max(1L, as.integer(ceiling(svi_steps / chunk_size)))
      svi_loss_chunks <- list()
      if (!is.null(svi_checkpoint_latest$loss_history) &&
          length(svi_checkpoint_latest$loss_history) > 0L) {
        svi_loss_chunks[[1L]] <- as.numeric(svi_checkpoint_latest$loss_history)
      }
      steps_completed <- as.integer(checkpoint_resume_completed)
      steps_remaining <- max(0L, as.integer(svi_steps) - steps_completed)
      svi_state <- NULL
      checkpoint_index <- max(0L, as.integer(ceiling(steps_completed / chunk_size)))
      while (steps_remaining > 0L) {
        chunk_steps <- min(chunk_size, steps_remaining)
        chunk_started_at <- proc.time()[["elapsed"]]
        run_result <- checkpoint_run_chunk(
          chunk_steps = chunk_steps,
          model_args_current = svi_model_args,
          init_state_current = svi_state,
          init_params_current = checkpoint_resume_params,
          progress_bar = FALSE
        )
        checkpoint_resume_params <- NULL
        strategize_jax_block_until_ready(run_result)
        chunk_run_seconds <- as.numeric(proc.time()[["elapsed"]] - chunk_started_at)
        run_parts <- parse_svi_run_result(run_result)
        if (is.null(run_parts$state)) {
          early_stopping_reason <- "update_failed"
          break
        }
        svi_state <- run_parts$state
        chunk_losses <- as.numeric(run_parts$losses)
        if (!length(chunk_losses)) {
          chunk_losses <- rep(NA_real_, chunk_steps)
        }
        svi_loss_chunks[[length(svi_loss_chunks) + 1L]] <- chunk_losses
        steps_completed <- steps_completed + as.integer(chunk_steps)
        steps_remaining <- steps_remaining - as.integer(chunk_steps)
        checkpoint_index <- checkpoint_index + 1L
        svi_steps_completed <- as.integer(steps_completed)
        current_loss_history <- unlist(svi_loss_chunks, use.names = FALSE)
        gradient_checkpoint <- record_svi_gradient_checkpoint(
          svi_state_current = svi_state,
          model_args_current = svi_model_args,
          step_current = steps_completed
        )
        iter_per_s_value <- if (is.finite(chunk_run_seconds) && chunk_run_seconds > 0) {
          as.numeric(chunk_steps) / chunk_run_seconds
        } else {
          NA_real_
        }
        message(sprintf(
          "SVI checkpoint %d/%d: step=%d/%d; train_elbo=%s; iter_per_s=%s%s.",
          checkpoint_index,
          total_checks_planned,
          steps_completed,
          svi_steps,
          format_svi_diag_value(tail(chunk_losses, 1L), digits = 2L),
          format_svi_diag_value(iter_per_s_value, digits = 3L),
          format_svi_gradient_fields(gradient_checkpoint)
        ))
        early_stopping_info$reason <- early_stopping_reason
        early_stopping_info$stop_step <- as.integer(svi_steps_completed)
        checkpoint_save(
          type = "latest",
          svi_state_current = svi_state,
          step_current = steps_completed,
          loss_history_current = current_loss_history,
          best_metric_current = NA_real_,
          best_step_current = NA_integer_,
          no_improve_checks_current = 0L
        )
      }
      svi_loss_curve <- if (length(svi_loss_chunks) > 0L) {
        unlist(svi_loss_chunks, use.names = FALSE)
      } else {
        NULL
      }
      early_stopping_info$reason <- early_stopping_reason
      early_stopping_info$stop_step <- as.integer(svi_steps_completed %||% svi_steps)
    } else if (!isTRUE(checkpoint_training_complete)) {
      svi_result <- do.call(
        svi$run,
        c(
          list(rng_key, ai(svi_steps)),
          svi_model_args
        )
      )
      strategize_jax_block_until_ready(svi_result)
      run_parts <- parse_svi_run_result(svi_result)
      svi_loss_curve <- run_parts$losses
      svi_state <- run_parts$state
      svi_steps_completed <- if (!is.null(svi_loss_curve) && length(svi_loss_curve) > 0L) {
        as.integer(length(svi_loss_curve))
      } else {
        as.integer(svi_steps)
      }
      if (!is.null(svi_state)) {
        record_svi_gradient_checkpoint(
          svi_state_current = svi_state,
          model_args_current = svi_model_args,
          step_current = svi_steps_completed
        )
      }
      early_stopping_info$reason <- early_stopping_reason
      early_stopping_info$stop_step <- as.integer(svi_steps_completed)
    }

    svi_loss_curve <- as.numeric(svi_loss_curve)
    if (length(svi_loss_curve) > 0L) {
      svi_loss_curve[!is.finite(svi_loss_curve)] <- NA_real_
    }
    if (!is.null(svi_loss_curve) && length(svi_loss_curve) > 0L &&
        identical(subsample_method, "batch_vi")) {
      svi_plot_title <- neural_format_svi_elbo_plot_title(svi_loss_curve)
      try(suppressWarnings(plot(svi_loss_curve,
                                type = "l",
                                main = svi_plot_title,
                                xlab = "Iteration",
                                log = ifelse(any(svi_loss_curve < 0), "", "y"),
                                ylab = "ELBO loss")), TRUE)
      finite_idx <- is.finite(svi_loss_curve)
      if (sum(finite_idx, na.rm = TRUE) >= 2L) {
        try(points(lowess(svi_loss_curve[finite_idx]),
                   type = "l",
                   lwd = 2,
                   col = "red"), TRUE)
      }
    }
    lr_trace_info <- neural_extract_lr_trace(
      lr_schedule = lr_schedule,
      steps_completed = svi_steps_completed %||% length(svi_loss_curve),
      fallback_lr = svi_lr
    )
    optimizer_diagnostics$steps_completed <- as.integer(svi_steps_completed %||% length(svi_loss_curve))
    optimizer_diagnostics$lr_trace <- lr_trace_info$lr_trace
    optimizer_diagnostics$lr_trace_status <- lr_trace_info$lr_trace_status
    skipped_update_steps <- suppressWarnings(as.integer(
      optimizer_diagnostics$compact_update_skipped_steps %||% 0L
    ))
    optimizer_diagnostics$optimizer_status <- if (length(skipped_update_steps) == 1L &&
                                                    !is.na(skipped_update_steps) &&
                                                    skipped_update_steps > 0L) {
      "ok_with_skipped_updates"
    } else {
      "ok"
    }
    optimizer_diagnostics$universal_loss_weighting <- universal_loss_weighting_diagnostics
    params <- if (!is.null(SVIParams)) {
      SVIParams
    } else {
      svi$get_params(svi_state)
    }
    SVIParams <- params
    n_draws <- ai(resolved_svi_num_draws)
    if (length(n_draws) != 1L || is.na(n_draws) || !is.finite(n_draws) || n_draws < 1L) {
      n_draws <- 1L
    }
    sample_key <- neural_training_prng_key(neural_training_seed, stream = 2L)
    if (isTRUE(run_mcmc_after_svi)) {
      SVIInitValues <- extract_svi_param_sites(params)
    } else if (!is.null(SVIPosteriorDraws)) {
      # Checkpoint-restored fits arrive here with single-point pseudo-draws
      # (guide medians), whose variance is exactly zero. Resample the guide
      # posterior from the restored variational params (which include the
      # *_auto_scale sites) so a rebuilt fit carries the same uncertainty as
      # an uninterrupted one. sample_posterior needs a prototype trace; on a
      # pure rebuild the guide was never initialized, so retry after one
      # throwaway svi$init.
      sample_restored_posterior <- function() {
        posterior_sample_args <- c(
          list(
            sample_key,
            params,
            sample_shape = reticulate::tuple(ai(n_draws))
          ),
          svi_model_args
        )
        posterior_samples <- do.call(guide$sample_posterior, posterior_sample_args)
        out <- lapply(posterior_samples, function(x) {
          strenv$jnp$expand_dims(x, 0L)
        })
        names(out) <- names(posterior_samples)
        out
      }
      rebuilt_draws <- tryCatch(
        sample_restored_posterior(),
        error = function(e) {
          tryCatch({
            invisible(do.call(svi$init, c(list(rng_key), svi_model_args)))
            sample_restored_posterior()
          }, error = function(e2) {
            warning(sprintf(paste0(
              "Could not resample the guide posterior from restored ",
              "checkpoint parameters (%s). Posterior variance is unavailable ",
              "for this fit: dependent standard errors will be NA."
            ), conditionMessage(e2)), call. = FALSE)
            NULL
          })
        }
      )
      if (!is.null(rebuilt_draws)) {
        PosteriorDraws <- rebuilt_draws
      } else {
        PosteriorDraws <- SVIPosteriorDraws
        posterior_draws_point_only <- TRUE
      }
    } else {
      posterior_sample_args <- c(
        list(
          sample_key,
          params,
          sample_shape = reticulate::tuple(ai(n_draws))
        ),
        svi_model_args
      )
      posterior_samples <- do.call(guide$sample_posterior, posterior_sample_args)
      PosteriorDraws <- lapply(posterior_samples, function(x) {
        strenv$jnp$expand_dims(x, 0L)
      })
      names(PosteriorDraws) <- names(posterior_samples)
    }
    message(sprintf("\n SVI Runtime: %.3f min",
                    as.numeric(difftime(Sys.time(), t0_, units = "secs"))/60))
    emit_svi_fit_summary()
    strenv$moe_aux_active <- FALSE
    # Release this fit's jitted SVI update/gradient closures: they capture the
    # svi object (and through it the full-dataset arrays), so without clearing,
    # every CV fold / OOS refit / adversarial fit grew session memory for the
    # session's lifetime (the clear helpers previously had no callers).
    tryCatch({
      if (!is.null(strenv$jax_svi_update_jit_cache_clear)) {
        strenv$jax_svi_update_jit_cache_clear()
      }
      if (!is.null(strenv$jax_svi_gradient_jit_cache_clear)) {
        strenv$jax_svi_gradient_jit_cache_clear()
      }
    }, error = function(e) NULL)
  }

  if (!isTRUE(use_svi) || isTRUE(run_mcmc_after_svi)) {
    strenv$numpyro$set_host_device_count(mcmc_control$n_chains)
    if (!isTRUE(use_svi)) {
      emit_transformer_structure_banner()
    }

    init_strategy <- NULL
    if (!is.null(SVIInitValues) && length(SVIInitValues) > 0L) {
      if (!is.null(init_to_value)) {
        init_strategy <- init_to_value(values = SVIInitValues)
        message("Initializing MCMC with SVI posterior means...")
      }
    } else if (!is.null(init_to_value) && length(output_site_init_values) > 0L) {
      init_strategy <- init_to_value(values = output_site_init_values)
      message("Initializing MCMC with data-informed output warm starts...")
    }

    if (identical(subsample_method, "batch")) {
      message("Enlisting HMCECS kernels for subsampled likelihood...")
      nuts_kernel <- if (is.null(init_strategy)) {
        strenv$numpyro$infer$NUTS(model_fn)
      } else {
        strenv$numpyro$infer$NUTS(model_fn, init_strategy = init_strategy)
      }
      kernel <- strenv$numpyro$infer$HMCECS(
        nuts_kernel,
        num_blocks = if (!is.null(mcmc_control$num_blocks)) ai(mcmc_control$num_blocks) else ai(4L)
      )
    } else {
      message("Enlisting NUTS kernels for full-data likelihood...")
      kernel <- if (is.null(init_strategy)) {
        strenv$numpyro$infer$NUTS(
          model_fn,
          max_tree_depth = ai(8L),
          target_accept_prob = 0.85
        )
      } else {
        strenv$numpyro$infer$NUTS(
          model_fn,
          max_tree_depth = ai(8L),
          target_accept_prob = 0.85,
          init_strategy = init_strategy
        )
      }
    }

    sampler <- strenv$numpyro$infer$MCMC(
      sampler = kernel,
      num_warmup = mcmc_control$n_samples_warmup,
      num_samples = mcmc_control$n_samples_mcmc,
      thinning   = mcmc_control$n_thin_by,
      chain_method = ifelse(!is.null(mcmc_control$chain_method), yes = mcmc_control$chain_method, no = "parallel"),
      num_chains = mcmc_control$n_chains,
      jit_model_args = TRUE,
      progress_bar = TRUE
    )

    if (pairwise_mode) {
      sampler$run(neural_training_prng_key(neural_training_seed, stream = 3L),
                  X_left = X_left_jnp,
                  X_right = X_right_jnp,
                  party_left = party_left_jnp,
                  party_right = party_right_jnp,
                  resp_party = resp_party_jnp,
                  resp_cov = resp_cov_jnp,
                  resp_cov_present = resp_cov_present_jnp,
                  experiment_index = experiment_index_jnp,
                  Y_obs = Y_jnp)
    } else {
      sampler$run(neural_training_prng_key(neural_training_seed, stream = 4L),
                  X = X_single_jnp,
                  party = party_single_jnp,
                  resp_party = resp_party_jnp,
                  resp_cov = resp_cov_jnp,
                  resp_cov_present = resp_cov_present_jnp,
                  experiment_index = experiment_index_jnp,
                  Y_obs = Y_jnp)
    }
    PosteriorDraws <- sampler$get_samples(group_by_chain = TRUE)
    message(sprintf("\n MCMC Runtime: %.3f min",
                    as.numeric(difftime(Sys.time(), t0_, units = "secs"))/60))
  }

  mean_param <- function(x) { strenv$jnp$mean(x, 0L:1L) }
  get_loc_scale_draws <- function(name, scale_name) {
    draws <- PosteriorDraws[[name]]
    if (!is.null(draws)) {
      return(draws)
    }
    base_names <- c(paste0(name, "_decentered"),
                    paste0(name, "_base"),
                    paste0(name, "_z"))
    base_draws <- NULL
    for (base_name in base_names) {
      if (!is.null(PosteriorDraws[[base_name]])) {
        base_draws <- PosteriorDraws[[base_name]]
        break
      }
    }
    if (is.null(base_draws)) {
      return(NULL)
    }
    scale_draws <- PosteriorDraws[[scale_name]]
    if (is.null(scale_draws)) {
      return(NULL)
    }
    scale_shape <- tryCatch(
      as.integer(reticulate::py_to_r(scale_draws$shape)),
      error = function(e) NULL
    )
    base_shape <- tryCatch(
      as.integer(reticulate::py_to_r(base_draws$shape)),
      error = function(e) NULL
    )
    if (is.null(scale_shape) || is.null(base_shape)) {
      return(scale_draws * base_draws)
    }
    extra_dims <- length(base_shape) - length(scale_shape)
    if (extra_dims > 0L) {
      reshape_dims <- c(scale_shape, rep(1L, extra_dims))
      scale_draws <- strenv$jnp$reshape(scale_draws, as.list(reshape_dims))
    }
    scale_draws * base_draws
  }
  get_cross_draws <- function() {
    draws <- PosteriorDraws$M_cross
    if (!is.null(draws)) {
      return(draws)
    }
    raw_draws <- PosteriorDraws$M_cross_raw
    if (is.null(raw_draws)) {
      raw_draws <- get_loc_scale_draws("M_cross_raw", "tau_cross")
    }
    if (is.null(raw_draws)) {
      return(NULL)
    }
    trans_axes <- list(0L, 1L, 3L, 2L)
    0.5 * (raw_draws - strenv$jnp$transpose(raw_draws, trans_axes))
  }
  get_stacked_layer_draws <- function(name) {
    draws <- PosteriorDraws[[name]]
    if (!is.null(draws)) {
      return(draws)
    }
    map <- neural_standard_transformer_stack_map()
    if (!name %in% names(map)) {
      return(NULL)
    }
    legacy_base <- unname(map[[name]])
    parts <- lapply(seq_len(ModelDepth), function(l_) {
      legacy_name <- paste0(legacy_base, l_)
      if (legacy_base %in% c("W_q_l", "W_k_l", "W_v_l", "W_o_l", "W_ff1_l", "W_ff2_l")) {
        return(get_loc_scale_draws(legacy_name, paste0("tau_w_", l_)))
      }
      PosteriorDraws[[legacy_name]]
    })
    if (any(vapply(parts, is.null, logical(1)))) {
      return(NULL)
    }
    strenv$jnp$stack(parts, axis = 2L)
  }
  get_param_draws <- function(name) {
    if (name %in% names(neural_standard_transformer_stack_map())) {
      return(get_stacked_layer_draws(name))
    }
    if (identical(name, "E_segment")) {
      draws <- PosteriorDraws$E_segment
      if (!is.null(draws)) {
        return(draws)
      }
      delta_draws <- PosteriorDraws$E_segment_delta
      if (is.null(delta_draws)) {
        return(NULL)
      }
      return(neural_build_symmetric_segment_embeddings(delta_draws))
    }
    if (identical(name, "M_cross")) {
      return(get_cross_draws())
    }
    if (identical(name, "W_out")) {
      return(get_loc_scale_draws("W_out", "tau_w_out"))
    }
    if (identical(name, "W_add_out")) {
      return(get_loc_scale_draws("W_add_out", "tau_w_out"))
    }
    if (identical(name, "b_out")) {
      return(get_loc_scale_draws("b_out", "tau_b"))
    }
    if (name %in% c("W_q_cross", "W_k_cross", "W_v_cross", "W_o_cross")) {
      return(get_loc_scale_draws(name, "tau_cross_attn"))
    }
    if (name %in% c("W_rc_r", "W_rc_c")) {
      return(get_loc_scale_draws(name, "tau_rc"))
    }
    if (grepl("^W_(q|k|v|o)_l\\d+$", name) ||
        grepl("^W_ff(1|2)_l\\d+$", name)) {
      layer_id <- sub(".*_l", "", name)
      tau_name <- paste0("tau_w_", layer_id)
      return(get_loc_scale_draws(name, tau_name))
    }
    PosteriorDraws[[name]]
  }

  p2d_output_only <- isTRUE(output_only_mode)
  get_svi_param <- function(name) {
    if (is.null(SVIParams)) {
      return(NULL)
    }
    value <- tryCatch(SVIParams[[name]], error = function(e) NULL)
    if (cs2step_has_reticulate() &&
        reticulate::is_py_object(value) &&
        inherits(value, "python.builtin.NoneType")) {
      return(NULL)
    }
    value
  }
  get_site_mean_or_param <- function(name) {
    draws <- get_param_draws(name)
    if (!is.null(draws)) {
      return(mean_param(draws))
    }
    if (isTRUE(p2d_output_only) && identical(name, "E_segment")) {
      delta_val <- get_svi_param("E_segment_delta")
      if (!is.null(delta_val)) {
        return(neural_build_symmetric_segment_embeddings(delta_val))
      }
    }
    get_svi_param(name)
  }

  ParamsMean <- list()
  ParamsMean$E_choice <- get_site_mean_or_param("E_choice")
  if (is.null(ParamsMean$E_choice)) {
    stop("Neural model is missing required embedding estimates.", call. = FALSE)
  }

  maybe_site <- function(name, assign_as = name) {
    value <- get_site_mean_or_param(name)
    if (!is.null(value)) {
      ParamsMean[[assign_as]] <<- value
    }
    invisible(value)
  }

  get_gate_mean_or_fixed <- function(name) {
    value <- get_site_mean_or_param(name)
    if (!is.null(value)) {
      return(value)
    }
    if (!isTRUE(use_svi)) {
      return(p2d_fixed_gate_value())
    }
    NULL
  }

  maybe_gate <- function(name, assign_as = name) {
    value <- get_gate_mean_or_fixed(name)
    if (!is.null(value)) {
      ParamsMean[[assign_as]] <<- value
    }
    invisible(value)
  }

  maybe_site("E_party")
  maybe_site("E_resp_party")
  maybe_site("E_respondent_cls")
  maybe_site("E_candidate_cls")
  maybe_site("E_sep")
  maybe_site("E_segment")
  maybe_site("E_factor_fused_base")
  maybe_site("W_factor_fuse_1")
  maybe_site("b_factor_fuse_1")
  maybe_site("W_factor_fuse_2")
  maybe_site("b_factor_fuse_2")
  maybe_site("E_rel")
  maybe_site("E_token_family")
  maybe_site("E_experiment")
  maybe_site("E_stage")
  maybe_site("E_matchup")
  maybe_site("W_factor_name_text")
  maybe_site("W_level_name_text")
  maybe_site("W_factor_struct")
  maybe_site("W_level_struct")
  maybe_site("W_covariate_name_text")
  maybe_site("W_experiment_text")
  maybe_site("W_place_context")
  maybe_site("W_time_context")
  if (isTRUE(pairwise_mode) && isTRUE(use_cross_attn)) {
    maybe_gate("alpha_cross")
  }
  maybe_site("RMS_cross")
  maybe_site("RMS_merge_cross")
  maybe_site("RMS_q_cross")
  maybe_site("RMS_k_cross")

  W_out_draws <- get_loc_scale_draws("W_out", "tau_w_out")
  if (!is.null(W_out_draws)) {
    ParamsMean$W_out <- mean_param(W_out_draws)
  }
  W_add_out_draws <- get_loc_scale_draws("W_add_out", "tau_w_out")
  if (!is.null(W_add_out_draws)) {
    ParamsMean$W_add_out <- mean_param(W_add_out_draws)
  } else {
    maybe_site("W_add_out")
  }
  b_out_draws <- get_loc_scale_draws("b_out", "tau_b")
  if (!is.null(b_out_draws)) {
    ParamsMean$b_out <- mean_param(b_out_draws)
  }
  ordinal_threshold_raw_mean <- get_site_mean_or_param("ordinal_threshold_raw")
  if (!is.null(ordinal_threshold_raw_mean)) {
    ParamsMean$ordinal_threshold_raw <- ordinal_threshold_raw_mean
    ParamsMean$ordinal_thresholds <- ordinal_thresholds_from_raw_jnp(ordinal_threshold_raw_mean)
  } else {
    maybe_site("ordinal_thresholds")
  }

  cross_draws <- get_cross_draws()
  if (!is.null(cross_draws)) {
    ParamsMean$M_cross <- mean_param(cross_draws)
  } else if (isTRUE(p2d_output_only) && isTRUE(pairwise_mode) && isTRUE(use_cross_term)) {
    M_cross_raw <- get_svi_param("M_cross_raw")
    if (!is.null(M_cross_raw)) {
      ParamsMean$M_cross <- 0.5 * (M_cross_raw - strenv$jnp$transpose(M_cross_raw))
    }
  }

  if (!is.null(PosteriorDraws$W_cross_out)) {
    ParamsMean$W_cross_out <- mean_param(PosteriorDraws$W_cross_out)
  }
  if (low_rank_interaction_rank > 0L) {
    maybe_gate("alpha_rc")
  }
  maybe_site("W_rc_r")
  maybe_site("W_rc_c")
  maybe_site("W_rc_out")
  maybe_site("log_pairwise_bernoulli_logit_scale")
  maybe_site("log_calibration_scale")
  if (!is.null(ParamsMean$log_calibration_scale)) {
    ParamsMean$calibration_scale <- strenv$jnp$exp(ParamsMean$log_calibration_scale)
  }
  if (likelihood == "normal" || isTRUE(universal_has_normal)) {
    ParamsMean$sigma <- mean_param(PosteriorDraws$sigma)
  }

  if (n_resp_covariates > 0L) {
    maybe_site("E_covariate_fused_base")
    maybe_site("E_covariate_missing")
    maybe_site("W_covariate_fuse_1")
    maybe_site("b_covariate_fuse_1")
    maybe_site("W_covariate_fuse_2")
    maybe_site("b_covariate_fuse_2")
    maybe_site("W_covariate_value_text")
    maybe_site("W_covariate_value_shared")
    maybe_site("W_covariate_value_basis")
    maybe_site("W_covariate_value_conditioner_1")
    maybe_site("b_covariate_value_conditioner_1")
    maybe_site("W_covariate_value_conditioner_2")
    maybe_site("b_covariate_value_conditioner_2")
  }

  for (l_ in 1L:ModelDepth) {
    maybe_site(paste0("pseudo_query_attn_l", l_))
    maybe_site(paste0("pseudo_query_ff_l", l_))
    alpha_attn_name <- paste0("alpha_attn_l", l_)
    alpha_ff_name <- paste0("alpha_ff_l", l_)
    if (!isTRUE(use_full_attn_residual)) {
      maybe_gate(alpha_attn_name)
      maybe_gate(alpha_ff_name)
    }

    maybe_site(paste0("RMS_attn_l", l_))
    maybe_site(paste0("RMS_q_l", l_))
    maybe_site(paste0("RMS_k_l", l_))
    maybe_site(paste0("RMS_ff_l", l_))

    for (base in c("W_q_l", "W_k_l", "W_v_l", "W_o_l", "W_ff1_l", "W_ff2_l")) {
      name <- paste0(base, l_)
      tau_name <- paste0("tau_w_", l_)
      draws <- get_loc_scale_draws(name, tau_name)
      if (!is.null(draws)) {
        ParamsMean[[name]] <- mean_param(draws)
      } else if (isTRUE(p2d_output_only)) {
        value <- get_svi_param(name)
        if (!is.null(value)) {
          ParamsMean[[name]] <- value
        }
      }
    }
  }
  if (!isTRUE(use_full_attn_residual)) {
    ParamsMean <- neural_stack_standard_transformer_layers(
      ParamsMean,
      model_depth = ModelDepth,
      drop_legacy = TRUE
    )
  }
  for (name in c("W_q_cross", "W_k_cross", "W_v_cross", "W_o_cross")) {
    draws <- get_loc_scale_draws(name, "tau_cross_attn")
    if (!is.null(draws)) {
      ParamsMean[[name]] <- mean_param(draws)
    } else if (isTRUE(p2d_output_only)) {
      value <- get_svi_param(name)
      if (!is.null(value)) {
        ParamsMean[[name]] <- value
      }
    }
  }
  maybe_site("pseudo_query_final")
  maybe_site("RMS_final")

  pairwise_bernoulli_logit_scale_mean <- 1.0
  if (!is.null(ParamsMean$log_pairwise_bernoulli_logit_scale)) {
    scale_jnp <- strenv$jnp$exp(ParamsMean$log_pairwise_bernoulli_logit_scale)
    pairwise_bernoulli_logit_scale_mean <- as.numeric(strenv$np$array(scale_jnp))[[1L]]
  }
  if (!is.finite(pairwise_bernoulli_logit_scale_mean) ||
      pairwise_bernoulli_logit_scale_mean <= 0) {
    pairwise_bernoulli_logit_scale_mean <- 1.0
  }
  low_rank_logit_model_info$pairwise_bernoulli_logit_scale <-
    pairwise_bernoulli_logit_scale_mean
  calibration_scale_mean <- 1.0
  if (!is.null(ParamsMean$log_calibration_scale)) {
    calibration_scale_jnp <- strenv$jnp$exp(ParamsMean$log_calibration_scale)
    calibration_scale_mean <- as.numeric(strenv$np$array(calibration_scale_jnp))[[1L]]
  } else if (!is.null(ParamsMean$calibration_scale)) {
    calibration_scale_mean <- as.numeric(strenv$np$array(ParamsMean$calibration_scale))[[1L]]
  }
  if (!is.finite(calibration_scale_mean) || calibration_scale_mean <= 0) {
    calibration_scale_mean <- 1.0
  }
  low_rank_logit_model_info$calibration_scale <- calibration_scale_mean

  has_qk_norm <- !is.null(ParamsMean$RMS_q_layers) || !is.null(ParamsMean$RMS_k_layers)
  for (l_ in 1L:ModelDepth) {
    if (!is.null(ParamsMean[[paste0("RMS_q_l", l_)]]) ||
        !is.null(ParamsMean[[paste0("RMS_k_l", l_)]])) {
      has_qk_norm <- TRUE
      break
    }
  }
  if (!isTRUE(has_qk_norm)) {
    has_qk_norm <- !is.null(ParamsMean$RMS_q_cross) || !is.null(ParamsMean$RMS_k_cross)
  }

  predict_model_info <- neural_make_prepared_prediction_model_info(
    model_depth = ModelDepth,
    model_dims = ModelDims,
    n_heads = TransformerHeads,
    head_dim = head_dim,
    residual_mode = residual_mode,
    attention_backend = attention_backend,
    attention_dtype = attention_dtype,
    attention_padding_multiple = attention_padding_multiple,
    attention_resolved_backend = attention_resolved_backend,
    attention_fallback_reason = attention_fallback_reason,
    cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
    n_party_levels = ai(n_party_levels),
    n_candidate_tokens = n_candidate_tokens,
    cross_candidate_encoder_mode = cross_candidate_encoder_mode,
    cross_candidate_encoder = !identical(cross_candidate_encoder_mode, "none"),
    likelihood = likelihood,
    shared_projection_value_encoder = shared_projection_value_encoder,
    covariate_value_stats_by_experiment = covariate_value_stats_by_experiment,
    default_covariate_value_stats = default_covariate_value_stats,
    covariate_value_metadata_by_experiment = covariate_value_metadata_by_experiment,
    default_covariate_value_metadata = default_covariate_value_metadata,
    covariate_value_text = covariate_value_text,
    covariate_value_text_present = covariate_value_text_present,
    covariate_value_type = covariate_value_type,
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    factor_order_by_experiment = factor_order_by_experiment,
    default_factor_order = default_factor_order,
    factor_struct_matrix = factor_struct_matrix,
    level_struct_matrices = level_struct_matrices,
    factor_struct_feature_names = factor_struct_feature_names,
    level_struct_feature_names = level_struct_feature_names,
    factor_tokenization = factor_tokenization,
    max_factor_tokens = max_factor_tokens,
    low_rank_interaction_rank = low_rank_interaction_rank,
    low_rank_logit_transform = low_rank_logit_transform,
    low_rank_logit_bound = low_rank_logit_bound,
    low_rank_logit_softness = low_rank_logit_softness,
    low_rank_logit_normalization = low_rank_logit_normalization,
    low_rank_head_weight_target_rms = low_rank_head_weight_target_rms,
    low_rank_rc_out_target_rms = low_rank_rc_out_target_rms,
    pairwise_antisymmetry = pairwise_antisymmetry_mode,
    additive_utility_mode = additive_utility_mode,
    additive_utility_normalization = additive_utility_normalization,
    additive_head_weight_target_rms = additive_head_weight_target_rms,
    calibration_enabled = isTRUE(calibration_control$enabled),
    calibration_method = calibration_control$method %||% "none",
    calibration_scale = calibration_scale_mean
  )
  predict_model_info <- neural_set_pairwise_context_model_info(
    info = predict_model_info,
    pairwise_context_mode = pairwise_context_mode,
    has_candidate_group_context = has_candidate_group_context,
    has_respondent_group_context = has_respondent_group_context,
    has_relation_token_context = has_relation_context,
    has_stage_context = stage_context_enabled,
    has_matchup_context = use_matchup_token,
    party_missing_label = party_missing_label,
    resp_party_missing_label = resp_party_missing_label,
    n_resp_party_levels = n_resp_party_levels,
    party_missing_index = party_missing_index,
    resp_party_missing_index = resp_party_missing_index,
    context_present_masking = context_present_masking
  )
  predict_model_info$init_policy <- init_policy
  predict_model_info$factor_name_text <- factor_name_text
  predict_model_info$level_name_text <- level_name_text
  predict_model_info$factor_struct_matrix <- factor_struct_matrix
  predict_model_info$level_struct_matrices <- level_struct_matrices
  predict_model_info$factor_struct_feature_names <- factor_struct_feature_names
  predict_model_info$level_struct_feature_names <- level_struct_feature_names
  predict_model_info$covariate_name_text <- covariate_name_text
  predict_model_info$experiment_description_text <- experiment_description_text
  predict_model_info$experiment_description_present <- experiment_description_present
  predict_model_info$default_experiment_text <- default_experiment_text
  predict_model_info$default_experiment_text_present <- default_experiment_text_present
  predict_model_info$place_embedding <- place_embedding
  predict_model_info$place_present <- place_present
  predict_model_info$place_context_enabled <- place_context_enabled
  predict_model_info$place_feature_names <- place_feature_names
  predict_model_info$default_place_embedding <- default_place_embedding
  predict_model_info$default_place_present <- default_place_present
  predict_model_info$place_context_dim <- place_context_dim
  predict_model_info$time_embedding <- time_embedding
  predict_model_info$time_present <- time_present
  predict_model_info$time_context_enabled <- time_context_enabled
  predict_model_info$time_feature_names <- time_feature_names
  predict_model_info$default_time_embedding <- default_time_embedding
  predict_model_info$default_time_present <- default_time_present
  predict_model_info$time_context_dim <- time_context_dim
  predict_model_info$covariate_names <- covariate_names_override
  predict_model_info$covariate_order_by_experiment <- covariate_order_by_experiment
  predict_model_info$default_covariate_order <- default_covariate_order
  predict_model_info$max_covariate_tokens <- max_covariate_tokens
  predict_model_info$resp_cov_mean <- resp_cov_mean
  predict_model_info$resp_cov_scale <- resp_cov_scale
  predict_model_info$resp_cov_default_present <- resp_cov_default_present
  predict_model_info$covariate_value_stats_by_experiment <- covariate_value_stats_by_experiment
  predict_model_info$default_covariate_value_stats <- default_covariate_value_stats
  predict_model_info$covariate_value_metadata_by_experiment <- covariate_value_metadata_by_experiment
  predict_model_info$default_covariate_value_metadata <- default_covariate_value_metadata
  predict_model_info$covariate_value_text <- covariate_value_text
  predict_model_info$covariate_value_text_present <- covariate_value_text_present
  predict_model_info$covariate_value_type <- covariate_value_type
  predict_model_info$n_resp_covariates <- n_resp_covariates
  predict_model_info$experiment_levels <- experiment_levels_override
  predict_model_info$default_experiment_index <- if (is.na(default_experiment_index)) NULL else as.integer(default_experiment_index)
  predict_model_info$token_family_levels <- token_family_levels
  predict_model_info$experiment_token_mode <- experiment_token_mode
  predict_model_info$covariate_value_encoding <- covariate_value_encoding
  predict_model_info$shared_projection_value_encoder <- shared_projection_value_encoder
  predict_pair_jit_response <- if (isTRUE(pairwise_mode)) {
    neural_get_predict_jit(
      model_info = predict_model_info,
      pairwise = TRUE,
      return_logits = FALSE
    )
  } else {
    NULL
  }
  predict_pair_jit_logits <- if (isTRUE(pairwise_mode)) {
    neural_get_predict_jit(
      model_info = predict_model_info,
      pairwise = TRUE,
      return_logits = TRUE
    )
  } else {
    NULL
  }
  predict_single_jit_response <- if (!isTRUE(pairwise_mode) || isTRUE(universal_mixed_mode)) {
    neural_get_predict_jit(
      model_info = predict_model_info,
      pairwise = FALSE,
      return_logits = FALSE
    )
  } else {
    NULL
  }
  predict_single_jit_logits <- if (!isTRUE(pairwise_mode) || isTRUE(universal_mixed_mode)) {
    neural_get_predict_jit(
      model_info = predict_model_info,
      pairwise = FALSE,
      return_logits = TRUE
    )
  } else {
    NULL
  }

  normalize_direct_resp_cov_prediction <- function(resp_cov_new = NULL,
                                                   resp_cov_present_new = NULL,
                                                   n_rows = 1L) {
    n_rows <- ai(n_rows)
    if (is.null(resp_cov_new)) {
      n_cov <- length(covariate_names_override)
      if (n_cov < 1L) {
        n_cov <- ai(n_resp_covariates)
      }
      values <- matrix(0, nrow = n_rows, ncol = n_cov)
      if (n_cov > 0L && length(covariate_names_override) == n_cov) {
        colnames(values) <- covariate_names_override
      }
      return(list(values = values, present = matrix(0, nrow = n_rows, ncol = n_cov)))
    }

    raw_mat <- as.matrix(resp_cov_new)
    if (nrow(raw_mat) != n_rows) {
      stop("resp_cov_new must have one row per prediction row.", call. = FALSE)
    }
    missing_raw <- is.na(raw_mat)
    values <- suppressWarnings(matrix(
      as.numeric(raw_mat),
      nrow = nrow(raw_mat),
      ncol = ncol(raw_mat),
      dimnames = dimnames(raw_mat)
    ))
    bad_non_numeric <- is.na(values) & !missing_raw
    if (any(bad_non_numeric)) {
      stop("resp_cov_new must contain numeric values or NA.", call. = FALSE)
    }
    bad_inf <- !is.na(values) & !is.finite(values)
    if (any(bad_inf)) {
      stop("resp_cov_new must not contain Inf or -Inf values.", call. = FALSE)
    }
    values[is.na(values)] <- 0

    if (is.null(resp_cov_present_new)) {
      present <- matrix(as.numeric(!missing_raw), nrow = nrow(values), ncol = ncol(values))
      dimnames(present) <- dimnames(values)
    } else {
      present <- as.matrix(resp_cov_present_new)
      if (!identical(dim(present), dim(values))) {
        stop("resp_cov_present_new must have the same dimensions as resp_cov_new.", call. = FALSE)
      }
      present <- matrix(as.numeric(present), nrow = nrow(values), ncol = ncol(values))
      dimnames(present) <- dimnames(values)
      present[is.na(present)] <- 0
    }
    list(values = values, present = present)
  }

  TransformerPredict_pair <- function(params, Xl_new, Xr_new, pl_new, pr_new,
                                      resp_party_new = NULL, resp_cov_new = NULL,
                                      resp_cov_present_new = NULL,
                                      resp_cov_order_new = NULL,
                                      experiment_idx_new = NULL,
                                      factor_order_new = NULL,
                                      return_logits = FALSE) {
    Xl <- strenv$jnp$array(to_index_matrix(Xl_new))$astype(strenv$jnp$int32)
    Xr <- strenv$jnp$array(to_index_matrix(Xr_new))$astype(strenv$jnp$int32)
    pl <- strenv$jnp$array(as.integer(pl_new))$astype(strenv$jnp$int32)
    pr <- strenv$jnp$array(as.integer(pr_new))$astype(strenv$jnp$int32)
    if (is.null(resp_party_new)) {
      # NULL means "unknown respondent group": use the explicit missing-group
      # embedding (as my_model's coercion does), not the first group level.
      resp_party_new <- rep(
        as.integer(resp_party_missing_index %||% 0L),
        nrow(Xl_new)
      )
    }
    resp_p <- strenv$jnp$array(as.integer(resp_party_new))$astype(strenv$jnp$int32)
    resp_cov_prepped <- normalize_direct_resp_cov_prediction(
      resp_cov_new = resp_cov_new,
      resp_cov_present_new = resp_cov_present_new,
      n_rows = nrow(Xl_new)
    )
    resp_cov_new <- resp_cov_prepped$values
    resp_cov_present_new <- resp_cov_prepped$present
    resp_c <- strenv$jnp$array(as.matrix(resp_cov_new))$astype(ddtype_)
    resp_c_present <- strenv$jnp$array(as.matrix(resp_cov_present_new))$astype(ddtype_)
    if (is.null(resp_cov_order_new)) {
      resp_cov_order_new <- build_resp_cov_order_new(
        resp_cov_new,
        nrow(Xl_new),
        experiment_idx_new = experiment_idx_new
      )
    }
    resp_c_order <- if (is.null(resp_cov_order_new)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(resp_cov_order_new))$astype(strenv$jnp$int32)
    }
    experiment_idx <- if (!is.null(experiment_idx_new)) {
      strenv$jnp$array(as.integer(experiment_idx_new))$astype(strenv$jnp$int32)
    } else if (!is.na(default_experiment_index)) {
      strenv$jnp$array(rep.int(as.integer(default_experiment_index), nrow(Xl_new)))$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    if (is.null(factor_order_new)) {
      factor_order_new <- build_factor_order_new(
        Xl_new,
        nrow(Xl_new),
        experiment_idx_new = experiment_idx_new
      )
    }
    factor_order <- if (is.null(factor_order_new)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(factor_order_new))$astype(strenv$jnp$int32)
    }
    predict_fn <- if (isTRUE(return_logits)) {
      predict_pair_jit_logits
    } else {
      predict_pair_jit_response
    }
    pred <- predict_fn(
      params, Xl, Xr, pl, pr, resp_p, resp_c, resp_c_present,
      resp_c_order, experiment_idx, NULL, NULL, factor_order
    )
    strategize_jax_block_until_ready(pred)
    pred
  }

  TransformerPredict_single <- function(params, X_new, party_new,
                                        resp_party_new = NULL, resp_cov_new = NULL,
                                        resp_cov_present_new = NULL,
                                        resp_cov_order_new = NULL,
                                        experiment_idx_new = NULL,
                                        factor_order_new = NULL,
                                        return_logits = FALSE) {
    Xb <- strenv$jnp$array(to_index_matrix(X_new))$astype(strenv$jnp$int32)
    pb <- strenv$jnp$array(as.integer(party_new))$astype(strenv$jnp$int32)
    if (is.null(resp_party_new)) {
      # NULL means "unknown respondent group": use the explicit missing-group
      # embedding (as my_model's coercion does), not the first group level.
      resp_party_new <- rep(
        as.integer(resp_party_missing_index %||% 0L),
        nrow(X_new)
      )
    }
    resp_p <- strenv$jnp$array(as.integer(resp_party_new))$astype(strenv$jnp$int32)
    resp_cov_prepped <- normalize_direct_resp_cov_prediction(
      resp_cov_new = resp_cov_new,
      resp_cov_present_new = resp_cov_present_new,
      n_rows = nrow(X_new)
    )
    resp_cov_new <- resp_cov_prepped$values
    resp_cov_present_new <- resp_cov_prepped$present
    resp_c <- strenv$jnp$array(as.matrix(resp_cov_new))$astype(ddtype_)
    resp_c_present <- strenv$jnp$array(as.matrix(resp_cov_present_new))$astype(ddtype_)
    if (is.null(resp_cov_order_new)) {
      resp_cov_order_new <- build_resp_cov_order_new(
        resp_cov_new,
        nrow(X_new),
        experiment_idx_new = experiment_idx_new
      )
    }
    resp_c_order <- if (is.null(resp_cov_order_new)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(resp_cov_order_new))$astype(strenv$jnp$int32)
    }
    experiment_idx <- if (!is.null(experiment_idx_new)) {
      strenv$jnp$array(as.integer(experiment_idx_new))$astype(strenv$jnp$int32)
    } else if (!is.na(default_experiment_index)) {
      strenv$jnp$array(rep.int(as.integer(default_experiment_index), nrow(X_new)))$astype(strenv$jnp$int32)
    } else {
      NULL
    }
    if (is.null(factor_order_new)) {
      factor_order_new <- build_factor_order_new(
        X_new,
        nrow(X_new),
        experiment_idx_new = experiment_idx_new
      )
    }
    factor_order <- if (is.null(factor_order_new)) {
      NULL
    } else {
      strenv$jnp$array(as.matrix(factor_order_new))$astype(strenv$jnp$int32)
    }
    predict_fn <- if (isTRUE(return_logits)) {
      predict_single_jit_logits
    } else {
      predict_single_jit_response
    }
    pred <- predict_fn(
      params, Xb, pb, resp_p, resp_c, resp_c_present,
      resp_c_order, experiment_idx, NULL, NULL, factor_order
    )
    strategize_jax_block_until_ready(pred)
    pred
  }

  coerce_party_idx_base <- function(party_vec, n_rows, levels, missing_label) {
    neural_coerce_group_index_base(
      values = party_vec,
      n_rows = n_rows,
      levels = levels,
      missing_label = missing_label
    )
  }
  coerce_party_idx <- function(party_vec, n_rows) {
    coerce_party_idx_base(party_vec, n_rows, party_levels, party_missing_label)
  }
  coerce_resp_party_idx <- function(party_vec, n_rows) {
    coerce_party_idx_base(party_vec, n_rows, resp_party_levels, resp_party_missing_label)
  }

  to_r_array <- function(x) {
    if (is.null(x) || is.numeric(x)) {
      return(x)
    }
    strategize_jax_block_until_ready(x)
    tryCatch(reticulate::py_to_r(strenv$np$array(x)),
             error = function(e) {
               tryCatch(reticulate::py_to_r(x), error = function(e2) x)
             })
  }

  coerce_mixed_prediction_output <- function(pred,
                                             target_likelihood = NULL,
                                             target_n_outcomes = NULL,
                                             pairwise_prediction = FALSE,
                                             target_experiment_index = NULL) {
    logits <- if (is.list(pred) && !is.null(pred$logits)) {
      pred$logits
    } else {
      pred
    }
    logits <- as.matrix(to_r_array(logits))
    if (ncol(logits) < 1L) {
      stop("Mixed-family prediction requires at least one output logit.", call. = FALSE)
    }
    target_likelihood <- tolower(as.character(target_likelihood %||% "bernoulli"))
    if (length(target_likelihood) != 1L || is.na(target_likelihood) || !nzchar(target_likelihood)) {
      target_likelihood <- "bernoulli"
    }
    target_n_outcomes <- suppressWarnings(as.integer(target_n_outcomes %||% 1L))
    if (length(target_n_outcomes) != 1L || is.na(target_n_outcomes) || target_n_outcomes < 1L) {
      target_n_outcomes <- 1L
    }
    # Train/serve parity: the jitted mixed cores skip the calibration
    # temperature (their jnp call lacks likelihood_code_obs, a no-op for
    # likelihood == "mixed"), and training applies it to class-family rows
    # (bernoulli/categorical/ordinal) only -- re-apply it here, never to
    # normal rows.
    if (identical(target_likelihood, "bernoulli")) {
      if (isTRUE(pairwise_prediction)) {
        logits[, 1L] <- neural_apply_pairwise_bernoulli_logit_adjustment_r(
          logits[, 1L],
          low_rank_logit_model_info
        )
      }
      logits[, 1L] <- neural_apply_classification_logit_calibration_r(
        logits[, 1L],
        low_rank_logit_model_info
      )
      return(stats::plogis(logits[, 1L]))
    }
    if (identical(target_likelihood, "categorical")) {
      k <- max(2L, min(as.integer(target_n_outcomes), ncol(logits)))
      z <- logits[, seq_len(k), drop = FALSE]
      z <- neural_apply_classification_logit_calibration_r(
        z,
        low_rank_logit_model_info
      )
      z <- sweep(z, 1L, apply(z, 1L, max), "-")
      p <- exp(z)
      return(sweep(p, 1L, rowSums(p), "/"))
    }
    if (identical(target_likelihood, "normal")) {
      sigma_source <- if (is.list(pred) && !is.null(pred$sigma)) {
        pred$sigma
      } else {
        ParamsMean$sigma %||% NULL
      }
      return(list(
        mu = as.numeric(logits[, 1L]),
        sigma = mixed_sigma_vector_r(to_r_array(sigma_source), nrow(logits))
      ))
    }
    if (target_likelihood %in% c("ordinal", "ordered", "ordered_logit", "ordinal_single")) {
      return(ordinal_prob_matrix_r(
        eta = neural_apply_classification_logit_calibration_r(
          logits[, 1L],
          low_rank_logit_model_info
        ),
        n_outcomes_obs = rep.int(as.integer(target_n_outcomes), nrow(logits)),
        experiment_index = target_experiment_index,
        ordinal_thresholds = to_r_array(ParamsMean$ordinal_thresholds %||% NULL),
        ordinal_threshold_raw = to_r_array(ParamsMean$ordinal_threshold_raw %||% NULL)
      ))
    }
    stop(
      sprintf("Unsupported target likelihood for mixed-family prediction: %s", target_likelihood),
      call. = FALSE
    )
  }

  coerce_prediction_output <- function(pred,
                                       target_likelihood = NULL,
                                       target_n_outcomes = NULL) {
    if (likelihood == "mixed") {
      return(coerce_mixed_prediction_output(
        pred = pred,
        target_likelihood = target_likelihood,
        target_n_outcomes = target_n_outcomes
      ))
    }
    if (likelihood == "bernoulli") {
      return(as.numeric(to_r_array(pred)))
    }
    if (likelihood == "categorical") {
      return(as.matrix(to_r_array(pred)))
    }
    if (likelihood == "normal") {
      return(list(
        mu = as.numeric(to_r_array(pred$mu)),
        sigma = as.numeric(to_r_array(pred$sigma))
      ))
    }
    if (likelihood == "ordinal") {
      logits <- if (is.list(pred) && !is.null(pred$logits)) {
        pred$logits
      } else {
        pred
      }
      logits <- as.matrix(to_r_array(logits))
      return(ordinal_prob_matrix_r(
        eta = logits[, 1L],
        n_outcomes_obs = rep.int(as.integer(target_n_outcomes %||% nOutcomes), nrow(logits)),
        experiment_index = NULL,
        ordinal_thresholds = to_r_array(ParamsMean$ordinal_thresholds %||% NULL),
        ordinal_threshold_raw = to_r_array(ParamsMean$ordinal_threshold_raw %||% NULL)
      ))
    }
    pred
  }

  my_model <- function(...) {
    args <- list(...)
    return_logits <- isTRUE(args$return_logits)
    args$return_logits <- NULL
    target_likelihood <- tolower(as.character(
      args$target_likelihood %||% args$likelihood %||% NULL
    ))
    if (length(target_likelihood) != 1L || is.na(target_likelihood) || !nzchar(target_likelihood)) {
      target_likelihood <- if (identical(likelihood, "mixed")) "bernoulli" else likelihood
    }
    target_n_outcomes <- suppressWarnings(as.integer(
      args$target_n_outcomes %||% args$n_outcomes %||% nOutcomes
    ))
    if (length(target_n_outcomes) != 1L || is.na(target_n_outcomes) || target_n_outcomes < 1L) {
      target_n_outcomes <- as.integer(nOutcomes)
    }
    predict_mode <- tolower(as.character(args$mode %||% args$target_mode %||% NULL))
    if (length(predict_mode) != 1L || is.na(predict_mode) || !nzchar(predict_mode)) {
      predict_mode <- NULL
    }
    args$target_likelihood <- NULL
    args$likelihood <- NULL
    args$target_n_outcomes <- NULL
    args$n_outcomes <- NULL
    args$mode <- NULL
    args$target_mode <- NULL

    use_pair_prediction <- if (!is.null(predict_mode)) {
      identical(predict_mode, "pairwise")
    } else if (isTRUE(pairwise_mode) && !isTRUE(universal_mixed_mode)) {
      TRUE
    } else {
      !is.null(args$X_left_new) || !is.null(args$X_right_new)
    }
    if (isTRUE(use_pair_prediction)) {
      X_left_new <- args$X_left_new
      X_right_new <- args$X_right_new
      party_left_new <- args$party_left_new
      party_right_new <- args$party_right_new
      resp_party_new <- args$resp_party_new
      resp_cov_new <- args$resp_cov_new
      resp_cov_present_new <- args$resp_cov_present_new
      resp_cov_order_new <- args$resp_cov_order_new
      experiment_idx_new <- args$experiment_idx_new
      factor_order_new <- args$factor_order_new

      if (is.null(X_left_new) || is.null(X_right_new)) {
        if (length(args) < 2L) {
          stop("pairwise my_model requires X_left_new and X_right_new.", call. = FALSE)
        }
        if (is.null(X_left_new)) X_left_new <- args[[1]]
        if (is.null(X_right_new)) X_right_new <- args[[2]]
        if (length(args) >= 3L && is.null(party_left_new)) party_left_new <- args[[3]]
        if (length(args) >= 4L && is.null(party_right_new)) party_right_new <- args[[4]]
        if (length(args) >= 5L && is.null(resp_party_new)) resp_party_new <- args[[5]]
        if (length(args) >= 6L && is.null(resp_cov_new)) resp_cov_new <- args[[6]]
        if (length(args) >= 7L && is.null(resp_cov_present_new)) resp_cov_present_new <- args[[7]]
        if (length(args) >= 8L) {
          arg8 <- args[[8]]
          if (is.null(resp_cov_order_new) &&
              (is.matrix(arg8) || is.data.frame(arg8))) {
            resp_cov_order_new <- arg8
          } else if (is.null(experiment_idx_new)) {
            experiment_idx_new <- arg8
          }
        }
        if (length(args) >= 9L && is.null(experiment_idx_new)) experiment_idx_new <- args[[9]]
        if (length(args) >= 10L && is.null(factor_order_new)) factor_order_new <- args[[10]]
      }

      party_left_new <- coerce_party_idx(party_left_new, nrow(X_left_new))
      party_right_new <- coerce_party_idx(party_right_new, nrow(X_right_new))
      resp_party_new <- coerce_resp_party_idx(resp_party_new, nrow(X_left_new))
      pred <- TransformerPredict_pair(ParamsMean, X_left_new, X_right_new,
                                      party_left_new, party_right_new,
                                      resp_party_new, resp_cov_new,
                                      resp_cov_present_new, resp_cov_order_new,
                                      experiment_idx_new, factor_order_new,
                                      return_logits = isTRUE(return_logits) || identical(likelihood, "mixed"))
      if (identical(likelihood, "mixed")) {
        logits <- as.matrix(to_r_array(pred))
        sigma_vec <- mixed_sigma_vector_r(to_r_array(ParamsMean$sigma %||% NULL), nrow(logits))
        raw_pred <- list(logits = logits, sigma = sigma_vec)
        if (isTRUE(return_logits)) {
          return(raw_pred)
        }
        return(coerce_mixed_prediction_output(
          pred = raw_pred,
          target_likelihood = target_likelihood,
          target_n_outcomes = target_n_outcomes,
          pairwise_prediction = TRUE,
          target_experiment_index = experiment_idx_new
        ))
      }
      if (isTRUE(return_logits)) {
        logits <- as.matrix(to_r_array(pred))
        sigma_vec <- mixed_sigma_vector_r(to_r_array(ParamsMean$sigma %||% NULL), nrow(logits))
        return(list(logits = logits, sigma = sigma_vec))
      }
      return(coerce_prediction_output(
        pred,
        target_likelihood = target_likelihood,
        target_n_outcomes = target_n_outcomes
      ))
    }

    X_new <- args$X_new
    party_new <- args$party_new
    resp_party_new <- args$resp_party_new
    resp_cov_new <- args$resp_cov_new
    resp_cov_present_new <- args$resp_cov_present_new
    resp_cov_order_new <- args$resp_cov_order_new
    experiment_idx_new <- args$experiment_idx_new
    factor_order_new <- args$factor_order_new

    if (is.null(X_new)) {
      if (!is.null(args$X_left_new)) {
        X_new <- args$X_left_new
        if (is.null(party_new) && !is.null(args$party_left_new)) {
          party_new <- args$party_left_new
        }
      } else if (length(args) >= 1L) {
        X_new <- args[[1]]
        if (length(args) >= 2L && is.null(party_new)) party_new <- args[[2]]
        if (length(args) >= 3L && is.null(resp_party_new)) resp_party_new <- args[[3]]
        if (length(args) >= 4L && is.null(resp_cov_new)) resp_cov_new <- args[[4]]
        if (length(args) >= 5L && is.null(resp_cov_present_new)) resp_cov_present_new <- args[[5]]
        if (length(args) >= 6L) {
          arg6 <- args[[6]]
          if (is.null(resp_cov_order_new) &&
              (is.matrix(arg6) || is.data.frame(arg6))) {
            resp_cov_order_new <- arg6
          } else if (is.null(experiment_idx_new)) {
            experiment_idx_new <- arg6
          }
        }
        if (length(args) >= 7L && is.null(experiment_idx_new)) experiment_idx_new <- args[[7]]
        if (length(args) >= 8L && is.null(factor_order_new)) factor_order_new <- args[[8]]
      }
    }

    if (is.null(X_new)) {
      stop("my_model requires X_new for single-candidate predictions.", call. = FALSE)
    }
    party_new <- coerce_party_idx(party_new, nrow(X_new))
    resp_party_new <- coerce_resp_party_idx(resp_party_new, nrow(X_new))
    pred <- TransformerPredict_single(ParamsMean, X_new, party_new,
                                      resp_party_new, resp_cov_new,
                                      resp_cov_present_new, resp_cov_order_new,
                                      experiment_idx_new, factor_order_new,
                                      return_logits = isTRUE(return_logits) || identical(likelihood, "mixed"))
    if (identical(likelihood, "mixed")) {
      logits <- as.matrix(to_r_array(pred))
      sigma_vec <- mixed_sigma_vector_r(to_r_array(ParamsMean$sigma %||% NULL), nrow(logits))
      raw_pred <- list(logits = logits, sigma = sigma_vec)
      if (isTRUE(return_logits)) {
        return(raw_pred)
      }
      return(coerce_mixed_prediction_output(
        pred = raw_pred,
        target_likelihood = target_likelihood,
        target_n_outcomes = target_n_outcomes,
        pairwise_prediction = FALSE,
        target_experiment_index = experiment_idx_new
      ))
    }
    if (isTRUE(return_logits)) {
      logits <- as.matrix(to_r_array(pred))
      sigma_vec <- mixed_sigma_vector_r(to_r_array(ParamsMean$sigma %||% NULL), nrow(logits))
      return(list(logits = logits, sigma = sigma_vec))
    }
    coerce_prediction_output(
      pred,
      target_likelihood = target_likelihood,
      target_n_outcomes = target_n_outcomes
    )
  }

  if (likelihood %in% c("bernoulli", "categorical", "normal", "ordinal") &&
      !is.null(fit_metrics) &&
      !is.null(fit_metrics$eval_index) &&
      length(fit_metrics$eval_index) > 0L) {
    eval_idx_in <- as.integer(fit_metrics$eval_index)
    pred_in_sample <- tryCatch({
      if (pairwise_mode) {
        my_model(
          X_left_new = X_left[eval_idx_in, , drop = FALSE],
          X_right_new = X_right[eval_idx_in, , drop = FALSE],
          party_left_new = party_left[eval_idx_in],
          party_right_new = party_right[eval_idx_in],
          resp_party_new = resp_party_use[eval_idx_in],
          resp_cov_new = if (!is.null(X_use) && n_resp_covariates > 0L) {
            X_use[eval_idx_in, , drop = FALSE]
          } else {
            NULL
          },
          resp_cov_present_new = if (!is.null(X_present_use) && n_resp_covariates > 0L) {
            X_present_use[eval_idx_in, , drop = FALSE]
          } else {
            NULL
          },
          experiment_idx_new = if (!is.null(experiment_index_use)) {
            experiment_index_use[eval_idx_in]
          } else {
            NULL
          }
        )
      } else {
        my_model(
          X_new = X_single[eval_idx_in, , drop = FALSE],
          party_new = party_single[eval_idx_in],
          resp_party_new = resp_party_use[eval_idx_in],
          resp_cov_new = if (!is.null(X_use) && n_resp_covariates > 0L) {
            X_use[eval_idx_in, , drop = FALSE]
          } else {
            NULL
          },
          resp_cov_present_new = if (!is.null(X_present_use) && n_resp_covariates > 0L) {
            X_present_use[eval_idx_in, , drop = FALSE]
          } else {
            NULL
          },
          experiment_idx_new = if (!is.null(experiment_index_use)) {
            experiment_index_use[eval_idx_in]
          } else {
            NULL
          }
        )
      }
    }, error = function(e) NULL)

    if (!is.null(pred_in_sample)) {
      in_sample_metrics <- if (identical(likelihood, "ordinal")) {
        ordinal_metrics_r(
          y_eval = Y_use[eval_idx_in],
          prob_mat = pred_in_sample,
          n_outcomes_obs = rep.int(as.integer(nOutcomes), length(eval_idx_in))
        )
      } else {
        cs_compute_outcome_metrics(
          y_eval = Y_use[eval_idx_in],
          pred_eval = pred_in_sample,
          likelihood = likelihood
        )
      }
      in_sample_metrics$eval_note <- "in_sample_full_fit"
      in_sample_metrics$eval_subset <- fit_metrics$eval_subset

      if (pairwise_mode && isTRUE(has_relation_context)) {
        stage_primary <- party_left == party_right
        if (length(stage_primary) == length(Y_use)) {
          stage_primary <- stage_primary[eval_idx_in]
          stage_keep <- context_present_pair[eval_idx_in]
          stage_keep[is.na(stage_keep)] <- FALSE
          stage_primary[!stage_keep] <- NA
          by_stage <- list()
          if (any(stage_primary, na.rm = TRUE)) {
            idx0 <- which(stage_primary)
            pred_stage <- if (likelihood == "bernoulli") {
              pred_in_sample[idx0]
            } else if (likelihood %in% c("categorical", "ordinal")) {
              pred_in_sample[idx0, , drop = FALSE]
            } else {
              list(
                mu = pred_in_sample$mu[idx0],
                sigma = pred_in_sample$sigma[idx0]
              )
            }
            by_stage$primary <- if (identical(likelihood, "ordinal")) {
              ordinal_metrics_r(
                y_eval = Y_use[eval_idx_in][idx0],
                prob_mat = pred_stage,
                n_outcomes_obs = rep.int(as.integer(nOutcomes), length(idx0))
              )
            } else {
              cs_compute_outcome_metrics(
                y_eval = Y_use[eval_idx_in][idx0],
                pred_eval = pred_stage,
                likelihood = likelihood
              )
            }
          }
          if (any(!stage_primary, na.rm = TRUE)) {
            idx1 <- which(!stage_primary)
            pred_stage <- if (likelihood == "bernoulli") {
              pred_in_sample[idx1]
            } else if (likelihood %in% c("categorical", "ordinal")) {
              pred_in_sample[idx1, , drop = FALSE]
            } else {
              list(
                mu = pred_in_sample$mu[idx1],
                sigma = pred_in_sample$sigma[idx1]
              )
            }
            by_stage$general <- if (identical(likelihood, "ordinal")) {
              ordinal_metrics_r(
                y_eval = Y_use[eval_idx_in][idx1],
                prob_mat = pred_stage,
                n_outcomes_obs = rep.int(as.integer(nOutcomes), length(idx1))
              )
            } else {
              cs_compute_outcome_metrics(
                y_eval = Y_use[eval_idx_in][idx1],
                pred_eval = pred_stage,
                likelihood = likelihood
              )
            }
          }
          if (length(by_stage) > 0L) {
            in_sample_metrics$by_stage <- by_stage
          }
        }
      }

      fit_metrics$in_sample_metrics <- in_sample_metrics
    }
    fit_metrics$eval_index <- NULL
  }

  # Neural parameter vector and diagonal posterior covariance
  param_schema <- neural_build_param_schema(
    params = ParamsMean,
    n_factors = length(factor_levels),
    model_depth = ModelDepth,
    factor_tokenization = factor_tokenization
  )
  param_names <- param_schema$param_names
  param_shapes <- param_schema$param_shapes
  param_sizes <- param_schema$param_sizes
  param_offsets <- param_schema$param_offsets
  param_total <- as.integer(param_schema$n_params)

  theta_mean <- neural_flatten_params(ParamsMean, param_schema, dtype = ddtype_)
  theta_mean_num <- as.numeric(strenv$np$array(theta_mean))

  var_parts <- lapply(seq_along(param_names), function(i_) {
    name <- param_names[[i_]]
    draws <- get_param_draws(name)
    if (is.null(draws)) {
      return(strenv$jnp$zeros(list(ai(param_sizes[[i_]])), dtype = ddtype_))
    }
    strenv$jnp$ravel(strenv$jnp$var(draws, 0L:1L))
  })
  param_var_vec <- if (length(var_parts) == 0L) {
    strenv$jnp$array(numeric(0), dtype = ddtype_)
  } else {
    strenv$jnp$concatenate(var_parts, axis = 0L)
  }
  param_var <- as.numeric(strenv$np$array(param_var_vec))
  if (length(param_var) < param_total) {
    param_var <- c(param_var, rep(0, param_total - length(param_var)))
  }
  if (length(param_var) > param_total) {
    param_var <- param_var[seq_len(param_total)]
  }
  if (isTRUE(posterior_draws_point_only)) {
    # Draws collapsed to a restored point estimate: zero variance would be
    # silently wrong, so propagate NA into every downstream SE instead.
    param_var <- rep(NA_real_, length(param_var))
  }
  if (uncertainty_scope == "output") {
    keep <- param_names %in% c(
      "W_out", "W_add_out", "b_out", "sigma", "W_cross_out", "alpha_rc", "W_rc_out",
      "log_pairwise_bernoulli_logit_scale", "log_calibration_scale",
      "ordinal_threshold_raw", "ordinal_thresholds"
    )
    mask <- unlist(mapply(function(keep_i, size_i) rep(keep_i, size_i),
                          keep, param_sizes))
    if (length(mask) == length(param_var)) {
      param_var <- param_var * as.numeric(mask)
    }
  }
  vcov_OutcomeModel <- c(0, param_var)
  vcov_OutcomeModel_by_k <- NULL

  EST_INTERCEPT_tf <- strenv$jnp$array(matrix(0, nrow = 1L, ncol = 1L), dtype = strenv$dtj)
  EST_COEFFICIENTS_tf <- strenv$jnp$reshape(theta_mean, list(-1L, 1L))
  my_mean <- numeric(0)
  my_mean_full <- NULL
  if (exists("K", inherits = TRUE) && is.numeric(K) && K > 1) {
    base_vec <- c(0, theta_mean_num)
    my_mean_full <- matrix(rep(base_vec, K), ncol = K)
    vcov_OutcomeModel_by_k <- replicate(K, vcov_OutcomeModel, simplify = FALSE)
  }

  resp_cov_mean_jnp <- if (!is.null(resp_cov_mean)) {
    strenv$jnp$array(as.numeric(resp_cov_mean))$astype(ddtype_)
  } else {
    NULL
  }
  resp_cov_scale_jnp <- if (!is.null(resp_cov_scale)) {
    strenv$jnp$array(as.numeric(resp_cov_scale))$astype(ddtype_)
  } else {
    NULL
  }
  resp_cov_default_present_jnp <- if (!is.null(resp_cov_default_present)) {
    strenv$jnp$array(as.numeric(resp_cov_default_present))$astype(ddtype_)
  } else {
    NULL
  }
  party_index_map <- NULL
  if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    party_index_map <- setNames(vapply(GroupsPool, function(grp) {
      idx <- match(as.character(grp), party_levels) - 1L
      if (is.na(idx)) party_missing_index else idx
    }, integer(1)), GroupsPool)
  }
  resp_party_index_map <- NULL
  if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    resp_party_index_map <- setNames(vapply(GroupsPool, function(grp) {
      idx <- match(as.character(grp), resp_party_levels) - 1L
      if (is.na(idx)) resp_party_missing_index else idx
    }, integer(1)), GroupsPool)
  }

  # fit_metrics is computed above via cross-fitting (unless eval is disabled).

  if (!isTRUE(use_svi) && identical(early_stopping_info$reason, "not_initialized")) {
    early_stopping_info$reason <- if (isTRUE(early_stopping_info$enabled)) {
      "not_svi"
    } else {
      "disabled"
    }
  }
  if (isTRUE(use_svi) && is.null(svi_steps_completed) && !is.null(resolved_svi_steps)) {
    svi_steps_completed <- as.integer(resolved_svi_steps)
  }

  if (!is.null(svi_loss_curve) && length(svi_loss_curve) > 0L) {
    finite_losses <- svi_loss_curve[is.finite(svi_loss_curve)]
    svi_summary <- list(
      svi_elbo = svi_loss_curve,
      svi_elbo_final = if (length(finite_losses)) tail(finite_losses, 1) else NA_real_
    )
    if (is.null(fit_metrics)) {
      fit_metrics <- svi_summary
    } else {
      fit_metrics <- c(fit_metrics, svi_summary)
    }
  }
  if (!is.null(universal_loss_weighting_diagnostics)) {
    if (is.null(fit_metrics)) {
      fit_metrics <- list()
    }
    fit_metrics$universal_loss_weighting <- universal_loss_weighting_diagnostics
  }

  parameter_diagnostics <- neural_build_parameter_diagnostics(ParamsMean)
  convergence_diagnostics <- neural_build_convergence_diagnostics(
    parameter_diagnostics = parameter_diagnostics,
    svi_loss_curve = svi_loss_curve,
    early_stopping = early_stopping_info,
    fit_metrics = fit_metrics,
    steps_completed = svi_steps_completed,
    steps_planned = resolved_svi_steps,
    use_svi = isTRUE(use_svi)
  )

  average_case_dr_training <- NULL
  average_case_linear_q <- NULL
  average_case_single_normal <- !isTRUE(pairwise_mode) &&
    !isTRUE(universal_enabled) &&
    !isTRUE(compact_training) &&
    identical(likelihood, "normal") &&
    !is.null(X_single_jnp) &&
    !is.null(Y_jnp) &&
    !isFALSE(mcmc_control$average_case_linear_q)
  if (isTRUE(average_case_single_normal)) {
    # This surrogate REPLACES the neural soft prediction inside both the
    # optimization objective (neural_getQStar_single) and the reported Q --
    # a deliberate population-average estimand under independent
    # randomization, but it must never be silent: dQ/dtheta through the
    # surrogate is zero, so the outcome-model uncertainty channel does not
    # propagate through it.
    message(
      "Single-candidate normal outcome: optimization and reported Q use the ",
      "average-case linear (main + 2-way OLS) Q surrogate rather than the ",
      "neural prediction. Set neural_mcmc_control$average_case_linear_q = FALSE ",
      "to optimize the neural prediction directly."
    )
  }
  if (isTRUE(average_case_single_normal)) {
    average_case_linear_q_r <- neural_build_average_case_linear_q_calibration(
      W_idx = X_single,
      Y = Y_use,
      factor_levels = factor_levels,
      holdout_indicator = holdout_indicator
    )
    if (!is.null(average_case_linear_q_r)) {
      average_case_linear_q <- list(
        intercept = strenv$jnp$array(average_case_linear_q_r$intercept, dtype = strenv$dtj),
        main_coef = strenv$jnp$array(average_case_linear_q_r$main_coef, dtype = strenv$dtj),
        inter_coef = strenv$jnp$array(average_case_linear_q_r$inter_coef, dtype = strenv$dtj),
        inter_left = strenv$jnp$array(average_case_linear_q_r$inter_left, dtype = strenv$jnp$int32),
        inter_right = strenv$jnp$array(average_case_linear_q_r$inter_right, dtype = strenv$jnp$int32)
      )
    }
  }
  if (isTRUE(average_case_single_normal) &&
      !is.null(p_list) &&
      length(p_list) == length(factor_levels)) {
    average_case_dr_training <- list(
      X_single = X_single_jnp,
      Y_single = Y_jnp,
      party_single = party_single_jnp,
      resp_party = resp_party_jnp,
      resp_cov = resp_cov_jnp,
      resp_cov_present = resp_cov_present_jnp,
      experiment_idx = experiment_index_jnp,
      baseline_level_probs = lapply(p_list, function(prob_vec) {
        strenv$jnp$array(as.numeric(prob_vec), dtype = strenv$dtj)
      })
    )
  }

  neural_model_info <- list(
    params = ParamsMean,
    param_names = param_names,
    param_shapes = param_shapes,
    param_sizes = param_sizes,
    param_offsets = param_offsets,
    n_params = ai(param_total),
    uncertainty_scope = uncertainty_scope,
    transformer_ffn = "swiglu",
    fused_token_mlp = "swiglu",
    factor_levels = factor_levels,
    factor_index_list = factor_index_list,
    implicit = isTRUE(holdout_indicator == 1L),
    pairwise_mode = pairwise_mode,
    pairwise_context_mode = pairwise_context_mode,
    n_factors = ai(length(factor_levels)),
    n_candidate_tokens = n_candidate_tokens,
    party_levels = party_levels,
    n_party_levels = ai(n_party_levels),
    n_matchup_levels = ai(n_matchup_levels),
    resp_party_levels = resp_party_levels,
    n_resp_party_levels = ai(n_resp_party_levels),
    group_context_tokenization = if (isTRUE(unified_group_context)) {
      group_context_tokenization
    } else {
      NULL
    },
    group_context_schema = if (isTRUE(unified_group_context)) {
      group_context_schema
    } else {
      NULL
    },
    party_missing_label = party_missing_label,
    resp_party_missing_label = resp_party_missing_label,
    party_missing_index = as.integer(party_missing_index),
    resp_party_missing_index = as.integer(resp_party_missing_index),
    party_index_map = party_index_map,
    resp_party_index_map = resp_party_index_map,
    cand_party_to_resp_idx = cand_party_to_resp_idx_jnp,
    resp_cov_mean = resp_cov_mean_jnp,
    resp_cov_scale = resp_cov_scale_jnp,
    resp_cov_default_present = resp_cov_default_present_jnp,
    n_resp_covariates = n_resp_covariates,
    covariate_names = covariate_names_override,
    covariate_order_by_experiment = covariate_order_by_experiment,
    default_covariate_order = default_covariate_order,
    max_covariate_tokens = max_covariate_tokens,
    covariate_value_stats_by_experiment = covariate_value_stats_by_experiment,
    default_covariate_value_stats = default_covariate_value_stats,
    covariate_value_metadata_by_experiment = covariate_value_metadata_by_experiment,
    default_covariate_value_metadata = default_covariate_value_metadata,
    covariate_value_text = covariate_value_text,
    covariate_value_text_present = covariate_value_text_present,
    covariate_value_type = covariate_value_type,
    factor_order_by_experiment = factor_order_by_experiment,
    default_factor_order = default_factor_order,
    factor_tokenization = factor_tokenization,
    max_factor_tokens = max_factor_tokens,
    factor_schema_supplied = neural_factor_schema_supplied(
      factor_name_text = factor_name_text,
      level_name_text = level_name_text,
      factor_struct_matrix = factor_struct_matrix,
      level_struct_matrices = level_struct_matrices,
      default_factor_order = default_factor_order,
      factor_order_by_experiment = factor_order_by_experiment
    ),
    factor_name_text = factor_name_text,
    level_name_text = level_name_text,
    factor_struct_matrix = factor_struct_matrix,
    factor_struct_feature_names = factor_struct_feature_names,
    level_struct_matrices = level_struct_matrices,
    level_struct_feature_names = level_struct_feature_names,
    covariate_name_text = covariate_name_text,
    experiment_description_text = experiment_description_text,
    experiment_description_present = experiment_description_present,
    default_experiment_text = default_experiment_text,
    default_experiment_text_present = default_experiment_text_present,
    place_embedding = place_embedding,
    place_present = place_present,
    place_context_enabled = place_context_enabled,
    place_feature_names = place_feature_names,
    default_place_embedding = default_place_embedding,
    default_place_present = default_place_present,
    time_embedding = time_embedding,
    time_present = time_present,
    time_context_enabled = time_context_enabled,
    time_feature_names = time_feature_names,
    default_time_embedding = default_time_embedding,
    default_time_present = default_time_present,
    experiment_levels = experiment_levels_override,
    n_experiment_levels = as.integer(n_experiment_levels),
    default_experiment_index = if (is.na(default_experiment_index)) NULL else as.integer(default_experiment_index),
    token_family_levels = token_family_levels,
    experiment_token_mode = experiment_token_mode,
    covariate_value_encoding = covariate_value_encoding,
    shared_projection_value_encoder = shared_projection_value_encoder,
    low_rank_interaction_rank = as.integer(low_rank_interaction_rank),
    low_rank_logit_transform = low_rank_logit_transform,
    low_rank_logit_bound = low_rank_logit_bound,
    low_rank_logit_softness = low_rank_logit_softness,
    low_rank_logit_normalization = low_rank_logit_normalization,
    low_rank_head_weight_target_rms = low_rank_head_weight_target_rms,
    low_rank_rc_out_target_rms = low_rank_rc_out_target_rms,
    pairwise_antisymmetry = pairwise_antisymmetry_mode,
    additive_utility_mode = additive_utility_mode,
    additive_utility_normalization = additive_utility_normalization,
    additive_head_weight_target_rms = additive_head_weight_target_rms,
    has_additive_utility = !is.null(ParamsMean$W_add_out),
    learned_pairwise_bernoulli_logit_scale = learned_pairwise_bernoulli_logit_scale,
    pairwise_bernoulli_logit_scale_prior_sd = pairwise_bernoulli_logit_scale_prior_sd,
    pairwise_bernoulli_logit_scale = pairwise_bernoulli_logit_scale_mean,
    calibration_enabled = isTRUE(calibration_control$enabled),
    calibration_method = calibration_control$method %||% "none",
    calibration_prior_sd = calibration_control$prior_sd %||% NULL,
    calibration_scale = calibration_scale_mean,
    schema_dropout = schema_dropout,
    text_semantic_dim = as.integer(text_semantic_dim),
    factor_struct_dim = as.integer(factor_struct_dim),
    level_struct_dim = as.integer(level_struct_dim),
    place_context_dim = as.integer(place_context_dim),
    time_context_dim = as.integer(time_context_dim),
    has_candidate_group_context = isTRUE(has_candidate_group_context),
    has_respondent_group_context = isTRUE(has_respondent_group_context),
    has_relation_token_context = isTRUE(has_relation_context),
    context_present_masking = context_present_masking,
    has_stage_context = isTRUE(stage_context_enabled),
    has_matchup_context = isTRUE(use_matchup_token),
    has_stage_token = !is.null(ParamsMean$E_stage),
    has_matchup_token = !is.null(ParamsMean$E_matchup),
    has_resp_party_token = !is.null(ParamsMean$E_resp_party),
    has_rel_token = !is.null(ParamsMean$E_rel),
    has_factor_fused_tokens = !is.null(ParamsMean$E_factor_fused_base),
    has_factor_span_tokens = FALSE,
    has_token_family_embedding = !is.null(ParamsMean$E_token_family),
    has_experiment_token = !is.null(ParamsMean$E_experiment) || !is.null(ParamsMean$W_experiment_text),
    has_experiment_id_embedding = !is.null(ParamsMean$E_experiment),
    has_experiment_text_projection = !is.null(ParamsMean$W_experiment_text),
    has_place_context = !is.null(ParamsMean$W_place_context),
    has_time_context = !is.null(ParamsMean$W_time_context),
    has_factor_name_text = !is.null(ParamsMean$W_factor_name_text) && text_semantic_dim > 0L,
    has_level_name_text = !is.null(ParamsMean$W_level_name_text) && text_semantic_dim > 0L,
    has_factor_struct_projection = !is.null(ParamsMean$W_factor_struct) && factor_struct_dim > 0L,
    has_level_struct_projection = !is.null(ParamsMean$W_level_struct) && level_struct_dim > 0L,
    has_covariate_name_text = !is.null(ParamsMean$W_covariate_name_text) && text_semantic_dim > 0L,
    has_covariate_tokens = !is.null(ParamsMean$E_covariate_fused_base),
    has_covariate_fused_tokens = !is.null(ParamsMean$E_covariate_fused_base),
    has_covariate_span_tokens = FALSE,
    has_covariate_missing_token = !is.null(ParamsMean$E_covariate_missing),
    has_covariate_value_text_projection = !is.null(ParamsMean$W_covariate_value_text),
    has_shared_covariate_value_projection = !is.null(ParamsMean$W_covariate_value_shared) ||
      !is.null(ParamsMean$W_covariate_value_basis),
    has_conditioned_covariate_value_encoder = !is.null(ParamsMean$W_covariate_value_basis),
    has_segment_embedding = !is.null(ParamsMean$E_segment),
    has_sep_token = !is.null(ParamsMean$E_sep),
    has_stage_head = !is.null(ParamsMean$W_stage),
    has_ctx_head = !is.null(ParamsMean$W_ctx),
    has_choice_token = !is.null(ParamsMean$E_choice),
    has_respondent_cls = low_rank_interaction_rank > 0L && !is.null(ParamsMean$E_respondent_cls),
    has_candidate_cls = low_rank_interaction_rank > 0L && !is.null(ParamsMean$E_candidate_cls),
    has_low_rank_interaction = isTRUE(neural_has_low_rank_interaction(ParamsMean, list(
      low_rank_interaction_rank = low_rank_interaction_rank
    ))),
    readout_embedding_families = neural_readout_embedding_families(
      low_rank_interaction_rank = low_rank_interaction_rank
    ),
    cross_candidate_encoder = !identical(cross_candidate_encoder_mode, "none"),
    cross_candidate_encoder_mode = cross_candidate_encoder_mode,
    cross_candidate_encoder_note = cross_candidate_encoder_note,
    has_cross_encoder = isTRUE(use_cross_encoder),
    has_cross_attn = !is.null(ParamsMean$W_q_cross),
    has_cross_term = !is.null(ParamsMean$M_cross),
    has_stacked_transformer_layers = isTRUE(neural_has_stacked_standard_transformer(ParamsMean)),
    has_qk_norm = isTRUE(has_qk_norm),
    choice_token_index = 0L,
    likelihood = likelihood,
    universal_foundation_training = isTRUE(universal_enabled),
    universal_mixed_mode = isTRUE(universal_mixed_mode),
    universal_loss_weighting = universal_loss_weighting_diagnostics,
    supported_modes = if (isTRUE(universal_enabled)) {
      unique(universal_task_mode_all)
    } else if (isTRUE(pairwise_mode)) {
      "pairwise"
    } else {
      "single"
    },
    supported_likelihoods = if (isTRUE(universal_enabled)) {
      unique(universal_likelihood_all)
    } else {
      likelihood
    },
    average_case_dr_training = average_case_dr_training,
    average_case_linear_q = average_case_linear_q,
    global_out_dim = as.integer(if (isTRUE(universal_enabled)) universal_global_out_dim else nOutcomes),
    has_ordinal_likelihood = isTRUE(has_ordinal_likelihood),
    ordinal_n_threshold_groups = as.integer(ordinal_n_threshold_groups),
    ordinal_max_n_outcomes = as.integer(ordinal_max_n_outcomes),
    ordinal_max_thresholds = as.integer(ordinal_max_thresholds),
    ordinal_threshold_schema = if (isTRUE(has_ordinal_likelihood)) {
      list(
        parameter = "ordinal_threshold_raw",
        transformed_parameter = "ordinal_thresholds",
        transform = "cumulative_softplus_spacing",
        indexed_by = "experiment_index"
      )
    } else {
      NULL
    },
    fit_metrics = fit_metrics,
    svi_loss_curve = svi_loss_curve,
    svi_steps = resolved_svi_steps,
    svi_steps_completed = svi_steps_completed,
    svi_num_draws = resolved_svi_num_draws,
    balanced_sampling = compact_balanced_sampling_summary(),
    early_stopping = early_stopping_info,
    svi_budget_info = svi_budget_info,
    convergence_diagnostics = convergence_diagnostics,
    optimizer_diagnostics = optimizer_diagnostics,
    gradient_diagnostics = gradient_diagnostics,
    parameter_diagnostics = parameter_diagnostics,
    stage_diagnostics = stage_diagnostics,
    training_seed = as.integer(neural_training_seed),
    training_seed_supplied = isTRUE(neural_training_seed_supplied),
    runtime_provenance = neural_runtime_info,
    model_dims = ModelDims,
    model_depth = ModelDepth,
    init_policy = init_policy,
    residual_mode = residual_mode,
    n_heads = TransformerHeads,
    head_dim = head_dim,
    attention_backend = attention_backend,
    attention_dtype = attention_dtype,
    attention_padding_multiple = as.integer(attention_padding_multiple),
    attention_resolved_backend = attention_resolved_backend,
    attention_fallback_reason = attention_fallback_reason
  )

  if (isTRUE(save_outcome_model) && !isTRUE(neural_oos_eval_internal_flag)) {
    dir.create("./StrategizeInternals", showWarnings = FALSE)
    bundle_meta <- list(
      outcome_model_key = outcome_model_key,
      group = GroupsPool[GroupCounter],
      round = Round_,
      adversarial = isTRUE(adversarial),
      adversarial_model_strategy = adversarial_model_strategy,
      training_seed = as.integer(neural_training_seed),
      runtime_provenance = neural_runtime_info,
      fingerprint = neural_outcome_fingerprint
    )
    tryCatch({
      save_neural_outcome_bundle(
        file = bundle_path,
        theta_mean = theta_mean_num,
        theta_var = param_var,
        neural_model_info = neural_model_info,
        names_list = names_list,
        factor_levels = factor_levels,
        mode = if (isTRUE(pairwise_mode)) "pairwise" else "single",
        fit_metrics = fit_metrics,
        conda_env = conda_env,
        conda_env_required = conda_env_required,
        overwrite = TRUE,
        metadata = bundle_meta
      )
    }, error = function(e) {
      warning(sprintf("Failed to save neural outcome bundle: %s", e$message),
              call. = FALSE)
    })
  }

}
