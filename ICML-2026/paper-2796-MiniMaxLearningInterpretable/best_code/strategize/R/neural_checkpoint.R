neural_svi_checkpoint_control <- function(mcmc_control) {
  path <- mcmc_control$checkpoint_path %||% NULL
  if (is.null(path)) {
    return(list(
      enabled = FALSE,
      path = NULL,
      resume = FALSE,
      n_checks = 10L,
      compress = FALSE
    ))
  }
  path <- as.character(path)
  if (length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("'neural_mcmc_control$checkpoint_path' must be a non-empty path.", call. = FALSE)
  }

  resume <- mcmc_control$checkpoint_resume
  if (is.null(resume)) {
    resume <- TRUE
  }
  if (!is.logical(resume) || length(resume) != 1L || is.na(resume)) {
    stop("'neural_mcmc_control$checkpoint_resume' must be TRUE or FALSE.", call. = FALSE)
  }

  n_checks <- mcmc_control$checkpoint_n_checks %||% 10L
  n_checks <- suppressWarnings(as.integer(n_checks))
  if (length(n_checks) != 1L || is.na(n_checks) || !is.finite(n_checks) || n_checks < 1L) {
    stop("'neural_mcmc_control$checkpoint_n_checks' must be an integer >= 1.", call. = FALSE)
  }

  compress <- mcmc_control$checkpoint_compress %||% FALSE
  if (!is.logical(compress) || length(compress) != 1L || is.na(compress)) {
    stop("'neural_mcmc_control$checkpoint_compress' must be TRUE or FALSE.", call. = FALSE)
  }

  list(
    enabled = TRUE,
    path = path,
    resume = isTRUE(resume),
    n_checks = as.integer(n_checks),
    compress = isTRUE(compress)
  )
}

neural_svi_checkpoint_strip_control <- function(mcmc_control) {
  out <- mcmc_control %||% list()
  out$checkpoint_path <- NULL
  out$checkpoint_resume <- NULL
  out$checkpoint_n_checks <- NULL
  out$checkpoint_compress <- NULL
  out
}

neural_svi_checkpoint_hash_value <- function(x) {
  x_r <- tryCatch(cs2step_neural_to_r_array(x), error = function(e) x)
  digest::digest(x_r, algo = "xxhash64", serialize = TRUE)
}

neural_svi_checkpoint_normalize_value <- function(x) {
  if (is.list(x) && is.null(class(x))) {
    return(lapply(x, neural_svi_checkpoint_normalize_value))
  }
  tryCatch(cs2step_neural_to_r_array(x), error = function(e) x)
}

neural_svi_checkpoint_fingerprint <- function(fields) {
  fields <- fields %||% list()
  normalized <- lapply(fields, neural_svi_checkpoint_normalize_value)
  hash <- digest::digest(normalized, algo = "xxhash64", serialize = TRUE)
  list(
    schema_version = 1L,
    hash = hash,
    fields = names(fields)
  )
}

# Same-directory tempfile + rename is an atomic overwrite on POSIX, but
# file.rename onto an existing destination fails on Windows -- which would
# abort a long run at the second snapshot save. Fall back to remove-then-
# rename there (a small non-atomic window, recovered by the tmp copy).
neural_svi_checkpoint_commit_tmp <- function(tmp, file, what) {
  if (file.rename(tmp, file)) {
    return(invisible(file))
  }
  if (file.exists(file)) {
    unlink(file, recursive = FALSE, force = TRUE)
    if (file.rename(tmp, file)) {
      return(invisible(file))
    }
  }
  stop(sprintf("Could not atomically write checkpoint %s '%s'.", what, file), call. = FALSE)
}

neural_svi_checkpoint_atomic_save_rds <- function(object, file, compress = FALSE) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile(pattern = paste0(".", basename(file), "-"), tmpdir = dirname(file))
  on.exit({
    if (file.exists(tmp)) {
      unlink(tmp, recursive = FALSE, force = TRUE)
    }
  }, add = TRUE)
  saveRDS(object, file = tmp, compress = compress)
  neural_svi_checkpoint_commit_tmp(tmp, file, "snapshot")
}

neural_svi_checkpoint_atomic_write_json <- function(object, file) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile(pattern = paste0(".", basename(file), "-"), tmpdir = dirname(file))
  on.exit({
    if (file.exists(tmp)) {
      unlink(tmp, recursive = FALSE, force = TRUE)
    }
  }, add = TRUE)
  jsonlite::write_json(object, path = tmp, auto_unbox = TRUE, null = "null", pretty = TRUE)
  neural_svi_checkpoint_commit_tmp(tmp, file, "manifest")
}

neural_svi_checkpoint_manifest_path <- function(path) {
  file.path(path, "manifest.json")
}

neural_svi_checkpoint_snapshot_path <- function(path, type = c("latest", "best")) {
  type <- match.arg(type)
  file.path(path, paste0(type, ".rds"))
}

neural_svi_checkpoint_read_manifest <- function(path) {
  manifest_file <- neural_svi_checkpoint_manifest_path(path)
  if (!file.exists(manifest_file)) {
    return(NULL)
  }
  tryCatch(
    jsonlite::read_json(manifest_file, simplifyVector = FALSE),
    error = function(e) NULL
  )
}

neural_svi_checkpoint_write_manifest <- function(path,
                                                 fingerprint,
                                                 latest = NULL,
                                                 best = NULL) {
  now <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  old <- neural_svi_checkpoint_read_manifest(path) %||% list()
  created_at <- old$created_at %||% now
  snapshots <- old$snapshots %||% list()
  snapshot_summary <- function(snapshot, file) {
    if (is.null(snapshot)) {
      return(NULL)
    }
    best_metric <- as.numeric(snapshot$best_metric %||% NA_real_)
    if (length(best_metric) != 1L || !is.finite(best_metric)) {
      best_metric <- NA_real_
    }
    list(
      file = file,
      completed_step = as.integer(snapshot$completed_step %||% NA_integer_),
      resolved_svi_steps = as.integer(snapshot$resolved_svi_steps %||% NA_integer_),
      best_step = as.integer(snapshot$best_step %||% NA_integer_),
      best_metric = best_metric,
      updated_at = now
    )
  }
  if (!is.null(latest)) {
    snapshots$latest <- snapshot_summary(latest, "latest.rds")
  }
  if (!is.null(best)) {
    snapshots$best <- snapshot_summary(best, "best.rds")
  }
  manifest <- list(
    schema_version = 1L,
    artifact_type = "strategize_neural_svi_checkpoint",
    checkpoint_semantics = "parameter_warm_start",
    created_at = created_at,
    updated_at = now,
    fingerprint = fingerprint,
    snapshots = snapshots
  )
  neural_svi_checkpoint_atomic_write_json(
    manifest,
    neural_svi_checkpoint_manifest_path(path)
  )
}

neural_svi_checkpoint_params_to_r <- function(params) {
  if (is.null(params)) {
    return(NULL)
  }
  params_list <- neural_svi_checkpoint_params_to_list(params)
  if (is.null(params_list)) {
    return(NULL)
  }
  lapply(params_list, function(x) {
    jax_shape <- if (cs2step_has_reticulate() && reticulate::is_py_object(x)) {
      tryCatch(as.integer(reticulate::py_to_r(x$shape)), error = function(e) NULL)
    } else {
      NULL
    }
    out <- tryCatch(cs2step_neural_to_r_array(x), error = function(e) x)
    if (is.array(out) || is.matrix(out)) {
      if (!is.null(jax_shape)) {
        attr(out, "strategize_jax_shape") <- jax_shape
      }
      return(out)
    }
    if (is.numeric(out) || is.integer(out) || is.logical(out)) {
      out_arr <- as.array(out)
      if (!is.null(jax_shape)) {
        attr(out_arr, "strategize_jax_shape") <- jax_shape
      }
      return(out_arr)
    }
    out
  })
}

neural_svi_checkpoint_params_to_list <- function(params) {
  if (is.null(params)) {
    return(NULL)
  }
  if (cs2step_has_reticulate() && reticulate::is_py_object(params)) {
    out <- tryCatch(reticulate::py_to_r(params), error = function(e) NULL)
    if (is.list(out)) {
      return(out)
    }
  }
  tryCatch(as.list(params), error = function(e) NULL)
}

neural_svi_checkpoint_params_to_jax <- function(params) {
  if (is.null(params)) {
    return(NULL)
  }
  params_list <- neural_svi_checkpoint_params_to_list(params)
  if (is.null(params_list)) {
    return(NULL)
  }
  converted <- lapply(params_list, function(x) {
    if (cs2step_has_reticulate() && reticulate::is_py_object(x)) {
      return(x)
    }
    arr <- strenv$jnp$array(x)$astype(strenv$dtj)
    jax_shape <- attr(x, "strategize_jax_shape", exact = TRUE)
    if (!is.null(jax_shape)) {
      arr <- strenv$jnp$reshape(arr, as.list(as.integer(jax_shape)))
    }
    arr
  })
  param_names <- names(params_list)
  if (cs2step_has_reticulate() &&
      !is.null(param_names) &&
      length(param_names) == length(converted) &&
      all(nzchar(param_names))) {
    out <- reticulate::dict()
    for (param_name in param_names) {
      out[[param_name]] <- converted[[param_name]]
    }
    return(out)
  }
  converted
}

neural_svi_checkpoint_make_payload <- function(snapshot_type,
                                               fingerprint,
                                               completed_step,
                                               resolved_svi_steps,
                                               svi_params,
                                               prediction_params = NULL,
                                               loss_history = NULL,
                                               validation_history = NULL,
                                               best_metric = NA_real_,
                                               best_step = NA_integer_,
                                               no_improve_checks = 0L,
                                               early_stopping = NULL,
                                               optimizer_diagnostics = NULL,
                                               svi_budget_info = NULL,
                                               checkpoint_context = NULL) {
  list(
    schema_version = 1L,
    artifact_type = "strategize_neural_svi_checkpoint_snapshot",
    checkpoint_semantics = "parameter_warm_start",
    snapshot_type = snapshot_type,
    created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    fingerprint = fingerprint,
    completed_step = as.integer(completed_step %||% 0L),
    resolved_svi_steps = as.integer(resolved_svi_steps %||% NA_integer_),
    svi_params = neural_svi_checkpoint_params_to_r(svi_params),
    prediction_params = neural_svi_checkpoint_params_to_r(prediction_params),
    loss_history = as.numeric(loss_history %||% numeric(0)),
    validation_history = as.numeric(validation_history %||% numeric(0)),
    best_metric = as.numeric(best_metric %||% NA_real_),
    best_step = as.integer(best_step %||% NA_integer_),
    no_improve_checks = as.integer(no_improve_checks %||% 0L),
    early_stopping = early_stopping %||% NULL,
    optimizer_diagnostics = optimizer_diagnostics %||% NULL,
    svi_budget_info = svi_budget_info %||% NULL,
    checkpoint_context = checkpoint_context %||% NULL
  )
}

neural_svi_checkpoint_save_snapshot <- function(path,
                                                type = c("latest", "best"),
                                                payload,
                                                compress = FALSE) {
  type <- match.arg(type)
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  payload$snapshot_type <- type
  neural_svi_checkpoint_atomic_save_rds(
    payload,
    neural_svi_checkpoint_snapshot_path(path, type),
    compress = compress
  )
  neural_svi_checkpoint_write_manifest(
    path,
    fingerprint = payload$fingerprint,
    latest = if (identical(type, "latest")) payload else NULL,
    best = if (identical(type, "best")) payload else NULL
  )
  invisible(payload)
}

neural_svi_checkpoint_load_snapshot <- function(path, type = c("latest", "best")) {
  type <- match.arg(type)
  file <- neural_svi_checkpoint_snapshot_path(path, type)
  if (!file.exists(file)) {
    return(NULL)
  }
  snapshot <- tryCatch(readRDS(file), error = function(e) NULL)
  if (!is.list(snapshot) ||
      !identical(snapshot$artifact_type, "strategize_neural_svi_checkpoint_snapshot")) {
    return(NULL)
  }
  snapshot
}

neural_svi_checkpoint_assert_fingerprint <- function(snapshot,
                                                     fingerprint,
                                                     checkpoint_path = NULL) {
  if (is.null(snapshot)) {
    return(invisible(TRUE))
  }
  old_hash <- snapshot$fingerprint$hash %||% NULL
  new_hash <- fingerprint$hash %||% NULL
  if (!identical(old_hash, new_hash)) {
    path_msg <- if (!is.null(checkpoint_path) && nzchar(checkpoint_path)) {
      sprintf(" at '%s'", checkpoint_path)
    } else {
      ""
    }
    stop(
      "Neural SVI checkpoint fingerprint mismatch",
      path_msg,
      ". Restartable training requires the same data and training controls. ",
      "Rerun with cache_overwrite = TRUE or delete the stale .inprogress checkpoint directory.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

neural_svi_checkpoint_restore_latest <- function(path, fingerprint) {
  latest <- neural_svi_checkpoint_load_snapshot(path, "latest")
  if (is.null(latest)) {
    return(NULL)
  }
  neural_svi_checkpoint_assert_fingerprint(latest, fingerprint, path)
  latest
}

neural_svi_checkpoint_restore_best <- function(path, fingerprint) {
  best <- neural_svi_checkpoint_load_snapshot(path, "best")
  if (is.null(best)) {
    return(NULL)
  }
  neural_svi_checkpoint_assert_fingerprint(best, fingerprint, path)
  best
}

neural_svi_checkpoint_remove_dir <- function(path) {
  if (is.null(path)) {
    return(invisible(NULL))
  }
  path <- as.character(path)
  if (length(path) != 1L || is.na(path) || !nzchar(path) || !dir.exists(path)) {
    return(invisible(NULL))
  }
  base <- basename(normalizePath(path, winslash = "/", mustWork = FALSE))
  if (!endsWith(base, ".inprogress")) {
    stop(
      "Refusing to remove checkpoint directory because it is not an '.inprogress' path: ",
      path,
      call. = FALSE
    )
  }
  manifest <- neural_svi_checkpoint_read_manifest(path)
  if (!is.list(manifest) ||
      !identical(manifest$artifact_type, "strategize_neural_svi_checkpoint") ||
      !identical(manifest$checkpoint_semantics %||% "parameter_warm_start",
                 "parameter_warm_start")) {
    stop(
      "Refusing to remove checkpoint directory without a valid strategize manifest: ",
      path,
      call. = FALSE
    )
  }
  unlink(path, recursive = TRUE, force = TRUE)
  invisible(NULL)
}
