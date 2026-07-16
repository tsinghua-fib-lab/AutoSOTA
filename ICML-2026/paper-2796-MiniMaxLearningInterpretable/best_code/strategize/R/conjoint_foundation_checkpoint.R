cs_foundation_is_checkpoint_dir <- function(path) {
  dir.exists(path) && file.exists(file.path(path, "manifest.json"))
}

cs_foundation_tuple <- function(x) {
  x <- as.integer(x %||% integer(0))
  if (length(x) == 0L) {
    return(reticulate::tuple())
  }
  do.call(reticulate::tuple, as.list(x))
}

cs_foundation_import_orbax_v1 <- function() {
  ocp <- tryCatch(
    reticulate::import("orbax.checkpoint.experimental.v1", convert = FALSE),
    error = function(e) NULL
  )
  if (!is.null(ocp) && reticulate::py_has_attr(ocp, "load_pytree")) {
    return(ocp)
  }
  tryCatch(
    reticulate::import("orbax.checkpoint", convert = FALSE),
    error = function(e) NULL
  )
}

cs_foundation_array_abstract_leaf <- function(meta) {
  shape <- as.integer(meta$shape %||% integer(0))
  dtype <- as.character(meta$dtype %||% "float32")
  strenv$jax$ShapeDtypeStruct(
    shape = cs_foundation_tuple(shape),
    dtype = strenv$np$dtype(dtype)
  )
}

cs_foundation_array_restore_leaf <- function(meta, ocp) {
  dtype <- as.character(meta$dtype %||% "float32")
  ocp$RestoreArgs(
    restore_type = strenv$np$ndarray,
    dtype = strenv$np$dtype(dtype)
  )
}

cs_foundation_build_abstract_tree <- function(array_manifest) {
  groups <- array_manifest$groups %||% list()
  tree_groups <- lapply(groups, function(group_meta) {
    group_tree <- list()
    params_meta <- group_meta$params %||% list()
    if (length(params_meta) > 0L) {
      group_tree$params <- lapply(params_meta, cs_foundation_array_abstract_leaf)
    }
    if (!is.null(group_meta$theta_mean)) {
      group_tree$theta_mean <- cs_foundation_array_abstract_leaf(group_meta$theta_mean)
    }
    if (!is.null(group_meta$theta_var)) {
      group_tree$theta_var <- cs_foundation_array_abstract_leaf(group_meta$theta_var)
    }
    text_meta <- group_meta$text_registry %||% list()
    if (length(text_meta) > 0L) {
      group_tree$text_registry <- lapply(text_meta, cs_foundation_array_abstract_leaf)
    }
    group_tree
  })
  list(groups = tree_groups)
}

cs_foundation_build_restore_args_tree <- function(array_manifest, ocp) {
  groups <- array_manifest$groups %||% list()
  tree_groups <- lapply(groups, function(group_meta) {
    group_tree <- list()
    params_meta <- group_meta$params %||% list()
    if (length(params_meta) > 0L) {
      group_tree$params <- lapply(params_meta, cs_foundation_array_restore_leaf, ocp = ocp)
    }
    if (!is.null(group_meta$theta_mean)) {
      group_tree$theta_mean <- cs_foundation_array_restore_leaf(group_meta$theta_mean, ocp)
    }
    if (!is.null(group_meta$theta_var)) {
      group_tree$theta_var <- cs_foundation_array_restore_leaf(group_meta$theta_var, ocp)
    }
    text_meta <- group_meta$text_registry %||% list()
    if (length(text_meta) > 0L) {
      group_tree$text_registry <- lapply(text_meta, cs_foundation_array_restore_leaf, ocp = ocp)
    }
    group_tree
  })
  list(groups = tree_groups)
}

cs_foundation_orbax_restore_path <- function(path) {
  pytree_path <- file.path(path, "pytree")
  if (dir.exists(pytree_path)) {
    return(pytree_path)
  }
  path
}

cs_foundation_orbax_load_tree <- function(path, abstract_tree, array_manifest) {
  ocp <- tryCatch(
    reticulate::import("orbax.checkpoint", convert = FALSE),
    error = function(e) NULL
  )
  if (is.null(ocp)) {
    stop(
      "Loading this foundation checkpoint requires Python module 'orbax.checkpoint'.\n",
      "Run strategize::build_backend() to install the pip package 'orbax-checkpoint'.",
      call. = FALSE
    )
  }
  if (reticulate::py_has_attr(ocp, "PyTreeCheckpointer")) {
    restore_path <- cs_foundation_orbax_restore_path(path)
    restore_args <- if (reticulate::py_has_attr(ocp, "RestoreArgs")) {
      cs_foundation_build_restore_args_tree(array_manifest, ocp)
    } else {
      NULL
    }
    if (!is.null(restore_args)) {
      return(ocp$PyTreeCheckpointer()$restore(
        restore_path,
        item = abstract_tree,
        restore_args = restore_args
      ))
    }
    return(ocp$PyTreeCheckpointer()$restore(restore_path, item = abstract_tree))
  }
  ocp_v1 <- cs_foundation_import_orbax_v1()
  if (!is.null(ocp_v1) && reticulate::py_has_attr(ocp_v1, "load_pytree")) {
    return(ocp_v1$load_pytree(path, abstract_tree))
  }
  stop("Installed 'orbax.checkpoint' does not expose a supported PyTree loader.", call. = FALSE)
}

cs_foundation_py_get_item <- function(x, key) {
  reticulate::py_get_item(x, as.character(key))
}

cs_foundation_restore_text_matrix <- function(py_group, name, meta) {
  text_tree <- tryCatch(cs_foundation_py_get_item(py_group, "text_registry"), error = function(e) NULL)
  if (is.null(text_tree)) {
    return(NULL)
  }
  value <- tryCatch(cs_foundation_py_get_item(text_tree, name), error = function(e) NULL)
  if (is.null(value)) {
    return(NULL)
  }
  out <- as.matrix(cs2step_neural_to_r_array(value))
  # Assigning character(0) dimnames to a non-empty matrix errors; only
  # restore names that were actually recorded.
  meta_rownames <- as.character(meta$rownames %||% character(0))
  if (length(meta_rownames) == nrow(out)) {
    rownames(out) <- meta_rownames
  }
  meta_colnames <- as.character(meta$colnames %||% character(0))
  if (length(meta_colnames) == ncol(out)) {
    colnames(out) <- meta_colnames
  }
  out
}

cs_foundation_param_schema <- function(params, model_info) {
  if (is.null(params) || !is.list(params)) {
    return(NULL)
  }
  model_info <- model_info %||% list()
  has_schema <- !is.null(model_info$param_names) &&
    !is.null(model_info$param_shapes) &&
    !is.null(model_info$param_sizes) &&
    !is.null(model_info$param_offsets)
  if (isTRUE(has_schema)) {
    param_sizes <- as.integer(unlist(model_info$param_sizes, use.names = FALSE))
    param_offsets <- as.integer(unlist(model_info$param_offsets, use.names = FALSE))
    return(list(
      param_names = as.character(model_info$param_names),
      param_shapes = lapply(model_info$param_shapes, as.integer),
      param_sizes = param_sizes,
      param_offsets = param_offsets,
      n_params = as.integer(model_info$n_params %||% sum(param_sizes))
    ))
  }
  if (!is.null(model_info$n_factors) && !is.null(model_info$model_depth)) {
    return(neural_build_param_schema(
      params = params,
      n_factors = model_info$n_factors,
      model_depth = model_info$model_depth,
      factor_tokenization = neural_factor_tokenization(model_info)
    ))
  }
  NULL
}

cs_foundation_restore_theta_mean <- function(params, model_info) {
  schema <- cs_foundation_param_schema(params = params, model_info = model_info)
  if (is.null(schema)) {
    return(NULL)
  }
  missing_params <- setdiff(as.character(schema$param_names), names(params))
  if (length(missing_params) > 0L) {
    return(NULL)
  }
  theta <- neural_flatten_params(params, schema, dtype = strenv$dtj)
  as.numeric(cs2step_neural_to_r_array(theta))
}

cs_foundation_load_checkpoint_dir <- function(path,
                                             conda_env = "strategize_env",
                                             conda_env_required = TRUE,
                                             preload_params = FALSE) {
  manifest_path <- file.path(path, "manifest.json")
  metadata_path <- file.path(path, "metadata.rds")
  arrays_path <- file.path(path, "arrays")
  if (!file.exists(metadata_path)) {
    stop("Foundation checkpoint is missing metadata.rds.", call. = FALSE)
  }
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  if (!identical(manifest$writer$package %||% NULL, "preference.fm")) {
    stop("Unrecognized foundation checkpoint writer.", call. = FALSE)
  }
  manifest_schema_version <- suppressWarnings(
    as.integer(manifest$schema_version %||% 1L)
  )
  if (is.na(manifest_schema_version) || manifest_schema_version > 2L) {
    stop(sprintf(paste0(
      "Foundation bundle manifest declares schema_version %s, newer than this ",
      "strategize supports (<= 2). Update strategize/preference.fm to load it."
    ), as.character(manifest$schema_version %||% NA)), call. = FALSE)
  }
  if (!dir.exists(arrays_path)) {
    stop("Foundation checkpoint is missing arrays/.", call. = FALSE)
  }
  if (!"jnp" %in% ls(envir = strenv) || !"np" %in% ls(envir = strenv)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
  }

  bundle <- readRDS(metadata_path)
  if (!is.list(bundle) || is.null(bundle$groups)) {
    stop("Unrecognized foundation checkpoint metadata.", call. = FALSE)
  }
  bundle$metadata <- cs2step_restore_text_embedding_metadata(
    bundle$metadata %||% list(),
    conda_env = conda_env,
    required = FALSE
  )

  abstract_tree <- cs_foundation_build_abstract_tree(manifest$arrays %||% list())
  restored <- cs_foundation_orbax_load_tree(
    arrays_path,
    abstract_tree,
    manifest$arrays %||% list()
  )
  restored_groups <- cs_foundation_py_get_item(restored, "groups")
  manifest_groups <- manifest$arrays$groups %||% list()
  group_keys <- vapply(manifest_groups, function(x) as.character(x$group_key), character(1))

  for (group_id in names(manifest_groups)) {
    group_key <- group_keys[[group_id]]
    if (is.null(bundle$groups[[group_key]])) {
      next
    }
    py_group <- cs_foundation_py_get_item(restored_groups, group_id)
    if (!is.null(manifest_groups[[group_id]]$theta_mean)) {
      theta_mean <- cs_foundation_py_get_item(py_group, "theta_mean")
      bundle$groups[[group_key]]$fit$theta_mean <- as.numeric(cs2step_neural_to_r_array(theta_mean))
    }
    params_meta <- manifest_groups[[group_id]]$params %||% list()
    if (length(params_meta) > 0L) {
      py_params <- cs_foundation_py_get_item(py_group, "params")
      params <- lapply(names(params_meta), function(param_name) {
        cs2step_neural_to_r_array(cs_foundation_py_get_item(py_params, param_name))
      })
      names(params) <- names(params_meta)
      model_info <- bundle$groups[[group_key]]$fit$neural_model_info %||% list()
      if (is.null(bundle$groups[[group_key]]$fit$theta_mean)) {
        theta_mean <- tryCatch(
          cs_foundation_restore_theta_mean(params = params, model_info = model_info),
          error = function(e) NULL
        )
      } else {
        theta_mean <- NULL
      }
      model_info$params <- params
      bundle$groups[[group_key]]$fit$neural_model_info <- model_info
      bundle$groups[[group_key]]$fit$params <- params
      if (!is.null(theta_mean)) {
        bundle$groups[[group_key]]$fit$theta_mean <- theta_mean
      }
    }
    if (!is.null(manifest_groups[[group_id]]$theta_var)) {
      theta_var <- cs_foundation_py_get_item(py_group, "theta_var")
      bundle$groups[[group_key]]$fit$theta_var <- as.numeric(cs2step_neural_to_r_array(theta_var))
    }

    text_placeholders <- bundle$groups[[group_key]]$text_registry %||% NULL
    text_meta <- manifest_groups[[group_id]]$text_registry %||% list()
    if (!is.null(text_placeholders) && length(text_meta) > 0L) {
      for (text_name in names(text_meta)) {
        restored_text <- cs_foundation_restore_text_matrix(
          py_group = py_group,
          name = text_name,
          meta = text_meta[[text_name]]
        )
        if (!is.null(restored_text)) {
          bundle$groups[[group_key]]$text_registry[[text_name]] <- restored_text
        }
      }
    }
  }

  bundle$groups <- lapply(bundle$groups, cs_foundation_unpack_group,
                          conda_env = conda_env,
                          conda_env_required = conda_env_required,
                          preload_params = preload_params)
  class(bundle) <- "conjoint_foundation_model"
  bundle
}
