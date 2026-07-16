#' Build the computational environment for `strategize`
#'
#' Creates the conda environment used by the package's JAX-backed optimization
#' workflow, neural APIs, and foundation checkpoint loading. Users may also
#' create and manage a compatible environment themselves.
#'
#' @details
#' The default \code{backend = "auto"} preserves the existing non-MPS behavior.
#' The \code{"mps"} backend is experimental, opt-in, and supported only on
#' macOS Apple Silicon. It creates a Python 3.13 environment, installs
#' \code{jax-mps}, verifies that JAX reports the MPS backend under
#' \code{JAX_PLATFORMS=mps}, and recreates an existing incompatible environment.
#'
#' @param conda_env Name of the conda environment in which to place the backends.
#'   Defaults to \code{"strategize_env"}.
#' @param conda The path to a conda executable. Using \code{"auto"} allows reticulate
#'   to attempt to automatically find an appropriate conda binary. Defaults to \code{"auto"}.
#'   If creation fails and a conda binary is found on PATH, the function retries with it.
#' @param backend JAX backend selector. Use \code{"auto"} to preserve the default
#'   host-aware behavior, \code{"cpu"} to install plain CPU JAX,
#'   \code{"cuda"} to require Linux and install CUDA-capable JAX wheels when
#'   supported by the NVIDIA driver, or \code{"mps"} to opt in to the
#'   experimental Apple Silicon \code{jax-mps} backend.
#' @param force_reinstall Logical. If \code{TRUE}, remove the named conda
#'   environment if it exists, then recreate and reinstall the backend from the
#'   requested input specifications.
#' @param text_embeddings Optional text embedding profile to install after the
#'   core backend is ready. Use \code{NULL} or \code{FALSE} for no text runtime,
#'   a profile name such as \code{"portable"},
#'   \code{"harrier_oss_v1_0.6b_1024"}, or \code{"qwen3_8b_4096"}, or a list
#'   with \code{profile}, \code{runtime}, and optional \code{model_id}.
#' @param text_embedding_runtime Runtime preference for \code{text_embeddings};
#'   one of \code{"auto"}, \code{"mlx"}, \code{"cuda"}, \code{"rocm"}, or
#'   \code{"cpu"}.
#' @param text_embedding_conda_env Name of the conda environment used for
#'   optional text embedding runtimes. Defaults to
#'   \code{"strategize_torch_env"} so Torch/sentence-transformers packages do
#'   not share the JAX backend environment. Set to \code{conda_env} to preserve
#'   legacy single-environment behavior.
#' @param verbose Logical; if \code{TRUE}, print install milestones and stream
#'   long-running pip/conda output.
#'
#' @return Invisibly returns \code{NULL}. This function is called for its side effects
#'   of creating and configuring a conda environment for \code{strategize}.
#'   This function requires an Internet connection.
#'   You can find a list of conda Python paths via: \code{Sys.which("python")}
#'
#' @examples
#' \dontrun{
#' # Create a conda environment named "strategize_env"
#' # and install the required Python packages (jax, numpy, orbax-checkpoint, etc.)
#' build_backend(conda_env = "strategize_env", conda = "auto")
#'
#' # If you want to specify a particular conda path:
#' # build_backend(conda_env = "strategize_env", conda = "/usr/local/bin/conda")
#'
#' # Experimental Apple Silicon MPS backend:
#' # build_backend(backend = "mps")
#' # Sys.setenv(JAX_PLATFORMS = "mps") may be needed before importing JAX
#' # directly through reticulate in an already-running R session.
#'
#' # Install the host-specific text embedding runtime for a profile:
#' # build_backend(text_embeddings = "harrier_oss_v1_0.6b_1024")
#' # This uses "strategize_torch_env" for Torch/sentence-transformers by default.
#'
#' # Keep scripted runs quiet:
#' # build_backend(text_embeddings = "harrier_oss_v1_0.6b_1024", verbose = FALSE)
#' }
#'
#' @export
#' @md

build_backend <- function(conda_env = "strategize_env", conda = "auto",
                          backend = c("auto", "cpu", "cuda", "mps"),
                          force_reinstall = FALSE,
                          text_embeddings = NULL,
                          text_embedding_runtime = "auto",
                          text_embedding_conda_env = "strategize_torch_env",
                          verbose = TRUE) {
  backend <- match.arg(backend)
  text_embedding_runtime <- cs2step_text_embedding_runtime(text_embedding_runtime)
  text_embedding_request <- cs2step_text_embedding_request(
    text_embeddings = text_embeddings,
    text_embedding_runtime = text_embedding_runtime
  )
  text_embedding_conda_env <- as.character(text_embedding_conda_env %||% conda_env)
  if (length(text_embedding_conda_env) != 1L ||
      is.na(text_embedding_conda_env) ||
      !nzchar(trimws(text_embedding_conda_env))) {
    stop("text_embedding_conda_env must be a non-empty scalar string.", call. = FALSE)
  }
  text_embedding_conda_env <- trimws(text_embedding_conda_env)
  verbose <- cs2step_backend_verbose(verbose)
  if (!is.logical(force_reinstall) || length(force_reinstall) != 1L ||
      is.na(force_reinstall)) {
    stop("force_reinstall must be TRUE or FALSE.", call. = FALSE)
  }
  host <- cs2step_backend_host_info()
  if (identical(backend, "mps") &&
      (!isTRUE(host$is_macos) || !isTRUE(host$is_arm64))) {
    stop(
      "backend = 'mps' requires macOS on Apple Silicon.",
      call. = FALSE
    )
  }
  if (identical(backend, "cuda") && !identical(host$os, "Linux")) {
    stop(
      "backend = 'cuda' requires Linux with a compatible NVIDIA driver.",
      call. = FALSE
    )
  }

  env_registered <- function(conda_bin) {
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
    isTRUE(state$registered)
  }

  create_env <- function(conda_bin) {
    python_version <- if (identical(backend, "mps")) "3.13" else "3.12"
    reticulate::conda_create(envname = conda_env, conda = conda_bin, python_version = python_version)
    invisible(TRUE)
  }

  remove_env <- function(conda_bin) {
    tryCatch(
      reticulate::conda_remove(envname = conda_env, conda = conda_bin),
      error = function(e) {
        if (nzchar(conda_bin %||% "")) {
          suppressWarnings(system2(
            conda_bin,
            c("env", "remove", "-n", conda_env, "-y"),
            stdout = TRUE,
            stderr = TRUE
          ))
        }
        invisible(FALSE)
      }
    )
    invisible(TRUE)
  }

  write_activation_scripts <- function(state) {
    try({
      if (!isTRUE(state$python_exists) || !nzchar(state$python %||% "")) {
        return(invisible(FALSE))
      }
      actdir <- file.path(dirname(dirname(state$python)), "etc", "conda", "activate.d")
      dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
      writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh"))
      mps_script <- file.path(actdir, "10-jax-mps.sh")
      if (identical(backend, "mps")) {
        writeLines("export JAX_PLATFORMS=mps", mps_script)
      } else if (file.exists(mps_script)) {
        unlink(mps_script)
      }
    }, silent = TRUE)
    invisible(TRUE)
  }

  mps_message <- function() {
    if (identical(backend, "mps")) {
      cs2step_backend_message(
        verbose,
        "%s",
        paste(
          "MPS backend selected. R sessions using reticulate directly may need",
          "Sys.setenv(JAX_PLATFORMS = \"mps\") before the first JAX import."
        )
      )
    }
  }

  conda_bin <- cs2step_resolve_conda_binary(conda) %||% conda
  state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  if (isTRUE(force_reinstall) && isTRUE(state$registered)) {
    cs2step_backend_message(verbose,
      "force_reinstall = TRUE; removing conda environment '%s' before rebuilding it.",
      conda_env
    )
    remove_env(conda_bin)
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
    if (isTRUE(state$registered)) {
      stop(
        sprintf(
          "Conda environment '%s' is still registered after forced reinstall removal.",
          conda_env
        ),
        call. = FALSE
      )
    }
  }
  mps_compatibility <- if (identical(backend, "mps") &&
      isTRUE(state$registered) && isTRUE(state$python_exists)) {
    cs2step_backend_mps_compatibility(state)
  } else {
    NULL
  }

  if (isTRUE(state$core_modules_ready) &&
      (!identical(backend, "mps") || isTRUE(mps_compatibility$compatible))) {
    write_activation_scripts(state)
    if (!is.null(text_embedding_request)) {
      cs2step_ensure_text_embedding_request(
        text_embeddings = text_embedding_request,
        text_embedding_runtime = text_embedding_runtime,
        conda_env = text_embedding_conda_env,
        conda = conda_bin,
        force_reinstall = isTRUE(force_reinstall) &&
          !identical(text_embedding_conda_env, conda_env),
        verbose = verbose
      )
    }
    cs2step_backend_message(verbose, "Environment '%s' is ready.", conda_env)
    mps_message()
    return(invisible(NULL))
  }

  if (identical(backend, "mps") &&
      isTRUE(state$registered) && isTRUE(state$python_exists) &&
      !isTRUE(mps_compatibility$compatible)) {
    cs2step_backend_message(verbose,
      "Conda environment '%s' is not compatible with backend = 'mps' (%s); recreating it.",
      conda_env,
      cs2step_describe_mps_compatibility(mps_compatibility)
    )
    remove_env(conda_bin)
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
    mps_compatibility <- NULL
  }

  if (isTRUE(state$registered) && !isTRUE(state$python_exists)) {
    cs2step_backend_message(verbose,
      "Conda environment '%s' is registered but its Python interpreter is missing; recreating it.",
      conda_env
    )
    remove_env(conda_bin)
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
    mps_compatibility <- NULL
  }

  ok <- TRUE
  if (!isTRUE(state$registered)) {
    ok <- tryCatch({
      create_env(conda_bin)
      TRUE
    }, error = function(e) {
      cs2step_backend_message(verbose, "conda_create failed using '%s': %s", conda_bin, e$message)
      FALSE
    })
  }

  if (!ok || !env_registered(conda_bin)) {
    conda_fallback <- Sys.which("conda")
    if (nzchar(conda_fallback) && conda_fallback != conda_bin) {
      cs2step_backend_message(verbose, "Retrying conda_create with: %s", conda_fallback)
      tryCatch({
        create_env(conda_fallback)
      }, error = function(e) {
        cs2step_backend_message(verbose, "conda_create failed using '%s': %s", conda_fallback, e$message)
      })
      if (env_registered(conda_fallback)) {
        conda_bin <- conda_fallback
      }
    }
  }

  state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  if (!isTRUE(state$registered)) {
    stop(
      sprintf("Failed to create conda environment '%s'.\n", conda_env),
      "Try passing conda = '/path/to/conda' or setting RETICULATE_CONDA.\n",
      call. = FALSE
    )
  }
  
  os <- host$os
  msg <- function(...) cs2step_backend_message(verbose, ...)
  
  pip_install <- function(pkgs, ...) {
    cs2step_pip_install_in_conda(
      conda = conda_bin,
      conda_env = conda_env,
      packages = pkgs,
      verbose = verbose,
      ...
    )
    TRUE
  }
  
  # --- (A) Choose CUDA 13 vs 12 *by driver version* and install JAX FIRST ---
  install_jax <- function() {
    if (identical(backend, "cpu")) {
      return(pip_install("jax"))
    }
    if (!identical(os, "Linux")){
      return(pip_install("jax")) 
    }
    
    driver <- cs2step_probe_nvidia_driver()
    drv <- driver$driver_version %||% "unknown"
    drv_major <- driver$driver_major %||% NA_integer_
    
    # Prefer CUDA 13 if the driver is new enough; otherwise CUDA 12; else CPU fallback
    if (!is.na(drv_major) && drv_major >= 580) {
      msg("Driver %s detected (>=580): installing JAX CUDA 13 wheels.", drv)
      tryCatch(pip_install('jax[cuda13]'), error = function(e) {
        msg("CUDA 13 wheels failed (%s); falling back to CUDA 12.", e$message)
        pip_install('jax[cuda12]')
      })
    } else if (!is.na(drv_major) && drv_major >= 525) {
      msg("Driver %s detected (>=525,<580): installing JAX CUDA 12 wheels.", drv)
      pip_install('jax[cuda12]')
    } else {
      msg("Driver %s too old for CUDA wheels; installing CPU-only JAX.", drv)
      pip_install('jax')
    }
  }

  if (identical(backend, "mps") &&
      isTRUE(state$registered) && isTRUE(state$python_exists)) {
    mps_compatibility <- cs2step_backend_mps_compatibility(state)
  }
  
  missing_core <- names(state$core_module_status)[!state$core_module_status]
  needs_mps_install <- identical(backend, "mps") && !isTRUE(mps_compatibility$compatible)
  if (length(missing_core) > 0L || !isTRUE(state$python_exists) || needs_mps_install) {
    # Install JAX first so later dependencies do not pin a CPU-only variant.
    if (identical(backend, "mps")) {
      if (needs_mps_install) {
        pip_install("jax-mps")
      }
    } else {
      install_jax()
    }
    pip_specs <- cs2step_backend_core_pip_packages()
    other_pkgs <- unname(pip_specs[setdiff(names(pip_specs), "jax")])
    pip_install(other_pkgs)
  }

  state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  if (!isTRUE(state$core_modules_ready)) {
    missing_now <- names(state$core_module_status)[!state$core_module_status]
    stop(
      sprintf(
        "Conda environment '%s' is still missing required backend modules: %s",
        conda_env,
        paste(missing_now, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (identical(backend, "mps")) {
    mps_compatibility <- cs2step_backend_mps_compatibility(state)
    if (!isTRUE(mps_compatibility$compatible)) {
      stop(
        sprintf(
          "Conda environment '%s' is not ready for backend = 'mps': %s",
          conda_env,
          cs2step_describe_mps_compatibility(mps_compatibility)
        ),
        call. = FALSE
      )
    }
  }

  write_activation_scripts(state)

  if (!is.null(text_embedding_request)) {
    cs2step_ensure_text_embedding_request(
      text_embeddings = text_embedding_request,
      text_embedding_runtime = text_embedding_runtime,
      conda_env = text_embedding_conda_env,
      conda = conda_bin,
      force_reinstall = isTRUE(force_reinstall) &&
        !identical(text_embedding_conda_env, conda_env),
      verbose = verbose
    )
  }
  
  msg("Environment '%s' is ready.", conda_env)
  mps_message()
}
