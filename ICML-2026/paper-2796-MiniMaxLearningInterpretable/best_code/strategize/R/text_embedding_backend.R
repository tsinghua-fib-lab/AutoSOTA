#' Inspect a host-aware text-embedding backend
#'
#' @param conda_env Conda env name used for the Python runtime. Defaults to
#'   \code{"strategize_torch_env"} so text embedding packages can be isolated
#'   from the JAX backend environment.
#' @param family Optional embedding-family selector. When \code{NULL}, the
#'   selected profile supplies the family.
#' @param runtime Runtime preference. Use \code{"auto"} to resolve per host, or
#'   choose one of \code{"mlx"}, \code{"cuda"}, \code{"rocm"}, or \code{"cpu"} explicitly.
#' @param profile Embedding profile. Use \code{"portable"} for the default
#'   portable profile or a registered profile such as
#'   \code{"harrier_oss_v1_0.6b_1024"} or \code{"qwen3_8b_4096"}.
#' @param model_id Optional model id override applied to every compatible
#'   candidate in the selected profile.
#' @param conda Conda binary to use. Defaults to \code{"auto"}.
#'
#' @return A serializable list describing the host, Python env health, candidate
#'   text-embedding runtimes, and the selected backend.
#' @export
inspect_text_embedding_backend <- function(conda_env = "strategize_torch_env",
                                           family = NULL,
                                           runtime = "auto",
                                           profile = "portable",
                                           model_id = NULL,
                                           conda = "auto") {
  runtime <- cs2step_text_embedding_runtime(runtime)
  profile <- cs2step_text_embedding_profile(profile)
  family <- cs2step_text_embedding_family(family %||%
    cs2step_text_embedding_profile_spec(profile)$family)
  host <- cs2step_collect_text_embedding_host_info(conda_env = conda_env, conda = conda)
  inspected <- cs2step_inspect_text_embedding_candidates(
    host = host,
    family = family,
    runtime = runtime,
    profile = profile,
    model_id = model_id
  )
  structure(inspected, class = "strategize_text_embedding_backend")
}

#' Build a host-aware text-embedding backend
#'
#' @param conda_env Conda env name used for the Python runtime. Defaults to
#'   \code{"strategize_torch_env"} so text embedding packages can be isolated
#'   from the JAX backend environment.
#' @param family Optional embedding-family selector. When \code{NULL}, the
#'   selected profile supplies the family.
#' @param runtime Runtime preference. Use \code{"auto"} to resolve per host, or
#'   choose one of \code{"mlx"}, \code{"cuda"}, \code{"rocm"}, or \code{"cpu"} explicitly.
#' @param profile Embedding profile. Use \code{"portable"} for the default
#'   portable profile or a registered profile such as
#'   \code{"harrier_oss_v1_0.6b_1024"} or \code{"qwen3_8b_4096"}.
#' @param model_id Optional model id override applied to every compatible
#'   candidate in the selected profile.
#' @param cache_dir Optional directory for cached text embeddings.
#' @param cache_only Logical; if \code{TRUE}, missing cache entries error before
#'   a model runtime is imported.
#' @param batch_size Number of texts to encode per model call.
#' @param normalize Logical; if \code{TRUE}, L2-normalize canonical embeddings.
#' @param required Logical; if \code{TRUE}, ensure the selected runtime is usable
#'   immediately. If \code{FALSE}, return a lazy backend that validates on first use.
#' @param verbose Logical; if \code{TRUE}, print install milestones and stream
#'   long-running pip/conda output.
#' @param conda Conda binary to use. Defaults to \code{"auto"}.
#'
#' @return A list with \code{$fn}, \code{$spec}, and \code{$runtime}.
#' @export
build_text_embedding_backend <- function(conda_env = "strategize_torch_env",
                                         family = NULL,
                                         runtime = "auto",
                                         profile = "portable",
                                         model_id = NULL,
                                         cache_dir = NULL,
                                         cache_only = FALSE,
                                         batch_size = 64L,
                                         normalize = TRUE,
                                         required = TRUE,
                                         verbose = TRUE,
                                         conda = "auto") {
  runtime <- cs2step_text_embedding_runtime(runtime)
  profile <- cs2step_text_embedding_profile(profile)
  family <- cs2step_text_embedding_family(family %||%
    cs2step_text_embedding_profile_spec(profile)$family)
  verbose <- cs2step_backend_verbose(verbose)

  if (isTRUE(required)) {
    cs2step_ensure_basic_python_conda_env(
      conda_env = conda_env,
      conda = conda,
      verbose = verbose,
      context = "text embedding Python runtime"
    )
  }

  inspected <- inspect_text_embedding_backend(
    conda_env = conda_env,
    family = family,
    runtime = runtime,
    profile = profile,
    model_id = model_id,
    conda = conda
  )
  spec <- inspected$selected

  if (is.null(spec) || identical(spec$status, "unavailable")) {
    stop(
      sprintf(
        "No compatible text embedding backend is available for family '%s' and runtime '%s'.\n%s",
        family,
        runtime,
        paste(unique(inspected$issues %||% character(0)), collapse = "\n")
      ),
      call. = FALSE
    )
  }

  if (isTRUE(required) && !cs2step_text_embedding_runtime_ready(spec)) {
    cs2step_ensure_text_embedding_runtime(spec, verbose = verbose)
    inspected <- inspect_text_embedding_backend(
      conda_env = conda_env,
      family = family,
      runtime = runtime,
      profile = profile,
      model_id = model_id,
      conda = conda
    )
    spec <- inspected$selected
    if (is.null(spec) || !identical(spec$status, "ready")) {
      label_use <- if (is.null(spec)) "unknown" else spec$label %||% "unknown"
      stop(
        sprintf(
          "Selected text embedding backend '%s' is not ready after installation.\n%s",
          label_use,
          paste(unique(inspected$issues %||% character(0)), collapse = "\n")
        ),
        call. = FALSE
      )
    }
  }

  spec <- modifyList(spec, list(
    normalize = isTRUE(normalize),
    frozen = TRUE,
    trainable = FALSE,
    embedding_strategy = "frozen_pretrained_schema_text"
  ))
  fn <- cs2step_build_text_embedding_fn(
    spec = spec,
    cache_dir = cache_dir,
    cache_only = cache_only,
    batch_size = batch_size,
    verbose = verbose
  )
  list(fn = fn, spec = spec, runtime = inspected)
}

cs2step_text_embedding_family <- function(family) {
  family <- tolower(as.character(family %||% "qwen3"))
  valid <- c("qwen3", "harrier")
  if (!family %in% valid) {
    stop(
      sprintf("Unsupported text embedding family '%s'. Valid families: %s.",
              family, paste(valid, collapse = ", ")),
      call. = FALSE
    )
  }
  family
}

cs2step_text_embedding_runtime <- function(runtime) {
  runtime <- tolower(as.character(runtime %||% "auto"))
  valid <- c("auto", "mlx", "cuda", "rocm", "cpu")
  if (!runtime %in% valid) {
    stop(
      sprintf("Unsupported text embedding runtime '%s'. Valid values: %s.",
              runtime, paste(valid, collapse = ", ")),
      call. = FALSE
    )
  }
  runtime
}

cs2step_backend_verbose <- function(verbose) {
  if (!is.logical(verbose) || length(verbose) != 1L || is.na(verbose)) {
    stop("verbose must be TRUE or FALSE.", call. = FALSE)
  }
  isTRUE(verbose)
}

cs2step_backend_message <- function(verbose, ...) {
  if (!isTRUE(verbose)) {
    return(invisible(FALSE))
  }
  message(sprintf(...))
  invisible(TRUE)
}

cs2step_text_embedding_profile_registry <- function() {
  st_modules <- c("sentence_transformers", "transformers", "torch", "numpy")
  list(
    portable = list(
      profile = "portable",
      family = "qwen3",
      canonical_dim = 1024L,
      cache_key_version = cs2step_text_embedding_backend_version(),
      auto_cpu_fallback = TRUE,
      candidates = list(
        mlx = list(
          label = "mlx",
          backend = "mlx",
          device = "metal",
          model_id = "mlx-community/Qwen3-Embedding-8B-mxfp8",
          raw_dim = 4096L,
          required_modules = c("mlx_embeddings", "mlx.core", "numpy"),
          install_packages = c("mlx", "mlx-embeddings"),
          host_constraints = list(os = "Darwin", machine = "arm64")
        ),
        cuda = list(
          label = "sentence_transformers_cuda",
          backend = "sentence_transformers",
          device = "cuda",
          model_id = "Qwen/Qwen3-Embedding-0.6B",
          raw_dim = 1024L,
          required_modules = st_modules,
          install_packages = c("sentence-transformers", "transformers")
        ),
        rocm = list(
          label = "sentence_transformers_rocm",
          backend = "sentence_transformers",
          device = "rocm",
          model_id = "Qwen/Qwen3-Embedding-0.6B",
          raw_dim = 1024L,
          required_modules = st_modules,
          install_packages = c("sentence-transformers", "transformers")
        ),
        cpu = list(
          label = "sentence_transformers_cpu",
          backend = "sentence_transformers",
          device = "cpu",
          model_id = "Qwen/Qwen3-Embedding-0.6B",
          raw_dim = 1024L,
          required_modules = st_modules,
          install_packages = c("torch", "sentence-transformers", "transformers")
        )
      )
    ),
    qwen3_0.6b_1024 = list(
      alias = "portable"
    ),
    harrier_oss_v1_0.6b_1024 = list(
      profile = "harrier_oss_v1_0.6b_1024",
      family = "harrier",
      canonical_dim = 1024L,
      cache_key_version = cs2step_text_embedding_backend_version(),
      auto_cpu_fallback = TRUE,
      candidates = list(
        cuda = list(
          label = "sentence_transformers_cuda_harrier_oss_v1_0.6b_1024",
          backend = "sentence_transformers",
          device = "cuda",
          model_id = "microsoft/harrier-oss-v1-0.6b",
          raw_dim = 1024L,
          required_modules = st_modules,
          install_packages = c("sentence-transformers", "transformers")
        ),
        rocm = list(
          label = "sentence_transformers_rocm_harrier_oss_v1_0.6b_1024",
          backend = "sentence_transformers",
          device = "rocm",
          model_id = "microsoft/harrier-oss-v1-0.6b",
          raw_dim = 1024L,
          required_modules = st_modules,
          install_packages = c("sentence-transformers", "transformers")
        ),
        cpu = list(
          label = "sentence_transformers_cpu_harrier_oss_v1_0.6b_1024",
          backend = "sentence_transformers",
          device = "cpu",
          model_id = "microsoft/harrier-oss-v1-0.6b",
          raw_dim = 1024L,
          required_modules = st_modules,
          install_packages = c("torch", "sentence-transformers", "transformers")
        )
      )
    ),
    qwen3_8b_4096 = list(
      profile = "qwen3_8b_4096",
      family = "qwen3",
      canonical_dim = 4096L,
      cache_key_version = cs2step_text_embedding_backend_version(),
      auto_cpu_fallback = FALSE,
      candidates = list(
        mlx = list(
          label = "mlx_qwen3_8b_4096",
          backend = "mlx",
          device = "metal",
          model_id = "mlx-community/Qwen3-Embedding-8B-mxfp8",
          raw_dim = 4096L,
          required_modules = c("mlx_embeddings", "mlx.core", "numpy"),
          install_packages = c("mlx", "mlx-embeddings"),
          host_constraints = list(os = "Darwin", machine = "arm64")
        ),
        cuda = list(
          label = "sentence_transformers_cuda_qwen3_8b_4096",
          backend = "sentence_transformers",
          device = "cuda",
          model_id = "Qwen/Qwen3-Embedding-8B",
          raw_dim = 4096L,
          required_modules = st_modules,
          install_packages = c("sentence-transformers", "transformers")
        ),
        rocm = list(
          label = "sentence_transformers_rocm_qwen3_8b_4096",
          backend = "sentence_transformers",
          device = "rocm",
          model_id = "Qwen/Qwen3-Embedding-8B",
          raw_dim = 4096L,
          required_modules = st_modules,
          install_packages = c("sentence-transformers", "transformers")
        ),
        cpu = list(
          label = "sentence_transformers_cpu_qwen3_8b_4096",
          backend = "sentence_transformers",
          device = "cpu",
          model_id = "Qwen/Qwen3-Embedding-8B",
          raw_dim = 4096L,
          required_modules = st_modules,
          install_packages = c("torch", "sentence-transformers", "transformers")
        )
      )
    )
  )
}

cs2step_text_embedding_profile_names <- function() {
  names(cs2step_text_embedding_profile_registry())
}

cs2step_text_embedding_profile <- function(profile) {
  profile <- tolower(as.character(profile %||% "portable"))
  valid <- cs2step_text_embedding_profile_names()
  if (!profile %in% valid) {
    stop(
      sprintf(
        "Unsupported text embedding profile '%s'. Valid profiles: %s.",
        profile,
        paste(valid, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  profile
}

cs2step_text_embedding_profile_spec <- function(profile) {
  registry <- cs2step_text_embedding_profile_registry()
  profile <- cs2step_text_embedding_profile(profile)
  spec <- registry[[profile]]
  if (!is.null(spec$alias)) {
    spec <- registry[[spec$alias]]
    spec$profile <- profile
  }
  spec
}

cs2step_text_embedding_install_packages <- function(spec, fallback = character(0)) {
  packages <- as.character(spec$install_packages %||% character(0))
  packages <- packages[nzchar(packages)]
  if (!length(packages)) {
    packages <- as.character(fallback %||% character(0))
  }
  packages
}

cs2step_text_embedding_runtime_ready <- function(spec) {
  identical(as.character(spec$status %||% ""), "ready")
}

cs2step_text_embedding_request <- function(text_embeddings = NULL,
                                           text_embedding_runtime = "auto") {
  runtime_default <- cs2step_text_embedding_runtime(text_embedding_runtime)
  if (is.null(text_embeddings) || identical(text_embeddings, FALSE)) {
    return(NULL)
  }
  if (is.character(text_embeddings) && length(text_embeddings) == 1L) {
    value <- tolower(trimws(text_embeddings))
    if (!nzchar(value) || value %in% c("none", "off", "false", "0", "no")) {
      return(NULL)
    }
    return(list(
      profile = cs2step_text_embedding_profile(value),
      runtime = runtime_default,
      family = NULL,
      model_id = NULL,
      normalize = TRUE,
      frozen = TRUE
    ))
  }
  if (!is.list(text_embeddings)) {
    stop(
      "text_embeddings must be NULL, FALSE, a profile name, or a list with profile/runtime/model_id fields.",
      call. = FALSE
    )
  }
  profile <- cs2step_text_embedding_profile(text_embeddings$profile %||% "portable")
  runtime <- cs2step_text_embedding_runtime(text_embeddings$runtime %||% runtime_default)
  family <- text_embeddings$family %||% NULL
  if (!is.null(family)) {
    family <- cs2step_text_embedding_family(family)
  }
  model_id <- text_embeddings$model_id %||% NULL
  if (!is.null(model_id)) {
    model_id <- as.character(model_id)
    if (length(model_id) != 1L || is.na(model_id) || !nzchar(trimws(model_id))) {
      stop("text_embeddings$model_id must be a non-empty scalar string when supplied.", call. = FALSE)
    }
  }
  normalize <- isTRUE(text_embeddings$normalize %||% TRUE)
  list(
    profile = profile,
    runtime = runtime,
    family = family,
    model_id = model_id,
    normalize = normalize,
    frozen = TRUE
  )
}

cs2step_ensure_text_embedding_request <- function(text_embeddings = NULL,
                                                  text_embedding_runtime = "auto",
                                                  conda_env = "strategize_torch_env",
                                                  conda = "auto",
                                                  force_reinstall = FALSE,
                                                  verbose = TRUE) {
  verbose <- cs2step_backend_verbose(verbose)
  request <- cs2step_text_embedding_request(
    text_embeddings = text_embeddings,
    text_embedding_runtime = text_embedding_runtime
  )
  if (is.null(request)) {
    return(invisible(NULL))
  }
  cs2step_ensure_basic_python_conda_env(
    conda_env = conda_env,
    conda = conda,
    force_reinstall = force_reinstall,
    verbose = verbose,
    context = "text embedding Python runtime"
  )
  inspected <- inspect_text_embedding_backend(
    conda_env = conda_env,
    family = request$family,
    runtime = request$runtime,
    profile = request$profile,
    model_id = request$model_id,
    conda = conda
  )
  spec <- inspected$selected
  if (is.null(spec) || identical(spec$status, "unavailable")) {
    stop(
      sprintf(
        "No compatible text embedding backend is available for profile '%s' and runtime '%s'.\n%s",
        request$profile,
        request$runtime,
        paste(unique(inspected$issues %||% character(0)), collapse = "\n")
      ),
      call. = FALSE
    )
  }
  if (!cs2step_text_embedding_runtime_ready(spec)) {
    cs2step_backend_message(
      verbose,
      "Preparing text embedding backend: profile=%s; backend=%s; device=%s; model=%s",
      spec$profile,
      spec$backend,
      spec$device %||% "unknown",
      spec$model_id
    )
    cs2step_ensure_text_embedding_runtime(spec, verbose = verbose)
    inspected <- inspect_text_embedding_backend(
      conda_env = conda_env,
      family = request$family,
      runtime = request$runtime,
      profile = request$profile,
      model_id = request$model_id,
      conda = conda
    )
    spec <- inspected$selected
    if (is.null(spec) || !identical(spec$status, "ready")) {
      label_use <- if (is.null(spec)) "unknown" else spec$label %||% "unknown"
      stop(
        sprintf(
          "Selected text embedding backend '%s' is not ready after installation.\n%s",
          label_use,
          paste(unique(inspected$issues %||% character(0)), collapse = "\n")
        ),
        call. = FALSE
      )
    }
  }
  spec <- modifyList(spec, list(
    normalize = isTRUE(request$normalize),
    frozen = TRUE,
    trainable = FALSE,
    embedding_strategy = "frozen_pretrained_schema_text"
  ))
  cs2step_backend_message(
    verbose,
    "Text embedding backend '%s' is ready (profile=%s, runtime=%s, model=%s).",
    spec$label %||% spec$backend,
    spec$profile,
    request$runtime,
    spec$model_id
  )
  invisible(spec)
}

cs2step_text_embedding_backend_version <- function() 2L

cs2step_resolve_conda_binary <- function(conda = "auto") {
  if (!cs2step_has_reticulate()) {
    return(NULL)
  }
  conda_use <- conda %||% "auto"
  conda_bin <- tryCatch(reticulate::conda_binary(conda = conda_use), error = function(e) NULL)
  if (!nzchar(conda_bin %||% "")) {
    env_bin <- Sys.getenv("RETICULATE_CONDA", unset = "")
    if (nzchar(env_bin)) {
      conda_bin <- path.expand(env_bin)
    }
  }
  if (!nzchar(conda_bin %||% "")) {
    path_bin <- Sys.which("conda")
    if (nzchar(path_bin)) {
      conda_bin <- path_bin
    }
  }
  if (!nzchar(conda_bin %||% "")) {
    return(NULL)
  }
  path.expand(conda_bin)
}

cs2step_python_probe <- function(python, code, env = character()) {
  python <- path.expand(as.character(python %||% ""))
  if (!nzchar(python) || !file.exists(python)) {
    return(list(status = 127L, output = "python interpreter not found"))
  }
  script <- tempfile(pattern = "strategize-python-probe-", fileext = ".py")
  on.exit(unlink(script), add = TRUE)
  writeLines(code, script, useBytes = TRUE)
  output <- tryCatch(
    suppressWarnings(system2(
      python,
      script,
      stdout = TRUE,
      stderr = TRUE,
      env = as.character(env %||% character())
    )),
    error = function(e) structure(conditionMessage(e), status = 127L)
  )
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  list(status = as.integer(status), output = as.character(output))
}

cs2step_python_version_major_minor <- function(python) {
  code <- paste(
    "import sys",
    "print('PYTHON_VERSION::{}.{}'.format(sys.version_info.major, sys.version_info.minor))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  line <- grep("^PYTHON_VERSION::", probe$output, value = TRUE)
  version <- if (length(line) > 0L) sub("^PYTHON_VERSION::", "", line[[1]]) else ""
  list(
    ok = identical(probe$status, 0L) && nzchar(version),
    major_minor = version,
    status = probe$status,
    output = probe$output
  )
}

cs2step_python_distribution_probe <- function(python, distributions) {
  distributions <- unique(as.character(distributions %||% character(0)))
  if (!length(distributions)) {
    return(list(ok = logical(0), version = character(0), details = character(0), status = 0L))
  }
  quoted <- paste(sprintf("'%s'", gsub("'", "\\\\'", distributions, fixed = TRUE)), collapse = ", ")
  code <- paste(
    "from importlib import metadata",
    sprintf("dists = [%s]", quoted),
    "for name in dists:",
    "    try:",
    "        print('OK::' + name + '::' + metadata.version(name))",
    "    except Exception as exc:",
    "        print('FAIL::' + name + '::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  ok <- setNames(rep(FALSE, length(distributions)), distributions)
  version <- setNames(rep("", length(distributions)), distributions)
  details <- setNames(rep("", length(distributions)), distributions)
  for (line in probe$output) {
    if (!nzchar(line)) {
      next
    }
    if (startsWith(line, "OK::")) {
      parts <- strsplit(line, "::", fixed = TRUE)[[1]]
      if (length(parts) >= 3L && parts[[2]] %in% distributions) {
        ok[[parts[[2]]]] <- TRUE
        version[[parts[[2]]]] <- paste(parts[-(1:2)], collapse = "::")
      }
    } else if (startsWith(line, "FAIL::")) {
      parts <- strsplit(line, "::", fixed = TRUE)[[1]]
      if (length(parts) >= 2L && parts[[2]] %in% distributions) {
        details[[parts[[2]]]] <- paste(parts[-(1:2)], collapse = "::")
      }
    }
  }
  list(ok = ok, version = version, details = details, status = probe$status)
}

cs2step_python_jax_backend_probe <- function(python, platform = NULL) {
  code <- paste(
    "import jax",
    "print('JAX_DEFAULT_BACKEND::' + str(jax.default_backend()))",
    sep = "\n"
  )
  platform <- as.character(platform %||% "")
  env <- if (nzchar(platform)) sprintf("JAX_PLATFORMS=%s", platform) else character()
  probe <- cs2step_python_probe(python, code, env = env)
  line <- grep("^JAX_DEFAULT_BACKEND::", probe$output, value = TRUE)
  backend <- if (length(line) > 0L) sub("^JAX_DEFAULT_BACKEND::", "", line[[1]]) else ""
  ok <- identical(probe$status, 0L) && nzchar(backend)
  if (nzchar(platform)) {
    ok <- ok && identical(backend, platform)
  }
  list(ok = ok, backend = backend, status = probe$status, output = probe$output)
}

cs2step_python_module_probe <- function(python, modules) {
  modules <- unique(as.character(modules %||% character(0)))
  if (!length(modules)) {
    return(list(ok = logical(0), details = character(0), status = 0L))
  }
  quoted <- paste(sprintf("'%s'", gsub("'", "\\\\'", modules, fixed = TRUE)), collapse = ", ")
  code <- paste(
    "import importlib",
    sprintf("mods = [%s]", quoted),
    "for name in mods:",
    "    try:",
    "        importlib.import_module(name)",
    "        print('OK::' + name)",
    "    except Exception as exc:",
    "        print('FAIL::' + name + '::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  ok <- setNames(rep(FALSE, length(modules)), modules)
  details <- setNames(rep("", length(modules)), modules)
  for (line in probe$output) {
    if (!nzchar(line)) {
      next
    }
    if (startsWith(line, "OK::")) {
      mod <- sub("^OK::", "", line)
      if (mod %in% modules) {
        ok[[mod]] <- TRUE
      }
    } else if (startsWith(line, "FAIL::")) {
      parts <- strsplit(line, "::", fixed = TRUE)[[1]]
      if (length(parts) >= 2L && parts[[2]] %in% modules) {
        details[[parts[[2]]]] <- paste(parts[-(1:2)], collapse = "::")
      }
    }
  }
  list(ok = ok, details = details, status = probe$status)
}

cs2step_backend_core_modules <- function() {
  c("jax", "numpyro", "optax", "equinox", "numpy", "orbax.checkpoint")
}

cs2step_backend_core_pip_packages <- function() {
  c(
    jax = "jax",
    numpyro = "numpyro",
    optax = "optax",
    equinox = "equinox",
    numpy = "numpy",
    "orbax.checkpoint" = "orbax-checkpoint"
  )
}

cs2step_backend_env_state <- function(conda_env = "strategize_env", conda = "auto") {
  conda_bin <- cs2step_resolve_conda_binary(conda)
  envs <- tryCatch(
    reticulate::conda_list(conda = conda_bin %||% conda),
    error = function(e) NULL
  )
  registered <- FALSE
  python <- ""
  if (!is.null(envs) && nrow(envs) > 0L && "name" %in% names(envs)) {
    idx <- match(conda_env, as.character(envs$name))
    if (!is.na(idx)) {
      registered <- TRUE
      if ("python" %in% names(envs)) {
        python <- as.character(envs$python[[idx]] %||% "")
      }
    }
  }
  python <- path.expand(python)
  python_exists <- nzchar(python) && file.exists(python)
  module_probe <- if (python_exists) {
    cs2step_python_module_probe(python, cs2step_backend_core_modules())
  } else {
    list(
      ok = setNames(rep(FALSE, length(cs2step_backend_core_modules())), cs2step_backend_core_modules()),
      details = setNames(rep("", length(cs2step_backend_core_modules())), cs2step_backend_core_modules()),
      status = 127L
    )
  }
  list(
    conda = conda_bin,
    conda_env = conda_env,
    registered = registered,
    python = python,
    python_exists = python_exists,
    core_module_status = module_probe$ok,
    core_module_details = module_probe$details,
    core_modules_ready = python_exists && all(module_probe$ok)
  )
}

cs2step_ensure_basic_python_conda_env <- function(conda_env,
                                                  conda = "auto",
                                                  python_version = "3.12",
                                                  force_reinstall = FALSE,
                                                  verbose = TRUE,
                                                  context = "Python runtime") {
  verbose <- cs2step_backend_verbose(verbose)
  conda_env <- as.character(conda_env %||% "")
  if (length(conda_env) != 1L || is.na(conda_env) || !nzchar(trimws(conda_env))) {
    stop("conda_env must be a non-empty scalar string.", call. = FALSE)
  }
  conda_env <- trimws(conda_env)
  if (!is.logical(force_reinstall) || length(force_reinstall) != 1L ||
      is.na(force_reinstall)) {
    stop("force_reinstall must be TRUE or FALSE.", call. = FALSE)
  }
  conda_bin <- cs2step_resolve_conda_binary(conda) %||% conda

  remove_env <- function(conda_use) {
    tryCatch(
      reticulate::conda_remove(envname = conda_env, conda = conda_use),
      error = function(e) {
        if (nzchar(conda_use %||% "")) {
          suppressWarnings(system2(
            conda_use,
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

  create_env <- function(conda_use) {
    cs2step_backend_message(
      verbose,
      "Creating conda environment '%s' for %s.",
      conda_env,
      context
    )
    reticulate::conda_create(
      envname = conda_env,
      conda = conda_use,
      python_version = python_version
    )
    invisible(TRUE)
  }

  env_registered <- function(conda_use) {
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_use)
    isTRUE(state$registered)
  }

  state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
  if (isTRUE(force_reinstall) && isTRUE(state$registered)) {
    cs2step_backend_message(
      verbose,
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

  if (isTRUE(state$registered) && !isTRUE(state$python_exists)) {
    cs2step_backend_message(
      verbose,
      "Conda environment '%s' is registered but its Python interpreter is missing; recreating it.",
      conda_env
    )
    remove_env(conda_bin)
    state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda_bin)
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
  if (!isTRUE(state$registered) || !isTRUE(state$python_exists)) {
    stop(
      sprintf("Failed to create conda environment '%s' for %s.", conda_env, context),
      call. = FALSE
    )
  }
  state
}

cs2step_backend_host_info <- function() {
  info <- Sys.info()
  os <- as.character(unname(info["sysname"]))
  if (!nzchar(os) || is.na(os)) {
    os <- .Platform$OS.type
  }
  machine <- as.character(unname(info["machine"]))
  if (!nzchar(machine) || is.na(machine)) {
    machine <- R.version$arch %||% ""
  }
  list(
    os = os,
    machine = machine,
    is_macos = identical(os, "Darwin"),
    is_arm64 = grepl("arm64|aarch64", machine, ignore.case = TRUE)
  )
}

cs2step_backend_mps_compatibility <- function(state) {
  python <- path.expand(as.character(state$python %||% ""))
  python_exists <- isTRUE(state$python_exists) && nzchar(python) && file.exists(python)
  if (!python_exists) {
    return(list(
      compatible = FALSE,
      python_major_minor = "",
      python_313 = FALSE,
      jax_mps_installed = FALSE,
      jax_mps_version = "",
      jax_backend = "",
      jax_backend_mps = FALSE,
      details = "python interpreter not found"
    ))
  }

  python_version <- cs2step_python_version_major_minor(python)
  python_313 <- identical(python_version$major_minor, "3.13")
  dist_probe <- cs2step_python_distribution_probe(python, "jax-mps")
  jax_mps_installed <- isTRUE(dist_probe$ok[["jax-mps"]])
  jax_backend_probe <- if (python_313 && jax_mps_installed) {
    cs2step_python_jax_backend_probe(python, platform = "mps")
  } else {
    list(ok = FALSE, backend = "", status = NA_integer_, output = character())
  }

  list(
    compatible = python_313 && jax_mps_installed && isTRUE(jax_backend_probe$ok),
    python_major_minor = python_version$major_minor,
    python_313 = python_313,
    jax_mps_installed = jax_mps_installed,
    jax_mps_version = dist_probe$version[["jax-mps"]] %||% "",
    jax_backend = jax_backend_probe$backend,
    jax_backend_mps = isTRUE(jax_backend_probe$ok),
    details = paste(c(python_version$output, dist_probe$details, jax_backend_probe$output), collapse = "\n")
  )
}

cs2step_describe_mps_compatibility <- function(compatibility) {
  issues <- character()
  if (!isTRUE(compatibility$python_313)) {
    found <- compatibility$python_major_minor %||% ""
    if (!nzchar(found)) {
      found <- "unknown"
    }
    issues <- c(issues, sprintf("Python 3.13 required, found %s", found))
  }
  if (!isTRUE(compatibility$jax_mps_installed)) {
    issues <- c(issues, "Python distribution 'jax-mps' is not installed")
  }
  if (!isTRUE(compatibility$jax_backend_mps)) {
    backend <- compatibility$jax_backend %||% ""
    if (!nzchar(backend)) {
      backend <- "unavailable"
    }
    issues <- c(issues, sprintf("JAX default backend under JAX_PLATFORMS=mps is %s", backend))
  }
  if (!length(issues)) {
    return("compatible")
  }
  paste(issues, collapse = "; ")
}

cs2step_command_probe <- function(command, args = character()) {
  output <- tryCatch(
    suppressWarnings(system2(command, args = args, stdout = TRUE, stderr = TRUE)),
    error = function(e) structure(conditionMessage(e), status = 127L)
  )
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  list(status = as.integer(status), output = as.character(output))
}

cs2step_command_stream <- function(command, args = character()) {
  status <- tryCatch(
    suppressWarnings(system2(command, args = args, stdout = "", stderr = "")),
    error = function(e) {
      out <- 127L
      attr(out, "output") <- conditionMessage(e)
      out
    }
  )
  output <- attr(status, "output") %||% character()
  if (is.null(status)) {
    status <- 0L
  }
  list(status = as.integer(status), output = as.character(output), streamed = TRUE)
}

cs2step_run_command_available <- function(cmd) {
  nzchar(Sys.which(cmd))
}

cs2step_collect_nvidia_tools <- function() {
  list(
    nvidia_smi = cs2step_run_command_available("nvidia-smi"),
    nvcc = cs2step_run_command_available("nvcc")
  )
}

cs2step_probe_nvidia_driver <- function() {
  if (!isTRUE(cs2step_run_command_available("nvidia-smi"))) {
    return(list(available = FALSE, driver_version = "", driver_major = NA_integer_, device_name = ""))
  }
  probe <- cs2step_command_probe(
    "nvidia-smi",
    c("--query-gpu=driver_version,name", "--format=csv,noheader")
  )
  line <- probe$output[[1]] %||% ""
  if (!nzchar(line)) {
    return(list(available = FALSE, driver_version = "", driver_major = NA_integer_, device_name = ""))
  }
  parts <- strsplit(line, ",", fixed = TRUE)[[1]]
  driver_version <- trimws(parts[[1]] %||% "")
  driver_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", driver_version)))
  device_name <- if (length(parts) >= 2L) trimws(paste(parts[-1], collapse = ",")) else ""
  list(
    available = identical(probe$status, 0L) && nzchar(driver_version),
    driver_version = driver_version,
    driver_major = driver_major,
    device_name = device_name
  )
}

cs2step_cuda_torch_wheel <- function(driver_major) {
  if (is.na(driver_major)) {
    return(NULL)
  }
  if (driver_major >= 580L) {
    return(list(label = "cu130", index_url = "https://download.pytorch.org/whl/cu130"))
  }
  if (driver_major >= 525L) {
    return(list(label = "cu128", index_url = "https://download.pytorch.org/whl/cu128"))
  }
  NULL
}

cs2step_collect_rocm_tools <- function() {
  list(
    rocminfo = cs2step_run_command_available("rocminfo"),
    hipcc = cs2step_run_command_available("hipcc"),
    rocm_smi = cs2step_run_command_available("rocm-smi"),
    rocm_root = dir.exists("/opt/rocm")
  )
}

cs2step_probe_rocm_runtime <- function(python) {
  if (!nzchar(python %||% "") || !file.exists(python)) {
    return(list(validated = FALSE, cuda_available = FALSE, hip_version = "", device_name = ""))
  }
  code <- paste(
    "try:",
    "    import torch",
    "    hip = getattr(getattr(torch, 'version', None), 'hip', None)",
    "    avail = bool(torch.cuda.is_available())",
    "    print('TORCH_OK')",
    "    print('CUDA_AVAILABLE::' + ('1' if avail else '0'))",
    "    print('HIP_VERSION::' + (str(hip) if hip is not None else ''))",
    "    if avail:",
    "        try:",
    "            print('DEVICE_NAME::' + str(torch.cuda.get_device_name(0)))",
    "        except Exception as exc:",
    "            print('DEVICE_NAME::' + exc.__class__.__name__)",
    "except Exception as exc:",
    "    print('TORCH_FAIL::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  lines <- probe$output %||% character(0)
  torch_ok <- any(startsWith(lines, "TORCH_OK"))
  cuda_available <- any(lines == "CUDA_AVAILABLE::1")
  hip_line <- lines[startsWith(lines, "HIP_VERSION::")]
  device_line <- lines[startsWith(lines, "DEVICE_NAME::")]
  hip_version <- if (length(hip_line)) sub("^HIP_VERSION::", "", hip_line[[1]]) else ""
  device_name <- if (length(device_line)) sub("^DEVICE_NAME::", "", device_line[[1]]) else ""
  list(
    validated = isTRUE(torch_ok) && isTRUE(cuda_available) && nzchar(hip_version),
    cuda_available = cuda_available,
    hip_version = hip_version,
    device_name = device_name
  )
}

cs2step_probe_cuda_runtime <- function(python) {
  if (!nzchar(python %||% "") || !file.exists(python)) {
    return(list(validated = FALSE, cuda_available = FALSE, cuda_version = "", hip_version = "", device_name = ""))
  }
  code <- paste(
    "try:",
    "    import torch",
    "    ver = getattr(torch, 'version', None)",
    "    cuda = getattr(ver, 'cuda', None)",
    "    hip = getattr(ver, 'hip', None)",
    "    avail = bool(torch.cuda.is_available())",
    "    print('TORCH_OK')",
    "    print('CUDA_AVAILABLE::' + ('1' if avail else '0'))",
    "    print('CUDA_VERSION::' + (str(cuda) if cuda is not None else ''))",
    "    print('HIP_VERSION::' + (str(hip) if hip is not None else ''))",
    "    if avail:",
    "        try:",
    "            print('DEVICE_NAME::' + str(torch.cuda.get_device_name(0)))",
    "        except Exception as exc:",
    "            print('DEVICE_NAME::' + exc.__class__.__name__)",
    "except Exception as exc:",
    "    print('TORCH_FAIL::' + exc.__class__.__name__ + '::' + str(exc))",
    sep = "\n"
  )
  probe <- cs2step_python_probe(python, code)
  lines <- probe$output %||% character(0)
  torch_ok <- any(startsWith(lines, "TORCH_OK"))
  cuda_available <- any(lines == "CUDA_AVAILABLE::1")
  cuda_line <- lines[startsWith(lines, "CUDA_VERSION::")]
  hip_line <- lines[startsWith(lines, "HIP_VERSION::")]
  device_line <- lines[startsWith(lines, "DEVICE_NAME::")]
  cuda_version <- if (length(cuda_line)) sub("^CUDA_VERSION::", "", cuda_line[[1]]) else ""
  hip_version <- if (length(hip_line)) sub("^HIP_VERSION::", "", hip_line[[1]]) else ""
  device_name <- if (length(device_line)) sub("^DEVICE_NAME::", "", device_line[[1]]) else ""
  list(
    validated = isTRUE(torch_ok) && isTRUE(cuda_available) && nzchar(cuda_version) && !nzchar(hip_version),
    cuda_available = cuda_available,
    cuda_version = cuda_version,
    hip_version = hip_version,
    device_name = device_name
  )
}

cs2step_collect_text_embedding_host_info <- function(conda_env = "strategize_torch_env", conda = "auto") {
  env_state <- cs2step_backend_env_state(conda_env = conda_env, conda = conda)
  os_name <- as.character(Sys.info()[["sysname"]] %||% .Platform$OS.type)
  machine <- as.character(Sys.info()[["machine"]] %||% R.version$arch)
  nvidia_tools <- cs2step_collect_nvidia_tools()
  nvidia_driver <- cs2step_probe_nvidia_driver()
  cuda_runtime <- cs2step_probe_cuda_runtime(env_state$python)
  rocm_tools <- cs2step_collect_rocm_tools()
  rocm_runtime <- cs2step_probe_rocm_runtime(env_state$python)
  list(
    os = os_name,
    machine = machine,
    conda = env_state$conda,
    conda_env = conda_env,
    conda_registered = env_state$registered,
    python = env_state$python,
    python_exists = env_state$python_exists,
    core_modules_ready = env_state$core_modules_ready,
    core_module_status = env_state$core_module_status,
    core_module_details = env_state$core_module_details,
    mlx_host_capable = identical(os_name, "Darwin") &&
      grepl("arm|aarch64", machine, ignore.case = TRUE),
    nvidia_tools = nvidia_tools,
    nvidia_driver = nvidia_driver,
    cuda_runtime = cuda_runtime,
    rocm_tools = rocm_tools,
    rocm_runtime = rocm_runtime
  )
}

cs2step_text_embedding_candidate <- function(label,
                                             backend,
                                             device,
                                             model_id,
                                             conda_env,
                                             conda,
                                             family = "qwen3",
                                             profile = "portable",
                                             canonical_dim = 1024L,
                                             raw_dim = canonical_dim,
                                             required_modules = character(0),
                                             install_packages = character(0),
                                             cache_key_version = 1L,
                                             host_constraints = list()) {
  list(
    version = cs2step_text_embedding_backend_version(),
    label = label,
    family = family,
    profile = profile,
    backend = backend,
    device = device,
    model_id = model_id,
    conda_env = conda_env,
    conda = conda,
    canonical_dim = as.integer(canonical_dim),
    raw_dim = as.integer(raw_dim),
    required_modules = as.character(required_modules),
    install_packages = as.character(install_packages),
    cache_key_version = as.integer(cache_key_version),
    host_constraints = host_constraints
  )
}

cs2step_evaluate_text_embedding_candidate <- function(candidate, host) {
  issues <- character(0)
  status <- "ready"
  installable <- FALSE

  if (!isTRUE(host$conda_registered)) {
    status <- "unavailable"
    issues <- c(issues, sprintf("Conda env '%s' is not registered.", candidate$conda_env))
  }
  if (!isTRUE(host$python_exists)) {
    status <- "unavailable"
    issues <- c(issues, sprintf("Python interpreter for env '%s' is missing.", candidate$conda_env))
  }

  if (identical(candidate$backend, "mlx")) {
    if (!isTRUE(host$mlx_host_capable)) {
      status <- "unavailable"
      issues <- c(issues, "MLX is only supported on Apple Silicon hosts.")
    } else {
      installable <- TRUE
    }
  } else if (identical(candidate$backend, "sentence_transformers")) {
    installable <- identical(candidate$device, "cpu")
    if (identical(candidate$device, "cuda")) {
      if (!identical(host$os, "Linux")) {
        status <- "unavailable"
        issues <- c(issues, "CUDA text embeddings are currently supported only on Linux hosts.")
      } else if (!isTRUE(any(unlist(host$nvidia_tools)))) {
        status <- "unavailable"
        issues <- c(issues, "Nvidia tooling is not present on this host.")
      } else if (!isTRUE(host$nvidia_driver$available)) {
        status <- "unavailable"
        issues <- c(issues, "Nvidia tooling is present but the GPU driver could not be resolved.")
      } else if (is.null(cs2step_cuda_torch_wheel(host$nvidia_driver$driver_major))) {
        status <- "unavailable"
        issues <- c(
          issues,
          sprintf(
            "Nvidia driver '%s' is too old for the supported CUDA torch wheels.",
            host$nvidia_driver$driver_version %||% "unknown"
          )
        )
      } else {
        installable <- TRUE
      }
    } else if (identical(candidate$device, "rocm")) {
      if (!isTRUE(any(unlist(host$rocm_tools)))) {
        status <- "unavailable"
        issues <- c(issues, "ROCm tooling is not present on this host.")
      } else if (!isTRUE(host$rocm_runtime$validated)) {
        status <- "unavailable"
        issues <- c(issues, "ROCm tooling is present but Python torch ROCm validation did not succeed.")
      }
    }
  }

  module_probe <- if (identical(status, "unavailable")) {
    list(ok = setNames(rep(FALSE, length(candidate$required_modules)), candidate$required_modules))
  } else {
    cs2step_python_module_probe(host$python, candidate$required_modules)
  }

  if (length(candidate$required_modules)) {
    missing <- names(module_probe$ok)[!module_probe$ok]
    if (length(missing)) {
      if (installable) {
        status <- "needs_install"
      } else {
        status <- "unavailable"
      }
      issues <- c(
        issues,
        sprintf(
          "%s runtime is missing Python modules: %s.",
          candidate$label,
          paste(missing, collapse = ", ")
        )
      )
    }
  }

  if (identical(candidate$device, "cuda") &&
      !identical(status, "unavailable") &&
      !isTRUE(host$cuda_runtime$validated)) {
    status <- "needs_install"
    issues <- c(issues, "Nvidia tooling is present but Python torch CUDA validation did not succeed.")
  }

  candidate$status <- status
  candidate$installable <- installable
  candidate$issues <- unique(issues)
  candidate$module_status <- module_probe$ok %||% logical(0)
  candidate
}

cs2step_resolve_text_embedding_candidates <- function(host,
                                                      profile = "portable",
                                                      family = NULL,
                                                      model_id = NULL) {
  profile <- cs2step_text_embedding_profile(profile)
  spec <- cs2step_text_embedding_profile_spec(profile)
  family <- cs2step_text_embedding_family(family %||% spec$family)
  lapply(spec$candidates, function(candidate) {
    model_use <- as.character(model_id %||% candidate$model_id)
    cs2step_text_embedding_candidate(
      label = candidate$label,
      backend = candidate$backend,
      device = candidate$device,
      model_id = model_use,
      conda_env = host$conda_env,
      conda = host$conda,
      family = family,
      profile = profile,
      canonical_dim = spec$canonical_dim,
      raw_dim = candidate$raw_dim %||% spec$canonical_dim,
      required_modules = candidate$required_modules %||% character(0),
      install_packages = candidate$install_packages %||% character(0),
      cache_key_version = spec$cache_key_version %||% 1L,
      host_constraints = candidate$host_constraints %||% list()
    )
  })
}

cs2step_select_text_embedding_candidate <- function(candidates,
                                                    host,
                                                    runtime = "auto",
                                                    family = "qwen3",
                                                    profile = "portable") {
  family <- cs2step_text_embedding_family(family)
  profile <- cs2step_text_embedding_profile(profile)
  profile_spec <- cs2step_text_embedding_profile_spec(profile)
  runtime <- cs2step_text_embedding_runtime(runtime)
  candidates <- lapply(candidates, cs2step_evaluate_text_embedding_candidate, host = host)

  selected <- NULL
  issues <- character(0)
  candidate_selectable <- function(name) {
    candidate <- candidates[[name]]
    is.list(candidate) && candidate$status %in% c("ready", "needs_install")
  }
  candidate_or_issue <- function(name) {
    candidate <- candidates[[name]]
    if (!is.list(candidate)) {
      issues <<- c(
        issues,
        sprintf("Profile '%s' does not define a %s text embedding candidate.", profile, name)
      )
      return(NULL)
    }
    candidate
  }
  if (identical(runtime, "auto")) {
    if (isTRUE(host$mlx_host_capable) && candidate_selectable("mlx")) {
      selected <- candidates$mlx
    } else if (identical(host$os, "Linux") &&
               candidate_selectable("cuda")) {
      selected <- candidates$cuda
    } else if (identical(host$os, "Linux") &&
               isTRUE(any(unlist(host$rocm_tools))) &&
               isTRUE(host$rocm_runtime$validated) &&
               candidate_selectable("rocm")) {
      selected <- candidates$rocm
    } else if (isTRUE(profile_spec$auto_cpu_fallback)) {
      selected <- candidate_or_issue("cpu")
      if (identical(host$os, "Linux") && isTRUE(any(unlist(host$nvidia_tools)))) {
        issues <- c(
          issues,
          "Nvidia tooling is present but CUDA text embeddings are not usable; falling back to the next runtime."
        )
      }
      if (identical(host$os, "Linux") && isTRUE(any(unlist(host$rocm_tools))) &&
          !isTRUE(host$rocm_runtime$validated)) {
        issues <- c(
          issues,
          "ROCm tooling is present but not validated in Python; falling back to CPU text embeddings."
        )
      }
    } else {
      issues <- c(
        issues,
        sprintf(
          "Profile '%s' requires an accelerator in runtime='auto'; use runtime='cpu' explicitly to force CPU.",
          profile
        )
      )
    }
  } else if (identical(runtime, "mlx")) {
    selected <- candidate_or_issue("mlx")
  } else if (identical(runtime, "cuda")) {
    selected <- candidate_or_issue("cuda")
  } else if (identical(runtime, "rocm")) {
    selected <- candidate_or_issue("rocm")
  } else {
    selected <- candidate_or_issue("cpu")
  }

  issues <- unique(c(
    issues,
    unlist(lapply(candidates, `[[`, "issues"), use.names = FALSE)
  ))

  list(
    family = family,
    profile = profile,
    runtime = runtime,
    host = host,
    candidates = candidates,
    selected = selected,
    issues = issues
  )
}

cs2step_inspect_text_embedding_candidates <- function(host,
                                                      family = "qwen3",
                                                      runtime = "auto",
                                                      profile = "portable",
                                                      model_id = NULL) {
  candidates <- cs2step_resolve_text_embedding_candidates(
    host = host,
    profile = profile,
    family = family,
    model_id = model_id
  )
  cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = runtime,
    family = family,
    profile = profile
  )
}

cs2step_conda_run_supports_no_capture <- function(conda_bin) {
  probe <- cs2step_command_probe(conda_bin, c("run", "--help"))
  identical(probe$status, 0L) &&
    any(grepl("--no-capture-output", probe$output, fixed = TRUE))
}

cs2step_conda_run <- function(conda,
                              conda_env,
                              args,
                              verbose = TRUE,
                              stream = verbose,
                              context = "preparing Python runtime") {
  verbose <- cs2step_backend_verbose(verbose)
  stream <- isTRUE(stream) && isTRUE(verbose)
  conda_bin <- cs2step_resolve_conda_binary(conda)
  if (!nzchar(conda_bin %||% "")) {
    stop("Unable to resolve a conda binary for text embedding runtime installation.", call. = FALSE)
  }
  run_args <- c("run")
  if (isTRUE(stream) && isTRUE(cs2step_conda_run_supports_no_capture(conda_bin))) {
    run_args <- c(run_args, "--no-capture-output")
  }
  run_args <- c(run_args, "-n", conda_env, args)
  command_text <- paste(c(conda_bin, run_args), collapse = " ")
  cs2step_backend_message(
    verbose,
    "Running in conda env '%s': %s",
    conda_env,
    paste(args, collapse = " ")
  )
  probe <- if (isTRUE(stream)) {
    cs2step_command_stream(conda_bin, run_args)
  } else {
    cs2step_command_probe(conda_bin, run_args)
  }
  if (!identical(probe$status, 0L)) {
    output <- paste(probe$output %||% character(0), collapse = "\n")
    if (!nzchar(output) && isTRUE(probe$streamed)) {
      output <- "Command output was streamed above."
    }
    stop(
      sprintf(
        "Command failed while %s in '%s'.\nCommand: %s\n%s",
        context,
        conda_env,
        command_text,
        output
      ),
      call. = FALSE
    )
  }
  invisible(probe$output)
}

cs2step_pip_install_in_conda <- function(conda,
                                         conda_env,
                                         packages,
                                         index_url = NULL,
                                         force_reinstall = FALSE,
                                         verbose = TRUE,
                                         context = "installing Python packages") {
  verbose <- cs2step_backend_verbose(verbose)
  packages <- as.character(packages)
  packages <- packages[nzchar(packages)]
  if (!length(packages)) {
    return(invisible(TRUE))
  }
  args <- c("python", "-m", "pip", "install", "--upgrade")
  if (isTRUE(force_reinstall)) {
    args <- c(args, "--force-reinstall")
  }
  if (!is.null(index_url) && nzchar(index_url)) {
    args <- c(args, "--index-url", index_url)
  }
  args <- c(args, packages)
  cs2step_backend_message(
    verbose,
    "Installing Python package(s) in '%s': %s",
    conda_env,
    paste(packages, collapse = ", ")
  )
  cs2step_conda_run(
    conda = conda,
    conda_env = conda_env,
    args = args,
    verbose = verbose,
    stream = verbose,
    context = context
  )
}

cs2step_install_cuda_sentence_transformers <- function(spec, verbose = TRUE) {
  verbose <- cs2step_backend_verbose(verbose)
  driver <- cs2step_probe_nvidia_driver()
  wheel <- cs2step_cuda_torch_wheel(driver$driver_major)
  if (is.null(wheel)) {
    stop(
      sprintf(
        "No supported CUDA torch wheel is available for Nvidia driver '%s'.",
        driver$driver_version %||% "unknown"
      ),
      call. = FALSE
    )
  }
  cs2step_pip_install_in_conda(
    conda = spec$conda,
    conda_env = spec$conda_env,
    packages = "torch",
    index_url = wheel$index_url,
    force_reinstall = TRUE,
    verbose = verbose,
    context = "installing CUDA torch for text embeddings"
  )
  cs2step_pip_install_in_conda(
    conda = spec$conda,
    conda_env = spec$conda_env,
    packages = cs2step_text_embedding_install_packages(
      spec,
      fallback = c("sentence-transformers", "transformers")
    ),
    verbose = verbose,
    context = "installing sentence-transformers for text embeddings"
  )
}

cs2step_ensure_text_embedding_runtime <- function(spec, verbose = TRUE) {
  verbose <- cs2step_backend_verbose(verbose)
  spec <- cs2step_normalize_text_embedding_spec(spec)
  env_state <- cs2step_backend_env_state(conda_env = spec$conda_env, conda = spec$conda)
  if (!isTRUE(env_state$registered) || !isTRUE(env_state$python_exists)) {
    cs2step_ensure_basic_python_conda_env(
      conda_env = spec$conda_env,
      conda = spec$conda,
      verbose = verbose,
      context = "text embedding Python runtime"
    )
  }

  if (identical(spec$backend, "mlx")) {
    cs2step_pip_install_in_conda(
      conda = spec$conda,
      conda_env = spec$conda_env,
      packages = cs2step_text_embedding_install_packages(
        spec,
        fallback = c("mlx", "mlx-embeddings")
      ),
      verbose = verbose,
      context = "installing MLX text embedding packages"
    )
  } else if (identical(spec$backend, "sentence_transformers")) {
    if (identical(spec$device, "cuda")) {
      cs2step_install_cuda_sentence_transformers(spec, verbose = verbose)
    } else if (identical(spec$device, "cpu")) {
      cs2step_pip_install_in_conda(
        conda = spec$conda,
        conda_env = spec$conda_env,
        packages = cs2step_text_embedding_install_packages(
          spec,
          fallback = c("torch", "sentence-transformers", "transformers")
        ),
        verbose = verbose,
        context = "installing CPU text embedding packages"
      )
    } else {
      cs2step_pip_install_in_conda(
        conda = spec$conda,
        conda_env = spec$conda_env,
        packages = cs2step_text_embedding_install_packages(
          spec,
          fallback = c("sentence-transformers", "transformers")
        ),
        verbose = verbose,
        context = "installing text embedding packages"
      )
    }
  }
  invisible(TRUE)
}

cs2step_normalize_text_embedding_spec <- function(spec) {
  if (!is.list(spec) || is.null(spec$backend) || is.null(spec$model_id)) {
    stop("Invalid text embedding backend specification.", call. = FALSE)
  }
  spec$family <- cs2step_text_embedding_family(spec$family %||% "qwen3")
  spec$profile <- cs2step_text_embedding_profile(spec$profile %||% "portable")
  spec$runtime <- cs2step_text_embedding_runtime(spec$runtime %||% "auto")
  spec$canonical_dim <- as.integer(spec$canonical_dim %||% spec$embedding_dim %||% 1024L)
  spec$raw_dim <- as.integer(spec$raw_dim %||% spec$canonical_dim)
  spec$conda_env <- as.character(spec$conda_env %||% "strategize_torch_env")
  spec$conda <- cs2step_resolve_conda_binary(spec$conda %||% "auto")
  spec$install_packages <- as.character(spec$install_packages %||% character(0))
  spec$normalize <- isTRUE(spec$normalize %||% TRUE)
  spec$frozen <- TRUE
  spec$trainable <- FALSE
  spec$embedding_strategy <- as.character(
    spec$embedding_strategy %||% "frozen_pretrained_schema_text"
  )
  spec$cache_key_version <- as.integer(
    spec$cache_key_version %||% cs2step_text_embedding_backend_version()
  )
  spec
}

cs2step_text_embedding_cache_file <- function(spec, cache_dir = NULL) {
  spec <- cs2step_normalize_text_embedding_spec(spec)
  root <- if (!is.null(cache_dir) && nzchar(cache_dir)) {
    path.expand(as.character(cache_dir))
  } else {
    file.path(tools::R_user_dir("strategize", which = "cache"), "text-embeddings")
  }
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  file.path(
    root,
    sprintf(
      "%s.rds",
      digest::digest(list(
        family = spec$family,
        profile = spec$profile,
        backend = spec$backend,
        device = spec$device,
        model = spec$model_id,
        dim = spec$canonical_dim,
        normalize = isTRUE(spec$normalize),
        frozen = isTRUE(spec$frozen),
        strategy = spec$embedding_strategy,
        version = spec$cache_key_version
      ))
    )
  )
}

cs2step_text_embedding_canonicalize_matrix <- function(emb, spec) {
  target_dim <- as.integer(spec$canonical_dim %||% 1024L)
  raw_dim <- as.integer(spec$raw_dim %||% target_dim)
  emb_dim <- dim(emb)
  if ((is.null(emb_dim) || length(emb_dim) == 1L) && length(emb) %in% c(target_dim, raw_dim)) {
    emb <- matrix(emb, nrow = 1L)
  } else {
    emb <- as.matrix(emb)
  }
  storage.mode(emb) <- "double"
  if (ncol(emb) < target_dim) {
    stop(
      sprintf(
        "Text embedding backend '%s' returned %d columns, below the required canonical width %d.",
        spec$label %||% spec$backend,
        ncol(emb),
        target_dim
      ),
      call. = FALSE
    )
  }
  if (ncol(emb) > target_dim) {
    emb <- emb[, seq_len(target_dim), drop = FALSE]
  }
  if (isTRUE(spec$normalize)) {
    norms <- sqrt(rowSums(emb * emb))
    norms[!is.finite(norms) | norms <= 0] <- 1
    emb <- emb / norms
  }
  emb
}

cs2step_build_text_embedding_fn <- function(spec,
                                            cache_dir = NULL,
                                            cache_only = FALSE,
                                            batch_size = 64L,
                                            verbose = TRUE) {
  spec <- cs2step_normalize_text_embedding_spec(spec)
  verbose <- cs2step_backend_verbose(verbose)
  cache_only <- isTRUE(cache_only)
  batch_size <- as.integer(batch_size %||% 64L)
  if (length(batch_size) != 1L || is.na(batch_size) || batch_size < 1L) {
    batch_size <- 64L
  }
  cache_file <- function() cs2step_text_embedding_cache_file(spec, cache_dir = cache_dir)
  state <- new.env(parent = emptyenv())
  state$loaded <- FALSE
  state$cache <- NULL

  load_cache <- function() {
    if (!is.null(state$cache)) {
      return(invisible(TRUE))
    }
    cache_path <- cache_file()
    state$cache <- if (file.exists(cache_path)) {
      tryCatch(readRDS(cache_path), error = function(e) list())
    } else {
      list()
    }
    if (!is.list(state$cache)) {
      state$cache <- list()
    }
    invisible(TRUE)
  }

  save_cache <- function() {
    cache_path <- cache_file()
    saveRDS(state$cache, cache_path)
    invisible(TRUE)
  }

  load_model <- function() {
    if (isTRUE(state$loaded)) {
      return(invisible(TRUE))
    }
    if (!cs2step_text_embedding_runtime_ready(spec)) {
      cs2step_ensure_text_embedding_runtime(spec, verbose = verbose)
    }
    reticulate::use_condaenv(spec$conda_env, required = TRUE)

    if (identical(spec$backend, "mlx")) {
      state$mlx_embeddings <- reticulate::import("mlx_embeddings", delay_load = FALSE)
      state$mx <- reticulate::import("mlx.core", delay_load = FALSE)
      state$np <- reticulate::import("numpy", delay_load = FALSE)
      model_bundle <- state$mlx_embeddings$load(spec$model_id)
      state$model <- model_bundle[[1]]
      state$processor <- model_bundle[[2]]
    } else {
      state$sentence_transformers <- reticulate::import("sentence_transformers", delay_load = FALSE)
      device_use <- if (spec$device %in% c("cuda", "rocm")) "cuda" else "cpu"
      state$model <- state$sentence_transformers$SentenceTransformer(
        spec$model_id,
        device = device_use
      )
    }
    state$loaded <- TRUE
    invisible(TRUE)
  }

  encode_mlx <- function(texts) {
    output <- state$mlx_embeddings$generate(
      state$model,
      state$processor,
      texts = as.list(texts)
    )
    emb_obj <- tryCatch(reticulate::py_get_attr(output, "text_embeds"), error = function(e) output)
    state$mx$eval(emb_obj)
    emb <- reticulate::py_to_r(state$np$array(emb_obj))
    cs2step_text_embedding_canonicalize_matrix(emb, spec)
  }

  encode_sentence_transformers <- function(texts) {
    emb <- reticulate::py_to_r(
      state$model$encode(
        as.list(texts),
        show_progress_bar = FALSE,
        convert_to_numpy = TRUE,
        batch_size = batch_size
      )
    )
    cs2step_text_embedding_canonicalize_matrix(emb, spec)
  }

  fn <- function(texts) {
    load_cache()
    texts <- trimws(as.character(texts))
    texts[is.na(texts) | !nzchar(texts)] <- "(missing)"
    missing <- setdiff(unique(texts), names(state$cache))
    if (length(missing)) {
      if (isTRUE(cache_only)) {
        stop(
          sprintf(
            "Text embedding cache-only backend '%s' is missing %d text embedding(s). First missing text: %s",
            spec$label %||% spec$backend,
            length(missing),
            substr(missing[[1L]], 1L, 160L)
          ),
          call. = FALSE
        )
      }
      load_model()
      starts <- seq.int(1L, length(missing), by = batch_size)
      for (start in starts) {
        idx <- seq.int(start, min(length(missing), start + batch_size - 1L))
        batch_texts <- missing[idx]
        emb <- if (identical(spec$backend, "mlx")) {
          encode_mlx(batch_texts)
        } else {
          encode_sentence_transformers(batch_texts)
        }
        if (nrow(emb) != length(batch_texts)) {
          stop("Text embedding backend returned a row count that does not match the input text count.", call. = FALSE)
        }
        for (i in seq_along(batch_texts)) {
          state$cache[[batch_texts[[i]]]] <- unname(emb[i, ])
        }
        save_cache()
      }
    }
    out <- do.call(rbind, lapply(texts, function(text) state$cache[[text]]))
    rownames(out) <- NULL
    storage.mode(out) <- "double"
    out
  }
  attr(fn, "text_embedding_backend") <- spec
  attr(fn, "text_embedding_cache") <- list(
    cache_path = normalizePath(cache_file(), mustWork = FALSE),
    cache_only = cache_only,
    batch_size = batch_size,
    normalize = isTRUE(spec$normalize),
    frozen = isTRUE(spec$frozen)
  )
  class(fn) <- c("strategize_text_embedding_fn", class(fn))
  fn
}

cs2step_capture_text_embedding_metadata <- function(text_embedding_fn = NULL,
                                                    text_embedding_backend = NULL) {
  backend <- text_embedding_backend %||%
    if (is.function(text_embedding_fn)) attr(text_embedding_fn, "text_embedding_backend") else NULL
  list(
    text_embedding_fn = text_embedding_fn,
    text_embedding_backend = backend
  )
}

cs2step_restore_text_embedding_metadata <- function(metadata,
                                                    conda_env = NULL,
                                                    required = FALSE) {
  metadata <- metadata %||% list()
  backend <- metadata$text_embedding_backend %||%
    if (is.function(metadata$text_embedding_fn)) {
      attr(metadata$text_embedding_fn, "text_embedding_backend")
    } else {
      NULL
    }
  if (is.null(backend)) {
    return(metadata)
  }
  backend <- cs2step_normalize_text_embedding_spec(backend)
  # Keep the RECORDED text-model conda env: every restore call site passes the
  # JAX env (e.g. "strategize_env"), and overriding the stored torch/mlx env
  # with it made cross-host restores import sentence_transformers from the
  # wrong environment. Only fill the env when the stored spec has none.
  if ((is.null(backend$conda_env) || !nzchar(backend$conda_env %||% "")) &&
      !is.null(conda_env) && nzchar(conda_env)) {
    backend$conda_env <- conda_env
  }
  # Build-time "ready" status is a property of the ORIGINAL host; carrying it
  # across a save/load lets the runtime skip dependency probing on a machine
  # that never installed the text stack. Force re-probing after restore.
  if (!is.null(backend$status)) {
    backend$status <- NULL
  }
  metadata$text_embedding_backend <- backend
  if (!is.function(metadata$text_embedding_fn)) {
    metadata$text_embedding_fn <- cs2step_build_text_embedding_fn(
      spec = backend,
      cache_dir = NULL
    )
  } else if (is.null(attr(metadata$text_embedding_fn, "text_embedding_backend"))) {
    attr(metadata$text_embedding_fn, "text_embedding_backend") <- backend
  }
  metadata
}
