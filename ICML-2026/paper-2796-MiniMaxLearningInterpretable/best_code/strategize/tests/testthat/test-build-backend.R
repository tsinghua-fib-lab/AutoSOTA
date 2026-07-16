build_backend_core_status <- function(ready) {
  setNames(
    rep(ready, length(strategize:::cs2step_backend_core_modules())),
    strategize:::cs2step_backend_core_modules()
  )
}

build_backend_mock_state <- function(conda_env,
                                     conda,
                                     registered = FALSE,
                                     python = "",
                                     python_exists = FALSE,
                                     core_ready = FALSE) {
  list(
    conda = conda,
    conda_env = conda_env,
    registered = registered,
    python = python,
    python_exists = python_exists,
    core_module_status = build_backend_core_status(core_ready),
    core_module_details = setNames(
      rep("", length(strategize:::cs2step_backend_core_modules())),
      strategize:::cs2step_backend_core_modules()
    ),
    core_modules_ready = isTRUE(python_exists) && isTRUE(core_ready)
  )
}

build_backend_mps_compat <- function(compatible = FALSE,
                                     python_major_minor = "3.12",
                                     jax_mps_installed = FALSE,
                                     jax_backend = "") {
  list(
    compatible = compatible,
    python_major_minor = python_major_minor,
    python_313 = identical(python_major_minor, "3.13"),
    jax_mps_installed = jax_mps_installed,
    jax_mps_version = if (jax_mps_installed) "0.1.0" else "",
    jax_backend = jax_backend,
    jax_backend_mps = identical(jax_backend, "mps"),
    details = ""
  )
}

build_backend_mock_pip_install_in_conda <- function(conda,
                                                    conda_env,
                                                    packages,
                                                    index_url = NULL,
                                                    force_reinstall = FALSE,
                                                    verbose = TRUE,
                                                    context = "installing Python packages") {
  reticulate::py_install(
    packages = packages,
    envname = conda_env,
    conda = conda,
    pip = TRUE,
    index_url = index_url,
    force_reinstall = force_reinstall
  )
  invisible(TRUE)
}

testthat::local_mocked_bindings(
  cs2step_pip_install_in_conda = build_backend_mock_pip_install_in_conda,
  .package = "strategize"
)

test_that("build_backend is idempotent when the core env is already healthy", {
  skip_on_cran()
  skip_if_not_installed("withr")

  py_path <- file.path(tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = function(conda_env, conda) {
      list(
        conda = "/usr/bin/conda",
        conda_env = conda_env,
        registered = TRUE,
        python = py_path,
        python_exists = TRUE,
        core_module_status = setNames(rep(TRUE, length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
        core_module_details = setNames(rep("", length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
        core_modules_ready = TRUE
      )
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(conda_env = "test_env", conda = "auto"))
})

test_that("build_backend leaves text embedding runtime untouched by default", {
  skip_on_cran()
  skip_if_not_installed("withr")

  py_path <- file.path(tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = function(conda_env, conda) {
      build_backend_mock_state(
        conda_env = conda_env,
        conda = conda,
        registered = TRUE,
        python = py_path,
        python_exists = TRUE,
        core_ready = TRUE
      )
    },
    cs2step_ensure_text_embedding_request = function(...) {
      stop("text embedding runtime should not be touched")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(conda_env = "test_env", conda = "auto"))
})

test_that("build_backend delegates requested text embedding profile for a ready env", {
  skip_on_cran()
  skip_if_not_installed("withr")

  calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = function(conda_env, conda) {
      build_backend_mock_state(
        conda_env = conda_env,
        conda = conda,
        registered = TRUE,
        python = py_path,
        python_exists = TRUE,
        core_ready = TRUE
      )
    },
    cs2step_ensure_text_embedding_request = function(text_embeddings,
                                                     text_embedding_runtime,
                                                     conda_env,
                                                     conda,
                                                     force_reinstall = FALSE,
                                                     verbose = TRUE) {
      calls <<- c(calls, list(list(
        text_embeddings = text_embeddings,
        text_embedding_runtime = text_embedding_runtime,
        conda_env = conda_env,
        conda = conda,
        force_reinstall = force_reinstall
      )))
      invisible(TRUE)
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(
    conda_env = "test_env",
    conda = "auto",
    text_embeddings = "harrier_oss_v1_0.6b_1024",
    text_embedding_runtime = "cuda"
  ))

  expect_length(calls, 1L)
  expect_equal(calls[[1]]$text_embeddings$profile, "harrier_oss_v1_0.6b_1024")
  expect_equal(calls[[1]]$text_embeddings$runtime, "cuda")
  expect_equal(calls[[1]]$text_embedding_runtime, "cuda")
  expect_equal(calls[[1]]$conda_env, "strategize_torch_env")
  expect_false(calls[[1]]$force_reinstall)
})

test_that("build_backend can install text embeddings into the JAX env when requested", {
  skip_on_cran()
  skip_if_not_installed("withr")

  calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = function(conda_env, conda) {
      build_backend_mock_state(
        conda_env = conda_env,
        conda = conda,
        registered = TRUE,
        python = py_path,
        python_exists = TRUE,
        core_ready = TRUE
      )
    },
    cs2step_ensure_text_embedding_request = function(text_embeddings,
                                                     text_embedding_runtime,
                                                     conda_env,
                                                     conda,
                                                     force_reinstall = FALSE,
                                                     verbose = TRUE) {
      calls <<- c(calls, list(list(
        text_embeddings = text_embeddings,
        text_embedding_runtime = text_embedding_runtime,
        conda_env = conda_env,
        conda = conda,
        force_reinstall = force_reinstall
      )))
      invisible(TRUE)
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(
    conda_env = "test_env",
    conda = "auto",
    text_embeddings = "harrier_oss_v1_0.6b_1024",
    text_embedding_conda_env = "test_env"
  ))

  expect_length(calls, 1L)
  expect_equal(calls[[1]]$conda_env, "test_env")
})

test_that("build_backend delegates requested text embedding profile after installing core env", {
  skip_on_cran()
  skip_if_not_installed("withr")

  calls <- list()
  install_calls <- list()
  registered <- FALSE
  installed <- FALSE
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Linux", machine = "x86_64", is_macos = FALSE, is_arm64 = FALSE)
    },
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = FALSE, driver_version = "unknown", driver_major = NA_integer_, device_name = "")
    },
    cs2step_ensure_text_embedding_request = function(text_embeddings,
                                                     text_embedding_runtime,
                                                     conda_env,
                                                     conda,
                                                     force_reinstall = FALSE,
                                                     verbose = TRUE) {
      calls <<- c(calls, list(list(
        text_embeddings = text_embeddings,
        text_embedding_runtime = text_embedding_runtime,
        conda_env = conda_env,
        conda = conda,
        force_reinstall = force_reinstall
      )))
      invisible(TRUE)
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(
    conda_env = "test_env",
    conda = "auto",
    text_embeddings = list(profile = "qwen3_8b_4096", runtime = "cpu")
  ))

  expect_true("jax" %in% unlist(install_calls))
  expect_length(calls, 1L)
  expect_equal(calls[[1]]$text_embeddings$profile, "qwen3_8b_4096")
  expect_equal(calls[[1]]$text_embeddings$runtime, "cpu")
  expect_equal(calls[[1]]$conda_env, "strategize_torch_env")
})

test_that("build_backend force_reinstall removes and rebuilds a healthy env", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  create_versions <- character()
  remove_calls <- 0L
  registered <- TRUE
  installed <- TRUE
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  states <- function(conda_env, conda) {
    if (registered && !file.exists(py_path)) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      file.create(py_path)
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "x86_64", is_macos = TRUE, is_arm64 = FALSE)
    },
    cs2step_backend_env_state = states,
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_remove = function(envname, conda) {
      remove_calls <<- remove_calls + 1L
      registered <<- FALSE
      installed <<- FALSE
      unlink(py_path)
      TRUE
    },
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      create_versions <<- c(create_versions, python_version)
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(
    conda_env = "test_env",
    conda = "auto",
    force_reinstall = TRUE
  ))

  installed_packages <- unlist(install_calls)
  expect_equal(remove_calls, 1L)
  expect_equal(create_versions, "3.12")
  expect_true("jax" %in% installed_packages)
  expect_true(all(c("numpy", "equinox", "numpyro", "optax", "orbax-checkpoint") %in% installed_packages))
})

test_that("build_backend force_reinstall skips removal for a missing env", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  create_versions <- character()
  registered <- FALSE
  installed <- FALSE
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")

  states <- function(conda_env, conda) {
    if (registered && !file.exists(py_path)) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      file.create(py_path)
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "x86_64", is_macos = TRUE, is_arm64 = FALSE)
    },
    cs2step_backend_env_state = states,
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_remove = function(...) stop("conda_remove should not run"),
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      create_versions <<- c(create_versions, python_version)
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto", force_reinstall = TRUE)

  expect_equal(create_versions, "3.12")
  expect_true("jax" %in% unlist(install_calls))
})

test_that("build_backend force_reinstall stops when removal leaves env registered", {
  skip_on_cran()
  skip_if_not_installed("withr")

  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)
  remove_calls <- 0L

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "x86_64", is_macos = TRUE, is_arm64 = FALSE)
    },
    cs2step_backend_env_state = function(conda_env, conda) {
      build_backend_mock_state(
        conda_env = conda_env,
        conda = conda,
        registered = TRUE,
        python = py_path,
        python_exists = TRUE,
        core_ready = TRUE
      )
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_remove = function(envname, conda) {
      remove_calls <<- remove_calls + 1L
      TRUE
    },
    conda_create = function(...) stop("conda_create should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  expect_error(
    build_backend(conda_env = "test_env", conda = "auto", force_reinstall = TRUE),
    "still registered after forced reinstall removal"
  )
  expect_equal(remove_calls, 1L)
})

test_that("build_backend force_reinstall validates scalar logical input", {
  expect_error(
    build_backend(force_reinstall = NA),
    "force_reinstall must be TRUE or FALSE"
  )
  expect_error(
    build_backend(force_reinstall = c(TRUE, FALSE)),
    "force_reinstall must be TRUE or FALSE"
  )
  expect_error(
    build_backend(verbose = NA),
    "verbose must be TRUE or FALSE"
  )
  expect_error(
    build_backend(verbose = c(TRUE, FALSE)),
    "verbose must be TRUE or FALSE"
  )
  expect_error(
    build_backend(text_embedding_conda_env = ""),
    "text_embedding_conda_env must be a non-empty scalar string"
  )
})

test_that("build_backend installs CPU JAX when nvidia-smi is unavailable", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  registered <- FALSE
  installed <- FALSE

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    ready <- registered && installed
    list(
      conda = "/usr/bin/conda",
      conda_env = conda_env,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_module_status = setNames(rep(ready, length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
      core_module_details = setNames(rep("", length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
      core_modules_ready = ready
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = FALSE, driver_version = "unknown", driver_major = NA_integer_, device_name = "")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(Sys.info = function() c(sysname = "Linux"), .package = "base")

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  expect_true("jax" %in% installed)
  expect_true(all(c("numpy", "equinox", "numpyro", "optax", "orbax-checkpoint") %in% installed))
  expect_false("orbax.checkpoint" %in% installed)
  expect_false("mlx-embeddings" %in% installed)
})

test_that("build_backend falls back to PATH conda and selects CUDA 13", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  registered <- FALSE
  installed <- FALSE

  states <- function(conda_env, conda) {
    if (identical(conda, "auto")) {
      return(list(
        conda = "auto",
        conda_env = conda_env,
        registered = FALSE,
        python = "",
        python_exists = FALSE,
        core_module_status = setNames(rep(FALSE, length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
        core_module_details = setNames(rep("", length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
        core_modules_ready = FALSE
      ))
    }
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    ready <- registered && installed
    list(
      conda = "/usr/bin/conda",
      conda_env = conda_env,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_module_status = setNames(rep(ready, length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
      core_module_details = setNames(rep("", length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
      core_modules_ready = ready
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "auto",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "580.12", driver_major = 580L, device_name = "NVIDIA GPU")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      if (identical(conda, "auto")) {
        stop("auto conda failed")
      }
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(
    Sys.info = function() c(sysname = "Linux"),
    Sys.which = function(x) "/usr/bin/conda",
    .package = "base"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  condas <- unlist(lapply(install_calls, `[[`, "conda"))
  expect_true("jax[cuda13]" %in% installed)
  expect_true(any(condas == "/usr/bin/conda"))
})

test_that("build_backend selects CUDA 12 for mid-range drivers", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  py_path <- file.path(tempdir(), "env", "bin", "python")
  registered <- FALSE
  installed <- FALSE

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    ready <- registered && installed
    list(
      conda = "/usr/bin/conda",
      conda_env = conda_env,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_module_status = setNames(rep(ready, length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
      core_module_details = setNames(rep("", length(strategize:::cs2step_backend_core_modules())), strategize:::cs2step_backend_core_modules()),
      core_modules_ready = ready
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "530.40", driver_major = 530L, device_name = "NVIDIA GPU")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(list(packages = packages, conda = conda)))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(Sys.info = function() c(sysname = "Linux"), .package = "base")

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto")

  installed <- unlist(lapply(install_calls, `[[`, "packages"))
  expect_true("jax[cuda12]" %in% installed)
  expect_false("jax[cuda13]" %in% installed)
})

test_that("build_backend with backend mps creates Python 3.13 and installs jax-mps first", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  create_versions <- character()
  registered <- FALSE
  mps_installed <- FALSE
  core_installed <- FALSE
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && mps_installed && core_installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "arm64", is_macos = TRUE, is_arm64 = TRUE)
    },
    cs2step_backend_env_state = states,
    cs2step_backend_mps_compatibility = function(state) {
      version <- if (length(create_versions) > 0L) create_versions[[length(create_versions)]] else "3.12"
      compatible <- identical(version, "3.13") && mps_installed
      build_backend_mps_compat(
        compatible = compatible,
        python_major_minor = version,
        jax_mps_installed = mps_installed,
        jax_backend = if (compatible) "mps" else ""
      )
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      create_versions <<- c(create_versions, python_version)
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      if ("jax-mps" %in% packages) {
        mps_installed <<- TRUE
      }
      if (any(c("numpyro", "optax", "equinox", "numpy", "orbax-checkpoint") %in% packages)) {
        core_installed <<- TRUE
      }
      TRUE
    },
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto", backend = "mps")

  installed <- unlist(install_calls)
  expect_equal(create_versions, "3.13")
  expect_equal(install_calls[[1]], "jax-mps")
  expect_false("jax" %in% installed)
  expect_true(all(c("numpy", "equinox", "numpyro", "optax", "orbax-checkpoint") %in% installed))
  mps_script <- file.path(dirname(dirname(py_path)), "etc", "conda", "activate.d", "10-jax-mps.sh")
  expect_true(file.exists(mps_script))
  expect_true(any(grepl("JAX_PLATFORMS=mps", readLines(mps_script), fixed = TRUE)))
})

test_that("build_backend with backend mps recreates an incompatible existing env", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  create_versions <- character()
  remove_calls <- 0L
  registered <- TRUE
  mps_installed <- FALSE
  core_installed <- TRUE
  python_version <- "3.12"
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  states <- function(conda_env, conda) {
    if (registered && !file.exists(py_path)) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      file.create(py_path)
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && core_installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "arm64", is_macos = TRUE, is_arm64 = TRUE)
    },
    cs2step_backend_env_state = states,
    cs2step_backend_mps_compatibility = function(state) {
      compatible <- identical(python_version, "3.13") && mps_installed
      build_backend_mps_compat(
        compatible = compatible,
        python_major_minor = python_version,
        jax_mps_installed = mps_installed,
        jax_backend = if (compatible) "mps" else ""
      )
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_remove = function(envname, conda) {
      remove_calls <<- remove_calls + 1L
      registered <<- FALSE
      core_installed <<- FALSE
      mps_installed <<- FALSE
      unlink(py_path)
      TRUE
    },
    conda_create = function(envname, conda, python_version) {
      version <- python_version
      registered <<- TRUE
      python_version <<- version
      create_versions <<- c(create_versions, version)
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      if ("jax-mps" %in% packages) {
        mps_installed <<- TRUE
      }
      if (any(c("numpyro", "optax", "equinox", "numpy", "orbax-checkpoint") %in% packages)) {
        core_installed <<- TRUE
      }
      TRUE
    },
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto", backend = "mps")

  expect_equal(remove_calls, 1L)
  expect_equal(create_versions, "3.13")
  expect_equal(install_calls[[1]], "jax-mps")
})

test_that("build_backend with backend mps is idempotent for a valid MPS env", {
  skip_on_cran()
  skip_if_not_installed("withr")

  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "arm64", is_macos = TRUE, is_arm64 = TRUE)
    },
    cs2step_backend_env_state = function(conda_env, conda) {
      build_backend_mock_state(
        conda_env = conda_env,
        conda = conda,
        registered = TRUE,
        python = py_path,
        python_exists = TRUE,
        core_ready = TRUE
      )
    },
    cs2step_backend_mps_compatibility = function(state) {
      build_backend_mps_compat(
        compatible = TRUE,
        python_major_minor = "3.13",
        jax_mps_installed = TRUE,
        jax_backend = "mps"
      )
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    conda_remove = function(...) stop("conda_remove should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  expect_invisible(build_backend(conda_env = "test_env", conda = "auto", backend = "mps"))
})

test_that("build_backend force_reinstall rebuilds a valid MPS env", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  create_versions <- character()
  remove_calls <- 0L
  registered <- TRUE
  mps_installed <- TRUE
  core_installed <- TRUE
  python_version <- "3.13"
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")
  dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
  file.create(py_path)

  states <- function(conda_env, conda) {
    if (registered && !file.exists(py_path)) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      file.create(py_path)
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && core_installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "arm64", is_macos = TRUE, is_arm64 = TRUE)
    },
    cs2step_backend_env_state = states,
    cs2step_backend_mps_compatibility = function(state) {
      compatible <- identical(python_version, "3.13") && mps_installed
      build_backend_mps_compat(
        compatible = compatible,
        python_major_minor = python_version,
        jax_mps_installed = mps_installed,
        jax_backend = if (compatible) "mps" else ""
      )
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_remove = function(envname, conda) {
      remove_calls <<- remove_calls + 1L
      registered <<- FALSE
      core_installed <<- FALSE
      mps_installed <<- FALSE
      unlink(py_path)
      TRUE
    },
    conda_create = function(envname, conda, python_version) {
      version <- python_version
      registered <<- TRUE
      python_version <<- version
      create_versions <<- c(create_versions, version)
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      if ("jax-mps" %in% packages) {
        mps_installed <<- TRUE
      }
      if (any(c("numpyro", "optax", "equinox", "numpy", "orbax-checkpoint") %in% packages)) {
        core_installed <<- TRUE
      }
      TRUE
    },
    .package = "reticulate"
  )

  withr::local_envvar(HOME = tempdir())

  build_backend(
    conda_env = "test_env",
    conda = "auto",
    backend = "mps",
    force_reinstall = TRUE
  )

  expect_equal(remove_calls, 1L)
  expect_equal(create_versions, "3.13")
  expect_equal(install_calls[[1]], "jax-mps")
})

test_that("build_backend with backend mps stops on unsupported hosts before env changes", {
  skip_on_cran()
  skip_if_not_installed("withr")

  testthat::local_mocked_bindings(
    cs2step_backend_host_info = function() {
      list(os = "Linux", machine = "x86_64", is_macos = FALSE, is_arm64 = FALSE)
    },
    cs2step_resolve_conda_binary = function(...) stop("conda resolution should not run"),
    cs2step_backend_env_state = function(...) stop("env state should not run"),
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    conda_remove = function(...) stop("conda_remove should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  expect_error(
    build_backend(conda_env = "test_env", conda = "auto", backend = "mps", force_reinstall = TRUE),
    "requires macOS on Apple Silicon"
  )
})

test_that("build_backend with backend cpu installs plain JAX even on CUDA-capable Linux", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  registered <- FALSE
  installed <- FALSE
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "580.12", driver_major = 580L, device_name = "NVIDIA GPU")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(Sys.info = function() c(sysname = "Linux"), .package = "base")
  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto", backend = "cpu")

  installed <- unlist(install_calls)
  expect_true("jax" %in% installed)
  expect_false("jax[cuda13]" %in% installed)
  expect_false("jax[cuda12]" %in% installed)
})

test_that("build_backend with backend cuda requires Linux before env changes", {
  skip_on_cran()
  skip_if_not_installed("withr")

  testthat::local_mocked_bindings(
    cs2step_backend_host_info = function() {
      list(os = "Darwin", machine = "arm64", is_macos = TRUE, is_arm64 = TRUE)
    },
    cs2step_resolve_conda_binary = function(...) stop("conda resolution should not run"),
    cs2step_backend_env_state = function(...) stop("env state should not run"),
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(...) stop("conda_create should not run"),
    py_install = function(...) stop("py_install should not run"),
    .package = "reticulate"
  )

  expect_error(
    build_backend(conda_env = "test_env", conda = "auto", backend = "cuda", force_reinstall = TRUE),
    "requires Linux"
  )
})

test_that("build_backend with backend cuda uses CUDA driver detection on Linux", {
  skip_on_cran()
  skip_if_not_installed("withr")

  install_calls <- list()
  registered <- FALSE
  installed <- FALSE
  py_path <- file.path(withr::local_tempdir(), "env", "bin", "python")

  states <- function(conda_env, conda) {
    if (registered) {
      dir.create(dirname(py_path), recursive = TRUE, showWarnings = FALSE)
      if (!file.exists(py_path)) {
        file.create(py_path)
      }
    }
    build_backend_mock_state(
      conda_env = conda_env,
      conda = conda,
      registered = registered,
      python = if (registered) py_path else "",
      python_exists = registered,
      core_ready = registered && installed
    )
  }

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_backend_env_state = states,
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "580.12", driver_major = 580L, device_name = "NVIDIA GPU")
    },
    .package = "strategize"
  )

  testthat::local_mocked_bindings(
    conda_create = function(envname, conda, python_version) {
      registered <<- TRUE
      TRUE
    },
    py_install = function(packages, envname, conda, pip, ...) {
      install_calls <<- c(install_calls, list(packages))
      installed <<- TRUE
      TRUE
    },
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(Sys.info = function() c(sysname = "Linux"), .package = "base")
  withr::local_envvar(HOME = tempdir())

  build_backend(conda_env = "test_env", conda = "auto", backend = "cuda")

  installed <- unlist(install_calls)
  expect_true("jax[cuda13]" %in% installed)
  expect_false("jax" %in% installed)
})
