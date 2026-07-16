make_text_embedding_host <- function(...) {
  core_modules <- strategize:::cs2step_backend_core_modules()
  defaults <- list(
    os = "Linux",
    machine = "x86_64",
    conda = "/usr/bin/conda",
    conda_env = "strategize_env",
    conda_registered = TRUE,
    python = tempfile("python"),
    python_exists = TRUE,
    core_modules_ready = TRUE,
    core_module_status = setNames(rep(TRUE, length(core_modules)), core_modules),
    core_module_details = setNames(rep("", length(core_modules)), core_modules),
    mlx_host_capable = FALSE,
    nvidia_tools = list(nvidia_smi = FALSE, nvcc = FALSE),
    nvidia_driver = list(available = FALSE, driver_version = "", driver_major = NA_integer_, device_name = ""),
    cuda_runtime = list(validated = FALSE, cuda_available = FALSE, cuda_version = "", hip_version = "", device_name = ""),
    rocm_tools = list(rocminfo = FALSE, hipcc = FALSE, rocm_smi = FALSE, rocm_root = FALSE),
    rocm_runtime = list(validated = FALSE, cuda_available = FALSE, hip_version = "", device_name = "")
  )
  modifyList(defaults, list(...))
}

test_that("text embedding selector chooses MLX on Apple Silicon in auto mode", {
  host <- make_text_embedding_host(
    os = "Darwin",
    machine = "arm64",
    mlx_host_capable = TRUE
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$backend, "mlx")) {
        candidate$status <- "ready"
      } else {
        candidate$status <- "needs_install"
      }
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "portable"
  )

  expect_equal(inspected$selected$backend, "mlx")
  expect_equal(inspected$selected$device, "metal")
})

test_that("qwen3 8B profile resolves platform-specific 4096-dimensional candidates", {
  linux_host <- make_text_embedding_host()
  linux_candidates <- strategize:::cs2step_resolve_text_embedding_candidates(
    linux_host,
    profile = "qwen3_8b_4096"
  )

  expect_equal(linux_candidates$cuda$profile, "qwen3_8b_4096")
  expect_equal(linux_candidates$cuda$model_id, "Qwen/Qwen3-Embedding-8B")
  expect_equal(linux_candidates$cuda$canonical_dim, 4096L)
  expect_equal(linux_candidates$cuda$raw_dim, 4096L)

  mac_host <- make_text_embedding_host(
    os = "Darwin",
    machine = "arm64",
    mlx_host_capable = TRUE
  )
  mac_candidates <- strategize:::cs2step_resolve_text_embedding_candidates(
    mac_host,
    profile = "qwen3_8b_4096"
  )
  expect_equal(mac_candidates$mlx$model_id, "mlx-community/Qwen3-Embedding-8B-mxfp8")
  expect_equal(mac_candidates$mlx$backend, "mlx")
  expect_equal(mac_candidates$mlx$canonical_dim, 4096L)
})

test_that("Harrier profile resolves 1024-dimensional sentence-transformers candidates", {
  host <- make_text_embedding_host()
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "harrier_oss_v1_0.6b_1024"
  )

  expect_null(candidates$mlx)
  expect_equal(candidates$cuda$family, "harrier")
  expect_equal(candidates$cuda$profile, "harrier_oss_v1_0.6b_1024")
  expect_equal(candidates$cuda$model_id, "microsoft/harrier-oss-v1-0.6b")
  expect_equal(candidates$cuda$backend, "sentence_transformers")
  expect_equal(candidates$cuda$canonical_dim, 1024L)
  expect_equal(candidates$cuda$raw_dim, 1024L)
  expect_equal(candidates$cpu$model_id, "microsoft/harrier-oss-v1-0.6b")
})

test_that("Harrier auto mode chooses CUDA on validated Nvidia hosts", {
  host <- make_text_embedding_host(
    nvidia_tools = list(nvidia_smi = TRUE, nvcc = TRUE),
    nvidia_driver = list(
      available = TRUE,
      driver_version = "580.12",
      driver_major = 580L,
      device_name = "NVIDIA RTX"
    ),
    cuda_runtime = list(
      validated = TRUE,
      cuda_available = TRUE,
      cuda_version = "13.0",
      hip_version = "",
      device_name = "NVIDIA RTX"
    )
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "harrier_oss_v1_0.6b_1024"
  )

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      candidate$status <- if (identical(candidate$device, "cuda")) "ready" else "needs_install"
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "harrier",
    profile = "harrier_oss_v1_0.6b_1024"
  )

  expect_equal(inspected$selected$backend, "sentence_transformers")
  expect_equal(inspected$selected$device, "cuda")
  expect_equal(inspected$selected$model_id, "microsoft/harrier-oss-v1-0.6b")
})

test_that("Harrier auto mode falls back to CPU on Apple Silicon without MLX candidate", {
  host <- make_text_embedding_host(
    os = "Darwin",
    machine = "arm64",
    mlx_host_capable = TRUE
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "harrier_oss_v1_0.6b_1024"
  )

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      candidate$status <- if (identical(candidate$device, "cpu")) "needs_install" else "unavailable"
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "harrier",
    profile = "harrier_oss_v1_0.6b_1024"
  )

  expect_equal(inspected$selected$backend, "sentence_transformers")
  expect_equal(inspected$selected$device, "cpu")
})

test_that("qwen3 8B profile requires an accelerator in auto mode but allows explicit CPU", {
  host <- make_text_embedding_host()
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "qwen3_8b_4096"
  )

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      candidate$status <- if (identical(candidate$device, "cpu")) "needs_install" else "unavailable"
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  auto_inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "qwen3_8b_4096"
  )
  expect_null(auto_inspected$selected)
  expect_true(any(grepl("requires an accelerator", auto_inspected$issues, fixed = TRUE)))

  cpu_inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "cpu",
    family = "qwen3",
    profile = "qwen3_8b_4096"
  )
  expect_equal(cpu_inspected$selected$device, "cpu")
  expect_equal(cpu_inspected$selected$model_id, "Qwen/Qwen3-Embedding-8B")
})

test_that("CUDA candidate is installable on supported Nvidia hosts even before validation", {
  host <- make_text_embedding_host(
    nvidia_tools = list(nvidia_smi = TRUE, nvcc = TRUE),
    nvidia_driver = list(
      available = TRUE,
      driver_version = "580.12",
      driver_major = 580L,
      device_name = "NVIDIA RTX"
    )
  )
  candidate <- strategize:::cs2step_resolve_text_embedding_candidates(host)$cuda

  testthat::local_mocked_bindings(
    cs2step_python_module_probe = function(python, modules) {
      list(
        ok = setNames(rep(TRUE, length(modules)), modules),
        details = setNames(rep("", length(modules)), modules),
        status = 0L
      )
    },
    .package = "strategize"
  )

  evaluated <- strategize:::cs2step_evaluate_text_embedding_candidate(candidate, host)

  expect_equal(evaluated$status, "needs_install")
  expect_true(isTRUE(evaluated$installable))
  expect_true(any(grepl("CUDA validation did not succeed", evaluated$issues, fixed = TRUE)))
})

test_that("text embedding candidates do not require JAX core modules", {
  host <- make_text_embedding_host(
    core_modules_ready = FALSE,
    core_module_status = setNames(
      rep(FALSE, length(strategize:::cs2step_backend_core_modules())),
      strategize:::cs2step_backend_core_modules()
    )
  )
  candidate <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "harrier_oss_v1_0.6b_1024"
  )$cpu

  testthat::local_mocked_bindings(
    cs2step_python_module_probe = function(python, modules) {
      list(
        ok = setNames(rep(FALSE, length(modules)), modules),
        details = setNames(rep("", length(modules)), modules),
        status = 1L
      )
    },
    .package = "strategize"
  )

  evaluated <- strategize:::cs2step_evaluate_text_embedding_candidate(candidate, host)

  expect_equal(evaluated$status, "needs_install")
  expect_true(isTRUE(evaluated$installable))
  expect_false(any(grepl("Core strategize", evaluated$issues, fixed = TRUE)))
})

test_that("text embedding selector prefers CUDA on validated Nvidia hosts", {
  host <- make_text_embedding_host(
    nvidia_tools = list(nvidia_smi = TRUE, nvcc = TRUE),
    nvidia_driver = list(
      available = TRUE,
      driver_version = "580.12",
      driver_major = 580L,
      device_name = "NVIDIA RTX"
    ),
    cuda_runtime = list(
      validated = TRUE,
      cuda_available = TRUE,
      cuda_version = "13.0",
      hip_version = "",
      device_name = "NVIDIA RTX"
    )
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$device, "cuda")) {
        candidate$status <- "ready"
      } else if (identical(candidate$device, "cpu")) {
        candidate$status <- "needs_install"
      } else {
        candidate$status <- "unavailable"
      }
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "portable"
  )

  expect_equal(inspected$selected$backend, "sentence_transformers")
  expect_equal(inspected$selected$device, "cuda")
})

test_that("text embedding selector prefers ROCm when CUDA is unavailable", {
  host <- make_text_embedding_host(
    rocm_tools = list(rocminfo = TRUE, hipcc = TRUE, rocm_smi = TRUE, rocm_root = TRUE),
    rocm_runtime = list(validated = TRUE, cuda_available = TRUE, hip_version = "6.3", device_name = "AMD GPU")
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$device, "rocm")) {
        candidate$status <- "ready"
      } else if (identical(candidate$device, "cpu")) {
        candidate$status <- "needs_install"
      } else {
        candidate$status <- "unavailable"
      }
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "portable"
  )

  expect_equal(inspected$selected$backend, "sentence_transformers")
  expect_equal(inspected$selected$device, "rocm")
})

test_that("text embedding selector falls back to CPU when GPU runtimes are unavailable", {
  host <- make_text_embedding_host(
    nvidia_tools = list(nvidia_smi = TRUE, nvcc = TRUE),
    nvidia_driver = list(
      available = TRUE,
      driver_version = "510.12",
      driver_major = 510L,
      device_name = "Old NVIDIA"
    ),
    rocm_tools = list(rocminfo = TRUE, hipcc = TRUE, rocm_smi = TRUE, rocm_root = TRUE),
    rocm_runtime = list(validated = FALSE, cuda_available = FALSE, hip_version = "", device_name = "")
  )
  candidates <- strategize:::cs2step_resolve_text_embedding_candidates(host)

  testthat::local_mocked_bindings(
    cs2step_evaluate_text_embedding_candidate = function(candidate, host) {
      if (identical(candidate$device, "cpu")) {
        candidate$status <- "needs_install"
      } else {
        candidate$status <- "unavailable"
      }
      candidate$issues <- character(0)
      candidate
    },
    .package = "strategize"
  )

  inspected <- strategize:::cs2step_select_text_embedding_candidate(
    candidates = candidates,
    host = host,
    runtime = "auto",
    family = "qwen3",
    profile = "portable"
  )

  expect_equal(inspected$selected$device, "cpu")
  expect_true(any(grepl("falling back", inspected$issues, fixed = TRUE)))
})

test_that("CUDA sentence-transformers install uses the CUDA 13 wheel index for new drivers", {
  calls <- list()
  spec <- list(
    family = "qwen3",
    profile = "portable",
    runtime = "cuda",
    backend = "sentence_transformers",
    label = "sentence_transformers_cuda",
    device = "cuda",
    model_id = "Qwen/Qwen3-Embedding-0.6B",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 1024L
  )

  testthat::local_mocked_bindings(
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "580.12", driver_major = 580L, device_name = "NVIDIA RTX")
    },
    cs2step_pip_install_in_conda = function(conda, conda_env, packages, index_url = NULL,
                                            force_reinstall = FALSE, verbose = TRUE,
                                            context = "installing Python packages") {
      calls <<- c(calls, list(list(packages = packages, index_url = index_url, force_reinstall = force_reinstall)))
      invisible(TRUE)
    },
    .package = "strategize"
  )

  strategize:::cs2step_install_cuda_sentence_transformers(spec)

  expect_equal(calls[[1]]$packages, "torch")
  expect_equal(calls[[1]]$index_url, "https://download.pytorch.org/whl/cu130")
  expect_true(isTRUE(calls[[1]]$force_reinstall))
  expect_equal(calls[[2]]$packages, c("sentence-transformers", "transformers"))
})

test_that("CUDA sentence-transformers install uses the CUDA 12 wheel index for mid-range drivers", {
  calls <- list()
  spec <- list(
    family = "qwen3",
    profile = "portable",
    runtime = "cuda",
    backend = "sentence_transformers",
    label = "sentence_transformers_cuda",
    device = "cuda",
    model_id = "Qwen/Qwen3-Embedding-0.6B",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 1024L
  )

  testthat::local_mocked_bindings(
    cs2step_probe_nvidia_driver = function() {
      list(available = TRUE, driver_version = "530.40", driver_major = 530L, device_name = "NVIDIA RTX")
    },
    cs2step_pip_install_in_conda = function(conda, conda_env, packages, index_url = NULL,
                                            force_reinstall = FALSE, verbose = TRUE,
                                            context = "installing Python packages") {
      calls <<- c(calls, list(list(packages = packages, index_url = index_url, force_reinstall = force_reinstall)))
      invisible(TRUE)
    },
    .package = "strategize"
  )

  strategize:::cs2step_install_cuda_sentence_transformers(spec)

  expect_equal(calls[[1]]$packages, "torch")
  expect_equal(calls[[1]]$index_url, "https://download.pytorch.org/whl/cu128")
})

test_that("conda run streams with no-capture output when supported", {
  stream_call <- NULL

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_command_probe = function(command, args = character()) {
      expect_equal(args, c("run", "--help"))
      list(status = 0L, output = "--no-capture-output")
    },
    cs2step_command_stream = function(command, args = character()) {
      stream_call <<- list(command = command, args = args)
      list(status = 0L, output = character(), streamed = TRUE)
    },
    .package = "strategize"
  )

  expect_invisible(strategize:::cs2step_conda_run(
    conda = "/usr/bin/conda",
    conda_env = "strategize_env",
    args = c("python", "--version"),
    verbose = TRUE
  ))
  expect_true("--no-capture-output" %in% stream_call$args)
})

test_that("conda run quiet mode captures output and does not stream", {
  probe_call <- NULL

  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_command_stream = function(...) stop("streaming should not run"),
    cs2step_command_probe = function(command, args = character()) {
      probe_call <<- list(command = command, args = args)
      list(status = 0L, output = "ok")
    },
    .package = "strategize"
  )

  expect_invisible(strategize:::cs2step_conda_run(
    conda = "/usr/bin/conda",
    conda_env = "strategize_env",
    args = c("python", "--version"),
    verbose = FALSE
  ))
  expect_false("--no-capture-output" %in% probe_call$args)
})

test_that("conda run streaming failures retain command context", {
  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    cs2step_command_probe = function(command, args = character()) {
      list(status = 0L, output = "--no-capture-output")
    },
    cs2step_command_stream = function(command, args = character()) {
      list(status = 1L, output = character(), streamed = TRUE)
    },
    .package = "strategize"
  )

  expect_error(
    strategize:::cs2step_conda_run(
      conda = "/usr/bin/conda",
      conda_env = "strategize_env",
      args = c("python", "--version"),
      verbose = TRUE,
      context = "testing streamed install"
    ),
    "Command failed while testing streamed install.*Command output was streamed above"
  )
})

test_that("canonical text embedding width truncates larger matrices", {
  spec <- list(
    family = "qwen3",
    profile = "portable",
    runtime = "auto",
    backend = "mlx",
    label = "mlx",
    model_id = "mlx-community/Qwen3-Embedding-8B-mxfp8",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 4096L
  )
  emb <- matrix(seq_len(2L * 4096L), nrow = 2L, ncol = 4096L)
  out <- strategize:::cs2step_text_embedding_canonicalize_matrix(emb, spec)

  expect_equal(dim(out), c(2L, 1024L))
  expect_equal(out[, 1], emb[, 1])
  expect_equal(out[, 1024], emb[, 1024])
})

test_that("canonical text embedding width preserves single target-width vectors", {
  spec <- list(
    family = "harrier",
    profile = "harrier_oss_v1_0.6b_1024",
    runtime = "auto",
    backend = "sentence_transformers",
    label = "sentence_transformers_cuda_harrier_oss_v1_0.6b_1024",
    model_id = "microsoft/harrier-oss-v1-0.6b",
    conda_env = "strategize_torch_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 1024L
  )
  emb <- seq_len(1024L)
  out <- strategize:::cs2step_text_embedding_canonicalize_matrix(emb, spec)

  expect_equal(dim(out), c(1L, 1024L))
  expect_equal(out[1, ], emb)
})

test_that("canonical text embedding width truncates single raw-width vectors", {
  spec <- list(
    family = "qwen3",
    profile = "portable",
    runtime = "auto",
    backend = "mlx",
    label = "mlx",
    model_id = "mlx-community/Qwen3-Embedding-8B-mxfp8",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 4096L
  )
  emb <- seq_len(4096L)
  out <- strategize:::cs2step_text_embedding_canonicalize_matrix(emb, spec)

  expect_equal(dim(out), c(1L, 1024L))
  expect_equal(out[1, ], emb[seq_len(1024L)])
})

test_that("normalized text embedding specs produce unit-length frozen embeddings", {
  spec <- strategize:::cs2step_normalize_text_embedding_spec(list(
    family = "qwen3",
    profile = "portable",
    runtime = "auto",
    backend = "sentence_transformers",
    label = "sentence_transformers_cpu",
    model_id = "Qwen/Qwen3-Embedding-0.6B",
    conda_env = "strategize_env",
    conda = "/usr/bin/conda",
    canonical_dim = 3L,
    raw_dim = 3L,
    normalize = TRUE
  ))
  emb <- rbind(c(3, 4, 0), c(0, 0, 0))
  out <- strategize:::cs2step_text_embedding_canonicalize_matrix(emb, spec)

  expect_equal(drop(sqrt(rowSums(out[1, , drop = FALSE]^2))), 1)
  expect_equal(out[1, ], c(0.6, 0.8, 0))
  expect_equal(out[2, ], c(0, 0, 0))
  expect_true(isTRUE(spec$frozen))
  expect_false(isTRUE(spec$trainable))
})

test_that("legacy text embedding metadata preserves embedding_dim during normalization", {
  testthat::local_mocked_bindings(
    cs2step_resolve_conda_binary = function(conda = "auto") "/usr/bin/conda",
    .package = "strategize"
  )
  spec <- strategize:::cs2step_normalize_text_embedding_spec(list(
    backend = "mlx_embeddings",
    model_id = "mlx-community/Qwen3-Embedding-8B-mxfp8",
    embedding_dim = 4096L
  ))

  expect_equal(spec$canonical_dim, 4096L)
})

test_that("cache-only text embedding backend errors before importing model runtime", {
  host <- make_text_embedding_host()
  spec <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "qwen3_8b_4096"
  )$cpu
  cache_dir <- tempfile("strategize-text-cache-")
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(cache_dir, recursive = TRUE), add = TRUE)
  fn <- strategize:::cs2step_build_text_embedding_fn(
    spec = spec,
    cache_dir = cache_dir,
    cache_only = TRUE
  )

  testthat::local_mocked_bindings(
    cs2step_ensure_text_embedding_runtime = function(spec) {
      stop("runtime import should not be attempted")
    },
    .package = "strategize"
  )

  expect_error(
    fn("new text"),
    "Text embedding cache-only backend"
  )
})

test_that("text embedding runtime installation only creates a basic Python env", {
  spec <- list(
    family = "harrier",
    profile = "harrier_oss_v1_0.6b_1024",
    runtime = "cpu",
    backend = "sentence_transformers",
    label = "sentence_transformers_cpu",
    device = "cpu",
    model_id = "microsoft/harrier-oss-v1-0.6b",
    conda_env = "strategize_torch_env",
    conda = "/usr/bin/conda",
    canonical_dim = 1024L,
    raw_dim = 1024L
  )
  calls <- list()

  testthat::local_mocked_bindings(
    cs2step_backend_env_state = function(conda_env, conda) {
      make_text_embedding_host(
        conda = conda,
        conda_env = conda_env,
        registered = FALSE,
        conda_registered = FALSE,
        python = "",
        python_exists = FALSE,
        core_modules_ready = FALSE
      )
    },
    cs2step_ensure_basic_python_conda_env = function(conda_env,
                                                     conda = "auto",
                                                     python_version = "3.12",
                                                     force_reinstall = FALSE,
                                                     verbose = TRUE,
                                                     context = "Python runtime") {
      calls <<- c(calls, list(list(
        kind = "create_python_env",
        conda_env = conda_env,
        conda = conda,
        context = context
      )))
      invisible(TRUE)
    },
    cs2step_pip_install_in_conda = function(conda,
                                            conda_env,
                                            packages,
                                            index_url = NULL,
                                            force_reinstall = FALSE,
                                            verbose = TRUE,
                                            context = "installing Python packages") {
      calls <<- c(calls, list(list(
        kind = "pip_install",
        conda_env = conda_env,
        packages = packages,
        context = context
      )))
      invisible(TRUE)
    },
    build_backend = function(...) {
      stop("build_backend should not be called for text-only runtime installation")
    },
    .package = "strategize"
  )

  expect_invisible(strategize:::cs2step_ensure_text_embedding_runtime(spec))
  expect_equal(calls[[1]]$kind, "create_python_env")
  expect_equal(calls[[1]]$conda_env, "strategize_torch_env")
  expect_true(any(vapply(calls, `[[`, character(1), "kind") == "pip_install"))
})

test_that("text embedding cache file includes the profile in the cache key", {
  host <- make_text_embedding_host()
  portable <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "portable"
  )$cpu
  alias <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "qwen3_0.6b_1024"
  )$cpu
  harrier <- strategize:::cs2step_resolve_text_embedding_candidates(
    host,
    profile = "harrier_oss_v1_0.6b_1024"
  )$cpu
  cache_dir <- tempfile("strategize-text-cache-")
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(cache_dir, recursive = TRUE), add = TRUE)

  expect_equal(portable$model_id, alias$model_id)
  expect_equal(portable$canonical_dim, alias$canonical_dim)
  expect_false(identical(
    strategize:::cs2step_text_embedding_cache_file(portable, cache_dir = cache_dir),
    strategize:::cs2step_text_embedding_cache_file(alias, cache_dir = cache_dir)
  ))
  expect_false(identical(
    strategize:::cs2step_text_embedding_cache_file(portable, cache_dir = cache_dir),
    strategize:::cs2step_text_embedding_cache_file(harrier, cache_dir = cache_dir)
  ))
})
