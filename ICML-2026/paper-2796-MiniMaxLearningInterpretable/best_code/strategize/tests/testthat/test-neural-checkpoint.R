test_that("neural SVI checkpoints write and read latest/best snapshots", {
  tmp <- tempfile()
  fingerprint <- strategize:::neural_svi_checkpoint_fingerprint(list(
    data = list(y = c(0, 1), x = matrix(1:4, nrow = 2)),
    control = list(svi_steps = 4L)
  ))
  params <- list(
    loc = matrix(seq_len(4), nrow = 2),
    scale = c(0.1, 0.2)
  )
  prediction_params <- list(
    W_out = matrix(c(1, 2), ncol = 1),
    b_out = 0
  )
  payload <- strategize:::neural_svi_checkpoint_make_payload(
    snapshot_type = "latest",
    fingerprint = fingerprint,
    completed_step = 2L,
    resolved_svi_steps = 4L,
    svi_params = params,
    prediction_params = prediction_params,
    loss_history = c(3, 2),
    validation_history = c(0.7),
    best_metric = 0.7,
    best_step = 2L
  )

  strategize:::neural_svi_checkpoint_save_snapshot(tmp, "latest", payload)
  strategize:::neural_svi_checkpoint_save_snapshot(tmp, "best", payload)

  expect_true(file.exists(file.path(tmp, "manifest.json")))
  expect_true(file.exists(file.path(tmp, "latest.rds")))
  expect_true(file.exists(file.path(tmp, "best.rds")))

  latest <- strategize:::neural_svi_checkpoint_load_snapshot(tmp, "latest")
  expect_equal(latest$svi_params$loc, params$loc)
  expect_equal(as.numeric(latest$svi_params$scale), params$scale)
  expect_equal(latest$prediction_params$W_out, prediction_params$W_out)
  expect_equal(latest$loss_history, c(3, 2))
  expect_equal(latest$validation_history, 0.7)

  manifest <- jsonlite::read_json(file.path(tmp, "manifest.json"))
  expect_identical(manifest$artifact_type, "strategize_neural_svi_checkpoint")
  expect_identical(manifest$checkpoint_semantics, "parameter_warm_start")
  expect_identical(manifest$snapshots$latest$file, "latest.rds")
  expect_identical(manifest$snapshots$best$file, "best.rds")
  expect_identical(latest$checkpoint_semantics, "parameter_warm_start")
})

test_that("neural SVI checkpoint atomic writes leave readable snapshots", {
  tmp <- tempfile()
  dir.create(tmp)
  target <- file.path(tmp, "latest.rds")
  strategize:::neural_svi_checkpoint_atomic_save_rds(list(version = 1L), target)
  expect_identical(readRDS(target)$version, 1L)

  strategize:::neural_svi_checkpoint_atomic_save_rds(list(version = 2L), target)
  expect_identical(readRDS(target)$version, 2L)

  leftovers <- list.files(tmp, all.files = TRUE, no.. = TRUE)
  expect_identical(sort(leftovers), "latest.rds")
})

test_that("neural SVI checkpoint params accept reticulate dictionaries", {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    testthat::skip("reticulate not available")
  }
  py_available <- tryCatch(reticulate::py_available(initialize = TRUE), error = function(e) FALSE)
  if (!isTRUE(py_available)) {
    testthat::skip("Python not available through reticulate")
  }

  params <- reticulate::dict(
    W_out = matrix(c(1, 2), ncol = 1),
    b_out = 0,
    missing_site = reticulate::py_none()
  )
  out <- strategize:::neural_svi_checkpoint_params_to_list(params)

  expect_type(out, "list")
  expect_named(out, c("W_out", "b_out", "missing_site"))
  expect_equal(out$W_out, matrix(c(1, 2), ncol = 1))
  expect_equal(out$b_out, 0)
  expect_null(out$missing_site)
})

test_that("neural SVI checkpoint fingerprint mismatch errors clearly", {
  snapshot <- list(
    artifact_type = "strategize_neural_svi_checkpoint_snapshot",
    fingerprint = list(hash = "old")
  )
  expect_error(
    strategize:::neural_svi_checkpoint_assert_fingerprint(
      snapshot,
      list(hash = "new"),
      tempfile("model.rds.inprogress")
    ),
    "fingerprint mismatch.*cache_overwrite = TRUE.*delete the stale \\.inprogress"
  )
})

test_that("neural SVI checkpoint cleanup is manifest guarded", {
  fingerprint <- strategize:::neural_svi_checkpoint_fingerprint(list(data = 1))
  payload <- strategize:::neural_svi_checkpoint_make_payload(
    snapshot_type = "latest",
    fingerprint = fingerprint,
    completed_step = 1L,
    resolved_svi_steps = 1L,
    svi_params = list(w = 1)
  )

  safe <- tempfile(fileext = ".inprogress")
  strategize:::neural_svi_checkpoint_save_snapshot(safe, "latest", payload)
  expect_true(dir.exists(safe))
  strategize:::neural_svi_checkpoint_remove_dir(safe)
  expect_false(dir.exists(safe))

  broad <- tempfile("checkpoint-")
  dir.create(broad)
  expect_error(
    strategize:::neural_svi_checkpoint_remove_dir(broad),
    "not an '\\.inprogress'"
  )

  missing_manifest <- tempfile(fileext = ".inprogress")
  dir.create(missing_manifest)
  expect_error(
    strategize:::neural_svi_checkpoint_remove_dir(missing_manifest),
    "valid strategize manifest"
  )
})
