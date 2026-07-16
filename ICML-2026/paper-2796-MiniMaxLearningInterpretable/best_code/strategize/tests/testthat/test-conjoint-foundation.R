test_that("pooled foundation training stub points to preference.fm", {
  expect_error(
    fit_conjoint_foundation_model(experiments = list()),
    "preference.fm::fit_conjoint_foundation_model",
    fixed = TRUE
  )
})

test_that("foundation bundle compatibility shims delegate to preference.fm", {
  calls <- list()
  testthat::local_mocked_bindings(
    cs_foundation_preference_fm_export = function(name) {
      function(...) {
        calls[[name]] <<- list(...)
        list(delegate = name)
      }
    },
    .package = "strategize"
  )

  save_out <- NULL
  expect_warning(
    save_out <- save_conjoint_foundation_bundle(
      file = "foundation_bundle",
      foundation_model = "fit",
      overwrite = TRUE,
      compress = FALSE
    ),
    "preference.fm::save_conjoint_foundation_bundle"
  )
  expect_identical(save_out$delegate, "save_conjoint_foundation_bundle")
  expect_identical(calls$save_conjoint_foundation_bundle$file, "foundation_bundle")
  expect_identical(calls$save_conjoint_foundation_bundle$foundation_model, "fit")
  expect_true(calls$save_conjoint_foundation_bundle$overwrite)
  expect_false(calls$save_conjoint_foundation_bundle$compress)

  load_out <- NULL
  expect_warning(
    load_out <- load_conjoint_foundation_bundle(
      file = "foundation_bundle",
      conda_env = "unit_env",
      conda_env_required = FALSE,
      preload_params = TRUE
    ),
    "preference.fm::load_conjoint_foundation_bundle"
  )
  expect_identical(load_out$delegate, "load_conjoint_foundation_bundle")
  expect_identical(calls$load_conjoint_foundation_bundle$file, "foundation_bundle")
  expect_identical(calls$load_conjoint_foundation_bundle$conda_env, "unit_env")
  expect_false(calls$load_conjoint_foundation_bundle$conda_env_required)
  expect_true(calls$load_conjoint_foundation_bundle$preload_params)
})

test_that("foundation wrapper defaults use the stable pairwise neural path", {
  control <- strategize:::cs_foundation_default_control()

  expect_identical(control$neural_mcmc_control$low_rank_interaction_rank, 16L)
  expect_identical(control$neural_mcmc_control$low_rank_logit_normalization, "rms")
  expect_identical(control$neural_mcmc_control$low_rank_logit_transform, "none")
  expect_identical(control$neural_mcmc_control$additive_utility, "off")
  expect_false(isTRUE(control$neural_mcmc_control$calibration$enabled))
})

test_that("current semantic foundation adaptation is routed to preference.fm", {
  group_key <- strategize:::cs_foundation_universal_group_key()
  fake <- structure(
    list(
      groups = stats::setNames(
        list(list(group_key = group_key, transfer_mode = "semantic_zero_overlap")),
        group_key
      )
    ),
    class = "conjoint_foundation_model"
  )

  expect_error(
    adapt_conjoint_foundation_model(
      foundation_model = fake,
      Y = c(0, 1),
      W = data.frame(price = c("Low", "High"), stringsAsFactors = FALSE),
      mode = "pairwise",
      pair_id = c(1, 1),
      profile_order = c(1, 2)
    ),
    "preference.fm::adapt_conjoint_foundation_model",
    fixed = TRUE
  )
})

test_that("foundation adaptation outcome normalization rejects invalid targets", {
  base <- list(
    experiment_id = "invalid_adapt_outcome",
    W = data.frame(policy = c("A", "B", "A"), stringsAsFactors = FALSE),
    Y = c(0, 1, 0)
  )

  all_missing <- base
  all_missing$Y <- c(NA_real_, NA_real_, NA_real_)
  expect_error(
    strategize:::cs_foundation_normalize_experiment(all_missing, index = 1L),
    "at least one non-missing"
  )

  character_auto <- base
  character_auto$Y <- c("A", "B", "A")
  expect_error(
    strategize:::cs_foundation_normalize_experiment(character_auto, index = 1L),
    "Cannot infer likelihood for non-numeric outcomes"
  )

  bernoulli_missing <- base
  bernoulli_missing$Y <- c(0, 1, NA_real_)
  bernoulli_missing$likelihood <- "bernoulli"
  expect_error(
    strategize:::cs_foundation_normalize_experiment(bernoulli_missing, index = 1L),
    "contains missing, non-finite, or non-numeric"
  )

  normal_inf <- base
  normal_inf$Y <- c(0.1, Inf, 0.3)
  normal_inf$likelihood <- "normal"
  expect_error(
    strategize:::cs_foundation_normalize_experiment(normal_inf, index = 1L),
    "contains missing, non-finite, or non-numeric"
  )
})

test_that("internal checkpoint directory loader restores direct params", {
  skip_on_cran()
  skip_if_no_jax()
  orbax_available <- tryCatch({
    reticulate::use_condaenv("strategize_env", required = TRUE)
    reticulate::py_module_available("orbax.checkpoint")
  }, error = function(e) FALSE)
  skip_if_not(orbax_available, "orbax.checkpoint not available")

  strategize:::initialize_jax(conda_env = "strategize_env", conda_env_required = TRUE)
  ocp <- tryCatch(
    reticulate::import("orbax.checkpoint.experimental.v1", convert = FALSE),
    error = function(e) NULL
  )
  if (is.null(ocp) || !reticulate::py_has_attr(ocp, "save_pytree")) {
    ocp <- reticulate::import("orbax.checkpoint", convert = FALSE)
  }
  skip_if_not(
    reticulate::py_has_attr(ocp, "save_pytree") ||
      reticulate::py_has_attr(ocp, "PyTreeCheckpointer"),
    "unsupported orbax checkpoint API"
  )
  rewrite_checkpoint_sharding <- function(arrays_path, device_str) {
    sharding_path <- file.path(arrays_path, "pytree", "_sharding")
    skip_if_not(file.exists(sharding_path), "checkpoint did not write sharding metadata")
    sharding <- jsonlite::read_json(sharding_path, simplifyVector = FALSE)
    sharding <- lapply(sharding, function(entry_json) {
      entry <- jsonlite::fromJSON(entry_json, simplifyVector = FALSE)
      entry$device_str <- device_str
      as.character(jsonlite::toJSON(entry, auto_unbox = TRUE))
    })
    jsonlite::write_json(sharding, sharding_path, auto_unbox = TRUE)
  }

  group_key <- strategize:::cs_foundation_universal_group_key()
  tmp <- tempfile()
  dir.create(tmp, recursive = TRUE)
  bundle <- structure(
    list(
      schema_version = 1L,
      model_type = "conjoint_foundation",
      groups = list(),
      metadata = list(created_at = Sys.time())
    ),
    class = c("conjoint_foundation_bundle", "list")
  )
  bundle$groups[[group_key]] <- list(
    group_key = group_key,
    mode = "universal",
    likelihood = "bernoulli",
    n_outcomes = 1L,
    supported_modes = "pairwise",
    supported_likelihoods = "bernoulli",
    supported_pairwise_context_modes = "stage_free",
    experiment_ids = "study_a",
    encoder = list(
      factor_names = "price",
      names_list = list(price = list(c("Low", "High"))),
      factor_levels = c(price = 2L),
      unknown_policy = "holdout"
    ),
    schema_registry = list(
      slot_table = data.frame(
        slot_name = "slot_001",
        slot_key = "canon::price",
        display_label = "price",
        stringsAsFactors = FALSE
      ),
      pooled_names_list = list(slot_001 = list(c("canon::low", "canon::high"))),
      slot_level_keys = list(slot_001 = c("canon::low", "canon::high")),
      slot_level_labels = list(slot_001 = c("Low", "High")),
      experiment_maps = list()
    ),
    x_feature_names = character(0),
    x_schema = list(
      base_x_names = character(0),
      experiment_indicator_names = character(0),
      semantic_feature_names = character(0),
      experiment_token_levels = "study_a"
    ),
    text_registry = NULL,
    token_control = list(),
    fit = list(
      theta_mean = NULL,
      theta_var = NULL,
      neural_model_info = list(
        params = NULL,
        param_names = "W_out",
        param_shapes = list(c(1L, 1L)),
        param_sizes = list(1L),
        param_offsets = list(0L),
        n_params = 1L
      ),
      fit_metrics = NULL
    )
  )
  saveRDS(bundle, file.path(tmp, "metadata.rds"))

  manifest <- list(
    schema_version = 1L,
    artifact_type = "conjoint_foundation_checkpoint",
    writer = list(package = "preference.fm", version = "0.0.1"),
    compatible = list(strategize_version = as.character(utils::packageVersion("strategize"))),
    arrays = list(
      format = "orbax_pytree",
      path = "arrays",
      groups = list(
        group_001 = list(
          group_key = group_key,
          params = list(W_out = list(shape = c(1L, 1L), dtype = "float32")),
          text_registry = list()
        )
      )
    )
  )
  jsonlite::write_json(
    manifest,
    file.path(tmp, "manifest.json"),
    auto_unbox = TRUE,
    null = "null"
  )
  tree <- list(
    groups = list(
      group_001 = list(
        params = list(
          W_out = strategize:::strenv$jnp$ones(reticulate::tuple(1L, 1L))
        )
      )
    )
  )
  if (reticulate::py_has_attr(ocp, "save_pytree")) {
    ocp$save_pytree(file.path(tmp, "arrays"), tree)
  } else {
    ocp$PyTreeCheckpointer()$save(file.path(tmp, "arrays"), item = tree)
  }
  rewrite_checkpoint_sharding(file.path(tmp, "arrays"), "cuda:987654")

  loaded <- strategize:::cs_foundation_load_checkpoint_dir(tmp, preload_params = TRUE)
  expect_s3_class(loaded, "conjoint_foundation_model")
  expect_false(is.null(loaded$groups[[group_key]]$fit$neural_model_info$params$W_out))
  expect_false(is.null(loaded$groups[[group_key]]$fit$params$W_out))
  expect_equal(loaded$groups[[group_key]]$fit$theta_mean, 1)

  manifest_with_theta <- manifest
  manifest_with_theta$arrays$groups$group_001$theta_mean <- list(shape = 1L, dtype = "float32")
  manifest_with_theta$arrays$groups$group_001$theta_var <- list(shape = 1L, dtype = "float32")
  abstract_tree <- strategize:::cs_foundation_build_abstract_tree(manifest_with_theta$arrays)
  expect_named(abstract_tree$groups$group_001, c("params", "theta_mean", "theta_var"))

  tmp_direct <- tempfile()
  dir.create(tmp_direct, recursive = TRUE)
  saveRDS(bundle, file.path(tmp_direct, "metadata.rds"))
  manifest_direct <- manifest
  manifest_direct$arrays$groups$group_001$theta_mean <- list(shape = 1L, dtype = "float32")
  jsonlite::write_json(
    manifest_direct,
    file.path(tmp_direct, "manifest.json"),
    auto_unbox = TRUE,
    null = "null"
  )
  tree_direct <- tree
  tree_direct$groups$group_001$theta_mean <- strategize:::strenv$jnp$multiply(
    strategize:::strenv$jnp$ones(reticulate::tuple(1L)),
    42L
  )$astype(strategize:::strenv$dtj)
  if (reticulate::py_has_attr(ocp, "save_pytree")) {
    ocp$save_pytree(file.path(tmp_direct, "arrays"), tree_direct)
  } else {
    ocp$PyTreeCheckpointer()$save(file.path(tmp_direct, "arrays"), item = tree_direct)
  }
  rewrite_checkpoint_sharding(file.path(tmp_direct, "arrays"), "mps:987654")

  loaded_direct <- strategize:::cs_foundation_load_checkpoint_dir(tmp_direct, preload_params = FALSE)
  expect_s3_class(loaded_direct, "conjoint_foundation_model")
  expect_false(is.null(loaded_direct$groups[[group_key]]$fit$neural_model_info$params$W_out))
  expect_equal(loaded_direct$groups[[group_key]]$fit$theta_mean, 42)
})
