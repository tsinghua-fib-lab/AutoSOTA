# =============================================================================
# Tests for Input Validation
# =============================================================================
# These tests verify that the input validation functions catch common
# errors and provide helpful error messages.
# =============================================================================

# =============================================================================
# Y and W Validation
# =============================================================================

test_that("check_jax_available is exported", {
  expect_true("check_jax_available" %in% getNamespaceExports("strategize"))
  expect_true(is.function(strategize::check_jax_available))
})

test_that("validate_strategize_inputs catches missing Y", {
  skip_on_cran()

  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(W = W, lambda = 0.1),
    "'Y' is required"
  )
})

test_that("validate_strategize_inputs catches missing W", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)

  expect_error(
    validate_strategize_inputs(Y = Y, lambda = 0.1),
    "'W' is required"
  )
})

test_that("validate_strategize_inputs catches Y/W dimension mismatch", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F"))  # Only 2 rows

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1),
    "Dimension mismatch"
  )
})

test_that("validate_strategize_inputs catches non-numeric Y", {
  skip_on_cran()

  Y <- c("a", "b", "c", "d")
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1),
    "'Y' must be numeric"
  )
})

test_that("validate_strategize_inputs catches missing W values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(
    Gender = c("M", "F", NA, "F"),
    Message = c("Jobs", "Taxes", "Jobs", NA),
    stringsAsFactors = FALSE
  )

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1),
    "Gender.*Message"
  )
})

test_that("validate_strategize_inputs warns about NA in Y", {
  skip_on_cran()

  Y <- c(1, 0, NA, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_warning(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1),
    "NA values"
  )
})

# =============================================================================
# Lambda Validation
# =============================================================================

test_that("validate_strategize_inputs catches missing lambda", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(Y = Y, W = W),
    "'lambda' is required"
  )
})

test_that("validate_strategize_inputs catches negative lambda", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = -0.1),
    "'lambda' must be non-negative"
  )
})

test_that("validate_strategize_inputs catches multi-value lambda", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = c(0.1, 0.2)),
    "'lambda' must be a single"
  )
})

# =============================================================================
# p_list Validation
# =============================================================================

test_that("validate_strategize_inputs catches p_list length mismatch", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"), Age = c("Y", "O", "Y", "O"))
  p_list <- list(Gender = c(M = 0.5, F = 0.5))  # Missing Age

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, p_list = p_list),
    "p_list.*elements.*columns"
  )
})

test_that("validate_strategize_inputs catches p_list level mismatch", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))  # Wrong level names

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, p_list = p_list),
    "level mismatch"
  )
})

test_that("validate_strategize_inputs catches unnamed p_list elements", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  p_list <- list(Gender = c(0.5, 0.5))  # No level names

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, p_list = p_list),
    "named elements"
  )
})

test_that("validate_strategize_inputs catches p_list not summing to 1", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  p_list <- list(Gender = c(M = 0.4, F = 0.4))  # Sum to 0.8

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, p_list = p_list),
    "sum to"
  )
})

test_that("validate_strategize_inputs catches non-finite p_list probabilities", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  p_list <- list(Gender = c(M = NA_real_, F = 1))

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, p_list = p_list),
    "finite, non-missing"
  )
})

test_that("validate_strategize_inputs catches negative probabilities in p_list", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  p_list <- list(Gender = c(M = -0.5, F = 1.5))

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, p_list = p_list),
    "negative"
  )
})

# =============================================================================
# Adversarial Mode Validation
# =============================================================================

test_that("validate_strategize_inputs catches missing adversarial variables", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  # Missing competing_group_variable_respondent
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      adversarial = TRUE
    ),
    "competing_group_variable_respondent"
  )

  # Missing competing_group_variable_candidate
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      adversarial = TRUE,
      competing_group_variable_respondent = c("D", "R", "D", "R")
    ),
    "competing_group_variable_candidate"
  )
})

test_that("validate_strategize_inputs catches non-binary groups in adversarial mode", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      adversarial = TRUE,
      competing_group_variable_respondent = c("A", "B", "C", "D"),  # 4 groups
      competing_group_variable_candidate = c("A", "B", "C", "D")
    ),
    "exactly 2 groups"
  )
})

test_that("validate_strategize_inputs catches adversarial group mismatches", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      adversarial = TRUE,
      competing_group_variable_respondent = c("D", "R", "D", "R"),
      competing_group_variable_candidate = c("A", "B", "A", "B")
    ),
    "matching group labels"
  )
})

test_that("validate_strategize_inputs catches mismatched adversarial variable length", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      adversarial = TRUE,
      competing_group_variable_respondent = c("D", "R"),  # Wrong length
      competing_group_variable_candidate = c("D", "R", "D", "R")
    ),
    "elements.*rows"
  )
})

# =============================================================================
# Model Type Validation
# =============================================================================

test_that("validate_strategize_inputs catches invalid outcome_model_type", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      outcome_model_type = "invalid_type"
    ),
    "outcome_model_type.*must be one of"
  )
})

test_that("validate_strategize_inputs catches invalid penalty_type", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      penalty_type = "invalid_penalty"
    ),
    "penalty_type.*must be one of"
  )
})

# =============================================================================
# K Validation
# =============================================================================

test_that("validate_strategize_inputs catches invalid K", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, K = 0),
    "'K' must be a positive integer"
  )

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, K = -1),
    "'K' must be a positive integer"
  )

  expect_error(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, K = 1.5),
    "'K' must be a positive integer"
  )
})

test_that("validate_strategize_inputs warns about K > 1 without X", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_warning(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, K = 2, X = NULL),
    "K > 1.*requires"
  )
})

# =============================================================================
# diff Mode Validation
# =============================================================================

test_that("validate_strategize_inputs warns about diff without pair_id", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_warning(
    validate_strategize_inputs(Y = Y, W = W, lambda = 0.1, diff = TRUE),
    "diff=TRUE.*pair_id"
  )
})

test_that("validate_strategize_inputs warns about diff without profile_order", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  pair_id <- c(1, 1, 2, 2)

  expect_warning(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      diff = TRUE, pair_id = pair_id
    ),
    "profile_order"
  )
})

test_that("validate_strategize_inputs errors on malformed neural diff metadata", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      diff = TRUE,
      outcome_model_type = "neural"
    ),
    "pair_id"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      diff = TRUE,
      pair_id = c(1, 1, 2, 2),
      profile_order = c(1, 1, 1, 2),
      outcome_model_type = "neural"
    ),
    "profile_order"
  )
})

test_that("validate_strategize_inputs validates neural seed", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_true(validate_strategize_inputs(
    Y = Y, W = W, lambda = 0.1,
    neural_mcmc_control = list(seed = 42L)
  ))
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(seed = NA_integer_)
    ),
    "seed"
  )
})

# =============================================================================
# Neural MCMC Control Validation
# =============================================================================

test_that("validate_strategize_inputs accepts cross_candidate_encoder options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  allowed <- list(TRUE, FALSE, "none", "term", "attn", "lite", "cross_attn",
                  "cls_attn", "cross-attn", "cls-attn", "full", "true", "false")

  for (val in allowed) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(cross_candidate_encoder = val)
      ),
      info = sprintf(
        "Expected cross_candidate_encoder=%s to be accepted",
        as.character(val)
      )
    )
  }
})

test_that("validate_strategize_inputs rejects invalid cross_candidate_encoder values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(cross_candidate_encoder = "bad_value")
    ),
    "cross_candidate_encoder"
  )
})

test_that("validate_strategize_inputs accepts low-rank logit control options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_true(validate_strategize_inputs(
    Y = Y, W = W, lambda = 0.1,
    neural_mcmc_control = list(
      low_rank_logit_transform = "softclip",
      low_rank_logit_bound = 1.5,
      low_rank_logit_softness = 0.25,
      low_rank_logit_normalization = "rms",
      low_rank_head_weight_target_rms = 0.1,
      low_rank_rc_out_target_rms = 0.25,
      additive_utility_normalization = "rms",
      additive_head_weight_target_rms = 0.25
    )
  ))
  expect_true(validate_strategize_inputs(
    Y = Y, W = W, lambda = 0.1,
    neural_mcmc_control = list(low_rank_logit_transform = "none")
  ))
})

test_that("validate_strategize_inputs rejects invalid low-rank logit controls", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(low_rank_logit_transform = "clip")
    ),
    "low_rank_logit_transform"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(low_rank_logit_bound = "wide")
    ),
    "low_rank_logit_bound"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(low_rank_logit_softness = 0)
    ),
    "low_rank_logit_softness"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(low_rank_logit_normalization = "batchnorm")
    ),
    "low_rank_logit_normalization"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(low_rank_head_weight_target_rms = 0)
    ),
    "low_rank_head_weight_target_rms"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(low_rank_rc_out_target_rms = -1)
    ),
    "low_rank_rc_out_target_rms"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(additive_utility_normalization = "batchnorm")
    ),
    "additive_utility_normalization"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(additive_head_weight_target_rms = 0)
    ),
    "additive_head_weight_target_rms"
  )
})

test_that("validate_strategize_inputs accepts qk_norm options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  for (val in list(TRUE, FALSE, "true", "false")) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(qk_norm = val)
      ),
      info = sprintf("Expected qk_norm=%s to be accepted", as.character(val))
    )
  }
})

test_that("validate_strategize_inputs rejects invalid qk_norm values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(qk_norm = "bad_value")
    ),
    "qk_norm"
  )
})

test_that("validate_strategize_inputs accepts residual_mode options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  for (val in list(TRUE, FALSE, "standard", "additive", "rezero", "full_attn", "full-attn", "attnres")) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(residual_mode = val)
      ),
      info = sprintf("Expected residual_mode=%s to be accepted", as.character(val))
    )
  }
})

test_that("validate_strategize_inputs rejects invalid residual_mode values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(residual_mode = "bad_value")
    ),
    "residual_mode"
  )
})

test_that("validate_strategize_inputs accepts optimizer options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  allowed <- c("adam", "adamw", "adabelief", "muon")

  for (val in allowed) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(optimizer = val)
      ),
      info = sprintf("Expected optimizer=%s to be accepted", val)
    )
  }
})

test_that("validate_strategize_inputs rejects invalid optimizer values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(optimizer = "bad_value")
    ),
    "optimizer"
  )
})

test_that("validate_strategize_inputs accepts svi_lr_schedule options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  allowed <- c("none", "constant", "cosine", "warmup_cosine")

  for (val in allowed) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(svi_lr_schedule = val)
      ),
      info = sprintf("Expected svi_lr_schedule=%s to be accepted", val)
    )
  }
})

test_that("validate_strategize_inputs rejects invalid svi_lr_schedule values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(svi_lr_schedule = "bad_value")
    ),
    "svi_lr_schedule"
  )
})

test_that("validate_strategize_inputs accepts svi_steps options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  allowed <- list(1L, 200L, 1000, "optimal")

  for (val in allowed) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(svi_steps = val)
      ),
      info = sprintf("Expected svi_steps=%s to be accepted", as.character(val))
    )
  }
})

test_that("validate_strategize_inputs rejects invalid svi_steps values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(svi_steps = 0)
    ),
    "svi_steps"
  )

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(svi_steps = "bad_value")
    ),
    "svi_steps"
  )
})

test_that("validate_strategize_inputs accepts early_stopping logical options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  for (val in c(TRUE, FALSE)) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(early_stopping = val)
      ),
      info = sprintf("Expected early_stopping=%s to be accepted", as.character(val))
    )
  }
})

test_that("validate_strategize_inputs rejects invalid early_stopping values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(early_stopping = "yes")
    ),
    "early_stopping"
  )
})

test_that("validate_strategize_inputs accepts gradient_diagnostics logical options", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  for (val in c(TRUE, FALSE)) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(gradient_diagnostics = val)
      ),
      info = sprintf("Expected gradient_diagnostics=%s to be accepted", as.character(val))
    )
  }
})

test_that("validate_strategize_inputs rejects invalid gradient_diagnostics values", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(gradient_diagnostics = "yes")
    ),
    "gradient_diagnostics"
  )
})

test_that("validate_strategize_inputs validates universal loss weighting controls", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  for (policy in c("empirical", "balanced_cell")) {
    expect_true(
      validate_strategize_inputs(
        Y = Y,
        W = W,
        lambda = 0.1,
        neural_mcmc_control = list(
          universal_loss_weighting = policy,
          universal_loss_weight_clip = c(0.25, 4)
        )
      ),
      info = sprintf("Expected universal_loss_weighting=%s to be accepted", policy)
    )
  }

  expect_error(
    validate_strategize_inputs(
      Y = Y,
      W = W,
      lambda = 0.1,
      neural_mcmc_control = list(universal_loss_weighting = "equal_cells")
    ),
    "universal_loss_weighting"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y,
      W = W,
      lambda = 0.1,
      neural_mcmc_control = list(universal_loss_weight_clip = c(4, 0.25))
    ),
    "universal_loss_weight_clip"
  )
})

test_that("validate_strategize_inputs accepts early stopping validation size controls", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_true(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(
        early_stopping_validation_frac = 0.25,
        early_stopping_validation_max_n = 128L,
        early_stopping_validation_batch_size = 64L
      )
    )
  )
  expect_true(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(
        early_stopping_validation_frac = 1,
        early_stopping_validation_max_n = NULL
      )
    )
  )
})

test_that("validate_strategize_inputs rejects invalid early stopping validation controls", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(early_stopping_validation_frac = 0)
    ),
    "early_stopping_validation_frac"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(early_stopping_validation_frac = 1.5)
    ),
    "early_stopping_validation_frac"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(early_stopping_validation_max_n = 0)
    ),
    "early_stopping_validation_max_n"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(early_stopping_validation_max_n = 3.5)
    ),
    "early_stopping_validation_max_n"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(early_stopping_validation_batch_size = 0)
    ),
    "early_stopping_validation_batch_size"
  )
})

test_that("validate_strategize_inputs validates compact update scan controls", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  for (val in c("required", "fallback")) {
    expect_true(
      validate_strategize_inputs(
        Y = Y, W = W, lambda = 0.1,
        neural_mcmc_control = list(
          compact_update_chunk_size = 2L,
          compact_update_scan = val
        )
      ),
      info = sprintf("Expected compact_update_scan=%s to be accepted", val)
    )
  }
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(compact_update_chunk_size = 0L)
    ),
    "compact_update_chunk_size"
  )
  expect_error(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      neural_mcmc_control = list(compact_update_scan = "silent")
    ),
    "compact_update_scan"
  )
})

# =============================================================================
# CV Validation
# =============================================================================

test_that("validate_cv_strategize_inputs catches invalid folds", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_cv_strategize_inputs(Y = Y, W = W, folds = 1),
    "'folds' must be.*>= 2"
  )
})

test_that("validate_cv_strategize_inputs catches folds > n", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_error(
    validate_cv_strategize_inputs(Y = Y, W = W, folds = 10),
    "'folds'.*cannot exceed"
  )
})

test_that("validate_cv_strategize_inputs warns about single lambda", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_warning(
    validate_cv_strategize_inputs(Y = Y, W = W, lambda_seq = c(0.1)),
    "only 1 value"
  )
})

# =============================================================================
# Valid Input Tests (should pass without error)
# =============================================================================

test_that("validate_strategize_inputs passes for valid inputs", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))
  p_list <- list(Gender = c(M = 0.5, F = 0.5))

  expect_true(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1, p_list = p_list
    )
  )
})

test_that("validate_strategize_inputs passes for valid adversarial inputs", {
  skip_on_cran()

  Y <- c(1, 0, 1, 0)
  W <- data.frame(Gender = c("M", "F", "M", "F"))

  expect_true(
    validate_strategize_inputs(
      Y = Y, W = W, lambda = 0.1,
      adversarial = TRUE,
      competing_group_variable_respondent = c("D", "R", "D", "R"),
      competing_group_variable_candidate = c("D", "R", "D", "R")
    )
  )
})
