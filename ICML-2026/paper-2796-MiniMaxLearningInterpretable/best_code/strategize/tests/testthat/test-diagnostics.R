test_that("post-screen GLM design diagnostic reports dimensions and regularization state", {
  msg <- cs_glm_design_size_error_message(
    glm_input = matrix(0, nrow = 4, ncol = 5),
    main_dat = matrix(0, nrow = 4, ncol = 2),
    interacted_dat = matrix(0, nrow = 4, ncol = 3),
    use_regularization_requested = TRUE,
    screening_applied = TRUE,
    outcome_model_key = "analysis_001"
  )

  expect_true(grepl("Post-screen GLM design is too large", msg, fixed = TRUE))
  expect_true(grepl("5 column(s) for 4 observation(s)", msg, fixed = TRUE))
  expect_true(grepl("limit is 2 column(s)", msg, fixed = TRUE))
  expect_true(grepl("use_regularization=TRUE", msg, fixed = TRUE))
  expect_true(grepl("screening_applied=TRUE", msg, fixed = TRUE))
  expect_true(grepl("main=2, interactions=3", msg, fixed = TRUE))
  expect_true(grepl("regularization/glinternet screening", msg, fixed = TRUE))
  expect_true(grepl("analysis_001", msg, fixed = TRUE))
})

test_that("crossfit fold diagnostic preserves fold context and inner error", {
  msg <- cs_crossfit_q_fold_error_message(
    fold = 2L,
    n_folds = 3L,
    split_by = "pair_id",
    train_pair_rows = 1:4,
    test_pair_rows = 5:6,
    train_rows = 1:8,
    use_regularization = TRUE,
    message = "inner failure"
  )

  expect_true(grepl("crossfit_q fold 2/3 training strategize() failed", msg, fixed = TRUE))
  expect_true(grepl("split_by=pair_id", msg, fixed = TRUE))
  expect_true(grepl("train_pairs=4", msg, fixed = TRUE))
  expect_true(grepl("test_pairs=2", msg, fixed = TRUE))
  expect_true(grepl("train_rows=8", msg, fixed = TRUE))
  expect_true(grepl("use_regularization=TRUE", msg, fixed = TRUE))
  expect_true(grepl("inner failure", msg, fixed = TRUE))
})
