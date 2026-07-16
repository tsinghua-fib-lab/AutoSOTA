test_that("stratified fold helper preserves clusters and class balance", {
  cluster <- rep(seq_len(12), each = 2L)
  y <- rep(c(rep(0, 6), rep(1, 6)), each = 2L)

  folds1 <- strategize:::cs_make_stratified_folds(
    n = length(y),
    n_folds = 3L,
    y = y,
    cluster = cluster,
    seed = 123L
  )
  folds2 <- strategize:::cs_make_stratified_folds(
    n = length(y),
    n_folds = 3L,
    y = y,
    cluster = cluster,
    seed = 123L
  )

  expect_false(is.null(folds1))
  expect_identical(folds1$fold_id, folds2$fold_id)
  expect_identical(folds1$n_folds, 3L)

  cluster_split <- vapply(split(folds1$fold_id, cluster), function(x) {
    length(unique(x))
  }, integer(1))
  expect_true(all(cluster_split == 1L))

  class_table <- table(folds1$fold_id, y)
  expect_equal(dim(class_table), c(3L, 2L))
  expect_true(all(class_table[, "0"] > 0L))
  expect_true(all(class_table[, "1"] > 0L))
})

test_that("pairwise performance fixture avoids tied or identical pairs", {
  dat <- generate_pairwise_performance_test_data(
    n_pairs = 1000L,
    n_factors = 3,
    n_levels = 2,
    seed = 20260327
  )

  expect_false(any(dat$identical_pair))
  expect_true(all(dat$pair_margin > 0))
  expect_equal(length(dat$pair_margin), length(unique(dat$pair_id)))
})
