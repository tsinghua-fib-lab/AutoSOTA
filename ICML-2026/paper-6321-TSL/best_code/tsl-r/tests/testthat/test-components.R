# Extraction of the glass-box structure via tsl_components(): one entry per
# boosting stage, each with OLS scalings and the two-tensor grid components
# (backbone b >= 0, tilt d, branch scalars lambda_+/lambda_-).

test_that("tsl_components() returns one well-formed entry per stage", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 6L, n_trees = 4L, seed = 123L, verbosity = 0L)
  comp <- tsl_components(fit)

  expect_type(comp, "list")
  expect_length(comp, 6L) # one stage per epoch

  stage <- comp[[1]]
  expect_named(
    stage,
    c("scaling_plus", "scaling_minus", "candidate_indices",
      "combined_grid_tensor", "grid_tensors"),
    ignore.order = TRUE
  )
  expect_length(stage$grid_tensors, 4L) # one component per tree

  g <- stage$combined_grid_tensor
  expect_named(
    g,
    c("splits", "backbone_values", "tilt_values", "observation_counts",
      "lambda_plus", "lambda_minus", "scaling"),
    ignore.order = TRUE
  )
})

test_that("non-tsl input is rejected", {
  expect_error(tsl_components(list()), "class")
})

test_that("grid-tensor shapes are consistent per feature", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 4L, n_trees = 3L, seed = 1L, verbosity = 0L)
  comp <- tsl_components(fit)

  for (stage in comp) {
    for (g in c(list(stage$combined_grid_tensor), stage$grid_tensors)) {
      # one axis per feature
      expect_length(g$backbone_values, 3L)
      expect_length(g$tilt_values, 3L)
      expect_length(g$splits, 3L)
      expect_length(g$observation_counts, 3L)

      for (j in 1:3) {
        n_intervals <- length(g$backbone_values[[j]])
        expect_gte(n_intervals, 1L)
        expect_length(g$tilt_values[[j]], n_intervals)
        expect_length(g$observation_counts[[j]], n_intervals)
        # one more interval than there are split points
        expect_length(g$splits[[j]], n_intervals - 1L)
      }
    }
  }
})

test_that("positivity and finiteness invariants hold", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 5L, n_trees = 4L, seed = 99L, verbosity = 0L)
  comp <- tsl_components(fit)

  for (stage in comp) {
    expect_true(is.finite(stage$scaling_plus))
    expect_true(is.finite(stage$scaling_minus))

    for (g in c(list(stage$combined_grid_tensor), stage$grid_tensors)) {
      expect_gte(g$lambda_plus, 0)   # lambda_+ >= 0
      expect_gte(g$lambda_minus, 0)  # lambda_- >= 0
      for (j in seq_along(g$backbone_values)) {
        # backbone is the non-negative gate magnitude; tilt is a real direction
        expect_true(all(g$backbone_values[[j]] >= 0))
        expect_true(all(is.finite(g$backbone_values[[j]])))
        expect_true(all(is.finite(g$tilt_values[[j]])))
        expect_true(all(g$observation_counts[[j]] >= 1))
      }
    }
  }
})

test_that("the combined grid tensor obeys the l2_identify gauge", {
  # With a single tree the stage's combined grid is the one identified
  # component, so the gauge is exact: per axis the backbone has unit weighted
  # L2 and the tilt has zero weighted mean (weights = observation_counts).
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 4L, n_trees = 1L, seed = 5L, verbosity = 0L)
  comp <- tsl_components(fit)

  for (stage in comp) {
    g <- stage$combined_grid_tensor
    for (j in seq_along(g$backbone_values)) {
      w <- g$observation_counts[[j]]
      b <- g$backbone_values[[j]]
      tilt <- g$tilt_values[[j]]
      wsum <- sum(w)
      expect_equal(sum(w * b^2) / wsum, 1, tolerance = 1e-6) # unit weighted L2
      expect_equal(sum(w * tilt) / wsum, 0, tolerance = 1e-6) # centered tilt
    }
  }
})

test_that("predictions reconstruct from the extracted components", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 6L, n_trees = 5L, seed = 321L, verbosity = 0L)
  comp <- tsl_components(fit)

  te <- make_data(n = 40, seed = 7)
  x_eval <- te$x

  # Rebuild the model prediction from the two-tensor fields, mirroring the
  # core's default ("Combined") aggregation:
  #   stage(x) = scaling_plus  * lambda_+ * prod_j b_j(x) * exp( d_j(x))
  #            - scaling_minus * lambda_- * prod_j b_j(x) * exp(-d_j(x))
  # and the model prediction is the sum over stages.
  reconstruct <- function(comp, x_eval) {
    n <- nrow(x_eval)
    p <- ncol(x_eval)
    out <- numeric(n)
    for (i in seq_len(n)) {
      total <- 0
      for (stage in comp) {
        g <- stage$combined_grid_tensor
        sp <- if (is.na(stage$scaling_plus)) 1 else stage$scaling_plus
        sm <- if (is.na(stage$scaling_minus)) 0 else stage$scaling_minus
        fp <- g$lambda_plus
        fm <- g$lambda_minus
        for (j in seq_len(p)) {
          splits_j <- g$splits[[j]]
          b_j <- g$backbone_values[[j]]
          tilt_j <- g$tilt_values[[j]]
          # interval index: count of splits <= x_j, clamped to the last
          idx <- min(sum(splits_j <= x_eval[i, j]) + 1L, length(b_j))
          b <- b_j[idx]
          dd <- tilt_j[idx]
          fp <- fp * b * exp(dd)
          fm <- fm * b * exp(-dd)
        }
        total <- total + sp * fp - sm * fm
      }
      out[i] <- total
    }
    out
  }

  expect_equal(reconstruct(comp, x_eval), predict(fit, x_eval),
               tolerance = 1e-8)
})

test_that("each per-tree grid obeys the l2_identify gauge", {
  # The combined grid is gauge-exact only for a single tree, but every per-tree
  # component is l2-identified at fit time, so the gauge holds for any n_trees.
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 4L, n_trees = 4L, seed = 13L, verbosity = 0L)
  comp <- tsl_components(fit)

  for (stage in comp) {
    for (g in stage$grid_tensors) {
      for (j in seq_along(g$backbone_values)) {
        w <- g$observation_counts[[j]]
        b <- g$backbone_values[[j]]
        tilt <- g$tilt_values[[j]]
        wsum <- sum(w)
        expect_equal(sum(w * b^2) / wsum, 1, tolerance = 1e-6)
        expect_equal(sum(w * tilt) / wsum, 0, tolerance = 1e-6)
      }
    }
  }
})

test_that("candidate_indices list every tree in the bag", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 3L, n_trees = 6L, seed = 7L, verbosity = 0L)
  comp <- tsl_components(fit)
  for (stage in comp) {
    expect_equal(sort(stage$candidate_indices), 1:6)
  }
})

test_that("observation counts partition the training rows per axis", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 3L, n_trees = 3L, seed = 21L, verbosity = 0L)
  comp <- tsl_components(fit)
  for (stage in comp) {
    for (g in c(list(stage$combined_grid_tensor), stage$grid_tensors)) {
      for (j in seq_along(g$observation_counts)) {
        expect_equal(sum(g$observation_counts[[j]]), nrow(d$x))
      }
    }
  }
})

test_that("similarity trimming still yields a valid model", {
  # similarity_threshold > 0 trims the bag when aggregating the combined grid;
  # the resulting model must still predict finitely and keep the invariants.
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 4L, n_trees = 6L, seed = 31L,
             similarity_threshold = 0.7, verbosity = 0L)

  te <- make_data(n = 40, seed = 8)
  expect_true(all(is.finite(predict(fit, te$x))))

  for (stage in tsl_components(fit)) {
    g <- stage$combined_grid_tensor
    expect_gte(g$lambda_plus, 0)
    expect_gte(g$lambda_minus, 0)
    for (j in seq_along(g$backbone_values)) {
      expect_true(all(g$backbone_values[[j]] >= 0))
      expect_true(all(is.finite(g$backbone_values[[j]])))
      expect_true(all(is.finite(g$tilt_values[[j]])))
    }
  }
})

test_that("the extracted structure is deterministic for a fixed seed", {
  d <- make_data()
  fit_args <- list(d$x, d$y, epochs = 3L, n_trees = 3L, seed = 55L,
                   verbosity = 0L)
  c1 <- tsl_components(do.call(tsl, fit_args))
  c2 <- tsl_components(do.call(tsl, fit_args))
  expect_identical(c1, c2)
})
