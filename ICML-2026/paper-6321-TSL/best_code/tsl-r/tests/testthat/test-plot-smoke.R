# The plot_*() functions are thin ggplot builders over the (separately tested)
# compute layer. Here we just confirm each returns a buildable ggplot/patchwork
# object across the normal path and a few degenerate inputs.

skip_if_not_installed("ggplot2")

build_ok <- function(p) {
  if (inherits(p, "patchwork")) {
    tmp <- tempfile(fileext = ".pdf")
    grDevices::pdf(tmp)
    on.exit({ grDevices::dev.off(); unlink(tmp) }, add = TRUE)
    print(p)
    expect_true(TRUE)
  } else {
    expect_s3_class(p, "ggplot")
    expect_no_error(ggplot2::ggplot_build(p))
  }
  invisible(p)
}

test_that("every plot_*() builds on a normal multi-stage model", {
  d <- make_data(n = 180)
  fit <- tsl(d$x, d$y, epochs = 4L, n_trees = 4L, seed = 2L, verbosity = 0L)

  build_ok(plot_first_order_pd(fit))
  build_ok(plot_first_order_pd(fit, show_data_density = TRUE))
  build_ok(pd_difference_plot(fit, features = c("a", "b")))
  build_ok(plot_2d_pd(fit, feature_x = "a", feature_y = "b", grid_points = 20L))
  build_ok(plot_ice(fit, feature = "a", n_ice = 15L, grid_points = 30L))
  build_ok(plot_tilt_1d(fit, features = c("a", "b")))
  build_ok(plot_2d_tilt(fit, feature_x = "a", feature_y = "b", grid_points = 20L))
  build_ok(plot_tilt_diagnostics(fit, features = "a"))
  build_ok(plot_2d_backbone(fit, feature_x = "a", feature_y = "b",
                            grid_points = 20L))
  build_ok(plot_feature_importance(fit))
  build_ok(plot_local_interpretation(fit, list(d$x[1, ], d$x[2, ]),
                                     titles = c("A", "B")))

  comp <- tsl_components(fit)
  build_ok(plot_grid_tensor_components(comp[[1]]$combined_grid_tensor,
                                       feature_names = colnames(d$x)))
  build_ok(plot_combined_grid_tensors(fit))
  build_ok(plot_epoch_components(fit, 1L))
})

test_that("autoplot.tsl dispatches by type", {
  d <- make_data(n = 120)
  fit <- tsl(d$x, d$y, epochs = 3L, n_trees = 3L, seed = 4L, verbosity = 0L)
  build_ok(ggplot2::autoplot(fit, type = "pd", features = "a"))
  build_ok(ggplot2::autoplot(fit, type = "tilt_2d", feature_x = "a",
                             feature_y = "b", grid_points = 15L))
  build_ok(ggplot2::autoplot(fit, type = "importance"))
})

test_that("plots build on degenerate inputs (single stage, component scale)", {
  d <- make_data(n = 120)
  fit1 <- tsl(d$x, d$y, epochs = 1L, n_trees = 3L, seed = 6L, verbosity = 0L)
  build_ok(plot_first_order_pd(fit1))
  build_ok(plot_tilt_1d(fit1, features = "a"))
  build_ok(plot_feature_importance(fit1))
  build_ok(plot_first_order_pd(fit1, features = "a", scale = "component"))
})

test_that("plots build with a binary feature", {
  set.seed(9)
  n <- 150
  x <- cbind(a = runif(n, -2, 2), b = rbinom(n, 1, 0.5), c = runif(n, -2, 2))
  y <- 2 * x[, "a"] - x[, "b"] + 0.5 * x[, "c"] + rnorm(n, sd = 0.1)
  fit <- tsl(x, y, epochs = 4L, n_trees = 4L, seed = 1L, verbosity = 0L)
  build_ok(plot_first_order_pd(fit, features = "b"))
  build_ok(plot_tilt_1d(fit, features = "b"))
})

test_that("plot_* attach recoverable data via tsl_plot_data()", {
  d <- make_data(n = 100)
  fit <- tsl(d$x, d$y, epochs = 3L, n_trees = 3L, seed = 1L, verbosity = 0L)
  p <- plot_first_order_pd(fit, features = "a")
  expect_s3_class(tsl_plot_data(p), "data.frame")
})
