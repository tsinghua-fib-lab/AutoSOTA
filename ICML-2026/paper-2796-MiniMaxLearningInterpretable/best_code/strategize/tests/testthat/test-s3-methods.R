make_strategize_result <- function(multi = FALSE) {
  p_list <- list(Gender = c(M = 0.5, F = 0.5))
  pi_star_point <- if (!multi) {
    list(k1 = list(Gender = c(M = 0.6, F = 0.4)))
  } else {
    list(
      k1 = list(Gender = c(M = 0.6, F = 0.4)),
      k2 = list(Gender = c(M = 0.4, F = 0.6))
    )
  }
  pi_star_se <- lapply(pi_star_point, function(cluster) {
    lapply(cluster, function(levels) rep(0.01, length(levels)))
  })
  res <- list(
    pi_star_point = pi_star_point,
    pi_star_se = pi_star_se,
    Q_point = 0.6,
    Q_se = 0.1,
    lambda = 0.1,
    penalty_type = "KL",
    p_list = p_list
  )
  class(res) <- "strategize_result"
  res
}

test_that("print.strategize_result outputs core sections", {
  skip_on_cran()

  res <- make_strategize_result()
  output <- capture.output(print(res))
  expect_true(any(grepl("strategize Result", output)))
  expect_true(any(grepl("Optimal Distribution", output)))
  expect_true(any(grepl("Expected Outcome", output)))
  expect_true(any(grepl("Settings: lambda", output)))
})

test_that("print.strategize_result handles multi-cluster output", {
  skip_on_cran()

  res <- make_strategize_result(multi = TRUE)
  output <- capture.output(print(res))
  expect_true(any(grepl("Cluster k1", output)))
  expect_true(any(grepl("Cluster k2", output)))
})

test_that("summary.strategize_result returns a comparison table", {
  skip_on_cran()

  res <- make_strategize_result()
  output <- capture.output(summary_res <- summary(res))
  expect_true(is.data.frame(summary_res))
  expect_true(all(c("Factor", "Level", "Baseline", "Optimal", "Change", "SE") %in% names(summary_res)))
  expect_true(any(grepl("Summary: Distribution Changes", output)))
})

test_that("summary.strategize_result handles missing data", {
  skip_on_cran()

  res <- list(pi_star_point = NULL, p_list = NULL)
  class(res) <- "strategize_result"
  output <- capture.output(summary_res <- summary(res))
  expect_null(summary_res)
  expect_true(any(grepl("Insufficient data", output)))
})

test_that("summary.strategize_result handles unnamed cluster list", {
  skip_on_cran()

  p_list <- list(Gender = c(M = 0.5, F = 0.5))
  res <- list(
    pi_star_point = list(list(Gender = c(M = 0.55, F = 0.45))),
    pi_star_se = list(list(Gender = c(0.01, 0.01))),
    Q_point = 0.6,
    p_list = p_list
  )
  class(res) <- "strategize_result"
  summary_res <- summary(res)
  expect_true(is.data.frame(summary_res))
})
