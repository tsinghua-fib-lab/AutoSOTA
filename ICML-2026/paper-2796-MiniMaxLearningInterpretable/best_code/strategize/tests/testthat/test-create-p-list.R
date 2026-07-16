test_that("create_p_list handles uniform and observed frequencies", {
  skip_on_cran()

  W <- data.frame(
    Gender = c("M", "F", "M", "F", "F"),
    Age = c("Y", "O", "Y", "Y", "O"),
    stringsAsFactors = FALSE
  )

  p_obs <- create_p_list(W, uniform = FALSE)
  expect_equal(unname(p_obs$Gender), as.numeric(prop.table(table(W$Gender))))
  expect_equal(names(p_obs$Gender), names(prop.table(table(W$Gender))))
  expect_equal(unname(p_obs$Age), as.numeric(prop.table(table(W$Age))))

  p_uniform <- create_p_list(W, uniform = TRUE)
  expect_equal(unname(p_uniform$Gender), rep(1 / 2, 2), tolerance = 1e-8)
  expect_equal(unname(p_uniform$Age), rep(1 / 2, 2), tolerance = 1e-8)
})

test_that("create_p_list assigns fallback factor names when columns unnamed", {
  skip_on_cran()

  W <- as.data.frame(matrix(c("A", "B", "A", "B"), ncol = 2))
  colnames(W) <- NULL

  p_list <- create_p_list(W, uniform = TRUE)
  expect_equal(names(p_list), c("Factor1", "Factor2"))
})

test_that("create_p_list errors on empty W", {
  skip_on_cran()

  W <- data.frame()
  expect_error(create_p_list(W), "must have at least one column")
})

test_that("create_p_list works with non-factor columns", {
  skip_on_cran()

  W <- data.frame(Score = c(1, 2, 1, 3, 3))
  p_list <- create_p_list(W, uniform = FALSE)
  expect_equal(sum(p_list$Score), 1)
  expect_true(all(p_list$Score >= 0))
})
