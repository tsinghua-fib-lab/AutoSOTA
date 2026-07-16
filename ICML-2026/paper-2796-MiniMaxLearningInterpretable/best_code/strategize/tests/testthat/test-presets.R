test_that("strategize_preset returns expected presets", {
  skip_on_cran()

  quick <- strategize_preset("quick_test")
  expect_equal(quick$nSGD, 20L)
  expect_false(quick$compute_se)
  expect_equal(quick$lambda, 0.1)

  standard <- strategize_preset("standard")
  expect_equal(standard$nSGD, 100L)
  expect_true(standard$compute_se)
  expect_equal(standard$lambda, 0.1)
  expect_equal(standard$conf_level, 0.95)

  publication <- strategize_preset("publication")
  expect_equal(publication$nSGD, 500L)
  expect_true(publication$compute_se)
  expect_true(is.null(publication$lambda))

  adversarial <- strategize_preset("adversarial")
  expect_true(adversarial$adversarial)
  expect_true(adversarial$diff)
  expect_equal(adversarial$lambda, 0.1)
})

test_that("strategize_preset validates preset names", {
  skip_on_cran()

  expect_error(strategize_preset("not-a-preset"))
})
