# =============================================================================
# Tests for Visualization Functions
# =============================================================================
# These tests cover strategize.plot() and plot_best_response_curves()
# Note: These tests primarily check that functions run without error,
# as visual output is difficult to test automatically.
# =============================================================================

# =============================================================================
# strategize.plot() Tests
# =============================================================================

local_trace_graphics_points <- function(mock, .local_envir = parent.frame()) {
  mock_name <- ".strategize_test_points_mock"
  assign(mock_name, mock, envir = .GlobalEnv)
  tracer <- substitute(MOCK(x, y, ...), list(MOCK = as.name(mock_name)))

  invisible(utils::capture.output(
    suppressMessages(trace("points",
                           where = asNamespace("graphics"),
                           tracer = tracer,
                           print = FALSE))
  ))
  withr::defer({
    invisible(utils::capture.output(
      suppressMessages(untrace("points", where = asNamespace("graphics")))
    ))
    if (exists(mock_name, envir = .GlobalEnv, inherits = FALSE)) {
      rm(list = mock_name, envir = .GlobalEnv)
    }
  }, envir = .local_envir)
}

test_that("strategize.plot runs without error on mock data", {
  skip_on_cran()

  # Create mock data (no JAX needed)
  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.4, Female = 0.6),
    Age = c(Young = 0.3, Middle = 0.3, Old = 0.4)
  ))
  pi_star_se_list <- list(k1 = list(
    Gender = c(Male = 0.05, Female = 0.05),
    Age = c(Young = 0.04, Middle = 0.04, Old = 0.04)
  ))
  p_list <- list(
    Gender = c(Male = 0.5, Female = 0.5),
    Age = c(Young = 0.33, Middle = 0.33, Old = 0.34)
  )

  # Should run without error (may produce warnings about graphics device)
  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list
      )
    }),
    NA
  )
})

test_that("strategize.plot handles various ticks_type options", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Factor1 = c(A = 0.6, B = 0.4)
  ))
  pi_star_se_list <- list(k1 = list(
    Factor1 = c(A = 0.05, B = 0.05)
  ))
  p_list <- list(Factor1 = c(A = 0.5, B = 0.5))

  for (ticks in c("assignmentProbs", "zero", "none")) {
    expect_error(
      suppressWarnings({
        strategize.plot(
          pi_star_list = pi_star_list,
          pi_star_se_list = pi_star_se_list,
          p_list = p_list,
          ticks_type = ticks
        )
      }),
      NA
    )
  }
})

test_that("strategize.plot handles multiple factors", {
  skip_on_cran()

  # Create mock data with 4 factors
  pi_star_list <- list(k1 = list(
    Factor1 = c(A = 0.6, B = 0.4),
    Factor2 = c(X = 0.3, Y = 0.7),
    Factor3 = c(P = 0.5, Q = 0.5),
    Factor4 = c(M = 0.4, N = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    Factor1 = c(A = 0.05, B = 0.05),
    Factor2 = c(X = 0.04, Y = 0.04),
    Factor3 = c(P = 0.03, Q = 0.03),
    Factor4 = c(M = 0.05, N = 0.05)
  ))
  p_list <- list(
    Factor1 = c(A = 0.5, B = 0.5),
    Factor2 = c(X = 0.5, Y = 0.5),
    Factor3 = c(P = 0.5, Q = 0.5),
    Factor4 = c(M = 0.5, N = 0.5)
  )

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list
      )
    }),
    NA
  )
})

test_that("strategize.plot handles multi-level factors", {
  skip_on_cran()

  # Factor with 5 levels
  pi_star_list <- list(k1 = list(
    Region = c(North = 0.15, South = 0.25, East = 0.20, West = 0.25, Central = 0.15)
  ))
  pi_star_se_list <- list(k1 = list(
    Region = c(North = 0.03, South = 0.04, East = 0.03, West = 0.04, Central = 0.03)
  ))
  p_list <- list(
    Region = c(North = 0.2, South = 0.2, East = 0.2, West = 0.2, Central = 0.2)
  )

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list
      )
    }),
    NA
  )
})

test_that("strategize.plot handles plot_ci = FALSE", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.4, Female = 0.6)
  ))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = NULL,
        p_list = p_list,
        plot_ci = FALSE
      )
    }),
    NA
  )
})

test_that("strategize.plot uses the supplied zStar for confidence intervals", {
  skip_on_cran()

  line_calls <- list()
  local_trace_graphics_points(function(x, y, ...) {
    args <- list(...)
    if (identical(args$type, "l")) {
      line_calls[[length(line_calls) + 1L]] <<- as.numeric(x)
    }
    invisible(NULL)
  })

  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.5, Female = 0.5)
  ))
  pi_star_se_list <- list(k1 = list(
    Gender = c(Male = 0.1, Female = 0.1)
  ))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  pdf(tempfile(fileext = ".pdf"))
  withr::defer(dev.off())

  strategize.plot(
    pi_star_list = pi_star_list,
    pi_star_se_list = pi_star_se_list,
    p_list = p_list,
    zStar = 2,
    xlim = c(0, 1)
  )

  expect_true(length(line_calls) > 0L)
  expect_equal(line_calls[[1]], c(0.7, 0.3), tolerance = 1e-10)
})

test_that("strategize.plot skips confidence intervals when SEs are omitted", {
  skip_on_cran()

  line_calls <- list()
  local_trace_graphics_points(function(x, y, ...) {
    args <- list(...)
    if (identical(args$type, "l")) {
      line_calls[[length(line_calls) + 1L]] <<- as.numeric(x)
    }
    invisible(NULL)
  })

  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.35, Female = 0.65)
  ))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  pdf(tempfile(fileext = ".pdf"))
  withr::defer(dev.off())

  strategize.plot(
    pi_star_list = pi_star_list,
    p_list = p_list,
    xlim = c(0, 1)
  )

  expect_length(line_calls, 0L)
})

test_that("strategize.plot validates ticks_type", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.35, Female = 0.65)
  ))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  expect_error(
    strategize.plot(
      pi_star_list = pi_star_list,
      p_list = p_list,
      ticks_type = "bad"
    ),
    "ticks_type"
  )
})

test_that("strategize.plot handles custom colors", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Gender = c(Male = 0.4, Female = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    Gender = c(Male = 0.05, Female = 0.05)
  ))
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        col_vec = c("darkblue")
      )
    }),
    NA
  )
})

test_that("strategize.plot handles multiple clusters", {
  skip_on_cran()

  # Two clusters
  pi_star_list <- list(
    k1 = list(Gender = c(Male = 0.4, Female = 0.6)),
    k2 = list(Gender = c(Male = 0.7, Female = 0.3))
  )
  pi_star_se_list <- list(
    k1 = list(Gender = c(Male = 0.05, Female = 0.05)),
    k2 = list(Gender = c(Male = 0.04, Female = 0.04))
  )
  p_list <- list(Gender = c(Male = 0.5, Female = 0.5))

  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        col_vec = c("blue", "red")
      )
    }),
    NA
  )
})

# =============================================================================
# plot_best_response_curves() Tests
# =============================================================================
# These tests require a full adversarial strategize result, so they
# are marked to skip unless JAX is available.

test_that("plot_best_response_extract_curves uses first maxima by grid direction", {
  grid_points <- c(0, 0.5, 1)
  ast_surface <- matrix(
    c(
      1, 4, 2,
      3, 4, 1,
      2, 1, 5
    ),
    nrow = 3L,
    byrow = TRUE
  )
  dag_surface <- matrix(
    c(
      1, 2, 3,
      5, 5, 4,
      1, 9, 0
    ),
    nrow = 3L,
    byrow = TRUE
  )

  curves <- strategize:::plot_best_response_extract_curves(
    grid_points = grid_points,
    ast_surface = ast_surface,
    dag_surface = dag_surface
  )

  expect_equal(curves$br_dag_given_ast, c(1, 0, 0.5))
  expect_equal(curves$br_ast_given_dag, c(0.5, 0, 1))
})

test_that("plot_best_response_curves avoids scalar objective grid loops", {
  body_text <- paste(deparse(body(plot_best_response_curves)), collapse = "\n")

  expect_false(grepl("FullGetQStar_jit", body_text, fixed = TRUE))
  expect_false(grepl("for\\s*\\(i_\\s+in\\s+seq_along\\(grid_points\\)", body_text, perl = TRUE))
  expect_false(grepl("for\\s*\\(ix\\s+in\\s+seq_along\\(xvals\\)", body_text, perl = TRUE))
})

test_that("plot_best_response_curves requires adversarial result", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_slow()

  # This test would require running a full adversarial strategize,
 # which is slow. Skip for normal testing.

  # Mock test: at minimum, function should exist and be callable
  expect_true(is.function(plot_best_response_curves))
})

# =============================================================================
# Name Transformer Tests
# =============================================================================

test_that("strategize.plot applies factor_name_transformer", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    gender = c(male = 0.4, female = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    gender = c(male = 0.05, female = 0.05)
  ))
  p_list <- list(gender = c(male = 0.5, female = 0.5))

  # Capitalize factor names
  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        factor_name_transformer = function(x) toupper(x)
      )
    }),
    NA
  )
})

test_that("strategize.plot applies level_name_transformer", {
  skip_on_cran()

  pi_star_list <- list(k1 = list(
    Gender = c(m = 0.4, f = 0.6)
  ))
  pi_star_se_list <- list(k1 = list(
    Gender = c(m = 0.05, f = 0.05)
  ))
  p_list <- list(Gender = c(m = 0.5, f = 0.5))

  # Expand level names
  label_map <- c(m = "Male", f = "Female")
  expect_error(
    suppressWarnings({
      strategize.plot(
        pi_star_list = pi_star_list,
        pi_star_se_list = pi_star_se_list,
        p_list = p_list,
        level_name_transformer = function(x) {
          ifelse(x %in% names(label_map), label_map[x], x)
        }
      )
    }),
    NA
  )
})
