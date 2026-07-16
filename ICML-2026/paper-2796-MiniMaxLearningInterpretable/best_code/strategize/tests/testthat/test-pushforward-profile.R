# testthat unit test for profile-specific pushforward vs scalar mixture

bar_pi <- function(pi, pi_field, kappa) {
  row_expect <- as.vector(kappa %*% pi_field)
  col_expect <- as.vector(t(pi) %*% kappa)
  pi * row_expect + pi_field * (1 - col_expect)
}

expect_matrix <- function(p, q, M) {
  sum(outer(p, q) * M)
}

q_true_barpi <- function(piA, piA_field, piB, piB_field, kA, kB, C) {
  barA <- bar_pi(piA, piA_field, kA)
  barB <- bar_pi(piB, piB_field, kB)
  sum(outer(barA, barB) * C)
}

q_code_scalar <- function(piA, piA_field, piB, piB_field, kA, kB, C) {
  P_A_entrant <- expect_matrix(piA, piA_field, kA)
  P_B_entrant <- expect_matrix(piB, piB_field, kB)
  w1 <- P_A_entrant * P_B_entrant
  w2 <- P_A_entrant * (1 - P_B_entrant)
  w3 <- (1 - P_A_entrant) * P_B_entrant
  w4 <- (1 - P_A_entrant) * (1 - P_B_entrant)

  C_tu <- expect_matrix(piA, piB, C)
  C_tu_field <- expect_matrix(piA, piB_field, C)
  C_field_u <- expect_matrix(piA_field, piB, C)
  C_field_field <- expect_matrix(piA_field, piB_field, C)

  w1 * C_tu + w2 * C_tu_field + w3 * C_field_u + w4 * C_field_field
}

if (requireNamespace("testthat", quietly = TRUE)) {
  testthat::test_that("pushforward equals scalar mixture when kappa is constant", {
    piA <- c(0.2, 0.8)
    piA_field <- c(0.8, 0.2)
    piB <- c(0.7, 0.3)
    piB_field <- c(0.4, 0.6)
    C <- matrix(c(0.7, 0.4, 0.6, 0.2), nrow = 2, byrow = TRUE)
    kA <- matrix(0.5, nrow = 2, ncol = 2)
    kB <- matrix(0.5, nrow = 2, ncol = 2)

    q_true <- q_true_barpi(piA, piA_field, piB, piB_field, kA, kB, C)
    q_scalar <- q_code_scalar(piA, piA_field, piB, piB_field, kA, kB, C)

    testthat::expect_lt(abs(q_true - q_scalar), 1e-10)
  })

  testthat::test_that("pushforward differs from scalar mixture when kappa varies", {
    piA <- c(0.2, 0.8)
    piA_field <- c(0.8, 0.2)
    piB <- c(0.7, 0.3)
    piB_field <- c(0.4, 0.6)
    C <- matrix(c(0.7, 0.4, 0.6, 0.2), nrow = 2, byrow = TRUE)
    kA <- matrix(c(0.5, 0.9, 0.1, 0.5), nrow = 2, byrow = TRUE)
    kB <- matrix(c(0.5, 0.8, 0.2, 0.5), nrow = 2, byrow = TRUE)

    q_true <- q_true_barpi(piA, piA_field, piB, piB_field, kA, kB, C)
    q_scalar <- q_code_scalar(piA, piA_field, piB, piB_field, kA, kB, C)

    testthat::expect_gt(abs(q_true - q_scalar), 1e-4)
  })
}
