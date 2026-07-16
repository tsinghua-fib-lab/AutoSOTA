#' Summarize Adversarial Strategize Results
#'
#' Prints a comprehensive summary of adversarial equilibrium results,
#' including strategies, equilibrium quality metrics, and four-quadrant breakdown.
#'
#' @param result Output from \code{\link{strategize}} with \code{adversarial = TRUE}
#' @param validate Logical. Whether to run equilibrium validation. Default is \code{TRUE}.
#'   Set to \code{FALSE} for faster output without validation.
#' @param check_hessian Logical. Whether to include Hessian geometry analysis.
#'   Default is \code{TRUE}. Only runs if Hessian functions are available in the result.
#' @param verbose Logical. Whether to print the summary. Default is \code{TRUE}.
#'
#' @return Invisibly returns a list containing all summary statistics.
#'
#' @details
#' This function provides a unified view of adversarial equilibrium results:
#' \itemize{
#'   \item Equilibrium vote share Q* with standard error
#'   \item Optimized strategies for both parties (AST and DAG)
#'   \item Equilibrium quality metrics (best-response errors)
#'   \item Four-quadrant contribution breakdown
#'   \item Voter proportion information
#'   \item Convergence status
#' }
#'
#' @examples
#' \dontrun{
#' # Run adversarial strategize
#' result <- strategize(Y = y, W = w, adversarial = TRUE, nSGD = 500)
#'
#' # Print summary
#' summarize_adversarial(result)
#'
#' # Quick summary without validation
#' summarize_adversarial(result, validate = FALSE)
#' }
#'
#' @export
summarize_adversarial <- function(result, validate = TRUE, check_hessian = TRUE, verbose = TRUE) {

  # Validate input
  if (!isTRUE(result$convergence_history$adversarial)) {
    stop("summarize_adversarial() requires an adversarial strategize result. ",
         "Set adversarial = TRUE in strategize().")
  }

  summary_data <- list()

  # ---- Basic info ----
  summary_data$Q_star <- as.numeric(result$Q_point)
  summary_data$Q_se <- if (!is.null(result$Q_se)) as.numeric(result$Q_se) else NA
  summary_data$lambda <- result$lambda
  summary_data$penalty_type <- result$penalty_type

  # ---- Party strategies ----
  pi_ast <- result$pi_star_point$k1
  pi_dag <- result$pi_star_point$k2

  summary_data$strategies <- list(
    AST = pi_ast,
    DAG = pi_dag
  )

  # ---- Voter proportions ----
  summary_data$voter_props <- list(
    AST = as.numeric(result$AstProp),
    DAG = as.numeric(result$DagProp)
  )

  # ---- Convergence info ----
  ch <- result$convergence_history
  final_grad_ast <- ch$grad_ast[ch$nSGD]
  final_grad_dag <- ch$grad_dag[ch$nSGD]

  summary_data$convergence <- list(
    nSGD = ch$nSGD,
    final_grad_ast = if (is.finite(final_grad_ast)) final_grad_ast else NA,
    final_grad_dag = if (is.finite(final_grad_dag)) final_grad_dag else NA
  )

  # ---- Equilibrium validation ----
  if (validate) {
    validation <- tryCatch({
      validate_equilibrium(result, method = "grid", resolution = 30,
                           plot = FALSE, verbose = FALSE)
    }, error = function(e) {
      list(
        br_error_ast = NA,
        br_error_dag = NA,
        is_equilibrium = NA
      )
    })
    summary_data$validation <- validation
  } else {
    summary_data$validation <- NULL
  }

  # ---- Quadrant breakdown ----
  quadrant <- tryCatch({
    breakdown <- compute_quadrant_breakdown(result, nMonte = 100, verbose = FALSE)
    breakdown
  }, error = function(e) {
    list(weights = c(E1 = 0.25, E2 = 0.25, E3 = 0.25, E4 = 0.25))
  })
  summary_data$quadrant <- quadrant

  # ---- Hessian geometry analysis ----
  if (check_hessian && isTRUE(result$hessian_available)) {
    hessian_result <- tryCatch({
      check_hessian_geometry(result, verbose = FALSE)
    }, error = function(e) {
      NULL
    })
    summary_data$hessian_analysis <- hessian_result
    summary_data$geometry_valid <- if (!is.null(hessian_result)) hessian_result$valid_saddle else NA
  } else if (check_hessian && !isTRUE(result$hessian_available)) {
    summary_data$hessian_analysis <- NULL
    summary_data$geometry_valid <- NA
    summary_data$hessian_skipped_reason <- result$hessian_skipped_reason
  } else {
    summary_data$hessian_analysis <- NULL
    summary_data$geometry_valid <- NA
  }

  # ---- Print summary ----
  if (verbose) {
    print_adversarial_summary(summary_data, pi_ast, pi_dag)
  }

  invisible(summary_data)
}


#' Internal: Print formatted adversarial summary
#' @keywords internal
#' @noRd
print_adversarial_summary <- function(summary_data, pi_ast, pi_dag) {

  cat("\n")
  cat("==============================================\n")
  cat("       Adversarial Equilibrium Summary        \n")
  cat("==============================================\n\n")

  # Equilibrium Q*
  Q_str <- sprintf("%.4f", summary_data$Q_star)
  if (!is.na(summary_data$Q_se) && summary_data$Q_se > 0) {
    Q_str <- sprintf("%s (SE: %.4f)", Q_str, summary_data$Q_se)
  }
  cat(sprintf("Equilibrium Q*: %s\n", Q_str))
  cat(sprintf("Lambda: %.4f (%s penalty)\n\n",
              summary_data$lambda, summary_data$penalty_type))

  # Party strategies
  cat("Party Strategies:\n")
  cat("-----------------\n")

  format_strategy <- function(pi_list, party_name) {
    for (factor_name in names(pi_list)) {
      probs <- pi_list[[factor_name]]
      level_names <- names(probs)
      if (is.null(level_names)) level_names <- seq_along(probs)

      prob_str <- paste(
        sapply(seq_along(probs), function(i) {
          sprintf("%s: %.2f", level_names[i], probs[i])
        }),
        collapse = ", "
      )

      cat(sprintf("  %s %s[%s]\n", party_name, factor_name, prob_str))
    }
  }

  format_strategy(pi_ast, "AST:")
  format_strategy(pi_dag, "DAG:")
  cat("\n")

  # Equilibrium quality
  cat("Equilibrium Quality:\n")
  cat("-------------------\n")

  if (!is.null(summary_data$validation)) {
    val <- summary_data$validation

    br_ast_str <- if (!is.na(val$br_error_ast)) {
      sprintf("%.4f %s", val$br_error_ast,
              ifelse(val$br_error_ast < 0.01, "[PASS]", "[FAIL]"))
    } else {
      "N/A"
    }

    br_dag_str <- if (!is.na(val$br_error_dag)) {
      sprintf("%.4f %s", val$br_error_dag,
              ifelse(val$br_error_dag < 0.01, "[PASS]", "[FAIL]"))
    } else {
      "N/A"
    }

    cat(sprintf("  BR Error (AST): %s\n", br_ast_str))
    cat(sprintf("  BR Error (DAG): %s\n", br_dag_str))
  } else {
    cat("  (validation skipped)\n")
  }

  conv <- summary_data$convergence
  grad_str <- if (!is.na(conv$final_grad_ast) && !is.na(conv$final_grad_dag)) {
    sprintf("%.2e (AST), %.2e (DAG)", conv$final_grad_ast, conv$final_grad_dag)
  } else {
    "N/A"
  }
  cat(sprintf("  Final gradient norm: %s\n", grad_str))
  cat(sprintf("  SGD iterations: %d\n\n", conv$nSGD))

  # Four-quadrant breakdown
  cat("Four-Quadrant Breakdown:\n")
  cat("-----------------------\n")

  if (!is.null(summary_data$quadrant) && !is.null(summary_data$quadrant$weights)) {
    weights <- summary_data$quadrant$weights
    cat(sprintf("  Both entrants:    %.1f%%\n", weights["E1"] * 100))
    cat(sprintf("  AST ent., DAG field: %.1f%%\n", weights["E2"] * 100))
    cat(sprintf("  AST field, DAG ent.: %.1f%%\n", weights["E3"] * 100))
    cat(sprintf("  Both field:       %.1f%%\n", weights["E4"] * 100))
  } else {
    cat("  (quadrant breakdown unavailable)\n")
  }
  cat("\n")

  # Voter proportions
  cat("Voter Proportions:\n")
  cat("-----------------\n")
  cat(sprintf("  AST voters: %.1f%%\n", summary_data$voter_props$AST * 100))
  cat(sprintf("  DAG voters: %.1f%%\n", summary_data$voter_props$DAG * 100))
  cat("\n")

  # Hessian geometry analysis
  if (!is.null(summary_data$hessian_analysis)) {
    hess <- summary_data$hessian_analysis
    cat("Hessian Geometry:\n")
    cat("-----------------\n")
    cat(sprintf("  Status: %s\n", hess$status))
    cat(sprintf("  Valid saddle point: %s\n", ifelse(hess$valid_saddle, "YES", "NO")))
    if (!is.na(hess$condition_number_ast)) {
      cat(sprintf("  Condition numbers: AST=%.1f, DAG=%.1f\n",
                  hess$condition_number_ast, hess$condition_number_dag))
    }
    cat("\n")
  } else if (!is.null(summary_data$hessian_skipped_reason)) {
    cat("Hessian Geometry:\n")
    cat("-----------------\n")
    cat(sprintf("  (skipped: %s)\n\n", summary_data$hessian_skipped_reason))
  }

  cat("==============================================\n")
}


#' Check if Result is Adversarial
#'
#' Utility function to check if a strategize result is from adversarial mode.
#'
#' @param result Output from \code{\link{strategize}}
#' @return Logical. TRUE if the result is from adversarial mode.
#'
#' @export
is_adversarial <- function(result) {
  isTRUE(result$convergence_history$adversarial)
}


#' Get Final Gradient Norms
#'
#' Extract the final gradient magnitudes from a strategize result.
#'
#' @param result Output from \code{\link{strategize}}
#' @return Named numeric vector with gradient norms for AST and DAG players.
#'
#' @export
get_final_gradients <- function(result) {

  if (is.null(result$convergence_history)) {
    stop("No convergence history found. Use a recent version of strategize().")
  }

  ch <- result$convergence_history
  nSGD <- ch$nSGD

  c(
    AST = if (length(ch$grad_ast) >= nSGD) ch$grad_ast[nSGD] else NA,
    DAG = if (length(ch$grad_dag) >= nSGD) ch$grad_dag[nSGD] else NA
  )
}
