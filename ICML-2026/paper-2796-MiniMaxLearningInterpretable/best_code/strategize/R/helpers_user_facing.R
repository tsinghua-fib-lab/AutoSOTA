#' User-Facing Helper Functions for strategize
#'
#' @description
#' Convenience functions to simplify common tasks when using the strategize package.
#' These functions reduce the complexity of the main API by providing sensible
#' defaults and automatic data processing.
#'
#' @name helpers
NULL


#' Create Baseline Probability List from Data
#'
#' Generates a \code{p_list} suitable for the \code{strategize()} function
#' from observed data or using uniform probabilities.
#'
#' @param W Factor matrix (data.frame or matrix) with one column per conjoint factor.
#' @param uniform Logical; if \code{TRUE}, use uniform probabilities across levels;
#'   if \code{FALSE} (default), use observed frequencies from the data.
#'
#' @return Named list where each element is a named numeric vector of probabilities.
#'   Names correspond to factor levels.
#'
#' @details
#' For conjoint experiments with balanced randomization, use \code{uniform = TRUE}.
#' For experiments with intentional imbalance or to match observed frequencies,
#' use \code{uniform = FALSE}.
#'
#' @examples
#' # Create sample factor matrix
#' W <- data.frame(
#'   Gender = c("Male", "Female", "Male", "Female"),
#'   Age = c("Young", "Old", "Young", "Old")
#' )
#'
#' # Uniform probabilities (for balanced designs)
#' p_list_uniform <- create_p_list(W, uniform = TRUE)
#' print(p_list_uniform)
#' # $Gender
#' #   Male Female
#' #    0.5    0.5
#' # $Age
#' #   Young    Old
#' #     0.5    0.5
#'
#' # Observed frequencies
#' p_list_observed <- create_p_list(W, uniform = FALSE)
#' print(p_list_observed)
#'
#' @export
create_p_list <- function(W, uniform = FALSE) {
  p_list <- cs_generate_p_list(W, uniform = uniform)

  # Print summary
  message(sprintf("Created p_list with %d factor(s):", length(p_list)))
  for (i in seq_along(p_list)) {
    levels_str <- paste(names(p_list[[i]]), collapse = ", ")
    message(sprintf("  %s: %d levels (%s)", names(p_list)[i], length(p_list[[i]]), levels_str))
  }

  p_list
}


#' Get Recommended Parameter Settings
#'
#' Returns a list of parameter values suitable for passing to \code{strategize()}.
#' This simplifies the API by providing sensible defaults for common use cases.
#'
#' @param preset One of:
#'   \describe{
#'     \item{\code{"quick_test"}}{Minimal iterations for testing code (not for inference)}
#'     \item{\code{"standard"}}{Reasonable defaults for most analyses (default)}
#'     \item{\code{"publication"}}{Higher accuracy for publication-quality results}
#'     \item{\code{"adversarial"}}{Settings tuned for adversarial/game-theoretic mode}
#'   }
#'
#' @return Named list of parameters that can be merged with \code{strategize()} arguments.
#'
#' @details
#' These presets provide starting points. You should still set data-specific

#' parameters like \code{Y}, \code{W}, \code{p_list}, and potentially \code{lambda}.
#'
#' For the "publication" preset, \code{lambda} is not included because you should
#' use \code{cv_strategize()} to select it via cross-validation.
#'
#' @examples
#' \donttest{
#' # Get standard settings
#' params <- strategize_preset("standard")
#' print(params)
#'
#' # Use with strategize (hypothetically)
#' # result <- do.call(strategize, c(list(Y = Y, W = W, p_list = p_list), params))
#' }
#'
#' @export
strategize_preset <- function(preset = c("standard", "quick_test", "publication", "adversarial")) {
  preset <- match.arg(preset)

  presets <- list(
    quick_test = list(
      nSGD = 20L,
      nMonte_adversarial = 5L,
      nMonte_Qglm = 20L,
      compute_se = FALSE,
      lambda = 0.1
    ),
    standard = list(
      nSGD = 100L,
      nMonte_adversarial = 24L,
      nMonte_Qglm = 100L,
      compute_se = TRUE,
      lambda = 0.1,
      conf_level = 0.95
    ),
    publication = list(
      nSGD = 500L,
      nMonte_adversarial = 100L,
      nMonte_Qglm = 500L,
      compute_se = TRUE,
      conf_level = 0.95
      # lambda intentionally omitted - use cv_strategize
    ),
    adversarial = list(
      nSGD = 200L,
      nMonte_adversarial = 50L,
      nMonte_Qglm = 200L,
      adversarial = TRUE,
      diff = TRUE,
      compute_se = TRUE,
      lambda = 0.1
    )
  )

  result <- presets[[preset]]

  message(sprintf("Using '%s' preset:", preset))
  message(sprintf("  nSGD = %d, compute_se = %s",
                  result$nSGD, result$compute_se))
  if (!is.null(result$lambda)) {
    message(sprintf("  lambda = %.2f (override with cv_strategize for optimal selection)",
                    result$lambda))
  } else {
    message("  lambda not set (use cv_strategize to select)")
  }

  result
}


#' Print Method for strategize Results
#'
#' @param x A strategize result object
#' @param digits Number of digits to display
#' @param ... Additional arguments (ignored)
#'
#' @return Invisibly returns the input object
#'
#' @export
#' @method print strategize_result
print.strategize_result <- function(x, digits = 3, ...) {
  cat("strategize Result\n")
  cat(strrep("=", 50), "\n\n")

  # Optimal distribution
  cat("Optimal Distribution (pi*):\n")
  pi_point <- x$pi_star_point

  if (!is.null(pi_point)) {
    # Handle different structures
    cluster_names <- names(pi_point)
    if (is.null(cluster_names)) cluster_names <- paste0("k", seq_along(pi_point))

    for (k in seq_along(pi_point)) {
      if (length(pi_point) > 1) {
        cat(sprintf("\n  Cluster %s:\n", cluster_names[k]))
      }
      factors <- pi_point[[k]]
      if (is.list(factors)) {
        for (factor_name in names(factors)) {
          cat(sprintf("  %s:\n", factor_name))
          probs <- factors[[factor_name]]
          baseline <- if (!is.null(x$p_list[[factor_name]])) x$p_list[[factor_name]] else NULL

          for (level in names(probs)) {
            if (!is.null(baseline) && level %in% names(baseline)) {
              change <- probs[level] - baseline[level]
              arrow <- if (change > 0.01) "+" else if (change < -0.01) "-" else "="
              cat(sprintf("    %s: %.3f (baseline: %.3f) %s\n",
                          level, probs[level], baseline[level], arrow))
            } else {
              cat(sprintf("    %s: %.3f\n", level, probs[level]))
            }
          }
        }
      }
    }
  }

  # Expected outcome
  q_val <- x$Q_point
  if (is.null(q_val)) q_val <- x$Q_point_mEst
  if (!is.null(q_val) && length(q_val) > 0) {
    cat(sprintf("\nExpected Outcome (Q*): %.4f\n", q_val[1]))
    q_se <- x$Q_se
    if (is.null(q_se)) q_se <- x$Q_se_mEst
    if (!is.null(q_se) && !all(is.na(q_se))) {
      cat(sprintf("  (SE: %.4f)\n", q_se[1]))
    }
  }

  # Settings
  if (!is.null(x$lambda)) {
    cat(sprintf("\nSettings: lambda = %.4f", x$lambda))
    if (!is.null(x$penalty_type)) {
      cat(sprintf(", penalty = %s", x$penalty_type))
    }
    cat("\n")
  }

  invisible(x)
}


#' Summary Method for strategize Results
#'
#' Creates a summary table comparing baseline and optimal distributions.
#'
#' @param object A strategize result object
#' @param ... Additional arguments (ignored)
#'
#' @return Invisibly returns a data.frame with the comparison
#'
#' @export
#' @method summary strategize_result
summary.strategize_result <- function(object, ...) {
  # Build comparison table
  comparison <- data.frame(
    Factor = character(),
    Level = character(),
    Baseline = numeric(),
    Optimal = numeric(),
    Change = numeric(),
    SE = numeric(),
    stringsAsFactors = FALSE
  )

  p_list <- object$p_list
  pi_point <- object$pi_star_point
  pi_se <- object$pi_star_se

  if (is.null(p_list) || is.null(pi_point)) {
    cat("Insufficient data for summary.\n")
    return(invisible(NULL))
  }

  # Get first cluster's results
  if (!is.null(pi_point$k1)) {
    optimal <- pi_point$k1
    se_vals <- if (!is.null(pi_se$k1)) pi_se$k1 else NULL
  } else {
    optimal <- pi_point[[1]]
    se_vals <- if (length(pi_se) > 0) pi_se[[1]] else NULL
  }

  for (factor_name in names(p_list)) {
    baseline <- p_list[[factor_name]]
    opt_probs <- optimal[[factor_name]]
    se_vec <- if (!is.null(se_vals[[factor_name]])) se_vals[[factor_name]] else rep(NA, length(baseline))

    for (j in seq_along(baseline)) {
      level <- names(baseline)[j]
      opt_val <- if (level %in% names(opt_probs)) opt_probs[level] else NA
      se_val <- if (length(se_vec) >= j) se_vec[j] else NA

      comparison <- rbind(comparison, data.frame(
        Factor = factor_name,
        Level = level,
        Baseline = round(baseline[level], 3),
        Optimal = round(opt_val, 3),
        Change = round(opt_val - baseline[level], 3),
        SE = round(se_val, 4),
        stringsAsFactors = FALSE
      ))
    }
  }

  cat("Summary: Distribution Changes from Baseline to Optimal\n")
  cat(strrep("-", 60), "\n")
  print(comparison, row.names = FALSE)
  cat(strrep("-", 60), "\n")

  q_val <- object$Q_point
  if (is.null(q_val)) q_val <- object$Q_point_mEst
  if (!is.null(q_val) && length(q_val) > 0) {
    cat(sprintf("Q* = %.4f", q_val[1]))
    q_se <- object$Q_se
    if (is.null(q_se)) q_se <- object$Q_se_mEst
    if (!is.null(q_se) && !all(is.na(q_se))) {
      cat(sprintf(" (SE = %.4f)", q_se[1]))
    }
    cat("\n")
  }

  invisible(comparison)
}
