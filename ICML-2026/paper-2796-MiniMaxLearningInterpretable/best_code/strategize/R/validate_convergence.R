#' Plot Convergence Diagnostics for Strategize Results
#'
#' Visualizes the optimization trajectory from gradient descent, showing
#' gradient magnitudes, loss values, and learning rate adaptation over iterations.
#'
#' @importFrom graphics grid
#'
#' @param result Output from \code{\link{strategize}} with \code{adversarial = TRUE}
#' @param metrics Character vector specifying which metrics to plot. Options are:
#'   \itemize{
#'     \item \code{"gradient"}: Gradient magnitude (L2 norm) over iterations
#'     \item \code{"loss"}: Objective function value over iterations
#'     \item \code{"lr"}: Learning rate adaptation over iterations
#'   }
#'   Default is \code{c("gradient", "loss")}.
#' @param log_scale Logical. Whether to use log scale for gradient magnitudes.
#'   Default is \code{TRUE}.
#' @param use_ggplot Logical. If \code{TRUE} and ggplot2 is available, uses ggplot2
#'   for visualization. Otherwise uses base R graphics. Default is \code{FALSE}.
#'
#' @return If \code{use_ggplot = TRUE} and ggplot2 is available, returns a ggplot
#'   object. Otherwise, plots are created as side effects and the function returns
#'   \code{invisible(NULL)}.
#'
#' @details
#' This function provides diagnostic plots to assess whether the gradient descent

#' optimization has converged to a Nash equilibrium. Key indicators of convergence:
#' \itemize{
#'   \item Gradient magnitudes should decrease toward zero for both players
#'   \item Loss values should stabilize (may oscillate slightly in adversarial settings)
#'   \item Learning rates should adapt appropriately (decrease as gradients shrink)
#' }
#'
#' In adversarial (two-player) mode, both AST and DAG players' metrics are shown.
#' In non-adversarial mode, only AST metrics are displayed.
#'
#' @examples
#' \dontrun{
#' # Run adversarial strategize
#' result <- strategize(Y = y, W = w, adversarial = TRUE, nSGD = 500)
#'
#' # Plot convergence diagnostics
#' plot_convergence(result)
#'
#' # Plot only gradient magnitudes with log scale
#' plot_convergence(result, metrics = "gradient", log_scale = TRUE)
#' }
#'
#' @export
plot_convergence <- function(result,
                             metrics = c("gradient", "loss"),
                             log_scale = TRUE,
                             use_ggplot = FALSE) {

  # Input validation
  if (is.null(result$convergence_history)) {
    stop("No convergence history found in result. ",
         "Make sure you are using the latest version of strategize().")
  }

  ch <- result$convergence_history
  adversarial <- isTRUE(ch$adversarial)
  nSGD <- ch$nSGD

  # Validate metrics argument
  valid_metrics <- c("gradient", "loss", "lr")
  metrics <- match.arg(metrics, valid_metrics, several.ok = TRUE)

  # Extract data
  iterations <- seq_len(nSGD)

  # Convert any remaining JAX arrays to numeric
  safe_numeric <- function(x) {
    if (is.null(x)) return(rep(NA_real_, nSGD))
    x <- as.numeric(x)
    x[!is.finite(x)] <- NA
    x
  }

  data_list <- list(
    grad_ast = safe_numeric(ch$grad_ast),
    grad_dag = safe_numeric(ch$grad_dag),
    loss_ast = safe_numeric(ch$loss_ast),
    loss_dag = safe_numeric(ch$loss_dag),
    lr_ast = if (!is.null(ch$inv_lr_ast)) 1 / safe_numeric(ch$inv_lr_ast) else NULL,
    lr_dag = if (!is.null(ch$inv_lr_dag)) 1 / safe_numeric(ch$inv_lr_dag) else NULL
  )

  # Use ggplot2 if requested and available
  if (use_ggplot && requireNamespace("ggplot2", quietly = TRUE)) {
    return(plot_convergence_ggplot(data_list, metrics, log_scale, adversarial, nSGD))
  }

  # Base R plotting
  n_panels <- length(metrics)
  if (adversarial) {
    # Two rows: AST and DAG
    old_par <- par(mfrow = c(2, n_panels), mar = c(4, 4, 3, 1))
  } else {
    old_par <- par(mfrow = c(1, n_panels), mar = c(4, 4, 3, 1))
  }
  on.exit(par(old_par))

  # Color scheme
  col_ast <- "#E41A1C"  # Red for AST/Republican

  col_dag <- "#377EB8"  # Blue for DAG/Democrat

  plot_metric_panels <- function(player_key, player_label, col_use) {
    for (metric in metrics) {
      if (metric == "gradient") {
        y_data <- data_list[[paste0("grad_", player_key)]]
        main_title <- sprintf("Gradient Magnitude (%s)", player_label)
        ylab <- expression("|" * nabla * "Q|")
        if (log_scale && any(y_data > 0, na.rm = TRUE)) {
          y_data <- log10(pmax(y_data, 1e-10))
          ylab <- expression("log"[10] * "(|" * nabla * "Q|)")
        }
      } else if (metric == "loss") {
        y_data <- data_list[[paste0("loss_", player_key)]]
        main_title <- sprintf("Loss/Objective (%s)", player_label)
        ylab <- "Q (objective)"
      } else if (metric == "lr") {
        y_data <- data_list[[paste0("lr_", player_key)]]
        if (is.null(y_data) || all(is.na(y_data))) {
          plot.new()
          text(0.5, 0.5, "Learning rate data\nnot available\n(optax used?)")
          next
        }
        main_title <- sprintf("Learning Rate (%s)", player_label)
        ylab <- "Learning Rate"
      }

      plot(iterations, y_data, type = "l", col = col_use, lwd = 1.5,
           main = main_title, xlab = "Iteration", ylab = ylab,
           xlim = c(1, nSGD))
      grid(col = "gray90")

      # Add reference line at 0 for gradients
      if (metric == "gradient") {
        abline(h = ifelse(log_scale, -10, 0), lty = 2, col = "gray50")
      }
    }
  }

  plot_metric_panels("ast", "AST", col_ast)

  if (adversarial) {
    plot_metric_panels("dag", "DAG", col_dag)
  }

  invisible(NULL)
}


#' Internal: ggplot2 version of convergence plot
#' @keywords internal
#' @noRd
plot_convergence_ggplot <- function(data_list, metrics, log_scale, adversarial, nSGD) {

  # Avoid R CMD check notes for ggplot2 NSE variables
  iteration <- value <- player <- NULL

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 is required for use_ggplot = TRUE")
  }

  iterations <- seq_len(nSGD)

  # Build data frame for plotting
  plot_data <- data.frame(iteration = iterations)

  # Add metrics columns
  if ("gradient" %in% metrics) {
    plot_data$grad_ast <- data_list$grad_ast
    if (adversarial) plot_data$grad_dag <- data_list$grad_dag
  }
  if ("loss" %in% metrics) {
    plot_data$loss_ast <- data_list$loss_ast
    if (adversarial) plot_data$loss_dag <- data_list$loss_dag
  }
  if ("lr" %in% metrics) {
    plot_data$lr_ast <- data_list$lr_ast
    if (adversarial) plot_data$lr_dag <- data_list$lr_dag
  }

  # Reshape to long format for ggplot
  long_data <- NULL

  if ("gradient" %in% metrics) {
    grad_df <- data.frame(
      iteration = iterations,
      value = c(data_list$grad_ast, if (adversarial) data_list$grad_dag else NULL),
      player = c(rep("AST", nSGD), if (adversarial) rep("DAG", nSGD) else NULL),
      metric = "Gradient Magnitude"
    )
    long_data <- rbind(long_data, grad_df)
  }

  if ("loss" %in% metrics) {
    loss_df <- data.frame(
      iteration = iterations,
      value = c(data_list$loss_ast, if (adversarial) data_list$loss_dag else NULL),
      player = c(rep("AST", nSGD), if (adversarial) rep("DAG", nSGD) else NULL),
      metric = "Objective Value"
    )
    long_data <- rbind(long_data, loss_df)
  }

  if ("lr" %in% metrics && !is.null(data_list$lr_ast)) {
    lr_df <- data.frame(
      iteration = iterations,
      value = c(data_list$lr_ast, if (adversarial) data_list$lr_dag else NULL),
      player = c(rep("AST", nSGD), if (adversarial) rep("DAG", nSGD) else NULL),
      metric = "Learning Rate"
    )
    long_data <- rbind(long_data, lr_df)
  }

  # Create plot
  p <- ggplot2::ggplot(long_data, ggplot2::aes(x = iteration, y = value, color = player)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::facet_wrap(~metric, scales = "free_y", ncol = length(metrics)) +
    ggplot2::scale_color_manual(values = c("AST" = "#E41A1C", "DAG" = "#377EB8")) +
    ggplot2::labs(
      title = "Convergence Diagnostics",
      x = "Iteration",
      y = "Value",
      color = "Player"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text = ggplot2::element_text(face = "bold")
    )

  # Apply log scale for gradients if requested
  if (log_scale && "gradient" %in% metrics) {
    # Note: facet_wrap with free scales makes this tricky;
    # for simplicity, we apply log scale to gradient panel only via coord transformation
    # In practice, users can customize the returned ggplot object
    message("Note: For log-scaled gradients with ggplot2, consider adding ",
            "+ scale_y_log10() to the gradient facet manually.")
  }

  return(p)
}
