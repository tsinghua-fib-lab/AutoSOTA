#' Plot Estimated Probabilities for Hypothetical Scenarios
#'
#' This function creates a grid of base R plots to visualize and compare
#' probabilities (and optionally their confidence intervals) across multiple
#' hypothetical or assignment scenarios. By default, it arranges the plots
#' in a 3xN grid, labeling each row according to the factor or condition
#' being displayed.
#'
#' @param pi_star_list A list of numeric vectors, each corresponding to a set of
#'   hypothetical probabilities to be plotted. These are typically model-based or
#'   derived values.
#' @param pi_star_se_list A list of numeric vectors of the same structure as
#'   \code{pi_star_list}, containing standard errors for each probability.
#'   Used to plot confidence intervals.
#' @param p_list A list of numeric vectors of "assignment" or
#'   baseline probabilities to be overlaid as vertical ticks on each plot
#'   (depending on \code{ticks_type}).
#' @param col.main Character. Color for the main title in each subplot.
#'   Default is \code{"black"}.
#' @param cex.main Numeric. Character expansion factor for main titles.
#'   Default is \code{1.5}.
#' @param zStar Numeric. Multiplier for the standard error bars (e.g., 1.96
#'   for approximately 95 percent confidence intervals). Default is \code{1}.
#' @param xlim Numeric vector of length 2. The x-axis limits for all subplots.
#'   Defaults to \code{c(0, 1)} if not specified.
#' @param ticks_type Character. Controls the type of reference ticks added. 
#'     \code{"assignmentProbs"}: Vertical ticks drawn at the positions from \code{p_list} (default). 
#'     \code{"zero"}: Vertical ticks drawn at 0. 
#'     \code{"none"}: No vertical reference ticks.
#' @param col_vec Optional character vector of colors (one per set of probabilities
#'   in \code{pi_star_list}). If \code{NULL}, uses sequential indexing
#'   for color.
#' @param plot_names Logical. If \code{TRUE} (default), factor/condition labels
#'   will be placed along the y-axis.
#' @param plot_ci Logical. If \code{TRUE} (default), error bars will be drawn
#'   using \code{pi_star_se_list}.
#' @param widths_vec,heights_vec Currently unused. Reserved for future layout
#'   expansions.
#' @param main_title Character. An overall title for the plot. Default is
#'   an empty string.
#' @param margins_vec Currently unused. Reserved for future layout expansions.
#' @param add Logical. If \code{FALSE} (default), a new plot is created. If
#'   \code{TRUE}, points/error bars are added to an existing plot space.
#' @param pch Numeric or character. The plotting symbol. Default is \code{20}.
#' @param factor_name_transformer Function to transform factor names for display.
#'   Should accept and return a character vector. Default is identity function.
#' @param level_name_transformer Function to transform level names for display.
#'   Should accept and return a character vector. Default is identity function.
#' @param label_wrap_width Numeric. If provided, long level labels are wrapped
#'   to this character width using line breaks. Default is \code{30}. Use
#'   \code{NULL} to disable wrapping.
#' @param label_line_height Numeric. Additional vertical spacing (in plot units)
#'   added per wrapped line beyond the first. Default is \code{0.6}.
#' @param cex.axis Numeric. Character expansion factor for axis labels.
#'   Default is \code{1.25}.
#' @param open_browser Logical. If \code{TRUE}, opens a browser for debugging.
#'   Default is \code{FALSE}. Intended for development use only.
#'
#' @details
#' \code{strategize.plot} arranges multiple subplots (3 columns by default) in a grid
#' that depends on the number of elements in \code{p_list}. Each subplot
#' will show a factor level or condition on the y-axis, with probabilities along
#' the x-axis. If confidence intervals are provided (\code{pi_star_se_list}), horizontal
#' error bars around each probability point will be displayed. Additionally, vertical
#' reference ticks can be added, showing values from \code{p_list} or zero
#' depending on \code{ticks_type}.
#'
#' @return Invisibly returns \code{NULL}. This function is primarily called for its
#' side effect: producing a multi-panel base R plot.
#'
#' @examples
#' # =============================================
#' # Visualize optimal vs baseline distributions
#' # =============================================
#' # This function works without JAX - just needs the result structure
#'
#' # Create mock strategize result for plotting
#' # (In practice, use output from strategize())
#' pi_star_list <- list(k1 = list(
#'   Gender = c(Male = 0.35, Female = 0.65),
#'   Age = c(Young = 0.45, Middle = 0.30, Old = 0.25),
#'   Party = c(Dem = 0.40, Rep = 0.60)
#' ))
#'
#' pi_star_se_list <- list(k1 = list(
#'   Gender = c(Male = 0.04, Female = 0.04),
#'   Age = c(Young = 0.03, Middle = 0.03, Old = 0.03),
#'   Party = c(Dem = 0.05, Rep = 0.05)
#' ))
#'
#' # Baseline (original assignment) probabilities
#' p_list <- list(
#'   Gender = c(Male = 0.5, Female = 0.5),
#'   Age = c(Young = 0.33, Middle = 0.33, Old = 0.34),
#'   Party = c(Dem = 0.5, Rep = 0.5)
#' )
#'
#' # Plot comparing optimal to baseline
#' strategize.plot(
#'   pi_star_list = pi_star_list,
#'   pi_star_se_list = pi_star_se_list,
#'   p_list = p_list,
#'   main_title = "Optimal vs Baseline Distribution",
#'   ticks_type = "assignmentProbs"  # Show baseline as reference ticks
#' )
#'
#' @importFrom graphics plot points par axis abline
#' @export
#' @md
strategize.plot <- function(pi_star_list=NULL, 
                            pi_star_se_list = NULL, 
                            p_list=NULL, 
                            col.main = "black", 
                            cex.main = 1.5, 
                            zStar = 1, 
                            xlim = NULL, 
                            ticks_type = "assignmentProbs",
                            col_vec = NULL, 
                            plot_names = TRUE, 
                            plot_ci = TRUE, 
                            widths_vec, heights_vec,
                            main_title = "", 
                            margins_vec = NULL, 
                            add = FALSE, 
                            pch = 20,
                            factor_name_transformer = function(x){x},
                            level_name_transformer = function(x){x},
                            label_wrap_width = 30,
                            label_line_height = 0.6,
                            cex.axis = 1.25,
                            open_browser = FALSE
                            ) {
  if(open_browser){browser()}
  valid_ticks_type <- c("assignmentProbs", "zero", "none")
  if (!is.character(ticks_type) || length(ticks_type) != 1L ||
      is.na(ticks_type) || !ticks_type %in% valid_ticks_type) {
    stop(
      "'ticks_type' must be one of: ",
      paste(valid_ticks_type, collapse = ", "),
      call. = FALSE
    )
  }
  if (!is.numeric(zStar) || length(zStar) != 1L || !is.finite(zStar)) {
    stop("'zStar' must be a single finite numeric value.", call. = FALSE)
  }
  wrap_labels <- function(labels, width) {
    if (is.null(width) || is.na(width) || !is.finite(width) || width <= 0) {
      return(labels)
    }
    vapply(labels, function(lbl) {
      if (is.na(lbl)) return(NA_character_)
      paste(strwrap(lbl, width = width), collapse = "\n")
    }, character(1))
  }
  p_list_ <- lapply(1:length(p_list), function(ze) {
    names(p_list[[ze]]) <- gsub(names(p_list[[ze]]),
                                            pattern = names(p_list)[ze],
                                            replacement = "")
    names(p_list[[ze]]) <- gsub(names(p_list[[ze]]),
                                            pattern = "\\.",
                                            replacement = "")
    p_list[[ze]]
  })
  
  ncols <- 3
  nrows <- ceiling(length(p_list_)/3)

  # Save original par settings and restore on exit
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)

  par(mfrow = c(nrows, ncols))

  for(d_ in 1:length(p_list_)) {
    pd_ <- p_list_[[d_]]
    ordering_d <- order(unlist(lapply(strsplit(names(pd_),
                                               split = ""),
                                      function(zer) { zer[1] })),
                        decreasing = TRUE)

    # Apply factor name transformation
    prettyFactorNames <- factor_name_transformer(names(p_list)[d_])

    # Apply level name transformation (and optional wrapping)
    raw_level_names <- level_name_transformer(names(pd_)[ordering_d])
    wrapped_level_names <- wrap_labels(raw_level_names, label_wrap_width)
    label_line_counts <- vapply(wrapped_level_names, function(lbl) {
      if (is.na(lbl)) return(1L)
      length(strsplit(lbl, "\n", fixed = TRUE)[[1]])
    }, integer(1))

    # Preliminary calculations
    yScale <- 0.2
    k_EST <- length(pi_star_list)
    p_d <- p_list[[d_]][ordering_d]
    ypos_grid <- expand.grid(k = 1:k_EST, l = 1:length(pd_))
    base_gap <- 1.5
    extra_gap <- ifelse(is.finite(label_line_height), label_line_height, 0)
    level_gaps <- base_gap + (label_line_counts - 1) * extra_gap
    level_centers <- cumsum(level_gaps)
    ypos_grid$ypos <- level_centers[ypos_grid$l] + (yScale) * (ypos_grid$k - 1)/1

    # Calculate adaptive margins based on label lengths
    max_label_len <- max(vapply(wrapped_level_names, function(lbl) {
      if (is.na(lbl)) return(0L)
      max(nchar(strsplit(lbl, "\n", fixed = TRUE)[[1]]))
    }, integer(1)), na.rm = TRUE)
    if (!is.finite(max_label_len)) max_label_len <- 0
    left_margin <- min(max(5, max_label_len * 0.6), 18)  # Cap at 18

    # Use provided margins if available, otherwise use adaptive defaults
    if (!is.null(margins_vec)) {
      par(mar = margins_vec)
    } else {
      par(mar = c(3.5, left_margin, 2, 1))
    }
    ylim <- c(0.8, max(ypos_grid$ypos) + 0.20)
    xlim_use <- if(is.null(xlim)) c(0, 1) else xlim
    plot(pd_[ordering_d], 1:length(pd_),
         ylim = ylim,
         main = prettyFactorNames,
         cex.main = cex.main, cex = 0,
         xlim = xlim_use,
         yaxt = "n", xlab = "", ylab = "")
    
    for(l_ in 1:length(pd_)) {
      shadowk_ <- 0
      for(k_ in k_EST:1) {
        shadowk_ <- shadowk_ + 1
        pi_kd <- pi_star_list[[k_]][[d_]][ordering_d]
        
        col_ <- if(is.null(col_vec)) k_ else col_vec[k_]
        y_loc <- ypos_grid[ypos_grid$l == l_ & ypos_grid$k == shadowk_, "ypos"]
        
        if(ticks_type == "assignmentProbs") {
          points(p_d[l_], y_loc, pch = "|", col = "gray", cex = 1.5)
        }
        if(ticks_type == "zero") {
          points(0, y_loc, pch = "|", col = "gray", cex = 1.5)
        }
        
        if(plot_ci && !is.null(pi_star_se_list) &&
           length(pi_star_se_list) >= k_ &&
           length(pi_star_se_list[[k_]]) >= d_ &&
           !is.null(pi_star_se_list[[k_]][[d_]])) {
          se_kd <- pi_star_se_list[[k_]][[d_]][ordering_d]
          points(c(min(xlim_use[2], min(1,pi_kd[l_] + zStar * se_kd[l_])),
                   max(xlim_use[1], max(0,pi_kd[l_] - zStar * se_kd[l_]))),
                 c(y_loc, y_loc), lwd = 2, type = "l", col = col_)
        }
        points(pi_kd[l_], y_loc, pch = pch, col = col_, cex = 1.5)
      }
    }
    
    # Apply level name transformation
    my_names <- wrapped_level_names
    
    if(plot_names && !add) {
      axis(2,
           at = tapply(ypos_grid$ypos, ypos_grid$l, mean),
           labels = my_names, 
           tick = FALSE,
           col = "black",
           lwd = 1,
           las = 2,
           cex.axis = cex.axis)
    }
  }
  
  invisible(NULL)
}
