plot_best_response_extract_curves <- function(grid_points,
                                              ast_surface,
                                              dag_surface) {
  grid_points <- as.numeric(grid_points)
  ast_surface <- as.matrix(ast_surface)
  dag_surface <- as.matrix(dag_surface)

  n_grid <- length(grid_points)
  if (n_grid < 1L ||
      nrow(ast_surface) != n_grid ||
      ncol(ast_surface) != n_grid ||
      nrow(dag_surface) != n_grid ||
      ncol(dag_surface) != n_grid) {
    stop("Best-response surfaces must be square matrices matching grid_points.", call. = FALSE)
  }

  ast_idx <- vapply(seq_len(ncol(ast_surface)), function(j) {
    which.max(ast_surface[, j])
  }, integer(1L))
  dag_idx <- max.col(dag_surface, ties.method = "first")

  list(
    br_dag_given_ast = grid_points[dag_idx],
    br_ast_given_dag = grid_points[ast_idx]
  )
}

#' Plot Dimension-by-Dimension Best-Response Curves from Adversarial \code{strategize()} Output
#'
#' @description
#' \code{plot_best_response_curves} takes the result of an adversarial 
#' \code{\link{strategize}} run (i.e., with \code{adversarial = TRUE}) and produces 
#' dimension-specific best-response curves. Specifically, for a chosen factor dimension 
#' \eqn{d}, it plots:
#' \enumerate{
#'   \item The curve of \eqn{\pi_{\mathrm{dag}, d}^{*}} as a function of 
#'         \eqn{\pi_{\mathrm{ast}, d}}.
#'   \item The curve of \eqn{\pi_{\mathrm{ast}, d}^{*}} as a function of 
#'         \eqn{\pi_{\mathrm{dag}, d}}.
#' }
#' Potential intersection points in this 2D space can indicate approximate equilibria 
#' for dimension \eqn{d}, holding the other dimensions fixed at the solution found by 
#' \code{strategize}.
#'
#' This function is computationally intensive: for each of \code{nPoints_br} grid values
#' of \eqn{\pi_{\mathrm{ast}, d}}, it searches over possible \eqn{\pi_{\mathrm{dag}, d}} 
#' (and vice versa) to find each side's best response through batched JAX grid evaluation.
#' Nonetheless, it provides a direct visualization of how each player (ast or dag) responds 
#' to changes in the other's distribution along a single factor dimension.
#'
#' @usage
#' plot_best_response_curves(
#'   res,
#'   d_ = 1,
#'   nPoints_br = 100L,
#'   nPoints_heat = 50L,
#'   title = NULL,
#'   col_ast = "blue",
#'   col_dag = "red",
#'   lwd_ast = 2,
#'   lwd_dag = 2,
#'   point_pch = 19,
#'   silent = FALSE
#' )
#'
#' @param res A list returned by \code{\link{strategize}}, which must include 
#'   adversarial references. Internally, \code{res} should contain items like 
#'   \code{res$a_i_ast}, \code{res$a_i_dag}, the JAX-based functions 
#'   (\code{dQ_da_ast}, \code{dQ_da_dag}, \code{QFXN}, \ldots), along with the 
#'   appropriate unconstrained parameter vectors.
#' @param d_ (Integer) The dimension of \eqn{\pi_{\mathrm{ast}}, \pi_{\mathrm{dag}}} to examine. 
#'   For example, if you have multiple factors (dimensions), each is indexed by a positive integer. 
#'   Defaults to 1.
#' @param nPoints_br (Integer) Number of equally spaced grid points in \eqn{[0,1]}
#'   to sample for \emph{the outer} loop. The code does an internal small search
#'   for best responses at each grid point. Larger \code{nPoints_br} => smoother curves
#'   but more computation. Defaults to \code{100L}.
#' @param nPoints_heat (Integer) Number of grid points for heat map calculations.
#'   Defaults to \code{50L}.
#' @param title (Character or \code{NULL}) Main plot title. If \code{NULL}, 
#'   an auto-generated title is used, e.g. "Best-Response Curves (Dimension d_=1)".
#' @param col_ast,col_dag (Character) Colors for ast’s and dag’s best-response curves, 
#'   respectively. Defaults to \code{"blue"} (ast) and \code{"red"} (dag).
#' @param lwd_ast,lwd_dag (Numeric) Line widths for ast and dag curves, respectively. 
#'   Default is 2.
#' @param point_pch (Numeric) Symbol for marking the approximate intersection 
#'   (if found) on the plot. Defaults to 19 (filled circle).
#' @param silent (Logical) If \code{TRUE}, suppresses printed messages during 
#'   the search for intersection. Defaults to \code{FALSE}.
#'
#' @return (Invisibly) A list containing:
#' \describe{
#'   \item{\code{grid_points}}{A numeric vector of the \code{nPoints_br} grid 
#'         values in \eqn{[0,1]} used for the outer loop.}
#'   \item{\code{br_dag_given_ast}}{A numeric vector of the same length, giving 
#'         the dag best-response \eqn{\pi_{\mathrm{dag}, d}} at each grid point 
#'         for \eqn{\pi_{\mathrm{ast}, d}}.}
#'   \item{\code{br_ast_given_dag}}{A numeric vector with the ast best-response 
#'         for each grid point \eqn{\pi_{\mathrm{dag}, d}}.}
#' }
#'
#' @details
#' \strong{Mechanics:}
#' For each \eqn{\pi_{\mathrm{ast}, d} \in \{0,\frac{1}{nPoints_br-1},\ldots,1\}}, 
#' the function temporarily fixes that dimension in the ast player’s unconstrained 
#' parameter vector. It then does an internal grid search over \eqn{\pi_{\mathrm{dag}, d}} 
#' to see which value yields the largest dag payoff (lowest ast payoff), consistent 
#' with \code{adversarial=TRUE}. The resulting curve is \eqn{\mathrm{BR}_{\mathrm{dag}}(\pi_{\mathrm{ast},d})}.
#'
#' Likewise, it holds \eqn{\pi_{\mathrm{dag}, d}} fixed and searches over 
#' \eqn{\pi_{\mathrm{ast}, d}} to get \eqn{\mathrm{BR}_{\mathrm{ast}}(\pi_{\mathrm{dag},d})}.
#'
#' The intersection in \eqn{\bigl(\pi_{\mathrm{ast}, d}, \pi_{\mathrm{dag}, d}\bigr)}-space 
#' (if one exists in the discretized grid) is a candidate local equilibrium for that factor 
#' dimension, \emph{given that the other factor dimensions remain at the solution from 
#' \code{\link{strategize}}.}
#'
#' \strong{Performance Caution:} This batched grid search evaluates many candidate
#' responses, which may still be slow for large \code{nPoints_br} or complex outcome
#' models. Consider using a smaller \code{nPoints_br} if performance is an issue, or
#' focusing on only a handful of crucial dimensions \eqn{d}.
#'
#' @seealso
#' \code{\link{strategize}} for obtaining the result object \code{res} in adversarial mode.
#' See also \code{\link{cv_strategize}}.
#'
#' @examples
#' \dontrun{
#' # =====================================================
#' # Visualize best-response curves in adversarial mode
#' # =====================================================
#' # First, fit an adversarial strategize model
#' set.seed(42)
#' n <- 400
#'
#' # Generate data with party structure
#' W <- data.frame(
#'   Gender = sample(c("Male", "Female"), n, replace = TRUE),
#'   Age = sample(c("Young", "Middle", "Old"), n, replace = TRUE)
#' )
#'
#' # Party affiliations for respondents and candidates
#' respondent_party <- sample(c("Dem", "Rep"), n/2, replace = TRUE)
#' candidate_party <- rep(c("Dem", "Rep"), n/2)
#'
#' Y <- rbinom(n, 1, 0.5)  # Simplified outcome
#'
#' # Fit adversarial model
#' adv_result <- strategize(
#'   Y = Y,
#'   W = W,
#'   lambda = 0.1,
#'   adversarial = TRUE,
#'   competing_group_variable_respondent = rep(respondent_party, each = 2),
#'   competing_group_variable_candidate = candidate_party,
#'   nSGD = 100
#' )
#'
#' # Plot best-response curves for Gender dimension (d_ = 1)
#' # Shows how each party's optimal Gender distribution responds
#' # to changes in the other party's Gender distribution
#' plot_best_response_curves(
#'   res = adv_result,
#'   d_ = 1,  # Gender is first factor
#'   nPoints_br = 50,
#'   title = "Gender: Best-Response Curves",
#'   col_ast = "blue",   # Democrats
#'   col_dag = "red"     # Republicans
#' )
#'
#' # Intersection point indicates Nash equilibrium for this dimension
#' }
#'
#' @md
#' @export
plot_best_response_curves  <- function(
                               res,
                               d_ = 1,
                               nPoints_br = 100L,
                               nPoints_heat = 50L,
                               title = NULL,
                               col_ast = "blue",
                               col_dag = "red",
                               lwd_ast = 2,
                               lwd_dag = 2,
                               point_pch = 19,
                               silent = FALSE){
  # define evaluation environment 
  evaluation_environment <- environment()
  package_environment <- environment(res$QFXN)
  strenv <- res$strenv
  compile_fxn <- function(x, static_argnums=NULL){return(strenv$jax$jit(x, static_argnums=static_argnums))}
  if(!silent) {
    message(sprintf(
      "[plot_best_response_curves] Building best-response curves for dimension d_ = %d...",
      d_))
  }
  
  ########################################################################
  ## 1) Extract from res the references we need
  ########################################################################
  a_ast_current <- res$a_i_ast
  a_dag_current <- res$a_i_dag
  
  # JAX function references
  #dQ_da_ast <- res$dQ_da_ast
  #dQ_da_dag <- res$dQ_da_dag
  #QFXN      <- res$QFXN
  #getQPiStar_gd      <- res$getQPiStar_gd
  #getMultinomialSamp <- res$getMultinomialSamp
  #getPrettyPi_diff <- res$getPrettyPi_diff
  #getPrettyPi_diff_R <- res$getPrettyPi_diff_R
  #environment(getPrettyPi_diff) <- strenv
  
  # Pi vectors + slates
  P_VEC_FULL_ast <- res$P_VEC_FULL_ast
  P_VEC_FULL_dag <- res$P_VEC_FULL_dag
  SLATE_VEC_ast  <- res$SLATE_VEC_ast
  SLATE_VEC_dag  <- res$SLATE_VEC_dag
  AstProp <- res$AstProp
  DagProp <- res$DagProp
  adversarial <- TRUE 
  
  # Regression param arrays
  REGRESSION_PARAMS_ast  <- res$REGRESSION_PARAMETERS_ast
  REGRESSION_PARAMS_dag  <- res$REGRESSION_PARAMETERS_dag
  REGRESSION_PARAMS_ast0 <- res$REGRESSION_PARAMETERS_ast0
  REGRESSION_PARAMS_dag0 <- res$REGRESSION_PARAMETERS_dag0
  
  # gather_fxn, a2Simplex, lambda
  gather_fxn          <- res$gather_fxn
  LAMBDA_             <- strenv$jnp$array(res$lambda, strenv$dtj)
  penalty_type        <- res$penalty_type
  
  d_locator <- res$d_locator
  d_locator_use <- res$d_locator_use
  main_comp_mat <- res$main_comp_mat
  shadow_comp_mat <- res$shadow_comp_mat
  intersection_dist_threshold <- 0.001
  
  START_VAL_SEARCH <- 0; STOP_VAL_SEARCH <- 1
  
  # setup environments 
  # multiround material
  for(DisaggreateQ in ifelse(adversarial, 
                             yes = list(c(F,T)), 
                             no = list(F))[[1]]){
    # general specifications
    getQStar_diff_ <- paste(deparse(getQStar_diff_BASE),collapse="\n")
    getQStar_diff_ <- gsub(getQStar_diff_, pattern = "Q_DISAGGREGATE", replacement = sprintf("T == %s", DisaggreateQ))
    getQStar_diff_ <- eval( parse( text = getQStar_diff_ ), envir = package_environment )
    
    # specifications for case (getQStar_diff_MultiGroup getQStar_diff_SingleGroup)
    eval(parse(text = sprintf("getQStar_diff_%sGroup <- compile_fxn( getQStar_diff_ )", 
                              ifelse(DisaggreateQ, yes = "Multi", no = "Single") )))
  }
  
  # environment management
  adversarial <- TRUE
  nMonte_adversarial <- 200L
  MNtemp <- res$temperature
  nMonte_Qglm <- 200L
  # d_locator_use already defined at line 186
  if (is.function(AstProp)) {
    environment(AstProp) <- evaluation_environment
  }
  if (is.function(DagProp)) {
    environment(DagProp) <- evaluation_environment
  }
  # rlang and reticulate are in Suggests/Imports - use :: syntax when needed
  FullGetQStar_eval <- res$FullGetQStar_
  if (is.null(FullGetQStar_eval)) {
    environment(FullGetQStar_) <- evaluation_environment
    FullGetQStar_eval <- strenv$jax$jit(FullGetQStar_)
  }

  REGRESSION_PARAMETERS_ast_parts <- gather_fxn(REGRESSION_PARAMS_ast)
  REGRESSION_PARAMETERS_dag_parts <- gather_fxn(REGRESSION_PARAMS_dag)
  REGRESSION_PARAMETERS_ast0_parts <- gather_fxn(REGRESSION_PARAMS_ast0)
  REGRESSION_PARAMETERS_dag0_parts <- gather_fxn(REGRESSION_PARAMS_dag0)
  INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast_parts[[1]]
  COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast_parts[[2]]
  INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag_parts[[1]]
  COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag_parts[[2]]
  INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0_parts[[1]]
  COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0_parts[[2]]
  INTERCEPT_dag0_ <- REGRESSION_PARAMETERS_dag0_parts[[1]]
  COEFFICIENTS_dag0_ <- REGRESSION_PARAMETERS_dag0_parts[[2]]

  dimension_index <- ai(d_ - 1L)
  plot_seed_key <- strenv$jax$random$PRNGKey(ai(0L))
  half_jnp <- strenv$jnp$array(0.5, dtype = strenv$dtj)
  lower_override <- strenv$jnp$array(-10.0, dtype = strenv$dtj)
  upper_override <- strenv$jnp$array(10.0, dtype = strenv$dtj)

  override_a_value_jax <- function(a_in, new_pi_value) {
    body <- function(iter_idx, carry) {
      a0 <- carry[[0L]]
      b0 <- carry[[1L]]
      mid <- strenv$jnp$multiply(strenv$jnp$add(a0, b0), half_jnp)
      a_trial <- a_in$at[[dimension_index]]$set(mid)
      pi_trial <- strenv$a2Simplex_diff_use(a_trial)
      pi_val <- strenv$jnp$take(pi_trial, dimension_index)
      update_low <- pi_val < new_pi_value
      list(
        strenv$jnp$where(update_low, mid, a0),
        strenv$jnp$where(update_low, b0, mid)
      )
    }
    carry <- strenv$jax$lax$fori_loop(
      ai(0L),
      ai(18L),
      body,
      list(lower_override, upper_override)
    )
    mid <- strenv$jnp$multiply(strenv$jnp$add(carry[[0L]], carry[[1L]]), half_jnp)
    a_in$at[[dimension_index]]$set(mid)
  }

  batch_override_ast <- strenv$jax$vmap(function(pi_value) {
    override_a_value_jax(a_ast_current, pi_value)
  }, in_axes = list(0L))
  batch_override_dag <- strenv$jax$vmap(function(pi_value) {
    override_a_value_jax(a_dag_current, pi_value)
  }, in_axes = list(0L))

  evaluate_surface <- function(ast_grid, dag_grid, seed_ids, q_sign) {
    a_ast_grid <- batch_override_ast(ast_grid)
    a_dag_grid <- batch_override_dag(dag_grid)

    evaluate_pair <- function(a_ast_test, a_dag_test, seed_id) {
      seed_key <- strenv$jax$random$fold_in(plot_seed_key, seed_id)
      value <- FullGetQStar_eval(
        a_ast_test,
        a_dag_test,
        INTERCEPT_ast_, COEFFICIENTS_ast_,
        INTERCEPT_dag_, COEFFICIENTS_dag_,
        INTERCEPT_ast0_, COEFFICIENTS_ast0_,
        INTERCEPT_dag0_, COEFFICIENTS_dag0_,
        P_VEC_FULL_ast, P_VEC_FULL_dag,
        SLATE_VEC_ast, SLATE_VEC_dag,
        LAMBDA_,
        q_sign,
        seed_key
      )
      strenv$jnp$take(strenv$jnp$ravel(value), 0L)
    }

    evaluate_row <- function(a_ast_test, seed_row) {
      strenv$jax$vmap(function(a_dag_test, seed_id) {
        evaluate_pair(a_ast_test, a_dag_test, seed_id)
      }, in_axes = list(0L, 0L))(a_dag_grid, seed_row)
    }

    strenv$jax$vmap(evaluate_row, in_axes = list(0L, 0L))(a_ast_grid, seed_ids)
  }
  evaluate_surface_jit <- strenv$jax$jit(evaluate_surface)

  evaluate_surface_r <- function(ast_grid, dag_grid, seed_ids, q_sign) {
    surface <- evaluate_surface_jit(
      strenv$jnp$array(as.numeric(ast_grid), dtype = strenv$dtj),
      strenv$jnp$array(as.numeric(dag_grid), dtype = strenv$dtj),
      strenv$jnp$array(as.matrix(seed_ids))$astype(strenv$jnp$int32),
      strenv$jnp$array(q_sign, dtype = strenv$dtj)
    )
    strategize_jax_block_until_ready(surface)
    as.matrix(strenv$np$array(surface))
  }
  
  ########################################################################
  ## 3) Evaluate best-response surfaces in batches.
  ########################################################################
  ########################################################################
  # 4) Evaluate these curves on an outer grid
  ########################################################################
  grid_points <- seq(0,1,length.out=nPoints_br)
  br_seed_ids <- outer(seq_len(nPoints_br), seq_len(nPoints_br), FUN = "*")
  if(!silent) {
    message("[plot_best_response_curves] Evaluating best-response surfaces with batched JAX...")
  }
  br_ast_surface <- evaluate_surface_r(grid_points, grid_points, br_seed_ids, 1.0)
  br_dag_surface <- evaluate_surface_r(grid_points, grid_points, br_seed_ids, -1.0)
  br_curves <- plot_best_response_extract_curves(
    grid_points = grid_points,
    ast_surface = br_ast_surface,
    dag_surface = br_dag_surface
  )
  br_dag_given_ast <- br_curves$br_dag_given_ast
  br_ast_given_dag <- br_curves$br_ast_given_dag
  
  ########################################################################
  # 5) Plot 
  ########################################################################
  if(is.null(title)){
    mainTitle <- sprintf("Best-Response Curves (Dimension d_=%d)", d_)
  } else {
    mainTitle <- title
  }
  
  # Try to find approximate intersection
  best_pt <- c(NA, NA)
  distMin <- Inf
  for(i in seq_along(grid_points)){
    x_ <- grid_points[i]
    y_ <- br_dag_given_ast[i]
    # find index i2 s.t. grid_points[i2] is close to y_
    i2 <- which.min(abs(grid_points - y_))
    x2_ <- br_ast_given_dag[i2]
    dist_ <- sqrt( (x_ - x2_)^2 + (y_ - grid_points[i2])^2 )
    if(dist_ < distMin){
      distMin <- dist_
      best_pt <- c(x_, y_)
    }
  }
  if(distMin < 0.05){
    if(!silent){
      message(sprintf(
        "Closest intersection is (x=%.3f, y=%.3f) with dist=%.4f in dimension d_=%d.",
        best_pt[1], best_pt[2], distMin, d_))
    }
  }
  
  # HEATMAP 
  {
    xvals <- seq(START_VAL_SEARCH, STOP_VAL_SEARCH, length.out=nPoints_heat)
    yvals <- seq(START_VAL_SEARCH, STOP_VAL_SEARCH, length.out=nPoints_heat)
    
    xvals_mat <- matrix(rep(xvals, times = nPoints_heat), nrow = nPoints_heat, ncol = nPoints_heat)
    yvals_mat <- matrix(rep(yvals, each = nPoints_heat), nrow = nPoints_heat, ncol = nPoints_heat)
    heat_seed_ids <- matrix(seq_len(nPoints_heat * nPoints_heat),
                            nrow = nPoints_heat,
                            ncol = nPoints_heat,
                            byrow = TRUE)
    if(!silent) {
      message("[plot_best_response_curves] Evaluating heatmap surfaces with batched JAX...")
    }
    z_ast <- evaluate_surface_r(xvals, yvals, heat_seed_ids, 1.0)
    z_dag <- evaluate_surface_r(xvals, yvals, heat_seed_ids, -1.0)
  }
  
  # helper fxn
  {
    geom_smooth2 <- function(mapping = NULL,
                             data = NULL,
                             threshold = 3,  # Number of unique points required in x and y
                             method = "loess",
                             se = FALSE,
                             color = "blue",
                             linewidth = 1.2,
                             na.rm = FALSE,
                             show.legend = NA,
                             inherit.aes = TRUE,
                             span = 1.5, 
                             ...) {
      
      # Make sure both 'mapping' and 'data' are provided
      if (is.null(mapping) || is.null(data)) {
        stop("Please specify both 'mapping' and 'data' for geom_smooth2().",
             call. = FALSE)
      }
      
      # Extract x and y variable names from the mapping
      x_var <- rlang::get_expr(mapping$x)
      y_var <- rlang::get_expr(mapping$y)
      
      if (is.null(x_var) || is.null(y_var)) {
        stop("Mapping must define both 'x' and 'y' for geom_smooth2().",
             call. = FALSE)
      }
      
      # Convert x_var and y_var to strings so we can use them to index the data
      x_col <- rlang::as_label(x_var)
      y_col <- rlang::as_label(y_var)
      
      # Count number of unique values for x and y
      n_unique_x <- length(unique(data[[x_col]]))
      n_unique_y <- length(unique(data[[y_col]]))
      
      # Conditionally return geom_smooth or geom_line
      if (n_unique_x > threshold && n_unique_y > threshold) {
        ggplot2::geom_smooth(
          mapping = mapping,
          data = data,
          method = method,
          se = se,
          color = color,
          linewidth = linewidth,
          na.rm = na.rm,
          show.legend = show.legend,
          inherit.aes = inherit.aes,
          span = span, 
          ...
        )
      } else {
        ggplot2::geom_line(
          mapping = mapping,
          data = data,
          color = color,
          linewidth = linewidth,
          na.rm = na.rm,
          show.legend = show.legend,
          inherit.aes = inherit.aes,
          ...
        )
      }
    }
  }
  
  # HEATMAP - MORE PLOTTING 
  {
    #############################################################################
    # 4) Convert these utility matrices to data.frames for ggplot
    ##############################################################################
    # We'll rely on row ix varying fastest along x, col iy along y
    # Actually, it’s standard to do expand.grid(xvals, yvals).
    # Then fill row by row: z_ast[ix, iy] => df_ast$utility_ast[row].
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
      stop("Package 'ggplot2' is required for plotting. Please install it.", call. = FALSE)
    }
    if (!requireNamespace("gridExtra", quietly = TRUE)) {
      stop("Package 'gridExtra' is required for plotting. Please install it.", call. = FALSE)
    }
    
    df_ast <- data.frame(
                    "pi_ast"=c(xvals_mat),
                    "pi_dag"=c(yvals_mat),
                    "utility_ast"=c(z_ast))
    df_dag <- data.frame(
                    "pi_ast"=c(xvals_mat),
                    "pi_dag"=c(yvals_mat),
                    "utility_dag"=c(z_dag))
    
    # Best-response lines data
    df_br_astGivenDag <- data.frame(
      pi_ast_br = br_ast_given_dag,
      pi_dag    = grid_points
    )
    df_br_dagGivenAst <- data.frame(
      pi_ast    = grid_points,
      pi_dag_br = br_dag_given_ast
    )
    
    # For labeling or highlighting intersection:
    intersection_point <- NULL
    showIntersection <- (!is.na(best_pt[1]) && distMin < intersection_dist_threshold)
    if(showIntersection){
      intersection_point <- data.frame(
        pi_ast = best_pt[1],
        pi_dag = best_pt[2]
      )
      if(!silent){
        message(sprintf(
          "Closest intersection ~ (%.3f, %.3f) in dimension d_=%d [dist=%.4f]",
          best_pt[1], best_pt[2], d_, distMin
        ))
      }
    } else {
      if(!silent){
        message("No intersection found under threshold; min distance was ", round(distMin,4))
      }
    }
    
    # Title logic
    if(is.null(title)){
      mainTitle <- sprintf("Best-Response Curves (Dimension d_=%d)", d_)
    } else {
      mainTitle <- title
    }
    
    ##############################################################################
    # 5) Build the two ggplot heatmaps side by side
    #    p_ast => ast’s utility, p_dag => dag’s utility
    #    Overlaid with both best-response curves
    ##############################################################################
    mid_ast <- mean(df_ast$utility_ast, na.rm = TRUE)
    mid_dag <- mean(df_dag$utility_dag, na.rm = TRUE)

    eq_pt <- data.frame(
      pi_ast = as.numeric(strenv$np$array(res$pi_star_ast_vec_jnp[d_-1L])),
      pi_dag = as.numeric(strenv$np$array(res$pi_star_dag_vec_jnp[d_-1L]))
    )
    

    model_br_ast <- lm(pi_ast_br~pi_dag+pi_dag^2, 
                       data = df_br_astGivenDag)
    df_br_astGivenDag$pi_ast_br_hat <- predict(model_br_ast,
                                            newx=data.frame("pi_dag"=seq(0,1,length.out=10)))
    
    p_ast <- ggplot2::ggplot(df_ast, ggplot2::aes(x = pi_ast, y = pi_dag, fill = utility_ast)) +
      ggplot2::geom_tile() +
      ggplot2::scale_fill_gradient2(
        midpoint = mid_ast,
        low = "white", mid = "skyblue", high = "blue"
      ) +
      # best-response lines
      geom_smooth2(
        data = df_br_astGivenDag,
        #aes(x = pi_ast_br, y = pi_dag),
        ggplot2::aes(x = pi_ast_br_hat, y = pi_dag), # to handle
        color = col_ast, linewidth = lwd_ast, inherit.aes = FALSE
      ) +
      geom_smooth2(
        data = df_br_dagGivenAst,
        ggplot2::aes(x = pi_ast, y = pi_dag_br),
        color = col_dag, linewidth = lwd_dag, inherit.aes = FALSE,
      ) +
      # intersection point if found
      {
        if(showIntersection) {
          ggplot2::annotate("point", x = intersection_point$pi_ast, y = intersection_point$pi_dag,
                   shape = point_pch, size = 3, color = "black")
        } else {
          NULL
        }
      } +
      # equilibrium point
      ggplot2::geom_point(
        data = eq_pt,
        ggplot2::aes(x = pi_ast, y = pi_dag),
        shape = 8, color = "green", size = 5, inherit.aes=FALSE
      ) +
      ggplot2::labs(
        title = mainTitle,
        subtitle = " -- ast's Utility",
        x = expression(pi[ast]^{(d)}),
        y = expression(pi[dag]^{(d)}),
        fill = "Utility(ast)"
      ) +
      ggplot2::theme_minimal()
    
    p_dag <- ggplot2::ggplot(df_dag, ggplot2::aes(x = pi_ast, y = pi_dag, fill = utility_dag+0.001)) +
      ggplot2::geom_tile() +
      ggplot2::scale_fill_gradient2(
        midpoint = mid_dag,
        low = "white", mid = "pink", high = "red"
      )  +
      # best response line
      geom_smooth2(
        data = df_br_astGivenDag,
        #aes(x = pi_ast_br, y = pi_dag),
        ggplot2::aes(x = pi_ast_br_hat, y = pi_dag),
        color = col_ast, linewidth = lwd_ast, inherit.aes = FALSE
      ) +
      geom_smooth2(
        data = df_br_dagGivenAst,
        ggplot2::aes(x = pi_ast, y = pi_dag_br),
        color = col_dag, linewidth = lwd_dag, inherit.aes = FALSE
      ) +
      {
        if(showIntersection) {
          ggplot2::annotate("point", x = intersection_point$pi_ast, y = intersection_point$pi_dag,
                   shape = point_pch, size = 3, color = "black")
        } else {
          NULL
        }
      } +
      # equilibrium point
      ggplot2::geom_point(
        data = eq_pt,
        ggplot2::aes(x = pi_ast, y = pi_dag),
        shape = 8, color = "green", size = 5, inherit.aes=FALSE
      ) +
      ggplot2::labs(
        title = mainTitle,
        subtitle = " -- dag's Utility",
        x = expression(pi[ast]^{(d)}),
        y = expression(pi[dag]^{(d)}),
        fill = "Utility(dag)"
      ) +
      ggplot2::theme_minimal()
    
    # Print side-by-side via grid.arrange (optional)
    p_grid <- gridExtra::grid.arrange(p_ast, p_dag, nrow = 1)
    
    # plot(df_br_astGivenDag$pi_ast_br, df_br_astGivenDag$pi_dag) # 
    # plot(df_br_dagGivenAst$pi_ast, df_br_dagGivenAst$pi_dag_br) # 
  }
  
  if(!silent) {
    message("Done plotting best-response curves for dimension d_=", d_)
  }
  
  invisible(list(grid_points = grid_points,
                 br_dag_given_ast = br_dag_given_ast,
                 br_ast_given_dag = br_ast_given_dag,
                 p_ast = p_ast, 
                 p_dag = p_dag, 
                 p_grid = p_grid
                 ))
}
