#' Plot Four-Quadrant Contribution Breakdown
#'
#' Decomposes the equilibrium vote share Q* into contributions from four
#' distinct election scenarios, providing insight into which primary-to-general
#' election pathways drive the Nash equilibrium outcome.
#'
#' In adversarial mode, two parties simultaneously optimize their candidate
#' attribute distributions. Each party's "entrant" candidate (drawn from the
#' optimized distribution) competes in a primary against a "field" candidate
#' (drawn from the baseline distribution). The general election outcome
#' depends on which candidates win their respective primaries, creating
#' four possible scenarios. This function visualizes the relative importance
#' of each scenario to the overall equilibrium.
#'
#' @importFrom graphics barplot legend pie plot.new text
#'
#' @param result Output from \code{\link{strategize}} with \code{adversarial = TRUE}
#' @param type Character string specifying the plot type:
#'   \itemize{
#'     \item \code{"bar"}: Stacked bar chart showing contributions
#'     \item \code{"pie"}: Pie chart showing proportions
#'   }
#'   Default is \code{"bar"}.
#' @param nMonte Integer. Number of Monte Carlo samples for estimation.
#'   Default is 500.
#' @param verbose Logical. Whether to print progress messages. Default is \code{TRUE}.
#'
#' @return A list containing:
#'   \describe{
#'     \item{weights}{Named vector of four-quadrant weights (sum to 1)}
#'     \item{contributions}{Named vector of contributions to Q* from each quadrant}
#'     \item{Q_star}{Total equilibrium vote share}
#'     \item{plot}{Base R plot (invisible return)}
#'   }
#'
#' @details
#' The four scenarios (quadrants) represent different primary election outcomes:
#' \describe{
#'   \item{E1: Both Entrants}{Both parties' "entrant" candidates (sampled from
#'     optimized distributions) win their primaries}
#'   \item{E2: A Entrant, B Field}{Party A's entrant wins, Party B's field
#'     candidate (sampled from baseline) wins}
#'   \item{E3: A Field, B Entrant}{Party A's field wins, Party B's entrant wins}
#'   \item{E4: Both Field}{Both parties' field candidates win their primaries}
#' }
#'
#' The weight of each scenario depends on the primary election probabilities
#' (kappa values), which in turn depend on the voter model.
#'
#' @section Interpretation:
#' \itemize{
#'   \item A dominant E1 contribution suggests entrant vs. entrant matchups
#'     are most important for the equilibrium
#'   \item Balanced contributions indicate a robust equilibrium across scenarios
#'   \item Large E4 contribution suggests field candidates often win primaries
#' }
#'
#' @examples
#' \dontrun{
#' # Run adversarial strategize
#' result <- strategize(Y = y, W = w, adversarial = TRUE, nSGD = 500)
#'
#' # Plot quadrant breakdown
#' breakdown <- plot_quadrant_breakdown(result)
#' print(breakdown$weights)
#' print(breakdown$contributions)
#' }
#'
#' @export
plot_quadrant_breakdown <- function(result,
                                     type = c("bar", "pie"),
                                     nMonte = 500,
                                     verbose = TRUE) {

  type <- match.arg(type)

  # Validate input
  if (!isTRUE(result$convergence_history$adversarial)) {
    stop("plot_quadrant_breakdown() requires an adversarial strategize result. ",
         "Set adversarial = TRUE in strategize().")
  }

  strenv <- result$strenv
  if (is.null(strenv) || is.null(strenv$jnp)) {
    stop("JAX environment not available. Cannot compute quadrant breakdown.")
  }

  if (verbose) message("Computing four-quadrant breakdown...")

  # Compute quadrant breakdown
  breakdown <- compute_quadrant_breakdown(result, nMonte, verbose)

  # Create visualization
  if (verbose) message("Generating plot...")

  if (type == "bar") {
    plot_quadrant_bar(breakdown)
  } else {
    plot_quadrant_pie(breakdown)
  }

  return(invisible(list(
    weights = breakdown$weights,
    contributions = breakdown$contributions,
    Q_star = breakdown$Q_star,
    primary_probs = breakdown$primary_probs
  )))
}


#' Internal: Compute four-quadrant breakdown
#' @keywords internal
#' @noRd
compute_quadrant_breakdown <- function(result, nMonte, verbose) {

  strenv <- result$strenv
  simple_quadrant <- function(Q_star = as.numeric(result$Q_point)) {
    P_A_entrant <- 0.5
    P_B_entrant <- 0.5
    weights <- c(
      E1 = P_A_entrant * P_B_entrant,
      E2 = P_A_entrant * (1 - P_B_entrant),
      E3 = (1 - P_A_entrant) * P_B_entrant,
      E4 = (1 - P_A_entrant) * (1 - P_B_entrant)
    )
    contributions <- weights * Q_star
    list(
      weights = weights,
      contributions = contributions,
      Q_star = sum(contributions),
      primary_probs = list(P_A_entrant = P_A_entrant, P_B_entrant = P_B_entrant)
    )
  }

  # Get optimized parameters
  a_i_ast <- result$a_i_ast
  a_i_dag <- result$a_i_dag

  # Get model parameters for primary elections
  params_ast0 <- result$gather_fxn(result$REGRESSION_PARAMETERS_ast0)
  params_dag0 <- result$gather_fxn(result$REGRESSION_PARAMETERS_dag0)
  params_ast <- result$gather_fxn(result$REGRESSION_PARAMETERS_ast)
  params_dag <- result$gather_fxn(result$REGRESSION_PARAMETERS_dag)

  INTERCEPT_ast0 <- params_ast0[[1]]
  COEFFICIENTS_ast0 <- params_ast0[[2]]
  INTERCEPT_dag0 <- params_dag0[[1]]
  COEFFICIENTS_dag0 <- params_dag0[[2]]
  INTERCEPT_ast <- params_ast[[1]]
  COEFFICIENTS_ast <- params_ast[[2]]
  INTERCEPT_dag <- params_dag[[1]]
  COEFFICIENTS_dag <- params_dag[[2]]

  # Get baseline distributions
  P_VEC_FULL_ast <- result$P_VEC_FULL_ast
  P_VEC_FULL_dag <- result$P_VEC_FULL_dag
  SLATE_VEC_ast <- result$SLATE_VEC_ast
  SLATE_VEC_dag <- result$SLATE_VEC_dag

  # Convert optimized parameters to probability distributions
  pi_star_ast <- strenv$a2Simplex_diff_use(a_i_ast)
  pi_star_dag <- strenv$a2Simplex_diff_use(a_i_dag)

  # Estimate primary win probabilities using Monte Carlo
  # These are kappa_A and kappa_B from the model
  SEED <- strenv$jax$random$PRNGKey(as.integer(42))

  # Sample profiles from entrant and field distributions
  n_samples <- as.integer(nMonte)

  # Get sampling function
  if (!is.null(strenv$getMultinomialSamp)) {
    getMNSamp <- strenv$getMultinomialSamp
    ParameterizationType <- strenv$ParameterizationType
    d_locator_use <- strenv$d_locator_use
    MNtemp <- result$temperature
  } else {
    # Fallback: simple uniform sampling (less accurate but won't crash)
    if (verbose) message("  Note: Using simplified quadrant estimation")
    return(simple_quadrant())
  }

  safe_draw <- function(pi_vec, seed_in) {
    res <- tryCatch({
      draw_profile_samples(pi_vec, n_samples, seed_in,
                           MNtemp, ParameterizationType, d_locator_use,
                           sampler = getMNSamp)
    }, error = function(e) NULL)
    if (is.null(res)) {
      list(samples = NULL, seed_next = strenv$jax$random$split(seed_in)[[1L]])
    } else {
      res
    }
  }

  # Sample from optimized (entrant) distributions
  res <- safe_draw(pi_star_ast, SEED)
  sample_entrant_A <- res$samples
  SEED <- res$seed_next

  res <- safe_draw(pi_star_dag, SEED)
  sample_entrant_B <- res$samples
  SEED <- res$seed_next

  # Sample from baseline (field) distributions
  res <- safe_draw(SLATE_VEC_ast, SEED)
  sample_field_A <- res$samples
  SEED <- res$seed_next

  res <- safe_draw(SLATE_VEC_dag, SEED)
  sample_field_B <- res$samples
  SEED <- res$seed_next

  # Check if sampling succeeded
  if (is.null(sample_entrant_A) || is.null(sample_entrant_B) ||
      is.null(sample_field_A) || is.null(sample_field_B)) {
    if (verbose) message("  Note: Sampling failed, using simplified estimation")
    return(simple_quadrant())
  }

  primary_pushforward <- if (!is.null(result$convergence_history$primary_pushforward)) {
    tolower(result$convergence_history$primary_pushforward)
  } else {
    "mc"
  }
  primary_strength <- result$primary_strength
  if (is.null(primary_strength)) {
    primary_strength <- strenv$jnp$array(1.0, strenv$dtj)
  } else if (is.numeric(primary_strength)) {
    primary_strength <- strenv$jnp$array(as.numeric(primary_strength), strenv$dtj)
  }

  q_env <- NULL
  if (!is.null(result$getQPiStar_gd)) {
    q_env <- environment(result$getQPiStar_gd)
  }
  getQStar_diff_SingleGroup <- NULL
  if (!is.null(q_env) && exists("getQStar_diff_SingleGroup", envir = q_env)) {
    getQStar_diff_SingleGroup <- get("getQStar_diff_SingleGroup", envir = q_env)
  }
  QFXN <- result$QFXN
  if (is.null(QFXN) && !is.null(q_env) && exists("getQStar_diff_MultiGroup", envir = q_env)) {
    QFXN <- get("getQStar_diff_MultiGroup", envir = q_env)
  }
  if (is.null(getQStar_diff_SingleGroup) || is.null(QFXN)) {
    if (verbose) message("  Note: Primary model unavailable, using simplified estimation")
    return(simple_quadrant())
  }

  kappa_pair <- function(v, v_prime, intercept, coeffs) {
    strenv$jnp$take(
      getQStar_diff_SingleGroup(
        v,
        v_prime,
        intercept * primary_strength, primary_strength * coeffs,
        intercept * primary_strength, primary_strength * coeffs
      ),
      0L
    )
  }
  Qpop_pair <- function(t, u) {
    strenv$jnp$take(
      QFXN(t, u,
           INTERCEPT_ast, COEFFICIENTS_ast,
           INTERCEPT_dag, COEFFICIENTS_dag),
      0L
    )
  }

  if (primary_pushforward == "multi") {
    n_entrants <- if (is.null(result$primary_n_entrants)) 1L else as.integer(result$primary_n_entrants)
    n_field <- if (is.null(result$primary_n_field)) 1L else as.integer(result$primary_n_field)
    n_entrants <- max(1L, n_entrants)
    n_field <- max(1L, n_field)

    samp_ast <- sample_pool_jax(
      pi_star_ast, n_samples, n_entrants, SEED,
      MNtemp, ParameterizationType, d_locator_use,
      sampler = getMNSamp
    )
    TSAMP_ast_all <- samp_ast$samples
    SEED <- samp_ast$seed_next

    samp_dag <- sample_pool_jax(
      pi_star_dag, n_samples, n_entrants, SEED,
      MNtemp, ParameterizationType, d_locator_use,
      sampler = getMNSamp
    )
    TSAMP_dag_all <- samp_dag$samples
    SEED <- samp_dag$seed_next

    samp_ast_field <- sample_pool_jax(
      SLATE_VEC_ast, n_samples, n_field, SEED,
      MNtemp, ParameterizationType, d_locator_use,
      sampler = getMNSamp
    )
    TSAMP_ast_field_all <- samp_ast_field$samples
    SEED <- samp_ast_field$seed_next

    samp_dag_field <- sample_pool_jax(
      SLATE_VEC_dag, n_samples, n_field, SEED,
      MNtemp, ParameterizationType, d_locator_use,
      sampler = getMNSamp
    )
    TSAMP_dag_field_all <- samp_dag_field$samples
    SEED <- samp_dag_field$seed_next

    draw_stats <- strenv$jax$vmap(function(tsamp_ast, tsamp_dag,
                                          tsamp_ast_field, tsamp_dag_field) {
      cand_ast <- strenv$jnp$concatenate(list(tsamp_ast, tsamp_ast_field), 0L)
      cand_dag <- strenv$jnp$concatenate(list(tsamp_dag, tsamp_dag_field), 0L)
      nA <- as.integer(n_entrants + n_field)
      nB <- as.integer(n_entrants + n_field)

      kA <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(t_j){
          kappa_pair(t_i, t_j, INTERCEPT_ast0, COEFFICIENTS_ast0)
        }, in_axes = 0L)(cand_ast)
      }, in_axes = 0L)(cand_ast)
      kB <- strenv$jax$vmap(function(u_i){
        strenv$jax$vmap(function(u_j){
          kappa_pair(u_i, u_j, INTERCEPT_dag0, COEFFICIENTS_dag0)
        }, in_axes = 0L)(cand_dag)
      }, in_axes = 0L)(cand_dag)

      maskA <- strenv$jnp$ones(list(nA, nA), dtype = strenv$dtj) - strenv$jnp$eye(nA, dtype = strenv$dtj)
      maskB <- strenv$jnp$ones(list(nB, nB), dtype = strenv$dtj) - strenv$jnp$eye(nB, dtype = strenv$dtj)
      eps <- strenv$jnp$maximum(
        strenv$jnp$array(1e-8, strenv$dtj),
        strenv$jnp$array(strenv$jnp$finfo(strenv$dtj)$eps, strenv$dtj)
      )
      one_bt <- strenv$jnp$array(1.0, strenv$dtj)
      kA_clip <- strenv$jnp$clip(kA, eps, one_bt - eps)
      kB_clip <- strenv$jnp$clip(kB, eps, one_bt - eps)
      logoddsA <- strenv$jnp$log(kA_clip) - strenv$jnp$log(one_bt - kA_clip)
      logoddsB <- strenv$jnp$log(kB_clip) - strenv$jnp$log(one_bt - kB_clip)
      denomA <- strenv$jnp$array(nA, strenv$dtj)
      denomB <- strenv$jnp$array(nB, strenv$dtj)
      utilityA <- (logoddsA * maskA)$sum(axis = 1L) / denomA
      utilityB <- (logoddsB * maskB)$sum(axis = 1L) / denomB
      pA <- strenv$jax$nn$softmax(utilityA)
      pB <- strenv$jax$nn$softmax(utilityB)

      idxA_e <- strenv$jnp$arange(as.integer(n_entrants))
      idxA_f <- strenv$jnp$arange(as.integer(n_entrants), as.integer(nA))
      idxB_e <- strenv$jnp$arange(as.integer(n_entrants))
      idxB_f <- strenv$jnp$arange(as.integer(n_entrants), as.integer(nB))

      pA_e <- strenv$jnp$take(pA, idxA_e)
      pA_f <- strenv$jnp$take(pA, idxA_f)
      pB_e <- strenv$jnp$take(pB, idxB_e)
      pB_f <- strenv$jnp$take(pB, idxB_f)

      C_ab <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(u_j){
          Qpop_pair(t_i, u_j)
        }, in_axes = 0L)(cand_dag)
      }, in_axes = 0L)(cand_ast)

      C_ee <- strenv$jnp$take(strenv$jnp$take(C_ab, idxA_e, axis = 0L), idxB_e, axis = 1L)
      C_ef <- strenv$jnp$take(strenv$jnp$take(C_ab, idxA_e, axis = 0L), idxB_f, axis = 1L)
      C_fe <- strenv$jnp$take(strenv$jnp$take(C_ab, idxA_f, axis = 0L), idxB_e, axis = 1L)
      C_ff <- strenv$jnp$take(strenv$jnp$take(C_ab, idxA_f, axis = 0L), idxB_f, axis = 1L)

      pA_e_col <- strenv$jnp$expand_dims(pA_e, 1L)
      pA_f_col <- strenv$jnp$expand_dims(pA_f, 1L)
      pB_e_row <- strenv$jnp$expand_dims(pB_e, 0L)
      pB_f_row <- strenv$jnp$expand_dims(pB_f, 0L)

      list(
        E1 = strenv$jnp$sum(C_ee * pA_e_col * pB_e_row),
        E2 = strenv$jnp$sum(C_ef * pA_e_col * pB_f_row),
        E3 = strenv$jnp$sum(C_fe * pA_f_col * pB_e_row),
        E4 = strenv$jnp$sum(C_ff * pA_f_col * pB_f_row),
        P_A_entrant = strenv$jnp$sum(pA_e),
        P_B_entrant = strenv$jnp$sum(pB_e)
      )
    }, in_axes = list(0L, 0L, 0L, 0L))(TSAMP_ast_all, TSAMP_dag_all,
                                      TSAMP_ast_field_all, TSAMP_dag_field_all)

    P_A_entrant <- as.numeric(strenv$np$array(draw_stats$P_A_entrant$mean()))
    P_B_entrant <- as.numeric(strenv$np$array(draw_stats$P_B_entrant$mean()))
    contributions <- c(
      E1 = as.numeric(strenv$np$array(draw_stats$E1$mean())),
      E2 = as.numeric(strenv$np$array(draw_stats$E2$mean())),
      E3 = as.numeric(strenv$np$array(draw_stats$E3$mean())),
      E4 = as.numeric(strenv$np$array(draw_stats$E4$mean()))
    )
  } else {
    kA_vec <- strenv$jax$vmap(function(t_i, t_j) {
      kappa_pair(t_i, t_j, INTERCEPT_ast0, COEFFICIENTS_ast0)
    }, in_axes = list(0L, 0L))(sample_entrant_A, sample_field_A)
    kB_vec <- strenv$jax$vmap(function(u_i, u_j) {
      kappa_pair(u_i, u_j, INTERCEPT_dag0, COEFFICIENTS_dag0)
    }, in_axes = list(0L, 0L))(sample_entrant_B, sample_field_B)

    P_A_entrant <- as.numeric(strenv$np$array(kA_vec$mean()))
    P_B_entrant <- as.numeric(strenv$np$array(kB_vec$mean()))

    Q_tt <- strenv$jax$vmap(function(t_i, u_i) {
      Qpop_pair(t_i, u_i)
    }, in_axes = list(0L, 0L))(sample_entrant_A, sample_entrant_B)$mean()
    Q_tf <- strenv$jax$vmap(function(t_i, u_i) {
      Qpop_pair(t_i, u_i)
    }, in_axes = list(0L, 0L))(sample_entrant_A, sample_field_B)$mean()
    Q_ft <- strenv$jax$vmap(function(t_i, u_i) {
      Qpop_pair(t_i, u_i)
    }, in_axes = list(0L, 0L))(sample_field_A, sample_entrant_B)$mean()
    Q_ff <- strenv$jax$vmap(function(t_i, u_i) {
      Qpop_pair(t_i, u_i)
    }, in_axes = list(0L, 0L))(sample_field_A, sample_field_B)$mean()

    weights <- c(
      E1 = P_A_entrant * P_B_entrant,
      E2 = P_A_entrant * (1 - P_B_entrant),
      E3 = (1 - P_A_entrant) * P_B_entrant,
      E4 = (1 - P_A_entrant) * (1 - P_B_entrant)
    )
    contributions <- weights * c(
      E1 = as.numeric(strenv$np$array(Q_tt)),
      E2 = as.numeric(strenv$np$array(Q_tf)),
      E3 = as.numeric(strenv$np$array(Q_ft)),
      E4 = as.numeric(strenv$np$array(Q_ff))
    )
  }

  # Compute weights
  weights <- c(
    E1 = P_A_entrant * P_B_entrant,
    E2 = P_A_entrant * (1 - P_B_entrant),
    E3 = (1 - P_A_entrant) * P_B_entrant,
    E4 = (1 - P_A_entrant) * (1 - P_B_entrant)
  )

  if (verbose) {
    message(sprintf("  P(A entrant wins): %.3f", P_A_entrant))
    message(sprintf("  P(B entrant wins): %.3f", P_B_entrant))
    message(sprintf("  Q*: %.3f", sum(contributions)))
  }

  return(list(
    weights = weights,
    contributions = contributions,
    Q_star = sum(contributions),
    primary_probs = list(P_A_entrant = P_A_entrant, P_B_entrant = P_B_entrant)
  ))
}


#' Internal: Bar plot for quadrant breakdown
#' @keywords internal
#' @noRd
plot_quadrant_bar <- function(breakdown) {

  old_par <- par(mar = c(5, 4, 4, 8), xpd = TRUE)
  on.exit(par(old_par))

  # Colors for the four quadrants
  colors <- c(
    E1 = "#E41A1C",  # Both entrants (red)
    E2 = "#FF7F00",  # A entrant, B field (orange)
    E3 = "#377EB8",  # A field, B entrant (blue)
    E4 = "#4DAF4A"   # Both field (green)
  )

  weights <- breakdown$weights
  contributions <- breakdown$contributions

  # Create stacked bar for weights (cbind creates 4 rows x 1 col = single stacked bar)
  barplot(
    cbind(weights),
    beside = FALSE,
    col = colors,
    main = "Four-Quadrant Breakdown",
    ylab = "Proportion",
    ylim = c(0, 1.1),
    names.arg = "Scenario\nWeights"
  )

  # Add legend
  legend("topright",
         inset = c(-0.3, 0),
         legend = c(
           "E1: Both Entrants",
           "E2: A Ent., B Field",
           "E3: A Field, B Ent.",
           "E4: Both Field"
         ),
         fill = colors,
         bty = "n",
         cex = 0.8)

  # Add percentages as labels
  cumweights <- cumsum(weights)
  midpoints <- c(0, cumweights[-4]) + weights / 2

  text(0.7, midpoints,
       sprintf("%.1f%%", weights * 100),
       col = "white",
       font = 2,
       cex = 0.9)

  # Add Q* annotation
  text(0.7, 1.05,
       sprintf("Q* = %.3f", breakdown$Q_star),
       font = 2,
       cex = 1.1)

  invisible(NULL)
}


#' Internal: Pie chart for quadrant breakdown
#' @keywords internal
#' @noRd
plot_quadrant_pie <- function(breakdown) {

  old_par <- par(mar = c(2, 2, 3, 2))
  on.exit(par(old_par))

  colors <- c(
    E1 = "#E41A1C",
    E2 = "#FF7F00",
    E3 = "#377EB8",
    E4 = "#4DAF4A"
  )

  weights <- breakdown$weights

  labels <- sprintf("%s\n%.1f%%",
                    c("Both Entrants", "A Ent/B Field",
                      "A Field/B Ent", "Both Field"),
                    weights * 100)

  pie(weights,
      labels = labels,
      col = colors,
      main = sprintf("Four-Quadrant Breakdown (Q* = %.3f)", breakdown$Q_star),
      clockwise = TRUE)

  invisible(NULL)
}
