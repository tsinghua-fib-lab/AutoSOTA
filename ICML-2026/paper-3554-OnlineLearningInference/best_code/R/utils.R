# Cache for intermediate computations to avoid repeated calculation
.colsa_cache <- new.env(parent = emptyenv())

# Internal helper: Get or create cached prox_forward matrix
get_prox_cache <- function(n_basis, n_basis_pre, n_features) {
  key <- paste0("prox_", n_basis, "_", n_basis_pre, "_", n_features)
  if (!exists(key, envir = .colsa_cache)) {
    prox_base <- prox_forward(n_basis, n_basis_pre)
    prox <- rbind(
      cbind(prox_base, matrix(0, nrow(prox_base), n_features)),
      cbind(matrix(0, n_features, ncol(prox_base)), diag(n_features))
    )
    assign(key, prox, envir = .colsa_cache)
  }
  get(key, envir = .colsa_cache)
}

# Internal helper: Get or create cached bpoly matrix for given times
get_bpoly_cache <- function(time, n_basis, boundary) {
  n <- length(time)
  key <- paste0(
    "bpoly_", n, "_", n_basis, "_",
    round(sum(time), 6), "_", round(time[1], 6), "_", round(time[n], 6)
  )
  if (!exists(key, envir = .colsa_cache)) {
    b <- if (n_basis == 1) {
      matrix(1, nrow = n, ncol = 1)
    } else {
      splines2::bpoly(
        time,
        degree = n_basis - 1, intercept = TRUE,
        Boundary.knots = boundary
      )
    }
    assign(key, b, envir = .colsa_cache)
  }
  get(key, envir = .colsa_cache)
}

# Internal helper: Get or create cached quadrature nodes
get_quad_cache <- function(n_nodes) {
  key <- paste0("quad_", n_nodes)
  if (!exists(key, envir = .colsa_cache)) {
    quad <- gauss.quad.prob(n_nodes, "uniform", l = 0, u = 1)
    assign(key, quad, envir = .colsa_cache)
  }

  get(key, envir = .colsa_cache)
}

# Internal helper: Get or create cached index vectors for outer product
get_idx_cache <- function(n_basis) {
  key <- paste0("idx_", n_basis)
  if (!exists(key, envir = .colsa_cache)) {
    idx <- list(
      idx1 = rep.int(seq_len(n_basis), n_basis),
      idx2 = rep(seq_len(n_basis), each = n_basis)
    )
    assign(key, idx, envir = .colsa_cache)
  }
  get(key, envir = .colsa_cache)
}

#' Objective Function for Cox Model Optimization
#'
#' Computes the objective value, gradient, and Hessian for optimizing a Cox
#' proportional hazards model with spline-based baseline hazard estimation.
#'
#' @param par Numeric vector of parameters to be optimized, including spline
#'        coefficients for the baseline hazard (`alpha`) and regression
#'        coefficients (`beta`).
#' @param time Numeric vector of observed survival times.
#' @param status Numeric vector of event indicators.
#' @param x Numeric matrix of covariates (features) for the model.
#' @param boundary Numeric vector specifying the boundary knots for the splines.
#' @param theta Numeric vector of prior parameters for regularization.
#' @param hessian_prev Numeric matrix of prior Hessian for regularization.
#'
#' @return A numeric value with attributes `gradient` and `hessian`.
#'
#' @details
#' The objective function combines two components:
#' - A regularization term based on prior parameters (`theta`) and prior Hessian
#'   (`hessian_prev`).
#' - A likelihood-based term for the Cox proportional hazards model, where the
#'   baseline hazard is modeled using Bernstein polynomials.
#'
#' @keywords internal
objective <- function(par, time, status, x, boundary, theta, hessian_prev) {
  n_parameters <- length(par)
  n_features <- ncol(x)
  n_basis <- n_parameters - n_features

  n_parameters_pre <- length(theta)
  n_basis_pre <- n_parameters_pre - n_features

  # Get cached prox matrix (only computed once per optimization)
  prox <- get_prox_cache(n_basis, n_basis_pre, n_features)

  # Use cached QR decomposition for solving
  theta_trans <- qr.solve(prox, theta)
  hess1 <- crossprod(prox, hessian_prev) %*% prox

  par_diff <- par - theta_trans
  hess1_par_diff <- hess1 %*% par_diff
  loss1 <- as.numeric(crossprod(par_diff, hess1_par_diff)) / 2
  grad1 <- as.vector(hess1_par_diff)

  alpha <- par[seq_len(n_basis)]
  beta <- par[(n_basis + 1):n_parameters]

  # Get cached bpoly matrix
  b <- get_bpoly_cache(time, n_basis, boundary)
  eta <- x %*% beta
  exp_eta <- exp(eta)

  # Vectorized computation of cumulative baseline hazard and derivatives
  int_result <- int_basehaz(as.vector(time), alpha, n_basis, boundary)
  cbh <- int_result$cbh
  dcbh <- int_result$dcbh
  d2cbh <- int_result$d2cbh

  # Optimized gradient and Hessian computation
  dalpha <- crossprod(dcbh, exp_eta) - crossprod(b, status)
  cbh_exp_eta <- cbh * exp_eta
  dbeta <- crossprod(x, cbh_exp_eta - status)

  # Hessian for alpha: use crossprod instead of sweep + colSums
  d2alpha <- matrix(crossprod(d2cbh, exp_eta), n_basis, n_basis)

  # Hessian for beta: use efficient weighted crossprod
  d2beta <- crossprod(x, as.vector(cbh_exp_eta) * x)

  # Cross-Hessian
  dalph_dabeta <- crossprod(dcbh, as.vector(exp_eta) * x)

  loss2 <- sum(cbh_exp_eta) - sum(status * (b %*% alpha + eta))
  grad2 <- c(dalpha, dbeta)
  hess2 <- cbind(
    rbind(d2alpha, t(dalph_dabeta)),
    rbind(dalph_dabeta, d2beta)
  )

  res <- loss1 + loss2
  attr(res, "gradient") <- grad1 + grad2
  attr(res, "hessian") <- hess1 + hess2
  res
}

#' Compute Integrated Baseline Hazard and Its Derivatives
#'
#' Computes the integrated baseline hazard and its derivatives for multiple
#' time points using Gaussian quadrature and Bernstein polynomial basis.
#'
#' @param times Numeric vector of time points (upper limits of integration).
#' @param alpha Numeric vector of baseline hazard coefficients.
#' @param n_basis Integer number of Bernstein polynomial basis functions.
#' @param boundary Numeric vector of length 2 for boundary knots.
#' @param n_nodes Integer number of quadrature nodes (default 15).
#'
#' @return A list containing:
#'   \item{cbh}{Numeric vector of cumulative baseline hazards.}
#'   \item{dcbh}{Matrix of gradients (n_times x n_basis).}
#'   \item{d2cbh}{Matrix of vectorized Hessians (n_times x n_basis^2).}
#'
#' @keywords internal
#' @importFrom statmod gauss.quad.prob
int_basehaz <- function(times, alpha, n_basis, boundary, n_nodes = 15) {
  n_times <- length(times)

  # Handle trivial case first (before quadrature lookup)
  if (n_basis == 1) {
    bh_val <- exp(alpha[1])
    cbh <- times * bh_val
    dcbh <- matrix(cbh, ncol = 1)
    d2cbh <- matrix(cbh, ncol = 1)
    return(list(cbh = cbh, dcbh = dcbh, d2cbh = d2cbh))
  }

  # Get cached quadrature nodes
  quad_unit <- get_quad_cache(n_nodes)
  nodes_unit <- quad_unit$nodes
  weights_unit <- quad_unit$weights

  # Create all quadrature nodes: nodes_all[i, j] = times[i] * nodes_unit[j]
  nodes_all <- outer(as.vector(times), nodes_unit)

  # Flatten and compute basis functions in one call
  nodes_flat <- as.vector(t(nodes_all))
  b_flat <- splines2::bpoly(
    nodes_flat,
    degree = n_basis - 1, intercept = TRUE,
    Boundary.knots = boundary
  )

  # Compute baseline hazard and weighted values
  bh_flat <- exp(as.vector(b_flat %*% alpha))
  w_bh_flat <- bh_flat * rep.int(weights_unit, n_times)

  # cbh: sum of weighted hazard (avoid matrix reshape)
  w_bh_mat <- matrix(w_bh_flat, nrow = n_times, ncol = n_nodes, byrow = TRUE)
  cbh <- times * .rowSums(w_bh_mat, n_times, n_nodes)

  # Group index for rowsum (use rep.int for speed)
  group_idx <- rep.int(seq_len(n_times), rep.int(n_nodes, n_times))

  # Gradient: use rowsum with pre-multiplied weights
  b_weighted <- b_flat * w_bh_flat
  dcbh <- rowsum(b_weighted, group_idx, reorder = FALSE) * times

  # Hessian: vectorized outer product with cached indices
  idx_cache <- get_idx_cache(n_basis)
  b_outer <- b_flat[, idx_cache$idx1, drop = FALSE] *
    b_flat[, idx_cache$idx2, drop = FALSE]
  b_outer_weighted <- b_outer * w_bh_flat
  d2cbh <- rowsum(b_outer_weighted, group_idx, reorder = FALSE) * times

  list(cbh = cbh, dcbh = dcbh, d2cbh = d2cbh)
}

#' Proximal Operator (Forward)
#'
#' Constructs a transformation matrix to map the coefficients of Bernstein
#' polynomial basis functions from one dimension to another, ensuring
#' consistency with the properties of Bernstein polynomial bases. Note that
#' the original dimension must be less than or equal to the target dimension.
#'
#' @param p Integer. The number of basis functions in the original dimension.
#' @param q Integer. The number of basis functions in the target dimension.
#'
#' @return A matrix representing the transformation from the original dimension
#'         to the target dimension.
#'
#' @details This function iteratively constructs the transformation matrix by
#'          expanding the basis functions from \code{p} to \code{q}, while
#'          preserving the mathematical properties of Bernstein polynomials.
#'
#' @seealso \code{\link{prox_reverse}}
#' @keywords internal
prox_forward <- function(p, q) {
  if (
    !is.numeric(p) || !is.numeric(q) || p != as.integer(p) || q != as.integer(q)
  ) {
    stop("p and q must be integers")
  }
  if (p <= 0 || q <= 0) stop("p and q must be positive integers")
  if (p > q) stop("p must be less than or equal to q")
  if (p == q) {
    return(diag(p))
  }
  prox_forward <- diag(p)
  for (i in p:(q - 1)) {
    if (i == 1) {
      prox <- matrix(1, 2, 1)
    } else {
      prox <- matrix(0, i + 1, i)
      diag(prox) <- (i - 0:(i - 1)) / i
      diag(prox[-1, ]) <- (1:i) / i
    }
    prox_forward <- prox %*% prox_forward
  }
  prox_forward
}

#' Proximal Operator (Reverse)
#'
#' Constructs a transformation matrix to map the coefficients of Bernstein
#' polynomial basis functions from one dimension to another, ensuring
#' consistency with the properties of Bernstein polynomial bases. Note that
#' the original dimension must be greater than or equal to the target dimension.
#'
#' @param q Integer. The number of basis functions in the original dimension.
#' @param p Integer. The number of basis functions in the target dimension.
#'
#' @return A matrix representing the transformation from the original dimension
#'         to the target dimension.
#'
#' @details This function iteratively constructs the transformation matrix by
#'          expanding the basis functions from \code{p} to \code{q}, while
#'          preserving the mathematical properties of Bernstein polynomials.
#'
#' @seealso \code{\link{prox_forward}}
#' @keywords internal
#' @importFrom MASS ginv
prox_reverse <- function(q, p) {
  if (
    !is.numeric(p) || !is.numeric(q) || p != as.integer(p) || q != as.integer(q)
  ) {
    stop("p and q must be integers")
  }
  if (p <= 0 || q <= 0) stop("p and q must be positive integers")
  if (p > q) stop("p must be less than or equal to q")
  if (p == q) {
    return(diag(p))
  }
  prox_forward <- prox_forward(p, q)
  ginv(prox_forward)
}
