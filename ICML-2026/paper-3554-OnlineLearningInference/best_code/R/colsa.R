#' @title Alias for Survival Function
#' @description This function creates an alias for the `Surv` function from the
#' `survival` package, allowing it to be used without explicitly referencing
#' the package namespace.
#'
#' @param time The follow-up time for the subject.
#' @param time2 The ending time for the interval (if applicable).
#' @param event The status indicator, normally 0=alive, 1=dead.
#'              Other values are allowed.
#' @param type Character string specifying the type of censoring.
#'             Possible values are "right", "left", "interval", or "counting".
#' @param origin The origin for counting time, used when `type = "counting"`.
#'
#' @return A `Surv` object representing the survival data.
#'
#' @importFrom survival Surv
#' @seealso \code{\link[survival]{Surv}}
#' @export
Surv <- survival::Surv # nolint: object_name_linter.

#' Collaborative Inference for Cox Proportional Hazards Model in Distributed
#' Data
#'
#' @description Performs collaborative inference for the Cox proportional
#' hazards model, tailored for distributed data environments.
#'
#' @param formula A formula object specifying the model. The response must be
#'   a survival object created using the `Surv` function from the `survival`
#'   package.
#' @param data A data frame containing the variables in the model.
#' @param n_basis An integer specifying the number of basis functions to use.
#' @param boundary A numeric vector specifying the boundary for the basis
#'   functions. It should be a two-element vector with the lower and upper
#'   bounds.
#' @param scale A numeric value specifying the scaling factor for the number of
#'   pre-estimation basis functions. Default is 2.0.
#' @param init A character string specifying the initialization method.
#'   \code{"zero"} (default) uses zero initialization for fast computation.
#'   \code{"flexsurv"} uses \code{flexsurvspline} for better initial values
#'   but slower computation.
#' @param ... Additional arguments passed to the optimization function.
#'
#' @return An object of class `"colsa"` containing the following components:
#'   \item{logLik}{The log-likelihood of the fitted model.}
#'   \item{theta}{The estimated model parameters.}
#'   \item{hessian}{The Hessian of the objective function at the solution.}
#'   \item{n_basis}{The number of basis functions used.}
#'   \item{n_features}{The number of features in the model.}
#'   \item{n_samples}{The number of samples in the dataset.}
#'   \item{formula}{The formula used to fit the model.}
#'   \item{boundary}{The boundary for the basis functions.}
#'   \item{scale}{The scaling factor for the number of pre-estimation basis
#'                functions.}
#'   \item{call}{The matched call.}
#'
#' @details This function employs the `nlm` optimization method to estimate
#'   the model parameters. If the optimization fails to converge, an error is
#'   raised. During the pre-estimation stage, parameters are projected to
#'   mitigate bias introduced in the early stage.
#'
#' @examples
#' formula <- Surv(time, status) ~ x1 + x2 + x31 + x42 + x43 + x44
#' boundary <- c(0, max(sim$time))
#' df_sub <- sim[sim$group == 1, , drop = FALSE]
#' fit <- colsa(formula, df_sub, n_basis = 3, boundary = boundary)
#' for (batch in 2:6) {
#'   df_sub <- sim[sim$group == batch, , drop = FALSE]
#'   fit <- update(fit, df_sub, n_basis = "auto")
#' }
#' summary(fit)
#'
#' @seealso \code{\link{update.colsa}}, \code{\link{summary.colsa}}.
#'
#' @importFrom stats model.frame model.response model.matrix nlm
#'
#' @export
colsa <- function(
    formula, data, n_basis, boundary, scale = 2.0, init = "zero", ...) {
  # Input validation
  if (!inherits(formula, "formula")) stop("formula must be a formula object")
  if (!inherits(data, "data.frame")) stop("data must be a data frame")
  if (!is.numeric(n_basis) || length(n_basis) != 1 || n_basis <= 0) {
    stop("n_basis must be a positive integer")
  }
  if (
    !is.numeric(boundary) || length(boundary) != 2 || boundary[1] >= boundary[2]
  ) {
    stop("boundary must be a numeric vector of length 2 with increasing values")
  }
  if (!is.numeric(scale) || length(scale) != 1 || scale <= 0) {
    stop("scale must be a positive numeric value")
  }
  if (!init %in% c("zero", "flexsurv")) {
    stop("init must be 'zero' or 'flexsurv'")
  }

  # Prepare model frame and extract response and predictors
  mf <- stats::model.frame(formula, data)
  y <- stats::model.response(mf)
  time <- y[, 1, drop = FALSE]
  status <- y[, 2, drop = FALSE]
  x <- stats::model.matrix(formula, mf)[, -1, drop = FALSE]

  # Initialize parameters
  n_samples <- nrow(x)
  n_features <- ncol(x)
  n_basis_pre <- round(n_basis * scale)
  n_parameters <- n_basis_pre + n_features

  # Initialize the Parameters

  if (init == "flexsurv") {
    init_fit <- flexsurv::flexsurvspline(formula, data = data, k = n_basis - 1)
    bh <- flexsurv::hsurvspline(
      time, init_fit$coefficients[seq_len(n_basis + 1)],
      knots = init_fit$knots
    )
    b <- if (n_basis == 1) {
      matrix(1, nrow = length(time), ncol = 1)
    } else {
      splines2::bpoly(
        time,
        degree = n_basis - 1, intercept = TRUE,
        Boundary.knots = boundary
      )
    }
    alpha_init <- stats::lm.fit(b, log(bh))$coefficients
    beta_init <- init_fit$coefficients[-seq_len(n_basis + 1)]
    theta_init <- c(alpha_init, beta_init)
  } else {
    theta_init <- numeric(n_basis + n_features)
  }

  theta <- numeric(n_parameters)
  hessian <- matrix(0, nrow = n_parameters, ncol = n_parameters)

  # Query stage
  res <- nlm(
    f = objective, p = theta_init, fscale = n_samples,
    time = time, status = status, x = x, boundary = boundary,
    theta = theta, hessian_prev = hessian
  )
  if (!res$code %in% c(1, 2)) {
    stop("Optimization failed to converge")
  }

  # Pre-estimation stage
  prox <- prox_forward(n_basis, n_basis_pre)
  prox <- rbind(
    cbind(prox, matrix(0, nrow(prox), n_features)),
    cbind(matrix(0, n_features, ncol(prox)), diag(n_features))
  )
  theta <- as.vector(prox %*% res$estimate)
  obj <- objective(theta, time, status, x, boundary, theta, hessian)
  names(theta) <- c(paste0("Basis", seq_len(n_basis_pre)), colnames(x))

  object <- list(
    logLik = as.numeric(-obj),
    theta = theta,
    hessian = attr(obj, "hessian"),
    n_basis = c(n_basis),
    n_features = n_features,
    n_samples = n_samples,
    formula = formula,
    boundary = boundary,
    scale = scale,
    call = match.call()
  )
  class(object) <- "colsa"
  object
}

#' Update a COLSA Model with New Data
#'
#' This function updates a COLSA model  with new data, allowing for dynamic
#' adjustment of the model parameters and basis functions.
#'
#' @param object An object of class \code{"colsa"} representing the fitted
#'   COLSA model.
#' @param newdata A \code{data.frame} containing the new data to update the
#'   model. It must include the variables specified in the original model
#'   formula.
#' @param n_basis Either a numeric value specifying the number of basis
#'   functions or \code{"auto"} to automatically determine the number of basis
#'   functions based on the sample size and scaling parameters.
#'   Defaults to \code{"auto"}, which means the number of basis functions will
#'   be determined by n_basis = round(alpha * n_samples^nu).
#' @param alpha A positive numeric value controlling the growth rate of the
#'   number of basis functions when \code{n_basis = "auto"}. Defaults to 1.0.
#' @param nu A positive numeric value controlling the scaling exponent for
#'   determining the number of basis functions when \code{n_basis = "auto"}.
#'   Defaults to 0.2.
#' @param ... Additional arguments passed to the optimization function.
#'
#' @details The function updates the COLSA model by incorporating new data and
#'   adjusting the number of basis functions dynamically. It ensures that the
#'   model parameters and Hessian matrix are updated accordingly. The
#'   optimization process is performed using the \code{nlm} method to
#'   maximize the log-likelihood of the updated model.
#'
#' @return An updated object of class \code{"colsa"} with the following
#' components:
#' \itemize{
#'   \item \code{logLik}: The log-likelihood of the updated model.
#'   \item \code{theta}: The updated parameter estimates.
#'   \item \code{hessian}: The updated Hessian matrix.
#'   \item \code{n_samples}: The updated total number of samples.
#'   \item \code{n_basis}: A vector tracking the number of basis functions used
#'         at each update step.
#'   \item \code{call}: The matched call of the update function.
#' }
#'
#' @importFrom stats update
#' @importFrom stats model.frame model.response model.matrix nlm
#' @importFrom utils tail
#'
#' @export
update.colsa <- function(
    object, newdata, n_basis = "auto", alpha = 1.0, nu = 0.2, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  if (!inherits(newdata, "data.frame")) stop("newdata must be a data frame")
  if (!is.numeric(n_basis) && n_basis != "auto") {
    stop("n_basis must be numeric or 'auto'")
  }
  if (!is.numeric(alpha) || length(alpha) != 1 || alpha <= 0) {
    stop("alpha must be a positive numeric value")
  }
  if (!is.numeric(nu) || length(nu) != 1 || nu <= 0) {
    stop("nu must be a positive numeric value")
  }

  # Retrieve Parameters
  theta <- object$theta
  hessian <- object$hessian

  n_samples <- object$n_samples
  n_features <- object$n_features
  n_basis_last <- tail(object$n_basis, 1)
  n_basis_pre_last <- length(theta) - n_features

  formula <- object$formula
  boundary <- object$boundary
  scale <- object$scale

  mf <- stats::model.frame(formula, newdata)
  y <- stats::model.response(mf)
  time <- y[, 1, drop = FALSE]
  status <- y[, 2, drop = FALSE]
  x <- stats::model.matrix(formula, mf)[, -1, drop = FALSE]

  # Update Parameters
  n_samples <- n_samples + nrow(x)
  if (n_basis == "auto") {
    n_basis <- max(n_basis_last, round(alpha * n_samples^nu))
  } else if (is.numeric(n_basis)) {
    n_basis <- max(n_basis_last, n_basis)
  } else {
    stop("n_basis must be numeric or 'auto'")
  }
  object$n_samples <- n_samples
  object$n_basis <- c(object$n_basis, n_basis)

  # Query Stage
  n_basis_pre <- round(n_basis * scale)
  if (n_basis_pre > n_basis_pre_last) {
    prox <- prox_forward(n_basis_pre_last, n_basis_pre)
    prox <- rbind(
      cbind(prox, matrix(0, nrow(prox), n_features)),
      cbind(matrix(0, n_features, ncol(prox)), diag(n_features))
    )
    theta <- as.vector(prox %*% theta)
    prox_inv <- MASS::ginv(prox)
    hessian <- t(prox_inv) %*% hessian %*% prox_inv
  }

  prox <- prox_forward(n_basis, n_basis_pre)
  prox <- rbind(
    cbind(prox, matrix(0, nrow(prox), n_features)),
    cbind(matrix(0, n_features, ncol(prox)), diag(n_features))
  )
  theta_init <- qr.solve(prox, theta)

  res <- nlm(
    f = objective, p = theta_init, fscale = n_samples,
    time = time, status = status, x = x, boundary = boundary,
    theta = theta, hessian_prev = hessian
  )
  if (!res$code %in% c(1, 2)) {
    stop("Optimization failed to converge")
  }

  # Pre-estimation Stage
  prox <- prox_forward(n_basis, n_basis_pre)
  prox <- rbind(
    cbind(prox, matrix(0, nrow(prox), n_features)),
    cbind(matrix(0, n_features, ncol(prox)), diag(n_features))
  )
  theta <- as.vector(prox %*% res$estimate)
  obj <- objective(theta, time, status, x, boundary, theta, hessian)
  names(theta) <- c(paste0("Basis", seq_len(n_basis_pre)), colnames(x))

  object$logLik <- as.numeric(-obj)
  object$theta <- theta
  object$hessian <- attr(obj, "hessian")
  object$call <- match.call()
  object
}

#' Compute Log-Likelihood for 'colsa' Objects
#'
#' This function retrieves the log-likelihood value from an object of class
#' 'colsa'.
#'
#' @param object An object of class 'colsa'.
#' @param ... Additional arguments (currently unused).
#'
#' @return The log-likelihood value stored in the 'colsa' object.
#'
#' @importFrom stats logLik
#'
#' @export
logLik.colsa <- function(object, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  object$logLik
}

#' Extract Coefficients from a 'colsa' Object
#'
#' This function retrieves the coefficients from a fitted 'colsa' object.
#'
#' @param object An object of class 'colsa'. The fitted model object from which
#'   coefficients are to be extracted.
#' @param ... Additional arguments (currently not used).
#'
#' @return A numeric vector containing the coefficients extracted from the
#'   'colsa' object.
#'
#' @details The function verifies that the input object is of class 'colsa'.
#'   It then calculates the coefficients by applying a transformation matrix
#'   (`prox`) to the `theta` parameter of the object. The transformation matrix
#'   accounts for the number of basis functions and features in the model.
#'
#' @importFrom stats coef
#' @importFrom utils tail
#'
#' @export
coef.colsa <- function(object, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  n_basis <- length(object$theta) - object$n_features
  object$theta[-seq_len(n_basis)]
}

#' Compute Variance-Covariance Matrix for 'colsa' Objects
#'
#' This function computes the variance-covariance matrix for objects of class
#' \code{"colsa"}.
#'
#' @param object An object of class \code{"colsa"}.
#' @param ... Additional arguments (currently not used).
#'
#' @return A matrix representing the variance-covariance matrix of the model
#'   parameters.
#'
#' @importFrom stats vcov
#' @importFrom utils tail
#'
#' @export
vcov.colsa <- function(object, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  n_basis_pre <- length(object$theta) - object$n_features
  n_basis <- tail(object$n_basis, 1)

  hessian <- object$hessian
  prox <- prox_forward(n_basis, n_basis_pre)
  prox <- rbind(
    cbind(prox, matrix(0, nrow(prox), object$n_features)),
    cbind(matrix(0, object$n_features, ncol(prox)), diag(object$n_features))
  )
  hess <- crossprod(prox, hessian) %*% prox

  idx <- seq_len(n_basis)
  hess_alpha <- hess[idx, idx, drop = FALSE]
  hess_beta <- hess[-idx, -idx, drop = FALSE]
  hess_gb <- hess[idx, -idx, drop = FALSE]
  hess_bg <- hess[-idx, idx, drop = FALSE]

  # Compute the marginal variance-covariance matrix for coefficients
  # Use Cholesky decomposition for faster matrix inversion
  chol_alpha <- tryCatch(
    chol(hess_alpha),
    error = function(e) NULL
  )
  if (!is.null(chol_alpha)) {
    hess_alpha_inv <- chol2inv(chol_alpha)
  } else {
    hess_alpha_inv <- solve(hess_alpha)
  }

  hess_prox <- hess_beta - hess_bg %*% hess_alpha_inv %*% hess_gb

  chol_prox <- tryCatch(
    chol(hess_prox),
    error = function(e) NULL
  )
  if (!is.null(chol_prox)) {
    chol2inv(chol_prox)
  } else {
    solve(hess_prox)
  }
}


#' Calculate the Akaike Information Criterion (AIC) for a 'colsa' object
#'
#' This function computes the AIC for a given 'colsa' object. The AIC is a
#' measure of the relative quality of a statistical model for a given dataset.
#' It is calculated as: \eqn{-2 * logLik + 2 * k}, where \code{k} is the number
#' of parameters in the model.
#'
#' @param object An object of class \code{"colsa"} for which the AIC is to be
#'   calculated.
#' @param ... Additional arguments (currently not used).
#'
#' @return A numeric value representing the AIC of the model.
#'
#' @importFrom stats AIC
#'
#' @seealso \code{\link{logLik.colsa}}, \code{\link{BIC.colsa}}
#'
#' @export
AIC.colsa <- function(object, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  n_parameters <- tail(object$n_basis, 1) + object$n_features
  -2 * logLik(object) + 2 * n_parameters
}

#' Calculate Bayesian Information Criterion (BIC) for a 'colsa' Object
#'
#' This function computes the Bayesian Information Criterion (BIC) for a fitted
#' 'colsa' model object. The BIC is a model selection criterion that balances
#' model fit and complexity.
#'
#' @param object An object of class 'colsa'. This is the fitted model for which
#'   the BIC is to be calculated.
#' @param ... Additional arguments (currently not used).
#'
#' @return A numeric value representing the BIC of the model.
#'
#' @details The BIC is calculated as:
#'   \deqn{-2 \times \log(\text{Likelihood}) + k \times \log(n)}
#'   where \eqn{k} is the number of parameters in the model,
#'   and \eqn{n} is the number of samples in the dataset.
#'
#' @seealso \code{\link{logLik.colsa}}, \code{\link{AIC.colsa}}
#'
#' @importFrom stats BIC
#'
#' @export
BIC.colsa <- function(object, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  n_parameters <- tail(object$n_basis, 1) + object$n_features
  -2 * logLik(object) + n_parameters * log(object$n_samples)
}

#' Summarize a 'colsa' Object
#'
#' This function provides a summary for objects of class \code{"colsa"}.
#'
#' @param object An object of class \code{"colsa"}.
#' @param conf.int A numeric value between 0 and 1 specifying the confidence
#'   level for the confidence intervals. Default is 0.95.
#' @param ... Additional arguments (currently not used).
#'
#' @return A list of class \code{"summary.colsa"} containing the following
#'   components:
#' \item{call}{The function call that created the \code{"colsa"} object.}
#' \item{coefficients}{A matrix with columns for the coefficients, exponentiated
#'   coefficients, standard errors, z-scores, and p-values.}
#' \item{conf.int}{A matrix with columns for the exponentiated coefficients,
#'   their reciprocals, and the lower and upper confidence interval bounds.}
#' \item{n_basis}{The number of basis functions used in the model.}
#' \item{boundary}{The boundary information from the \code{"colsa"} object.}
#'
#' @details The function calculates the z-scores and p-values for the
#'   coefficients, as well as confidence intervals for the exponentiated
#'   coefficients.
#'
#' @importFrom stats qnorm pchisq
#'
#' @export
summary.colsa <- function(object, conf.int = 0.95, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  if (
    !is.numeric(conf.int) || length(conf.int) != 1 ||
      conf.int <= 0 || conf.int >= 1
  ) {
    stop("conf.int must be a numeric value between 0 and 1")
  }
  coef <- coef(object)
  vcov <- vcov(object)
  se <- sqrt(diag(vcov))
  z_scores <- coef / se
  p_values <- pchisq(z_scores^2, df = 1, lower.tail = FALSE)
  coef_matrix <- cbind(coef, exp(coef), se, z_scores, p_values)
  dimnames(coef_matrix) <- list(
    names(coef), c("coef", "exp(coef)", "se", "z", "p")
  )

  z <- qnorm((1 + conf.int) / 2)
  conf_int_matrix <- cbind(
    exp(coef), exp(-coef), exp(coef - z * se), exp(coef + z * se)
  )
  dimnames(conf_int_matrix) <- list(
    names(coef), c(
      "exp(coef)", "exp(-coef)",
      paste("lower .", round(100 * conf.int, 2), sep = ""),
      paste("upper .", round(100 * conf.int, 2), sep = "")
    )
  )

  summary_list <- list(
    call = object$call, coefficients = coef_matrix, conf.int = conf_int_matrix,
    n_basis = tail(object$n_basis, 1), boundary = object$boundary
  )
  class(summary_list) <- "summary.colsa"
  return(summary_list)
}

#' @importFrom stats printCoefmat
#'
#' @export
print.summary.colsa <- function(
    x, digits = max(3, getOption("digits") - 3),
    signif.stars = getOption("show.signif.stars"), ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nNumber of basis functions: ", x$n_basis, "\n\n")

  idx <- !grepl("^Basis", rownames(x$coefficients))
  printCoefmat(
    x$coefficients[idx, ],
    digits = digits, signif.stars = signif.stars,
    cs.ind = 1:3, tst.ind = 4, P.values = TRUE, has.Pvalue = TRUE
  )
  print(
    format(x$conf.int[idx, ], digits = digits),
    quote = FALSE
  )

  invisible(x)
}


#' Generic Function for Baseline Hazard
#'
#' This is a generic function to compute the baseline hazard for a given object.
#'
#' @param object An object for which the baseline hazard is to be computed.
#' @param ... Additional arguments passed to specific methods.
#'
#' @return A numeric vector representing the baseline hazard.
#' @export
basehaz <- function(object, ...) {
  UseMethod("basehaz")
}

#' Compute the Baseline Hazard Function with Confidence Intervals
#'
#' This function calculates the baseline hazard function for a fitted `colsa`
#' model and optionally provides confidence intervals for the estimated hazard.
#'
#' @param object An object of class `colsa`. This is the fitted model object.
#' @param newdata A numeric vector of time points at which to evaluate the
#'   baseline hazard. If `NULL`, the function generates a sequence of time
#'   points based on the model's boundary.
#' @param conf.int A numeric value between 0 and 1 specifying the confidence
#'   level for the confidence intervals. Default is 0.95.
#' @param ... Additional arguments (currently not used).
#'
#' @return A matrix with columns:
#'   \describe{
#'     \item{time}{The time points at which the baseline hazard is evaluated.}
#'     \item{basehaz}{The estimated baseline hazard at each time point.}
#'     \item{se}{The standard error of the baseline hazard estimates.}
#'     \item{lower}{The lower bound of the confidence interval.}
#'     \item{upper}{The upper bound of the confidence interval.}
#'   }
#'
#' @details The function uses the model's boundary and basis functions to
#'   compute the cumulative baseline hazard. Confidence intervals are derived
#'   using the variance-covariance matrix of the model parameters.
#'
#' @export
basehaz.colsa <- function(
    object, newdata = NULL, conf.int = 0.95, ...) {
  if (!inherits(object, "colsa")) stop("object must be of class 'colsa'")
  if (
    !is.numeric(conf.int) || length(conf.int) != 1 ||
      conf.int <= 0 || conf.int >= 1
  ) {
    stop("conf.int must be a numeric value between 0 and 1")
  }
  boundary <- object$boundary
  time <- if (!is.null(newdata)) {
    newdata
  } else {
    seq(boundary[1], boundary[2], length.out = 100)
  }

  coef <- object$theta
  n_basis <- length(coef) - object$n_features
  idx <- seq_len(n_basis)
  alpha <- coef[idx]

  cbh <- int_basehaz(time, alpha, n_basis, boundary)$cbh

  hess <- object$hessian
  hess_alpha <- hess[idx, idx, drop = FALSE]
  hess_beta <- hess[-idx, -idx, drop = FALSE]
  hess_bg <- hess[idx, -idx, drop = FALSE]
  hess_gb <- hess[-idx, idx, drop = FALSE]

  # Use Cholesky decomposition for faster matrix inversion
  chol_beta <- tryCatch(chol(hess_beta), error = function(e) NULL)
  hess_beta_inv <- if (!is.null(chol_beta)) {
    chol2inv(chol_beta)
  } else {
    solve(hess_beta)
  }

  hess_schur <- hess_alpha - hess_bg %*% hess_beta_inv %*% hess_gb
  chol_schur <- tryCatch(chol(hess_schur), error = function(e) NULL)
  vcov_alpha <- if (!is.null(chol_schur)) {
    chol2inv(chol_schur)
  } else {
    solve(hess_schur)
  }

  b <- if (n_basis == 1) {
    matrix(1, nrow = length(time), ncol = 1)
  } else {
    splines2::bpoly(
      time,
      degree = n_basis - 1, intercept = TRUE,
      Boundary.knots = boundary
    )
  }
  vcov_lp <- b %*% vcov_alpha %*% t(b)
  se_cbh <- sqrt(diag(vcov_lp) * cbh^2)
  z <- qnorm((1 + conf.int) / 2)
  conf_int <- cbind(
    time = time,
    basehaz = cbh,
    se = se_cbh,
    lower = cbh - z * se_cbh,
    upper = cbh + z * se_cbh
  )

  colnames(conf_int) <- c(
    "time", "basehaz", "se",
    paste0("lower .", round(100 * conf.int, 2)),
    paste0("upper .", round(100 * conf.int, 2))
  )
  as.data.frame(conf_int)
}
