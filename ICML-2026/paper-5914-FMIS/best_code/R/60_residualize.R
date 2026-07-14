# Residualize values so only (FWL orthogonalized) 'variables' in X are kept
update_fwl <- function(X, y, variables, rm = NULL) {
  if (!any(variables == 0)) {
    if (is.null(rm)) {
      Q_fwl <- qr.Q(qr(X[, -variables, drop = FALSE]))
      y <- y - Q_fwl %*% crossprod(Q_fwl, y)
      X <- X[, variables, drop = FALSE] -
        Q_fwl %*%
          crossprod(Q_fwl, X[, variables, drop = FALSE])
    } else {
      Q_fwl <- qr.Q(qr(X[-rm, -variables, drop = FALSE]))
      y[-rm] <- y[-rm] - Q_fwl %*% crossprod(Q_fwl, y[-rm])
      X[-rm, variables] <- X[-rm, variables, drop = FALSE] -
        Q_fwl %*%
          crossprod(Q_fwl, X[-rm, variables, drop = FALSE])
      X <- X[, variables]
    }
  }
  return(list("y" = y, "X" = X))
}

# Residualize using a partial linear model
update_plm <- function(
  X,
  y,
  variables,
  method = c("gbm", "randomForest", "xgboost"),
  params = list()
) {
  message("Interface for PLM residualization is not yet fully implemented.")
  # Input validation
  method <- match.arg(method)
  stopifnot(has_package(method))
  y <- as.vector(y)
  X <- as.matrix(X[, variables])
  Z <- as.matrix(X[, -variables])
  n <- length(y)
  p <- ncol(Z)
  if (NROW(Z) != n) {
    stop("Dimensions of y, X, and Z must match")
  }

  # Default parameters
  defaults <- list(
    randomForest = list(
      ntree = 500,
      mtry = max(floor(p / 3), 1),
      nodesize = 5
    ),
    gbm = list(
      n.trees = 500,
      depth = 3,
      shrinkage = 0.01,
      bag.frac = 0.5
    ),
    xgboost = list(
      nrounds = 500,
      max_depth = 6,
      eta = 0.05,
      subsample = 0.8,
      colsample = 0.8
    )
  )
  parameters <- modifyList(defaults[[method]], params)

  # Flexible fitting function
  fit_np <- function(response, Z_mat) {
    Z_df <- as.data.frame(Z_mat)
    d_mt <- cbind(y = response, Z_df)
    switch(
      method,
      gbm = {
        fit <- gbm::gbm(
          y ~ .,
          data = d_mt,
          distribution = "gaussian",
          n.trees = parameters$n.trees,
          interaction.depth = parameters$depth,
          shrinkage = parameters$shrinkage,
          bag.fraction = parameters$bag.frac,
          verbose = FALSE
        )
        predict(fit, d_mt, n.trees = parameters$n.trees)
      },
      randomForest = {
        fit <- randomForest::randomForest(
          x = Z_df,
          y = response,
          ntree = parameters$ntree,
          mtry = parameters$mtry,
          nodesize = parameters$nodesize
        )
        predict(fit, Z_df)
      },
      xgboost = {
        dtrain <- xgboost::xgb.DMatrix(data = Z_mat, label = response)
        fit <- xgboost::xgb.train(
          params = list(
            objective = "reg:squarederror",
            max_depth = parameters$max_depth,
            eta = parameters$eta,
            subsample = parameters$subsample,
            colsample_bytree = parameters$colsample
          ),
          data = dtrain,
          nrounds = parameters$nrounds,
          verbose = 0
        )
        predict(fit, dtrain)
      }
    )
  }

  # Robinson's (1988) double residual procedure
  y_resid <- y - fit_np(y, Z)
  X_resid <- apply(X, 2, function(x) x - fit_np(x, Z))

  return(list("y" = y_resid, "X" = X_resid))
}
