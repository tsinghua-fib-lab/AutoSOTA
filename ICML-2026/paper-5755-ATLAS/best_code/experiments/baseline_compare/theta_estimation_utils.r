# Theta Estimation Utilities for p-IRT
# Provides mirt-based theta estimation for use in p-IRT scripts

# Load required library
if (!require(mirt, quietly = TRUE)) {
  stop("mirt package required. Install with: install.packages('mirt')")
}

#' Estimate theta using mirt EAP method
#' 
#' @param responses Numeric vector of binary responses (0/1)
#' @param item_params Data frame with columns: item_id, a1, d, g (mirt format)
#' @return List with theta (ability estimate) and se (standard error)
estimate_theta_mirt <- function(responses, item_params) {
  # Validate inputs
  if (length(responses) == 0 || nrow(item_params) == 0) {
    warning("Empty responses or item parameters")
    return(list(theta = 0, se = 1))
  }
  
  # Check if responses match item parameters
  if (length(responses) != nrow(item_params)) {
    warning(paste0("Mismatch: ", length(responses), " responses vs ", 
                   nrow(item_params), " item parameters"))
    # Use only matching items
    n_items <- min(length(responses), nrow(item_params))
    responses <- responses[1:n_items]
    item_params <- item_params[1:n_items, ]
  }
  
  # Remove NA responses
  valid_idx <- !is.na(responses)
  if (sum(valid_idx) == 0) {
    warning("All responses are NA")
    return(list(theta = 0, se = 1))
  }
  
  responses <- responses[valid_idx]
  item_params <- item_params[valid_idx, ]
  
  n_items <- length(responses)
  
  # Create dummy response matrix with multiple rows to avoid mirt's variance check
  # We need at least 2 rows with different response patterns
  dummy_data <- rbind(
    rep(0, n_items),  # All incorrect
    rep(1, n_items)   # All correct
  )
  dummy_df <- as.data.frame(dummy_data)
  colnames(dummy_df) <- paste0("Item", 1:n_items)
  
  # Get parameter template from dummy data
  pars <- mirt(dummy_df, 1, itemtype = "3PL", pars = 'values', verbose = FALSE)
  
  # Set fixed item parameters (a1, d, g format)
  for (i in 1:nrow(item_params)) {
    item_name <- paste0("Item", i)
    item_rows <- which(pars$item == item_name)
    
    # Set a1 (discrimination)
    a1_idx <- item_rows[pars$name[item_rows] == "a1"]
    if (length(a1_idx) > 0) {
      pars$value[a1_idx] <- item_params$a1[i]
      pars$est[a1_idx] <- FALSE
    }
    
    # Set d (difficulty)
    d_idx <- item_rows[pars$name[item_rows] == "d"]
    if (length(d_idx) > 0) {
      pars$value[d_idx] <- item_params$d[i]
      pars$est[d_idx] <- FALSE
    }
    
    # Set g (guessing)
    g_idx <- item_rows[pars$name[item_rows] == "g"]
    if (length(g_idx) > 0) {
      pars$value[g_idx] <- item_params$g[i]
      pars$est[g_idx] <- FALSE
    }
  }
  
  # Build model with fixed parameters
  mod_fixed <- mirt(dummy_df, 1, itemtype = "3PL", pars = pars,
                    technical = list(NCYCLES = 1000),
                    verbose = FALSE)
  
  # Score the actual response pattern
  response_pattern <- matrix(responses, nrow = 1)
  theta_scores <- fscores(
    mod_fixed,
    method = "EAP",
    response.pattern = response_pattern,
    full.scores.SE = TRUE,
    quadpts = 61,
    verbose = FALSE
  )
  
  # Extract theta and SE
  theta <- theta_scores[1, 1]
  se <- theta_scores[1, 2]
  
  return(list(theta = theta, se = se))
}

#' Estimate theta using mirt EAP method (legacy function for backward compatibility)
#' 
#' @param responses Numeric vector of binary responses (0/1)
#' @param item_params Data frame with columns: item_id, a, b, c
#' @return List with theta (ability estimate) and se (standard error)
estimate_theta_mirt_abc <- function(responses, item_params) {
  # Convert a, b, c to a1, d, g
  item_params_converted <- data.frame(
    item_id = item_params$item_id,
    a1 = item_params$a,
    d = -item_params$a * item_params$b,
    g = item_params$c,
    stringsAsFactors = FALSE
  )
  
  # Call the main function
  return(estimate_theta_mirt(responses, item_params_converted))
}
