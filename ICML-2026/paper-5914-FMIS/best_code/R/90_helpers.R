# Check whether a package is available
has_package <- \(name, verbose = TRUE, purpose = "") {
  has_pkg <- requireNamespace(name, quietly = TRUE)
  if (!has_pkg && isTRUE(verbose)) {
    message("Package", name, "not available.", purpose)
  }
  has_pkg
}

# Extract (univariate) regressor and residuals from an lm
get_lm_xr <- function(model) {
  stopifnot(inherits(model, "lm"))
  if (!is.null(model$na.action)) {
    stop("lm contains missingness; refit with na.action = na.exclude/na.omit")
  }

  X <- model.matrix(model)
  if (
    !is.null(attr(terms(model), "intercept")) &&
      attr(terms(model), "intercept") != 0L
  ) {
    stop("Model must have no intercept (use y ~ 0 + x)")
  }

  if (NCOL(X) != 1L) {
    stop("Model must be univariate/residualized: exactly one regressor column")
  }

  x <- drop(X[, 1L, drop = FALSE])
  r <- residuals(model)

  stopifnot(length(x) == length(r))
  return(list(X = x, R = r))
}

# Jaccard similarity between two sets: |intersection| / |union|
jaccard_index <- function(set1, set2) {
  if (length(set1) == 0L && length(set2) == 0L) {
    return(1)
  }
  if (length(set1) == 0L || length(set2) == 0L) {
    return(0)
  }

  inter <- length(intersect(set1, set2))
  uni <- length(union(set1, set2))
  return(inter / uni)
}

# Compute Jaccard index between a list of consecutive sets
jaccard_consecutive <- function(sets) {
  if (length(sets) <= 1L) {
    return(numeric(0L))
  }
  vapply(
    seq_len(length(sets) - 1L),
    \(i) {
      jaccard_index(sets[[i]], sets[[i + 1L]])
    },
    numeric(1L)
  )
}

# Format a set of indices as a compact string
format_set <- function(s, max_show = 5L) {
  if (length(s) == 0L) {
    return("{}")
  }
  max_show <- as.integer(max_show)
  if (length(s) <= max_show) {
    return(paste0("{", paste(s, collapse = ", "), "}"))
  }
  return(paste0("{", paste(s[seq_len(max_show)], collapse = ", "), ", ...}"))
}

# Where are earlier sets not nested?
find_breaks <- function(sets) {
  n_sets <- length(sets)
  if (n_sets <= 1L) {
    return(integer(0L))
  }

  breaks <- integer(0L)
  for (j in seq_len(n_sets - 1L)) {
    a <- sort(sets[[j]])
    b <- sort(sets[[j + 1L]])

    i <- 1L
    k <- 1L
    is_subset <- TRUE
    while (i <= length(a) && k <= length(b)) {
      if (a[i] == b[k]) {
        i <- i + 1L
        k <- k + 1L
      } else if (a[i] > b[k]) {
        k <- k + 1L
      } else {
        is_subset <- FALSE
        break
      }
    }
    if (i <= length(a)) {
      is_subset <- FALSE
    }

    if (!is_subset) {
      breaks <- c(breaks, j)
    }
  }

  return(breaks)
}

# Top and bottom of a vector
headtail <- \(x) c(head = head(x), tail = tail(x))

# Estimate time from number of operations
to_time <- function(x, unit = c("ns", "us", "ms", "s"), digits = 2) {
  unit <- match.arg(unit)
  # Convert to seconds
  sec <- x *
    c(
      ns = 1e-9,
      us = 1e-6,
      ms = 1e-3,
      s = 1
    )[unit]

  scales <- c(
    s = 1,
    m = 60,
    h = 3600,
    d = 86400,
    y = 365.2425 * 86400
  )
  abs_sec <- abs(sec)

  idx <- findInterval(abs_sec, scales, rightmost.closed = TRUE)
  idx <- max(1L, idx)

  value <- sec / scales[idx]
  out_unit <- names(scales)[idx]

  paste0(round(value, digits), " ", out_unit)
}
