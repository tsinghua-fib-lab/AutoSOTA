#' Country latitude/longitude cache
#'
#' @description
#' Return the bundled country lookup table used by foundation-model place
#' context tokens. The cache contains ISO codes, country names and aliases, and
#' representative latitude/longitude values.
#'
#' @return A data frame with one row per country.
#' @export
country_latlong_cache <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      path <- system.file(
        "extdata",
        "country_latlong_cache.rds",
        package = "strategize",
        mustWork = FALSE
      )
      if (!nzchar(path) || !file.exists(path)) {
        stop(
          "Bundled country latitude/longitude cache is missing.",
          call. = FALSE
        )
      }
      cache <<- readRDS(path)
    }
    cache
  }
})

cs2step_missing_country_key <- "__missing_country__"

cs2step_context_ascii <- function(x) {
  x <- as.character(x %||% "")
  x <- iconv(x, from = "", to = "ASCII//TRANSLIT", sub = "")
  x[is.na(x)] <- ""
  x
}

cs2step_normalize_country_alias <- function(x) {
  x <- tolower(cs2step_context_ascii(x))
  x <- gsub("[^a-z0-9]+", " ", x)
  trimws(gsub("\\s+", " ", x))
}

cs2step_country_alias_frame <- function(cache = country_latlong_cache()) {
  if (!nrow(cache)) {
    return(data.frame(alias = character(0), country_key = character(0)))
  }
  rows <- lapply(seq_len(nrow(cache)), function(i) {
    aliases <- unique(c(
      cache$country_key[[i]],
      cache$iso3[[i]],
      cache$iso2[[i]],
      cache$name[[i]],
      unlist(cache$aliases[[i]] %||% character(0), use.names = FALSE)
    ))
    aliases <- aliases[!is.na(aliases) & nzchar(aliases)]
    data.frame(
      alias = cs2step_normalize_country_alias(aliases),
      country_key = cache$country_key[[i]],
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out <- out[nzchar(out$alias), , drop = FALSE]
  out[!duplicated(out$alias), , drop = FALSE]
}

cs2step_missing_country_record <- function() {
  list(
    experiment_country = cs2step_missing_country_key,
    country_key = cs2step_missing_country_key,
    country_iso3 = NA_character_,
    country_iso2 = NA_character_,
    country_name = "Missing country",
    country_latitude = NA_real_,
    country_longitude = NA_real_,
    country_present = FALSE
  )
}

cs2step_country_suggestion <- function(value, cache = country_latlong_cache()) {
  alias_frame <- cs2step_country_alias_frame(cache)
  if (!nrow(alias_frame)) {
    return(NULL)
  }
  value_norm <- cs2step_normalize_country_alias(value)
  if (!nzchar(value_norm)) {
    return(NULL)
  }
  distances <- utils::adist(value_norm, alias_frame$alias, partial = FALSE)
  best <- which.min(distances)
  if (!length(best) || is.na(best)) {
    return(NULL)
  }
  key <- alias_frame$country_key[[best]]
  row <- cache[match(key, cache$country_key), , drop = FALSE]
  if (!nrow(row) || is.na(row$name[[1L]]) || !nzchar(row$name[[1L]])) {
    return(NULL)
  }
  row$name[[1L]]
}

cs2step_normalize_country <- function(value, arg = "experiment_country") {
  if (is.null(value) || length(value) < 1L) {
    return(cs2step_missing_country_record())
  }
  if (length(value) != 1L) {
    stop(sprintf("'%s' must be length 1 when supplied.", arg), call. = FALSE)
  }
  value_chr <- as.character(value)
  if (is.na(value_chr) || !nzchar(trimws(value_chr))) {
    return(cs2step_missing_country_record())
  }
  if (identical(value_chr, cs2step_missing_country_key)) {
    return(cs2step_missing_country_record())
  }

  cache <- country_latlong_cache()
  value_trim <- trimws(value_chr)
  value_upper <- toupper(value_trim)
  idx <- match(value_upper, toupper(cache$iso3 %||% character(0)))
  if (is.na(idx)) {
    idx <- match(value_upper, toupper(cache$iso2 %||% character(0)))
  }
  if (is.na(idx)) {
    idx <- match(
      cs2step_normalize_country_alias(value_trim),
      cs2step_normalize_country_alias(cache$name %||% character(0))
    )
  }
  if (is.na(idx)) {
    alias_frame <- cs2step_country_alias_frame(cache)
    alias_norm <- cs2step_normalize_country_alias(value_trim)
    key <- alias_frame$country_key[match(alias_norm, alias_frame$alias)]
    if (!is.na(key)) {
      idx <- match(key, cache$country_key)
    }
  }
  if (is.na(idx) || idx < 1L || idx > nrow(cache)) {
    suggestion <- cs2step_country_suggestion(value_trim, cache)
    msg <- sprintf("Unknown %s '%s'.", arg, value_trim)
    if (!is.null(suggestion) && nzchar(suggestion)) {
      msg <- paste0(msg, sprintf(" Did you mean '%s'?", suggestion))
    }
    stop(msg, call. = FALSE)
  }
  row <- cache[idx, , drop = FALSE]
  list(
    experiment_country = as.character(row$country_key[[1L]]),
    country_key = as.character(row$country_key[[1L]]),
    country_iso3 = as.character(row$iso3[[1L]] %||% NA_character_),
    country_iso2 = as.character(row$iso2[[1L]] %||% NA_character_),
    country_name = as.character(row$name[[1L]] %||% NA_character_),
    country_latitude = as.numeric(row$latitude[[1L]] %||% NA_real_),
    country_longitude = as.numeric(row$longitude[[1L]] %||% NA_real_),
    country_present = TRUE
  )
}
