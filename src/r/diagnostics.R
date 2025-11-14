suppressPackageStartupMessages(library(tseries))
suppressPackageStartupMessages(library(dplyr))

run_diagnostics <- function(df, column = "Close", lag = 20, auto_fix = TRUE) {
   if (!(column %in% names(df))) {
    stop(paste("Missing required column:", column))
  }


    df[[column]] <- as.numeric(df[[column]])
  df <- df %>% filter(!is.na(.data[[column]]))

  adf_res <- tryCatch({
    adf.test(df$Close, alternative = "stationary")
  }, error = function(e) NA)

  box_res <- tryCatch({
    Box.test(df$Close, lag = 20, type = "Ljung-Box")
  }, error = function(e) NA)

  adf_p <- if (!is.null(adf_res) && "p.value" %in% names(adf_res)) adf_res$p.value else NA_real_
  box_p <- if (!is.null(box_res) && "p.value" %in% names(box_res)) box_res$p.value else NA_real_



    stationary <- !is.na(adf_p) && length(adf_p) == 1 && adf_p < 0.05

  if (auto_fix && !stationary) {
    message("[Diagnostics] Series appears non-stationary → applying differencing...")

    df <- df %>%
      mutate(Stationary_Close = c(NA, diff(log1p(.data[[column]])))) %>%
      filter(!is.na(Stationary_Close))

    stationary <- TRUE
    column <- "Stationary_Close"
  }

  list(
    df = df,
    column_used = column,
    adf_pvalue = adf_p,
    adf_stationary = stationary,
    box_pvalue = box_p,
    box_autocorrelated = if (!is.na(box_p)) box_p < 0.05 else NA
  )
}
