suppressPackageStartupMessages({
  library(tseries)
  library(dplyr)
  source("src/config/paths.R", encoding = "UTF-8")

})

run_diagnostics <- function(df, column = "Close", lag = 20, auto_fix = TRUE) {
  if (is.null(df) || nrow(df) == 0) {
    stop("[Diagnostics] DataFrame is empty or NULL.")
  }

  if (!(column %in% names(df))) {
    stop(paste("[Diagnostics] Missing required column:", column))
  }

  # Konwersja do wartości numerycznych i czyszczenie braków
  df[[column]] <- suppressWarnings(as.numeric(df[[column]]))
  df <- df %>% filter(!is.na(.data[[column]]))

  if (nrow(df) < lag) {
    stop("[Diagnostics] Not enough data for statistical tests.")
  }

  # --- Test ADF ---
  adf_res <- tryCatch({
    adf.test(df[[column]], alternative = "stationary")
  }, error = function(e) {
    message("[Diagnostics] ADF test failed: ", e$message)
    NULL
  })

  # --- Test Ljung-Box ---
  box_res <- tryCatch({
    Box.test(df[[column]], lag = lag, type = "Ljung-Box")
  }, error = function(e) {
    message("[Diagnostics] Ljung-Box test failed: ", e$message)
    NULL
  })

  adf_p <- if (!is.null(adf_res) && "p.value" %in% names(adf_res)) adf_res$p.value else NA_real_
  box_p <- if (!is.null(box_res) && "p.value" %in% names(box_res)) box_res$p.value else NA_real_

  stationary <- !is.na(adf_p) && length(adf_p) == 1 && adf_p < 0.05

  # --- Automatyczne poprawianie niestacjonarności ---
  if (auto_fix && !stationary) {
    message("[Diagnostics] Series appears non-stationary → applying log-differencing...")

    df <- df %>%
      mutate(Stationary_Close = c(NA, diff(log1p(.data[[column]])))) %>%
      filter(!is.na(Stationary_Close))

    stationary <- TRUE
    column <- "Stationary_Close"
  }

  message("[Diagnostics] Completed successfully.")
  message("[Diagnostics] ADF p-value: ", round(adf_p, 5))
  message("[Diagnostics] Ljung-Box p-value: ", round(box_p, 5))

  list(
    df = df,
    column_used = column,
    adf_pvalue = adf_p,
    adf_stationary = stationary,
    box_pvalue = box_p,
    box_autocorrelated = if (!is.na(box_p)) box_p < 0.05 else NA
  )
}
