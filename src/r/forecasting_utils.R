config_path <- file.path(dirname(getwd()), "src", "config", "paths.R")
if (!file.exists(config_path)) {
  # Fallback — if executed from main project root
  config_path <- file.path("src", "config", "paths.R")
}

if (file.exists(config_path)) {
  source(config_path, encoding = "UTF-8")
} else {
  stop("[PathError] Could not locate paths.R for forecasting_utils.R")
}

cat("[Config] Paths loaded successfully in forecasting_utils.R\n")

# --- Now your forecasting helper functions below ---
# e.g.
run_combined_forecast <- function(df, horizon = 30) {
  cat("[ARIMA] Fitting ARIMA model...\n")

  # Upewnij się, że dane są poprawne
  if (!"Close" %in% names(df)) stop("[Forecast] Missing 'Close' column.")
  df <- df[!is.na(df$Close), ]
  if (nrow(df) < 10) stop("[Forecast] Not enough observations for forecasting.")

  ts_data <- ts(df$Close, frequency = 365)

  # Fit ARIMA
  fit_arima <- tryCatch({
    auto.arima(ts_data)
  }, error = function(e) {
    warning(paste("[ARIMA] Model fitting failed:", e$message))
    NULL
  })

  # Fit STLF
  cat("[STLF] Fitting STLF model...\n")
  fit_stlf <- tryCatch({
    stlf(ts_data, h = horizon)
  }, error = function(e) {
    warning(paste("[STLF] Model fitting failed:", e$message))
    NULL
  })

  # Extract forecasts safely
  arima_forecast <- if (!is.null(fit_arima)) {
    forecast(fit_arima, h = horizon)$mean
  } else rep(NA, horizon)

  stlf_forecast <- if (!is.null(fit_stlf)) {
    fit_stlf$mean
  } else rep(NA, horizon)

  # Ensure numeric consistency
  arima_forecast <- as.numeric(arima_forecast)
  stlf_forecast <- as.numeric(stlf_forecast)

  # Defensive check
  if (length(arima_forecast) == 0 || all(is.na(arima_forecast))) {
    warning("[Forecast] ARIMA forecast failed, filling with NAs.")
    arima_forecast <- rep(NA, horizon)
  }
  if (length(stlf_forecast) == 0 || all(is.na(stlf_forecast))) {
    warning("[Forecast] STLF forecast failed, filling with NAs.")
    stlf_forecast <- rep(NA, horizon)
  }

  # Combined forecast
  combined_forecast <- data.frame(
    Date = seq(max(df$Date, na.rm = TRUE) + 1, by = "day", length.out = horizon),
    ARIMA = arima_forecast,
    STLF = stlf_forecast,
    Combined = rowMeans(cbind(arima_forecast, stlf_forecast), na.rm = TRUE)
  )

  cat("[Forecast] Combined ARIMA + STLF forecast generated.\n")

  list(
    arima = fit_arima,
    stlf = fit_stlf,
    combined = combined_forecast,
    future = combined_forecast
  )
}
