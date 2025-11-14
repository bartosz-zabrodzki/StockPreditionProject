suppressPackageStartupMessages({
  library(forecast)
  library(ggplot2)
})

run_forecast <- function(df, horizon = 30) {
  if (!"Close" %in% names(df)) {
    stop("DataFrame does not contain 'Close' column.")
  }

  ts_data <- ts(df$Close, frequency = 252)  # 252 = trading days per year

  cat("Fitting ARIMA model...\n")
  fit <- tryCatch({
    auto.arima(ts_data, seasonal = FALSE)
  }, error = function(e) {
    warning(paste("auto.arima failed:", e))
    arima(ts_data, order = c(1, 1, 1))
  })

  future <- forecast(fit, h = horizon)
  list(fit = fit, forecast = future)
}

run_ets <- function(df, horizon = 30) {
  if (!"Close" %in% names(df)) {
    stop("DataFrame does not contain 'Close' column.")
  }

  ts_data <- ts(df$Close, frequency = 252)

  cat("Fitting ETS model...\n")
  fit <- tryCatch({
    ets(ts_data)
  }, error = function(e) {
    warning(paste("ETS model failed:", e))
    NULL
  })

  if (is.null(fit)) {
    return(NULL)
  }

  future <- forecast(fit, h = horizon)
  list(fit = fit, forecast = future)
}

run_combined_forecast <- function(df, horizon = 30) {
  cat("Running ARIMA and ETS models...\n")

  arima_result <- run_forecast(df, horizon)
  ets_result <- run_ets(df, horizon)

  if (is.null(ets_result)) {
    cat("ETS model unavailable, returning ARIMA only.\n")
    return(arima_result)
  }

  combined <- data.frame(
    Date = seq(max(df$Date), by = "days", length.out = horizon),
    ARIMA = as.numeric(arima_result$forecast$mean),
    ETS = as.numeric(ets_result$forecast$mean)
  )

  combined$Average <- rowMeans(combined[, c("ARIMA", "ETS")], na.rm = TRUE)
  list(arima = arima_result, ets = ets_result, combined = combined)
}
