library(forecast)

run_forecast <- function(df) {
  ts_data <- ts(df$Close, frequency = 252)
  fit <- tryCatch({
    auto.arima(ts_data, seasonal = FALSE)
  }, error = function(e) {
    arima(ts_data, order = c(1,1,1))
  })
  future <- forecast(fit, h = 30)
  list(fit = fit, future = future)
}
