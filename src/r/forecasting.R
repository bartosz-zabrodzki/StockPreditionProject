suppressPackageStartupMessages({
  library(forecast)
  library(ggplot2)
  library(readr)
})

DATA_PATH <- file.path("src", "data", "data_cache", "AAPL_1d.csv")
MODEL_PATH <- file.path("src", "models", "arima_forecast.csv")
PLOT_PATH  <- file.path("src", "models", "arima_plot.png")
HORIZON    <- 30

if(!file.exists(DATA_PATH)){
  stop(paste("Data file not found:", DATA_PATH))
}

df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)


df$Close <- suppressWarnings(as.numeric(df$Close))

df <- df[!is.na(df$Close), ]

cat("Rows after cleaning:", nrow(df), "\n")

ts_data <- ts(df$Close, frequency = 252)

fit <- tryCatch({
  auto.arima(ts_data, seasonal = FALSE)
}, error = function(e) {
  warning("auto.arima failed: ", e$message)
  arima(ts_data, order = c(1,1,1))
})

model <- auto.arima(ts_data)

forecast_vals <- forecast(model, h = HORIZON)

pred_df <- data.frame(
  Day = 1:HORIZON,
  Forecast = forecast_vals$mean,
  Lower80 = forecast_vals$lower[, 1],
  Upper80 = forecast_vals$upper[, 1],
  Lower95 = forecast_vals$lower[, 2],
  Upper95 = forecast_vals$upper[, 2]
)

write_csv(pred_df, MODEL_PATH)
cat(paste("Saved forecast to:", MODEL_PATH, "\n"))


p <- autoplot(forecast_vals) +
  labs(
    title = "ARIMA Forecast for AAPL (Next 30 Days)",
    x = "Day",
    y = "Predicted Close Price"
  ) +
  theme_minimal()

ggsave(PLOT_PATH, p, width = 8, height = 4)
cat(paste("Saved plot to:", PLOT_PATH, "\n"))

cat("Done.\n")
