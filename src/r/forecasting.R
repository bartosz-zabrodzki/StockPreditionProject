suppressPackageStartupMessages({
  library(ggplot2)
  library(forecast)
})

source("src/r/read_data.R")
source("src/r/forecasting_utils.R")

DATA_PATH <- "src/data/data_cache/AAPL_1d.csv"
OUTPUT_PATH <- "src/data/data_processed/forecasts/AAPL_forecast.csv"

df <- read_data(DATA_PATH)
res <- run_combined_forecast(df, horizon = 30)

dir.create(dirname(OUTPUT_PATH), recursive = TRUE, showWarnings = FALSE)
write.csv(res$combined, OUTPUT_PATH, row.names = FALSE)
cat("Forecasts saved →", OUTPUT_PATH, "\n")

autoplot(res$arima$forecast) + ggtitle("AAPL ARIMA Forecast")
autoplot(res$ets$forecast) + ggtitle("AAPL ETS Forecast")
