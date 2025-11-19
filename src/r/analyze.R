
suppressPackageStartupMessages({
Sys.setlocale("LC_ALL", "English_United States.utf8")
options(encoding = "UTF-8")
setwd(normalizePath("src", winslash = "/"))

  library(forecast)
  library(TTR)
  library(dplyr)
})


source("src/r/read_data.R")
source("src/r/features.R")
source("src/r/forecasting.R")
source("src/r/diagnostics.R")

args <- commandArgs(trailingOnly = TRUE)
DATA_PATH <- ifelse(length(args) > 0, args[1], "src/data/data_cache/AAPL_1d.csv")


df <- read_data(DATA_PATH)

diagnostics <- run_diagnostics(df, column = "Close", auto_fix = TRUE)


cat("ADF p-value:", diagnostics$adf_pvalue, "\n")
cat("Ljung-Box p-value:", diagnostics$box_pvalue, "\n")
cat("Using column:", diagnostics$column_used, "\n")

if (!diagnostics$adf_stationary) {
  cat("Warning: Series is non-stationary. Consider differencing or log-transform.\n")
}
df_clean <- diagnostics$df
df_features <- generate_features(df_clean)
forecast_result <- run_forecast(df_features)

dir.create("src/data/data_processed", showWarnings = FALSE, recursive = TRUE)
dir.create("src/models", showWarnings = FALSE, recursive = TRUE)

if (!dir.exists("src/data/data_processed/features")) {
  dir.create("src/data/data_processed/features", recursive = TRUE)
}
if (!dir.exists("src/data/data_processed/forecasts")) {
  dir.create("src/data/data_processed/forecasts", recursive = TRUE)
}

write.csv(df, "src/data/data_processed/features/AAPL_features.csv", row.names = FALSE)
write.csv(forecast_result$future, "src/data/data_processed/forecasts/AAPL_forecast.csv", row.names = FALSE)

cat("[Pipeline completed successfully]\n")
cat("Saved features → src/data/data_processed/features/AAPL_features.csv\n")
cat("Saved ARIMA forecast → src/data/data_processed/forecasts/AAPL_forecast.csv\n")