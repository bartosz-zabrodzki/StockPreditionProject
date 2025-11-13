source("src/r/read_data.R")
source("src/r/features.R")
source("src/r/forecasting.R")
source("src/r/diagnostics.R")

args <- commandArgs(trailingOnly = TRUE)
DATA_PATH <- ifelse(length(args) > 0, args[1], "src/data/data_cache/AAPL_1d.csv")

df <- read_data()
df <- generate_features(df)
diag <- run_diagnostics(df)
fcast <- run_forecast(df)

write.csv(df, "src/data/data_processed/AAPL_features.csv", row.names = FALSE)
write.csv(fcast$future, "src/models/arima_forecast.csv", row.names = FALSE)
