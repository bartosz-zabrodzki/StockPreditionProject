suppressPackageStartupMessages({
  library(forecast)
  library(ggplot2)
})

# --- Load paths configuration ---
source(file.path("src", "config", "paths.R"), encoding = "UTF-8")

# --- Import helper module ---
utils_path <- file.path(R_DIR, "forecasting_utils.R")
if (!file.exists(utils_path)) {
  stop("[Forecasting] ERROR: Cannot locate forecasting_utils.R")
}

cat("[Forecasting] Sourcing:", normalizePath(utils_path), "\n")
source(utils_path, encoding = "UTF-8", local = .GlobalEnv)

# --- Main forecasting wrapper ---
run_forecast <- function(df, horizon = 30) {
  cat("[Status] Running combined forecast...\n")
  result <- run_combined_forecast(df, horizon)
  cat("[Forecasting pipeline completed successfully.]\n")
  return(result)
}

