# ============================================================
# analyze.R — główny pipeline R
# ============================================================

suppressPackageStartupMessages({
  library(forecast)
  library(TTR)
  library(dplyr)
  library(quantmod)
})

# --- Wczytaj konfigurację ścieżek ---
config_path <- normalizePath(file.path("src", "config", "paths.R"), mustWork = FALSE)
if (!file.exists(config_path)) {
  config_path <- normalizePath(file.path(dirname(getwd()), "src", "config", "paths.R"), mustWork = FALSE)
}
if (!file.exists(config_path)) stop("[ConfigError] Cannot locate paths.R")

source(config_path, encoding = "UTF-8")
cat("[Config] paths.R successfully loaded globally.\n")

cat("\n========== R PIPELINE START ==========\n")

# --- Definicja bezpiecznego ładowania modułów ---
safe_source <- function(path) {
  full_path <- normalizePath(path, winslash = "/", mustWork = FALSE)

  if (!file.exists(full_path)) {
    alt_paths <- c(
      file.path(R_DIR, basename(path)),
      file.path(CONFIG_DIR, basename(path)),
      file.path(SRC_DIR, "r", basename(path)),
      file.path(SRC_DIR, "config", basename(path))
    )
    for (alt in alt_paths) {
      if (file.exists(alt)) {
        full_path <- normalizePath(alt, winslash = "/", mustWork = FALSE)
        break
      }
    }
  }

  if (file.exists(full_path)) {
    cat("[Source] Loading:", full_path, "\n")
    tryCatch({
      source(full_path, encoding = "UTF-8", local = .GlobalEnv)
    }, error = function(e) {
      stop(paste("[SourceError] Failed to source:", full_path, "\nReason:", e$message))
    })
  } else {
    warning("[SourceWarning] File not found:", path)
  }
}

# --- Wczytaj moduły R ---
safe_source(file.path(R_DIR, "read_data.R"))
safe_source(file.path(R_DIR, "features.R"))
safe_source(file.path(R_DIR, "forecasting.R"))
safe_source(file.path(R_DIR, "diagnostics.R"))

# --- Dane wejściowe ---
DATA_PATH <- file.path(CACHE_DIR, "AAPL_1d.csv")
if (!file.exists(DATA_PATH)) stop(paste("[DataError] File not found:", DATA_PATH))
cat("[Info] Using data file:", DATA_PATH, "\n")

# ============================================================
#   1. Wczytywanie i diagnostyka danych
# ============================================================
df <- read_data(DATA_PATH)
diagnostics <- run_diagnostics(df, column = "Close", auto_fix = TRUE)

cat("ADF p-value:", diagnostics$adf_pvalue, "\n")
cat("Ljung-Box p-value:", diagnostics$box_pvalue, "\n")

df_clean <- diagnostics$df

# ============================================================
#   2. Generacja cech i prognozowanie
# ============================================================
df_features <- generate_features(df_clean)
forecast_result <- run_forecast(df_features)

# ============================================================
#   3. Zapis wyników
# ============================================================
write.csv(df_features, file.path(FEATURES_DIR, "AAPL_features.csv"), row.names = FALSE)
write.csv(forecast_result$future, file.path(FORECASTS_DIR, "AAPL_forecast.csv"), row.names = FALSE)

cat("\n[Pipeline completed successfully]\n")
cat("Saved features →", FEATURES_DIR, "\n")
cat("Saved forecast →", FORECASTS_DIR, "\n")
cat("========== R PIPELINE END   ==========\n")
