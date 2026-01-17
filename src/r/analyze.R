# ============================================================
# analyze.R — główny pipeline R
# ============================================================

# ============================================================
# analyze.R — unified header (robust path & JSON-safe)
# ============================================================

suppressPackageStartupMessages({
  library(forecast)
  library(TTR)
  library(dplyr)
  library(quantmod)
  library(jsonlite)
  library(fs)
})

# --- Global error trap for Python Bridge ---
options(error = function(e) {
  cat("[BridgeError]", e$message, "\n")
  cat(toJSON(list(error = e$message), auto_unbox = TRUE))
  q("no", 1, FALSE)
})

# --- Parse CLI args or env ---
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NA_character_) {
  i <- match(flag, args)
  if (!is.na(i) && length(args) >= i + 1) return(args[[i + 1]])
  default
}

ticker_arg   <- get_arg("--ticker",   Sys.getenv("STOCK_TICKER", "AAPL"))
interval_arg <- get_arg("--interval", Sys.getenv("STOCK_INTERVAL", "1d"))

Sys.setenv(STOCK_TICKER = ticker_arg)
Sys.setenv(STOCK_INTERVAL = interval_arg)

cat("[Config] Using TICKER=", ticker_arg, " INTERVAL=", interval_arg, "\n")

# ============================================================
# Locate configuration root dynamically
# ============================================================
project_candidates <- c(
  Sys.getenv("PROJECT_ROOT", unset = "/app"),
  getwd(),
  dirname(getwd()),
  "/"
)
config_path <- NA_character_
for (base in project_candidates) {
  trial <- file.path(base, "src", "config", "paths.R")
  if (file.exists(trial)) {
    config_path <- normalizePath(trial, mustWork = TRUE)
    break
  }
}



if (is.na(config_path)) {
  stop("[ConfigError] Cannot locate src/config/paths.R in any known root.")
}

source(config_path, encoding = "UTF-8")
cat("[Config] paths.R successfully loaded globally.\n")

cat("\n========== R PIPELINE START ==========\n")

# ============================================================
# Safe source helper (robust module imports)
# ============================================================
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
# paths.R powinien mieć już TICKER i INTERVAL na podstawie Sys.getenv(...)
DATA_PATH <- file.path(CACHE_DIR, paste0(TICKER, "_", INTERVAL, ".csv"))
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
write.csv(df_features, file.path(FEATURES_DIR, paste0(TICKER, "_features.csv")), row.names = FALSE)
write.csv(forecast_result$future, file.path(FORECASTS_DIR, paste0(TICKER, "_forecast.csv")), row.names = FALSE)

cat("\n[Pipeline completed successfully]\n")
cat("Saved features →", FEATURES_DIR, "\n")
cat("Saved forecast →", FORECASTS_DIR, "\n")
cat("========== R PIPELINE END   ==========\n")

# ============================================================
# Final output section for Python bridge
# ============================================================

suppressPackageStartupMessages(library(jsonlite))

cat("[Bridge] Preparing JSON output for Python API...\n")

# Retrieve environment and ticker name
ticker_arg <- Sys.getenv("STOCK_TICKER", "AAPL")
forecast_dir <- Sys.getenv("FORECASTS_DIR", "/data/data_processed/forecasts")
forecast_path <- file.path(forecast_dir, paste0(ticker_arg, "_forecast.csv"))

if (!file.exists(forecast_path)) {
  cat("[OutputError] Forecast file not found at: ", forecast_path, "\n")
  stop("Forecast file missing, cannot emit JSON.")
}

# Read forecast data
forecast_data <- tryCatch({
  read.csv(forecast_path, stringsAsFactors = FALSE)
}, error = function(e) {
  stop("[ReadError] Unable to read forecast file: ", e$message)
})

# Normalize column names
if (!("Date" %in% names(forecast_data))) {
  names(forecast_data)[1] <- "Date"
}
if (!("R_Forecast" %in% names(forecast_data))) {
  names(forecast_data)[ncol(forecast_data)] <- "R_Forecast"
}

# Minimal check
if (nrow(forecast_data) == 0) {
  stop("[OutputError] Forecast file is empty: ", forecast_path)
}

# Prepare result frame
result_df <- forecast_data[, c("Date", "R_Forecast")]

# ============================================================
# Final output section for Python bridge (auto-detect column)
# ============================================================

suppressPackageStartupMessages(library(jsonlite))

cat("[Bridge] Preparing JSON output for Python API...\n")

ticker_arg <- Sys.getenv("STOCK_TICKER", "AAPL")
forecast_dir <- Sys.getenv("FORECASTS_DIR", "/data/data_processed/forecasts")
forecast_path <- file.path(forecast_dir, paste0(ticker_arg, "_forecast.csv"))

if (!file.exists(forecast_path)) {
  cat("[OutputError] Forecast file not found: ", forecast_path, "\n")
  cat("[]")
  quit(save = "no", status = 0)
}

forecast_data <- tryCatch({
  read.csv(forecast_path, stringsAsFactors = FALSE)
}, error = function(e) {
  cat("[ReadError] Unable to read forecast file: ", e$message, "\n")
  cat("[]")
  quit(save = "no", status = 0)
})

if (nrow(forecast_data) == 0) {
  cat("[OutputError] Forecast file is empty: ", forecast_path, "\n")
  cat("[]")
  quit(save = "no", status = 0)
}

# Automatically pick the right forecast column
if ("Combined" %in% names(forecast_data)) {
  forecast_data$R_Forecast <- forecast_data$Combined
} else if ("Forecast" %in% names(forecast_data)) {
  forecast_data$R_Forecast <- forecast_data$Forecast
} else if ("R_Forecast" %in% names(forecast_data)) {
  # already OK
} else {
  cat("[ColumnError] No forecast-like column found. Columns:", paste(names(forecast_data), collapse = ", "), "\n")
  cat("[]")
  quit(save = "no", status = 0)
}

result_df <- forecast_data[, c("Date", "R_Forecast")]
result_df$R_Forecast <- as.numeric(result_df$R_Forecast)
result_df <- result_df[is.finite(result_df$R_Forecast), ]

json_output <- tryCatch({
  toJSON(result_df, pretty = FALSE, auto_unbox = TRUE, na = "null")
}, error = function(e) {
  cat("[JSONError] Failed to encode JSON: ", e$message, "\n")
  cat("[]")
  quit(save = "no", status = 0)
})

cat(json_output, "\n[Bridge] JSON output completed successfully.\n")
flush.console()
