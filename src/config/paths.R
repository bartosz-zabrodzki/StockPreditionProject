# ============================================================
# paths.R — unified R configuration for Docker environment
# ============================================================

# Root project directory
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "/app")

# -----------------------------------------------------------------
# Key directories (mirrors Python’s src/config/paths.py)
# -----------------------------------------------------------------
SRC_DIR        <- file.path(PROJECT_ROOT, "src")
R_DIR          <- file.path(SRC_DIR, "r")
CONFIG_DIR     <- file.path(SRC_DIR, "config")

DATA_ROOT      <- Sys.getenv("DATA_ROOT", unset = "/data")
DATA_PROCESSED <- file.path(DATA_ROOT, "data_processed")
CACHE_DIR      <- file.path(DATA_ROOT, "data_cache")
FEATURES_DIR   <- file.path(DATA_PROCESSED, "features")
FORECASTS_DIR  <- file.path(DATA_PROCESSED, "forecasts")

MODEL_DIR      <- Sys.getenv("MODEL_DIR", unset = "/models")
OUTPUT_DIR     <- Sys.getenv("OUTPUT_DIR", unset = "/output")
LOG_DIR        <- Sys.getenv("LOG_DIR", unset = "/logs")

# -----------------------------------------------------------------
# Create directories if they don't exist
# -----------------------------------------------------------------
dir.create(CACHE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FEATURES_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FORECASTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------
# Context variables (exported from Python / environment)
# -----------------------------------------------------------------
TICKER   <- Sys.getenv("STOCK_TICKER", unset = "AAPL")
INTERVAL <- Sys.getenv("STOCK_INTERVAL", unset = "1d")

# -----------------------------------------------------------------
# Debug info for logs
# -----------------------------------------------------------------
cat("[paths.R] Loaded configuration\n")
cat("PROJECT_ROOT:", PROJECT_ROOT, "\n")
cat("SRC_DIR:", SRC_DIR, "\n")
cat("R_DIR:", R_DIR, "\n")
cat("DATA_ROOT:", DATA_ROOT, "\n")
cat("CACHE_DIR:", CACHE_DIR, "\n")
cat("FEATURES_DIR:", FEATURES_DIR, "\n")
cat("FORECASTS_DIR:", FORECASTS_DIR, "\n")
cat("OUTPUT_DIR:", OUTPUT_DIR, "\n")
cat("LOG_DIR:", LOG_DIR, "\n")
cat("Ticker:", TICKER, " Interval:", INTERVAL, "\n")
