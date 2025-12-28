# ============================================================
# paths.R — Centralized project path configuration (final version)
# ============================================================

# Skip reinitialization if paths are already defined globally
if (exists("PROJECT_ROOT", envir = .GlobalEnv)) {
  cat("[Config] paths.R already loaded — skipping reinitialization.\n")
  return(invisible(TRUE))
}

# --- Detect project root robustly ---
detect_project_root <- function() {
  wd <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  paths_to_try <- c(
    wd,
    dirname(wd),
    file.path(dirname(wd), "StockPreditionProject"),
    file.path(dirname(dirname(wd)), "StockPreditionProject")
  )

  for (p in paths_to_try) {
    if (dir.exists(file.path(p, "src", "r"))) {
      return(normalizePath(p, winslash = "/", mustWork = TRUE))
    }
  }

  stop("[PathError] Cannot determine project root — 'src/r' not found.")
}

PROJECT_ROOT <- detect_project_root()
SRC_DIR <- file.path(PROJECT_ROOT, "src")
R_DIR <- file.path(SRC_DIR, "r")
DATA_DIR <- file.path(SRC_DIR, "data")
CACHE_DIR <- file.path(DATA_DIR, "data_cache")
PROCESSED_DIR <- file.path(DATA_DIR, "data_processed")
FEATURES_DIR <- file.path(PROCESSED_DIR, "features")
FORECASTS_DIR <- file.path(PROCESSED_DIR, "forecasts")
OUTPUTS_DIR <- file.path(SRC_DIR, "outputs")
CONFIG_DIR <- file.path(SRC_DIR, "config")

# --- Ensure directories exist ---
dirs <- c(DATA_DIR, CACHE_DIR, PROCESSED_DIR, FEATURES_DIR, FORECASTS_DIR, OUTPUTS_DIR)
for (d in dirs) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# --- Diagnostics ---
cat("[Config] Project root set to:", PROJECT_ROOT, "\n")
cat("[Config] R module path:", R_DIR, "\n")
cat("[Config] Data cache path:", CACHE_DIR, "\n")
cat("[Config] Processed data path:", PROCESSED_DIR, "\n")

# --- Export globally ---
assign("PROJECT_ROOT", PROJECT_ROOT, envir = .GlobalEnv)
assign("SRC_DIR", SRC_DIR, envir = .GlobalEnv)
assign("R_DIR", R_DIR, envir = .GlobalEnv)
assign("DATA_DIR", DATA_DIR, envir = .GlobalEnv)
assign("CACHE_DIR", CACHE_DIR, envir = .GlobalEnv)
assign("PROCESSED_DIR", PROCESSED_DIR, envir = .GlobalEnv)
assign("FEATURES_DIR", FEATURES_DIR, envir = .GlobalEnv)
assign("FORECASTS_DIR", FORECASTS_DIR, envir = .GlobalEnv)
assign("OUTPUTS_DIR", OUTPUTS_DIR, envir = .GlobalEnv)
assign("CONFIG_DIR", CONFIG_DIR, envir = .GlobalEnv)

cat("[Config] Paths.R successfully initialized in global environment.\n")
