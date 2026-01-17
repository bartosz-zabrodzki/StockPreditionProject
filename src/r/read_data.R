# ============================================================
# read_data.R — data loader (inherits global config)
# ============================================================
# Inherit global config from analyze.R instead of reloading paths.R
if (!exists("PROJECT_ROOT")) {
  PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "/app")
}
if (!exists("R_DIR")) {
  R_DIR <- file.path(PROJECT_ROOT, "src", "r")
}
if (!exists("CONFIG_DIR")) {
  CONFIG_DIR <- file.path(PROJECT_ROOT, "src", "config")
}

# Only load paths.R once if missing
if (!exists("PATHS_LOADED") || isFALSE(PATHS_LOADED)) {
  config_path <- file.path(CONFIG_DIR, "paths.R")
  if (!file.exists(config_path)) {
    stop(paste("[ConfigError] Cannot locate paths.R at", config_path))
  }
  source(config_path, encoding = "UTF-8")
  PATHS_LOADED <- TRUE
}

cat("[Config] paths.R loaded\n")
cat("[Config] PROJECT_ROOT=", PROJECT_ROOT, "\n")
cat("[Config] R_DIR=", R_DIR, "\n")

# ============================================================
# Functions
# ============================================================

default_data_filename <- function(ticker) {
  paste0(ticker, "_1d.csv")
}

find_data_file <- function(filename = default_data_filename(TICKER)) {
  # Candidate search paths (relative and absolute)
  paths <- c(
    file.path("src", "data", "data_cache", filename),
    file.path("data", "data_cache", filename),
    file.path("..", "data", "data_cache", filename)
  )

  for (p in paths) {
    if (file.exists(p)) {
      return(normalizePath(p, winslash = "/", mustWork = TRUE))
    }
  }

  stop(paste("Nie znaleziono pliku danych:", filename))
}


read_data <- function(file_path = NULL) {
  # Try auto-locate if no path provided
  if (is.null(file_path) || file_path == "" || !file.exists(file_path)) {
    file_path <- find_data_file(default_data_filename(TICKER))
  }

  if (!file.exists(file_path)) {
    stop(paste("BŁĄD: Data file not found:", file_path))
  }

  cat("Znaleziono plik danych:", normalizePath(file_path, winslash = "/"), "\n")

  # Read header preview
  header_lines <- suppressWarnings(readLines(file_path, n = 5, warn = FALSE))

  # --- Auto-detect format ---
  if (length(header_lines) >= 2 && grepl("^Price,", header_lines[1]) && grepl("^Ticker,", header_lines[2])) {
    col_names <- unlist(strsplit(header_lines[1], ","))
    df <- read.csv(file_path, skip = 2, header = FALSE, stringsAsFactors = FALSE, col.names = col_names)
    cat("Wczytano dane w formacie yfinance. Kolumny:", paste(col_names, collapse = ", "), "\n")

  } else if (any(grepl("^Date", header_lines))) {
    skip_lines <- which(grepl("^Date", header_lines)) - 1
    df <- read.csv(file_path, skip = skip_lines, stringsAsFactors = FALSE)
    cat("Wczytano dane w formacie standardowym. Kolumny:", paste(names(df), collapse = ", "), "\n")

  } else {
    df <- read.csv(file_path, stringsAsFactors = FALSE)
    cat("Wczytano dane w formacie niestandardowym.\n")
  }

  # --- Normalize column names ---
  names(df) <- gsub("[^[:alnum:]_]+", "", trimws(names(df)))

  # --- Ensure Date column exists ---
  if (!"Date" %in% names(df)) {
    warning("Brak kolumny 'Date' – użyto pierwszej kolumny jako daty.")
    df$Date <- df[[1]]
  }

  # --- Parse Date formats robustly ---
  suppressWarnings({
    df$Date <- as.Date(df$Date, format = "%Y-%m-%d")
    if (any(is.na(df$Date))) {
      df$Date <- as.Date(df$Date, tryFormats = c("%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"))
    }
  })

  if (any(is.na(df$Date))) {
    warning("Niektóre daty nadal nie zostały rozpoznane — pozostały tekstowo.")
  } else {
    cat("Daty rozpoznane poprawnie w formacie ISO.\n")
  }

  # --- Validate Close column ---
  if (!"Close" %in% names(df)) {
    stop("BŁĄD: Brak wymaganej kolumny 'Close' w danych!")
  }

  df$Close <- suppressWarnings(as.numeric(df$Close))
  df <- df[!is.na(df$Date) & !is.na(df$Close), ]

  # --- Summary ---
  cat("Wczytano poprawnie wierszy:", nrow(df), "\n")
  invisible(df)
}
