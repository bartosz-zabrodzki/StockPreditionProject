cat("[Diagnostic] Checking R environment...\n")

required <- c("forecast", "tseries", "TTR", "dplyr", "quantmod")
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]

if (length(missing) > 0) {
  cat("Missing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
}

TICKER <- toupper(trimws(Sys.getenv("STOCK_TICKER", "AAPL")))
if (is.na(TICKER) || TICKER == "") TICKER <- "AAPL"
cat("[Diagnostic] Session info:\n")
print(sessionInfo())

cat("[Diagnostic] Checking data file...\n")
data_file <- file.path("src", "data", "data_cache", paste0(TICKER, "_1d.csv"))
if (!file.exists(data_file)) {
  cat("File missing:", data_file, "\n")
} else {
  cat("File exists:", data_file, "\n")
}

cat("[Diagnostic] Finished. R environment OK.\n")
