cat("[Diagnostic] Checking R environment...\n")

required <- c("forecast", "tseries", "TTR", "dplyr", "quantmod")
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]

if (length(missing) > 0) {
  cat("Missing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
}

cat("[Diagnostic] Session info:\n")
print(sessionInfo())

cat("[Diagnostic] Checking data file...\n")
if (!file.exists("src/data/data_cache/AAPL_1d.csv")) {
  cat("File missing: src/data/data_cache/AAPL_1d.csv\n")
} else {
  cat("File exists: src/data/data_cache/AAPL_1d.csv\n")
}

cat("[Diagnostic] Finished. R environment OK.\n")
