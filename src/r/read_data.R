find_data_file <- function(filename = "AAPL_1d.csv") {
  paths <- c(
    file.path("data", "data_cache", filename),
    file.path("src", "data", "data_cache", filename)
  )
  for (p in paths) {
    if (file.exists(p)) return(p)
  }
  stop(paste("Nie znaleziono pliku danych:", filename))
}

read_data <- function(file_path = NULL) {
  if (is.null(file_path) || file_path == "") {
    file_path <- find_data_file("AAPL_1d.csv")
  }
  cat("Znaleziono plik danych:", normalizePath(file_path), "\n")

  header_lines <- readLines(file_path, n = 5)


  if (grepl("^Price,", header_lines[1]) && grepl("^Ticker,", header_lines[2])) {
  # YFinance-style multi-header file
  col_names <- unlist(strsplit(header_lines[1], ","))
  df <- read.csv(file_path, skip = 2, header = FALSE, stringsAsFactors = FALSE, col.names = col_names)
  cat("Wczytano dane w formacie yfinance. Kolumny:", paste(col_names, collapse = ", "), "\n")

} else if (any(grepl("^Date", header_lines))) {
  # Standard CSV (with Date header somewhere in top lines)
  skip_lines <- which(grepl("^Date", header_lines)) - 1
  df <- read.csv(file_path, skip = skip_lines, stringsAsFactors = FALSE)
  cat("Wczytano dane w formacie standardowym. Kolumny:", paste(names(df), collapse = ", "), "\n")

} else {
  # Fallback: assume simple CSV
  df <- read.csv(file_path, stringsAsFactors = FALSE)
  cat("Wczytano dane w formacie niestandardowym.\n")
}

# --- UNIVERSAL DATE PARSING FIX ---
if ("Date" %in% names(df)) {
  suppressWarnings({
    df$Date <- as.Date(df$Date, format = "%Y-%m-%d")
    if (any(is.na(df$Date))) {
      # Retry with a few common alternate formats (defensive)
      df$Date <- as.Date(df$Date, tryFormats = c("%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"))
    }
  })
  if (any(is.na(df$Date))) {
    warning("Niektóre daty nadal nie zostały rozpoznane — pozostały tekstowo.")
  } else {
    cat("Daty rozpoznane poprawnie w formacie ISO.\n")
  }
} else {
  warning("Brak kolumny 'Date' w danych.")
}


  names(df) <- trimws(names(df))
  names(df) <- gsub("[^[:alnum:]]+", "", names(df))

  if ("Price" %in% names(df) && !"Date" %in% names(df)) {
    names(df)[names(df) == "Price"] <- "Date"
  }

  if (!"Date" %in% names(df)) {
    df$Date <- df[[1]]
  }
df$Date <- as.character(df$Date)
df$Date <- trimws(gsub("[^0-9\\-]", "", df$Date))
parsed <- as.Date(df$Date, format = "%Y-%m-%d")
if (any(is.na(parsed))) {
  cat("Niektóre daty nadal nie zostały rozpoznane — pozostały tekstowo.\n")
} else {
  df$Date <- parsed
}

df <- df[!is.na(df$Close) & !is.na(df$Date), ]

if (all(grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", df$Date))) {
  df$Date <- as.Date(df$Date, format = "%Y-%m-%d")
} else if (all(grepl("^[0-9]{2}/[0-9]{2}/[0-9]{4}$", df$Date))) {
  df$Date <- as.Date(df$Date, format = "%m/%d/%Y")
} else {
  warning("Nie udało się jednoznacznie rozpoznać formatu daty. Pozostawiono tekstowo.")
}

df <- df[!is.na(df$Date) & !is.na(df$Close), ]

  if (any(is.na(df$Date))) {
    cat("Ostrzeżenie: niektóre daty nie zostały rozpoznane. Przykładowe wartości:\n")
    print(unique(df$Date[is.na(df$Date)]))
  }

  df$Close <- suppressWarnings(as.numeric(df$Close))
  df <- df[!is.na(df$Close) & !is.na(df$Date), ]

  cat("Wczytano poprawnie wierszy:", nrow(df), "\n")
  return(df)

}