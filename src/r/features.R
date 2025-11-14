suppressPackageStartupMessages({
  library(TTR)
  library(dplyr)
})

generate_features <- function(df) {
  if (!"Close" %in% names(df)) {
    stop("Missing 'Close' column in input data.")
  }

  df <- df %>%
    mutate(
      Close = as.numeric(Close),
      SMA_20 = SMA(Close, n = 20),
      EMA_20 = EMA(Close, n = 20),
      RSI_14 = RSI(Close, n = 14),
      Volatility = runSD(Close, n = 20),
      Momentum = momentum(Close, n = 10)
    )

  df <- df %>% filter(!is.na(SMA_20))

  return(df)
}
