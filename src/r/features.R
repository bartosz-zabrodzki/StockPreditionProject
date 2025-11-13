library(TTR)

generate_features <- function(df) {
  df$SMA_20 <- SMA(df$Close, n = 20)
  df$EMA_20 <- EMA(df$Close, n = 20)
  df$RSI_14 <- RSI(df$Close, n = 14)
  df$Volatility <- runSD(df$Close, n = 20)
  df$Momentum <- momentum(df$Close, n = 10)
  df
}