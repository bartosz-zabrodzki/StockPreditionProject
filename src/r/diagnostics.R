library(tseries)

run_diagnostics <- function(df) {
  adf <- adf.test(df$Close, alternative = "stationary")
  box <- Box.test(df$Close, lag = 20, type = "Ljung-Box")
  list(adf = adf$p.value, box = box$p.value)
}
