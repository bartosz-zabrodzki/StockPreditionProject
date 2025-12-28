## Configuring the stock ticker

Set the `STOCK_TICKER` environment variable to change which company is downloaded, processed, trained, and predicted. All cache, model, scaler, and output filenames now include the ticker to keep multi-ticker runs isolated.

```bash
export STOCK_TICKER=MSFT
python -m src.data.data            # download and cache new ticker data
Rscript src/r/analyze.R            # generate features and forecasts for the ticker
python -m src.training.train_model # train models using the ticker-specific features
python -m src.prediction.predict   # create predictions and plots for the ticker
```
