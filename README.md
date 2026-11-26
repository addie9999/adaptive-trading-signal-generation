# Adaptive Trading Signal Generation System

A Python research project that uses market conditions to decide which forecasting model to trust for a ticker. The training pipeline fits a hidden Markov model (HMM) to price history and scores candidate forecasters within each market regime. At prediction time, the system reports the detected regime, the selected model, its expected return, and a buy, sell, or hold signal.

## Run the offline demo

Use Python 3.12 or later. From the repository root, create and activate a virtual environment, then install the project with its test dependencies.

~~~bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS or Linux
# source .venv/bin/activate

python -m pip install -e ".[dev]"
python -m pytest tests/test_pipeline.py -v
~~~

The two pipeline tests generate 1,200 synthetic trading days, train candidate models, produce a signal, and check that a saved model bundle can be loaded for another prediction. They run without a market-data account. Run the full suite with `python -m pytest`.

## Work with historical market data

The default data provider is yfinance. These commands need internet access for the initial download. ARIMA and XGBoost are included in the core installation. Optional neural-network dependencies are available through `python -m pip install -e ".[deep]"`.

~~~bash
python main.py train --ticker AAPL
python main.py predict --ticker AAPL
python main.py walkforward --ticker AAPL --model xgboost
python main.py backtest --ticker AAPL --model xgboost
~~~

| Command | What it produces |
| --- | --- |
| `train` | A saved bundle containing the regime detector, fitted forecasters, feature columns, and model selector |
| `predict` | A JSON record with the observation date, regime, selected model, forecast, and signal. Directional signals also carry risk estimates. |
| `walkforward` | Forecast errors and directional accuracy for successive test windows, with a configurable gap after each training window |
| `backtest` | A historical simulation of the named model with commission and slippage, plus equity and drawdown plots |

The backtest evaluates a single forecaster. The adaptive selector uses a chronological holdout to compare models within regimes, but the regime detector sees the full training history, including that holdout. Its selection scores should therefore be read as exploratory. The project does not report a verified live trading record.

## Monte Carlo and signal accuracy

To explore how the order of historical outcomes affects the backtest, run:

```sh
python main.py backtest --ticker AAPL --model xgboost --simulations 2000 --block-size 5
```

This resamples consecutive blocks of the single-model backtest's net returns, including its configured costs. Each path spans the observed evaluation period. Block starts are sampled uniformly with replacement and wrap at the end of the history. Within each block, returns keep their original order. The JSON report gives 5th, 50th and 95th percentiles of compounded terminal return and maximum drawdown, plus the fraction of simulated paths that end below their starting value. Drawdown is a positive loss magnitude, measured from initial equity or a later peak. A value of 0.12 means 12%.

Five bars is an illustrative block length. Compare several lengths because the results depend on which historical patterns the blocks preserve. The scenarios reuse observed returns and assume those blocks are representative. They do not estimate the probability that the next signal is correct or cover shocks absent from the sample.

For a direction forecast, run the `walkforward` command above. Each fold reports the model's directional accuracy, the accuracy of always predicting the most frequent training-period direction, and the difference (`directional_accuracy_lift`). A lift of 0.03 means three percentage points. Use a final untouched time period after model selection. Both evaluation commands retain extreme return bars, avoiding a full-history outlier filter, and separate training labels from evaluation by at least the forward-label horizon. Results produced before these evaluation changes can differ.

## Explore the code

| Path | What to look for |
| --- | --- |
| [src/pipeline.py](src/pipeline.py) | Training, inference, and model bundle persistence |
| [src/regime](src/regime) and [src/adaptive](src/adaptive) | Regime detection and model selection |
| [src/features](src/features) and [src/models](src/models) | Feature generation and forecasting implementations |
| [src/validation](src/validation) and [src/backtesting](src/backtesting) | Time-ordered evaluation and transaction-cost simulation |
| [research](research) | Nine numbered notebooks, from exploratory analysis through backtesting and explainability |
| [tests](tests) | Synthetic-data tests covering features, regimes, signals, risk, metrics, and the end-to-end pipeline |

## API and configuration

Start the FastAPI service with `python main.py serve --port 8000`. Open [the interactive API documentation](http://localhost:8000/docs) to inspect the request and response schemas. Use `GET /health` to check service status. The training and prediction endpoints accept a POST request to `/train` or `/predict` with a JSON body such as `{"ticker": "AAPL"}`.

Default settings live in [configs/default.yaml](configs/default.yaml). Apply the research or production settings with the CLI's `--env` option. Environment-variable examples are in [.env.example](.env.example). A [Docker Compose configuration](docker/docker-compose.yml) is included for running the API, PostgreSQL, and MLflow together locally.

## License

Released under the [MIT License](LICENSE). This project is for research and education.
