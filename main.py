"""Command-line entrypoint for the AQAF framework.

Examples
--------
    python main.py train      --ticker AAPL --env research
    python main.py predict    --ticker AAPL
    python main.py walkforward --ticker AAPL --model xgboost
    python main.py backtest   --ticker AAPL
    python main.py serve      --port 8000

Each sub-command is a thin wrapper around the library modules so the same code
paths back both the CLI and the FastAPI service.
"""

from __future__ import annotations

import argparse
import json

from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
def _cmd_train(args: argparse.Namespace) -> None:
    from src.deployment.service import PredictionService

    config = load_config(env=args.env)
    service = PredictionService(config)
    path = service.train(args.ticker)
    print(f"Trained bundle saved to: {path}")


def _cmd_predict(args: argparse.Namespace) -> None:
    from src.deployment.service import PredictionService
    from src.storage import SignalRepository

    config = load_config(env=args.env)
    service = PredictionService(config)
    if not service.is_trained(args.ticker):
        logger.info("No bundle found for %s; training first", args.ticker)
        service.train(args.ticker)
    result = service.predict(args.ticker).to_dict()
    print(json.dumps(result, indent=2, default=str))
    try:
        SignalRepository(config).save_prediction(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist prediction: %s", exc)


def _cmd_walkforward(args: argparse.Namespace) -> None:
    from src.cleaning import DataCleaner
    from src.data import MarketDataLoader
    from src.features import FeatureEngineer
    from src.models import build_model
    from src.reporting import plot_walk_forward
    from src.validation import WalkForwardValidator

    config = load_config(env=args.env)
    raw = MarketDataLoader(config).load_one(args.ticker)
    clean = DataCleaner(config).clean(raw)
    X, y = FeatureEngineer(config).build_dataset(clean)

    validator = WalkForwardValidator(config)
    results = validator.evaluate(lambda: build_model(args.model, config), X, y)
    if results.empty:
        print("No folds produced — check validation window vs. data length.")
        return
    print(results.to_string(index=False))
    print("\nMean:", results[["rmse", "mae", "directional_accuracy"]].mean().round(5).to_dict())
    plot_walk_forward(results)


def _cmd_backtest(args: argparse.Namespace) -> None:
    import numpy as np

    from src.backtesting import BacktestEngine
    from src.cleaning import DataCleaner
    from src.data import MarketDataLoader
    from src.features import FeatureEngineer
    from src.models import build_model
    from src.reporting import plot_drawdown, plot_equity_curve

    config = load_config(env=args.env)
    raw = MarketDataLoader(config).load_one(args.ticker)
    clean = DataCleaner(config).clean(raw)
    X, y = FeatureEngineer(config).build_dataset(clean)

    split = int(len(X) * 0.7)
    model = build_model(args.model, config).fit(X.iloc[:split], y.iloc[:split])
    preds = model.predict(X.iloc[split:])

    # Map predicted forward return to a long/flat/short target position.
    import pandas as pd

    prices = clean["close"].reindex(X.index).iloc[split:]
    preds = preds[-len(prices):]
    positions = pd.Series(np.sign(preds), index=prices.index)
    result = BacktestEngine(config).run(prices, positions)

    print(json.dumps({k: round(float(v), 4) for k, v in result.metrics.items()}, indent=2))
    plot_equity_curve(result.equity_curve, benchmark=prices)
    plot_drawdown(result.equity_curve)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    config = load_config(env=args.env)
    uvicorn.run(
        "src.deployment.api:app",
        host=config.api.host,
        port=args.port or config.api.port,
        reload=args.reload,
    )


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive Quantitative Alpha Framework")
    # Shared options available on every sub-command (so `--env` may appear
    # either before or after the sub-command name).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env", default=None, help="Config environment (research/production)")
    parser.add_argument("--env", default=None, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", parents=[common], help="Train and persist a ticker bundle")
    p_train.add_argument("--ticker", required=True)
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", parents=[common], help="Generate a signal for a ticker")
    p_pred.add_argument("--ticker", required=True)
    p_pred.set_defaults(func=_cmd_predict)

    p_wf = sub.add_parser("walkforward", parents=[common], help="Walk-forward validate a model")
    p_wf.add_argument("--ticker", required=True)
    p_wf.add_argument("--model", default="xgboost")
    p_wf.set_defaults(func=_cmd_walkforward)

    p_bt = sub.add_parser("backtest", parents=[common], help="Backtest a model's signals")
    p_bt.add_argument("--ticker", required=True)
    p_bt.add_argument("--model", default="xgboost")
    p_bt.set_defaults(func=_cmd_backtest)

    p_serve = sub.add_parser("serve", parents=[common], help="Run the FastAPI service")
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def cli() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
