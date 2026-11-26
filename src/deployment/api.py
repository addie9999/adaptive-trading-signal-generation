"""FastAPI application exposing the prediction service.

Run with::

    uvicorn src.deployment.api:app --reload

Endpoints
---------
GET  /health            liveness + loaded tickers
POST /train             train (or retrain) a ticker's models
POST /predict           regime-aware signal + risk decision for a ticker
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.deployment.service import PredictionService
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Equity ticker symbol")


class TrainRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])


class PredictResponse(BaseModel):
    ticker: str
    as_of: str
    market_regime: str
    selected_model: str
    signal: str
    confidence: float
    expected_return: float
    expected_holding_period: int
    stop_loss: float | None
    take_profit: float | None
    position_size: float | None
    risk_reward_ratio: float | None
    last_price: float


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    config = load_config()
    setup_logging(config.project.log_level)
    service = PredictionService(config)

    app = FastAPI(
        title="Adaptive Quantitative Alpha Generation Framework",
        description="Regime-aware ML signal service (BUY/SELL/HOLD).",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "loaded_tickers": service.loaded_tickers()}

    @app.post("/train")
    def train(req: TrainRequest) -> dict:
        try:
            path = service.train(req.ticker)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed for %s", req.ticker)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "trained", "ticker": req.ticker.upper(), "artifact": str(path)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        try:
            result = service.predict(req.ticker)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed for %s", req.ticker)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PredictResponse(**result.to_dict())

    return app


app = create_app()
