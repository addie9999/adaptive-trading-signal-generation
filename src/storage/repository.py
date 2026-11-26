"""Repository pattern over the signal store.

Defaults to the configured PostgreSQL URL but transparently falls back to a
local SQLite file when Postgres is unreachable, so research never blocks on a
database being up.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import Base, SignalRecord
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SQLITE_FALLBACK = "sqlite:///artifacts/signals.db"


class SignalRepository:
    """CRUD access to :class:`SignalRecord` rows."""

    def __init__(self, config: AppConfig) -> None:
        self.engine = self._make_engine(config)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def _make_engine(self, config: AppConfig) -> Engine:
        try:
            engine = create_engine(config.database.url, echo=config.database.echo)
            engine.connect().close()
            logger.info("Connected to signal store: %s", config.database.url.split("@")[-1])
            return engine
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB unavailable (%s); falling back to SQLite", exc)
            from pathlib import Path

            Path("artifacts").mkdir(exist_ok=True)
            return create_engine(_SQLITE_FALLBACK)

    # ------------------------------------------------------------------ #
    def save_prediction(self, prediction: dict) -> int:
        """Persist a prediction dict (as produced by ``PredictionOutput``)."""
        record = SignalRecord(
            ticker=prediction["ticker"],
            as_of=prediction["as_of"],
            regime=prediction["market_regime"],
            model=prediction["selected_model"],
            signal=prediction["signal"],
            confidence=prediction["confidence"],
            expected_return=prediction["expected_return"],
            last_price=prediction["last_price"],
            stop_loss=prediction.get("stop_loss"),
            take_profit=prediction.get("take_profit"),
            position_size=prediction.get("position_size"),
        )
        with self._Session() as session:  # type: Session
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def recent(self, ticker: str | None = None, limit: int = 50) -> list[SignalRecord]:
        with self._Session() as session:
            query = session.query(SignalRecord)
            if ticker:
                query = query.filter(SignalRecord.ticker == ticker.upper())
            return query.order_by(SignalRecord.id.desc()).limit(limit).all()
