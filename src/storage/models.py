"""SQLAlchemy ORM models for the signal/prediction store."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class SignalRecord(Base):
    """A persisted trading signal / prediction event."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    regime: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(32))
    signal: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    expected_return: Mapped[float] = mapped_column(Float)
    last_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_size: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SignalRecord {self.ticker} {self.as_of} {self.signal} ({self.regime})>"
