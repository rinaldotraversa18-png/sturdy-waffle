"""
Risk configuration models for Tradeify funded account evaluations.

Defines per-instrument contract limits and the global evaluation
parameters (profit target, drawdown, daily loss).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InstrumentLimit(BaseModel):
    """Per-instrument contract caps and tick-value metadata.

    Attributes:
        max_contracts: Maximum absolute net position allowed for this
            instrument (e.g. 4 for mini-index futures, 40 for micros).
        tick_value: Dollar value of one tick (e.g. $1.25 for MES,
            $0.50 for MNQ).
    """

    max_contracts: int = Field(ge=1, description="Maximum absolute net position")
    tick_value: float = Field(gt=0, description="Dollar value per tick")


class RiskConfig(BaseModel):
    """Global risk parameters for a Tradeify funded account evaluation.

    Defaults correspond to the **$50,000 Growth Funded Account** tier.

    Attributes:
        profit_target: Total realized P&L needed to pass ($3,000).
        max_eod_drawdown: Maximum drawdown from peak equity ($2,000).
        daily_loss_limit: Maximum daily realized loss ($1,250).
        max_mini_contracts: Hard cap on mini-index contracts (4).
        max_micro_contracts: Hard cap on micro-index contracts (40).
        instrument_limits: Per-symbol overrides.  Falls back to
            ``max_mini_contracts`` / ``max_micro_contracts`` when a
            symbol is not listed.
    """

    profit_target: float = Field(default=3000.0, gt=0)
    max_eod_drawdown: float = Field(default=2000.0, gt=0)
    daily_loss_limit: float = Field(default=1250.0, gt=0)
    max_mini_contracts: int = Field(default=4, ge=1)
    max_micro_contracts: int = Field(default=40, ge=1)
    instrument_limits: dict[str, InstrumentLimit] = Field(default_factory=dict)

    def get_limit(self, symbol: str) -> int:
        """Return the max-contracts limit for *symbol*.

        Checks ``instrument_limits`` first; if the symbol is not
        present, we fall back to a heuristic based on common micro/mini
        prefixes (``M`` prefix → micro → 40; otherwise → mini → 4).
        """
        if symbol in self.instrument_limits:
            return self.instrument_limits[symbol].max_contracts

        # Heuristic: symbols starting with "M" (e.g. MES, MNQ, M2K) are
        # micro contracts (40), everything else is a mini (4).
        if symbol.startswith("M"):
            return self.max_micro_contracts
        return self.max_mini_contracts
