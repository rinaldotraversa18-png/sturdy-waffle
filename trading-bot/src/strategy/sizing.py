"""
Position sizing — risk-based contract quantity calculation.

Every sizing decision is driven by the **remaining daily loss
headroom**.  The core principle: never risk more than a configurable
fraction of what the evaluation still allows us to lose that day.
"""

from __future__ import annotations

import math

from src.client.models import Account, Position
from src.risk.limits import InstrumentLimit
from src.risk.state import RiskState
from src.strategy.signals import Signal

# ---------------------------------------------------------------------------
# Default point values for CME Micro Bitcoin (MBT) and Micro Ether (MET).
# Micro BTC: 0.1 BTC × $1.00/BTC point = $0.10 per point
# Micro ETH: 0.1 ETH × $1.00/ETH point = $0.10 per point
# These can be overridden via *instrument_limits* if the config carries
# an explicit per-instrument ``point_value``.
# ---------------------------------------------------------------------------
_DEFAULT_POINT_VALUES: dict[str, float] = {
    "MBT": 0.10,
    "MET": 0.10,
}


class StrategyConfig:
    """Strategy parameter container.

    Attributes:
        symbols: Contract symbols to trade (default ``["MBT", "MET"]``).
        primary_symbol: Preferred symbol when only one trade is allowed.
        risk_per_trade_pct: Fraction of remaining daily loss to risk per
            trade (0.0–1.0, default 0.25 = 25%).
        max_risk_per_trade: Hard dollar cap on risk per trade ($300).
        mean_reversion_std_dev: Std-dev threshold for mean-reversion
            signals (default 2.0).
        breakout_consecutive_ticks: How many same-direction ticks
            constitute a breakout (default 5).
        volume_spike_multiplier: Volume threshold ratio versus rolling
            average (default 2.0).
        quote_window_size: Max quotes kept in the rolling buffer (200).
        min_spread_bps: Max spread in bps for a symbol to be eligible.
        cooldown_seconds: Minimum seconds between signals per symbol.
        min_confidence_threshold: Signals below this confidence are
            discarded.
        point_values: Per-symbol point value overrides (falls back to
            ``_DEFAULT_POINT_VALUES``).
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        primary_symbol: str = "MBT",
        risk_per_trade_pct: float = 0.25,
        max_risk_per_trade: float = 300.0,
        mean_reversion_std_dev: float = 2.0,
        breakout_consecutive_ticks: int = 5,
        volume_spike_multiplier: float = 2.0,
        quote_window_size: int = 200,
        min_spread_bps: float = 5.0,
        cooldown_seconds: float = 30.0,
        min_confidence_threshold: float = 0.6,
        point_values: dict[str, float] | None = None,
    ) -> None:
        self.symbols = symbols or ["MBT", "MET"]
        self.primary_symbol = primary_symbol
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_risk_per_trade = max_risk_per_trade
        self.mean_reversion_std_dev = mean_reversion_std_dev
        self.breakout_consecutive_ticks = breakout_consecutive_ticks
        self.volume_spike_multiplier = volume_spike_multiplier
        self.quote_window_size = quote_window_size
        self.min_spread_bps = min_spread_bps
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence_threshold = min_confidence_threshold
        self.point_values = point_values or _DEFAULT_POINT_VALUES


def _get_point_value(symbol: str, config: StrategyConfig) -> float:
    """Resolve the dollar value of a 1-point move for *symbol*."""
    if symbol in config.point_values:
        return config.point_values[symbol]
    # Fallback: assume micro contract = $0.10/pt.
    return 0.10


def calculate_position_size(
    signal: Signal,
    account: Account,
    risk_state: RiskState,
    current_positions: list[Position],
    config: StrategyConfig,
    instrument_limits: dict[str, InstrumentLimit],
    signal_confidence: float | None = None,
) -> int:
    """Risk-based position sizing for a single trade signal.

    Algorithm
    ---------
    1. Determine the dollar risk budget:
       ``risk_budget = min(remaining_daily_loss × risk_per_trade_pct,
       max_risk_per_trade)``.
    2. Convert the stop distance (price units) to dollars:
       ``stop_distance_$ = |entry − stop| × point_value``.
    3. Compute raw contracts:
       ``contracts = floor(risk_budget / stop_distance_$)``.
    4. Apply confidence scaling (if provided):
       ``confidence_scalar = 0.5 + (signal_confidence × 0.5)`` → range 0.5–1.0.
       Contracts are multiplied by this scalar and floored.
    5. Cap at ``instrument_max − |current_position|``.
    6. Return at least 1 if the risk budget allows, otherwise 0.

    Args:
        signal: The trade signal (must have a non-``None`` stop_price
            and a defined direction).
        account: Current account snapshot (used for equity awareness).
        risk_state: Live risk state from the :class:`RiskEngine`.
        current_positions: All open positions.
        config: Strategy parameters.
        instrument_limits: Per-symbol contract caps.
        signal_confidence: Optional 0.0–1.0 score that scales position
            size linearly.  0.6 → 0.8×, 0.9 → 0.95×, 1.0 → 1.0×.

    Returns:
        Number of contracts to trade (positive integer), or 0 to skip.
    """
    if signal.direction == "flat":
        return 0

    # ---- 1. Risk budget ---------------------------------------------------
    remaining_daily = _remaining_daily_loss(risk_state, config)
    if remaining_daily <= 0:
        return 0

    risk_per_trade_dollars = min(
        remaining_daily * config.risk_per_trade_pct,
        config.max_risk_per_trade,
    )

    # ---- 2. Stop distance in dollars --------------------------------------
    entry = signal.entry_price if signal.entry_price is not None else signal.stop_price
    stop_distance = abs(entry - signal.stop_price)
    if stop_distance <= 0:
        return 0

    point_value = _get_point_value(signal.symbol, config)
    stop_distance_dollars = stop_distance * point_value
    if stop_distance_dollars <= 0:
        return 0

    # ---- 3. Raw contracts -------------------------------------------------
    raw_contracts = int(risk_per_trade_dollars / stop_distance_dollars)
    if raw_contracts < 1:
        return 0

    # ---- 3b. Confidence-based scaling (NEW) --------------------------------
    if signal_confidence is not None:
        confidence_scalar = 0.5 + (signal_confidence * 0.5)
        # Clamp to [0.0, 1.0] for safety.
        confidence_scalar = max(0.0, min(1.0, confidence_scalar))
        raw_contracts = int(math.floor(raw_contracts * confidence_scalar))
        if raw_contracts < 1:
            raw_contracts = 0

    if raw_contracts < 1:
        return 0

    # ---- 4. Instrument limit cap ------------------------------------------
    limit = _get_max_contracts(signal.symbol, instrument_limits)
    current_net = _get_net_position(signal.symbol, current_positions)
    available = limit - abs(current_net)
    if available <= 0:
        return 0

    return min(raw_contracts, available)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remaining_daily_loss(
    risk_state: RiskState,
    config: StrategyConfig,
) -> float:
    """Compute remaining daily loss headroom.

    Mirrors :meth:`RiskEngine.remaining_daily_loss` so the sizing
    function can operate on a plain ``RiskState`` snapshot without a
    live engine reference.
    """
    if risk_state.session_realized_pnl >= 0:
        return 1250.0  # Tradeify $50k daily loss limit
    remaining = 1250.0 + risk_state.session_realized_pnl
    return max(0.0, remaining)


def _get_max_contracts(
    symbol: str,
    instrument_limits: dict[str, InstrumentLimit],
) -> int:
    """Return the max-contracts cap for *symbol*."""
    if symbol in instrument_limits:
        return instrument_limits[symbol].max_contracts
    # Heuristic: M-prefix = micro = 40, else mini = 4.
    return 40 if symbol.startswith("M") else 4


def _get_net_position(symbol: str, positions: list[Position]) -> int:
    """Extract ``net_pos`` for *symbol* from *positions*."""
    for pos in positions:
        if pos.symbol == symbol:
            return pos.net_pos
    return 0
