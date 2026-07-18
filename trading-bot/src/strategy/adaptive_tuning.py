"""
Adaptive parameter tuning — drift strategy parameters toward profitability.

Over time, the strategy observes which parameter combinations produce
winning trades and uses an exponential moving average (EMA) to gradually
adapt.  Safety bounds ensure parameters never drift more than 50% from
their defaults, preventing over-fitting to transient conditions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# EMA alpha for updating optimal parameters (0.15 ≈ ~13 trades to half-life).
DEFAULT_ALPHA: float = 0.15

# Minimum number of winning trades before adapted parameters are used.
MIN_WINNING_TRADES: int = 10

# Maximum drift from defaults (50%).
MAX_DRIFT_PCT: float = 0.50

# How many recent winning trade params to keep for inspection.
RECENT_WINS_CAP: int = 50


# ---------------------------------------------------------------------------
# Adaptive state
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveState:
    """Mutable state that tracks trade outcomes and drifts parameters.

    Attributes:
        total_trades: Total completed round-trip trades.
        winning_trades: Number of winning trades.
        optimal_std_dev: EMA-tuned VWAP std-dev threshold.
        optimal_conf_threshold: EMA-tuned minimum confidence.
        optimal_risk_pct: EMA-tuned risk-per-trade percentage.
        recent_win_params: Rolling buffer of winning trade parameters
            for external inspection / debugging.
        _alpha: EMA smoothing factor.
    """

    total_trades: int = 0
    winning_trades: int = 0

    # Per-parameter EMA values — start at the defaults from StrategyConfig.
    optimal_std_dev: float = 2.0
    optimal_conf_threshold: float = 0.6
    optimal_risk_pct: float = 0.25

    recent_win_params: deque[dict] = field(default_factory=deque)

    _alpha: float = DEFAULT_ALPHA

    def update(self, trade_result: dict) -> None:
        """Ingest a completed trade and recompute optimal parameters.

        Only winning trades contribute to the EMA — losses do not shift
        the parameters.  The idea: "keep doing what won; don't chase losses."

        Args:
            trade_result: Dict with keys:

                - ``was_winner`` (bool)
                - ``signal_std_dev`` (float) — std-dev that triggered signal
                - ``signal_confidence`` (float) — confidence at entry
                - ``risk_pct`` (float) — risk % used for sizing
                - ``pnl`` (float) — realized P&L
        """
        self.total_trades += 1

        was_winner = bool(trade_result.get("was_winner", False))

        if not was_winner:
            return

        self.winning_trades += 1

        # Extract parameters from the winning trade.
        win_std_dev = float(trade_result.get("signal_std_dev", self.optimal_std_dev))
        win_conf = float(trade_result.get("signal_confidence", self.optimal_conf_threshold))
        win_risk = float(trade_result.get("risk_pct", self.optimal_risk_pct))
        win_pnl = float(trade_result.get("pnl", 0.0))

        # Push to recent wins buffer.
        self.recent_win_params.append({
            "std_dev": win_std_dev,
            "confidence": win_conf,
            "risk_pct": win_risk,
            "pnl": win_pnl,
        })
        if len(self.recent_win_params) > RECENT_WINS_CAP:
            self.recent_win_params.popleft()

        # Update EMAs with safety bounds.
        self.optimal_std_dev = _bounded_ema(
            self.optimal_std_dev, win_std_dev, self._alpha, 2.0, MAX_DRIFT_PCT,
        )
        self.optimal_conf_threshold = _bounded_ema(
            self.optimal_conf_threshold, win_conf, self._alpha, 0.6, MAX_DRIFT_PCT,
        )
        self.optimal_risk_pct = _bounded_ema(
            self.optimal_risk_pct, win_risk, self._alpha, 0.25, MAX_DRIFT_PCT,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_adapted_params(
    state: AdaptiveState,
    defaults: dict,
) -> dict:
    """Merge adapted parameters with *defaults*, returning a complete dict.

    Adapted values are **only** used when we have at least
    ``MIN_WINNING_TRADES`` winning trades to learn from.  Before that
    threshold the defaults are returned unchanged.

    Args:
        state: The adaptive state to read from.
        defaults: Dict of default parameter values (keys: ``mean_reversion_std_dev``,
            ``min_confidence_threshold``, ``risk_per_trade_pct``).

    Returns:
        Dict with the same keys, populated from either the adaptive state
        or the defaults.
    """
    if state.winning_trades < MIN_WINNING_TRADES:
        return dict(defaults)

    result: dict = {}
    result["mean_reversion_std_dev"] = state.optimal_std_dev
    result["min_confidence_threshold"] = state.optimal_conf_threshold
    result["risk_per_trade_pct"] = state.optimal_risk_pct

    # Fill in any keys from defaults that aren't adapted.
    for key, value in defaults.items():
        if key not in result:
            result[key] = value

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bounded_ema(
    current: float,
    new_value: float,
    alpha: float,
    default: float,
    max_drift_pct: float,
) -> float:
    """Apply an EMA update, then clamp to ``default ± (default × max_drift_pct)``.

    The lower bound is ``default * (1 - max_drift_pct)`` and the upper
    bound is ``default * (1 + max_drift_pct)``.

    Args:
        current: Current EMA value.
        new_value: The new observation.
        alpha: Smoothing factor (0.0–1.0).
        default: The parameter's default value (used for bounds).
        max_drift_pct: Maximum allowed drift from default (0.50 = 50%).

    Returns:
        Bounded EMA value.
    """
    raw = current * (1.0 - alpha) + new_value * alpha

    lower = default * (1.0 - max_drift_pct)
    upper = default * (1.0 + max_drift_pct)

    return max(lower, min(upper, raw))
