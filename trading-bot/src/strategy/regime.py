"""
Market regime detection — classify price action as trending, ranging, or choppy.

Uses linear regression slope and price efficiency ratio on the rolling
quote buffer to determine the prevailing market structure.  Choppy markets
suppress all signal generation (safety-first principle).
"""

from __future__ import annotations

from collections import deque
from enum import Enum

from src.client.models import Quote


class MarketRegime(Enum):
    """Three market structure categories."""

    TRENDING = "trending"  # Clear directional bias
    RANGING = "ranging"  # Oscillating within a band
    CHOPPY = "choppy"  # No clear pattern — avoid trading


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_regime(
    quotes: deque[Quote],
    n: int | None = None,
) -> tuple[MarketRegime, float]:
    """Analyze recent quotes to determine market regime.

    Method
    ------
    1. Compute **linear regression slope** over the last *n* prices to
       measure direction strength (positive = uptrend, negative = downtrend).
    2. Compute **price efficiency**::

           efficiency = |price_last - price_first|
                        ---------------------------------
                        sum(|price[t] - price[t-1]|)

       - High efficiency (> 0.4) → TRENDING
       - Medium efficiency (0.2–0.4) → RANGING
       - Low efficiency (< 0.2) → CHOPPY
    3. Return ``(regime, confidence)`` where confidence ∈ [0.0, 1.0]
       reflects how strongly the regime is identified.

    Choppy markets → no signals generated (safety first).

    Args:
        quotes: Rolling quote buffer (at least 20 elements recommended).
        n: Number of most recent quotes to analyze (default: all).

    Returns:
        Tuple of ``(MarketRegime, float_confidence)``.
    """
    if n is None:
        n = len(quotes)

    if len(quotes) < 2:
        return MarketRegime.CHOPPY, 0.0

    # Work on the last *n* prices for recency.
    qlist = list(quotes)[-n:] if n < len(quotes) else list(quotes)
    prices = [q.last for q in qlist]
    m = len(prices)

    if m < 2:
        return MarketRegime.CHOPPY, 0.0

    # ---- 1. Linear regression slope -----------------------------------------
    slope = _compute_slope(prices)

    # ---- 2. Price efficiency ------------------------------------------------
    efficiency = _compute_efficiency(prices)

    # ---- 3. Classify regime -------------------------------------------------
    if efficiency > 0.4:
        regime = MarketRegime.TRENDING
        # Confidence: combine efficiency (0.4–1.0) and slope magnitude.
        confidence = min(1.0, 0.5 + efficiency * 0.5)
    elif efficiency >= 0.2:
        regime = MarketRegime.RANGING
        confidence = min(1.0, 0.4 + (efficiency - 0.2) / 0.2 * 0.3)
    else:
        regime = MarketRegime.CHOPPY
        # Confidence: inversion — lower efficiency → more confidently choppy.
        confidence = min(1.0, 1.0 - efficiency / 0.2 * 0.8)

    return regime, confidence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_slope(prices: list[float]) -> float:
    """Ordinary least-squares slope of *prices* against index 0..n-1.

    Returns:
        Slope (positive = uptrend, negative = downtrend, near-zero = flat).
    """
    n = len(prices)
    if n < 2:
        return 0.0

    # Use integer indices as x-values.
    x_mean = (n - 1) / 2.0
    y_mean = sum(prices) / n

    num = 0.0
    den = 0.0
    for i, y in enumerate(prices):
        dx = i - x_mean
        num += dx * (y - y_mean)
        den += dx * dx

    if den == 0:
        return 0.0
    return num / den


def _compute_efficiency(prices: list[float]) -> float:
    """Price efficiency ratio.

    .. math::

        E = \\frac{|P_n - P_0|}{\\sum_{i=1}^{n-1} |P_i - P_{i-1}|}

    Returns 0.0 when the denominator is zero (flat prices).
    """
    n = len(prices)
    if n < 2:
        return 0.0

    net_move = abs(prices[-1] - prices[0])
    path_length = 0.0
    for i in range(1, n):
        path_length += abs(prices[i] - prices[i - 1])

    if path_length == 0:
        # Flat prices → no movement at all → maximum efficiency (but no
        # trading opportunity).  Treat as ranging for safety.
        return 0.3

    return net_move / path_length
