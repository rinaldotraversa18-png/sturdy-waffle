"""
Volatility filter — ATR-based safety gate for the strategy engine.

Computes Average True Range from the quote buffer and compares current
volatility against historical levels.  Blocks trading when volatility
exceeds account-safety thresholds.
"""

from __future__ import annotations

from collections import deque

from src.client.models import Account, Quote


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_atr(quotes: deque[Quote], period: int = 14) -> float:
    """Average True Range computed from the last *period* quotes.

    Uses ``bid`` / ``ask`` spread as a volatility proxy when true high/low
    data is not available in the Quote model.  The "true range" for each
    tick is approximated as::

        TR = max(ask - bid, |last - prev_last|)

    Args:
        quotes: Rolling quote buffer (at least *period* + 1 elements).
        period: Lookback window (default 14).

    Returns:
        ATR value in price units, or 0.0 when insufficient data.
    """
    if len(quotes) < period + 1:
        return 0.0

    qlist = list(quotes)[-period - 1:]
    true_ranges: list[float] = []

    for i in range(1, len(qlist)):
        prev = qlist[i - 1]
        curr = qlist[i]

        # True Range proxies when high/low are not available:
        # 1. Bid/ask spread magnitude
        # 2. Tick-to-tick absolute change
        spread_range = abs(curr.ask - curr.bid)
        tick_range = abs(curr.last - prev.last)
        tr = max(spread_range, tick_range)
        true_ranges.append(tr)

    if not true_ranges:
        return 0.0

    # Initial ATR = simple average, then Wilder's smoothing.
    atr = sum(true_ranges) / len(true_ranges)

    # For a rolling estimate we use the simple average; Wilder's smoothing
    # requires a much longer history and is equivalent for first-pass.
    return atr


def compute_volatility_ratio(
    current_atr: float,
    historical_atr: deque[float],
) -> float:
    """Ratio of current ATR to median of historical ATR values.

    Interpretation
    --------------
    - ``> 1.5`` → high volatility (relative to recent history).
    - ``< 0.5`` → low volatility.
    - ``0.5–1.5`` → normal.

    Args:
        current_atr: Latest ATR value.
        historical_atr: Rolling history of ATR values (at least 1 entry).

    Returns:
        Volatility ratio (current / median).  Returns **1.0** when
        *historical_atr* is empty as a neutral fallback.
    """
    if not historical_atr or len(historical_atr) == 0:
        return 1.0

    # Compute median from sorted history.
    sorted_atr = sorted(historical_atr)
    n = len(sorted_atr)
    if n % 2 == 1:
        median = sorted_atr[n // 2]
    else:
        median = (sorted_atr[n // 2 - 1] + sorted_atr[n // 2]) / 2.0

    if median == 0:
        return 1.0

    return current_atr / median


def is_safe_to_trade(
    volatility_ratio: float,
    current_atr: float,
    account: Account,
    max_volatility_ratio: float = 1.5,
    min_volatility_ratio: float = 0.5,
) -> tuple[bool, str]:
    """Determine whether market volatility is safe for trading.

    Checks
    ------
    1. Volatility ratio > *max_volatility_ratio* → blocked
       (reason: ``"volatility too high, sitting out"``).
    2. ATR > 1% of account ``net_liq`` → blocked
       (reason: ``"volatility exceeds account safety threshold"``).
    3. Both checks pass → ``(True, "")``.

    Args:
        volatility_ratio: Current ATR / median historical ATR.
        current_atr: Latest ATR in price units.
        account: Current account snapshot (for net_liq safety check).
        max_volatility_ratio: Upper bound for acceptable vol ratio.
        min_volatility_ratio: Lower bound (low vol is okay but noted).

    Returns:
        ``(is_safe: bool, reason: str)``.
    """
    # Check 1: high volatility.
    if volatility_ratio > max_volatility_ratio:
        return False, "volatility too high, sitting out"

    # Check 2: ATR relative to account equity.
    if account.net_liq > 0:
        atr_pct = current_atr / account.net_liq
        if atr_pct > 0.01:  # 1% of account value
            return False, "volatility exceeds account safety threshold"

    # Low vol is fine — just weaker signals; the caller can adjust.
    return True, ""
