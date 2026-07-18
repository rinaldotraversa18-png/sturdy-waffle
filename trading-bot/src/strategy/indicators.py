"""
Pure-function technical indicators for the strategy engine.

Every function accepts standard Python collections (``deque``, ``list``,
``Sequence``) and returns a plain value — no side effects, no shared
state.  This makes the indicators trivially testable and easy to swap
out later for ML-based equivalents.

All price inputs are expected to be :class:`Quote` objects from
``src.client.models``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

from src.client.models import Quote


# ---------------------------------------------------------------------------
# Micro-price (bid/ask midpoint)
# ---------------------------------------------------------------------------

def compute_micro_price(bid: float, ask: float) -> float:
    """Return the bid/ask midpoint.

    Returns:
        ``(bid + ask) / 2.0``.

    Raises:
        ValueError: If *bid* > *ask* (crossed market).
    """
    if bid > ask:
        raise ValueError(f"Crossed market: bid {bid} > ask {ask}")
    return (bid + ask) / 2.0


# ---------------------------------------------------------------------------
# Rolling VWAP
# ---------------------------------------------------------------------------

def compute_vwap(quotes: Sequence[Quote]) -> float | None:
    """Volume-weighted average price over *quotes*.

    Uses each quote's ``last`` price weighted by the **incremental**
    volume contributed by that tick.  When ``volume`` is a cumulative
    value we estimate tick-volume as ``max(1, volume - prev_volume)``.

    Returns:
        VWAP as a float, or ``None`` when *quotes* is empty.
    """
    if not quotes:
        return None

    total_pv = 0.0
    total_vol = 0.0
    prev_volume = quotes[0].volume

    for i, q in enumerate(quotes):
        tick_vol = max(1, q.volume - prev_volume) if i > 0 else 1
        total_pv += q.last * tick_vol
        total_vol += tick_vol
        prev_volume = q.volume

    return total_pv / total_vol if total_vol > 0 else None


# ---------------------------------------------------------------------------
# Rolling standard deviation
# ---------------------------------------------------------------------------

def compute_std_dev(
    quotes: Sequence[Quote],
    vwap: float | None = None,
) -> float | None:
    """Rolling standard deviation of ``last`` prices from their VWAP.

    Args:
        quotes: Sequence of quotes.
        vwap: Pre-computed VWAP.  When ``None``, VWAP is computed from
            *quotes* on the fly.

    Returns:
        Population standard deviation, or ``None`` when there are fewer
        than 2 quotes.
    """
    n = len(quotes)
    if n < 2:
        return None

    mean = vwap if vwap is not None else compute_vwap(quotes)
    if mean is None:
        return None

    sum_sq = 0.0
    for q in quotes:
        diff = q.last - mean
        sum_sq += diff * diff

    return math.sqrt(sum_sq / n)


# ---------------------------------------------------------------------------
# Volume spike detection
# ---------------------------------------------------------------------------

def detect_volume_spike(
    quotes: Sequence[Quote],
    multiplier: float = 2.0,
) -> bool:
    """Return ``True`` when the latest tick volume is > *multiplier* ×
    the rolling average tick volume.

    The latest tick is the **last** element of *quotes*.  The rolling
    average is computed over all preceding ticks (at least 1).

    Args:
        quotes: At least 2 quotes (we need a baseline).
        multiplier: Threshold ratio (default ``2.0`` → 2× average).

    Returns:
        ``True`` if the most recent tick is a volume spike.
    """
    if len(quotes) < 2:
        return False

    # Compute tick volumes.
    tick_vols: list[int] = []
    prev_vol = quotes[0].volume
    for i, q in enumerate(quotes):
        tv = max(1, q.volume - prev_vol) if i > 0 else 1
        tick_vols.append(tv)
        prev_vol = q.volume

    latest_vol = tick_vols[-1]
    # Average over all ticks *except* the latest for the baseline.
    baseline_vols = tick_vols[:-1]
    avg_vol = sum(baseline_vols) / len(baseline_vols) if baseline_vols else 1

    return avg_vol > 0 and (latest_vol / avg_vol) > multiplier


# ---------------------------------------------------------------------------
# Consecutive direction detection
# ---------------------------------------------------------------------------

def detect_consecutive_direction(
    quotes: Sequence[Quote],
    n: int = 5,
) -> bool:
    """Return ``True`` when the last *n* ticks moved in the same
    direction (all higher or all lower than the previous tick).

    Direction is determined by comparing each tick's ``last`` price to
    the previous tick's ``last`` price.

    Args:
        quotes: At least *n*+1 quotes.
        n: Number of consecutive same-direction ticks required.

    Returns:
        ``True`` if a directional run of length *n* is detected.
    """
    if len(quotes) < n + 1:
        return False

    # Look at the last n+1 prices to detect n consecutive moves.
    recent = list(quotes)[-(n + 1):]
    # Determine direction from the first pair.
    first_diff = recent[1].last - recent[0].last
    if first_diff == 0:
        return False

    direction = 1 if first_diff > 0 else -1

    for i in range(1, n):
        diff = recent[i + 1].last - recent[i].last
        if (direction > 0 and diff <= 0) or (direction < 0 and diff >= 0):
            return False

    return True


def get_consecutive_direction_sign(
    quotes: Sequence[Quote],
    n: int = 5,
) -> int:
    """Return +1 for consecutive up-ticks, -1 for consecutive
    down-ticks, or 0 when the condition is not met.

    Args:
        quotes: At least *n*+1 quotes.
        n: Number of consecutive same-direction ticks.

    Returns:
        +1, -1, or 0.
    """
    if len(quotes) < n + 1:
        return 0

    recent = list(quotes)[-(n + 1):]
    first_diff = recent[1].last - recent[0].last
    if first_diff == 0:
        return 0

    direction = 1 if first_diff > 0 else -1

    for i in range(1, n):
        diff = recent[i + 1].last - recent[i].last
        if (direction > 0 and diff <= 0) or (direction < 0 and diff >= 0):
            return 0

    return direction


# ---------------------------------------------------------------------------
# Spread filtering
# ---------------------------------------------------------------------------

def is_spread_eligible(bid: float, ask: float, max_spread_bps: float = 5.0) -> bool:
    """Return ``True`` when the spread is within acceptable bounds.

    Args:
        bid: Best bid.
        ask: Best ask.
        max_spread_bps: Maximum spread in basis points.

    Returns:
        ``True`` if the spread ≤ *max_spread_bps*.
    """
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return False
    spread = ask - bid
    spread_bps = (spread / mid) * 10_000.0
    return spread_bps <= max_spread_bps
