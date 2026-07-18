"""
Trailing stop management for funded account evaluations.

Since Tradovate supports bracket orders with fixed TP/SL, we implement
trailing by **modifying existing stop orders** as price moves favorably.

This module provides the decision logic for:
- When to activate trailing (price must move ``activation_pct`` toward TP)
- Where to place the trail stop relative to current price
- When an update is warranted (``step_ticks`` minimum movement to avoid churn)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrailingConfig:
    """Parameters controlling trailing stop behavior.

    Attributes:
        activation_pct: Fraction of target move required before trailing
            begins (e.g. 0.3 → 30% toward TP).
        trail_distance_ticks: How many ticks behind current price the
            stop trails once activated.
        step_ticks: Minimum price movement (in ticks) before the stop is
            actually adjusted.  Prevents churn from tiny wiggles.
    """

    activation_pct: float = 0.3
    trail_distance_ticks: int = 20
    step_ticks: int = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_activate_trail(
    entry_price: float,
    current_price: float,
    target_price: float,
    config: TrailingConfig,
) -> bool:
    """Return ``True`` once price has moved far enough toward the target.

    The condition::

        distance_moved / distance_to_target >= activation_pct

    where ``distance_moved`` is the *favorable* move from entry and
    ``distance_to_target`` is the total journey to the take-profit level.

    Args:
        entry_price: Fill price of the position.
        current_price: Most recent price (bid for longs, ask for shorts).
        target_price: Take-profit price.
        config: Trailing stop parameters.

    Returns:
        ``True`` if trailing should be activated, ``False`` otherwise.
    """
    total_distance = abs(target_price - entry_price)
    if total_distance <= 0:
        return False

    favorable_move = _favorable_move(entry_price, current_price, target_price)
    if favorable_move <= 0:
        return False

    return (favorable_move / total_distance) >= config.activation_pct


def compute_trail_stop(
    current_price: float,
    direction: Literal["long", "short"],
    trail_distance_ticks: int,
    tick_size: float,
    instrument_config: dict | None = None,
) -> float:
    """Calculate where the trail stop should sit right now.

    For longs the stop trails **below** price; for shorts it trails
    **above** price.  The distance is ``trail_distance_ticks × tick_size``.

    Args:
        current_price: The most recent favorable price reference
            (bid for longs, ask for shorts).
        direction: ``"long"`` or ``"short"``.
        trail_distance_ticks: Number of ticks of breathing room.
        tick_size: Dollar value of one tick for the instrument.
        instrument_config: Optional dict with ``tick_size`` override.

    Returns:
        The new stop price.
    """
    # Allow tick_size override from instrument_config
    if instrument_config and "tick_size" in instrument_config:
        tick_size = float(instrument_config["tick_size"])

    distance = trail_distance_ticks * tick_size

    if direction == "long":
        return current_price - distance
    else:
        return current_price + distance


def should_update_trail(
    current_stop: float,
    new_stop: float,
    direction: Literal["long", "short"],
    step_ticks: int,
    tick_size: float,
) -> bool:
    """Decide whether the stop has moved enough to warrant an update.

    For longs the stop ratchets **up** — an update is warranted when
    ``new_stop > current_stop`` by at least ``step_ticks × tick_size``.
    For shorts the stop ratchets **down** — an update is warranted when
    ``new_stop < current_stop`` by at least ``step_ticks × tick_size``.

    Args:
        current_stop: Price of the currently placed stop order.
        new_stop: Candidate stop price computed from latest market.
        direction: ``"long"`` or ``"short"``.
        step_ticks: Minimum ticks the stop must move to trigger an update.
        tick_size: Dollar value of one tick for this instrument.

    Returns:
        ``True`` if the stop should be modified, ``False`` otherwise.
    """
    min_move = step_ticks * tick_size

    if direction == "long":
        # Stop should move UP for longs — only update if new stop is
        # significantly higher than current.
        return (new_stop - current_stop) >= min_move
    else:
        # Stop should move DOWN for shorts — only update if new stop is
        # significantly lower than current.
        return (current_stop - new_stop) >= min_move


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _favorable_move(
    entry_price: float,
    current_price: float,
    target_price: float,
) -> float:
    """Return the absolute favorable price move from entry."""
    # Determine direction from entry → target.
    if target_price > entry_price:
        # Long trade: favorable move = current - entry (only if positive).
        return max(0.0, current_price - entry_price)
    else:
        # Short trade: favorable move = entry - current (only if positive).
        return max(0.0, entry_price - current_price)
