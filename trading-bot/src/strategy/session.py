"""
Time-of-day trading session awareness.

Determines the active CME session (CT timezone) and returns per-session
parameter overrides for risk, confidence, and regime preference.

Session schedule (Central Time, approximate):
- Asian:      17:00–02:00 CT  — lower vol, tighter ranges
- European:   02:00–08:00 CT  — moderate vol, momentum develops
- US Morning: 08:00–11:00 CT  — highest vol, breakouts common
- US Aftern.: 11:00–15:30 CT  — mean reversion dominant
- Pre-Close:  15:30–16:00 CT  — stop trading, maintenance soon
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import pytz


class TradingSession(Enum):
    """Five CME futures trading sessions (Central Time)."""

    ASIAN = "asian"          # 5 PM – 2 AM CT
    EUROPEAN = "european"    # 2 AM – 8 AM CT
    US_MORNING = "us_am"     # 8 AM – 11 AM CT
    US_AFTERNOON = "us_pm"   # 11 AM – 3:30 PM CT
    PRE_CLOSE = "pre_close"  # 3:30 PM – 4 PM CT — no trading


# ---------------------------------------------------------------------------
# Session parameters
# ---------------------------------------------------------------------------

SESSION_PARAMS: dict[TradingSession, dict] = {
    TradingSession.ASIAN: {
        "risk_multiplier": 0.5,
        "min_confidence": 0.75,
        "prefer_regime": "ranging",
    },
    TradingSession.EUROPEAN: {
        "risk_multiplier": 0.75,
        "min_confidence": 0.65,
        "prefer_regime": None,
    },
    TradingSession.US_MORNING: {
        "risk_multiplier": 1.0,
        "min_confidence": 0.60,
        "prefer_regime": "trending",
    },
    TradingSession.US_AFTERNOON: {
        "risk_multiplier": 0.75,
        "min_confidence": 0.65,
        "prefer_regime": "ranging",
    },
    TradingSession.PRE_CLOSE: {
        "risk_multiplier": 0.0,
        "min_confidence": 1.0,
        "prefer_regime": None,
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_current_session(
    now: datetime | None = None,
) -> TradingSession:
    """Determine the active CME trading session from the current CT time.

    Args:
        now: Timezone-aware datetime to use for the lookup.  When ``None``,
            the current wall-clock time is used (UTC, converted to CT).

    Returns:
        The matching :class:`TradingSession`.
    """
    central = pytz.timezone("America/Chicago")

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        # Naive datetime → treat as UTC for safety.
        now = now.replace(tzinfo=timezone.utc)

    ct_time = now.astimezone(central)
    hour = ct_time.hour
    minute = ct_time.minute
    # Decimal hour for easier comparison.
    time_val = hour + minute / 60.0

    if 17.0 <= time_val < 24.0 or 0.0 <= time_val < 2.0:
        return TradingSession.ASIAN
    elif 2.0 <= time_val < 8.0:
        return TradingSession.EUROPEAN
    elif 8.0 <= time_val < 11.0:
        return TradingSession.US_MORNING
    elif 11.0 <= time_val < 15.5:
        return TradingSession.US_AFTERNOON
    else:  # 15.5 (3:30 PM) – 17.0 (5:00 PM)
        return TradingSession.PRE_CLOSE


def get_session_params(session: TradingSession) -> dict:
    """Return per-session parameter overrides.

    The returned dictionary always contains:

    * ``risk_multiplier`` — multiply ``risk_per_trade`` by this factor
    * ``min_confidence`` — adjusted minimum confidence threshold
    * ``prefer_regime`` — preferred market regime (string or ``None``)

    Args:
        session: The trading session to look up.

    Returns:
        Dict with keys ``risk_multiplier``, ``min_confidence``,
        ``prefer_regime``.
    """
    return SESSION_PARAMS[session]
