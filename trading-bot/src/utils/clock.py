"""
Trading-hours clock with CT timezone awareness.

All time checks use **US Central Time** because CME futures and
Tradovate follow the CME Globex schedule:

- **Sunday** 17:00 CT – **Friday** 16:00 CT: open
- **Daily maintenance** 16:00–17:00 CT: closed

The 1-hour maintenance window happens every weekday (Mon–Thu) and on
Sunday there is no maintenance because the market *opens* at 17:00.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import zoneinfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CT_TZ: zoneinfo.ZoneInfo = zoneinfo.ZoneInfo("America/Chicago")

_MAINTENANCE_START_HOUR: int = 16  # 4 PM CT
_MAINTENANCE_END_HOUR: int = 17  # 5 PM CT

_MARKET_OPEN_WEEKDAY: int = 0  # Monday
_MARKET_CLOSE_WEEKDAY: int = 4  # Friday
_MARKET_CLOSE_HOUR: int = 16  # 4 PM CT
_MARKET_OPEN_HOUR: int = 17  # 5 PM CT (Sunday)
_MARKET_OPEN_WEEKEND_DAY: int = 6  # Sunday


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_current_ct_time() -> datetime:
    """Return the current time in US Central Time."""
    return datetime.now(timezone.utc).astimezone(_CT_TZ)


def is_market_open() -> bool:
    """Return ``True`` if the CME Globex market is currently open for trading.

    Rules:

    * Sunday 17:00 CT through Friday 16:00 CT the market is open.
    * Monday–Thursday 16:00–17:00 CT is the daily maintenance window
      (market closed).
    * Friday 16:00 CT through Sunday 17:00 CT the market is closed
      for the weekend.
    """
    now = get_current_ct_time()
    wd = now.weekday()  # Monday=0, Sunday=6
    hour = now.hour

    # Weekend: Friday 16:00+ or Saturday
    if wd == _MARKET_CLOSE_WEEKDAY and hour >= _MARKET_CLOSE_HOUR:
        return False
    if wd == 5:  # Saturday
        return False
    # Sunday before 17:00
    if wd == _MARKET_OPEN_WEEKEND_DAY and hour < _MARKET_OPEN_HOUR:
        return False

    # Daily maintenance: Mon–Thu 16:00–16:59
    if (
        _MARKET_OPEN_WEEKDAY <= wd <= _MARKET_CLOSE_WEEKDAY
        and _MAINTENANCE_START_HOUR <= hour < _MAINTENANCE_END_HOUR
    ):
        return False

    return True


def next_session_start() -> datetime:
    """Return the datetime of the next market-open event.

    If we are currently within a session:
        returns the next maintenance end (or next day's open after maintenance).
    If we are in the weekend:
        returns Sunday 17:00 CT.
    If we are in daily maintenance:
        returns today at 17:00 CT.
    """
    now = get_current_ct_time()
    wd = now.weekday()

    # If we're in daily maintenance (Mon–Thu 16:00–16:59), return today 17:00.
    if (
        _MARKET_OPEN_WEEKDAY <= wd <= _MARKET_CLOSE_WEEKDAY
        and _MAINTENANCE_START_HOUR <= now.hour < _MAINTENANCE_END_HOUR
    ):
        return now.replace(hour=_MAINTENANCE_END_HOUR, minute=0, second=0, microsecond=0)

    # If the market is open now, the "next" session start is tomorrow at 17:00
    # (maintenance end) — unless it's Friday, then Sunday 17:00.
    if is_market_open():
        if wd == _MARKET_CLOSE_WEEKDAY:
            # Next session is Sunday 17:00
            days_until_sunday = (6 - wd) % 7 or 7
            nxt = now + timedelta(days=days_until_sunday)
            return nxt.replace(hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)
        # Otherwise: tomorrow 17:00 (after maintenance)
        nxt = now + timedelta(days=1)
        return nxt.replace(hour=_MAINTENANCE_END_HOUR, minute=0, second=0, microsecond=0)

    # Market is closed — figure out when it reopens.
    # Weekend: advance to Sunday 17:00.
    if wd >= _MARKET_CLOSE_WEEKDAY:  # Friday after 16:00, Saturday, Sunday < 17:00
        days_until_sunday = (_MARKET_OPEN_WEEKEND_DAY - wd) % 7 or 7
        nxt = now + timedelta(days=days_until_sunday)
        return nxt.replace(hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)

    # Weekday before 17:00 but market closed → next session is today 17:00.
    return now.replace(hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)


def seconds_until_market_close() -> float:
    """Return seconds until the next market-close event.

    During the week: Friday 16:00 CT or today's maintenance start (16:00 CT).
    If already closed: returns 0.0.
    """
    if not is_market_open():
        return 0.0

    now = get_current_ct_time()
    wd = now.weekday()
    today_16 = now.replace(hour=_MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)

    if wd == _MARKET_CLOSE_WEEKDAY:
        # Friday — closes at 16:00
        return max(0.0, (today_16 - now).total_seconds())

    # Mon–Thu — maintenance at 16:00
    return max(0.0, (today_16 - now).total_seconds())


def seconds_until_maintenance() -> float:
    """Return seconds until the next daily maintenance window (16:00 CT).

    Returns 0.0 if currently within maintenance or on a weekend.
    """
    if not is_market_open():
        return 0.0

    now = get_current_ct_time()
    wd = now.weekday()

    if wd == _MARKET_CLOSE_WEEKDAY:
        # Friday: closes at 16:00, no maintenance return needed separately
        today_16 = now.replace(hour=_MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)
        return max(0.0, (today_16 - now).total_seconds())

    # Mon–Thu
    today_16 = now.replace(hour=_MAINTENANCE_START_HOUR, minute=0, second=0, microsecond=0)
    return max(0.0, (today_16 - now).total_seconds())
