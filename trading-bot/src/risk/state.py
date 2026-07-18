"""
RiskState dataclass and StateManager for persisting evaluation progress.

The StateManager provides atomic read/write operations so the bot can
resume safely after a restart without corrupting state.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass
class RiskState:
    """Snapshot of the evaluation's current risk state.

    All monetary values are in dollars.  The session_date tracks the
    trading day the state belongs to, so the bot can detect a new day
    and reset daily counters automatically.

    Attributes:
        session_date: Trading day this state belongs to.
        session_realized_pnl: Cumulative **realized** P&L for the current
            trading session (resets daily).  Unrealised / open P&L is
            *not* included.
        peak_equity: Highest ``net_liq`` observed since the evaluation
            started.  Trails upward only — it is the watermark used for
            the EOD trailing drawdown calculation.
        starting_equity: ``net_liq`` at evaluation start (typically
            $50,000 for the Tradeify $50k account).
        total_realized_pnl: Cumulative realised P&L across **all**
            sessions since the evaluation began.
        profit_target_reached: ``True`` when ``total_realized_pnl >=
            profit_target``.
        drawdown_breached: ``True`` when ``net_liq`` falls below
            ``peak_equity - max_eod_drawdown``.
        daily_loss_breached: ``True`` when ``session_realized_pnl <=
            -daily_loss_limit``.
    """

    session_date: date
    session_realized_pnl: float = 0.0
    peak_equity: float = 50000.0
    starting_equity: float = 50000.0
    total_realized_pnl: float = 0.0
    profit_target_reached: bool = False
    drawdown_breached: bool = False
    daily_loss_breached: bool = False


class StateManager:
    """Atomic JSON file persistence for :class:`RiskState`.

    Uses *write-to-temp + atomic-rename* to avoid corruption on crash.
    The file path defaults to ``<project_root>/state.json`` but can be
    overridden via the constructor.

    Typical usage::

        mgr = StateManager()
        mgr.save(state)
        loaded = mgr.load()      # None if file missing
        mgr.clear()              # delete state file
    """

    def __init__(self, file_path: Optional[str] = None) -> None:
        if file_path is None:
            # Resolve relative to the *project* root (trading-bot/)
            project_root = Path(__file__).resolve().parents[2]
            file_path = str(project_root / "state.json")
        self._path = Path(file_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state: RiskState) -> None:
        """Persist *state* atomically.

        Writes JSON to a temporary file in the same directory and then
        atomically renames it onto the target path.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "session_date": state.session_date.isoformat(),
            "session_realized_pnl": state.session_realized_pnl,
            "peak_equity": state.peak_equity,
            "starting_equity": state.starting_equity,
            "total_realized_pnl": state.total_realized_pnl,
            "profit_target_reached": state.profit_target_reached,
            "drawdown_breached": state.drawdown_breached,
            "daily_loss_breached": state.daily_loss_breached,
        }

        # Write to a temp file in the *same* directory so rename is atomic.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent),
            prefix=".state.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, self._path)  # atomic on POSIX
        except Exception:
            # Clean up the temp file on any error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> Optional[RiskState]:
        """Load persisted state, returning ``None`` when no file exists."""
        if not self._path.exists():
            return None

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # corrupted / unreadable → treat as missing

        try:
            return RiskState(
                session_date=date.fromisoformat(raw["session_date"]),
                session_realized_pnl=float(raw["session_realized_pnl"]),
                peak_equity=float(raw["peak_equity"]),
                starting_equity=float(raw["starting_equity"]),
                total_realized_pnl=float(raw["total_realized_pnl"]),
                profit_target_reached=bool(raw.get("profit_target_reached", False)),
                drawdown_breached=bool(raw.get("drawdown_breached", False)),
                daily_loss_breached=bool(raw.get("daily_loss_breached", False)),
            )
        except (KeyError, ValueError, TypeError):
            return None

    def clear(self) -> None:
        """Delete the state file if it exists."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass
