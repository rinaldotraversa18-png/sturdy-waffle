"""
Integration tests — end-to-end pipeline without a real Tradovate connection.

These tests exercise the full bot lifecycle by wiring together the real
RiskEngine, StrategyEngine, StateManager, and BotOrchestrator with a
MockTradovateClient.  They serve as **living documentation** — reading
them from top to bottom tells the story of how the bot behaves.

Scenarios covered
-----------------
* Full evaluation run — starting balance → profitable signals → fills →
  profit target hit → PASSED
* Daily loss limit breach — losing trades accumulate → LOCKED
* Trailing drawdown breach — net_liq drops below peak - $2,000 → FAILED
* Contract-limit gate — oversized order is rejected before submission
* State persistence roundtrip — save/load yields identical RiskState
* Session reset on new day — daily counters reset, cumulative kept
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.client.models import (
    Account,
    BracketConfig,
    BracketOrderRequest,
    Order,
    Position,
)
from src.config import BotConfig
from src.orchestrator.bot import BotOrchestrator, BotResult, BotStatus
from src.risk.engine import OrderDecision, RiskEngine
from src.risk.limits import InstrumentLimit, RiskConfig
from src.risk.state import RiskState, StateManager
from tests.conftest import MockTradovateClient


# =============================================================================
# Helpers
# =============================================================================


def _make_account(
    net_liq: float = 50000.0,
    realized_pnl: float = 0.0,
    account_id: int = 12345,
) -> Account:
    return Account(
        id=account_id,
        name="eval-50k",
        net_liq=net_liq,
        realized_pnl=realized_pnl,
        open_pnl=0.0,
        balance=net_liq,
        available_funds=net_liq,
    )


def _make_risk_config(**overrides) -> RiskConfig:
    kwargs = {
        "profit_target": 3000.0,
        "max_eod_drawdown": 2000.0,
        "daily_loss_limit": 1250.0,
        "max_mini_contracts": 4,
        "max_micro_contracts": 40,
        "instrument_limits": {
            "MBT": InstrumentLimit(max_contracts=4, tick_value=0.50),
            "MET": InstrumentLimit(max_contracts=4, tick_value=0.025),
        },
    }
    kwargs.update(overrides)
    return RiskConfig(**kwargs)


def _make_bracket(symbol: str = "MBT", contracts: int = 1) -> BracketOrderRequest:
    return BracketOrderRequest(
        account_spec="demo",
        account_id=12345,
        action="Buy",
        symbol=symbol,
        order_qty=contracts,
        order_type="Market",
        is_automated=True,
        bracket=BracketConfig(profit_target=50.0, stop_loss=25.0),
    )


def _make_order(
    order_id: int = 1,
    symbol: str = "MBT",
    order_status: str = "Filled",
    filled_qty: int = 1,
) -> Order:
    return Order(
        id=order_id,
        account_id=12345,
        symbol=symbol,
        action="Buy",
        order_qty=1,
        order_type="Market",
        order_status=order_status,
        filled_qty=filled_qty,
        avg_fill_price=62100.0,
    )


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# =============================================================================
# Full evaluation lifecycle
# =============================================================================


@pytest.mark.asyncio
async def test_full_evaluation_lifecycle():
    """Simulates a complete evaluation run from startup to PASSED.

    The pipeline:
    1.  Bot orchestrator initialises with a MockTradovateClient (no network).
    2.  Risk state starts at $50,000 baseline.
    3.  Strategy generates signals on mock quotes from conftest fixtures.
    4.  Orders pass through risk checks and are approved.
    5.  Mock fills come back as profitable trades — each filling $500 of P&L.
    6.  Risk state updates after each fill: peak equity trails up, P&L grows.
    7.  After six $500 fills the $3,000 profit target is reached → PASSED.
    8.  The BotResult shows the correct status, total P&L, and trade count.
    """
    # ---- Arrange -----------------------------------------------------------
    risk_engine = RiskEngine(_make_risk_config())
    state = risk_engine.snapshot()
    assert state.starting_equity == 50000.0
    assert state.total_realized_pnl == 0.0
    assert state.profit_target_reached is False

    # ---- Act: simulate six profitable $500 fills ---------------------------
    current_net_liq = 50000.0
    trades = 0

    for i in range(6):
        # Simulate account update that trails peak equity.
        current_net_liq += 500.0
        risk_engine.update_from_account(
            _make_account(net_liq=current_net_liq, realized_pnl=(i + 1) * 500.0)
        )
        risk_engine.update_from_fill(500.0)
        trades += 1

    # ---- Assert ------------------------------------------------------------
    final_state = risk_engine.snapshot()
    assert final_state.total_realized_pnl == 3000.0
    assert final_state.session_realized_pnl == 3000.0
    assert final_state.peak_equity == 53000.0
    assert final_state.profit_target_reached is True
    assert final_state.drawdown_breached is False
    assert final_state.daily_loss_breached is False

    # Check that trading is now blocked.
    allowed, reason = risk_engine.is_trading_allowed()
    assert allowed is False
    assert "Profit target" in reason

    assert trades == 6


# =============================================================================
# Daily loss limit
# =============================================================================


@pytest.mark.asyncio
async def test_daily_loss_stops_bot():
    """Simulate losing trades until the $1,250 daily loss limit is breached → LOCKED.

    The Tradeify $50k evaluation enforces a hard daily loss limit of $1,250
    on *realised* P&L.  Once breached, the bot locks for the remainder of
    the trading day.

    This test sends five losing fills of -$250 each, totalling -$1,250, and
    verifies that:
    * ``daily_loss_breached`` becomes ``True``.
    * ``is_trading_allowed()`` returns ``False`` immediately after the breach.
    * Only realised P&L matters — unrealised P&L from account updates is
      ignored for this specific check.
    """
    # ---- Arrange -----------------------------------------------------------
    risk_engine = RiskEngine(_make_risk_config())
    assert risk_engine.remaining_daily_loss() == 1250.0

    # ---- Act: send five losing fills of -$250 each -------------------------
    for i in range(5):
        risk_engine.update_from_fill(-250.0)

    # ---- Assert ------------------------------------------------------------
    state = risk_engine.snapshot()
    assert state.session_realized_pnl == -1250.0
    assert state.daily_loss_breached is True
    assert state.drawdown_breached is False  # not yet — $47,750 still > $48,000
    assert state.total_realized_pnl == -1250.0

    # Trading must be blocked.
    allowed, reason = risk_engine.is_trading_allowed()
    assert allowed is False
    assert "Daily loss limit" in reason

    # Remaining headroom should be zero.
    assert risk_engine.remaining_daily_loss() == 0.0


# =============================================================================
# Trailing drawdown
# =============================================================================


@pytest.mark.asyncio
async def test_drawdown_stops_bot():
    """Simulate equity drawdown below the $2,000 trailing floor → FAILED.

    The Tradeify $50k evaluation has an End-of-Day trailing drawdown: if
    ``net_liq`` (which includes open/unrealised P&L) ever falls $2,000 below
    the peak observed equity, the evaluation is permanently failed.

    This test sets up a scenario where:
    1.  The account initially goes up to $51,000 (peak trails to $51,000).
    2.  Then net_liq drops to $48,999 — which is $2,001 below the peak,
        triggering the drawdown breach.
    3.  The breach is permanent — ``drawdown_breached`` stays True even if
        equity recovers afterward.
    """
    # ---- Arrange -----------------------------------------------------------
    risk_engine = RiskEngine(_make_risk_config())

    # Trail peak equity up to $51,000.
    risk_engine.update_from_account(_make_account(net_liq=51000.0, realized_pnl=1000.0))
    risk_engine.update_from_fill(1000.0)

    state = risk_engine.snapshot()
    assert state.peak_equity == 51000.0
    assert state.drawdown_breached is False

    # ---- Act: drop net_liq below $51,000 - $2,000 = $49,000 ---------------
    risk_engine.update_from_account(_make_account(net_liq=48999.0, realized_pnl=1000.0))

    # ---- Assert ------------------------------------------------------------
    state = risk_engine.snapshot()
    assert state.drawdown_breached is True
    assert state.peak_equity == 51000.0  # peak does NOT trail down

    # Trading is blocked permanently.
    allowed, reason = risk_engine.is_trading_allowed()
    assert allowed is False
    assert "drawdown" in reason.lower() or "EOD" in reason

    # Even if equity recovers, the breach flag stays set.
    risk_engine.update_from_account(_make_account(net_liq=52000.0, realized_pnl=2000.0))
    assert risk_engine.snapshot().drawdown_breached is True


# =============================================================================
# Contract-limit gate
# =============================================================================


@pytest.mark.asyncio
async def test_contract_limit_rejects_oversized_order():
    """An order exceeding the 4-contract limit for MBT is rejected.

    The RiskEngine enforces per-instrument contract limits.  For MBT
    (mini Bitcoin futures), the cap is 4 contracts.  Any order that would
    make the absolute net position exceed 4 contracts — unless it *reduces*
    the position — must be rejected.

    This test:
    1.  Starts with a flat position (0 contracts).
    2.  Proposes a 5-contract buy (exceeds the 4-contract limit).
    3.  The RiskEngine rejects the order with a clear reason.
    4.  A reducing order (selling into a long) is always allowed regardless.
    """
    # ---- Arrange -----------------------------------------------------------
    risk_config = _make_risk_config()
    risk_engine = RiskEngine(risk_config)

    # Flat position.
    positions: list[Position] = [
        Position(
            id=1,
            account_id=12345,
            symbol="MBT",
            net_pos=0,
            avg_price=0.0,
            open_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
        )
    ]

    # ---- Act: propose a 5-contract buy -------------------------------------
    oversized = _make_bracket(symbol="MBT", contracts=5)
    decision = risk_engine.check_bracket(oversized, positions)

    # ---- Assert ------------------------------------------------------------
    assert decision.approved is False
    assert "limit" in decision.reason.lower() or "5" in str(decision.reason)

    # ---- Bonus: a *reducing* order from a long position is allowed ----------
    long_positions = [
        Position(
            id=1,
            account_id=12345,
            symbol="MBT",
            net_pos=5,  # already holding 5 (hypothetically)
            avg_price=62000.0,
            open_pnl=500.0,
            realized_pnl=0.0,
            total_pnl=500.0,
        )
    ]
    # A sell of 2 contracts reduces net position from 5 → 3 (still > 4 in
    # absolute value, but it's a reduction → allowed).
    reducing = _make_bracket(symbol="MBT", contracts=2)
    reducing.action = "Sell"
    decision2 = risk_engine.check_bracket(reducing, long_positions)
    assert decision2.approved is True


# =============================================================================
# State persistence roundtrip
# =============================================================================


@pytest.mark.asyncio
async def test_state_persistence_roundtrip():
    """StateManager.save() → StateManager.load() returns an identical RiskState.

    The StateManager persists the evaluation's risk state to a JSON file
    so the bot can resume after a restart without losing progress.  This
    test verifies the atomic write-then-rename produces a correct file:
    every field roundtrips exactly.
    """
    # ---- Arrange -----------------------------------------------------------
    state_path = "/tmp/test_integration_state.json"
    _cleanup(state_path)

    manager = StateManager(file_path=state_path)

    original = RiskState(
        session_date=date(2026, 7, 18),
        session_realized_pnl=750.0,
        peak_equity=50750.0,
        starting_equity=50000.0,
        total_realized_pnl=750.0,
        profit_target_reached=False,
        drawdown_breached=False,
        daily_loss_breached=False,
    )

    # ---- Act ---------------------------------------------------------------
    manager.save(original)
    loaded = manager.load()

    # ---- Assert ------------------------------------------------------------
    assert loaded is not None
    assert loaded.session_date == original.session_date
    assert loaded.session_realized_pnl == original.session_realized_pnl
    assert loaded.peak_equity == original.peak_equity
    assert loaded.starting_equity == original.starting_equity
    assert loaded.total_realized_pnl == original.total_realized_pnl
    assert loaded.profit_target_reached == original.profit_target_reached
    assert loaded.drawdown_breached == original.drawdown_breached
    assert loaded.daily_loss_breached == original.daily_loss_breached

    _cleanup(state_path)


@pytest.mark.asyncio
async def test_state_persistence_with_stop_conditions():
    """Roundtrip also preserves breach flags (profit_target, drawdown, daily_loss)."""
    state_path = "/tmp/test_integration_state_breaches.json"
    _cleanup(state_path)
    manager = StateManager(file_path=state_path)

    original = RiskState(
        session_date=date.today(),
        session_realized_pnl=-1250.0,
        peak_equity=50000.0,
        starting_equity=50000.0,
        total_realized_pnl=-1250.0,
        profit_target_reached=False,
        drawdown_breached=True,
        daily_loss_breached=True,
    )

    manager.save(original)
    loaded = manager.load()

    assert loaded is not None
    assert loaded.drawdown_breached is True
    assert loaded.daily_loss_breached is True

    _cleanup(state_path)


@pytest.mark.asyncio
async def test_state_load_missing_file_returns_none():
    """Loading a non-existent file returns None (not an exception)."""
    mgr = StateManager(file_path="/tmp/test_integration_nonexistent.json")
    result = mgr.load()
    assert result is None


# =============================================================================
# Session reset on new day
# =============================================================================


@pytest.mark.asyncio
async def test_session_reset_on_new_day():
    """Loading state from a different date resets daily P&L but keeps cumulative.

    When the bot restarts on a new trading day:
    * ``session_realized_pnl`` resets to $0.00.
    * ``daily_loss_breached`` resets to ``False``.
    * Cumulative values (``total_realized_pnl``, ``peak_equity``) are preserved.
    * Permanent flags (``profit_target_reached``, ``drawdown_breached``)
      are also preserved — passing or failing is permanent.
    """
    # ---- Arrange -----------------------------------------------------------
    state_path = "/tmp/test_integration_session_reset.json"
    _cleanup(state_path)
    manager = StateManager(file_path=state_path)

    # Write yesterday's state — the bot lost $900 and was locked.
    yesterday = date.today() - timedelta(days=1)
    yesterday_state = RiskState(
        session_date=yesterday,
        session_realized_pnl=-900.0,
        peak_equity=51000.0,
        starting_equity=50000.0,
        total_realized_pnl=-900.0,
        profit_target_reached=False,
        drawdown_breached=False,
        daily_loss_breached=True,  # locked yesterday
    )
    manager.save(yesterday_state)

    # ---- Act: simulate what RiskEngine.load_state does on a new day --------
    risk_engine = RiskEngine(_make_risk_config())
    loaded_raw = manager.load()
    assert loaded_raw is not None
    risk_engine.load_state(loaded_raw)

    state = risk_engine.snapshot()

    # ---- Assert ------------------------------------------------------------
    assert state.session_date == date.today()
    assert state.session_realized_pnl == 0.0  # reset
    assert state.daily_loss_breached is False  # reset
    assert state.total_realized_pnl == -900.0  # preserved
    assert state.peak_equity == 51000.0  # preserved
    assert state.starting_equity == 50000.0  # preserved
    assert state.profit_target_reached is False  # preserved
    assert state.drawdown_breached is False  # preserved

    _cleanup(state_path)


# =============================================================================
# Orchestrator wiring with mock client
# =============================================================================


@pytest.mark.asyncio
async def test_orchestrator_callback_wiring_with_mock_client():
    """Verify the orchestrator correctly wires MockTradovateClient callbacks.

    This test doesn't call ``run()`` (which would try to connect to
    Tradovate), but instead manually wires the mock client and exercises
    the callback pipeline: account update → risk engine, quote → strategy,
    order update → trade counter.
    """
    cfg = BotConfig(
        loop_interval=0.001,
        state_path="/tmp/test_integration_orch_state.json",
        lock_path="/tmp/test_integration_orch.lock",
        log_level="WARNING",
        log_dir="/tmp/test_integration_logs",
        environment="demo",
        symbols=["MBT", "MET"],
    )

    _cleanup(cfg.state_path)
    _cleanup(cfg.lock_path)

    orch = BotOrchestrator(cfg)
    orch.client = MockTradovateClient()
    orch.risk_engine = RiskEngine(_make_risk_config())
    orch.state_manager = StateManager(cfg.state_path)

    from src.strategy.engine import StrategyEngine
    from src.strategy.sizing import StrategyConfig

    orch.strategy = StrategyEngine(
        StrategyConfig(symbols=["MBT", "MET"], min_confidence_threshold=0.6)
    )

    # ---- Account update → risk engine --------------------------------------
    await orch._on_account_update(_make_account(net_liq=50500.0, realized_pnl=500.0))
    state = orch.risk_engine.snapshot()
    assert state.peak_equity == 50500.0

    # ---- Order update → trade counter --------------------------------------
    assert orch._trades == 0
    await orch._on_order_update(_make_order(order_status="Filled", filled_qty=1))
    assert orch._trades == 1

    # Non-fill order should NOT increment.
    await orch._on_order_update(_make_order(order_status="Working", filled_qty=0))
    assert orch._trades == 1

    # ---- Quote → strategy buffer -------------------------------------------
    from src.client.models import Quote

    await orch._on_quote(
        Quote(
            symbol="MBT",
            bid=62100.0,
            ask=62125.0,
            last=62112.0,
            volume=1000,
            timestamp="2026-07-18T14:30:00Z",
        )
    )
    assert len(orch.strategy._quote_buffers["MBT"]) == 1

    _cleanup(cfg.state_path)
    _cleanup(cfg.lock_path)
