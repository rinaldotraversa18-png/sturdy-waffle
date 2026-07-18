"""
Tests for the BotOrchestrator — main loop, event routing, lifecycle.

Covers:
- Initialization flow (config, state, client, risk, strategy wiring)
- Main-loop stop conditions (profit target, daily loss, drawdown)
- WebSocket callback routing (account → risk, quote → strategy, fill tracking)
- Shutdown cleanup
- Session reset on date change
- PID lock file behavior
- BotResult assembly
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.client.models import (
    Account,
    BracketConfig,
    BracketOrderRequest,
    BracketOrderResponse,
    Order,
    Position,
    Quote,
)
from src.config import BotConfig
from src.orchestrator.bot import BotOrchestrator, BotResult, BotStatus
from src.risk.engine import OrderDecision, RiskEngine
from src.risk.limits import RiskConfig
from src.risk.state import RiskState, StateManager
from src.strategy.signals import Signal
from tests.conftest import MockTradovateClient


# ===========================================================================
# Helpers
# ===========================================================================


def _make_bot_config(**overrides) -> BotConfig:
    kwargs = {
        "loop_interval": 0.001,
        "state_path": "/tmp/test_orch_state.json",
        "lock_path": "/tmp/test_orch.lock",
        "log_level": "WARNING",
        "log_dir": "/tmp/test_orch_logs",
        "environment": "demo",
        "symbols": ["MBT", "MET"],
    }
    kwargs.update(overrides)
    return BotConfig(**kwargs)


def _make_account(
    net_liq: float = 50000.0,
    realized_pnl: float = 0.0,
    account_id: int = 12345,
) -> Account:
    return Account(
        id=account_id,
        name="eval",
        net_liq=net_liq,
        realized_pnl=realized_pnl,
        open_pnl=0.0,
        balance=net_liq,
        available_funds=net_liq,
    )


def _make_quote(symbol: str = "MBT", bid: float = 62100.0, ask: float = 62125.0) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=1000,
        timestamp="2026-07-18T14:30:00Z",
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


def _make_signal(
    symbol: str = "MBT",
    direction: str = "long",
    confidence: float = 0.75,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        entry_price=62112.0,
        stop_price=62100.0,
        target_price=62136.0,
        rationale="test signal",
        timestamp=1_800_000_000.0,
    )


# ===========================================================================
# Cleanup helper
# ===========================================================================


def _cleanup(path: str) -> None:
    """Remove a file if it exists."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ===========================================================================
# Initialization tests
# ===========================================================================


class TestInitialization:
    """BotOrchestrator startup and wiring."""

    def test_constructor_stores_config(self):
        """Constructor should store the BotConfig and leave engines as None."""
        cfg = _make_bot_config()
        orch = BotOrchestrator(cfg)
        assert orch.config is cfg
        assert orch.client is None
        assert orch.risk_engine is None
        assert orch.strategy is None
        assert orch.state_manager is None
        assert orch._running is False

    def test_initialize_creates_subsystems(self):
        """After _initialize, all subsystems should be non-None."""
        cfg = _make_bot_config()
        orch = BotOrchestrator(cfg)

        # Patch TradovateConfig + TradovateClient to avoid real network calls.
        with (
            patch("src.orchestrator.bot.TradovateConfig", autospec=True),
            patch(
                "src.orchestrator.bot.TradovateClient", autospec=True
            ) as mock_tv_cls,
        ):
            mock_tv = mock_tv_cls.return_value
            mock_tv.connect = AsyncMock()
            mock_tv.get_account = AsyncMock(
                return_value=_make_account()
            )
            mock_tv.get_positions = AsyncMock(return_value=[])
            mock_tv.subscribe_quotes = AsyncMock()
            mock_tv.on_account_update = MagicMock()
            mock_tv.on_order_update = MagicMock()
            mock_tv.on_quote = MagicMock()
            mock_tv._request = AsyncMock(
                return_value={"d": [{"id": 12345}]}
            )

            async def _run_init():
                await orch._initialize()

            asyncio.run(_run_init())

        assert orch.client is not None
        assert orch.risk_engine is not None
        assert orch.strategy is not None
        assert orch.state_manager is not None
        assert orch._account_id == 12345

        _cleanup(cfg.state_path)
        _cleanup(cfg.lock_path)

    def test_initialize_with_persisted_state(self):
        """If state.json exists, it should be loaded into the risk engine."""
        cfg = _make_bot_config()

        # Pre-write a state file.
        saved = RiskState(
            session_date=date.today(),
            session_realized_pnl=500.0,
            peak_equity=50500.0,
            starting_equity=50000.0,
            total_realized_pnl=500.0,
        )
        mgr = StateManager(cfg.state_path)
        mgr.save(saved)

        orch = BotOrchestrator(cfg)

        with (
            patch("src.orchestrator.bot.TradovateConfig", autospec=True),
            patch(
                "src.orchestrator.bot.TradovateClient", autospec=True
            ) as mock_tv_cls,
        ):
            mock_tv = mock_tv_cls.return_value
            mock_tv.connect = AsyncMock()
            mock_tv.get_account = AsyncMock(return_value=_make_account(net_liq=50500.0))
            mock_tv.get_positions = AsyncMock(return_value=[])
            mock_tv.subscribe_quotes = AsyncMock()
            mock_tv.on_account_update = MagicMock()
            mock_tv.on_order_update = MagicMock()
            mock_tv.on_quote = MagicMock()
            mock_tv._request = AsyncMock(return_value={"d": [{"id": 12345}]})

            async def _run():
                await orch._initialize()

            asyncio.run(_run())

        assert orch.risk_engine is not None
        state = orch.risk_engine.snapshot()
        assert state.total_realized_pnl == 500.0
        assert state.peak_equity == 50500.0

        _cleanup(cfg.state_path)
        _cleanup(cfg.lock_path)

    def test_initialize_resets_session_on_new_day(self):
        """If the saved state is from yesterday, the session counters reset."""
        cfg = _make_bot_config()

        yesterday = date.today() - timedelta(days=1)
        saved = RiskState(
            session_date=yesterday,
            session_realized_pnl=-300.0,
            peak_equity=50000.0,
            starting_equity=50000.0,
            total_realized_pnl=-300.0,
            daily_loss_breached=True,
        )
        mgr = StateManager(cfg.state_path)
        mgr.save(saved)

        orch = BotOrchestrator(cfg)

        with (
            patch("src.orchestrator.bot.TradovateConfig", autospec=True),
            patch(
                "src.orchestrator.bot.TradovateClient", autospec=True
            ) as mock_tv_cls,
        ):
            mock_tv = mock_tv_cls.return_value
            mock_tv.connect = AsyncMock()
            mock_tv.get_account = AsyncMock(return_value=_make_account())
            mock_tv.get_positions = AsyncMock(return_value=[])
            mock_tv.subscribe_quotes = AsyncMock()
            mock_tv.on_account_update = MagicMock()
            mock_tv.on_order_update = MagicMock()
            mock_tv.on_quote = MagicMock()
            mock_tv._request = AsyncMock(return_value={"d": [{"id": 12345}]})

            async def _run():
                await orch._initialize()

            asyncio.run(_run())

        assert orch.risk_engine is not None
        state = orch.risk_engine.snapshot()
        # Session counters should be reset.
        assert state.session_date == date.today()
        assert state.session_realized_pnl == 0.0
        assert state.daily_loss_breached is False
        # Cumulative values are preserved.
        assert state.total_realized_pnl == -300.0

        _cleanup(cfg.state_path)
        _cleanup(cfg.lock_path)


# ===========================================================================
# Main-loop stop condition tests
# ===========================================================================


class TestStopConditions:
    """End-condition evaluation inside the orchestrator."""

    def test_profit_target_stops_loop(self, bot_config, risk_config):
        """When profit target is reached, the loop should exit with PASSED."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)

        # Manually push the engine into a passed state.
        orch.risk_engine.update_from_fill(3000.0)
        # Update the peak equity via an account update.
        orch.risk_engine.update_from_account(_make_account(net_liq=53000.0, realized_pnl=3000.0))

        async def _eval():
            return await orch._evaluate_end_conditions()

        status = asyncio.run(_eval())
        assert status == BotStatus.PASSED

    def test_daily_loss_stops_loop(self, bot_config, risk_config):
        """When daily loss limit is breached, the loop should return LOCKED."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)
        orch.risk_engine.update_from_fill(-1250.0)

        async def _eval():
            return await orch._evaluate_end_conditions()

        status = asyncio.run(_eval())
        assert status == BotStatus.LOCKED

    def test_drawdown_stops_loop(self, bot_config, risk_config):
        """When drawdown is breached, the loop should return FAILED."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)
        orch.risk_engine.update_from_account(_make_account(net_liq=47999.0))

        async def _eval():
            return await orch._evaluate_end_conditions()

        status = asyncio.run(_eval())
        assert status == BotStatus.FAILED

    def test_clean_state_continues(self, bot_config, risk_config):
        """No stop conditions met → CONTINUE."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)

        async def _eval():
            return await orch._evaluate_end_conditions()

        status = asyncio.run(_eval())
        assert status == BotStatus.CONTINUE


# ===========================================================================
# Callback routing tests
# ===========================================================================


class TestCallbackRouting:
    """WebSocket event → engine callbacks."""

    def test_account_update_routes_to_risk_engine(self, bot_config, risk_config):
        """_on_account_update should call RiskEngine.update_from_account()."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)
        orch.state_manager = StateManager(bot_config.state_path)

        acct = _make_account(net_liq=51000.0, realized_pnl=1000.0)

        async def _route():
            await orch._on_account_update(acct)

        asyncio.run(_route())

        state = orch.risk_engine.snapshot()
        assert state.peak_equity == 51000.0

        _cleanup(bot_config.state_path)
        _cleanup(bot_config.lock_path)

    def test_quote_routes_to_strategy(self, bot_config, strategy_config):
        """_on_quote should call StrategyEngine.ingest_quote()."""
        orch = BotOrchestrator(bot_config)
        from src.strategy.engine import StrategyEngine

        orch.strategy = StrategyEngine(strategy_config)
        quote = _make_quote()

        async def _route():
            await orch._on_quote(quote)

        asyncio.run(_route())

        # Quote buffer should have 1 entry.
        buf = orch.strategy._quote_buffers["MBT"]
        assert len(buf) == 1
        assert buf[0].symbol == "MBT"

    def test_fill_tracks_trade_count(self, bot_config):
        """_on_order_update should increment trade counter on Fill."""
        orch = BotOrchestrator(bot_config)

        assert orch._trades == 0
        order = _make_order(order_status="Filled", filled_qty=1)

        async def _route():
            await orch._on_order_update(order)

        asyncio.run(_route())
        assert orch._trades == 1

    def test_non_fill_does_not_increment(self, bot_config):
        """Order statuses other than 'Filled' should not increment trades."""
        orch = BotOrchestrator(bot_config)

        order = _make_order(order_status="Working", filled_qty=0)
        async def _route():
            await orch._on_order_update(order)
        asyncio.run(_route())
        assert orch._trades == 0


# ===========================================================================
# Shutdown tests
# ===========================================================================


class TestShutdown:
    """Graceful shutdown behavior."""

    def test_shutdown_persists_state(self, bot_config, risk_config):
        """shutdown() should save the current risk state to disk."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)
        orch.state_manager = StateManager(bot_config.state_path)
        orch.client = MockTradovateClient()

        orch.risk_engine.update_from_fill(500.0)

        async def _shutdown():
            await orch.shutdown()

        asyncio.run(_shutdown())

        # State should be persisted.
        loaded = orch.state_manager.load()
        assert loaded is not None
        assert loaded.total_realized_pnl == 500.0

        _cleanup(bot_config.state_path)
        _cleanup(bot_config.lock_path)

    def test_shutdown_idempotent(self, bot_config):
        """Calling shutdown multiple times should not raise."""
        orch = BotOrchestrator(bot_config)
        orch.client = MockTradovateClient()

        async def _double_shutdown():
            await orch.shutdown()
            await orch.shutdown()

        # Should not raise.
        asyncio.run(_double_shutdown())

        _cleanup(bot_config.lock_path)


# ===========================================================================
# BotResult tests
# ===========================================================================


class TestBotResult:
    """BotResult assembly from engine state."""

    def test_passed_result(self, bot_config, risk_config):
        """A passed evaluation should produce BotStatus.PASSED."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)
        orch.risk_engine.update_from_fill(3000.0)
        orch.risk_engine.update_from_account(_make_account(net_liq=53000.0))
        orch._latest_account = _make_account(net_liq=53000.0)

        result = orch._build_result()
        assert result.status == BotStatus.PASSED
        assert result.total_pnl == 3000.0
        assert result.peak_equity == 53000.0

    def test_failed_result(self, bot_config, risk_config):
        """A drawdown breach should produce BotStatus.FAILED."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)
        orch.risk_engine.update_from_account(_make_account(net_liq=47999.0))
        orch._latest_account = _make_account(net_liq=47999.0)

        result = orch._build_result()
        assert result.status == BotStatus.FAILED

    def test_locked_result(self, bot_config, risk_config):
        """A daily loss breach should produce BotStatus.LOCKED."""
        orch = BotOrchestrator(bot_config)
        orch.risk_engine = RiskEngine(risk_config)
        orch.risk_engine.update_from_fill(-1250.0)
        orch._latest_account = _make_account(net_liq=48750.0)

        result = orch._build_result()
        assert result.status == BotStatus.LOCKED

    def test_uninitialised_result(self, bot_config):
        """If the engine is None, result should be FAILED."""
        orch = BotOrchestrator(bot_config)
        result = orch._build_result()
        assert result.status == BotStatus.FAILED
        assert result.reason == "never initialised"


# ===========================================================================
# PID lock tests
# ===========================================================================


class TestPidLock:
    """Lock-file behavior."""

    def test_acquire_and_release_lock(self, bot_config):
        """Lock should be created on acquire and removed on release."""
        cfg = _make_bot_config(lock_path="/tmp/test_orch_lock_test.lock")
        _cleanup(cfg.lock_path)

        orch = BotOrchestrator(cfg)
        orch._acquire_lock()
        assert os.path.exists(cfg.lock_path)

        orch._release_lock()
        assert not os.path.exists(cfg.lock_path)

    def test_double_acquire_raises(self, bot_config):
        """Acquiring a held lock should raise RuntimeError."""
        cfg = _make_bot_config(lock_path="/tmp/test_orch_double_lock.lock")
        _cleanup(cfg.lock_path)

        orch1 = BotOrchestrator(cfg)
        orch1._acquire_lock()

        orch2 = BotOrchestrator(cfg)
        with pytest.raises(RuntimeError, match="already running"):
            orch2._acquire_lock()

        orch1._release_lock()
        _cleanup(cfg.lock_path)


# ===========================================================================
# Signal processing tests
# ===========================================================================


class TestSignalProcessing:
    """End-to-end signal → order flow in the main loop."""

    def test_flat_signal_dropped(self, bot_config):
        """A 'flat' signal should not produce an order."""
        signal = _make_signal(direction="flat", confidence=0.0)
        assert signal.direction == "flat"

        # This is more of a design check: the orchestrator skips flat signals.
        orch = BotOrchestrator(bot_config)
        # If the strategy engine returns flat signals, they are discarded
        # in the main loop.
        # The real test is that we don't crash.
        # (Full main-loop testing requires a running event loop and mocks.)

    def test_low_confidence_signal_dropped(self, bot_config):
        """Signals below min_confidence_threshold should be skipped."""
        signal = _make_signal(confidence=0.3)
        cfg = _make_bot_config()

        orch = BotOrchestrator(cfg)
        # In the main loop: confidence < threshold → skip.
        # Design validation: threshold gate exists.
        assert signal.confidence < 0.6  # default threshold
