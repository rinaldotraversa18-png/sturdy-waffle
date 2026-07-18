"""
Shared pytest fixtures for the trading-bot test suite.

Provides mock/stub factories for TradovateClient, RiskEngine,
RiskState, Account, Position, Quote, and config objects so every
test module has a consistent, lightweight test harness.
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.client.models import (
    Account,
    BracketOrderRequest,
    BracketOrderResponse,
    Order,
    OrderResponse,
    Position,
    Quote,
)
from src.config import BotConfig
from src.orchestrator.bot import BotOrchestrator, BotResult, BotStatus
from src.risk.engine import OrderDecision, RiskEngine
from src.risk.limits import InstrumentLimit, RiskConfig
from src.risk.state import RiskState, StateManager
from src.strategy.engine import StrategyEngine
from src.strategy.signals import Signal
from src.strategy.sizing import StrategyConfig


# ===========================================================================
# Config fixtures
# ===========================================================================


@pytest.fixture
def risk_config() -> RiskConfig:
    """A :class:`RiskConfig` with Tradeify $50k defaults and MBT/MET limits."""
    return RiskConfig(
        profit_target=3000.0,
        max_eod_drawdown=2000.0,
        daily_loss_limit=1250.0,
        max_mini_contracts=4,
        max_micro_contracts=40,
        instrument_limits={
            "MBT": InstrumentLimit(max_contracts=4, tick_value=0.50),
            "MET": InstrumentLimit(max_contracts=4, tick_value=0.025),
        },
    )


@pytest.fixture
def strategy_config() -> StrategyConfig:
    """Default :class:`StrategyConfig` for tests."""
    return StrategyConfig(
        symbols=["MBT", "MET"],
        primary_symbol="MBT",
        risk_per_trade_pct=0.25,
        max_risk_per_trade=300.0,
        mean_reversion_std_dev=2.0,
        min_confidence_threshold=0.6,
    )


@pytest.fixture
def bot_config() -> BotConfig:
    """A :class:`BotConfig` suitable for unit tests."""
    return BotConfig(
        loop_interval=0.01,  # fast for tests
        state_path="/tmp/test_state.json",
        lock_path="/tmp/test_bot.lock",
        log_level="WARNING",
        log_dir="/tmp/test_logs",
        environment="demo",
        symbols=["MBT", "MET"],
    )


# ===========================================================================
# Data fixtures
# ===========================================================================


@pytest.fixture
def sample_account() -> Account:
    """A fresh $50,000 evaluation account with no P&L."""
    return Account(
        id=12345,
        name="eval-account",
        net_liq=50000.0,
        realized_pnl=0.0,
        open_pnl=0.0,
        balance=50000.0,
        available_funds=50000.0,
    )


@pytest.fixture
def profitable_account() -> Account:
    """An account that has reached the $3,000 profit target."""
    return Account(
        id=12345,
        name="eval-account",
        net_liq=53000.0,
        realized_pnl=3000.0,
        open_pnl=0.0,
        balance=53000.0,
        available_funds=53000.0,
    )


@pytest.fixture
def drawdown_account() -> Account:
    """An account deep in drawdown ($2,000 below starting)."""
    return Account(
        id=12345,
        name="eval-account",
        net_liq=48000.0,
        realized_pnl=-2000.0,
        open_pnl=0.0,
        balance=48000.0,
        available_funds=48000.0,
    )


@pytest.fixture
def sample_position() -> Position:
    """A flat MBT position."""
    return Position(
        id=1,
        account_id=12345,
        symbol="MBT",
        net_pos=0,
        avg_price=0.0,
        open_pnl=0.0,
        realized_pnl=0.0,
        total_pnl=0.0,
    )


@pytest.fixture
def sample_quote() -> Quote:
    """A realistic MBT quote."""
    return Quote(
        symbol="MBT",
        bid=62100.0,
        ask=62125.0,
        last=62112.0,
        volume=1500,
        timestamp="2026-07-18T14:30:00Z",
    )


@pytest.fixture
def sample_signal() -> Signal:
    """A long MBT signal with moderate confidence."""
    return Signal(
        symbol="MBT",
        direction="long",
        confidence=0.75,
        entry_price=62112.0,
        stop_price=62100.0,
        target_price=62136.0,
        rationale="Mean-reversion: price -2.5σ from VWAP",
        timestamp=1_800_000_000.0,
    )


# ===========================================================================
# Risk state fixtures
# ===========================================================================


@pytest.fixture
def fresh_risk_state() -> RiskState:
    """Brand-new, untouched risk state."""
    return RiskState(
        session_date=date.today(),
        session_realized_pnl=0.0,
        peak_equity=50000.0,
        starting_equity=50000.0,
        total_realized_pnl=0.0,
        profit_target_reached=False,
        drawdown_breached=False,
        daily_loss_breached=False,
    )


@pytest.fixture
def passed_risk_state() -> RiskState:
    """Risk state where the profit target has been hit."""
    return RiskState(
        session_date=date.today(),
        session_realized_pnl=3000.0,
        peak_equity=53000.0,
        starting_equity=50000.0,
        total_realized_pnl=3000.0,
        profit_target_reached=True,
        drawdown_breached=False,
        daily_loss_breached=False,
    )


@pytest.fixture
def drawdown_risk_state() -> RiskState:
    """Risk state where the drawdown floor was breached."""
    return RiskState(
        session_date=date.today(),
        session_realized_pnl=-2000.0,
        peak_equity=50000.0,
        starting_equity=50000.0,
        total_realized_pnl=-2000.0,
        profit_target_reached=False,
        drawdown_breached=True,
        daily_loss_breached=False,
    )


@pytest.fixture
def daily_loss_risk_state() -> RiskState:
    """Risk state where the daily loss limit was breached."""
    return RiskState(
        session_date=date.today(),
        session_realized_pnl=-1250.0,
        peak_equity=50000.0,
        starting_equity=50000.0,
        total_realized_pnl=-1250.0,
        profit_target_reached=False,
        drawdown_breached=False,
        daily_loss_breached=True,
    )


# ===========================================================================
# Engine fixtures
# ===========================================================================


@pytest.fixture
def risk_engine(risk_config: RiskConfig) -> RiskEngine:
    """A :class:`RiskEngine` primed with Tradeify $50k defaults."""
    return RiskEngine(risk_config)


@pytest.fixture
def strategy_engine(strategy_config: StrategyConfig) -> StrategyEngine:
    """A :class:`StrategyEngine` primed with default config."""
    return StrategyEngine(strategy_config)


# ===========================================================================
# Mock / stub fixtures
# ===========================================================================


class MockTradovateClient:
    """In-memory stub of :class:`TradovateClient` for orchestrator tests.

    Records method calls and lets tests inject return values or side-
    effects without touching the network.
    """

    def __init__(self) -> None:
        self.connected: bool = False
        self._callbacks: dict[str, list] = {
            "account_update": [],
            "order_update": [],
            "position_update": [],
            "quote": [],
        }

        # Stub return values.
        self._get_account_return: Account | None = None
        self._get_positions_return: list[Position] = []
        self._place_bracket_return: BracketOrderResponse = BracketOrderResponse(
            entry_order_id=1,
            profit_target_order_id=2,
            stop_loss_order_id=3,
            status="Ok",
        )
        self._request_return: dict = {"d": [{"id": 12345}]}

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    # -- REST helpers --------------------------------------------------------

    async def _request(self, method: str, endpoint: str, data: dict | None = None) -> dict:
        return self._request_return

    # -- Query ---------------------------------------------------------------

    async def get_account(self, account_id: int) -> Account:
        if self._get_account_return is not None:
            return self._get_account_return
        return Account(
            id=account_id,
            name="eval",
            net_liq=50000.0,
            realized_pnl=0.0,
            open_pnl=0.0,
            balance=50000.0,
            available_funds=50000.0,
        )

    async def get_positions(self) -> list[Position]:
        return self._get_positions_return

    async def subscribe_quotes(self, symbols: list[str]) -> None:
        pass

    # -- Orders --------------------------------------------------------------

    async def place_bracket(self, bracket: BracketOrderRequest) -> BracketOrderResponse:
        return self._place_bracket_return

    # -- Callbacks -----------------------------------------------------------

    def on_account_update(self, cb) -> None:
        self._callbacks["account_update"].append(cb)

    def on_order_update(self, cb) -> None:
        self._callbacks["order_update"].append(cb)

    def on_quote(self, cb) -> None:
        self._callbacks["quote"].append(cb)

    # -- Test helpers --------------------------------------------------------

    async def trigger_account_update(self, account: Account) -> None:
        """Fire all account-update callbacks with *account*."""
        for cb in self._callbacks["account_update"]:
            result = cb(account)
            if hasattr(result, "__await__"):
                await result

    async def trigger_order_update(self, order: Order) -> None:
        for cb in self._callbacks["order_update"]:
            result = cb(order)
            if hasattr(result, "__await__"):
                await result

    async def trigger_quote(self, quote: Quote) -> None:
        for cb in self._callbacks["quote"]:
            result = cb(quote)
            if hasattr(result, "__await__"):
                await result


@pytest.fixture
def mock_client() -> MockTradovateClient:
    """An in-memory mock Tradovate client."""
    return MockTradovateClient()
