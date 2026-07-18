"""
Tests for the RiskEngine — the safety gatekeeper for Tradeify evaluations.

Covers:
- Daily loss tracking (realised only, not open P&L)
- EOD trailing drawdown
- Profit target detection
- Contract limit enforcement
- Pre-trade rejection scenarios
- State persistence (save / load / clear / atomic writes)
- Session reset behaviour
- Edge cases (flattening trades, multiple symbols, boundaries)
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pytest

from src.client.models import (
    Account,
    BracketConfig,
    BracketOrderRequest,
    OrderRequest,
    Position,
)
from src.risk.engine import OrderDecision, RiskEngine
from src.risk.limits import InstrumentLimit, RiskConfig
from src.risk.state import RiskState, StateManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> RiskConfig:
    kwargs: dict = {
        "profit_target": 3000.0,
        "max_eod_drawdown": 2000.0,
        "daily_loss_limit": 1250.0,
        "max_mini_contracts": 4,
        "max_micro_contracts": 40,
        "instrument_limits": {
            "MES": InstrumentLimit(max_contracts=40, tick_value=1.25),
            "MNQ": InstrumentLimit(max_contracts=40, tick_value=0.50),
            "ES": InstrumentLimit(max_contracts=4, tick_value=12.50),
            "NQ": InstrumentLimit(max_contracts=4, tick_value=5.00),
        },
    }
    kwargs.update(overrides)
    return RiskConfig(**kwargs)


def _make_account(
    net_liq: float = 50000.0,
    realized_pnl: float = 0.0,
    open_pnl: float = 0.0,
) -> Account:
    return Account(
        id=12345,
        name="eval-account",
        net_liq=net_liq,
        realized_pnl=realized_pnl,
        open_pnl=open_pnl,
        balance=net_liq,
        available_funds=net_liq,
    )


def _make_position(
    symbol: str,
    net_pos: int,
    avg_price: float = 100.0,
    open_pnl: float = 0.0,
    realized_pnl: float = 0.0,
) -> Position:
    return Position(
        id=hash(symbol) % 10000,
        account_id=12345,
        symbol=symbol,
        net_pos=net_pos,
        avg_price=avg_price,
        open_pnl=open_pnl,
        realized_pnl=realized_pnl,
        total_pnl=open_pnl + realized_pnl,
    )


def _make_order(
    symbol: str = "MES",
    action: str = "Buy",
    order_qty: int = 1,
) -> OrderRequest:
    return OrderRequest(
        account_spec="eval",
        account_id=12345,
        action=action,  # type: ignore[arg-type]
        symbol=symbol,
        order_qty=order_qty,
        order_type="Market",
        is_automated=True,
    )


def _make_bracket(
    symbol: str = "MES",
    action: str = "Buy",
    order_qty: int = 1,
    profit_target: float = 10.0,
    stop_loss: float = 5.0,
) -> BracketOrderRequest:
    return BracketOrderRequest(
        account_spec="eval",
        account_id=12345,
        action=action,  # type: ignore[arg-type]
        symbol=symbol,
        order_qty=order_qty,
        order_type="Market",
        is_automated=True,
        bracket=BracketConfig(profit_target=profit_target, stop_loss=stop_loss),
    )


# ===================================================================
# RiskState
# ===================================================================


class TestRiskStateDefaults:
    def test_default_values(self) -> None:
        s = RiskState(session_date=date(2026, 7, 18))
        assert s.session_realized_pnl == 0.0
        assert s.peak_equity == 50000.0
        assert s.starting_equity == 50000.0
        assert s.total_realized_pnl == 0.0
        assert s.profit_target_reached is False
        assert s.drawdown_breached is False
        assert s.daily_loss_breached is False

    def test_custom_starting_equity(self) -> None:
        s = RiskState(
            session_date=date(2026, 7, 18),
            starting_equity=150000.0,
            peak_equity=150000.0,
        )
        assert s.starting_equity == 150000.0
        assert s.peak_equity == 150000.0


# ===================================================================
# StateManager
# ===================================================================


class TestStateManager:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        mgr = StateManager(str(tmp_path / "state.json"))
        now = date.today()
        state = RiskState(
            session_date=now,
            session_realized_pnl=-200.0,
            peak_equity=50200.0,
            starting_equity=50000.0,
            total_realized_pnl=500.0,
            profit_target_reached=False,
            drawdown_breached=False,
            daily_loss_breached=False,
        )
        mgr.save(state)

        loaded = mgr.load()
        assert loaded is not None
        assert loaded.session_date == now
        assert loaded.session_realized_pnl == -200.0
        assert loaded.peak_equity == 50200.0
        assert loaded.starting_equity == 50000.0
        assert loaded.total_realized_pnl == 500.0
        assert loaded.profit_target_reached is False

    def test_load_returns_none_when_no_file(self, tmp_path: Path) -> None:
        mgr = StateManager(str(tmp_path / "nonexistent.json"))
        assert mgr.load() is None

    def test_load_returns_none_for_corrupt_file(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.json"
        p.write_text("not-json")
        mgr = StateManager(str(p))
        assert mgr.load() is None

    def test_clear_removes_file(self, tmp_path: Path) -> None:
        mgr = StateManager(str(tmp_path / "state.json"))
        mgr.save(RiskState(session_date=date.today()))
        assert mgr.load() is not None
        mgr.clear()
        assert mgr.load() is None

    def test_clear_no_file_does_not_raise(self, tmp_path: Path) -> None:
        mgr = StateManager(str(tmp_path / "ghost.json"))
        mgr.clear()  # should not raise

    def test_atomic_write_does_not_corrupt(self, tmp_path: Path) -> None:
        """Ensure interrupted writes do not leave a partial file."""
        p = tmp_path / "state.json"
        mgr = StateManager(str(p))
        state = RiskState(
            session_date=date.today(),
            total_realized_pnl=1234.56,
        )
        mgr.save(state)

        # File must contain valid JSON
        raw = json.loads(p.read_text())
        assert raw["total_realized_pnl"] == 1234.56

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        mgr = StateManager(str(tmp_path / "state.json"))
        s1 = RiskState(session_date=date.today(), total_realized_pnl=100.0)
        s2 = RiskState(session_date=date.today(), total_realized_pnl=999.0)
        mgr.save(s1)
        mgr.save(s2)
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.total_realized_pnl == 999.0

    def test_load_missing_fields_default_to_false(self, tmp_path: Path) -> None:
        """Old state files missing newer boolean fields should not break."""
        p = tmp_path / "state.json"
        p.write_text(
            json.dumps(
                {
                    "session_date": "2026-07-18",
                    "session_realized_pnl": 0.0,
                    "peak_equity": 50000.0,
                    "starting_equity": 50000.0,
                    "total_realized_pnl": 0.0,
                    # missing boolean flags
                }
            )
        )
        mgr = StateManager(str(p))
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.profit_target_reached is False
        assert loaded.drawdown_breached is False
        assert loaded.daily_loss_breached is False


# ===================================================================
# RiskConfig
# ===================================================================


class TestRiskConfig:
    def test_defaults(self) -> None:
        cfg = RiskConfig()
        assert cfg.profit_target == 3000.0
        assert cfg.max_eod_drawdown == 2000.0
        assert cfg.daily_loss_limit == 1250.0

    def test_get_limit_explicit(self) -> None:
        cfg = _make_config()
        assert cfg.get_limit("ES") == 4
        assert cfg.get_limit("MES") == 40

    def test_get_limit_fallback_micro(self) -> None:
        cfg = RiskConfig()
        assert cfg.get_limit("MNQ") == 40  # starts with M → micro

    def test_get_limit_fallback_mini(self) -> None:
        cfg = RiskConfig()
        assert cfg.get_limit("ES") == 4  # starts with E → mini

    def test_get_limit_unknown_fallback(self) -> None:
        cfg = RiskConfig(max_micro_contracts=40, max_mini_contracts=4)
        assert cfg.get_limit("ZB") == 4  # bond futures → mini fallback


# ===================================================================
# RiskEngine — daily loss (realised only)
# ===================================================================


class TestDailyLoss:
    def test_fills_accumulate_session_pnl(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-500.0)
        assert engine.snapshot().session_realized_pnl == -500.0

    def test_breach_at_exact_limit(self) -> None:
        engine = RiskEngine(_make_config())
        # -1250 exactly is a breach
        engine.update_from_fill(-1250.0)
        assert engine.snapshot().daily_loss_breached is True

    def test_breach_below_limit(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-600.0)
        engine.update_from_fill(-700.0)  # total -1300
        assert engine.snapshot().daily_loss_breached is True

    def test_no_breach_above_limit(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-1249.99)
        assert engine.snapshot().daily_loss_breached is False

    def test_positive_pnl_does_not_breach(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(500.0)
        engine.update_from_fill(700.0)
        assert engine.snapshot().daily_loss_breached is False
        assert engine.snapshot().session_realized_pnl == 1200.0

    def test_mixed_pnl_still_breaches(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(300.0)  # +300
        engine.update_from_fill(-1600.0)  # net -1300
        assert engine.snapshot().daily_loss_breached is True
        assert engine.snapshot().session_realized_pnl == -1300.0

    def test_remaining_daily_loss_no_losses(self) -> None:
        engine = RiskEngine(_make_config())
        assert engine.remaining_daily_loss() == 1250.0

    def test_remaining_daily_loss_partial(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-400.0)
        assert engine.remaining_daily_loss() == 850.0

    def test_remaining_daily_loss_exhausted(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-1250.0)
        assert engine.remaining_daily_loss() == 0.0

    def test_remaining_daily_loss_floors_at_zero(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-2000.0)
        assert engine.remaining_daily_loss() == 0.0

    def test_unrealized_pnl_does_not_affect_daily_loss(self) -> None:
        """Account updates (which include open P&L) do NOT affect daily loss."""
        engine = RiskEngine(_make_config())
        # Simulate an account where net_liq dropped due to open P&L only
        acct = _make_account(net_liq=48000.0, realized_pnl=0.0, open_pnl=-2000.0)
        engine.update_from_account(acct)
        # Daily loss must still be 0 because realized_pnl hasn't changed
        assert engine.snapshot().daily_loss_breached is False
        assert engine.snapshot().session_realized_pnl == 0.0


# ===================================================================
# RiskEngine — EOD trailing drawdown
# ===================================================================


class TestDrawdown:
    def test_peak_equity_trails_up(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        assert engine.snapshot().peak_equity == 50200.0

    def test_peak_equity_does_not_trail_down(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        engine.update_from_account(_make_account(net_liq=49900.0))
        assert engine.snapshot().peak_equity == 50200.0

    def test_multiple_updates_trail_correctly(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50100.0))
        engine.update_from_account(_make_account(net_liq=50400.0))
        engine.update_from_account(_make_account(net_liq=50300.0))
        engine.update_from_account(_make_account(net_liq=50500.0))
        assert engine.snapshot().peak_equity == 50500.0

    def test_breach_when_below_floor(self) -> None:
        engine = RiskEngine(_make_config())
        # Trail peak to 50200
        engine.update_from_account(_make_account(net_liq=50200.0))
        # Now drop below floor: 50200 - 2000 = 48200
        engine.update_from_account(_make_account(net_liq=48199.0))
        assert engine.snapshot().drawdown_breached is True

    def test_no_breach_at_floor(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        # Exactly at floor: 50200 - 2000 = 48200
        engine.update_from_account(_make_account(net_liq=48200.0))
        assert engine.snapshot().drawdown_breached is True  # <= means breach

    def test_no_breach_above_floor(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        engine.update_from_account(_make_account(net_liq=48200.01))
        # 48200 is the floor, 48200.01 is just above → no breach
        assert engine.snapshot().drawdown_breached is False

    def test_remaining_drawdown_normal(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        # floor = 50200 - 2000 = 48200; current = 49000
        assert engine.remaining_drawdown(49000.0) == 800.0

    def test_remaining_drawdown_zero(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        # floor = 48200
        assert engine.remaining_drawdown(48200.0) == 0.0

    def test_remaining_drawdown_negative(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        assert engine.remaining_drawdown(48000.0) == -200.0


# ===================================================================
# RiskEngine — profit target
# ===================================================================


class TestProfitTarget:
    def test_reached_via_fills(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(1500.0)
        engine.update_from_fill(1500.0)  # total 3000
        assert engine.snapshot().profit_target_reached is True

    def test_reached_via_single_fill(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(3000.0)
        assert engine.snapshot().profit_target_reached is True

    def test_not_reached_below_target(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(2999.99)
        assert engine.snapshot().profit_target_reached is False

    def test_reached_via_total_accumulated(self) -> None:
        """total_realized_pnl spans sessions — if previously accumulated,
        a new fill can push it over."""
        engine = RiskEngine(_make_config())
        engine.load_state(
            RiskState(
                session_date=date.today(),
                total_realized_pnl=2800.0,
                peak_equity=52800.0,
                starting_equity=50000.0,
            )
        )
        engine.update_from_fill(200.0)
        assert engine.snapshot().profit_target_reached is True


# ===================================================================
# RiskEngine — contract limits
# ===================================================================


class TestContractLimits:
    def test_single_order_within_limit(self) -> None:
        engine = RiskEngine(_make_config())
        order = _make_order("ES", "Buy", 4)
        decision = engine.check_order(order, [])
        assert decision.approved is True

    def test_single_order_exceeds_limit(self) -> None:
        engine = RiskEngine(_make_config())
        order = _make_order("ES", "Buy", 5)
        decision = engine.check_order(order, [])
        assert decision.approved is False
        assert "limit exceeded" in (decision.reason or "")

    def test_existing_position_plus_order_within_limit(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("ES", net_pos=2)]
        order = _make_order("ES", "Buy", 2)  # 2 + 2 = 4 → ok
        decision = engine.check_order(order, positions)
        assert decision.approved is True

    def test_existing_position_plus_order_exceeds_limit(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("ES", net_pos=3)]
        order = _make_order("ES", "Buy", 2)  # 3 + 2 = 5 → exceeds 4
        decision = engine.check_order(order, positions)
        assert decision.approved is False

    def test_flattening_reduces_position_allowed(self) -> None:
        """Even if the account is over the limit, an order that reduces
        absolute position must be allowed (flattening escape hatch)."""
        engine = RiskEngine(_make_config())
        # Somehow position became 6 (over 4 limit)
        positions = [_make_position("ES", net_pos=6)]
        order = _make_order("ES", "Sell", 2)  # 6 - 2 = 4 → still at limit, reducing
        decision = engine.check_order(order, positions)
        # Reducing from 6 to 4 — allowed
        assert decision.approved is True

    def test_flattening_fully_closes_position(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("ES", net_pos=4)]
        order = _make_order("ES", "Sell", 4)  # close
        decision = engine.check_order(order, positions)
        assert decision.approved is True

    def test_increasing_beyond_limit_rejected(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("ES", net_pos=4)]
        order = _make_order("ES", "Buy", 1)  # 4 + 1 = 5
        decision = engine.check_order(order, positions)
        assert decision.approved is False

    def test_short_position_within_limit(self) -> None:
        engine = RiskEngine(_make_config())
        order = _make_order("ES", "Sell", 4)  # net_pos = -4
        decision = engine.check_order(order, [])
        assert decision.approved is True

    def test_short_exceeds_limit(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("ES", net_pos=-3)]
        order = _make_order("ES", "Sell", 2)  # -3 + (-2) = -5 → abs=5 > 4
        decision = engine.check_order(order, positions)
        assert decision.approved is False

    def test_short_flattening_allowed(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("ES", net_pos=-6)]  # over limit
        order = _make_order("ES", "Buy", 2)  # -6 + 2 = -4 → reducing
        decision = engine.check_order(order, positions)
        assert decision.approved is True

    def test_multiple_symbols_independent(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [
            _make_position("ES", net_pos=4),
            _make_position("MES", net_pos=38),
        ]
        # ES at limit, but MES order should still be checked independently
        order = _make_order("MES", "Buy", 2)  # 38 + 2 = 40 -> at limit, ok
        decision = engine.check_order(order, positions)
        assert decision.approved is True

    def test_multiple_symbols_mes_exceeds(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("MES", net_pos=39)]
        order = _make_order("MES", "Buy", 2)  # 39 + 2 = 41 > 40
        decision = engine.check_order(order, positions)
        assert decision.approved is False

    def test_symbol_not_in_positions_treated_as_zero(self) -> None:
        engine = RiskEngine(_make_config())
        positions = [_make_position("ES", net_pos=4)]
        order = _make_order("NQ", "Buy", 4)
        decision = engine.check_order(order, positions)
        assert decision.approved is True


# ===================================================================
# RiskEngine — pre-trade rejection (stop conditions)
# ===================================================================


class TestPreTradeRejection:
    def test_rejects_when_daily_loss_breached(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-1250.0)
        order = _make_order("MES", "Buy", 1)
        decision = engine.check_order(order, [])
        assert decision.approved is False
        assert "Daily loss" in (decision.reason or "")

    def test_rejects_when_drawdown_breached(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        engine.update_from_account(_make_account(net_liq=48199.0))
        order = _make_order("MES", "Buy", 1)
        decision = engine.check_order(order, [])
        assert decision.approved is False
        assert "drawdown" in (decision.reason or "")

    def test_rejects_when_profit_target_reached(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(3000.0)
        order = _make_order("MES", "Buy", 1)
        decision = engine.check_order(order, [])
        assert decision.approved is False
        assert "Profit target" in (decision.reason or "")


# ===================================================================
# RiskEngine — is_trading_allowed / get_stop_reason
# ===================================================================


class TestTradingAllowed:
    def test_allowed_initially(self) -> None:
        engine = RiskEngine(_make_config())
        allowed, reason = engine.is_trading_allowed()
        assert allowed is True
        assert reason == ""

    def test_blocked_daily_loss(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-1250.0)
        allowed, reason = engine.is_trading_allowed()
        assert allowed is False
        assert "Daily loss" in reason

    def test_blocked_drawdown(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        engine.update_from_account(_make_account(net_liq=48199.0))
        allowed, reason = engine.is_trading_allowed()
        assert allowed is False
        assert "drawdown" in reason

    def test_blocked_profit_target(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(3000.0)
        allowed, reason = engine.is_trading_allowed()
        assert allowed is False
        assert "Profit target" in reason

    def test_get_stop_reason_none_initially(self) -> None:
        engine = RiskEngine(_make_config())
        assert engine.get_stop_reason() is None

    def test_get_stop_reason_daily_loss(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-1250.0)
        assert engine.get_stop_reason() == "daily_loss"

    def test_get_stop_reason_drawdown(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50200.0))
        engine.update_from_account(_make_account(net_liq=48199.0))
        assert engine.get_stop_reason() == "drawdown"

    def test_get_stop_reason_profit_target(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(3000.0)
        assert engine.get_stop_reason() == "profit_target"


# ===================================================================
# RiskEngine — check_bracket
# ===================================================================


class TestBracketChecks:
    def test_bracket_passes_standard_checks(self) -> None:
        engine = RiskEngine(_make_config())
        bracket = _make_bracket("MES", "Buy", 5, stop_loss=5.0)
        decision = engine.check_bracket(bracket, [])
        assert decision.approved is True

    def test_bracket_rejects_on_contract_limit(self) -> None:
        engine = RiskEngine(_make_config())
        bracket = _make_bracket("ES", "Buy", 5, stop_loss=5.0)
        decision = engine.check_bracket(bracket, [])
        assert decision.approved is False
        assert "limit exceeded" in (decision.reason or "")

    def test_bracket_rejects_on_daily_loss(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-1250.0)
        bracket = _make_bracket("MES", "Buy", 1)
        decision = engine.check_bracket(bracket, [])
        assert decision.approved is False

    def test_bracket_rejects_negative_stop_loss_at_model_level(self) -> None:
        """BracketConfig enforces stop_loss > 0 at the Pydantic level.
        The RiskEngine's <= 0 check is a redundant safety net."""
        with pytest.raises(ValueError):
            BracketConfig(profit_target=10.0, stop_loss=-5.0)
        with pytest.raises(ValueError):
            BracketConfig(profit_target=10.0, stop_loss=0.0)

    def test_bracket_worst_case_within_drawdown(self) -> None:
        """Small stop loss on micros should not breach drawdown."""
        engine = RiskEngine(_make_config())
        engine.update_from_account(_make_account(net_liq=50000.0))
        bracket = _make_bracket("MES", "Buy", 2, stop_loss=5.0)
        # worst_case = 5 * 1.25 * 2 = 12.50, well within drawdown
        decision = engine.check_bracket(bracket, [])
        assert decision.approved is True

    def test_bracket_worst_case_breaches_drawdown(self) -> None:
        """Large stop on ES near drawdown floor should reject."""
        engine = RiskEngine(_make_config())
        # Current equity very close to floor: peak 50000, floor 48000
        engine.update_from_account(_make_account(net_liq=48500.0))
        # Big bracket: ES with 4 contracts at 100 tick stop
        # worst_case = 100 * 12.50 * 4 = 5000 loss
        bracket = _make_bracket("ES", "Buy", 4, stop_loss=100.0)
        decision = engine.check_bracket(bracket, [])
        # projected = 48500 - 5000 = 43500, floor = 48000
        assert decision.approved is False
        assert "drawdown" in (decision.reason or "").lower()

    def test_bracket_no_instrument_limit_skips_worst_case(self) -> None:
        """Symbol without instrument_limits entry skips worst-case check."""
        engine = RiskEngine(_make_config())
        bracket = _make_bracket("ZB", "Buy", 3, stop_loss=500.0)
        decision = engine.check_bracket(bracket, [])
        # Should pass — no instrument_limit for ZB, so worst-case skipped
        assert decision.approved is True


# ===================================================================
# RiskEngine — session reset
# ===================================================================


class TestSessionReset:
    def test_reset_clears_daily_pnl(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-500.0)
        engine.update_from_account(_make_account(net_liq=50200.0))

        engine.reset_session()

        s = engine.snapshot()
        assert s.session_realized_pnl == 0.0
        assert s.daily_loss_breached is False
        # Cumulative fields persist
        assert s.peak_equity == 50200.0
        assert s.total_realized_pnl == -500.0

    def test_reset_does_not_clear_permanent_breach(self) -> None:
        engine = RiskEngine(_make_config())
        engine.update_from_fill(3000.0)  # profit target hit
        engine.reset_session()
        s = engine.snapshot()
        assert s.profit_target_reached is True  # permanent

    def test_load_state_with_new_date_resets_session(self) -> None:
        engine = RiskEngine(_make_config())
        yesterday = date.today() - timedelta(days=1)
        old_state = RiskState(
            session_date=yesterday,
            session_realized_pnl=-800.0,
            peak_equity=51000.0,
            starting_equity=50000.0,
            total_realized_pnl=1200.0,
            daily_loss_breached=False,
        )
        engine.load_state(old_state)
        s = engine.snapshot()
        assert s.session_date == date.today()
        assert s.session_realized_pnl == 0.0
        assert s.daily_loss_breached is False
        assert s.peak_equity == 51000.0  # preserved
        assert s.total_realized_pnl == 1200.0  # preserved

    def test_load_state_same_day_preserves_all(self) -> None:
        engine = RiskEngine(_make_config())
        today = date.today()
        old_state = RiskState(
            session_date=today,
            session_realized_pnl=-400.0,
            peak_equity=50200.0,
            starting_equity=50000.0,
            total_realized_pnl=600.0,
            daily_loss_breached=False,
        )
        engine.load_state(old_state)
        s = engine.snapshot()
        assert s.session_realized_pnl == -400.0
        assert s.peak_equity == 50200.0
        assert s.total_realized_pnl == 600.0


# ===================================================================
# RiskEngine — edge cases
# ===================================================================


class TestEdgeCases:
    def test_zero_quantity_order_is_invalid_by_model(self) -> None:
        """OrderRequest requires order_qty > 0, so this is guarded by Pydantic."""
        with pytest.raises(ValueError):
            _make_order("MES", "Buy", 0)

    def test_update_from_account_evaluates_drawdown_only(self) -> None:
        """Account updates do NOT change realized P&L — only fills do."""
        engine = RiskEngine(_make_config())
        engine.update_from_fill(-300.0)
        engine.update_from_account(_make_account(net_liq=49000.0, realized_pnl=-1500.0))
        # session_realized_pnl should still be -300 (from fills only)
        assert engine.snapshot().session_realized_pnl == -300.0

    def test_exact_boundary_profit_target(self) -> None:
        engine = RiskEngine(_make_config(profit_target=3000.0))
        engine.update_from_fill(3000.0)
        assert engine.snapshot().profit_target_reached is True

    def test_exact_boundary_daily_loss(self) -> None:
        engine = RiskEngine(_make_config(daily_loss_limit=1250.0))
        engine.update_from_fill(-1250.0)
        assert engine.snapshot().daily_loss_breached is True

    def test_exact_boundary_drawdown(self) -> None:
        engine = RiskEngine(_make_config(max_eod_drawdown=2000.0))
        # floor = 50000 - 2000 = 48000
        engine.update_from_account(_make_account(net_liq=48000.0))
        # Exactly at floor — breach (<=)
        assert engine.snapshot().drawdown_breached is True

    def test_snapshot_is_independent(self) -> None:
        engine = RiskEngine(_make_config())
        snap1 = engine.snapshot()
        engine.update_from_fill(-100.0)
        snap2 = engine.snapshot()
        assert snap1.session_realized_pnl == 0.0
        assert snap2.session_realized_pnl == -100.0

    def test_load_state_preserves_breach_flags(self) -> None:
        engine = RiskEngine(_make_config())
        state = RiskState(
            session_date=date.today(),
            profit_target_reached=True,
            drawdown_breached=False,
            daily_loss_breached=False,
            peak_equity=53000.0,
            total_realized_pnl=3000.0,
        )
        engine.load_state(state)
        assert engine.snapshot().profit_target_reached is True

    def test_reducing_position_on_different_symbol_is_ok(self) -> None:
        """Flattening on one symbol doesn't interfere with limits on another."""
        engine = RiskEngine(_make_config())
        positions = [
            _make_position("ES", net_pos=6),
            _make_position("MES", net_pos=40),
        ]
        order = _make_order("ES", "Sell", 2)  # 6→4, reducing
        decision = engine.check_order(order, positions)
        assert decision.approved is True

    def test_heuristic_fallback_for_unknown_symbol(self) -> None:
        """Symbols not in instrument_limits use mini/micro heuristic."""
        engine = RiskEngine(_make_config())
        # "CL" doesn't start with M → treated as mini, limit 4
        order = _make_order("CL", "Buy", 4)
        decision = engine.check_order(order, [])
        assert decision.approved is True

        order2 = _make_order("CL", "Buy", 5)
        decision2 = engine.check_order(order2, [])
        assert decision2.approved is False
