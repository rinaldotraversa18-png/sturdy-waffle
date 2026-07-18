"""
Tests for Phase 2.2 adaptive strategy modules.

Covers:
- Trailing stop activation (not yet, just past threshold, deep into profit)
- Trail stop computation (long vs short directions)
- Trail update gating (small moves ignored, significant moves trigger update)
- Adaptive state update (win updates EMA, loss doesn't change params)
- EMA drift toward winning parameters, safety bounds (50% cap from defaults)
- Confidence sizing: low confidence -> smaller size, high -> near full size
- get_adapted_params returns defaults before 10 winning trades
- Integration: engine passes confidence to sizing
- Engine: record_trade_result, last_adapted_params
"""

from __future__ import annotations

import math
import time
from collections import deque
from datetime import date

import pytest

from src.client.models import Account
from src.risk.limits import InstrumentLimit
from src.risk.state import RiskState
from src.strategy.adaptive_tuning import (
    DEFAULT_ALPHA,
    MAX_DRIFT_PCT,
    MIN_WINNING_TRADES,
    AdaptiveState,
    _bounded_ema,
    get_adapted_params,
)
from src.strategy.engine import StrategyEngine
from src.strategy.signals import Signal
from src.strategy.sizing import StrategyConfig, calculate_position_size
from src.strategy.trailing import (
    TrailingConfig,
    _favorable_move,
    compute_trail_stop,
    should_activate_trail,
    should_update_trail,
)


# ===========================================================================
# Test data helpers
# ===========================================================================


def _make_account(
    net_liq: float = 50000.0,
    realized_pnl: float = 0.0,
    balance: float = 50000.0,
    available_funds: float = 50000.0,
) -> Account:
    return Account(
        id=1,
        name="test",
        net_liq=net_liq,
        realized_pnl=realized_pnl,
        balance=balance,
        available_funds=available_funds,
    )


def _make_risk_state(
    session_pnl: float = 0.0,
    peak_equity: float = 50000.0,
) -> RiskState:
    return RiskState(
        session_date=date.today(),
        session_realized_pnl=session_pnl,
        peak_equity=peak_equity,
    )


def _make_signal(
    symbol: str = "MBT",
    direction: str = "long",
    confidence: float = 0.8,
    entry_price: float = 50000.0,
    stop_price: float = 49950.0,
    target_price: float = 50100.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        rationale="test",
        timestamp=time.time(),
    )


# ===========================================================================
# Trailing stop activation
# ===========================================================================


class TestTrailingActivation:
    """should_activate_trail: price must move activation_pct toward target."""

    def test_not_activated_below_threshold(self) -> None:
        """Price hasn't moved enough — no activation."""
        config = TrailingConfig(activation_pct=0.3)
        # Entry 100, target 200, current 110 = 10/100 = 10% < 30%.
        assert should_activate_trail(100.0, 110.0, 200.0, config) is False

    def test_activated_just_past_threshold(self) -> None:
        """Price just crossed the threshold — activation."""
        config = TrailingConfig(activation_pct=0.3)
        # Entry 100, target 200, current 130 = 30/100 = 30% == threshold.
        assert should_activate_trail(100.0, 130.0, 200.0, config) is True

    def test_activated_deep_into_profit(self) -> None:
        """Price well past threshold — definitely activated."""
        config = TrailingConfig(activation_pct=0.3)
        assert should_activate_trail(100.0, 190.0, 200.0, config) is True

    def test_short_trade_activation(self) -> None:
        """Short trade: favorable move is price going down."""
        config = TrailingConfig(activation_pct=0.3)
        # Entry 200, target 100, current 170 = favorable 30, total 100 = 30%.
        assert should_activate_trail(200.0, 170.0, 100.0, config) is True

    def test_short_trade_not_activated(self) -> None:
        """Short trade: price hasn't moved down enough."""
        config = TrailingConfig(activation_pct=0.3)
        # Entry 200, target 100, current 190 = favorable 10, total 100 = 10%.
        assert should_activate_trail(200.0, 190.0, 100.0, config) is False

    def test_zero_distance_to_target(self) -> None:
        """Degenerate: entry == target — should return False."""
        config = TrailingConfig(activation_pct=0.3)
        assert should_activate_trail(100.0, 110.0, 100.0, config) is False

    def test_price_against_position(self) -> None:
        """Price moved against the position — no activation."""
        config = TrailingConfig(activation_pct=0.3)
        # Long: entry 100, target 200, current 90 = -10 favorable = 0.
        assert should_activate_trail(100.0, 90.0, 200.0, config) is False

    def test_custom_activation_pct(self) -> None:
        """Custom activation threshold respected."""
        config = TrailingConfig(activation_pct=0.5)
        # 40% < 50% → no activation.
        assert should_activate_trail(100.0, 140.0, 200.0, config) is False
        # 51% > 50% → activation.
        assert should_activate_trail(100.0, 151.0, 200.0, config) is True


# ===========================================================================
# Trail stop computation
# ===========================================================================


class TestComputeTrailStop:
    """compute_trail_stop: calculate stop price relative to current price."""

    def test_long_trail_stop(self) -> None:
        """Long stop is below current price by trail_distance_ticks × tick_size."""
        # 20 ticks × 0.25 = 5.0 below price.
        stop = compute_trail_stop(50000.0, "long", 20, 0.25)
        assert stop == 50000.0 - 5.0

    def test_short_trail_stop(self) -> None:
        """Short stop is above current price by trail_distance_ticks × tick_size."""
        stop = compute_trail_stop(50000.0, "short", 20, 0.25)
        assert stop == 50000.0 + 5.0

    def test_custom_trail_distance(self) -> None:
        """Different trail distance gives proportional offset."""
        stop = compute_trail_stop(50000.0, "long", 40, 0.25)
        assert stop == 50000.0 - 10.0

    def test_different_tick_size(self) -> None:
        """Different tick size affects the distance."""
        stop = compute_trail_stop(50000.0, "long", 20, 0.10)
        assert stop == 50000.0 - 2.0

    def test_instrument_config_override(self) -> None:
        """Instrument config tick_size overrides the parameter."""
        inst = {"tick_size": 0.125}
        stop = compute_trail_stop(50000.0, "long", 20, 0.25, inst)
        assert stop == 50000.0 - (20 * 0.125)

    def test_instrument_config_without_tick_size(self) -> None:
        """Instrument config without tick_size falls back to parameter."""
        inst = {"max_contracts": 40}
        stop = compute_trail_stop(50000.0, "long", 20, 0.25, inst)
        assert stop == 50000.0 - 5.0

    def test_short_custom_distance(self) -> None:
        """Short stop above price with custom distance."""
        stop = compute_trail_stop(4000.0, "short", 10, 0.50)
        assert stop == 4000.0 + 5.0


# ===========================================================================
# Trail update gating
# ===========================================================================


class TestShouldUpdateTrail:
    """should_update_trail: only update when move >= step_ticks."""

    def test_long_significant_move_triggers_update(self) -> None:
        """Long stop moves up by enough ticks — update."""
        # step_ticks=5, tick_size=0.25 → min_move=1.25
        # new_stop(50005) - current_stop(50000) = 5 > 1.25
        assert should_update_trail(50000.0, 50005.0, "long", 5, 0.25) is True

    def test_long_tiny_move_ignored(self) -> None:
        """Long stop moves up by less than step_ticks — skip."""
        # min_move = 5 * 0.25 = 1.25, delta = 1.0 < 1.25
        assert should_update_trail(50000.0, 50001.0, "long", 5, 0.25) is False

    def test_long_exactly_at_threshold(self) -> None:
        """Long stop moves exactly step_ticks — update (>= check)."""
        min_move = 5 * 0.25  # 1.25
        current = 50000.0
        new_stop = current + min_move
        assert should_update_trail(current, new_stop, "long", 5, 0.25) is True

    def test_long_move_down_ignored(self) -> None:
        """Long stop moving down (loosening) — never update."""
        assert should_update_trail(50005.0, 50000.0, "long", 5, 0.25) is False

    def test_short_significant_move_triggers_update(self) -> None:
        """Short stop moves down by enough ticks — update."""
        # current(50005) - new_stop(50000) = 5 > 1.25
        assert should_update_trail(50005.0, 50000.0, "short", 5, 0.25) is True

    def test_short_tiny_move_ignored(self) -> None:
        """Short stop moves down by less than step_ticks — skip."""
        assert should_update_trail(50005.0, 50004.0, "short", 5, 0.25) is False

    def test_short_move_up_ignored(self) -> None:
        """Short stop moving up (loosening) — never update."""
        assert should_update_trail(50000.0, 50005.0, "short", 5, 0.25) is False

    def test_custom_step_ticks(self) -> None:
        """Larger step_ticks means more tolerance before updating."""
        # step=10, tick_size=0.25 → min=2.5, delta=2.0 < 2.5
        assert should_update_trail(50000.0, 50002.0, "long", 10, 0.25) is False
        # delta=3.0 > 2.5
        assert should_update_trail(50000.0, 50003.0, "long", 10, 0.25) is True


# ===========================================================================
# Internal: _favorable_move
# ===========================================================================


class TestFavorableMove:
    """Internal helper: compute absolute favorable price move."""

    def test_long_favorable(self) -> None:
        """Long: price above entry → positive favorable move."""
        assert _favorable_move(100.0, 120.0, 200.0) == 20.0

    def test_long_unfavorable_returns_zero(self) -> None:
        """Long: price below entry → zero."""
        assert _favorable_move(100.0, 90.0, 200.0) == 0.0

    def test_short_favorable(self) -> None:
        """Short: price below entry → positive favorable move."""
        assert _favorable_move(200.0, 170.0, 100.0) == 30.0

    def test_short_unfavorable_returns_zero(self) -> None:
        """Short: price above entry → zero."""
        assert _favorable_move(200.0, 210.0, 100.0) == 0.0

    def test_flat_returns_zero(self) -> None:
        """No movement → zero."""
        assert _favorable_move(100.0, 100.0, 200.0) == 0.0


# ===========================================================================
# Adaptive state unit tests
# ===========================================================================


class TestAdaptiveState:
    """AdaptiveState: tracking trade results and EMA drift."""

    def test_initial_state_defaults(self) -> None:
        """Fresh state has default values."""
        state = AdaptiveState()
        assert state.total_trades == 0
        assert state.winning_trades == 0
        assert state.optimal_std_dev == 2.0
        assert state.optimal_conf_threshold == 0.6
        assert state.optimal_risk_pct == 0.25
        assert len(state.recent_win_params) == 0

    def test_loss_does_not_change_params(self) -> None:
        """A losing trade does not affect EMA parameters."""
        state = AdaptiveState()
        state.update({
            "was_winner": False,
            "signal_std_dev": 3.0,
            "signal_confidence": 0.9,
            "risk_pct": 0.5,
            "pnl": -100.0,
        })
        assert state.total_trades == 1
        assert state.winning_trades == 0
        # Params unchanged.
        assert state.optimal_std_dev == 2.0
        assert state.optimal_conf_threshold == 0.6
        assert state.optimal_risk_pct == 0.25

    def test_single_win_updates_ema(self) -> None:
        """A winning trade shifts EMA toward its parameters."""
        state = AdaptiveState(optimal_std_dev=2.0, _alpha=0.15)
        # Win with std_dev=3.0 → EMA: 2.0*0.85 + 3.0*0.15 = 1.7 + 0.45 = 2.15
        state.update({
            "was_winner": True,
            "signal_std_dev": 3.0,
            "signal_confidence": 0.8,
            "risk_pct": 0.3,
            "pnl": 50.0,
        })
        assert state.total_trades == 1
        assert state.winning_trades == 1
        assert state.optimal_std_dev == pytest.approx(2.15)
        # confidence: 0.6*0.85 + 0.8*0.15 = 0.51 + 0.12 = 0.63
        assert state.optimal_conf_threshold == pytest.approx(0.63)
        # risk: 0.25*0.85 + 0.3*0.15 = 0.2125 + 0.045 = 0.2575
        assert state.optimal_risk_pct == pytest.approx(0.2575)

    def test_multiple_wins_converge(self) -> None:
        """Multiple wins with same params converge EMA toward those values."""
        state = AdaptiveState(optimal_std_dev=2.0, _alpha=0.15)
        for _ in range(20):
            state.update({
                "was_winner": True,
                "signal_std_dev": 3.0,
                "signal_confidence": 0.75,
                "risk_pct": 0.3,
                "pnl": 25.0,
            })
        # After many updates, EMA should be close to 3.0.
        assert state.optimal_std_dev == pytest.approx(3.0, abs=0.05)
        assert state.optimal_conf_threshold == pytest.approx(0.75, abs=0.05)
        assert state.optimal_risk_pct == pytest.approx(0.3, abs=0.05)

    def test_recent_win_params_capped(self) -> None:
        """recent_win_params buffer is capped at RECENT_WINS_CAP."""
        state = AdaptiveState()
        for i in range(60):
            state.update({
                "was_winner": True,
                "signal_std_dev": 2.0 + i * 0.01,
                "signal_confidence": 0.6,
                "risk_pct": 0.25,
                "pnl": 10.0,
            })
        assert len(state.recent_win_params) == 50  # RECENT_WINS_CAP

    def test_total_trades_includes_losses(self) -> None:
        """total_trades counts both wins and losses."""
        state = AdaptiveState()
        state.update({"was_winner": True, "signal_std_dev": 2.0, "signal_confidence": 0.6, "risk_pct": 0.25, "pnl": 10.0})
        state.update({"was_winner": False, "signal_std_dev": 2.0, "signal_confidence": 0.6, "risk_pct": 0.25, "pnl": -10.0})
        state.update({"was_winner": True, "signal_std_dev": 2.0, "signal_confidence": 0.6, "risk_pct": 0.25, "pnl": 15.0})
        assert state.total_trades == 3
        assert state.winning_trades == 2

    def test_missing_keys_default_to_current(self) -> None:
        """Missing keys in trade_result use current optimal values."""
        state = AdaptiveState()
        state.update({
            "was_winner": True,
            "pnl": 20.0,
            # No signal_std_dev, signal_confidence, risk_pct
        })
        # All should still be at defaults — EMA toward itself = no-op.
        assert state.winning_trades == 1
        assert state.optimal_std_dev == 2.0
        assert state.optimal_conf_threshold == 0.6
        assert state.optimal_risk_pct == 0.25


# ===========================================================================
# Adaptive parameter bounds
# ===========================================================================


class TestAdaptiveBounds:
    """Parameters must never drift more than 50% from defaults."""

    def test_std_dev_upper_bound(self) -> None:
        """optimal_std_dev capped at 2.0 × 1.5 = 3.0."""
        state = AdaptiveState(optimal_std_dev=2.95, _alpha=1.0)  # alpha=1.0 = jump fully
        state.update({
            "was_winner": True,
            "signal_std_dev": 10.0,  # extreme
            "signal_confidence": 0.6,
            "risk_pct": 0.25,
            "pnl": 30.0,
        })
        assert state.optimal_std_dev == pytest.approx(3.0)  # clamped

    def test_std_dev_lower_bound(self) -> None:
        """optimal_std_dev floor at 2.0 × 0.5 = 1.0."""
        state = AdaptiveState(optimal_std_dev=1.05, _alpha=1.0)
        state.update({
            "was_winner": True,
            "signal_std_dev": 0.1,
            "signal_confidence": 0.6,
            "risk_pct": 0.25,
            "pnl": 30.0,
        })
        assert state.optimal_std_dev == pytest.approx(1.0)  # clamped

    def test_confidence_upper_bound(self) -> None:
        """optimal_conf_threshold capped at 0.6 × 1.5 = 0.9."""
        state = AdaptiveState(optimal_conf_threshold=0.85, _alpha=1.0)
        state.update({
            "was_winner": True,
            "signal_std_dev": 2.0,
            "signal_confidence": 2.0,
            "risk_pct": 0.25,
            "pnl": 30.0,
        })
        assert state.optimal_conf_threshold == pytest.approx(0.9)

    def test_confidence_lower_bound(self) -> None:
        """optimal_conf_threshold floor at 0.6 × 0.5 = 0.3."""
        state = AdaptiveState(optimal_conf_threshold=0.35, _alpha=1.0)
        state.update({
            "was_winner": True,
            "signal_std_dev": 2.0,
            "signal_confidence": 0.0,
            "risk_pct": 0.25,
            "pnl": 30.0,
        })
        assert state.optimal_conf_threshold == pytest.approx(0.3)

    def test_risk_pct_upper_bound(self) -> None:
        """optimal_risk_pct capped at 0.25 × 1.5 = 0.375."""
        state = AdaptiveState(optimal_risk_pct=0.35, _alpha=1.0)
        state.update({
            "was_winner": True,
            "signal_std_dev": 2.0,
            "signal_confidence": 0.6,
            "risk_pct": 1.0,
            "pnl": 30.0,
        })
        assert state.optimal_risk_pct == pytest.approx(0.375)

    def test_risk_pct_lower_bound(self) -> None:
        """optimal_risk_pct floor at 0.25 × 0.5 = 0.125."""
        state = AdaptiveState(optimal_risk_pct=0.13, _alpha=1.0)
        state.update({
            "was_winner": True,
            "signal_std_dev": 2.0,
            "signal_confidence": 0.6,
            "risk_pct": 0.0,
            "pnl": 30.0,
        })
        assert state.optimal_risk_pct == pytest.approx(0.125)


# ===========================================================================
# _bounded_ema helper
# ===========================================================================


class TestBoundedEma:
    """Internal _bounded_ema: EMA + clamp to default ± drift."""

    def test_basic_ema_no_clamp(self) -> None:
        """EMA within bounds passes through."""
        result = _bounded_ema(2.0, 2.5, 0.15, 2.0, 0.5)
        # raw = 2.0*0.85 + 2.5*0.15 = 1.7 + 0.375 = 2.075
        assert result == pytest.approx(2.075)

    def test_clamped_high(self) -> None:
        """Value above upper bound gets clamped."""
        result = _bounded_ema(2.95, 5.0, 0.5, 2.0, 0.5)
        # raw = 2.95*0.5 + 5.0*0.5 = 3.975. Upper = 2.0*1.5 = 3.0.
        assert result == pytest.approx(3.0)

    def test_clamped_low(self) -> None:
        """Value below lower bound gets clamped."""
        result = _bounded_ema(1.05, -5.0, 0.5, 2.0, 0.5)
        # raw = 1.05*0.5 + (-5.0)*0.5 = -1.975. Lower = 2.0*0.5 = 1.0.
        assert result == pytest.approx(1.0)

    def test_different_default(self) -> None:
        """Bounds are calculated from the default parameter."""
        result = _bounded_ema(9.0, 20.0, 0.5, 10.0, 0.2)
        # raw = 9.0*0.5 + 20.0*0.5 = 14.5. Upper = 10.0*1.2 = 12.0.
        assert result == pytest.approx(12.0)


# ===========================================================================
# get_adapted_params
# ===========================================================================


class TestGetAdaptedParams:
    """get_adapted_params: merge adapted values with defaults."""

    def test_returns_defaults_before_threshold(self) -> None:
        """Fewer than 10 wins → defaults only."""
        state = AdaptiveState(winning_trades=9)
        defaults = {
            "mean_reversion_std_dev": 2.0,
            "min_confidence_threshold": 0.6,
            "risk_per_trade_pct": 0.25,
            "extra_key": 42,
        }
        result = get_adapted_params(state, defaults)
        assert result == defaults

    def test_returns_adapted_after_threshold(self) -> None:
        """10+ wins → adapted values override defaults."""
        state = AdaptiveState(
            winning_trades=10,
            optimal_std_dev=2.5,
            optimal_conf_threshold=0.7,
            optimal_risk_pct=0.3,
        )
        defaults = {
            "mean_reversion_std_dev": 2.0,
            "min_confidence_threshold": 0.6,
            "risk_per_trade_pct": 0.25,
            "extra_key": 99,
        }
        result = get_adapted_params(state, defaults)
        assert result["mean_reversion_std_dev"] == 2.5
        assert result["min_confidence_threshold"] == 0.7
        assert result["risk_per_trade_pct"] == 0.3
        # Keys not managed by adaptive state fall through to defaults.
        assert result["extra_key"] == 99

    def test_exactly_at_threshold(self) -> None:
        """Exactly 10 wins → adapted params used."""
        state = AdaptiveState(winning_trades=10)
        defaults = {"mean_reversion_std_dev": 2.0, "min_confidence_threshold": 0.6, "risk_per_trade_pct": 0.25}
        result = get_adapted_params(state, defaults)
        assert result["mean_reversion_std_dev"] == state.optimal_std_dev


# ===========================================================================
# Confidence-based position sizing
# ===========================================================================


class TestConfidenceSizing:
    """calculate_position_size with signal_confidence scales contracts."""

    def test_low_confidence_smaller_size(self) -> None:
        """0.6 confidence → scalar 0.8 → fewer contracts."""
        signal = _make_signal(confidence=0.6, stop_price=49900.0)
        config = StrategyConfig(risk_per_trade_pct=1.0, max_risk_per_trade=1000.0)
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}
        # Without confidence: risk_budget=1250, stop_distance=100, pdv=0.10 → 125 contracts
        # Capped at 40 from instrument limit.
        size_no_conf = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
        )
        assert size_no_conf == 40

        # With confidence=0.6: scalar=0.8, 125*0.8=100 → capped at 40.
        size_with_conf = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
            signal_confidence=0.6,
        )
        assert size_with_conf == 40  # Still capped by instrument limit

    def test_high_confidence_near_full_size(self) -> None:
        """0.95 confidence → scalar 0.975 → very near full size."""
        signal = _make_signal(confidence=0.95, stop_price=49990.0)
        config = StrategyConfig(risk_per_trade_pct=0.5, max_risk_per_trade=500.0)
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}
        # Without: risk_budget=min(625,500)=500, stop=10, pdv=0.10 → 500 contracts
        # Capped at 40.
        size_no_conf = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
        )
        assert size_no_conf == 40

        # With: scalar=0.975 → 500*0.975=487.5 → floor 487 → capped at 40.
        size_with = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
            signal_confidence=0.95,
        )
        assert size_with == 40  # Still capped

    def test_confidence_reduces_below_instrument_limit(self) -> None:
        """Confidence scaling can push size below instrument cap."""
        signal = _make_signal(confidence=0.6, stop_price=49950.0)
        config = StrategyConfig(risk_per_trade_pct=0.25, max_risk_per_trade=300.0)
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}
        # Without confidence: 300 / (50*0.10) = 60 → cap at 40
        size_orig = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
        )
        assert size_orig == 40

        # With confidence=0.6: scalar=0.8, 60*0.8 = 48 → cap at 40 → still 40
        size_low = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
            signal_confidence=0.6,
        )
        assert size_low == 40  # Same

        # With confidence=0.4: scalar=0.7, 60*0.7 = 42 → cap at 40
        size_vlow = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
            signal_confidence=0.4,
        )
        assert size_vlow == 40

        # Using a scenario where raw contracts are low enough.
        # risk_budget=300, stop_dist=200, pdv=0.10 → raw=15
        signal2 = _make_signal(confidence=0.6, stop_price=49800.0)  # stop dist=200
        # raw = floor(300 / (200*0.10)) = 15
        # With conf=0.6: scalar=0.8 → floor(15*0.8)=12
        size2 = calculate_position_size(
            signal2, _make_account(), _make_risk_state(), [], config, limits,
            signal_confidence=0.6,
        )
        assert size2 == 12  # Clearly below limit now

    def test_perfect_confidence_full_size(self) -> None:
        """1.0 confidence → scalar 1.0 → full size."""
        signal = _make_signal(confidence=1.0, stop_price=49900.0)
        config = StrategyConfig(risk_per_trade_pct=0.25, max_risk_per_trade=300.0)
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}
        size = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
            signal_confidence=1.0,
        )
        # 300 / (100*0.10) = 30, scalar=1.0 → 30
        assert size == 30

    def test_none_confidence_no_scaling(self) -> None:
        """signal_confidence=None → no scaling applied (backward compat)."""
        signal = _make_signal(confidence=0.8, stop_price=49900.0)
        config = StrategyConfig(risk_per_trade_pct=0.25, max_risk_per_trade=300.0)
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}
        size = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], config, limits,
        )
        assert size == 30  # No confidence scaling, full 30


# ===========================================================================
# Engine integration — confidence passed to sizing
# ===========================================================================


class TestEngineConfidenceSizing:
    """StrategyEngine.calculate_size passes signal.confidence."""

    def test_engine_passes_confidence(self) -> None:
        """The engine's calculate_size delegates with signal_confidence."""
        engine = StrategyEngine(StrategyConfig(symbols=["MBT"]))
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50100.0,
            rationale="test",
            timestamp=time.time(),
        )
        size = engine.calculate_size(
            signal, _make_account(), _make_risk_state(), [],
        )
        # Without confidence: 30. With confidence=0.8: scalar=0.9 → 27.
        assert size == 27

    def test_low_confidence_reduces_engine_size(self) -> None:
        """Lower confidence → fewer contracts from engine."""
        engine = StrategyEngine(StrategyConfig(symbols=["MBT"]))
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.3,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50100.0,
            rationale="test",
            timestamp=time.time(),
        )
        size = engine.calculate_size(
            signal, _make_account(), _make_risk_state(), [],
        )
        # scalar = 0.5 + 0.3*0.5 = 0.65, 30*0.65=19.5 → 19
        assert size < 25  # definitely less than without scaling


# ===========================================================================
# Engine — record_trade_result and last_adapted_params
# ===========================================================================


class TestEngineAdaptiveIntegration:
    """StrategyEngine.record_trade_result and last_adapted_params."""

    def test_record_trade_result_updates_state(self) -> None:
        """record_trade_result feeds into adaptive_state."""
        engine = StrategyEngine(StrategyConfig())
        assert engine.adaptive_state.total_trades == 0

        engine.record_trade_result({
            "was_winner": True,
            "signal_std_dev": 2.5,
            "signal_confidence": 0.7,
            "risk_pct": 0.3,
            "pnl": 25.0,
        })
        assert engine.adaptive_state.total_trades == 1
        assert engine.adaptive_state.winning_trades == 1

    def test_last_adapted_params_none_before_threshold(self) -> None:
        """last_adapted_params is None when < 10 wins."""
        engine = StrategyEngine(StrategyConfig())
        assert engine.last_adapted_params is None

        # Add 9 wins.
        for _ in range(9):
            engine.record_trade_result({
                "was_winner": True,
                "signal_std_dev": 2.0,
                "signal_confidence": 0.6,
                "risk_pct": 0.25,
                "pnl": 10.0,
            })
        assert engine.last_adapted_params is None

    def test_last_adapted_params_returns_dict_after_threshold(self) -> None:
        """last_adapted_params returns adapted values after 10 wins."""
        engine = StrategyEngine(StrategyConfig())
        for _ in range(10):
            engine.record_trade_result({
                "was_winner": True,
                "signal_std_dev": 2.5,
                "signal_confidence": 0.7,
                "risk_pct": 0.3,
                "pnl": 15.0,
            })
        params = engine.last_adapted_params
        assert params is not None
        assert "mean_reversion_std_dev" in params
        assert "min_confidence_threshold" in params
        assert "risk_per_trade_pct" in params
        # After 10 identical wins, EMA should have drifted toward 2.5.
        assert params["mean_reversion_std_dev"] > 2.0

    def test_engine_has_trailing_config(self) -> None:
        """Engine initializes with default TrailingConfig."""
        engine = StrategyEngine(StrategyConfig())
        assert engine.trailing_config is not None
        assert engine.trailing_config.activation_pct == 0.3
        assert engine.trailing_config.trail_distance_ticks == 20
        assert engine.trailing_config.step_ticks == 5

    def test_engine_has_adaptive_state(self) -> None:
        """Engine initializes with fresh AdaptiveState."""
        engine = StrategyEngine(StrategyConfig())
        assert engine.adaptive_state is not None
        assert engine.adaptive_state.total_trades == 0
        assert engine.adaptive_state.winning_trades == 0
