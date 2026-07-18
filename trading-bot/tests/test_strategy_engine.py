"""
Tests for the strategy engine — indicators, signals, sizing, and the
full StrategyEngine integration.

Covers:
- VWAP / std-dev calculation
- Volume spike detection
- Consecutive direction detection
- Mean-reversion signal generation
- Breakout signal generation
- Confidence scoring
- Cooldown enforcement
- Position sizing with various risk budgets
- Spread filtering
- Empty buffer handling
- Max position capping
"""

from __future__ import annotations

import math
import time
from collections import deque
from datetime import date
from unittest.mock import patch

import pytest

from src.client.models import Account, Position, Quote
from src.risk.limits import InstrumentLimit
from src.risk.state import RiskState
from src.strategy.engine import StrategyEngine
from src.strategy.indicators import (
    compute_micro_price,
    compute_std_dev,
    compute_vwap,
    detect_consecutive_direction,
    detect_volume_spike,
    get_consecutive_direction_sign,
    is_spread_eligible,
)
from src.strategy.signals import MarketSnapshot, Signal
from src.strategy.sizing import StrategyConfig, calculate_position_size


# ===========================================================================
# Test data helpers
# ===========================================================================


def _make_quote(
    symbol: str = "MBT",
    bid: float = 50000.0,
    ask: float = 50005.0,
    last: float = 50002.0,
    volume: int = 1000,
    timestamp: str = "2026-07-18T12:00:00Z",
) -> Quote:
    """Create a minimal Quote for testing."""
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        timestamp=timestamp,
    )


def _make_quote_sequence(
    symbol: str = "MBT",
    base_price: float = 50000.0,
    n: int = 60,
    *,
    trend: float = 0.0,
    noise_std: float = 5.0,
    volume_start: int = 1000,
    volume_step: int = 1,
) -> list[Quote]:
    """Generate *n* synthetic quotes with optional trend and noise.

    Args:
        symbol: Contract symbol.
        base_price: Starting price.
        n: Number of quotes.
        trend: Drift per tick (positive = uptrend).
        noise_std: Standard deviation of gaussian price noise.
        volume_start: Starting cumulative volume.
        volume_step: Volume increment per tick.

    Returns:
        List of Quote objects.
    """
    import random
    rng = random.Random(42)  # deterministic
    quotes = []
    price = base_price
    vol = volume_start
    for i in range(n):
        price += trend + rng.gauss(0, noise_std)
        vol += volume_step
        ts = f"2026-07-18T12:{i//60:02d}:{i%60:02d}Z"
        quotes.append(
            _make_quote(
                symbol=symbol,
                bid=price - 2.5,
                ask=price + 2.5,
                last=price,
                volume=vol,
                timestamp=ts,
            )
        )
    return quotes


def _make_account(
    net_liq: float = 50000.0,
    balance: float = 50000.0,
    available_funds: float = 50000.0,
) -> Account:
    return Account(
        id=1,
        name="Test",
        net_liq=net_liq,
        balance=balance,
        available_funds=available_funds,
    )


def _make_risk_state(
    session_realized_pnl: float = 0.0,
    peak_equity: float = 50000.0,
) -> RiskState:
    return RiskState(
        session_date=date.today(),
        session_realized_pnl=session_realized_pnl,
        peak_equity=peak_equity,
        starting_equity=50000.0,
    )


def _make_position(symbol: str = "MBT", net_pos: int = 0) -> Position:
    return Position(
        id=1,
        account_id=1,
        symbol=symbol,
        net_pos=net_pos,
        avg_price=50000.0,
    )


# ===========================================================================
# MarketSnapshot tests
# ===========================================================================


class TestMarketSnapshot:
    def test_defaults_derive_micro_price_and_spread_bps(self) -> None:
        snap = MarketSnapshot(
            symbol="MBT",
            bid=50000.0,
            ask=50005.0,
            last=50002.0,
            spread=5.0,
            volume=1000,
            timestamp=time.time(),
        )
        assert snap.micro_price == pytest.approx(50002.5)
        # spread_bps = (5.0 / 50002.5) * 10000 ≈ 1.0
        assert snap.spread_bps == pytest.approx(1.0, rel=0.1)

    def test_explicit_micro_price_respected(self) -> None:
        snap = MarketSnapshot(
            symbol="MBT",
            bid=50000.0,
            ask=50005.0,
            last=50002.0,
            spread=5.0,
            volume=1000,
            timestamp=time.time(),
            micro_price=50002.0,
            spread_bps=1.0,
        )
        assert snap.micro_price == 50002.0
        assert snap.spread_bps == 1.0

    def test_zero_reference_price_avoids_division_by_zero(self) -> None:
        snap = MarketSnapshot(
            symbol="MBT",
            bid=0.0,
            ask=0.0,
            last=0.0,
            spread=0.0,
            volume=0,
            timestamp=time.time(),
            micro_price=None,
        )
        assert snap.spread_bps == 0.0


# ===========================================================================
# Signal tests
# ===========================================================================


class TestSignal:
    def test_signal_creation(self) -> None:
        sig = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.85,
            entry_price=50000.5,
            stop_price=49950.0,
            target_price=50100.0,
            rationale="Test signal",
            timestamp=time.time(),
        )
        assert sig.symbol == "MBT"
        assert sig.direction == "long"
        assert sig.confidence == 0.85

    def test_flat_signal(self) -> None:
        sig = Signal(
            symbol="MBT",
            direction="flat",
            confidence=0.0,
            entry_price=None,
            stop_price=50000.0,
            target_price=50000.0,
            rationale="No action",
            timestamp=time.time(),
        )
        assert sig.direction == "flat"


# ===========================================================================
# Indicator tests
# ===========================================================================


class TestMicroPrice:
    def test_normal_market(self) -> None:
        assert compute_micro_price(50000.0, 50005.0) == 50002.5

    def test_crossed_market_raises(self) -> None:
        with pytest.raises(ValueError, match="Crossed"):
            compute_micro_price(50005.0, 50000.0)

    def test_tight_spread(self) -> None:
        assert compute_micro_price(50000.0, 50000.5) == 50000.25


class TestVWAP:
    def test_single_quote(self) -> None:
        quotes = [_make_quote(last=50000.0, volume=1000)]
        vwap = compute_vwap(quotes)
        assert vwap == pytest.approx(50000.0)

    def test_multiple_quotes(self) -> None:
        quotes = _make_quote_sequence(base_price=50000.0, n=10, trend=0, noise_std=0)
        # All same price → VWAP = that price.
        vwap = compute_vwap(quotes)
        assert vwap == pytest.approx(50000.0)

    def test_empty_quotes(self) -> None:
        assert compute_vwap([]) is None

    def test_vwap_with_volume_weighting(self) -> None:
        q1 = _make_quote(last=100.0, volume=10)
        q2 = _make_quote(last=200.0, volume=20)
        # tick_vol for q1=1, q2=10. VWAP = (100*1 + 200*10) / 11 = 2100/11 ≈ 190.91
        vwap = compute_vwap([q1, q2])
        assert vwap is not None
        # tick_vol for first = 1, for second = max(1, 20-10) = 10
        expected = (100.0 * 1 + 200.0 * 10) / 11
        assert vwap == pytest.approx(expected)


class TestStdDev:
    def test_zero_dispersion(self) -> None:
        quotes = _make_quote_sequence(base_price=50000.0, n=10, trend=0, noise_std=0)
        vwap = compute_vwap(quotes)
        std = compute_std_dev(quotes, vwap)
        assert std == pytest.approx(0.0, abs=1e-9)

    def test_positive_dispersion(self) -> None:
        quotes = [
            _make_quote(last=100.0),
            _make_quote(last=105.0),
            _make_quote(last=95.0),
        ]
        std = compute_std_dev(quotes)
        # Mean = 100, variance = ((0)² + (5)² + (-5)²)/3 = 50/3, std = sqrt(50/3) ≈ 4.08
        assert std is not None
        assert std == pytest.approx(math.sqrt(50 / 3))

    def test_too_few_quotes(self) -> None:
        assert compute_std_dev([_make_quote()]) is None

    def test_auto_computes_vwap(self) -> None:
        quotes = _make_quote_sequence(base_price=50000.0, n=10, trend=0, noise_std=0)
        std = compute_std_dev(quotes)  # vwap computed on the fly
        assert std is not None


class TestVolumeSpike:
    def test_no_spike_normal_volume(self) -> None:
        quotes = _make_quote_sequence(n=10, volume_step=1)
        assert not detect_volume_spike(quotes, multiplier=2.0)

    def test_spike_detected(self) -> None:
        # Regular volumes then a big jump.
        quotes = _make_quote_sequence(n=10, volume_step=1)  # tick vols of 1
        # Add a quote with a huge volume increase.
        last = quotes[-1]
        spike_q = _make_quote(
            last=last.last,
            volume=last.volume + 100,  # tick vol = 100
            timestamp="2026-07-18T12:10:00Z",
        )
        quotes.append(spike_q)
        assert detect_volume_spike(quotes, multiplier=2.0)

    def test_insufficient_quotes(self) -> None:
        assert not detect_volume_spike([_make_quote()], multiplier=2.0)

    def test_spike_at_exact_threshold_not_spike(self) -> None:
        quotes = _make_quote_sequence(n=5, volume_step=1)
        # tick vols: 1, 1, 1, 1, 1 → avg baseline = 1. spike needs > 2.0 * 1 = 2
        # So tick vol = 2 is NOT a spike.
        last = quotes[-1]
        spike_q = _make_quote(last=last.last, volume=last.volume + 2)
        quotes.append(spike_q)
        assert not detect_volume_spike(quotes, multiplier=2.0)


class TestConsecutiveDirection:
    def test_consecutive_up_noise_sequence(self) -> None:
        quotes = _make_quote_sequence(n=7, trend=10.0, noise_std=0)
        assert detect_consecutive_direction(quotes, n=5)
        assert get_consecutive_direction_sign(quotes, n=5) == 1

    def test_consecutive_down(self) -> None:
        quotes = _make_quote_sequence(n=7, trend=-10.0, noise_std=0)
        assert detect_consecutive_direction(quotes, n=5)
        assert get_consecutive_direction_sign(quotes, n=5) == -1

    def test_mixed_direction_no_signal(self) -> None:
        # Alternating up/down.
        prices = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0]
        quotes = [
            _make_quote(last=p, volume=1000 + i)
            for i, p in enumerate(prices)
        ]
        assert not detect_consecutive_direction(quotes, n=5)

    def test_insufficient_quotes(self) -> None:
        quotes = _make_quote_sequence(n=3)
        assert not detect_consecutive_direction(quotes, n=5)

    def test_zero_diff_direction(self) -> None:
        # Flat price — no direction.
        quotes = _make_quote_sequence(base_price=50000.0, n=7, trend=0, noise_std=0)
        # All prices = 50000.0 exactly.
        assert not detect_consecutive_direction(quotes, n=5)
        assert get_consecutive_direction_sign(quotes, n=5) == 0


class TestSpreadFilter:
    def test_eligible(self) -> None:
        assert is_spread_eligible(50000.0, 50001.0, max_spread_bps=5.0)
        # spread = 1.0, mid = 50000.5, bps = (1/50000.5)*10000 ≈ 0.2 < 5

    def test_not_eligible(self) -> None:
        assert not is_spread_eligible(50000.0, 50050.0, max_spread_bps=5.0)
        # Also test with 100 bps spread with a 5 bps limit
        assert not is_spread_eligible(50000.0, 50500.0, max_spread_bps=5.0)
        # spread = 50, mid = 50025, bps = (50/50025)*10000 ≈ 9.995 > 5 → not eligible

    def test_zero_price(self) -> None:
        assert not is_spread_eligible(0.0, 0.0, max_spread_bps=5.0)

    def test_exact_boundary(self) -> None:
        # Mid = 50000, max bps = 5 → max spread = 50000 * 5 / 10000 = 25
        assert is_spread_eligible(50000.0, 50025.0, max_spread_bps=5.0)
        assert not is_spread_eligible(50000.0, 50025.1, max_spread_bps=5.0)


# ===========================================================================
# Position sizing tests
# ===========================================================================


class TestPositionSizing:
    def test_basic_sizing(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,  # $100 stop distance
            target_price=50200.0,
            rationale="test",
            timestamp=time.time(),
        )
        account = _make_account()
        risk_state = _make_risk_state()
        positions: list[Position] = []
        config = StrategyConfig()
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}

        size = calculate_position_size(
            signal, account, risk_state, positions, config, limits,
        )
        # risk_budget = min(1250*0.25, 300) = min(312.5, 300) = 300
        # stop_distance_$ = 100 * 0.10 = 10.0
        # contracts = floor(300/10) = 30
        assert size == 30

    def test_sizing_hit_max_risk_cap(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49999.0,  # $1 stop distance
            target_price=50010.0,
            rationale="test",
            timestamp=time.time(),
        )
        account = _make_account()
        risk_state = _make_risk_state()
        positions: list[Position] = []
        config = StrategyConfig(max_risk_per_trade=50.0)
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}

        size = calculate_position_size(
            signal, account, risk_state, positions, config, limits,
        )
        # risk_budget = min(1250*0.25, 50) = 50
        # stop_distance_$ = 1 * 0.10 = 0.10
        # contracts = floor(50/0.10) = 500, cap at 40
        assert size == 40

    def test_sizing_zero_when_flat_signal(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="flat",
            confidence=0.0,
            entry_price=None,
            stop_price=50000.0,
            target_price=50000.0,
            rationale="no trade",
            timestamp=time.time(),
        )
        assert calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], StrategyConfig(), {}
        ) == 0

    def test_sizing_zero_when_no_remaining_daily_loss(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50200.0,
            rationale="test",
            timestamp=time.time(),
        )
        risk_state = _make_risk_state(session_realized_pnl=-1250.0)  # daily loss blown
        assert calculate_position_size(
            signal, _make_account(), risk_state, [], StrategyConfig(), {}
        ) == 0

    def test_sizing_capped_by_existing_position(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50200.0,
            rationale="test",
            timestamp=time.time(),
        )
        positions = [_make_position(symbol="MBT", net_pos=35)]  # only 5 left
        config = StrategyConfig()
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}

        size = calculate_position_size(
            signal, _make_account(), _make_risk_state(), positions, config, limits,
        )
        assert size == 5

    def test_sizing_zero_when_position_at_limit(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50200.0,
            rationale="test",
            timestamp=time.time(),
        )
        positions = [_make_position(symbol="MBT", net_pos=40)]
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}
        assert calculate_position_size(
            signal, _make_account(), _make_risk_state(), positions,
            StrategyConfig(), limits,
        ) == 0

    def test_sizing_fallback_limit_for_unknown_symbol(self) -> None:
        signal = Signal(
            symbol="XYZ",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50200.0,
            rationale="test",
            timestamp=time.time(),
        )
        # XYZ not in limits → heuristic: doesn't start with M → 4
        size = calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], StrategyConfig(), {}
        )
        assert size <= 4

    def test_sizing_zero_when_stop_distance_zero(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=50000.0,  # no stop distance
            target_price=50100.0,
            rationale="test",
            timestamp=time.time(),
        )
        assert calculate_position_size(
            signal, _make_account(), _make_risk_state(), [], StrategyConfig(), {}
        ) == 0

    def test_sizing_with_partial_daily_loss(self) -> None:
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50200.0,
            rationale="test",
            timestamp=time.time(),
        )
        risk_state = _make_risk_state(session_realized_pnl=-500.0)
        # remaining daily loss = 1250 - 500 = 750
        # risk_budget = min(750*0.25, 300) = min(187.5, 300) = 187.5
        # stop_distance_$ = 100 * 0.10 = 10
        # contracts = floor(187.5 / 10) = 18
        limits = {"MBT": InstrumentLimit(max_contracts=40, tick_value=0.50)}
        size = calculate_position_size(
            signal, _make_account(), risk_state, [], StrategyConfig(), limits,
        )
        assert size == 18


# ===========================================================================
# StrategyConfig tests
# ===========================================================================


class TestStrategyConfig:
    def test_defaults(self) -> None:
        cfg = StrategyConfig()
        assert cfg.symbols == ["MBT", "MET"]
        assert cfg.primary_symbol == "MBT"
        assert cfg.risk_per_trade_pct == 0.25
        assert cfg.max_risk_per_trade == 300.0
        assert cfg.mean_reversion_std_dev == 2.0
        assert cfg.breakout_consecutive_ticks == 5
        assert cfg.quote_window_size == 200

    def test_custom_config(self) -> None:
        cfg = StrategyConfig(
            symbols=["MBT"],
            primary_symbol="MBT",
            risk_per_trade_pct=0.10,
        )
        assert cfg.symbols == ["MBT"]
        assert cfg.risk_per_trade_pct == 0.10

    def test_point_values_default(self) -> None:
        cfg = StrategyConfig()
        assert cfg.point_values["MBT"] == 0.10
        assert cfg.point_values["MET"] == 0.10

    def test_custom_point_values(self) -> None:
        cfg = StrategyConfig(point_values={"MBT": 0.50})
        assert cfg.point_values["MBT"] == 0.50


# ===========================================================================
# StrategyEngine integration tests
# ===========================================================================


class TestStrategyEngine:
    def test_init_creates_buffers(self) -> None:
        engine = StrategyEngine(StrategyConfig(symbols=["MBT", "MET"]))
        assert len(engine._quote_buffers) == 2
        assert "MBT" in engine._quote_buffers
        assert "MET" in engine._quote_buffers

    def test_ingest_quote_appends_to_buffer(self) -> None:
        engine = StrategyEngine(StrategyConfig(symbols=["MBT"]))
        q = _make_quote(symbol="MBT")
        engine.ingest_quote(q)
        assert len(engine._quote_buffers["MBT"]) == 1

    def test_ingest_unknown_symbol_dropped(self) -> None:
        engine = StrategyEngine(StrategyConfig(symbols=["MBT"]))
        q = _make_quote(symbol="XYZ")
        engine.ingest_quote(q)
        # Should not raise, buffer should not exist.
        assert "XYZ" not in engine._quote_buffers

    def test_clear_buffers_flushes_data(self) -> None:
        engine = StrategyEngine(StrategyConfig(symbols=["MBT", "MET"]))
        for _ in range(10):
            engine.ingest_quote(_make_quote(symbol="MBT"))
            engine.ingest_quote(_make_quote(symbol="MET"))
        assert len(engine._quote_buffers["MBT"]) == 10
        assert len(engine._quote_buffers["MET"]) == 10
        engine.clear_buffers()
        assert len(engine._quote_buffers["MBT"]) == 0
        assert len(engine._quote_buffers["MET"]) == 0

    def test_buffer_respects_maxlen(self) -> None:
        maxlen = 5
        engine = StrategyEngine(
            StrategyConfig(symbols=["MBT"], quote_window_size=maxlen)
        )
        for i in range(maxlen + 3):
            q = _make_quote(symbol="MBT", last=float(i))
            engine.ingest_quote(q)
        assert len(engine._quote_buffers["MBT"]) == maxlen
        # Oldest should have been evicted.
        prices = [q.last for q in engine._quote_buffers["MBT"]]
        assert prices[0] == 3.0  # index 3 = the 4th quote (0,1,2 evicted)

    @pytest.mark.asyncio
    async def test_generate_signals_insufficient_data(self) -> None:
        engine = StrategyEngine(StrategyConfig(symbols=["MBT"]))
        # Only 10 quotes, need 50.
        for q in _make_quote_sequence(n=10):
            engine.ingest_quote(q)
        signals = await engine.generate_signals()
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_mean_reversion_long(self) -> None:
        """Price significantly below VWAP → long signal."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT"],
                mean_reversion_std_dev=1.5,
                min_confidence_threshold=0.3,
                cooldown_seconds=0.0,
            )
        )
        # Generate 50 quotes around 50000, then push price far below.
        quotes = _make_quote_sequence(base_price=50000.0, n=50, trend=0, noise_std=3)
        for q in quotes:
            engine.ingest_quote(q)
        # Add a few very low quotes to trigger mean reversion.
        low_price = 49900.0
        for i in range(5):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    bid=low_price - 2.5,
                    ask=low_price + 2.5,
                    last=low_price,
                    volume=1050 + i,
                )
            )
        signals = await engine.generate_signals()
        assert len(signals) >= 1
        # Should be a "long" signal (price below VWAP → buy).
        long_sigs = [s for s in signals if s.direction == "long"]
        assert len(long_sigs) >= 1

    @pytest.mark.asyncio
    async def test_generate_signals_mean_reversion_short(self) -> None:
        """Price significantly above VWAP → short signal."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT"],
                mean_reversion_std_dev=1.5,
                min_confidence_threshold=0.3,
                cooldown_seconds=0.0,
            )
        )
        quotes = _make_quote_sequence(base_price=50000.0, n=50, trend=0, noise_std=3)
        for q in quotes:
            engine.ingest_quote(q)
        # Push price far above VWAP.
        high_price = 50100.0
        for i in range(5):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    bid=high_price - 2.5,
                    ask=high_price + 2.5,
                    last=high_price,
                    volume=1050 + i,
                )
            )
        signals = await engine.generate_signals()
        short_sigs = [s for s in signals if s.direction == "short"]
        assert len(short_sigs) >= 1

    @pytest.mark.asyncio
    async def test_generate_signals_breakout(self) -> None:
        """Consecutive same-direction ticks + volume spike → breakout."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT"],
                breakout_consecutive_ticks=4,
                volume_spike_multiplier=2.0,
                mean_reversion_std_dev=5.0,  # high threshold to suppress MR
                min_confidence_threshold=0.3,
                cooldown_seconds=0.0,
            )
        )
        # 50 normal quotes.
        quotes = _make_quote_sequence(base_price=50000.0, n=50, trend=0, noise_std=2)
        for q in quotes:
            engine.ingest_quote(q)
        # Now push a breakout: 5 consecutive up-ticks with volume spike.
        base = quotes[-1]
        for i in range(6):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    bid=base.last + i * 10,
                    ask=base.last + i * 10 + 5,
                    last=base.last + i * 10,
                    volume=base.volume + (100 if i == 5 else 1),  # spike on last
                )
            )
        signals = await engine.generate_signals()
        assert len(signals) >= 1

    @pytest.mark.asyncio
    async def test_cooldown_enforcement(self) -> None:
        """After a signal, no new signal within cooldown period."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT"],
                mean_reversion_std_dev=0.5,  # very sensitive
                min_confidence_threshold=0.3,
                cooldown_seconds=999.0,  # huge cooldown
            )
        )
        quotes = _make_quote_sequence(base_price=50000.0, n=55, trend=0, noise_std=3)
        for q in quotes:
            engine.ingest_quote(q)
        # Push extreme prices.
        for i in range(5):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    last=48000.0,
                    bid=47997.5,
                    ask=48002.5,
                    volume=1100 + i,
                )
            )
        sigs1 = await engine.generate_signals()
        assert len(sigs1) >= 1

        # Immediately try again — cooldown should block.
        sigs2 = await engine.generate_signals()
        assert sigs2 == []

    @pytest.mark.asyncio
    async def test_spread_filter_blocks_wide_spreads(self) -> None:
        """Quotes with wide spreads should not produce signals."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT"],
                min_spread_bps=1.0,  # very tight
                mean_reversion_std_dev=0.5,
                min_confidence_threshold=0.3,
                cooldown_seconds=0.0,
            )
        )
        quotes = _make_quote_sequence(base_price=50000.0, n=50, trend=0, noise_std=3)
        for q in quotes:
            engine.ingest_quote(q)
        # Add extreme price with wide spread.
        for i in range(5):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    bid=45000.0,
                    ask=55000.0,  # 10000 spread → huge bps
                    last=48000.0,
                    volume=1100 + i,
                )
            )
        signals = await engine.generate_signals()
        # Should be empty because spread is too wide.
        assert signals == []

    @pytest.mark.asyncio
    async def test_get_current_signal_returns_none(self) -> None:
        engine = StrategyEngine(StrategyConfig(symbols=["MBT"]))
        assert engine.get_current_signal("MBT") is None

    @pytest.mark.asyncio
    async def test_confidence_threshold_filters_weak_signals(self) -> None:
        """Signals below min_confidence_threshold are dropped."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT"],
                mean_reversion_std_dev=2.0,
                min_confidence_threshold=0.95,  # very high threshold
                cooldown_seconds=0.0,
            )
        )
        quotes = _make_quote_sequence(base_price=50000.0, n=55, trend=0, noise_std=2)
        for q in quotes:
            engine.ingest_quote(q)
        # Push somewhat extreme but not very extreme.
        for i in range(5):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    last=49900.0,
                    bid=49897.5,
                    ask=49902.5,
                    volume=1100 + i,
                )
            )
        signals = await engine.generate_signals()
        # Signals should be below threshold → dropped.
        assert signals == []

    def test_calculate_size_delegates(self) -> None:
        engine = StrategyEngine(StrategyConfig(symbols=["MBT"]))
        signal = Signal(
            symbol="MBT",
            direction="long",
            confidence=0.8,
            entry_price=50000.0,
            stop_price=49900.0,
            target_price=50200.0,
            rationale="test",
            timestamp=time.time(),
        )
        size = engine.calculate_size(
            signal, _make_account(), _make_risk_state(), [],
        )
        # risk_budget = min(1250*0.25, 300) = 300
        # stop_distance_$ = 100 * 0.10 = 10
        # contracts = floor(300/10) = 30
        # MBT starts with M → micro → limit 40
        assert size == 30

    @pytest.mark.asyncio
    async def test_multiple_symbols_evaluated(self) -> None:
        """Both MBT and MET are evaluated independently."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT", "MET"],
                mean_reversion_std_dev=0.5,
                min_confidence_threshold=0.3,
                cooldown_seconds=0.0,
            )
        )
        # Feed both symbols.
        mbt_quotes = _make_quote_sequence(
            symbol="MBT", base_price=50000.0, n=51, trend=0, noise_std=3
        )
        met_quotes = _make_quote_sequence(
            symbol="MET", base_price=3000.0, n=51, trend=0, noise_std=0.5
        )
        for mbt, met in zip(mbt_quotes, met_quotes):
            engine.ingest_quote(mbt)
            engine.ingest_quote(met)
        # Push extreme on MBT only.
        for i in range(5):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    last=48000.0,
                    bid=47997.5,
                    ask=48002.5,
                    volume=1100 + i,
                )
            )
        signals = await engine.generate_signals()
        # MBT should have at least one signal, MET may or may not.
        mbt_signals = [s for s in signals if s.symbol == "MBT"]
        assert len(mbt_signals) >= 1

    @pytest.mark.asyncio
    async def test_signal_has_stop_and_target(self) -> None:
        """Every generated signal must have valid stop and target prices."""
        engine = StrategyEngine(
            StrategyConfig(
                symbols=["MBT"],
                mean_reversion_std_dev=1.0,
                min_confidence_threshold=0.3,
                cooldown_seconds=0.0,
            )
        )
        quotes = _make_quote_sequence(base_price=50000.0, n=55, trend=0, noise_std=3)
        for q in quotes:
            engine.ingest_quote(q)
        for i in range(5):
            engine.ingest_quote(
                _make_quote(
                    symbol="MBT",
                    last=49800.0,
                    bid=49797.5,
                    ask=49802.5,
                    volume=1100 + i,
                )
            )
        signals = await engine.generate_signals()
        for sig in signals:
            if sig.direction == "long":
                assert sig.stop_price < sig.target_price
                assert sig.stop_price < (sig.entry_price or sig.stop_price)
            elif sig.direction == "short":
                assert sig.stop_price > sig.target_price
                assert sig.stop_price > (sig.entry_price or sig.stop_price)
