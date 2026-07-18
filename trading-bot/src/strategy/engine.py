"""
Strategy engine — rule-based mean-reversion + breakout hybrid.

The engine maintains a rolling quote buffer per symbol and evaluates
two independent signal types on every new tick:

**Mean reversion**
    When the last price deviates more than ``mean_reversion_std_dev``
    standard deviations from the rolling VWAP, we bet on reversion.

**Breakout**
    When we see *N* consecutive ticks in the same direction *and* the
    latest tick carries unusually high volume, we bet on continuation.

**Adaptive gating (Phase 2)**
    Before generating signals, the engine consults three filters:

    1. **Regime detection** — choppy markets suppress all signals.
    2. **Volatility filter** — excessive ATR blocks trading.
    3. **Session awareness** — adjusts risk, confidence, and regime
       preference by time of day.

**Trailing stops & adaptive tuning (Phase 2.2)**
    When enough winning trades have accumulated:

    - Trailing stop parameters can be applied to open positions.
    - Strategy parameters (std-dev threshold, confidence, risk %) drift
      toward what's produced winning trades via an EMA.
    - Position sizing scales with signal confidence.

The engine also enforces a per-symbol cooldown to avoid overtrading
and discards signals whose spread exceeds the configured limit.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from src.client.models import Account, Quote
from src.strategy.adaptive_tuning import AdaptiveState, get_adapted_params
from src.strategy.indicators import (
    compute_micro_price,
    compute_std_dev,
    compute_vwap,
    detect_consecutive_direction,
    detect_volume_spike,
    get_consecutive_direction_sign,
    is_spread_eligible,
)
from src.strategy.regime import MarketRegime, detect_regime
from src.strategy.session import (
    TradingSession,
    get_current_session,
    get_session_params,
)
from src.strategy.signals import MarketSnapshot, Signal
from src.strategy.sizing import StrategyConfig, calculate_position_size
from src.strategy.trailing import TrailingConfig
from src.strategy.volatility import (
    compute_atr,
    compute_volatility_ratio,
    is_safe_to_trade,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """Generates trade signals from streaming quote data.

    Usage sketch::

        engine = StrategyEngine(StrategyConfig())
        engine.ingest_quote(quote)               # push every tick
        signals = await engine.generate_signals(account)  # evaluate all symbols
        for sig in signals:
            if sig.confidence >= engine.config.min_confidence_threshold:
                size = engine.calculate_size(sig, acct, risk, positions)
                ...

    Attributes:
        config: Active strategy configuration (immutable after init).
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.symbols: list[str] = list(config.symbols)
        # Per-symbol rolling quote buffer (bounded deque).
        self._quote_buffers: dict[str, deque[Quote]] = {
            sym: deque(maxlen=config.quote_window_size) for sym in self.symbols
        }
        # Per-symbol cooldown — epoch seconds of last signal.
        self._last_signal_time: dict[str, float] = {
            sym: 0.0 for sym in self.symbols
        }
        # Historical ATR values for volatility ratio computation.
        self._historical_atr: deque[float] = deque(maxlen=50)
        # Cached latest regime for external inspection.
        self._last_regime: MarketRegime | None = None
        self._last_regime_confidence: float = 0.0
        # Phase 2.2: trailing stop configuration.
        self.trailing_config: TrailingConfig = TrailingConfig()
        # Phase 2.2: adaptive parameter tuning state.
        self.adaptive_state: AdaptiveState = AdaptiveState()

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest_quote(self, quote: Quote) -> None:
        """Push a quote into the per-symbol rolling buffer.

        If *symbol* is not tracked, the quote is silently dropped.
        """
        buf = self._quote_buffers.get(quote.symbol)
        if buf is not None:
            buf.append(quote)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        account: Account | None = None,
    ) -> list[Signal]:
        """Evaluate all tracked symbols and return actionable signals.

        Each symbol is evaluated independently:

        0. **Adaptive gates** (regime, volatility, session) run first.
        1. Skip if the buffer has fewer than 50 quotes.
        2. Skip if the spread is too wide (``min_spread_bps`` config).
        3. Compute VWAP, std-dev, micro-price, volume-spike flag.
        4. Check mean-reversion: price N std from VWAP → reversal.
        5. Check breakout: consecutive same-direction ticks + volume
           spike → momentum.
        6. Combine confidences (higher when both agree).
        7. Enforce per-symbol cooldown.

        Args:
            account: Optional account snapshot for volatility safety
                check.  When ``None``, the account-based volatility
                gate is skipped.

        Returns:
            List of :class:`Signal` objects (may be empty).
        """
        signals: list[Signal] = []
        now = time.time()

        # ---- 0a. Regime gate (symbol-agnostic) ---------------------------
        # Use the first symbol's buffer for regime detection.
        primary_buf = self._quote_buffers.get(self.symbols[0]) if self.symbols else None
        if primary_buf is not None and len(primary_buf) >= 50:
            regime, regime_conf = detect_regime(primary_buf)
            self._last_regime = regime
            self._last_regime_confidence = regime_conf
            if regime == MarketRegime.CHOPPY:
                return []

        # ---- 0b. Volatility gate (symbol-agnostic) -----------------------
        if primary_buf is not None and len(primary_buf) >= 15:
            atr = compute_atr(primary_buf, period=14)
            if atr > 0:
                self._historical_atr.append(atr)
                vol_ratio = compute_volatility_ratio(atr, self._historical_atr)
                if account is not None:
                    safe, reason = is_safe_to_trade(vol_ratio, atr, account)
                    if not safe:
                        return []

        # ---- 0c. Session parameters --------------------------------------
        session = get_current_session()
        session_params = get_session_params(session)

        # PRE_CLOSE → no trading.
        if session == TradingSession.PRE_CLOSE:
            return []

        risk_mult = session_params["risk_multiplier"]
        session_min_confidence = session_params["min_confidence"]
        prefer_regime = session_params.get("prefer_regime")

        # ---- 0d. Adaptive parameter tuning (Phase 2.2) --------------------
        adapted_std_dev = self.config.mean_reversion_std_dev
        adapted_min_confidence = self.config.min_confidence_threshold
        adapted_risk_pct = self.config.risk_per_trade_pct

        if self.adaptive_state.winning_trades >= 10:
            defaults = {
                "mean_reversion_std_dev": self.config.mean_reversion_std_dev,
                "min_confidence_threshold": self.config.min_confidence_threshold,
                "risk_per_trade_pct": self.config.risk_per_trade_pct,
            }
            adapted = get_adapted_params(self.adaptive_state, defaults)
            adapted_std_dev = float(adapted.get("mean_reversion_std_dev", defaults["mean_reversion_std_dev"]))
            adapted_min_confidence = float(adapted.get("min_confidence_threshold", defaults["min_confidence_threshold"]))
            adapted_risk_pct = float(adapted.get("risk_per_trade_pct", defaults["risk_per_trade_pct"]))

        # ---- Per-symbol evaluation ---------------------------------------
        for symbol in self.symbols:
            buf = self._quote_buffers.get(symbol)
            if buf is None or len(buf) < 50:
                continue

            # Convert to list for indicator calls.
            quotes_list = list(buf)
            latest = quotes_list[-1]

            # ---- Spread gate ----------------------------------------------
            if not is_spread_eligible(
                latest.bid, latest.ask, self.config.min_spread_bps
            ):
                continue

            # ---- Cooldown -------------------------------------------------
            last_time = self._last_signal_time.get(symbol, 0.0)
            if now - last_time < self.config.cooldown_seconds:
                continue

            # ---- Indicators -----------------------------------------------
            vwap = compute_vwap(quotes_list)
            std_dev = compute_std_dev(quotes_list, vwap)
            micro_price = compute_micro_price(latest.bid, latest.ask)
            vol_spike = detect_volume_spike(
                quotes_list, self.config.volume_spike_multiplier
            )
            consecutive_up = detect_consecutive_direction(
                quotes_list, self.config.breakout_consecutive_ticks
            )
            consecutive_dir = get_consecutive_direction_sign(
                quotes_list, self.config.breakout_consecutive_ticks
            )

            if vwap is None or std_dev is None or std_dev == 0:
                continue

            price = latest.last
            deviation = (price - vwap) / std_dev

            # ---- Mean-reversion signal ------------------------------------
            mr_direction: str = "flat"
            mr_confidence: float = 0.0
            threshold = adapted_std_dev

            if deviation > threshold:
                # Price is stretched above VWAP → bet on reversal (short).
                mr_direction = "short"
                mr_confidence = min(1.0, (deviation - threshold) / threshold + 0.5)
            elif deviation < -threshold:
                # Price is stretched below VWAP → bet on reversal (long).
                mr_direction = "long"
                mr_confidence = min(1.0, (abs(deviation) - threshold) / threshold + 0.5)

            # ---- Breakout signal ------------------------------------------
            bo_direction: str = "flat"
            bo_confidence: float = 0.0

            if consecutive_up and vol_spike and consecutive_dir != 0:
                bo_direction = "long" if consecutive_dir > 0 else "short"
                # Breakout confidence: 0.6 base + 0.2 for volume spike
                # + 0.2 for longer runs beyond the minimum.
                extra_ticks = min(
                    len(quotes_list) - self.config.breakout_consecutive_ticks - 1,
                    10,
                )
                bo_confidence = min(1.0, 0.6 + 0.2 + extra_ticks * 0.02)

            # ---- Signal fusion --------------------------------------------
            final_direction: str = "flat"
            final_confidence: float = 0.0
            rationale_parts: list[str] = []

            if mr_direction != "flat" and bo_direction != "flat":
                if mr_direction == bo_direction:
                    # Both agree → high confidence.
                    final_direction = mr_direction
                    final_confidence = min(1.0, 0.6 + mr_confidence * 0.2 + bo_confidence * 0.2)
                    rationale_parts.append(
                        f"Mean-reversion ({mr_direction}, {deviation:+.2f}σ) "
                        f"+ breakout ({bo_direction}, {consecutive_dir:+d}) converge"
                    )
                else:
                    # Conflicting — trust the signal with higher individual confidence.
                    if mr_confidence >= bo_confidence:
                        final_direction = mr_direction
                        final_confidence = mr_confidence * 0.8  # penalty for conflict
                        rationale_parts.append(
                            f"Mean-reversion ({mr_direction}, {deviation:+.2f}σ) "
                            f"overrides breakout ({bo_direction})"
                        )
                    else:
                        final_direction = bo_direction
                        final_confidence = bo_confidence * 0.8
                        rationale_parts.append(
                            f"Breakout ({bo_direction}, {consecutive_dir:+d}) "
                            f"overrides mean-reversion ({mr_direction})"
                        )
            elif mr_direction != "flat":
                final_direction = mr_direction
                final_confidence = mr_confidence
                rationale_parts.append(
                    f"Mean-reversion: price {deviation:+.2f}σ from VWAP={vwap:.2f}"
                )
            elif bo_direction != "flat":
                final_direction = bo_direction
                final_confidence = bo_confidence
                rationale_parts.append(
                    f"Breakout: {consecutive_dir:+d} consecutive ticks + volume spike"
                )

            # ---- Adaptive confidence adjustment (Phase 2) -----------------
            # Regime preference boost/penalty.
            if prefer_regime is not None and self._last_regime is not None:
                regime_value = self._last_regime.value
                if regime_value == prefer_regime:
                    # Boost confidence when regime matches session preference.
                    final_confidence = min(1.0, final_confidence * 1.15)
                    rationale_parts.append(
                        f"Regime boost: {regime_value} matches "
                        f"session preference ({prefer_regime})"
                    )
                else:
                    # Penalize when regime conflicts with session preference.
                    final_confidence = max(0.0, final_confidence * 0.75)
                    rationale_parts.append(
                        f"Regime penalty: {regime_value} conflicts with "
                        f"session preference ({prefer_regime})"
                    )

            # ---- Session confidence threshold (overrides config) ----------
            effective_min_confidence = max(
                adapted_min_confidence,
                session_min_confidence,
            )

            if final_confidence < effective_min_confidence:
                continue

            # ---- Construct signal -----------------------------------------
            entry_price = micro_price  # use mid-price as entry reference
            stop_offset = abs(deviation * std_dev) * 0.5  # half the deviation as stop
            if stop_offset <= 0:
                stop_offset = std_dev  # fallback: 1 std-dev stop

            # Apply session risk multiplier to stop distance.
            stop_offset *= risk_mult

            if final_direction == "long":
                stop_price = price - stop_offset
                target_price = price + stop_offset * 2.0  # 2:1 reward-to-risk
            elif final_direction == "short":
                stop_price = price + stop_offset
                target_price = price - stop_offset * 2.0
            else:
                stop_price = price
                target_price = price

            signals.append(
                Signal(
                    symbol=symbol,
                    direction=final_direction,  # type: ignore[arg-type]
                    confidence=final_confidence,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    rationale=" | ".join(rationale_parts),
                    timestamp=now,
                )
            )

            # Update cooldown timer.
            self._last_signal_time[symbol] = now

        return signals

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_current_signal(self, symbol: str) -> Signal | None:
        """Return the most recent signal for *symbol*, or ``None``.

        Note that this only works for signals generated in the current
        session — the engine does not persist signal history across
        restarts.

        This is a convenience method for the orchestrator to check the
        last-known state of a symbol without re-evaluating.
        """
        # We don't store signal history; return None for now.
        # The orchestrator should call generate_signals() for fresh signals.
        return None

    @property
    def last_regime(self) -> MarketRegime | None:
        """Cached regime from the most recent ``generate_signals()`` call."""
        return self._last_regime

    @property
    def last_regime_confidence(self) -> float:
        """Cached regime confidence from the most recent call."""
        return self._last_regime_confidence

    # ------------------------------------------------------------------
    # Position sizing (delegated)
    # ------------------------------------------------------------------

    def calculate_size(
        self,
        signal: Signal,
        account: "Account",  # type: ignore[valid-type]  # noqa: F821
        risk_state: "RiskState",  # type: ignore[valid-type]  # noqa: F821
        positions: list,  # list[Position]
    ) -> int:
        """Delegate to :func:`calculate_position_size`.

        Args:
            signal: The trade signal.
            account: Current account snapshot.
            risk_state: Live risk state.
            positions: All open positions.

        Returns:
            Number of contracts to trade (0 = skip).
        """
        # Import inline to avoid circular imports at module level.
        from src.risk.limits import InstrumentLimit  # noqa: F811

        # Build instrument limits from config defaults.
        instrument_limits: dict[str, InstrumentLimit] = {}
        for sym in self.symbols:
            instrument_limits[sym] = InstrumentLimit(
                max_contracts=40 if sym.startswith("M") else 4,
                tick_value=0.50,
            )

        return calculate_position_size(
            signal=signal,
            account=account,
            risk_state=risk_state,
            current_positions=positions,
            config=self.config,
            instrument_limits=instrument_limits,
            signal_confidence=signal.confidence,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def record_trade_result(self, result: dict) -> None:
        """Feed a completed trade result into adaptive tuning.

        The orchestrator should call this after every round-trip trade so
        the adaptive state can learn which parameters produce wins.

        Args:
            result: Dict with keys matching :meth:`AdaptiveState.update`.
        """
        self.adaptive_state.update(result)

    @property
    def last_adapted_params(self) -> dict | None:
        """Current adapted parameters, or ``None`` if not enough data.

        Returns a dict with keys ``mean_reversion_std_dev``,
        ``min_confidence_threshold``, ``risk_per_trade_pct`` — or
        ``None`` when fewer than 10 winning trades have been recorded.
        """
        if self.adaptive_state.winning_trades < 10:
            return None

        defaults = {
            "mean_reversion_std_dev": self.config.mean_reversion_std_dev,
            "min_confidence_threshold": self.config.min_confidence_threshold,
            "risk_per_trade_pct": self.config.risk_per_trade_pct,
        }
        return get_adapted_params(self.adaptive_state, defaults)

    def clear_buffers(self) -> None:
        """Flush all quote buffers (e.g. after disconnect/reconnect).

        Cooldown timers are **not** reset — a reconnect should not allow
        immediate re-entry on stale signals.
        """
        for sym in self.symbols:
            self._quote_buffers[sym] = deque(
                maxlen=self.config.quote_window_size
            )
