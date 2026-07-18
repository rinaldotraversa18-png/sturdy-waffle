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

The engine also enforces a per-symbol cooldown to avoid overtrading
and discards signals whose spread exceeds the configured limit.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from src.client.models import Quote
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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """Generates trade signals from streaming quote data.

    Usage sketch::

        engine = StrategyEngine(StrategyConfig())
        engine.ingest_quote(quote)               # push every tick
        signals = await engine.generate_signals()  # evaluate all symbols
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

    async def generate_signals(self) -> list[Signal]:
        """Evaluate all tracked symbols and return actionable signals.

        Each symbol is evaluated independently:

        1. Skip if the buffer has fewer than 50 quotes.
        2. Skip if the spread is too wide (``min_spread_bps`` config).
        3. Compute VWAP, std-dev, micro-price, volume-spike flag.
        4. Check mean-reversion: price N std from VWAP → reversal.
        5. Check breakout: consecutive same-direction ticks + volume
           spike → momentum.
        6. Combine confidences (higher when both agree).
        7. Enforce per-symbol cooldown.

        Returns:
            List of :class:`Signal` objects (may be empty).
        """
        signals: list[Signal] = []
        now = time.time()

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
            threshold = self.config.mean_reversion_std_dev

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

            # ---- Threshold gate -------------------------------------------
            if final_confidence < self.config.min_confidence_threshold:
                continue

            # ---- Construct signal -----------------------------------------
            entry_price = micro_price  # use mid-price as entry reference
            stop_offset = abs(deviation * std_dev) * 0.5  # half the deviation as stop
            if stop_offset <= 0:
                stop_offset = std_dev  # fallback: 1 std-dev stop

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
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_buffers(self) -> None:
        """Flush all quote buffers (e.g. after disconnect/reconnect).

        Cooldown timers are **not** reset — a reconnect should not allow
        immediate re-entry on stale signals.
        """
        for sym in self.symbols:
            self._quote_buffers[sym] = deque(
                maxlen=self.config.quote_window_size
            )
