"""
Bot orchestrator — main loop, event routing, and lifecycle management.

The :class:`BotOrchestrator` is the central nervous system of the
trading bot.  It owns the Tradovate client, RiskEngine, StrategyEngine,
and StateManager, and wires them together via WebSocket callbacks and a
tick-driven main loop.

Usage::

    config = BotConfig(...)
    orchestrator = BotOrchestrator(config)
    result = await orchestrator.run()
    if result.status == BotStatus.PASSED:
        print("Evaluation passed!")
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import date
from enum import Enum
from typing import Optional

import structlog
from pydantic import BaseModel

from src.client.models import Account, BracketOrderRequest, Order, OrderRequest, Position, Quote
from src.client.tradovate_client import TradovateClient
from src.config import BotConfig, TradovateConfig
from src.risk.engine import OrderDecision, RiskEngine
from src.risk.limits import RiskConfig
from src.risk.state import RiskState, StateManager
from src.strategy.engine import StrategyEngine
from src.strategy.sizing import StrategyConfig
from src.strategy.trailing import (
    TrailingConfig,
    compute_trail_stop,
    should_activate_trail,
    should_update_trail,
)
from src.utils.clock import is_market_open, seconds_until_maintenance

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public enums / models
# ---------------------------------------------------------------------------


class BotStatus(str, Enum):
    """Terminal (and near-terminal) status codes for a bot run."""

    CONTINUE = "continue"
    PASSED = "passed"
    FAILED = "failed"
    LOCKED = "locked"


class BotResult(BaseModel):
    """Immutable summary produced at the end of every orchestrator run.

    Attributes:
        status: How the run ended.
        total_pnl: Cumulative realised P&L across all sessions.
        peak_equity: Highest ``net_liq`` watermark.
        final_equity: ``net_liq`` at shutdown.
        trades: Total number of round-trip trades (fills).
        winning_trades: Trades with positive realised P&L.
        losing_trades: Trades with negative realised P&L.
        reason: Human-readable explanation (empty on ``PASSED``).
    """

    status: BotStatus
    total_pnl: float
    peak_equity: float
    final_equity: float
    trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BotOrchestrator:
    """Central coordinator for the Tradeify evaluation trading bot.

    Responsibilities
    ----------------
    * Initialise all subsystems (client, risk, strategy, state).
    * Connect to Tradovate (REST auth + WebSocket + quote subscription).
    * Route WebSocket events → risk engine & strategy engine.
    * Run the main loop: evaluate stops → generate signals → risk-check →
      execute.
    * Graceful shutdown on stop / signal / error.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, config: BotConfig) -> None:
        self.config: BotConfig = config
        self.client: Optional[TradovateClient] = None
        self.risk_engine: Optional[RiskEngine] = None
        self.strategy: Optional[StrategyEngine] = None
        self.state_manager: Optional[StateManager] = None
        self._running: bool = False
        self._account_id: Optional[int] = None
        self._latest_account: Optional[Account] = None
        self._latest_positions: list[Position] = []
        self._trades: int = 0
        self._winning: int = 0
        self._losing: int = 0
        self._result_reason: str = ""
        self._lock_fd: Optional[int] = None

        # -- Phase 2: Adaptive feature state -----------------------------------
        self._iteration_count: int = 0
        # Track bracket orders: entry_order_id → {entry_price, symbol, direction, ...}
        self._active_brackets: dict[int, dict] = {}
        # Track active trailing stops: symbol → {direction, entry_price, stop_order_id, current_stop, target_price}
        self._trailing_stops: dict[str, dict] = {}
        # Track last-known position P&L baseline for win/loss detection.
        self._last_position_pnl: dict[str, float] = {}

    # ------------------------------------------------------------------
    # run() — entry point
    # ------------------------------------------------------------------

    async def run(self) -> BotResult:
        """Execute one full bot session from startup to shutdown.

        Returns:
            A :class:`BotResult` summarising what happened.
        """
        try:
            self._acquire_lock()
            await self._initialize()

            self._running = True
            logger.info("orchestrator.running", account_id=self._account_id)

            await self._main_loop()

        except asyncio.CancelledError:
            logger.warning("orchestrator.cancelled")
            self._result_reason = "cancelled"
        except Exception as exc:
            logger.exception("orchestrator.fatal_error", error=str(exc))
            self._result_reason = f"fatal error: {exc}"
        finally:
            await self.shutdown()

        return self._build_result()

    # ------------------------------------------------------------------
    # _initialize — subsystem wiring
    # ------------------------------------------------------------------

    async def _initialize(self) -> None:
        """Create and wire all subsystems, connect to Tradovate.

        Steps:

        1. Set up the Tradovate client (demo or live).
        2. Create RiskEngine, StrategyEngine, StateManager.
        3. Load persisted state; reset session if the date changed.
        4. Connect to Tradovate (REST auth + WS).
        5. Fetch initial account / position snapshots.
        6. Subscribe to real-time quotes.
        7. Register WebSocket → engine callbacks.
        """
        # ---- Tradovate client -------------------------------------------------
        tv_config = TradovateConfig(environment=self.config.environment)  # type: ignore[call-arg]
        self.client = TradovateClient(tv_config)

        # ---- Risk engine ------------------------------------------------------
        risk_config = RiskConfig()
        self.risk_engine = RiskEngine(risk_config)

        # ---- Strategy engine --------------------------------------------------
        strat_config = StrategyConfig(symbols=self.config.symbols)
        self.strategy = StrategyEngine(strat_config)

        # ---- Phase 2: Wire adaptive configs into strategy engine --------------
        if self.config.adaptive_enabled:
            from src.strategy.trailing import TrailingConfig as StratTrailingConfig
            from src.strategy.adaptive_tuning import AdaptiveState

            # Override trailing config with values from bot config.
            self.strategy.trailing_config = StratTrailingConfig(
                activation_pct=self.config.trailing.activation_pct,
                trail_distance_ticks=self.config.trailing.trail_distance_ticks,
                step_ticks=self.config.trailing.step_ticks,
            )
            # AdaptiveState uses module-level constants; update them from config.
            self.strategy.adaptive_state._alpha = self.config.adaptive.ema_alpha
            logger.info(
                "orchestrator.adaptive_wired",
                trailing_activation_pct=self.config.trailing.activation_pct,
                adaptive_alpha=self.config.adaptive.ema_alpha,
                min_wins_to_adapt=self.config.adaptive.min_wins_to_adapt,
            )

        # ---- State manager ----------------------------------------------------
        self.state_manager = StateManager(file_path=self.config.state_path)

        # ---- Load persisted state ---------------------------------------------
        saved_state = self.state_manager.load()
        if saved_state is not None:
            self.risk_engine.load_state(saved_state)
            logger.info(
                "state.loaded",
                session_date=str(saved_state.session_date),
                total_realized_pnl=saved_state.total_realized_pnl,
                peak_equity=saved_state.peak_equity,
            )
        else:
            logger.info("state.fresh_start")

        # ---- Connect to Tradovate ---------------------------------------------
        assert self.client is not None
        await self.client.connect()

        # ---- Fetch initial account info ---------------------------------------
        # Discover account ID from list, or use a well-known ID.
        self._account_id = await self._discover_account_id()
        if self._account_id is not None:
            acct = await self.client.get_account(self._account_id)
            self._latest_account = acct
            self.risk_engine.update_from_account(acct)
            logger.info(
                "account.initial",
                account_id=acct.id,
                net_liq=acct.net_liq,
                realized_pnl=acct.realized_pnl,
            )

        # ---- Fetch positions --------------------------------------------------
        self._latest_positions = await self.client.get_positions()
        logger.info("positions.initial", count=len(self._latest_positions))

        # ---- Subscribe to quotes ----------------------------------------------
        await self.client.subscribe_quotes(self.config.symbols)

        # ---- Register WebSocket callbacks -------------------------------------
        self.client.on_account_update(self._on_account_update)
        self.client.on_order_update(self._on_order_update)
        self.client.on_quote(self._on_quote)

        # Persist state after initialisation.
        self._persist_state()

        logger.info("orchestrator.initialized")

    # ------------------------------------------------------------------
    # _main_loop — tick-driven core
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        """Run the main trading loop until a stop condition is met.

        Each iteration:

        1. Evaluate end-conditions (profit target / daily loss / drawdown /
           session timeout).
        2. If not ``CONTINUE`` → shutdown.
        3. Check market hours — sleep if closed or in maintenance.
        4. Generate signals from the strategy engine.
        5. For each actionable signal:
           a. Calculate position size.
           b. Build a bracket order request.
           c. Run through the risk engine.
           d. If approved → submit; if rejected → log.
        6. Sleep for ``loop_interval`` seconds.
        """
        assert self.risk_engine is not None
        assert self.strategy is not None
        assert self.client is not None

        while self._running:
            # ---- 1. Evaluate end conditions ----------------------------------
            status = await self._evaluate_end_conditions()
            if status != BotStatus.CONTINUE:
                self._result_reason = self.risk_engine.get_stop_reason() or "unknown"
                logger.info("orchestrator.stop_condition", status=status.value, reason=self._result_reason)
                self._running = False
                break

            # ---- 2. Market hours check ---------------------------------------
            if not is_market_open():
                sleep_time = min(60.0, seconds_until_maintenance())
                logger.debug("orchestrator.market_closed", sleep_s=sleep_time)
                await asyncio.sleep(sleep_time)
                continue

            # ---- 3. Trading gate ---------------------------------------------
            allowed, block_reason = self.risk_engine.is_trading_allowed()
            if not allowed:
                logger.debug("orchestrator.trading_blocked", reason=block_reason)
                await asyncio.sleep(self.config.loop_interval)
                continue

            # ---- 4. Generate signals -----------------------------------------
            try:
                signals = await self.strategy.generate_signals(
                    account=self._latest_account,
                )
            except Exception:
                logger.exception("orchestrator.signal_error")
                await asyncio.sleep(self.config.loop_interval)
                continue

            # ---- 5. Process each signal --------------------------------------
            for sig in signals:
                if sig.direction == "flat" or sig.confidence < self.strategy.config.min_confidence_threshold:
                    continue

                # 5a. Calculate size
                positions = self._latest_positions
                risk_snapshot = self.risk_engine.snapshot()
                if self._latest_account is None:
                    continue

                contracts = self.strategy.calculate_size(
                    sig, self._latest_account, risk_snapshot, positions
                )
                if contracts <= 0:
                    logger.debug(
                        "orchestrator.size_zero",
                        symbol=sig.symbol,
                        direction=sig.direction,
                    )
                    continue

                # 5b. Build bracket order request
                action: str = "Buy" if sig.direction == "long" else "Sell"
                bracket_req = BracketOrderRequest(
                    account_spec=self.config.environment,
                    account_id=self._account_id or 0,
                    action=action,  # type: ignore[arg-type]
                    symbol=sig.symbol,
                    order_qty=contracts,
                    order_type="Market",
                    is_automated=True,
                    bracket={
                        "profit_target": abs(sig.target_price - (sig.entry_price or sig.stop_price)),
                        "stop_loss": abs(sig.entry_price - sig.stop_price) if sig.entry_price else abs(sig.stop_price - sig.target_price) * 0.5,
                    },
                )

                # 5c. Risk check
                decision = self.risk_engine.check_bracket(bracket_req, positions)
                if not decision.approved:
                    logger.warning(
                        "orchestrator.signal_rejected",
                        symbol=sig.symbol,
                        direction=sig.direction,
                        reason=decision.reason,
                    )
                    continue

                # 5d. Execute
                try:
                    response = await self.client.place_bracket(bracket_req)
                    logger.info(
                        "orchestrator.order_placed",
                        symbol=sig.symbol,
                        direction=sig.direction,
                        contracts=contracts,
                        entry_order_id=response.entry_order_id,
                        status=response.status,
                    )
                    # Track bracket for trade-result recording.
                    self._active_brackets[response.entry_order_id] = {
                        "symbol": sig.symbol,
                        "direction": sig.direction,
                        "entry_price": sig.entry_price,
                        "target_price": sig.target_price,
                        "stop_price": sig.stop_price,
                        "confidence": sig.confidence,
                        "contracts": contracts,
                        "stop_loss_order_id": response.stop_loss_order_id,
                        "profit_target_order_id": response.profit_target_order_id,
                    }
                except Exception:
                    logger.exception(
                        "orchestrator.order_failed",
                        symbol=sig.symbol,
                    )

            # ---- 6. Adaptive logging (every 10 iterations) -------------------
            self._iteration_count += 1
            if self._iteration_count % 10 == 0 and self.config.adaptive_enabled:
                adapted = self.strategy.last_adapted_params
                if adapted is not None:
                    logger.info(
                        "orchestrator.adaptive_params",
                        iteration=self._iteration_count,
                        winning_trades=self.strategy.adaptive_state.winning_trades,
                        mean_reversion_std_dev=adapted.get("mean_reversion_std_dev"),
                        min_confidence_threshold=adapted.get("min_confidence_threshold"),
                        risk_per_trade_pct=adapted.get("risk_per_trade_pct"),
                    )

            # ---- 7. Sleep ----------------------------------------------------
            await asyncio.sleep(self.config.loop_interval)

    # ------------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------------

    async def _on_account_update(self, account: Account) -> None:
        """Route account updates → risk engine, persist state, evaluate stops."""
        if self.risk_engine is None:
            return

        self._latest_account = account
        self.risk_engine.update_from_account(account)
        self._persist_state()

        logger.debug(
            "callback.account",
            net_liq=account.net_liq,
            realized_pnl=account.realized_pnl,
        )

    async def _on_order_update(self, order: Order) -> None:
        """Route order updates — track fills for trade stats and adaptive tuning."""
        if order.order_status == "Filled" and order.filled_qty > 0:
            self._trades += 1
            logger.info(
                "callback.fill",
                order_id=order.id,
                symbol=order.symbol,
                action=order.action,
                filled_qty=order.filled_qty,
                avg_fill_price=order.avg_fill_price,
            )

            # -- Phase 2: Trade-close detection for adaptive tuning -------------
            if self.config.adaptive_enabled and self.strategy is not None:
                # Check if this fill closes a bracket leg (stop-loss or TP).
                for entry_id, bracket_info in list(self._active_brackets.items()):
                    if order.id in (
                        bracket_info.get("stop_loss_order_id"),
                        bracket_info.get("profit_target_order_id"),
                    ):
                        # Trade closed — compute result and feed adaptive.
                        entry_price = bracket_info.get("entry_price", 0.0)
                        fill_price = order.avg_fill_price or 0.0
                        direction = bracket_info.get("direction", "flat")
                        pnl: float = 0.0
                        if direction == "long":
                            pnl = (fill_price - entry_price) * bracket_info.get("contracts", 0)
                        elif direction == "short":
                            pnl = (entry_price - fill_price) * bracket_info.get("contracts", 0)

                        was_winner = pnl > 0
                        if was_winner:
                            self._winning += 1
                        else:
                            self._losing += 1

                        self.strategy.record_trade_result({
                            "was_winner": was_winner,
                            "signal_std_dev": 2.0,  # default; can be enriched later
                            "signal_confidence": bracket_info.get("confidence", 0.6),
                            "risk_pct": self.strategy.config.risk_per_trade_pct,
                            "pnl": pnl,
                        })

                        logger.info(
                            "orchestrator.trade_closed",
                            symbol=bracket_info.get("symbol"),
                            direction=direction,
                            pnl=pnl,
                            was_winner=was_winner,
                        )

                        # Clean up bracket tracking and trailing stop.
                        self._active_brackets.pop(entry_id, None)
                        symbol = bracket_info.get("symbol", "")
                        self._trailing_stops.pop(symbol, None)
                        break

            # -- Phase 2: Entry fill → start trailing stop tracking -------------
            if self.config.adaptive_enabled and self.config.trailing.enabled:
                for entry_id, bracket_info in list(self._active_brackets.items()):
                    if order.id == entry_id and order.avg_fill_price:
                        # Entry filled — start tracking for trailing stop.
                        symbol = bracket_info.get("symbol", "")
                        self._trailing_stops[symbol] = {
                            "direction": bracket_info.get("direction", "flat"),
                            "entry_price": order.avg_fill_price,
                            "stop_order_id": bracket_info.get("stop_loss_order_id"),
                            "current_stop": bracket_info.get("stop_price", 0.0),
                            "target_price": bracket_info.get("target_price", 0.0),
                        }
                        logger.info(
                            "orchestrator.trailing_armed",
                            symbol=symbol,
                            entry_price=order.avg_fill_price,
                            initial_stop=bracket_info.get("stop_price"),
                        )
                        break

        # If this fill carries P&L info (some brokers embed it), feed
        # it to the risk engine.  For Tradovate, fills don't directly
        # carry P&L; that arrives via account updates.  We keep the
        # hook here for future use.
        if self.risk_engine is not None:
            # Order status transitions can trigger a re-fetch of account
            # data which is handled by _on_account_update.
            pass

    async def _on_quote(self, quote: Quote) -> None:
        """Route real-time quotes → strategy engine and trailing stop logic."""
        if self.strategy is None:
            return
        self.strategy.ingest_quote(quote)

        # -- Phase 2: Trailing stop check ---------------------------------------
        if not self.config.adaptive_enabled or not self.config.trailing.enabled:
            return
        if self.client is None:
            return

        trail_info = self._trailing_stops.get(quote.symbol)
        if trail_info is None:
            return

        direction = trail_info["direction"]
        entry_price = trail_info["entry_price"]
        target_price = trail_info["target_price"]
        current_stop = trail_info["current_stop"]
        stop_order_id = trail_info.get("stop_order_id")

        if direction not in ("long", "short") or stop_order_id is None:
            return

        # Use bid for longs, ask for shorts as the reference price.
        current_price = quote.bid if direction == "long" else quote.ask

        # Build a TrailingConfig from the bot config.
        trail_cfg = TrailingConfig(
            activation_pct=self.config.trailing.activation_pct,
            trail_distance_ticks=self.config.trailing.trail_distance_ticks,
            step_ticks=self.config.trailing.step_ticks,
        )
        tick_size = 0.50  # default for MBT; could be from instrument config

        if not should_activate_trail(entry_price, current_price, target_price, trail_cfg):
            return

        new_stop = compute_trail_stop(current_price, direction, trail_cfg.trail_distance_ticks, tick_size)

        if not should_update_trail(current_stop, new_stop, direction, trail_cfg.step_ticks, tick_size):
            return

        # Modify the stop order.
        try:
            await self.client.modify_order(stop_order_id, {"stopPrice": new_stop})
            trail_info["current_stop"] = new_stop
            logger.debug(
                "orchestrator.trail_updated",
                symbol=quote.symbol,
                direction=direction,
                old_stop=current_stop,
                new_stop=new_stop,
            )
        except Exception:
            logger.exception(
                "orchestrator.trail_failed",
                symbol=quote.symbol,
                stop_order_id=stop_order_id,
            )

    # ------------------------------------------------------------------
    # End-condition evaluation
    # ------------------------------------------------------------------

    async def _evaluate_end_conditions(self) -> BotStatus:
        """Check all stop conditions and return the appropriate status.

        Order of precedence:
        1. Profit target reached → PASSED.
        2. EOD drawdown breached → FAILED.
        3. Daily loss limit breached → LOCKED.
        4. Market session timeout → LOCKED.
        5. Otherwise → CONTINUE.
        """
        if self.risk_engine is None:
            return BotStatus.CONTINUE

        state = self.risk_engine.snapshot()

        # 1. Profit target
        if state.profit_target_reached:
            return BotStatus.PASSED

        # 2. EOD drawdown — permanent failure
        if state.drawdown_breached:
            return BotStatus.FAILED

        # 3. Daily loss limit — locked for the day
        if state.daily_loss_breached:
            return BotStatus.LOCKED

        # 4. Session timeout — market closed for the weekend
        if not is_market_open():
            # Check if it's a weekend shutdown (not just maintenance).
            from src.utils.clock import get_current_ct_time

            now = get_current_ct_time()
            wd = now.weekday()
            if wd >= 4 and now.hour >= 16:  # Friday 16:00+ or Saturday
                return BotStatus.LOCKED

        return BotStatus.CONTINUE

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully tear down all subsystems.

        Idempotent — safe to call multiple times.
        """
        self._running = False

        # Persist final state.
        self._persist_state()

        # Disconnect Tradovate client.
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception as exc:
                logger.warning("orchestrator.disconnect_error", error=str(exc))
            self.client = None

        # Release PID lock.
        self._release_lock()

        logger.info("orchestrator.shutdown_complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Save the current risk state to disk."""
        if self.risk_engine is None or self.state_manager is None:
            return
        try:
            self.state_manager.save(self.risk_engine.snapshot())
        except Exception:
            logger.exception("state.save_failed")

    def _build_result(self) -> BotResult:
        """Assemble a :class:`BotResult` from the current engine state."""
        if self.risk_engine is None:
            return BotResult(
                status=BotStatus.FAILED,
                total_pnl=0.0,
                peak_equity=0.0,
                final_equity=0.0,
                reason="never initialised",
            )

        state = self.risk_engine.snapshot()
        final_equity = self._latest_account.net_liq if self._latest_account else state.peak_equity

        if state.profit_target_reached:
            status = BotStatus.PASSED
        elif state.drawdown_breached:
            status = BotStatus.FAILED
        elif state.daily_loss_breached:
            status = BotStatus.LOCKED
        else:
            status = BotStatus.LOCKED  # Shutdown without explicit stop = locked

        return BotResult(
            status=status,
            total_pnl=state.total_realized_pnl,
            peak_equity=state.peak_equity,
            final_equity=final_equity,
            trades=self._trades,
            winning_trades=self._winning,
            losing_trades=self._losing,
            reason=self._result_reason or self.risk_engine.get_stop_reason() or "shutdown",
        )

    async def _discover_account_id(self) -> Optional[int]:
        """Discover the evaluation account ID from Tradovate.

        Since the bot is designed for a single evaluation account, we
        search contracts for the symbols we trade and use the account
        associated with the first match.  Falls back to a REST call
        that lists accounts.
        """
        if self.client is None:
            return None

        # Try fetching via the accounts endpoint — Tradovate has a
        # GET /account/list endpoint.
        try:
            # Use low-level _request since get_account needs an ID first.
            data = await self.client._request("GET", "account/list")
            items = data.get("d", data) if isinstance(data, dict) else data
            if isinstance(items, list) and len(items) > 0:
                acct_data = items[0]
                if isinstance(acct_data, dict):
                    return int(acct_data.get("id", 0))
        except Exception as exc:
            logger.warning("account.discovery_failed", error=str(exc))

        return None

    # ------------------------------------------------------------------
    # PID lock
    # ------------------------------------------------------------------

    def _acquire_lock(self) -> None:
        """Create a PID lock file to prevent concurrent bot instances."""
        lock_path = self.config.lock_path
        try:
            import fcntl

            self._lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.error("orchestrator.lock_conflict", lock_path=lock_path)
                raise RuntimeError(
                    f"Another bot instance is already running (lock: {lock_path}). "
                    "Remove the lock file if you are certain no other instance is active."
                ) from None
            # Write PID.
            os.write(self._lock_fd, f"{os.getpid()}\n".encode())
        except ImportError:
            # fcntl not available (e.g. Windows) — fall back to simple PID file.
            if os.path.exists(lock_path):
                try:
                    with open(lock_path, "r") as f:
                        old_pid = int(f.read().strip())
                    # Check if the process is still alive.
                    try:
                        os.kill(old_pid, 0)
                        raise RuntimeError(
                            f"Another bot instance is running with PID {old_pid} "
                            f"(lock: {lock_path})"
                        )
                    except OSError:
                        # Stale lock — remove it.
                        os.unlink(lock_path)
                except (ValueError, FileNotFoundError):
                    pass
            self._lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            os.write(self._lock_fd, f"{os.getpid()}\n".encode())

    def _release_lock(self) -> None:
        """Remove the PID lock file."""
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        try:
            os.unlink(self.config.lock_path)
        except FileNotFoundError:
            pass
