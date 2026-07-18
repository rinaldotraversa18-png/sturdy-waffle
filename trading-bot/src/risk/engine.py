"""
Risk engine — the safety gatekeeper for Tradeify funded evaluations.

Every order, account update, and fill passes through the risk engine
before the bot acts.  It enforces the three critical rules:

1. **Daily loss limit** — realised P&L only, resets each session.
2. **EOD trailing drawdown** — trails peak equity upward only.
3. **Contract limits** — per-instrument position caps.

The engine also detects stop conditions (profit target hit, daily loss
breached, drawdown breached) so the orchestrator can shut down trading.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from pydantic import BaseModel

from src.client.models import Account, BracketOrderRequest, OrderRequest, Position
from src.risk.limits import RiskConfig
from src.risk.state import RiskState

# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class OrderDecision(BaseModel):
    """Result of a pre-trade risk check.

    Attributes:
        approved: ``True`` when the order is safe to submit.
        reason: Human-readable explanation, set on rejection (``None``
            when approved).
    """

    approved: bool
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RiskEngine:
    """Central risk gate for the trading bot.

    Usage sketch::

        engine = RiskEngine(RiskConfig())
        engine.load_state(saved_state)         # on startup
        engine.update_from_account(acct)        # on every WS push
        engine.update_from_fill(fill_pnl)       # on every fill
        decision = engine.check_order(order, positions)
        if decision.approved:
            await client.place_order(order)
    """

    # ---- life-cycle -------------------------------------------------------

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        today = date.today()
        self._state = RiskState(
            session_date=today,
            starting_equity=50000.0,
            peak_equity=50000.0,
        )

    # ---- state management -------------------------------------------------

    def load_state(self, state: RiskState) -> None:
        """Restore a previously-persisted :class:`RiskState`.

        Called once at bot startup.  If the loaded state belongs to a
        different trading day, the session-level fields are reset
        automatically.
        """
        today = date.today()
        if state.session_date != today:
            # New day — carry forward cumulative values but reset session.
            self._state = RiskState(
                session_date=today,
                session_realized_pnl=0.0,
                peak_equity=state.peak_equity,
                starting_equity=state.starting_equity,
                total_realized_pnl=state.total_realized_pnl,
                profit_target_reached=state.profit_target_reached,
                drawdown_breached=state.drawdown_breached,
                daily_loss_breached=False,
            )
        else:
            self._state = copy.deepcopy(state)

    def snapshot(self) -> RiskState:
        """Return a deep copy of the current risk state."""
        return copy.deepcopy(self._state)

    def reset_session(self) -> None:
        """Start a fresh trading day — resets daily P&L & loss flag.

        Does **not** reset ``peak_equity``, ``starting_equity``,
        ``total_realized_pnl``, or the permanent stop conditions
        (``drawdown_breached``, ``profit_target_reached``).
        """
        today = date.today()
        self._state.session_date = today
        self._state.session_realized_pnl = 0.0
        self._state.daily_loss_breached = False

    # ---- real-time updates ------------------------------------------------

    def update_from_account(self, account: Account) -> None:
        """Ingest an account snapshot from the Tradovate WebSocket.

        This is the **only** path that trails ``peak_equity`` and
        evaluates the drawdown stop condition because only
        ``account.net_liq`` reflects both realised *and* open P&L.

        Side-effects:
        * ``peak_equity`` trails up when ``net_liq`` exceeds it.
        * ``drawdown_breached`` flag is set when ``net_liq`` falls below
          ``peak_equity - max_eod_drawdown``.
        * ``profit_target_reached`` is set when ``total_realized_pnl``
          meets the target.
        """
        net_liq = account.net_liq

        # Trail peak equity upward only.
        if net_liq > self._state.peak_equity:
            self._state.peak_equity = net_liq

        # Drawdown check — uses net_liq (includes open P&L).
        drawdown_floor = self._state.peak_equity - self._config.max_eod_drawdown
        if net_liq <= drawdown_floor:
            self._state.drawdown_breached = True

        # Profit target check.
        if self._state.total_realized_pnl >= self._config.profit_target:
            self._state.profit_target_reached = True

    def update_from_fill(self, fill_pnl: float) -> None:
        """Add *fill_pnl* to both session and cumulative realised P&L.

        Called on every fill confirmation.  Fills are the **only** way
        ``session_realized_pnl`` changes — open/unrealised P&L is
        deliberately excluded from the daily-loss calculation.

        Evaluates the daily-loss stop condition after the increment.
        """
        self._state.session_realized_pnl += fill_pnl
        self._state.total_realized_pnl += fill_pnl

        # Daily loss check — realised P&L only.
        if self._state.session_realized_pnl <= -self._config.daily_loss_limit:
            self._state.daily_loss_breached = True

        # Profit target may now be reached.
        if self._state.total_realized_pnl >= self._config.profit_target:
            self._state.profit_target_reached = True

    # ---- pre-trade checks -------------------------------------------------

    def check_order(
        self,
        order: OrderRequest,
        positions: list[Position],
    ) -> OrderDecision:
        """Run all risk checks against a proposed single order.

        Checks (in order):
        1.  Contract limit — proposed net position must not exceed cap
            *unless* the order reduces the absolute position.
        2.  Daily loss limit — reject if already breached.
        3.  EOD drawdown — reject if already breached.
        4.  Profit target — reject if already reached.
        """
        # 1. Contract limit
        limit = self._config.get_limit(order.symbol)
        current_net = self._get_net_position(order.symbol, positions)
        proposed_net = self._compute_proposed_net(current_net, order)

        if abs(proposed_net) > limit and abs(proposed_net) >= abs(current_net):
            return OrderDecision(
                approved=False,
                reason=(
                    f"Contract limit exceeded for {order.symbol}: "
                    f"proposed {abs(proposed_net)} > limit {limit}"
                ),
            )

        # 2. Daily loss limit
        if self._state.daily_loss_breached:
            return OrderDecision(
                approved=False,
                reason="Daily loss limit breached",
            )

        # 3. EOD drawdown
        if self._state.drawdown_breached:
            return OrderDecision(
                approved=False,
                reason="EOD drawdown breached",
            )

        # 4. Profit target
        if self._state.profit_target_reached:
            return OrderDecision(
                approved=False,
                reason="Profit target already reached",
            )

        return OrderDecision(approved=True)

    def check_bracket(
        self,
        bracket: BracketOrderRequest,
        positions: list[Position],
    ) -> OrderDecision:
        """Run risk checks against a proposed bracket (OCO) order.

        Same checks as :meth:`check_order` **plus** a worst-case
        drawdown analysis: if the stop-loss leg is hit immediately, the
        resulting ``net_liq`` must not breach the drawdown floor.
        """
        # Run standard checks first.
        decision = self.check_order(bracket, positions)
        if not decision.approved:
            return decision

        # Validate TP/SL prices are reasonable.
        if bracket.bracket.stop_loss <= 0:
            return OrderDecision(
                approved=False,
                reason="Stop-loss must be positive",
            )

        # Worst-case drawdown: if the stop is hit at the worst price,
        # would we breach drawdown?  We estimate the loss as:
        #   loss = stop_loss_offset × tick_value × contracts
        # The actual tick_value per instrument would come from the
        # config, but here we use a conservative estimate based on
        # the stop-loss distance times order quantity.
        limit_cfg = self._config.instrument_limits.get(bracket.symbol)
        if limit_cfg is not None:
            tick_value = limit_cfg.tick_value
            worst_case_loss = bracket.bracket.stop_loss * tick_value * bracket.order_qty

            current_equity = self._state.peak_equity  # best recent estimate
            drawdown_floor = self._state.peak_equity - self._config.max_eod_drawdown
            projected_equity = current_equity - worst_case_loss

            if projected_equity <= drawdown_floor:
                return OrderDecision(
                    approved=False,
                    reason=(
                        "Worst-case stop-loss would breach drawdown: "
                        f"projected {projected_equity:.2f} <= floor {drawdown_floor:.2f}"
                    ),
                )

        return OrderDecision(approved=True)

    # ---- limit queries ----------------------------------------------------

    def is_trading_allowed(self) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for the current state.

        Trading is blocked when *any* stop condition has fired.
        """
        if self._state.profit_target_reached:
            return False, "Profit target reached"
        if self._state.daily_loss_breached:
            return False, "Daily loss limit breached"
        if self._state.drawdown_breached:
            return False, "EOD drawdown breached"
        return True, ""

    def remaining_daily_loss(self) -> float:
        """Remaining loss headroom for today.

        Returns ``$1,250`` when there are no losses yet.  Shrinks as
        ``session_realized_pnl`` goes negative.  Returns 0 when the
        daily loss limit has been hit.
        """
        if self._state.session_realized_pnl >= 0:
            return self._config.daily_loss_limit
        remaining = self._config.daily_loss_limit + self._state.session_realized_pnl
        return max(0.0, remaining)

    def remaining_drawdown(self, current_equity: float) -> float:
        """Drawdown headroom: how far *current_equity* is above the
        trailing drawdown floor.

        Returns
        -------
        float
            ``current_equity - (peak_equity - $2,000)``.  Zero or
            negative means the floor has been hit.
        """
        floor = self._state.peak_equity - self._config.max_eod_drawdown
        return current_equity - floor

    def get_stop_reason(self) -> Optional[str]:
        """Return the human-readable stop reason, or ``None``."""
        if self._state.daily_loss_breached:
            return "daily_loss"
        if self._state.drawdown_breached:
            return "drawdown"
        if self._state.profit_target_reached:
            return "profit_target"
        return None

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _get_net_position(symbol: str, positions: list[Position]) -> int:
        """Extract ``net_pos`` for *symbol* from the position list."""
        for pos in positions:
            if pos.symbol == symbol:
                return pos.net_pos
        return 0

    @staticmethod
    def _compute_proposed_net(current_net: int, order: OrderRequest) -> int:
        """Project the net position after *order* fills."""
        if order.action == "Buy":
            return current_net + order.order_qty
        else:  # Sell
            return current_net - order.order_qty
