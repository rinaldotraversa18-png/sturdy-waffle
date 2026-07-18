"""
Tests for Pydantic v2 models in trading_bot.src.client.models.

Covers:
  - Construction from dicts (mimicking JSON API responses)
  - Validation rejecting invalid/bad data
  - Default values
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.client.models import (
    Account,
    BracketConfig,
    BracketOrderRequest,
    BracketOrderResponse,
    Contract,
    Order,
    OrderRequest,
    OrderResponse,
    Position,
    Quote,
)


# ---------------------------------------------------------------------------
# OrderRequest
# ---------------------------------------------------------------------------

class TestOrderRequest:
    """Unit tests for OrderRequest."""

    def test_construct_from_dict_minimal(self) -> None:
        """Minimal valid dict should produce an OrderRequest with defaults."""
        data = {
            "account_spec": "demo",
            "account_id": 12345,
            "action": "Buy",
            "symbol": "MESM5",
            "order_qty": 1,
            "order_type": "Market",
        }
        req = OrderRequest(**data)
        assert req.account_spec == "demo"
        assert req.account_id == 12345
        assert req.action == "Buy"
        assert req.symbol == "MESM5"
        assert req.order_qty == 1
        assert req.order_type == "Market"
        assert req.price is None
        assert req.stop_price is None
        assert req.time_in_force == {"tif": "Day"}
        assert req.is_automated is True

    def test_construct_from_dict_full(self) -> None:
        """Full dict including optional fields."""
        data = {
            "account_spec": "live",
            "account_id": 999,
            "action": "Sell",
            "symbol": "MNQM5",
            "order_qty": 2,
            "order_type": "Limit",
            "price": 18450.25,
            "stop_price": None,
            "time_in_force": {"tif": "GTC"},
            "is_automated": False,
        }
        req = OrderRequest(**data)
        assert req.price == 18450.25
        assert req.time_in_force == {"tif": "GTC"}
        assert req.is_automated is False

    def test_order_qty_must_be_positive(self) -> None:
        """order_qty <= 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            OrderRequest(
                account_spec="demo",
                account_id=1,
                action="Buy",
                symbol="MESM5",
                order_qty=0,
                order_type="Market",
            )
        with pytest.raises(ValidationError):
            OrderRequest(
                account_spec="demo",
                account_id=1,
                action="Buy",
                symbol="MESM5",
                order_qty=-1,
                order_type="Market",
            )

    def test_invalid_action_rejected(self) -> None:
        """Invalid action literal must be rejected."""
        with pytest.raises(ValidationError):
            OrderRequest(
                account_spec="demo",
                account_id=1,
                action="Hold",  # not a valid literal
                symbol="MESM5",
                order_qty=1,
                order_type="Market",
            )

    def test_invalid_order_type_rejected(self) -> None:
        """Invalid order_type literal must be rejected."""
        with pytest.raises(ValidationError):
            OrderRequest(
                account_spec="demo",
                account_id=1,
                action="Buy",
                symbol="MESM5",
                order_qty=1,
                order_type="TrailingStop",  # not supported
            )

    def test_default_time_in_force(self) -> None:
        """When time_in_force is omitted, defaults to {"tif": "Day"}."""
        req = OrderRequest(
            account_spec="demo",
            account_id=1,
            action="Buy",
            symbol="MESM5",
            order_qty=1,
            order_type="Market",
        )
        assert req.time_in_force == {"tif": "Day"}

    def test_default_is_automated(self) -> None:
        """is_automated defaults to True."""
        req = OrderRequest(
            account_spec="demo",
            account_id=1,
            action="Buy",
            symbol="MESM5",
            order_qty=1,
            order_type="Market",
        )
        assert req.is_automated is True


# ---------------------------------------------------------------------------
# BracketConfig
# ---------------------------------------------------------------------------

class TestBracketConfig:
    """Unit tests for BracketConfig."""

    def test_construct_from_dict(self) -> None:
        cfg = BracketConfig(profit_target=50.0, stop_loss=25.0)
        assert cfg.profit_target == 50.0
        assert cfg.stop_loss == 25.0

    def test_negative_profit_target_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BracketConfig(profit_target=-1.0, stop_loss=25.0)

    def test_zero_profit_target_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BracketConfig(profit_target=0.0, stop_loss=25.0)

    def test_negative_stop_loss_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BracketConfig(profit_target=50.0, stop_loss=-1.0)

    def test_zero_stop_loss_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BracketConfig(profit_target=50.0, stop_loss=0.0)


# ---------------------------------------------------------------------------
# BracketOrderRequest
# ---------------------------------------------------------------------------

class TestBracketOrderRequest:
    """Unit tests for BracketOrderRequest (inherits from OrderRequest)."""

    def test_construct_from_dict(self) -> None:
        data = {
            "account_spec": "demo",
            "account_id": 12345,
            "action": "Sell",
            "symbol": "MESM5",
            "order_qty": 1,
            "order_type": "Market",
            "bracket": {"profit_target": 50.0, "stop_loss": 25.0},
        }
        req = BracketOrderRequest(**data)
        assert req.action == "Sell"
        assert req.bracket.profit_target == 50.0
        assert req.bracket.stop_loss == 25.0
        assert req.is_automated is True  # inherited default

    def test_missing_bracket_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BracketOrderRequest(
                account_spec="demo",
                account_id=1,
                action="Buy",
                symbol="MESM5",
                order_qty=1,
                order_type="Market",
            )

    def test_bracket_with_bad_stop_loss_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BracketOrderRequest(
                account_spec="demo",
                account_id=1,
                action="Buy",
                symbol="MESM5",
                order_qty=1,
                order_type="Market",
                bracket={"profit_target": 50.0, "stop_loss": 0},
            )


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class TestOrder:
    """Unit tests for Order (API response model)."""

    def test_construct_from_dict_working(self) -> None:
        data = {
            "id": 1001,
            "account_id": 12345,
            "symbol": "MESM5",
            "action": "Buy",
            "order_qty": 2,
            "order_type": "Limit",
            "order_status": "Working",
            "price": 5600.50,
            "stop_price": None,
            "filled_qty": 0,
            "avg_fill_price": None,
        }
        order = Order(**data)
        assert order.id == 1001
        assert order.order_status == "Working"
        assert order.filled_qty == 0
        assert order.avg_fill_price is None

    def test_construct_from_dict_filled(self) -> None:
        data = {
            "id": 1002,
            "account_id": 12345,
            "symbol": "MNQM5",
            "action": "Sell",
            "order_qty": 1,
            "order_type": "Market",
            "order_status": "Filled",
            "price": None,
            "stop_price": None,
            "filled_qty": 1,
            "avg_fill_price": 18450.25,
        }
        order = Order(**data)
        assert order.order_status == "Filled"
        assert order.filled_qty == 1
        assert order.avg_fill_price == 18450.25

    def test_defaults(self) -> None:
        data = {
            "id": 1003,
            "account_id": 1,
            "symbol": "MESM5",
            "action": "Buy",
            "order_qty": 1,
            "order_type": "Market",
            "order_status": "Cancelled",
        }
        order = Order(**data)
        assert order.filled_qty == 0
        assert order.avg_fill_price is None
        assert order.price is None
        assert order.stop_price is None

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Order(
                id=1,
                account_id=1,
                symbol="MESM5",
                action="Hold",
                order_qty=1,
                order_type="Market",
                order_status="Working",
            )

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Order(id=1, account_id=1, symbol="MESM5", action="Buy")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    """Unit tests for Position."""

    def test_construct_from_dict_long(self) -> None:
        data = {
            "id": 500,
            "account_id": 12345,
            "symbol": "MESM5",
            "net_pos": 2,
            "avg_price": 5600.25,
            "open_pnl": 125.50,
            "realized_pnl": 0.0,
            "total_pnl": 125.50,
        }
        pos = Position(**data)
        assert pos.net_pos == 2
        assert pos.open_pnl == 125.50

    def test_construct_from_dict_short(self) -> None:
        data = {
            "id": 501,
            "account_id": 12345,
            "symbol": "MNQM5",
            "net_pos": -1,
            "avg_price": 18450.00,
            "open_pnl": -75.00,
            "realized_pnl": 200.00,
            "total_pnl": 125.00,
        }
        pos = Position(**data)
        assert pos.net_pos == -1
        assert pos.total_pnl == 125.00

    def test_defaults(self) -> None:
        data = {
            "id": 502,
            "account_id": 1,
            "symbol": "MESM5",
            "net_pos": 0,
            "avg_price": 0.0,
        }
        pos = Position(**data)
        assert pos.open_pnl == 0.0
        assert pos.realized_pnl == 0.0
        assert pos.total_pnl == 0.0

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Position(id=1, account_id=1, symbol="MESM5")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class TestAccount:
    """Unit tests for Account."""

    def test_construct_from_dict(self) -> None:
        data = {
            "id": 12345,
            "name": "Funded-50k-001",
            "net_liq": 50250.00,
            "realized_pnl": 250.00,
            "open_pnl": 125.00,
            "balance": 50000.00,
            "available_funds": 47000.00,
        }
        acct = Account(**data)
        assert acct.id == 12345
        assert acct.name == "Funded-50k-001"
        assert acct.net_liq == 50250.00
        assert acct.realized_pnl == 250.00

    def test_defaults(self) -> None:
        data = {
            "id": 1,
            "name": "Test",
            "net_liq": 50000.0,
            "balance": 50000.0,
            "available_funds": 50000.0,
        }
        acct = Account(**data)
        assert acct.realized_pnl == 0.0
        assert acct.open_pnl == 0.0

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Account(id=1, name="Test")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

class TestQuote:
    """Unit tests for Quote."""

    def test_construct_from_dict(self) -> None:
        data = {
            "symbol": "MESM5",
            "bid": 5600.00,
            "ask": 5600.25,
            "last": 5600.25,
            "volume": 15000,
            "timestamp": "2026-07-18T14:30:00Z",
        }
        q = Quote(**data)
        assert q.bid == 5600.00
        assert q.ask == 5600.25
        assert q.volume == 15000

    def test_defaults(self) -> None:
        data = {
            "symbol": "MESM5",
            "bid": 5600.00,
            "ask": 5600.25,
            "last": 5600.25,
            "timestamp": "2026-07-18T14:30:00Z",
        }
        q = Quote(**data)
        assert q.volume == 0

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Quote(symbol="MESM5", bid=1.0, ask=1.0, last=1.0)  # type: ignore[call-arg]

    def test_bid_ask_as_floats(self) -> None:
        """Integers should be coerced to float."""
        data = {
            "symbol": "MESM5",
            "bid": 5600,
            "ask": 5601,
            "last": 5600,
            "timestamp": "2026-07-18T14:30:00Z",
        }
        q = Quote(**data)
        assert isinstance(q.bid, float)
        assert isinstance(q.ask, float)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class TestContract:
    """Unit tests for Contract."""

    def test_construct_from_dict(self) -> None:
        data = {
            "id": 270639,
            "name": "MESU5",
            "contract_maturity": "2025-09",
            "product_type": "Futures",
            "tick_size": 0.25,
            "tick_value": 1.25,
            "point_value": 5.00,
            "multiplier": 5,
        }
        c = Contract(**data)
        assert c.id == 270639
        assert c.name == "MESU5"
        assert c.tick_size == 0.25
        assert c.multiplier == 5

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Contract(id=1, name="MESU5")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# OrderResponse
# ---------------------------------------------------------------------------

class TestOrderResponse:
    """Unit tests for OrderResponse."""

    def test_construct_from_dict_ok(self) -> None:
        data = {"order_id": 10042, "status": "Ok"}
        resp = OrderResponse(**data)
        assert resp.order_id == 10042
        assert resp.status == "Ok"
        assert resp.message is None

    def test_construct_from_dict_rejected(self) -> None:
        data = {
            "order_id": 0,
            "status": "Rejected",
            "message": "Not enough buying power",
        }
        resp = OrderResponse(**data)
        assert resp.order_id == 0
        assert resp.status == "Rejected"
        assert resp.message == "Not enough buying power"

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrderResponse(order_id=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# BracketOrderResponse
# ---------------------------------------------------------------------------

class TestBracketOrderResponse:
    """Unit tests for BracketOrderResponse."""

    def test_construct_from_dict(self) -> None:
        data = {
            "entry_order_id": 1100,
            "profit_target_order_id": 1101,
            "stop_loss_order_id": 1102,
            "status": "Ok",
        }
        resp = BracketOrderResponse(**data)
        assert resp.entry_order_id == 1100
        assert resp.profit_target_order_id == 1101
        assert resp.stop_loss_order_id == 1102
        assert resp.status == "Ok"

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BracketOrderResponse(entry_order_id=1)  # type: ignore[call-arg]
