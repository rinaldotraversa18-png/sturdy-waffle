"""Tests for the TradovateClient."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

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
from src.client.tradovate_client import TradovateAuthError, TradovateClient
from src.client.ws_manager import WebSocketManager
from src.config import TradovateConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> TradovateConfig:
    """A test config pointed at the demo environment."""
    return TradovateConfig(
        TRADOVATE_USERNAME="testuser",
        TRADOVATE_PASSWORD="testpass",
        TRADOVATE_APP_ID="test-app",
        TRADOVATE_DEVICE_ID="test-device",
        environment="demo",
    )


@pytest.fixture
def client(config: TradovateConfig) -> TradovateClient:
    """A fresh TradovateClient (not yet connected)."""
    return TradovateClient(config)


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Return a MagicMock with an AsyncMock .request method."""
    mock = MagicMock(spec=httpx.AsyncClient)
    mock.aclose = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_response(token: str = "test-token-123") -> dict[str, Any]:
    return {"accessToken": token, "expirationTime": 86400}


def _make_httpx_response(
    status_code: int = 200, json_data: dict[str, Any] | None = None
) -> httpx.Response:
    """Build a synthetic httpx.Response."""
    if json_data is None:
        json_data = {}
    request = httpx.Request("GET", "https://demo.tradovate.com/v1/test")
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=request,
    )


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestURLConstruction:
    """Tests that config produces the correct URLs."""

    def test_demo_urls(self) -> None:
        cfg = TradovateConfig(
            TRADOVATE_USERNAME="u",
            TRADOVATE_PASSWORD="p",
            TRADOVATE_APP_ID="a",
            TRADOVATE_DEVICE_ID="d",
            environment="demo",
        )
        assert cfg.api_base_url == "https://demo.tradovate.com/v1"
        assert cfg.ws_base_url == "wss://demo.tradovate.com/v1/websocket"
        assert cfg.md_ws_base_url == "wss://md.demo.tradovate.com/v1/websocket"

    def test_live_urls(self) -> None:
        cfg = TradovateConfig(
            TRADOVATE_USERNAME="u",
            TRADOVATE_PASSWORD="p",
            TRADOVATE_APP_ID="a",
            TRADOVATE_DEVICE_ID="d",
            environment="live",
        )
        assert cfg.api_base_url == "https://live.tradovate.com/v1"
        assert cfg.ws_base_url == "wss://live.tradovate.com/v1/websocket"
        assert cfg.md_ws_base_url == "wss://md.tradovate.com/v1/websocket"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Tests for TradovateClient.authenticate()."""

    @pytest.mark.asyncio
    async def test_authenticate_success(self, client: TradovateClient) -> None:
        resp = _make_httpx_response(200, _auth_response("tok-abc"))
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=resp)

        with patch.object(
            client, "_http_client", mock_client
        ):
            token = await client.authenticate()

        assert token == "tok-abc"
        assert client._access_token == "tok-abc"
        assert client._token_expiry is not None

    @pytest.mark.asyncio
    async def test_authenticate_bad_credentials(
        self, client: TradovateClient
    ) -> None:
        resp = _make_httpx_response(403, {"message": "Forbidden"})
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=resp)
        # httpx raises HTTPStatusError for non-2xx when raise_for_status is
        # called inside _do_request.  Simulate that.
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Forbidden", request=resp.request, response=resp
            )
        )

        with patch.object(
            client, "_http_client", mock_client
        ):
            with pytest.raises(TradovateAuthError, match="Authentication failed"):
                await client.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_no_token_in_response(
        self, client: TradovateClient
    ) -> None:
        resp = _make_httpx_response(200, {"unexpected": "response"})
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=resp)

        with patch.object(
            client, "_http_client", mock_client
        ):
            with pytest.raises(TradovateAuthError, match="No access token"):
                await client.authenticate()


# ---------------------------------------------------------------------------
# REST operations (happy-path, using mocked HTTP)
# ---------------------------------------------------------------------------


class TestRESTOperations:
    """Tests for place_order, place_bracket, cancel_order, modify_order,
    get_account, get_positions, get_quote, search_contract."""

    @pytest.mark.asyncio
    async def test_place_order(self, client: TradovateClient) -> None:
        order_req = OrderRequest(
            account_spec="test",
            account_id=100,
            action="Buy",
            symbol="MESU5",
            order_qty=1,
            order_type="Market",
        )
        resp_data = {"order_id": 42, "status": "Ok"}
        resp = _make_httpx_response(200, resp_data)

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.place_order(order_req)
        assert isinstance(result, OrderResponse)
        assert result.order_id == 42
        assert result.status == "Ok"

    @pytest.mark.asyncio
    async def test_place_bracket(self, client: TradovateClient) -> None:
        bracket = BracketOrderRequest(
            account_spec="test",
            account_id=100,
            action="Buy",
            symbol="MBTU5",
            order_qty=2,
            order_type="Market",
            bracket=BracketConfig(profit_target=50.0, stop_loss=25.0),
        )
        resp_data = {
            "entry_order_id": 1,
            "profit_target_order_id": 2,
            "stop_loss_order_id": 3,
            "status": "Ok",
        }
        resp = _make_httpx_response(200, resp_data)

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.place_bracket(bracket)
        assert isinstance(result, BracketOrderResponse)
        assert result.entry_order_id == 1
        assert result.status == "Ok"

    @pytest.mark.asyncio
    async def test_cancel_order(self, client: TradovateClient) -> None:
        resp = _make_httpx_response(200, {"status": "Cancelled"})
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        # cancel_order returns None on success.
        result = await client.cancel_order(99)
        assert result is None
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_modify_order(self, client: TradovateClient) -> None:
        resp = _make_httpx_response(200, {"status": "Ok"})
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.modify_order(55, {"price": 5100.0})
        assert result is None

    @pytest.mark.asyncio
    async def test_get_account(self, client: TradovateClient) -> None:
        acct_data = {
            "id": 100,
            "name": "Test Acct",
            "net_liq": 150000.0,
            "balance": 150000.0,
            "available_funds": 149000.0,
            "realized_pnl": 500.0,
            "open_pnl": -200.0,
        }
        resp = _make_httpx_response(200, acct_data)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.get_account(100)
        assert isinstance(result, Account)
        assert result.id == 100
        assert result.net_liq == 150000.0

    @pytest.mark.asyncio
    async def test_get_account_wrapped_in_d(self, client: TradovateClient) -> None:
        """Account returned inside 'd' field."""
        acct_data = {
            "d": {
                "id": 100,
                "name": "Test Acct",
                "net_liq": 150000.0,
                "balance": 150000.0,
                "available_funds": 149000.0,
            }
        }
        resp = _make_httpx_response(200, acct_data)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.get_account(100)
        assert result.id == 100

    @pytest.mark.asyncio
    async def test_get_positions(self, client: TradovateClient) -> None:
        pos_data = {
            "d": [
                {
                    "id": 1,
                    "account_id": 100,
                    "symbol": "MESU5",
                    "net_pos": 2,
                    "avg_price": 5020.0,
                }
            ]
        }
        resp = _make_httpx_response(200, pos_data)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.get_positions()
        assert len(result) == 1
        assert isinstance(result[0], Position)
        assert result[0].symbol == "MESU5"

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, client: TradovateClient) -> None:
        resp = _make_httpx_response(200, {"d": []})
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.get_positions()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_quote(self, client: TradovateClient) -> None:
        quote_data = {
            "symbol": "MESU5",
            "bid": 5010.25,
            "ask": 5011.00,
            "last": 5010.75,
            "volume": 5000,
            "timestamp": "2026-07-18T12:00:00Z",
        }
        resp = _make_httpx_response(200, {"d": quote_data})
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.get_quote("MESU5")
        assert isinstance(result, Quote)
        assert result.symbol == "MESU5"
        assert result.bid == 5010.25

    @pytest.mark.asyncio
    async def test_search_contract(self, client: TradovateClient) -> None:
        contracts_data = {
            "d": [
                {
                    "id": 1001,
                    "name": "MESU5",
                    "contract_maturity": "202509",
                    "product_type": "Futures",
                    "tick_size": 0.25,
                    "tick_value": 1.25,
                    "point_value": 5.0,
                    "multiplier": 5.0,
                }
            ]
        }
        resp = _make_httpx_response(200, contracts_data)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=resp)

        client._access_token = "tok"
        client._http_client = mock_http

        result = await client.search_contract("MES")
        assert len(result) == 1
        assert isinstance(result[0], Contract)
        assert result[0].name == "MESU5"


# ---------------------------------------------------------------------------
# Token refresh on 401
# ---------------------------------------------------------------------------


class TestTokenRefresh:
    """Tests that 401 responses trigger a token refresh and retry."""

    @pytest.mark.asyncio
    async def test_401_triggers_token_refresh_and_retry(
        self, client: TradovateClient
    ) -> None:
        """First request gets 401; second (after re-auth) succeeds."""
        # Create responses.
        unauth_resp = httpx.Response(
            status_code=401,
            json={"error": "unauthorized"},
            request=httpx.Request("GET", "https://demo.tradovate.com/v1/test"),
        )
        ok_resp = _make_httpx_response(
            200,
            {
                "d": [
                    {"id": 1, "account_id": 100, "symbol": "MESU5", "net_pos": 0, "avg_price": 5000.0}
                ]
            },
        )

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        # First call returns 401, second returns success.
        mock_http.get = AsyncMock(side_effect=[
            httpx.HTTPStatusError(
                "Unauthorized", request=unauth_resp.request, response=unauth_resp
            ),
            ok_resp,
        ])
        mock_http.post = AsyncMock(
            return_value=_make_httpx_response(200, _auth_response("new-token"))
        )

        client._http_client = mock_http
        client._access_token = "old-token"

        result = await client.get_positions()
        assert len(result) == 1
        # Token should have been refreshed.
        assert client._access_token == "new-token"
        # Two GETs: one 401, one success; one POST for auth.
        assert mock_http.get.call_count == 2
        assert mock_http.post.call_count == 1

    @pytest.mark.asyncio
    async def test_401_twice_raises(self, client: TradovateClient) -> None:
        """If re-auth also gets 401, the error should propagate."""
        unauth_resp = httpx.Response(
            status_code=401,
            json={"error": "unauthorized"},
            request=httpx.Request("GET", "https://demo.tradovate.com/v1/test"),
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Unauthorized", request=unauth_resp.request, response=unauth_resp
            )
        )
        # Auth also fails — but _request calls authenticate which does POST.
        # authenticate raises TradovateAuthError on non-200.
        mock_http.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Forbidden",
                request=unauth_resp.request,
                response=httpx.Response(
                    403,
                    json={"error": "forbidden"},
                    request=unauth_resp.request,
                ),
            )
        )

        client._http_client = mock_http
        client._access_token = "expired"

        with pytest.raises(TradovateAuthError, match="Authentication failed"):
            await client.get_positions()


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    """Tests for the WebSocket connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_authenticates_if_needed(
        self, client: TradovateClient
    ) -> None:
        """connect() calls authenticate() when no token is cached."""
        auth_resp = _make_httpx_response(200, _auth_response("connect-tok"))
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=auth_resp)

        client._http_client = mock_http

        with patch(
            "src.client.tradovate_client.WebSocketManager"
        ) as MockWSMgr:
            mock_ws_instance = AsyncMock(spec=WebSocketManager)
            mock_ws_instance.connect = AsyncMock()
            mock_ws_instance.on = MagicMock()
            mock_ws_instance.disconnect = AsyncMock()
            MockWSMgr.side_effect = [mock_ws_instance, mock_ws_instance]

            await client.connect()

        MockWSMgr.assert_called()  # Called twice (trading + market).
        assert client._access_token == "connect-tok"
        assert client._trading_ws is not None
        assert client._market_ws is not None

    @pytest.mark.asyncio
    async def test_connect_skips_auth_when_token_valid(
        self, client: TradovateClient
    ) -> None:
        """connect() reuses cached token when not expired."""
        import time

        client._access_token = "cached-tok"
        client._token_expiry = time.time() + 3600  # valid for 1 hr

        with patch(
            "src.client.tradovate_client.WebSocketManager"
        ) as MockWSMgr:
            mock_ws_instance = AsyncMock(spec=WebSocketManager)
            mock_ws_instance.connect = AsyncMock()
            mock_ws_instance.on = MagicMock()
            mock_ws_instance.disconnect = AsyncMock()
            MockWSMgr.side_effect = [mock_ws_instance, mock_ws_instance]

            await client.connect()

        # Token unchanged.
        assert client._access_token == "cached-tok"

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(
        self, client: TradovateClient
    ) -> None:
        """disconnect() closes both WS connections and the HTTP client."""
        import time

        client._access_token = "tok"
        client._token_expiry = time.time() + 3600

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.aclose = AsyncMock()
        client._http_client = mock_http

        with patch(
            "src.client.tradovate_client.WebSocketManager"
        ) as MockWSMgr:
            mock_ws = AsyncMock(spec=WebSocketManager)
            mock_ws.connect = AsyncMock()
            mock_ws.on = MagicMock()
            mock_ws.disconnect = AsyncMock()
            MockWSMgr.side_effect = [mock_ws, mock_ws]

            await client.connect()
            await client.disconnect()

        assert client._trading_ws is None
        assert client._market_ws is None
        mock_http.aclose.assert_called_once()
        # Each WS disconnect called once.
        assert mock_ws.disconnect.call_count == 2

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, client: TradovateClient
    ) -> None:
        """__aenter__ / __aexit__ work correctly."""
        import time

        client._access_token = "tok"
        client._token_expiry = time.time() + 3600

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.aclose = AsyncMock()
        client._http_client = mock_http

        with patch(
            "src.client.tradovate_client.WebSocketManager"
        ) as MockWSMgr:
            mock_ws = AsyncMock(spec=WebSocketManager)
            mock_ws.connect = AsyncMock()
            mock_ws.on = MagicMock()
            mock_ws.disconnect = AsyncMock()
            MockWSMgr.side_effect = [mock_ws, mock_ws]

            async with client as c:
                await c.connect()
                assert c._trading_ws is not None

        # After __aexit__, everything should be cleaned up.
        assert client._trading_ws is None
        assert client._market_ws is None


# ---------------------------------------------------------------------------
# subscribe_quotes
# ---------------------------------------------------------------------------


class TestSubscribeQuotes:
    """Tests for the subscribe_quotes WebSocket method."""

    @pytest.mark.asyncio
    async def test_subscribe_quotes_sends_message(
        self, client: TradovateClient
    ) -> None:
        """subscribe_quotes sends the right subscription payload."""
        import time

        client._access_token = "tok"
        client._token_expiry = time.time() + 3600

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.aclose = AsyncMock()
        client._http_client = mock_http

        with patch(
            "src.client.tradovate_client.WebSocketManager"
        ) as MockWSMgr:
            mock_trading = AsyncMock(spec=WebSocketManager)
            mock_trading.connect = AsyncMock()
            mock_trading.on = MagicMock()
            mock_trading.disconnect = AsyncMock()

            mock_md = AsyncMock(spec=WebSocketManager)
            mock_md.connect = AsyncMock()
            mock_md.on = MagicMock()
            mock_md.disconnect = AsyncMock()
            mock_md.send = AsyncMock(return_value={"s": 200})

            MockWSMgr.side_effect = [mock_trading, mock_md]

            await client.connect()
            await client.subscribe_quotes(["MESU5", "MNQU5"])

        mock_md.send.assert_called_once()
        call_args = mock_md.send.call_args[0][0]
        assert call_args["url"] == "md/subscribequotes"
        assert "MESU5" in call_args["body"]["symbol"]

    @pytest.mark.asyncio
    async def test_subscribe_quotes_not_connected_raises(
        self, client: TradovateClient
    ) -> None:
        """subscribe_quotes raises RuntimeError when market WS is None."""
        with pytest.raises(
            RuntimeError, match="Market data WebSocket is not connected"
        ):
            await client.subscribe_quotes(["MESU5"])


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


class TestCallbacks:
    """Tests for on_* callback registration and dispatch."""

    def test_on_order_update_registration(self, client: TradovateClient) -> None:
        received: list[Order] = []

        def handler(o: Order) -> None:
            received.append(o)

        client.on_order_update(handler)
        assert "order_update" in client._callbacks
        assert len(client._callbacks["order_update"]) == 1

    def test_on_position_update_registration(self, client: TradovateClient) -> None:
        received: list[Position] = []

        def handler(p: Position) -> None:
            received.append(p)

        client.on_position_update(handler)
        assert "position_update" in client._callbacks

    def test_on_account_update_registration(self, client: TradovateClient) -> None:
        received: list[Account] = []

        def handler(a: Account) -> None:
            received.append(a)

        client.on_account_update(handler)
        assert "account_update" in client._callbacks

    def test_on_quote_registration(self, client: TradovateClient) -> None:
        received: list[Quote] = []

        def handler(q: Quote) -> None:
            received.append(q)

        client.on_quote(handler)
        assert "quote" in client._callbacks


# ---------------------------------------------------------------------------
# Dispatcher (internal) — _mk_dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Tests for the internal _mk_dispatcher callback pipeline."""

    def test_dispatcher_order(self, client: TradovateClient) -> None:
        received: list[Order] = []
        client.on_order_update(received.append)

        disp = client._mk_dispatcher("order_update")
        msg = {
            "e": "orders",
            "d": {
                "id": 42,
                "account_id": 100,
                "symbol": "MESU5",
                "action": "Buy",
                "order_qty": 1,
                "order_type": "Market",
                "order_status": "Filled",
                "filled_qty": 1,
            },
        }
        disp(msg)
        assert len(received) == 1
        assert isinstance(received[0], Order)
        assert received[0].id == 42

    def test_dispatcher_quote(self, client: TradovateClient) -> None:
        received: list[Quote] = []
        client.on_quote(received.append)

        disp = client._mk_dispatcher("quote")
        msg = {
            "e": "md",
            "d": {
                "symbol": "MESU5",
                "bid": 5010.0,
                "ask": 5011.0,
                "last": 5010.5,
                "volume": 100,
                "timestamp": "2026-07-18T12:00:00Z",
            },
        }
        disp(msg)
        assert len(received) == 1
        assert isinstance(received[0], Quote)
        assert received[0].bid == 5010.0

    def test_dispatcher_bad_data_does_not_crash(
        self, client: TradovateClient
    ) -> None:
        """A malformed payload is logged and skipped — callbacks not invoked."""
        received: list[Order] = []
        client.on_order_update(received.append)

        disp = client._mk_dispatcher("order_update")
        # Missing required fields.
        disp({"e": "orders", "d": {"id": "not-an-int"}})
        assert len(received) == 0


# ---------------------------------------------------------------------------
# _build_url
# ---------------------------------------------------------------------------


class TestBuildURL:
    """Tests for the _build_url helper."""

    def test_basic(self, client: TradovateClient) -> None:
        url = client._build_url("order/placeorder")
        assert url == "https://demo.tradovate.com/v1/order/placeorder"

    def test_trailing_slash_on_base(self, client: TradovateClient) -> None:
        # Even if base has a trailing slash, it should work.
        url = client._build_url("/order/placeorder")
        assert url == "https://demo.tradovate.com/v1/order/placeorder"


# ---------------------------------------------------------------------------
# _auth_headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    """Tests for _auth_headers."""

    def test_with_token(self, client: TradovateClient) -> None:
        client._access_token = "my-token"
        h = client._auth_headers()
        assert h == {"Authorization": "Bearer my-token"}

    def test_without_token(self, client: TradovateClient) -> None:
        h = client._auth_headers()
        assert h == {"Authorization": "Bearer "}


# ---------------------------------------------------------------------------
# _is_token_expired
# ---------------------------------------------------------------------------


class TestTokenExpired:
    """Tests for _is_token_expired."""

    def test_no_token(self, client: TradovateClient) -> None:
        assert client._is_token_expired() is True

    def test_expired(self, client: TradovateClient) -> None:
        import time

        client._token_expiry = time.time() - 3600
        assert client._is_token_expired() is True

    def test_valid(self, client: TradovateClient) -> None:
        import time

        client._token_expiry = time.time() + 3600
        assert client._is_token_expired() is False

    def test_grace_period(self, client: TradovateClient) -> None:
        """Token within 60 s of expiry is considered expired."""
        import time

        client._token_expiry = time.time() + 30
        assert client._is_token_expired() is True
