"""
Async REST + WebSocket client for the Tradovate API.

Provides a single :class:`TradovateClient` that manages authentication,
REST HTTP calls, and two WebSocket connections (trading + market data).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import TracebackType
from typing import Any, Callable

import httpx

from src.client.models import (
    Account,
    BracketOrderRequest,
    BracketOrderResponse,
    Contract,
    Order,
    OrderRequest,
    OrderResponse,
    Position,
    Quote,
)
from src.client.ws_manager import WebSocketManager
from src.config import TradovateConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TradovateAuthError(Exception):
    """Raised when REST authentication fails (e.g. bad credentials)."""


# ---------------------------------------------------------------------------
# TradovateClient
# ---------------------------------------------------------------------------


class TradovateClient:
    """Async client for Tradovate REST + WebSocket API.

    Use as an **async context manager**::

        async with TradovateClient(config) as client:
            await client.connect()
            acct = await client.get_account(12345)
            ...

    Closing the context manager calls :meth:`disconnect`, which tears down
    both WebSocket connections and closes the HTTP session.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, config: TradovateConfig) -> None:
        self.config: TradovateConfig = config
        self._http_client: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._token_expiry: float | None = None

        # WebSocket managers are created lazily by connect().
        self._trading_ws: WebSocketManager | None = None
        self._market_ws: WebSocketManager | None = None

        # Callback registries — thin wrappers around WebSocketManager.on().
        self._callbacks: dict[str, list[Callable[..., Any]]] = {}

    async def __aenter__(self) -> "TradovateClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # REST: Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> str:
        """Obtain (or refresh) an access token via ``POST /auth/accesstokenrequest``.

        Returns the access-token string.  The token is cached in
        ``self._access_token`` and re-used until expiry.

        Raises:
            TradovateAuthError: If credentials are rejected (any non-200).
        """
        path = "auth/accesstokenrequest"
        payload = {
            "name": self.config.username,
            "password": self.config.password,
            "appId": self.config.app_id,
            "appVersion": "1.0",
            "deviceId": self.config.device_id,
        }
        try:
            resp = await self._request("POST", path, data=payload)
        except httpx.HTTPStatusError as exc:
            raise TradovateAuthError(
                f"Authentication failed: {exc.response.status_code} "
                f"— {exc.response.text[:200]}"
            ) from exc

        token: str = resp.get("accessToken", "")
        if not token:
            # Some deployments place it under a different key.
            token = resp.get("d", {}).get("accessToken", "")
        if not token:
            raise TradovateAuthError(
                f"No access token in response: {json.dumps(resp)[:300]}"
            )

        expires_in: int = resp.get("expirationTime", 86400)
        self._access_token = token
        self._token_expiry = time.time() + expires_in

        # If we don't have an HTTP client yet, create one now.
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
            )

        logger.info("Authenticated — token expires in %ds", expires_in)
        return token

    # ------------------------------------------------------------------
    # REST: Trading operations
    # ------------------------------------------------------------------

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a single order via ``POST /order/placeorder``."""
        data = await self._request(
            "POST", "order/placeorder", data=order.model_dump()
        )
        return OrderResponse(**data)

    async def place_bracket(
        self, bracket: BracketOrderRequest
    ) -> BracketOrderResponse:
        """Place a bracket (OCO) order via ``POST /order/strategy``."""
        # The bracket-order endpoint expects the entry + bracket config
        # under a specific structure.
        data = await self._request(
            "POST", "order/strategy", data=bracket.model_dump()
        )
        return BracketOrderResponse(**data)

    async def cancel_order(self, order_id: int) -> None:
        """Cancel an open order via ``POST /order/cancelorder``."""
        await self._request(
            "POST", "order/cancelorder", data={"orderId": order_id}
        )

    async def modify_order(self, order_id: int, changes: dict[str, Any]) -> None:
        """Modify an existing order via ``POST /order/modifyorder``."""
        payload: dict[str, Any] = {"orderId": order_id, **changes}
        await self._request("POST", "order/modifyorder", data=payload)

    # ------------------------------------------------------------------
    # REST: Query operations
    # ------------------------------------------------------------------

    async def get_account(self, account_id: int) -> Account:
        """Fetch account summary via ``GET /account/item``."""
        data = await self._request("GET", f"account/item?id={account_id}")
        # The response may wrap the account under 'd' or be the object itself.
        if "d" in data and isinstance(data["d"], dict):
            return Account(**data["d"])
        return Account(**data)

    async def get_positions(self) -> list[Position]:
        """Fetch all open positions via ``GET /position/list``."""
        data = await self._request("GET", "position/list")
        items: list[dict[str, Any]] = data.get("d", data.get("positions", []))
        if not isinstance(items, list):
            items = []
        return [Position(**p) for p in items]

    async def get_quote(self, symbol: str) -> Quote:
        """Fetch a snapshot quote via ``GET /md/quote``."""
        data = await self._request("GET", f"md/quote?s={symbol}")
        d: dict[str, Any] = data.get("d", data)
        return Quote(**d)

    async def search_contract(self, name: str) -> list[Contract]:
        """Search contracts via ``GET /contract/search``."""
        data = await self._request("GET", f"contract/search?name={name}")
        items: list[dict[str, Any]] = data.get("d", data.get("contracts", []))
        if not isinstance(items, list):
            items = []
        return [Contract(**c) for c in items]

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate (if needed) and open both WebSocket connections.

        Trading WS carries orders, positions, account updates.
        Market-data WS carries real-time quotes.
        """
        # Ensure we have a valid token.
        if self._access_token is None or self._is_token_expired():
            await self.authenticate()

        assert self._access_token is not None

        # --- Trading WebSocket ------------------------------------------------
        self._trading_ws = WebSocketManager(self.config.ws_base_url)
        await self._trading_ws.connect(self._access_token)

        # Wire up callback forwarding.
        self._trading_ws.on("orders", self._mk_dispatcher("order_update"))
        self._trading_ws.on("positions", self._mk_dispatcher("position_update"))
        self._trading_ws.on("accounts", self._mk_dispatcher("account_update"))

        # --- Market Data WebSocket --------------------------------------------
        self._market_ws = WebSocketManager(self.config.md_ws_base_url)
        await self._market_ws.connect(self._access_token)

        self._market_ws.on("md", self._mk_dispatcher("quote"))

        logger.info("Both WebSocket connections established")

    async def disconnect(self) -> None:
        """Close both WebSocket connections and the HTTP client."""
        for ws in (self._trading_ws, self._market_ws):
            if ws is not None:
                try:
                    await ws.disconnect()
                except Exception as exc:
                    logger.warning("Error closing WebSocket: %s", exc)

        self._trading_ws = None
        self._market_ws = None

        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("All connections closed")

    async def subscribe_quotes(self, symbols: list[str]) -> None:
        """Subscribe to real-time quotes for the given *symbols*.

        Sends a market-data subscription message via the market-data
        WebSocket connection.

        Raises:
            RuntimeError: If the market-data WebSocket is not connected.
        """
        if self._market_ws is None:
            raise RuntimeError(
                "Market data WebSocket is not connected — call connect() first"
            )
        sub_msg: dict[str, Any] = {
            "url": "md/subscribequotes",
            "body": {"symbol": ",".join(symbols)},
        }
        await self._market_ws.send(sub_msg, expect_response=True)
        logger.info("Subscribed to quotes: %s", symbols)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_order_update(
        self, cb: Callable[[Order], Any]
    ) -> None:
        """Register a callback for order updates (fills, status changes)."""
        self._register_callback("order_update", cb)

    def on_position_update(
        self, cb: Callable[[Position], Any]
    ) -> None:
        """Register a callback for position updates."""
        self._register_callback("position_update", cb)

    def on_account_update(
        self, cb: Callable[[Account], Any]
    ) -> None:
        """Register a callback for account balance / PnL updates."""
        self._register_callback("account_update", cb)

    def on_quote(
        self, cb: Callable[[Quote], Any]
    ) -> None:
        """Register a callback for real-time quote updates."""
        self._register_callback("quote", cb)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generic REST helper with auth header and 401 retry.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``).
            endpoint: API path relative to the base URL (e.g. ``"account/list"``).
            data: Optional JSON body for POST requests.

        Returns:
            The deserialised JSON response body.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
            )

        url = self._build_url(endpoint)
        headers = self._auth_headers()

        try:
            return await self._do_request(method, url, headers, data)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                logger.info("Got 401 — refreshing token and retrying")
                await self.authenticate()
                headers = self._auth_headers()
                return await self._do_request(method, url, headers, data)
            raise

    async def _do_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute a single HTTP request and return parsed JSON.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
        """
        assert self._http_client is not None
        if method == "GET":
            resp = await self._http_client.get(url, headers=headers)
        else:
            resp = await self._http_client.post(
                url, headers=headers, json=data
            )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def _build_url(self, endpoint: str) -> str:
        """Construct a full REST URL from the config base URL and *endpoint*."""
        base = self.config.api_base_url.rstrip("/")
        ep = endpoint.lstrip("/")
        return f"{base}/{ep}"

    def _auth_headers(self) -> dict[str, str]:
        """Return the ``Authorization: Bearer`` header dict."""
        token = self._access_token or ""
        return {"Authorization": f"Bearer {token}"}

    def _is_token_expired(self) -> bool:
        """Check whether the cached access token has expired."""
        if self._token_expiry is None:
            return True
        return time.time() >= (self._token_expiry - 60)  # 60 s grace

    def _register_callback(
        self, event: str, cb: Callable[..., Any]
    ) -> None:
        """Store a user callback so it can be wired during :meth:`connect`."""
        self._callbacks.setdefault(event, []).append(cb)

    def _mk_dispatcher(
        self, event: str
    ) -> Callable[[dict[str, Any]], None]:
        """Create a dispatcher that deserialises a WS message and fires callbacks.

        The dispatcher extracts the ``"d"`` payload from the incoming
        WebSocket message, constructs the appropriate Pydantic model
        (:class:`Order`, :class:`Position`, :class:`Account`, or
        :class:`Quote`), and passes it to every callback registered under
        *event*.
        """
        model_map: dict[str, type[Order | Position | Account | Quote]] = {
            "order_update": Order,
            "position_update": Position,
            "account_update": Account,
            "quote": Quote,
        }
        model_cls = model_map.get(event)

        if model_cls is None:
            # Fallback: pass raw dict.
            def _dispatch(msg: dict[str, Any]) -> None:
                for cb in self._callbacks.get(event, []):
                    self._safe_call(cb, msg)

            return _dispatch

        def _dispatch(msg: dict[str, Any]) -> None:
            payload = msg.get("d", msg)
            # If the payload is a list, iterate and dispatch each item.
            items: list[dict[str, Any]] = (
                payload if isinstance(payload, list) else [payload]
            )
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    obj = model_cls(**item)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse %s from %s: %s",
                        model_cls.__name__,
                        str(item)[:200],
                        exc,
                    )
                    continue
                for cb in self._callbacks.get(event, []):
                    self._safe_call(cb, obj)

        return _dispatch

    @staticmethod
    def _safe_call(cb: Callable[..., Any], arg: Any) -> None:
        """Invoke *cb(arg)*, catching exceptions so one bad callback
        can't break the dispatch chain.

        If the callback returns a coroutine it is scheduled as a
        fire-and-forget background task.
        """
        try:
            result = cb(arg)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception as exc:
            logger.error(
                "Callback %s raised: %s",
                getattr(cb, "__name__", str(cb)),
                exc,
                exc_info=True,
            )
