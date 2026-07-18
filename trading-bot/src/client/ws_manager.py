"""
WebSocket Manager for Tradovate API.

Manages a single WebSocket connection lifecycle with automatic reconnection,
message correlation via client IDs (cid), and event-based callback routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class WebSocketTimeoutError(Exception):
    """Raised when a WebSocket request times out waiting for a response."""

    def __init__(self, message: str = "WebSocket request timed out") -> None:
        super().__init__(message)


class AuthenticationError(Exception):
    """Raised when WebSocket authentication fails."""

    def __init__(self, message: str = "WebSocket authentication failed") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# WebSocketManager
# ---------------------------------------------------------------------------


class WebSocketManager:
    """Manages a single WebSocket connection lifecycle to Tradovate.

    Handles connection establishment, authentication, message correlation via
    client-assigned IDs (cid), event-based callback routing, and automatic
    reconnection with exponential backoff.

    Typical usage::

        manager = WebSocketManager("wss://demo.tradovate.com/v1/websocket")

        @manager.on("md")
        def handle_market_data(msg: dict[str, Any]) -> None:
            ...

        await manager.connect(access_token)
        response = await manager.send(
            {"url": "execution/listOrders"}, expect_response=True
        )
        await manager.disconnect()
    """

    # Exponential backoff bounds (seconds)
    _INITIAL_DELAY: float = 1.0
    _MAX_DELAY: float = 30.0
    _AUTH_TIMEOUT: float = 10.0
    _REQUEST_TIMEOUT: float = 30.0

    def __init__(self, url: str) -> None:
        self.url: str = url
        self.ws: ClientConnection | None = None
        self.cid_counter: int = 0
        self.pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.callbacks: dict[str, list[Callable[..., Any]]] = {}
        self._reconnect_delay: float = self._INITIAL_DELAY
        self._running: bool = False
        self._receive_task: asyncio.Task[Any] | None = None
        self._access_token: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self, access_token: str) -> None:
        """Open a WebSocket connection and authenticate.

        Sends ``{"token": access_token}`` as the first message and waits
        for the server to confirm authentication.

        Args:
            access_token: Tradovate API access token obtained via the
                ``/auth/accesstokenrequest`` REST endpoint.

        Raises:
            AuthenticationError: If the server responds with an
                authentication-failure indicator.
            OSError: If the underlying TCP/TLS connection cannot be
                established.
        """
        self._access_token = access_token
        self.ws = await websockets.connect(self.url)

        # Send authentication message per spec.
        auth_msg = json.dumps({"token": access_token})
        await self.ws.send(auth_msg)

        # Block until the server confirms (or rejects) authentication.
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=self._AUTH_TIMEOUT)
        except asyncio.TimeoutError:
            await self.ws.close()
            self.ws = None
            raise AuthenticationError(
                f"No auth confirmation received within {self._AUTH_TIMEOUT}s"
            )

        response: dict[str, Any] = json.loads(raw)

        if self._is_auth_failure(response):
            await self.ws.close()
            self.ws = None
            raise AuthenticationError(
                f"WebSocket authentication rejected: {json.dumps(response)}"
            )

        self._running = True
        self._receive_task = asyncio.create_task(self._listen())
        logger.info("WebSocket connected and authenticated to %s", self.url)

    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully and cancel the receive loop.

        Any pending request futures are cancelled and the ``_running`` flag
        is cleared.  Safe to call multiple times.
        """
        self._running = False

        # Cancel the receive task.
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        # Close the WebSocket.
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

        # Resolve every outstanding future so nothing leaks.
        for cid, future in list(self.pending.items()):
            if not future.done():
                future.cancel()
        self.pending.clear()

        logger.info("WebSocket disconnected from %s", self.url)

    async def send(
        self, msg: dict[str, Any], expect_response: bool = True
    ) -> dict[str, Any] | None:
        """Send a JSON message over the WebSocket.

        When *expect_response* is ``True`` (the default) a unique **cid**
        is attached to the outgoing message and the call blocks until the
        correlated response arrives or the request times out.

        Args:
            msg: The message dictionary to send (must be JSON-serialisable).
            expect_response: If ``True``, wait for a response; otherwise
                fire-and-forget.

        Returns:
            The response dictionary when *expect_response* is ``True``;
            ``None`` otherwise.

        Raises:
            RuntimeError: If the WebSocket is not connected.
            WebSocketTimeoutError: If no response arrives within the
                request timeout window.
        """
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")

        if not expect_response:
            await self.ws.send(json.dumps(msg))
            return None

        # --- expect_response path -------------------------------------------
        self.cid_counter += 1
        cid = self.cid_counter
        msg["cid"] = cid

        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_event_loop().create_future()
        )
        self.pending[cid] = future

        await self.ws.send(json.dumps(msg))

        try:
            return await asyncio.wait_for(future, timeout=self._REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            self.pending.pop(cid, None)
            raise WebSocketTimeoutError(
                f"Request cid={cid} timed out after {self._REQUEST_TIMEOUT}s"
            )

    def on(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register a callback for a specific event type.

        Multiple callbacks may be registered for the same event type;
        they will be invoked in registration order when a matching
        server-push message arrives.

        Args:
            event_type: The event type string (e.g. ``"md"``, ``"orders"``,
                ``"positions"``, ``"accounts"``, ``"clock"``).
            callback: A synchronous or async callable that receives the
                full deserialised message dictionary as its sole argument.
        """
        self.callbacks.setdefault(event_type, []).append(callback)

    async def reconnect(self) -> None:
        """Disconnect, wait with exponential backoff, then reconnect.

        The backoff delay starts at 1 s and doubles on every call, capped
        at 30 s.  It resets to 1 s after a successful :meth:`connect`.
        """
        await self.disconnect()

        logger.info("Reconnecting in %.1f seconds...", self._reconnect_delay)
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self._MAX_DELAY)

        if self._access_token is None:
            raise RuntimeError("Cannot reconnect: no access token stored")

        await self.connect(self._access_token)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _listen(self) -> None:
        """Main receive loop — runs as a background :class:`asyncio.Task`.

        Incoming messages fall into one of two buckets:

        1. **Correlated responses** — if the message carries a ``cid`` that
           matches an outstanding :class:`~asyncio.Future`, that future is
           resolved and the message is *not* forwarded to callbacks.

        2. **Server-push events** — all other messages are dispatched to
           every callback registered for the event type indicated by the
           message's ``"e"`` field.
        """
        while self._running:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                # Periodic wake-up to check the _running flag.
                continue
            except ConnectionClosed:
                if not self._running:
                    break
                logger.warning(
                    "WebSocket connection closed unexpectedly — reconnecting"
                )
                await self._do_reconnect()
                # After a successful inline reconnect the loop continues.
                continue
            except Exception as exc:
                logger.error(
                    "Unexpected error in receive loop: %s", exc, exc_info=True
                )
                if self._running:
                    await self._do_reconnect()
                    continue
                break

            try:
                data: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed WebSocket message: %.200s",
                    str(raw)[:200],
                )
                continue

            # --- CID-based correlation --------------------------------------
            cid = data.get("cid")
            if cid is not None and cid in self.pending:
                future = self.pending.pop(cid)
                if not future.done():
                    future.set_result(data)
                continue

            # --- Event-callback routing -------------------------------------
            event_type: str | None = data.get("e")
            if event_type is None:
                # Some messages use the 'd' field to carry the event type.
                d_val = data.get("d")
                if isinstance(d_val, str) and d_val in self.callbacks:
                    event_type = d_val

            if event_type and event_type in self.callbacks:
                for cb in self.callbacks[event_type]:
                    try:
                        result = cb(data)
                        if asyncio.iscoroutine(result):
                            # Fire background task so a slow callback does
                            # not block the receive loop.
                            asyncio.create_task(result)
                    except Exception as exc:
                        logger.error(
                            "Callback for event '%s' raised: %s",
                            event_type,
                            exc,
                            exc_info=True,
                        )

    async def _do_reconnect(self) -> None:
        """Reconnect without cancelling the current receive task.

        Closes the old WebSocket, waits with exponential backoff, then
        opens a fresh connection and re-authenticates.  Unlike
        :meth:`reconnect`, this does **not** call :meth:`disconnect`, so
        it is safe to invoke from within :meth:`_listen`.
        """
        # Close the old connection quietly.
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

        # Resolve any pending futures — they won't get a response now.
        for cid, future in list(self.pending.items()):
            if not future.done():
                future.cancel()
        self.pending.clear()

        logger.info("Reconnecting in %.1f seconds...", self._reconnect_delay)
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self._MAX_DELAY)

        if self._access_token is None:
            logger.error("Cannot reconnect: no access token stored")
            self._running = False
            return

        # Open a new connection and re-authenticate *without* creating a
        # second receive task (we are already inside the listen loop).
        try:
            self.ws = await websockets.connect(self.url)
            await self.ws.send(json.dumps({"token": self._access_token}))
            raw = await asyncio.wait_for(
                self.ws.recv(), timeout=self._AUTH_TIMEOUT
            )
            response: dict[str, Any] = json.loads(raw)

            if self._is_auth_failure(response):
                await self.ws.close()
                self.ws = None
                logger.error("Reconnection auth failed: %s", response)
                self._running = False
                return

            logger.info("Reconnected successfully to %s", self.url)
        except asyncio.TimeoutError:
            logger.error("Reconnection auth timed out")
            if self.ws is not None:
                await self.ws.close()
                self.ws = None
        except Exception as exc:
            logger.error("Reconnection failed: %s", exc)
            if self.ws is not None:
                await self.ws.close()
                self.ws = None

    @staticmethod
    def _is_auth_failure(response: dict[str, Any]) -> bool:
        """Return ``True`` if ``response`` indicates an auth rejection."""
        # Tradovate may signal auth failure in several ways.
        if response.get("s") == 401:
            return True
        if response.get("e") == "auth_fail":
            return True
        # Some responses nest the failure inside 'd'.
        d = response.get("d")
        if isinstance(d, dict) and d.get("s") == 401:
            return True
        return False
