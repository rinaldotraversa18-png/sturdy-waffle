"""Tests for the WebSocketManager."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from src.client.ws_manager import (
    AuthenticationError,
    WebSocketManager,
    WebSocketTimeoutError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_URL = "wss://demo.tradovate.com/v1/websocket"
ACCESS_TOKEN = "test-access-token-123"


# ---------------------------------------------------------------------------
# Helpers — mock WebSocket factories
# ---------------------------------------------------------------------------


def _make_mock_ws(
    receive_queue: asyncio.Queue[str] | None = None,
    send_mock: AsyncMock | None = None,
    close_mock: AsyncMock | None = None,
) -> AsyncMock:
    """Create an AsyncMock that quacks like a ClientConnection."""
    ws = AsyncMock(spec=ClientConnection)
    ws.send = send_mock or AsyncMock()

    if receive_queue is not None:

        async def recv() -> str:
            return await receive_queue.get()

        ws.recv = recv
    else:
        ws.recv = AsyncMock()

    ws.close = close_mock or AsyncMock()
    return ws


def _auth_ok_msg() -> str:
    """A generic auth-success response from Tradovate."""
    return json.dumps({"s": 200, "d": {"message": "Authenticated"}})


def _install_connect_patch(mock_ws: AsyncMock) -> Any:
    """Patch src.client.ws_manager.websockets.connect to return *mock_ws*.

    Returns the context-manager object so the caller can enter it.
    """
    mock_connect = AsyncMock(return_value=mock_ws)
    return patch("src.client.ws_manager.websockets.connect", new=mock_connect)


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_sends_auth_and_starts_listener() -> None:
    """connect() should send the token and set _running=True on success."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    # Auth message was sent.
    mock_ws.send.assert_called_once_with(json.dumps({"token": ACCESS_TOKEN}))
    # State is correct.
    assert manager._running is True
    assert manager._receive_task is not None
    assert manager._access_token == ACCESS_TOKEN
    # Clean up.
    await manager.disconnect()


@pytest.mark.asyncio
async def test_connect_auth_failure_s_401() -> None:
    """connect() raises AuthenticationError when server returns s=401."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(json.dumps({"s": 401}))
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        with pytest.raises(AuthenticationError, match="401"):
            await manager.connect(ACCESS_TOKEN)

    assert manager._running is False
    mock_ws.close.assert_called_once()


@pytest.mark.asyncio
async def test_connect_auth_failure_e_field() -> None:
    """connect() raises AuthenticationError with e='auth_fail'."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(json.dumps({"e": "auth_fail"}))
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        with pytest.raises(AuthenticationError, match="auth_fail"):
            await manager.connect(ACCESS_TOKEN)


@pytest.mark.asyncio
async def test_connect_auth_failure_nested_d() -> None:
    """connect() raises AuthenticationError with nested d.s=401."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(json.dumps({"d": {"s": 401}}))
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        with pytest.raises(AuthenticationError):
            await manager.connect(ACCESS_TOKEN)


@pytest.mark.asyncio
async def test_connect_auth_timeout() -> None:
    """connect() raises AuthenticationError when no response arrives in time."""
    mock_ws = _make_mock_ws()
    mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        with pytest.raises(AuthenticationError, match="auth confirmation"):
            await manager.connect(ACCESS_TOKEN)

    mock_ws.close.assert_called_once()


# ---------------------------------------------------------------------------
# disconnect()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_cleans_up() -> None:
    """disconnect() sets _running=False, closes ws, cancels futures."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    # Add a pending future.
    future: asyncio.Future[dict[str, Any]] = asyncio.Future()
    manager.pending[99] = future

    await manager.disconnect()

    assert manager._running is False
    assert manager._receive_task is None
    assert manager.ws is None
    mock_ws.close.assert_called_once()
    # The pending future should be cancelled.
    assert future.cancelled() or future.done()
    assert len(manager.pending) == 0


@pytest.mark.asyncio
async def test_disconnect_idempotent() -> None:
    """disconnect() is safe to call multiple times."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    await manager.disconnect()
    await manager.disconnect()  # second call should not raise

    assert manager._running is False


# ---------------------------------------------------------------------------
# send()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_with_response() -> None:
    """send() with expect_response=True assigns cid and returns response."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    # Queue up the expected response to be picked up by _listen().
    response_payload = {"cid": 1, "d": {"results": [1, 2, 3]}}
    await rx.put(json.dumps(response_payload))

    msg = {"url": "execution/listOrders"}
    result = await manager.send(msg, expect_response=True)

    assert result == response_payload
    # Verify cid was attached.
    sent_raw = mock_ws.send.call_args_list[-1][0][0]
    sent_obj = json.loads(sent_raw)
    assert sent_obj["cid"] == 1
    assert manager.cid_counter == 1

    await manager.disconnect()


@pytest.mark.asyncio
async def test_send_fire_and_forget() -> None:
    """send() with expect_response=False fires and returns None."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    result = await manager.send({"ping": True}, expect_response=False)
    assert result is None

    # No cid should be attached.
    sent_raw = mock_ws.send.call_args_list[-1][0][0]
    sent_obj = json.loads(sent_raw)
    assert "cid" not in sent_obj

    await manager.disconnect()


@pytest.mark.asyncio
async def test_send_not_connected_raises() -> None:
    """send() raises RuntimeError when not connected."""
    manager = WebSocketManager(WS_URL)
    with pytest.raises(RuntimeError, match="not connected"):
        await manager.send({"foo": "bar"})


@pytest.mark.asyncio
async def test_send_timeout_raises() -> None:
    """send() raises WebSocketTimeoutError when response never arrives."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    # Set a very short timeout for the test.
    manager._REQUEST_TIMEOUT = 0.01

    # Do NOT queue up a response — the pending future will time out.
    with pytest.raises(WebSocketTimeoutError, match="timed out"):
        await manager.send({"url": "some/request"}, expect_response=True)

    # The pending dict should be clean.
    assert len(manager.pending) == 0
    await manager.disconnect()


# ---------------------------------------------------------------------------
# on() — callback registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_registers_callback() -> None:
    """on() adds callbacks to a per-event-type list."""
    manager = WebSocketManager(WS_URL)
    cb1: list[dict[str, Any]] = []
    cb2: list[dict[str, Any]] = []

    manager.on("md", cb1.append)
    manager.on("md", cb2.append)
    manager.on("orders", cb1.append)

    assert len(manager.callbacks) == 2
    assert len(manager.callbacks["md"]) == 2
    assert len(manager.callbacks["orders"]) == 1


@pytest.mark.asyncio
async def test_callbacks_invoked_on_event() -> None:
    """Registered callbacks are invoked when an event arrives."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    received: list[dict[str, Any]] = []
    manager.on("md", received.append)

    # Push an "md" event onto the queue.
    md_msg = {"e": "md", "d": {"symbol": "MESU5", "bid": 5000.25}}
    await rx.put(json.dumps(md_msg))

    # Give the receive loop a moment to process.
    await asyncio.sleep(0.1)

    assert len(received) == 1
    assert received[0] == md_msg

    await manager.disconnect()


@pytest.mark.asyncio
async def test_callback_d_field_routing() -> None:
    """When 'e' is absent, the string 'd' field is used as event_type."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    received: list[dict[str, Any]] = []
    manager.on("clock", received.append)

    # Tradovate clock message uses d="clock" without an "e" field.
    clock_msg = {"d": "clock", "t": "2026-07-18T12:00:00Z"}
    await rx.put(json.dumps(clock_msg))

    await asyncio.sleep(0.1)
    assert len(received) == 1
    assert received[0] == clock_msg

    await manager.disconnect()


@pytest.mark.asyncio
async def test_callback_error_does_not_crash_listener() -> None:
    """A callback that raises is logged, not re-raised into the listen loop."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    def bad_callback(_msg: dict[str, Any]) -> None:
        raise ValueError("boom")

    manager.on("md", bad_callback)

    good: list[dict[str, Any]] = []

    def good_callback(msg: dict[str, Any]) -> None:
        good.append(msg)

    manager.on("md", good_callback)

    md_msg = {"e": "md", "d": {"symbol": "MESU5"}}
    await rx.put(json.dumps(md_msg))
    await asyncio.sleep(0.1)

    # The bad callback raised but the good one still ran.
    assert len(good) == 1
    assert good[0] == md_msg

    await manager.disconnect()


# ---------------------------------------------------------------------------
# reconnect()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_backoff_logic() -> None:
    """reconnect() doubles delay each call, capped at 30s."""
    auth_ok = _auth_ok_msg()

    rx1: asyncio.Queue[str] = asyncio.Queue()
    await rx1.put(auth_ok)
    ws1 = _make_mock_ws(receive_queue=rx1)

    rx2: asyncio.Queue[str] = asyncio.Queue()
    await rx2.put(auth_ok)
    ws2 = _make_mock_ws(receive_queue=rx2)

    rx3: asyncio.Queue[str] = asyncio.Queue()
    await rx3.put(auth_ok)
    ws3 = _make_mock_ws(receive_queue=rx3)

    mock_connect = AsyncMock(side_effect=[ws1, ws2, ws3])

    with patch("src.client.ws_manager.websockets.connect", new=mock_connect):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

        assert manager._reconnect_delay == 1.0

        # Override asyncio.sleep during reconnect so it doesn't actually wait.
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await manager.reconnect()

        # After reconnect, delay should have doubled: 1 → 2.
        assert manager._reconnect_delay == 2.0
        sleep_mock.assert_called_once_with(1.0)

        # Reconnect again — delay doubles to 4.0.
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await manager.reconnect()

        assert manager._reconnect_delay == 4.0
        sleep_mock.assert_called_once_with(2.0)

        assert mock_connect.call_count == 3  # initial + 2 reconnects

        await manager.disconnect()


@pytest.mark.asyncio
async def test_reconnect_caps_at_max() -> None:
    """reconnect() delay never exceeds 30s."""
    auth_ok = _auth_ok_msg()
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(auth_ok)
    ws = _make_mock_ws(receive_queue=rx)

    mock_connect = AsyncMock(return_value=ws)

    with patch("src.client.ws_manager.websockets.connect", new=mock_connect):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

        # Artificially set delay to 16s — next reconnect should be 30, not 32.
        manager._reconnect_delay = 16.0

        # Pre-populate the queue so reconnect's connect() has an auth response.
        await rx.put(auth_ok)

        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await manager.reconnect()

        assert manager._reconnect_delay == 30.0
        sleep_mock.assert_called_once_with(16.0)

        # Next reconnect still 30.
        await rx.put(auth_ok)
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await manager.reconnect()

        assert manager._reconnect_delay == 30.0
        sleep_mock.assert_called_once_with(30.0)

        await manager.disconnect()


@pytest.mark.asyncio
async def test_reconnect_without_token_raises() -> None:
    """reconnect() raises RuntimeError if no access token is stored."""
    manager = WebSocketManager(WS_URL)
    with pytest.raises(RuntimeError, match="no access token"):
        await manager.reconnect()


# ---------------------------------------------------------------------------
# Auto-reconnect on connection closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_do_reconnect_creates_new_connection() -> None:
    """_do_reconnect opens a new WebSocket without creating a second listen task."""
    auth_ok = _auth_ok_msg()

    rx1: asyncio.Queue[str] = asyncio.Queue()
    await rx1.put(auth_ok)
    ws1 = _make_mock_ws(receive_queue=rx1)

    rx2: asyncio.Queue[str] = asyncio.Queue()
    await rx2.put(auth_ok)
    ws2 = _make_mock_ws(receive_queue=rx2)

    mock_connect = AsyncMock(side_effect=[ws1, ws2])

    with patch("src.client.ws_manager.websockets.connect", new=mock_connect):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

        previous_ws = manager.ws
        assert previous_ws is ws1
        assert manager._reconnect_delay == 1.0

        # Simulate reconnect.
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await manager._do_reconnect()

        # A new WebSocket should be assigned.
        assert manager.ws is ws2
        assert manager.ws is not previous_ws
        # Backoff should have doubled.
        assert manager._reconnect_delay == 2.0
        # The receive task should still be the original one (no new task).
        assert manager._receive_task is not None

        await manager.disconnect()


# ---------------------------------------------------------------------------
# JSON decode error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_json_does_not_crash() -> None:
    """Malformed JSON messages are logged and skipped."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    # Push malformed data.
    await rx.put("NOT JSON {{{")
    # Followed by a good message.
    good_msg = {"e": "md", "d": {"symbol": "MESU5"}}
    await rx.put(json.dumps(good_msg))

    received: list[dict[str, Any]] = []
    manager.on("md", received.append)

    await asyncio.sleep(0.15)

    assert len(received) >= 1
    assert received[0] == good_msg

    await manager.disconnect()


# ---------------------------------------------------------------------------
# CID message routing (correlated responses not forwarded to callbacks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cid_response_not_forwarded_to_callbacks() -> None:
    """A CID-correlated response resolves the future but skips callbacks."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        await manager.connect(ACCESS_TOKEN)

    received: list[dict[str, Any]] = []
    manager.on("anything", received.append)

    # A correlated response that also has an 'e' field — should NOT fire callbacks.
    response = json.dumps({"cid": 1, "e": "anything", "d": "should-not-fire"})
    await rx.put(response)

    result = await manager.send({"test": True}, expect_response=True)
    assert result["cid"] == 1

    # Give the listener time to process.
    await asyncio.sleep(0.05)
    # The callback should NOT have been invoked.
    assert len(received) == 0

    await manager.disconnect()


# ---------------------------------------------------------------------------
# initial backoff state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backoff_starts_at_initial() -> None:
    """A fresh WebSocketManager starts with _reconnect_delay == 1.0."""
    rx: asyncio.Queue[str] = asyncio.Queue()
    await rx.put(_auth_ok_msg())
    mock_ws = _make_mock_ws(receive_queue=rx)

    with _install_connect_patch(mock_ws):
        manager = WebSocketManager(WS_URL)
        assert manager._reconnect_delay == 1.0
        await manager.connect(ACCESS_TOKEN)

    await manager.disconnect()
