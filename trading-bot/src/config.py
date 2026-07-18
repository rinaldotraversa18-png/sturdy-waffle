"""
Application configuration models.

Uses pydantic-settings to load Tradovate credentials and environment
settings from a ``.env`` file and/or environment variables.
"""

from __future__ import annotations

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradovateConfig(BaseSettings):
    """Configuration for the Tradovate API client.

    All secrets (username, password, app_id, device_id) are loaded from
    environment variables or a ``.env`` file — never hard-coded.

    Environment:
        TRADOVATE_USERNAME
        TRADOVATE_PASSWORD
        TRADOVATE_APP_ID
        TRADOVATE_DEVICE_ID
        TRADOVATE_ENVIRONMENT (optional, default ``"demo"``)
    """

    environment: str = "demo"
    username: str = Field(..., alias="TRADOVATE_USERNAME")
    password: str = Field(..., alias="TRADOVATE_PASSWORD")
    app_id: str = Field(..., alias="TRADOVATE_APP_ID")
    device_id: str = Field(..., alias="TRADOVATE_DEVICE_ID")

    # ------------------------------------------------------------------
    # Computed URL properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def api_base_url(self) -> str:
        """REST API base URL (e.g. ``https://demo.tradovate.com/v1``)."""
        base = "demo.tradovate.com" if self.environment == "demo" else "live.tradovate.com"
        return f"https://{base}/v1"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ws_base_url(self) -> str:
        """Trading WebSocket base URL."""
        base = "demo.tradovate.com" if self.environment == "demo" else "live.tradovate.com"
        return f"wss://{base}/v1/websocket"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def md_ws_base_url(self) -> str:
        """Market data WebSocket base URL."""
        base = "md.demo.tradovate.com" if self.environment == "demo" else "md.tradovate.com"
        return f"wss://{base}/v1/websocket"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )
