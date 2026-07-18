"""
Application configuration models.

Uses pydantic-settings to load Tradovate credentials and environment
settings from a ``.env`` file and/or environment variables.
"""

from __future__ import annotations

from pydantic import BaseModel as PydanticBaseModel
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


# ---------------------------------------------------------------------------
# Bot orchestrator configuration
# ---------------------------------------------------------------------------


class BotConfig(PydanticBaseModel):
    """Top-level configuration for the :class:`BotOrchestrator`.

    All values have sensible defaults so the bot can start with a
    minimal ``config.yaml``.
    """

    # -- Main loop -----------------------------------------------------------
    loop_interval: float = Field(
        default=1.0,
        ge=0.001,
        description="Seconds between main-loop iterations",
    )
    state_path: str = Field(
        default="state.json",
        description="Path to the persisted state JSON file",
    )
    lock_path: str = Field(
        default="bot.lock",
        description="PID lock file to prevent double-runs",
    )
    log_level: str = Field(
        default="INFO",
        description="Log level for structlog",
    )
    log_dir: str = Field(
        default="logs",
        description="Directory for JSON log files",
    )

    # -- Trading environment -------------------------------------------------
    environment: str = Field(
        default="demo",
        description="Tradovate environment: 'demo' or 'live'",
    )
    symbols: list[str] = Field(
        default_factory=lambda: ["MBT", "MET"],
        description="Contract symbols to trade",
    )

    # -- Stats tracking ------------------------------------------------------
    track_trade_stats: bool = Field(
        default=True,
        description="Track winning/losing trade counts",
    )
