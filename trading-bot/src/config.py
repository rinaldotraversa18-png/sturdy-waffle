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

    # -- Adaptive features ----------------------------------------------------
    adaptive_enabled: bool = Field(
        default=True,
        description="Enable all Phase 2 adaptive features",
    )

    # -- Adaptive sub-configs (loaded from config.yaml) ------------------------
    adaptive: AdaptiveConfig = Field(
        default_factory=lambda: AdaptiveConfig(),
        description="Adaptive parameter tuning configuration",
    )
    trailing: TrailingStopConfig = Field(
        default_factory=lambda: TrailingStopConfig(),
        description="Trailing stop configuration",
    )
    regime: RegimeConfig = Field(
        default_factory=lambda: RegimeConfig(),
        description="Market regime detection configuration",
    )
    volatility: VolatilityConfig = Field(
        default_factory=lambda: VolatilityConfig(),
        description="Volatility filter configuration",
    )


# ---------------------------------------------------------------------------
# Phase 2: Adaptive feature configuration models
# ---------------------------------------------------------------------------


class AdaptiveConfig(PydanticBaseModel):
    """Configuration for adaptive parameter tuning (Phase 2)."""

    enabled: bool = Field(
        default=True,
        description="Enable adaptive parameter tuning",
    )
    min_wins_to_adapt: int = Field(
        default=10,
        ge=1,
        description="Minimum winning trades before adapted params are used",
    )
    ema_alpha: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="EMA smoothing factor for parameter drift",
    )
    param_drift_cap_pct: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Maximum drift from defaults (±50% = 0.50)",
    )


class TrailingStopConfig(PydanticBaseModel):
    """Configuration for trailing stop management (Phase 2)."""

    enabled: bool = Field(
        default=True,
        description="Enable trailing stops",
    )
    activation_pct: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Start trailing after N% toward target",
    )
    trail_distance_ticks: int = Field(
        default=20,
        ge=1,
        description="Stop lag behind current price in ticks",
    )
    step_ticks: int = Field(
        default=5,
        ge=1,
        description="Minimum tick movement to adjust stop",
    )


class RegimeConfig(PydanticBaseModel):
    """Configuration for market regime detection (Phase 2)."""

    trending_efficiency: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Efficiency ratio above this = trending",
    )
    ranging_efficiency: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Efficiency ratio below trending, above this = ranging",
    )
    choppy_efficiency: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Efficiency ratio below this = choppy (block trading)",
    )


class VolatilityConfig(PydanticBaseModel):
    """Configuration for volatility filter (Phase 2)."""

    max_ratio: float = Field(
        default=1.50,
        gt=0.0,
        description="Current/median ATR ratio max",
    )
    atr_period: int = Field(
        default=14,
        ge=2,
        description="ATR calculation window",
    )
