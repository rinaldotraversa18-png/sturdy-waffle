# Strategy engine — signal generation, position sizing, and adaptive filters

from src.strategy.adaptive_tuning import AdaptiveState, get_adapted_params
from src.strategy.engine import StrategyEngine
from src.strategy.regime import MarketRegime, detect_regime
from src.strategy.session import (
    TradingSession,
    get_current_session,
    get_session_params,
)
from src.strategy.signals import MarketSnapshot, Signal
from src.strategy.sizing import StrategyConfig, calculate_position_size
from src.strategy.trailing import (
    TrailingConfig,
    compute_trail_stop,
    should_activate_trail,
    should_update_trail,
)
from src.strategy.volatility import (
    compute_atr,
    compute_volatility_ratio,
    is_safe_to_trade,
)

__all__ = [
    # Engine
    "StrategyEngine",
    # Adaptive tuning
    "AdaptiveState",
    "get_adapted_params",
    # Regime detection
    "MarketRegime",
    "detect_regime",
    # Session awareness
    "TradingSession",
    "get_current_session",
    "get_session_params",
    # Signal structures
    "MarketSnapshot",
    "Signal",
    # Sizing
    "StrategyConfig",
    "calculate_position_size",
    # Trailing stops
    "TrailingConfig",
    "should_activate_trail",
    "compute_trail_stop",
    "should_update_trail",
    # Volatility
    "compute_atr",
    "compute_volatility_ratio",
    "is_safe_to_trade",
]
