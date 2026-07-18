#!/usr/bin/env python3
"""
Entry point for the Tradeify evaluation trading bot.

Usage::

    python -m src.main                    # paper trading (default)
    python -m src.main --live             # live evaluation
    python -m src.main --config cfg.yaml  # custom config path

The bot always runs against ``demo.tradovate.com`` unless you pass
``--live``.  Paper trading is the default — no real orders are submitted
on the demo environment; risk and position tracking are simulated.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

from src.config import BotConfig
from src.orchestrator.bot import BotOrchestrator, BotStatus
from src.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tradeify evaluation trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                      # paper trading (demo env, default)
  python -m src.main --live               # live evaluation on real Tradovate
  python -m src.main --config prod.yaml   # custom config file
        """,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--env",
        default="demo",
        choices=["demo", "live"],
        help="Tradovate environment: 'demo' (paper trading) or 'live' (real) (default: demo)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading mode (default — the bot runs against demo.tradovate.com)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Live evaluation mode — submits real orders to live.tradovate.com",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=True,
        help="Enable all adaptive features (default)",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        default=False,
        help="Disable all Phase 2 adaptive features",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class Config:
    """Simple YAML-config loader that returns a :class:`BotConfig`.

    If the YAML file is missing or unreadable we fall back to defaults
    so the bot can still start in development / testing.
    """

    @staticmethod
    def load(path: str) -> BotConfig:
        """Load a :class:`BotConfig` from a YAML file.

        Args:
            path: Filesystem path to ``config.yaml``.

        Returns:
            A populated config object.  Falls back to defaults if the
            file cannot be read.
        """
        try:
            import yaml

            from src.config import AdaptiveConfig, RegimeConfig, TrailingStopConfig, VolatilityConfig

            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
            bot_raw = raw.get("bot", {})
            strategy_raw = raw.get("strategy", {})

            # Build BotConfig with sub-configs from strategy section.
            adaptive_raw = strategy_raw.get("adaptive", {})
            trailing_raw = strategy_raw.get("trailing", {})
            regime_raw = strategy_raw.get("regime", {})
            volatility_raw = strategy_raw.get("volatility", {})

            return BotConfig(
                **bot_raw,
                adaptive=AdaptiveConfig(**adaptive_raw) if adaptive_raw else AdaptiveConfig(),
                trailing=TrailingStopConfig(**trailing_raw) if trailing_raw else TrailingStopConfig(),
                regime=RegimeConfig(**regime_raw) if regime_raw else RegimeConfig(),
                volatility=VolatilityConfig(**volatility_raw) if volatility_raw else VolatilityConfig(),
            )
        except ImportError:
            logger.warning("config.no_yaml", path=path)
            return BotConfig()
        except FileNotFoundError:
            logger.warning("config.missing", path=path)
            return BotConfig()
        except Exception as exc:
            logger.error("config.parse_error", path=path, error=str(exc))
            return BotConfig()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


async def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, load config, run the orchestrator, print result.

    Returns:
        Exit code: 0 on ``PASSED``, 1 otherwise.
    """
    args = _parse_args(argv)

    # Resolve environment override.
    if args.live:
        args.env = "live"
        args.paper = False

    # 1. Load config.
    config = Config.load(args.config)
    config.environment = args.env
    config.log_level = args.log_level

    # Apply --no-adaptive flag.
    if args.no_adaptive:
        config.adaptive_enabled = False
        config.adaptive.enabled = False
        config.trailing.enabled = False
        logger.info("adaptive.disabled", reason="--no-adaptive flag")

    # 2. Set up structured logging.
    setup_logging(level=config.log_level, log_dir=config.log_dir)

    logger.info(
        "bot.starting",
        environment=config.environment,
        paper=args.paper,
        symbols=config.symbols,
    )

    # 3. Create and run the orchestrator.
    orchestrator = BotOrchestrator(config)
    result = await orchestrator.run()

    # 4. Report.
    if result.status == BotStatus.PASSED:
        logger.info(
            "✅ Evaluation PASSED!",
            pnl=result.total_pnl,
            trades=result.trades,
            winning=result.winning_trades,
            losing=result.losing_trades,
        )
    elif result.status == BotStatus.FAILED:
        logger.error(
            "❌ Evaluation FAILED",
            reason=result.reason,
            pnl=result.total_pnl,
            peak_equity=result.peak_equity,
            final_equity=result.final_equity,
        )
    elif result.status == BotStatus.LOCKED:
        logger.info(
            "🔒 Session locked",
            reason=result.reason,
            pnl=result.total_pnl,
        )

    return 0 if result.status == BotStatus.PASSED else 1


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def entry_point() -> None:
    """Console-script entry point (set via ``pyproject.toml``)."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
