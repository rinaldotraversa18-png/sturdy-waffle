"""
Structlog configuration with JSON file rendering and coloured console output.

Call :func:`setup_logging` once at application startup.  All subsequent
``structlog.get_logger()`` calls will produce structured logs that are:

* **Console** — human-readable, colourised (via ``structlog.dev``).
* **File** — machine-parseable JSON lines (one JSON object per log event)
  written to ``<log_dir>/bot_<date>.log``.

Environment
-----------
Set ``LOG_LEVEL`` to override the default ``INFO`` level at runtime.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
) -> None:
    """Configure structlog with dual output (console + file).

    Args:
        level: Log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``).  Can be overridden by the
            ``LOG_LEVEL`` environment variable.
        log_dir: Directory for JSON log files.  Created if missing.
    """
    resolved_level = os.environ.get("LOG_LEVEL", level).upper()

    # Ensure the log directory exists.
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Build a file name with today's date so logs rotate naturally.
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_path / f"bot_{today_str}.log"

    # ------------------------------------------------------------------
    # Shared processors applied to *every* event before rendering.
    # ------------------------------------------------------------------
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # ------------------------------------------------------------------
    # Console renderer: colourised, human-readable.
    # ------------------------------------------------------------------
    console_processor = structlog.dev.ConsoleRenderer(
        colors=True,
        exception_formatter=structlog.dev.rich_traceback,
    )

    # ------------------------------------------------------------------
    # JSON renderer: one JSON line per event (good for grep / jq).
    # ------------------------------------------------------------------
    json_file = open(str(log_file), "a", encoding="utf-8")  # noqa: SIM115 — kept open intentionally

    # ------------------------------------------------------------------
    # Configure standard-library logging as a fallback / bridge.
    # ------------------------------------------------------------------
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, resolved_level, logging.INFO),
    )

    root_logger = logging.getLogger()
    root_logger.handlers = []  # clear defaults so structlog owns the pipe
    root_logger.addHandler(
        logging.StreamHandler(sys.stdout),
    )

    # Also redirect stdlib logs to the JSON file.
    file_handler = logging.StreamHandler(json_file)
    file_handler.setLevel(getattr(logging, resolved_level, logging.INFO))
    root_logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Wire structlog.
    # ------------------------------------------------------------------
    structlog.configure(
        processors=shared_processors
        + [
            # Route stdlib logs through structlog.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Also install a console formatter for stdlib logs so they look nice.
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_processor,
        foreign_pre_chain=shared_processors,
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream is sys.stdout:
                handler.setFormatter(console_formatter)
            else:
                handler.setFormatter(json_formatter)
