"""Logging configuration for the ML platform."""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import settings

if TYPE_CHECKING:
    from loguru import Record


class JSONFormatter:
    """JSON log formatter for structured logging."""

    def __init__(self, extra_fields: dict[str, Any] | None = None):
        """Initialize JSON formatter."""
        self.extra_fields = extra_fields or {}

    def format(self, record: "Record") -> str:
        """Format log record as JSON and escape for Loguru."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record["time"].timestamp()).isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            **self.extra_fields,
        }

        if record["exception"] is not None:
            exc_type, exc_value, exc_traceback = record["exception"]
            log_entry["exception"] = {
                "type": exc_type.__name__ if exc_type else "UnknownError",
                "value": str(exc_value),
                "traceback": "".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                ),
            }

        if "extra" in record:
            log_entry.update(record["extra"])

        # Serialize to JSON
        json_str = json.dumps(log_entry)

        # CRITICAL: Escape braces so Loguru doesn't treat them as placeholders
        # Also, Loguru expects the format function to return a 'template' string.
        # By escaping { to {{, the final output will be a single {
        return json_str.replace("{", "{{").replace("}", "}}") + "\n"


def configure_logging(
    log_level: str | None = None, log_dir: Path | None = None
) -> None:
    """Configure logging for the application."""
    log_level = log_level or settings.log_level
    log_dir = log_dir or settings.log_dir

    logger.remove()

    # 1. Console Logging
    if settings.environment == "development":
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
            colorize=True,
        )
    else:
        # Production JSON Console
        proc_formatter = JSONFormatter(
            {"environment": settings.environment, "service": "rayscale-ml"}
        )
        logger.add(sys.stderr, level=log_level, format=proc_formatter.format)

    # 2. File Logging
    log_dir.mkdir(parents=True, exist_ok=True)

    file_formatter = JSONFormatter(
        {"environment": settings.environment, "service": "rayscale-ml"}
    )

    # App Log (JSON)
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        format=file_formatter.format,
    )

    # Error Log (JSON)
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="1 day",
        retention="90 days",
        compression="zip",
        format=file_formatter.format,
    )

    # Debug Log (Plain text for dev)
    if settings.environment == "development":
        logger.add(
            log_dir / "debug_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation="1 day",
            retention="7 days",
            compression="zip",
        )

    logger.info(
        f"Logging configured. Environment: {settings.environment}, Level: {log_level}"
    )


def log_exception(context: str = "", **kwargs: Any) -> None:
    """Log an exception with context."""
    exc_info = sys.exc_info()
    if exc_info[0] is not None:
        error_msg = (
            f"Exception in {context}: {exc_info[1]}" if context else str(exc_info[1])
        )
        logger.error(
            error_msg,
            exc_info=True,
            extra={
                "exception_type": exc_info[0].__name__,
                "exception_value": str(exc_info[1]),
                "traceback": traceback.format_exc(),
                **kwargs,
            },
        )


def log_metrics(metrics: dict[str, Any], prefix: str = "", **kwargs: Any) -> None:
    """Log metrics in a structured way."""
    for name, value in metrics.items():
        metric_name = f"{prefix}_{name}" if prefix else name
        logger.info(
            f"Metric: {metric_name} = {value}",
            extra={"metric_name": metric_name, "metric_value": value, **kwargs},
        )


def log_data_info(data: Any, name: str = "data", **kwargs: Any) -> None:
    """Log data information."""
    try:
        if hasattr(data, 'shape'):
            shape = data.shape
            dtype = str(data.dtype) if hasattr(data, 'dtype') else str(type(data))
            logger.info(
                f"Data info: {name}",
                extra={"data_name": name, "shape": shape, "dtype": dtype, **kwargs},
            )
        elif hasattr(data, 'count'):
            # Works for Spark or similar
            count = data.count()
            logger.info(
                f"Data info: {name}",
                extra={"data_name": name, "count": count, **kwargs},
            )
        else:
            logger.debug(
                f"Data info: {name}",
                extra={"data_name": name, "type": str(type(data)), **kwargs},
            )
    except Exception as e:
        logger.warning(f"Could not log data info for {name}: {e}")
