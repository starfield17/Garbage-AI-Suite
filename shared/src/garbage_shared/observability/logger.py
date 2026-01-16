"""Structured logging with correlation ID support."""

import contextlib
from typing import Any

import structlog

_loggers: dict[str, structlog.stdlib.BoundLogger] = {}


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    import logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = structlog.get_logger(name)
    return _loggers[name]


@contextlib.contextmanager
def log_context(**kwargs: Any):
    """Temporarily add context to all log messages."""
    logger = structlog.get_logger()
    try:
        with logger.contextvars.bind_context(**kwargs):
            yield
    finally:
        pass
