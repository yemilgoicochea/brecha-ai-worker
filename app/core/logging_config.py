"""Logging configuration."""

import logging
import sys

import colorlog

from app.core.config import settings

LOG_COLORS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

LOG_FORMAT = (
    "%(log_color)s%(asctime)s%(reset)s "
    "%(blue)s[%(filename)s:%(lineno)d]%(reset)s "
    "%(log_color)s%(levelname)-8s%(reset)s "
    "%(white)s%(name)s%(reset)s "
    "- %(message_log_color)s%(message)s%(reset)s"
)

SECONDARY_COLORS = {
    "message": {
        "DEBUG":    "cyan",
        "INFO":     "white",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    }
}


def setup_logging():
    """Configure colored logging for the application."""
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            LOG_FORMAT,
            datefmt="%H:%M:%S",
            log_colors=LOG_COLORS,
            secondary_log_colors=SECONDARY_COLORS,
            reset=True,
        )
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL))
    root.handlers = []
    root.addHandler(handler)

    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured for {settings.APP_NAME} ({settings.ENVIRONMENT})")

    return logger
