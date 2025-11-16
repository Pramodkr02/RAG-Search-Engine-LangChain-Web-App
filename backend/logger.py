"""Simple logging configuration for the project.

Provides a configured `logging.Logger` instance for use across modules.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path("data") / "rag_app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger. Logs to both console and rotating file.

    Args:
        name: Logger name.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


root_logger = get_logger("rag_app")