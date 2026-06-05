"""
utils.py
--------
Shared helper utilities: logging factory and directory setup.
"""

import logging
import sys
from pathlib import Path

from config import DATA_DIR, VECTORSTORE_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Return a consistently formatted logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated imports
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def ensure_dirs() -> None:
    """
    Create required data and vectorstore directories if they do not exist.
    Safe to call multiple times (idempotent).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
