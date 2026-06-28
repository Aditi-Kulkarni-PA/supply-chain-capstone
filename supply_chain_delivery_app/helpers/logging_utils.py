"""Logging helpers for the supply chain delivery app."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_run_logger(app_dir: Path) -> tuple[logging.Logger, Path]:
    """Create one timestamped log file for the current app process run."""
    log_dir = app_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"delivery_chat_run_{ts}.log"

    logger = logging.getLogger("supply_chain_delivery_app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("logger.initialized path=%s", log_path)
    return logger, log_path
