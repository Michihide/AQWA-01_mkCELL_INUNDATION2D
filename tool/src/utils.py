"""ロギングと計測の共通ユーティリティ。"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "meshgen"


def setup_logging(output_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


@contextmanager
def stage(name: str):
    """処理段階の所要時間をログに出す。"""
    logger = get_logger()
    logger.info("%s ...", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s 完了 (%.1f s)", name, time.perf_counter() - t0)


def format_count(n: int) -> str:
    return f"{n:,}"
