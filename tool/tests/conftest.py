from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_synthetic_data import write_all  # noqa: E402
from src.utils import setup_logging  # noqa: E402

DATA_DIR = ROOT / "tests" / "data"


@pytest.fixture(scope="session", autouse=True)
def _logging():
    import logging

    setup_logging(level=logging.WARNING)


@pytest.fixture(scope="session")
def synthetic_data() -> Path:
    if not (DATA_DIR / "synthetic_domain.gpkg").exists():
        write_all(DATA_DIR)
    return DATA_DIR
