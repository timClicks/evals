from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def models_csv() -> Path:
    return BASE_DIR / "tables" / "models.csv"
