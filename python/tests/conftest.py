import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
warnings.filterwarnings("ignore")

from dmx import common  # noqa: E402

SAMPLES = ROOT / "public" / "data" / "samples"
LOCAL = ROOT / ".localdata"


def load(key: str, path: Path):
    common.load_csv(key, path.read_text(encoding="utf-8"))
    return key


@pytest.fixture(scope="session")
def t01():
    return load("t01", SAMPLES / "t01_nvidia_stock_1999_2026.csv")


@pytest.fixture(scope="session")
def t02():
    return load("t02", SAMPLES / "t02_llm_agent_failure_benchmark.csv")


@pytest.fixture(scope="session")
def t03():
    return load("t03", SAMPLES / "t03_chip_companies_financials.csv")


@pytest.fixture(scope="session")
def aapl():
    p = LOCAL / "AAPL.csv"
    if not p.exists():
        pytest.skip("Data NASDAQ (lisensi © pemilik) tidak didistribusikan; letakkan AAPL.csv di .localdata/")
    return load("aapl", p)


@pytest.fixture(scope="session")
def t05():
    return load("t05", SAMPLES / "t05_trending_topics_2026_synthetic.csv")


@pytest.fixture(scope="session")
def t06():
    return load("t06", SAMPLES / "t06_indonesia_multiasset_macro_2010_2026.csv")


@pytest.fixture(scope="session")
def t07():
    return load("t07", SAMPLES / "t07_instagram_reels_knn_clean.csv")


@pytest.fixture(scope="session")
def t08():
    return load("t08", SAMPLES / "t08_aapl_2026_stock_data.csv")
