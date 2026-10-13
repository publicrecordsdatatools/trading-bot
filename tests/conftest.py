import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradingbot.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def closes():
    """Synthetic random-walk prices so tests never hit the network."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=900)
    cols = ["SPY", "QQQ", "EFA", "IEF"]
    rets = rng.normal(0.0004, 0.01, size=(len(idx), len(cols)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)
