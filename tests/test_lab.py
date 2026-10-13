import numpy as np
import pandas as pd
import pytest

from tradingbot import lab


@pytest.fixture
def lev_closes():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2012-01-02", periods=800)
    base = {s: 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, len(idx))) for s in ["QQQ", "SPY", "IWM", "SOXX"]}
    df = pd.DataFrame(base, index=idx)
    for lev, one in {"TQQQ": "QQQ", "UPRO": "SPY", "TNA": "IWM", "SOXL": "SOXX"}.items():
        df[lev] = lab.simulate_leveraged(df[one], 3.0, 0.0)
    return df


def test_simulated_fund_moves_three_times_the_index():
    prices = pd.Series([100.0, 101.0, 99.99], index=pd.bdate_range("2024-01-01", periods=3))
    sim = lab.simulate_leveraged(prices, 3.0, annual_drag=0.0)
    assert sim.pct_change().iloc[1:].tolist() == pytest.approx([0.03, -0.03])


def test_simulated_fund_cannot_go_negative():
    prices = pd.Series([100.0, 60.0, 61.0], index=pd.bdate_range("2024-01-01", periods=3))
    sim = lab.simulate_leveraged(prices, 3.0, annual_drag=0.0)
    assert (sim >= 0).all()
    assert sim.iloc[1] == 0.0


def test_goal_odds_counts_hits_and_ruin():
    # Steady 1% a day reaches 10x in about 232 days; never falls.
    up = pd.Series(0.01, index=range(1000))
    o = lab.goal_odds(up, horizon_days=300, step=10)
    assert o["hit_target"] == 1.0 and o["hit_ruin"] == 0.0
    assert o["median_years_to_target"] == pytest.approx(232 / 252, abs=0.01)
    down = pd.Series(-0.01, index=range(1000))
    o = lab.goal_odds(down, horizon_days=300, step=10)
    assert o["hit_target"] == 0.0 and o["hit_ruin"] == 1.0


def test_lab_strategies_weights_valid(lev_closes):
    for name, fn in lab.LAB_STRATEGIES.items():
        w = fn(lev_closes)
        assert (w.sum(axis=1) <= 1.0 + 1e-9).all(), name
        assert (w >= 0).all().all(), name
        assert w.values.max() == 1.0, f"{name} never invests"


def test_lab_strategies_only_use_past_data(lev_closes):
    cut = 600
    for name, fn in lab.LAB_STRATEGIES.items():
        full = fn(lev_closes)
        trunc = fn(lev_closes.iloc[:cut])
        # The last few rows of a truncated run may differ only by the weekly rebalance-day guess.
        pd.testing.assert_frame_equal(full.iloc[: cut - 5], trunc.iloc[: cut - 5], obj=name)


def test_lev_rotation_holds_one_fund_or_cash(lev_closes):
    w = lab.lev_rotation(lev_closes)
    lev = ["TQQQ", "UPRO", "TNA", "SOXL"]
    assert set((w[lev] > 0).sum(axis=1).unique()) <= {0, 1}


def test_symbols_for_lab_strategies():
    assert lab.symbols_for({"signal": "QQQ", "trade": "TQQQ"}) == ["QQQ", "TQQQ"]
    assert set(lab.symbols_for({"pairs": {"TQQQ": "QQQ", "UPRO": "SPY"}})) == {"TQQQ", "QQQ", "UPRO", "SPY"}
