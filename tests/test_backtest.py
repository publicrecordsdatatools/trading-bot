import pandas as pd
import pytest

from tradingbot import backtest, strategies


def test_no_lookahead_weights_apply_next_day():
    closes = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    # Decide to buy at the close of day 0: earn day 1 and day 2 returns, not day 0's.
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=closes.index)
    result = backtest.run(closes, weights, cost_per_trade=0.0)
    assert result.returns.tolist() == pytest.approx([0.0, 0.10, 0.10])


def test_signal_on_last_day_earns_nothing():
    closes = pd.DataFrame({"A": [100.0, 50.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    weights = pd.DataFrame({"A": [0.0, 1.0]}, index=closes.index)
    result = backtest.run(closes, weights, cost_per_trade=0.0)
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_costs_are_charged_on_turnover():
    closes = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=closes.index)
    result = backtest.run(closes, weights, cost_per_trade=0.001)
    assert result.equity.iloc[-1] == pytest.approx(0.999)


def test_strategy_weights_never_exceed_one(closes):
    for name, params in {
        "trend_sma": {"symbol": "SPY", "safe_asset": "IEF", "sma_days": 200},
        "dual_momentum": {"risk_assets": ["SPY", "QQQ", "EFA"], "safe_asset": "IEF", "lookback_days": 252},
        "rsi_reversion": {"symbol": "SPY"},
    }.items():
        w = strategies.target_weights(name, closes, params)
        assert (w.sum(axis=1) <= 1.0 + 1e-9).all(), name
        assert (w >= 0).all().all(), name


def test_strategies_only_use_past_data(closes):
    """Changing future prices must not change today's weights."""
    params = {"risk_assets": ["SPY", "QQQ", "EFA"], "safe_asset": "IEF", "lookback_days": 252}
    full = strategies.dual_momentum(closes, **params)
    cut = 600
    truncated = strategies.dual_momentum(closes.iloc[:cut], **params)
    pd.testing.assert_frame_equal(full.iloc[: cut - 25], truncated.iloc[: cut - 25])

    for fn in (strategies.trend_sma, strategies.rsi_reversion):
        full = fn(closes)
        truncated = fn(closes.iloc[:cut])
        pd.testing.assert_frame_equal(full.iloc[:cut], truncated)


def test_metrics_drawdown():
    returns = pd.Series([0.0, 0.10, -0.50, 0.20])
    m = backtest.metrics(returns)
    assert m["max_drawdown"] == pytest.approx(-0.50)
