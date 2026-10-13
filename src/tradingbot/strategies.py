"""Strategies turn a table of daily closes into a table of target portfolio weights.

Contract: the weights on row t may only use closes up to and including day t.
The backtester applies them to returns from day t+1 onward, so there is no look-ahead.
Each row of weights sums to at most 1 (any remainder is cash).
"""

import pandas as pd


def trend_sma(closes: pd.DataFrame, symbol: str = "SPY", safe_asset: str = "IEF", sma_days: int = 200) -> pd.DataFrame:
    """Hold `symbol` while it closes above its moving average, otherwise hold `safe_asset`."""
    price = closes[symbol]
    above = price > price.rolling(sma_days).mean()
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    weights[symbol] = above.astype(float)
    if safe_asset in closes:
        weights[safe_asset] = (~above).astype(float)
    weights.iloc[: sma_days - 1] = 0.0
    return weights


def dual_momentum(
    closes: pd.DataFrame,
    risk_assets: list[str] = ("SPY", "QQQ", "EFA"),
    safe_asset: str = "IEF",
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Each month, hold the risk asset with the best trailing return if that return beats
    the safe asset's; otherwise hold the safe asset. Decisions happen on month-end closes."""
    risk_assets = list(risk_assets)
    momentum = closes.pct_change(lookback_days)
    # A day is month-end when the next trading day falls in another month. For the latest row we
    # can't see tomorrow, so use the next weekday; this keeps live runs from treating every day as month-end.
    next_day = closes.index.to_series().shift(-1)
    next_day.iloc[-1] = closes.index[-1] + pd.offsets.BDay(1)
    month_end = pd.Series(next_day.dt.month.values != closes.index.month, index=closes.index)

    weights = pd.DataFrame(float("nan"), index=closes.index, columns=closes.columns)
    for day in closes.index[month_end.values]:
        row = momentum.loc[day]
        if row[risk_assets].isna().any():
            continue
        best = row[risk_assets].idxmax()
        pick = best if row[best] > row.get(safe_asset, 0.0) else safe_asset
        weights.loc[day] = 0.0
        weights.loc[day, pick] = 1.0
    return weights.ffill().fillna(0.0)


def rsi(series: pd.Series, days: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / days, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / days, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def rsi_reversion(
    closes: pd.DataFrame,
    symbol: str = "SPY",
    rsi_days: int = 2,
    entry_below: float = 10,
    exit_above: float = 70,
    trend_sma_days: int = 200,
) -> pd.DataFrame:
    """Buy short-term dips (low RSI) only while the long-term trend is up; sell on the bounce."""
    price = closes[symbol]
    r = rsi(price, rsi_days)
    uptrend = price > price.rolling(trend_sma_days).mean()

    holding = False
    position = []
    for value, up in zip(r, uptrend):
        if not holding and up and value < entry_below:
            holding = True
        elif holding and (value > exit_above or not up):
            holding = False
        position.append(1.0 if holding else 0.0)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    weights[symbol] = position
    return weights


STRATEGIES = {
    "trend_sma": trend_sma,
    "dual_momentum": dual_momentum,
    "rsi_reversion": rsi_reversion,
}


def symbols_for(name: str, params: dict) -> list[str]:
    """Every symbol a strategy needs price data for."""
    syms = []
    for key in ("symbol", "safe_asset"):
        if key in params:
            syms.append(params[key])
    syms += list(params.get("risk_assets", []))
    return list(dict.fromkeys(syms))


def target_weights(name: str, closes: pd.DataFrame, params: dict) -> pd.DataFrame:
    return STRATEGIES[name](closes, **params)
