"""Aggressive strategy lab: paper-only experiments with 3x leveraged ETFs.

Nothing here is traded by the daily runner. These strategies exist to test, honestly, whether
higher risk buys enough extra return to be worth it. Same contract as strategies.py:
weights on day t use closes through day t only, and each row sums to at most 1 (the rest is cash).
"""

import numpy as np
import pandas as pd

from .strategies import rsi


def _period_end(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on the last trading day of each week ("W") or month ("M"). The latest row is judged
    against the next weekday, so a live run never treats every day as a rebalance day."""
    next_day = index.to_series().shift(-1)
    next_day.iloc[-1] = index[-1] + pd.offsets.BDay(1)
    if freq == "W":
        return next_day.dt.isocalendar().week.values != index.isocalendar().week.values
    return next_day.dt.month.values != index.month


def lev_trend(closes: pd.DataFrame, signal: str = "QQQ", trade: str = "TQQQ", sma_days: int = 200) -> pd.DataFrame:
    """Hold the 3x ETF while the 1x index is above its moving average; otherwise sit in cash."""
    above = closes[signal] > closes[signal].rolling(sma_days).mean()
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    weights[trade] = above.astype(float)
    return weights


def lev_rotation(
    closes: pd.DataFrame,
    pairs: dict[str, str] | None = None,
    lookback_days: int = 63,
    sma_days: int = 200,
) -> pd.DataFrame:
    """Each week, put everything in the single 3x ETF whose 1x index has the best recent return,
    as long as that index is above its moving average. Otherwise cash. `pairs` maps 3x -> 1x."""
    pairs = pairs or {"TQQQ": "QQQ", "UPRO": "SPY", "TNA": "IWM", "SOXL": "SOXX"}
    signals = list(pairs.values())
    momentum = closes[signals].pct_change(lookback_days)
    above = closes[signals] > closes[signals].rolling(sma_days).mean()
    rebalance = _period_end(closes.index, "W")

    weights = pd.DataFrame(float("nan"), index=closes.index, columns=closes.columns)
    for i in range(max(lookback_days, sma_days), len(closes)):
        if not rebalance[i]:
            continue
        day = closes.index[i]
        weights.loc[day] = 0.0
        best = momentum.loc[day].idxmax()
        if above.loc[day, best] and momentum.loc[day, best] > 0:
            lev = next(k for k, v in pairs.items() if v == best)
            weights.loc[day, lev] = 1.0
    return weights.ffill().fillna(0.0)


def lev_dip(
    closes: pd.DataFrame,
    signal: str = "QQQ",
    trade: str = "TQQQ",
    rsi_days: int = 2,
    entry_below: float = 10,
    exit_above: float = 70,
    trend_sma_days: int = 200,
) -> pd.DataFrame:
    """Buy the 3x ETF after a sharp short-term dip in the 1x index during an uptrend; sell on the bounce."""
    price = closes[signal]
    r = rsi(price, rsi_days)
    uptrend = price > price.rolling(trend_sma_days).mean()
    holding, position = False, []
    for value, up in zip(r, uptrend):
        if not holding and up and value < entry_below:
            holding = True
        elif holding and (value > exit_above or not up):
            holding = False
        position.append(1.0 if holding else 0.0)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    weights[trade] = position
    return weights


def hold(closes: pd.DataFrame, symbol: str) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    weights[symbol] = 1.0
    return weights


LAB_STRATEGIES = {
    "lev_trend": lev_trend,
    "lev_rotation": lev_rotation,
    "lev_dip": lev_dip,
}


def simulate_leveraged(prices: pd.Series, factor: float = 3.0, annual_drag: float = 0.025) -> pd.Series:
    """A synthetic daily-reset leveraged ETF built from an index's prices, for years before the
    real fund existed. `annual_drag` stands in for fees plus borrowing costs (an approximation)."""
    daily = prices.pct_change().fillna(0.0) * factor - annual_drag / 252
    daily = daily.clip(lower=-1.0)  # a fund can't lose more than everything
    return prices.iloc[0] * (1 + daily).cumprod()


def goal_odds(
    returns: pd.Series,
    target: float = 10.0,
    ruin: float = 0.5,
    horizon_days: int = 1260,
    step: int = 5,
) -> dict:
    """Over every start date (every `step` days) with a full horizon ahead: how often did the
    account reach `target` times its start within the horizon, and how often did it fall to `ruin`
    times its start at some point along the way?"""
    log_eq = np.log1p(returns.clip(lower=-0.999999).values).cumsum()
    starts = range(0, len(log_eq) - horizon_days, step)
    hit = ruined = 0
    days_to_hit = []
    for i in starts:
        base = log_eq[i]
        path = log_eq[i + 1 : i + 1 + horizon_days] - base
        crossed = np.nonzero(path >= np.log(target))[0]
        if len(crossed):
            hit += 1
            days_to_hit.append(crossed[0] + 1)
        if path.min() <= np.log(ruin):
            ruined += 1
    n = len(starts)
    return {
        "windows": n,
        "hit_target": hit / n if n else float("nan"),
        "hit_ruin": ruined / n if n else float("nan"),
        "median_years_to_target": float(np.median(days_to_hit)) / 252 if days_to_hit else float("nan"),
    }
