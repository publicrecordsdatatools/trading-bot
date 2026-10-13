"""Vectorized daily backtester.

Weights decided at the close of day t are held from the close of day t to the close of day t+1,
so day t+1's return is earned with weights from day t. Trading costs are charged on turnover.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Result:
    equity: pd.Series  # growth of $1
    returns: pd.Series  # daily net returns
    weights: pd.DataFrame  # weights actually held each day
    trades: int


def run(closes: pd.DataFrame, weights: pd.DataFrame, cost_per_trade: float = 0.0005) -> Result:
    weights = weights.reindex(closes.index).fillna(0.0)[closes.columns]
    asset_returns = closes.pct_change().fillna(0.0)
    held = weights.shift(1).fillna(0.0)

    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    costs = turnover.shift(1).fillna(0.0) * cost_per_trade
    net = (held * asset_returns).sum(axis=1) - costs

    changes = (weights.diff().abs() > 1e-9).any(axis=1)
    return Result(equity=(1 + net).cumprod(), returns=net, weights=held, trades=int(changes.sum()))


def buy_and_hold(closes: pd.Series) -> pd.Series:
    """Daily returns of holding one asset, for the benchmark."""
    return closes.pct_change().fillna(0.0)


def metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS) if returns.std() > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    active = returns[returns != 0]
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "win_rate": (active > 0).mean() if len(active) else 0.0,
    }
