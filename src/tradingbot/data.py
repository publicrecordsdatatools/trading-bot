"""Daily price data: download from Yahoo Finance (via yfinance) and cache locally as CSV.

Prices are split- and dividend-adjusted, so returns include dividends.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from .config import ROOT

CACHE_DIR = ROOT / "data"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.csv"


def _download(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, start=start, auto_adjust=True, progress=False, multi_level_index=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_bars(symbol: str, start: str = "2000-01-01", refresh: bool = False) -> pd.DataFrame:
    """Return daily OHLCV bars for one symbol, using the cache when it is current."""
    path = _cache_path(symbol)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        if df.index[-1].date() >= _last_expected_close():
            return df.loc[start:]
    df = _download(symbol, "2000-01-01")
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(path)
    return df.loc[start:]


def load_closes(symbols: list[str], start: str = "2000-01-01", refresh: bool = False) -> pd.DataFrame:
    """Adjusted closes, one column per symbol, only on dates where every symbol traded."""
    closes = {s: load_bars(s, start, refresh)["Close"] for s in symbols}
    return pd.DataFrame(closes).dropna()


def load_volumes(symbols: list[str], start: str = "2000-01-01") -> pd.DataFrame:
    return pd.DataFrame({s: load_bars(s, start)["Volume"] for s in symbols}).dropna()


def _last_expected_close() -> date:
    """Most recent weekday before today. Holidays just cause one extra download."""
    d = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    return d.date()
