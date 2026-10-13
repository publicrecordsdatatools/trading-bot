"""Backtest every strategy in config.toml against buy-and-hold SPY.

Writes reports/backtest-<date>.md and reports/backtest-<date>.png.
Usage: python scripts/backtest.py [--refresh]
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tradingbot import backtest, data, strategies  # noqa: E402
from tradingbot.config import ROOT, load_config  # noqa: E402


def fmt(m: dict) -> list[str]:
    return [
        f"{m['total_return']:.0%}",
        f"{m['cagr']:.1%}",
        f"{m['sharpe']:.2f}",
        f"{m['max_drawdown']:.1%}",
        f"{m['volatility']:.1%}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    args = parser.parse_args()

    cfg = load_config()
    g = cfg["general"]
    bench = g["benchmark"]
    split = pd.Timestamp(g["out_of_sample_start"])

    all_syms = {bench}
    for name in strategies.STRATEGIES:
        all_syms |= set(strategies.symbols_for(name, cfg["strategy"][name]))
    closes = data.load_closes(sorted(all_syms), start=g["backtest_start"], refresh=args.refresh)

    series = {f"{bench} buy & hold": backtest.buy_and_hold(closes[bench])}
    trades = {}
    for name in strategies.STRATEGIES:
        params = cfg["strategy"][name]
        weights = strategies.target_weights(name, closes, params)
        result = backtest.run(closes, weights, g["cost_per_trade"])
        series[name] = result.returns
        trades[name] = result.trades

    # Start every series once all strategies have warmed up (longest lookback).
    warmup = 260
    returns = pd.DataFrame(series).iloc[warmup:]

    rows = []
    header = "| Strategy | Period | Total | CAGR | Sharpe | Max drawdown | Volatility | Trades |"
    rows += [header, "|" + "---|" * 8]
    for label, period in (("in-sample", returns.loc[:split - pd.Timedelta(days=1)]), ("out-of-sample", returns.loc[split:])):
        for col in returns.columns:
            m = backtest.metrics(period[col])
            rows.append(f"| {col} | {label} | " + " | ".join(fmt(m)) + f" | {trades.get(col, '-')} |")

    today = date.today().isoformat()
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    chart = out / f"backtest-{today}.png"
    _plot((1 + returns).cumprod(), split, chart)

    start, end = returns.index[0].date(), returns.index[-1].date()
    report = [
        f"# Backtest {today}",
        "",
        f"Data: {start} to {end}, daily adjusted closes. Costs: {g['cost_per_trade']:.2%} per trade.",
        f"In-sample: before {split.date()}. Out-of-sample: {split.date()} onward. Trades are over the full period.",
        "",
        *rows,
        "",
        f"![equity curves]({chart.name})",
        "",
    ]
    md = out / f"backtest-{today}.md"
    md.write_text("\n".join(report))
    print("\n".join(rows))
    print(f"\nWrote {md.relative_to(ROOT)} and {chart.relative_to(ROOT)}")


def _plot(equity: pd.DataFrame, split: pd.Timestamp, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in equity.columns:
        ax.plot(equity.index, equity[col], label=col, linewidth=1.2)
    ax.axvline(split, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
