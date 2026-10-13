"""Backtest the aggressive lab strategies.

Two tests:
1. Real 3x ETF prices (since the funds launched around 2010), against SPY, QQQ and TQQQ buy-and-hold.
2. A crash test from 1999 using a simulated 3x QQQ fund, so the strategies live through the
   dot-com crash and 2008, which the real funds never saw.

Both report the odds of turning $100 into $1,000 within 5 years, and of falling to $50 first.
Writes reports/lab-<date>.md and reports/lab-<date>.png.
Usage: python scripts/lab_backtest.py [--refresh]
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tradingbot import backtest, data, lab  # noqa: E402
from tradingbot.config import ROOT, load_config  # noqa: E402

WARMUP = 260
HEADER = [
    "| Strategy | Total | CAGR | Sharpe | Max drawdown | $100→$1,000 in 5y | Fell to $50 | Trades |",
    "|---|---|---|---|---|---|---|---|",
]


def row(label: str, returns: pd.Series, trades) -> str:
    m = backtest.metrics(returns)
    odds = lab.goal_odds(returns)
    return (
        f"| {label} | {m['total_return']:,.0%} | {m['cagr']:.1%} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} "
        f"| {odds['hit_target']:.0%} | {odds['hit_ruin']:.0%} | {trades} |"
    )


def split_rows(label: str, returns: pd.Series, split: pd.Timestamp) -> list[str]:
    out = []
    for name, part in (("in-sample", returns.loc[: split - pd.Timedelta(days=1)]), ("out-of-sample", returns.loc[split:])):
        m = backtest.metrics(part)
        out.append(f"| {label} | {name} | {m['cagr']:.1%} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} |")
    return out


def run_set(closes: pd.DataFrame, strategies: dict, cost: float) -> tuple[pd.DataFrame, dict]:
    series, trades = {}, {}
    for label, weights in strategies.items():
        result = backtest.run(closes, weights, cost)
        series[label] = result.returns
        trades[label] = result.trades
    return pd.DataFrame(series).iloc[WARMUP:], trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    args = parser.parse_args()

    cfg = load_config()
    g, lcfg = cfg["general"], cfg["lab"]
    cost = g["cost_per_trade"]
    split = pd.Timestamp(g["out_of_sample_start"])
    p_trend, p_rot, p_dip = lcfg["lev_trend"], lcfg["lev_rotation"], lcfg["lev_dip"]

    # 1. Real leveraged ETF prices.
    pairs = p_rot["pairs"]
    syms = sorted({"SPY", "QQQ", *pairs.keys(), *pairs.values()})
    real = data.load_closes(syms, refresh=args.refresh)
    real_sets = {
        "SPY buy & hold": lab.hold(real, "SPY"),
        "QQQ buy & hold": lab.hold(real, "QQQ"),
        "TQQQ buy & hold (3x)": lab.hold(real, "TQQQ"),
        "lev_trend": lab.lev_trend(real, **p_trend),
        "lev_rotation": lab.lev_rotation(real, **p_rot),
        "lev_dip": lab.lev_dip(real, **p_dip),
    }
    real_ret, real_trades = run_set(real, real_sets, cost)

    # 2. Crash test with a simulated 3x QQQ fund.
    qqq = data.load_closes(["QQQ"], start=lcfg["crash_test_start"], refresh=args.refresh)
    sim = qqq.assign(TQQQ=lab.simulate_leveraged(qqq["QQQ"], 3.0, lcfg["leveraged_annual_drag"]))
    sim_sets = {
        "QQQ buy & hold": lab.hold(sim, "QQQ"),
        "Simulated 3x QQQ buy & hold": lab.hold(sim, "TQQQ"),
        "lev_trend (simulated)": lab.lev_trend(sim, **p_trend),
        "lev_dip (simulated)": lab.lev_dip(sim, **p_dip),
    }
    sim_ret, sim_trades = run_set(sim, sim_sets, cost)

    # How close is the simulation to the real fund where they overlap?
    overlap = real.index.intersection(sim.index)
    real_cagr = backtest.metrics(real["TQQQ"].loc[overlap].pct_change().dropna())["cagr"]
    sim_cagr = backtest.metrics(sim["TQQQ"].loc[overlap].pct_change().dropna())["cagr"]

    today = date.today().isoformat()
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    chart = out / f"lab-{today}.png"
    _plot((1 + real_ret).cumprod(), (1 + sim_ret).cumprod(), chart)

    r0, r1 = real_ret.index[0].date(), real_ret.index[-1].date()
    s0, s1 = sim_ret.index[0].date(), sim_ret.index[-1].date()
    lines = [
        f"# Aggressive lab backtest {today}",
        "",
        "Paper-only experiments. 3x ETFs move three times the index each day, both ways.",
        f"Costs: {cost:.2%} per trade. \"$100→$1,000 in 5y\" is the share of all 5-year windows (start dates every week)",
        "where the account reached 10x at some point; \"Fell to $50\" is the share where it dropped to half its start at some point.",
        "",
        f"## 1. Real leveraged ETF prices ({r0} to {r1})",
        "",
        *HEADER,
        *[row(c, real_ret[c], real_trades[c] if "hold" not in c else "-") for c in real_ret.columns],
        "",
        f"In-sample before {split.date()}, out-of-sample after:",
        "",
        "| Strategy | Period | CAGR | Sharpe | Max drawdown |",
        "|---|---|---|---|---|",
        *[r for c in real_ret.columns for r in split_rows(c, real_ret[c], split)],
        "",
        f"## 2. Crash test with simulated 3x QQQ ({s0} to {s1})",
        "",
        "Covers the 2000–2002 dot-com crash and 2008. The simulated fund uses QQQ's daily moves x3 minus "
        f"{lcfg['leveraged_annual_drag']:.1%}/yr for fees and borrowing. Where both exist, real TQQQ returned "
        f"{real_cagr:.1%}/yr and the simulation {sim_cagr:.1%}/yr.",
        "",
        *HEADER,
        *[row(c, sim_ret[c], sim_trades[c] if "hold" not in c else "-") for c in sim_ret.columns],
        "",
        f"![equity curves]({chart.name})",
        "",
    ]
    md = out / f"lab-{today}.md"
    md.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {md.relative_to(ROOT)} and {chart.relative_to(ROOT)}")


def _plot(real: pd.DataFrame, sim: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    for ax, eq, title in ((axes[0], real, "Real 3x ETFs"), (axes[1], sim, "Crash test: simulated 3x QQQ since 1999")):
        for col in eq.columns:
            ax.plot(eq.index, eq[col], label=col, linewidth=1.1)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("Growth of $1 (log scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
