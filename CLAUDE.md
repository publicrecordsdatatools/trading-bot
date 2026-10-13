# Notes for Claude Code

This repo is a long-running project. Read this file first in every session and keep it current.

## Owner's goal and ground rules

- Goal: a low-effort system with a realistic chance of growing a small account. Not guaranteed income.
- Always compare against buy-and-hold SPY on risk-adjusted terms after costs. If a strategy doesn't beat it, say so plainly.
- Prefer simple, robust strategies. Push back on overfitting, leverage, options, crypto, or anything that looks like gambling.
- Ask before adding any paid service or data source.
- Never commit API keys. Keys live in `.env` locally and in GitHub Actions secrets.

## Live trading gate

Do **not** enable live trading until both are true:
1. The strategy has paper-traded for at least 2–3 months with results roughly matching the backtest.
2. The owner writes an explicit approval like "approve live trading with $X".

Then: set `live.enabled = true` and `live.max_capital` in `config.toml`, add `TRADING_MODE=live` and live keys as secrets, keep all risk rules, and keep paper trading running alongside for comparison.

## Layout

- `config.toml` – strategy params and risk limits (single source of truth).
- `src/tradingbot/data.py` – yfinance daily bars, cached to `data/` (git-ignored).
- `src/tradingbot/strategies.py` – closes → target weights. Weights on day t use data through t only.
- `src/tradingbot/backtest.py` – weights at t earn returns on t+1; costs on turnover.
- `src/tradingbot/risk.py` – hard limits, kill switch state in `logs/risk_state.json`.
- `src/tradingbot/broker.py` – Alpaca wrapper. `runner.py` – the daily job.
- `scripts/backtest.py` – writes `reports/backtest-<date>.md/png`.
- `.github/workflows/` – `tests` (on push), `backtest` (manual), `paper-trade` (weekdays ~15:45 ET).

## Current state

- Active strategy: `trend_sma` (SPY above 200-day SMA, else IEF). Chosen as the simplest default, not yet on evidence.
- Backtests have **not** been run on real data yet: the cloud session that built this couldn't reach market data hosts. First job is to run `python scripts/backtest.py` (locally or via the `backtest` workflow) and pick the active strategy from the out-of-sample results.
- Paper trading starts once Alpaca paper keys are added as repo secrets.

## Maintenance checklist ("check on the trading system")

1. Read this file, the latest `reports/`, and the tail of `logs/trades.jsonl`.
2. Summarize performance since last check vs SPY, errors, skipped trades, and whether the kill switch or daily loss limit fired.
3. Fix bugs and dependency breakage, with tests.
4. Suggest (don't silently make) strategy changes. Backtest and check out-of-sample first; log the decision below. Don't tweak parameters because of one bad week.
5. Flag anything needing the owner: failed scheduled runs, expiring keys, unusual results.

## Next steps

- Run real backtests and choose the active strategy.
- Add a weekly summary report (positions, P&L vs SPY, errors) generated from the Alpaca account and trade log.
- Consider a regime check on data quality (missing bars, stale prices) before each run.

## Decision log

- 2026-10-13: Initial build. Alpaca paper, US ETFs, daily bars, $1,000 planned live capital. Default strategy `trend_sma`.
