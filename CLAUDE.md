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
- `src/tradingbot/broker.py` – Alpaca wrapper. `runner.py` – the daily job; `--profile main|safe|lab` picks the account.
- `scripts/backtest.py` – writes `reports/backtest-<date>.md/png`.
- `src/tradingbot/lab.py` + `scripts/lab_backtest.py` – aggressive lab (3x leveraged ETFs), paper-only experiments, never traded by the runner. Includes a 1999+ crash test with a simulated 3x QQQ fund and $100→$1,000 odds. Params under `[lab]` in `config.toml`.
- `.github/workflows/` – `tests` (on push), `backtest` and `lab-backtest` (manual), `paper-trade` (weekdays ~15:45 ET).

## Current state

- Active strategy: `sector_rotation` (top 3 of 9 sector ETFs by 3-month return, rebalanced weekly; IEF when SPY is below its 200-day SMA). Switched from `trend_sma` because Sam wants more active trading.
- First real backtest (`reports/backtest-2026-10-13.md`, 2006–2026): **no strategy beat buy-and-hold SPY on return out-of-sample** (2019+: SPY 17.5% CAGR vs trend_sma 11.5%, dual_momentum 15.8%). trend_sma's edge is risk, not return: in-sample it matched SPY's return with a -22% max drawdown vs SPY's -55% (2008), and out-of-sample Sharpe was about equal (0.90 vs 0.93). Kept as the paper strategy for that reason; revisit after paper results.
- Lab results (`reports/lab-2026-10-13.md`): real 3x ETFs 2011–2026: TQQQ hold 40.9%/yr but -82% max DD; lev_trend 33.2%/yr, -56%; lev_rotation 26.4%, -71%; lev_dip 14.1%, -46%. Crash test from the March 2000 peak (simulated 3x QQQ, drag calibrated to match real TQQQ): 3x hold went to ~zero (-100%), lev_trend -95% DD, lev_dip held up best (12.0%/yr, -45%). Best odds of $100→$1,000 in 5 years: TQQQ hold 24% of windows (2011+) / 12% (2000+), with 16% / 43% of windows halving first.
- Lab paper trading (connected 2026-10-13, $1,000 fake balance): `lev_dip` (RSI(2) dips in QQQ, trades TQQQ) in a second Alpaca paper account, keys `ALPACA_LAB_API_KEY`/`ALPACA_LAB_SECRET_KEY`. Run with `python -m tradingbot.runner --profile lab`; own log `logs/lab_trades.jsonl` and state `logs/lab_risk_state.json`. Looser limits in `[lab.paper]` (20% daily loss, 60% kill switch) because 3x ETFs swing hard. Always paper; the code refuses live for the lab profile.
- Three paper accounts by risk level (Sam's request; all connected 2026-10-13, safe and lab at $1,000 fake balance): high = lab `lev_dip`; middle = main `sector_rotation`; low = safe `conservative_trend` (40% SPY in uptrends, rest IEF; 100% IEF in downtrends). Safe account keys `ALPACA_SAFE_API_KEY`/`ALPACA_SAFE_SECRET_KEY`, `--profile safe`, own log `logs/safe_trades.jsonl` and state `logs/safe_risk_state.json`, main `[risk]` limits, paper only, fractional shares (all three accounts now use them) (a $1,000 account can't afford a whole SPY share for its 40% slice). Backtest: conservative_trend 6.0%/yr, -9% max DD in-sample; 5.0%/yr, -20.5% out-of-sample (2022, when bonds fell with stocks). Half the volatility of sector_rotation (7% vs 14%) but its worst drop was not smaller. rsi_basket is the alternative if smallest drops matter most (-9% worst, ~3.5%/yr).
- Main paper account reset from $100k to $1,000 on 2026-10-13 (Sam's call, so all three accounts start equal). Cleared `peak_equity` in `logs/risk_state.json` at the same time. If an account is ever reset or resized again, clear that profile's state file too, or the kill switch reads the drop as a loss.

## Maintenance checklist ("check on the trading system")

1. Read this file, the latest `reports/`, and the tail of `logs/trades.jsonl`.
2. Summarize performance since last check vs SPY, errors, skipped trades, and whether the kill switch or daily loss limit fired.
3. Fix bugs and dependency breakage, with tests.
4. Suggest (don't silently make) strategy changes. Backtest and check out-of-sample first; log the decision below. Don't tweak parameters because of one bad week.
5. Flag anything needing the owner: failed scheduled runs, expiring keys, unusual results.

## Next steps

- Add a weekly summary report (positions, P&L vs SPY, errors) generated from the Alpaca account and trade log.
- Consider a regime check on data quality (missing bars, stale prices) before each run.

## Decision log

- 2026-10-13: Main paper account resized to $1,000 to match the other two and the planned live size.
- 2026-10-13: Safe account's first run bought only IEF because whole shares couldn't fit SPY into $400. Added fractional shares (`fractional = true`) for the safe and lab accounts; main stays whole-share. Broker rejections are now logged as `order_error` instead of crashing the run.
- 2026-10-13: Sam asked for three accounts at high, middle and low risk. Added `conservative_trend` and a third paper account (`safe`) for it. Chose it over `trend_sma` for low risk because trend_sma's risk was about the same as sector_rotation's.
- 2026-10-13: Sam chose to paper trade `lev_dip` from the lab, alongside main `sector_rotation`, in a separate paper account.
- 2026-10-13: Sam asked how to turn $100 into $1,000 with day trading and penny stocks. Told them plainly there's no reliable way and most day traders lose. They asked for aggressive strategies anyway, so built a separate paper-only lab with 3x leveraged ETF strategies. Penny stocks left out: no free historical data that includes delisted stocks, so any backtest would be badly biased. Leverage is allowed in the lab only.
- 2026-10-13: Sam asked for more active trading. Added `sector_rotation` (weekly) and `rsi_basket` (RSI(2) dips across SPY/QQQ/IWM/DIA). Backtest 2019+: sector_rotation 10.6% CAGR, -15% max DD; rsi_basket 3.7%, -9%; SPY 17.5%, -34%. More trading did not earn more. Switched paper to sector_rotation (smallest drawdown, return close to trend_sma).
- 2026-10-13: First real backtest run; kept trend_sma for its lower drawdown, accepting lower return than SPY in the 2019+ bull market.
- 2026-10-13: Initial build. Alpaca paper, US ETFs, daily bars, $1,000 planned live capital. Default strategy `trend_sma`.
