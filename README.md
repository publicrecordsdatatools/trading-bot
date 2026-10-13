# trading-bot

A small, conservative daily trading system for US ETFs, built and maintained with Claude Code.
It runs in **paper trading** (fake money) on Alpaca. Live trading is off and needs two separate switches to turn on.

No strategy here is guaranteed to make money. Every strategy is judged against simply holding SPY.

## How it works

1. `scripts/backtest.py` downloads daily prices, runs each strategy in `config.toml`, and writes a report to `reports/` comparing them with buy-and-hold SPY, split into in-sample and out-of-sample periods.
2. `src/tradingbot/runner.py` runs once each weekday about 15 minutes before the close. It computes the active strategy's target portfolio, checks the risk rules, and places paper orders on Alpaca for the difference.
3. Every decision, order and skip is appended to `logs/trades.jsonl`.

## Strategies

| Name | Idea |
|---|---|
| `trend_sma` | Hold SPY above its 200-day average, otherwise hold IEF (bonds). |
| `dual_momentum` | Each month hold whichever of SPY, QQQ, EFA has the best 12-month return, or IEF if none beats bonds. |
| `rsi_reversion` | Buy SPY after a sharp 2-day dip while the long-term trend is up; sell on the bounce. |

The active one is `strategy.active` in `config.toml` (default `trend_sma`).

## Risk rules (enforced in `src/tradingbot/risk.py`)

- No margin, shorting, options or leverage. Orders never exceed available cash.
- Single stocks capped at 25% of the account; listed ETFs may go to 100%.
- Stop trading for the day after a 3% daily loss.
- Kill switch at a 15% drawdown from peak: sells everything and stays off until a human resets `logs/risk_state.json`.
- No stocks under $5 or with under $5M average daily dollar volume.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Alpaca paper keys
pytest                 # all tests should pass
python scripts/backtest.py
PYTHONPATH=src python -m tradingbot.runner   # one paper run (market must be open)
```

### Automatic daily runs (GitHub Actions)

Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper keys) under repo **Settings → Secrets and variables → Actions**.
The `paper-trade` workflow then runs every weekday and commits its log. The `backtest` workflow runs on demand from the Actions tab.

## Going live

Not yet. See `CLAUDE.md` for the conditions.
