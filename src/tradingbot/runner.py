"""Daily paper-trading job.

Meant to run once a day about 15 minutes before the US market close, so the signal uses
(nearly) the same closing price the backtest assumes. Safe to run more than once a day:
it only trades the difference between target and current positions, and does nothing
while earlier orders are still open.
"""

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from . import data, risk, strategies
from .config import ROOT, load_config

log = logging.getLogger("tradingbot")
TRADE_LOG = ROOT / "logs" / "trades.jsonl"


def trading_mode(cfg: dict) -> str:
    """Live only when both the config flag and the env var say so."""
    if os.environ.get("TRADING_MODE") == "live":
        if not cfg["live"]["enabled"]:
            raise RuntimeError("TRADING_MODE=live but live.enabled is false in config.toml; refusing to trade")
        return "live"
    return "paper"


def plan_orders(
    targets: dict[str, float],
    positions: dict[str, int],
    prices: dict[str, float],
    equity: float,
    min_trade_fraction: float,
) -> list[risk.Order]:
    """Whole-share orders that move current positions to target weights. Sells come first."""
    orders = []
    for sym in sorted(set(targets) | set(positions)):
        price = prices[sym]
        target_qty = int(targets.get(sym, 0.0) * equity // price)
        diff = target_qty - positions.get(sym, 0)
        if diff == 0:
            continue
        # Skip tiny rebalances, but always fully exit a position the strategy no longer wants.
        if abs(diff) * price < min_trade_fraction * equity and target_qty != 0:
            continue
        orders.append(risk.Order(sym, "buy" if diff > 0 else "sell", abs(diff), price))
    return sorted(orders, key=lambda o: o.side != "sell")


def record(event: dict) -> None:
    TRADE_LOG.parent.mkdir(exist_ok=True)
    event = {"time": datetime.now(timezone.utc).isoformat(), **event}
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")
    log.info(json.dumps(event))


def run(
    broker=None,
    closes: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> list[dict]:
    cfg = cfg or load_config()
    mode = trading_mode(cfg)
    if broker is None:
        from .broker import AlpacaBroker

        broker = AlpacaBroker(live=mode == "live")

    if not broker.market_open():
        record({"event": "skip", "reason": "market closed"})
        return []
    if broker.has_open_orders():
        record({"event": "skip", "reason": "orders from an earlier run are still open"})
        return []

    rcfg = cfg["risk"]
    state = risk.RiskState.load()
    account = broker.account()
    state = risk.update_drawdown(state, account.equity, rcfg)
    state.save()

    if state.kill_switch:
        if broker.positions():
            broker.close_all()
            record({"event": "kill_switch", "action": "closed all positions", "reason": state.kill_reason})
        else:
            record({"event": "skip", "reason": f"kill switch on: {state.kill_reason}"})
        return []
    if risk.daily_loss_breached(account.equity, account.last_equity, rcfg):
        record({"event": "skip", "reason": "daily loss limit reached", "equity": account.equity})
        return []

    name = cfg["strategy"]["active"]
    params = cfg["strategy"][name]
    symbols = strategies.symbols_for(name, params)
    if closes is None:
        closes = data.load_closes(symbols, refresh=True)
    weights = strategies.target_weights(name, closes, params).iloc[-1]
    targets = risk.clamp_weights({s: float(w) for s, w in weights.items() if w > 0}, rcfg)

    positions = broker.positions()
    held_prices = {s: float(closes[s].iloc[-1]) for s in closes.columns}
    missing = set(positions) - set(held_prices)
    if missing:
        # Holding something this strategy doesn't price. Leave it alone and tell a human.
        record({"event": "warning", "reason": f"positions not managed by strategy: {sorted(missing)}"})
        positions = {s: q for s, q in positions.items() if s not in missing}

    equity = min(account.equity, cfg["live"]["max_capital"]) if mode == "live" else account.equity
    orders = plan_orders(targets, positions, held_prices, equity, rcfg["min_trade_fraction"])

    placed = []
    cash = account.cash
    for order in orders:
        if order.side == "buy" and volumes is None:
            volumes = data.load_volumes(list(closes.columns))
        adv = float((volumes[order.symbol] * closes[order.symbol]).tail(20).mean()) if order.side == "buy" else 0.0
        check = risk.check_order(order, cash, adv, rcfg)
        if not check.ok:
            record({"event": "order_rejected", "order": order.__dict__, "reasons": check.reasons})
            continue
        order_id = broker.submit(order.symbol, order.side, order.qty)
        cash += order.qty * order.price * (1 if order.side == "sell" else -1)
        event = {"event": "order", "mode": mode, "id": order_id, **order.__dict__}
        record(event)
        placed.append(event)

    if not orders:
        record({"event": "no_change", "strategy": name, "targets": targets})
    return placed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    run()


if __name__ == "__main__":
    main()
