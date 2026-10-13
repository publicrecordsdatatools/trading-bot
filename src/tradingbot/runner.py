"""Daily paper-trading job.

Meant to run once a day about 15 minutes before the US market close, so the signal uses
(nearly) the same closing price the backtest assumes. Safe to run more than once a day:
it only trades the difference between target and current positions, and does nothing
while earlier orders are still open.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone

import pandas as pd

from . import data, lab, risk, strategies
from .config import ROOT, load_config

log = logging.getLogger("tradingbot")
TRADE_LOG = ROOT / "logs" / "trades.jsonl"
LAB_TRADE_LOG = ROOT / "logs" / "lab_trades.jsonl"
LAB_STATE_FILE = ROOT / "logs" / "lab_risk_state.json"
SAFE_TRADE_LOG = ROOT / "logs" / "safe_trades.jsonl"
SAFE_STATE_FILE = ROOT / "logs" / "safe_risk_state.json"


def trading_mode(cfg: dict) -> str:
    """Live only when both the config flag and the env var say so."""
    if os.environ.get("TRADING_MODE") == "live":
        if not cfg["live"]["enabled"]:
            raise RuntimeError("TRADING_MODE=live but live.enabled is false in config.toml; refusing to trade")
        return "live"
    return "paper"


def plan_orders(
    targets: dict[str, float],
    positions: dict[str, float],
    prices: dict[str, float],
    equity: float,
    min_trade_fraction: float,
    fractional: bool = False,
) -> list[risk.Order]:
    """Orders that move current positions to target weights. Sells come first.
    Whole shares by default; with `fractional`, quantities are rounded down to 0.001 of a share
    so small accounts can hold expensive ETFs."""
    orders = []
    for sym in sorted(set(targets) | set(positions)):
        price = prices[sym]
        shares = targets.get(sym, 0.0) * equity / price
        target_qty = math.floor(shares * 1000) / 1000 if fractional else int(shares)
        diff = round(target_qty - positions.get(sym, 0), 3)
        if diff == 0:
            continue
        # Skip tiny rebalances, but always fully exit a position the strategy no longer wants.
        if abs(diff) * price < min_trade_fraction * equity and target_qty != 0:
            continue
        orders.append(risk.Order(sym, "buy" if diff > 0 else "sell", abs(diff), price))
    return sorted(orders, key=lambda o: o.side != "sell")


def _record(event: dict, path=None) -> None:
    path = path or TRADE_LOG
    path.parent.mkdir(exist_ok=True)
    event = {"time": datetime.now(timezone.utc).isoformat(), **event}
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")
    log.info(json.dumps(event))


def resolve_profile(cfg: dict, profile: str) -> dict:
    """What a run trades and where it keeps its state. Three accounts, one per risk level:
    "main" (middle) is the main strategy; "safe" (low) and "lab" (high) each have their own
    Alpaca paper account. Only "main" can ever go live."""
    if profile == "main":
        name = cfg["strategy"]["active"]
        params = cfg["strategy"][name]
        return {
            "name": name,
            "params": params,
            "weights": lambda closes: strategies.target_weights(name, closes, params),
            "symbols": strategies.symbols_for(name, params),
            "risk": cfg["risk"],
            "state_file": None,  # risk.STATE_FILE
            "log": None,  # TRADE_LOG
            "mode": trading_mode(cfg),
            "keys": ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
            "fractional": False,
        }
    if profile == "safe":
        name = cfg["safe"]["strategy"]
        params = cfg["strategy"][name]
        return {
            "name": name,
            "params": params,
            "weights": lambda closes: strategies.target_weights(name, closes, params),
            "symbols": strategies.symbols_for(name, params),
            "risk": cfg["risk"],
            "state_file": SAFE_STATE_FILE,
            "log": SAFE_TRADE_LOG,
            "mode": "paper",
            "keys": ("ALPACA_SAFE_API_KEY", "ALPACA_SAFE_SECRET_KEY"),
            "fractional": cfg["safe"].get("fractional", False),
        }
    if profile == "lab":
        lp = cfg["lab"]["paper"]
        name = lp["strategy"]
        params = cfg["lab"][name]
        rcfg = {
            **cfg["risk"],
            "daily_loss_limit": lp["daily_loss_limit"],
            "kill_switch_drawdown": lp["kill_switch_drawdown"],
            "etfs": cfg["risk"]["etfs"] + lp["extra_etfs"],
        }
        return {
            "name": name,
            "params": params,
            "weights": lambda closes: lab.LAB_STRATEGIES[name](closes, **params),
            "symbols": lab.symbols_for(params),
            "risk": rcfg,
            "state_file": LAB_STATE_FILE,
            "log": LAB_TRADE_LOG,
            "mode": "paper",
            "keys": ("ALPACA_LAB_API_KEY", "ALPACA_LAB_SECRET_KEY"),
            "fractional": lp.get("fractional", False),
        }
    raise ValueError(f"unknown profile {profile}")


def run(
    broker=None,
    closes: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    cfg: dict | None = None,
    profile: str = "main",
) -> list[dict]:
    cfg = cfg or load_config()
    prof = resolve_profile(cfg, profile)
    mode = prof["mode"]
    if broker is None:
        from .broker import AlpacaBroker

        broker = AlpacaBroker(live=mode == "live", key_env=prof["keys"][0], secret_env=prof["keys"][1])

    def record(event: dict) -> None:
        _record(event, prof["log"])

    if not broker.market_open():
        record({"event": "skip", "reason": "market closed"})
        return []
    if broker.has_open_orders():
        record({"event": "skip", "reason": "orders from an earlier run are still open"})
        return []

    rcfg = prof["risk"]
    state = risk.RiskState.load(prof["state_file"])
    account = broker.account()
    state = risk.update_drawdown(state, account.equity, rcfg)
    state.save(prof["state_file"])

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

    name = prof["name"]
    if closes is None:
        closes = data.load_closes(prof["symbols"], refresh=True)
    weights = prof["weights"](closes).iloc[-1]
    targets = risk.clamp_weights({s: float(w) for s, w in weights.items() if w > 0}, rcfg)

    positions = broker.positions()
    held_prices = {s: float(closes[s].iloc[-1]) for s in closes.columns}
    missing = set(positions) - set(held_prices)
    if missing:
        # Holding something this strategy doesn't price. Leave it alone and tell a human.
        record({"event": "warning", "reason": f"positions not managed by strategy: {sorted(missing)}"})
        positions = {s: q for s, q in positions.items() if s not in missing}

    equity = min(account.equity, cfg["live"]["max_capital"]) if mode == "live" else account.equity
    orders = plan_orders(targets, positions, held_prices, equity, rcfg["min_trade_fraction"], prof["fractional"])

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
        try:
            order_id = broker.submit(order.symbol, order.side, order.qty)
        except Exception as e:  # e.g. the broker rejects the order; log it and carry on
            record({"event": "order_error", "order": order.__dict__, "error": str(e)})
            continue
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["main", "safe", "lab"], default="main")
    run(profile=parser.parse_args().profile)


if __name__ == "__main__":
    main()
