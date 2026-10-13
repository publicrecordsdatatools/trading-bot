"""Hard risk rules. Every order the runner places passes through here first.

These rules are deliberately simple and conservative; tests in tests/test_risk.py cover each one.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

from .config import ROOT

STATE_FILE = ROOT / "logs" / "risk_state.json"


@dataclass
class RiskState:
    """Persisted between runs so the kill switch survives restarts."""

    peak_equity: float = 0.0
    kill_switch: bool = False
    kill_reason: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> "RiskState":
        path = path or STATE_FILE
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or STATE_FILE
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))


@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    price: float


@dataclass
class Check:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def clamp_weights(weights: dict[str, float], cfg: dict) -> dict[str, float]:
    """Cap single-stock weights, drop negatives (no shorting), and scale down so the total is at most 100%."""
    etfs = set(cfg["etfs"])
    capped = {}
    for sym, w in weights.items():
        w = max(0.0, w)
        if sym not in etfs:
            w = min(w, cfg["max_single_stock_weight"])
        capped[sym] = w
    total = sum(capped.values())
    if total > 1.0:
        capped = {s: w / total for s, w in capped.items()}
    return capped


def update_drawdown(state: RiskState, equity: float, cfg: dict) -> RiskState:
    """Track peak equity and trip the kill switch on a large drawdown. Only a human resets it."""
    state.peak_equity = max(state.peak_equity, equity)
    if state.peak_equity > 0:
        drawdown = 1 - equity / state.peak_equity
        if drawdown >= cfg["kill_switch_drawdown"] and not state.kill_switch:
            state.kill_switch = True
            state.kill_reason = f"Drawdown {drawdown:.1%} from peak {state.peak_equity:,.2f}"
    return state


def daily_loss_breached(equity: float, last_equity: float, cfg: dict) -> bool:
    """True when the account is down more than the daily limit versus yesterday's close."""
    if last_equity <= 0:
        return False
    return (equity / last_equity - 1) <= -cfg["daily_loss_limit"]


def check_order(order: Order, cash: float, avg_dollar_volume: float, cfg: dict) -> Check:
    reasons = []
    if order.side not in ("buy", "sell"):
        reasons.append(f"unknown side {order.side}")
    if order.qty <= 0:
        reasons.append("quantity must be positive")
    if order.price <= 0:
        reasons.append("price must be positive")
    elif order.side == "buy":
        if order.price < cfg["min_price"]:
            reasons.append(f"price {order.price} below minimum {cfg['min_price']}")
        if order.qty * order.price > cash:
            reasons.append(f"order value {order.qty * order.price:,.2f} exceeds cash {cash:,.2f} (no margin)")
        if avg_dollar_volume < cfg["min_avg_dollar_volume"]:
            reasons.append(f"average dollar volume {avg_dollar_volume:,.0f} too low")
    return Check(ok=not reasons, reasons=reasons)
