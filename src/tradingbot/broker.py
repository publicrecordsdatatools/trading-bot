"""Thin wrapper around the Alpaca API, so the runner can be tested with a fake broker."""

import os
from dataclasses import dataclass


@dataclass
class Account:
    equity: float
    last_equity: float  # equity at the previous close
    cash: float


class AlpacaBroker:
    def __init__(self, live: bool = False):
        from alpaca.trading.client import TradingClient

        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (see .env.example)")
        self.live = live
        self.client = TradingClient(key, secret, paper=not live)

    def market_open(self) -> bool:
        return self.client.get_clock().is_open

    def account(self) -> Account:
        a = self.client.get_account()
        return Account(equity=float(a.equity), last_equity=float(a.last_equity), cash=float(a.cash))

    def positions(self) -> dict[str, int]:
        return {p.symbol: int(float(p.qty)) for p in self.client.get_all_positions()}

    def has_open_orders(self) -> bool:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return bool(self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)))

    def submit(self, symbol: str, side: str, qty: int) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        return str(order.id)

    def close_all(self) -> None:
        self.client.close_all_positions(cancel_orders=True)
