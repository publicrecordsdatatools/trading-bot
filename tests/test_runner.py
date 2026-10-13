import pandas as pd
import pytest

from tradingbot import risk, runner
from tradingbot.broker import Account


class FakeBroker:
    def __init__(self, equity=10_000.0, last_equity=10_000.0, cash=10_000.0, positions=None, is_open=True):
        self.acct = Account(equity, last_equity, cash)
        self.pos = dict(positions or {})
        self.is_open = is_open
        self.submitted = []
        self.closed_all = False

    def market_open(self):
        return self.is_open

    def has_open_orders(self):
        return False

    def account(self):
        return self.acct

    def positions(self):
        return dict(self.pos)

    def submit(self, symbol, side, qty):
        self.submitted.append((symbol, side, qty))
        self.pos[symbol] = self.pos.get(symbol, 0) + (qty if side == "buy" else -qty)
        return f"id-{len(self.submitted)}"

    def close_all(self):
        self.closed_all = True
        self.pos = {}


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "STATE_FILE", tmp_path / "risk_state.json")
    monkeypatch.setattr(runner, "TRADE_LOG", tmp_path / "trades.jsonl")
    monkeypatch.delenv("TRADING_MODE", raising=False)


@pytest.fixture
def uptrend():
    idx = pd.bdate_range("2023-01-02", periods=300)
    spy = pd.Series(range(300), index=idx, dtype=float) + 300
    ief = pd.Series(100.0, index=idx)
    return pd.DataFrame({"SPY": spy, "IEF": ief})


@pytest.fixture
def volumes(uptrend):
    return pd.DataFrame(1e8, index=uptrend.index, columns=uptrend.columns)


def test_buys_target_in_uptrend(cfg, uptrend, volumes):
    broker = FakeBroker()
    runner.run(broker, uptrend, volumes, cfg)
    assert broker.submitted == [("SPY", "buy", int(10_000 // 599))]


def test_second_run_same_day_does_nothing(cfg, uptrend, volumes):
    broker = FakeBroker()
    runner.run(broker, uptrend, volumes, cfg)
    broker.submitted.clear()
    runner.run(broker, uptrend, volumes, cfg)
    assert broker.submitted == []


def test_market_closed_does_nothing(cfg, uptrend, volumes):
    broker = FakeBroker(is_open=False)
    runner.run(broker, uptrend, volumes, cfg)
    assert broker.submitted == []


def test_sells_before_buys_on_switch(cfg, uptrend, volumes):
    down = uptrend.copy()
    down.iloc[-1, 0] = 100.0  # SPY crashes below its 200-day average
    broker = FakeBroker(cash=0.0, positions={"SPY": 20})
    broker.acct = Account(equity=2000.0, last_equity=2000.0, cash=0.0)
    runner.run(broker, down, volumes, cfg)
    assert broker.submitted[0] == ("SPY", "sell", 20)
    assert broker.submitted[1][:2] == ("IEF", "buy")


def test_daily_loss_limit_blocks_trading(cfg, uptrend, volumes):
    broker = FakeBroker(equity=9_600.0, last_equity=10_000.0)
    runner.run(broker, uptrend, volumes, cfg)
    assert broker.submitted == []


def test_kill_switch_closes_everything(cfg, uptrend, volumes):
    state = risk.RiskState(peak_equity=20_000.0)
    state.save(risk.STATE_FILE)
    broker = FakeBroker(positions={"SPY": 10})
    runner.run(broker, uptrend, volumes, cfg)
    assert broker.closed_all
    assert broker.submitted == []


def test_live_requires_config_flag(cfg, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    with pytest.raises(RuntimeError):
        runner.trading_mode(cfg)
    cfg["live"]["enabled"] = True
    assert runner.trading_mode(cfg) == "live"


def test_paper_is_default(cfg):
    assert runner.trading_mode(cfg) == "paper"


def test_small_rebalances_skipped():
    orders = runner.plan_orders({"SPY": 1.0}, {"SPY": 99}, {"SPY": 100.0}, 10_000.0, 0.02)
    assert orders == []
