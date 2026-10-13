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


@pytest.fixture
def cfg(cfg):
    """Runner mechanics are tested with the simple SPY/IEF trend strategy, whatever config.toml has active."""
    cfg["strategy"]["active"] = "trend_sma"
    return cfg


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "STATE_FILE", tmp_path / "risk_state.json")
    monkeypatch.setattr(runner, "TRADE_LOG", tmp_path / "trades.jsonl")
    monkeypatch.setattr(runner, "LAB_TRADE_LOG", tmp_path / "lab_trades.jsonl")
    monkeypatch.setattr(runner, "LAB_STATE_FILE", tmp_path / "lab_risk_state.json")
    monkeypatch.setattr(runner, "SAFE_TRADE_LOG", tmp_path / "safe_trades.jsonl")
    monkeypatch.setattr(runner, "SAFE_STATE_FILE", tmp_path / "safe_risk_state.json")
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


def test_switch_to_sector_rotation_sells_spy_and_buys_sectors(cfg):
    import numpy as np

    cfg["strategy"]["active"] = "sector_rotation"
    sectors = cfg["strategy"]["sector_rotation"]["risk_assets"]
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(2)
    data = {s: 50 * np.cumprod(1 + rng.normal(0.001, 0.01, len(idx))) for s in sectors}
    data["SPY"] = np.linspace(300, 600, len(idx))  # market in an uptrend
    data["IEF"] = np.full(len(idx), 100.0)
    closes = pd.DataFrame(data, index=idx)
    volumes = pd.DataFrame(1e8, index=idx, columns=closes.columns)

    broker = FakeBroker(positions={"SPY": 16})
    broker.acct = Account(equity=10_000.0, last_equity=10_000.0, cash=400.0)
    runner.run(broker, closes, volumes, cfg)

    assert broker.submitted[0] == ("SPY", "sell", 16)
    bought = [s for s, side, _ in broker.submitted if side == "buy"]
    assert len(bought) == 3 and set(bought) <= set(sectors)


@pytest.fixture
def qqq_dip():
    """Nasdaq in a long uptrend, then a sharp 3-day drop that is still above the 200-day average."""
    from tradingbot import lab

    idx = pd.bdate_range("2023-01-02", periods=300)
    qqq = pd.Series(range(300), index=idx, dtype=float) + 300
    qqq.iloc[-3:] = [590.0, 570.0, 550.0]
    closes = pd.DataFrame({"QQQ": qqq})
    closes["TQQQ"] = lab.simulate_leveraged(closes["QQQ"], 3.0, 0.0)
    return closes


def test_lab_buys_tqqq_on_dip_with_its_own_log_and_state(cfg, qqq_dip):
    volumes = pd.DataFrame(1e9, index=qqq_dip.index, columns=qqq_dip.columns)
    broker = FakeBroker(equity=100_000.0, last_equity=104_000.0, cash=100_000.0)
    runner.run(broker, qqq_dip, volumes, cfg, profile="lab")

    assert [(s, side) for s, side, _ in broker.submitted] == [("TQQQ", "buy")]
    assert runner.LAB_TRADE_LOG.exists() and not runner.TRADE_LOG.exists()
    assert runner.LAB_STATE_FILE.exists() and not risk.STATE_FILE.exists()


def test_lab_never_goes_live(cfg, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    cfg["live"]["enabled"] = True
    assert runner.resolve_profile(cfg, "lab")["mode"] == "paper"
    assert runner.resolve_profile(cfg, "lab")["keys"] == ("ALPACA_LAB_API_KEY", "ALPACA_LAB_SECRET_KEY")


def test_lab_kill_switch_is_separate_from_main(cfg, qqq_dip):
    risk.RiskState(peak_equity=1e9, kill_switch=True, kill_reason="main account").save(risk.STATE_FILE)
    volumes = pd.DataFrame(1e9, index=qqq_dip.index, columns=qqq_dip.columns)
    broker = FakeBroker(equity=100_000.0, last_equity=100_000.0, cash=100_000.0)
    runner.run(broker, qqq_dip, volumes, cfg, profile="lab")
    assert broker.submitted and not broker.closed_all


def test_safe_account_trades_conservative_mix_separately(cfg, uptrend, volumes):
    broker = FakeBroker()
    runner.run(broker, uptrend, volumes, cfg, profile="safe")
    bought = {s: q for s, side, q in broker.submitted if side == "buy"}
    assert bought == {"IEF": int(0.6 * 10_000 // 100), "SPY": int(0.4 * 10_000 // 599)}
    assert runner.SAFE_TRADE_LOG.exists() and not runner.TRADE_LOG.exists()
    assert runner.SAFE_STATE_FILE.exists() and not risk.STATE_FILE.exists()


def test_safe_account_never_goes_live(cfg, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    cfg["live"]["enabled"] = True
    prof = runner.resolve_profile(cfg, "safe")
    assert prof["mode"] == "paper"
    assert prof["keys"] == ("ALPACA_SAFE_API_KEY", "ALPACA_SAFE_SECRET_KEY")
