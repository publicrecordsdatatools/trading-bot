import pytest

from tradingbot import risk


def test_single_stock_capped_but_etf_not(cfg):
    w = risk.clamp_weights({"AAPL": 0.6, "SPY": 0.4}, cfg["risk"])
    assert w["AAPL"] == pytest.approx(0.25)
    assert w["SPY"] == pytest.approx(0.4)


def test_etf_can_be_full_account(cfg):
    assert risk.clamp_weights({"SPY": 1.0}, cfg["risk"]) == {"SPY": 1.0}


def test_no_shorting_and_no_leverage(cfg):
    w = risk.clamp_weights({"SPY": 0.9, "QQQ": 0.6, "IEF": -0.5}, cfg["risk"])
    assert w["IEF"] == 0.0
    assert sum(w.values()) == pytest.approx(1.0)


def test_kill_switch_trips_at_drawdown_and_stays_on(cfg):
    state = risk.RiskState()
    risk.update_drawdown(state, 1000, cfg["risk"])
    risk.update_drawdown(state, 900, cfg["risk"])
    assert not state.kill_switch
    risk.update_drawdown(state, 850, cfg["risk"])
    assert state.kill_switch
    risk.update_drawdown(state, 2000, cfg["risk"])
    assert state.kill_switch, "only a human may reset the kill switch"


def test_risk_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    risk.RiskState(peak_equity=123.0, kill_switch=True, kill_reason="x").save(path)
    assert risk.RiskState.load(path).kill_switch


def test_daily_loss_limit(cfg):
    assert risk.daily_loss_breached(969, 1000, cfg["risk"])
    assert not risk.daily_loss_breached(980, 1000, cfg["risk"])
    assert not risk.daily_loss_breached(1000, 0, cfg["risk"])


def test_buy_larger_than_cash_rejected(cfg):
    c = risk.check_order(risk.Order("SPY", "buy", 10, 500.0), cash=1000, avg_dollar_volume=1e10, cfg=cfg["risk"])
    assert not c.ok and "no margin" in c.reasons[0]


def test_penny_stock_rejected(cfg):
    c = risk.check_order(risk.Order("XYZ", "buy", 10, 2.0), cash=1000, avg_dollar_volume=1e10, cfg=cfg["risk"])
    assert not c.ok


def test_illiquid_rejected(cfg):
    c = risk.check_order(risk.Order("XYZ", "buy", 1, 50.0), cash=1000, avg_dollar_volume=1000, cfg=cfg["risk"])
    assert not c.ok


def test_bad_quantity_and_side_rejected(cfg):
    assert not risk.check_order(risk.Order("SPY", "buy", 0, 500.0), 1e6, 1e10, cfg["risk"]).ok
    assert not risk.check_order(risk.Order("SPY", "short", 1, 500.0), 1e6, 1e10, cfg["risk"]).ok


def test_sells_allowed_without_cash(cfg):
    assert risk.check_order(risk.Order("SPY", "sell", 5, 500.0), 0, 0, cfg["risk"]).ok
