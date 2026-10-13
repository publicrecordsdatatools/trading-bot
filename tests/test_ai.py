import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_runner import FakeBroker
from tradingbot import ai_analyst, risk, runner
from tradingbot.broker import Account


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "STATE_FILE", tmp_path / "risk_state.json")
    monkeypatch.setattr(runner, "TRADE_LOG", tmp_path / "trades.jsonl")
    monkeypatch.setattr(runner, "AI_TRADE_LOG", tmp_path / "ai_trades.jsonl")
    monkeypatch.setattr(runner, "AI_STATE_FILE", tmp_path / "ai_risk_state.json")
    monkeypatch.setattr(ai_analyst, "SCORES_DIR", tmp_path / "ai_scores")
    monkeypatch.setattr(ai_analyst, "today", lambda: "2026-10-14")


@pytest.fixture
def params(cfg):
    return cfg["ai"]


def scores(**by_symbol):
    """Scores in the order given, as if Claude ranked them that way."""
    return {s: {"score": v, "reason": "", "rank": i} for i, (s, v) in enumerate(by_symbol.items())}


def test_top_scores_bought_and_spy_fills_empty_slots():
    w = ai_analyst.weights_from_scores(scores(AAPL=8, NVDA=5, XOM=2, KO=-4), top_n=5, min_score=3, filler="SPY")
    assert w == pytest.approx({"AAPL": 0.2, "NVDA": 0.2, "SPY": 0.6})


def test_quiet_day_is_all_spy():
    w = ai_analyst.weights_from_scores(scores(AAPL=0, MSFT=1), top_n=5, min_score=3, filler="SPY")
    assert w == {"SPY": 1.0}


def test_only_top_n_bought_and_ties_follow_claudes_order():
    s = scores(MSFT=6, AAPL=6, NVDA=9, KO=6, XOM=6, JPM=6, V=4)
    w = ai_analyst.weights_from_scores(s, top_n=5, min_score=3, filler="SPY")
    assert list(w) == ["NVDA", "MSFT", "AAPL", "KO", "XOM"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_clean_scores_drops_unknown_clamps_and_fills_missing():
    raw = {
        "rankings": [
            {"symbol": "NVDA", "score": 14, "reason": "beat"},
            {"symbol": "ZZZZ", "score": 9, "reason": "not on the list"},
            {"symbol": "NVDA", "score": -3, "reason": "duplicate"},
            {"symbol": "KO", "score": -12, "reason": "recall"},
        ]
    }
    s = ai_analyst.clean_scores(raw, ["NVDA", "KO", "AAPL"])
    assert {k: v["score"] for k, v in s.items()} == {"NVDA": 10, "KO": -10, "AAPL": 0}
    assert s["NVDA"]["reason"] == "beat" and s["AAPL"]["reason"] == "not scored"
    assert [s[k]["rank"] for k in ("NVDA", "KO", "AAPL")] == [0, 1, 2]


def test_prompt_wraps_news_as_data_and_covers_every_symbol():
    news = {
        "AAPL": [{"time": "2026-10-14 13:00 UTC", "source": "benzinga", "headline": "Apple beats", "summary": "x" * 500}],
        "KO": [],
    }
    prompt = ai_analyst.build_prompt(news, {"AAPL": (0.012, -0.03)}, "2026-10-14")
    assert prompt.index("<news>") < prompt.index("Apple beats") < prompt.index("</news>")
    assert "## AAPL (latest day +1.2%, last 5 days -3.0%)" in prompt
    assert "## KO\nNo news in this window." in prompt
    assert "x" * ai_analyst.SUMMARY_CHARS in prompt and "x" * (ai_analyst.SUMMARY_CHARS + 1) not in prompt


class FakeClaude:
    """Stands in for anthropic.Anthropic(): records the request and returns a canned message."""

    def __init__(self, answer: dict, stop_reason="end_turn"):
        self.requests = []
        self.message = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=json.dumps(answer))],
            usage=SimpleNamespace(input_tokens=30_000, output_tokens=5_000),
        )
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **request):
        self.requests.append(request)
        message = self.message

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return message

        return Stream()


def test_ask_claude_sends_schema_and_parses_answer(params):
    answer = {"rankings": [{"symbol": "AAPL", "score": 4, "reason": "beat"}], "market_note": "calm"}
    client = FakeClaude(answer)
    raw, usage = ai_analyst.ask_claude("prompt", ["AAPL", "KO"], params, client)
    assert raw == answer and usage == {"input_tokens": 30_000, "output_tokens": 5_000}
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["effort"] == "medium"
    schema = req["output_config"]["format"]["schema"]
    assert schema["properties"]["rankings"]["items"]["properties"]["symbol"]["enum"] == ["AAPL", "KO"]


def test_ask_claude_refuses_to_trade_on_a_cut_off_answer(params):
    with pytest.raises(RuntimeError, match="max_tokens"):
        ai_analyst.ask_claude("prompt", ["AAPL"], params, FakeClaude({}, stop_reason="max_tokens"))


@pytest.fixture
def watchlist_closes(params):
    syms = ai_analyst.symbols_for(params)
    idx = pd.bdate_range("2025-01-02", periods=60)
    rng = np.random.default_rng(1)
    data = {s: 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))) for s in syms}
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def fake_ai(monkeypatch):
    """Fake news feed and Claude. AAPL and NVDA look bullish, XOM bearish, the rest quiet."""
    calls = {"news": 0, "claude": 0}

    def fetch_news(symbols, *args):
        calls["news"] += 1
        return {s: [{"time": "t", "source": "s", "headline": f"{s} headline", "summary": ""}] for s in symbols}

    def ask_claude(prompt, symbols, params, client=None):
        calls["claude"] += 1
        ranked = [{"symbol": "AAPL", "score": 8, "reason": "beat"}, {"symbol": "NVDA", "score": 5, "reason": "deal"}]
        ranked += [{"symbol": s, "score": 0, "reason": "quiet"} for s in symbols if s not in ("AAPL", "NVDA", "XOM")]
        ranked += [{"symbol": "XOM", "score": -4, "reason": "oil slump"}]
        return {"rankings": ranked, "market_note": "calm"}, {"input_tokens": 30_000, "output_tokens": 5_000}

    monkeypatch.setattr(ai_analyst, "fetch_news", fetch_news)
    monkeypatch.setattr(ai_analyst, "ask_claude", ask_claude)
    return calls


def test_scores_saved_once_per_day_without_article_text(params, watchlist_closes, fake_ai):
    keys = ("K", "S")
    first = ai_analyst.daily_scores(watchlist_closes, params, keys)
    second = ai_analyst.daily_scores(watchlist_closes, params, keys)
    assert fake_ai == {"news": 1, "claude": 1}, "second run the same day must not pay for another call"
    assert first == second
    saved = (ai_analyst.SCORES_DIR / "2026-10-14.json").read_text()
    assert "AAPL headline" not in saved
    assert first["cost_usd"] == pytest.approx(30_000 * 4 / 1e6 + 5_000 * 20 / 1e6)


def test_ai_account_buys_picks_and_spy_with_its_own_log_and_state(cfg, watchlist_closes, fake_ai):
    volumes = pd.DataFrame(1e8, index=watchlist_closes.index, columns=watchlist_closes.columns)
    broker = FakeBroker(equity=1_000.0, last_equity=1_000.0, cash=1_000.0)
    runner.run(broker, watchlist_closes, volumes, cfg, profile="ai")

    bought = {s: q * watchlist_closes[s].iloc[-1] for s, side, q in broker.submitted if side == "buy"}
    assert set(bought) == {"AAPL", "NVDA", "SPY"}
    assert bought["AAPL"] == pytest.approx(200, abs=1) and bought["SPY"] == pytest.approx(600, abs=1)
    assert runner.AI_TRADE_LOG.exists() and not runner.TRADE_LOG.exists()
    assert runner.AI_STATE_FILE.exists() and not risk.STATE_FILE.exists()


def test_market_closed_skips_before_paying_for_claude(cfg, watchlist_closes, fake_ai):
    runner.run(FakeBroker(is_open=False), watchlist_closes, None, cfg, profile="ai")
    assert fake_ai["claude"] == 0


def test_ai_failure_is_logged_and_trades_nothing(cfg, watchlist_closes, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("news feed down")

    monkeypatch.setattr(ai_analyst, "fetch_news", broken)
    broker = FakeBroker(equity=1_000.0, last_equity=1_000.0, cash=1_000.0)
    with pytest.raises(RuntimeError):
        runner.run(broker, watchlist_closes, None, cfg, profile="ai")
    assert broker.submitted == []
    assert "news feed down" in runner.AI_TRADE_LOG.read_text()


def test_ai_account_is_paper_only_and_borrows_the_low_risk_keys(cfg, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    cfg["live"]["enabled"] = True
    prof = runner.resolve_profile(cfg, "ai")
    assert prof["mode"] == "paper"
    assert prof["keys"] == ("ALPACA_SAFE_API_KEY", "ALPACA_SAFE_SECRET_KEY")


def test_watchlist_stocks_are_capped_as_single_stocks(cfg):
    etfs = set(cfg["risk"]["etfs"])
    assert not etfs & set(cfg["ai"]["watchlist"])
    assert 1 / cfg["ai"]["top_n"] <= cfg["risk"]["max_single_stock_weight"]
