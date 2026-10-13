"""Score the AI analyst's watchlist with Claude right now, without trading or saving anything.

Use it to check the setup (Alpaca news keys + ANTHROPIC_API_KEY) or to see what Claude would buy.
Costs one API call. Usage: python scripts/ai_preview.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradingbot import ai_analyst, data  # noqa: E402
from tradingbot.config import ROOT, load_config  # noqa: E402


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    params = load_config()["ai"]
    closes = data.load_closes(ai_analyst.symbols_for(params), start="2025-01-01", refresh=True)
    record = ai_analyst.score_today(closes, params, ("ALPACA_SAFE_API_KEY", "ALPACA_SAFE_SECRET_KEY"))
    weights = ai_analyst.weights_from_scores(record["scores"], params["top_n"], params["min_score"], params["filler"])

    ranked = sorted(record["scores"].items(), key=lambda kv: (-kv[1]["score"], kv[1]["rank"]))
    lines = [
        f"# AI analyst preview, {record['date']}",
        "",
        f"Model {record['model']}, cost ${record['cost_usd']:.2f} "
        f"({record['usage']['input_tokens']:,} input / {record['usage']['output_tokens']:,} output tokens).",
        "",
        f"Market: {record['market_note']}",
        "",
        "Would hold: " + ", ".join(f"{s} {w:.0%}" for s, w in weights.items()),
        "",
        "| Symbol | Score | Articles | Reason |",
        "|---|---|---|---|",
    ]
    for sym, s in ranked:
        reason = s["reason"].replace("|", "/")
        lines.append(f"| {sym} | {s['score']:+d} | {record['articles'].get(sym, 0)} | {reason} |")
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
