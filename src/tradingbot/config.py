"""Load config.toml from the repository root."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)
