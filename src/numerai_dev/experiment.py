"""Append-only experiment log: one JSON line per run, with git hash and config."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "experiments" / "log.jsonl"


def git_hash() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True)
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return sha.strip() + ("-dirty" if dirty.strip() else "")
    except subprocess.CalledProcessError:
        return "no-commit"


def log_run(name: str, config: dict, metrics: dict, started: float, **extra) -> dict:
    record = {
        "name": name,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_hash(),
        "seconds": round(time.time() - started, 1),
        "config": config,
        "metrics": metrics,
        **extra,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
