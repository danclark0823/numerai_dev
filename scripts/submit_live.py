"""Download the current live round, predict with each model in configs/live.yaml,
validate, and upload.

Usage:
  python scripts/submit_live.py              # predict + validate only (dry run)
  python scripts/submit_live.py --upload     # also upload to Numerai
  python scripts/submit_live.py --check-validation  # reproduce validated preds

Credentials come from .env (NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numerapi import NumerAPI

from numerai_dev.data import data_path, era_mask, feature_set, load_features, load_index
from numerai_dev.experiment import ROOT
from numerai_dev.live import load_members, predict_model, validate_submission, write_submission

SUBMISSION_LOG = ROOT / "experiments" / "submissions.jsonl"


def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    return env


def all_members(cfg: dict) -> dict:
    names = {n for spec in cfg["models"].values() for n in spec["members"]}
    return {n: load_members(n) for n in names}


def check_validation(cfg: dict) -> None:
    """Run the live code path on a few validation eras and compare with the
    validated artifacts' saved predictions (rank correlation should be ~1)."""
    version = cfg["version"]
    members = all_members(cfg)
    idx = load_index(version, "validation", [])
    eras = ["1100", "1150", "1160"]
    mask = era_mask(idx, eras=eras, data_type="validation")
    feats = feature_set(version, cfg["feature_set"])
    x_all = pd.DataFrame(load_features(version, "validation", feats, mask),
                         index=idx.index[mask], columns=feats)
    era_s = idx.loc[mask, "era"]
    saved = {"lgbm_bag4": "lgbm_all_bag4", "dc_resid_ensemble": "ens_targets_resid_neutral"}
    for model, spec in cfg["models"].items():
        x_neu = None
        if spec.get("neutralize"):
            x_neu = x_all[feature_set(version, spec["neutralize"]["feature_set"])]
        live = predict_model(spec, x_all, x_neu, era_s, members)
        ref = pd.read_parquet(ROOT / "artifacts" / saved[model] / "validation_preds.parquet")
        ref = ref.loc[ref.index.intersection(live.index)]
        corr = pd.concat([live, ref["prediction"]], axis=1, join="inner").groupby(
            era_s).apply(lambda d: d.rank().corr().iloc[0, 1])
        print(f"{model}: per-era rank corr live-path vs validated = "
              f"{corr.round(5).to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live.yaml")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--check-validation", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    version = cfg["version"]

    if args.check_validation:
        check_validation(cfg)
        return

    env = load_env()
    napi = NumerAPI(env["NUMERAI_PUBLIC_ID"], env["NUMERAI_SECRET_KEY"])
    rnd = napi.get_current_round()
    live_path = data_path(version, f"live_{rnd}.parquet")
    if not live_path.exists():
        napi.download_dataset(f"{version}/live.parquet", str(live_path))
    feats = feature_set(version, cfg["feature_set"])
    live = pd.read_parquet(live_path)
    x_all = live[feats].astype(np.int8)
    eras = pd.Series("live", index=live.index)
    print(f"round {rnd}: {len(live)} live rows")

    members = all_members(cfg)
    model_ids = napi.get_models()
    for model, spec in cfg["models"].items():
        x_neu = None
        if spec.get("neutralize"):
            x_neu = live[feature_set(version, spec["neutralize"]["feature_set"])]
        preds = predict_model(spec, x_all, x_neu, eras, members)
        validate_submission(preds, live.index)
        path = write_submission(preds, ROOT / "predictions" / str(rnd) / f"{model}.csv")
        status = "dry-run"
        if args.upload:
            if model not in model_ids:
                raise KeyError(f"model {model} not found on account")
            sub_id = napi.upload_predictions(str(path), model_id=model_ids[model])
            status = f"uploaded ({sub_id})"
        print(f"  {model}: {path.relative_to(ROOT)} {status}")
        SUBMISSION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SUBMISSION_LOG.open("a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "round": rnd, "model": model, "members": spec["members"],
                "rows": len(preds), "status": status,
            }) + "\n")


if __name__ == "__main__":
    main()
