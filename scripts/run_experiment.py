"""Train a model from a YAML config, score it on dev eras, and log the result.

Usage: python scripts/run_experiment.py configs/baseline.yaml [--holdout]
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from numerai_dev.data import data_path, feature_set, load_split
from numerai_dev.evaluation import BENCHMARK_COL, evaluate, flatten, prediction_correlation
from numerai_dev.experiment import ROOT, log_run
from numerai_dev.splits import make_split


def build_model(cfg: dict):
    if cfg["type"] == "lgbm":
        return lgb.LGBMRegressor(**cfg["params"])
    raise ValueError(f"unknown model type {cfg['type']}")


def peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--holdout", action="store_true", help="also score the locked holdout")
    args = parser.parse_args()

    started = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]
    target = d["target"]
    features = feature_set(d["version"], d["feature_set"])

    train = load_split(d["version"], "train", features, [target], era_stride=d["era_stride"])
    train = train.dropna(subset=[target])
    train_eras = train["era"].unique().tolist()
    print(f"train: {train.shape}, {len(train_eras)} eras, peak {peak_rss_gb():.1f} GB")

    model = build_model(cfg["model"])
    model.fit(train[features], train[target])
    del train
    gc.collect()
    print(f"trained in {time.time() - started:.0f}s, peak {peak_rss_gb():.1f} GB")

    val = load_split(d["version"], "validation", features, [target])
    val = val.dropna(subset=[target])
    split = make_split(train_eras, val["era"].unique().tolist(), target,
                       cfg["validation"]["holdout_eras"])
    scored_eras = split.dev + (split.holdout if args.holdout else [])
    val = val[val["era"].isin(scored_eras)]
    val["prediction"] = model.predict(val[features])
    val = val.drop(columns=features)
    gc.collect()

    meta = pd.read_parquet(data_path(d["version"], "meta_model.parquet"))
    example = pd.read_parquet(data_path(d["version"], "validation_example_preds.parquet"))
    bench = pd.read_parquet(data_path(d["version"], "validation_benchmark_models.parquet"))
    val = val.join(meta[["numerai_meta_model"]], how="left")
    val = val.join(bench[[BENCHMARK_COL]], how="left")
    val = val.join(example[["prediction"]].rename(columns={"prediction": "example"}), how="left")

    preds = ["prediction", "example"]
    results = {}
    for part, eras in [("dev", split.dev)] + ([("holdout", split.holdout)] if args.holdout else []):
        sub = val[val["era"].isin(eras)]
        results[part] = evaluate(sub, preds, target)
        results[part]["pred_corr_with_example"] = float(
            prediction_correlation(sub, preds).loc["prediction", "example"]
        )

    out_dir = ROOT / "artifacts" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "model.txt"))
    val[["era", "prediction"]].to_parquet(out_dir / "validation_preds.parquet")

    record = log_run(
        cfg["name"], cfg, flatten(results), started,
        dev_eras=[split.dev[0], split.dev[-1], len(split.dev)],
        holdout_scored=args.holdout,
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    print(json.dumps(results, indent=2))
    print(f"logged {record['name']} in {record['seconds']}s, peak {record['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
