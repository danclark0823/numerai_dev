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
import numpy as np
import pandas as pd
import yaml

from numerai_dev.data import (
    data_path,
    era_mask,
    feature_set,
    iter_features,
    load_features,
    load_index,
)
from numerai_dev.evaluation import BENCHMARK_COL, evaluate, flatten, prediction_correlation
from numerai_dev.experiment import ROOT, log_run
from numerai_dev.splits import make_split
from numerai_dev.targets import residualize


def peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9


def train_lgbm(version: str, features: list[str], mask: np.ndarray, y: np.ndarray,
               model_cfg: dict) -> lgb.Booster:
    x = load_features(version, "train", features, mask)
    print(f"train: {x.shape}, {x.nbytes / 1e9:.1f} GB raw", flush=True)
    params = {"objective": "regression", **model_cfg["params"]}
    num_rounds = params.pop("n_estimators")
    # Bin into LightGBM's compact format, then free the raw int8 matrix before training.
    ds = lgb.Dataset(x, label=y, free_raw_data=True).construct()
    del x, y
    gc.collect()
    model = lgb.train(params, ds, num_boost_round=num_rounds)
    del ds
    gc.collect()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--holdout", action="store_true", help="also score the locked holdout")
    args = parser.parse_args()

    started = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]
    version, target = d["version"], d["target"]
    features = feature_set(version, d["feature_set"])

    if cfg["model"]["type"] != "lgbm":
        raise ValueError(f"unknown model type {cfg['model']['type']}")

    # --- training label: another target, optionally residualized against a benchmark ---
    train_target = d.get("train_target", target)
    tidx = load_index(version, "train", [train_target])
    label = train_target
    if ref := d.get("residualize_against"):
        bench = pd.read_parquet(data_path(version, "train_benchmark_models.parquet"),
                                columns=[ref])
        tidx = tidx.join(bench)
        has = tidx[[train_target, ref]].notna().all(axis=1)
        tidx["label"] = np.nan
        tidx.loc[has, "label"] = residualize(tidx.loc[has], train_target, ref)
        label = "label"
        del bench

    # --- train: one model per era offset, averaged (all data at constant memory) ---
    models, train_eras = [], set()
    for offset in d.get("era_offsets", [0]):
        tmask = era_mask(tidx, era_stride=d["era_stride"], era_offset=offset, require=[label])
        train_eras |= set(tidx.loc[tmask, "era"])
        y = tidx.loc[tmask, label].to_numpy(dtype=np.float32)
        models.append(train_lgbm(version, features, tmask, y, cfg["model"]))
        print(f"offset {offset} trained at {time.time() - started:.0f}s, "
              f"peak {peak_rss_gb():.1f} GB", flush=True)
    del tidx

    # --- predict validation in batches ---
    vidx = load_index(version, "validation", [target])
    resolved = era_mask(vidx, require=[target], data_type="validation")
    split = make_split(train_eras, vidx.loc[resolved, "era"].unique().tolist(), target,
                       cfg["validation"]["holdout_eras"])
    scored_eras = split.dev + (split.holdout if args.holdout else [])
    vmask = era_mask(vidx, eras=scored_eras, require=[target], data_type="validation")
    preds = np.full(len(vidx), np.nan, dtype=np.float32)
    for pos, xb in iter_features(version, "validation", features, vmask):
        preds[pos] = np.mean([m.predict(xb) for m in models], axis=0)
    val = vidx.loc[vmask, ["era", target]].copy()
    val["prediction"] = preds[vmask]
    del vidx, preds
    gc.collect()

    meta = pd.read_parquet(data_path(version, "meta_model.parquet"))
    example = pd.read_parquet(data_path(version, "validation_example_preds.parquet"))
    bench = pd.read_parquet(data_path(version, "validation_benchmark_models.parquet"))
    val = val.join(meta[["numerai_meta_model"]], how="left")
    val = val.join(bench[[BENCHMARK_COL]], how="left")
    val = val.join(example[["prediction"]].rename(columns={"prediction": "example"}), how="left")

    pred_cols = ["prediction", "example"]
    results = {}
    for part, eras in [("dev", split.dev)] + ([("holdout", split.holdout)] if args.holdout else []):
        sub = val[val["era"].isin(eras)]
        results[part] = evaluate(sub, pred_cols, target)
        results[part]["pred_corr_with_example"] = float(
            prediction_correlation(sub, pred_cols).loc["prediction", "example"]
        )

    out_dir = ROOT / "artifacts" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(str(out_dir / f"model_{i}.txt"))
    val[["era", "prediction"]].to_parquet(out_dir / "validation_preds.parquet")

    record = log_run(
        cfg["name"], cfg, flatten(results), started,
        dev_eras=[split.dev[0], split.dev[-1], len(split.dev)],
        holdout_scored=args.holdout,
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    dev = results["dev"]["prediction"]
    print(json.dumps({k: {m: round(v[m], 4) for m in ("mean", "sharpe", "pct_positive")}
                      for k, v in dev.items()}, indent=1))
    print(f"logged {record['name']} in {record['seconds']}s, peak {record['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
