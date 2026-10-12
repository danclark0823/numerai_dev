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
    ParquetRows,
    data_path,
    era_key,
    era_mask,
    feature_set,
    iter_features,
    load_features,
    load_index,
)
from numerai_dev.evaluation import (
    BENCHMARK_COL,
    META_COL,
    evaluate,
    flatten,
    prediction_correlation,
)
from numerai_dev.experiment import ROOT, log_run
from numerai_dev.models import RidgeModel
from numerai_dev.splits import assert_no_overlap, embargo_for, make_split
from numerai_dev.targets import residualize


def peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9


def train_lgbm(x: np.ndarray, y: np.ndarray, model_cfg: dict) -> lgb.Booster:
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


def label_frame(version: str, split: str, train_target: str, refs: list[str]) -> pd.DataFrame:
    """Index frame for a split with a `label` column: the training target, or its
    per-era residual against benchmark predictions when `refs` is given."""
    idx = load_index(version, split, [train_target])
    if not refs:
        idx["label"] = idx[train_target]
        return idx
    bench = pd.read_parquet(data_path(version, f"{split}_benchmark_models.parquet"),
                            columns=refs)
    idx = idx.join(bench)
    has = idx[[train_target, *refs]].notna().all(axis=1)
    idx["label"] = np.nan
    idx.loc[has, "label"] = residualize(idx.loc[has], train_target, refs)
    return idx


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

    if cfg["model"]["type"] not in ("lgbm", "ridge"):
        raise ValueError(f"unknown model type {cfg['model']['type']}")

    # --- holdout boundary (fixed by the validation file, independent of training) ---
    vidx = load_index(version, "validation", [target])
    resolved_eras = sorted(vidx.loc[era_mask(vidx, require=[target], data_type="validation"),
                                    "era"].unique(), key=era_key)
    holdout = resolved_eras[-cfg["validation"]["holdout_eras"]:]
    gap = embargo_for(target)
    include_val = d.get("include_validation", False)
    stream = d.get("stream", False)
    if stream and cfg["model"]["type"] != "lgbm":
        raise SystemExit("stream is only supported for lgbm")
    final = include_val == "all"  # production fit: every resolved era, nothing left to score
    if include_val and not final and not args.holdout:
        raise SystemExit("include_validation trains on dev eras; only --holdout can score it")

    # --- training rows: train eras, plus validation eras up to the holdout embargo ---
    train_target = d.get("train_target", target)
    ref = d.get("residualize_against")
    refs = [] if not ref else ([ref] if isinstance(ref, str) else list(ref))
    parts = [("train", label_frame(version, "train", train_target, refs), None)]
    if include_val:
        last_ok = 10**9 if final else era_key(holdout[0]) - gap - 1
        allowed = [e for e in resolved_eras if era_key(e) <= last_ok]
        parts.append(("validation", label_frame(version, "validation", train_target, refs),
                      allowed))

    # --- train: one model per era offset, averaged (all data at constant memory) ---
    models, train_eras = [], set()
    for offset in d.get("era_offsets", [0]):
        masks = [era_mask(f, era_stride=d["era_stride"], era_offset=offset, eras=allowed,
                          require=["label"], data_type="validation" if s == "validation" else None)
                 for s, f, allowed in parts]
        n = [int(m.sum()) for m in masks]
        ys = [frame.loc[m, "label"].to_numpy(dtype=np.float32)
              for (_, frame, _), m in zip(parts, masks, strict=True)]
        for (_, frame, _), m in zip(parts, masks, strict=True):
            train_eras |= set(frame.loc[m, "era"])
        y = np.concatenate(ys)
        if stream:
            # No raw matrix: LightGBM bins rows streamed from parquet (~half the memory).
            x = [ParquetRows(version, split_name, features, m)
                 for (split_name, _, _), m in zip(parts, masks, strict=True)]
            print(f"offset {offset}: {sum(n):,} rows streamed", flush=True)
        else:
            x = np.empty((sum(n), len(features)), dtype=np.int8)
            start = 0
            for (split_name, _, _), m, k in zip(parts, masks, n, strict=True):
                load_features(version, split_name, features, m, out=x[start:start + k])
                start += k
            print(f"offset {offset}: {x.shape}, {x.nbytes / 1e9:.1f} GB raw", flush=True)
        if cfg["model"]["type"] == "ridge":
            models.append(RidgeModel(cfg["model"]["params"]["alpha"]).fit(x, y))
        else:
            models.append(train_lgbm(x, y, cfg["model"]))
        del x, y
        gc.collect()
        print(f"offset {offset} trained at {time.time() - started:.0f}s, "
              f"peak {peak_rss_gb():.1f} GB", flush=True)
    del parts
    if final:
        out_dir = ROOT / "artifacts" / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(models):
            m.save_model(str(out_dir / f"model_{i}.txt"))
        record = log_run(cfg["name"], cfg, {}, started, kind="final_fit",
                         train_eras=[min(train_eras, key=era_key), max(train_eras, key=era_key),
                                     len(train_eras)],
                         peak_rss_gb=round(peak_rss_gb(), 2))
        print(f"final fit saved to artifacts/{cfg['name']} in {record['seconds']}s "
              f"(eras {record['train_eras']}), not scored")
        return
    assert_no_overlap(sorted(train_eras, key=era_key), holdout, gap)

    # --- predict validation in batches ---
    split = make_split(sorted(train_eras, key=era_key), resolved_eras, target,
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
    parts_to_score = [("dev", split.dev)] if split.dev else []
    if args.holdout:
        parts_to_score.append(("holdout", split.holdout))
    for part, eras in parts_to_score:
        sub = val[val["era"].isin(eras)]
        results[part] = evaluate(sub, pred_cols, target)
        results[part]["pred_corr_with_example"] = float(
            prediction_correlation(sub, pred_cols).loc["prediction", "example"]
        )
        results[part]["pred_corr_with_benchmark"] = float(
            prediction_correlation(sub, ["prediction", BENCHMARK_COL]).iloc[0, 1]
        )
        with_meta = sub.dropna(subset=[META_COL])
        if len(with_meta):
            results[part]["pred_corr_with_meta"] = float(
                prediction_correlation(with_meta, ["prediction", META_COL]).iloc[0, 1]
            )

    out_dir = ROOT / "artifacts" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        if isinstance(m, RidgeModel):
            m.save(out_dir / f"model_{i}")
        else:
            m.save_model(str(out_dir / f"model_{i}.txt"))
    val[["era", "prediction"]].to_parquet(out_dir / "validation_preds.parquet")

    record = log_run(
        cfg["name"], cfg, flatten(results), started,
        dev_eras=[split.dev[0], split.dev[-1], len(split.dev)] if split.dev else None,
        holdout_scored=args.holdout,
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    for part, res in results.items():
        print(part, json.dumps({k: {m: round(v[m], 4) for m in ("mean", "sharpe", "pct_positive")}
                                for k, v in res["prediction"].items()}, indent=1))
        print({k: round(v, 3) for k, v in res.items() if k.startswith("pred_corr")})
    print(f"logged {record['name']} in {record['seconds']}s, peak {record['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
