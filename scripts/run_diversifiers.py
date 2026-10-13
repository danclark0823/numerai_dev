"""Nested chronological research slate. Usage: python scripts/run_diversifiers.py CONFIG.

Writes a new research directory, never production artifacts. Row sampling makes
the first experiments affordable; these results are not full-universe diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits

from numerai_dev.data import data_path, feature_set, iter_features, load_index
from numerai_dev.diversifiers import (
    FactorizationMachine,
    StablePortfolio,
    consensus,
    crossfit_consensus,
    era_gaussian,
    residual_labels,
    teacher_predictions,
)
from numerai_dev.evaluation import (
    BENCHMARK_COL,
    META_COL,
    evaluate,
    flatten,
    per_era_contribution,
    per_era_corr,
    prediction_correlation,
)
from numerai_dev.experiment import ROOT, git_hash, log_run
from numerai_dev.splits import embargo_for


def sample_split(version, split, target, features, last_era, rows_per_era, seed):
    idx = load_index(version, split, [target])
    eligible = idx[target].notna() & (idx.era.astype(int) <= last_era)
    if split == "validation" and "data_type" in idx:
        eligible &= idx.data_type.eq("validation")
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(idx), dtype=bool)
    for positions in idx.assign(position=np.arange(len(idx))).loc[eligible].groupby("era"):
        pos = positions[1].position.to_numpy()
        mask[rng.choice(pos, min(rows_per_era, len(pos)), replace=False)] = True
    x = np.empty((int(mask.sum()), len(features)), dtype=np.int8)
    cursor = 0
    for _, batch in iter_features(version, split, features, mask):
        x[cursor:cursor + len(batch)] = batch
        cursor += len(batch)
    assert cursor == len(x)
    return idx.loc[mask, ["era", target]], x


def run(cfg):
    started = time.time()
    version, target = cfg["data"]["version"], cfg["data"]["target"]
    gap = embargo_for(target)
    validation = load_index(version, "validation", [target])
    resolved = validation.loc[validation[target].notna()
                              & validation.data_type.eq("validation"), "era"].astype(int)
    holdout_start = sorted(resolved.unique())[-cfg["holdout_eras"]]
    windows = [(start, start + cfg["test_eras"] - 1) for start in cfg["test_starts"]]
    if not windows or cfg["test_eras"] < 1:
        raise ValueError("At least one nonempty evaluation window is required")
    if any(end >= holdout_start - gap for _, end in windows):
        raise ValueError("Evaluation touches the reserved holdout or its embargo")
    if any(b[0] <= a[1] for a, b in pairwise(windows)):
        raise ValueError("Evaluation windows must be ordered and disjoint")
    if cfg["data"]["rows_per_era"] < 20 or cfg["data"]["train_stride"] < 1:
        raise ValueError("Need at least 20 rows per era and a positive stride")
    features = feature_set(version, cfg["data"]["feature_set"])
    limit = cfg["data"].get("feature_limit", len(features))
    if limit < 1:
        raise ValueError("feature_limit must be positive")
    rng = np.random.default_rng(cfg["seed"])
    features = [features[i] for i in sorted(rng.choice(len(features), min(limit, len(features)),
                                                      replace=False))]
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"config": cfg, "features": features, "git": git_hash(),
                "holdout_start": int(holdout_start), "embargo": gap,
                "status": "running", "sampled_research_only": True}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    frames, arrays = [], []
    for split in ("train", "validation"):
        frame, x = sample_split(version, split, target, features, windows[-1][1],
                                cfg["data"]["rows_per_era"], cfg["seed"])
        frames.append(frame)
        arrays.append(x)
    frame = pd.concat(frames)
    x = np.concatenate(arrays)
    del arrays, frames
    if not frame.index.is_unique:
        raise ValueError("Duplicate row IDs across train/validation")
    eras = frame.era.astype(int).to_numpy()
    y = frame[target].to_numpy(dtype=float)
    print(f"Loaded {len(y):,} rows × {len(features)} features; holdout starts {holdout_start}",
          flush=True)
    predictions, audits = [], []
    for start, end in windows:
        train_eras = np.unique(eras[eras < start - gap])[::cfg["data"]["train_stride"]]
        train = np.isin(eras, train_eras)
        test = (eras >= start) & (eras <= end)
        if not test.any() or not train.any():
            raise ValueError("Empty training or test window")
        tx, ty, te = x[train], y[train], eras[train]
        print(f"Window {start}–{end}: cross-fitting on {len(ty):,} historical rows", flush=True)
        oof, inner = crossfit_consensus(tx, ty, te, gap=gap, **cfg["crossfit"])
        valid = np.isfinite(oof)
        if not valid.any():
            raise ValueError("No out-of-fold rows: reduce inner warm-up or expand history")
        residual = residual_labels(ty[valid], oof[valid], te[valid])
        sx, se = tx[valid], te[valid]
        portfolio = StablePortfolio(**cfg["portfolio"]).fit(sx, residual, se, oof[valid])
        fm = FactorizationMachine(seed=cfg["seed"], **cfg["factorization_machine"]).fit(sx,
                                                                                        residual)
        anti = LGBMRegressor(random_state=cfg["seed"], n_jobs=2, verbosity=-1,
                            **cfg["anti_consensus"]).fit(sx, residual)
        teachers = teacher_predictions(tx, ty, x[test], trees=cfg["crossfit"]["trees"],
                                       seed=cfg["seed"])
        result = frame.loc[test].copy()
        for i, name in enumerate(("teacher_ridge", "teacher_extra_trees", "teacher_lgbm")):
            result[name] = teachers[:, i]
        result["consensus"] = consensus(teachers, eras[test])
        result["portfolio"] = portfolio.predict(x[test], eras[test])
        result["factorization_machine"] = fm.predict(x[test])
        result["anti_consensus"] = anti.predict(x[test])
        result["ensemble"] = np.mean([era_gaussian(result[c].to_numpy(), eras[test])
                                      for c in ("portfolio", "factorization_machine",
                                                "anti_consensus")], axis=0)
        result["window"] = start
        predictions.append(result)
        audit = {"test_start": start, "test_end": end, "train_end": int(te.max()),
                 "residual_rows": int(valid.sum()), "crossfit": inner,
                 "portfolio_features": [features[i] for i in portfolio.columns]}
        audits.append(audit)
        # Each bundle is tied to its feature order and chronological window.
        joblib.dump({"features": features, "portfolio": portfolio,
                     "factorization_machine": fm, "anti_consensus": anti, "audit": audit},
                    out / f"window_{start}.joblib")
        pd.DataFrame({"era": te[valid], "consensus": oof[valid], "residual": residual},
                     index=frame.index[train][valid]).to_parquet(out / f"oof_{start}.parquet")
        print(f"Window {start}–{end}: saved models and predictions", flush=True)
    result = pd.concat(predictions)
    columns = ["teacher_ridge", "teacher_extra_trees", "teacher_lgbm", "consensus",
               "portfolio", "factorization_machine", "anti_consensus", "ensemble"]
    for filename, column in (("meta_model.parquet", META_COL),
                              ("validation_benchmark_models.parquet", BENCHMARK_COL)):
        result = result.join(pd.read_parquet(data_path(version, filename), columns=[column]))
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL, "consensus_contribution": "consensus"}
    def score(rows):
        return {col: evaluate(rows, [col], target,
                              {name: ref for name, ref in refs.items() if ref != col})[col]
                for col in columns}

    metrics = {"all": score(result)}
    for window, rows in result.groupby("window"):
        metrics[str(window)] = score(rows)
    per_era = per_era_corr(result, columns, target).stack().rename("corr").to_frame()
    for name, ref in refs.items():
        if result[ref].notna().any():
            per_era[name] = per_era_contribution(result, [c for c in columns if c != ref],
                                               target, ref).stack()
    per_era.to_csv(out / "per_era.csv")
    prediction_correlation(result, columns).to_csv(out / "prediction_correlations.csv")
    result.to_parquet(out / "validation_preds.parquet")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest.update(status="complete", folds=audits, seconds=round(time.time() - started, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log_run(cfg["name"], cfg, flatten(metrics), started, kind="diversifier_walk_forward",
            holdout_scored=False, features=features)
    for name, scores in metrics["all"].items():
        print(name, {metric: round(stats["mean"], 5) for metric, stats in scores.items()
                     if not metric.startswith("payout_")})
    print(f"Saved research artifacts to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(yaml.safe_load(Path(args.config).read_text()))
