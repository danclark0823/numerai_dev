"""Research-only full-era evaluation for dc_wenek. No production writes or uploads."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from numerai_dev.data import data_path, feature_set, iter_features, load_index
from numerai_dev.diversifiers import consensus, teacher_predictions
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
from numerai_dev.live import validate_submission
from numerai_dev.splits import embargo_for
from numerai_dev.wenek import fit_window, paired_block_ci


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def production_snapshot():
    config = ROOT / "configs/live.yaml"
    cfg = yaml.safe_load(config.read_text())
    paths = [config]
    for model, spec in cfg["models"].items():
        paths.extend((ROOT / "artifacts/hosted").glob(f"{model}.pkl"))
        for member in spec["members"]:
            paths.extend((ROOT / "artifacts" / member).glob("model*"))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))}


def sample_positions(frame, last_train, stride, rows_per_era, seed):
    """Nested row samples: doubling row count retains every previous sampled ID."""
    numeric = frame.era.astype(int)
    eligible = np.unique(numeric[numeric <= last_train])[::stride]
    sample = np.zeros(len(frame), dtype=bool)
    for era, positions in frame.groupby(numeric).indices.items():
        if era in eligible:
            rng = np.random.default_rng(seed + int(era))
            selected = rng.permutation(positions)[:rows_per_era]
            sample[selected] = True
    return sample


def load_data(cfg, features, gap):
    d = cfg["data"]
    version, target = d["version"], d["target"]
    parts = []
    for split in ("train", "validation"):
        frame = load_index(version, split, [target])
        keep = frame[target].notna()
        if split == "validation":
            keep &= frame.data_type.eq("validation")
        frame = frame.loc[keep, ["era", target]].copy()
        frame["split"] = split
        parts.append(frame)
    frame = pd.concat(parts)
    if not frame.index.is_unique:
        raise ValueError("Duplicate IDs across training and validation")
    start = max(cfg["test_starts"])
    train_mask = sample_positions(frame, start - gap - 1, d["train_stride"],
                                  d["train_rows_per_era"], cfg["seed"])
    e = frame.era.astype(int).to_numpy()
    test_mask = np.zeros(len(frame), dtype=bool)
    for start in cfg["test_starts"]:
        test_mask |= (e >= start) & (e < start + cfg["test_eras"])
    if (test_mask & frame.split.ne("validation").to_numpy()).any():
        raise ValueError("Test windows must belong to validation")
    keep = train_mask | test_mask
    frame["sampled_train"] = train_mask
    frame = frame.loc[keep]
    x = np.empty((len(frame), len(features)), dtype=np.int8)
    # Fill by explicit IDs; no assumption about parquet order across files.
    for split in ("train", "validation"):
        idx = load_index(version, split, [])
        mask = idx.index.isin(frame.index[frame.split.eq(split)])
        for pos, batch in iter_features(version, split, features, mask):
            destination = frame.index.get_indexer(idx.index[pos])
            if (destination < 0).any():
                raise ValueError("Feature rows do not match selected index")
            x[destination] = batch
    return frame, x


def validate_windows(cfg):
    d = cfg["data"]
    gap = embargo_for(d["target"])
    val = load_index(d["version"], "validation", [d["target"]])
    resolved = val.loc[val[d["target"]].notna() & val.data_type.eq("validation"), "era"]
    reserve = sorted(resolved.astype(int).unique())[-cfg["holdout_eras"]]
    windows = [(s, s + cfg["test_eras"] - 1) for s in cfg["test_starts"]]
    if not windows or cfg["test_eras"] < 1 or any(e >= reserve - gap for _, e in windows):
        raise ValueError("Invalid windows or reserved holdout/embargo accessed")
    if any(b[0] <= a[1] for a, b in pairwise(windows)):
        raise ValueError("Windows must be ordered and disjoint")
    if d["train_rows_per_era"] < 20 or d["train_stride"] < 1:
        raise ValueError("Invalid sampling settings")
    return gap, int(reserve)


def make_report(result, out, target, gap):
    columns = ["ridge", "raw_fm", "matched_fm", "interaction", "blend_00", "blend_10",
               "blend_25", "teacher_consensus"]
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL,
            "ridge_contribution": "ridge", "teacher_contribution": "teacher_consensus"}

    def score(rows):
        return {c: evaluate(rows, [c], target, {k: v for k, v in refs.items() if v != c})[c]
                for c in columns}

    metrics = {"all": score(result)}
    paired, era_scores = {}, []
    for window, rows in result.groupby("window"):
        metrics[str(window)] = score(rows)
        corr = per_era_corr(rows, columns, target)
        corr = corr.loc[sorted(corr.index, key=int)]
        measures = {"corr": corr}
        for metric, ref in refs.items():
            if rows[ref].notna().any():
                measures[metric] = per_era_contribution(rows, [c for c in columns if c != ref],
                                                       target, ref)
        # Payout is computed only on the identical eras with actual MMC.
        if "mmc" in measures:
            measures["payout_mmc"] = 3 * corr.loc[measures["mmc"].index] + 15 * measures["mmc"]
        paired[str(window)] = {}
        for metric, scores in measures.items():
            scores = scores.loc[sorted(scores.index, key=int)]
            tidy = scores.rename_axis("era").stack().rename(metric).to_frame()
            tidy.index.names = ["era", "model"]
            era_scores.append(tidy)
            if "ridge" in scores:
                paired[str(window)][metric] = {
                    c: paired_block_ci((scores[c] - scores.ridge).to_numpy(),
                                       scores.index.to_numpy(), block=gap)
                    for c in scores if c != "ridge"}
    # Group identical era/model rows across disjoint metric frames.
    pd.concat(era_scores).groupby(level=[0, 1]).first().to_csv(out / "per_era.csv")
    correlations = prediction_correlation(result, columns)
    correlations.to_csv(out / "prediction_correlations.csv")
    for window, rows in result.groupby("window"):
        prediction_correlation(rows, columns).to_csv(out / f"correlations_{window}.csv")
    windows = [str(w) for w in sorted(result.window.unique())]
    positive_interaction = sum(metrics[w]["interaction"]["ridge_contribution"]["mean"] > 0
                               for w in windows)
    qualify = []
    for candidate in ("blend_10", "blend_25"):
        corr_wins = sum(paired[w]["corr"][candidate]["mean"] > 0 for w in windows)
        mmc_deltas = [paired[w]["mmc"][candidate]["mean"] for w in windows if "mmc" in paired[w]]
        if corr_wins >= 2 and mmc_deltas and min(mmc_deltas) >= 0 and positive_interaction >= 2:
            qualify.append(candidate)
    decision = {"scale_to_2048": bool(qualify), "qualifying_blends": qualify,
                "positive_interaction_windows": positive_interaction,
                "promotion": False,
                "reason": "Screening only; prior development windows are not fresh confirmation"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "paired_vs_ridge.json").write_text(json.dumps(paired, indent=2))
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    lines = ["# dc_wenek development results", "",
             "Full test eras; CORR covers 90 eras and MMC only the 30 eras with meta data.",
             "These are previously examined development periods, not a fresh final test.", "",
             "| Model | CORR | MMC | Worst-window CORR |",
             "|---|---:|---:|---:|"]
    for c in columns:
        r = metrics["all"][c]
        worst = min(metrics[w][c]["corr"]["mean"] for w in windows)
        lines.append(f"| {c} | {r['corr']['mean']:.5f} | "
                     f"{r.get('mmc', {}).get('mean', float('nan')):.5f} | {worst:.5f} |")
    lines.extend(["", f"Scaling screen passed: {bool(qualify)}; qualifying blends: {qualify}.",
                  "", "Paired bootstrap intervals are in paired_vs_ridge.json, separately per",
                  "contiguous window. Each 30-era window has only about 2.3 blocks; treat",
                  "intervals as exploratory. No bootstrap crosses gaps between windows."])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return metrics, decision


def run(cfg):
    started = time.time()
    gap, reserve = validate_windows(cfg)
    if Path(cfg["name"]).name != cfg["name"] or not cfg["name"].startswith("dc_wenek_"):
        raise ValueError("Use a dc_wenek_ research name without path separators")
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    production = production_snapshot()
    d = cfg["data"]
    features = feature_set(d["version"], d["feature_set"])
    rng = np.random.default_rng(cfg["seed"])
    features = [features[i] for i in sorted(rng.choice(len(features), d["feature_limit"],
                                                      replace=False))]
    source_paths = [Path(__file__).resolve(), ROOT / "src/numerai_dev/wenek.py",
                    ROOT / "src/numerai_dev/diversifiers.py", ROOT / "src/numerai_dev/models.py",
                    ROOT / "docs/wenek_plan.md"]
    manifest = {"model": "dc_wenek", "config": cfg, "features": features,
                "git": git_hash(), "created": datetime.now(UTC).isoformat(),
                "status": "running", "holdout_start": reserve, "embargo": gap,
                "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                "production_before": production, "folds": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        frame, x = load_data(cfg, features, gap)
        eras = frame.era.astype(int).to_numpy()
        y = frame[d["target"]].to_numpy(dtype=float)
        sample = frame.sampled_train.to_numpy()
        sx, sy, se = x[sample], y[sample], eras[sample]
        print(f"Loaded {len(x):,} rows, including {len(sx):,} sampled training rows; "
              f"{len(features)} features", flush=True)
        predictions = []
        for start in cfg["test_starts"]:
            end = start + cfg["test_eras"] - 1
            test = (eras >= start) & (eras <= end)
            if np.unique(eras[test]).size != cfg["test_eras"]:
                raise ValueError("Evaluation window has missing eras")
            print(f"Window {start}–{end}: fitting ridge and three FM controls", flush=True)
            bundle, train, valid, oof, residual, inner = fit_window(
                sx, sy, se, start, gap, cfg["ridge_alpha"], cfg["crossfit"],
                {**cfg["factorization_machine"], "seed": cfg["seed"]})
            audit = {"test_start": start, "test_end": end,
                     "train_start": int(se[train].min()), "train_end": int(se[train].max()),
                     "train_rows": int(train.sum()), "residual_rows": int(valid.sum()),
                     "test_rows": int(test.sum()), "inner": inner}
            pd.DataFrame({"era": se[train][valid], "target": sy[train][valid],
                          "ridge_oof": oof[valid], "residual": residual},
                         index=frame.index[sample][train][valid]).to_parquet(
                             out / f"oof_{start}.parquet")
            scores = bundle.predict(x[test], eras[test])
            teachers = teacher_predictions(sx[train], sy[train], x[test],
                                           trees=cfg["teacher_trees"], seed=cfg["seed"])
            scores["teacher_consensus"] = consensus(teachers, eras[test])
            result = frame.loc[test, ["era", d["target"]]].copy()
            for name, pred in scores.items():
                result[name] = pred
            result["window"] = start
            path = out / f"window_{start}.joblib"
            joblib.dump({"bundle": bundle, "features": features, "audit": audit}, path)
            reloaded = joblib.load(path)["bundle"]
            reproduced = reloaded.predict(x[test], eras[test], batch_size=3071)
            for name, pred in reproduced.items():
                np.testing.assert_allclose(pred, scores[name], atol=1e-10, rtol=1e-10)
            np.testing.assert_array_equal(scores["blend_00"], scores["ridge"])
            # Validate each full era, matching the single-era live submission contract.
            xf = pd.DataFrame(x[test], index=result.index, columns=features)
            es = pd.Series(eras[test], index=result.index)
            for era in np.unique(eras[test]):
                mask = es.eq(era)
                submission = reloaded.submission(xf.loc[mask], es.loc[mask], features)
                validate_submission(submission, xf.index[mask])
            manifest["folds"].append(audit)
            predictions.append(result)
            print(f"Window {start}–{end}: {len(result):,} full-era predictions; "
                  "reload/batch/submission checks passed", flush=True)
        result = pd.concat(predictions)
        for name, col in (("meta_model.parquet", META_COL),
                          ("validation_benchmark_models.parquet", BENCHMARK_COL)):
            result = result.join(pd.read_parquet(data_path(d["version"], name), columns=[col]))
        result.to_parquet(out / "validation_preds.parquet")
        metrics, decision = make_report(result, out, d["target"], gap)
        manifest.update(status="complete", decision=decision,
                        full_era_rows=len(result), sampled_training_rows=len(sx))
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="wenek_walk_forward",
                holdout_scored=False, decision=decision)
        print((out / "report.md").read_text(), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["seconds"] = round(time.time() - started, 1)
        manifest["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024) / 1e9
        manifest["memory_note"] = "Peak RSS may understate macOS compressed physical footprint"
        manifest["production_after"] = production_snapshot()
        manifest["production_unchanged"] = production == manifest["production_after"]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if not manifest["production_unchanged"]:
            raise RuntimeError("Production files changed during the experiment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(yaml.safe_load(Path(args.config).read_text()))
