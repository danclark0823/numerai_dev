"""Frozen additive-model research; never uploads or writes production artifacts."""

# Archived experiment: dc_bendonal_1024_v1 tested curvature penalties 10 and 100
# against matched ridge across three embargoed development windows. Both gained
# CORR in every window, but neither achieved nonnegative MMC in the 30 eras where
# it was available. The frozen screen selected neither candidate, so no larger
# run or production build followed. Keep outputs intact rather than tuning this
# completed trial after seeing its results. See docs/bendonal.md for limitations
# and metrics; the four existing live models remain outside this experiment.

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from run_wenek import digest, production_snapshot, sample_positions, validate_windows
from threadpoolctl import threadpool_limits

from numerai_dev.bendonal import AdditiveModel, screening_decision, sufficient_stats, validate_bins
from numerai_dev.data import data_path, load_index
from numerai_dev.diversifiers import era_gaussian
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
from numerai_dev.models import RidgeModel
from numerai_dev.wenek import paired_block_ci


def checked_data(cfg, features, gap):
    """Validate selected raw parquet values before any potentially lossy int8 cast."""
    version, target = cfg["data"]["version"], cfg["data"]["target"]
    frames = []
    for split in ("train", "validation"):
        f = load_index(version, split, [target])
        mask = f[target].notna()
        if split == "validation":
            mask &= f.data_type.eq("validation")
        f = f.loc[mask, ["era", target]].copy()
        f["split"] = split
        frames.append(f)
    frame = pd.concat(frames)
    if not frame.index.is_unique:
        raise ValueError("Duplicate source IDs")
    eras = frame.era.astype(int).to_numpy()
    sample = sample_positions(frame, max(cfg["test_starts"]) - gap - 1,
                              cfg["data"]["train_stride"], cfg["data"]["train_rows_per_era"],
                              cfg["seed"])
    test = np.zeros(len(frame), dtype=bool)
    for start in cfg["test_starts"]:
        test |= (eras >= start) & (eras < start + cfg["test_eras"])
    if (test & frame.split.ne("validation").to_numpy()).any():
        raise ValueError("Test observations must be validation rows")
    frame["sampled_train"] = sample
    frame = frame.loc[sample | test]
    x = np.empty((len(frame), len(features)), dtype=np.int8)
    filled = np.zeros(len(frame), dtype=bool)
    for split in ("train", "validation"):
        idx = load_index(version, split, [])
        mask = idx.index.isin(frame.index[frame.split.eq(split)])
        parquet = pq.ParquetFile(data_path(version, f"{split}.parquet"), pre_buffer=False)
        offset = 0
        for batch in parquet.iter_batches(batch_size=16384, columns=features):
            selected = mask[offset:offset + len(batch)]
            if selected.any():
                raw = np.column_stack([column.to_numpy(zero_copy_only=False)
                                       for column in batch.columns])[selected]
                values = validate_bins(raw)
                ids = idx.index[offset:offset + len(batch)][selected]
                pos = frame.index.get_indexer(ids)
                if (pos < 0).any() or filled[pos].any():
                    raise ValueError("Parquet/index alignment failed")
                x[pos], filled[pos] = values, True
            offset += len(batch)
        if offset != len(idx):
            raise ValueError("Parquet/index row counts differ")
    if not filled.all():
        raise ValueError("Selected feature rows were not filled")
    return frame, x


def report(result, out, target, gap, candidates):
    columns = ["ridge", *candidates, *[f"{name}_curvature" for name in candidates]]
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL,
            "ridge_contribution": "ridge", "teacher_contribution": "teacher_consensus"}

    def score(rows):
        return {name: evaluate(rows, [name], target,
                               {k: v for k, v in refs.items() if v != name})[name]
                for name in columns}

    metrics = {"all": score(result)}
    paired, scores = {}, []
    for start, rows in result.groupby("window"):
        metrics[str(start)] = score(rows)
        measures = {"corr": per_era_corr(rows, columns, target)}
        for name, ref in refs.items():
            if rows[ref].notna().any():
                measures[name] = per_era_contribution(rows, [n for n in columns if n != ref],
                                                     target, ref)
        if "mmc" in measures:
            measures["payout_mmc"] = (3 * measures["corr"].loc[measures["mmc"].index]
                                       + 15 * measures["mmc"])
        paired[str(start)] = {}
        for name, values in measures.items():
            values = values.loc[sorted(values.index, key=int)]
            stacked = values.stack().rename(name).to_frame()
            stacked.index.names = ["era", "model"]
            scores.append(stacked)
            if "ridge" in values:
                paired[str(start)][name] = {
                    c: paired_block_ci((values[c] - values.ridge).to_numpy(), values.index,
                                       block=gap) for c in candidates}
        prediction_correlation(rows, columns).to_csv(out / f"correlations_{start}.csv")
    decision = screening_decision(metrics, list(candidates), candidates)
    prediction_correlation(result, columns).to_csv(out / "prediction_correlations.csv")
    pd.concat(scores).groupby(level=[0, 1]).first().to_csv(out / "per_era.csv")
    for name, obj in (("metrics", metrics), ("paired_vs_ridge", paired), ("decision", decision)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=2))
    lines = ["# dc_bendonal development results", "",
             "CORR covers 90 complete eras; MMC covers only 30. All are previously examined.",
             "Curvature-only outputs are diagnostics, not additional selectable candidates.", "",
             "| Model | CORR | MMC | Worst-window CORR | Positive ridge-contribution windows |",
             "|---|---:|---:|---:|---:|"]
    for name in ["ridge", *candidates]:
        m = metrics["all"][name]
        worst = min(metrics[str(w)][name]["corr"]["mean"] for w in result.window.unique())
        count = decision["candidates"].get(name, {}).get("positive_contribution_windows", "—")
        lines.append(f"| {name} | {m['corr']['mean']:.6f} | "
                     f"{m.get('mmc', {}).get('mean', float('nan')):.6f} | {worst:.6f} | {count} |")
    lines += ["", f"Screen decision: {json.dumps(decision)}", "",
              "Paired bootstrap intervals are separate per contiguous window, with only",
              "about 2.3 thirteen-era blocks each. A passing screen is not confirmation.",
              "The teacher reference is the frozen 1,024-row Wenek historical ensemble."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return metrics, decision


def run(cfg):
    started = time.time()
    gap, reserve = validate_windows(cfg)
    if Path(cfg["name"]).name != cfg["name"] or not cfg["name"].startswith("dc_bendonal_"):
        raise ValueError("Invalid research artifact name")
    if Path(cfg["baseline"]).name != cfg["baseline"]:
        raise ValueError("Invalid baseline artifact name")
    base = ROOT / "artifacts" / cfg["baseline"]
    baseline_manifest = json.loads((base / "manifest.json").read_text())
    features = baseline_manifest["features"]
    oldcfg = baseline_manifest["config"]
    for key in ("seed", "test_starts", "test_eras", "holdout_eras"):
        if cfg[key] != oldcfg[key]:
            raise ValueError(f"Baseline design mismatch: {key}")
    for key in ("version", "target", "feature_set", "feature_limit", "train_stride"):
        if cfg["data"][key] != oldcfg["data"][key]:
            raise ValueError(f"Baseline data mismatch: {key}")
    reference = pd.read_parquet(base / "validation_preds.parquet")
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    sources = [Path(__file__).resolve(), ROOT / "src/numerai_dev/bendonal.py",
               ROOT / "src/numerai_dev/models.py", ROOT / "scripts/run_wenek.py",
               ROOT / "docs/bendonal_plan.md"]
    (out / "sources").mkdir()
    for path in sources:
        (out / "sources" / path.name).write_bytes(path.read_bytes())
    manifest = {"config": cfg, "features": features, "git": git_hash(), "status": "running",
                "holdout_start": reserve, "embargo": gap, "production_before": before,
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "baseline_manifest_sha256": digest(base / "manifest.json"), "folds": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        frame, x = checked_data(cfg, features, gap)
        eras, y = frame.era.astype(int).to_numpy(), frame[cfg["data"]["target"]].to_numpy()
        sample = frame.sampled_train.to_numpy()
        print(f"Verified {len(x):,} raw feature rows: finite bins 0–4; missing policy=reject", flush=True)
        predictions, diagnostics = [], {}
        for start in cfg["test_starts"]:
            print(f"Window {start}: accumulating streamed additive statistics", flush=True)
            train = sample & (eras < start - gap)
            test = (eras >= start) & (eras < start + cfg["test_eras"])
            if np.unique(eras[test]).size != cfg["test_eras"]:
                raise ValueError("Incomplete test window")
            train_ids = frame.index[train]
            pd.DataFrame({"era": eras[train], "target": y[train]}, index=train_ids).to_parquet(
                out / f"training_rows_{start}.parquet")
            tx, ty, vx = x[train], y[train], x[test]
            ridge = RidgeModel(cfg["linear_penalty"]).fit_stream([tx], ty)
            stats = sufficient_stats([tx], ty)
            models = {name: AdditiveModel(cfg["linear_penalty"], penalty).fit_stats(stats)
                      for name, penalty in cfg["candidates"].items()}
            linear = AdditiveModel(cfg["linear_penalty"]).fit_stream([tx], ty, nonlinear=False)
            np.testing.assert_allclose(linear.predict(vx), ridge.predict(vx), atol=1e-10, rtol=1e-10)
            rows = reference.loc[frame.index[test], ["era", cfg["data"]["target"], "window",
                                                      META_COL, BENCHMARK_COL, "teacher_consensus"]].copy()
            np.testing.assert_array_equal(rows[cfg["data"]["target"]], y[test])
            rows["ridge"] = ridge.predict(vx)
            if cfg["data"]["train_rows_per_era"] == oldcfg["data"]["train_rows_per_era"]:
                np.testing.assert_allclose(era_gaussian(rows.ridge.to_numpy(), eras[test]),
                                           reference.loc[rows.index, "ridge"], atol=1e-10)
                saved_oof = pd.read_parquet(base / f"oof_{start}.parquet")
                assert saved_oof.index.isin(train_ids).all()
                saved_ridge = joblib.load(base / f"window_{start}.joblib")["bundle"].ridge
                np.testing.assert_allclose(ridge.beta, saved_ridge.beta, atol=1e-12)
            diagnostics[str(start)] = {}
            for name, model in models.items():
                raw = model.predict(vx)
                rows[name] = raw
                rows[f"{name}_curvature"] = model.predict(vx, component="nonlinear")
                path = out / f"{name}_{start}.joblib"
                joblib.dump({"model": model, "ridge": ridge, "features": features,
                             "train_end": int(eras[train].max()), "test_start": start}, path)
                loaded = joblib.load(path)["model"]
                np.testing.assert_allclose(loaded.predict(vx, batch_size=3071), raw, atol=1e-10)
                curves = model.curves()
                np.testing.assert_allclose(model.intercept + curves[np.arange(len(features)), vx].sum(axis=1),
                                           raw, atol=1e-10)
                pd.DataFrame(curves, index=features, columns=range(5)).to_csv(out / f"curves_{name}_{start}.csv")
                xf = pd.DataFrame(vx, index=rows.index, columns=features)
                es = pd.Series(eras[test], index=rows.index)
                submission = loaded.submission(xf[features[::-1]], es, features, batch_size=3071)
                np.testing.assert_array_equal(submission, pd.Series(raw, index=rows.index).groupby(es).rank(pct=True))
                for era in es.unique():
                    mask = es.eq(era)
                    validate_submission(submission.loc[mask], rows.index[mask])
                training = pd.DataFrame({"era": frame.loc[train, "era"], "target": ty,
                                         "prediction": model.predict(tx)})
                diagnostics[str(start)][name] = {
                    "in_sample_corr": float(per_era_corr(training, ["prediction"], "target").prediction.mean()),
                    "train_mse": float(np.mean((training.prediction.to_numpy() - ty)**2)),
                    "test_mse": float(np.mean((raw - y[test])**2)),
                    "linear_norm": float(np.linalg.norm(model.beta[:model.p])),
                    "nonlinear_norm": float(np.linalg.norm(model.beta[model.p:])),
                    "missing_training_bins": int((model.bin_counts == 0).sum())}
            pd.DataFrame(stats.bin_counts, index=features).to_csv(out / f"training_bins_{start}.csv")
            predictions.append(rows)
            manifest["folds"].append({"test_start": start, "train_end": int(eras[train].max()),
                                       "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                                       "checks_passed": True})
            print(f"Window {start}: {len(rows):,} predictions; ridge, curves, reload and submission parity passed", flush=True)
        result = pd.concat(predictions)
        result.to_parquet(out / "validation_preds.parquet")
        (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
        metrics, decision = report(result, out, cfg["data"]["target"], gap, cfg["candidates"])
        manifest.update(status="complete", decision=decision, full_era_rows=len(result))
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="bendonal_walk_forward",
                holdout_scored=False, decision=decision)
        print((out / "report.md").read_text(), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["seconds"] = round(time.time() - started, 1)
        manifest["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024) / 1e9
        manifest["memory_note"] = "Peak RSS can understate macOS compressed footprint"
        manifest["production_after"] = production_snapshot()
        manifest["production_unchanged"] = before == manifest["production_after"]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if not manifest["production_unchanged"]:
            raise RuntimeError("Production files changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(yaml.safe_load(Path(args.config).read_text()))
