"""Frozen feature-specialist experiment; no production fitting or uploads."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from run_bendonal import checked_data
from run_wenek import digest, production_snapshot, validate_windows
from threadpoolctl import threadpool_limits

from numerai_dev.data import data_path, feature_set
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
from numerai_dev.hesia import RunningStats, SpecialistBundle, feature_subsets
from numerai_dev.live import validate_submission
from numerai_dev.wenek import paired_block_ci


def screen(metrics):
    windows = [key for key in metrics if key != "all"]
    positive = sum(metrics[w]["hesia"]["all_contribution"]["mean"] > 0 for w in windows)
    wins = sum(metrics[w]["hesia"]["corr"]["mean"] > metrics[w]["ridge_all"]["corr"]["mean"]
               for w in windows)
    pooled = metrics["all"]["hesia"]["corr"]["mean"] > metrics["all"]["ridge_all"]["corr"]["mean"]
    mmc = [(metrics[w]["hesia"]["mmc"]["mean"], metrics[w]["ridge_all"]["mmc"]["mean"])
           for w in windows if "mmc" in metrics[w]["hesia"]]
    mmc_passed = bool(mmc) and all(value >= 0 and value >= ref for value, ref in mmc)
    passed = positive >= 2 and wins >= 2 and pooled and mmc_passed
    return {"screen_passed": bool(passed), "positive_all_contribution_windows": positive,
            "corr_wins_vs_all": wins, "pooled_corr_improved_vs_all": bool(pooled),
            "mmc_passed": bool(mmc_passed), "deployment": False, "shelve": not passed}


def report(result, out, target, gap):
    columns = ["ridge128", "ridge_all", "hesia"]
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL}

    def score(rows):
        scores = evaluate(rows, columns, target, refs)
        extra = evaluate(rows, ["hesia"], target,
                         {"all_contribution": "ridge_all", "small_contribution": "ridge128"})
        scores["hesia"].update(extra["hesia"])
        return scores

    metrics = {"all": score(result)}
    paired, per_era = {}, []
    for start, rows in result.groupby("window"):
        metrics[str(start)] = score(rows)
        measures = {"corr": per_era_corr(rows, columns, target)}
        for name, ref in refs.items():
            if rows[ref].notna().any():
                measures[name] = per_era_contribution(rows, columns, target, ref)
        paired[str(start)] = {}
        for name, values in measures.items():
            values = values.loc[sorted(values.index, key=int)]
            paired[str(start)][name] = {
                f"{candidate}_minus_{control}": paired_block_ci(
                    (values[candidate] - values[control]).to_numpy(), values.index, block=gap)
                for candidate, control in [("hesia", "ridge_all"), ("hesia", "ridge128"),
                                            ("ridge_all", "ridge128")]}
            stacked = values.stack().rename(name).to_frame()
            stacked.index.names = ["era", "model"]
            per_era.append(stacked)
        prediction_correlation(rows, columns).to_csv(out / f"correlations_{start}.csv")
    decision = screen(metrics)
    pd.concat(per_era).groupby(level=[0, 1]).first().to_csv(out / "per_era.csv")
    prediction_correlation(result, columns).to_csv(out / "prediction_correlations.csv")
    for name, value in (("metrics", metrics), ("decision", decision), ("paired", paired)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2))
    lines = ["# Hesia development results", "",
             "CORR: 90 full eras. Actual MMC: latest 30 only. Previously inspected windows.",
             "Fixed 16-member equal blend; primary control is ALL-feature ridge.", "",
             "| Model | CORR | MMC | Worst-window CORR |", "|---|---:|---:|---:|"]
    for name in columns:
        m = metrics["all"][name]
        worst = min(metrics[str(w)][name]["corr"]["mean"] for w in result.window.unique())
        lines.append(f"| {name} | {m['corr']['mean']:.6f} | "
                     f"{m.get('mmc', {}).get('mean', float('nan')):.6f} | {worst:.6f} |")
    lines += ["", f"Decision: {json.dumps(decision)}", "",
              "Paired block intervals are exploratory (about 2.3 blocks per window).",
              "No holdout confirmation, production edits, packaging or upload."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return metrics, decision


def run(cfg):
    started = time.time()
    gap, reserve = validate_windows(cfg)
    if Path(cfg["name"]).name != cfg["name"] or not cfg["name"].startswith("dc_hesia_"):
        raise ValueError("Invalid run name")
    if Path(cfg["baseline"]).name != cfg["baseline"]:
        raise ValueError("Invalid baseline")
    base = ROOT / "artifacts" / cfg["baseline"]
    old = json.loads((base / "manifest.json").read_text())
    for key in ("seed", "test_starts", "test_eras", "holdout_eras"):
        if cfg[key] != old["config"][key]:
            raise ValueError(f"Baseline mismatch: {key}")
    for key in ("version", "target", "train_rows_per_era", "train_stride"):
        if cfg["data"][key] != old["config"]["data"][key]:
            raise ValueError(f"Sampling mismatch: {key}")
    features = feature_set(cfg["data"]["version"], cfg["data"]["feature_set"])
    small = np.array([features.index(name) for name in old["features"]])
    subsets = feature_subsets(len(features), cfg["partitions"], cfg["groups_per_partition"], cfg["seed"])
    reference = pd.read_parquet(base / "validation_preds.parquet")
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    sources = [Path(__file__).resolve(), ROOT / "src/numerai_dev/hesia.py",
               ROOT / "scripts/run_bendonal.py", ROOT / "scripts/run_wenek.py",
               ROOT / "src/numerai_dev/bendonal.py", ROOT / "src/numerai_dev/diversifiers.py",
               ROOT / "src/numerai_dev/evaluation.py", ROOT / "src/numerai_dev/wenek.py",
               ROOT / "docs/hesia_plan.md", ROOT / "tests/test_hesia.py"]
    (out / "sources").mkdir()
    for path in sources:
        (out / "sources" / path.name).write_bytes(path.read_bytes())
    manifest = {"config": cfg, "features": features, "small_features": old["features"],
                "subsets": [[features[j] for j in subset] for subset in subsets],
                "git": git_hash(), "status": "running", "holdout_start": reserve,
                "embargo": gap, "production_before": before, "folds": [],
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "metadata_sha256": digest(data_path(cfg["data"]["version"], "features.json")),
                "baseline_manifest_sha256": digest(base / "manifest.json")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        print(f"Loading validated full-feature data ({len(features)} columns)", flush=True)
        frame, x = checked_data(cfg, features, gap)
        eras, target = frame.era.astype(int).to_numpy(), cfg["data"]["target"]
        y, sample = frame[target].to_numpy(), frame.sampled_train.to_numpy()
        stats, seen = RunningStats(len(features)), np.zeros(len(frame), dtype=bool)
        predictions = []
        for start in cfg["test_starts"]:
            train = sample & (eras < start - gap)
            test = (eras >= start) & (eras < start + cfg["test_eras"])
            assert not (seen & ~train).any() and np.unique(eras[test]).size == cfg["test_eras"]
            positions = np.flatnonzero(train & ~seen)
            print(f"Window {start}: accumulating {len(positions):,} new training rows", flush=True)
            for offset in range(0, len(positions), 4096):
                idx = positions[offset:offset + 4096]
                stats.update(x[idx], y[idx])
            seen |= train
            assert stats.n == train.sum()
            print(f"Window {start}: fitting controls and {len(subsets)} specialists", flush=True)
            bundle = SpecialistBundle.fit_stats(stats, features, subsets, small, cfg["alpha"])
            rows = reference.loc[frame.index[test], ["era", target, "window", META_COL, BENCHMARK_COL]].copy()
            assert rows.index.is_unique and rows.window.eq(start).all()
            np.testing.assert_array_equal(rows.era.astype(int), eras[test])
            np.testing.assert_array_equal(rows[target], y[test])
            vx = x[test]
            values = bundle.predict(vx, eras[test])
            rows[["ridge128", "ridge_all", "hesia"]] = values
            np.testing.assert_allclose(era_gaussian(values[:, 0], eras[test]),
                                       reference.loc[rows.index, "ridge"], atol=1e-10)
            path = out / f"window_{start}.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            assert loaded.features == features
            np.testing.assert_allclose(loaded.predict(vx, eras[test], 3071), values, atol=1e-10)
            assert np.isfinite(values).all()
            for _, era in rows.groupby("era"):
                for name in ("ridge128", "ridge_all", "hesia"):
                    validate_submission(era[name].rank(pct=True).rename("prediction"), era.index)
            # Exercise named-column selection and ranks on a real complete era.
            first = eras[test] == eras[test][0]
            xf = pd.DataFrame(vx[first], index=rows.index[first], columns=features)
            es = rows.loc[xf.index, "era"]
            submission = loaded.submission(xf[features[::-1]], es, 3071)
            np.testing.assert_allclose(submission, rows.loc[xf.index, "hesia"].rank(pct=True))
            pd.DataFrame({"era": eras[train], "target": y[train]}, index=frame.index[train]).to_parquet(
                out / f"training_rows_{start}.parquet")
            audit = {"test_start": start, "train_end": int(eras[train].max()),
                     "train_rows": int(train.sum()), "test_rows": len(rows), "checks_passed": True}
            assert audit["train_end"] < start - gap
            manifest["folds"].append(audit)
            predictions.append(rows)
            print(f"Window {start}: full-era predictions, ridge parity and reload checks passed", flush=True)
            del vx, xf, values, loaded, bundle
        result = pd.concat(predictions)
        result.to_parquet(out / "validation_preds.parquet")
        metrics, decision = report(result, out, target, gap)
        manifest.update(status="complete", decision=decision, full_era_rows=len(result))
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="hesia_walk_forward",
                holdout_scored=False, decision=decision)
        print((out / "report.md").read_text(), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["seconds"] = round(time.time() - started, 1)
        manifest["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024) / 1e9
        manifest["memory_note"] = "Peak RSS can understate macOS compressed physical footprint"
        manifest["production_after"] = production_snapshot()
        manifest["production_unchanged"] = before == manifest["production_after"]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if not manifest["production_unchanged"]:
            raise RuntimeError("Protected production files changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(yaml.safe_load(Path(args.config).read_text()))
