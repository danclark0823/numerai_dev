"""Frozen stability research. Does not package, upload, or edit live models."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from run_bendonal import checked_data
from run_wenek import digest, production_snapshot, validate_windows
from threadpoolctl import threadpool_limits

from numerai_dev.alscot import StableRidge, select_strength
from numerai_dev.bendonal import screening_decision
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


def report(result, out, target, gap):
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL}

    def score(rows):
        scores = evaluate(rows, ["ridge", "alscot"], target, refs)
        scores["alscot"].update(evaluate(rows, ["alscot"], target,
                                        {"ridge_contribution": "ridge"})["alscot"])
        return scores

    metrics = {"all": score(result)}
    paired, era_scores = {}, []
    for start, rows in result.groupby("window"):
        metrics[str(start)] = score(rows)
        measures = {"corr": per_era_corr(rows, ["ridge", "alscot"], target)}
        for name, ref in refs.items():
            if rows[ref].notna().any():
                measures[name] = per_era_contribution(rows, ["ridge", "alscot"], target, ref)
        paired[str(start)] = {}
        for name, values in measures.items():
            values = values.loc[sorted(values.index, key=int)]
            paired[str(start)][name] = paired_block_ci(
                (values.alscot - values.ridge).to_numpy(), values.index, block=gap)
            stacked = values.stack().rename(name).to_frame()
            stacked.index.names = ["era", "model"]
            era_scores.append(stacked)
    decision = screening_decision(metrics, ["alscot"], {"alscot": 0})
    worst = {name: min(metrics[str(start)][name]["corr"]["mean"]
                       for start in result.window.unique()) for name in ["ridge", "alscot"]}
    improved = worst["alscot"] > worst["ridge"]
    decision["worst_window_corr"] = worst
    decision["worst_window_improved"] = improved
    decision["screen_passed"] = bool(decision["screen_passed"] and improved)
    decision["candidates"]["alscot"]["passed"] = decision["screen_passed"]
    decision["selected"] = "alscot" if decision["screen_passed"] else None
    decision["shelve"] = not decision["screen_passed"]
    pd.concat(era_scores).groupby(level=[0, 1]).first().to_csv(out / "per_era.csv")
    prediction_correlation(result, ["ridge", "alscot"]).to_csv(out / "prediction_correlations.csv")
    for name, value in (("metrics", metrics), ("decision", decision), ("paired_vs_ridge", paired)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2))
    lines = ["# Alscot development results", "",
             "Previously inspected windows: 90 CORR eras; actual MMC only in the latest 30.",
             "Only the inner-selected pipeline is screened; no outer-grid selection.", "",
             "| Model | CORR | MMC | Worst-window CORR |", "|---|---:|---:|---:|"]
    for name in ["ridge", "alscot"]:
        m = metrics["all"][name]
        lines.append(f"| {name} | {m['corr']['mean']:.6f} | "
                     f"{m.get('mmc', {}).get('mean', float('nan')):.6f} | {worst[name]:.6f} |")
    lines += ["", f"Decision: {json.dumps(decision)}", "",
              "Bootstrap intervals are exploratory: only about 2.3 13-era blocks/window.",
              "Covariance stability is a proxy, not direct MMC optimization. No deployment."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return metrics, decision


def run(cfg):
    started = time.time()
    gap, reserve = validate_windows(cfg)
    if Path(cfg["name"]).name != cfg["name"] or not cfg["name"].startswith("dc_alscot_"):
        raise ValueError("Invalid artifact name")
    if Path(cfg["baseline"]).name != cfg["baseline"]:
        raise ValueError("Invalid baseline")
    base = ROOT / "artifacts" / cfg["baseline"]
    old = json.loads((base / "manifest.json").read_text())
    for key in ("seed", "test_starts", "test_eras", "holdout_eras", "data"):
        if cfg[key] != old["config"][key]:
            raise ValueError(f"Baseline mismatch: {key}")
    features = old["features"]
    reference = pd.read_parquet(base / "validation_preds.parquet")
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    sources = [Path(__file__).resolve(), ROOT / "src/numerai_dev/alscot.py",
               ROOT / "src/numerai_dev/bendonal.py", ROOT / "scripts/run_bendonal.py",
               ROOT / "scripts/run_wenek.py", ROOT / "src/numerai_dev/models.py",
               ROOT / "src/numerai_dev/evaluation.py", ROOT / "src/numerai_dev/wenek.py",
               ROOT / "docs/alscot_plan.md", ROOT / "tests/test_alscot.py"]
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
        eras = frame.era.astype(int).to_numpy()
        target = cfg["data"]["target"]
        y, sample = frame[target].to_numpy(), frame.sampled_train.to_numpy()
        print(f"Loaded {len(frame):,} validated feature rows", flush=True)
        predictions = []
        for start in cfg["test_starts"]:
            train = sample & (eras < start - gap)
            test = (eras >= start) & (eras < start + cfg["test_eras"])
            assert np.unique(eras[test]).size == cfg["test_eras"]
            tx, ty, te = x[train], y[train], eras[train]
            print(f"Window {start}: nested selection on {len(ty):,} historical rows", flush=True)
            strength, audit = select_strength(
                tx, ty, te, start, cfg["strengths"], cfg["alpha"], cfg["block_eras"],
                gap, cfg["inner_eras"], cfg["inner_folds"])
            assert strength in cfg["strengths"]
            ridge = RidgeModel(cfg["alpha"]).fit_stream([tx], ty)
            zero = StableRidge(cfg["alpha"], 0, cfg["block_eras"]).fit(tx, ty, te)
            model = StableRidge(cfg["alpha"], strength, cfg["block_eras"]).fit(tx, ty, te)
            vx = x[test]
            np.testing.assert_allclose(zero.predict(vx), ridge.predict(vx), atol=1e-12)
            rows = reference.loc[frame.index[test], ["era", target, "window", META_COL, BENCHMARK_COL]].copy()
            np.testing.assert_array_equal(rows[target], y[test])
            np.testing.assert_array_equal(rows.era.astype(int), eras[test])
            assert rows.window.eq(start).all() and rows.index.is_unique
            rows["ridge"], rows["alscot"] = ridge.predict(vx), model.predict(vx)
            np.testing.assert_allclose(era_gaussian(rows.ridge.to_numpy(), eras[test]),
                                       reference.loc[rows.index, "ridge"], atol=1e-10)
            path = out / f"window_{start}.joblib"
            joblib.dump({"model": model, "ridge": ridge, "features": features, "audit": audit}, path)
            loaded = joblib.load(path)
            assert loaded["features"] == features
            np.testing.assert_allclose(loaded["model"].predict(vx, batch_size=3071), rows.alscot, atol=1e-12)
            assert np.isfinite(rows[["ridge", "alscot"]]).all().all()
            for _, era in rows.groupby("era"):
                assert era.alscot.nunique() > 1
                validate_submission(era.alscot.rank(pct=True).rename("prediction"), era.index)
            pd.DataFrame({"era": te, "target": ty}, index=frame.index[train]).to_parquet(
                out / f"training_rows_{start}.parquet")
            audit.update(test_start=start, train_end=int(te.max()), train_rows=len(ty),
                         test_rows=len(rows), coefficient_norm=float(np.linalg.norm(model.beta)),
                         penalty_scale=model.penalty_scale,
                         stability_penalty=float(model.beta @ model.penalty @ model.beta),
                         ridge_stability_penalty=float(ridge.beta @ model.penalty @ ridge.beta),
                         checks_passed=True)
            manifest["folds"].append(audit)
            predictions.append(rows)
            print(f"Window {start}: selected lambda={strength:g}; parity and submission checks passed", flush=True)
        result = pd.concat(predictions)
        result.to_parquet(out / "validation_preds.parquet")
        metrics, decision = report(result, out, target, gap)
        manifest.update(status="complete", decision=decision, full_era_rows=len(result))
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="alscot_walk_forward",
                holdout_scored=False, decision=decision)
        print((out / "report.md").read_text(), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["seconds"] = round(time.time() - started, 1)
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
