"""Controlled simplification using the exact saved Wenek OOF residuals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from run_wenek import digest, load_data, production_snapshot, validate_windows
from threadpoolctl import threadpool_limits

from numerai_dev.diversifiers import FactorizationMachine, era_gaussian
from numerai_dev.evaluation import evaluate, flatten, per_era_contribution, per_era_corr
from numerai_dev.experiment import ROOT, log_run
from numerai_dev.wenek import blend, paired_block_ci


def diagnostics(y, prediction, eras):
    corrs = []
    for era in np.unique(eras):
        mask = eras == era
        if np.std(y[mask]) > 0 and np.std(prediction[mask]) > 0:
            corrs.append(float(np.corrcoef(y[mask], prediction[mask])[0, 1]))
    return {"mean_era_residual_pearson": float(np.mean(corrs)) if corrs else None,
            "mse": float(np.mean((y - prediction)**2)),
            "zero_prediction_mse": float(np.mean(y**2)),
            "prediction_std": float(np.std(prediction)), "rows": len(y)}


def run(cfg):
    started = time.time()
    for name in (cfg["name"], cfg["baseline"]):
        if Path(name).name != name or not name.startswith("dc_wenek_"):
            raise ValueError("Require a dc_wenek_ artifact name without path separators")
    base = ROOT / "artifacts" / cfg["baseline"]
    previous = json.loads((base / "manifest.json").read_text())
    if previous["status"] != "complete":
        raise ValueError("Baseline must be a completed experiment")
    for path, checksum in previous["sources"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Baseline source changed: {path}")
    original = previous["config"]
    gap, _ = validate_windows(original)
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    manifest = {"config": cfg, "baseline_manifest_sha256": digest(base / "manifest.json"),
                "status": "running", "production_before": before,
                "source_sha256": digest(Path(__file__).resolve()),
                "plan_sha256": digest(ROOT / "docs/wenek_simplify_plan.md"),
                "checks": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        frame, x = load_data(original, previous["features"], gap)
        eras = frame.era.astype(int).to_numpy()
        target = original["data"]["target"]
        result = pd.read_parquet(base / "validation_preds.parquet")
        metrics, diag, paired, per_era = {}, {}, {}, []
        for start in original["test_starts"]:
            print(f"Window {start}: verifying saved baseline and fitting two variants", flush=True)
            saved = joblib.load(base / f"window_{start}.joblib")
            assert saved["features"] == previous["features"]
            oof = pd.read_parquet(base / f"oof_{start}.parquet")
            pos = frame.index.get_indexer(oof.index)
            if (pos < 0).any() or not oof.index.is_unique:
                raise ValueError("OOF training IDs do not match current data")
            np.testing.assert_array_equal(frame.iloc[pos][target], oof.target)
            np.testing.assert_array_equal(eras[pos], oof.era)
            np.testing.assert_array_equal(oof.residual, oof.target - oof.ridge_oof)
            assert frame.iloc[pos].sampled_train.all()
            assert eras[pos].max() + gap < start
            assert all(a["train_end"] + gap < a["test_start"] for a in saved["audit"]["inner"])
            rows = result.loc[result.window.eq(start)].copy()
            tpos = frame.index.get_indexer(rows.index)
            assert (tpos >= 0).all()
            tx, vx = x[pos], x[tpos]
            te, ve = eras[pos], eras[tpos]
            residual = oof.residual.to_numpy()
            baseline = saved["bundle"]
            for name, values in baseline.predict(vx, ve).items():
                np.testing.assert_allclose(values, rows[name], atol=1e-10, rtol=1e-10)
            test_residual = rows[target].to_numpy() - baseline.ridge.predict(vx)
            diag[str(start)] = {}
            models = {"original": baseline.interaction}
            for label, override in cfg["variants"].items():
                params = {**original["factorization_machine"], "seed": original["seed"],
                          **override, "interaction_only": True}
                models[label] = FactorizationMachine(**params).fit(tx, residual)
            columns = []
            for label, model in models.items():
                np.testing.assert_array_equal(model.w, np.zeros(tx.shape[1]))
                raw_train, raw_test = model.predict(tx), model.predict(vx)
                diag[str(start)][label] = {
                    "in_sample_interaction": diagnostics(residual, raw_train, te),
                    "later_test": diagnostics(test_residual, raw_test, ve)}
                score = era_gaussian(raw_test, ve)
                rows[label] = score
                columns.append(label)
                for weight in (0.1, 0.25):
                    name = f"{label}_blend_{int(100 * weight)}"
                    rows[name] = blend(rows.ridge.to_numpy(), score, weight)
                    columns.append(name)
                if label != "original":
                    path = out / f"{label}_{start}.joblib"
                    joblib.dump({"model": model, "features": saved["features"],
                                 "audit": saved["audit"],
                                 "params": {**original["factorization_machine"],
                                            **cfg["variants"][label], "seed": original["seed"],
                                            "interaction_only": True}}, path)
                    reloaded = joblib.load(path)["model"]
                    chunked = np.concatenate([reloaded.predict(vx[i:i + 3071])
                                              for i in range(0, len(vx), 3071)])
                    np.testing.assert_allclose(raw_test, chunked, atol=1e-10, rtol=1e-10)
            refs = {"mmc": "numerai_meta_model", "ridge_contribution": "ridge",
                    "teacher_contribution": "teacher_consensus"}
            metrics[str(start)] = evaluate(rows, columns, target, refs)
            scores = {"corr": per_era_corr(rows, columns, target)}
            for name, ref in refs.items():
                if rows[ref].notna().any():
                    scores[name] = per_era_contribution(rows, columns, target, ref)
            paired[str(start)] = {}
            for name, values in scores.items():
                values = values.loc[sorted(values.index, key=int)]
                per_era.append(values.stack().rename(name).to_frame())
                paired[str(start)][name] = {
                    label: paired_block_ci((values[label] - values.original).to_numpy(),
                                            values.index, block=gap)
                    for label in cfg["variants"]}
            for col in columns:
                result.loc[rows.index, col] = rows[col]
            manifest["checks"].append({"window": start, "training_rows": len(tx),
                                       "test_rows": len(vx), "parity_passed": True})
            print(f"Window {start}: complete; baseline, residual and reload checks passed", flush=True)
        counts = {label: sum(m[label]["ridge_contribution"]["mean"] > 0
                            for m in metrics.values())
                  for label in ["original", *cfg["variants"]]}
        passes = [label for label in cfg["variants"] if counts[label] >= 2]
        decision = {"positive_contribution_windows": counts, "passing_variants": passes,
                    "shelve_formulation": not passes, "deployment": False}
        result.to_parquet(out / "validation_preds.parquet")
        pd.concat(per_era).groupby(level=[0, 1]).first().to_csv(out / "per_era.csv")
        for name, value in (("metrics", metrics), ("diagnostics", diag),
                             ("paired_vs_original", paired), ("decision", decision)):
            (out / f"{name}.json").write_text(json.dumps(value, indent=2))
        lines = ["# Wenek simplification results", "",
                 "Full test eras; all settings fixed except the stated rank/penalty change.",
                 "", "| Window | Variant | CORR | MMC | Ridge contribution | Train residual r | Test residual r |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for window, values in metrics.items():
            for label in ["original", *cfg["variants"]]:
                m, d = values[label], diag[window][label]
                mmc = f"{m['mmc']['mean']:.5f}" if "mmc" in m else "unavailable"
                lines.append(f"| {window} | {label} | {m['corr']['mean']:.5f} | {mmc} | "
                             f"{m['ridge_contribution']['mean']:.5f} | "
                             f"{d['in_sample_interaction']['mean_era_residual_pearson']:.5f} | "
                             f"{d['later_test']['mean_era_residual_pearson']:.5f} |")
        lines += ["", f"Decision: {json.dumps(decision)}", "",
                  "Training diagnostics are in-sample for the interaction model. Test",
                  "residuals are measured later against the outer ridge. These repeatedly",
                  "examined windows do not provide fresh confirmation. Paired confidence",
                  "intervals use only about 2.3 blocks per window and are exploratory."]
        (out / "report.md").write_text("\n".join(lines) + "\n")
        manifest.update(status="complete", decision=decision)
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="wenek_simplification",
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
            raise RuntimeError("Production files changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(yaml.safe_load(Path(args.config).read_text()))
