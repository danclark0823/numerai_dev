"""Frozen alternate-target residual ridge research (dc_azar); see docs/azar_plan.md.

dc_azar would be a separate model. Never uploads, packages hosted models, or edits
live configuration; live dc_rondal is only recreated walk-forward as a yardstick.

Usage: python scripts/run_azar.py configs/azar_full.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from run_experiment import peak_rss_gb
from run_taren import check_windows, design, feature_batches
from run_wenek import digest, production_snapshot
from threadpoolctl import threadpool_limits

from numerai_dev.azar import (
    EndpointFitter,
    mean_contribution,
    residualize_many,
    screening_decision,
    select_target,
)
from numerai_dev.data import data_path, era_key, feature_set, load_index
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
from numerai_dev.targets import residualize
from numerai_dev.wenek import paired_block_ci

PREDICT_ROWS = 8_192
PARITY_ROWS = 512
PARITY_ERAS = 3


def short(target: str) -> str:
    return target.removeprefix("target_").removesuffix("_60")


def labels_for(cfg, split, last_train):
    """Index frame plus residual labels (rows x targets) for eligible training rows,
    with a parity check against `targets.residualize` on a few eras."""
    d = cfg["data"]
    version, targets, refs = d["version"], d["candidate_targets"], d["residualize_against"]
    base = d["target"]
    frame = load_index(version, split, targets)
    bench = pd.read_parquet(data_path(version, f"{split}_benchmark_models.parquet"),
                            columns=refs)
    frame = frame.join(bench)
    eras = frame.era.map(era_key).to_numpy()
    typed = (frame.data_type.eq("validation").to_numpy() if split == "validation"
             else np.ones(len(frame), dtype=bool))
    rondal_rows = frame[[base, *refs]].notna().all(axis=1).to_numpy() & typed
    complete = frame[[*targets, *refs]].notna().all(axis=1).to_numpy() & typed
    eligible = eras <= last_train
    if (rondal_rows & eligible & ~complete).any():
        raise ValueError("A candidate target is missing on rows where Rondal has a label")
    train_m = complete & eligible
    labels = residualize_many(frame.loc[train_m, targets].to_numpy(),
                              frame.loc[train_m, refs].to_numpy(), eras[train_m])
    sub = frame.loc[train_m]
    rng = np.random.default_rng(0)
    checked = sorted(rng.choice(sub.era.unique(), PARITY_ERAS, replace=False).tolist())
    for era in checked:
        rows = sub.era.eq(era).to_numpy()
        for j, t in enumerate(targets):
            # A single-era groupby-apply returns a 1 x n frame; values are in row order.
            # targets.residualize centers float32 targets in float32 (~1e-8 rounding).
            expected = np.asarray(residualize(sub[rows], t, refs)).ravel()
            np.testing.assert_allclose(labels[rows, j], expected, atol=1e-6)
    return frame, eras, typed, train_m, labels, checked


def check_rondal_parity(frame, taren):
    """The fallback must reproduce dc_taren_full_v1's walk-forward Rondal exactly."""
    np.testing.assert_allclose(frame.rondal, taren.loc[frame.index, "rondal"],
                               rtol=1e-5, atol=1e-7)


def stream(cfg, features, endpoints, windows):
    d = cfg["data"]
    version, base, targets = d["version"], d["target"], d["candidate_targets"]
    names = [short(t) for t in targets]
    fitter = EndpointFitter(len(features), names, endpoints, cfg["alpha"])
    coef: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    spans = [(key, a, b, e) for key, (a, b, e) in windows.items()]
    last_train = max(endpoints)
    last_pred = max(b for _, _, b, _ in spans)
    pieces = {key: [] for key in windows}
    parity = {key: [] for key in windows if key[0] == "outer"}
    audit = {}

    def refresh():
        for endpoint, models in fitter.models.items():
            if endpoint not in coef:
                coef[endpoint] = (np.column_stack([models[n].beta for n in names]),
                                  np.array([models[n].offset for n in names]))

    for split in ("train", "validation"):
        frame, eras, typed, train_m, labels, checked = labels_for(cfg, split, last_train)
        label_pos = np.cumsum(train_m) - 1
        pred_m = (frame[base].notna().to_numpy() & typed & (eras <= last_pred)
                  & (split == "validation"))
        raw = frame[base].to_numpy(dtype=np.float64)
        audit[split] = {"rows": len(frame), "training_rows": int(train_m.sum()),
                        "prediction_candidate_rows": int(pred_m.sum()),
                        "label_parity_eras": checked}
        print(f"{split}: {train_m.sum():,} training rows; label parity eras {checked}",
              flush=True)
        row = 0
        for x in feature_batches(version, split, features):
            sl = slice(row, row + len(x))
            row += len(x)
            e, tm, pm = eras[sl], train_m[sl], pred_m[sl]
            if not (tm | pm).any():
                if e.min() > max(last_train, last_pred):
                    break
                continue
            for era in np.unique(e[tm | pm]):
                fitter.advance(era)
                refresh()
                m = e == era
                t = m & tm
                if t.any():
                    fitter.add_rows(era, x[t], labels[label_pos[sl][t]])
                q = m & pm
                if not q.any():
                    continue
                xq, ids = x[q], frame.index[sl][q]
                for key, a, b, endpoint in spans:
                    if not a <= era <= b:
                        continue
                    beta, offset = coef[endpoint]
                    values = np.empty((len(xq), len(names)), dtype=np.float32)
                    for i in range(0, len(xq), PREDICT_ROWS):
                        values[i:i + PREDICT_ROWS] = (
                            xq[i:i + PREDICT_ROWS].astype(np.float64) @ beta + offset)
                    pieces[key].append(pd.DataFrame(values, index=ids, columns=names)
                                       .assign(era=int(era), target=raw[sl][q]))
                    if key in parity and sum(len(p[0]) for p in parity[key]) < PARITY_ROWS:
                        parity[key].append((xq[:PARITY_ROWS].copy(), ids[:PARITY_ROWS]))
            if row % (16_384 * 40) == 0:
                print(f"  {split} row {row:,} era {e.max()}", flush=True)
        del frame, labels
    models = fitter.finish()
    preds = {key: pd.concat(v) for key, v in pieces.items()}
    parity = {key: (np.concatenate([p[0] for p in v]), np.concatenate([p[1] for p in v]))
              for key, v in parity.items()}
    return models, preds, parity, audit


def report(result, out, target, gap, grid_cols):
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL}
    cols = ["rondal", "azar", *grid_cols]

    def score(rows):
        scores = evaluate(rows, cols, target, refs)
        extra = evaluate(rows, ["azar", *grid_cols], target, {"rondal_contribution": "rondal"})
        for name, values in extra.items():
            scores[name]["rondal_contribution"] = values["rondal_contribution"]
        return scores

    metrics = {"all": score(result)}
    paired, era_scores, correlations = {}, [], {}
    for start, rows in result.groupby("window"):
        metrics[str(start)] = score(rows)
        measures = {"corr": per_era_corr(rows, ["rondal", "azar"], target)}
        for name, ref in refs.items():
            if rows[ref].notna().any():
                measures[name] = per_era_contribution(rows, ["rondal", "azar"], target, ref)
        paired[str(start)] = {}
        for name, values in measures.items():
            values = values.loc[sorted(values.index, key=int)]
            paired[str(start)][name] = paired_block_ci(
                (values.azar - values.rondal).to_numpy(), values.index, block=gap)
            stacked = values.stack().rename(name).to_frame()
            stacked.index.names = ["era", "model"]
            era_scores.append(stacked.assign(window=start))
        correlations[str(start)] = {
            "azar_vs_rondal": float(prediction_correlation(rows, ["azar", "rondal"]).iloc[0, 1])}
        with_meta = rows.dropna(subset=[META_COL])
        if len(with_meta):
            c = prediction_correlation(with_meta, ["azar", "rondal", META_COL])
            correlations[str(start)].update(azar_vs_meta=float(c.loc["azar", META_COL]),
                                            rondal_vs_meta=float(c.loc["rondal", META_COL]))
    decision = screening_decision(metrics)
    pd.concat(era_scores).to_csv(out / "per_era.csv")
    prediction_correlation(result, cols).to_csv(out / "prediction_correlations.csv")
    for name, value in (("metrics", metrics), ("paired_vs_rondal", paired),
                        ("correlations", correlations)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2))
    return metrics, decision, paired, correlations


def write_report(out, metrics, decision, paired, correlations, folds, grid_cols):
    def fmt(m, key):
        return f"{m[key]['mean']:.6f}" if key in m else "—"

    lines = ["# Azar development results", "",
             "Inner-selected alternate-target residual ridge (a separate model) versus a",
             "walk-forward Rondal recreation. Windows were inspected by earlier research;",
             "CORR covers 90 eras and actual MMC only the latest 30.", "",
             "| Window | Selected target | Best three inner contributions |", "|---|---|---|"]
    for f in folds:
        top = sorted(f["inner_mean_contribution"].items(), key=lambda kv: -kv[1])[:3]
        lines.append(f"| {f['test_start']} | {f['selected']} | "
                     + ", ".join(f"{k}: {v:.5f}" for k, v in top) + " |")
    lines += ["", "| Model | CORR | MMC | BMC | Contribution beyond Rondal |",
              "|---|---:|---:|---:|---:|"]
    for name in ["rondal", "azar", *grid_cols]:
        m = metrics["all"][name]
        lines.append(f"| {name} | {fmt(m, 'corr')} | {fmt(m, 'mmc')} | {fmt(m, 'bmc')} | "
                     f"{fmt(m, 'rondal_contribution')} |")
    lines += ["", ("| Window | Rondal CORR | Azar CORR | Contribution beyond Rondal | "
                   "Azar MMC | Rondal MMC | Corr(Azar, Rondal) |"),
              "|---|---:|---:|---:|---:|---:|---:|"]
    for w in [k for k in metrics if k != "all"]:
        a, r = metrics[w]["azar"], metrics[w]["rondal"]
        lines.append(f"| {w} | {fmt(r, 'corr')} | {fmt(a, 'corr')} | "
                     f"{fmt(a, 'rondal_contribution')} | {fmt(a, 'mmc')} | {fmt(r, 'mmc')} | "
                     f"{correlations[w]['azar_vs_rondal']:.4f} |")
    lines += ["", f"Paired Azar - Rondal (13-era blocks): {json.dumps(paired)}", "",
              f"Decision: {json.dumps(decision)}", "",
              "Bootstrap intervals are exploratory (about 2.3 blocks per window). No deployment."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(cfg):
    started = time.time()
    gap, reserve = check_windows(cfg)
    if Path(cfg["name"]).name != cfg["name"] or not cfg["name"].startswith("dc_azar_"):
        raise ValueError("Invalid artifact name")
    d = cfg["data"]
    target, version, targets = d["target"], d["version"], d["candidate_targets"]
    if cfg["alpha"] != 10 or target not in targets or len(set(targets)) != len(targets):
        raise ValueError("Frozen design requires alpha=10 and Rondal's target as fallback")
    fallback = short(target)
    features = feature_set(version, d["feature_set"])
    plan, endpoints, _ = design(cfg, gap)
    windows = {}
    for s, w in plan.items():
        for k, inner in enumerate(w["inner"]):
            windows[("inner", s, k)] = inner
        windows[("outer", s)] = w["outer"]
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    sources = [Path(__file__).resolve(), ROOT / "scripts/run_taren.py",
               ROOT / "src/numerai_dev/azar.py", ROOT / "src/numerai_dev/taren.py",
               ROOT / "src/numerai_dev/evaluation.py", ROOT / "src/numerai_dev/targets.py",
               ROOT / "docs/azar_plan.md", ROOT / "tests/test_azar.py"]
    (out / "sources").mkdir()
    for path in sources:
        (out / "sources" / path.name).write_bytes(path.read_bytes())
    manifest = {"config": cfg, "features": features, "git": git_hash(), "status": "running",
                "holdout_start": reserve, "embargo": gap, "endpoints": endpoints,
                "windows": {"/".join(map(str, k)): v for k, v in windows.items()},
                "production_before": before,
                "metadata_sha256": digest(data_path(version, "features.json")),
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "folds": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        models, preds, parity, audit = stream(cfg, features, endpoints, windows)
        manifest["stream_audit"] = audit
        print(f"Streamed in {time.time() - started:.0f}s; peak RSS {peak_rss_gb():.1f} GB",
              flush=True)
        names = [short(t) for t in targets]
        alternates = [n for n in names if n != fallback]
        grid_cols = [f"grid_{n}" for n in alternates]
        meta = pd.read_parquet(data_path(version, "meta_model.parquet"), columns=[META_COL])
        bench = pd.read_parquet(data_path(version, "validation_benchmark_models.parquet"),
                                columns=[BENCHMARK_COL])
        taren = pd.read_parquet(ROOT / "artifacts/dc_taren_full_v1/validation_preds.parquet",
                                columns=["rondal"])
        results = []
        for s, w in plan.items():
            scores = {n: [] for n in names}
            for k, (a, b, endpoint) in enumerate(w["inner"]):
                rows = preds[("inner", s, k)]
                assert rows.era.nunique() == b - a + 1, "incomplete inner window"
                assert all(models[endpoint][n].train_end <= endpoint < a - gap for n in names)
                contrib = mean_contribution(rows, alternates, "target", fallback)
                for n in alternates:
                    scores[n].append(float(contrib[n]))
                scores[fallback].append(0.0)
            chosen = select_target(scores, fallback)
            a, b, endpoint = w["outer"]
            rows = preds[("outer", s)]
            assert rows.era.nunique() == b - a + 1, "incomplete outer window"
            assert all(models[endpoint][n].train_end <= endpoint < a - gap for n in names)
            frame = pd.DataFrame({"era": rows.era.astype(str), target: rows.target,
                                  "window": s}, index=rows.index)
            frame["azar"] = rows[chosen].astype(np.float64)
            frame["rondal"] = rows[fallback].astype(np.float64)
            for n, col in zip(alternates, grid_cols, strict=True):
                frame[col] = rows[n].astype(np.float64)
            assert np.isfinite(frame[["azar", "rondal", *grid_cols]]).all().all()
            check_rondal_parity(frame, taren)
            for _, era in frame.groupby("era"):
                assert era.azar.nunique() > 1
                validate_submission(era.azar.rank(pct=True).rename("prediction"), era.index)
            bundle = {"azar": models[endpoint][chosen], "rondal": models[endpoint][fallback],
                      "grid": models[endpoint], "features": features, "selected": chosen,
                      "inner": {e: models[e] for _, _, e in w["inner"]}}
            path = out / f"window_{s}.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            assert loaded["features"] == features
            px, pids = parity[("outer", s)]
            np.testing.assert_allclose(loaded["azar"].predict(px, batch_size=3071),
                                       frame.loc[pids, "azar"], rtol=1e-5, atol=1e-7)
            fold = {"test_start": s, "test_end": b, "train_end": endpoint, "selected": chosen,
                    "inner_windows": [{"validation_start": a_, "validation_end": b_,
                                       "train_end": e_,
                                       "model": models[e_][fallback].audit()}
                                      for a_, b_, e_ in w["inner"]],
                    "inner_scores": scores,
                    "inner_mean_contribution": {n: float(np.mean(v)) for n, v in scores.items()},
                    "outer_model": models[endpoint][chosen].audit(),
                    "rondal_parity_with_taren": True, "test_rows": len(frame),
                    "checks_passed": True}
            manifest["folds"].append(fold)
            results.append(frame)
            print(f"Window {s}: selected {chosen}; top inner "
                  f"{sorted(fold['inner_mean_contribution'].items(), key=lambda kv: -kv[1])[:3]}",
                  flush=True)
        result = pd.concat(results).join(meta, how="left").join(bench, how="left")
        result.to_parquet(out / "validation_preds.parquet")
        metrics, decision, paired, correlations = report(result, out, target, gap, grid_cols)
        decision["selected_targets"] = {f["test_start"]: f["selected"] for f in manifest["folds"]}
        decision["all_folds_selected_fallback"] = all(f["selected"] == fallback
                                                      for f in manifest["folds"])
        (out / "decision.json").write_text(json.dumps(decision, indent=2))
        write_report(out, metrics, decision, paired, correlations, manifest["folds"], grid_cols)
        manifest.update(status="complete", decision=decision, full_era_rows=len(result),
                        peak_rss_gb=round(peak_rss_gb(), 2))
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="azar_walk_forward",
                holdout_scored=False, decision=decision, peak_rss_gb=round(peak_rss_gb(), 2))
        print((out / "report.md").read_text(), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["seconds"] = round(time.time() - started, 1)
        manifest["production_after"] = production_snapshot()
        manifest["production_unchanged"] = before == manifest["production_after"]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        if not manifest["production_unchanged"]:
            raise RuntimeError("Protected production files changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(yaml.safe_load(Path(args.config).read_text()))
