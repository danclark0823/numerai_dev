"""Frozen recency-weighted residual ridge research (dc_taren); see docs/taren_plan.md.

Never uploads, packages hosted models, or edits live configuration.

Usage: python scripts/run_taren.py configs/taren_full.yaml
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
import pyarrow.parquet as pq
import yaml
from run_experiment import label_frame, peak_rss_gb
from run_wenek import digest, production_snapshot
from threadpoolctl import threadpool_limits

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
from numerai_dev.splits import embargo_for
from numerai_dev.taren import (
    INF,
    EndpointFitter,
    half_life_label,
    inner_windows,
    screening_decision,
    select_half_life,
    window_corr,
)
from numerai_dev.wenek import paired_block_ci

BATCH_ROWS = 16_384
PREDICT_ROWS = 8_192
PARITY_ROWS = 512


def check_windows(cfg):
    d = cfg["data"]
    gap = embargo_for(d["target"])
    val = load_index(d["version"], "validation", [d["target"]])
    resolved = val.loc[val[d["target"]].notna() & val.data_type.eq("validation"), "era"]
    reserve = sorted(resolved.map(era_key).unique())[-cfg["holdout_eras"]]
    windows = [(s, s + cfg["test_eras"] - 1) for s in cfg["test_starts"]]
    if not windows or cfg["test_eras"] < 1 or any(e >= reserve - gap for _, e in windows):
        raise ValueError("Invalid windows or reserved holdout/embargo accessed")
    if any(b[0] <= a[1] for a, b in pairwise(windows)):
        raise ValueError("Windows must be ordered and disjoint")
    return gap, int(reserve)


def feature_batches(version, split, features):
    """int8 feature batches in file order; rejects missing or out-of-range bins."""
    pf = pq.ParquetFile(data_path(version, f"{split}.parquet"), pre_buffer=False,
                        buffer_size=1 << 24)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=features):
        if any(col.null_count for col in batch.columns):
            raise ValueError(f"Missing feature values in {split}")
        x = np.column_stack([col.to_numpy(zero_copy_only=False) for col in batch.columns])
        if not np.issubdtype(x.dtype, np.integer) and np.any(x != np.floor(x)):
            raise ValueError("Fractional feature bins")
        if x.min() < 0 or x.max() > 4:
            raise ValueError("Feature bins outside 0-4")
        yield x.astype(np.int8, copy=False)


def design(cfg, gap):
    """Every fit endpoint and the eras each prediction window needs."""
    windows = {}
    for s in cfg["test_starts"]:
        inner = inner_windows(s, gap, cfg["inner_eras"], cfg["inner_folds"])
        outer = (s, s + cfg["test_eras"] - 1, s - gap - 1)
        for start, end, train_end in [*inner, outer]:
            assert train_end < start - gap
        windows[s] = {"inner": inner, "outer": outer}
    endpoints = sorted({w[2] for v in windows.values() for w in [*v["inner"], v["outer"]]})
    eras = sorted({e for v in windows.values() for a, b, _ in [*v["inner"], v["outer"]]
                   for e in range(a, b + 1)})
    return windows, endpoints, eras


def stream(cfg, features, endpoints, pred_eras, half_lives, outer_ranges):
    """One pass over train then validation in era order: accumulate decayed statistics,
    solve each endpoint before later eras arrive, and predict needed rows with every
    model solved so far."""
    d = cfg["data"]
    version, target, refs = d["version"], d["target"], list(d["residualize_against"])
    fitter = EndpointFitter(len(features), half_lives, endpoints, cfg["alpha"])
    keys = [(e, h) for e in endpoints for h in half_lives]
    column = {k: i for i, k in enumerate(keys)}
    coef = np.zeros((len(features), len(keys)))
    offset = np.zeros(len(keys))
    solved = np.zeros(len(keys), dtype=bool)
    last_train, last_pred = max(endpoints), max(pred_eras)
    wanted = np.zeros(last_pred + 1, dtype=bool)
    wanted[pred_eras] = True
    pieces, parity = [], {s: [] for s in outer_ranges}
    audit = {}

    def refresh():
        for endpoint, models in fitter.models.items():
            for h, model in models.items():
                i = column[(endpoint, h)]
                if not solved[i]:
                    coef[:, i], offset[i], solved[i] = model.beta, model.offset, True

    for split in ("train", "validation"):
        frame = label_frame(version, split, target, refs)
        eras = frame.era.map(era_key).to_numpy()
        typed = (frame.data_type.eq("validation").to_numpy() if split == "validation"
                 else np.ones(len(frame), dtype=bool))
        train_m = frame.label.notna().to_numpy() & (eras <= last_train) & typed
        pred_m = (frame[target].notna().to_numpy() & typed
                  & (eras <= last_pred) & wanted[np.minimum(eras, last_pred)])
        if split == "train":
            pred_m[:] = False
        labels = frame.label.to_numpy(dtype=np.float64)
        raw = frame[target].to_numpy(dtype=np.float64)
        audit[split] = {"rows": len(frame), "training_rows": int(train_m.sum()),
                        "prediction_rows": int(pred_m.sum()),
                        "rows_missing_label": int(frame.label.isna().sum())}
        print(f"{split}: {train_m.sum():,} training rows, {pred_m.sum():,} prediction rows",
              flush=True)
        row = 0
        for x in feature_batches(version, split, features):
            sl = slice(row, row + len(x))
            row += len(x)
            e, tm, pm = eras[sl], train_m[sl], pred_m[sl]
            use = tm | pm
            if not use.any():
                if e.min() > max(last_train, last_pred):
                    break
                continue
            for era in np.unique(e[use]):
                fitter.advance(era)
                refresh()
                m = e == era
                t = m & tm
                if t.any():
                    fitter.add_rows(era, x[t], labels[sl][t])
                q = m & pm
                if q.any():
                    xq = x[q]
                    values = np.full((len(xq), len(keys)), np.nan, dtype=np.float32)
                    for i in range(0, len(xq), PREDICT_ROWS):
                        xb = xq[i:i + PREDICT_ROWS].astype(np.float64)
                        values[i:i + PREDICT_ROWS, solved] = xb @ coef[:, solved] + offset[solved]
                    ids = frame.index[sl][q]
                    pieces.append(pd.DataFrame(values, index=ids,
                                               columns=[f"{k[0]}_{half_life_label(k[1])}"
                                                        for k in keys])
                                  .assign(era=int(era), target=raw[sl][q]))
                    for s, (a, b) in outer_ranges.items():
                        if a <= era <= b and sum(len(p[0]) for p in parity[s]) < PARITY_ROWS:
                            parity[s].append((xq[:PARITY_ROWS].copy(), ids[:PARITY_ROWS]))
            if row % (BATCH_ROWS * 40) == 0:
                print(f"  {split} row {row:,} era {e.max()}", flush=True)
        del frame
    models = fitter.finish()
    preds = pd.concat(pieces)
    if not preds.index.is_unique:
        raise ValueError("Duplicate prediction IDs")
    parity = {s: (np.concatenate([p[0] for p in v]), np.concatenate([p[1] for p in v]))
              for s, v in parity.items() if v}
    return models, preds, parity, audit


def report(result, out, target, gap, grid_cols, corr_tolerance):
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL}
    finite = [c for c in grid_cols if c != "grid_hinf"]
    cols = ["rondal", "taren", *grid_cols]

    def score(rows):
        scores = evaluate(rows, cols, target, refs)
        extra = evaluate(rows, ["taren", *finite], target, {"rondal_contribution": "rondal"})
        for name, values in extra.items():
            scores[name]["rondal_contribution"] = values["rondal_contribution"]
        return scores

    metrics = {"all": score(result)}
    paired, era_scores, correlations = {}, [], {}
    for start, rows in result.groupby("window"):
        metrics[str(start)] = score(rows)
        measures = {"corr": per_era_corr(rows, ["rondal", "taren"], target)}
        for name, ref in refs.items():
            if rows[ref].notna().any():
                measures[name] = per_era_contribution(rows, ["rondal", "taren"], target, ref)
        paired[str(start)] = {}
        for name, values in measures.items():
            values = values.loc[sorted(values.index, key=int)]
            paired[str(start)][name] = paired_block_ci(
                (values.taren - values.rondal).to_numpy(), values.index, block=gap)
            stacked = values.stack().rename(name).to_frame()
            stacked.index.names = ["era", "model"]
            era_scores.append(stacked.assign(window=start))
        correlations[str(start)] = {
            "taren_vs_rondal": float(prediction_correlation(rows, ["taren", "rondal"]).iloc[0, 1])}
        with_meta = rows.dropna(subset=[META_COL])
        if len(with_meta):
            c = prediction_correlation(with_meta, ["taren", "rondal", META_COL])
            correlations[str(start)].update(taren_vs_meta=float(c.loc["taren", META_COL]),
                                            rondal_vs_meta=float(c.loc["rondal", META_COL]))
    decision = screening_decision(metrics, corr_tolerance)
    pd.concat(era_scores).to_csv(out / "per_era.csv")
    prediction_correlation(result, cols).to_csv(out / "prediction_correlations.csv")
    for name, value in (("metrics", metrics), ("decision", decision),
                        ("paired_vs_rondal", paired), ("correlations", correlations)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2))
    return metrics, decision, paired, correlations


def write_report(out, metrics, decision, paired, correlations, folds, grid_cols):
    def fmt(m, key):
        return f"{m[key]['mean']:.6f}" if key in m else "—"

    lines = ["# Taren development results", "",
             "Inner-selected recency-weighted residual ridge versus a walk-forward Rondal",
             "recreation (h = inf). Windows were inspected by earlier research; CORR covers",
             "90 eras and actual MMC only the latest 30.", "",
             "| Window | Selected h | Inner mean CORR by h |", "|---|---|---|"]
    for f in folds:
        inner = ", ".join(f"{k}: {np.mean(v):.5f}" for k, v in f["inner_scores"].items())
        lines.append(f"| {f['test_start']} | {f['selected']} | {inner} |")
    lines += ["", "| Model | CORR | MMC | BMC | Contribution beyond Rondal |",
              "|---|---:|---:|---:|---:|"]
    for name in ["rondal", "taren", *grid_cols]:
        m = metrics["all"][name]
        lines.append(f"| {name} | {fmt(m, 'corr')} | {fmt(m, 'mmc')} | {fmt(m, 'bmc')} | "
                     f"{fmt(m, 'rondal_contribution')} |")
    lines += ["", ("| Window | Rondal CORR | Taren CORR | Contribution beyond Rondal | "
                   "Taren MMC | Rondal MMC | Corr(Taren, Rondal) |"),
              "|---|---:|---:|---:|---:|---:|---:|"]
    for w in [k for k in metrics if k != "all"]:
        t, r = metrics[w]["taren"], metrics[w]["rondal"]
        lines.append(f"| {w} | {fmt(r, 'corr')} | {fmt(t, 'corr')} | "
                     f"{fmt(t, 'rondal_contribution')} | {fmt(t, 'mmc')} | {fmt(r, 'mmc')} | "
                     f"{correlations[w]['taren_vs_rondal']:.4f} |")
    lines += ["", f"Paired Taren - Rondal (13-era blocks): {json.dumps(paired)}", "",
              f"Decision: {json.dumps(decision)}", "",
              "Bootstrap intervals are exploratory (about 2.3 blocks per window). No deployment."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(cfg):
    started = time.time()
    gap, reserve = check_windows(cfg)
    if Path(cfg["name"]).name != cfg["name"] or not cfg["name"].startswith("dc_taren_"):
        raise ValueError("Invalid artifact name")
    half_lives = [float(h) for h in cfg["half_lives"]]
    if INF not in half_lives or cfg["alpha"] != 10:
        raise ValueError("Frozen design requires alpha=10 and an h=inf fallback")
    d = cfg["data"]
    target, version = d["target"], d["version"]
    features = feature_set(version, d["feature_set"])
    windows, endpoints, pred_eras = design(cfg, gap)
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    sources = [Path(__file__).resolve(), ROOT / "src/numerai_dev/taren.py",
               ROOT / "src/numerai_dev/evaluation.py", ROOT / "src/numerai_dev/targets.py",
               ROOT / "src/numerai_dev/models.py", ROOT / "scripts/run_experiment.py",
               ROOT / "docs/taren_plan.md", ROOT / "tests/test_taren.py"]
    (out / "sources").mkdir()
    for path in sources:
        (out / "sources" / path.name).write_bytes(path.read_bytes())
    manifest = {"config": cfg, "features": features, "git": git_hash(), "status": "running",
                "holdout_start": reserve, "embargo": gap, "endpoints": endpoints,
                "windows": {str(k): v for k, v in windows.items()},
                "production_before": before,
                "metadata_sha256": digest(data_path(version, "features.json")),
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "folds": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        outer_ranges = {s: v["outer"][:2] for s, v in windows.items()}
        models, preds, parity, audit = stream(cfg, features, endpoints, pred_eras,
                                              half_lives, outer_ranges)
        manifest["stream_audit"] = audit
        print(f"Streamed in {time.time() - started:.0f}s; {len(preds):,} predicted rows; "
              f"peak RSS {peak_rss_gb():.1f} GB", flush=True)
        meta = pd.read_parquet(data_path(version, "meta_model.parquet"), columns=[META_COL])
        bench = pd.read_parquet(data_path(version, "validation_benchmark_models.parquet"),
                                columns=[BENCHMARK_COL])
        grid_cols = [f"grid_{half_life_label(h)}" for h in half_lives]
        results = []
        for s, w in windows.items():
            scores = {h: [] for h in half_lives}
            for start, end, train_end in w["inner"]:
                rows = preds[(preds.era >= start) & (preds.era <= end)]
                assert rows.era.nunique() == end - start + 1, "incomplete inner window"
                for h in half_lives:
                    model = models[train_end][h]
                    assert model.train_end <= train_end < start - gap
                    scores[h].append(window_corr(rows[f"{train_end}_{half_life_label(h)}"],
                                                 rows.target, rows.era))
            chosen = select_half_life(scores)
            start, end, train_end = w["outer"]
            rows = preds[(preds.era >= start) & (preds.era <= end)]
            assert rows.era.nunique() == end - start + 1, "incomplete outer window"
            frame = pd.DataFrame({"era": rows.era.astype(str), target: rows.target,
                                  "window": s}, index=rows.index)
            frame["taren"] = rows[f"{train_end}_{half_life_label(chosen)}"].astype(np.float64)
            frame["rondal"] = rows[f"{train_end}_hinf"].astype(np.float64)
            for h, col in zip(half_lives, grid_cols, strict=True):
                frame[col] = rows[f"{train_end}_{half_life_label(h)}"].astype(np.float64)
            assert np.isfinite(frame[["taren", "rondal", *grid_cols]]).all().all()
            for h in half_lives:
                assert models[train_end][h].train_end <= train_end < start - gap
            for _, era in frame.groupby("era"):
                assert era.taren.nunique() > 1
                validate_submission(era.taren.rank(pct=True).rename("prediction"), era.index)
            bundle = {"taren": models[train_end][chosen], "rondal": models[train_end][INF],
                      "grid": models[train_end], "features": features,
                      "inner": {e: models[e] for _, _, e in w["inner"]}}
            path = out / f"window_{s}.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            assert loaded["features"] == features
            px, pids = parity[s]
            np.testing.assert_allclose(loaded["taren"].predict(px, batch_size=3071),
                                       frame.loc[pids, "taren"], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(loaded["rondal"].predict(px, batch_size=1000),
                                       frame.loc[pids, "rondal"], rtol=1e-5, atol=1e-7)
            fold = {"test_start": s, "test_end": end, "train_end": train_end,
                    "selected": half_life_label(chosen),
                    "inner_windows": [{"validation_start": a, "validation_end": b,
                                       "train_end": e, "models": {
                                           half_life_label(h): models[e][h].audit()
                                           for h in half_lives}} for a, b, e in w["inner"]],
                    "inner_scores": {half_life_label(h): v for h, v in scores.items()},
                    "outer_models": {half_life_label(h): models[train_end][h].audit()
                                     for h in half_lives},
                    "test_rows": len(frame), "checks_passed": True}
            manifest["folds"].append(fold)
            results.append(frame)
            print(f"Window {s}: selected {fold['selected']}; inner means "
                  f"{ {k: round(float(np.mean(v)), 5) for k, v in fold['inner_scores'].items()} }",
                  flush=True)
        result = pd.concat(results).join(meta, how="left").join(bench, how="left")
        result.to_parquet(out / "validation_preds.parquet")
        metrics, decision, paired, correlations = report(
            result, out, target, gap, grid_cols, cfg["corr_tolerance"])
        decision["selected_half_lives"] = {f["test_start"]: f["selected"]
                                           for f in manifest["folds"]}
        decision["all_folds_selected_inf"] = all(f["selected"] == "hinf"
                                                 for f in manifest["folds"])
        (out / "decision.json").write_text(json.dumps(decision, indent=2))
        write_report(out, metrics, decision, paired, correlations, manifest["folds"], grid_cols)
        manifest.update(status="complete", decision=decision, full_era_rows=len(result),
                        peak_rss_gb=round(peak_rss_gb(), 2))
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="taren_walk_forward",
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
