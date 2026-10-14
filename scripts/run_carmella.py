"""Frozen portfolio-residual research (dc_carmella); see docs/carmella_plan.md.

dc_carmella would be a separate model. Never uploads, packages hosted models, or
edits live configuration; Rondal and Tyndal are only recreated walk-forward.

Usage: python scripts/run_carmella.py configs/carmella_full.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from run_experiment import label_frame, peak_rss_gb
from run_taren import check_windows, feature_batches
from run_wenek import digest, production_snapshot
from threadpoolctl import threadpool_limits

from numerai_dev.azar import residualize_many
from numerai_dev.carmella import ForwardChain, block_geometry, screening_decision
from numerai_dev.data import ParquetRows, data_path, era_key, feature_set, iter_features
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
from numerai_dev.targets import residualize
from numerai_dev.wenek import paired_block_ci

PARITY_ROWS = 512
LABEL_PARITY_ERAS = 2


def check_rondal_parity(frame, taren):
    """The outer Rondal recreation must reproduce dc_taren_full_v1's exactly."""
    np.testing.assert_allclose(frame.rondal, taren.loc[frame.index, "rondal"],
                               rtol=1e-5, atol=1e-7)


def pass_one(cfg, features, outer):
    """Forward-chained Rondal predictions for every labeled row (NaN in warm-up), and
    outer-window Rondal recreations, from one era-ordered pass."""
    d = cfg["data"]
    version, target, refs = d["version"], d["target"], list(d["residualize_against"])
    last_train = max(e for _, _, e in outer.values())
    last_outer = max(b for _, b, _ in outer.values())
    frames = {}
    for split in ("train", "validation"):
        f = label_frame(version, split, target, refs)
        f["era_n"] = f.era.map(era_key).to_numpy()
        f["typed"] = (f.data_type.eq("validation") if split == "validation"
                      else pd.Series(True, index=f.index))
        frames[split] = f
    first = int(min(f.era_n[f.label.notna() & f.typed].min() for f in frames.values()))
    blocks = block_geometry(first, last_train, cfg["warmup_eras"], 13, cfg["block_eras"])
    chain = ForwardChain(len(features), blocks, cfg["alpha"],
                         [e for _, _, e in outer.values()])
    outer_rows = {s: [] for s in outer}
    for split, f in frames.items():
        eras = f.era_n.to_numpy()
        typed = f.typed.to_numpy()
        labels = f.label.to_numpy(dtype=np.float64)
        train_m = ~np.isnan(labels) & typed & (eras <= last_train)
        in_outer = np.zeros(len(f), dtype=bool)
        for a, b, _ in outer.values():
            in_outer |= (eras >= a) & (eras <= b)
        outer_m = (split == "validation") & typed & f[target].notna().to_numpy() & in_outer
        oof = np.full(len(f), np.nan)
        print(f"{split}: {train_m.sum():,} ridge training rows", flush=True)
        row = 0
        for x in feature_batches(version, split, features):
            sl = slice(row, row + len(x))
            row += len(x)
            e, tm, om = eras[sl], train_m[sl], outer_m[sl]
            if not (tm | om).any():
                if e.min() > max(last_train, last_outer):
                    break
                continue
            for era in np.unique(e[tm | om]):
                chain.advance(era)
                m = e == era
                t = m & tm
                if t.any():
                    model = chain.model_for(era)
                    if model is not None:
                        oof[sl.start + np.flatnonzero(t)] = model.predict(x[t])
                    chain.add_rows(era, x[t], labels[sl][t])
                q = m & om
                if q.any():
                    s = next(s for s, (a, b, _) in outer.items() if a <= era <= b)
                    model = chain.endpoint_model(outer[s][2])
                    assert model.train_end <= outer[s][2] < outer[s][0] - 13
                    outer_rows[s].append(pd.DataFrame(
                        {"rondal": model.predict(x[q])}, index=f.index[sl][q]))
            if row % (16_384 * 40) == 0:
                print(f"  {split} row {row:,} era {e.max()}", flush=True)
        f["oof"] = oof
    chain.finish()
    rondal = {s: pd.concat(v) for s, v in outer_rows.items()}
    return frames, blocks, chain, rondal


def add_labels(cfg, frames):
    """Carmella label: target residualized against both benchmarks and the forward-
    chained Rondal prediction; Tyndal label: `label` from label_frame (benchmarks only)."""
    d = cfg["data"]
    target, refs = d["target"], list(d["residualize_against"])
    audit = {}
    rng = np.random.default_rng(0)
    for split, f in frames.items():
        ok = f.oof.notna().to_numpy()
        f["carmella_label"] = np.nan
        sub = f.loc[ok]
        f.loc[ok, "carmella_label"] = residualize_many(
            sub[[target]].to_numpy(), sub[[*refs, "oof"]].to_numpy(), sub.era_n.to_numpy())[:, 0]
        checked = sorted(rng.choice(sub.era.unique(), LABEL_PARITY_ERAS, replace=False).tolist())
        for era in checked:
            rows = f.loc[ok & f.era.eq(era).to_numpy()]
            # targets.residualize centers float32 targets in float32 (~1e-8 rounding).
            expected = np.asarray(residualize(rows, target, [*refs, "oof"])).ravel()
            np.testing.assert_allclose(rows.carmella_label, expected, atol=1e-6)
        assert f.loc[ok, "label"].notna().all() and np.isfinite(f.loc[ok, "carmella_label"]).all()
        audit[split] = {"rows_with_forward_rondal": int(ok.sum()), "label_parity_eras": checked}
    pooled = pd.concat([f.loc[f.oof.notna(), ["era", "oof", target]] for f in frames.values()])
    corr = per_era_corr(pooled, ["oof"], target)["oof"]
    audit["forward_rondal_train_corr"] = {"mean": float(corr.mean()), "n_eras": len(corr)}
    return audit


def fit_fold(cfg, features, frames, endpoint):
    """Tyndal control and Carmella on identical stride-4 rows, from one binned Dataset."""
    version = cfg["data"]["version"]
    eligible = sorted({int(e) for f in frames.values()
                       for e in f.era_n[f.oof.notna() & (f.era_n <= endpoint)].unique()})
    chosen = set(eligible[::cfg["train_stride"]])
    seqs, y_t, y_c, used = [], [], [], set()
    for split, f in frames.items():
        mask = (f.oof.notna() & f.era_n.isin(chosen)).to_numpy()
        if not mask.any():
            continue
        seqs.append(ParquetRows(version, split, features, mask))
        y_t.append(f.label.to_numpy(np.float64)[mask])
        y_c.append(f.carmella_label.to_numpy(np.float64)[mask])
        used |= set(f.era_n[mask].unique())
    y_t, y_c = np.concatenate(y_t), np.concatenate(y_c)
    assert max(used) <= endpoint and np.isfinite(y_t).all() and np.isfinite(y_c).all()
    params = {"objective": "regression", **cfg["lgbm"]}
    rounds = params.pop("n_estimators")
    ds = lgb.Dataset(seqs, label=y_t, free_raw_data=True).construct()
    print(f"  dataset: {ds.num_data():,} rows, eras {min(used)}-{max(used)} "
          f"({len(used)}), peak RSS {peak_rss_gb():.1f} GB", flush=True)
    tyndal = lgb.train(params, ds, num_boost_round=rounds)
    ds.set_label(y_c)
    carmella = lgb.train(params, ds, num_boost_round=rounds)
    del ds, seqs
    gc.collect()
    audit = {"train_rows": len(y_t), "train_eras": len(used), "first_era": min(used),
             "last_era": max(used), "stride": cfg["train_stride"]}
    return tyndal, carmella, audit


def predict_outer(cfg, features, frame, window, boosters):
    version, target = cfg["data"]["version"], cfg["data"]["target"]
    a, b, _ = window
    mask = (frame.typed & frame[target].notna() & frame.era_n.between(a, b)).to_numpy()
    out = np.full((len(frame), len(boosters)), np.nan)
    parity = None
    for pos, xb in iter_features(version, "validation", features, mask):
        out[pos] = np.column_stack([m.predict(xb) for m in boosters])
        if parity is None:
            parity = (xb[:PARITY_ROWS].copy(), frame.index[pos[:PARITY_ROWS]])
    return pd.DataFrame(out[mask], index=frame.index[mask]), parity


def report(result, out, target, gap):
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL}
    cols = ["rondal", "tyndal", "portfolio", "carmella"]
    contribs = {"portfolio_contribution": "portfolio", "tyndal_contribution": "tyndal",
                "rondal_contribution": "rondal"}

    def score(rows):
        scores = evaluate(rows, cols, target, refs)
        for name, values in evaluate(rows, ["carmella"], target, contribs)["carmella"].items():
            if name in contribs:
                scores["carmella"][name] = values
        scores["tyndal"]["rondal_contribution"] = evaluate(
            rows, ["tyndal"], target, {"rondal_contribution": "rondal"})["tyndal"][
            "rondal_contribution"]
        return scores

    metrics = {"all": score(result)}
    paired, era_scores, correlations = {}, [], {}
    for start, rows in result.groupby("window"):
        metrics[str(start)] = score(rows)
        measures = {"corr": per_era_corr(rows, ["tyndal", "carmella"], target)}
        for name, ref in refs.items():
            if rows[ref].notna().any():
                measures[name] = per_era_contribution(rows, ["tyndal", "carmella"], target, ref)
        paired[str(start)] = {}
        for name, values in measures.items():
            values = values.loc[sorted(values.index, key=int)]
            paired[str(start)][name] = paired_block_ci(
                (values.carmella - values.tyndal).to_numpy(), values.index, block=gap)
            stacked = values.stack().rename(name).to_frame()
            stacked.index.names = ["era", "model"]
            era_scores.append(stacked.assign(window=start))
        c = prediction_correlation(rows, ["carmella", "tyndal", "rondal"])
        correlations[str(start)] = {"carmella_vs_tyndal": float(c.loc["carmella", "tyndal"]),
                                    "carmella_vs_rondal": float(c.loc["carmella", "rondal"]),
                                    "tyndal_vs_rondal": float(c.loc["tyndal", "rondal"])}
        with_meta = rows.dropna(subset=[META_COL])
        if len(with_meta):
            c = prediction_correlation(with_meta, ["carmella", "tyndal", "rondal", META_COL])
            correlations[str(start)].update(
                {f"{k}_vs_meta": float(c.loc[k, META_COL]) for k in ["carmella", "tyndal", "rondal"]})
    decision = screening_decision(metrics)
    pd.concat(era_scores).to_csv(out / "per_era.csv")
    prediction_correlation(result, cols).to_csv(out / "prediction_correlations.csv")
    for name, value in (("metrics", metrics), ("paired_vs_tyndal", paired),
                        ("correlations", correlations)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2))
    return metrics, decision, paired, correlations


def write_report(out, metrics, decision, paired, correlations, label_audit):
    def fmt(m, key):
        return f"{m[key]['mean']:.6f}" if key in m else "—"

    lines = ["# Carmella development results", "",
             "LightGBM on target residualized against both benchmarks and a forward-chained",
             "Rondal prediction (a separate model), versus walk-forward Rondal and a matched",
             "Tyndal recreation. Windows were inspected by earlier research; CORR covers 90",
             "eras and actual MMC only the latest 30.", "",
             (f"Forward-chained Rondal CORR on its training eras: "
              f"{label_audit['forward_rondal_train_corr']}"), "",
             "| Model | CORR | MMC | BMC |", "|---|---:|---:|---:|"]
    for name in ["rondal", "tyndal", "portfolio", "carmella"]:
        m = metrics["all"][name]
        lines.append(f"| {name} | {fmt(m, 'corr')} | {fmt(m, 'mmc')} | {fmt(m, 'bmc')} |")
    lines += ["", ("| Window | Carmella CORR | Tyndal CORR | Beyond portfolio | Beyond Tyndal | "
                   "Beyond Rondal | Carmella MMC | Tyndal MMC | Corr(C, T) | Corr(C, R) |"),
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w in [k for k in metrics if k != "all"]:
        c, t = metrics[w]["carmella"], metrics[w]["tyndal"]
        lines.append(f"| {w} | {fmt(c, 'corr')} | {fmt(t, 'corr')} | "
                     f"{fmt(c, 'portfolio_contribution')} | {fmt(c, 'tyndal_contribution')} | "
                     f"{fmt(c, 'rondal_contribution')} | {fmt(c, 'mmc')} | {fmt(t, 'mmc')} | "
                     f"{correlations[w]['carmella_vs_tyndal']:.4f} | "
                     f"{correlations[w]['carmella_vs_rondal']:.4f} |")
    lines += ["", f"Paired Carmella - Tyndal (13-era blocks): {json.dumps(paired)}", "",
              f"Decision: {json.dumps(decision)}", "",
              "Bootstrap intervals are exploratory (about 2.3 blocks per window). No deployment."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(cfg):
    started = time.time()
    gap, reserve = check_windows(cfg)
    if Path(cfg["name"]).name != cfg["name"] or not cfg["name"].startswith("dc_carmella_"):
        raise ValueError("Invalid artifact name")
    if cfg["alpha"] != 10 or cfg["train_stride"] < 1:
        raise ValueError("Frozen design requires alpha=10")
    d = cfg["data"]
    version, target = d["version"], d["target"]
    features = feature_set(version, d["feature_set"])
    outer = {s: (s, s + cfg["test_eras"] - 1, s - gap - 1) for s in cfg["test_starts"]}
    out = ROOT / "artifacts" / cfg["name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    sources = [Path(__file__).resolve(), ROOT / "scripts/run_taren.py",
               ROOT / "scripts/run_experiment.py", ROOT / "src/numerai_dev/carmella.py",
               ROOT / "src/numerai_dev/taren.py", ROOT / "src/numerai_dev/azar.py",
               ROOT / "src/numerai_dev/data.py", ROOT / "src/numerai_dev/evaluation.py",
               ROOT / "src/numerai_dev/targets.py", ROOT / "docs/carmella_plan.md",
               ROOT / "tests/test_carmella.py"]
    (out / "sources").mkdir()
    for path in sources:
        (out / "sources" / path.name).write_bytes(path.read_bytes())
    manifest = {"config": cfg, "features": features, "git": git_hash(), "status": "running",
                "holdout_start": reserve, "embargo": gap,
                "outer": {str(k): v for k, v in outer.items()},
                "production_before": before,
                "metadata_sha256": digest(data_path(version, "features.json")),
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "folds": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        with threadpool_limits(limits=2, user_api="blas"):
            frames, blocks, chain, rondal = pass_one(cfg, features, outer)
            label_audit = add_labels(cfg, frames)
        manifest.update(blocks=blocks, label_audit=label_audit)
        print(f"Pass 1 done in {time.time() - started:.0f}s; {len(blocks)} blocks; "
              f"forward Rondal CORR {label_audit['forward_rondal_train_corr']}; "
              f"peak RSS {peak_rss_gb():.1f} GB", flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        meta = pd.read_parquet(data_path(version, "meta_model.parquet"), columns=[META_COL])
        bench = pd.read_parquet(data_path(version, "validation_benchmark_models.parquet"),
                                columns=[BENCHMARK_COL])
        taren = pd.read_parquet(ROOT / "artifacts/dc_taren_full_v1/validation_preds.parquet",
                                columns=["rondal"])
        val = frames["validation"]
        results = []
        for s, window in outer.items():
            fold_start = time.time()
            print(f"Window {s}: fitting on eras <= {window[2]}", flush=True)
            tyndal, carmella, audit = fit_fold(cfg, features, frames, window[2])
            assert audit["last_era"] <= window[2] < s - gap
            for name, booster in (("tyndal", tyndal), ("carmella", carmella)):
                booster.save_model(str(out / f"{name}_{s}.txt"))
            preds, (px, pids) = predict_outer(cfg, features, val, window, [tyndal, carmella])
            frame = pd.DataFrame({"era": val.loc[preds.index, "era"],
                                  target: val.loc[preds.index, target], "window": s},
                                 index=preds.index)
            frame["tyndal"], frame["carmella"] = preds[0], preds[1]
            frame["rondal"] = rondal[s].loc[frame.index, "rondal"]
            assert frame.era.map(era_key).nunique() == cfg["test_eras"], "incomplete window"
            eras = frame.era.to_numpy()
            frame["portfolio"] = (era_gaussian(frame.rondal.to_numpy(), eras)
                                  + era_gaussian(frame.tyndal.to_numpy(), eras)) / 2
            assert np.isfinite(frame[["tyndal", "carmella", "rondal"]]).all().all()
            check_rondal_parity(frame, taren)
            for _, era in frame.groupby("era"):
                assert era.carmella.nunique() > 1
                validate_submission(era.carmella.rank(pct=True).rename("prediction"), era.index)
            for name in ("tyndal", "carmella"):
                loaded = lgb.Booster(model_file=str(out / f"{name}_{s}.txt"))
                np.testing.assert_allclose(loaded.predict(px), frame.loc[pids, name],
                                           atol=1e-10)
            joblib.dump({"rondal": chain.endpoint_model(window[2]), "features": features},
                        out / f"rondal_{s}.joblib")
            audit.update(test_start=s, test_end=window[1], fit_endpoint=window[2],
                         test_rows=len(frame), seconds=round(time.time() - fold_start, 1),
                         peak_rss_gb=round(peak_rss_gb(), 2), checks_passed=True)
            manifest["folds"].append(audit)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            results.append(frame)
            print(f"Window {s}: done in {audit['seconds']}s, peak RSS {audit['peak_rss_gb']} GB",
                  flush=True)
        result = pd.concat(results).join(meta, how="left").join(bench, how="left")
        result.to_parquet(out / "validation_preds.parquet")
        metrics, decision, paired, correlations = report(result, out, target, gap)
        (out / "decision.json").write_text(json.dumps(decision, indent=2))
        write_report(out, metrics, decision, paired, correlations, label_audit)
        manifest.update(status="complete", decision=decision, full_era_rows=len(result),
                        peak_rss_gb=round(peak_rss_gb(), 2))
        log_run(cfg["name"], cfg, flatten(metrics), started, kind="carmella_walk_forward",
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
    run(yaml.safe_load(Path(args.config).read_text()))
