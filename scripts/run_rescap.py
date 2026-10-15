"""Residual capacity and fixed-ensemble comparison (dc_rescap); see
docs/residual_capacity_ensemble_plan.md.

A separate research experiment. Never uploads, packages hosted models, or edits live
configuration; the four live models' files are only hashed before and after. Eras 1228+
are reserved: nothing here predicts or scores them.

Usage:
  python scripts/run_rescap.py timing configs/rescap_research.yaml
  python scripts/run_rescap.py forward configs/rescap_research.yaml --budget-hours H [--resume]
  python scripts/run_rescap.py evaluate configs/rescap_research.yaml
Add --smoke for the 64-feature integration check (separate artifact name).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from run_experiment import label_frame, peak_rss_gb
from run_mavone import environment, md_table
from run_taren import feature_batches
from run_wenek import production_snapshot

from numerai_dev.azar import residualize_many
from numerai_dev.data import (
    DATA_DIR,
    ParquetRows,
    data_path,
    era_key,
    feature_set,
    iter_features,
    load_index,
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
from numerai_dev.live import validate_submission
from numerai_dev.mavone import assert_complete, blend, heatmap_svg, training_mask
from numerai_dev.provenance import (
    PROVENANCE_VERSION,
    digest,
    identity,
    ids_hash,
    open_for_evaluation,
    open_for_forward,
)
from numerai_dev.rescap import (
    BLENDS,
    COMPARISONS,
    OUTPUTS,
    adds_beyond_both,
    advancement,
    check_recipes,
    evaluation_blocks,
)
from numerai_dev.taren import INF, EndpointFitter
from numerai_dev.targets import residualize
from numerai_dev.wenek import paired_block_ci

IMPLEMENTATION = ["scripts/run_rescap.py", "scripts/run_mavone.py", "scripts/run_experiment.py",
                  "scripts/run_taren.py", "scripts/run_wenek.py", "src/numerai_dev/rescap.py",
                  "src/numerai_dev/mavone.py", "src/numerai_dev/provenance.py",
                  "src/numerai_dev/data.py", "src/numerai_dev/evaluation.py",
                  "src/numerai_dev/azar.py", "src/numerai_dev/taren.py",
                  "src/numerai_dev/targets.py", "src/numerai_dev/live.py",
                  "src/numerai_dev/wenek.py"]
DATASET_FILES = ["features.json", "train.parquet", "validation.parquet",
                 "train_benchmark_models.parquet", "validation_benchmark_models.parquet",
                 "meta_model.parquet"]
SOURCES = [*IMPLEMENTATION, "configs/rescap_research.yaml",
           "docs/residual_capacity_ensemble_plan.md", "tests/test_rescap.py"]
SMOKE = {"name": "dc_rescap_smoke_v1", "features": 64, "trees": {"current": 20, "large": 40},
         "blocks": [1133, 1211]}
TIMING = {"current": 200, "large": 400, "predict_rows": 20_000, "block": 1211}
PARITY_ROWS = 512
TREES = {"current": "tree_current", "large": "tree_large"}


def check_cfg(cfg):
    for key in ("name", "timing_name"):
        if Path(cfg[key]).name != cfg[key] or not cfg[key].startswith("dc_rescap_"):
            raise ValueError(f"Invalid artifact name {cfg[key]}")
    d, s, t, r = cfg["data"], cfg["schedule"], cfg["trees"], cfg["rondal"]
    if d["scoring_target"] != "target_ender_60" or d["feature_set"] != "all":
        raise ValueError("Frozen design scores target_ender_60 with all features")
    if (s["gap"], s["block_eras"], t["stride"], t["anchor"]) != (13, 26, 4, 1):
        raise ValueError("Frozen design requires gap 13, 26-era blocks, stride 4, anchor 1")
    if t["residualize_against"] != [BENCHMARK_COL]:
        raise ValueError("Tree label must match Mavone's ender_resid")
    if r["residualize_against"] != [BENCHMARK_COL, "v53_lgbm_ender20"] or r["alpha"] != 10:
        raise ValueError("Rondal recreation must use its established recipe")
    if cfg["blend_weights"] != [0.5, 0.5]:
        raise ValueError("Blends are fixed 50/50")
    check_recipes(t["recipes"], cfg["num_threads"])
    evaluation_blocks(s["first_block_start"], s["last_block_start"], s["last_eval_era"])


def smoke_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["name"] = SMOKE["name"]
    for recipe, n in SMOKE["trees"].items():
        cfg["trees"]["recipes"][recipe]["params"]["n_estimators"] = n
    return cfg


def prepare(cfg, smoke):
    check_cfg(cfg)
    if smoke:
        cfg = smoke_cfg(cfg)
    features = feature_set(cfg["data"]["version"], cfg["data"]["feature_set"])
    if smoke:
        features = features[:SMOKE["features"]]
    s = cfg["schedule"]
    blocks = evaluation_blocks(s["first_block_start"], s["last_block_start"], s["last_eval_era"],
                               s["block_eras"], s["gap"])
    if smoke:
        blocks = [b for b in blocks if b["start"] in SMOKE["blocks"]]
    return cfg, features, blocks


def current_identity(cfg, features):
    return identity(cfg, features, IMPLEMENTATION, DATASET_FILES, ROOT,
                    DATA_DIR / cfg["data"]["version"])


# ----------------------------------------------------------------- tree members

def tree_frames(cfg):
    """Mavone's ender_resid label on every eligible row with a benchmark prediction."""
    version, target = cfg["data"]["version"], cfg["data"]["scoring_target"]
    refs = cfg["trees"]["residualize_against"]
    frames, audit = {}, {}
    rng = np.random.default_rng(0)
    for split in ("train", "validation"):
        f = load_index(version, split, [target])
        n = len(f)
        f = f.join(pd.read_parquet(data_path(version, f"{split}_benchmark_models.parquet"),
                                   columns=refs), how="left")
        if len(f) != n or f.index.has_duplicates:
            raise ValueError(f"{split}: benchmark join changed or duplicated rows")
        f["era_n"] = f.era.map(era_key)
        f["eligible"] = f.data_type.isin(["train", "validation"]).to_numpy()
        has = (f[[target, *refs]].notna().all(axis=1) & f.eligible).to_numpy()
        f["resid_label"] = np.nan
        f.loc[has, "resid_label"] = residualize_many(
            f.loc[has, [target]].to_numpy(), f.loc[has, refs].to_numpy(),
            f.loc[has, "era_n"].to_numpy())[:, 0]
        checked = sorted(rng.choice(f.loc[has, "era"].unique(), 2, replace=False).tolist())
        for era in checked:
            rows = f.loc[has & f.era.eq(era).to_numpy()]
            np.testing.assert_allclose(rows.resid_label,
                                       np.asarray(residualize(rows, target, refs)).ravel(),
                                       atol=1e-6)
        audit[split] = {"rows": n, "resid_rows": int(has.sum()), "label_parity_eras": checked}
        frames[split] = f
    return frames, audit


def tree_rows(cfg, features, frames, fit_end):
    """Identical training rows for both recipes: eligible, matured, global stride-4."""
    version, target = cfg["data"]["version"], cfg["data"]["scoring_target"]
    t = cfg["trees"]
    seqs, labels, ids, eras = [], [], [], []
    for split, f in frames.items():
        m = training_mask(f.era_n, f[[target]], f.eligible, fit_end, t["stride"], t["anchor"])
        if m.any():
            seqs.append(ParquetRows(version, split, features, m))
            labels.append(f.resid_label.to_numpy()[m])
            ids.append(np.asarray(f.index[m]))
            eras.append(f.era_n.to_numpy()[m])
    r, eras, ids = np.concatenate(labels), np.concatenate(eras), np.concatenate(ids)
    if eras.max() > fit_end:
        raise ValueError("training rows beyond the fit endpoint")
    if not (eras % t["stride"] == t["anchor"] % t["stride"]).all():
        raise ValueError("training eras off the global stride")
    w = np.isfinite(r).astype(np.float64)
    return seqs, np.where(np.isfinite(r), r, 0.0), w, ids, eras


def build_dataset(seqs, label, weight):
    ds = lgb.Dataset(seqs, label=label, free_raw_data=True).construct()
    ds.set_weight(weight)
    return ds


def train(params, ds):
    p = dict(params)
    rounds = p.pop("n_estimators")
    return lgb.train(p, ds, num_boost_round=rounds)


# ----------------------------------------------------------------- Rondal recreation

def rondal_pass(cfg, features, blocks):
    """One era-ordered pass: ridge at each block endpoint (solved before later eras
    arrive); every evaluation row is predicted by its own block's endpoint."""
    d, r = cfg["data"], cfg["rondal"]
    version, target = d["version"], d["scoring_target"]
    endpoints = [b["fit_end"] for b in blocks]
    fit_end_of = {e: b["fit_end"] for b in blocks for e in range(b["start"], b["eval_end"] + 1)}
    last_train, last_pred = max(endpoints), max(fit_end_of)
    fitter = EndpointFitter(len(features), [INF], endpoints, r["alpha"])
    pieces, audit, parity = [], {}, None
    for split in ("train", "validation"):
        frame = label_frame(version, split, target, list(r["residualize_against"]))
        eras = frame.era.map(era_key).to_numpy()
        typed = (frame.data_type.eq("validation").to_numpy() if split == "validation"
                 else np.ones(len(frame), dtype=bool))
        train_m = frame.label.notna().to_numpy() & typed & (eras <= last_train)
        pred_m = (np.isin(eras, list(fit_end_of)) & typed if split == "validation"
                  else np.zeros(len(frame), dtype=bool))
        labels = frame.label.to_numpy(dtype=np.float64)
        audit[split] = {"training_rows": int(train_m.sum()), "prediction_rows": int(pred_m.sum())}
        print(f"rondal {split}: {train_m.sum():,} training rows, {pred_m.sum():,} prediction rows",
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
                q = (e == era) & pm
                if q.any():
                    model = fitter.models[fit_end_of[era]][INF]
                    if not model.train_end <= fit_end_of[era] < era - 13:
                        raise ValueError(f"era {era}: Rondal fit violates the maturity gap")
                    xq = x[q]
                    pieces.append(pd.DataFrame({"rondal": model.predict(xq), "era_n": int(era)},
                                               index=frame.index[sl][q]))
                    if parity is None:
                        parity = (xq[:PARITY_ROWS].copy(), fit_end_of[era])
                t = (e == era) & tm
                if t.any():
                    fitter.add_rows(era, x[t], labels[sl][t])
        del frame
        gc.collect()
    models = {int(k): v[INF] for k, v in fitter.finish().items()}
    preds = pd.concat(pieces)
    if preds.index.has_duplicates:
        raise ValueError("duplicate Rondal prediction ids")
    px, pend = parity
    a, b = models[pend].predict(px, batch_size=8192), models[pend].predict(px, batch_size=7)
    if not np.allclose(a, b, atol=1e-10):
        raise ValueError("Rondal batch-size parity failed")
    audit["models"] = {str(k): m.audit() for k, m in models.items()}
    return preds, models, audit


def check_rondal_parity(preds, blocks):
    """Endpoint-1119 Rondal must reproduce dc_taren_full_v1's walk-forward Rondal."""
    path = ROOT / "artifacts/dc_taren_full_v1/validation_preds.parquet"
    first = next(b for b in blocks if b["fit_end"] == 1119)
    taren = pd.read_parquet(path, columns=["era", "rondal"])
    taren = taren[taren.era.astype(int).between(first["start"], first["end"])]
    ours = preds.loc[preds.era_n.between(first["start"], first["end"]), "rondal"]
    if not ours.index.sort_values().equals(taren.index.sort_values()):
        raise ValueError("Rondal parity rows differ from dc_taren_full_v1")
    if not np.allclose(ours, taren.loc[ours.index, "rondal"], rtol=1e-5, atol=1e-7):
        raise ValueError("Rondal recreation does not match dc_taren_full_v1")
    return {"rows": len(ours), "max_abs_diff": float((ours - taren.loc[ours.index, "rondal"])
                                                     .abs().max())}


# ----------------------------------------------------------------- forward run

def save_manifest(out, manifest):
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def predict_block(cfg, features, frame, block, boosters):
    version = cfg["data"]["version"]
    mask = (frame.data_type.eq("validation")
            & frame.era_n.between(block["start"], block["eval_end"])).to_numpy()
    rows = np.flatnonzero(mask)
    names = list(boosters)
    out = np.full((len(rows), len(names)), np.nan)
    parity = None
    for pos, xb in iter_features(version, "validation", features, mask):
        out[np.searchsorted(rows, pos)] = np.column_stack([boosters[n].predict(xb) for n in names])
        if parity is None:
            parity = (xb[:PARITY_ROWS].copy(), frame.index[pos[:PARITY_ROWS]])
    preds = frame.iloc[rows][["era", "era_n", "data_type"]].join(
        pd.DataFrame(out, columns=names, index=frame.index[rows]))
    return preds, parity


def mavone_parity(block, preds):
    """Diagnostic only: recipe A here vs Mavone's ender_resid (same label, rows, endpoint)."""
    path = ROOT / f"artifacts/dc_mavone_research_v1/blocks/{block['start']}/preds.parquet"
    if not path.exists():
        return None
    old = pd.read_parquet(path, columns=["ender_resid"]).reindex(preds.index)
    diff = (preds.tree_current - old.ender_resid).abs()
    return {"rows": int(old.ender_resid.notna().sum()), "max_abs_diff": float(diff.max())}


def run_forward(cfg, budget_hours, resume=False, smoke=False):
    started = time.time()
    if not budget_hours or budget_hours <= 0:
        raise ValueError("--budget-hours is required (the approved compute budget)")
    cfg, features, blocks = prepare(cfg, smoke)
    out = ROOT / "artifacts" / cfg["name"]
    ident = current_identity(cfg, features)
    manifest = open_for_forward(out, ident, resume)
    if manifest is None:
        out.mkdir(exist_ok=False)
        (out / "sources").mkdir()
        for p in SOURCES:
            shutil.copy2(ROOT / p, out / "sources" / Path(p).name)
        manifest = {"provenance_version": PROVENANCE_VERSION, **ident, "features": features,
                    "git": git_hash(), "blocks": blocks, "environment": environment(),
                    "label_availability": "retrospective forward-label simulation: era e "
                                          "labels assumed available at era e + 13",
                    "completed_blocks": {}, "checkpoint_sha256": {}, "model_seconds": {},
                    "runs": []}
    spent_before = sum(r.get("seconds", 0) for r in manifest["runs"])
    budget = budget_hours * 3600
    before = production_snapshot()
    run = {"started": datetime.now().isoformat(timespec="seconds"), "resume": resume,
           "budget_hours": budget_hours, "production_before": before}
    manifest["runs"].append(run)
    manifest["status"] = "running"
    save_manifest(out, manifest)

    def over_budget(stage):
        spent = spent_before + time.time() - started
        if spent > budget:
            manifest["status"] = "stopped_budget"
            run["stopped_before"] = stage
            print(f"Budget of {budget_hours} h exhausted ({spent / 3600:.2f} h) before {stage}; "
                  "stopping. Resume only with explicit approval.", flush=True)
            return True
        return False

    try:
        if "rondal" not in manifest:
            if over_budget("rondal"):
                return
            t0 = time.time()
            preds, models, audit = rondal_pass(cfg, features, blocks)
            if not smoke:
                audit["taren_parity"] = check_rondal_parity(preds, blocks)
            (out / "rondal").mkdir()
            preds.to_parquet(out / "rondal" / "preds.parquet")
            joblib.dump({"models": models, "features": features}, out / "rondal" / "models.joblib")
            for name in ("preds.parquet", "models.joblib"):
                manifest["checkpoint_sha256"][f"rondal/{name}"] = digest(out / "rondal" / name)
            audit.update(seconds=round(time.time() - t0, 1), peak_rss_gb=round(peak_rss_gb(), 2))
            manifest["rondal"] = audit
            save_manifest(out, manifest)
            print(f"rondal done in {audit['seconds']}s", flush=True)
        rondal = pd.read_parquet(out / "rondal" / "preds.parquet")
        frames, label_audit = tree_frames(cfg)
        manifest["label_audit"] = label_audit
        recipes = cfg["trees"]["recipes"]
        for block in blocks:
            key = str(block["start"])
            if key in manifest["completed_blocks"]:
                print(f"block {key}: complete, skipping", flush=True)
                continue
            bdir = out / "blocks" / key
            bdir.mkdir(parents=True, exist_ok=True)
            seqs, label, weight, ids, eras = tree_rows(cfg, features, frames, block["fit_end"])
            ds = None
            boosters, seconds = {}, manifest["model_seconds"].setdefault(key, {})
            for recipe, col in TREES.items():
                ck = f"blocks/{key}/{col}.txt"
                if ck in manifest["checkpoint_sha256"]:
                    boosters[col] = lgb.Booster(model_file=str(out / ck))
                    continue
                if over_budget(f"block {key} {recipe}"):
                    return
                if ds is None:
                    t0 = time.time()
                    ds = build_dataset(seqs, label, weight)
                    seconds["dataset"] = round(time.time() - t0, 1)
                    print(f"block {key}: {ds.num_data():,} rows, eras {eras.min()}-{eras.max()}, "
                          f"peak RSS {peak_rss_gb():.1f} GB", flush=True)
                t0 = time.time()
                boosters[col] = train(recipes[recipe]["params"], ds)
                boosters[col].save_model(str(out / ck))
                seconds[recipe] = round(time.time() - t0, 1)
                manifest["checkpoint_sha256"][ck] = digest(out / ck)
                save_manifest(out, manifest)
                print(f"block {key} {recipe}: {seconds[recipe]}s, peak RSS {peak_rss_gb():.1f} GB",
                      flush=True)
            del ds, seqs
            gc.collect()
            t0 = time.time()
            preds, (px, pids) = predict_block(cfg, features, frames["validation"], block, boosters)
            seconds["predict"] = round(time.time() - t0, 1)
            for col in boosters:
                loaded = lgb.Booster(model_file=str(bdir / f"{col}.txt"))
                for got in (loaded.predict(px),
                            np.concatenate([loaded.predict(px[:7]), loaded.predict(px[7:])])):
                    if not np.allclose(got, preds.loc[pids, col], atol=1e-10):
                        raise ValueError(f"block {key} {col}: reload/batch parity failed")
            preds = preds.join(rondal[["rondal"]], how="left")
            assert_complete(preds, [*TREES.values(), "rondal"], preds.index)
            if not np.isfinite(preds[[*TREES.values(), "rondal"]].to_numpy()).all():
                raise ValueError(f"block {key}: non-finite predictions")
            preds = preds.assign(block_start=block["start"], fit_end=block["fit_end"],
                                 label_available_by_era=block["label_available_by_era"])
            preds.index.name = "id"
            preds.to_parquet(bdir / "preds.parquet")
            files = {n: digest(bdir / n) for n in
                     ["preds.parquet", *[f"{c}.txt" for c in TREES.values()]]}
            manifest["completed_blocks"][key] = {
                "block": block, "files_sha256": files, "train_rows": len(label),
                "train_eras": len(np.unique(eras)), "train_ids_sha256": ids_hash(ids),
                "zero_weight_rows": int((weight == 0).sum()), "pred_rows": len(preds),
                "pred_eras": int(preds.era.nunique()), "seconds": dict(seconds),
                "peak_rss_gb": round(peak_rss_gb(), 2),
                "mavone_parity": None if smoke else mavone_parity(block, preds)}
            save_manifest(out, manifest)
            print(f"block {key}: done; {json.dumps(seconds)}", flush=True)
        done = all(str(b["start"]) in manifest["completed_blocks"] for b in blocks)
        manifest["status"] = "forward_complete" if done else "partial"
    except Exception as exc:
        manifest["status"] = "failed"
        run["error"] = repr(exc)
        raise
    finally:
        run["seconds"] = round(time.time() - started, 1)
        run["production_after"] = production_snapshot()
        run["production_unchanged"] = before == run["production_after"]
        save_manifest(out, manifest)
        if not run["production_unchanged"]:
            raise RuntimeError("Protected production files changed")


# ----------------------------------------------------------------- bounded timing

def run_timing(cfg, smoke=False):
    """Seconds per tree and memory on the largest block's training rows. Models are
    discarded; nothing is scored."""
    cfg, features, blocks = prepare(cfg, smoke)
    out = ROOT / "artifacts" / (cfg["timing_name"] + ("_smoke" if smoke else ""))
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    block = next(b for b in blocks if b["start"] == TIMING["block"])
    frames, _ = tree_frames(cfg)
    t, target = cfg["trees"], cfg["data"]["scoring_target"]
    rows_by_block = {b["start"]: sum(int(training_mask(f.era_n, f[[target]], f.eligible,
                                                       b["fit_end"], t["stride"],
                                                       t["anchor"]).sum())
                                     for f in frames.values()) for b in blocks}
    eval_rows = {b["start"]: int((frames["validation"].data_type.eq("validation")
                                  & frames["validation"].era_n.between(b["start"], b["eval_end"]))
                                 .sum()) for b in blocks}
    seqs, label, weight, _, _ = tree_rows(cfg, features, frames, block["fit_end"])
    t0 = time.time()
    ds = build_dataset(seqs, label, weight)
    result = {"block": block["start"], "train_rows": ds.num_data(),
              "dataset_seconds": round(time.time() - t0, 1), "rows_by_block": rows_by_block,
              "eval_rows_by_block": eval_rows,
              "peak_rss_after_dataset_gb": round(peak_rss_gb(), 2), "recipes": {},
              "environment": environment()}
    val = frames["validation"]
    pmask = (val.data_type.eq("validation")
             & val.era_n.between(block["start"], block["eval_end"])).to_numpy()
    pmask[np.flatnonzero(pmask)[TIMING["predict_rows"]:]] = False
    xp = np.concatenate([x for _, x in iter_features(cfg["data"]["version"], "validation",
                                                     features, pmask)])
    for recipe, n in (("current", TIMING["current"]), ("large", TIMING["large"])):
        params = {**cfg["trees"]["recipes"][recipe]["params"], "n_estimators": n}
        t0 = time.time()
        booster = train(params, ds)
        fit = time.time() - t0
        t0 = time.time()
        booster.predict(xp)
        pred = time.time() - t0
        result["recipes"][recipe] = {"trees_timed": n, "fit_seconds": round(fit, 1),
                                     "seconds_per_tree": fit / n,
                                     "predict_seconds_per_tree_per_1k_rows":
                                         pred / n / (len(xp) / 1000),
                                     "peak_rss_gb": round(peak_rss_gb(), 2)}
        print(f"{recipe}: {n} trees in {fit:.1f}s", flush=True)
        del booster
    result["estimate"] = estimate(cfg, result, blocks)
    if production_snapshot() != before:
        raise RuntimeError("Protected production files changed")
    (out / "timing.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["estimate"], indent=2), flush=True)


def estimate(cfg, timing, blocks):
    """Linear extrapolation from the timed block: per-tree time scales with rows."""
    ref = timing["rows_by_block"][timing["block"]]
    per_block, total = {}, 0.0
    for b in blocks:
        scale = timing["rows_by_block"][b["start"]] / ref
        n_pred = timing["eval_rows_by_block"][b["start"]]
        sec = timing["dataset_seconds"] * scale
        for recipe, t in timing["recipes"].items():
            trees = cfg["trees"]["recipes"][recipe]["params"]["n_estimators"]
            sec += trees * t["seconds_per_tree"] * scale
            sec += trees * t["predict_seconds_per_tree_per_1k_rows"] * n_pred / 1000
        per_block[b["start"]] = round(sec / 3600, 2)
        total += sec
    rondal, evaluation = 15 * 60, 5 * 60
    return {"hours_by_block": per_block, "rondal_hours": rondal / 3600,
            "evaluation_hours": evaluation / 3600,
            "total_hours": round((total + rondal + evaluation) / 3600, 2),
            "peak_rss_gb": max(t["peak_rss_gb"] for t in timing["recipes"].values())}


# ----------------------------------------------------------------- evaluation

def final_outputs(preds):
    eras = preds.era.to_numpy()
    out = pd.DataFrame(index=preds.index)
    for col in ("tree_current", "tree_large", "rondal"):
        out[col] = blend(preds[[col]], {col: 1.0}, eras)
    for name, tree in BLENDS.items():
        out[name] = blend(preds[[tree, "rondal"]], {tree: 0.5, "rondal": 0.5}, eras)
    return out


def paired(per_era, a, b, blocks):
    out = {}
    for k, v in per_era.items():
        d = (v[a] - v[b]).dropna()
        eras = d.index.astype(int)
        by_block = {str(bl["start"]): float(d[(eras >= bl["start"]) & (eras <= bl["eval_end"])]
                                            .mean()) for bl in blocks}
        contiguous = bool(np.all(np.diff(eras) == 1))
        out[k] = {"mean": float(d.mean()), "std": float(d.std()), "n_eras": len(d),
                  "pct_positive": float((d > 0).mean()), "by_block": by_block,
                  "worst_block": min(by_block.values()),
                  "ci_13": paired_block_ci(d.to_numpy(), eras, block=13)["ci"] if contiguous
                  else None,
                  "ci_26": paired_block_ci(d.to_numpy(), eras, block=26)["ci"] if contiguous
                  else None}
    return out


def run_evaluate(cfg, smoke=False):
    started = time.time()
    cfg, features, blocks = prepare(cfg, smoke)
    version, target = cfg["data"]["version"], cfg["data"]["scoring_target"]
    out = ROOT / "artifacts" / cfg["name"]
    manifest = open_for_evaluation(out, current_identity(cfg, features))
    ev = out / "evaluation"
    ev.mkdir(exist_ok=False)
    before = production_snapshot()
    preds = pd.concat([pd.read_parquet(out / "blocks" / str(b["start"]) / "preds.parquet")
                       for b in blocks])
    val = load_index(version, "validation", [target])
    val["era_n"] = val.era.map(era_key)
    in_eval = np.zeros(len(val), dtype=bool)
    for b in blocks:
        in_eval |= val.era_n.between(b["start"], b["eval_end"]).to_numpy()
    in_eval &= val.data_type.eq("validation").to_numpy()
    expected = val.index[in_eval]
    members = [*TREES.values(), "rondal"]
    assert_complete(preds, members, expected)
    preds = preds.loc[expected]
    for b in blocks:
        sel = preds.block_start.eq(b["start"])
        if not preds.loc[sel, "era_n"].between(b["start"], b["eval_end"]).all():
            raise ValueError(f"block {b['start']}: predictions outside its eras")
        if not (preds.loc[sel, "fit_end"] < b["start"] - 13).all():
            raise ValueError(f"block {b['start']}: fit endpoint violates the maturity gap")
    if preds.era_n.max() > 1227:
        raise ValueError("reserved eras present")
    outputs = final_outputs(preds)
    for _, era in outputs.groupby(preds.era):
        for col in OUTPUTS:
            validate_submission(era[col].rename("prediction"), era.index)
    scored = outputs.join(preds[["era", "era_n", "block_start"]])
    scored[target] = val.loc[scored.index, target]
    n = len(scored)
    scored = scored.join(pd.read_parquet(data_path(version, "meta_model.parquet"),
                                         columns=[META_COL]), how="left")
    scored = scored.join(pd.read_parquet(data_path(version, "validation_benchmark_models.parquet"),
                                         columns=[BENCHMARK_COL]), how="left")
    if len(scored) != n or scored[target].isna().any():
        raise ValueError("evaluation rows changed or lack resolved targets")
    preds.join(outputs.add_prefix("final_")).to_parquet(ev / "eval_preds.parquet")
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL}
    metrics = {"all": evaluate(scored, OUTPUTS, target, refs)}
    for start, rows in scored.groupby("block_start"):
        metrics[str(start)] = evaluate(rows, OUTPUTS, target, refs)
    per_era = {"corr": per_era_corr(scored, OUTPUTS, target)}
    for k, ref in refs.items():
        per_era[k] = per_era_contribution(scored, OUTPUTS, target, ref)
    per_era = {k: v.loc[sorted(v.index, key=int)] for k, v in per_era.items()}
    stacked = pd.concat({k: v.stack() for k, v in per_era.items()}, axis=1)
    stacked.index.names = ["era", "output"]
    stacked.to_csv(ev / "per_era.csv")
    comparisons, decisions = {}, {}
    for a, b in COMPARISONS:
        key = f"{a}_vs_{b}"
        comparisons[key] = paired(per_era, a, b, blocks)
        m = comparisons[key]["mmc"]
        decisions[key] = advancement(metrics["all"][a]["corr"]["mean"],
                                     metrics["all"][a]["mmc"]["mean"],
                                     metrics["all"][b]["corr"]["mean"], m["mean"],
                                     list(m["by_block"].values()))
    decisions["beyond_both"] = {blend_name: adds_beyond_both(decisions, blend_name)
                                for blend_name in BLENDS}
    pcorr = prediction_correlation(scored.dropna(subset=[META_COL]),
                                   [*OUTPUTS, META_COL, BENCHMARK_COL])
    pcorr.to_csv(ev / "prediction_rank_corr.csv")
    labels = {META_COL: "meta", BENCHMARK_COL: "bench"}
    (ev / "prediction_rank_corr.svg").write_text(heatmap_svg(
        pcorr.rename(index=labels, columns=labels), "Mean per-era prediction rank correlation"))
    score_corr = {k: v[OUTPUTS].corr() for k, v in per_era.items()}
    cost = {"rondal": {k: manifest["rondal"][k] for k in ("seconds", "peak_rss_gb")},
            "blocks": {k: {"seconds": v["seconds"], "peak_rss_gb": v["peak_rss_gb"],
                           "train_rows": v["train_rows"]}
                       for k, v in manifest["completed_blocks"].items()}}
    parity = {"rondal_vs_taren": manifest["rondal"].get("taren_parity"),
              "tree_current_vs_mavone": {k: v["mavone_parity"]
                                         for k, v in manifest["completed_blocks"].items()}}
    for name, value in (("metrics", metrics), ("comparisons", comparisons),
                        ("decisions", decisions), ("compute_cost", cost), ("parity", parity)):
        (ev / f"{name}.json").write_text(json.dumps(value, indent=2, default=str))
    write_report(ev, metrics, comparisons, decisions, pcorr.rename(index=labels, columns=labels),
                 score_corr, cost, parity, scored)
    if production_snapshot() != before:
        raise RuntimeError("Protected production files changed")
    if not smoke:
        log_run(cfg["name"], cfg, flatten(metrics["all"]), started, kind="rescap_forward",
                holdout_scored=False, decision=decisions, peak_rss_gb=round(peak_rss_gb(), 2))
    print((ev / "report.md").read_text(), flush=True)


def write_report(ev, metrics, comparisons, decisions, pcorr, score_corr, cost, parity, scored):
    def f(m, k, stat="mean", fmt="{:.4f}"):
        return fmt.format(m[k][stat]) if k in m else "—"

    lines = ["# dc_rescap: residual tree capacity and fixed Rondal blends", "",
             (f"Eras {int(scored.era_n.min())}-{int(scored.era_n.max())} "
              f"({scored.era.nunique()} eras), all previously inspected development data, "
              "retrospective forward-label simulation. No result here establishes production "
              "readiness. Reserved eras 1228+ were not predicted or scored."), "",
             ("| Output | CORR | CORR std | CORR % pos | Worst era CORR | MMC | MMC std | "
              "MMC % pos | Worst era MMC | BMC |"), "|---|" + "---:|" * 9]
    for c in OUTPUTS:
        m = metrics["all"][c]
        lines.append(f"| {c} | {f(m, 'corr')} | {f(m, 'corr', 'std')} | "
                     f"{f(m, 'corr', 'pct_positive', '{:.2f}')} | {f(m, 'corr', 'worst_era')} | "
                     f"{f(m, 'mmc')} | {f(m, 'mmc', 'std')} | "
                     f"{f(m, 'mmc', 'pct_positive', '{:.2f}')} | {f(m, 'mmc', 'worst_era')} | "
                     f"{f(m, 'bmc')} |")
    lines += ["", "## By block (CORR / MMC / BMC)", "",
              "| Block | " + " | ".join(OUTPUTS) + " |", "|---|" + "---|" * len(OUTPUTS)]
    for blk in [k for k in metrics if k != "all"]:
        lines.append(f"| {blk} | " + " | ".join(
            f"{f(metrics[blk][c], 'corr')} / {f(metrics[blk][c], 'mmc')} / "
            f"{f(metrics[blk][c], 'bmc')}" for c in OUTPUTS) + " |")
    lines += ["", "## Paired differences (identical eras)", "",
              ("| Comparison | Score | Mean | 95% CI (13-era) | 95% CI (26-era) | % pos | "
               "Worst block | By block |"), "|---|---|---:|---|---|---:|---:|---|"]
    for key, scores in comparisons.items():
        for k, v in scores.items():
            ci = (lambda c: f"[{c[0]:.4f}, {c[1]:.4f}]" if c else "—")
            lines.append(f"| {key} | {k} | {v['mean']:.4f} | {ci(v['ci_13'])} | "
                         f"{ci(v['ci_26'])} | {v['pct_positive']:.2f} | {v['worst_block']:.4f} | "
                         + ", ".join(f"{b}: {x:.4f}" for b, x in v["by_block"].items()) + " |")
    lines += ["", "## Development advancement rule", "",
              ("| Comparison | MMC > 0 | Paired MMC gain > 0 | Block majority | CORR ≥ 90% | "
               "Inadequate anchor | Advances |"), "|---|---|---|---|---|---|---|"]
    for key, d in decisions.items():
        if key == "beyond_both":
            continue
        lines.append(f"| {key} | {d['mmc_positive']} | {d['paired_gain_positive']} "
                     f"({d['paired_mmc_gain']:.4f}) | {d['block_majority']} "
                     f"({d['positive_blocks']}/{d['blocks']}) | {d['corr_floor_ok']} | "
                     f"{d['inadequate_anchor']} | **{d['advances']}** |")
    lines += ["", f"Blend adds value beyond both members: `{json.dumps(decisions['beyond_both'])}`",
              "", ("Confidence intervals spanning zero mean the difference is uncertain. Higher "
                   "CORR with lower MMC is a trade-off, not a win."), "",
              "## Prediction rank correlations (meta-model eras)", ""]
    pcorr.index.name = "output"
    lines += md_table(pcorr, "{:.3f}")
    lines += ["", "## Per-era MMC score correlations", ""]
    sc = score_corr["mmc"].copy()
    sc.index.name = "output"
    lines += md_table(sc, "{:.3f}")
    lines += ["", "## Compute and parity", "", f"```\n{json.dumps(cost, indent=2)}\n```", "",
              f"```\n{json.dumps(parity, indent=2)}\n```", "",
              "Stopped for review. No deployment, upload, staking or live change."]
    (ev / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["timing", "forward", "evaluate"])
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--budget-hours", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.command == "timing":
        run_timing(config, smoke=args.smoke)
    elif args.command == "forward":
        run_forward(config, args.budget_hours, resume=args.resume, smoke=args.smoke)
    else:
        run_evaluate(config, smoke=args.smoke)
    sys.exit(0)
