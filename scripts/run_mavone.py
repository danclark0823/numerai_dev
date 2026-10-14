"""Frozen multi-target ensemble research (dc_mavone): Phases 0-3 of
docs/multi_target_ensemble_plan.md.

dc_mavone would be a separate model. Never uploads, packages hosted models, or edits
live configuration; the four live models' files are only hashed before and after.

Usage:
  python scripts/run_mavone.py audit configs/mavone_research.yaml
  python scripts/run_mavone.py forward configs/mavone_research.yaml [--resume] [--smoke]
  python scripts/run_mavone.py evaluate configs/mavone_research.yaml [--smoke]
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numerai_tools
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml
from run_experiment import peak_rss_gb
from run_wenek import digest, production_snapshot

from numerai_dev.azar import residualize_many
from numerai_dev.data import (
    DATA_DIR,
    ParquetRows,
    data_path,
    era_key,
    feature_set,
    iter_features,
    load_features,
    load_index,
    load_metadata,
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
from numerai_dev.live import neutralize, validate_submission
from numerai_dev.mavone import (
    assert_complete,
    blend,
    block_schedule,
    exclusions,
    gaussian_rank,
    heatmap_svg,
    hypothesis1_decision,
    incremental_sets,
    longest_run,
    mmc_gate,
    stride_mask,
    training_mask,
)
from numerai_dev.targets import residualize
from numerai_dev.wenek import paired_block_ci

RECIPE = {"objective": "regression", "n_estimators": 2000, "learning_rate": 0.01,
          "max_depth": 5, "num_leaves": 31, "colsample_bytree": 0.1, "random_state": 42}
PARITY_ROWS = 512
SMOKE = {"name": "dc_mavone_smoke_v1", "features": 64, "n_estimators": 20,
         "blocks": [1133, 1211]}
SOURCES = ["scripts/run_mavone.py", "src/numerai_dev/mavone.py", "src/numerai_dev/data.py",
           "src/numerai_dev/evaluation.py", "src/numerai_dev/azar.py",
           "src/numerai_dev/targets.py", "src/numerai_dev/live.py",
           "configs/mavone_research.yaml", "docs/multi_target_ensemble_plan.md",
           "tests/test_mavone.py"]
# Every period below influenced earlier decisions or live training; none is fresh.
PERIODS_USED = [
    {"eras": "1-574", "use": "training for every earlier model", "status": "training"},
    {"eras": "587-1162", "use": "validation scoring for round 1-3 LightGBM, ridge and "
     "target sweeps (including jasper, rowan, teager2b) and ensemble selection",
     "status": "development"},
    {"eras": "850-879, 1000-1029, 1133-1162", "use": "walk-forward windows: diversifiers, "
     "Wenek, Bendonal, Alscot, Hesia, Taren, Azar, Carmella", "status": "development"},
    {"eras": "1176-1227", "use": "52-era holdout, inspected 3 times; chose recent_* live "
     "members", "status": "development"},
    {"eras": "1-1162", "use": "training of recent_* live members", "status": "training"},
    {"eras": "1-1227", "use": "training of final_ender60 and final_ridge_resid_a10",
     "status": "training"},
    {"eras": "1133-1227", "use": "every era with local meta-model coverage; all previously "
     "inspected", "status": "development"},
    {"eras": ">=1228", "use": "unresolved in the 2026-10-10 snapshot; reserved for fresh "
     "confirmation, outcomes not inspected", "status": "reserved"},
]


def check_cfg(cfg):
    for key in ("name", "audit_name"):
        if Path(cfg[key]).name != cfg[key] or not cfg[key].startswith("dc_mavone_"):
            raise ValueError(f"Invalid artifact name {cfg[key]}")
    members = cfg["members"]
    if next(iter(members)) != "ender" or members["ender"] != cfg["data"]["scoring_target"]:
        raise ValueError("The anchor member must be the scoring target")
    s, t = cfg["schedule"], cfg["training"]
    if (s["gap"], s["block_eras"], t["stride"]) != (13, 26, 4):
        raise ValueError("Frozen design requires gap 13, 26-era blocks, stride 4")
    if any(cfg["lgbm"].get(k) != v for k, v in RECIPE.items()):
        raise ValueError("LightGBM recipe differs from the frozen all-feature recipe")
    if cfg["data"]["feature_set"] != "all":
        raise ValueError("Frozen design uses all features")


def smoke_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["name"] = SMOKE["name"]
    cfg["lgbm"]["n_estimators"] = SMOKE["n_estimators"]
    return cfg


def environment():
    return {"python": platform.python_version(), "lightgbm": lgb.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__, "scipy": scipy.__version__,
            "numerai_tools": getattr(numerai_tools, "__version__", "0.7.2"),
            "cpu_count": os.cpu_count(), "machine": platform.machine(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> list[str]:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or "", *cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in row]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return lines


# ----------------------------------------------------------------- Phase 0 / 1 audit

def target_inventory(version, targets, scoring_target):
    rows, alias, by_type = [], {}, {}
    for split in ("train", "validation"):
        df = load_index(version, split, targets)
        eras = df.era.map(era_key)
        if "data_type" in df:
            by_type[split] = {k: {"first": int(v.min()), "last": int(v.max()),
                                  "eras": int(v.nunique()), "rows": len(v)}
                              for k, v in eras.groupby(df.data_type)}
        for t in targets:
            frac = df[t].notna().groupby(eras).mean()
            has = frac[frac > 0]
            rows.append({"split": split, "target": t, "rows_nonnull": int(df[t].notna().sum()),
                         "rows": len(df), "first_era": int(has.index.min()) if len(has) else None,
                         "last_era": int(has.index.max()) if len(has) else None,
                         "full_eras": int((frac == 1).sum()),
                         "partial_eras": int(((frac > 0) & (frac < 1)).sum()),
                         "empty_eras": int((frac == 0).sum())})
        if "target" in df:
            both = df.target.notna() & df[scoring_target].notna()
            alias[split] = {"same_missingness": bool((df.target.isna()
                                                      == df[scoring_target].isna()).all()),
                            "equal_values": bool((df.target[both] == df[scoring_target][both]).all())}
        del df
        gc.collect()
    return pd.DataFrame(rows), alias, by_type


def coverage(version, val_eras):
    out = {}
    for split in ("train", "validation"):
        b = pd.read_parquet(data_path(version, f"{split}_benchmark_models.parquet"))
        idx = load_index(version, split, [])
        joined = idx.join(b[[BENCHMARK_COL]], how="left")
        eras = joined.era.map(era_key)
        has = eras[joined[BENCHMARK_COL].notna()]
        out[f"{split}_benchmark"] = {"columns": list(b.columns), "eras_with_rows": int(has.nunique()),
                                     "first": int(has.min()), "last": int(has.max()),
                                     "row_coverage": float(joined[BENCHMARK_COL].notna().mean())}
    meta = pd.read_parquet(data_path(version, "meta_model.parquet"))
    me = meta.era.map(era_key)
    rows_per_era = val_eras.value_counts()
    meta_rows = me.value_counts()
    length, first, last = longest_run(me.unique())
    out["meta_model"] = {"eras": int(me.nunique()), "first": int(me.min()), "last": int(me.max()),
                         "contiguous": [length, first, last], "rows": len(meta),
                         "min_era_row_coverage": float((meta_rows / rows_per_era.reindex(
                             meta_rows.index)).min())}
    return out, sorted(int(e) for e in me.unique())


def provenance_table():
    rows = []
    for line in (ROOT / "experiments/log.jsonl").read_text().splitlines():
        r = json.loads(line)
        cfg, d = r.get("config") or {}, (r.get("config") or {}).get("data", {})
        inc = d.get("include_validation", False)
        rows.append({"name": r["name"], "timestamp": r["timestamp"][:19], "git": r.get("git"),
                     "kind": r.get("kind", "experiment"),
                     "train_target": d.get("train_target", d.get("target")),
                     "residualize_against": d.get("residualize_against"),
                     "feature_set": d.get("feature_set"), "era_stride": d.get("era_stride"),
                     "include_validation": inc,
                     "trained_on_evaluation_eras": bool(inc),
                     "holdout_scored": r.get("holdout_scored"),
                     "reusable_as_forward_control": False,
                     "model": (cfg.get("model") or {}).get("type")})
    return pd.DataFrame(rows)


def history_audit(cfg):
    """Phase 1 statistics on the historical training period only (eras <= history end)."""
    version, end = cfg["data"]["version"], cfg["training"]["history_period_end"]
    cols = list(cfg["members"].values())
    parts = []
    for split in ("train", "validation"):
        f = load_index(version, split, cols)
        f["era_n"] = f.era.map(era_key)
        typed = f.data_type.isin(["train", "validation"]) if "data_type" in f else True
        parts.append(f.loc[typed & (f.era_n <= end)])
    h = pd.concat(parts)
    summary = []
    for name, t in cfg["members"].items():
        per_era = h.groupby("era_n")[t]
        vals = h[t].dropna()
        summary.append({"member": name, "target": t, "rows": len(h),
                        "nonnull": int(h[t].notna().sum()), "first_era": int(h.loc[h[t].notna(), "era_n"].min()),
                        "eras_partial": int(((per_era.count() > 0) & (per_era.count() < per_era.size())).sum()),
                        "eras_empty": int((per_era.count() == 0).sum()),
                        "distinct_values": int(vals.nunique()),
                        "values": ", ".join(f"{v:g}:{p:.3f}" for v, p in
                                            vals.value_counts(normalize=True).sort_index().items()),
                        "mean_era_mean": float(per_era.mean().mean()),
                        "mean_era_std": float(per_era.std().mean()),
                        "min_era_std": float(per_era.std().min()),
                        "max_era_std": float(per_era.std().max())})
    joint = h.dropna(subset=cols)
    names = list(cfg["members"])
    per_era_corr_mats = joint.groupby("era_n")[cols].apply(lambda d: d.rank().corr())
    corr = per_era_corr_mats.groupby(level=1).mean().loc[cols, cols]
    corr_std = per_era_corr_mats.groupby(level=1).std().loc[cols, cols]
    corr.index = corr.columns = corr_std.index = corr_std.columns = names
    excl = exclusions(h.era_n, h[cols], np.ones(len(h), bool), end, cfg["training"]["stride"],
                      cfg["training"]["anchor"])
    return pd.DataFrame(summary).set_index("member"), corr, corr_std, excl


def run_audit(cfg):
    started = time.time()
    check_cfg(cfg)
    version = cfg["data"]["version"]
    out = ROOT / "artifacts" / cfg["audit_name"]
    out.mkdir(exist_ok=False)
    before = production_snapshot()
    meta_json = load_metadata(version)
    targets = meta_json["targets"]
    features = feature_set(version, "all")
    files = []
    for p in sorted((DATA_DIR / version).iterdir()):
        st = p.stat()
        files.append({"file": p.name, "bytes": st.st_size,
                      "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                      "sha256": digest(p)})
    print("hashed data files", flush=True)
    inv, alias, by_type = target_inventory(version, targets, cfg["data"]["scoring_target"])
    print("target inventory done", flush=True)
    val = load_index(version, "validation", [cfg["data"]["scoring_target"]])
    val_eras = val.era.map(era_key)
    resolved = sorted(val_eras[val[cfg["data"]["scoring_target"]].notna()
                               & val.data_type.eq("validation")].unique())
    cov, meta_eras = coverage(version, val_eras)
    remote = pd.read_parquet(ROOT / cfg["data"]["remote_check"], columns=["era"]).era.map(era_key)
    cov["remote_meta_model_check_2026_10_14"] = {
        "eras": int(remote.nunique()), "first": int(remote.min()), "last": int(remote.max()),
        "note": "era column only; newer eras are reserved and their values were not read"}
    live = pd.read_parquet(data_path(version, "live_1374.parquet"), columns=["era"])
    prov = provenance_table()
    hist, tcorr, tcorr_std, excl = history_audit(cfg)
    print("history audit done", flush=True)
    s = cfg["schedule"]
    blocks = block_schedule(s["first_block_start"], s["last_block_start"], s["block_eras"], s["gap"])
    present = {int(e) for e in val_eras.unique()}
    stride_hist = {}
    for b in blocks:
        eras = range(b["start"], b["end"] + 1)
        b["eras_in_file"] = sum(e in present for e in eras)
        b["resolved_eras"] = sum(e in set(resolved) for e in eras)
        b["meta_eras"] = sum(e in set(meta_eras) for e in eras)
        b["prospective_eras"] = b["eras_in_file"] - b["resolved_eras"]
        stride_hist[b["start"]] = int(stride_mask(np.arange(1, b["fit_end"] + 1), 4,
                                                  cfg["training"]["anchor"]).sum())
        b["stride_training_eras"] = stride_hist[b["start"]]
    forward_meta = [e for e in meta_eras if e in set(resolved) and e >= blocks[0]["start"]]
    gate = mmc_gate(forward_meta)
    audit = {
        "dataset_version": version, "snapshot_acquired": "2026-10-10/11 (file mtimes)",
        "files": files, "n_features": len(features),
        "feature_order_sha256": ids_hash(features),
        "metadata_sha256": digest(data_path(version, "features.json")),
        "named_targets": len([t for t in targets if t != "target"]), "target_alias": alias,
        "data_types": by_type, "resolved_validation_eras": [int(resolved[0]), int(resolved[-1])],
        "coverage": cov, "live_era_values": sorted(live.era.astype(str).unique().tolist()),
        "periods_used": PERIODS_USED,
        "label_availability": (
            "No point-in-time publication dates exist locally; a label for era e is assumed "
            "available 13 weekly eras later (60D2L resolves in about 12 weeks). Results are a "
            "retrospective forward-label simulation, not an exact historical live replay."),
        "scoring_rules_checked": {
            "source": "https://docs.numer.ai/numerai-tournament/scoring/ (fetched 2026-10-14)",
            "summary": "From round 1343 CORR, MMC and BMC use 60-day Ender; v5.3 diagnostics "
                       "use target_ender_60 and v53_lgbm_ender60. Payout 3*CORR + 15*MMC "
                       "(repo constant)."},
        "remote_snapshot_check": {"latest_remote_version": "v5.3", "current_round": 1376,
                                  "local_snapshot_kept": True},
        "blocks": blocks, "mmc_gate": gate, "environment": environment(),
    }
    inv.to_csv(out / "target_inventory.csv", index=False)
    prov.to_csv(out / "artifact_provenance.csv", index=False)
    hist.to_csv(out / "history_target_summary.csv")
    tcorr.to_csv(out / "history_target_rank_corr.csv")
    excl.to_csv(out / "history_exclusions_by_era.csv")
    (out / "history_target_rank_corr.svg").write_text(
        heatmap_svg(tcorr, f"Mean per-era target rank correlation, eras <= "
                           f"{cfg['training']['history_period_end']}"))
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    write_audit_report(out, audit, inv, prov, hist, tcorr, tcorr_std, excl, cfg)
    after = production_snapshot()
    if before != after:
        raise RuntimeError("Protected production files changed")
    print(f"audit done in {time.time() - started:.0f}s", flush=True)


def write_audit_report(out, a, inv, prov, hist, tcorr, tcorr_std, excl, cfg):
    cov = a["coverage"]
    lines = ["# dc_mavone Phase 0-1 audit", "",
             (f"Dataset {a['dataset_version']}, {a['n_features']} features "
             f"(order sha256 {a['feature_order_sha256'][:16]}), {a['named_targets']} named "
             f"targets plus the `target` alias ({json.dumps(a['target_alias'])})."), "",
             "## Files", "", "| File | Bytes | Modified | sha256 |", "|---|---:|---|---|"]
    lines += [f"| {f['file']} | {f['bytes']:,} | {f['modified']} | {f['sha256'][:16]} |"
              for f in a["files"]]
    lines += ["", "## Availability", "", "| Source | Eras | First | Last | Notes |",
              "|---|---:|---:|---:|---|"]
    for k in ("train_benchmark", "validation_benchmark"):
        c = cov[k]
        lines.append(f"| {k} ({BENCHMARK_COL}) | {c['eras_with_rows']} | {c['first']} | "
                     f"{c['last']} | row coverage {c['row_coverage']:.3f} |")
    m = cov["meta_model"]
    lines.append(f"| meta model (local) | {m['eras']} | {m['first']} | {m['last']} | contiguous "
                 f"{m['contiguous'][0]}; min era row coverage {m['min_era_row_coverage']:.3f} |")
    r = cov["remote_meta_model_check_2026_10_14"]
    lines.append(f"| meta model (remote, 2026-10-14) | {r['eras']} | {r['first']} | {r['last']} "
                 f"| {r['note']} |")
    lines.append(f"| resolved {cfg['data']['scoring_target']} (validation) | | "
                 f"{a['resolved_validation_eras'][0]} | {a['resolved_validation_eras'][1]} | |")
    lines += ["", f"Data types: `{json.dumps(a['data_types'])}`", "",
              "## Target maturity and missingness", ""]
    member_targets = set(cfg["members"].values())
    sub = inv[inv.target.isin(member_targets)].set_index("target")
    lines += md_table(sub, "{}")
    lines += ["", f"All {len(inv) // 2} target columns are in `target_inventory.csv`.", "",
              "## Periods already used (all development evidence)", "",
              "| Eras | Use | Status |", "|---|---|---|"]
    lines += [f"| {p['eras']} | {p['use']} | {p['status']} |" for p in a["periods_used"]]
    lines += ["", ("No resolved era in the local snapshot is unused. Eras from 1228 on are "
              "reserved for fresh confirmation; their outcomes are not inspected here."), "",
              f"Label availability: {a['label_availability']}", "",
              "## Prediction-artifact provenance", "",
              (f"{len(prov)} logged runs (`artifact_provenance.csv`). "
              f"{int(prov.trained_on_evaluation_eras.sum())} trained on validation eras "
              "(recent_*/final_*), so they are not valid forward controls. Earlier walk-forward "
              "artifacts used different targets, features or labels. No cached prediction is "
              "reused; every Mavone prediction is generated fresh with recorded provenance."), "",
              ("## Phase 1: target audit on eras <= "
              f"{cfg['training']['history_period_end']} (historical training only)"), ""]
    lines += md_table(hist.drop(columns=["values"]), "{:.4f}")
    lines += ["", "Value distributions: "]
    lines += [f"- {k}: {v}" for k, v in hist["values"].items()]
    lines += ["", ("Mean per-era Spearman correlation between targets (std over eras in "
              "brackets):"), ""]
    fmt = tcorr.copy().astype(object)
    for i in tcorr.index:
        for j in tcorr.columns:
            fmt.loc[i, j] = f"{tcorr.loc[i, j]:.3f} ({tcorr_std.loc[i, j]:.3f})"
    fmt.index.name = "target"
    lines += md_table(fmt, "{}")
    lines += ["", (f"Joint-intersection exclusions on stride eras <= "
              f"{cfg['training']['history_period_end']}: {int(excl.excluded_rows.sum()):,} of "
              f"{int(excl.candidate_rows.sum()):,} rows; by target "
              f"{excl.filter(like='missing_').sum().to_dict()} (`history_exclusions_by_era.csv`)."),
              "", "## Split calendar", "",
              ("| Block | Eras | Fit on eras <= | Labels available by (approx.) | In file | "
              "Resolved | Meta | Prospective | Stride training eras |"),
              "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for b in a["blocks"]:
        lines.append(f"| {b['start']} | {b['start']}-{b['end']} | {b['fit_end']} | era "
                     f"{b['label_available_by_era']} ({b['label_available_by_date']}) | "
                     f"{b['eras_in_file']} | {b['resolved_eras']} | {b['meta_eras']} | "
                     f"{b['prospective_eras']} | {b['stride_training_eras']} |")
    g = a["mmc_gate"]
    lines += ["", (f"Phase 4 MMC-history gate: needs {g['required_contiguous_eras']} contiguous "
              f"forward eras with MMC, has {g['available_contiguous_eras']} "
              f"({g['first']}-{g['last']}). Gate passed: {g['gate_passed']}; learned MMC "
              f"weights {g['mmc_weight_optimization']}."), "",
              f"Scoring rules: {json.dumps(a['scoring_rules_checked'])}", "",
              f"Environment: `{json.dumps(a['environment'])}`"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- Phase 2 forward fits

def load_frames(cfg):
    version = cfg["data"]["version"]
    cols = list(cfg["members"].values())
    refs = list(cfg["control"]["residualize_against"])
    ender = cfg["members"]["ender"]
    frames, audit = {}, {}
    rng = np.random.default_rng(0)
    for split in ("train", "validation"):
        f = load_index(version, split, cols)
        bench = pd.read_parquet(data_path(version, f"{split}_benchmark_models.parquet"),
                                columns=refs)
        n = len(f)
        f = f.join(bench, how="left")
        assert len(f) == n and not f.index.has_duplicates
        f["era_n"] = f.era.map(era_key)
        f["eligible"] = f.data_type.isin(["train", "validation"]).to_numpy()
        has = (f[[ender, *refs]].notna().all(axis=1) & f.eligible).to_numpy()
        f["resid_label"] = np.nan
        f.loc[has, "resid_label"] = residualize_many(
            f.loc[has, [ender]].to_numpy(), f.loc[has, refs].to_numpy(),
            f.loc[has, "era_n"].to_numpy())[:, 0]
        checked = sorted(rng.choice(f.loc[has, "era"].unique(), 2, replace=False).tolist())
        for era in checked:
            rows = f.loc[has & f.era.eq(era).to_numpy()]
            expected = np.asarray(residualize(rows, ender, refs)).ravel()
            np.testing.assert_allclose(rows.resid_label, expected, atol=1e-6)
        audit[split] = {"rows": n, "resid_rows": int(has.sum()), "label_parity_eras": checked}
        frames[split] = f
    return frames, audit


def fit_block(cfg, features, frames, block, out_dir):
    version = cfg["data"]["version"]
    names, cols = list(cfg["members"]), list(cfg["members"].values())
    t = cfg["training"]
    seqs, ys, resid, ids, eras, excl = [], [], [], [], [], []
    for split, f in frames.items():
        m = training_mask(f.era_n, f[cols], f.eligible, block["fit_end"], t["stride"], t["anchor"])
        excl.append(exclusions(f.era_n, f[cols], f.eligible, block["fit_end"], t["stride"],
                               t["anchor"]).assign(split=split))
        if not m.any():
            continue
        seqs.append(ParquetRows(version, split, features, m))
        ys.append(f.loc[m, cols].to_numpy(np.float64))
        resid.append(f.resid_label.to_numpy()[m])
        ids.append(f.index[m])
        eras.append(f.era_n.to_numpy()[m])
    y, r, eras = np.vstack(ys), np.concatenate(resid), np.concatenate(eras)
    ids = np.concatenate([np.asarray(i) for i in ids])
    assert np.isfinite(y).all() and eras.max() <= block["fit_end"] < block["start"] - 13
    assert (eras % t["stride"] == t["anchor"] % t["stride"]).all()
    w = np.isfinite(r).astype(np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    pd.concat(excl).to_csv(out_dir / "exclusions_by_era.csv")
    params = dict(cfg["lgbm"])
    rounds = params.pop("n_estimators")
    ds = lgb.Dataset(seqs, label=y[:, 0], free_raw_data=True).construct()
    print(f"  dataset: {ds.num_data():,} rows, eras {eras.min()}-{eras.max()} "
          f"({len(np.unique(eras))}), peak RSS {peak_rss_gb():.1f} GB", flush=True)
    boosters, seconds = {}, {}
    for j, name in enumerate(names):
        t0 = time.time()
        ds.set_label(y[:, j])
        boosters[name] = lgb.train(params, ds, num_boost_round=rounds)
        seconds[name] = round(time.time() - t0, 1)
        print(f"  {name}: {seconds[name]}s, peak RSS {peak_rss_gb():.1f} GB", flush=True)
    t0 = time.time()
    ds.set_label(r)
    ds.set_weight(w)
    resid_name = cfg["control"]["resid_member"]
    boosters[resid_name] = lgb.train(params, ds, num_boost_round=rounds)
    seconds[resid_name] = round(time.time() - t0, 1)
    print(f"  {resid_name}: {seconds[resid_name]}s", flush=True)
    del ds, seqs
    gc.collect()
    audit = {"train_rows": len(y), "train_eras": len(np.unique(eras)),
             "first_era": int(eras.min()), "last_era": int(eras.max()),
             "train_ids_sha256": ids_hash(ids), "resid_zero_weight_rows": int((w == 0).sum()),
             "fit_seconds": seconds}
    return boosters, audit


def predict_block(cfg, features, frame, block, boosters):
    """Raw predictions for every validation-file row in the block's eras."""
    version = cfg["data"]["version"]
    mask = frame.era_n.between(block["start"], block["end"]).to_numpy()
    rows = np.flatnonzero(mask)
    names = list(boosters)
    out = np.full((len(rows), len(names)), np.nan)
    parity = None
    for pos, xb in iter_features(version, "validation", features, mask):
        out[np.searchsorted(rows, pos)] = np.column_stack([boosters[n].predict(xb) for n in names])
        if parity is None:
            parity = (xb[:PARITY_ROWS].copy(), pos[:PARITY_ROWS])
    preds = pd.DataFrame(out, columns=names, index=frame.index[rows])
    meta = frame.iloc[rows][["era", "data_type"]]
    preds = meta.join(preds)
    return preds, parity


def run_forward(cfg, resume=False, smoke=False):
    started = time.time()
    check_cfg(cfg)
    if smoke:
        cfg = smoke_cfg(cfg)
    version = cfg["data"]["version"]
    features = feature_set(version, cfg["data"]["feature_set"])
    if smoke:
        features = features[:SMOKE["features"]]
    s = cfg["schedule"]
    blocks = block_schedule(s["first_block_start"], s["last_block_start"], s["block_eras"], s["gap"])
    if smoke:
        blocks = [b for b in blocks if b["start"] in SMOKE["blocks"]]
    out = ROOT / "artifacts" / cfg["name"]
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        if not resume or manifest["config"] != cfg:
            raise FileExistsError(f"{out} exists; use --resume with an identical config")
        if manifest["status"] == "complete":
            raise FileExistsError(f"{out} is complete; never overwrite a completed run")
    else:
        out.mkdir(exist_ok=False)
        (out / "sources").mkdir()
        for p in SOURCES:
            shutil.copy2(ROOT / p, out / "sources" / Path(p).name)
        manifest = {"config": cfg, "features": features, "feature_order_sha256": ids_hash(features),
                    "git": git_hash(), "blocks": blocks, "environment": environment(),
                    "metadata_sha256": digest(data_path(version, "features.json")),
                    "source_hashes": {p: digest(ROOT / p) for p in SOURCES},
                    "label_availability": "retrospective forward-label simulation: era e "
                                          "labels assumed available at era e + 13",
                    "completed_blocks": {}, "runs": []}
    before = production_snapshot()
    manifest.update(status="running")
    run = {"started": datetime.now().isoformat(timespec="seconds"), "resume": resume,
           "source_hashes": {p: digest(ROOT / p) for p in SOURCES}, "production_before": before}
    manifest["runs"].append(run)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    try:
        frames, label_audit = load_frames(cfg)
        manifest["label_audit"] = label_audit
        print(f"labels ready in {time.time() - started:.0f}s: {label_audit}", flush=True)
        for block in blocks:
            key = str(block["start"])
            if key in manifest["completed_blocks"]:
                print(f"block {key}: already complete, skipping", flush=True)
                continue
            t0 = time.time()
            bdir = out / "blocks" / key
            if bdir.exists():
                shutil.rmtree(bdir)  # an interrupted, never-completed block
            bdir.mkdir(parents=True)
            print(f"block {key}: fitting on eras <= {block['fit_end']}", flush=True)
            boosters, audit = fit_block(cfg, features, frames, block, bdir)
            hashes = {}
            for name, booster in boosters.items():
                path = bdir / f"{name}.txt"
                booster.save_model(str(path))
                hashes[name] = digest(path)
            preds, (px, ppos) = predict_block(cfg, features, frames["validation"], block, boosters)
            assert np.isfinite(preds[list(boosters)].to_numpy()).all()
            for name in boosters:
                loaded = lgb.Booster(model_file=str(bdir / f"{name}.txt"))
                pid = frames["validation"].index[ppos]
                np.testing.assert_allclose(loaded.predict(px), preds.loc[pid, name], atol=1e-10)
                halves = np.concatenate([loaded.predict(px[:7]), loaded.predict(px[7:])])
                np.testing.assert_allclose(halves, preds.loc[pid, name], atol=1e-10)
            preds["block_start"] = block["start"]
            preds["fit_end"] = block["fit_end"]
            preds["label_available_by_era"] = block["label_available_by_era"]
            preds["label_available_by_date"] = block["label_available_by_date"]
            preds.index.name = "id"
            preds.to_parquet(bdir / "preds.parquet")
            audit.update(block=block, model_sha256=hashes, pred_rows=len(preds),
                         pred_eras=int(preds.era.nunique()),
                         preds_sha256=digest(bdir / "preds.parquet"),
                         seconds=round(time.time() - t0, 1), peak_rss_gb=round(peak_rss_gb(), 2),
                         checks_passed=True)
            manifest["completed_blocks"][key] = audit
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            print(f"block {key}: done in {audit['seconds']}s, peak RSS {audit['peak_rss_gb']} GB",
                  flush=True)
            del boosters
            gc.collect()
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
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        if not run["production_unchanged"]:
            raise RuntimeError("Protected production files changed")


# ----------------------------------------------------------------- Phase 3 evaluation

def derive(cfg, preds, features_medium):
    """Normalized members and fixed blends for every row; the normalization universe is
    each complete era in the validation file."""
    names = list(cfg["members"])
    eras = preds.era.to_numpy()
    out = pd.DataFrame(index=preds.index)
    for n in [*names, cfg["control"]["resid_member"]]:
        out[f"g_{n}"] = gaussian_rank(preds[n].to_numpy(), eras)
    out["ender_only"] = blend(preds[names], {"ender": 1.0}, eras)
    for subset in incremental_sets(names)[1:]:
        label = "eq4" if len(subset) == len(names) else "inc_" + "_".join(subset)
        out[label] = blend(preds[names], {n: 1 / len(subset) for n in subset}, eras)
    c = cfg["control"]
    mixed = pd.Series((out[f"g_{c['resid_member']}"] + out[f"g_{c['partner']}"]).to_numpy() / 2,
                      index=preds.index)
    neutral = neutralize(mixed, features_medium, preds.era, c["neutralize"]["proportion"])
    out[c["name"]] = neutral.groupby(preds.era).rank(pct=True)
    return out


def run_evaluate(cfg, smoke=False):
    started = time.time()
    check_cfg(cfg)
    if smoke:
        cfg = smoke_cfg(cfg)
    version, target = cfg["data"]["version"], cfg["data"]["scoring_target"]
    out = ROOT / "artifacts" / cfg["name"]
    manifest = json.loads((out / "manifest.json").read_text())
    blocks = manifest["blocks"]
    if not smoke and manifest["status"] != "forward_complete":
        raise RuntimeError("forward predictions are incomplete")
    ev = out / "evaluation"
    ev.mkdir(exist_ok=False)
    before = production_snapshot()
    names = list(cfg["members"])
    raw_cols = [*names, cfg["control"]["resid_member"]]
    parts = []
    for b in blocks:
        audit = manifest["completed_blocks"][str(b["start"])]
        path = out / "blocks" / str(b["start"]) / "preds.parquet"
        assert digest(path) == audit["preds_sha256"], "prediction artifact changed"
        p = pd.read_parquet(path)
        p["preds_sha256"] = audit["preds_sha256"]
        parts.append(p)
    preds = pd.concat(parts)
    val = load_index(version, "validation", [target])
    val["era_n"] = val.era.map(era_key)
    starts = {b["start"]: b for b in blocks}
    in_blocks = np.zeros(len(val), bool)
    for b in blocks:
        in_blocks |= val.era_n.between(b["start"], b["end"]).to_numpy()
    expected = val.index[in_blocks]
    assert_complete(preds, raw_cols, expected)
    preds = preds.loc[expected]  # file order
    preds[target] = val.loc[expected, target]
    preds["era_n"] = val.loc[expected, "era_n"]
    for start, b in starts.items():
        sel = preds.block_start.eq(start)
        assert preds.loc[sel, "era_n"].between(b["start"], b["end"]).all()
        assert (preds.loc[sel, "fit_end"] < b["start"] - 13).all()
    medium = feature_set(version, cfg["control"]["neutralize"]["feature_set"])
    xm = pd.DataFrame(load_features(version, "validation", medium, in_blocks),
                      index=expected, columns=medium)
    derived = derive(cfg, preds, xm)
    del xm
    gc.collect()
    full = preds.join(derived)
    blends = ["ender_only", *[c for c in derived.columns if c.startswith("inc_")], "eq4",
              cfg["control"]["name"]]
    for _, era in full.groupby("era"):
        for col in blends:
            validate_submission(era[col].rename("prediction"), era.index)
    meta = pd.read_parquet(data_path(version, "meta_model.parquet"), columns=[META_COL])
    bench = pd.read_parquet(data_path(version, "validation_benchmark_models.parquet"),
                            columns=[BENCHMARK_COL])
    n = len(full)
    full = full.join(meta, how="left").join(bench, how="left")
    assert len(full) == n
    resolved = full[target].notna() & val.loc[full.index, "data_type"].eq("validation")
    scored = full.loc[resolved].copy()
    prospective = full.loc[~resolved]
    last_resolved = int(scored.era_n.max())
    assert prospective.empty or prospective.era_n.min() > last_resolved
    assert prospective[target].isna().all()
    prospective.drop(columns=[target, META_COL, BENCHMARK_COL]).to_parquet(
        ev / "prospective_preds.parquet")
    full.drop(columns=[META_COL, BENCHMARK_COL]).loc[resolved].to_parquet(ev / "forward_preds.parquet")
    cols = [*names, cfg["control"]["resid_member"], *blends]
    scored["ender_ref"] = scored["ender"]  # a reference column must not also be scored
    refs = {"mmc": META_COL, "bmc": BENCHMARK_COL, "ender_contribution": "ender_ref"}

    def score(rows):
        return evaluate(rows, cols, target, refs)

    metrics = {"all": score(scored)}
    for start, rows in scored.groupby("block_start"):
        metrics[str(start)] = score(rows)
    per_era = {"corr": per_era_corr(scored, cols, target)}
    for k, ref in (("mmc", META_COL), ("bmc", BENCHMARK_COL)):
        per_era[k] = per_era_contribution(scored, cols, target, ref)
    for k in per_era:
        per_era[k] = per_era[k].loc[sorted(per_era[k].index, key=int)]
    stacked = pd.concat({k: v.stack() for k, v in per_era.items()}, axis=1)
    stacked.index.names = ["era", "model"]
    stacked.to_csv(ev / "per_era.csv")
    paired = {}
    comparisons = [("eq4", "ender_only"), (cfg["control"]["name"], "ender_only"),
                   ("eq4", cfg["control"]["name"])]
    comparisons += [(c, "ender_only") for c in blends if c.startswith("inc_")]
    for a, b in comparisons:
        paired[f"{a}_minus_{b}"] = {}
        for k, v in per_era.items():
            d = (v[a] - v[b]).dropna()
            if np.all(np.diff(d.index.astype(int)) == 1):
                ci = paired_block_ci(d.to_numpy(), d.index, block=13)
            else:
                ci = {"mean": float(d.mean()), "ci": None, "note": "non-contiguous eras"}
            by_block = {str(s): float(d[[e for e in d.index if starts[s]["start"] <= int(e)
                                         <= starts[s]["end"]]].mean())
                        for s in starts if any(starts[s]["start"] <= int(e) <= starts[s]["end"]
                                               for e in d.index)}
            paired[f"{a}_minus_{b}"][k] = {**ci, "n_eras": len(d), "by_block": by_block,
                                           "pct_positive": float((d > 0).mean())}
    pcorr = prediction_correlation(scored, cols)
    with_meta = scored.dropna(subset=[META_COL])
    pcorr_meta = prediction_correlation(with_meta, [*cols, META_COL, BENCHMARK_COL])
    score_corr = {k: v[cols].corr() for k, v in per_era.items()}
    pcorr.to_csv(ev / "prediction_rank_corr.csv")
    pcorr_meta.to_csv(ev / "prediction_rank_corr_meta_eras.csv")
    for k, v in score_corr.items():
        v.to_csv(ev / f"score_corr_{k}.csv")
    heat_cols = [*names, cfg["control"]["resid_member"], "eq4", cfg["control"]["name"],
                 META_COL, BENCHMARK_COL]
    (ev / "prediction_rank_corr.svg").write_text(heatmap_svg(
        pcorr_meta.loc[heat_cols, heat_cols].rename(index={META_COL: "meta", BENCHMARK_COL: "bench"},
                                                    columns={META_COL: "meta", BENCHMARK_COL: "bench"}),
        f"Mean per-era prediction rank correlation, eras {int(with_meta.era_n.min())}-"
        f"{int(with_meta.era_n.max())}"))
    eq = metrics["all"]["eq4"]
    pm = paired["eq4_minus_ender_only"]["mmc"]
    decision = hypothesis1_decision({"corr": eq["corr"]["mean"],
                                     "mmc": eq["mmc"]["mean"] if "mmc" in eq else None,
                                     "ender_contribution": eq["ender_contribution"]["mean"],
                                     "paired_mmc_improvement": pm["mean"] if pm["n_eras"] else None})
    gate = mmc_gate(sorted(int(e) for e in with_meta.era.unique()))
    decision["mmc_gate"] = gate
    cost = {k: {"seconds": v["seconds"], "peak_rss_gb": v["peak_rss_gb"],
                "train_rows": v["train_rows"], "fit_seconds": v["fit_seconds"]}
            for k, v in manifest["completed_blocks"].items()}
    for name, value in (("metrics", metrics), ("paired", paired), ("decision", decision),
                        ("compute_cost", cost)):
        (ev / f"{name}.json").write_text(json.dumps(value, indent=2, default=str))
    write_eval_report(ev, cfg, metrics, paired, decision, pcorr, pcorr_meta, score_corr, cost,
                      scored, prospective, blocks)
    after = production_snapshot()
    if before != after:
        raise RuntimeError("Protected production files changed")
    if not smoke:
        log_run(cfg["name"], cfg, flatten(metrics["all"]), started, kind="mavone_forward",
                holdout_scored=False, decision={k: v for k, v in decision.items()
                                                if k != "mmc_gate"},
                peak_rss_gb=round(peak_rss_gb(), 2))
    print((ev / "report.md").read_text(), flush=True)


def write_eval_report(ev, cfg, metrics, paired, decision, pcorr, pcorr_meta, score_corr, cost,
                      scored, prospective, blocks):
    names = list(cfg["members"])
    resid, control = cfg["control"]["resid_member"], cfg["control"]["name"]

    def cell(m, k, stat="mean", fmt="{:.4f}"):
        return fmt.format(m[k][stat]) if k in m else "—"

    def table(rows_metrics, cols):
        lines = [("| Model | CORR | CORR sharpe | MMC | MMC sharpe | BMC | Beyond Ender | "
                 "Payout (3C+15M) |"), "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in cols:
            m = rows_metrics[c]
            lines.append(f"| {c} | {cell(m, 'corr')} | {cell(m, 'corr', 'sharpe', '{:.2f}')} | "
                         f"{cell(m, 'mmc')} | {cell(m, 'mmc', 'sharpe', '{:.2f}')} | "
                         f"{cell(m, 'bmc')} | {cell(m, 'ender_contribution')} | "
                         f"{cell(m, 'payout_mmc')} |")
        return lines

    blends = [c for c in metrics["all"] if c not in names and c != resid]
    first, last = int(scored.era_n.min()), int(scored.era_n.max())
    meta_eras = scored.dropna(subset=[META_COL]).era.nunique()
    lines = ["# dc_mavone Phase 3: forward multi-target ensemble", "",
             (f"Forward predictions for eras {first}-{last} ({scored.era.nunique()} resolved eras, "
             f"{meta_eras} with meta-model MMC) from {len(blocks)} fixed 26-era blocks, each fit "
             "on stride-4 eras before a 13-era gap. Every evaluated era was inspected by earlier "
             "research, so this is development evidence from a retrospective forward-label "
             "simulation, not fresh confirmation."), "",
             (f"Prospective predictions for {prospective.era.nunique()} unresolved eras are saved "
             "unscored for later confirmation."), "",
             "## Members and blends (all forward eras)", ""]
    lines += table(metrics["all"], [*names, resid, *blends])
    lines += ["", (f"`{control}` is the contextual dc_resid_ensemble recreation (residual labels "
              "and neutralization differ from the controlled experiment; stride 4, unlike live)."),
              "", "## By block", ""]
    for b in [k for k in metrics if k != "all"]:
        lines += [f"### Block {b}", ""] + table(metrics[b], ["ender_only", "eq4", control])
        lines.append("")
    lines += ["## Paired common-era differences (13-era block bootstrap)", "",
              "| Comparison | Score | Mean | 95% CI | Eras | % positive | By block |",
              "|---|---|---:|---|---:|---:|---|"]
    for comp, scores in paired.items():
        for k, v in scores.items():
            ci = f"[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]" if v.get("ci") else "—"
            lines.append(f"| {comp} | {k} | {v['mean']:.4f} | {ci} | "
                         f"{v['n_eras']} | {v['pct_positive']:.2f} | "
                         f"{', '.join(f'{s}: {x:.4f}' for s, x in v['by_block'].items())} |")
    lines += ["", "## Prediction rank correlations (eras with meta model)", ""]
    pm = pcorr_meta.rename(index={META_COL: "meta", BENCHMARK_COL: "bench"},
                           columns={META_COL: "meta", BENCHMARK_COL: "bench"})
    keep = [*names, resid, "eq4", control, "meta", "bench"]
    pm = pm.loc[keep, keep]
    pm.index.name = "model"
    lines += md_table(pm, "{:.3f}")
    lines += ["", ("Heatmap: `prediction_rank_corr.svg`. All-era matrix: "
              "`prediction_rank_corr.csv`."), "", "## Per-era score correlations (CORR)", ""]
    sc = score_corr["corr"].loc[[*names, resid], [*names, resid]]
    sc.index.name = "model"
    lines += md_table(sc, "{:.3f}")
    lines += ["", "## Hypothesis 1 (controlled equal weight vs matched Ender)", "",
              f"```\n{json.dumps({k: v for k, v in decision.items() if k != 'mmc_gate'}, indent=2)}\n```",
              "", "## Phase 4 MMC-history gate", "", f"```\n{json.dumps(decision['mmc_gate'], indent=2)}\n```",
              "", "## Compute cost", "", "| Block | Seconds | Peak RSS GB | Train rows | Per-model fit seconds |",
              "|---|---:|---:|---:|---|"]
    for k, v in cost.items():
        lines.append(f"| {k} | {v['seconds']} | {v['peak_rss_gb']} | {v['train_rows']:,} | "
                     f"{json.dumps(v['fit_seconds'])} |")
    lines += ["", "Stopped for review before Phase 4. No deployment, upload or live change."]
    (ev / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "forward", "evaluate"])
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.command == "audit":
        run_audit(config)
    elif args.command == "forward":
        run_forward(config, resume=args.resume, smoke=args.smoke)
    else:
        run_evaluate(config, smoke=args.smoke)
    sys.exit(0)
