"""Ensemble saved predictions and apply feature neutralization, without retraining.

Usage:
  python scripts/ensemble.py lgbm_all_stride4 lgbm_all_tyler_60 ... [--neutralize medium]

Selection-bias guard: dev eras are split in half chronologically. Ensemble members
(greedy forward selection) and the neutralization proportion are chosen on the
first half (A) by 3*CORR + 15*BMC; everything is reported on the second half (B).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from numerai_dev.data import (
    data_path,
    era_key,
    era_mask,
    feature_set,
    iter_features,
    load_index,
)
from numerai_dev.evaluation import (
    BENCHMARK_COL,
    CORR_WEIGHT,
    META_COL,
    MMC_WEIGHT,
    evaluate,
    per_era_contribution,
    per_era_corr,
)
from numerai_dev.experiment import ROOT, log_run
from numerai_dev.targets import gaussianize

TARGET = "target_ender_60"
VERSION = "v5.3"


def load_preds(names: list[str]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(ROOT / "artifacts" / n / "validation_preds.parquet")
        .rename(columns={"prediction": n})
        for n in names
    ]
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f[[f.columns[-1]]], how="inner")
    df = df.join(pd.read_parquet(data_path(VERSION, "validation.parquet"), columns=[TARGET]))
    df = df.join(pd.read_parquet(data_path(VERSION, "validation_benchmark_models.parquet"),
                                 columns=[BENCHMARK_COL]))
    df = df.join(pd.read_parquet(data_path(VERSION, "meta_model.parquet"), columns=[META_COL]))
    df = df.dropna(subset=[TARGET])
    # Gaussianize each model per era so equal weights mean equal influence.
    for n in names:
        df[n] = df.groupby("era")[n].transform(gaussianize)
    return df


def objective(df: pd.DataFrame, col: str) -> float:
    corr = per_era_corr(df, [col], TARGET)[col].mean()
    bmc = per_era_contribution(df, [col], TARGET, BENCHMARK_COL)[col].mean()
    return CORR_WEIGHT * corr + MMC_WEIGHT * bmc


def greedy_select(df: pd.DataFrame, names: list[str], max_members: int = 6) -> list[str]:
    """Forward selection with replacement-free equal weights, by objective on df."""
    chosen: list[str] = []
    best = -np.inf
    while len(chosen) < max_members:
        scores = {}
        for n in names:
            if n in chosen:
                continue
            df["_cand"] = df[[*chosen, n]].mean(axis=1)
            scores[n] = objective(df, "_cand")
        if not scores:
            break
        n, s = max(scores.items(), key=lambda kv: kv[1])
        if s <= best:
            break
        chosen.append(n)
        best = s
        print(f"  + {n:28s} objective(A) = {s:.4f}")
    df.drop(columns="_cand", inplace=True, errors="ignore")
    return chosen


def neutral_fit(df: pd.DataFrame, col: str, features: list[str]) -> pd.Series:
    """Per era, the least-squares projection of `col` onto the features. Neutralized
    predictions are col - p * fit for proportion p."""
    idx = load_index(VERSION, "validation", [])
    mask = era_mask(idx, eras=sorted(df["era"].unique()))
    pos_of_id = pd.Series(np.arange(len(idx)), index=idx.index)
    want = pos_of_id.reindex(df.index).to_numpy()
    rows = np.empty((len(df), len(features)), dtype=np.int8)
    order = np.argsort(want)
    filled = 0
    for pos, xb in iter_features(VERSION, "validation", features, mask):
        keep = np.isin(pos, want)
        n = int(keep.sum())
        rows[order[filled:filled + n]] = xb[keep]
        filled += n
    assert filled == len(df)

    fit = np.empty(len(df), dtype=np.float64)
    eras = df["era"].to_numpy()
    for era in np.unique(eras):
        m = eras == era
        x = rows[m].astype(np.float32)
        x -= x.mean(axis=0)
        y = df.loc[m, col].to_numpy(dtype=np.float32)
        y = y - y.mean()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fit[m] = x @ beta
    return pd.Series(fit, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--neutralize", default=None, help="feature set for neutralization")
    parser.add_argument("--name", default="ensemble")
    parser.add_argument("--no-log", action="store_true", help="don't append to the log")
    args = parser.parse_args()

    started = time.time()
    df = load_preds(args.names)
    eras = sorted(df["era"].unique(), key=era_key)
    half = len(eras) // 2
    a_eras, b_eras = set(eras[:half]), set(eras[half:])
    a = df[df["era"].isin(a_eras)].copy()
    print(f"A: eras {eras[0]}-{eras[half - 1]}  B: eras {eras[half]}-{eras[-1]}")

    print("greedy selection on A:")
    chosen = greedy_select(a, args.names)
    df["ens_greedy"] = df[chosen].mean(axis=1)
    df["ens_all"] = df[args.names].mean(axis=1)
    cols = [*args.names, "ens_all", "ens_greedy"]

    best_p = None
    if args.neutralize:
        feats = feature_set(VERSION, args.neutralize)
        print(f"neutralizing ens_greedy against {args.neutralize} ({len(feats)} features)")
        fit = neutral_fit(df, "ens_greedy", feats)
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        for p in grid:
            df[f"neut_{p}"] = df["ens_greedy"] - p * fit
        a = df[df["era"].isin(a_eras)]
        scores = {p: objective(a, f"neut_{p}") for p in grid}
        print("  objective(A) by proportion:", {p: round(s, 4) for p, s in scores.items()})
        best_p = max(scores, key=scores.get)
        df["ens_neutral"] = df[f"neut_{best_p}"]
        cols.append("ens_neutral")

    b = df[df["era"].isin(b_eras)]
    res = evaluate(b, cols, TARGET)
    rows = []
    for c in cols:
        r = res[c]
        rows.append({
            "model": c,
            "corr": r["corr"]["mean"], "sharpe": r["corr"]["sharpe"],
            "bmc": r["bmc"]["mean"], "mmc": r.get("mmc", {}).get("mean", np.nan),
            "payout_bmc": r["payout_bmc"]["mean"],
            "payout_mmc": r.get("payout_mmc", {}).get("mean", np.nan),
        })
    table = pd.DataFrame(rows).set_index("model")
    print(f"\nscored on B ({len(b_eras)} eras; MMC on {res[cols[0]].get('mmc', {}).get('n_eras', 0)}"
          " eras with meta model):")
    print(table.round(4).to_string())

    final = "ens_neutral" if "ens_neutral" in df.columns else "ens_greedy"
    out_dir = ROOT / "artifacts" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df[["era", final]].rename(columns={final: "prediction"}).to_parquet(
        out_dir / "validation_preds.parquet")

    if not args.no_log:
        log_run(args.name, {"members": args.names, "chosen": chosen,
                            "neutralize": args.neutralize, "proportion": best_p},
                {f"B.{c}.{k}": float(v) for c, row in table.iterrows() for k, v in row.items()},
                started, kind="ensemble")
    print(json.dumps({"chosen": chosen, "neutral_proportion": best_p}))


if __name__ == "__main__":
    main()
