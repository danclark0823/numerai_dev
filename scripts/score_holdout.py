"""Score live-model specs on the locked holdout eras. Every call is logged as a
holdout look in experiments/log.jsonl, so the number of looks stays visible.

Usage: python scripts/score_holdout.py --reason "why we're looking"
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import yaml

from numerai_dev.data import data_path, era_key, era_mask, feature_set, load_features, load_index
from numerai_dev.evaluation import (
    BENCHMARK_COL,
    CORR_WEIGHT,
    META_COL,
    MMC_WEIGHT,
    evaluate,
    per_era_contribution,
    per_era_corr,
    prediction_correlation,
)
from numerai_dev.experiment import ROOT, log_run
from numerai_dev.live import load_members, predict_model
from numerai_dev.splits import make_split

TARGET = "target_ender_60"
HOLDOUT_ERAS = 52

# Live models plus candidates under evaluation.
CANDIDATES = {
    "recent_bag4": {"members": ["recent_ender60"], "neutralize": None},
    "recent_resid_ensemble": {"members": ["recent_resid_bench", "recent_teager2b_60"],
                              "neutralize": {"feature_set": "medium", "proportion": 0.5}},
    "recent_tyndal": {"members": ["recent_resid_both"], "neutralize": None},
}


def block_ci(x: np.ndarray, block: int = 13, n_boot: int = 5000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    n = len(x)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means <= 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", required=True)
    parser.add_argument("--baseline", default="lgbm_bag4")
    args = parser.parse_args()
    started = time.time()

    cfg = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text())
    specs = {**cfg["models"], **CANDIDATES}
    version = cfg["version"]

    vidx = load_index(version, "validation", [TARGET])
    resolved = era_mask(vidx, require=[TARGET], data_type="validation")
    tidx = load_index(version, "train", [])
    split = make_split(tidx["era"].unique().tolist(),
                       vidx.loc[resolved, "era"].unique().tolist(), TARGET, HOLDOUT_ERAS)
    mask = era_mask(vidx, eras=split.holdout, require=[TARGET], data_type="validation")
    feats = feature_set(version, cfg["feature_set"])
    x_all = pd.DataFrame(load_features(version, "validation", feats, mask),
                         index=vidx.index[mask], columns=feats)
    eras = vidx.loc[mask, "era"]
    print(f"holdout eras {split.holdout[0]}-{split.holdout[-1]} ({len(split.holdout)}), "
          f"{len(x_all)} rows")

    df = vidx.loc[mask, ["era", TARGET]].copy()
    df = df.join(pd.read_parquet(data_path(version, "meta_model.parquet"), columns=[META_COL]))
    df = df.join(pd.read_parquet(data_path(version, "validation_benchmark_models.parquet"),
                                 columns=[BENCHMARK_COL]))
    members = {n: load_members(n) for s in specs.values() for n in s["members"]}
    for name, spec in specs.items():
        x_neu = None
        if spec.get("neutralize"):
            x_neu = x_all[feature_set(version, spec["neutralize"]["feature_set"])]
        df[name] = predict_model(spec, x_all, x_neu, eras, members)

    names = list(specs)
    res = evaluate(df, names, TARGET)
    rows = []
    for n in names:
        r = res[n]
        rows.append({
            "model": n, "corr": r["corr"]["mean"], "corr_sharpe": r["corr"]["sharpe"],
            "mmc": r["mmc"]["mean"], "mmc_pct_pos": r["mmc"]["pct_positive"],
            "bmc": r["bmc"]["mean"], "payout_mmc": r["payout_mmc"]["mean"],
            "corr_w_meta": prediction_correlation(df, [n, META_COL]).iloc[0, 1],
        })
    table = pd.DataFrame(rows).set_index("model")
    print(table.round(4).to_string())

    corr = per_era_corr(df, names, TARGET)
    mmc = per_era_contribution(df, names, TARGET, META_COL)
    order = sorted(corr.index, key=era_key)
    print(f"\npaired vs {args.baseline} (13-era block bootstrap; only "
          f"{len(order) // 13} blocks, so CIs are wide):")
    for n in names:
        if n == args.baseline:
            continue
        for metric, s in [("corr", corr), ("mmc", mmc),
                          ("payout", CORR_WEIGHT * corr + MMC_WEIGHT * mmc)]:
            d = (s.loc[order, n] - s.loc[order, args.baseline]).to_numpy()
            lo, hi, p = block_ci(d)
            print(f"  {n:18s} {metric:6s} diff={d.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}] "
                  f"P(diff<=0)={p:.3f}")

    log_run("holdout_look", {"reason": args.reason, "models": specs},
            {f"holdout.{m}.{k}": float(v) for m, row in table.iterrows() for k, v in row.items()},
            started, kind="holdout", holdout_scored=True,
            holdout_eras=[split.holdout[0], split.holdout[-1]])


if __name__ == "__main__":
    main()
