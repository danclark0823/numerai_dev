"""Paired comparison of two experiments' dev predictions, era by era.

Usage: python scripts/compare.py <artifact_a> <artifact_b> [--target target_ender_60]

Reports mean per-era difference (b - a) in CORR and BMC with a moving-block bootstrap
CI. Blocks of 13 eras account for the autocorrelation that overlapping 60-day targets
induce between neighboring eras.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from numerai_dev.data import data_path, era_key
from numerai_dev.evaluation import BENCHMARK_COL, per_era_contribution, per_era_corr
from numerai_dev.experiment import ROOT


def block_bootstrap_ci(x: np.ndarray, block: int = 13, n_boot: int = 5000,
                       seed: int = 0) -> tuple[float, float, float]:
    """95% CI for the mean and the fraction of bootstrap means <= 0."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means <= 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--target", default="target_ender_60")
    parser.add_argument("--version", default="v5.3")
    args = parser.parse_args()

    pa = pd.read_parquet(ROOT / "artifacts" / args.a / "validation_preds.parquet")
    pb = pd.read_parquet(ROOT / "artifacts" / args.b / "validation_preds.parquet")
    df = pa.rename(columns={"prediction": "a"}).join(pb[["prediction"]].rename(
        columns={"prediction": "b"}), how="inner")
    df = df.join(pd.read_parquet(data_path(args.version, "validation.parquet"),
                                 columns=[args.target]))
    df = df.join(pd.read_parquet(data_path(args.version, "validation_benchmark_models.parquet"),
                                 columns=[BENCHMARK_COL]))
    df = df.dropna(subset=[args.target])

    scores = {
        "corr": per_era_corr(df, ["a", "b"], args.target),
        "bmc": per_era_contribution(df, ["a", "b"], args.target, BENCHMARK_COL),
    }
    print(f"{args.b} - {args.a}  ({df['era'].nunique()} eras)")
    for name, s in scores.items():
        s = s.loc[sorted(s.index, key=era_key)]
        diff = (s["b"] - s["a"]).to_numpy()
        lo, hi, p = block_bootstrap_ci(diff)
        print(f"  {name:5s} a={s['a'].mean():.4f} b={s['b'].mean():.4f} "
              f"diff={diff.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}] "
              f"P(diff<=0)={p:.3f} b wins {np.mean(diff > 0):.0%} of eras")


if __name__ == "__main__":
    main()
