"""Per-era scoring using Numerai's official numerai-tools implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr

# Numerai Classic payout weights (atomic staking, round 1343+): 3*CORR60 + 15*MMC60.
CORR_WEIGHT = 3.0
MMC_WEIGHT = 15.0

META_COL = "numerai_meta_model"
BENCHMARK_COL = "v53_lgbm_ender60"


def per_era_corr(df: pd.DataFrame, pred_cols: list[str], target: str) -> pd.DataFrame:
    return df.groupby("era")[[*pred_cols, target]].apply(
        lambda d: numerai_corr(d[pred_cols], d[target])
    )


def per_era_contribution(
    df: pd.DataFrame, pred_cols: list[str], target: str, ref_col: str
) -> pd.DataFrame:
    """Contribution of each prediction to `ref_col`'s correlation with the target.
    With the meta model as ref this is MMC; with a benchmark model it is BMC."""
    sub = df.dropna(subset=[ref_col])
    return sub.groupby("era")[[*pred_cols, ref_col, target]].apply(
        lambda d: correlation_contribution(d[pred_cols], d[ref_col], d[target])
    )


def summarize(scores: pd.Series) -> dict[str, float]:
    s = scores.dropna()
    cum = s.cumsum()
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "sharpe": float(s.mean() / s.std()) if s.std() > 0 else float("nan"),
        "pct_positive": float((s > 0).mean()),
        "worst_era": float(s.min()),
        "max_drawdown": float((cum - cum.cummax()).min()),
        "n_eras": len(s),
    }


def evaluate(
    df: pd.DataFrame,
    pred_cols: list[str],
    target: str,
    refs: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return {pred_col: {"corr": summary, <ref>: summary, "payout_<ref>": summary}}.

    `refs` maps a name to a reference column, e.g. {"mmc": "numerai_meta_model",
    "bmc": "v53_lgbm_ender60"}. Refs missing from df (or all-null) are skipped.
    payout_<ref> = 3*CORR + 15*contribution, the Numerai Classic payout formula.
    """
    refs = refs if refs is not None else {"mmc": META_COL, "bmc": BENCHMARK_COL}
    df = df.dropna(subset=[target])
    corr = per_era_corr(df, pred_cols, target)
    contribs = {
        name: per_era_contribution(df, pred_cols, target, col)
        for name, col in refs.items()
        if col in df.columns and df[col].notna().any()
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for col in pred_cols:
        res = {"corr": summarize(corr[col])}
        for name, c in contribs.items():
            res[name] = summarize(c[col])
            joined = pd.concat([corr[col], c[col]], axis=1, join="inner").dropna()
            payout = CORR_WEIGHT * joined.iloc[:, 0] + MMC_WEIGHT * joined.iloc[:, 1]
            res[f"payout_{name}"] = summarize(payout)
        out[col] = res
    return out


def prediction_correlation(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean per-era Spearman correlation between prediction columns."""
    per_era = df.groupby("era")[cols].apply(lambda d: d.rank().corr())
    return per_era.groupby(level=1).mean().loc[cols, cols]


def flatten(results: dict, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in results.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, f"{key}."))
        elif isinstance(v, (int, float, np.floating)):
            flat[key] = float(v)
    return flat
