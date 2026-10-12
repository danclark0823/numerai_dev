"""Derived training targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def gaussianize(s: pd.Series) -> pd.Series:
    """Rank to (0, 1) then map through the normal inverse CDF."""
    r = s.rank(method="average")
    return pd.Series(norm.ppf((r - 0.5) / len(r)), index=s.index)


def residualize(df: pd.DataFrame, target: str, ref: str | list[str]) -> pd.Series:
    """Per era, the part of the (centered) target not linearly explained by the
    gaussianized reference prediction(s). Training on this pushes a model toward
    signal the reference models don't already capture."""
    refs = [ref] if isinstance(ref, str) else list(ref)

    def one_era(d: pd.DataFrame) -> pd.Series:
        t = d[target] - d[target].mean()
        b = np.column_stack([gaussianize(d[r]) for r in refs])
        beta, *_ = np.linalg.lstsq(b, t.to_numpy(), rcond=None)
        return t - b @ beta

    return df.groupby("era", group_keys=False)[[target, *refs]].apply(one_era)
