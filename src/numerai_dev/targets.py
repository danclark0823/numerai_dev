"""Derived training targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def gaussianize(s: pd.Series) -> pd.Series:
    """Rank to (0, 1) then map through the normal inverse CDF."""
    r = s.rank(method="average")
    return pd.Series(norm.ppf((r - 0.5) / len(r)), index=s.index)


def residualize(df: pd.DataFrame, target: str, ref: str) -> pd.Series:
    """Per era, the part of the (centered) target not linearly explained by the
    gaussianized reference predictions. Training on this pushes a model toward
    signal the reference model doesn't already capture."""

    def one_era(d: pd.DataFrame) -> pd.Series:
        t = d[target] - d[target].mean()
        b = gaussianize(d[ref])
        beta = float(np.dot(t, b) / np.dot(b, b))
        return t - beta * b

    return df.groupby("era", group_keys=False)[[target, ref]].apply(one_era)
