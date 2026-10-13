"""Frozen, label-free random-subspace ridge ensemble; research only."""

# Research outcome (2026-10-13): shelved, not deployed. Sixteen fixed specialists
# reached CORR 0.022130 / MMC 0.005688, but matched all-feature ridge achieved
# 0.025220 / 0.007350. The ensemble lost CORR in every outer window and contributed
# positively beyond all-feature ridge in only 1/3. Both beat the 128-feature
# control: wider input coverage helped here, specialist partitioning did not.
# No outcome-based subset/weight tuning followed. See docs/hesia.md; frozen run
# sources remain in artifacts/dc_hesia_1024_v1 for reproducibility.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from numerai_dev.bendonal import validate_bins
from numerai_dev.diversifiers import era_gaussian


def feature_subsets(n_features, partitions=2, groups=8, seed=42):
    if n_features < groups or groups < 1 or partitions < 1:
        raise ValueError("Invalid partition dimensions")
    rng = np.random.default_rng(seed)
    return [np.sort(part) for _ in range(partitions)
            for part in np.array_split(rng.permutation(n_features), groups)]


class RunningStats:
    """Uncentered summaries allow exact incremental chronological accumulation."""

    def __init__(self, width):
        if width < 1:
            raise ValueError("Need features")
        self.n, self.sy = 0, 0.0
        self.sx = np.zeros(width)
        self.gram = np.zeros((width, width))
        self.cross = np.zeros(width)

    def update(self, x, y, batch_size=4096):
        y = np.asarray(y, dtype=float)
        if (np.ndim(x) != 2 or x.shape[1] != len(self.sx) or y.shape != (len(x),)
                or not np.isfinite(y).all() or batch_size < 1):
            raise ValueError("Invalid feature/label dimensions or batch size")
        for start in range(0, len(x), batch_size):
            batch = validate_bins(x[start:start + batch_size]).astype(float)
            target = y[start:start + len(batch)]
            self.gram += batch.T @ batch
            self.cross += batch.T @ target
            self.sx += batch.sum(axis=0)
            self.sy += float(target.sum())
            self.n += len(batch)
        return self

    def fit(self, columns, alpha=10):
        columns = np.asarray(columns, dtype=int)
        if (not self.n or not np.isfinite(alpha) or alpha <= 0 or columns.ndim != 1
                or not len(columns) or len(np.unique(columns)) != len(columns)
                or columns.min() < 0 or columns.max() >= len(self.sx)):
            raise ValueError("Invalid ridge fit")
        mean, intercept = self.sx[columns] / self.n, self.sy / self.n
        gram = self.gram[np.ix_(columns, columns)] - self.n * np.outer(mean, mean)
        cross = self.cross[columns] - self.n * mean * intercept
        gram[np.diag_indices_from(gram)] += self.n * alpha
        beta = np.linalg.solve(gram, cross)
        if not np.isfinite(beta).all():
            raise ValueError("Nonfinite coefficients")
        return mean, intercept, beta


@dataclass
class SpecialistBundle:
    features: list
    subsets: list
    coefficients: np.ndarray
    offsets: np.ndarray

    @classmethod
    def fit_stats(cls, stats, features, subsets, small_columns, alpha=10):
        if len(features) != len(stats.sx) or len(set(features)) != len(features) or not subsets:
            raise ValueError("Invalid feature names or subsets")
        groups = [np.asarray(small_columns), np.arange(len(features)), *subsets]
        coefficients = np.zeros((len(features), len(groups)))
        offsets = np.empty(len(groups))
        for j, columns in enumerate(groups):
            mean, intercept, beta = stats.fit(columns, alpha)
            coefficients[columns, j] = beta
            offsets[j] = intercept - mean @ beta
        return cls(list(features), subsets, coefficients, offsets)

    def raw_predict(self, x, batch_size=4096):
        if np.ndim(x) != 2 or x.shape[1] != len(self.features) or batch_size < 1:
            raise ValueError("Invalid feature dimensions or batch size")
        out = np.empty((len(x), self.coefficients.shape[1]))
        for start in range(0, len(x), batch_size):
            batch = validate_bins(x[start:start + batch_size]).astype(float)
            out[start:start + len(batch)] = batch @ self.coefficients + self.offsets
        return out

    def predict(self, x, eras, batch_size=4096):
        eras = np.asarray(eras)
        if eras.shape != (len(x),) or pd.isna(eras).any():
            raise ValueError("Need aligned nonmissing eras")
        raw = self.raw_predict(x, batch_size)
        # Normalize only AFTER collecting each complete era, never per numerical batch.
        normalized = np.column_stack([era_gaussian(raw[:, j], eras)
                                      for j in range(2, raw.shape[1])])
        return np.column_stack([raw[:, :2], normalized.mean(axis=1)])

    def submission(self, features, eras, batch_size=4096):
        if (not features.index.is_unique or not features.columns.is_unique
                or not features.index.equals(eras.index)):
            raise ValueError("Require unique aligned IDs and columns")
        raw = self.predict(features.loc[:, self.features].to_numpy(), eras.to_numpy(), batch_size)[:, 2]
        return pd.Series(raw, index=features.index).groupby(eras).rank(pct=True).rename("prediction")
