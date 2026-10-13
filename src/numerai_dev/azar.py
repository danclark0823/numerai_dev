"""Research-only ridge fitted on several residualized targets that share one X'X.

Unweighted, penalty alpha * n, so each target's fit equals `models.RidgeModel` on
that target. One Cholesky factorization per endpoint serves every target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, rankdata

from numerai_dev.evaluation import per_era_contribution
from numerai_dev.taren import INF, MAX_EXACT_ROWS, WeightedRidge


def era_stats(x: np.ndarray, y: np.ndarray):
    """Exact (X'X, X'Y, sum x, sum Y, n) for rows of one era; Y is (n, k)."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(y) != len(x) or not len(x):
        raise ValueError("Need a non-empty 2-D feature array and aligned 2-D labels")
    if len(x) > MAX_EXACT_ROWS:
        raise ValueError("Too many rows for an exact float32 Gram matrix; split the era")
    if x.min() < 0 or x.max() > 4 or not np.isfinite(y).all():
        raise ValueError("Features must be bins 0-4 and labels finite")
    xf = x.astype(np.float32)
    xd = x.astype(np.float64)
    return (xf.T @ xf).astype(np.float64), xd.T @ y, xd.sum(axis=0), y.sum(axis=0), len(y)


class MultiTargetStats:
    def __init__(self, n_features: int, n_targets: int):
        self.gram = np.zeros((n_features, n_features))
        self.cross = np.zeros((n_features, n_targets))
        self.xsum = np.zeros(n_features)
        self.ysum = np.zeros(n_targets)
        self.rows = self.n_eras = 0
        self.first_era = self.last_era = None

    def add(self, era, gram, cross, xsum, ysum, n):
        era = int(era)
        if self.last_era is not None and era <= self.last_era:
            raise ValueError("Eras must be added once each, in increasing order")
        self.gram += gram
        self.cross += cross
        self.xsum += xsum
        self.ysum += ysum
        self.rows += int(n)
        self.n_eras += 1
        self.first_era = era if self.first_era is None else self.first_era
        self.last_era = era

    def fit(self, alpha: float, names: list[str]) -> dict[str, WeightedRidge]:
        if self.last_era is None:
            raise ValueError("No training eras accumulated")
        if not np.isfinite(alpha) or alpha <= 0 or len(names) != len(self.ysum):
            raise ValueError("Require positive alpha and one name per target")
        n = self.rows
        mean = self.xsum / n
        y_mean = self.ysum / n
        matrix = self.gram - n * np.outer(mean, mean)
        matrix[np.diag_indices_from(matrix)] += alpha * n
        beta = cho_solve(cho_factor(matrix), self.cross - n * np.outer(mean, y_mean))
        return {name: WeightedRidge(mean, beta[:, j].copy(), y_mean[j], INF, alpha,
                                    self.last_era, self.first_era, self.n_eras, n, float(n),
                                    float(n))
                for j, name in enumerate(names)}


class EndpointFitter:
    """Streams rows in era order; fits every target at each endpoint (last usable era)
    before any later era is added."""

    def __init__(self, n_features: int, names: list[str], endpoints, alpha: float):
        if not names or len(set(names)) != len(names):
            raise ValueError("Target names must be distinct")
        self.names = list(names)
        self.stats = MultiTargetStats(n_features, len(names))
        self.alpha = alpha
        self.pending = sorted({int(e) for e in endpoints})
        self.models: dict[int, dict[str, WeightedRidge]] = {}
        self.current: int | None = None
        self._buffer = None

    def _flush(self):
        if self._buffer is not None:
            self.stats.add(self.current, *self._buffer)
            self._buffer = None

    def _solve_through(self, last_allowed: float):
        while self.pending and self.pending[0] <= last_allowed:
            endpoint = self.pending.pop(0)
            self.models[endpoint] = self.stats.fit(self.alpha, self.names)
            assert self.stats.last_era <= endpoint

    def advance(self, era: int):
        era = int(era)
        if self.current is not None and era < self.current:
            raise ValueError("Eras must arrive in nondecreasing order")
        if era == self.current:
            return
        self._flush()
        self._solve_through(era - 1)
        self.current = era

    def add_rows(self, era: int, x, y):
        if int(era) != self.current:
            raise ValueError("Call advance(era) before adding its rows")
        for i in range(0, len(x), MAX_EXACT_ROWS):
            stats = era_stats(x[i:i + MAX_EXACT_ROWS], y[i:i + MAX_EXACT_ROWS])
            if self._buffer is None:
                self._buffer = list(stats)
            else:
                for k in range(4):
                    self._buffer[k] = self._buffer[k] + stats[k]
                self._buffer[4] += stats[4]

    def finish(self):
        self._flush()
        self._solve_through(INF)
        return self.models


def fit_through(x, y, eras, endpoints, names, alpha):
    eras = np.asarray(eras, dtype=int)
    y = np.asarray(y)
    fitter = EndpointFitter(np.asarray(x).shape[1], names, endpoints, alpha)
    for era in np.unique(eras):
        mask = eras == era
        fitter.advance(era)
        fitter.add_rows(era, x[mask], y[mask])
    return fitter.finish()


def _gaussianize(values: np.ndarray) -> np.ndarray:
    return norm.ppf((rankdata(values, method="average") - 0.5) / len(values))


def residualize_many(targets: np.ndarray, refs: np.ndarray, eras: np.ndarray) -> np.ndarray:
    """Per era, each centered target minus its least-squares fit on the gaussianized
    references; equals `targets.residualize` column by column when every target is
    defined on the same rows within each era."""
    targets = np.asarray(targets, dtype=np.float64)
    refs = np.asarray(refs, dtype=np.float64)
    eras = np.asarray(eras)
    if targets.ndim != 2 or refs.ndim != 2 or not len(targets) == len(refs) == len(eras):
        raise ValueError("Targets, references and eras must be aligned 2-D arrays")
    if not (np.isfinite(targets).all() and np.isfinite(refs).all()):
        raise ValueError("Residualize only rows where every target and reference is set")
    out = np.empty_like(targets)
    for idx in pd.Series(np.arange(len(eras))).groupby(eras).indices.values():
        t = targets[idx] - targets[idx].mean(axis=0)
        b = np.column_stack([_gaussianize(refs[idx, j]) for j in range(refs.shape[1])])
        beta, *_ = np.linalg.lstsq(b, t, rcond=None)
        out[idx] = t - b @ beta
    return out


def mean_contribution(frame: pd.DataFrame, cols: list[str], target: str, ref: str) -> pd.Series:
    """Mean per-era contribution of each column to `ref`'s correlation with `target`."""
    return per_era_contribution(frame, cols, target, ref).mean()


def select_target(scores: dict[str, list[float]], fallback: str, zero: float = 1e-12) -> str:
    """Highest mean inner contribution; the fallback (Rondal's own target) unless some
    alternative is positive."""
    if fallback not in scores:
        raise ValueError("The fallback target must be in the grid")
    means = {t: float(np.mean(v)) for t, v in scores.items() if t != fallback}
    best = max(means, key=lambda t: (means[t], t)) if means else fallback
    return best if means and means[best] > zero else fallback


def screening_decision(metrics: dict, zero: float = 1e-12) -> dict:
    """Frozen screen for the inner-selected `azar` column (a separate model) against
    the Rondal recreation."""
    windows = [w for w in metrics if w != "all"]
    contributions = {w: metrics[w]["azar"]["rondal_contribution"]["mean"] for w in windows}
    positive = sum(v > zero for v in contributions.values())
    mmc = {w: metrics[w]["azar"]["mmc"]["mean"] for w in windows if "mmc" in metrics[w]["azar"]}
    mmc_ok = bool(mmc) and all(v >= 0 for v in mmc.values())
    corr = {w: metrics[w]["azar"]["corr"]["mean"] for w in windows}
    corr_ok = all(v > 0 for v in corr.values())
    passed = bool(positive >= 2 and mmc_ok and corr_ok)
    return {"positive_contribution_windows": positive, "contribution_by_window": contributions,
            "contribution_ok": positive >= 2, "mmc_by_window": mmc, "mmc_ok": mmc_ok,
            "corr_by_window": corr, "positive_corr_every_window": corr_ok,
            "screen_passed": passed, "deployment": False, "shelve": not passed}
