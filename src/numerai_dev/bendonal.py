"""Additive five-bin response curves with separately penalized slope and curvature."""

# Research outcome (2026-10-13): shelved, not selected as the fifth live model.
# After Wenek's inconsistent interaction signal, we tested a simpler nonlinear
# alternative: independent five-bin feature curves, without pairwise interactions.
# The frozen trial used 128 features, 1,024 rows per sampled training era, linear
# penalty 10, and curvature penalties 10 / 100 against a matched ridge control.
# Both variants improved CORR and contributed beyond ridge in all three windows.
# Pooled CORR was 0.007442 / 0.006296 versus ridge's 0.006146, but available MMC
# remained negative: -0.002517 / -0.004171 versus ridge's -0.004378. Improving a
# weak baseline did not satisfy the predefined requirement for nonnegative MMC.
# MMC covered only the latest 30 eras; these repeatedly inspected development
# windows do not establish fresh confirmation or explain the underlying cause.
# We therefore stopped before scale-up, final refit, packaging, or deployment.
# This rejects the tested configuration, not every possible additive model.
# Preserve this implementation for research history; see docs/bendonal.md and
# artifacts/dc_bendonal_1024_v1 for the full results and frozen source snapshots.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def validate_bins(x):
    x = np.asarray(x)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.number):
        raise ValueError("Features must be a numeric two-dimensional array")
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 4)) or np.any(x != np.floor(x)):
        raise ValueError("Features must be finite integer bins 0–4; missing values are rejected")
    return x.astype(np.int8, copy=False)


def curvature_basis():
    """Three orthonormal contrasts, orthogonal to constant and raw linear bin value."""
    bins = np.arange(5, dtype=float) - 2
    q, _ = np.linalg.qr(np.column_stack([bins**power for power in range(5)]))
    basis = q[:, 2:]
    for j in range(3):
        if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    return basis


def encode(x, basis):
    x = validate_bins(x)
    if basis is None:
        return x.astype(float)
    return np.column_stack([x, basis[x].reshape(len(x), -1)])


@dataclass
class SufficientStats:
    n: int
    p: int
    basis: np.ndarray | None
    mean: np.ndarray
    y_mean: float
    gram: np.ndarray
    cross: np.ndarray
    bin_counts: np.ndarray


def sufficient_stats(sequences, y, nonlinear=True, chunk_rows=4096):
    """One pass over raw batches; retain a small centered Gram matrix, not the design."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or chunk_rows < 1:
        raise ValueError("Need finite nonempty labels and a positive chunk size")
    basis = curvature_basis() if nonlinear else None
    gram = cross = total = counts = None
    pos, p = 0, None
    for sequence in sequences:
        for i in range(0, len(sequence), chunk_rows):
            raw = validate_bins(sequence[i:i + chunk_rows])
            if p is None:
                p = raw.shape[1]
                if not p:
                    raise ValueError("Need at least one feature")
                width = p * (4 if nonlinear else 1)
                gram, cross, total = np.zeros((width, width)), np.zeros(width), np.zeros(width)
                counts = np.zeros((p, 5), dtype=np.int64)
            if raw.shape[1] != p or pos + len(raw) > len(y):
                raise ValueError("Feature or label count mismatch")
            z = encode(raw, basis)
            gram += z.T @ z
            cross += z.T @ y[pos:pos + len(raw)]
            total += z.sum(axis=0)
            for value in range(5):
                counts[:, value] += (raw == value).sum(axis=0)
            pos += len(raw)
    if pos != len(y) or p is None:
        raise ValueError("Feature or label count mismatch")
    mean, y_mean = total / pos, float(y.mean())
    gram -= pos * np.outer(mean, mean)
    cross -= pos * mean * y_mean
    return SufficientStats(pos, p, basis, mean, y_mean, gram, cross, counts)


class AdditiveModel:
    def __init__(self, linear_penalty=10.0, nonlinear_penalty=10.0):
        if linear_penalty <= 0 or nonlinear_penalty <= 0:
            raise ValueError("Penalties must be positive")
        self.linear_penalty, self.nonlinear_penalty = linear_penalty, nonlinear_penalty

    def fit_stats(self, stats):
        self.p = stats.p
        self.basis = None if stats.basis is None else stats.basis.copy()
        self.mean, self.intercept = stats.mean.copy(), stats.y_mean
        self.bin_counts = stats.bin_counts.copy()
        penalties = np.full(len(stats.mean), self.nonlinear_penalty)
        penalties[:self.p] = self.linear_penalty
        gram = stats.gram.copy()
        gram[np.diag_indices_from(gram)] += stats.n * penalties
        self.beta = np.linalg.solve(gram, stats.cross)
        if not np.isfinite(self.beta).all():
            raise ValueError("Nonfinite solution")
        return self

    def fit_stream(self, sequences, y, nonlinear=True, chunk_rows=4096):
        return self.fit_stats(sufficient_stats(sequences, y, nonlinear, chunk_rows))

    def predict(self, x, batch_size=4096, component="all"):
        if component not in ("all", "nonlinear") or batch_size < 1:
            raise ValueError("Invalid component or batch size")
        if np.ndim(x) != 2 or x.shape[1] != self.p:
            raise ValueError("Feature count mismatch")
        out = np.empty(len(x))
        for i in range(0, len(x), batch_size):
            z = encode(x[i:i + batch_size], self.basis) - self.mean
            out[i:i + len(z)] = (z @ self.beta + self.intercept if component == "all"
                                 else z[:, self.p:] @ self.beta[self.p:])
        return out

    def curves(self):
        """Training-centered lookup curves; sum plus intercept reproduces prediction."""
        curves = (np.arange(5)[None, :] - self.mean[:self.p, None]) * self.beta[:self.p, None]
        if self.basis is not None:
            nonlinear = self.basis[None, :, :] - self.mean[self.p:].reshape(self.p, 1, 3)
            curves += (nonlinear * self.beta[self.p:].reshape(self.p, 1, 3)).sum(axis=2)
        return curves

    def submission(self, features, eras, feature_order, batch_size=4096):
        if not features.index.is_unique or not features.index.equals(eras.index) or eras.isna().any():
            raise ValueError("Require aligned unique feature/era IDs and nonmissing eras")
        raw = pd.Series(self.predict(features.loc[:, feature_order].to_numpy(), batch_size),
                        index=features.index)
        return raw.groupby(eras).rank(pct=True).rename("prediction")


def fit_before(x, y, eras, test_start, gap=13, **params):
    mask = np.asarray(eras) < test_start - gap
    return AdditiveModel(**params).fit_stream([x[mask]], np.asarray(y)[mask])


def screening_decision(metrics, candidates, penalties):
    windows = [w for w in metrics if w != "all"]
    diagnostics, passing = {}, []
    for name in candidates:
        contributions = [metrics[w][name]["ridge_contribution"]["mean"] for w in windows]
        wins = sum(metrics[w][name]["corr"]["mean"] > metrics[w]["ridge"]["corr"]["mean"]
                   for w in windows)
        positive = sum(value > 0 for value in contributions)
        pooled_win = metrics["all"][name]["corr"]["mean"] > metrics["all"]["ridge"]["corr"]["mean"]
        mmc = [(metrics[w][name]["mmc"]["mean"], metrics[w]["ridge"]["mmc"]["mean"])
               for w in windows if "mmc" in metrics[w][name]]
        passed = (positive >= 2 and wins >= 2 and pooled_win and bool(mmc)
                  and all(value >= 0 and value >= baseline for value, baseline in mmc))
        diagnostics[name] = {"positive_contribution_windows": positive, "corr_wins": wins,
                             "pooled_corr_improved": pooled_win,
                             "median_contribution": float(np.median(contributions)),
                             "passed": passed}
        if passed:
            passing.append(name)
    selected = max(passing, key=lambda n: (diagnostics[n]["positive_contribution_windows"],
                                          diagnostics[n]["median_contribution"], penalties[n])) if passing else None
    return {"candidates": diagnostics, "selected": selected, "screen_passed": bool(passing),
            "deployment": False, "shelve": not passing}
