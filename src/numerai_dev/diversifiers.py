"""Experimental diversifiers; fit only on embargoed, historical observations."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import norm, rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge


def era_gaussian(values, eras):
    out = np.empty(len(values), dtype=float)
    for era in np.unique(eras):
        mask = eras == era
        out[mask] = norm.ppf((rankdata(np.asarray(values)[mask]) - 0.5) / mask.sum())
    return out


def residual_labels(y, reference, eras):
    out = np.empty(len(y), dtype=float)
    ref = era_gaussian(reference, eras)
    for era in np.unique(eras):
        m = eras == era
        t, r = y[m] - y[m].mean(), ref[m]
        out[m] = t - r * (r @ t) / max(r @ r, 1e-12)
    return out


class StablePortfolio:
    """Signed rank portfolio selected by consistency across chronological blocks."""

    def __init__(self, top_k=20, min_consistency=0.75, blocks=8, max_exposure=0.5):
        self.top_k, self.min_consistency = top_k, min_consistency
        self.blocks, self.max_exposure = blocks, max_exposure

    def fit(self, x, y, eras, reference):
        scores, exposures = [], []
        for era in np.unique(eras):
            m = eras == era
            z = rankdata(x[m], axis=0).astype(float)
            z -= z.mean(axis=0)
            z /= np.maximum(np.linalg.norm(z, axis=0), 1e-12)
            t = y[m] - y[m].mean()
            r = rankdata(reference[m]).astype(float)
            r -= r.mean()
            scores.append(z.T @ t / max(np.linalg.norm(t), 1e-12))
            exposures.append(z.T @ r / max(np.linalg.norm(r), 1e-12))
        scores = np.asarray(scores)
        blocks = np.array_split(scores, min(self.blocks, len(scores)))
        block_scores = np.array([b.mean(axis=0) for b in blocks])
        direction = np.sign(np.median(block_scores, axis=0))
        consistency = (block_scores * direction > 0).mean(axis=0)
        eligible = ((consistency >= self.min_consistency)
                    & (np.abs(exposures).mean(axis=0) <= self.max_exposure))
        strength = np.abs(np.median(block_scores, axis=0))
        selected = np.flatnonzero(eligible & (strength > 0))
        selected = selected[np.argsort(strength[selected])[-self.top_k:]]
        if not len(selected):
            raise ValueError("No stable features qualify; report this fold as failed")
        self.columns = selected
        self.weights = direction[selected] / len(selected)
        self.consistency = consistency[selected]
        return self

    def predict(self, x, eras):
        out = np.empty(len(x))
        for era in np.unique(eras):
            m = eras == era
            ranks = rankdata(x[m][:, self.columns], axis=0) / m.sum() - 0.5
            out[m] = ranks @ self.weights
        return out


class FactorizationMachine:
    """Mini-batch Adam on squared error with a low-rank pairwise interaction term."""

    def __init__(self, rank=8, epochs=8, batch_size=1024, learning_rate=0.003,
                 penalty=0.01, seed=42, interaction_only=False):
        self.rank, self.epochs, self.batch_size = rank, epochs, batch_size
        self.learning_rate, self.penalty, self.seed = learning_rate, penalty, seed
        self.interaction_only = interaction_only

    def _forward(self, x):
        xv = x @ self.v
        return self.bias[0] + x @ self.w + 0.5 * (
            (xv * xv).sum(axis=1) - (x * x) @ (self.v * self.v).sum(axis=1))

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 0.1)
        self.w = np.zeros(x.shape[1])
        self.v = rng.normal(0, 0.01 / np.sqrt(self.rank), (x.shape[1], self.rank))
        self.bias = np.array([y.mean()])
        params = [self.w, self.v, self.bias]
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                ids = order[start:start + self.batch_size]
                z = (x[ids].astype(float) - self.mean) / self.scale
                error = (self._forward(z) - y[ids]) / len(ids)
                grads = [z.T @ error + self.penalty * self.w,
                         z.T @ (error[:, None] * (z @ self.v))
                         - ((z * z).T @ error)[:, None] * self.v
                         + self.penalty * self.v,
                         np.array([error.sum()])]
                if self.interaction_only:
                    grads[0].fill(0)
                step += 1
                for p, g, m, s in zip(params, grads, first, second, strict=True):
                    m *= 0.9
                    m += 0.1 * g
                    s *= 0.999
                    s += 0.001 * g * g
                    p -= self.learning_rate * (m / (1 - 0.9**step)) / (
                        np.sqrt(s / (1 - 0.999**step)) + 1e-8)
        if not all(np.isfinite(p).all() for p in params):
            raise ValueError("Factorization machine diverged")
        return self

    def predict(self, x):
        return np.concatenate([self._forward((x[i:i + self.batch_size] - self.mean)
                                             / self.scale)
                               for i in range(0, len(x), self.batch_size)])


def teacher_predictions(x, y, test_x, trees=80, seed=42):
    """Three distinct teachers; all receive the same strictly historical rows."""
    teachers = [Ridge(alpha=10 * len(y)),
                ExtraTreesRegressor(n_estimators=trees, max_depth=7,
                                    min_samples_leaf=50, max_features=0.5,
                                    random_state=seed, n_jobs=2),
                LGBMRegressor(n_estimators=trees, max_depth=4, num_leaves=15,
                              learning_rate=0.05, colsample_bytree=0.5,
                              random_state=seed, verbosity=-1, n_jobs=2)]
    return np.column_stack([m.fit(x, y).predict(test_x) for m in teachers])


def consensus(predictions, eras):
    return np.mean([era_gaussian(predictions[:, i], eras)
                    for i in range(predictions.shape[1])], axis=0)


def historical_folds(eras, min_train_eras, block_eras, gap):
    """Expanding windows; embargo measured in era numbers, never sampled row counts."""
    unique = np.unique(eras)
    for start in range(min_train_eras, len(unique), block_eras):
        test_eras = unique[start:start + block_eras]
        train = eras < test_eras[0] - gap
        test = np.isin(eras, test_eras)
        if np.unique(eras[train]).size >= min_train_eras:
            yield train, test


def crossfit_consensus(x, y, eras, gap=13, min_train_eras=40, block_eras=32, trees=80):
    predictions = np.full(len(y), np.nan)
    audit = []
    for train, test in historical_folds(eras, min_train_eras, block_eras, gap):
        predictions[test] = consensus(teacher_predictions(x[train], y[train], x[test], trees),
                                      eras[test])
        audit.append({"train_end": int(eras[train].max()),
                      "test_start": int(eras[test].min()),
                      "test_end": int(eras[test].max())})
    return predictions, audit
