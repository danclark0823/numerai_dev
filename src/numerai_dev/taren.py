"""Research-only recency-weighted ridge built from exact per-era sufficient statistics.

Era e in a fit whose last training era is L has weight 0.5 ** ((L - e) / h). The
penalty is alpha * sum(weights), so h = inf reproduces `models.RidgeModel`. Scaling
every weight by a constant leaves the solution unchanged, so accumulators are
decayed relative to the most recently added era.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numerai_dev.evaluation import per_era_corr

INF = float("inf")
# float32 products of bins 0-4 are exact while every partial sum stays below 2**24.
MAX_EXACT_ROWS = (1 << 24) // 16


def half_life_label(h: float) -> str:
    return "hinf" if h == INF else f"h{int(h)}"


def era_stats(x: np.ndarray, y: np.ndarray):
    """Exact (X'X, X'y, sum x, sum y, n) for rows of a single era."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("Need a non-empty 2-D feature array and aligned labels")
    if len(x) > MAX_EXACT_ROWS:
        raise ValueError("Too many rows for an exact float32 Gram matrix; split the era")
    if x.min() < 0 or x.max() > 4 or not np.isfinite(y).all():
        raise ValueError("Features must be bins 0-4 and labels finite")
    xf = x.astype(np.float32)
    gram = (xf.T @ xf).astype(np.float64)
    xd = x.astype(np.float64)
    return gram, xd.T @ y, xd.sum(axis=0), float(y.sum()), len(y)


class WeightedRidge:
    """Fitted recency-weighted ridge: prediction = (x - mean) @ beta + intercept."""

    def __init__(self, mean, beta, intercept, half_life, alpha, train_end, train_start,
                 n_eras, rows, weight_sum, effective_rows):
        self.mean, self.beta, self.intercept = mean, beta, float(intercept)
        self.half_life, self.alpha = float(half_life), float(alpha)
        self.train_end, self.train_start, self.n_eras = int(train_end), int(train_start), n_eras
        self.rows, self.weight_sum, self.effective_rows = rows, weight_sum, effective_rows

    @property
    def offset(self) -> float:
        return self.intercept - float(self.mean @ self.beta)

    def predict(self, x, batch_size=8192):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != len(self.beta) or batch_size < 1:
            raise ValueError("Invalid feature count or batch size")
        out = np.empty(len(x))
        for i in range(0, len(x), batch_size):
            out[i:i + batch_size] = (x[i:i + batch_size].astype(np.float64) - self.mean) @ self.beta
        return out + self.intercept

    def audit(self) -> dict:
        return {"half_life": half_life_label(self.half_life), "train_start": self.train_start,
                "train_end": self.train_end, "train_eras": self.n_eras, "rows": self.rows,
                "weight_sum": self.weight_sum, "effective_rows": self.effective_rows,
                "coefficient_norm": float(np.linalg.norm(self.beta))}


class DecayedStats:
    """One decayed accumulator per half-life, fed one complete era at a time."""

    def __init__(self, n_features: int, half_lives):
        half_lives = [float(h) for h in half_lives]
        if not half_lives or len(set(half_lives)) != len(half_lives) or any(
                not h > 0 for h in half_lives):
            raise ValueError("Half-lives must be distinct and positive (inf allowed)")
        p = n_features
        self.half_lives = half_lives
        self.gram = {h: np.zeros((p, p)) for h in half_lives}
        self.cross = {h: np.zeros(p) for h in half_lives}
        self.xsum = {h: np.zeros(p) for h in half_lives}
        self.ysum = dict.fromkeys(half_lives, 0.0)
        self.wsum = dict.fromkeys(half_lives, 0.0)
        self.wsq = dict.fromkeys(half_lives, 0.0)
        self.first_era = self.last_era = None
        self.n_eras = self.rows = 0

    def add(self, era: int, gram, cross, xsum, ysum, n):
        era = int(era)
        if self.last_era is not None and era <= self.last_era:
            raise ValueError("Eras must be added once each, in increasing order")
        for h in self.half_lives:
            if self.last_era is not None and h != INF:
                f = 0.5 ** ((era - self.last_era) / h)
                self.gram[h] *= f
                self.cross[h] *= f
                self.xsum[h] *= f
                self.ysum[h] *= f
                self.wsum[h] *= f
                self.wsq[h] *= f * f
            self.gram[h] += gram
            self.cross[h] += cross
            self.xsum[h] += xsum
            self.ysum[h] += ysum
            self.wsum[h] += n
            self.wsq[h] += n
        if self.first_era is None:
            self.first_era = era
        self.last_era = era
        self.n_eras += 1
        self.rows += int(n)

    def fit(self, alpha: float) -> dict[float, WeightedRidge]:
        if self.last_era is None:
            raise ValueError("No training eras accumulated")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Require positive alpha")
        models = {}
        for h in self.half_lives:
            w = self.wsum[h]
            mean = self.xsum[h] / w
            y_mean = self.ysum[h] / w
            matrix = self.gram[h] - w * np.outer(mean, mean)
            matrix[np.diag_indices_from(matrix)] += alpha * w
            beta = np.linalg.solve(matrix, self.cross[h] - w * mean * y_mean)
            models[h] = WeightedRidge(mean, beta, y_mean, h, alpha, self.last_era,
                                      self.first_era, self.n_eras, self.rows, w,
                                      w * w / self.wsq[h])
        return models


class EndpointFitter:
    """Streams rows in era order and fits every half-life at each requested endpoint
    (the last era a fit may use). A fit is solved before any later era is added."""

    def __init__(self, n_features: int, half_lives, endpoints, alpha: float):
        self.stats = DecayedStats(n_features, half_lives)
        self.alpha = alpha
        self.pending = sorted({int(e) for e in endpoints})
        self.models: dict[int, dict[float, WeightedRidge]] = {}
        self.current: int | None = None
        self._buffer = None

    def _flush(self):
        if self._buffer is not None:
            self.stats.add(self.current, *self._buffer)
            self._buffer = None

    def _solve_through(self, last_allowed: float):
        while self.pending and self.pending[0] <= last_allowed:
            endpoint = self.pending.pop(0)
            self.models[endpoint] = self.stats.fit(self.alpha)
            assert self.stats.last_era <= endpoint

    def advance(self, era: int):
        """Call before using rows of `era` (training or prediction)."""
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


def fit_through(x, y, eras, endpoints, half_lives, alpha):
    """In-memory convenience wrapper around EndpointFitter."""
    eras = np.asarray(eras, dtype=int)
    fitter = EndpointFitter(np.asarray(x).shape[1], half_lives, endpoints, alpha)
    for era in np.unique(eras):
        mask = eras == era
        fitter.advance(era)
        fitter.add_rows(era, x[mask], np.asarray(y)[mask])
    return fitter.finish()


def inner_windows(outer_start: int, gap: int = 13, inner_eras: int = 52, folds: int = 3):
    """[(validation_start, validation_end, train_end)], same geometry as Alscot."""
    if folds < 1 or inner_eras < 1 or gap < 1:
        raise ValueError("Invalid inner design")
    end = outer_start - gap
    starts = [end - inner_eras - i * (inner_eras + gap) for i in range(folds)][::-1]
    return [(s, s + inner_eras - 1, s - gap - 1) for s in starts]


def window_corr(predictions, target, eras) -> float:
    frame = pd.DataFrame({"era": eras, "prediction": predictions, "target": target})
    score = float(per_era_corr(frame, ["prediction"], "target").prediction.mean())
    if not np.isfinite(score):
        raise ValueError("Nonfinite inner score")
    return score


def select_half_life(scores: dict[float, list[float]], tolerance: float = 1e-6) -> float:
    """Highest mean inner CORR; near-ties go to the longer half-life (inf = Rondal)."""
    if INF not in scores:
        raise ValueError("The grid must include h = inf (the Rondal fallback)")
    means = {h: float(np.mean(v)) for h, v in scores.items()}
    best = max(means.values())
    return max(h for h, m in means.items() if m >= best - tolerance)


def screening_decision(metrics: dict, corr_tolerance: float = 0.002,
                       zero: float = 1e-12) -> dict:
    """Frozen screen for the inner-selected `taren` column against `rondal`. A
    contribution within `zero` of 0 (e.g. Taren identical to Rondal) is not positive."""
    windows = [w for w in metrics if w != "all"]
    contributions = {w: metrics[w]["taren"]["rondal_contribution"]["mean"] for w in windows}
    positive = sum(v > zero for v in contributions.values())
    mmc = {w: (metrics[w]["taren"]["mmc"]["mean"], metrics[w]["rondal"]["mmc"]["mean"])
           for w in windows if "mmc" in metrics[w]["taren"]}
    mmc_ok = bool(mmc) and all(t >= 0 and t >= r for t, r in mmc.values())
    window_corr_ok = all(metrics[w]["taren"]["corr"]["mean"] > 0 for w in windows)
    pooled = (metrics["all"]["taren"]["corr"]["mean"], metrics["all"]["rondal"]["corr"]["mean"])
    pooled_ok = pooled[0] >= pooled[1] - corr_tolerance
    checks = {"positive_contribution_windows": positive,
              "contribution_by_window": contributions,
              "contribution_ok": positive >= 2,
              "mmc_by_window": {w: {"taren": t, "rondal": r} for w, (t, r) in mmc.items()},
              "mmc_ok": mmc_ok,
              "positive_corr_every_window": window_corr_ok,
              "pooled_corr": {"taren": pooled[0], "rondal": pooled[1]},
              "pooled_corr_ok": pooled_ok}
    passed = bool(checks["contribution_ok"] and mmc_ok and window_corr_ok and pooled_ok)
    return {**checks, "screen_passed": passed, "deployment": False, "shelve": not passed}
