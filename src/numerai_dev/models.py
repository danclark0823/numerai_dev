"""Model types beyond LightGBM. Each exposes predict(x) and save(path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RidgeModel:
    """Closed-form ridge regression on int8 features, built from X'X in row chunks
    so the float copy never exceeds one chunk. Penalty is alpha * n_rows, so alpha
    is comparable across data sizes (features have variance ~2)."""

    def __init__(self, alpha: float, chunk_rows: int = 100_000):
        self.alpha = alpha
        self.chunk_rows = chunk_rows
        self.mean: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeModel:
        n, p = x.shape
        self.mean = x.mean(axis=0, dtype=np.float64)
        y_mean = float(y.mean())
        xtx = np.zeros((p, p))
        xty = np.zeros(p)
        for i in range(0, n, self.chunk_rows):
            xc = x[i : i + self.chunk_rows].astype(np.float64) - self.mean
            xtx += xc.T @ xc
            xty += xc.T @ (y[i : i + self.chunk_rows] - y_mean)
        xtx[np.diag_indices(p)] += self.alpha * n
        self.beta = np.linalg.solve(xtx, xty)
        self.intercept = y_mean
        return self

    def fit_stream(self, seqs: list, y: np.ndarray) -> RidgeModel:
        """Same fit from row batches of sequences (e.g. ParquetRows), one pass, so
        memory is the p x p summary rather than the data. Centering is applied
        afterwards: X'X_c = X'X - n mu mu', X'y_c = X'y - n mu ybar."""
        p = None
        xtx = xty = sx = None
        n, pos = 0, 0
        y = y.astype(np.float64)
        for seq in seqs:
            for i in range(0, len(seq), self.chunk_rows):
                xb = np.asarray(seq[i : i + self.chunk_rows], dtype=np.float64)
                yb = y[pos : pos + len(xb)]
                if p is None:
                    p = xb.shape[1]
                    xtx, xty, sx = np.zeros((p, p)), np.zeros(p), np.zeros(p)
                xtx += xb.T @ xb
                xty += xb.T @ yb
                sx += xb.sum(axis=0)
                n += len(xb)
                pos += len(xb)
        assert pos == len(y), "labels and streamed rows are misaligned"
        self.mean = sx / n
        y_mean = float(y.mean())
        xtx -= n * np.outer(self.mean, self.mean)
        xty -= n * self.mean * y_mean
        xtx[np.diag_indices(p)] += self.alpha * n
        self.beta = np.linalg.solve(xtx, xty)
        self.intercept = y_mean
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float64) - self.mean) @ self.beta + self.intercept

    def save(self, path: Path) -> None:
        np.savez(path.with_suffix(".npz"), mean=self.mean, beta=self.beta,
                 intercept=self.intercept, alpha=self.alpha)

    @classmethod
    def load(cls, path: Path) -> RidgeModel:
        d = np.load(path)
        m = cls(float(d["alpha"]))
        m.mean, m.beta, m.intercept = d["mean"], d["beta"], float(d["intercept"])
        return m
