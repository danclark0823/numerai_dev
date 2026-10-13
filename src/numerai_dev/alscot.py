"""Research-only ridge with a joint historical covariance-stability penalty."""

# Shelved after dc_alscot_1024_v1 (2026-10-13). Nested historical selection chose
# lambda=100 in all three folds, but outer CORR and contribution lost to ridge
# in every window. MMC was -0.005447 vs ridge -0.004378; worst-window CORR also
# deteriorated. The training covariance proxy decreased without generalizing.
# No tuning after outer evaluation, scale-up, or deployment. This does not reject
# every stability objective; see docs/alscot.md and the frozen source snapshots.

import numpy as np
import pandas as pd

from numerai_dev.bendonal import sufficient_stats, validate_bins
from numerai_dev.evaluation import per_era_corr


def stability_matrix(x, y, eras, block_eras=52):
    """Equal-era covariances, then equal-block population covariance (PSD)."""
    x = validate_bins(x)
    y, eras = np.asarray(y, dtype=float), np.asarray(eras)
    if (y.shape != (len(x),) or eras.shape != y.shape or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(eras).all()
            or np.any(eras != np.floor(eras)) or block_eras < 1):
        raise ValueError("Invalid labels, eras or block size")
    blocks = {}
    for era in np.unique(eras):
        mask = eras == era
        xe, ye = x[mask].astype(float), y[mask]
        if len(ye) < 2:
            raise ValueError("Need at least two rows per era")
        covariance = (xe - xe.mean(axis=0)).T @ (ye - ye.mean()) / len(ye)
        blocks.setdefault(int(era) // block_eras, []).append(covariance)
    if len(blocks) < 2:
        raise ValueError("Need at least two historical blocks")
    block_covariances = np.array([np.mean(blocks[b], axis=0) for b in sorted(blocks)])
    centered = block_covariances - block_covariances.mean(axis=0)
    penalty = centered.T @ centered / len(centered)
    scale = float(np.trace(penalty) / x.shape[1])
    if scale > 0:
        penalty /= scale
    return penalty, block_covariances, scale


class StableRidge:
    def __init__(self, alpha=10, strength=0, block_eras=52):
        if not np.isfinite([alpha, strength]).all() or alpha <= 0 or strength < 0:
            raise ValueError("Require positive alpha and nonnegative strength")
        self.alpha, self.strength, self.block_eras = alpha, strength, block_eras

    def fit(self, x, y, eras):
        stats = sufficient_stats([x], y, nonlinear=False)
        self.penalty, self.block_covariances, self.penalty_scale = stability_matrix(
            x, y, eras, self.block_eras)
        self.mean, self.intercept = stats.mean, stats.y_mean
        matrix = stats.gram / stats.n + self.strength * self.penalty
        matrix[np.diag_indices_from(matrix)] += self.alpha
        self.beta = np.linalg.solve(matrix, stats.cross / stats.n)
        return self

    def predict(self, x, batch_size=4096):
        x = validate_bins(x)
        if x.shape[1] != len(self.mean) or batch_size < 1:
            raise ValueError("Invalid feature count or batch size")
        result = np.empty(len(x))
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].astype(float)
            result[start:start + len(batch)] = (batch - self.mean) @ self.beta + self.intercept
        return result


def select_strength(x, y, eras, outer_start, strengths, alpha=10, block_eras=52,
                    gap=13, inner_eras=52, inner_folds=3):
    """Only rows before the outer embargo can influence nested model selection."""
    eras, y = np.asarray(eras), np.asarray(y)
    if inner_folds < 1 or inner_eras < 1 or gap < 1 or not strengths or 0 not in strengths:
        raise ValueError("Invalid selection design; include zero fallback")
    records = {float(value): [] for value in strengths}
    audits = []
    end = outer_start - gap
    starts = [end - inner_eras - i * (inner_eras + gap) for i in range(inner_folds)][::-1]
    for start in starts:
        train = eras < start - gap
        valid = (eras >= start) & (eras < start + inner_eras)
        if not train.any() or not valid.any():
            raise ValueError("Empty inner split")
        audits.append({"validation_start": int(start), "validation_end": int(start + inner_eras - 1),
                       "train_end": int(eras[train].max()), "train_rows": int(train.sum()),
                       "validation_rows": int(valid.sum()),
                       "validation_sampled_eras": int(np.unique(eras[valid]).size)})
        for strength, scores in records.items():
            model = StableRidge(alpha, strength, block_eras).fit(x[train], y[train], eras[train])
            frame = pd.DataFrame({"era": eras[valid], "target": y[valid],
                                  "prediction": model.predict(x[valid])})
            score = float(per_era_corr(frame, ["prediction"], "target").prediction.mean())
            if not np.isfinite(score):
                raise ValueError("Nonfinite inner score")
            scores.append(score)
    selected = max(records, key=lambda s: (min(records[s]), np.mean(records[s]), -s))
    return selected, {"selected": selected, "scores": records, "folds": audits}
