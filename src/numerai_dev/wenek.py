"""dc_wenek: chronological ridge residuals plus pairwise interactions.

All prediction normalization needs complete era groups. Numerical batches are
used only to compute raw predictions, never to rank partial cross-sections.
"""

# RESEARCH STATUS (2026-10-13): SHELVED; never promoted to a fifth live model.
# Tried ridge + an interaction-only FM trained on chronological OOF ridge errors,
# plus fixed 10%/25% blends. Used 128 features, 1,024 rows per sampled training era,
# and full test eras in three historical windows (90 eras; MMC available for 30).
# Original rank=8/penalty=0.01 interaction model: MMC +0.00371, but positive
# contribution beyond ridge in only 1/3 windows. Both blends failed the screen.
# Follow-up rank=2 gave MMC +0.00186; penalty=0.1 gave MMC +0.00401. Both still
# contributed positively beyond ridge in only 1/3 windows. Stronger regularization
# reduced early-window losses without establishing a repeatable incremental signal.
# Training/test gaps suggest unstable fitting; they do not prove its sole cause.
# Keep code/artifacts for audit; no 2,048-row run, production refit, or upload.
# Details: docs/wenek.md and docs/wenek_simplify_plan.md; artifacts:
# dc_wenek_1024_v1 and dc_wenek_simplify_v1. This rejects the tested formulation,
# not all interaction models. The four running models were unchanged.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from numerai_dev.diversifiers import FactorizationMachine, era_gaussian, historical_folds
from numerai_dev.models import RidgeModel


def fit_ridge(x, y, alpha):
    return RidgeModel(alpha=alpha).fit_stream([x], y)


def crossfit_ridge(x, y, eras, alpha=10, gap=13, min_train_eras=40, block_eras=32):
    """Raw y - raw ridge prediction defines the interaction training target."""
    predictions = np.full(len(y), np.nan)
    audit = []
    for train, test in historical_folds(eras, min_train_eras, block_eras, gap):
        predictions[test] = fit_ridge(x[train], y[train], alpha).predict(x[test])
        audit.append({"train_start": int(eras[train].min()),
                      "train_end": int(eras[train].max()),
                      "test_start": int(eras[test].min()),
                      "test_end": int(eras[test].max()),
                      "train_rows": int(train.sum()), "test_rows": int(test.sum())})
    return predictions, audit


def fit_window(x, y, eras, start, gap, alpha, crossfit, fm):
    """Apply the outer embargo here so callers cannot accidentally pass future labels."""
    mask = eras < start - gap
    tx, ty, te = x[mask], y[mask], eras[mask]
    if not len(ty):
        raise ValueError("No eligible historical training observations")
    oof, audit = crossfit_ridge(tx, ty, te, alpha=alpha, gap=gap, **crossfit)
    valid = np.isfinite(oof)
    if not valid.any():
        raise ValueError("Not enough history to construct out-of-fold ridge residuals")
    residual = ty[valid] - oof[valid]
    interaction = FactorizationMachine(interaction_only=True, **fm).fit(tx[valid], residual)
    raw_fm = FactorizationMachine(**fm).fit(tx, ty)
    # Matched-row control separates residualization from warm-up row exclusion.
    matched_fm = FactorizationMachine(**fm).fit(tx[valid], ty[valid])
    bundle = WenekBundle(fit_ridge(tx, ty, alpha), interaction, raw_fm, matched_fm)
    return bundle, mask, valid, oof, residual, audit


class WenekBundle:
    def __init__(self, ridge, interaction, raw_fm, matched_fm):
        self.ridge, self.interaction = ridge, interaction
        self.raw_fm, self.matched_fm = raw_fm, matched_fm

    def raw_predictions(self, x, batch_size=8192):
        if batch_size < 1 or len(x) == 0:
            raise ValueError("Need nonempty observations and positive batch_size")
        models = {"ridge": self.ridge, "interaction": self.interaction,
                  "raw_fm": self.raw_fm, "matched_fm": self.matched_fm}
        return {name: np.concatenate([model.predict(x[i:i + batch_size])
                                     for i in range(0, len(x), batch_size)])
                for name, model in models.items()}

    def predict(self, x, eras, batch_size=8192):
        raw = self.raw_predictions(x, batch_size)
        scores = {name: era_gaussian(pred, eras) for name, pred in raw.items()}
        for weight in (0.0, 0.1, 0.25):
            scores[f"blend_{int(100 * weight):02d}"] = blend(
                scores["ridge"], scores["interaction"], weight)
        return scores

    def submission(self, features, eras, feature_order, weight=0.1, batch_size=8192):
        """Research serving path, not connected to the four production models."""
        if not features.index.equals(eras.index) or not features.index.is_unique:
            raise ValueError("Era and feature IDs must match and be unique")
        x = features.loc[:, feature_order].to_numpy()
        raw = self.raw_predictions(x, batch_size)
        era = eras.to_numpy()
        score = blend(era_gaussian(raw["ridge"], era),
                      era_gaussian(raw["interaction"], era), weight)
        out = np.empty(len(score))
        for e in np.unique(era):
            mask = era == e
            out[mask] = rankdata(score[mask]) / mask.sum()
        return pd.Series(out, index=features.index, name="prediction")


def blend(ridge, interaction, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Blend weight must lie in [0, 1]")
    return (1 - weight) * ridge + weight * interaction


def paired_block_ci(values, eras, block=13, n_boot=3000, seed=42):
    """One contiguous window only. Never bridge missing eras or separate regimes."""
    values, eras = np.asarray(values), np.asarray(eras, dtype=int)
    if len(values) != len(eras) or not np.isfinite(values).all():
        raise ValueError("Paired scores must be finite and aligned")
    if np.any(np.diff(eras) != 1):
        raise ValueError("Bootstrap requires a contiguous chronological window")
    if len(values) < 2 * block:
        return {"mean": float(values.mean()), "ci": None, "n_eras": len(values)}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values) - block + 1,
                          (n_boot, int(np.ceil(len(values) / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :len(values)]
    means = values[idx].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci": np.percentile(means, [2.5, 97.5]).tolist(),
            "n_eras": len(values), "block": block,
            "effective_blocks_approx": len(values) / block}
