"""Turn saved experiment models into tournament predictions.

The same `predict_model` is used for live rounds and for checking that the live
path reproduces validated predictions, so what we validated is what we submit.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from numerai_dev.experiment import ROOT
from numerai_dev.targets import gaussianize


def load_members(artifact: str) -> list[lgb.Booster]:
    paths = sorted((ROOT / "artifacts" / artifact).glob("model*.txt"))
    if not paths:
        raise FileNotFoundError(f"no saved models in artifacts/{artifact}")
    return [lgb.Booster(model_file=str(p)) for p in paths]


def neutralize(preds: pd.Series, features: pd.DataFrame, eras: pd.Series,
               proportion: float) -> pd.Series:
    """Per era, subtract `proportion` of the least-squares projection onto features."""
    out = pd.Series(np.nan, index=preds.index)
    for idx in preds.groupby(eras).groups.values():
        x = features.loc[idx].to_numpy(dtype=np.float32)
        x = x - x.mean(axis=0)
        y = preds.loc[idx].to_numpy(dtype=np.float32)
        y = y - y.mean()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        out.loc[idx] = y - proportion * (x @ beta)
    return out


def predict_model(spec: dict, x_all: pd.DataFrame, x_neutral: pd.DataFrame | None,
                  eras: pd.Series, members: dict[str, list[lgb.Booster]]) -> pd.Series:
    """Predictions in (0, 1) for one live model spec from configs/live.yaml."""
    parts = []
    for name in spec["members"]:
        raw = np.mean([m.predict(x_all.to_numpy()) for m in members[name]], axis=0)
        s = pd.Series(raw, index=x_all.index)
        parts.append(s.groupby(eras).transform(gaussianize))
    ens = pd.concat(parts, axis=1).mean(axis=1)
    if spec.get("neutralize"):
        ens = neutralize(ens, x_neutral, eras, spec["neutralize"]["proportion"])
    # Per-era rank into (0, 1), the format Numerai expects.
    return ens.groupby(eras).rank(pct=True)


def validate_submission(preds: pd.Series, live_ids: pd.Index) -> None:
    if preds.isna().any():
        raise ValueError("submission has NaN predictions")
    if not preds.index.equals(live_ids):
        raise ValueError("submission ids don't match live ids")
    if not ((preds > 0) & (preds <= 1)).all():
        raise ValueError("predictions must be in (0, 1]")
    if preds.nunique() < 0.9 * len(preds):
        raise ValueError("too many tied predictions")


def write_submission(preds: pd.Series, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.rename("prediction").rename_axis("id").to_frame().to_csv(path)
    return path
