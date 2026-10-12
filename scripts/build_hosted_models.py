"""Package the live models in configs/live.yaml as Numerai-hosted pickles.

Run with the `numerai-predict` conda env, which pins the same package versions as
Numerai's hosted runtime (Python 3.12, lightgbm 4.5.0, pandas 2.3.1):

  ~/miniforge3/envs/numerai-predict/bin/python scripts/build_hosted_models.py
  ~/miniforge3/envs/numerai-predict/bin/python scripts/build_hosted_models.py --check 1374

The pickled `predict` is self-contained (no imports from this repo): models travel
as LightGBM text and are rebuilt on first call, which is robust across LightGBM
versions. Its logic mirrors numerai_dev.live.predict_model; --check verifies the
pickle reproduces the predictions the local pipeline produced for a round.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "hosted"


def make_predict(member_models: list[list], features: list[str],
                 neutral_features: list[str] | None, proportion: float):
    """Build a predict(live_features, live_benchmark_models) closure. Each member is
    a list of LightGBM model strings or of ridge (mean, beta, intercept) tuples."""

    boosters: list = []

    def predict(live_features: pd.DataFrame,
                live_benchmark_models: pd.DataFrame | None = None) -> pd.DataFrame:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
        from scipy.stats import norm

        if not boosters:
            for member in member_models:
                boosters.append([lgb.Booster(model_str=m) if isinstance(m, str) else m
                                 for m in member])

        x = live_features[features].to_numpy(dtype=np.float32)
        parts = []
        for member in boosters:
            raw = np.mean([
                m.predict(x) if hasattr(m, "predict")
                else (x.astype(np.float64) - m[0]) @ m[1] + m[2]  # ridge
                for m in member
            ], axis=0)
            r = pd.Series(raw).rank(method="average").to_numpy()
            parts.append(norm.ppf((r - 0.5) / len(r)))  # gaussianize
        ens = np.mean(parts, axis=0)

        if neutral_features:
            xn = live_features[neutral_features].to_numpy(dtype=np.float32)
            xn = xn - xn.mean(axis=0)
            y = (ens - ens.mean()).astype(np.float32)
            beta, *_ = np.linalg.lstsq(xn, y, rcond=None)
            ens = y - proportion * (xn @ beta)

        pred = pd.Series(ens, index=live_features.index).rank(pct=True)
        return pred.to_frame("prediction")

    return predict


def build(cfg: dict) -> dict[str, Path]:
    meta = json.loads((ROOT / "data" / cfg["version"] / "features.json").read_text())
    fsets = meta["feature_sets"]
    features = fsets[cfg["feature_set"]]
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for model, spec in cfg["models"].items():
        member_models = []
        for m in spec["members"]:
            d = ROOT / "artifacts" / m
            models = [p.read_text() for p in sorted(d.glob("model*.txt"))]
            for p in sorted(d.glob("model*.npz")):
                z = np.load(p)
                models.append((z["mean"], z["beta"], float(z["intercept"])))
            member_models.append(models)
        if not all(member_models):
            raise FileNotFoundError(f"missing saved models for {model}")
        neu = spec.get("neutralize")
        fn = make_predict(member_models, features,
                          fsets[neu["feature_set"]] if neu else None,
                          neu["proportion"] if neu else 0.0)
        path = OUT / f"{model}.pkl"
        path.write_bytes(cloudpickle.dumps(fn))
        paths[model] = path
        n_trees = sum(len(m) for m in member_models)
        print(f"{model}: {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB, "
              f"{n_trees} boosters)")
    return paths


def check(cfg: dict, rnd: int) -> None:
    import time

    live = pd.read_parquet(ROOT / "data" / cfg["version"] / f"live_{rnd}.parquet")
    for model in cfg["models"]:
        fn = cloudpickle.loads((OUT / f"{model}.pkl").read_bytes())
        t = time.time()
        out = fn(live, None)
        secs = time.time() - t
        ref = pd.read_csv(ROOT / "predictions" / str(rnd) / f"{model}.csv", index_col=0)
        corr = out["prediction"].corr(ref["prediction"].reindex(out.index), method="spearman")
        maxdiff = (out["prediction"] - ref["prediction"].reindex(out.index)).abs().max()
        print(f"{model}: {secs:.1f}s, rank corr vs submitted = {corr:.6f}, "
              f"max abs diff = {maxdiff:.2e}, rows = {len(out)}, "
              f"NaN = {int(out['prediction'].isna().sum())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live.yaml")
    parser.add_argument("--check", type=int, metavar="ROUND",
                        help="compare pickles against predictions/<ROUND>/*.csv")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.check:
        check(cfg, args.check)
    else:
        build(cfg)


if __name__ == "__main__":
    main()
