import numpy as np
import pandas as pd

from numerai_dev.models import RidgeModel
from numerai_dev.targets import gaussianize, residualize


def test_ridge_recovers_linear_signal(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(50_000, 20)).astype(np.int8)
    beta = rng.normal(size=20)
    y = (x - 2) @ beta + rng.normal(scale=0.1, size=len(x))
    m = RidgeModel(alpha=1e-6, chunk_rows=7_000).fit(x, y)
    np.testing.assert_allclose(m.beta, beta, atol=0.01)
    m.save(tmp_path / "m")
    m2 = RidgeModel.load(tmp_path / "m.npz")
    np.testing.assert_allclose(m2.predict(x[:100]), m.predict(x[:100]))


def test_ridge_alpha_shrinks():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, size=(10_000, 5)).astype(np.int8)
    y = x[:, 0].astype(float)
    small = RidgeModel(alpha=1e-6).fit(x, y).beta[0]
    big = RidgeModel(alpha=10).fit(x, y).beta[0]
    assert abs(big) < abs(small)


def test_residualize_multiple_refs_orthogonal():
    rng = np.random.default_rng(2)
    n = 2_000
    df = pd.DataFrame({
        "era": np.repeat(["0001", "0002"], n // 2),
        "t": rng.normal(size=n), "a": rng.normal(size=n), "b": rng.normal(size=n),
    })
    df["t"] += df["a"] + 0.5 * df["b"]
    r = residualize(df, "t", ["a", "b"])
    for _, d in df.assign(r=r).groupby("era"):
        for ref in ("a", "b"):
            assert abs(np.corrcoef(d["r"], gaussianize(d[ref]))[0, 1]) < 1e-8
