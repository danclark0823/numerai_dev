import joblib
import numpy as np
import pandas as pd
import pytest

from numerai_dev.diversifiers import era_gaussian
from numerai_dev.hesia import RunningStats, SpecialistBundle, feature_subsets
from numerai_dev.models import RidgeModel


def sample():
    rng = np.random.default_rng(9)
    x = rng.integers(0, 5, (600, 17), dtype=np.int8)
    y = x[:, 0] * 0.1 + rng.normal(0, 0.1, len(x))
    return x, y, np.repeat(np.arange(6), 100)


def test_fixed_partitions_exact_coverage():
    parts = feature_subsets(3555)
    assert len(parts) == 16 and {len(p) for p in parts} == {444, 445}
    assert all(len(np.unique(p)) == len(p) for p in parts)
    np.testing.assert_array_equal(np.bincount(np.concatenate(parts)), np.full(3555, 2))
    for a, b in zip(parts, feature_subsets(3555), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(parts[0], feature_subsets(3555, seed=43)[0])


def test_incremental_and_marginal_fit_parity():
    x, y, _ = sample()
    stats = RunningStats(17).update(x[:200], y[:200], 31).update(x[200:], y[200:], 71)
    whole = RunningStats(17).update(x, y)
    np.testing.assert_allclose(stats.gram, whole.gram)
    np.testing.assert_allclose(stats.cross, whole.cross)
    for cols in [np.arange(17), np.array([1, 7, 3])]:
        mean, intercept, beta = stats.fit(cols)
        ridge = RidgeModel(10).fit(x[:, cols], y)
        np.testing.assert_allclose(mean, ridge.mean)
        np.testing.assert_allclose(beta, ridge.beta, atol=1e-12)
        assert intercept == pytest.approx(ridge.intercept)


def test_blend_formula_reload_batch_and_feature_order(tmp_path):
    x, y, eras = sample()
    features = [f"f{i}" for i in range(17)]
    subsets = feature_subsets(17, groups=4)
    bundle = SpecialistBundle.fit_stats(RunningStats(17).update(x, y), features, subsets, [0, 1])
    predictions = bundle.predict(x, eras)
    independent = [era_gaussian(RidgeModel(10).fit(x[:, c], y).predict(x[:, c]), eras)
                   for c in subsets]
    np.testing.assert_allclose(predictions[:, 2], np.mean(independent, axis=0), atol=1e-12)
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(x, eras, 29), predictions, atol=1e-12)
    xf, es = pd.DataFrame(x, columns=features), pd.Series(eras)
    expected = pd.Series(predictions[:, 2]).groupby(es).rank(pct=True)
    np.testing.assert_allclose(loaded.submission(xf[features[::-1]], es, 29), expected)


def test_future_labels_do_not_enter_stats():
    x, y, eras = sample()
    mask = eras < 3
    a = RunningStats(17).update(x[mask], y[mask]).fit(np.arange(17))
    y[~mask] = 1e10
    b = RunningStats(17).update(x[mask], y[mask]).fit(np.arange(17))
    np.testing.assert_array_equal(a[2], b[2])


def test_invalid_inputs():
    x, y, _ = sample()
    with pytest.raises(ValueError):
        feature_subsets(3, groups=8)
    with pytest.raises(ValueError):
        RunningStats(17).update(x, y[:-1])
    with pytest.raises(ValueError):
        RunningStats(17).fit([0, 1])
    stats = RunningStats(17).update(x, y)
    with pytest.raises(ValueError):
        stats.fit([0, 0])
    with pytest.raises(ValueError):
        stats.fit([0], alpha=-1)
    bad = x.astype(float)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        RunningStats(17).update(bad, y)
