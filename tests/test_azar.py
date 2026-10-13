import joblib
import numpy as np
import pandas as pd
import pytest

from numerai_dev.azar import (
    EndpointFitter,
    MultiTargetStats,
    era_stats,
    fit_through,
    residualize_many,
    screening_decision,
    select_target,
)
from numerai_dev.models import RidgeModel
from numerai_dev.targets import residualize

NAMES = ["ender", "alt_a", "alt_b"]


def sample(n_eras=60, rows=40, p=5, seed=5):
    rng = np.random.default_rng(seed)
    eras = np.repeat(np.cumsum(rng.integers(1, 3, n_eras)), rows)
    x = rng.integers(0, 5, (len(eras), p), dtype=np.int8)
    y = np.column_stack([x @ rng.normal(0, 0.05, p) + rng.normal(0, 0.1, len(x))
                         for _ in NAMES])
    return x, y, eras


def test_era_stats_exact():
    x, y, _ = sample()
    gram, cross, xsum, ysum, n = era_stats(x, y)
    np.testing.assert_array_equal(gram, x.astype(float).T @ x.astype(float))
    np.testing.assert_allclose(cross, x.astype(float).T @ y)
    np.testing.assert_array_equal(xsum, x.sum(axis=0))
    np.testing.assert_allclose(ysum, y.sum(axis=0))
    assert n == len(x)


def test_each_target_matches_independent_ridge():
    x, y, eras = sample()
    models = fit_through(x, y, eras, [eras.max()], NAMES, 10)[eras.max()]
    for j, name in enumerate(NAMES):
        ridge = RidgeModel(10).fit(x, y[:, j])
        np.testing.assert_allclose(models[name].predict(x), ridge.predict(x), atol=1e-10)
        assert models[name].train_end == eras.max()


def test_vectorized_residuals_match_targets_residualize():
    rng = np.random.default_rng(1)
    eras = np.repeat(np.arange(1, 9), 50)
    df = pd.DataFrame({"era": eras.astype(str), "r1": rng.normal(size=len(eras)),
                       "r2": rng.normal(size=len(eras))})
    for name in NAMES:
        df[name] = rng.choice([0, 0.25, 0.5, 0.75, 1.0], len(eras)) + 0.1 * df.r1
    many = residualize_many(df[NAMES].to_numpy(), df[["r1", "r2"]].to_numpy(), eras)
    for j, name in enumerate(NAMES):
        np.testing.assert_allclose(many[:, j], residualize(df, name, ["r1", "r2"]), atol=1e-12)


def test_future_labels_cannot_change_fit():
    x, y, eras = sample()
    cut = int(np.unique(eras)[30])
    first = fit_through(x, y, eras, [cut], NAMES, 10)[cut]
    changed = y.copy()
    changed[eras > cut] = 1e6
    second = fit_through(x, changed, eras, [cut], NAMES, 10)[cut]
    for name in NAMES:
        np.testing.assert_array_equal(first[name].beta, second[name].beta)
        assert first[name].train_end <= cut


def test_selection_rule_and_fallback():
    assert select_target({"ender": [0.0], "alt_a": [0.002, 0.0], "alt_b": [0.003, -0.002]},
                         "ender") == "alt_a"
    assert select_target({"ender": [0.0], "alt_a": [-0.001], "alt_b": [0.0]}, "ender") == "ender"
    with pytest.raises(ValueError):
        select_target({"alt_a": [0.01]}, "ender")


def test_serialization_and_batch_parity(tmp_path):
    x, y, eras = sample()
    model = fit_through(x, y, eras, [eras.max()], NAMES, 10)[eras.max()]["alt_a"]
    joblib.dump(model, tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(loaded.predict(x, batch_size=9), model.predict(x), atol=1e-12)
    np.testing.assert_allclose(x @ model.beta + model.offset, model.predict(x), atol=1e-12)


def test_invalid_inputs():
    x, y, _ = sample()
    with pytest.raises(ValueError):
        era_stats(x, y[:, 0])
    with pytest.raises(ValueError):
        era_stats(np.full((3, 2), 7, np.int8), np.zeros((3, 1)))
    with pytest.raises(ValueError):
        EndpointFitter(5, ["a", "a"], [3], 10)
    with pytest.raises(ValueError):
        MultiTargetStats(5, 3).fit(10, NAMES)
    with pytest.raises(ValueError):
        residualize_many(np.array([[np.nan]]), np.zeros((1, 1)), np.array([1]))
    with pytest.raises(ValueError):
        residualize_many(np.zeros((3, 1)), np.zeros((2, 1)), np.array([1, 1, 1]))
    fitter = EndpointFitter(5, NAMES, [100], 10)
    fitter.advance(4)
    with pytest.raises(ValueError):
        fitter.add_rows(5, x[:3], y[:3])
    with pytest.raises(ValueError):
        fitter.advance(3)


def metrics(contrib, mmc, corr):
    out = {w: {"azar": {"rondal_contribution": {"mean": c}, "corr": {"mean": k}}}
           for w, c, k in zip(["850", "1000", "1133"], contrib, corr, strict=True)}
    out["1133"]["azar"]["mmc"] = {"mean": mmc}
    out["all"] = {}
    return out


def test_screen():
    assert screening_decision(metrics([0.001, -0.001, 0.002], 0.0, [0.01] * 3))["screen_passed"]
    assert not screening_decision(metrics([1e-19, 2e-19, 0.002], 0.01, [0.01] * 3))["screen_passed"]
    assert not screening_decision(metrics([0.001] * 3, -0.001, [0.01] * 3))["screen_passed"]
    assert not screening_decision(metrics([0.001] * 3, 0.01, [0.01, -0.01, 0.01]))["screen_passed"]
