import joblib
import numpy as np
import pytest

from numerai_dev.models import RidgeModel
from numerai_dev.taren import (
    INF,
    DecayedStats,
    EndpointFitter,
    era_stats,
    fit_through,
    inner_windows,
    screening_decision,
    select_half_life,
    window_corr,
)


def sample(n_eras=120, rows=30, p=4, seed=7, gaps=False):
    rng = np.random.default_rng(seed)
    era_ids = np.arange(1, n_eras + 1)
    if gaps:
        era_ids = np.cumsum(rng.integers(1, 4, n_eras))  # irregular calendar gaps
    eras = np.repeat(era_ids, rows)
    x = rng.integers(0, 5, (len(eras), p), dtype=np.int8)
    y = x @ rng.normal(0, 0.05, p) + rng.normal(0, 0.1, len(x))
    return x, y, eras


def weighted_ridge(x, y, w, alpha):
    xf = x.astype(float)
    mean = w @ xf / w.sum()
    ybar = w @ y / w.sum()
    xc = xf - mean
    matrix = xc.T @ (w[:, None] * xc) + alpha * w.sum() * np.eye(x.shape[1])
    beta = np.linalg.solve(matrix, xc.T @ (w * (y - ybar)))
    return mean, beta, ybar


def test_era_stats_exact():
    x, y, _ = sample()
    gram, cross, xsum, ysum, n = era_stats(x, y)
    np.testing.assert_array_equal(gram, x.astype(float).T @ x.astype(float))
    np.testing.assert_allclose(cross, x.astype(float).T @ y)
    assert n == len(y) and np.isclose(ysum, y.sum())
    np.testing.assert_array_equal(xsum, x.sum(axis=0))


def test_infinite_half_life_matches_ridge():
    x, y, eras = sample(gaps=True)
    model = fit_through(x, y, eras, [eras.max()], [INF, 10], alpha=10)[eras.max()][INF]
    ridge = RidgeModel(10).fit(x, y)
    np.testing.assert_allclose(model.predict(x), ridge.predict(x), atol=1e-10)
    stream = RidgeModel(10).fit_stream([x], y)
    np.testing.assert_allclose(model.beta, stream.beta, atol=1e-10)


@pytest.mark.parametrize("h", [3.0, 20.0, 75.0])
def test_recursive_decay_equals_direct_weighting(h):
    x, y, eras = sample(gaps=True)
    end = int(np.unique(eras)[80])
    model = fit_through(x, y, eras, [end], [h, INF], alpha=10)[end][h]
    keep = eras <= end
    last = eras[keep].max()
    w = 0.5 ** ((last - eras[keep]) / h)
    mean, beta, ybar = weighted_ridge(x[keep], y[keep], w, 10)
    np.testing.assert_allclose(model.beta, beta, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(model.mean, mean)
    assert model.train_end == last and np.isclose(model.intercept, ybar)
    assert model.effective_rows < keep.sum()


def test_future_labels_cannot_change_fit():
    x, y, eras = sample()
    endpoints = [40, 77]
    first = fit_through(x, y, eras, endpoints, [8, INF], 10)
    changed = y.copy()
    changed[eras > 40] = 1e6
    second = fit_through(x, changed, eras, endpoints, [8, INF], 10)
    for h in (8, INF):
        np.testing.assert_array_equal(first[40][h].beta, second[40][h].beta)
        assert first[40][h].train_end == 40
        assert not np.allclose(first[77][h].beta, second[77][h].beta)


def test_inner_window_geometry():
    windows = inner_windows(850)
    assert windows == [(655, 706, 641), (720, 771, 706), (785, 836, 771)]
    for start, end, train_end in windows:
        assert train_end < start - 13 and end < 850 - 13 and end - start + 1 == 52


def drifting(stationary, seed=3):
    rng = np.random.default_rng(seed)
    eras = np.repeat(np.arange(1, 401), 60)
    x = rng.integers(0, 5, (len(eras), 10), dtype=np.int8)
    base = np.random.default_rng(0).normal(0, 0.05, 10)
    trend = np.ones(len(eras)) if stationary else 1 - 2 * eras / 400
    y = (x - 2) @ base + 0.15 * trend * (x[:, 0] - 2) + rng.normal(0, 1.0, len(x))
    return x, y, eras


def nested_choice(x, y, eras, grid, outer=400):
    windows = inner_windows(outer, inner_eras=30)
    models = fit_through(x, y, eras, [w[2] for w in windows], grid, 10)
    scores = {h: [] for h in grid}
    for start, end, train_end in windows:
        m = (eras >= start) & (eras <= end)
        for h in grid:
            scores[h].append(window_corr(models[train_end][h].predict(x[m]), y[m], eras[m]))
    return select_half_life(scores), scores


def test_selection_prefers_recency_only_under_drift():
    grid = [15.0, 120.0, INF]
    chosen, _ = nested_choice(*drifting(stationary=False), grid)
    assert chosen == 15.0
    chosen, scores = nested_choice(*drifting(stationary=True), grid)
    assert chosen != 15.0 and np.mean(scores[INF]) > np.mean(scores[15.0])


def test_select_ties_go_to_longer_half_life():
    assert select_half_life({52.0: [0.01, 0.02], INF: [0.02, 0.01]}) == INF
    assert select_half_life({52.0: [0.03], INF: [0.01]}) == 52.0
    with pytest.raises(ValueError):
        select_half_life({52.0: [0.01]})


def test_serialization_and_batch_parity(tmp_path):
    x, y, eras = sample()
    model = fit_through(x, y, eras, [eras.max()], [20, INF], 10)[eras.max()][20]
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(x, batch_size=7), model.predict(x), atol=1e-12)
    np.testing.assert_allclose(x @ model.beta + model.offset, model.predict(x), atol=1e-12)


def test_invalid_inputs():
    x, y, eras = sample()
    with pytest.raises(ValueError):
        era_stats(np.full((3, 2), 5, np.int8), np.zeros(3))
    with pytest.raises(ValueError):
        era_stats(x, y[:-1])
    with pytest.raises(ValueError):
        DecayedStats(3, [0, INF])
    with pytest.raises(ValueError):
        DecayedStats(3, [5, 5])
    stats = DecayedStats(4, [INF])
    stats.add(5, *era_stats(x[:10], y[:10]))
    with pytest.raises(ValueError):
        stats.add(5, *era_stats(x[:10], y[:10]))
    fitter = EndpointFitter(4, [INF], [3], 10)
    with pytest.raises(ValueError):
        fitter.advance(10)  # endpoint 3 precedes all data
    fitter = EndpointFitter(4, [INF], [30], 10)
    fitter.advance(10)
    with pytest.raises(ValueError):
        fitter.add_rows(11, x[:5], y[:5])
    fitter.add_rows(10, x[:5], y[:5])
    with pytest.raises(ValueError):
        fitter.advance(9)
    model = fit_through(x, y, eras, [eras.max()], [INF], 10)[eras.max()][INF]
    with pytest.raises(ValueError):
        model.predict(x[:, :2])


def metrics(contrib, mmc, corr, rondal_corr=0.02, rondal_mmc=0.0):
    out = {}
    for w, c, k in zip(["850", "1000", "1133"], contrib, corr, strict=True):
        out[w] = {"taren": {"rondal_contribution": {"mean": c}, "corr": {"mean": k}},
                  "rondal": {"corr": {"mean": rondal_corr}}}
    out["1133"]["taren"]["mmc"] = {"mean": mmc}
    out["1133"]["rondal"]["mmc"] = {"mean": rondal_mmc}
    out["all"] = {"taren": {"corr": {"mean": float(np.mean(corr))}},
                  "rondal": {"corr": {"mean": rondal_corr}}}
    return out


def test_screen():
    assert screening_decision(metrics([0.001, 0.002, -0.001], 0.003, [0.02] * 3))["screen_passed"]
    assert not screening_decision(metrics([0.0, 0.0, 0.0], 0.0, [0.02] * 3))["screen_passed"]
    identical = metrics([2.7e-19, 1.6e-19, -1.7e-18], 0.0, [0.02] * 3)
    assert not screening_decision(identical)["screen_passed"]
    assert not screening_decision(metrics([0.001] * 3, -0.001, [0.02] * 3))["screen_passed"]
    assert not screening_decision(metrics([0.001] * 3, 0.01, [0.02, 0.02, -0.001]))["screen_passed"]
    assert not screening_decision(metrics([0.001] * 3, 0.01, [0.015] * 3, 0.02))["screen_passed"]
