import runpy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from numerai_dev import diversifiers as d


def test_train_sampling_keeps_train_data_type(monkeypatch):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1]
                               / "scripts" / "run_diversifiers.py"))
    sample = script["sample_split"]
    frame = pd.DataFrame({"era": ["0001"] * 30, "data_type": ["train"] * 30,
                          "target": np.ones(30)}, index=[f"id{i}" for i in range(30)])
    monkeypatch.setitem(sample.__globals__, "load_index", lambda *args: frame)
    monkeypatch.setitem(sample.__globals__, "iter_features",
                        lambda v, s, f, m: iter([(np.flatnonzero(m),
                                                 np.ones((int(m.sum()), 2), dtype=np.int8))]))
    rows, x = sample("vX", "train", "target", ["a", "b"], 10, 20, 42)
    assert len(rows) == 20 and x.shape == (20, 2)


def test_runner_rejects_holdout_before_loading_features(monkeypatch):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1]
                               / "scripts" / "run_diversifiers.py"))
    run = script["run"]
    frame = pd.DataFrame({"era": [f"{i:04d}" for i in range(1, 201)],
                          "data_type": "validation", "target_ender_60": 0.5})
    monkeypatch.setitem(run.__globals__, "load_index", lambda *args: frame)
    cfg = {"data": {"version": "vX", "target": "target_ender_60"},
           "holdout_eras": 52, "test_eras": 10, "test_starts": [140]}
    with pytest.raises(ValueError, match="reserved holdout"):
        run(cfg)


def test_crossfit_uses_only_embargoed_past(monkeypatch):
    eras = np.repeat(np.arange(1, 401, 4), 4)
    x = eras[:, None].astype(float)
    y = np.sin(eras)
    calls = []

    def teacher(tx, ty, vx, trees):
        assert tx[:, 0].max() + 13 < vx[:, 0].min()
        calls.append((tx.copy(), ty.copy()))
        return np.column_stack([np.arange(len(vx))] * 3)

    monkeypatch.setattr(d, "teacher_predictions", teacher)
    predictions, audit = d.crossfit_consensus(x, y, eras, min_train_eras=10, block_eras=8)
    assert calls and np.isnan(predictions[eras < 50]).all()
    assert all(a["train_end"] + 13 < a["test_start"] for a in audit)
    assert np.isfinite(predictions).any()


def test_residual_labels_orthogonal_per_era():
    rng = np.random.default_rng(4)
    eras = np.repeat(np.arange(6), 100)
    reference = rng.normal(size=len(eras))
    y = reference * 0.4 + rng.normal(size=len(eras))
    residual = d.residual_labels(y, reference, eras)
    normalized = d.era_gaussian(reference, eras)
    for era in np.unique(eras):
        m = eras == era
        assert abs(residual[m].mean()) < 1e-12
        assert abs(residual[m] @ normalized[m]) < 1e-10


def test_stable_portfolio_rejects_sign_flipping_signal():
    rng = np.random.default_rng(5)
    eras = np.repeat(np.arange(16), 300)
    x = rng.normal(size=(len(eras), 3))
    y = x[:, 0] + 2 * x[:, 1] * np.where(eras < 8, 1, -1)
    model = d.StablePortfolio(top_k=2, min_consistency=1).fit(
        x, y, eras, rng.normal(size=len(eras)))
    assert 0 in model.columns
    assert 1 not in model.columns
    assert np.corrcoef(model.predict(x, eras), x[:, 0])[0, 1] > 0.8


def test_no_qualifying_features_fails_explicitly():
    x = np.ones((40, 3))
    with pytest.raises(ValueError, match="No stable features"):
        d.StablePortfolio().fit(x, np.arange(40), np.repeat([1, 2], 20), np.arange(40))


def test_factorization_learns_interaction_and_roundtrips(tmp_path):
    rng = np.random.default_rng(6)
    x = rng.integers(0, 5, size=(6000, 4)).astype(float)
    y = (x[:, 0] - 2) * (x[:, 1] - 2) * 0.2
    model = d.FactorizationMachine(rank=3, epochs=30, learning_rate=0.01,
                                   batch_size=256, penalty=0.001).fit(x[:5000], y[:5000])
    pred = model.predict(x[5000:])
    assert np.corrcoef(pred, y[5000:])[0, 1] > 0.95
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(x[5000:]), pred)
