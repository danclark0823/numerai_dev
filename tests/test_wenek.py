import runpy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from numerai_dev.diversifiers import FactorizationMachine
from numerai_dev.live import validate_submission
from numerai_dev.wenek import crossfit_ridge, fit_window, paired_block_ci


@pytest.fixture
def history():
    rng = np.random.default_rng(17)
    eras = np.repeat(np.arange(1, 51), 25)
    x = rng.integers(0, 5, (len(eras), 4)).astype(np.int8)
    y = 0.5 + 0.03 * x[:, 0] + 0.04 * (x[:, 1] - 2) * (x[:, 2] - 2)
    return x, y, eras


def fit_small(x, y, eras):
    return fit_window(x, y, eras, start=35, gap=2, alpha=1,
                      crossfit={"min_train_eras": 6, "block_eras": 8},
                      fm={"epochs": 2, "rank": 2, "seed": 19, "batch_size": 128})


def test_future_labels_cannot_change_earlier_models(history):
    x, y, eras = history
    a, mask, valid, oof, residual, audit = fit_small(x, y, eras)
    changed = y.copy()
    changed[eras >= 33] = -1000
    b, *_ = fit_small(x, changed, eras)
    assert eras[mask].max() + 2 < 35
    assert all(f["train_end"] + 2 < f["test_start"] for f in audit)
    np.testing.assert_array_equal(residual, y[mask][valid] - oof[valid])
    for name, pred in a.predict(x, eras).items():
        np.testing.assert_array_equal(pred, b.predict(x, eras)[name])


def test_inner_predictions_are_isolated_from_future_labels(history):
    x, y, eras = history
    a, audit = crossfit_ridge(x, y, eras, gap=2, min_train_eras=6, block_eras=8)
    y2 = y.copy()
    y2[eras >= 25] += 100
    b, _ = crossfit_ridge(x, y2, eras, gap=2, min_train_eras=6, block_eras=8)
    np.testing.assert_array_equal(a[eras < 25], b[eras < 25])
    assert np.isnan(a[eras <= 6]).all()
    assert audit


def test_interaction_only_recovers_pairwise_signal_without_linear_terms():
    rng = np.random.default_rng(27)
    x = rng.integers(0, 5, (6000, 4)).astype(float)
    y = 0.2 * (x[:, 0] - 2) * (x[:, 1] - 2)
    m = FactorizationMachine(interaction_only=True, rank=3, epochs=30,
                             learning_rate=0.01, penalty=0.001, batch_size=256).fit(
                                 x[:5000], y[:5000])
    np.testing.assert_array_equal(m.w, np.zeros(4))
    assert np.corrcoef(m.predict(x[5000:]), y[5000:])[0, 1] > 0.95


def test_bundle_reload_batch_feature_order_and_submission(history, tmp_path):
    x, y, eras = history
    model, *_ = fit_small(x, y, eras)
    expected = model.predict(x, eras)
    path = tmp_path / "wenek.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    for name, values in reloaded.predict(x, eras, batch_size=31).items():
        np.testing.assert_allclose(values, expected[name], atol=1e-12)
    np.testing.assert_array_equal(expected["blend_00"], expected["ridge"])
    # The rank transformation must be global within each era, not numerical batches.
    index = pd.Index([f"row{i}" for i in range(len(x))])
    features = pd.DataFrame(x, index=index, columns=list("abcd"))
    es = pd.Series(eras, index=index)
    zero = reloaded.submission(features[list("dcba")], es, list("abcd"), weight=0,
                               batch_size=31)
    reference = pd.Series(expected["ridge"], index=index).groupby(es).rank(pct=True)
    np.testing.assert_array_equal(zero.to_numpy(), reference.to_numpy())
    for e in es.unique():
        mask = es.eq(e)
        validate_submission(zero.loc[mask], index[mask])
    with pytest.raises(ValueError, match="IDs"):
        reloaded.submission(features, es.iloc[::-1], list("abcd"))


def test_bootstrap_rejects_gaps_and_short_windows():
    with pytest.raises(ValueError, match="contiguous"):
        paired_block_ci(np.ones(30), np.r_[np.arange(15), np.arange(100, 115)])
    assert paired_block_ci(np.ones(10), np.arange(10))["ci"] is None
    assert paired_block_ci(np.ones(30), np.arange(30))["ci"] == [1.0, 1.0]


def test_wenek_rejects_reserved_holdout_and_embargo(monkeypatch):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_wenek.py"))
    validate = script["validate_windows"]
    frame = pd.DataFrame({"era": [f"{i:04d}" for i in range(1, 201)],
                          "data_type": "validation", "target_ender_60": 0.5})
    monkeypatch.setitem(validate.__globals__, "load_index", lambda *args: frame)
    cfg = {"data": {"version": "vX", "target": "target_ender_60",
                    "train_rows_per_era": 1024, "train_stride": 4},
           "holdout_eras": 52, "test_starts": [130], "test_eras": 10}
    with pytest.raises(ValueError, match="holdout/embargo"):
        validate(cfg)  # last era 139 encroaches on embargo before holdout 149
    cfg["test_starts"] = [126]
    assert validate(cfg) == (13, 149)  # ending 135 is permissible


def test_sample_nested_and_feature_alignment(monkeypatch):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_wenek.py"))
    sample = script["sample_positions"]
    frame = pd.DataFrame({"era": np.repeat(["0001", "0002"], 100)})
    small = sample(frame, 2, 1, 20, 42)
    large = sample(frame, 2, 1, 40, 42)
    assert (small <= large).all()
    frames = {}
    for split, numbers in (("train", range(1, 11)), ("validation", range(11, 21))):
        numbers = np.repeat(list(numbers), 25)
        frames[split] = pd.DataFrame({"era": [f"{n:04d}" for n in numbers],
                                     "target": numbers / 30,
                                     "data_type": split},
                                    index=[f"{split}_{i}" for i in range(len(numbers))])
    load = script["load_data"]
    monkeypatch.setitem(load.__globals__, "load_index", lambda v, s, t: frames[s])

    def stream(version, split, features, mask):
        pos = np.flatnonzero(mask)[::-1]  # deliberately reverse feature delivery
        yield pos, frames[split].era.astype(int).to_numpy()[pos, None].astype(np.int8)

    monkeypatch.setitem(load.__globals__, "iter_features", stream)
    cfg = {"seed": 42, "data": {"version": "vX", "target": "target",
                               "train_stride": 2, "train_rows_per_era": 20},
           "test_starts": [16, 19], "test_eras": 2}
    rows, x = load(cfg, ["feature"], gap=2)
    np.testing.assert_array_equal(x[:, 0], rows.era.astype(int))
    assert len(rows.loc[rows.era.eq("0019")]) == 25  # full era, not the training sample
    assert rows.loc[rows.split.eq("train"), "sampled_train"].all()
