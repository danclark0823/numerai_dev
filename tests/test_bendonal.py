import copy
import itertools

import joblib
import numpy as np
import pandas as pd
import pytest

from numerai_dev.bendonal import (
    AdditiveModel,
    curvature_basis,
    encode,
    fit_before,
    screening_decision,
    sufficient_stats,
    validate_bins,
)
from numerai_dev.live import validate_submission
from numerai_dev.models import RidgeModel


def test_contrast_basis_and_lookup_span():
    b = curvature_basis()
    np.testing.assert_allclose(b.T @ b, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(b.T @ np.ones(5), 0, atol=1e-14)
    np.testing.assert_allclose(b.T @ np.arange(5), 0, atol=1e-14)
    design = np.column_stack([np.ones(5), np.arange(5), b])
    values = np.array([0.4, -0.7, 0.1, 0.9, -0.2])
    np.testing.assert_allclose(design @ np.linalg.solve(design, values), values, atol=1e-14)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, 5, 1.5, 256])
def test_invalid_bins_rejected_before_cast(bad):
    with pytest.raises(ValueError, match="finite integer bins"):
        validate_bins(np.array([[bad, 2.0]]))


def test_streamed_reference_linear_parity_and_absent_bins():
    rng = np.random.default_rng(15)
    x = rng.integers(0, 4, (1000, 5))  # fifth bin absent from training
    y = rng.normal(size=1000) + 0.2 * x[:, 0]**2
    stats = sufficient_stats([x[:433], x[433:]], y, chunk_rows=97)
    z = encode(x, stats.basis)
    centered = z - z.mean(axis=0)
    np.testing.assert_allclose(stats.gram, centered.T @ centered, atol=1e-10)
    m = AdditiveModel(0.1, 0.7).fit_stats(stats)
    penalty = len(y) * np.r_[np.repeat(0.1, 5), np.repeat(0.7, 15)]
    beta = np.linalg.solve(centered.T @ centered + np.diag(penalty),
                           centered.T @ (y - y.mean()))
    np.testing.assert_allclose(m.predict(x), centered @ beta + y.mean(), atol=1e-10)
    linear = AdditiveModel(10).fit_stream([x[:433], x[433:]], y, nonlinear=False)
    ridge = RidgeModel(10).fit_stream([x], y)
    np.testing.assert_allclose(linear.predict(x), ridge.predict(x), atol=1e-12)
    assert np.isfinite(m.predict(np.full((2, 5), 4))).all()
    assert (m.bin_counts[:, 4] == 0).all()


def test_synthetic_u_shape_and_linear_signal_and_shrinkage():
    rng = np.random.default_rng(18)
    x = rng.integers(0, 5, (6000, 3))
    y = 0.4 * (x[:, 0] - 2)**2 + 0.2 * x[:, 1]
    stats = sufficient_stats([x[:5000]], y[:5000])
    small = AdditiveModel(1e-6, 1e-6).fit_stats(stats)
    large = AdditiveModel(100, 100).fit_stats(stats)
    np.testing.assert_allclose(small.predict(x[5000:]), y[5000:], atol=1e-4)
    assert np.linalg.norm(large.beta) < np.linalg.norm(small.beta)
    # The sufficient statistics remain unregularized and reusable across candidates.
    again = AdditiveModel(1e-6, 1e-6).fit_stats(stats)
    np.testing.assert_array_equal(small.beta, again.beta)


def test_cannot_fit_pure_centered_interaction():
    x = np.tile(np.array(list(itertools.product(range(5), repeat=2))), (30, 1))
    y = (x[:, 0] - 2) * (x[:, 1] - 2)
    m = AdditiveModel(1e-6, 1e-6).fit_stream([x], y)
    np.testing.assert_allclose(m.predict(x), 0, atol=1e-12)


def test_future_labels_do_not_change_model():
    rng = np.random.default_rng(21)
    x = rng.integers(0, 5, (1000, 3))
    y, eras = rng.normal(size=1000), np.repeat(np.arange(1, 101), 10)
    a = fit_before(x, y, eras, test_start=80)
    y[eras >= 67] += 999
    b = fit_before(x, y, eras, test_start=80)
    np.testing.assert_array_equal(a.beta, b.beta)


def test_serialization_curves_batch_order_and_submission(tmp_path):
    rng = np.random.default_rng(22)
    x = rng.integers(0, 5, (1000, 20))
    y = rng.normal(size=1000)
    m = AdditiveModel().fit_stream([x], y, chunk_rows=123)
    raw = m.predict(x)
    path = tmp_path / "model.joblib"
    joblib.dump(m, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(x, batch_size=73), raw, atol=1e-12)
    np.testing.assert_allclose(m.intercept + m.curves()[np.arange(20), x].sum(axis=1), raw,
                               atol=1e-12)
    np.testing.assert_allclose(m.predict(x, component="nonlinear"),
                               (encode(x, m.basis) - m.mean)[:, 20:] @ m.beta[20:])
    columns = [f"f{i}" for i in range(20)]
    frame = pd.DataFrame(x, columns=columns)
    eras = pd.Series(np.repeat([1, 2], 500), index=frame.index)
    submission = loaded.submission(frame[columns[::-1]], eras, columns, batch_size=73)
    expected = pd.Series(raw).groupby(eras).rank(pct=True)
    np.testing.assert_array_equal(submission, expected)
    for era in [1, 2]:
        mask = eras.eq(era)
        validate_submission(submission[mask], frame.index[mask])
    with pytest.raises(ValueError, match="aligned"):
        loaded.submission(frame, eras.iloc[::-1], columns)
    with pytest.raises(ValueError, match="count mismatch"):
        sufficient_stats([x], y[:-1])


def test_screen_requires_all_criteria_and_prefers_stronger_penalty_on_tie():
    def scores(corr, mmc, contribution):
        return {"corr": {"mean": corr}, "mmc": {"mean": mmc},
                "ridge_contribution": {"mean": contribution}}
    window = {"ridge": scores(0.01, -0.01, 0), "a": scores(0.02, 0.001, 0.002),
              "b": scores(0.02, 0.001, 0.002)}
    metrics = {key: copy.deepcopy(window) for key in ["all", "850", "1000", "1133"]}
    assert screening_decision(metrics, ["a", "b"], {"a": 10, "b": 100})["selected"] == "b"
    for w in metrics.values():
        for name in ("a", "b"):
            w[name]["mmc"]["mean"] = -0.001  # beats ridge, but still negative
    assert screening_decision(metrics, ["a", "b"], {"a": 10, "b": 100})["shelve"]
