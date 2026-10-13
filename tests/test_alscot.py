import joblib
import numpy as np
import pytest

from numerai_dev.alscot import StableRidge, select_strength, stability_matrix
from numerai_dev.models import RidgeModel


def sample():
    rng = np.random.default_rng(43)
    eras = np.repeat(np.arange(1, 301), 24)
    x = rng.integers(0, 5, (len(eras), 3), dtype=np.int8)
    sign = np.where((eras // 52) % 2 == 0, 1, -1)
    y = 0.2 * x[:, 0] + (0.3 + 0.8 * sign) * x[:, 1] + rng.normal(0, 0.1, len(x))
    return x, y, eras


def test_zero_matches_ridge_and_reload(tmp_path):
    x, y, eras = sample()
    ridge = RidgeModel(10).fit(x, y)
    model = StableRidge().fit(x, y, eras)
    np.testing.assert_allclose(model.predict(x), ridge.predict(x), atol=1e-12)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(x, 17), model.predict(x), atol=1e-12)


def test_penalty_psd_and_covariance_reconstruction():
    x, y, eras = sample()
    penalty, blocks, scale = stability_matrix(x, y, eras)
    assert np.linalg.eigvalsh(penalty).min() > -1e-12
    np.testing.assert_allclose(penalty * scale, np.cov(blocks, rowvar=False, bias=True))
    np.testing.assert_allclose(np.trace(penalty), x.shape[1])


def test_reversing_signal_is_shrunk_more_than_stable_signal():
    x, y, eras = sample()
    base = StableRidge(strength=0).fit(x, y, eras)
    robust = StableRidge(strength=100).fit(x, y, eras)
    assert abs(robust.beta[1] / base.beta[1]) < abs(robust.beta[0] / base.beta[0])
    assert robust.beta @ robust.penalty @ robust.beta < base.beta @ robust.penalty @ base.beta


def test_future_labels_cannot_change_selection():
    x, y, eras = sample()
    kwargs = {"outer_start": 240, "strengths": [0, 1, 10], "inner_eras": 30, "inner_folds": 2}
    first = select_strength(x, y, eras, **kwargs)
    changed = y.copy()
    changed[eras >= 227] = 1e8
    assert first == select_strength(x, changed, eras, **kwargs)
    assert first[0] in kwargs["strengths"]
    for fold in first[1]["folds"]:
        assert fold["train_end"] < fold["validation_start"] - 13
        assert fold["validation_end"] < 227


@pytest.mark.parametrize("alpha,strength", [(0, 1), (10, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_penalties(alpha, strength):
    with pytest.raises(ValueError):
        StableRidge(alpha, strength)


def test_invalid_rows_and_splits():
    x, y, eras = sample()
    with pytest.raises(ValueError):
        StableRidge().fit(x, y[:-1], eras)
    with pytest.raises(ValueError):
        stability_matrix(x, y, np.ones(len(y)))
    with pytest.raises(ValueError):
        select_strength(x, y, eras, 240, [1, 10])
    bad = x.astype(float)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        StableRidge().fit(bad, y, eras)
