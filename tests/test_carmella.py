import numpy as np
import pandas as pd
import pytest

from numerai_dev.azar import residualize_many
from numerai_dev.carmella import block_geometry, forward_chain, screening_decision
from numerai_dev.models import RidgeModel
from numerai_dev.targets import residualize


def sample(n_eras=200, rows=30, p=4, seed=2):
    rng = np.random.default_rng(seed)
    eras = np.repeat(np.arange(10, 10 + n_eras), rows)
    x = rng.integers(0, 5, (len(eras), p), dtype=np.int8)
    y = x @ rng.normal(0, 0.05, p) + rng.normal(0, 0.1, len(x))
    return x, y, eras


def test_block_geometry():
    blocks = block_geometry(158, 1119)
    assert blocks[0] == (275, 300, 261)
    for (s, e, f), nxt in zip(blocks, [*blocks[1:], None], strict=True):
        assert f == s - 14 and f < s - 13 and e - s + 1 <= 26
        if nxt:
            assert nxt[0] == e + 1
    assert blocks[-1][1] == 1119
    with pytest.raises(ValueError):
        block_geometry(1, 100, warmup=0)


def test_forward_predictions_use_only_embargoed_history():
    x, y, eras = sample()
    blocks = block_geometry(10, eras.max(), warmup=40, block=20)
    preds, _ = forward_chain(x, y, eras, blocks, 10)
    assert np.isnan(preds[eras < blocks[0][0]]).all()
    assert np.isfinite(preds[eras >= blocks[0][0]]).all()
    start, end, fit_end = blocks[2]
    changed = y.copy()
    changed[eras > fit_end] = 1e6
    again, _ = forward_chain(x, changed, eras, blocks, 10)
    m = (eras >= start) & (eras <= end)
    np.testing.assert_array_equal(preds[m], again[m])
    keep = eras <= fit_end
    np.testing.assert_allclose(preds[m], RidgeModel(10).fit(x[keep], y[keep]).predict(x[m]),
                               atol=1e-10)


def test_extra_endpoint_matches_ridge():
    x, y, eras = sample()
    blocks = block_geometry(10, eras.max(), warmup=40, block=20)
    _, chain = forward_chain(x, y, eras, blocks, 10, extra_endpoints=[150])
    keep = eras <= 150
    np.testing.assert_allclose(chain.endpoint_model(150).predict(x),
                               RidgeModel(10).fit(x[keep], y[keep]).predict(x), atol=1e-10)


def test_three_reference_residuals_match_targets_residualize():
    rng = np.random.default_rng(3)
    eras = np.repeat(np.arange(1, 6), 40)
    df = pd.DataFrame({"era": eras.astype(str), "t": rng.choice([0, .25, .5, .75, 1], len(eras)),
                       "a": rng.normal(size=len(eras)), "b": rng.normal(size=len(eras)),
                       "c": rng.normal(size=len(eras))})
    many = residualize_many(df[["t"]].to_numpy(), df[["a", "b", "c"]].to_numpy(), eras)
    np.testing.assert_allclose(many[:, 0], residualize(df, "t", ["a", "b", "c"]), atol=1e-12)


def metrics(portfolio, tyndal, mmc, corr):
    out = {w: {"carmella": {"portfolio_contribution": {"mean": p},
                            "tyndal_contribution": {"mean": t}, "corr": {"mean": k}}}
           for w, p, t, k in zip(["850", "1000", "1133"], portfolio, tyndal, corr, strict=True)}
    out["1133"]["carmella"]["mmc"] = {"mean": mmc}
    out["all"] = {}
    return out


def test_screen():
    ok = [0.001, 0.002, -0.001]
    assert screening_decision(metrics(ok, ok, 0.0, [0.01] * 3))["screen_passed"]
    assert not screening_decision(metrics(ok, [-0.001, -0.001, 0.002], 0.01, [0.01] * 3))[
        "screen_passed"]
    assert not screening_decision(metrics([-0.001, 1e-19, 0.002], ok, 0.01, [0.01] * 3))[
        "screen_passed"]
    assert not screening_decision(metrics(ok, ok, -0.001, [0.01] * 3))["screen_passed"]
    assert not screening_decision(metrics(ok, ok, 0.01, [0.01, -0.01, 0.01]))["screen_passed"]
