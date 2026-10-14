import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numerai_tools.scoring import numerai_corr

from numerai_dev.evaluation import per_era_corr
from numerai_dev.live import validate_submission
from numerai_dev.mavone import (
    assert_complete,
    blend,
    block_schedule,
    exclusions,
    gaussian_rank,
    heatmap_svg,
    hypothesis1_decision,
    incremental_sets,
    longest_run,
    mmc_gate,
    stride_mask,
    training_mask,
)
from numerai_dev.targets import gaussianize

PARAMS = {"objective": "regression", "learning_rate": 0.1, "max_depth": 3, "num_leaves": 7,
          "colsample_bytree": 0.5, "random_state": 42, "num_threads": 1, "verbose": -1,
          "min_data_in_leaf": 5}


def synthetic(n_eras=40, rows=60, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    eras = np.repeat(np.arange(1, n_eras + 1), rows)
    x = rng.integers(0, 5, (len(eras), n_features)).astype(np.float32)
    y = np.column_stack([x[:, 0] - x[:, 1] + rng.normal(size=len(eras)),
                         x[:, 2] + rng.normal(size=len(eras))]) / 4 + 0.5
    return eras, x, y


def test_block_schedule_maturity():
    blocks = block_schedule(1133, 1211)
    assert [b["start"] for b in blocks] == [1133, 1159, 1185, 1211]
    for b in blocks:
        assert b["end"] - b["start"] == 25
        assert b["fit_end"] + 13 < b["start"]
        assert b["label_available_by_era"] == b["start"] - 1
    with pytest.raises(ValueError):
        block_schedule(10, 5)


def test_global_stride_anchor():
    assert stride_mask([1, 2, 5, 573, 577, 1117], 4, 1).tolist() == [True, False, True, True,
                                                                     True, True]
    with pytest.raises(ValueError):
        stride_mask([1], 0, 1)


def test_training_mask_future_labels_and_intersection():
    eras, _, y = synthetic()
    eligible = np.ones(len(eras), bool)
    base = training_mask(eras, y, eligible, 20, 4, 1)
    future = y.copy()
    future[eras > 20] = np.nan
    future[eras > 25] = 99
    assert np.array_equal(base, training_mask(eras, future, eligible, 20, 4, 1))
    assert set(eras[base]) == {1, 5, 9, 13, 17}
    gaps = y.copy()
    gaps[0, 1] = np.nan
    m = training_mask(eras, gaps, eligible, 20, 4, 1)
    assert not m[0] and m[1]
    ineligible = eligible.copy()
    ineligible[eras == 5] = False
    assert not training_mask(eras, y, ineligible, 20, 4, 1)[eras == 5].any()
    with pytest.raises(ValueError):
        training_mask(eras, y[:, 0], eligible, 20, 4, 1)


def test_exclusions_report():
    eras, _, y = synthetic(n_eras=8, rows=5)
    labels = pd.DataFrame(y, columns=["a", "b"])
    labels.loc[0, "a"] = np.nan
    labels.loc[1, ["a", "b"]] = np.nan
    labels.loc[7, "b"] = np.nan  # era 2, off the stride
    report = exclusions(eras, labels, np.ones(len(eras), bool), 8, 4, 1)
    assert report.index.tolist() == [1, 5]
    assert report.loc[1].tolist() == [5, 2, 1, 2]


def test_gaussian_rank_ties_order_and_reference():
    eras = np.array([1, 1, 1, 1, 2, 2, 2])
    v = np.array([0.1, 0.5, 0.5, 0.9, 3.0, 1.0, 2.0])
    g = gaussian_rank(v, eras)
    assert g[1] == g[2]
    expected = pd.Series(v).groupby(eras).transform(gaussianize).to_numpy()
    np.testing.assert_allclose(g, expected)
    perm = np.random.default_rng(1).permutation(len(v))
    np.testing.assert_allclose(gaussian_rank(v[perm], eras[perm]), g[perm])
    with pytest.raises(ValueError):
        gaussian_rank([np.nan, 1.0], [1, 1])


def test_blend_formulas_and_submission():
    rng = np.random.default_rng(2)
    eras = np.repeat([1, 2, 3], 50)
    preds = pd.DataFrame(rng.normal(size=(150, 4)), columns=["ender", "b", "c", "d"])
    eq = blend(preds, {k: 0.25 for k in preds}, eras)
    manual = sum(gaussian_rank(preds[k].to_numpy(), eras) for k in preds) / 4
    np.testing.assert_allclose(eq, pd.Series(manual).groupby(eras).rank(pct=True).to_numpy())
    alone = blend(preds, {"ender": 1.0}, eras)
    np.testing.assert_allclose(alone, preds.ender.groupby(eras).rank(pct=True).to_numpy())
    for era in np.unique(eras):
        m = eras == era
        validate_submission(pd.Series(eq[m]), pd.RangeIndex(m.sum()))
    perm = rng.permutation(150)
    np.testing.assert_allclose(blend(preds.iloc[perm].reset_index(drop=True),
                                     {k: 0.25 for k in preds}, eras[perm]), eq[perm])
    for bad in ({"ender": 0.5}, {"ender": 1.5, "b": -0.5}, {"zzz": 1.0}, {}):
        with pytest.raises(ValueError):
            blend(preds, bad, eras)
    assert incremental_sets(["ender", "b", "c"]) == [["ender"], ["ender", "b"],
                                                     ["ender", "b", "c"]]


def test_scoring_matches_official_with_ties():
    rng = np.random.default_rng(3)
    eras = np.repeat(["0001", "0002"], 40)
    preds = pd.DataFrame(rng.integers(0, 5, (80, 2)).astype(float), columns=["ender", "b"])
    frame = pd.DataFrame({"era": eras, "target": rng.choice([0, .25, .5, .75, 1], 80)})
    frame["eq"] = blend(preds, {"ender": .5, "b": .5}, eras)
    ours = per_era_corr(frame, ["eq"], "target")["eq"]
    for era, rows in frame.groupby("era"):
        official = numerai_corr(rows[["eq"]], rows["target"])["eq"]
        assert ours[era] == pytest.approx(official)


def test_assert_complete_explicit_rows():
    ids = pd.Index(["a", "b", "c"])
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=["c", "a", "b"])
    assert_complete(frame, ["x"], ids)
    with pytest.raises(ValueError):
        assert_complete(frame.iloc[:2], ["x"], ids)
    with pytest.raises(ValueError):
        assert_complete(pd.concat([frame, frame.iloc[:1]]), ["x"], ids)
    with pytest.raises(ValueError):
        assert_complete(frame.assign(x=[1.0, np.nan, 3.0]), ["x"], ids)
    with pytest.raises(ValueError):
        assert_complete(frame.rename(index={"c": "z"}), ["x"], ids)


def test_hypothesis1_decision():
    good = {"corr": 0.02, "mmc": 0.0, "ender_contribution": 0.001,
            "paired_mmc_improvement": 0.0005}
    assert hypothesis1_decision(good)["hypothesis1_passed"]
    for key, value in (("corr", 0.0), ("mmc", -1e-4), ("mmc", None),
                       ("ender_contribution", 1e-15), ("paired_mmc_improvement", None),
                       ("paired_mmc_improvement", -1e-5)):
        d = hypothesis1_decision({**good, key: value})
        assert not d["hypothesis1_passed"] and not d["deployment"]


def test_mmc_gate_and_runs():
    assert longest_run([5, 1, 2, 3, 7, 8]) == (3, 1, 3)
    assert longest_run([]) == (0, -1, -1)
    fail = mmc_gate(range(1133, 1228))
    assert fail["required_contiguous_eras"] == 195 and not fail["gate_passed"]
    assert fail["mmc_weight_optimization"] == "deferred"
    assert mmc_gate(range(1000, 1195))["gate_passed"]
    assert not mmc_gate([*range(1000, 1100), *range(1101, 1200)])["gate_passed"]
    assert not mmc_gate([])["gate_passed"]


def test_shared_dataset_reuse_matches_fresh_fit():
    _eras, x, y = synthetic()
    ds = lgb.Dataset(x, label=y[:, 0], free_raw_data=False).construct()
    lgb.train(PARAMS, ds, num_boost_round=20)
    ds.set_label(y[:, 1])
    reused = lgb.train(PARAMS, ds, num_boost_round=20)
    fresh = lgb.train(PARAMS, lgb.Dataset(x, label=y[:, 1]), num_boost_round=20)
    np.testing.assert_allclose(reused.predict(x), fresh.predict(x))


def test_zero_weight_labels_do_not_matter():
    eras, x, y = synthetic()
    w = (eras > 10).astype(float)
    preds = []
    for fill in (0.0, 123.0):
        label = np.where(w > 0, y[:, 0], fill)
        ds = lgb.Dataset(x, label=label, free_raw_data=False).construct()
        ds.set_weight(w)
        preds.append(lgb.train(PARAMS, ds, num_boost_round=20).predict(x))
    np.testing.assert_allclose(preds[0], preds[1])


def test_future_label_perturbation_leaves_fit_unchanged():
    eras, x, y = synthetic()
    perturbed = y.copy()
    perturbed[eras > 20] = np.random.default_rng(9).normal(size=perturbed[eras > 20].shape)
    out = []
    for labels in (y, perturbed):
        m = training_mask(eras, labels, np.ones(len(eras), bool), 20, 4, 1)
        model = lgb.train(PARAMS, lgb.Dataset(x[m], label=labels[m, 0]), num_boost_round=20)
        out.append(model.predict(x[eras > 33]))
    np.testing.assert_array_equal(out[0], out[1])


def test_serialization_feature_order_and_batches(tmp_path):
    _eras, x, y = synthetic()
    names = [f"feature_{i}" for i in range(x.shape[1])]
    model = lgb.train(PARAMS, lgb.Dataset(pd.DataFrame(x, columns=names), label=y[:, 0]),
                      num_boost_round=20)
    path = tmp_path / "m.txt"
    model.save_model(str(path))
    loaded = lgb.Booster(model_file=str(path))
    frame = pd.DataFrame(x, columns=names)
    shuffled = frame[names[::-1]]
    expected = model.predict(frame)
    np.testing.assert_allclose(loaded.predict(shuffled[loaded.feature_name()]), expected)
    batched = np.concatenate([loaded.predict(x[:17]), loaded.predict(x[17:])])
    np.testing.assert_allclose(batched, expected)


def test_heatmap_svg():
    m = pd.DataFrame([[1.0, -0.5], [-0.5, 1.0]], index=["a", "b<"], columns=["a", "b<"])
    svg = heatmap_svg(m, "t")
    assert svg.startswith("<svg") and svg.count("<rect") == 4 and "b&lt;" in svg
