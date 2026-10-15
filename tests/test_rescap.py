import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml
from numerai_tools.scoring import numerai_corr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rescap as rr

from numerai_dev.evaluation import per_era_corr
from numerai_dev.live import validate_submission
from numerai_dev.mavone import gaussian_rank, training_mask
from numerai_dev.provenance import ProvenanceError, ids_hash
from numerai_dev.rescap import (
    COMPARISONS,
    CURRENT,
    STANDARD_LARGE,
    adds_beyond_both,
    advancement,
    check_recipes,
    evaluation_blocks,
)
from numerai_dev.taren import INF, fit_through

CFG = yaml.safe_load((ROOT / "configs/rescap_research.yaml").read_text())
TINY = {"objective": "regression", "learning_rate": 0.1, "max_depth": 3, "num_leaves": 7,
        "colsample_bytree": 0.5, "random_state": 42, "num_threads": 1, "verbose": -1}


class Rows(lgb.Sequence):
    def __init__(self, x):
        self.x, self.batch_size = x, 64

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


def synthetic(n_eras=40, rows=50, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    eras = np.repeat(np.arange(1, n_eras + 1), rows)
    x = rng.integers(0, 5, (len(eras), n_features)).astype(np.float64)
    y = (x[:, 0] - x[:, 1]) / 8 + rng.normal(scale=0.2, size=len(eras))
    return eras, x, y


def test_config_and_published_recipes():
    rr.check_cfg(CFG)
    assert STANDARD_LARGE["n_estimators"] == 20000 and STANDARD_LARGE["learning_rate"] == 0.001
    assert STANDARD_LARGE["max_depth"] == 6 and STANDARD_LARGE["num_leaves"] == 2 ** 6
    assert STANDARD_LARGE["colsample_bytree"] == CURRENT["colsample_bytree"] == 0.1
    for k in ("objective", "random_state"):
        assert STANDARD_LARGE[k] == CURRENT[k]
    recipes = json.loads(json.dumps(CFG["trees"]["recipes"]))
    check_recipes(recipes, 10)
    recipes["large"]["params"]["learning_rate"] = 0.002
    with pytest.raises(ValueError):
        check_recipes(recipes, 10)
    with pytest.raises(ValueError):
        check_recipes({**CFG["trees"]["recipes"], "deep": {"params": {}}}, 10)
    for bad in ({"blend_weights": [0.6, 0.4]}, {"name": "dc_other_v1"},
                {"rondal": {**CFG["rondal"], "residualize_against": ["v53_lgbm_ender60"]}}):
        with pytest.raises(ValueError):
            rr.check_cfg({**CFG, **bad})


def test_evaluation_blocks_clip_and_reserve():
    blocks = evaluation_blocks(1133, 1211, 1227)
    assert [(b["start"], b["eval_end"]) for b in blocks] == [
        (1133, 1158), (1159, 1184), (1185, 1210), (1211, 1227)]
    for b in blocks:
        assert b["fit_end"] + 13 < b["start"]
    with pytest.raises(ValueError, match="reserved"):
        evaluation_blocks(1133, 1211, 1236)
    with pytest.raises(ValueError):
        evaluation_blocks(1133, 1211, 1200)


def test_advancement_rule():
    ok = advancement(0.025, 0.012, 0.026, 0.001, [0.002, 0.001, -0.001, 0.003])
    assert ok["advances"] and ok["corr_floor"] == pytest.approx(0.0234)
    assert not advancement(0.025, -0.001, 0.026, 0.001, [1, 1, 1, 1])["advances"]
    assert not advancement(0.025, 0.012, 0.026, 0.0, [1, 1, 1, 1])["advances"]
    assert not advancement(0.025, 0.012, 0.026, 0.001, [1, 1, -1, -1])["advances"]
    assert not advancement(0.0233, 0.012, 0.026, 0.001, [1, 1, 1, 1])["advances"]
    anchor = advancement(0.01, 0.012, -0.002, 0.001, [1, 1, 1, 1])
    assert anchor["inadequate_anchor"] and anchor["corr_floor_ok"] is None
    assert not anchor["advances"] and not anchor["deployment"]
    decisions = {"blend_current_vs_tree_current": {"advances": True},
                 "blend_current_vs_rondal": {"advances": False}}
    assert not adds_beyond_both(decisions, "blend_current")
    decisions["blend_current_vs_rondal"]["advances"] = True
    assert adds_beyond_both(decisions, "blend_current")
    assert ("tree_large", "tree_current") in COMPARISONS and len(COMPARISONS) == 5


def test_final_outputs_blend_formula_and_submission():
    rng = np.random.default_rng(4)
    eras = np.repeat(["1133", "1134", "1135"], 60)
    preds = pd.DataFrame(rng.normal(size=(180, 3)), columns=["tree_current", "tree_large",
                                                             "rondal"])
    preds["tree_current"] = preds.tree_current.round(1)  # ties
    preds["era"] = eras
    out = rr.final_outputs(preds)
    for tree, name in (("tree_current", "blend_current"), ("tree_large", "blend_large")):
        z = (gaussian_rank(preds[tree], eras) + gaussian_rank(preds.rondal, eras)) / 2
        np.testing.assert_allclose(out[name], pd.Series(z).groupby(eras).rank(pct=True))
    np.testing.assert_allclose(out.rondal, preds.rondal.groupby(eras).rank(pct=True))
    for era in np.unique(eras):
        m = eras == era
        for col in out.columns.drop("tree_current"):
            validate_submission(out.loc[m, col].rename("prediction"), out.index[m])
    perm = rng.permutation(len(preds))
    shuffled = rr.final_outputs(preds.iloc[perm])
    pd.testing.assert_frame_equal(shuffled.loc[out.index], out)
    frame = out.assign(era=eras, target=rng.choice([0, .25, .5, .75, 1], 180))
    ours = per_era_corr(frame, ["blend_current"], "target")["blend_current"]
    for era, rows in frame.groupby("era"):
        official = numerai_corr(rows[["blend_current"]], rows["target"])["blend_current"]
        assert ours[era] == pytest.approx(official)


def test_matched_training_rows_and_zero_weight_isolation():
    eras, x, y = synthetic()
    mask = training_mask(eras, y[:, None], np.ones(len(eras), bool), 30, 4, 1)
    label = np.where(eras[mask] < 9, np.nan, y[mask])
    w = np.isfinite(label).astype(float)
    preds = {}
    for fill in (0.0, 50.0):
        ds = rr.build_dataset([Rows(x[mask])], np.where(w > 0, label, fill), w)
        assert ds.num_data() == mask.sum()
        preds[fill] = {name: rr.train({**params, "n_estimators": n}, ds).predict(x)
                       for name, params, n in (("current", TINY, 15),
                                               ("large", {**TINY, "num_leaves": 9}, 25))}
    for name in ("current", "large"):
        np.testing.assert_allclose(preds[0.0][name], preds[50.0][name])
    assert ids_hash(np.flatnonzero(mask)) == ids_hash(np.flatnonzero(mask))


def test_future_labels_cannot_change_tree_or_ridge_fits():
    eras, x, y = synthetic()
    perturbed = y.copy()
    perturbed[eras > 30] += 100
    out = []
    for labels in (y, perturbed):
        m = training_mask(eras, labels[:, None], np.ones(len(eras), bool), 30, 4, 1)
        tree = rr.train({**TINY, "n_estimators": 15},
                        rr.build_dataset([Rows(x[m])], labels[m], np.ones(m.sum())))
        ridge = fit_through(x, labels, eras, [30], [INF], 10.0)[30][INF]
        out.append((tree.predict(x[eras > 33]), ridge.predict(x[eras > 33])))
    for a, b in zip(out[0], out[1], strict=True):
        np.testing.assert_array_equal(a, b)


def test_serialization_feature_order_and_batches(tmp_path):
    _, x, y = synthetic()
    names = [f"feature_{i}" for i in range(x.shape[1])]
    model = rr.train({**TINY, "n_estimators": 15},
                     lgb.Dataset(pd.DataFrame(x, columns=names), label=y))
    model.save_model(str(tmp_path / "m.txt"))
    loaded = lgb.Booster(model_file=str(tmp_path / "m.txt"))
    frame = pd.DataFrame(x, columns=names)
    expected = model.predict(frame)
    np.testing.assert_allclose(loaded.predict(frame[names[::-1]][loaded.feature_name()]),
                               expected)
    np.testing.assert_allclose(np.concatenate([loaded.predict(x[:9]), loaded.predict(x[9:])]),
                               expected)


def test_paired_differences_and_intervals():
    eras = [str(e) for e in range(1133, 1228)]
    rng = np.random.default_rng(5)
    per_era = {k: pd.DataFrame({"a": rng.normal(0.01, 0.01, 95), "b": rng.normal(0, 0.01, 95)},
                               index=eras) for k in ("corr", "mmc")}
    blocks = evaluation_blocks(1133, 1211, 1227)
    out = rr.paired(per_era, "a", "b", blocks)["mmc"]
    d = per_era["mmc"].a - per_era["mmc"].b
    assert out["mean"] == pytest.approx(d.mean()) and out["n_eras"] == 95
    assert out["ci_13"] is not None and out["ci_26"] is not None
    assert set(out["by_block"]) == {"1133", "1159", "1185", "1211"}
    assert out["worst_block"] == min(out["by_block"].values())
    assert out["by_block"]["1211"] == pytest.approx(d.iloc[-17:].mean())


def test_identity_files_exist_and_budget_required():
    for p in rr.SOURCES:
        assert (ROOT / p).exists(), p
    with pytest.raises(ValueError, match="budget"):
        rr.run_forward(CFG, None)


def test_runner_refuses_completed_run_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "ROOT", tmp_path)
    monkeypatch.setattr(rr, "feature_set", lambda version, name: ["f1", "f2"])
    ident = {"config": None, "feature_order_sha256": ids_hash(["f1", "f2"]),
             "implementation_sha256": {"x": "1"}, "dataset_sha256": {"y": "2"}}
    monkeypatch.setattr(rr, "current_identity", lambda cfg, f: {**ident, "config": cfg})
    out = tmp_path / "artifacts" / CFG["name"]
    out.mkdir(parents=True)
    manifest = {"provenance_version": 2, **ident, "config": CFG, "status": "forward_complete",
                "completed_blocks": {}, "checkpoint_sha256": {}}
    (out / "manifest.json").write_text(json.dumps(manifest))
    (out / "evaluation").mkdir()
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(ProvenanceError, match="never reopened"):
        rr.run_forward(CFG, 10, resume=True)
    with pytest.raises(ProvenanceError, match="never overwritten"):
        rr.run_evaluate(CFG)
    manifest["config"] = {**CFG, "num_threads": 4}
    (out / "manifest.json").write_text(json.dumps({**manifest, "status": "partial"}))
    with pytest.raises(ProvenanceError, match="config differs"):
        rr.run_forward(CFG, 10, resume=True)
    assert (out / "evaluation").exists()
    (out / "manifest.json").write_bytes(before)
    assert (out / "manifest.json").read_bytes() == before


def test_checkpointed_model_corruption_detected(tmp_path):
    from numerai_dev.provenance import digest, verify_blocks
    out = tmp_path / "run"
    (out / "blocks/1133").mkdir(parents=True)
    model = out / "blocks/1133/tree_large.txt"
    model.write_text("model")
    manifest = {"checkpoint_sha256": {"blocks/1133/tree_large.txt": digest(model)},
                "completed_blocks": {}}
    verify_blocks(out, manifest)
    model.write_text("tampered")
    with pytest.raises(ProvenanceError, match="checkpoint"):
        verify_blocks(out, manifest)
