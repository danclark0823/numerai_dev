import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_mavone as rm

FEATURES = ["feature_a", "feature_b", "feature_c"]


@pytest.fixture
def env(tmp_path):
    root, data = tmp_path / "repo", tmp_path / "data"
    for p in rm.IMPLEMENTATION:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(f"code {p}")
    data.mkdir()
    for f in rm.DATASET_FILES:
        (data / f).write_text(f"data {f}")
    cfg = yaml.safe_load((ROOT / "configs/mavone_research.yaml").read_text())
    identity = rm.current_identity(cfg, FEATURES, root, data)
    return {"root": root, "data": data, "cfg": cfg, "identity": identity, "tmp": tmp_path}


def make_run(out, identity, status="partial", blocks=("1133",), version=rm.PROVENANCE_VERSION):
    completed = {}
    for key in blocks:
        bdir = out / "blocks" / key
        bdir.mkdir(parents=True)
        (bdir / "preds.parquet").write_bytes(b"preds " + key.encode())
        (bdir / "ender.txt").write_text("model ender " + key)
        completed[key] = {"preds_sha256": rm.digest(bdir / "preds.parquet"),
                          "model_sha256": {"ender": rm.digest(bdir / "ender.txt")}}
    manifest = {**identity, "status": status, "completed_blocks": completed,
                "blocks": [{"start": int(k)} for k in blocks], "runs": []}
    if version is not None:
        manifest["provenance_version"] = version
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def snapshot(out):
    return {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def changed(env, **kw):
    new = copy.deepcopy(env["identity"])
    new.update(kw)
    return new


def test_matching_resume_returns_manifest_unchanged(env):
    out = make_run(env["tmp"] / "dc_mavone_x", env["identity"])
    before = snapshot(out)
    manifest = rm.open_for_forward(out, env["identity"], resume=True)
    assert manifest["status"] == "partial" and snapshot(out) == before


def test_new_run_and_bare_directory(env):
    assert rm.open_for_forward(env["tmp"] / "fresh", env["identity"], resume=False) is None
    bare = env["tmp"] / "bare"
    bare.mkdir()
    with pytest.raises(rm.ProvenanceError, match="without a manifest"):
        rm.open_for_forward(bare, env["identity"], resume=True)


@pytest.mark.parametrize("which", ["config", "source", "features", "dataset"])
def test_identity_mismatch_rejected_without_changes(env, which):
    out = make_run(env["tmp"] / "dc_mavone_x", env["identity"])
    before = snapshot(out)
    if which == "config":
        cfg = copy.deepcopy(env["cfg"])
        cfg["lgbm"]["num_threads"] = 4
        expected = changed(env, config=cfg)
    elif which == "source":
        (env["root"] / "src/numerai_dev/mavone.py").write_text("edited")
        expected = rm.current_identity(env["cfg"], FEATURES, env["root"], env["data"])
    elif which == "features":
        expected = rm.current_identity(env["cfg"], FEATURES[::-1], env["root"], env["data"])
    else:
        (env["data"] / "validation.parquet").write_text("refreshed snapshot")
        expected = rm.current_identity(env["cfg"], FEATURES, env["root"], env["data"])
    with pytest.raises(rm.ProvenanceError, match="new `name`") as err:
        rm.open_for_forward(out, expected, resume=True)
    key = {"config": "config", "source": "implementation_sha256",
           "features": "feature_order_sha256", "dataset": "dataset_sha256"}[which]
    assert key in str(err.value)
    with pytest.raises(rm.ProvenanceError):
        rm.open_for_evaluation(make_run(env["tmp"] / f"done_{which}", env["identity"],
                                        status="forward_complete"), expected)
    assert snapshot(out) == before


@pytest.mark.parametrize("target", ["preds.parquet", "ender.txt", "missing"])
def test_corrupted_completed_block_rejected(env, target):
    out = make_run(env["tmp"] / "dc_mavone_x", env["identity"], blocks=("1133", "1159"))
    bdir = out / "blocks" / "1159"
    if target == "missing":
        (bdir / "ender.txt").unlink()
    else:
        (bdir / target).write_bytes(b"tampered")
    before = snapshot(out)
    with pytest.raises(rm.ProvenanceError, match="block 1159"):
        rm.open_for_forward(out, env["identity"], resume=True)
    assert snapshot(out) == before


def test_legacy_manifest_fails_safely(env):
    out = make_run(env["tmp"] / "dc_mavone_old", env["identity"], status="forward_complete",
                   version=None)
    before = snapshot(out)
    with pytest.raises(rm.ProvenanceError, match="legacy manifest"):
        rm.open_for_evaluation(out, env["identity"])
    partial = make_run(env["tmp"] / "dc_mavone_old2", env["identity"], version=None)
    with pytest.raises(rm.ProvenanceError, match="legacy manifest"):
        rm.open_for_forward(partial, env["identity"], resume=True)
    assert snapshot(out) == before


def test_completed_runs_are_never_reopened(env):
    for status in sorted(rm.COMPLETED):
        out = make_run(env["tmp"] / f"dc_mavone_{status}", env["identity"], status=status)
        before = snapshot(out)
        for resume in (True, False):
            with pytest.raises(rm.ProvenanceError, match="never reopened"):
                rm.open_for_forward(out, env["identity"], resume=resume)
        assert snapshot(out) == before
    partial = make_run(env["tmp"] / "dc_mavone_partial", env["identity"])
    with pytest.raises(rm.ProvenanceError, match="--resume"):
        rm.open_for_forward(partial, env["identity"], resume=False)


def test_evaluation_guards(env):
    out = make_run(env["tmp"] / "dc_mavone_x", env["identity"], status="forward_complete")
    assert rm.open_for_evaluation(out, env["identity"])["status"] == "forward_complete"
    partial = make_run(env["tmp"] / "dc_mavone_p", env["identity"])
    with pytest.raises(rm.ProvenanceError, match="not forward_complete"):
        rm.open_for_evaluation(partial, env["identity"])
    (out / "evaluation").mkdir()
    before = snapshot(out)
    with pytest.raises(rm.ProvenanceError, match="never overwritten"):
        rm.open_for_evaluation(out, env["identity"])
    assert snapshot(out) == before
    with pytest.raises(rm.ProvenanceError, match="no training manifest"):
        rm.open_for_evaluation(env["tmp"] / "nothing", env["identity"])


def test_runner_refuses_before_writing(env, monkeypatch):
    artifacts = env["tmp"] / "artifacts"
    cfg = env["cfg"]
    monkeypatch.setattr(rm, "ROOT", env["tmp"])
    monkeypatch.setattr(rm, "run_features", lambda c, smoke=False: FEATURES)

    def identity(c, f):
        return {**env["identity"], "config": c, "feature_order_sha256": rm.ids_hash(f)}

    monkeypatch.setattr(rm, "current_identity", identity)
    done = make_run(artifacts / cfg["name"], identity(cfg, FEATURES), status="forward_complete")
    (done / "evaluation").mkdir()
    before = snapshot(done)
    with pytest.raises(rm.ProvenanceError):
        rm.run_forward(cfg, resume=True)
    with pytest.raises(rm.ProvenanceError):
        rm.run_evaluate(cfg)
    assert snapshot(done) == before


def test_completed_mavone_run_is_legacy_and_untouched():
    out = ROOT / "artifacts/dc_mavone_research_v1"
    if not (out / "manifest.json").exists():
        pytest.skip("completed Mavone artifact not present")
    before = (out / "manifest.json").read_bytes()
    manifest = json.loads(before)
    with pytest.raises(rm.ProvenanceError, match="legacy manifest"):
        rm.check_identity(out, manifest, {k: manifest.get(k) for k in rm.IDENTITY_KEYS})
    assert (out / "manifest.json").read_bytes() == before
