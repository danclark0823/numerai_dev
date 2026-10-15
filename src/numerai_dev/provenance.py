"""Run-identity guards shared by research runners: a run may only be resumed or evaluated
with the config, feature order, implementation and dataset files it was started with."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROVENANCE_VERSION = 2
IDENTITY_KEYS = ["config", "feature_order_sha256", "implementation_sha256", "dataset_sha256"]
COMPLETED = {"forward_complete", "complete"}


class ProvenanceError(RuntimeError):
    """A run's recorded identity does not match what would be used now."""

    def __init__(self, out: Path, problem: str):
        super().__init__(f"{out.name}: {problem}. Existing artifacts were left unchanged; set "
                         "a new `name` in the config to start a fresh run.")


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def identity(cfg, features, implementation, dataset_files, root: Path, data_dir: Path) -> dict:
    return {"config": cfg, "feature_order_sha256": ids_hash(features),
            "implementation_sha256": {p: digest(root / p) for p in implementation},
            "dataset_sha256": {f: digest(data_dir / f) for f in dataset_files}}


def check_identity(out: Path, manifest: dict, expected: dict):
    """Raise ProvenanceError for a legacy manifest or any identity mismatch."""
    missing = [k for k in ["provenance_version", *IDENTITY_KEYS] if k not in manifest]
    if missing or manifest.get("provenance_version") != PROVENANCE_VERSION:
        raise ProvenanceError(out, "legacy manifest without the recorded provenance needed "
                                   f"for resume or evaluation (missing {missing}); its "
                                   "sources/ snapshot is its reproduction reference")
    diffs = []
    for key in IDENTITY_KEYS:
        recorded, now = manifest[key], expected[key]
        if isinstance(now, dict) and key != "config":
            changed = sorted(k for k in set(recorded) | set(now) if recorded.get(k) != now.get(k))
            if changed:
                diffs.append(f"{key} differs for {changed}")
        elif recorded != now:
            diffs.append(f"{key} differs")
    if diffs:
        raise ProvenanceError(out, "; ".join(diffs))


def verify_files(out: Path, label: str, files: dict[str, str]):
    """Every relative path in `files` must exist with its recorded sha256."""
    for name, recorded in files.items():
        path = out / name
        if not path.exists():
            raise ProvenanceError(out, f"{label} is missing {name}")
        if digest(path) != recorded:
            raise ProvenanceError(out, f"{label} {name} does not match its recorded hash")


def block_files(key: str, audit: dict) -> dict[str, str]:
    if "files_sha256" in audit:
        return {f"blocks/{key}/{n}": h for n, h in audit["files_sha256"].items()}
    return {f"blocks/{key}/preds.parquet": audit["preds_sha256"],
            **{f"blocks/{key}/{n}.txt": h for n, h in audit["model_sha256"].items()}}


def verify_blocks(out: Path, manifest: dict):
    """Completed blocks, checkpointed models and other recorded files must match."""
    for key, audit in manifest.get("completed_blocks", {}).items():
        verify_files(out, f"block {key}", block_files(key, audit))
    verify_files(out, "checkpoint", manifest.get("checkpoint_sha256", {}))


def open_for_forward(out: Path, expected: dict, resume: bool) -> dict | None:
    """Existing manifest to continue (verified, unmodified), or None for a new run."""
    path = out / "manifest.json"
    if not path.exists():
        if out.exists():
            raise ProvenanceError(out, "directory exists without a manifest")
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("status") in COMPLETED:
        raise ProvenanceError(out, f"run is already {manifest['status']}; completed runs are "
                                   "never reopened")
    if not resume:
        raise ProvenanceError(out, "run exists; pass --resume to continue it")
    check_identity(out, manifest, expected)
    verify_blocks(out, manifest)
    return manifest


def open_for_evaluation(out: Path, expected: dict) -> dict:
    path = out / "manifest.json"
    if not path.exists():
        raise ProvenanceError(out, "no training manifest")
    manifest = json.loads(path.read_text())
    check_identity(out, manifest, expected)
    if manifest.get("status") != "forward_complete":
        raise ProvenanceError(out, f"forward predictions are {manifest.get('status')}, "
                                   "not forward_complete")
    if (out / "evaluation").exists():
        raise ProvenanceError(out, "evaluation already exists; completed evaluations are "
                                   "never overwritten")
    verify_blocks(out, manifest)
    return manifest
