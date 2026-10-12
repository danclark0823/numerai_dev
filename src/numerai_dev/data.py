"""Loading Numerai data with only the columns we need, in memory-friendly form.

Features are streamed from parquet straight into a preallocated int8 numpy array
(no pandas copy), so the full 3.5k-feature set fits in 24 GB.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BATCH_ROWS = 65_536


def data_path(version: str, name: str) -> Path:
    return DATA_DIR / version / name


def load_metadata(version: str) -> dict:
    return json.loads(data_path(version, "features.json").read_text())


def feature_set(version: str, name: str) -> list[str]:
    return load_metadata(version)["feature_sets"][name]


def era_key(era: str) -> int:
    """Eras are zero-padded numeric strings in train/validation; live uses 'X'."""
    return int(era) if era.isdigit() else 10**9


def load_index(version: str, split: str, targets: list[str]) -> pd.DataFrame:
    """id-indexed frame of era, data_type (if present) and the requested targets,
    in file row order. Use it to build a row mask for `load_features`."""
    path = data_path(version, f"{split}.parquet")
    names = set(pq.read_schema(path).names)
    cols = ["era", *[c for c in ("data_type",) if c in names], *[t for t in targets if t in names]]
    df = pd.read_parquet(path, columns=cols)
    df["era"] = df["era"].astype(str)
    return df


def era_mask(
    index: pd.DataFrame,
    era_stride: int = 1,
    era_offset: int = 0,
    eras: list[str] | None = None,
    require: list[str] | None = None,
    data_type: str | None = None,
) -> np.ndarray:
    """Boolean row mask: keep every `era_stride`-th era (from `era_offset`), optionally
    restricted to `eras`, rows of `data_type`, and rows where `require` columns are set."""
    mask = np.ones(len(index), dtype=bool)
    if data_type is not None and "data_type" in index.columns:
        mask &= (index["data_type"] == data_type).to_numpy()
    if require:
        mask &= index[require].notna().all(axis=1).to_numpy()
    candidates = sorted(index.loc[mask, "era"].unique(), key=era_key)
    if eras is not None:
        wanted = set(eras)
        candidates = [e for e in candidates if e in wanted]
    keep = set(candidates[era_offset::era_stride])
    return mask & index["era"].isin(keep).to_numpy()


def _batches(version: str, split: str, features: list[str]) -> Iterator[np.ndarray]:
    pf = pq.ParquetFile(data_path(version, f"{split}.parquet"))
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=features):
        yield np.column_stack(
            [col.to_numpy(zero_copy_only=False) for col in batch.columns]
        ).astype(np.int8, copy=False)


def load_features(
    version: str, split: str, features: list[str], mask: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """int8 array of shape (mask.sum(), len(features)) for the masked rows.
    Pass `out` (e.g. a slice of a bigger preallocated array) to fill in place."""
    if out is None:
        out = np.empty((int(mask.sum()), len(features)), dtype=np.int8)
    row = pos = 0
    for x in _batches(version, split, features):
        m = mask[row : row + len(x)]
        n = int(m.sum())
        out[pos : pos + n] = x[m]
        row += len(x)
        pos += n
    assert pos == len(out), "row count mismatch while streaming features"
    return out


def iter_features(
    version: str, split: str, features: list[str], mask: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (row_positions, int8 features) for masked rows, batch by batch.
    For prediction on large splits without materializing the whole matrix."""
    row = 0
    for x in _batches(version, split, features):
        m = mask[row : row + len(x)]
        if m.any():
            yield np.flatnonzero(m) + row, x[m]
        row += len(x)


class ParquetRows(lgb.Sequence):
    """LightGBM `Sequence` over the masked rows of a parquet split, streamed one
    file batch at a time. LightGBM reads rows in increasing order (sampling, then
    construction), so only the current batch is ever held: the binned Dataset is
    built without materializing the raw feature matrix."""

    def __init__(self, version: str, split: str, features: list[str], mask: np.ndarray,
                 batch_size: int = BATCH_ROWS):
        self.version, self.split, self.features, self.mask = version, split, features, mask
        self.batch_size = batch_size  # rows per range read requested by LightGBM
        n_batches = -(-len(mask) // BATCH_ROWS)
        counts = [int(mask[i * BATCH_ROWS:(i + 1) * BATCH_ROWS].sum()) for i in range(n_batches)]
        self.cum = np.concatenate([[0], np.cumsum(counts)])
        self._gen: Iterator[np.ndarray] | None = None
        self._gen_pos = -1
        self._cache_id = -1
        self._cache: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.cum[-1])

    def _chunk(self, k: int) -> np.ndarray:
        if k == self._cache_id:
            return self._cache
        if self._gen is None or k <= self._gen_pos:
            self._gen = _batches(self.version, self.split, self.features)
            self._gen_pos = -1
        while self._gen_pos < k:
            x = next(self._gen)
            self._gen_pos += 1
        expected = len(self.mask[k * BATCH_ROWS:(k + 1) * BATCH_ROWS])
        assert len(x) == expected, "parquet batches not aligned to BATCH_ROWS"
        self._cache = x[self.mask[k * BATCH_ROWS:(k + 1) * BATCH_ROWS]]
        self._cache_id = k
        return self._cache

    def _locate(self, i: int) -> int:
        return int(np.searchsorted(self.cum, i, side="right")) - 1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, _ = idx.indices(len(self))
            pieces, i = [], start
            while i < stop:
                k = self._locate(i)
                chunk = self._chunk(k)
                lo = i - self.cum[k]
                hi = min(stop - self.cum[k], len(chunk))
                pieces.append(chunk[lo:hi])
                i += hi - lo
            # LightGBM pushes float rows; converting per batch keeps the cache int8.
            if not pieces:
                return np.empty((0, len(self.features)), np.float32)
            return np.concatenate(pieces).astype(np.float32)
        k = self._locate(int(idx))
        return self._chunk(k)[int(idx) - self.cum[k]].astype(np.float64)  # sampling needs double
