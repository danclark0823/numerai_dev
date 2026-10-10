"""Loading Numerai data with only the columns we need, in memory-friendly dtypes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def data_path(version: str, name: str) -> Path:
    return DATA_DIR / version / name


def load_metadata(version: str) -> dict:
    return json.loads(data_path(version, "features.json").read_text())


def feature_set(version: str, name: str) -> list[str]:
    return load_metadata(version)["feature_sets"][name]


def load_split(
    version: str,
    split: str,
    features: list[str],
    targets: list[str],
    era_stride: int = 1,
    era_offset: int = 0,
) -> pd.DataFrame:
    """Load train/validation/live parquet with an era column, features as int8.

    era_stride > 1 keeps every Nth era (counting from era_offset), which cuts memory
    and reduces overlap between consecutive eras' targets.
    """
    path = data_path(version, f"{split}.parquet")
    schema_cols = set(pq.read_schema(path).names)
    extra = [c for c in ("data_type",) if c in schema_cols]
    cols = ["era", *extra, *features, *[t for t in targets if t in schema_cols]]
    df = pd.read_parquet(path, columns=cols)
    if "data_type" in df.columns and split == "validation":
        df = df[df["data_type"] == "validation"].drop(columns="data_type")
    df[features] = df[features].astype("int8")
    df["era"] = df["era"].astype(str)
    if era_stride > 1:
        eras = sorted(df["era"].unique(), key=era_key)
        keep = set(eras[era_offset::era_stride])
        df = df[df["era"].isin(keep)]
    return df


def era_key(era: str) -> int:
    """Eras are zero-padded numeric strings in train/validation; live uses 'X'."""
    return int(era) if era.isdigit() else 10**9
