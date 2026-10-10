import numpy as np
import pandas as pd

from numerai_dev import data


def write_fake(tmp_path, monkeypatch, n=200_000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "era": np.repeat([f"{i:04d}" for i in range(1, 11)], n // 10),
        "data_type": "validation",
        "feature_a": rng.integers(0, 5, n).astype("int8"),
        "feature_b": rng.integers(0, 5, n).astype("int8"),
        "target_x_20": rng.choice([0, 0.25, 0.5, 0.75, 1.0, np.nan], n),
    }, index=pd.Index([f"id{i}" for i in range(n)], name="id"))
    (tmp_path / "vX").mkdir()
    df.to_parquet(tmp_path / "vX" / "validation.parquet")
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "BATCH_ROWS", 7_000)  # force many uneven batches
    return df


def test_streamed_features_match_pandas(tmp_path, monkeypatch):
    df = write_fake(tmp_path, monkeypatch)
    idx = data.load_index("vX", "validation", ["target_x_20"])
    mask = data.era_mask(idx, era_stride=3, require=["target_x_20"])
    x = data.load_features("vX", "validation", ["feature_a", "feature_b"], mask)
    expected = df.loc[mask, ["feature_a", "feature_b"]].to_numpy()
    np.testing.assert_array_equal(x, expected)
    assert set(idx.loc[mask, "era"]) == {"0001", "0004", "0007", "0010"}


def test_iter_features_covers_mask(tmp_path, monkeypatch):
    df = write_fake(tmp_path, monkeypatch)
    idx = data.load_index("vX", "validation", [])
    mask = data.era_mask(idx, eras=["0002", "0009"])
    rows = np.concatenate([pos for pos, _ in data.iter_features("vX", "validation",
                                                                 ["feature_a"], mask)])
    np.testing.assert_array_equal(rows, np.flatnonzero(mask))
    assert len(rows) == (df["era"].isin(["0002", "0009"])).sum()
