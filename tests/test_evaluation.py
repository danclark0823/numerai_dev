import numpy as np
import pandas as pd

from numerai_dev.evaluation import evaluate, summarize


def fake_frame(n_eras: int = 20, n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_eras):
        signal = rng.normal(size=n)
        target = np.clip(np.round((signal + rng.normal(size=n)) * 0.5 + 0.5, 2) // 0.25 * 0.25, 0, 1)
        meta = signal + rng.normal(scale=2, size=n)
        rows.append(pd.DataFrame({
            "era": f"{e:04d}",
            "target": target,
            "good": signal,
            "noise": rng.normal(size=n),
            "numerai_meta_model": meta,
        }))
    df = pd.concat(rows, ignore_index=True)
    df.index = df.index.astype(str)
    return df


def test_signal_beats_noise():
    res = evaluate(fake_frame(), ["good", "noise"], "target")
    assert res["good"]["corr"]["mean"] > 0.2
    assert abs(res["noise"]["corr"]["mean"]) < 0.05
    assert res["good"]["mmc"]["mean"] > res["noise"]["mmc"]["mean"]
    assert set(res["good"]) == {"corr", "mmc", "payout_mmc"}


def test_summarize_drawdown():
    s = summarize(pd.Series([1.0, -2.0, 1.0, -1.0]))
    assert s["max_drawdown"] == -2.0
    assert s["pct_positive"] == 0.5
