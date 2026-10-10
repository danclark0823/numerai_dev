"""Chronological dev / holdout split of validation eras, with embargo gaps."""

from __future__ import annotations

from dataclasses import dataclass

from numerai_dev.data import era_key

# 60 market days of forward returns + ~2 day lag ≈ 13 weekly eras of overlap.
EMBARGO_ERAS = {"20": 5, "60": 13}


def embargo_for(target: str) -> int:
    return EMBARGO_ERAS[target.rsplit("_", 1)[-1]]


@dataclass(frozen=True)
class Split:
    dev: list[str]
    holdout: list[str]


def make_split(
    train_eras: list[str],
    val_eras: list[str],
    target: str,
    holdout_eras: int,
) -> Split:
    """Dev = validation eras after an embargo from the end of training, stopping an
    embargo before the holdout. Holdout = the last `holdout_eras` resolved eras."""
    gap = embargo_for(target)
    val = sorted(val_eras, key=era_key)
    last_train = max(era_key(e) for e in train_eras)
    holdout = val[-holdout_eras:]
    holdout_start = era_key(holdout[0])
    dev = [e for e in val if last_train + gap < era_key(e) < holdout_start - gap]
    assert_no_overlap(train_eras, dev, gap)
    assert_no_overlap(dev, holdout, gap)
    return Split(dev=dev, holdout=holdout)


def assert_no_overlap(earlier: list[str], later: list[str], gap: int) -> None:
    """Raise if any era in `later` is within `gap` eras of the end of `earlier`."""
    if not earlier or not later:
        return
    end = max(era_key(e) for e in earlier)
    start = min(era_key(e) for e in later)
    if start - end <= gap:
        raise ValueError(f"leakage: eras {end} and {start} are within embargo of {gap}")
