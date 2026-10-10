import pytest

from numerai_dev.splits import assert_no_overlap, embargo_for, make_split


def eras(start: int, end: int) -> list[str]:
    return [f"{i:04d}" for i in range(start, end + 1)]


def test_embargo_by_horizon():
    assert embargo_for("target_ender_60") == 13
    assert embargo_for("target_ender_20") == 5


def test_split_respects_embargo():
    split = make_split(eras(1, 100), eras(101, 300), "target_ender_60", holdout_eras=50)
    assert split.holdout == eras(251, 300)
    assert split.dev[0] == "0114"  # 100 + 13 gap + 1
    assert split.dev[-1] == "0237"  # 251 - 13 gap - 1


def test_overlapping_windows_are_rejected():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_overlap(eras(1, 100), eras(105, 200), gap=13)


def test_train_eras_inside_validation_are_rejected():
    # Training that extends into the validation period must not produce a dev set
    # that starts within the embargo.
    split = make_split(eras(1, 150), eras(101, 300), "target_ender_60", holdout_eras=50)
    assert int(split.dev[0]) > 150 + 13
