import pytest

from vit_gaze.splits import recording_kfolds, select_splits


def test_recording_kfolds_covers_every_recording_in_val_once():
    recordings = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    splits = recording_kfolds(recordings, folds=5, seed=42)

    assert len(splits) == 5
    val_seen = []
    for split in splits:
        train = set(split["train_recordings"])
        val = set(split["val_recordings"])
        assert train.isdisjoint(val)
        assert train | val == set(recordings)
        val_seen.extend(val)

    assert sorted(val_seen) == sorted(recordings)


def test_recording_kfolds_is_deterministic_with_seed():
    recordings = [10, 20, 30, 40, 50, 60]
    first = recording_kfolds(recordings, folds=3, seed=7)
    second = recording_kfolds(recordings, folds=3, seed=7)
    assert first == second


def test_recording_kfolds_deduplicates_input():
    splits = recording_kfolds([1, 1, 2, 2, 3, 3], folds=3, seed=0)
    all_recordings = set()
    for split in splits:
        all_recordings |= set(split["val_recordings"])
    assert all_recordings == {1, 2, 3}


def test_recording_kfolds_rejects_too_few_folds():
    with pytest.raises(ValueError):
        recording_kfolds([1, 2, 3], folds=1)


def test_recording_kfolds_rejects_more_folds_than_recordings():
    with pytest.raises(ValueError):
        recording_kfolds([1, 2, 3], folds=5)


def test_select_splits_returns_all_when_index_is_none():
    splits = recording_kfolds([1, 2, 3, 4], folds=2, seed=0)
    assert select_splits(splits, None) == splits


def test_select_splits_returns_single_fold_by_index():
    splits = recording_kfolds([1, 2, 3, 4], folds=2, seed=0)
    selected = select_splits(splits, 1)
    assert selected == [splits[1]]


def test_select_splits_rejects_out_of_range_index():
    splits = recording_kfolds([1, 2, 3, 4], folds=2, seed=0)
    with pytest.raises(ValueError):
        select_splits(splits, 5)
