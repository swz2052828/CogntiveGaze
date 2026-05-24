import numpy as np


def recording_kfolds(recording_ids, folds=5, seed=42):
    unique_recordings = np.asarray(sorted(set(int(rec) for rec in recording_ids)), dtype=np.int32)
    if folds < 2:
        raise ValueError("--folds must be at least 2 for cross validation.")
    if folds > len(unique_recordings):
        raise ValueError(f"--folds={folds} is larger than the number of recordings ({len(unique_recordings)}).")

    rng = np.random.default_rng(seed)
    shuffled = unique_recordings.copy()
    rng.shuffle(shuffled)
    val_folds = np.array_split(shuffled, folds)

    splits = []
    for fold_idx, val_recordings in enumerate(val_folds):
        train_recordings = np.setdiff1d(unique_recordings, val_recordings, assume_unique=False)
        splits.append(
            {
                "fold": fold_idx,
                "train_recordings": train_recordings.tolist(),
                "val_recordings": val_recordings.tolist(),
            }
        )
    return splits


def select_splits(splits, fold_index):
    if fold_index is None:
        return splits
    if fold_index < 0 or fold_index >= len(splits):
        raise ValueError(f"--fold-index must be between 0 and {len(splits) - 1}.")
    return [splits[fold_index]]
